"""Measure disjoint original-shard SDK transfers, preserve partials, then resume recovery."""

import argparse
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit

import sa1b_recovery as recovery


OUTPUT = recovery.EVIDENCE / "sa1b-throughput-benchmark"
RESULT = OUTPUT / "results.json"
MIB = 1024 ** 2


def events(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def payload_bytes(row):
    archive = Path(row["archive"])
    sizes = [archive.stat().st_size] if archive.exists() else []
    cache = archive.parent / ".cache/huggingface/download"
    pattern = "*." + row["expected_lfs_sha256"] + ".incomplete" if row.get("expected_lfs_sha256") else row["filename"] + ".*.incomplete"
    sizes.extend(path.stat().st_size for path in cache.glob(pattern))
    return max(sizes, default=0)


def select_stages(rows):
    candidates = [row for row in rows if row["filename"] != "sa_000014.tar"
                  and row.get("download_status") not in ("verified", recovery.QUARANTINED)
                  and row.get("extraction_status") != "complete"
                  and not Path(row["archive"]).exists()]
    if len(candidates) < 20:
        raise RuntimeError("Need twenty distinct unfinished non-quarantined shards")
    stages = []
    offset = 0
    for concurrency in (2, 4, 6, 8):
        stages.append((concurrency, candidates[offset:offset + concurrency]))
        offset += concurrency
    return stages


def choose_best(stages):
    stable = [stage for stage in stages if stage["stable"]]
    if not stable:
        return 2, "No stable tier; retain safe original concurrency without claiming a winner"
    fastest = max(stage["total_mib_s"] for stage in stable)
    eligible = [stage for stage in stable if stage["total_mib_s"] >= fastest * 0.95]
    chosen = min(eligible, key=lambda stage: stage["concurrency"])
    return chosen["concurrency"], "Lowest stable concurrency within 5% of highest aggregate throughput"


def worker(filename, event_path):
    import requests
    from huggingface_hub import configure_http_backend, get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.utils._runtime import is_xet_available

    state = recovery.load(recovery.STATE)
    row = next(entry for entry in state["shards"] if entry["filename"] == filename)
    if row.get("download_status") in ("verified", recovery.QUARANTINED) or Path(row["archive"]).exists():
        raise RuntimeError("Refusing completed or quarantined archive")
    stream = Path(event_path).open("a", buffering=1)

    def emit(kind, **fields):
        stream.write(json.dumps(dict(kind=kind, utc=recovery.now(), **fields)) + "\n")

    class Session(requests.Session):
        def request(self, method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (10, 45)
            emit("http_request", method=method, host=urlsplit(url).hostname,
                 range=kwargs.get("headers", {}).get("Range"))
            try:
                response = super().request(method, url, **kwargs)
            except Exception as error:
                emit("http_exception", error_type=type(error).__name__)
                raise
            emit("http_response", method=method, status=response.status_code)
            if method == "GET":
                original_iterator = response.iter_content

                def monitored_content(*args, **arguments):
                    try:
                        yield from original_iterator(*args, **arguments)
                    except requests.RequestException as error:
                        emit("http_stream_exception", error_type=type(error).__name__)
                        raise

                response.iter_content = monitored_content
            return response

    class RetryHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage().lower()
            if "retry" in message or "resum" in message:
                emit("retry_warning", logger=record.name)

    configure_http_backend(backend_factory=Session)
    logging.getLogger("huggingface_hub").addHandler(RetryHandler())
    logging.getLogger("urllib3").addHandler(RetryHandler())
    try:
        metadata = get_hf_file_metadata(hf_hub_url(recovery.REPO, row["repo_path"],
            repo_type="dataset", revision=state["revision"], endpoint=state["endpoint"]), token=False)
        xet_active = bool(metadata.xet_file_data and is_xet_available())
        emit("metadata", xet_metadata_present=metadata.xet_file_data is not None,
             xet_package_available=is_xet_available(), actual_transport="hf_xet" if xet_active else "SDK resumable HTTP",
             size=metadata.size, etag=metadata.etag)
        if metadata.size != row["expected_size_bytes"] or metadata.etag != row["expected_lfs_sha256"]:
            raise RuntimeError("Pinned immutable object metadata mismatch")
        if xet_active:
            raise RuntimeError("Xet sparse file sizes are not valid benchmark payload counters")
        destination = Path(row["archive"])
        path = hf_hub_download(recovery.REPO, row["repo_path"], repo_type="dataset", token=False,
            revision=state["revision"], endpoint=state["endpoint"], local_dir=destination.parent,
            force_download=False)
        emit("download_complete", size=Path(path).stat().st_size,
             validation="Deferred to normal checksum/tar/extraction pipeline; not marked verified")
    except KeyboardInterrupt:
        emit("interrupted", partial_preserved=True)
    except Exception as error:
        emit("worker_failed", error_type=type(error).__name__)
        raise
    finally:
        stream.close()


def system_counters(pids):
    cpu_line = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    cpu = [int(value) for value in cpu_line[:8]]
    network = [0, 0]
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        interface, values = line.split(":", 1)
        if interface.strip() != "lo":
            values = values.split()
            network[0] += int(values[0])
            network[1] += int(values[8])
    process_ticks = 0
    process_write = 0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            process_ticks += int(fields[11]) + int(fields[12])
            io = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
            process_write += int(io["write_bytes"])
        except (FileNotFoundError, ProcessLookupError):
            pass
    nfs_writes = None
    selected = False
    for line in Path("/proc/self/mountstats").read_text().splitlines():
        if line.startswith("device "):
            selected = " mounted on /opt/data/private " in line
        elif selected and line.strip().startswith("bytes:"):
            nfs_writes = int(line.split()[2])
            break
    return dict(cpu_total=sum(cpu), cpu_idle=cpu[3] + cpu[4], net_rx=network[0], net_tx=network[1],
                process_ticks=process_ticks, process_write_bytes=process_write, nfs_write_bytes=nfs_writes)


def stop_workers(workers):
    for process in workers:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in workers:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def stage(concurrency, rows, duration):
    directory = OUTPUT / f"concurrency-{concurrency}"
    directory.mkdir(parents=True, exist_ok=False)
    starts = {row["filename"]: payload_bytes(row) for row in rows}
    workers = []
    logs = []
    started = time.monotonic()
    start_utc = recovery.now()
    try:
        for row in rows:
            log = (directory / (row["filename"] + ".log")).open("w")
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, "-u", __file__, "worker", "--filename", row["filename"],
                "--events", str(directory / (row["filename"] + ".jsonl"))], cwd=recovery.ROOT,
                stdout=log, stderr=subprocess.STDOUT))
        previous = system_counters([process.pid for process in workers])
        previous_sizes = starts.copy()
        previous_time = started
        samples = []
        early_stop = None
        with (directory / "samples.jsonl").open("w", buffering=1) as output:
            while time.monotonic() - started < duration:
                time.sleep(min(5, max(0, duration - (time.monotonic() - started))))
                current_time = time.monotonic()
                interval = current_time - previous_time
                current = system_counters([process.pid for process in workers])
                sizes = {row["filename"]: payload_bytes(row) for row in rows}
                per_shard = {name: max(0, sizes[name] - previous_sizes[name]) / interval / MIB for name in starts}
                total_cpu = max(1, current["cpu_total"] - previous["cpu_total"])
                sample = dict(utc=recovery.now(), elapsed_seconds=current_time - started,
                    total_mib_s=sum(per_shard.values()), per_shard_mib_s=per_shard,
                    host_cpu_percent=100 * (1 - (current["cpu_idle"] - previous["cpu_idle"]) / total_cpu),
                    worker_cpu_percent_one_core=100 * max(0, current["process_ticks"] - previous["process_ticks"]) /
                        os.sysconf("SC_CLK_TCK") / interval,
                    network_rx_mib_s=max(0, current["net_rx"] - previous["net_rx"]) / interval / MIB,
                    network_tx_mib_s=max(0, current["net_tx"] - previous["net_tx"]) / interval / MIB,
                    worker_physical_disk_write_mib_s=max(0, current["process_write_bytes"] - previous["process_write_bytes"]) / interval / MIB,
                    logical_download_file_write_mib_s=sum(per_shard.values()),
                    mount_nfs_write_mib_s=None if current["nfs_write_bytes"] is None else
                        max(0, current["nfs_write_bytes"] - previous["nfs_write_bytes"]) / interval / MIB)
                samples.append(sample)
                output.write(json.dumps(sample) + "\n")
                previous, previous_sizes, previous_time = current, sizes, current_time
                all_events = [event for row in rows for event in events(directory / (row["filename"] + ".jsonl"))]
                severe = sum(event.get("status", 0) == 429 or event.get("status", 0) >= 500 for event in all_events)
                failures = sum(event["kind"] == "worker_failed" for event in all_events)
                if current_time - started <= 120 and (severe >= 5 or failures >= math.ceil(concurrency / 2)):
                    early_stop = "Major early failures or repeated 429/5xx"
                    break
                if all(process.poll() is not None for process in workers) and failures:
                    early_stop = "All workers stopped after failure"
                    break
                if len(samples) % 6 == 0:
                    print(f"tier={concurrency} seconds={current_time-started:.0f} total={sample['total_mib_s']:.2f} MiB/s", flush=True)
        measured_seconds = time.monotonic() - started
        ends = {row["filename"]: payload_bytes(row) for row in rows}
    finally:
        stop_workers(workers)
        for log in logs:
            log.close()
    all_events = [event for row in rows for event in events(directory / (row["filename"] + ".jsonl"))]
    errors = sum(event["kind"] in ("http_exception", "http_stream_exception", "worker_failed") or
                 event.get("status", 0) >= 400 for event in all_events)
    retries = 0
    for row in rows:
        shard_events = events(directory / (row["filename"] + ".jsonl"))
        retries += max(sum(event["kind"] == "retry_warning" for event in shard_events),
                       max(0, sum(event["kind"] == "http_request" and event.get("method") == "GET" for event in shard_events) - 1))
    transferred = {name: max(0, ends[name] - starts[name]) for name in starts}
    total_rate = sum(transferred.values()) / measured_seconds / MIB
    rates = [sample["total_mib_s"] for sample in samples]
    longest_stall = current_stall = 0
    for sample in samples:
        current_stall = current_stall + 5 if sample["total_mib_s"] < 0.1 else 0
        longest_stall = max(longest_stall, current_stall)
    severe_errors = sum(event.get("status", 0) == 429 or event.get("status", 0) >= 500 for event in all_events)
    worker_failures = sum(event["kind"] == "worker_failed" for event in all_events)
    metadata = [event for event in all_events if event["kind"] == "metadata"]
    passed_duration = measured_seconds >= duration - 0.1
    stable = passed_duration and total_rate > 0 and severe_errors == 0 and worker_failures == 0 and retries <= concurrency and longest_stall < 30
    result = dict(concurrency=concurrency, filenames=list(starts), started_utc=start_utc,
        finished_utc=recovery.now(), measured_seconds=measured_seconds, total_bytes=sum(transferred.values()),
        total_mib_s=total_rate, average_per_shard_mib_s=total_rate / concurrency,
        per_shard_mib_s={name: count / measured_seconds / MIB for name, count in transferred.items()},
        initial_bytes=starts, end_bytes=ends, errors=errors, retries=retries,
        severe_http_errors=severe_errors, worker_failures=worker_failures, stable=stable,
        early_stop=early_stop, longest_stall_seconds=longest_stall,
        rate_coefficient_of_variation=statistics.pstdev(rates) / statistics.mean(rates) if rates and statistics.mean(rates) else None,
        xet_metadata_present=any(event["xet_metadata_present"] for event in metadata), metadata_responses=len(metadata),
        transport="SDK resumable HTTP" if len(metadata) == concurrency and not any(event["xet_metadata_present"] for event in metadata) else "Inspect worker evidence",
        metrics={key: statistics.mean(sample[key] for sample in samples if sample[key] is not None)
                 for key in ("host_cpu_percent", "worker_cpu_percent_one_core", "network_rx_mib_s", "network_tx_mib_s",
                             "worker_physical_disk_write_mib_s", "logical_download_file_write_mib_s", "mount_nfs_write_mib_s")
                 if any(sample[key] is not None for sample in samples)},
        sampling_interval_seconds=5, evidence_directory=str(directory))
    recovery.atomic_json(directory / "summary.json", result)
    print("STAGE_COMPLETE " + json.dumps(result), flush=True)
    return result


def resume(group, concurrency):
    try:
        members = subprocess.check_output(["ps", "-o", "pid=,args=", "-g", str(group)], text=True)
    except subprocess.CalledProcessError:
        members = ""
    if members:
        if "sa1b_recovery.py run --downloads-only" not in members or "continue_sa1b_recovery.sh" not in members:
            raise RuntimeError("Refusing to signal unexpected process group")
        os.killpg(group, signal.SIGKILL)
    for attempt in range(100):
        lock = (recovery.EVIDENCE / "sa1b-download.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            time.sleep(0.1)
        else:
            lock.close()
            break
    else:
        raise RuntimeError("Paused supervisor lock not released")
    state = recovery.load(recovery.STATE)
    state.update(download_concurrency=concurrency, throughput_benchmark=str(RESULT),
                 recovery_policy="Download-only; selected benchmark concurrency; quarantined shards skipped; no decode/smoke/training")
    recovery.atomic_json(recovery.STATE, state)
    environment = os.environ.copy()
    environment["SA1B_DOWNLOAD_CONCURRENCY"] = str(concurrency)
    log_path = recovery.EVIDENCE / "sa1b-download-benchmark-resumed.log"
    with log_path.open("a") as output:
        process = subprocess.Popen(["flock", "-n", "recovery/evidence/sa1b-download.lock", "bash", "recovery/continue_sa1b_recovery.sh"],
            cwd=recovery.ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
    job = dict(pid=process.pid, started_utc=recovery.now(), concurrency=concurrency,
               command="flock -n recovery/evidence/sa1b-download.lock bash recovery/continue_sa1b_recovery.sh",
               log=str(log_path), smoke_allowed=False, training_allowed=False)
    recovery.atomic_json(recovery.EVIDENCE / "sa1b-resumed-job.json", job)
    time.sleep(3)
    job["alive_after_launch"] = process.poll() is None
    recovery.atomic_json(recovery.EVIDENCE / "sa1b-resumed-job.json", job)
    return job


def retire_paused_supervisor(group):
    members = subprocess.check_output(["ps", "-o", "pid=,stat=,args=", "-g", str(group)], text=True)
    if "sa1b_recovery.py run --downloads-only" not in members or "continue_sa1b_recovery.sh" not in members:
        raise RuntimeError("Refusing unexpected supervisor process group")
    if not all(line.split()[1].startswith("T") for line in members.splitlines() if line.strip()):
        raise RuntimeError("Supervisor must be fully stopped before releasing SDK locks")
    os.killpg(group, signal.SIGKILL)
    time.sleep(1)
    return dict(group=group, retired_utc=recovery.now(), partials_preserved=True,
                reason="Release supervisor and SDK cache locks before benchmark; restart after selecting concurrency")


def report(result):
    state = recovery.load(recovery.STATE)
    remaining = sum(max(0, row["expected_size_bytes"] - payload_bytes(row)) for row in state["shards"]
                    if row.get("download_status") != recovery.QUARANTINED and row.get("extraction_status") != "complete")
    chosen = next((stage for stage in result["stages"] if stage["concurrency"] == result["best_concurrency"] and stage["stable"]), None)
    rate = chosen["total_mib_s"] if chosen else None
    result.update(remaining_normal_download_bytes=remaining,
                  pure_download_hours=remaining / (rate * MIB) / 3600 if rate else None,
                  best_stable_total_mib_s=rate)
    recovery.atomic_json(RESULT, result)
    lines = ["# SA-1B throughput benchmark", "", f"Started: {result['started_utc']}; updated: {recovery.now()}.",
        f"Repository `{recovery.REPO}`, immutable revision `{state['revision']}`, endpoint `{state['endpoint']}`.",
        "No training/smoke. Existing supervisor process group stopped before transfers; no partial/cache/completed archive deleted.",
        "Each tier uses disjoint unfinished shards, excluding 000014 and all checksum-matching tar-invalid quarantines.",
        "Measured rates are net SDK HTTP payload persisted to archive/partial files, not total NIC traffic or sparse Xet allocation.",
        "Every successful tier runs at least 300 seconds; start-up is included. Five-second raw samples retained.",
        "Errors count HTTP >=400, request exceptions and worker failures; retries count max of SDK retry warnings and repeated GET attempts per shard.",
        "Stable: full duration, nonzero rate, no 429/5xx/final failure, at most one retry per connection, no >=30-second aggregate stall.",
        "Lowest stable concurrency within 5% of the fastest wins. Short-run results are provisional, not proof of sustained all-night performance.", "",
        "| Concurrency | Total MiB/s | Avg per shard | Errors | Retries | Stable |",
        "|---:|---:|---:|---:|---:|---|" ]
    for stage_result in result["stages"]:
        lines.append(f"| {stage_result['concurrency']} | {stage_result['total_mib_s']:.3f} | {stage_result['average_per_shard_mib_s']:.3f} | {stage_result['errors']} | {stage_result['retries']} | {stage_result['stable']} |")
    lines.extend(["", "## Per-shard and host metrics", "",
        "CPU: host CPU busy percentage and benchmark-worker percentage of one core. Network: network-namespace interfaces except loopback.",
        "Storage is NFS. File-growth/logical write rate is attributable to this benchmark; kernel physical disk write_bytes may be zero on NFS.",
        "NFS mount write counters and NIC rates include other work/shared traffic and are not exclusive benchmark throughput.",
        "File-growth rates have 10MiB SDK-buffer quantization; zero individual samples alone do not prove a connection dropped.", "",
        "| Concurrency | Host CPU % | Workers CPU % of one core | NIC RX MiB/s | NIC TX MiB/s | Logical writes MiB/s | NFS writes MiB/s |",
        "|---:|---:|---:|---:|---:|---:|---:|"])
    for stage_result in result["stages"]:
        metrics = stage_result["metrics"]
        lines.append(f"| {stage_result['concurrency']} | {metrics['host_cpu_percent']:.2f} | {metrics['worker_cpu_percent_one_core']:.2f} | {metrics['network_rx_mib_s']:.3f} | {metrics['network_tx_mib_s']:.3f} | {metrics['logical_download_file_write_mib_s']:.3f} | {metrics.get('mount_nfs_write_mib_s', 0):.3f} |")
    for stage_result in result["stages"]:
        lines.extend(["", f"### Concurrency {stage_result['concurrency']}",
            f"Duration: {stage_result['measured_seconds']:.2f}s; actual transport: {stage_result['transport']}; Xet metadata: {stage_result['xet_metadata_present']}.",
            f"Longest aggregate stall: {stage_result['longest_stall_seconds']}s; five-second rate CV: {stage_result['rate_coefficient_of_variation']}.",
            f"Mean host/worker/network/storage metrics: `{json.dumps(stage_result['metrics'], sort_keys=True)}`.",
            f"Per-shard MiB/s: `{json.dumps(stage_result['per_shard_mib_s'], sort_keys=True)}`.",
            f"Raw evidence: `{stage_result['evidence_directory']}`."])
    lines.extend(["", "## Selection and resumption", "", f"Best concurrency: {result['best_concurrency']}; {result.get('selection_reason', 'not yet selected')}.",
        f"Best stable aggregate: {rate} MiB/s. Baseline was approximate 4–5 MiB/s, not an identical-shard controlled test.",
        f"Improvement versus baseline range: {rate / 5:.2f}–{rate / 4:.2f}x." if rate else "No stable winning rate established.",
        f"Remaining normal download bytes (excluding quarantine): {remaining}; pure download estimate: {result['pure_download_hours']} hours.",
        "Pure-download ETA excludes checksum/tar/extraction/existence/decode audits and unresolved invalid archives 000014/000016/000017.",
        f"Resumed supervisor: `{json.dumps(result.get('resumed_supervisor', {}), sort_keys=True)}`.",
        "hf_xet is installed/enabled but HIGH_PERFORMANCE on/off is not compared when HEAD provides no Xet metadata.",
        f"Official HF speed test: `{json.dumps(result.get('official_speed_test', {}), sort_keys=True)}`."])
    if result.get("safety_audit"):
        lines.extend(["", "## Safety acceptance", "",
            f"Protected verified/quarantined archives unchanged: {result['safety_audit']['protected_archives_unchanged']}.",
            f"All benchmark partial byte counts preserved or increased: {result['safety_audit']['benchmark_partial_bytes_retained']}.",
            "Instrumentation and cache-lock preflights are excluded from tier results; evidence and all downloaded bytes were retained.",
            f"Post-resume download sample: `{json.dumps(result.get('post_resume_sample', {}), sort_keys=True)}`.",
            "No training/smoke started; invalid shards remain unresolved and are not counted as recovered."])
    (recovery.RECOVERY / "SA1B_THROUGHPUT_BENCHMARK.md").write_text("\n".join(lines) + "\n")


def benchmark(group, duration):
    if duration < 300:
        raise ValueError("Each benchmark tier needs at least 300 seconds")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "benchmark.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = recovery.load(recovery.STATE)
    stages = select_stages(state["shards"])
    result = dict(started_utc=recovery.now(), supervisor_paused_group=group, stages=[], best_concurrency=2,
                  selection_reason="Benchmark in progress", training_allowed=False, smoke_allowed=False,
                  official_speed_test=recovery.load(OUTPUT / "official-speed-test.json", {}))
    recovery.atomic_json(RESULT, result)
    try:
        result["supervisor_retired"] = retire_paused_supervisor(group)
        recovery.atomic_json(RESULT, result)
        for concurrency, rows in stages:
            result["stages"].append(stage(concurrency, rows, duration))
            result["best_concurrency"], result["selection_reason"] = choose_best(result["stages"])
            report(result)
    except BaseException as error:
        result["interruption"] = type(error).__name__
        raise
    finally:
        try:
            result["resumed_supervisor"] = resume(group, result["best_concurrency"])
        except Exception as error:
            result["resume_error"] = str(error)
            try:
                os.killpg(group, signal.SIGCONT)
                result["fallback_original_supervisor_resumed"] = True
            except ProcessLookupError:
                result["fallback_original_supervisor_resumed"] = False
        result["finished_utc"] = recovery.now()
        report(result)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--paused-group", type=int)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--filename")
    parser.add_argument("--events")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.filename, args.events)
    else:
        benchmark(args.paused_group, args.seconds)


if __name__ == "__main__":
    main()
