"""Authorized single-shard OpenXLab benchmark; never modify the HF supervisor."""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import sa1b_recovery as recovery
from sa1b_throughput_benchmark import payload_bytes


FOLDER = recovery.EVIDENCE / "sa1b-opendatalab/authorized-probe"
FILENAME = "sa_000038.tar"
TARGET = recovery.ASSETS / "downloads/sa1b-opendatalab/test-sa_000038"
ARCHIVE = TARGET / "OpenDataLab___SA-1B/raw/sa_000038.tar"
MIB = 1024 ** 2


def recover_complete_cache(downloader, row):
    ranges = downloader._BigFileDownloader__get_ranges_from_cache()
    if not ranges or downloader.LOG or ranges[0][0] != 0:
        return None
    if any(previous[1]+1 != current[0] for previous, current in zip(ranges, ranges[1:])) or ranges[-1][1]+1 != row["expected_size_bytes"]:
        return None
    archive = Path(downloader.download_dir) / downloader.prefix / downloader.filename
    proof = dict(mechanism="Official SDK start() otherwise waits forever when all resume ranges are already complete; invoke SDK native assembler after exact contiguous coverage validation",
                 cached_ranges=len(ranges), cached_bytes=row["expected_size_bytes"], cache_preserved=True,
                 network_payload_required=False, archive=str(archive))
    if archive.is_file() and archive.stat().st_size == row["expected_size_bytes"]:
        try:
            proof.update(recovery.verify_archive(archive, row), existing_full_archive_reused=True)
            downloader._BigFileDownloader__done.set()
            downloader._BigFileDownloader__main_thread_done.set()
            return proof
        except ValueError:
            pass
    if archive.is_file():
        preserved = archive.with_name(archive.name+f".preassembly-{os.getpid()}-{time.time_ns()}.partial")
        archive.rename(preserved)
        proof["preserved_preassembly_partial"] = str(preserved)
    clear = downloader.clear
    downloader.clear = lambda: None
    try:
        downloader._BigFileDownloader__sew()
    finally:
        downloader.clear = clear
    proof.update(recovery.verify_archive(archive, row), existing_full_archive_reused=False)
    return proof


def payload_stalled(counters, started, now, archive_recent=False):
    if archive_recent:
        return False
    last = counters.get("last_payload_monotonic")
    return now-started >= 600 if last is None else now-last >= 300


def worker():
    import requests
    from openxlab.dataset import download
    from openxlab.dataset.io.downloader import BigFileDownloader

    FOLDER.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    lock = threading.RLock()
    counters = dict(bytes=0, errors=0, retries=0, statuses={}, hosts=[], ranges=[],
                    payload_started_monotonic=None, last_payload_monotonic=None)
    row = next(entry for entry in recovery.load(recovery.STATE)["shards"] if entry["filename"] == FILENAME)
    original_start = BigFileDownloader.start

    def resumable_start(downloader):
        proof = recover_complete_cache(downloader, row)
        if proof:
            recovery.atomic_json(FOLDER / "completed-cache-resume.json", proof)
            return
        return original_start(downloader)

    BigFileDownloader.start = resumable_start
    request = requests.Session.request

    def wrapped(session, method, url, **kwargs):
        kwargs["timeout"] = (15, 45)
        address = urlsplit(url)
        range_header = kwargs.get("headers", {}).get("Range")
        if kwargs.get("stream"):
            with lock:
                location = address.hostname + address.path
                if location not in counters["hosts"]:
                    counters["hosts"].append(location)
                if range_header:
                    counters["ranges"].append(range_header)
        try:
            response = request(session, method, url, **kwargs)
        except requests.RequestException:
            with lock:
                counters["errors"] += 1
            raise
        with lock:
            status = str(response.status_code)
            counters["statuses"][status] = counters["statuses"].get(status, 0) + 1
            if response.status_code >= 400:
                counters["errors"] += 1
        if kwargs.get("stream"):
            original = response.iter_content

            def content(*args, **arguments):
                try:
                    for chunk in original(*args, **arguments):
                        with lock:
                            now = time.monotonic()
                            if counters["payload_started_monotonic"] is None:
                                counters["payload_started_monotonic"] = now
                            counters["last_payload_monotonic"] = now
                            counters["bytes"] += len(chunk)
                        yield chunk
                except requests.RequestException:
                    with lock:
                        counters["errors"] += 1
                    raise

            response.iter_content = content
        return response

    class RetryHandler(logging.Handler):
        def emit(self, record):
            if "retry" in record.getMessage().lower():
                with lock:
                    counters["retries"] += 1

    requests.Session.request = wrapped
    logging.getLogger("openxlab").addHandler(RetryHandler())
    stop = threading.Event()
    worker_started = time.monotonic()

    def publish():
        while not stop.is_set():
            with lock:
                recovery.atomic_json(FOLDER / "worker-counters.json", dict(utc=recovery.now(), **counters))
                snapshot = counters.copy()
            archive_recent = ARCHIVE.is_file() and time.time()-ARCHIVE.stat().st_mtime < 300
            if payload_stalled(snapshot, worker_started, time.monotonic(), archive_recent):
                recovery.atomic_json(FOLDER / "worker-result.json", dict(completed=False,
                    error_type="PayloadStallTimeout", finished_utc=recovery.now(), partials_preserved=True,
                    reason="No payload for5minutes, or no initial payload for10minutes; SDK may be waiting on a failed range thread",
                    payload_bytes=snapshot["bytes"], resume_required=True))
                os._exit(75)
            stop.wait(2)

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    result = dict(started_utc=recovery.now(), completed=False)
    try:
        with open(os.devnull, "w") as suppressed, contextlib.redirect_stdout(suppressed), contextlib.redirect_stderr(suppressed):
            download(dataset_repo="OpenDataLab/SA-1B", source_path="/raw/" + FILENAME, target_path=str(TARGET))
        result.update(completed=ARCHIVE.exists(), finished_utc=recovery.now())
    except BaseException as error:
        result.update(error_type=type(error).__name__, exit_code=getattr(error, "code", None),
                      partials_preserved=True, finished_utc=recovery.now())
    finally:
        stop.set()
        publisher.join(timeout=3)
        with lock:
            recovery.atomic_json(FOLDER / "worker-counters.json", dict(utc=recovery.now(), **counters))
        recovery.atomic_json(FOLDER / "worker-result.json", result)


def finish_worker(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run():
    FOLDER.mkdir(parents=True, exist_ok=False)
    state = recovery.load(recovery.STATE)
    row = next(row for row in state["shards"] if row["filename"] == "sa_000038.tar")
    if row["download_status"] != "pending" or payload_bytes(row) != 0:
        raise RuntimeError("Refuse shard already started by HF")
    start_hf = {entry["filename"]: payload_bytes(entry) for entry in state["shards"]}
    beginning = time.monotonic()
    result = dict(started_utc=recovery.now(), repository="OpenDataLab/SA-1B", dataset_id=6248,
                  filename=row["filename"], account_login_passed=True, download_check_passed=True,
                  minimum_test_seconds=600, training_allowed=False, smoke_allowed=False,
                  source=str(ARCHIVE), split_recovery_enabled=False)
    recovery.atomic_json(FOLDER / "result.json", result)
    process = subprocess.Popen([str(recovery.ROOT / ".opendatalab-venv/bin/python"), "-u", __file__, "worker"],
                               cwd=recovery.ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples = []
    previous_bytes = 0
    previous_hf = start_hf
    previous_time = beginning
    try:
        with (FOLDER / "samples.jsonl").open("w", buffering=1) as output:
            while time.monotonic() - beginning < 600 or process.poll() is None:
                time.sleep(5)
                now = time.monotonic()
                counters = recovery.load(FOLDER / "worker-counters.json", {})
                current_hf = {entry["filename"]: payload_bytes(entry) for entry in state["shards"]}
                interval = now - previous_time
                sample = dict(utc=recovery.now(), elapsed_seconds=now - beginning, odl_bytes=counters.get("bytes", 0),
                    odl_mib_s=max(0, counters.get("bytes", 0) - previous_bytes) / interval / MIB,
                    hf_mib_s=sum(max(0, current_hf[name] - previous_hf[name]) for name in previous_hf) / interval / MIB,
                    errors=counters.get("errors", 0), retries=counters.get("retries", 0))
                samples.append(sample)
                output.write(json.dumps(sample) + "\n")
                previous_bytes = counters.get("bytes", 0)
                previous_time, previous_hf = now, current_hf
                elapsed = now - beginning
                rate = counters.get("bytes", 0) / elapsed / MIB
                if len(samples) % 12 == 0:
                    print(f"seconds={elapsed:.0f} ODL_avg={rate:.2f} HF_last={sample['hf_mib_s']:.2f} MiB/s errors={sample['errors']}", flush=True)
                if elapsed >= 600 and process.poll() is None and (rate < 8 or counters.get("errors", 0) > 2):
                    result["stopped_reason"] = "Ten-minute throughput below 8MiB/s or repeated errors; partials retained"
                    finish_worker(process)
                if elapsed > 7200:
                    result["stopped_reason"] = "Bounded two-hour probe; partials retained"
                    finish_worker(process)
    finally:
        finish_worker(process)
    duration = time.monotonic() - beginning
    counters = recovery.load(FOLDER / "worker-counters.json", {})
    hf_bytes = sum(max(0, previous_hf[name] - start_hf[name]) for name in start_hf)
    result.update(finished_utc=recovery.now(), duration_seconds=duration, counters=counters,
                  average_mib_s=counters.get("bytes", 0) / duration / MIB,
                  peak_five_second_mib_s=max((sample["odl_mib_s"] for sample in samples), default=0),
                  hf_average_mib_s=hf_bytes / duration / MIB, combined_mib_s=(hf_bytes + counters.get("bytes", 0)) / duration / MIB,
                  worker_result=recovery.load(FOLDER / "worker-result.json", {}),
                  archive_complete=ARCHIVE.exists() and ARCHIVE.stat().st_size == row["expected_size_bytes"],
                  resume_actual_range_206=counters.get("statuses", {}).get("206", 0) > 0,
                  partials_preserved=True, hf_supervisor_untouched=True)
    if result["archive_complete"]:
        try:
            directory = FOLDER / "full-audit"
            subprocess.run([str(recovery.ROOT / ".venv/bin/python"),
                str(recovery.RECOVERY / "sa1b_opendatalab_split.py"), "audit", "--filename", row["filename"],
                "--archive", str(ARCHIVE), "--directory", str(directory)],
                cwd=recovery.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            result["protocol_audit"] = recovery.load(directory / "audit.json")
        except Exception as error:
            result["protocol_audit"] = dict(protocol_passed=False, error_type=type(error).__name__, archive_preserved=True)
    else:
        result["protocol_audit"] = dict(protocol_passed=False, status="NOT_RUN_INCOMPLETE_ARCHIVE")
    result["speed_gate"] = result["average_mib_s"] >= 8 and counters.get("errors", 0) <= 2
    result["combined_gain"] = result["combined_mib_s"] / 5.299793803521094
    result["split_gates_passed"] = result["speed_gate"] and result["combined_gain"] >= 1.3 and result["protocol_audit"].get("protocol_passed", False)
    recovery.atomic_json(FOLDER / "result.json", result)
    print("FINISHED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    args = parser.parse_args()
    if args.command == "worker":
        worker()
    else:
        run()
