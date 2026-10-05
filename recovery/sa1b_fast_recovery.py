"""Disjoint source queues, short ODL benchmark and original-image ingestion."""

import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import sa1b_recovery as recovery
import sa1b_opendatalab_probe as probe
from sa1b_throughput_benchmark import payload_bytes


FOLDER = recovery.EVIDENCE / "sa1b-fast-recovery"
PLAN = FOLDER / "source-plan.json"
LIVE = FOLDER / "state.json"
BENCH = FOLDER / "benchmark.json"
BAD = {14, 16, 17}
MIB = 1024 ** 2
OWNERSHIP_LOCK = threading.Lock()


def source_path(filename):
    if filename == "sa_000038.tar":
        cached = recovery.ASSETS / "downloads/sa1b-opendatalab/test-sa_000038/OpenDataLab___SA-1B/raw" / filename
        if cached.is_file():
            return cached
    return recovery.ASSETS / "downloads/sa1b-opendatalab" / ("split-" + filename.removesuffix(".tar")) / "OpenDataLab___SA-1B/raw" / filename


def plan_sources(state):
    owners = {}
    snapshots = {}
    for position, row in enumerate(state["shards"]):
        count = payload_bytes(row)
        snapshots[row["filename"]] = count
        if position in BAD or row.get("download_status") == recovery.QUARANTINED:
            owners[row["filename"]] = "RESCUE_ONLY"
        elif row.get("extraction_status") == "complete":
            owners[row["filename"]] = "COMPLETE"
        elif source_path(row["filename"]).is_file() or Path(row["archive"] + ".opendatalab-provenance.json").is_file():
            owners[row["filename"]] = "ODL"
        elif count:
            owners[row["filename"]] = "HF_EXISTING_PARTIAL_ONLY"
        else:
            owners[row["filename"]] = "ODL"
    return dict(created_utc=recovery.now(), owners=owners, initial_hf_bytes=snapshots,
                hf_concurrency=4, odl_benchmark_seconds=180, formal_training_allowed=False,
                smoke_only_after_full_existence_and_decode=True)


def hf_job(state, row, plan):
    if plan["owners"][row["filename"]] != "HF_EXISTING_PARTIAL_ONLY" or not plan["initial_hf_bytes"][row["filename"]]:
        raise ValueError("HF cannot acquire a zero-progress or ODL-owned object")
    result = recovery.download_shard(state, row)
    if result.get("download_status") not in ("verified", recovery.QUARANTINED):
        raise RuntimeError("HF download failed; partial retained: " + row["filename"])
    return result


def start_odl(filename, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([str(recovery.ROOT / ".opendatalab-venv/bin/python"), "-u",
        str(recovery.RECOVERY / "sa1b_opendatalab_split.py"), "worker", "--filename", filename,
        "--directory", str(directory)], cwd=recovery.ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def counters(directory):
    return recovery.load(directory / "worker-counters.json", {})


def odl_ready(row):
    filename = row["filename"]
    path = source_path(filename)
    if not path.is_file() or path.stat().st_size != row["expected_size_bytes"]:
        return False
    if filename == "sa_000038.tar":
        audit = recovery.load(recovery.EVIDENCE / "sa1b-opendatalab/authorized-probe/full-audit/audit.json", {})
        return audit.get("protocol_passed") and audit.get("observed_sha256") == row["expected_lfs_sha256"]
    records = [recovery.EVIDENCE / "sa1b-opendatalab/split-recovery" / filename.removesuffix(".tar") / "worker-result.json"]
    records.extend(FOLDER / "odl-workers" / filename / str(attempt) / "worker-result.json" for attempt in (1, 2, 3))
    return any(recovery.load(record, {}).get("completed") for record in records)


def odl_job(row, plan):
    filename = row["filename"]
    if plan["owners"][filename] != "ODL":
        raise ValueError("ODL refuses a HF-owned object")
    if odl_ready(row):
        return dict(filename=filename, source="ODL", archive=str(source_path(filename)))
    for attempt in range(1, 4):
        directory = FOLDER / "odl-workers" / filename / str(attempt)
        process = start_odl(filename, directory)
        process.wait()
        result = recovery.load(directory / "worker-result.json", {})
        if result.get("completed") and source_path(filename).is_file():
            return dict(filename=filename, source="ODL", archive=str(source_path(filename)),
                        attempt=attempt, counters=counters(directory), worker_result=result)
        if result.get("error_type") in ("KeyboardInterrupt", "SystemExit"):
            raise RuntimeError("ODL interrupted; partial retained")
        time.sleep(10)
    raise RuntimeError("ODL bounded retries exhausted; partial retained: " + filename)


def benchmark(state, plan):
    saved = recovery.load(BENCH, {})
    if saved.get("status") == "COMPLETE":
        return normalize_benchmark(saved)
    stages = [(1, [41]), (2, [42, 43]), (3, [44, 45, 46])]
    results = []
    for concurrency, positions in stages:
        for position in positions:
            row = state["shards"][position]
            if plan["owners"][row["filename"]] != "ODL" or plan["initial_hf_bytes"][row["filename"]] or source_path(row["filename"]).exists():
                raise ValueError("Benchmark shard is not exclusively unstarted ODL work")
        workers = []
        begin = time.monotonic()
        hf_begin = sum(payload_bytes(row) for row in state["shards"] if plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY")
        for position in positions:
            filename = state["shards"][position]["filename"]
            directory = FOLDER / "benchmark" / str(concurrency) / filename
            workers.append((filename, directory, start_odl(filename, directory)))
        samples = []
        previous = 0
        previous_time = begin
        try:
            while time.monotonic() - begin < 180:
                time.sleep(10)
                values = {filename: counters(directory) for filename, directory, process in workers}
                total = sum(value.get("bytes", 0) for value in values.values())
                stamp = time.monotonic()
                elapsed = stamp - begin
                samples.append(dict(elapsed_seconds=elapsed, total_mib_s=(total-previous)/(stamp-previous_time)/MIB,
                    cumulative_bytes=total, per_shard=values))
                previous, previous_time = total, stamp
                recovery.atomic_json(BENCH, dict(status="RUNNING", stage=concurrency, stages=results, samples=samples,
                    started_utc=recovery.now(), shard_ownership=plan["owners"]))
                if elapsed >= 120 and sum(value.get("errors", 0) for value in values.values()) >= 20:
                    break
        finally:
            for filename, directory, process in workers:
                probe.finish_worker(process)
        elapsed = samples[-1]["elapsed_seconds"]
        values = {filename: counters(directory) for filename, directory, process in workers}
        total = samples[-1]["cumulative_bytes"]
        hf_end = sum(payload_bytes(row) for row in state["shards"] if plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY")
        errors = sum(value.get("errors", 0) for value in values.values())
        rates = [sample["total_mib_s"] for sample in samples[2:]]
        results.append(dict(concurrency=concurrency, seconds=elapsed, total_mib_s=total/elapsed/MIB,
            avg_per_shard_mib_s=total/elapsed/MIB/concurrency,
            per_shard_mib_s={filename: value.get("bytes", 0)/elapsed/MIB for filename, value in values.items()},
            errors=errors, retries=sum(value.get("retries", 0) for value in values.values()),
            stable=elapsed >= 180 and errors == 0 and bool(rates) and min(rates) > 0.5,
            hf_total_mib_s=max(0, hf_end-hf_begin)/elapsed/MIB, samples=samples, partials_preserved=True))
    eligible = [row for row in results if row["stable"]]
    if not eligible:
        raise RuntimeError("No stable ODL benchmark stage; do not enable unproven concurrency")
    peak = max(row["total_mib_s"] for row in eligible)
    selected = min((row for row in eligible if row["total_mib_s"] >= peak * .95), key=lambda row: row["concurrency"])
    result = dict(status="COMPLETE", finished_utc=recovery.now(), stages=results, selected=selected,
                  policy="Lowest stable concurrency within5% of highest aggregate throughput", partials_preserved=True)
    recovery.atomic_json(BENCH, result)
    lines = ["# ODL short concurrency benchmark", "", "Original official SDK; eight HTTP range threads per shard. Three180-second stages; distinct zero-HF-progress shards.",
        "HF continues only pre-existing partials. Benchmark partials are retained for ODL resume.", "",
        "| Concurrency | Total MiB/s | Avg per shard | Errors | Retries | Stable | HF MiB/s |", "|---:|---:|---:|---:|---:|---|---:|"]
    for row in results:
        lines.append(f"| {row['concurrency']} | {row['total_mib_s']:.3f} | {row['avg_per_shard_mib_s']:.3f} | {row['errors']} | {row['retries']} | {row['stable']} | {row['hf_total_mib_s']:.3f} |")
    lines.extend(["", f"Selected concurrency: {selected['concurrency']}; stable aggregate: {selected['total_mib_s']:.3f}MiB/s.",
                  "Live combined throughput and remaining-byte ETA: evidence/sa1b-fast-recovery/state.json."])
    (recovery.RECOVERY / "SA1B_ODL_SHORT_BENCHMARK.md").write_text("\n".join(lines)+"\n")
    return result


def normalize_benchmark(value):
    if value.get("clock_correction"):
        return value
    original = copy.deepcopy(value)
    recovery.atomic_json(FOLDER / "benchmark-original-timing.json", original)
    for row in value["stages"]:
        previous = 0
        seconds = 0
        for sample in row["samples"]:
            increment = sample["cumulative_bytes"]-previous
            rate = sample["total_mib_s"]
            if increment <= 0 or rate <= 0:
                raise ValueError("Cannot reconstruct an ambiguous timing interval")
            seconds += increment/rate/MIB
            previous = sample["cumulative_bytes"]
        factor = row["seconds"]/seconds
        row.update(seconds=seconds, total_mib_s=previous/seconds/MIB,
            avg_per_shard_mib_s=previous/seconds/MIB/row["concurrency"],
            per_shard_mib_s={filename: rate*factor for filename, rate in row["per_shard_mib_s"].items()},
            hf_total_mib_s=row["hf_total_mib_s"]*factor,
            stable=seconds >= 180 and row["errors"] == 0 and min(sample["total_mib_s"] for sample in row["samples"][2:]) > .5)
    eligible = [row for row in value["stages"] if row["stable"]]
    peak = max(row["total_mib_s"] for row in eligible)
    value["selected"] = min((row for row in eligible if row["total_mib_s"] >= peak*.95), key=lambda row: row["concurrency"])
    value["clock_correction"] = "Original sample elapsed was read before slow NFS counter reads. Actual monotonic intervals reconstructed exactly from recorded byte deltas / interval rates; no rebenchmark or invented bytes."
    recovery.atomic_json(BENCH, value)
    lines = ["# ODL short concurrency benchmark", "", value["clock_correction"], "",
        "| Concurrency | Total MiB/s | Avg per shard | Errors | Retries | Stable | HF MiB/s |", "|---:|---:|---:|---:|---:|---|---:|"]
    for row in value["stages"]:
        lines.append(f"| {row['concurrency']} | {row['total_mib_s']:.3f} | {row['avg_per_shard_mib_s']:.3f} | {row['errors']} | {row['retries']} | {row['stable']} | {row['hf_total_mib_s']:.3f} |")
    lines.extend(["", f"Selected ODL concurrency: {value['selected']['concurrency']}. All benchmark SDK partials retained.",
        "Eight Range threads per shard; requests used official OpenXLab SDK. Zero observed HTTP errors; retry count is the instrumented SDK counter, not a guarantee about unlogged internal retries.",
        "Live combined rates and source-partitioned ETA: evidence/sa1b-fast-recovery/state.json."])
    (recovery.RECOVERY / "SA1B_ODL_SHORT_BENCHMARK.md").write_text("\n".join(lines)+"\n")
    return value


def remaining_bytes(state, plan):
    totals = dict(HF_EXISTING_PARTIAL_ONLY=0, ODL=0, ODL_FORENSIC=0)
    for row in state["shards"]:
        source = plan["owners"][row["filename"]]
        if source not in totals or row.get("extraction_status") == "complete":
            continue
        expected = row["expected_size_bytes"]
        if source == "HF_EXISTING_PARTIAL_ONLY":
            received = payload_bytes(row)
        elif source == "ODL_FORENSIC":
            from sa1b_wallclock_recovery import forensic_archive
            path = forensic_archive(row["filename"])
            received = path.stat().st_size if path.is_file() else sum(part.stat().st_size for part in (path.parent / ".cache").glob(row["filename"]+".*.odl"))
        else:
            path = source_path(row["filename"])
            if path.is_file():
                received = path.stat().st_size
            else:
                received = sum(path.stat().st_size for path in (path.parent / ".cache").glob(row["filename"]+".*.odl"))
        totals[source] += max(0, expected-received)
    return totals


def ingest(row, downloaded, ownership):
    row = copy.deepcopy(row)
    filename = row["filename"]
    if downloaded.get("source") == "ODL":
        row["archive"] = downloaded["archive"]
        row.update(recovery.verify_archive(row["archive"], row))
        folder = FOLDER / "ingestion" / filename
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "tar-members.txt").open("w") as listing, (folder / "tar-stderr.txt").open("wb") as errors:
            process = subprocess.run(["tar", "-tf", row["archive"]], stdout=listing, stderr=errors)
        row.update(download_status="verified" if process.returncode == 0 else recovery.QUARANTINED,
            tar_integrity_passed=process.returncode == 0, tar_listing=str(folder / "tar-members.txt"),
            tar_stderr=str(folder / "tar-stderr.txt"), source_repository="OpenDataLab/SA-1B", source_dataset_id=6248,
            download_provenance=downloaded, mirror_classification="BITWISE_EQUIVALENT_MIRROR")
    if row.get("download_status") == recovery.QUARANTINED:
        return row
    staging = Path("/tmp/said-sa1b-original-jpeg-staging") / filename.removesuffix(".tar")
    row["training_images_installed"] = False
    local_ownership = ownership.copy()
    recovery.extract_shard(row, local_ownership, destination_root=staging)
    output = FOLDER / "ingestion" / filename / "decode.json"
    subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "sa1b_jpeg_rescue.py"),
        "transfer-inventory", "--source-root", str(staging), "--inventory", row["image_inventory"], "--output", str(output)], check=True)
    audit = recovery.load(output)
    row["jpeg_decode_audit"] = audit
    if not audit["passed"]:
        raise ValueError("Required original-image decode failed: " + filename)
    names = [json.loads(line)["basename"] for line in Path(row["image_inventory"]).open()]
    commit_ownership(ownership, names, filename)
    row.update(training_images_installed=True, local_original_jpeg_staging=str(staging))
    recovery.atomic_json(recovery.EVIDENCE / "sa1b-shards" / (filename+".json"), row)
    return row


def commit_ownership(ownership, names, filename):
    with OWNERSHIP_LOCK:
        if any(name in ownership and ownership[name] != filename for name in names):
            raise ValueError("Cross-shard duplicate original-image provenance")
        ownership.update(dict.fromkeys(names, filename))


def adopt_external_install(entry, ownership):
    output = recovery.load(Path(entry["output"]), {})
    if not output:
        if not Path(f"/proc/{entry['pid']}").exists():
            raise RuntimeError("External install exited without a complete decode/transfer proof")
        return None
    if output.get("passed") is not True or output.get("image_transform") != "NONE" or output.get("destination_root") != str(recovery.SAM_ROOT) or output.get("source_root") != entry["source_root"]:
        raise ValueError("External original-JPEG transfer/decode failed")
    row = recovery.load(recovery.EVIDENCE / "sa1b-shards" / (entry["filename"]+".json"))
    if (row.get("observed_sha256") != row["expected_lfs_sha256"] or
        row.get("observed_md5") != row["expected_checklist_checksum"] or not row.get("tar_integrity_passed") or
        recovery.digest(Path(row["image_inventory"])) != row["image_inventory_sha256"] or
        output.get("inventory_sha256") != row["image_inventory_sha256"]):
        raise ValueError("External install provenance differs from pinned original archive/inventory")
    names = [json.loads(line)["basename"] for line in Path(row["image_inventory"]).open()]
    if output.get("checked_images") != len(names) or output.get("failures"):
        raise ValueError("External install coverage is incomplete")
    commit_ownership(ownership, names, row["filename"])
    row.update(training_images_installed=True, jpeg_decode_audit=output,
        local_original_jpeg_staging=entry["source_root"], external_install_adopted_utc=recovery.now())
    recovery.atomic_json(recovery.EVIDENCE / "sa1b-shards" / (row["filename"]+".json"), row)
    return row


def quick_completeness(families, state, ownership=None):
    previous = recovery.load(recovery.RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    for family, count in (("coco", 118287), ("llava", 558128)):
        if previous["families"][family]["recovered_paths"] != count or previous["families"][family]["missing_paths"]:
            raise ValueError("Unverified retained COCO/LLaVA snapshot")
    names = set((ownership or {}).copy())
    for row in state["shards"]:
        rescue = row.get("required_jpeg_rescue", {})
        inventory = Path(rescue.get("image_inventory", "/nonexistent"))
        if inventory.is_file() and recovery.digest(inventory) == rescue.get("image_inventory_sha256"):
            names.update(json.loads(line)["basename"] for line in inventory.open())
    missing = sorted(path for path in families["sam"] if Path(path).name not in names)
    (recovery.EVIDENCE / "sa1b-missing-training-paths.txt").write_text("\n".join(missing)+("\n" if missing else ""))
    previous["families"]["sam"].update(required_paths=569486, recovered_paths=569486-len(missing), missing_paths=len(missing), missing_examples=missing[:12])
    previous.update(checked_utc=recovery.now(), training_index_sha256=recovery.RECORDS_SHA,
        missing_training_paths=len(missing), training_index_missing=len(missing), recovered_training_paths=1245901-len(missing),
        passed=False, decode_audit=dict(executed=False, passed=False),
        update_scope="Byte-verified installed-image inventories joined to immutable index; avoids blocking downloads on large NFS directory scans; final full physical existence/type/decode audit mandatory; unchanged verified COCO/LLaVA snapshots retained")
    recovery.atomic_json(recovery.RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", previous)
    recovery.atomic_json(recovery.EVIDENCE / "training-image-completeness-audit.json", previous)
    state.update(sam_required_paths_recovered=569486-len(missing), sam_required_paths_missing=len(missing), last_batch_audit_utc=recovery.now())
    return previous


def publish_progress(families, state, ownership, plan, selected, activity, previous):
    completeness = quick_completeness(families, state, ownership)
    recovery.atomic_json(recovery.STATE, state)
    recovery.report(state, completeness)
    highwater_path = FOLDER / "hf-byte-highwater.json"
    highwater = recovery.load(highwater_path, dict(plan["initial_hf_bytes"]))
    for row in state["shards"]:
        if plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY":
            highwater[row["filename"]] = max(highwater.get(row["filename"], 0), payload_bytes(row))
    recovery.atomic_json(highwater_path, highwater)
    hf_bytes = sum(highwater[row["filename"]] for row in state["shards"] if plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY")
    odl_bytes = sum(counters(path.parent).get("bytes", 0) for path in FOLDER.glob("odl-workers/*/*/worker-counters.json"))
    odl_bytes += sum(counters(path.parent).get("bytes", 0) for path in (recovery.EVIDENCE / "sa1b-wallclock/forensic").glob("*/worker/worker-counters.json"))
    stamp = time.monotonic()
    elapsed = stamp-previous[0]
    hf_rate = max(0, hf_bytes-previous[1])/elapsed/MIB
    odl_rate = max(0, odl_bytes-previous[2])/elapsed/MIB
    remaining = remaining_bytes(state, plan)
    hf_reference = hf_rate if hf_rate > .5 else (selected["hf_total_mib_s"] if selected else 5)
    odl_reference = odl_rate if odl_rate > .5 else (selected["total_mib_s"] if selected else 24)
    live = dict(checked_utc=recovery.now(), pid=os.getpid(), supervisor_running=True,
        benchmark_status="COMPLETE" if selected else "RUNNING", selected_odl=selected,
        hf_mib_s=hf_rate, odl_mib_s=odl_rate, combined_mib_s=hf_rate+odl_rate,
        training_index_missing=completeness["training_index_missing"],
        normal_shards_completed=sum(row.get("extraction_status")=="complete" for row in state["shards"]),
        formal_training_started=False, remaining_download_bytes=remaining,
        remaining_pure_download_seconds=max(remaining["HF_EXISTING_PARTIAL_ONLY"]/hf_reference/MIB, (remaining["ODL"]+remaining["ODL_FORENSIC"])/odl_reference/MIB),
        eta_caveat="Source partition maximum, not remaining total divided by combined speed. Excludes checksum, extraction, NFS latency, JPEG rescue, final decode and smoke.", **activity)
    recovery.atomic_json(LIVE, live)
    return stamp, hf_bytes, odl_bytes


def run():
    import sa1b_wallclock_recovery as wallclock
    FOLDER.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in (FOLDER / "supervisor.lock", recovery.EVIDENCE / "sa1b-download.lock"):
        descriptor = path.open("a")
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(descriptor)
    os.environ.setdefault("HF_HOME", str(recovery.ASSETS / "hf_cache"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    state = recovery.load(recovery.STATE)
    for position, row in enumerate(state["shards"]):
        installed = recovery.load(recovery.EVIDENCE / "sa1b-shards" / (row["filename"]+".json"), {})
        if (installed.get("training_images_installed") is True and installed.get("extraction_status") == "complete"
            and installed.get("observed_md5") == row["expected_checklist_checksum"]
            and installed.get("observed_sha256") == row["expected_lfs_sha256"]):
            state["shards"][position] = installed
    plan = recovery.load(PLAN)
    recovery.configure_client(state["endpoint"])
    recovery.disk_check(sum(row["expected_size_bytes"] for row in state["shards"]))
    families = recovery.requirements()
    ownership = {}
    for row in state["shards"]:
        if row.get("extraction_status") == "complete":
            if recovery.digest(Path(row["image_inventory"])) != row["image_inventory_sha256"]:
                raise ValueError("Frozen image inventory identity mismatch")
            for line in Path(row["image_inventory"]).open():
                name = json.loads(line)["basename"]
                if name in ownership:
                    raise ValueError("Duplicate original-image provenance")
                ownership[name] = row["filename"]
    queues = {source: [row["filename"] for row in state["shards"] if plan["owners"][row["filename"]] == source
                      and row.get("extraction_status") != "complete" and row.get("download_status") != recovery.QUARANTINED]
        for source in ("HF_EXISTING_PARTIAL_ONLY", "ODL")}
    external_installs = copy.deepcopy(plan.get("external_installs", []))
    for entry in external_installs:
        for queue in queues.values():
            if entry["filename"] in queue:
                queue.remove(entry["filename"])
    for filename in list(queues["HF_EXISTING_PARTIAL_ONLY"]):
        row = next(row for row in state["shards"] if row["filename"] == filename)
        if row.get("download_status") == "verified" and Path(row["archive"]).is_file():
            queues["HF_EXISTING_PARTIAL_ONLY"].remove(filename)
    pending_ingestion = [(row["filename"], dict(source="HF")) for row in state["shards"]
        if row.get("download_status") == "verified" and row.get("extraction_status") != "complete"
        and plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY"]
    forensic_queue = list(plan.get("forensic_downloads", []))
    local_rescue = wallclock.start_rescue("sa_000017.tar", Path(state["shards"][17]["archive"]), "local") if plan.get("try_local_17_first") else None
    local_checked = local_rescue is None
    focused_rescues = {}
    for filename in list(queues["ODL"]):
        row = next(row for row in state["shards"] if row["filename"] == filename)
        archive = source_path(filename)
        if odl_ready(row):
            pending_ingestion.append((filename, dict(source="ODL", archive=str(archive))))
            queues["ODL"].remove(filename)
    failures = []
    download_futures = {}
    rescue_jobs = {}
    ingestion_futures = {}
    last_audit = 0
    previous = (time.monotonic(), sum(max(plan["initial_hf_bytes"][row["filename"]], payload_bytes(row)) for row in state["shards"] if plan["owners"][row["filename"]] == "HF_EXISTING_PARTIAL_ONLY"),
                sum(counters(path.parent).get("bytes", 0) for path in FOLDER.glob("odl-workers/*/*/worker-counters.json")))
    publication = None
    ingestion_limit = min(3, plan.get("ingestion_concurrency", 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan.get("hf_concurrency", 4)) as hf_pool, concurrent.futures.ThreadPoolExecutor(max_workers=3) as odl_pool, concurrent.futures.ThreadPoolExecutor(max_workers=ingestion_limit) as ingester, concurrent.futures.ThreadPoolExecutor(max_workers=1) as tester, concurrent.futures.ThreadPoolExecutor(max_workers=1) as monitor:
        benchmark_future = tester.submit(benchmark, copy.deepcopy(state), plan)
        selected = None
        while True:
            if benchmark_future.done() and selected is None:
                selected = benchmark_future.result()["selected"]
                state.update(odl_download_concurrency=selected["concurrency"], source_assignment_plan=str(PLAN))
            if local_rescue is not None and local_rescue.poll() is not None and not local_checked:
                proof = recovery.load(recovery.EVIDENCE / "sa1b-jpeg-rescue/sa_000017/result.json", {})
                if proof.get("missing_required_image_ids"):
                    forensic_queue.insert(0, "sa_000017.tar")
                local_checked = True
            for source, pool, limit in (("HF_EXISTING_PARTIAL_ONLY", hf_pool, plan.get("hf_concurrency", 4)), ("ODL", odl_pool, min(3, selected["concurrency"]) if selected else 0)):
                active = sum(value[0] == source or source == "ODL" and value[0] == "ODL_FORENSIC" for value in download_futures.values())
                while (queues[source] or source == "ODL" and forensic_queue) and active < limit:
                    forensic = source == "ODL" and bool(forensic_queue)
                    filename = forensic_queue.pop(0) if forensic else queues[source].pop(0)
                    row = next(row for row in state["shards"] if row["filename"] == filename)
                    if not forensic:
                        row.update(download_status="downloading", assigned_source=source)
                    job = pool.submit(wallclock.forensic_job, copy.deepcopy(row)) if forensic else pool.submit(hf_job, copy.deepcopy(state), copy.deepcopy(row), plan) if source == "HF_EXISTING_PARTIAL_ONLY" else pool.submit(odl_job, copy.deepcopy(row), plan)
                    download_futures[job] = ("ODL_FORENSIC" if forensic else source, filename)
                    active += 1
            for job in list(download_futures):
                if job.done():
                    source, filename = download_futures.pop(job)
                    try:
                        result = job.result()
                        if source == "HF_EXISTING_PARTIAL_ONLY":
                            state["shards"][int(filename[3:9])] = result
                        if source == "ODL_FORENSIC":
                            focused_rescues[filename] = wallclock.start_rescue(filename, Path(result["archive"]), "odl")
                        else:
                            pending_ingestion.append((filename, result if source == "ODL" else dict(source="HF")))
                    except Exception as error:
                        failures.append(dict(filename=filename, phase="download", error_type=type(error).__name__, error=str(error)))
            for entry in list(external_installs):
                try:
                    installed = adopt_external_install(entry, ownership)
                    if installed is not None:
                        state["shards"][int(entry["filename"][3:9])] = installed
                        external_installs.remove(entry)
                except Exception as error:
                    failures.append(dict(filename=entry["filename"], phase="external-ingestion", error_type=type(error).__name__, error=str(error)))
                    external_installs.remove(entry)
            for job in list(ingestion_futures):
                if not job.done():
                    continue
                filename = ingestion_futures.pop(job)
                try:
                    state["shards"][int(filename[3:9])] = job.result()
                    print(filename, "ingested", recovery.now(), flush=True)
                except Exception as error:
                    failures.append(dict(filename=filename, phase="ingestion", error_type=type(error).__name__, error=str(error)))
            while pending_ingestion and len(ingestion_futures)+len(external_installs) < ingestion_limit:
                filename, downloaded = pending_ingestion.pop(0)
                row = state["shards"][int(filename[3:9])]
                row["extraction_status"] = "extracting"
                job = ingester.submit(ingest, copy.deepcopy(row), downloaded, ownership)
                ingestion_futures[job] = filename
            for position, row in enumerate(state["shards"]):
                if row.get("download_status") != recovery.QUARANTINED:
                    continue
                rescue = recovery.load(recovery.EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}" / "result.json", {})
                if rescue:
                    state["shards"][position]["required_jpeg_rescue"] = rescue
                    state["shards"][position]["training_images_status"] = rescue["status"]
                if position not in BAD and position not in rescue_jobs and not rescue and Path(row["archive"]).is_file():
                    folder = recovery.EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder / "worker.log").open("a") as output:
                        rescue_jobs[position] = subprocess.Popen([str(recovery.ROOT / ".venv/bin/python"), "-u",
                            str(recovery.RECOVERY / "sa1b_jpeg_rescue.py"), "run", "--shard", str(position)],
                            cwd=recovery.ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)
            if publication is not None and publication.done():
                try:
                    previous = publication.result()
                except Exception as error:
                    print("progress publication failed; download scheduler continues", type(error).__name__, flush=True)
                publication = None
            if publication is None and time.monotonic()-last_audit > 60:
                activity = dict(snapshot_utc=recovery.now(),
                    active_downloads=[dict(source=source, filename=filename) for source, filename in download_futures.values()],
                    ingestion=",".join(list(ingestion_futures.values())+[entry["filename"] for entry in external_installs]) or None,
                    active_ingestions=list(ingestion_futures.values())+[entry["filename"] for entry in external_installs],
                    pending_ingestion=len(pending_ingestion), ingestion_concurrency=ingestion_limit,
                    queues=copy.deepcopy(queues), forensic_queue=forensic_queue.copy(),
                    focused_rescue_running=[name for name, process in focused_rescues.items() if process.poll() is None],
                    local_17_rescue_running=bool(local_rescue and local_rescue.poll() is None), failures=copy.deepcopy(failures))
                publication = monitor.submit(publish_progress, families, copy.deepcopy(state), ownership.copy(), plan, selected, activity, previous)
                last_audit = time.monotonic()
            if selected and not download_futures and not pending_ingestion and not ingestion_futures and not external_installs and not any(queues.values()) and not forensic_queue and local_checked and all(process.poll() is not None for process in list(rescue_jobs.values())+list(focused_rescues.values())):
                break
            time.sleep(2)
    completeness = quick_completeness(families, state, ownership)
    recovery.atomic_json(recovery.STATE, state)
    recovery.report(state, completeness)
    if completeness["training_index_missing"] == 0 and not failures:
        live = recovery.load(LIVE, {})
        live.update(checked_utc=recovery.now(), training_index_missing=0, active_downloads=[],
            active_ingestions=[], ingestion=None, pending_ingestion=0,
            phase="FINAL_PHYSICAL_EXISTENCE_AND_REQUIRED_IMAGE_DECODE_AUDIT", formal_training_started=False)
        recovery.atomic_json(LIVE, live)
        subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "sa1b_recovery.py"), "audit"], check=True)
        completeness = recovery.load(recovery.RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
        if completeness.get("passed"):
            subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "audit.py"), "training"], check=True)
            policy = recovery.load(recovery.EVIDENCE / "recovery-operation-policy.json", {})
            if policy.get("smoke_authorized"):
                subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "validate.py"), "smoke"], check=True)
    final = recovery.load(LIVE, {})
    ready = bool(completeness.get("passed") and recovery.load(recovery.EVIDENCE / "smoke-audit.json", {}).get("passed")
                 and recovery.load(recovery.EVIDENCE / "export-audit.json", {}).get("passed"))
    final.update(supervisor_running=False, finished_utc=recovery.now(), training_index_missing=completeness["training_index_missing"], failures=failures,
                 resume_status="READY_TO_START_S02_FULL" if ready else "NOT_READY_TO_START_S02_FULL", formal_training_started=False)
    recovery.atomic_json(LIVE, final)
    print(final["resume_status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    args = parser.parse_args()
    if args.command == "plan":
        if PLAN.exists():
            raise RuntimeError("Do not overwrite an established source ownership plan")
        recovery.atomic_json(PLAN, plan_sources(recovery.load(recovery.STATE)))
    else:
        run()
