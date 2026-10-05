"""Recover original COCO/LLaVA archives only; never invoke training or smoke."""

import argparse
from contextlib import nullcontext
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile

from audit import ASSETS, EVIDENCE, RECOVERY, ROOT, digest, load
from sa1b_recovery import RECORDS_SHA, atomic_json, configure_client, now, requirements


STATE = Path(os.environ.get("SAID_HF_STATE", str(RECOVERY / "HF_TRAINING_ASSETS.json")))
DISCOVERY = EVIDENCE / "hf-training-discovery.json"
RATE_LOG = EVIDENCE / "hf-training-throughput.jsonl"
EXPECTED_COUNTS = {"coco": 118287, "llava": 558128}
SAMPLE_SECONDS = 30
SLOW_SECONDS = 600
MINIMUM_RATE = 1024**2


def worker_path(asset):
    return EVIDENCE / f"hf-training-{asset}-worker.json"


def install_progress_callback(file_download, on_payload):
    original_tqdm = file_download.tqdm

    class PayloadProgress(original_tqdm):
        def update(self, amount=1):
            on_payload(amount)
            return super().update(amount)

        def close(self):
            on_payload(0, force=True)
            return super().close()

    def progress_context(*, log_level, _tqdm_bar=None, **values):
        if _tqdm_bar is not None:
            return nullcontext(_tqdm_bar)
        return PayloadProgress(**values, disable=True)

    file_download._get_progress_bar_context = progress_context


def archive_directory(row):
    return ASSETS / "downloads/hf_training" / row["asset"] / row["repo"].replace("/", "--") / row["revision"]


def extraction_root(asset):
    return ASSETS / "training/ShareGPT4V" / ("coco" if asset == "coco" else "llava/llava_pretrain")


def test_zip(path, asset, expected_count):
    with zipfile.ZipFile(path) as archive:
        members = audit_zip_structure(archive, asset, expected_count)
        failed_member = archive.testzip()
        if failed_member is not None:
            raise ValueError("ZIP CRC check failed: " + failed_member)
    return dict(method="Python standard library ZipFile.testzip", bad_member=None,
                checked_images=len(members), passed=True)


def disk_check(rows):
    total = sum(row["expected_size_bytes"] for row in rows)
    required = total * 3 + 64 * 1024**3
    free = shutil.disk_usage(ROOT).free
    output = subprocess.check_output(["df", "-h", str(ROOT)], text=True)
    record = dict(checked_utc=now(), df_h=output, free_bytes=free,
                  planned_archive_bytes=total, minimum_free_bytes=required,
                  reservation="3x COCO/LLaVA archive sizes +64GiB; SA1B retains its independent disk budget",
                  sufficient=free >= required)
    atomic_json(EVIDENCE / "hf-training-disk-audit.json", record)
    if not record["sufficient"]:
        raise RuntimeError("Insufficient space; preserving all partials and not starting downloads")


def installed_image_path(filename, asset):
    relative = PurePosixPath(filename)
    if asset == "llava" and relative.parts[0] != "images":
        return PurePosixPath("images") / relative
    return relative


def audit_zip_structure(archive, asset, expected_count):
    members = []
    seen = set()
    layouts = set()
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
            raise ValueError("Unsafe ZIP path: " + member.filename)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("ZIP symlink: " + member.filename)
        if member.is_dir():
            continue
        if asset == "coco":
            valid_layout = len(relative.parts) == 2 and relative.parts[0] == "train2017"
        else:
            image_parts = relative.parts[1:] if relative.parts[0] == "images" else relative.parts
            valid_layout = (len(image_parts) == 2 and len(image_parts[0]) == 5 and image_parts[0].isdigit()
                            and len(relative.stem) == 9 and relative.stem.isdigit())
            layouts.add("wrapped" if relative.parts[0] == "images" else "bare")
        if not valid_layout or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Unexpected original image layout: " + member.filename)
        installed = str(installed_image_path(member.filename, asset))
        if installed in seen or member.file_size <= 0:
            raise ValueError("Duplicate/empty ZIP image: " + member.filename)
        seen.add(installed)
        members.append(member)
    if len(layouts) > 1:
        raise ValueError("Mixed original image ZIP layouts")
    if len(members) != expected_count:
        raise ValueError(f"Archive image count differs: {len(members)} != {expected_count}")
    return members


def extract_images(path, asset, expected_count, publish):
    target_root = extraction_root(asset)
    target_root.mkdir(parents=True, exist_ok=True)
    resolved_root = target_root.resolve()
    verified_parents = set()
    with zipfile.ZipFile(path) as archive:
        members = audit_zip_structure(archive, asset, expected_count)
        publish(phase="extracting", archive_images=len(members), extracted_images=0)
        for position, member in enumerate(members, 1):
            target = target_root / installed_image_path(member.filename, asset)
            if target.parent not in verified_parents:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(resolved_root):
                    raise ValueError("ZIP destination escapes original image root")
                verified_parents.add(target.parent)
            temporary = target.with_name(target.name + f".hf-recovery-{os.getpid()}.tmp")
            try:
                with archive.open(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024**2)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError("Extracted image size mismatch")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            if position % 1000 == 0:
                publish(extracted_images=position)
        return len(members)


def publish_family_completeness(asset, progress):
    count = EXPECTED_COUNTS[asset]
    if not (progress.get("completed") and progress.get("sha256_passed") and progress.get("zip_integrity_passed")
            and progress.get("index_missing") == 0 and progress.get("recovered_required_images") == count
            and progress.get("physical_image_count") == count):
        raise ValueError("Family recovery is not independently verified complete")
    record = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    if record.get("training_index_sha256") != RECORDS_SHA or record.get("training_records") != 1245901:
        raise ValueError("Frozen training index identity missing from completeness report")
    previous_checked = record["checked_utc"]
    checked = now()
    record.setdefault("family_checked_utc", {family: previous_checked for family in record["families"]})
    record["family_checked_utc"][asset] = checked
    record["families"][asset] = dict(required_paths=count, recovered_paths=count, missing_paths=0,
        duplicate_training_index_paths=0, missing_examples=[],
        audit_evidence=progress.get("independent_audit_evidence", str(worker_path(asset))))
    missing = sum(row["missing_paths"] for row in record["families"].values())
    record.update(checked_utc=checked, filtered_records=1245901, training_index_missing=missing,
        missing_training_paths=missing,
        recovered_training_paths=sum(row["recovered_paths"] for row in record["families"].values()),
        decode_audit=dict(executed=False, passed=False), passed=False,
        update_scope="Independent family completion; other families retain their recorded snapshot times")
    atomic_json(EVIDENCE / "training-image-completeness-audit.json", record)
    atomic_json(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", record)
    return record


def await_family(asset):
    while True:
        progress = load(worker_path(asset), {})
        if progress.get("completed"):
            record = publish_family_completeness(asset, progress)
            print(json.dumps(dict(asset=asset, recovered=EXPECTED_COUNTS[asset],
                                  training_index_missing=record["training_index_missing"])), flush=True)
            return
        if progress.get("phase") == "failed":
            raise RuntimeError("Family worker failed: " + progress.get("error", "unknown"))
        if progress.get("pid"):
            os.kill(progress["pid"], 0)
        time.sleep(SAMPLE_SECONDS)


def worker(asset):
    from huggingface_hub import file_download, hf_hub_download
    from huggingface_hub.utils._runtime import is_xet_available

    row = load(STATE)["assets"][asset]
    configure_client(row["endpoint"])
    directory = archive_directory(row)
    directory.mkdir(parents=True, exist_ok=True)
    archive = Path(row.get("archive_override", directory / row["filename"]))
    record = dict(asset=asset, repo=row["repo"], revision=row["revision"], pid=os.getpid(),
                  archive=str(archive), received_bytes=0, phase="downloading", started_utc=now(),
                  xet_enabled=is_xet_available(), xet_metadata_available=row["xet_metadata_available"],
                  throughput_metric="SDK progress callback payload bytes; never sparse file length")
    if not row["xet_metadata_available"]:
        record["resumed_payload_bytes"] = sum(path.stat().st_size for path in
            (directory / ".cache/huggingface/download").glob("*.incomplete"))
    last_publish = 0.0

    def publish(**values):
        nonlocal last_publish
        record.update(values, updated_utc=now())
        atomic_json(worker_path(asset), record)
        last_publish = time.monotonic()

    def on_payload(amount, force=False):
        record["received_bytes"] += amount
        if force or time.monotonic() - last_publish >= 2:
            publish()

    install_progress_callback(file_download, on_payload)
    publish()
    try:
        for attempt in range(1, 4):
            try:
                publish(phase="sha256_check" if archive.is_file() else "downloading", attempt=attempt)
                if not archive.is_file():
                    hf_hub_download(row["repo"], row["filename"], repo_type="dataset",
                                    revision=row["revision"], local_dir=directory,
                                    endpoint=row["endpoint"], token=False)
                publish(phase="sha256_check")
                actual_size = archive.stat().st_size
                actual_sha = digest(archive)
                if actual_size != row["expected_size_bytes"] or actual_sha != row["expected_sha256"]:
                    raise ValueError("Downloaded archive size/SHA256 mismatch")
                publish(phase="zip_integrity_check", actual_size_bytes=actual_size,
                        actual_sha256=actual_sha, sha256_passed=True)
                integrity = test_zip(archive, asset, EXPECTED_COUNTS[asset])
                atomic_json(EVIDENCE / f"hf-training-{asset}-zipfile-test.json", integrity)
                publish(zip_integrity_passed=True, zip_integrity=integrity)
                break
            except (ValueError, zipfile.BadZipFile):
                if archive.exists():
                    rejected = archive.with_name(archive.name + f".rejected-{time.time_ns()}")
                    archive.replace(rejected)
                    publish(rejected_archive=str(rejected))
                if attempt == 3:
                    raise
            except Exception as error:
                if record["phase"] != "downloading":
                    raise
                publish(last_download_error=str(error))
                if attempt == 3:
                    raise
                time.sleep(20)
        extracted = extract_images(archive, asset, EXPECTED_COUNTS[asset], publish)
        publish(phase="index_check", extracted_images=extracted)
        required = requirements()[asset]
        family_root = ASSETS / "training/ShareGPT4V"
        directories = {}
        missing = []
        for relative in sorted(required):
            parent = str(PurePosixPath(relative).parent)
            if parent not in directories:
                directories[parent] = {entry.name for entry in os.scandir(family_root / parent) if entry.is_file()}
            if PurePosixPath(relative).name not in directories[parent]:
                missing.append(relative)
        physical_count = sum(1 for directory, subdirectories, filenames in os.walk(target_root_images(asset))
                             for filename in filenames if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"})
        if missing or len(required) != EXPECTED_COUNTS[asset] or physical_count != EXPECTED_COUNTS[asset]:
            raise ValueError(f"Frozen index/image count failure: missing={len(missing)}, actual={physical_count}")
        publish(phase="complete", completed=True, finished_utc=now(),
                required_images=len(required), recovered_required_images=len(required),
                index_missing=0, physical_image_count=physical_count,
                original_bytes_preserved=True)
        publish_family_completeness(asset, record)
    except Exception as error:
        publish(failed_stage=record["phase"], phase="failed", completed=False,
                error_type=type(error).__name__, error=str(error))
        raise


def target_root_images(asset):
    return extraction_root(asset) / ("train2017" if asset == "coco" else "images")


def source_search(asset):
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    row = load(STATE)["assets"][asset]
    configure_client(row["endpoint"])
    api = HfApi(endpoint=row["endpoint"], token=False)
    searches = (["coco-2017", "coco2017"] if asset == "coco" else
                ["LLaVA-Pretrain", "LCS-558K", "llava_pretrain", "laion_cc_sbu"])
    report = dict(asset=asset, expected_sha256=row["expected_sha256"], started_utc=now(),
                  candidates=[], queries=[], selected=None,
                  acceptance="Exact original ZIP filename, size and LFS SHA256 identity only; no repacks")
    report_path = EVIDENCE / f"hf-training-{asset}-alternative-sources.json"
    excluded = {source["repo"] for source in row["source_history"]}
    candidates = []
    atomic_json(report_path, report)
    for search in searches:
        query = dict(search=search, limit=15, attempts=[])
        for attempt in range(2):
            try:
                datasets = list(api.list_datasets(search=search, limit=15))
                for dataset in datasets:
                    if dataset.id not in candidates and dataset.id not in excluded:
                        candidates.append(dataset.id)
                query.update(passed=True, results=len(datasets))
                break
            except Exception as error:
                query["attempts"].append(str(error))
                query["passed"] = False
                time.sleep(1)
        report["queries"].append(query)
        atomic_json(report_path, report)
    for repo in candidates[:50]:
        candidate = dict(repo=repo)
        try:
            info = api.dataset_info(repo)
            entries = api.get_paths_info(repo, [row["filename"]], repo_type="dataset", revision=info.sha)
            if not entries:
                raise ValueError("Original archive filename absent")
            entry = entries[0]
            candidate.update(revision=info.sha, size_bytes=entry.size,
                             sha256=entry.lfs.sha256 if entry.lfs else None)
            if entry.size == row["expected_size_bytes"] and candidate["sha256"] == row["expected_sha256"]:
                metadata = get_hf_file_metadata(hf_hub_url(repo, row["filename"], repo_type="dataset",
                    revision=info.sha, endpoint=row["endpoint"]), token=False)
                candidate["xet_metadata_available"] = metadata.xet_file_data is not None
                report["selected"] = candidate
        except Exception as error:
            candidate["error"] = str(error)
        report["candidates"].append(candidate)
        atomic_json(report_path, report)
        if report["selected"]:
            break
    report["finished_utc"] = now()
    report["status"] = ("IDENTICAL_ARCHIVE_FOUND" if report["selected"] else
                        "SEARCH_INCOMPLETE" if any(not query["passed"] for query in report["queries"]) else
                        "NO_IDENTICAL_ARCHIVE_IN_CHECKED_CANDIDATES")
    atomic_json(report_path, report)


def slow_source(low_since, rate, timestamp, phase):
    if phase != "downloading":
        return None, False
    if rate >= MINIMUM_RATE:
        return timestamp, False
    return low_since, timestamp - low_since >= SLOW_SECONDS


def window_rate(events, asset, repo, revision, end_timestamp):
    received = 0.0
    for event in events:
        if (event["asset"], event["repo"], event["revision"]) != (asset, repo, revision):
            continue
        end = datetime.datetime.fromisoformat(event["checked_utc"]).timestamp()
        duration = event["interval_seconds"]
        overlap = max(0, min(end, end_timestamp) - max(end-duration, end_timestamp-SLOW_SECONDS))
        if duration > 0:
            received += event["interval_payload_bytes"] * overlap / duration
    return received / SLOW_SECONDS


def stop_owned_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def render(state):
    completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
    lines = ["# Original COCO/LLaVA HF recovery", "", f"Updated UTC: {now()}", "",
             "No training/smoke commands in this supervisor. The SA1B supervisor is untouched.",
             "All old curl partials are retained; SDK resumes use separate revision-pinned directories.",
             "hf_xet is installed/enabled. Mirror HEAD currently supplies no Xet metadata; actual transfer uses SDK HTTP fallback.",
             "Rate metric: SDK payload callback deltas / wall time, logged every 30 seconds.",
             "Sources stop after 600 consecutive low-rate seconds or a 600-second payload average below 1MiB/s, including stalls.",
             "Alternative ZIPs require identical SHA256/size. The 600-second average prevents brief buffered bursts evading the slow-source limit.", ""]
    for asset, row in state["assets"].items():
        progress = row.get("worker", {})
        cached = progress.get("resumed_payload_bytes", 0) + progress.get("received_bytes", 0)
        lines.extend([f"## {asset.upper()}", f"Repo: `{row['repo']}`",
                      f"Pinned revision: `{row['revision']}`", f"Filename: `{row['filename']}`",
                      f"Expected SHA256: `{row['expected_sha256']}`",
                      f"Expected archive bytes: {row['expected_size_bytes']}",
                      f"Status: {row.get('status')}; phase: {progress.get('phase', 'pending')}",
                      f"SDK payload bytes (including HTTP resume): {cached}/{row['expected_size_bytes']}",
                      f"Latest payload throughput MiB/s: {row.get('rate_mib_s', 0):.3f}",
                      f"Verified/extracted required images: {progress.get('recovered_required_images', 0)}/{EXPECTED_COUNTS[asset]}", ""])
    lines.extend(["## Frozen training index", f"Snapshot UTC: {completeness.get('checked_utc')}",
                  f"Records: {completeness.get('training_records')}",
                  f"Missing paths: {completeness.get('missing_training_paths')}",
                  "Live rates: `evidence/hf-training-throughput.jsonl`",
                  "Worker logs: `evidence/hf-training-coco.log`, `evidence/hf-training-llava.log`"])
    report_name = "HF_TRAINING_RECOVERY.md" if STATE.name == "HF_TRAINING_ASSETS.json" else "HF_LLAVA_RECOVERY.md"
    (RECOVERY / report_name).write_text("\n".join(lines) + "\n")


def supervise():
    def handle_stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    if STATE.exists():
        state = load(STATE)
    else:
        rows = load(DISCOVERY)["assets"]
        disk_check(rows)
        partials = [ASSETS / "downloads/train2017.zip.part", ASSETS / "downloads/llava_pretrain_images.zip.part"]
        state = dict(started_utc=now(), supervisor_pid=os.getpid(), old_partials_preserved=[
            dict(path=str(path), size_bytes=path.stat().st_size) for path in partials if path.exists()],
            assets={row["asset"]: dict(row, source_history=[dict(row)], status="pending") for row in rows})
    state["supervisor_pid"] = os.getpid()
    atomic_json(STATE, state)
    disk_check(list(state["assets"].values()))
    processes = {}
    searches = {}
    monitoring = {}
    logs = []
    audit_process = None
    audit_pending = False
    selected_assets = os.environ.get("SAID_RECOVER_ASSETS", "coco,llava").split(",")

    def launch(asset):
        row = state["assets"][asset]
        row["source_history"][-1].setdefault("started_utc", row.get("started_utc", now()))
        worker_path(asset).unlink(missing_ok=True)
        output = (EVIDENCE / f"hf-training-{asset}.log").open("a")
        logs.append(output)
        processes[asset] = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--asset", asset],
                                            stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
        row.update(status="running", worker_pid=processes[asset].pid, started_utc=now(),
                   previous_received_bytes=0)
        timestamp = time.monotonic()
        monitoring[asset] = dict(last_time=timestamp, low_since=timestamp, last_bytes=0)

    def schedule_audit():
        nonlocal audit_process, audit_pending
        if audit_process is not None and audit_process.poll() is None:
            audit_pending = True
            return
        output = (EVIDENCE / "hf-training-completeness-audit.log").open("a")
        logs.append(output)
        audit_process = subprocess.Popen([ROOT / ".venv/bin/python", RECOVERY / "sa1b_recovery.py", "audit"],
                                         stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
        audit_pending = False

    for asset in selected_assets:
        row = state["assets"][asset]
        progress = row.get("worker", {})
        if (row.get("status") == "complete" and progress.get("completed") and progress.get("sha256_passed")
                and progress.get("zip_integrity_passed") and progress.get("index_missing") == 0
                and progress.get("recovered_required_images") == EXPECTED_COUNTS[asset]):
            continue
        if row.get("live_asset_state"):
            external = load(Path(row["live_asset_state"]), {})
            external_pid = external.get("supervisor_pid")
            if external_pid:
                try:
                    os.kill(external_pid, 0)
                    row["status"] = "externally_running"
                    continue
                except ProcessLookupError:
                    pass
        launch(asset)
    atomic_json(STATE, state)
    render(state)
    try:
        while processes or searches:
            time.sleep(SAMPLE_SECONDS)
            for asset, process in list(processes.items()):
                row = state["assets"][asset]
                progress = load(worker_path(asset), {})
                row["worker"] = progress
                timestamp = time.monotonic()
                previous = monitoring[asset]
                received = progress.get("received_bytes", previous["last_bytes"])
                elapsed = timestamp - previous["last_time"]
                rate = max(0, received - previous["last_bytes"]) / elapsed
                phase = progress.get("phase", "downloading")
                low_since, stop = slow_source(previous["low_since"], rate, timestamp, phase)
                previous.update(last_time=timestamp, last_bytes=received,
                                low_since=timestamp if low_since is None else low_since)
                row["rate_mib_s"] = rate / 1024**2
                event = dict(checked_utc=now(), asset=asset, repo=row["repo"], revision=row["revision"],
                             phase=phase, received_bytes=received, interval_seconds=elapsed,
                             interval_payload_bytes=max(0, received - row.get("previous_received_bytes", 0)),
                             mib_s=row["rate_mib_s"], low_rate_seconds=timestamp-previous["low_since"])
                row["previous_received_bytes"] = received
                with RATE_LOG.open("a") as output:
                    output.write(json.dumps(event) + "\n")
                end_timestamp = datetime.datetime.fromisoformat(event["checked_utc"]).timestamp()
                source_started = datetime.datetime.fromisoformat(row["source_history"][-1]["started_utc"]).timestamp()
                if phase == "downloading" and end_timestamp-source_started >= SLOW_SECONDS:
                    history = [json.loads(line) for line in RATE_LOG.read_text().splitlines()]
                    rolling_rate = window_rate(history, asset, row["repo"], row["revision"], end_timestamp)
                    row["rolling_600s_mib_s"] = rolling_rate / 1024**2
                    event["rolling_600s_mib_s"] = row["rolling_600s_mib_s"]
                    stop = stop or rolling_rate < MINIMUM_RATE
                print(json.dumps(event), flush=True)
                if stop:
                    stop_owned_process(process)
                    row.update(status="STOPPED_SLOW_SOURCE", stopped_utc=now(),
                               stop_reason="600-second payload average or consecutive rates below 1MiB/s; SDK incomplete preserved")
                    row["source_history"][-1].update(status=row["status"], stopped_utc=now())
                elif process.poll() is not None:
                    row["status"] = "complete" if process.returncode == 0 and progress.get("completed") else "failed"
                    row["source_history"][-1]["status"] = row["status"]
                else:
                    continue
                del processes[asset]
                completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
                if row["status"] == "complete" and completeness.get("training_index_missing") == 0:
                    schedule_audit()
                atomic_json(STATE, state)
                source_failed = row["status"] == "failed" and progress.get("failed_stage") == "downloading"
                if asset == "llava" and row["repo"] == "liuhaotian/LLaVA-Pretrain" and (row["status"] == "STOPPED_SLOW_SOURCE" or source_failed):
                    output = (EVIDENCE / f"hf-training-{asset}-source-search.log").open("a")
                    logs.append(output)
                    searches[asset] = (subprocess.Popen([sys.executable, "-u", str(RECOVERY / "llava_copy_audit.py")],
                        stdout=output, stderr=subprocess.STDOUT, cwd=ROOT), time.monotonic())
            for asset, (process, started) in list(searches.items()):
                if process.poll() is None and time.monotonic() - started < 900:
                    continue
                if process.poll() is None:
                    stop_owned_process(process)
                result = load(EVIDENCE / "llava-copy-identity-audit.json", {})
                selected = result.get("duplicate") if process.returncode == 0 and result.get("passed") else None
                if selected:
                    row = state["assets"][asset]
                    row.update(repo=selected["repo"], revision=selected["revision"],
                               endpoint=selected["endpoint"], filename=selected["filename"],
                               xet_metadata_available=selected["xet_metadata_available"],
                               duplicate_identity_audit=str(EVIDENCE / "llava-copy-identity-audit.json"))
                    row["source_history"].append(dict(repo=row["repo"], revision=row["revision"],
                        expected_sha256=row["expected_sha256"], status="pending", started_utc=now()))
                    atomic_json(STATE, state)
                    launch(asset)
                else:
                    state["assets"][asset]["alternative_source_status"] = "332F_IDENTITY_AUDIT_NOT_PASSED"
                del searches[asset]
            if audit_pending and audit_process.poll() is not None:
                schedule_audit()
            atomic_json(STATE, state)
            render(state)
        if audit_process:
            audit_process.wait()
        if audit_pending:
            schedule_audit()
            audit_process.wait()
        state.update(finished_utc=now(), completed=all(row["status"] == "complete" for row in state["assets"].values()))
        atomic_json(STATE, state)
        render(state)
        subprocess.run([ROOT / ".venv/bin/python", RECOVERY / "report.py"], check=True)
    finally:
        for process in processes.values():
            stop_owned_process(process)
        for process, started in searches.values():
            stop_owned_process(process)
        for output in logs:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["supervise", "worker", "search", "await-family"])
    parser.add_argument("--asset", choices=list(EXPECTED_COUNTS))
    args = parser.parse_args()
    if args.command == "supervise":
        supervise()
    elif args.command == "worker":
        worker(args.asset)
    elif args.command == "await-family":
        await_family(args.asset)
    else:
        source_search(args.asset)


if __name__ == "__main__":
    main()
