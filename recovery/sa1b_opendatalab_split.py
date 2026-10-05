"""Stage independently verified ODL archives for existing HF recovery without restarting it."""

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

import sa1b_recovery as recovery
import sa1b_opendatalab_probe as probe
from sa1b_throughput_benchmark import payload_bytes


FOLDER = recovery.EVIDENCE / "sa1b-opendatalab/split-recovery"
STATE = FOLDER / "state.json"
LOCAL_ODL = Path("/root/.cache/said-recovery/odl-original-shards")


def eligible(row, rows):
    active = [int(entry["filename"][3:9]) for entry in rows if entry.get("download_status") == "downloading"]
    return (int(row["filename"][3:9]) >= 38 and row.get("download_status") == "pending"
            and row.get("extraction_status") != "complete" and payload_bytes(row) == 0
            and bool(active) and int(row["filename"][3:9]) - max(active) >= 8)


def full_audit(path, row, output):
    from PIL import Image

    path = Path(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = recovery.verify_archive(path, row)
    recovery.atomic_json(output / "progress.json", dict(stage="CHECKSUM_PASSED_TAR_CHECK", checked_utc=recovery.now(), **result))
    with (output / "tar-members.txt").open("w") as listing, (output / "tar-stderr.txt").open("wb") as errors:
        process = subprocess.run(["tar", "-tf", str(path)], stdout=listing, stderr=errors)
    result.update(archive=str(path), tar_integrity_passed=process.returncode == 0,
                  tar_returncode=process.returncode, protocol_passed=False, archive_preserved=True)
    if process.returncode:
        recovery.atomic_json(output / "audit.json", result)
        return result
    recovery.atomic_json(output / "progress.json", dict(stage="TAR_PASSED_FULL_JPEG_DECODE", checked_utc=recovery.now(), **result))
    names = set()
    failures = []
    decoded = 0
    samples = []
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe original archive structure")
            if not member.isfile() or relative.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            basename = relative.name
            if not re.fullmatch(r"sa_\d+\.(jpg|jpeg)", basename) or basename in names:
                raise ValueError("Unexpected or duplicate original image basename")
            names.add(basename)
            data = archive.extractfile(member).read()
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    decoded += 1
                    if len(samples) < 16:
                        samples.append(dict(basename=basename, sha256=recovery.hashlib.sha256(data).hexdigest(),
                                            size=list(image.size), format=image.format))
            except Exception as error:
                failures.append(dict(basename=basename, error_type=type(error).__name__))
            if len(names) % 256 == 0:
                recovery.atomic_json(output / "progress.json", dict(stage="FULL_JPEG_DECODE", checked_utc=recovery.now(),
                    image_members=len(names), decoded_jpegs=decoded, decode_failures=len(failures)))
    requirements = recovery.requirements()["sam"]
    required_names = {Path(name).name for name in requirements}
    ids = sorted(int(name[3:].split(".")[0]) for name in names)
    required_in_range = {name for name in required_names if ids and ids[0] <= int(name[3:].split(".")[0]) <= ids[-1]}
    result.update(classification="BITWISE_EQUIVALENT_MIRROR", jpeg_count=len(names),
                  decoded_jpegs=decoded, decode_failures=failures, sample_byte_hashes=samples,
                  min_image_id=ids[0] if ids else None, max_image_id=ids[-1] if ids else None,
                  required_images_in_id_range=len(required_in_range),
                  required_images_found=len(required_in_range.intersection(names)),
                  required_missing_in_id_range=sorted(required_in_range.difference(names)),
                  training_index_sha256=recovery.RECORDS_SHA, image_transform="NONE",
                  protocol_passed=bool(names) and not failures and required_in_range.issubset(names))
    recovery.atomic_json(output / "audit.json", result)
    recovery.atomic_json(output / "progress.json", dict(stage="AUDIT_FINISHED", checked_utc=recovery.now(),
        passed=result["protocol_passed"], decoded_jpegs=decoded, decode_failures=len(failures)))
    return result


def handoff(source, row, revision, audit):
    if not audit.get("protocol_passed") or audit.get("observed_sha256") != row["expected_lfs_sha256"]:
        raise ValueError("Cannot hand off unverified original archive")
    from huggingface_hub._local_folder import write_download_metadata
    destination = Path(row["archive"])
    if payload_bytes(row) != 0:
        raise ValueError("HF already owns bytes; preserve both sources without overwriting")
    os.link(source, destination)
    write_download_metadata(destination.parent, row["repo_path"], revision, row["expected_lfs_sha256"])
    proof = dict(repository="OpenDataLab/SA-1B", dataset_id=6248, downloaded_source=str(source),
                 hf_local_path=str(destination), hf_revision=revision, filename=row["filename"],
                 checksum_audit=audit, handed_off_utc=recovery.now(),
                 mechanism="No-overwrite hardlink + official HF local metadata for byte-identical verified LFS object",
                 training_images_installed=False, native_ingestion="Existing HF supervisor will reverify/check tar/extract/audit at this queue position")
    recovery.atomic_json(destination.with_name(destination.name + ".opendatalab-provenance.json"), proof)
    return proof


def choose_download_target(filename, canonical_target, target=None):
    if target:
        return Path(target)
    raw = Path(canonical_target) / "OpenDataLab___SA-1B/raw"
    local = LOCAL_ODL / filename.removesuffix(".tar")
    local_raw = local / "OpenDataLab___SA-1B/raw"
    local_cache = local_raw / ".cache"
    if (raw / ".cache").is_symlink() and (raw / ".cache").resolve() == local_cache.resolve():
        return local
    if (raw / filename).exists() or (raw / ".cache").exists():
        return Path(canonical_target)
    local.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(local).free < 128 * 1024**3:
        return Path(canonical_target)
    local_cache.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    cache = raw / ".cache"
    if not cache.exists() and not cache.is_symlink():
        cache.symlink_to(local_cache, target_is_directory=True)
    return local


def publish_local_archive(path, destination, row, directory):
    proof = recovery.verify_archive(path, row)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or recovery.digest(destination) != proof["observed_sha256"]:
            raise ValueError("Canonical archive already exists and differs; refusing overwrite")
    else:
        destination.symlink_to(Path(path).resolve())
    proof.update(local_original_archive=str(path), canonical_archive=str(destination),
        mechanism="Official SDK payload/cache on local SSD; canonical symlink only after pinned size/MD5/SHA256 pass",
        image_destination_unchanged=str(recovery.SAM_ROOT), local_archive_must_be_retained=True)
    recovery.atomic_json(Path(directory) / "local-ssd-provenance.json", proof)


def download_worker(filename, directory, target=None):
    probe.FOLDER = Path(directory)
    probe.FILENAME = filename
    canonical_target = recovery.ASSETS / "downloads/sa1b-opendatalab" / ("split-" + filename.removesuffix(".tar"))
    probe.TARGET = choose_download_target(filename, canonical_target, target)
    probe.ARCHIVE = probe.TARGET / "OpenDataLab___SA-1B/raw" / filename
    row = next(row for row in recovery.load(recovery.STATE)["shards"] if row["filename"] == filename)
    if not target and probe.TARGET != canonical_target and probe.ARCHIVE.is_file() and probe.ARCHIVE.stat().st_size == row["expected_size_bytes"]:
        publish_local_archive(probe.ARCHIVE, canonical_target / "OpenDataLab___SA-1B/raw" / filename, row, directory)
        recovery.atomic_json(probe.FOLDER / "worker-result.json", dict(completed=True, existing_verified_archive_reused=True, finished_utc=recovery.now()))
        return
    probe.worker()
    if not target and probe.TARGET != canonical_target and recovery.load(probe.FOLDER / "worker-result.json", {}).get("completed"):
        publish_local_archive(probe.ARCHIVE, canonical_target / "OpenDataLab___SA-1B/raw" / filename,
            row, directory)


def split_run():
    FOLDER.mkdir(parents=True, exist_ok=True)
    accepted = recovery.load(recovery.EVIDENCE / "sa1b-opendatalab/authorized-probe/result.json")
    audit = recovery.load(recovery.EVIDENCE / "sa1b-opendatalab/authorized-probe/full-audit/audit.json")
    if not accepted.get("speed_gate") or accepted.get("combined_gain", 0) < 1.3 or not audit.get("protocol_passed"):
        raise ValueError("Authorized speed/protocol/combined-throughput gates must pass first")
    split = dict(started_utc=recovery.now(), enabled=True, odl_shard_concurrency=1,
                 sdk_ranges_per_shard=8, hf_concurrency=4, hf_supervisor_stopped=False,
                 assignments=[], training_allowed=False, smoke_allowed=False)
    recovery.atomic_json(STATE, split)
    for position in range(39, 51):
        state = recovery.load(recovery.STATE)
        row = next(row for row in state["shards"] if row["filename"] == f"sa_{position:06d}.tar")
        if not eligible(row, state["shards"]):
            split["assignments"].append(dict(filename=row["filename"], status="SKIPPED_HF_OWNS_OR_TOO_CLOSE"))
            recovery.atomic_json(STATE, split)
            continue
        directory = FOLDER / row["filename"].removesuffix(".tar")
        directory.mkdir(exist_ok=True)
        assignment = dict(filename=row["filename"], status="DOWNLOADING_ODL", started_utc=recovery.now(),
                          zero_hf_bytes_at_assignment=True)
        split["assignments"].append(assignment)
        recovery.atomic_json(STATE, split)
        process = subprocess.Popen([str(recovery.ROOT / ".opendatalab-venv/bin/python"), __file__, "worker", "--filename", row["filename"],
                                    "--directory", str(directory)], cwd=recovery.ROOT,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        beginning = time.monotonic()
        initial_hf_bytes = {entry["filename"]: payload_bytes(entry) for entry in state["shards"]}
        peak = 0
        previous_bytes = 0
        previous_time = beginning
        stopped = False
        while process.poll() is None:
            time.sleep(10)
            counters = recovery.load(directory / "worker-counters.json", {})
            current_time = time.monotonic()
            elapsed = current_time - beginning
            peak = max(peak, max(0, counters.get("bytes", 0) - previous_bytes) / (current_time - previous_time) / probe.MIB)
            rate = counters.get("bytes", 0) / elapsed / probe.MIB
            current_hf_bytes = {entry["filename"]: payload_bytes(entry) for entry in state["shards"]}
            hf_rate = sum(max(0, current_hf_bytes[name] - initial_hf_bytes[name]) for name in initial_hf_bytes) / elapsed / probe.MIB
            assignment.update(elapsed_seconds=elapsed, bytes=counters.get("bytes", 0), average_mib_s=rate,
                              hf_average_mib_s=hf_rate, combined_gain=(rate + hf_rate) / 5.299793803521094,
                              errors=counters.get("errors", 0), retries=counters.get("retries", 0))
            recovery.atomic_json(STATE, split)
            previous_time, previous_bytes = current_time, counters.get("bytes", 0)
            fresh = recovery.load(recovery.STATE)
            fresh_row = fresh["shards"][position]
            if payload_bytes(fresh_row) or fresh_row.get("download_status") == "downloading":
                probe.finish_worker(process)
                assignment["status"] = "STOPPED_HF_APPROACH_OR_OWNS_BYTES"
                stopped = True
                break
            if elapsed >= 600 and (rate < 8 or counters.get("errors", 0) > 2 or assignment["combined_gain"] < 1.3):
                probe.finish_worker(process)
                assignment["status"] = "STOPPED_SPEED_OR_ERROR_GATE"
                stopped = True
                break
        if stopped:
            split.update(enabled=False, stopped_utc=recovery.now(), stopped_reason=assignment["status"])
            recovery.atomic_json(STATE, split)
            return
        source = recovery.ASSETS / "downloads/sa1b-opendatalab" / ("split-" + row["filename"].removesuffix(".tar")) / "OpenDataLab___SA-1B/raw" / row["filename"]
        if not source.is_file() or source.stat().st_size != row["expected_size_bytes"]:
            assignment["status"] = "FAILED_DOWNLOAD_PARTIAL_PRESERVED"
            split.update(enabled=False, stopped_utc=recovery.now())
            recovery.atomic_json(STATE, split)
            return
        assignment["status"] = "CHECKSUM_TAR_FULL_DECODE_AUDIT"
        recovery.atomic_json(STATE, split)
        subprocess.run([str(recovery.ROOT / ".venv/bin/python"), __file__, "audit", "--filename", row["filename"],
                        "--archive", str(source), "--directory", str(directory)], check=True, cwd=recovery.ROOT)
        audit = recovery.load(directory / "audit.json")
        assignment.update(audit=str(directory / "audit.json"), audit_passed=audit.get("protocol_passed", False),
                          peak_ten_second_mib_s=peak)
        if not audit.get("protocol_passed"):
            assignment["status"] = "QUARANTINED_ODL_ARCHIVE_OR_IMAGE_INVALID"
            recovery.atomic_json(STATE, split)
            continue
        fresh = recovery.load(recovery.STATE)
        if not eligible(fresh["shards"][position], fresh["shards"]):
            assignment["status"] = "VERIFIED_SOURCE_RETAINED_HF_OWNS_OR_TOO_CLOSE"
            recovery.atomic_json(STATE, split)
            continue
        handoff(source, row, state["revision"], audit)
        assignment.update(status="VERIFIED_STAGED_FOR_NATIVE_HF_INGESTION", finished_utc=recovery.now(),
                          hf_payload_redownload_required=False)
        recovery.atomic_json(STATE, split)
    split.update(enabled=False, completed_utc=recovery.now(), stopped_reason="Eligible future-shard subqueue completed")
    recovery.atomic_json(STATE, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "audit"))
    parser.add_argument("--filename")
    parser.add_argument("--archive")
    parser.add_argument("--directory")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.command == "worker":
        download_worker(args.filename, args.directory, args.target)
    elif args.command == "audit":
        row = next(row for row in recovery.load(recovery.STATE)["shards"] if row["filename"] == args.filename)
        print(json.dumps(full_audit(args.archive, row, args.directory)), flush=True)
    else:
        split_run()
