"""Allocation-aware source migration and focused, byte-preserving JPEG rescue."""

import argparse
import copy
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
import zlib

import sa1b_recovery as recovery
from sa1b_throughput_benchmark import payload_bytes


FOLDER = recovery.EVIDENCE / "sa1b-wallclock"
MIB = 1024 ** 2


def forensic_target(filename):
    return FOLDER / "forensic" / filename.removesuffix(".tar") / "download"


def forensic_archive(filename):
    return forensic_target(filename) / "OpenDataLab___SA-1B/raw" / filename


def migration_decision(full_bytes, received, hf_mib_s, odl_total_mib_s, odl_concurrency=3):
    if min(hf_mib_s, odl_total_mib_s, odl_concurrency) <= 0:
        raise ValueError("Measured source speeds and concurrency must be positive")
    remaining = max(0, full_bytes-received)
    hf_seconds = remaining/hf_mib_s/MIB
    odl_seconds = full_bytes/(odl_total_mib_s/odl_concurrency)/MIB
    saving = 1-odl_seconds/hf_seconds if hf_seconds else 0
    return dict(full_bytes=full_bytes, downloaded_bytes=received, remaining_bytes=remaining,
        hf_continue_seconds=hf_seconds, odl_restart_seconds=odl_seconds,
        odl_effective_per_shard_mib_s=odl_total_mib_s/odl_concurrency,
        predicted_time_saving_fraction=saving, migrate=bool(remaining and saving >= .30))


def prepare_plan():
    from sa1b_fast_recovery import PLAN, LIVE, BENCH

    old = recovery.load(PLAN)
    state = recovery.load(recovery.STATE)
    live = recovery.load(LIVE)
    selected = recovery.load(BENCH)["selected"]
    hf_speed = live["hf_mib_s"]
    if hf_speed < .5:
        raise RuntimeError("Cannot migrate based on a missing live HF rate")
    odl_speed = selected["total_mib_s"]
    FOLDER.mkdir(parents=True, exist_ok=True)
    snapshot = FOLDER / ("plan-before-" + str(time.time_ns()) + ".json")
    recovery.atomic_json(snapshot, old)
    plan = copy.deepcopy(old)
    rows = []
    for row in state["shards"]:
        filename = row["filename"]
        if old["owners"][filename] != "HF_EXISTING_PARTIAL_ONLY":
            continue
        received = payload_bytes(row)
        if row.get("extraction_status") == "complete" or received >= row["expected_size_bytes"] or row.get("download_status") == recovery.QUARANTINED:
            continue
        decision = dict(filename=filename, **migration_decision(row["expected_size_bytes"], received, hf_speed, odl_speed))
        rows.append(decision)
        plan["owners"][filename] = "ODL" if decision["migrate"] else "HF_EXISTING_PARTIAL_ONLY"
        plan["initial_hf_bytes"][filename] = received
    retained = [row for row in rows if not row["migrate"]]
    plan.update(updated_utc=recovery.now(), hf_concurrency=max(1, min(2, len(retained))),
        forensic_downloads=["sa_000014.tar"], try_local_17_first=True,
        maximum_global_odl_concurrency=3, formal_training_allowed=False,
        smoke_only_after_full_existence_and_decode=False, smoke_allowed=False,
        previous_plan_snapshot=str(snapshot), wallclock_migrations=rows)
    plan["owners"]["sa_000014.tar"] = "ODL_FORENSIC"
    plan["owners"]["sa_000017.tar"] = "ODL_FORENSIC"
    recovery.atomic_json(PLAN, plan)
    report = dict(created_utc=recovery.now(), hf_measured_mib_s=hf_speed,
        odl_last_stable_aggregate_mib_s=odl_speed, odl_concurrency=3,
        calculation="A=remaining/measured HF aggregate; B=full/(ODL aggregate/3), conservative per-slot ODL estimate",
        speed_caveat="ODL48.35MiB/s is the prior stable measurement, not a new instantaneous rate",
        rows=rows, partials_deleted=0, confirmed_404_retries_allowed=False,
        forensic_required_missing=117, smoke_allowed=False, formal_training_allowed=False)
    recovery.atomic_json(FOLDER / "migration.json", report)
    lines = ["# SA-1B minimum-wall-clock recovery", "", "Measured-rate snapshot: " + report["created_utc"], "",
        f"HF aggregate: {hf_speed:.3f}MiB/s; previous stable ODL3 aggregate: {odl_speed:.3f}MiB/s.",
        "B uses ODL aggregate divided by3, avoiding the false assumption that each of3 shards gets48MiB/s.",
        "A uses the entire measured HF aggregate per shard, conservatively favoring retaining partials.",
        "No new speed benchmark. All old HF partials are preserved. No confirmed ModelScope404 retries.", "",
        "| Shard | Received bytes | Remaining bytes | Full bytes | HF A minutes | ODL B minutes | Saving | Assignment |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['filename']} | {row['downloaded_bytes']} | {row['remaining_bytes']} | {row['full_bytes']} | {row['hf_continue_seconds']/60:.2f} | {row['odl_restart_seconds']/60:.2f} | {row['predicted_time_saving_fraction']:.1%} | {'ODL' if row['migrate'] else 'HF resume'} |")
    odl_bytes = sum(row["full_bytes"] for row in rows if row["migrate"])+state["shards"][14]["expected_size_bytes"]+state["shards"][17]["expected_size_bytes"]
    hf_bytes = sum(row["remaining_bytes"] for row in retained)
    eta = max(odl_bytes/odl_speed/MIB, hf_bytes/hf_speed/MIB)
    report.update(projected_odl_bytes_including_conditional_17=odl_bytes,
        projected_hf_remaining_bytes=hf_bytes, projected_pure_download_seconds=eta)
    recovery.atomic_json(FOLDER / "migration.json", report)
    lines.extend(["", f"Conservative pure-transfer projection including conditional17: {eta/60:.1f}minutes.",
        "This is not completion ETA: checksum, gzip rescue, NAS extraction/installation and final audit are additional.",
        "14 is an explicitly authorized independent full ODL download.17 is downloaded only if the local focused extraction fails.",
        "16 is already complete and is never touched. ODL globally has at most3 downloads, including forensic jobs.",
        "Latest live status: evidence/sa1b-fast-recovery/state.json. No smoke or training is authorized."])
    (recovery.RECOVERY / "SA1B_WALLCLOCK_RECOVERY.md").write_text("\n".join(lines)+"\n")
    return report


def forensic_job(row):
    filename = row["filename"]
    if filename not in ("sa_000014.tar", "sa_000017.tar"):
        raise ValueError("Only14 and conditional17 may be independently redownloaded")
    target = forensic_target(filename)
    if shutil.disk_usage(FOLDER).free < 64 * 1024**3:
        raise RuntimeError("Insufficient forensic tar and extraction space")
    directory = target.parent / "worker"
    directory.mkdir(parents=True, exist_ok=True)
    path = forensic_archive(filename)
    for attempt in range(1, 4):
        if not path.is_file() or path.stat().st_size != row["expected_size_bytes"]:
            process = subprocess.Popen([str(recovery.ROOT / ".opendatalab-venv/bin/python"), "-u",
                str(recovery.RECOVERY / "sa1b_opendatalab_split.py"), "worker", "--filename", filename,
                "--directory", str(directory), "--target", str(target)], cwd=recovery.ROOT,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            process.wait()
        if path.is_file() and path.stat().st_size == row["expected_size_bytes"]:
            proof = recovery.verify_archive(path, row)
            proof.update(filename=filename, archive=str(path), source="OpenDataLab/SA-1B",
                source_dataset_id=6248, independent_forensic_copy=True,
                container_validity_not_a_rescue_precondition=True, attempt=attempt)
            recovery.atomic_json(target.parent / "checksum.json", proof)
            return proof
        time.sleep(10)
    raise RuntimeError("Forensic ODL bounded retries exhausted; all partials preserved")


def start_rescue(filename, archive, source):
    directory = FOLDER / "focused-rescue" / filename.removesuffix(".tar") / source
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.log").open("a") as output:
        return subprocess.Popen([str(recovery.ROOT / ".venv/bin/python"), "-u", __file__, "rescue",
            "--filename", filename, "--archive", str(archive), "--source", source], cwd=recovery.ROOT,
            stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)


def scan_tar(stream, required, output):
    output.mkdir(parents=True, exist_ok=True)
    position = 0
    recovered = []
    invalid_blocks = 0
    last = None
    try:
        while True:
            header = stream.read(512)
            if not header:
                break
            if len(header) != 512:
                raise EOFError("Partial tar header")
            position += 512
            if not any(header):
                continue
            try:
                member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
                if not member.isreg() and not member.isdir():
                    raise tarfile.InvalidHeaderError("Unsupported member type")
            except tarfile.HeaderError:
                invalid_blocks += 1
                continue
            last = member.name
            relative = PurePosixPath(member.name)
            safe = not relative.is_absolute() and ".." not in relative.parts
            wanted = member.isreg() and safe and relative.name in required
            if wanted and member.size > 256 * 1024**2:
                raise ValueError("Implausible JPEG member size")
            destination = output / relative.name if wanted else None
            temporary = destination.with_suffix(".pending") if wanted else None
            handle = temporary.open("wb") if wanted else None
            try:
                remaining = member.size
                while remaining:
                    data = stream.read(min(1024**2, remaining))
                    if not data:
                        raise EOFError("Partial member payload")
                    position += len(data)
                    remaining -= len(data)
                    if handle:
                        handle.write(data)
                padding = (-member.size) % 512
                if len(stream.read(padding)) != padding:
                    raise EOFError("Partial member padding")
                position += padding
            finally:
                if handle:
                    handle.close()
            if wanted:
                os.replace(temporary, destination)
                recovered.append(relative.name)
                if required.issubset(recovered):
                    break
        return dict(returncode=0, recovered_members=recovered, last_member=last,
                    decompressed_offset=position, invalid_header_blocks=invalid_blocks)
    except Exception as error:
        return dict(returncode=1, error_type=type(error).__name__, error=str(error),
            recovered_members=recovered, last_member=last, decompressed_offset=position,
            invalid_header_blocks=invalid_blocks)


def refresh_missing():
    missing = {}
    for position in (14, 16, 17, 19):
        proof = recovery.load(recovery.EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}/result.json")
        missing[proof["filename"]] = proof["missing_required_image_ids"]
    report = dict(checked_utc=recovery.now(), training_index_sha256=recovery.RECORDS_SHA,
        scope="Precisely missing required images from installed original/rescue inventories",
        missing_count=sum(len(ids) for ids in missing.values()), shards=missing,
        confirmed_modelscope_404_retry_allowed=False)
    recovery.atomic_json(recovery.RECOVERY / "MISSING_REQUIRED_SA_IMAGES.json", report)
    return report


def extract_deflate_tail(archive, required, output, maximum_seconds=180):
    archive = Path(archive)
    output.mkdir(parents=True, exist_ok=True)
    length = archive.stat().st_size
    beginning = max(0, length-64 * MIB)
    with archive.open("rb") as source:
        if source.read(2) != b"\x1f\x8b":
            return dict(status="NOT_GZIP", recovered_members=[])
        source.seek(beginning)
        region = source.read(64 * MIB)
    deadline = time.monotonic()+maximum_seconds
    attempts = 0
    position = 0
    recovered = []
    proofs = []
    candidates = []
    while position+5 < len(region) and time.monotonic() < deadline:
        zero = region.find(b"\x00", position)
        one = region.find(b"\x01", position)
        available = [offset for offset in (zero, one) if offset >= 0]
        if not available:
            break
        candidate = min(available)
        position = candidate+1
        stored_size = int.from_bytes(region[candidate+1:candidate+3], "little")
        complement = int.from_bytes(region[candidate+3:candidate+5], "little")
        if stored_size < 32768 or stored_size ^ complement != 65535 or candidate+5+stored_size > len(region):
            continue
        candidates.append((candidate, stored_size))
    for candidate, stored_size in reversed(candidates):
        if attempts >= 1024 or time.monotonic() >= deadline:
            break
        attempts += 1
        try:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(region[candidate:], 256 * MIB)
        except zlib.error:
            continue
        if not decoder.eof or decoder.unused_data != region[-8:] or data[:stored_size] != region[candidate+5:candidate+5+stored_size]:
            continue
        for name in sorted(required.difference(recovered)):
            offset = data.find(name.encode()+b"\x00")
            while offset >= 0:
                header_offset = offset-2 if data[max(0, offset-2):offset] == b"./" else offset
                try:
                    member = tarfile.TarInfo.frombuf(data[header_offset:header_offset+512], "utf-8", "surrogateescape")
                    relative = PurePosixPath(member.name)
                    if not member.isreg() or relative.name != name or relative.is_absolute() or ".." in relative.parts:
                        raise tarfile.InvalidHeaderError("Unsafe or unrelated member")
                    payload = data[header_offset+512:header_offset+512+member.size]
                    if not 0 < member.size <= 256 * MIB or len(payload) != member.size:
                        raise tarfile.InvalidHeaderError("Incomplete member")
                except (tarfile.HeaderError, ValueError):
                    offset = data.find(name.encode()+b"\x00", offset+1)
                    continue
                (output / name).write_bytes(payload)
                recovered.append(name)
                proofs.append(dict(basename=name, compressed_resync_offset=beginning+candidate,
                    first_stored_block_bytes=stored_size, initial_dictionary="EMPTY; never guessed",
                    raw_deflate_eof=True, gzip_footer_exact=True, tar_header_checksum_valid=True,
                    suffix_tar_header_offset=header_offset, size_bytes=member.size,
                    sha256=recovery.hashlib.sha256(payload).hexdigest()))
                break
        if required.issubset(recovered):
            break
    return dict(status="RECOVERED" if required.issubset(recovered) else "STILL_MISSING",
        recovered_members=recovered, attempts=attempts, scan_window_bytes=len(region),
        original_archive_modified=False, compression_specification="RFC1951 stored blocks, LEN/NLEN and32768-byte distance window",
        maximum_seconds=maximum_seconds, maximum_attempts=1024,
        maximum_decoded_suffix_bytes=256 * MIB, member_proofs=proofs)


def focused_rescue(filename, archive, source):
    import sa1b_jpeg_rescue as rescue

    position = int(filename[3:9])
    if position not in (14, 17):
        raise ValueError("Do not touch already-complete16")
    proof_path = rescue.FOLDER / filename.removesuffix(".tar") / "result.json"
    proof = recovery.load(proof_path)
    original_inventory = Path(proof["image_inventory"])
    if recovery.digest(original_inventory) != proof["image_inventory_sha256"]:
        raise ValueError("Installed inventory digest mismatch")
    inventory = {entry["basename"]: entry for entry in map(json.loads, original_inventory.open())}
    wanted = {f"sa_{image_id}.jpg" for image_id in proof["missing_required_image_ids"]}
    directory = FOLDER / "focused-rescue" / filename.removesuffix(".tar") / source
    directory.mkdir(parents=True, exist_ok=True)
    archive = Path(archive)
    identity = dict(size=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns, inode=archive.stat().st_ino)
    tests, failures = [], []
    started = recovery.now()
    def collect(output, label):
        rescue.collect(output, wanted, inventory, failures, label)
    if source == "odl":
        names = directory / "required-members.txt"
        names.write_text("".join("./"+name+"\n" for name in sorted(wanted)))
        for tool in ("tar", "bsdtar"):
            if not shutil.which(tool) or wanted.issubset(inventory):
                continue
            output = directory / tool
            output.mkdir(exist_ok=True)
            command = [tool, "-xf", str(archive), "-C", str(output), "-T", str(names)]
            if tool == "tar":
                command.insert(1, "--ignore-zeros")
                command.extend(["--no-same-owner", "--no-same-permissions"])
            with (directory / (tool+".stderr.txt")).open("wb") as errors:
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors)
            tests.append(dict(tool=tool, returncode=result.returncode, stderr=str(directory / (tool+".stderr.txt"))))
            collect(output, tool+" focused original JPEG extraction")
        if not wanted.issubset(inventory):
            output = directory / "python-stream"
            result = rescue.python_extract(archive, output, wanted.difference(inventory), "r|*", True)
            tests.append(dict(tool="python-tarfile-stream-ignore-zeros", **result))
            collect(output, "Python stream original JPEG extraction")
    if source == "tail" and not wanted.issubset(inventory):
        output = directory / "deflate-tail"
        result = extract_deflate_tail(archive, wanted.difference(inventory), output)
        tests.append(dict(tool="bounded-read-only-deflate-tail-resync", **result))
        collect(output, "Original stored-block bytes and dictionary-free RFC1951 suffix decode; checksum-valid tar member")
    if source != "tail" and not wanted.issubset(inventory):
        output = directory / "tolerant-512-scanner"
        with archive.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(archive, "rb") as stream:
            result = scan_tar(stream, wanted.difference(inventory), output)
        tests.append(dict(tool="checksum-validating-tolerant-512-scanner", compression="gzip" if compressed else "none", **result))
        collect(output, "Validated tar header and original JPEG payload; tolerant512 scanner")
    remaining = sorted(int(Path(name).stem[3:]) for name in wanted.difference(inventory))
    data = "".join(json.dumps(inventory[name], sort_keys=True)+"\n" for name in sorted(inventory))
    immutable = directory / ("installed-"+recovery.hashlib.sha256(data.encode()).hexdigest()+".jsonl")
    immutable.write_text(data)
    after = dict(size=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns, inode=archive.stat().st_ino)
    if identity != after:
        raise ValueError("Original archive evidence identity changed")
    audit = dict(filename=filename, source=source, archive=str(archive), started_utc=started,
        finished_utc=recovery.now(), requested_missing_count=len(wanted), recovered=len(wanted)-len(remaining),
        missing_required_image_ids=remaining, tests=tests, decode_failures=failures,
        original_archive_modified=False, archive_identity=identity, image_transform="NONE")
    recovery.atomic_json(directory / "result.json", audit)
    proof.update(recovered_required_images=len(inventory), missing_required_image_ids=remaining,
        remaining_decode_failures=0, image_inventory=str(immutable), image_inventory_sha256=recovery.digest(immutable),
        status="TRAINING_IMAGES_FULLY_RECOVERED" if not remaining else "REQUIRED_JPEGS_STILL_MISSING",
        last_repaired_utc=recovery.now(), focused_original_archive_rescue=audit)
    if source == "odl":
        proof.update(full_original_redownload=True, independent_forensic_archive=str(archive),
            independent_forensic_checksum=recovery.load(forensic_target(filename).parent / "checksum.json"))
    recovery.atomic_json(proof_path, proof)
    rescue.write_reports([recovery.load(rescue.FOLDER / f"sa_{item:06d}/result.json") for item in (14, 16, 17, 19)])
    refresh_missing()
    print(json.dumps(dict(filename=filename, recovered=audit["recovered"], missing=len(remaining))), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "missing", "rescue"))
    parser.add_argument("--filename")
    parser.add_argument("--archive")
    parser.add_argument("--source", choices=("local", "odl", "tail"))
    arguments = parser.parse_args()
    if arguments.command == "plan":
        print(json.dumps(prepare_plan(), indent=2))
    elif arguments.command == "missing":
        print(json.dumps(refresh_missing(), indent=2))
    else:
        focused_rescue(arguments.filename, arguments.archive, arguments.source)
