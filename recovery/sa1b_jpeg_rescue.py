"""Recover required original JPEG bytes without modifying archive evidence."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

from PIL import Image
import sa1b_recovery as recovery


FOLDER = recovery.EVIDENCE / "sa1b-jpeg-rescue"
RANGES = {14: (156625, 167810), 16: (179000, 190187), 17: (190188, 201375), 19: (212562, 223749)}


def required_range(position, state, families):
    if position in RANGES:
        return RANGES[position]
    bounds = []
    for neighbor in (position-1, position+1):
        if neighbor < 0:
            bounds.append(-1)
            continue
        if neighbor >= len(state["shards"]):
            bounds.append(max(int(Path(path).stem[3:]) for path in families["sam"])+1)
            continue
        row = state["shards"][neighbor]
        proof = recovery.load(recovery.EVIDENCE / "sa1b-shards" / (row["filename"]+".json"), row)
        if not proof.get("tar_integrity_passed") or proof.get("observed_md5") != row["expected_checklist_checksum"]:
            raise RuntimeError("Verified adjacent original shard boundaries are not yet available")
        listing = Path(proof["tar_listing"])
        ids = [int(Path(name).stem[3:]) for name in listing.read_text().splitlines()
               if re.fullmatch(r"sa_\d+\.jpg", Path(name).name)]
        bounds.append(max(ids) if neighbor < position else min(ids))
    return bounds[0]+1, bounds[1]-1


def check_jpeg(path):
    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG" or image.width <= 0 or image.height <= 0:
            raise ValueError("Not a valid original JPEG")
        size = [image.width, image.height]
    return dict(basename=path.name, sha256=recovery.digest(path), size_bytes=path.stat().st_size,
                width=size[0], height=size[1], decode_passed=True, image_transform="NONE")


def decode_inventory(inventory, output):
    entries = [json.loads(line) for line in Path(inventory).open()]
    def check(entry):
        try:
            path = recovery.SAM_ROOT / entry["basename"]
            result = check_jpeg(path)
            if result["sha256"] != entry["sha256"]:
                raise ValueError("Extracted image byte digest differs from archive")
            return None
        except Exception as error:
            return dict(basename=entry["basename"], error_type=type(error).__name__)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        failures = [result for result in executor.map(check, entries) if result]
    recovery.atomic_json(output, dict(checked_utc=recovery.now(), checked_images=len(entries), failures=failures,
        passed=bool(entries) and not failures, inventory_sha256=recovery.digest(Path(inventory)), image_transform="NONE"))


def install(path, name, inventory, source, verified=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Unsafe rescue source")
    record = verified or check_jpeg(path)
    target = recovery.SAM_ROOT / name
    if target.is_symlink():
        raise ValueError("Refusing target symlink")
    if target.exists():
        if recovery.digest(target) != record["sha256"]:
            raise ValueError("Existing original image differs; not overwritten")
    else:
        temporary = target.with_name("." + name + ".rescue-" + str(os.getpid()))
        shutil.copyfile(path, temporary)
        if recovery.digest(temporary) != record["sha256"]:
            raise ValueError("Copy byte digest mismatch")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if recovery.digest(target) != record["sha256"]:
                raise ValueError("Concurrent target differs; not overwritten")
        temporary.unlink()
    record.update(source=str(path), source_tool=source)
    inventory[name] = record


def transfer_inventory(inventory_path, source_root, output):
    entries = [json.loads(line) for line in Path(inventory_path).open()]
    inventory = {}
    def transfer(entry):
        path = Path(source_root) / entry["basename"]
        try:
            record = check_jpeg(path)
            if record["sha256"] != entry["sha256"] or record["size_bytes"] != entry["size_bytes"]:
                raise ValueError("Local original member byte digest mismatch")
            install(path, path.name, inventory, "Original archive member staged without transformation", verified=record)
            return None
        except Exception as error:
            return dict(basename=entry["basename"], error_type=type(error).__name__)
    failures = []
    checked = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for result in executor.map(transfer, entries):
            checked += 1
            if result:
                failures.append(result)
            if checked % 256 == 0:
                recovery.atomic_json(Path(output).with_name("transfer-progress.json"), dict(checked=checked, required=len(entries), failures=len(failures), checked_utc=recovery.now()))
    recovery.atomic_json(output, dict(checked_utc=recovery.now(), checked_images=checked, failures=failures,
        passed=checked == len(entries) and bool(entries) and not failures, inventory_sha256=recovery.digest(Path(inventory_path)),
        source_root=str(source_root), destination_root=str(recovery.SAM_ROOT), transfer_workers=32, image_transform="NONE"))


def collect(directory, required, inventory, failures, source):
    paths = [path for path in directory.rglob("*.jpg") if path.name in required]
    def recover(path):
        try:
            install(path, path.name, inventory, source)
        except Exception as error:
            failures.append(dict(basename=path.name, source=source, error_type=type(error).__name__))
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(recover, paths))


def python_extract(archive, output, missing, mode, ignore_zeros):
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    last = None
    offset = None
    try:
        with tarfile.open(archive, mode=mode, ignore_zeros=ignore_zeros) as container:
            for member in container:
                last, offset = member.name, member.offset
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                    raise ValueError("Unsafe tar member")
                if not member.isfile() or relative.name not in missing:
                    continue
                destination = output / relative.name
                with container.extractfile(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                count += 1
        return dict(returncode=0, extracted_members=count, last_member=last, offset=offset)
    except Exception as error:
        return dict(returncode=1, error_type=type(error).__name__, error=str(error), extracted_members=count,
                    last_member=last, offset=offset)


def write_reports(results):
    missing = {result["filename"]: result["missing_required_image_ids"] for result in results}
    recovery.atomic_json(recovery.RECOVERY / "MISSING_REQUIRED_SA_IMAGES.json", dict(checked_utc=recovery.now(),
        training_index_sha256=recovery.RECORDS_SHA, scope="Only actually missing required JPEGs in quarantined shards14/16/17",
        missing_count=sum(len(value) for value in missing.values()), shards=missing,
        pending_shards=[row["filename"] for row in recovery.load(recovery.STATE)["shards"]
            if row.get("download_status") == recovery.QUARANTINED and row["filename"] not in missing],
        quarantined_shards_audit_complete=all(row["filename"] in missing for row in recovery.load(recovery.STATE)["shards"]
            if row.get("download_status") == recovery.QUARANTINED)))
    lines = ["# Corrupt SA-1B shard JPEG rescue", "", "Training consumes original JPEG bytes, not tar container validity.",
        "Evidence originals are never modified. Extraction uses independent copies and retains only necessary image data.",
        "No resize, recompression, rename or format conversion. Frozen training index SHA256: `"+recovery.RECORDS_SHA+"`.", "",
        "| Shard | Required | Valid original JPEGs recovered | Missing | Decode failures remaining | Status |",
        "|---|---:|---:|---:|---:|---|"]
    for result in results:
        lines.append(f"| {result['filename']} | {result['required_images']} | {result['recovered_required_images']} | {len(result['missing_required_image_ids'])} | {result['remaining_decode_failures']} | {result['status']} |")
    lines.extend(["", "Exact missing IDs: MISSING_REQUIRED_SA_IMAGES.json. Per-image hashes/dimensions: evidence/sa1b-jpeg-rescue/<shard>/images.jsonl.",
        "000014's historical local archive was lost; historical1MiB fragments were not a recoverable full archive.",
        "The latest minimum-wall-clock instruction authorizes an independent full ODL14 download and conditional17 download; old evidence and all installed JPEGs are preserved.",
        "Fresh checksum and focused extraction evidence: evidence/sa1b-wallclock/forensic/ and evidence/sa1b-wallclock/focused-rescue/.",
        "The archives are gzip compressed despite .tar names: compressed size modulo512 and zero padding do not repair gzip corruption.",
        "Tail repair is attempted only on an independent decompressed tar copy if member bytes are demonstrably complete.",
        "Pending rescue proofs are not counted as completed rescue. Original-byte staged JPEGs may exist before full decode/atomic installation proof finishes.",
        "Additional000019 is now checksum-matching but container-invalid. Its ID range212562..223749 is bounded by fully listed/verified18 and verified20; it gets independent JPEG rescue without a same-object redownload.",
        "Candidate single-image access investigations: evidence/sa1b-jpeg-rescue/third-source-search.json and third-source-candidates.json.",
        "ODL official file catalog exposes original raw tar objects, not individual JPEG objects. No individual-JPEG object API has been established.",
        "Existing DCI/Urban archives have no exact SA image-ID overlap with the quarantined ranges.",
        "Candidate kkkkkcm/SA-1B-400k has seekable JPEG members, but original-byte equivalence is UNPROVEN; only bounded header/member probes are permitted and it is not used for recovery.",
        "k-m-irfan/sa1b is gated; current metadata probes time out. Authorization/network status remains unresolved, and no permission gate is bypassed."])
    (recovery.RECOVERY / "SA1B_CORRUPT_SHARD_RESCUE.md").write_text("\n".join(lines)+"\n")


def rescue_one(position, families):
    state = recovery.load(recovery.STATE)
    row = state["shards"][position]
    filename = row["filename"]
    directory = FOLDER / filename.removesuffix(".tar")
    directory.mkdir(parents=True, exist_ok=True)
    staging_root = Path("/tmp/said-sa1b-corrupt-rescue") / filename.removesuffix(".tar")
    staging_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(staging_root).free < 64 * 1024**3:
        raise RuntimeError("Insufficient local JPEG rescue staging space")
    low, high = required_range(position, state, families)
    required = {Path(path).name for path in families["sam"] if low <= int(Path(path).stem[3:]) <= high}
    original = Path(row["archive"])
    inventory, failures, tests = {}, [], []
    result = dict(filename=filename, started_utc=recovery.now(), required_images=len(required),
        required_id_range=[low, high], training_index_sha256=recovery.RECORDS_SHA, original_archive=str(original),
        original_present=original.is_file(), original_archive_modified=False, full_original_redownload=False,
        expected_md5=row["expected_checklist_checksum"], expected_sha256=row["expected_lfs_sha256"], tests=tests)
    recovery.atomic_json(directory / "progress.json", result)
    recovery.SAM_ROOT.mkdir(parents=True, exist_ok=True)
    previous_inventory = directory / "images.jsonl"
    if previous_inventory.is_file():
        for line in previous_inventory.open():
            entry = json.loads(line)
            if entry["basename"] in required:
                target = recovery.SAM_ROOT / entry["basename"]
                install(target, target.name, inventory, "Previously byte-verified rescue")
    if position == 14:
        collect(recovery.EVIDENCE / "sa1b-000014-forensics/original-prefix-samples", required, inventory, failures, "Historical original gzip prefix")
        tests.append(dict(status="FULL_ARCHIVE_MISSING", method="No same-object redownload; prefix rescue only"))
    elif original.is_file():
        observed = recovery.verify_archive(original, row)
        result["original_checksum_audit"] = observed
        before = original.stat()
        with original.open("rb") as stream:
            magic = stream.read(8)
            stream.seek(-4096, os.SEEK_END)
            tail = stream.read()
        result.update(file_size_modulo512=before.st_size % 512, compression="gzip" if magic.startswith(b"\x1f\x8b") else "uncompressed_tar",
            last4kb_sha256=hashlib.sha256(tail).hexdigest(), last4kb_zero_bytes=tail.count(b"\0"),
            original_inode=before.st_ino, original_mtime_ns=before.st_mtime_ns)
        (directory / "original-last4kb.bin").write_bytes(tail)
        archive = directory / "working-copy.tar"
        if not archive.exists():
            subprocess.run(["cp", "--reflink=auto", "--", str(original), str(archive)], check=True)
        if archive.stat().st_ino == original.stat().st_ino or recovery.digest(archive) != observed["observed_sha256"]:
            raise ValueError("Working copy is not an independent byte-identical archive")
        listing = Path(row.get("tar_listing", recovery.EVIDENCE / "sa1b-shards" / (filename+".tar-members.txt")))
        member_names = []
        for name in listing.read_text().splitlines():
            relative = PurePosixPath(name)
            if relative.name in required and not relative.is_absolute() and ".." not in relative.parts:
                member_names.append(name)
        selected = directory / "required-members.txt"
        selected.write_text("\n".join(member_names)+"\n")
        for tool, executable in (("gnu-ignore-zeros", shutil.which("tar")), ("bsdtar", shutil.which("bsdtar") or "/root/miniconda3/bin/bsdtar")):
            if required.issubset(inventory):
                tests.append(dict(tool=tool, status="NOT_NEEDED_ALL_REQUIRED_JPEGS_RECOVERED"))
                continue
            if not executable or not Path(executable).is_file():
                tests.append(dict(tool=tool, status="NOT_AVAILABLE"))
                continue
            output = staging_root / tool
            output.mkdir(exist_ok=True)
            command = [executable] + (["--ignore-zeros"] if tool == "gnu-ignore-zeros" else []) + ["-xf", str(archive), "-C", str(output), "-T", str(selected)]
            with (directory / (tool+".stderr.txt")).open("wb") as errors, (directory / (tool+".stdout.txt")).open("wb") as stdout:
                process = subprocess.run(command, stdout=stdout, stderr=errors)
            tests.append(dict(tool=tool, returncode=process.returncode, stderr=str(directory / (tool+".stderr.txt")), original_jpeg_staging=str(output)))
            collect(output, required, inventory, failures, tool)
            recovery.atomic_json(directory / "progress.json", dict(result, extracted_required_images=len(inventory), tests=tests))
        for mode, ignore in (("r:*", False), ("r|*", False), ("r|*", True)):
            label = "python-" + ("stream" if "|" in mode else "normal") + ("-ignore-zeros" if ignore else "")
            if required.issubset(inventory):
                tests.append(dict(tool=label, status="NOT_NEEDED_ALL_REQUIRED_JPEGS_RECOVERED"))
                continue
            output = staging_root / label
            diagnostic = python_extract(archive, output, required.difference(inventory), mode, ignore)
            tests.append(dict(tool=label, **diagnostic))
            collect(output, required, inventory, failures, label)
        executable = shutil.which("7z") or shutil.which("7zz")
        if executable and not required.issubset(inventory):
            output = staging_root / "7z"
            output.mkdir(exist_ok=True)
            with (directory / "7z.stderr.txt").open("wb") as errors:
                process = subprocess.run([executable, "x", str(archive), "-o"+str(output), "-y", "-ir!*.jpg"], stdout=subprocess.DEVNULL, stderr=errors)
            tests.append(dict(tool="7z", returncode=process.returncode))
            collect(output, required, inventory, failures, "7z")
        else:
            tests.append(dict(tool="7z", status="NOT_NEEDED" if required.issubset(inventory) else "NOT_AVAILABLE"))
        result["tail_repair"] = dict(attempted=False, reason="Compressed stream error; appending zeros to gzip is not a valid tar EOF repair" if result["compression"] == "gzip" else "No demonstrated tail-only defect")
        after = original.stat()
        if before.st_ino != after.st_ino or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("Original forensic evidence changed")
    missing = sorted(required.difference(inventory), key=lambda name: int(Path(name).stem[3:]))
    with (directory / "images.jsonl").open("w") as stream:
        for name in sorted(inventory):
            stream.write(json.dumps(inventory[name])+"\n")
    result.update(finished_utc=recovery.now(), recovered_required_images=len(inventory), missing_required_image_ids=[int(Path(name).stem[3:]) for name in missing],
        duplicate_ids=[], remaining_decode_failures=sum(name not in inventory for name in {entry["basename"] for entry in failures}),
        extraction_decode_failures=failures, image_inventory=str(directory / "images.jsonl"), image_inventory_sha256=recovery.digest(directory / "images.jsonl"),
        status="TRAINING_IMAGES_FULLY_RECOVERED" if not missing else "REQUIRED_JPEGS_STILL_MISSING",
        acceptance_reason="Original JPEGs independently decoded and byte hashed; container validity is not required for training", image_transform="NONE")
    recovery.atomic_json(directory / "result.json", result)
    return result


def run(shard=None):
    families = recovery.requirements()
    results = []
    for position in ([shard] if shard is not None else (14, 16, 17)):
        results.append(rescue_one(position, families))
        current = [recovery.load(path) for path in FOLDER.glob("sa_*/result.json")]
        write_reports(sorted(current, key=lambda result: result["filename"]))
        print(results[-1]["filename"], results[-1]["status"], results[-1]["recovered_required_images"], "/", results[-1]["required_images"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "decode-inventory", "transfer-inventory"))
    parser.add_argument("--inventory")
    parser.add_argument("--output")
    parser.add_argument("--source-root")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.command == "decode-inventory":
        decode_inventory(args.inventory, args.output)
    elif args.command == "transfer-inventory":
        transfer_inventory(args.inventory, args.source_root, args.output)
    else:
        run(args.shard)
