"""Independent, bounded shard14 diagnostics; never download the Aber-r object again."""

from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tarfile
import zlib

from audit import EVIDENCE, RECOVERY, load, digest
from llava_copy_audit import fetch_range
from sa1b_recovery import REPO, SAM_ROOT, STATE, atomic_json, configure_client, now, requirements


RESULT = EVIDENCE / "sa1b-000014-forensics.json"
DIRECTORY = EVIDENCE / "sa1b-000014-forensics"
MD5 = "45e15f9ff5ded968abe4eb01b95be29a"
SHA256 = "b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3"
SIZE = 11213714588
RANGE_BYTES = 1024**2


def compression_signature(payload):
    if payload.startswith(b"\x1f\x8b"):
        return "gzip"
    if payload.startswith(b"BZh"):
        return "bzip2"
    if payload.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if payload.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    return "uncompressed_or_unknown"


def inspect_prefix(payload):
    compression = compression_signature(payload)
    if compression == "gzip":
        data = zlib.decompressobj(31).decompress(payload, 32 * 1024**2)
    elif compression == "uncompressed_or_unknown":
        data = payload
    else:
        return dict(compression=compression, parser_status="unsupported_prefix_compression", samples=[])
    position = 0
    members = 0
    samples = []
    while position + 512 <= len(data):
        header = data[position:position+512]
        if header == bytes(512):
            break
        try:
            member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        except tarfile.HeaderError as error:
            return dict(compression=compression, logical_prefix_failure_offset=position,
                        prefix_error=str(error), complete_prefix_members=members, samples=samples)
        end = position + 512 + member.size
        if end > len(data):
            break
        members += 1
        if member.isfile() and re.fullmatch(r"sa_\d+\.jpg", PurePosixPath(member.name).name) and len(samples) < 5:
            content = data[position+512:end]
            basename = PurePosixPath(member.name).name
            sample_path = DIRECTORY / "original-prefix-samples" / basename
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_bytes(content)
            samples.append(dict(basename=basename, logical_tar_offset=position, bytes=len(content),
                                sha256=hashlib.sha256(content).hexdigest(), path=str(sample_path),
                                use="forensic comparison only; not installed as recovered training data"))
        position = position + 512 + ((member.size + 511) // 512) * 512
    return dict(compression=compression, complete_prefix_members=members, samples=samples,
                partial_prefix_only=True, whole_archive_integrity_passed=False)


def local_diagnostics(path):
    commands = [("file", ["file", str(path)]), ("gnu_tar", ["tar", "-tvf", str(path)]),
                ("bsdtar", ["bsdtar", "-tf", str(path)]), ("7z", ["7z", "l", str(path)])]
    results = []
    for name, command in commands:
        binary = shutil.which(command[0])
        if not path.is_file() or binary is None:
            results.append(dict(tool=name, executable=binary,
                status="NOT_RUN_ORIGINAL_ARCHIVE_MISSING" if not path.is_file() else "TOOL_NOT_INSTALLED"))
            continue
        command[0] = binary
        output_path = DIRECTORY / (name + ".stdout.txt")
        error_path = DIRECTORY / (name + ".stderr.txt")
        with output_path.open("wb") as output, error_path.open("wb") as errors:
            result = subprocess.run(command, stdout=output, stderr=errors, timeout=1800)
        results.append(dict(tool=name, command=command, returncode=result.returncode,
                            stdout=str(output_path), complete_stderr=str(error_path)))
    if not path.is_file():
        results.insert(2, dict(tool="Python tarfile.open(...).getmembers()", status="NOT_RUN_ORIGINAL_ARCHIVE_MISSING"))
        return results
    archive = None
    try:
        archive = tarfile.open(path, "r:*")
        members = archive.getmembers()
        last_end = max((member.offset_data + ((member.size+511)//512)*512 for member in members), default=0)
        archive.fileobj.seek(last_end)
        end_blocks = archive.fileobj.read(1024)
        results.insert(2, dict(tool="Python tarfile.open(...).getmembers()", listed_members=len(members),
            two_zero_end_blocks_read=end_blocks == bytes(1024), logical_end_offset=last_end,
            status="MEMBER_PARSE_ONLY_NOT_FULL_INTEGRITY_ACCEPTANCE"))
    except Exception as error:
        results.insert(2, dict(tool="Python tarfile.open(...).getmembers()", error=str(error),
            listed_members=len(archive.members) if archive else 0,
            first_failure_logical_offset=archive.offset if archive else None))
    finally:
        if archive:
            archive.close()
    return results


def audit_sam_namespace(state, families, forensic_record):
    owned = set()
    inventories = []
    for row in state["shards"]:
        if row.get("extraction_status") != "complete":
            continue
        inventory = Path(row["image_inventory"])
        if digest(inventory) != row["image_inventory_sha256"]:
            raise ValueError("Complete original-image inventory hash mismatch")
        names = {json.loads(line)["basename"] for line in inventory.read_text().splitlines()}
        if len(names) != row["extracted_images"] or owned & names:
            raise ValueError("Incomplete or duplicate original-image inventory")
        owned.update(names)
        inventories.append(dict(filename=row["filename"], path=str(inventory), sha256=row["image_inventory_sha256"]))
    with os.scandir(SAM_ROOT) as entries:
        observed = {entry.name for entry in entries if re.fullmatch(r"sa_\d+\.(jpg|jpeg|png|webp)", entry.name)}
    if observed != owned:
        return dict(passed=False, reason="Live physical namespace differs from completed source inventories; do not replace completeness snapshot", physical_count=len(observed), inventory_count=len(owned))
    required = families["sam"]
    missing = sorted(relative for relative in required if PurePosixPath(relative).name not in observed)
    evidence = EVIDENCE / "sam-independent-namespace-audit.json"
    proof = dict(checked_utc=now(), passed=True, source_inventories=inventories,
        physical_image_count=len(observed), recovered_required_images=len(required)-len(missing), missing_required_images=len(missing),
        method="Exact physical image namespace equals all hash-verified completed atomic-extraction inventories; frozen required paths checked by name", decode_executed=False)
    atomic_json(evidence, proof)
    completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    prior_checked = completeness["checked_utc"]
    checked = now()
    completeness.setdefault("family_checked_utc", {family:prior_checked for family in completeness["families"]})
    completeness["family_checked_utc"]["sam"] = checked
    completeness["families"]["sam"] = dict(required_paths=len(required), recovered_paths=proof["recovered_required_images"],
        missing_paths=len(missing), duplicate_training_index_paths=0, missing_examples=missing[:12], audit_evidence=str(evidence))
    missing_total = sum(row["missing_paths"] for row in completeness["families"].values())
    missing_path = EVIDENCE / "sa1b-missing-training-paths.txt"
    missing_path.write_text("\n".join(missing)+("\n" if missing else ""))
    completeness.update(checked_utc=checked, filtered_records=1245901, missing_training_paths=missing_total,
        training_index_missing=missing_total, recovered_training_paths=1245901-missing_total,
        sam_extracted_inventory_images=len(owned), sam_original_image_files_present=len(observed),
        shard14_forensics=dict(status="QUARANTINED_ARCHIVE_INVALID", required_images=forensic_record["required_images"],
            missing_required_images=forensic_record["missing_required_images"], missing_ids_path=forensic_record["missing_ids_path"], evidence=str(RESULT)),
        decode_audit=dict(executed=False, passed=False), passed=False,
        update_scope="Independent SAM physical namespace / verified extraction inventory audit; COCO/LLaVA retain verified snapshots")
    completeness["required_sam_image_ids"].update(recovered_count=proof["recovered_required_images"], missing_paths_file=str(missing_path))
    atomic_json(EVIDENCE / "training-image-completeness-audit.json", completeness)
    atomic_json(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", completeness)
    return proof


def report(record):
    history = record.get("historical_listing", {})
    access = load(EVIDENCE / "sa1b-000014-second-mirror-access.json", {})
    lines = ["# SA-1B sa_000014 archive forensics", "", f"Updated UTC: {now()}",
             "Status: QUARANTINED_ARCHIVE_INVALID. Other shards continue independently.",
             "No smoke, 500/4868 training, full Aber-r shard14 redownload, or gate bypass is performed.", "",
             "## Preserved identity and evidence loss", "",
             f"Original repository/revision: {record.get('repo')} / {record.get('revision')}.",
             f"Pinned size: {SIZE}; historical actual MD5: `{MD5}`; historical actual SHA256: `{SHA256}`.",
             f"Known archive size modulo512: {SIZE % 512}.",
             f"Original complete local archive present: {record.get('original_present')}.",
             "The previous supervisor deleted each checksum-matching copy after GNU tar failure.",
             "No complete cache copy was found. Those deleted original bytes and discarded historical GNU stderr cannot be recreated from checksum records.",
             "Existing checklist, whole-file checksum results, failed attempts and last GNU stdout listing remain preserved.",
             "The resumed downloader now saves complete GNU stderr and preserves checksum-matching tar failures instead of deleting/retrying them.", "",
             "## Full-archive diagnostics", "",
             "| Tool | Status |", "| --- | --- |"]
    for row in record.get("local_diagnostics", []):
        lines.append(f"| {row['tool']} | {row.get('status', 'returncode='+str(row.get('returncode')))} |")
    lines.extend(["", f"Historical GNU listed entries: {history.get('listed_entries')}; JPEG names: {history.get('jpeg_names')}.",
                  f"Historical last listed member: `{history.get('last_member')}`.",
                  "First whole-archive failure offset/member: UNKNOWN; original file and original stderr are missing. Last listed member is not asserted to be the failed member.",
                  "Two 512-byte uncompressed tar zero end blocks: NOT VERIFIED. A compressed-file byte size modulo512 is not by itself a tar validity test.", "",
                  "## Bounded remote forensic evidence", "",
                  "Only up to1MiB head and1MiB tail of the immutable public Aber-r object may be fetched as forensic fragments; no complete original archive is downloaded.",
                  f"Range result: {record.get('remote_fragments', {}).get('status', 'pending')}.",
                  f"Detected header signature: {record.get('remote_fragments', {}).get('compression', 'not available')}.",
                  "Fragments/prefix samples are not full archive acceptance and are not installed in the training tree.", "",
                  "## Required shard14 paths", "",
                  f"ID range inferred from verified neighboring shard inventories: {record.get('id_range')}.",
                  f"Required paths in frozen training index: {record.get('required_images', 'pending')}.",
                  f"Current shard14 required paths missing: {record.get('missing_required_images', 'pending')}.",
                  f"Exact missing image IDs: `{record.get('missing_ids_path', 'pending')}`.",
                  "The inferred neighbor gap is cross-checked against the surviving GNU name list; it is not a replacement split.", "",
                  "## Second original mirror", "",
                  "Candidate: k-m-irfan/sa1b; filename: dataset/sa_000014.tar.",
                  f"Access probe status: {access.get('status', 'pending')}; existing HF token present: {access.get('token_present')}.",
                  "Authenticated official-HF requests timed out; anonymous relay requests returned401. These do not establish whether the existing token has authorized access.",
                  "User action: confirm the Hugging Face access conditions have been accepted for this repository and the server's token has read access; official HF connectivity must also work.",
                  "No private HF token is sent to the third-party API relay. No gated file is downloaded without an authenticated official permission check.",
                  "No second archive has been obtained, so cases A/B and full second-file hash/member/JPEG/decode comparisons remain pending (case C: source unavailable).", "",
                  "## Acceptance remains fail-closed", "",
                  "An authorized independent archive must be stored separately, with size/MD5/SHA256, full tar validation, all members/image IDs, every JPEG decode, all required-path matches and available original-sample byte comparisons recorded.",
                  "No partial listing or non-GNU parser result alone is treated as complete recovery. No repacked/resized substitute is used."])
    samples = record.get("remote_fragments", {}).get("prefix", {}).get("samples", [])
    lines.extend(["", "## Available original-byte comparison samples", "",
                  "Samples remain forensic-only, outside the training tree.",
                  "| Image | Encoded-byte SHA256 | JPEG decode |", "| --- | --- | --- |"])
    for sample in samples:
        lines.append(f"| {sample['basename']} | `{sample['sha256']}` | {sample.get('jpeg_decode_audit', {}).get('passed', 'not run')} |")
    (RECOVERY / "SA1B_000014_FORENSICS.md").write_text("\n".join(lines)+"\n")


def checkpoint(record):
    atomic_json(RESULT, record)
    report(record)


def main():
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    DIRECTORY.mkdir(parents=True, exist_ok=True)
    state = load(STATE)
    row = state["shards"][14]
    path = Path(row["archive"])
    record = dict(started_utc=now(), repo=REPO, revision=state["revision"], original_archive=str(path),
        original_present=path.is_file(), original_size=SIZE, original_size_modulo512=SIZE % 512,
        checksum_md5=MD5, checksum_sha256=SHA256, phase="local_evidence", full_original_redownload=False,
        first_whole_archive_failure_offset=None, first_whole_archive_failure_member=None,
        two_uncompressed_zero_end_blocks_verified=False, archive_accepted=False)
    checkpoint(record)
    record["local_diagnostics"] = local_diagnostics(path)
    listing = EVIDENCE / "sa1b-shards/sa_000014.tar.tar-members.txt"
    names = listing.read_text().splitlines()
    jpeg_ids = {int(PurePosixPath(name).stem.removeprefix("sa_")) for name in names
                if re.fullmatch(r"sa_\d+\.jpg", PurePosixPath(name).name)}
    record["historical_listing"] = dict(path=str(listing), sha256=digest(listing), listed_entries=len(names),
        jpeg_names=len(jpeg_ids), last_member=names[-1], minimum_jpeg_id=min(jpeg_ids), maximum_jpeg_id=max(jpeg_ids))
    record["historical_listing"]["extensions"] = dict(Counter(PurePosixPath(name).suffix.lower() or "directory_or_other" for name in names))
    checkpoint(record)
    adjacent = []
    for position in [13, 15]:
        inventory = Path(state["shards"][position]["image_inventory"])
        if digest(inventory) != state["shards"][position]["image_inventory_sha256"]:
            raise ValueError("Neighboring original-image inventory identity mismatch")
        adjacent.append({int(json.loads(line)["basename"].removeprefix("sa_").removesuffix(".jpg"))
                         for line in inventory.read_text().splitlines()})
    minimum = max(adjacent[0])+1
    maximum = min(adjacent[1])-1
    if not jpeg_ids <= set(range(minimum, maximum+1)):
        raise ValueError("Historical original names contradict neighboring shard ID gap")
    record["id_range"] = dict(minimum=minimum, maximum=maximum, inferred_from="verified original shards13/15", historical_names_within_gap=True)
    families = requirements()
    required = sorted(path for path in families["sam"]
                      if minimum <= int(PurePosixPath(path).stem.removeprefix("sa_")) <= maximum)
    record.update(required_images=len(required), phase="required_path_existence_audit")
    checkpoint(record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        present = list(executor.map(lambda relative: (SAM_ROOT / PurePosixPath(relative).name).is_file(), required))
    missing = [relative for relative, exists in zip(required, present) if not exists]
    ids_path = DIRECTORY / "required-missing-image-ids.txt"
    ids_path.write_text("\n".join(PurePosixPath(relative).stem.removeprefix("sa_") for relative in missing)+("\n" if missing else ""))
    paths_path = DIRECTORY / "required-missing-image-paths.txt"
    paths_path.write_text("\n".join(missing)+("\n" if missing else ""))
    record.update(missing_required_images=len(missing), recovered_required_images=len(required)-len(missing),
        missing_ids_path=str(ids_path), missing_paths_path=str(paths_path), missing_ids_sha256=digest(ids_path),
        phase="bounded_public_object_fragments")
    checkpoint(record)
    record["sam_namespace_snapshot"] = audit_sam_namespace(state, families, record)
    checkpoint(record)
    fragment_record = dict(status="pending", maximum_transfer_bytes=2*RANGE_BYTES, not_a_full_archive=True)
    record["remote_fragments"] = fragment_record
    try:
        configure_client(state["endpoint"])
        metadata = get_hf_file_metadata(hf_hub_url(REPO,row["repo_path"],repo_type="dataset",
            revision=state["revision"],endpoint=state["endpoint"]),token=False,timeout=10)
        if metadata.size != SIZE or metadata.etag != SHA256:
            raise ValueError("Immutable public object metadata differs from historical original identity")
        session = requests.Session()
        head = fetch_range(session,metadata.location,0,RANGE_BYTES)
        tail = fetch_range(session,metadata.location,SIZE-RANGE_BYTES,RANGE_BYTES)
        for name,payload in [("head.bin",head),("tail.bin",tail)]:
            (DIRECTORY/name).write_bytes(payload)
        fragment_record.update(status="saved",received_bytes=len(head)+len(tail),
            head_sha256=hashlib.sha256(head).hexdigest(),tail_sha256=hashlib.sha256(tail).hexdigest(),
            source_lfs_identity=metadata.etag,compression=compression_signature(head))
        if fragment_record["compression"] == "gzip":
            crc, size_mod32 = struct.unpack("<II",tail[-8:])
            fragment_record.update(gzip_footer_crc32=crc,gzip_footer_uncompressed_size_mod2pow32=size_mod32,
                                  footer_is_not_full_crc_validation=True)
        fragment_record["prefix"] = inspect_prefix(head)
    except Exception as error:
        fragment_record.update(status="failed",error_type=type(error).__name__,error=str(error))
    record.update(phase="diagnosed_with_missing_original_and_second_source_blocker", finished_utc=now())
    checkpoint(record)
    print(json.dumps(dict(phase=record["phase"],required_images=record["required_images"],
        missing_required_images=record["missing_required_images"],remote_fragments=fragment_record)),flush=True)


if __name__ == "__main__":
    main()
