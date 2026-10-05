"""Recover only the original first51 SA-1B shards; never start formal training."""

import argparse
from collections import Counter
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

from audit import ASSETS, EVIDENCE, RECOVERY, RECORDS_SHA, ROOT, RUNTIME, digest, load


REPO = "Aber-r/SA-1B_backup"
NAMES = [f"sa_{position:06d}.tar" for position in range(51)]
STATE = RECOVERY / "SA1B_SHARDS.json"
SAM_ROOT = ASSETS / "training/ShareGPT4V/sam/images"
ARCHIVES = ASSETS / "downloads/sa1b"
EXAMPLES = {"sa_000000.tar": "78d0f487c735a3de86ae6e1b0ff24ea9",
            "sa_000001.tar": "c69be32127c9b4a6dc89ff95b036e44d",
            "sa_000050.tar": "5a2ea71c804dceed1b00a8ba6dc659e7"}
QUARANTINED = "QUARANTINED_ARCHIVE_INVALID"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parse_checklist(text):
    checksums = {}
    for line in text.splitlines():
        names = re.findall(r"\bsa_\d{6}\.tar\b", line)
        values = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])", line)
        if not names:
            continue
        if len(names) != 1 or len(values) != 1 or names[0] in checksums:
            raise ValueError("Ambiguous or duplicate checklist entry: " + line)
        checksums[names[0]] = values[0].lower()
    if set(checksums) != {f"sa_{position:06d}.tar" for position in range(1000)}:
        raise ValueError("Checklist must identify exactly the original 1000 shards")
    for name, expected in EXAMPLES.items():
        if checksums[name] != expected:
            raise ValueError("Checklist differs from supplied historical example: " + name)
    return checksums


def select_revision(api, commits, history):
    for commit in commits:
        revision = commit.commit_id
        try:
            files = api.list_repo_files(REPO, repo_type="dataset", revision=revision)
        except Exception as error:
            history.append(dict(revision=revision, checked=False, error=str(error)))
            continue
        mapping = {}
        for filename in files:
            basename = PurePosixPath(filename).name
            if basename in NAMES or basename == "checklist.chk":
                if basename in mapping:
                    raise ValueError("Ambiguous shard/checklist basename in revision " + revision)
                mapping[basename] = filename
        missing = [name for name in NAMES if name not in mapping]
        history.append(dict(revision=revision, checked=True, files=len(files),
                            required_shards_present=51-len(missing), missing=missing,
                            checklist_present="checklist.chk" in mapping))
        if not missing and "checklist.chk" in mapping:
            return revision, mapping
    return None, None


def configure_client(endpoint=None):
    import requests
    from huggingface_hub import configure_http_backend

    class BoundedSession(requests.Session):
        def request(self, method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (10, 45)
            if endpoint and endpoint != "https://huggingface.co" and url.startswith("https://huggingface.co/api/"):
                url = endpoint.rstrip("/") + url.removeprefix("https://huggingface.co")
            return super().request(method, url, **kwargs)

    configure_http_backend(backend_factory=BoundedSession)


def disk_check(total_bytes):
    output = subprocess.check_output(["df", "-h", str(ROOT)]).decode()
    (EVIDENCE / "sa1b-disk-before.txt").write_text(output)
    usage = shutil.disk_usage(ROOT)
    required = 3 * total_bytes + 64 * 1024**3
    record = dict(checked_utc=now(), df_h=output, free_bytes=usage.free,
                  planned_tar_bytes=total_bytes, minimum_free_bytes=required,
                  reservation="Three times total tar size (archives, original images, resume/cache headroom) +64GiB for other assets/reserve",
                  sufficient=usage.free >= required)
    atomic_json(EVIDENCE / "sa1b-disk-audit.json", record)
    if not record["sufficient"]:
        raise RuntimeError("Insufficient disk space; no shard download started")
    return record


def discover(endpoint=None):
    from huggingface_hub import HfApi, hf_hub_download

    configure_client()
    discovery = dict(repo=REPO, started_utc=now(), endpoint_attempts=[], history=[], candidates=[])
    atomic_json(EVIDENCE / "sa1b-revision-discovery.json", discovery)
    endpoints = [endpoint] if endpoint else ["https://huggingface.co", "https://hf-mirror.com"]
    for address in endpoints:
        configure_client(address)
        api = HfApi(endpoint=address, token=False)
        try:
            references = api.list_repo_refs(REPO, repo_type="dataset")
            refs = [dict(name=reference.name, revision=reference.target_commit)
                    for reference in references.branches + references.tags]
            if not refs:
                raise ValueError("Repository has no branch/tag references")
            seen = {}
            for reference in refs:
                for commit in api.list_repo_commits(REPO, repo_type="dataset", revision=reference["revision"]):
                    seen[commit.commit_id] = commit
            commits = sorted(seen.values(), key=lambda commit: commit.created_at, reverse=True)
            discovery.update(refs=refs, history=[dict(revision=commit.commit_id,
                created_at=commit.created_at.isoformat(), title=commit.title) for commit in commits], endpoint=address)
            revision, mapping = select_revision(api, commits, discovery["candidates"])
            atomic_json(EVIDENCE / "sa1b-revision-discovery.json", discovery)
            if revision is None:
                discovery["status"] = "NO_COMPLETE_REVISION" if all(row["checked"] for row in discovery["candidates"]) else "REVISION_CHECK_INCOMPLETE"
                atomic_json(EVIDENCE / "sa1b-revision-discovery.json", discovery)
                raise RuntimeError(discovery["status"] + "; no shards will be downloaded from this source")
            metadata = api.get_paths_info(REPO, paths=[mapping[name] for name in NAMES], revision=revision, repo_type="dataset")
            details = {entry.path: entry for entry in metadata}
            directory = ARCHIVES / revision
            checklist = Path(hf_hub_download(REPO, mapping["checklist.chk"], repo_type="dataset",
                revision=revision, endpoint=address, local_dir=directory, token=False))
            checksums = parse_checklist(checklist.read_text())
            shards = []
            for name in NAMES:
                entry = details[mapping[name]]
                lfs = entry.lfs
                shards.append(dict(filename=name, repo_path=mapping[name], expected_size_bytes=entry.size,
                    expected_checklist_checksum=checksums[name], checksum_algorithm_verified=False,
                    expected_lfs_sha256=lfs.sha256 if lfs else None,
                    archive=str(directory / mapping[name]), download_status="pending",
                    extraction_status="pending", attempts=[]))
            state = dict(repo=REPO, revision=revision, endpoint=address, discovered_utc=now(),
                required_files=NAMES, checklist=dict(path=str(checklist), sha256=digest(checklist),
                entries=len(checksums), candidate_algorithm="MD5", algorithm_verified_by_actual_file=False,
                verification_note="32 hexadecimal characters and examples are not actual-file algorithm verification"),
                downloader=dict(library="huggingface_hub", version="0.36.0", hf_xet_version="1.1.10",
                                xet_preferred=True, local_dir_resume=True), shards=shards,
                formal_training_started=False, smoke_started=False)
            state["disk"] = disk_check(sum(shard["expected_size_bytes"] for shard in shards))
            atomic_json(STATE, state)
            discovery.update(status="COMPLETE_REVISION_SELECTED", selected_revision=revision, finished_utc=now())
            atomic_json(EVIDENCE / "sa1b-revision-discovery.json", discovery)
            print(json.dumps(dict(repo=REPO, revision=revision, candidates_checked=len(discovery["candidates"]),
                                 historical_revisions=len(commits), total_tar_bytes=state["disk"]["planned_tar_bytes"])), flush=True)
            return state
        except Exception as error:
            discovery["endpoint_attempts"].append(dict(endpoint=address, error=str(error)))
            atomic_json(EVIDENCE / "sa1b-revision-discovery.json", discovery)
            if discovery.get("status") == "NO_COMPLETE_REVISION":
                raise
    raise RuntimeError("Revision discovery failed; inspect sa1b-revision-discovery.json")


def archive_hashes(path):
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as archive:
        for chunk in iter(lambda: archive.read(16 * 1024**2), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return dict(md5=md5.hexdigest(), sha256=sha256.hexdigest())


def verify_archive(path, row):
    size = Path(path).stat().st_size
    hashes = archive_hashes(path)
    if not size or size != row["expected_size_bytes"]:
        raise ValueError(f"Archive size differs from pinned repository metadata: expected {row['expected_size_bytes']}, observed {size}")
    if hashes["md5"] != row["expected_checklist_checksum"]:
        raise ValueError(f"Actual-file MD5 differs from original checklist: expected {row['expected_checklist_checksum']}, observed {hashes['md5']}")
    if row.get("expected_lfs_sha256") and hashes["sha256"] != row["expected_lfs_sha256"]:
        raise ValueError(f"Archive SHA256 differs from pinned LFS identity: expected {row['expected_lfs_sha256']}, observed {hashes['sha256']}")
    return dict(size_bytes=size, observed_md5=hashes["md5"], observed_sha256=hashes["sha256"],
                md5_matches_checklist=True, checksum_algorithm_verified=True, verified_utc=now())


def download_shard(state, original):
    if original.get("download_status") == QUARANTINED:
        return json.loads(json.dumps(original))
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.utils._runtime import is_xet_available

    row = json.loads(json.dumps(original))
    evidence = EVIDENCE / "sa1b-shards" / (row["filename"] + ".json")
    path = Path(row["archive"])
    for attempt in range(1, 4):
        row["download_status"] = "downloading"
        event = dict(attempt=attempt, started_utc=now(), revision=state["revision"],
                     endpoint=state["endpoint"], library="huggingface_hub.hf_hub_download",
                     xet_enabled=os.environ.get("HF_HUB_DISABLE_XET") != "1")
        row["attempts"].append(event)
        atomic_json(evidence, row)
        try:
            metadata = get_hf_file_metadata(hf_hub_url(REPO, row["repo_path"], repo_type="dataset",
                revision=state["revision"], endpoint=state["endpoint"]), token=False)
            event.update(xet_metadata_present=metadata.xet_file_data is not None,
                xet_package_available=is_xet_available(),
                sdk_selected_transport="hf_xet" if metadata.xet_file_data and is_xet_available() else "SDK resumable HTTP")
            atomic_json(evidence, row)
            path = Path(hf_hub_download(REPO, row["repo_path"], repo_type="dataset", token=False,
                revision=state["revision"], endpoint=state["endpoint"], local_dir=path.parents[len(PurePosixPath(row["repo_path"]).parts)-1],
                force_download=bool(row.pop("force_redownload", False))))
            try:
                row.update(verify_archive(path, row))
                listing = EVIDENCE / "sa1b-shards" / (row["filename"] + ".tar-members.txt")
                errors = EVIDENCE / "sa1b-shards" / (row["filename"] + ".tar-stderr.txt")
                with listing.open("w") as output, errors.open("wb") as stderr:
                    result = subprocess.run(["tar", "-tf", str(path)], stdout=output, stderr=stderr)
                if result.returncode:
                    event.update(finished_utc=now(), passed=False, checksum_matched=True,
                                 archive_preserved=True, same_object_retry_disabled=True,
                                 tar_returncode=result.returncode, tar_stderr=str(errors))
                    row.update(download_status=QUARANTINED, tar_integrity_passed=False,
                               tar_listing=str(listing), tar_stderr=str(errors), quarantined_utc=now(),
                               quarantine_reason="Pinned size/MD5/SHA256 match, but GNU tar integrity check fails")
                    atomic_json(evidence, row)
                    return row
                row.update(tar_integrity_passed=True, tar_listing=str(listing), download_status="verified")
            except ValueError as error:
                event.update(integrity_error=str(error), corrupt_shard_deleted=True)
                path.unlink()
                row["force_redownload"] = True
                raise
            event.update(finished_utc=now(), passed=True)
            atomic_json(evidence, row)
            return row
        except Exception as error:
            event.update(finished_utc=now(), passed=False, error_type=type(error).__name__, error=str(error))
            row["download_status"] = "retry_pending" if attempt < 3 else "failed"
            atomic_json(evidence, row)
            print(row["filename"], row["download_status"], str(error), flush=True)
            if attempt < 3:
                time.sleep(10)
    return row


def audit_structure(path):
    import tarfile

    extensions = Counter()
    types = Counter()
    names = set()
    sample = []
    image_bytes = 0
    image_members = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Archive member escapes original relative layout")
            if not member.isfile() and not member.isdir():
                raise ValueError("Unexpected archive link/device: " + member.name)
            types["file" if member.isfile() else "directory"] += 1
            if len(sample) < 24:
                sample.append(dict(name=member.name, size=member.size, type="file" if member.isfile() else "directory"))
            if member.isfile():
                suffix = relative.suffix.lower()
                extensions[suffix] += 1
                if suffix in (".jpg", ".jpeg", ".png", ".webp"):
                    if not re.fullmatch(r"sa_\d+\.(jpg|jpeg|png|webp)", relative.name):
                        raise ValueError("Unexpected original image basename: " + member.name)
                    if relative.name in names:
                        raise ValueError("Duplicate image basename within archive: " + relative.name)
                    names.add(relative.name)
                    image_members += 1
                    image_bytes += member.size
                elif suffix != ".json":
                    raise ValueError("Unrecognized non-image structure; inspect before skipping: " + member.name)
    if not image_members:
        raise ValueError("Archive contains no original SA-1B images")
    return dict(member_types=dict(types), extensions=dict(extensions), sample=sample,
                image_members=image_members, image_bytes=image_bytes,
                skipped_json=extensions.get(".json", 0), policy="Only regular original images; JSON mask annotations not extracted")


def extract_shard(row, ownership, destination_root=None):
    destination_root = Path(destination_root or SAM_ROOT)
    import tarfile

    inventory = EVIDENCE / "sa1b-shards" / (row["filename"] + ".images.jsonl")
    structure = audit_structure(row["archive"])
    with Path(row["archive"]).open("rb") as archive:
        structure["compression"] = "gzip" if archive.read(2) == b"\x1f\x8b" else "tarfile autodetection"
    row["archive_structure"] = structure
    atomic_json(EVIDENCE / "sa1b-shards" / (row["filename"] + ".structure.json"), structure)
    if shutil.disk_usage(ROOT).free < structure["image_bytes"] + 64 * 1024**3:
        raise RuntimeError("Insufficient space for audited original image bytes; extraction not started")
    destination_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination_root).free < structure["image_bytes"] + 64 * 1024**3:
        raise RuntimeError("Insufficient space for original-image extraction staging")
    temporary_inventory = inventory.with_suffix(".jsonl.part")
    extracted = 0
    duplicates = []
    row["duplicate_paths"] = duplicates
    with temporary_inventory.open("w") as output, tarfile.open(row["archive"], "r:*") as archive:
        for member in archive:
            basename = PurePosixPath(member.name).name
            if not member.isfile() or PurePosixPath(basename).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if basename in ownership and ownership[basename] != row["filename"]:
                duplicates.append(dict(basename=basename, prior_shard=ownership[basename], shard=row["filename"]))
                raise ValueError("Cross-shard duplicate image basename; refusing overwrite: " + basename)
            target = destination_root / basename
            if target.is_symlink():
                raise ValueError("Refusing to write through existing image symlink: " + basename)
            temporary = target.with_name("." + basename + ".sa1b-part")
            content_sha = hashlib.sha256()
            copied = 0
            with archive.extractfile(member) as source, temporary.open("wb") as destination:
                for chunk in iter(lambda: source.read(4 * 1024**2), b""):
                    content_sha.update(chunk)
                    destination.write(chunk)
                    copied += len(chunk)
            if copied != member.size:
                temporary.unlink()
                raise ValueError("Incomplete image extraction: " + basename)
            observed = content_sha.hexdigest()
            if target.exists():
                if target.stat().st_size != copied or digest(target) != observed:
                    temporary.unlink()
                    raise ValueError("Existing training image differs; left untouched: " + basename)
                temporary.unlink()
            else:
                temporary.replace(target)
            ownership[basename] = row["filename"]
            output.write(json.dumps(dict(basename=basename, source_member=member.name,
                shard=row["filename"], size_bytes=copied, sha256=observed)) + "\n")
            extracted += 1
    temporary_inventory.replace(inventory)
    row.update(extraction_status="complete", extracted_images=extracted,
               image_inventory=str(inventory), image_inventory_sha256=digest(inventory),
               skipped_mask_json=structure["skipped_json"], duplicate_paths=duplicates,
               extraction_finished_utc=now(), image_transform="NONE: original basename and raw bytes")
    atomic_json(EVIDENCE / "sa1b-shards" / (row["filename"] + ".json"), row)


def requirements():
    records = RUNTIME / "data_index/records.jsonl"
    if digest(records) != RECORDS_SHA:
        raise ValueError("Training index identity differs from historical frozen index")
    families = {"sam": set(), "coco": set(), "llava": set()}
    count = 0
    duplicate_references = 0
    for line in records.open():
        path = json.loads(line)["image"]
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe training image path")
        family = relative.parts[0]
        if family not in families:
            raise ValueError("Unrecognized frozen training image family: " + family)
        duplicate_references += path in families[family]
        families[family].add(path)
        count += 1
    assert count == 1245901 and len(families["sam"]) == 569486
    assert not duplicate_references
    path = EVIDENCE / "sa1b-required-training-paths.txt"
    if not path.is_file():
        path.write_text("\n".join(sorted(families["sam"])) + "\n")
    return families


def image_completeness(families, ownership=None, decode=False):
    root = ASSETS / "training/ShareGPT4V"
    directory_cache = {}
    results = {}
    all_present = []
    for family, paths in families.items():
        present = 0
        missing = []
        for relative in sorted(paths):
            path = PurePosixPath(relative)
            parent = str(path.parent)
            if parent not in directory_cache:
                directory = root / parent
                directory_cache[parent] = ({entry.name for entry in os.scandir(directory) if entry.is_file()}
                                           if directory.is_dir() else set())
            if path.name in directory_cache[parent]:
                present += 1
                if decode:
                    all_present.append(root / relative)
            else:
                missing.append(relative)
        results[family] = dict(required_paths=len(paths), recovered_paths=present, missing_paths=len(missing),
                              duplicate_training_index_paths=0, missing_examples=missing[:12])
        if family == "sam":
            (EVIDENCE / "sa1b-missing-training-paths.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    state = load(STATE, {})
    duplicates = [entry for row in state.get("shards", []) for entry in row.get("duplicate_paths", [])]
    inventory_count = len(ownership) if ownership is not None else sum(
        row.get("extracted_images", 0) for row in state.get("shards", []) if row.get("extraction_status") == "complete")
    record = dict(checked_utc=now(), training_index_sha256=RECORDS_SHA, training_records=1245901,
        filtered_records=1245901,
        families=results, recovered_training_paths=sum(row["recovered_paths"] for row in results.values()),
        missing_training_paths=sum(row["missing_paths"] for row in results.values()),
        training_index_missing=sum(row["missing_paths"] for row in results.values()),
        duplicate_training_index_paths=0, sam_duplicate_source_paths=duplicates,
        sam_extracted_inventory_images=inventory_count,
        sam_original_image_files_present=sum(bool(re.fullmatch(r"sa_\d+\.(jpg|jpeg|png|webp)", name))
            for name in directory_cache.get("sam/images", set())),
        decode_audit=dict(executed=False, passed=False), passed=False)
    sam_ids = [int(PurePosixPath(path).stem.removeprefix("sa_")) for path in families["sam"]]
    record["required_sam_image_ids"] = dict(count=len(sam_ids), minimum=min(sam_ids), maximum=max(sam_ids),
        source_path_list=str(EVIDENCE / "sa1b-required-training-paths.txt"),
        recovered_count=results["sam"]["recovered_paths"], missing_paths_file=str(EVIDENCE / "sa1b-missing-training-paths.txt"))
    if decode and record["missing_training_paths"] == 0:
        def check_image(path):
            from PIL import Image

            try:
                with Image.open(path) as image:
                    image.load()
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("Empty image dimensions")
                return None
            except Exception as error:
                return dict(path=str(path), error=str(error))

        failures = []
        checked = 0
        policy = load(EVIDENCE / "recovery-operation-policy.json", {})
        decode_workers = min(64, os.cpu_count() or 1, max(1, int(policy.get("full_decode_workers", 16))))
        with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as executor:
            for offset in range(0, len(all_present), 2048):
                batch = all_present[offset:offset+2048]
                failures.extend(result for result in executor.map(check_image, batch) if result is not None)
                checked += len(batch)
                atomic_json(EVIDENCE / "training-image-decode-progress.json", dict(checked=checked,
                            required=1245901, decode_failures=len(failures), checked_utc=now()))
        record["decode_audit"] = dict(executed=True, checked_images=checked, workers=decode_workers, failures=failures,
                                      passed=checked == 1245901 and not failures)
    record["passed"] = record["missing_training_paths"] == 0 and record["decode_audit"]["passed"] and not duplicates
    atomic_json(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", record)
    atomic_json(EVIDENCE / "training-image-completeness-audit.json", record)
    return record


def training_shard_ready(row):
    normal = (row.get("download_status") == "verified" and row.get("extraction_status") == "complete"
        and row.get("training_images_installed") is not False
        and row.get("checksum_algorithm_verified", False) and row.get("tar_integrity_passed", False)
        and row.get("observed_md5") == row.get("expected_checklist_checksum"))
    if normal:
        return True
    rescue = row.get("required_jpeg_rescue", {})
    if not (row.get("download_status") == QUARANTINED and rescue.get("status") == "TRAINING_IMAGES_FULLY_RECOVERED"
        and rescue.get("training_index_sha256") == RECORDS_SHA and rescue.get("required_images", 0) > 0
        and rescue.get("required_images") == rescue.get("recovered_required_images")
        and not rescue.get("missing_required_image_ids", ["unknown"])
        and rescue.get("remaining_decode_failures", 1) == 0 and not rescue.get("duplicate_ids", ["unknown"])
        and rescue.get("image_transform") == "NONE" and rescue.get("original_archive_modified") is False):
        return False
    inventory = Path(rescue.get("image_inventory", ""))
    return inventory.is_file() and digest(inventory) == rescue.get("image_inventory_sha256")


def report(state=None, completeness=None):
    state = state or load(STATE, {})
    discovery = load(EVIDENCE / "sa1b-revision-discovery.json", {})
    completeness = completeness or load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
    sam = completeness.get("families", {}).get("sam", {})
    duplicates = [entry for row in state.get("shards", []) for entry in row.get("duplicate_paths", [])]
    mirror_verified = (len(state.get("shards", [])) == 51 and
        state.get("checklist", {}).get("algorithm_verified_by_actual_file", False) and not duplicates and
        all(training_shard_ready(row) for row in state.get("shards", []))
        and sam.get("recovered_paths") == 569486 and sam.get("missing_paths") == 0)
    atomic_json(EVIDENCE / "sa1b-mirror-audit.json", dict(passed=mirror_verified,
        repo=REPO, revision=state.get("revision"), recovered_required_sam_images=sam.get("recovered_paths", 0),
        missing_required_sam_images=sam.get("missing_paths", 569486), duplicate_source_paths=duplicates,
        jpeg_rescued_shards=[row["filename"] for row in state.get("shards", [])
            if row.get("training_images_status") == "TRAINING_IMAGES_FULLY_RECOVERED"],
        container_exception="Original JPEG bytes with frozen-index coverage, per-image SHA256 and decode proof may pass despite invalid original container",
        verified_extracted_shards=sum(row.get("download_status") == "verified" and row.get("extraction_status") == "complete"
                                     for row in state.get("shards", [])), checked_utc=now()))
    if duplicates:
        completeness.update(sam_duplicate_source_paths=duplicates, passed=False)
        atomic_json(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", completeness)
        atomic_json(EVIDENCE / "training-image-completeness-audit.json", completeness)
    lines = ["# SA-1B mirror recovery", "", f"Repository: `{REPO}`.",
        f"Fixed revision: `{state.get('revision', 'NOT SELECTED')}`.",
        f"Discovery: {discovery.get('status', 'IN PROGRESS')}; revisions listed: {len(discovery.get('history', []))}; candidate file lists checked: {len(discovery.get('candidates', []))}.",
        "Main was not assumed complete. HfApi.list_repo_commits and list_repo_files were used on immutable revisions.",
        f"API/download endpoint: `{state.get('endpoint', 'pending')}`. End-to-end TLS verification remains enabled.",
        "", f"Checklist entries: {state.get('checklist', {}).get('entries', 'pending')}.",
        f"Checklist SHA256: `{state.get('checklist', {}).get('sha256', 'pending')}`.",
        f"MD5 algorithm verified against an actual downloaded archive: {state.get('checklist', {}).get('algorithm_verified_by_actual_file', False)}.",
        "32-character checksums and matching examples alone are not reported as MD5 file verification.",
        "", "## Capacity and transfer", "",
        f"Free space at planning: {state.get('disk', {}).get('free_bytes', 'pending')} bytes.",
        f"Total pinned tar sizes: {state.get('disk', {}).get('planned_tar_bytes', 'pending')} bytes.",
        f"Reserved minimum free capacity: {state.get('disk', {}).get('minimum_free_bytes', 'pending')} bytes.",
        "Official huggingface_hub0.36.0 / hf_xet1.1.10 in isolated `.download-venv`; training dependencies unchanged.",
        f"HF parallelism: {state.get('download_concurrency', 2)}; ODL parallelism: {state.get('odl_download_concurrency', 0)}. SDK partials/cache support resumption.",
        "Per-attempt provenance records whether server Xet metadata is present and which SDK transport is selected.",
        "Mirror API pagination links pointing to unreachable huggingface.co are routed through the same verified-TLS mirror endpoint.",
        "Normal shards require nonzero exact size, actual MD5, pinned LFS SHA256 and tar -tf before ingestion.",
        "Quarantined shards use independently decoded/byte-hashed required JPEG rescue on copies; original containers are never declared valid by that exception.",
        "Checksum mismatches delete only this job's corrupt tar and retry up to three times.",
        "Checksum-matching tar failures are QUARANTINED_ARCHIVE_INVALID, preserved, skipped, and never retried from the same object.",
        f"Resumed recovery policy: {state.get('recovery_policy', 'historical recovery job')}",
        "Shard14 independent diagnostics/access status: SA1B_000014_FORENSICS.md and evidence/sa1b-000014-forensics.json.",
        "Original image bytes/basenames are retained. Mask JSON is skipped only after full structure audit.",
        "No resize, recompression, rename, format conversion or processed-dataset substitution.",
        "COCO/LLaVA existing download jobs are retained; this job never redownloads those archives.",
        "", "## Shards", "", "| Filename | Checklist checksum | Actual MD5 | Download | Extraction | Images |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in state.get("shards", []):
        lines.append(f"| {row['filename']} | `{row['expected_checklist_checksum']}` | `{row.get('observed_md5', 'NOT VERIFIED')}` | {row['download_status']} | {row['extraction_status']} | {row.get('extracted_images', 0)} |")
    lines.extend(["", "## Training index completeness", "",
        f"Required SAM paths: {sam.get('required_paths', 569486)}.",
        f"Recovered required SAM paths: {sam.get('recovered_paths', 'not audited')}.",
        f"Missing required SAM paths: {sam.get('missing_paths', 'not audited')}.",
        f"Duplicate index paths: {completeness.get('duplicate_training_index_paths', 'not audited')}.",
        f"Duplicate source basenames: {len(duplicates)}.",
        f"Extracted image inventory count: {completeness.get('sam_extracted_inventory_images', 0)}.",
        f"Full training index missing paths: {completeness.get('missing_training_paths', 'not audited')} /1245901.",
        f"Full original-image decode audit passed: {completeness.get('decode_audit', {}).get('passed', False)}.",
        "Missing SAM path list: `evidence/sa1b-missing-training-paths.txt`; per-shard raw-byte inventory/provenance: `evidence/sa1b-shards/`.",
        f"All51 shard training-image protocols and569486 required SAM images verified: {mirror_verified}.",
        "No training is started. The latest operation policy also forbids smoke until separately authorized.",
        "Archive provenance, candidate revisions and failures are recorded even when recovery is incomplete."])
    forensic = load(EVIDENCE / "sa1b-000014-forensics.json", {})
    access = load(EVIDENCE / "sa1b-000014-second-mirror-access.json", {})
    lines.extend(["", "## Independent shard14 repair", "",
        "Status: QUARANTINED_ARCHIVE_INVALID; current rescue coverage is reported below.",
        "Latest minimum-wall-clock instruction explicitly authorizes independent full ODL14 and conditional17 forensic downloads, even when the object hash matches.",
        "Completed shards are reused; normal source queues continue independently. No training or smoke is authorized.",
        f"Complete original forensic archive present: {forensic.get('original_present', 'not checked')}.",
        "The prior supervisor deleted checksum-matching tar-failure copies; its historical GNU stderr was not saved.",
        f"Historical shard14 forensic snapshot required images: {forensic.get('required_images', 'pending')}; historical missing: {forensic.get('missing_required_images', 'pending')}. Current JPEG rescue coverage is listed below.",
        f"Second source k-m-irfan/sa1b access status: {access.get('status', 'pending')}.",
        "Official authenticated access timed out; anonymous relay401 does not prove the existing token is unauthorized.",
        "Confirm repository access conditions/token read permission and official HF connectivity; no gate bypass or substituted image data is used.",
        "Detailed tests, original evidence limitations, bounded range fragments and exact missing-ID manifest: SA1B_000014_FORENSICS.md."])
    source_plan = load(EVIDENCE / "sa1b-fast-recovery/source-plan.json", {})
    benchmark = load(EVIDENCE / "sa1b-fast-recovery/benchmark.json", {})
    selected = benchmark.get("selected", {})
    lines.extend(["", "## Fast disjoint source recovery", "",
        "Official OpenDataLab/SA-1B dataset6248 is BITWISE_EQUIVALENT_MIRROR. Source-plan.json is authoritative: slow partial ownership may migrate to ODL under the minimum-wall-clock policy; old partials are preserved.",
        "ODL has at most3 simultaneous downloads globally, including independent forensic14/17. A shard is never downloaded concurrently from both sources.",
        f"Short benchmark selected ODL concurrency: {selected.get('concurrency', 'RUNNING')}; stable aggregate MiB/s: {selected.get('total_mib_s', 'RUNNING')}.",
        "Fixed source ownership: evidence/sa1b-fast-recovery/source-plan.json. Live actual combined speed and source-partitioned ETA: evidence/sa1b-fast-recovery/state.json.",
        "Benchmark raw/corrected timing and all retained partials: evidence/sa1b-fast-recovery/; summary: SA1B_ODL_SHORT_BENCHMARK.md.",
        "Incremental inventory coverage is not a completed whole-index physical/decode audit. Final physical existence and full decode remain mandatory before smoke.",
        "### Required JPEG rescue", "", "| Shard | Required | Recovered | Missing | Status |", "|---|---:|---:|---:|---|"])
    for position in sorted({14, 16, 17}.union(int(row["filename"][3:9]) for row in state.get("shards", []) if row.get("download_status") == QUARANTINED)):
        rescue = load(EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}" / "result.json", {})
        progress = load(EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}" / "progress.json", {})
        lines.append(f"| {position:06d} | {rescue.get('required_images', progress.get('required_images', 'PENDING'))} | {rescue.get('recovered_required_images', 'IN PROGRESS')} | {len(rescue['missing_required_image_ids']) if rescue else 'IN PROGRESS'} | {rescue.get('status', 'RESCUE_PENDING_OR_RUNNING')} |")
    lines.extend(["", "Detailed rescue: SA1B_CORRUPT_SHARD_RESCUE.md. Only truly missing quarantine JPEG IDs: MISSING_REQUIRED_SA_IMAGES.json.",
        f"Source plan created: {source_plan.get('created_utc', 'PENDING')}. No500/4868 training is launched.",
        "### Completed-cache and NAS recovery", "",
        "Official OpenXLab0.1.3 start() can wait indefinitely when every resume range already exists and no worker triggers its assembler. Exact contiguous cached coverage is checked before invoking the SDK's own assembler; no archive bytes are downloaded again.",
        "NAS canonical/alternate bulk assembly returned EIO. Shards41/42 were assembled on persistent local SSD, pinned size/MD5/SHA256 rechecked, and exposed through canonical project symlinks. All old NAS partials and completed-cache chunks remain preserved.",
        "Assembly proof and original-worker handoff: evidence/sa1b-fast-recovery/odl-workers/sa_000041.tar/1/completed-cache-resume.json and corresponding42 proof.",
        "Local storage: /root/.cache/said-recovery/odl-completed/. These original archives must remain until native ingestion finishes; JPEG training paths and bytes are unchanged.",
        "NAS metadata I/O can delay telemetry; latest confirmed installed-image audit is evidence/sa1b-fast-recovery/latest-installed-image-audit.json. Do not infer image completeness from archive byte progress."])
    (RECOVERY / "SA1B_MIRROR_RECOVERY.md").write_text("\n".join(lines) + "\n")


def pending_shards(rows):
    return [position for position, row in enumerate(rows)
            if row.get("extraction_status") != "complete" and row.get("download_status") != QUARANTINED]


def run(smoke_on_ready=False, downloads_only=False, concurrency=2):
    if smoke_on_ready and downloads_only:
        raise ValueError("Download-only recovery forbids smoke")
    if concurrency not in (2, 4, 6, 8):
        raise ValueError("Download concurrency must be 2, 4, 6 or 8")
    state = load(STATE)
    if not state or len(state.get("shards", [])) != 51:
        raise RuntimeError("Run revision discovery before downloading shards")
    state["download_concurrency"] = concurrency
    configure_client(state["endpoint"])
    disk_check(sum(row["expected_size_bytes"] for row in state["shards"]))
    families = requirements()
    ownership = {}
    for row in state["shards"]:
        if row.get("extraction_status") == "complete":
            inventory = Path(row["image_inventory"])
            if digest(inventory) != row["image_inventory_sha256"]:
                raise ValueError("Extracted inventory identity mismatch")
            for line in inventory.open():
                entry = json.loads(line)
                if entry["basename"] in ownership:
                    raise ValueError("Duplicate image provenance across completed shards")
                ownership[entry["basename"]] = row["filename"]
    completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
    if completeness.get("training_index_sha256") != RECORDS_SHA:
        completeness = image_completeness(families, ownership)
    report(state, completeness)
    pending = pending_shards(state["shards"])
    for offset in range(0, len(pending), concurrency):
        disk_check(sum(row["expected_size_bytes"] for row in state["shards"]))
        batch = pending[offset:offset+concurrency]
        for position in batch:
            state["shards"][position]["download_status"] = "downloading"
        atomic_json(STATE, state)
        report(state, completeness)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(download_shard, state, state["shards"][position]): position for position in batch}
            for future in concurrent.futures.as_completed(futures):
                position = futures[future]
                row = future.result()
                state["shards"][position] = row
                if row["download_status"] == QUARANTINED:
                    atomic_json(STATE, state)
                    report(state, completeness)
                    print(row["filename"], QUARANTINED, "preserved; continuing other shards", flush=True)
                    continue
                if row["download_status"] != "verified":
                    atomic_json(STATE, state)
                    report(state, completeness)
                    raise RuntimeError("Shard download failed after bounded retries; remaining batches not started")
                state["checklist"].update(algorithm_verified_by_actual_file=True,
                    verified_algorithm="MD5", first_actual_archive=row["filename"],
                    verification_note="Actual whole-archive MD5 matches checklist; not inferred from hexadecimal length")
                row["extraction_status"] = "extracting"
                atomic_json(STATE, state)
                try:
                    extract_shard(row, ownership)
                except Exception as error:
                    row.update(extraction_status="failed", extraction_error=str(error))
                    atomic_json(STATE, state)
                    report(state, completeness)
                    raise
                atomic_json(STATE, state)
                print(row["filename"], "verified/extracted", row["extracted_images"], flush=True)
        completeness = image_completeness(families, ownership)
        state["sam_required_paths_recovered"] = completeness["families"]["sam"]["recovered_paths"]
        state["sam_required_paths_missing"] = completeness["families"]["sam"]["missing_paths"]
        state["last_batch_audit_utc"] = now()
        atomic_json(STATE, state)
        report(state, completeness)
    completeness = image_completeness(families, ownership)
    if not downloads_only and completeness["missing_training_paths"] == 0:
        subprocess.run([ROOT / ".venv/bin/python", RECOVERY / "sa1b_recovery.py", "audit"], check=True)
        completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    if not downloads_only:
        subprocess.run([ROOT / ".venv/bin/python", RECOVERY / "audit.py", "training"], check=True)
    if smoke_on_ready and completeness["passed"]:
        subprocess.run([ROOT / ".venv/bin/python", RECOVERY / "validate.py", "smoke"], check=True)
        state["smoke_started"] = True
    atomic_json(STATE, state)
    report(state, completeness)
    subprocess.run([ROOT / ".venv/bin/python", RECOVERY / "report.py"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover", "run", "audit", "report"])
    parser.add_argument("--endpoint")
    parser.add_argument("--smoke-on-ready", action="store_true")
    parser.add_argument("--downloads-only", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=(2, 4, 6, 8), default=2)
    args = parser.parse_args()
    if args.command == "discover":
        discover(args.endpoint)
        report()
    elif args.command == "run":
        run(args.smoke_on_ready, args.downloads_only, args.concurrency)
    elif args.command == "audit":
        report(completeness=image_completeness(requirements(), decode=True))
    else:
        report()


if __name__ == "__main__":
    main()
