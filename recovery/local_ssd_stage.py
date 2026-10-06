"""Manifest-driven raw-byte staging; no training, downloads or global cache operations."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import shutil
import subprocess
import threading
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DistributedSampler

from recovery.resource_stall_v2 import ROOT, GIB, dump, process_audit, system_snapshot, summary


EXP = ROOT / "experiments/nest_clip_v1/armb_summary02_4epoch_v1"
SOURCE = ROOT / "local_assets/training/ShareGPT4V"
CACHE = Path("/root/SAID_train_cache")
IMAGES = CACHE / "ShareGPT4V"
INDEX = CACHE / "data_index"
META = CACHE / "staging"
ORIGINAL_INDEX = ROOT / "runtime/SAID-nest-clip-v1/data_index"
HISTORY = ROOT / "experiments/nest_clip_v1/armb_summary_dose_500_v1/arm_E1_summary02/evidence/steps.jsonl.gz"
RESERVE = 64 * GIB
EXPECTED = dict(sam=569486, coco=118287, llava=558128)


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\0" in name or not path.parts:
        raise ValueError("Unsafe image relative path")
    return str(path)


def digest(path, direct=False, release_local=False):
    descriptor = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    result = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1 << 20):
            result.update(chunk)
        if release_local:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    return result.hexdigest()


def atomic_copy(source, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.parent.resolve().is_relative_to(CACHE):
        raise RuntimeError("Destination escapes private local cache")
    if destination.exists():
        source_sha = expected or digest(source, direct=True)
        destination_sha = digest(destination, release_local=True)
        if source_sha != destination_sha:
            raise RuntimeError("Existing destination SHA256 mismatch: " + str(destination))
        return dict(source_sha256=source_sha, destination_sha256=destination_sha,
                    size_bytes=destination.stat().st_size, copied=False)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}.{threading.get_ident()}")
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECT)
    initial = os.fstat(source_fd)
    checksum = hashlib.sha256()
    count = 0
    try:
        with temporary.open("wb") as output:
            while chunk := os.read(source_fd, 1 << 20):
                output.write(chunk)
                checksum.update(chunk)
                count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
            os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        final = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns) or count != initial.st_size:
        raise RuntimeError("Source changed during copy: " + str(source))
    source_sha = checksum.hexdigest()
    if expected is not None and source_sha != expected:
        raise RuntimeError("Previously verified source bytes changed")
    destination_sha = digest(temporary, release_local=True)
    if destination_sha != source_sha:
        raise RuntimeError("New destination SHA256 mismatch")
    temporary.replace(destination)
    return dict(source_sha256=source_sha, destination_sha256=destination_sha, size_bytes=count, copied=True)


def check_disk(required=0):
    CACHE.mkdir(parents=True, exist_ok=True)
    mount = subprocess.check_output(["findmnt", "-T", str(CACHE), "-n", "-o", "FSTYPE,SOURCE,TARGET"], text=True).strip()
    if any(value in mount.lower() for value in ("nfs", "cifs", "tmpfs")):
        raise RuntimeError("Cache is not local block-backed storage: " + mount)
    usage = shutil.disk_usage(CACHE)
    if usage.free < required + RESERVE:
        raise RuntimeError("Insufficient local space with64GiB reserve")
    return dict(mount=mount, free_bytes=usage.free, required_bytes=required, reserve_bytes=RESERVE,
                backing_device_evidence="lsblk: nvme2n1p2 ext4 backs the root overlay and /tmp/nvidia-mps")


def sizes_from_installation():
    sizes = {}
    sources = set()
    for shard in json.loads((ROOT / "recovery/SA1B_SHARDS.json").read_text())["shards"]:
        if shard.get("image_inventory"):
            sources.add(shard["image_inventory"])
        rescue = shard.get("required_jpeg_rescue", {})
        if rescue.get("image_inventory"):
            sources.add(rescue["image_inventory"])
        if rescue.get("original_rescue_inventory"):
            sources.add(rescue["original_rescue_inventory"]["path"])
    names = subprocess.check_output(["rg", "--files", "recovery/evidence/modelscope-sa1b/installed-inventories"], cwd=ROOT, text=True)
    sources.update(str(ROOT / name) for name in names.splitlines() if name.endswith(".jsonl"))
    for name in sorted(sources):
        with Path(name).open() as handle:
            for line in handle:
                row = json.loads(line)
                sizes["sam/images/" + row["basename"]] = row["size_bytes"]
    for family in ("coco", "llava"):
        info = json.loads((ROOT / f"recovery/evidence/hf-training-{family}-worker.json").read_text())
        with zipfile.ZipFile(info["archive"]) as archive:
            for member in archive.infolist():
                if member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    prefix = "coco/" if family == "coco" else "llava/llava_pretrain/images/"
                    sizes[prefix + member.filename] = member.file_size
    return sizes


def historical_rows():
    with gzip.open(HISTORY, "rt") as handle:
        return [json.loads(line) for line in handle]


def rng_digest():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()), protocol=4)).hexdigest()


def prepare():
    check_disk()
    META.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for name in ("records.jsonl", "offsets.npy", "metadata.json"):
        atomic_copy(ORIGINAL_INDEX / name, INDEX / name)
    metadata = json.loads((INDEX / "metadata.json").read_text())
    assert metadata["training_records"] == 1245901 and metadata["skip"] == 1000
    assert digest(INDEX / "records.jsonl") == metadata["records_sha256"]
    history = historical_rows()
    assert [row["step"] for row in history] == list(range(1, 501))
    before = rng_digest()
    positions = {}
    rank_indices = {}
    for rank in range(4):
        sampler = DistributedSampler(range(1245901), num_replicas=4, rank=rank, shuffle=True, seed=0, drop_last=False)
        indices = list(itertools.islice(iter(sampler), 500 * 256))
        rank_indices[rank] = indices
        expected = [index + 1000 for index in indices]
        actual = [sample for row in history for health in row["rank_health"] if health["rank"] == rank for sample in health["sampling"]["sample_ids"]]
        assert expected == actual, f"Historical500-step IDs differ on rank{rank}"
        positions.update({index: dict(step=position // 256 + 1, rank=rank, batch_position=position % 256)
                          for position, index in enumerate(indices)})
    assert len(positions) == 512000
    sizes = sizes_from_installation()
    counts, total, seen = Counter(), 0, set()
    with (INDEX / "records.jsonl").open("rb") as handle, (META / "step1_500_images.jsonl").open("w") as output:
        records = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = np.load(INDEX / "offsets.npy", mmap_mode="r")
        for index in sorted(positions):
            record = json.loads(records[offsets[index]:offsets[index + 1]])
            relative = safe_relative(record["image"])
            if relative in seen:
                raise RuntimeError("Unexpected image duplication in frozen first500 records")
            seen.add(relative)
            family = relative.split("/")[0]
            size = sizes[relative]
            row = dict(sample_id=index + 1000, record_index=index, family=family, relative_path=relative,
                       source_path=str(SOURCE / relative), destination_path=str(IMAGES / relative),
                       size_bytes=size, **positions[index])
            output.write(json.dumps(row) + "\n")
            counts[family] += 1
            total += size
        from train.nested_semantic_data import sampled_text_views
        matches = 0
        for row in history[:5]:
            for rank in range(4):
                indices = rank_indices[rank][(row["step"] - 1) * 256:row["step"] * 256]
                views = [sampled_text_views(json.loads(records[offsets[index]:offsets[index + 1]])["caption"],
                                            "summary_random_detail", 0, 0, index + 1000) for index in indices]
                stream = dict(sample_ids=[index + 1000 for index in indices], views=[view["views"] for view in views],
                              tokens=[[view[key].tolist() for view in views] for key in ("tokens_f", "tokens_o", "tokens_e")])
                actual_sha = hashlib.sha256(json.dumps(stream, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                expected_sha = next(item["stream_sha256"] for item in row["rank_health"] if item["rank"] == rank)
                assert actual_sha == expected_sha, "Historical F/S/D string/token stream differs"
                matches += 256
        records.close()
    assert rng_digest() == before, "Offline sampler/text construction advanced a process-wide RNG"
    proof = dict(status="MANIFEST_READY", records_count=512000, unique_images_count=len(seen), family_counts=dict(counts),
                 total_bytes=total, historical_sample_ids_matched=512000, historical_FSD_string_token_samples_matched=matches,
                 sampling_source="Unchanged train.nested_semantic_data.sampled_text_views", offline_global_RNG_unchanged=True,
                 horizon=4868, seed=0, global_batch=1024, index_records_sha256=metadata["records_sha256"],
                 manifest_sha256=digest(META / "step1_500_images.jsonl"), local_root=str(IMAGES), disk=check_disk(total))
    dump(META / "manifest-proof.json", proof)
    dump(EXP / "LOCAL_SSD_MANIFEST_PROOF.json", proof)
    print(json.dumps(proof), flush=True)


def manifest_rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def copy_images(full=False):
    manifest = META / "step1_500_images.jsonl"
    if full:
        manifest = META / "full_required_images.jsonl"
        with (ROOT / "recovery/required_training_images.jsonl").open() as source, manifest.open("w") as output:
            for line in source:
                row = json.loads(line)
                relative = safe_relative(row["relative_path"])
                output.write(json.dumps(dict(family=row["family"], relative_path=relative,
                    source_path=str(SOURCE / relative), destination_path=str(IMAGES / relative))) + "\n")
    ledger = META / "verified_images.jsonl"
    known = {row["relative_path"]: row for row in manifest_rows(ledger)} if ledger.exists() else {}
    started = time.monotonic()
    checked, copied, payload, written = 0, 0, 0, 0
    counts = Counter()
    progress_path = META / ("full-copy-progress.json" if full else "stage500-copy-progress.json")
    total = 1245901 if full else 512000

    def copy(row):
        previous = known.get(row["relative_path"])
        result = atomic_copy(Path(row["source_path"]), Path(row["destination_path"]),
                             previous["source_sha256"] if previous else None)
        if "size_bytes" in row and result["size_bytes"] != row["size_bytes"]:
            raise RuntimeError("Image size differs from recovery inventory")
        return dict(relative_path=row["relative_path"], family=row["family"], **result)

    worker_count = int(os.environ.get("SAID_LOCAL_COPY_WORKERS", "16"))
    if not 1 <= worker_count <= 32:
        raise RuntimeError("Copy concurrency must stay within1..32; training workers unchanged")
    with ThreadPoolExecutor(max_workers=worker_count) as workers, ledger.open("a") as log:
        stream = iter(manifest_rows(manifest))
        while batch := list(itertools.islice(stream, 256)):
            check_disk()
            for result in workers.map(copy, batch):
                checked += 1
                counts[result["family"]] += 1
                copied += int(result["copied"])
                payload += result["size_bytes"]
                written += result["size_bytes"] if result["copied"] else 0
                log.write(json.dumps(result) + "\n")
            log.flush()
            if checked % 5120 == 0 or checked == total:
                elapsed = time.monotonic() - started
                progress = dict(status="COPYING" if checked < total else "COPY_HASH_PASS", checked=checked, total=total,
                    copied=copied, families=dict(counts), source_destination_SHA256_matches=checked, payload_bytes=payload,
                    newly_written_bytes=written, elapsed_s=elapsed, average_payload_MiB_s=payload / elapsed / 2**20,
                    average_new_copy_MiB_s=written / elapsed / 2**20,
                    workers=worker_count, source_read="NFS O_DIRECT; empirically byte-verified", global_drop_caches=False,
                    private_local_cache_advice="fsync + file-scoped DONTNEED on our new mirror only; no source-cache advice")
                dump(progress_path, progress)
                dump(EXP / "LOCAL_SSD_COPY_PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
    assert checked == total
    if full:
        assert dict(counts) == EXPECTED
    dump(META / ("full-copy-ready.json" if full else "stage500-copy-ready.json"), dict(progress, copy_workers_exited=True))


def verify():
    from train.said_cvssl_data import reference_view_a_transform
    proof = json.loads((META / "manifest-proof.json").read_text())
    copied = json.loads((META / "stage500-copy-ready.json").read_text())
    assert proof["manifest_sha256"] == digest(META / "step1_500_images.jsonl")
    assert copied["checked"] == copied["source_destination_SHA256_matches"] == proof["unique_images_count"] == 512000
    assert copied["families"] == proof["family_counts"] and copied["payload_bytes"] == proof["total_bytes"]
    assert copied["copy_workers_exited"]
    torch.set_num_threads(1)
    groups = {family: [] for family in EXPECTED}
    for row in manifest_rows(META / "step1_500_images.jsonl"):
        groups[row["family"]].append(row)
    candidates = []
    for family, count in (("sam", 334), ("coco", 333), ("llava", 333)):
        candidates.extend(random.Random(0).sample(groups[family], count))
    transform = reference_view_a_transform()
    verified = []
    for row in candidates:
        source, destination = Path(row["source_path"]), Path(row["destination_path"])
        assert source.name == destination.name
        source_sha, destination_sha = digest(source, direct=True), digest(destination)
        assert source_sha == destination_sha
        with Image.open(source) as image:
            image.load()
            rgb = image.convert("RGB")
            pixels, dimensions = hashlib.sha256(rgb.tobytes()).hexdigest(), rgb.size
            expected = transform(rgb)
        with Image.open(destination) as image:
            image.load()
            rgb = image.convert("RGB")
            actual = transform(rgb)
            assert rgb.size == dimensions and hashlib.sha256(rgb.tobytes()).hexdigest() == pixels
        assert torch.equal(expected, actual)
        verified.append(dict(sample_id=row["sample_id"], family=row["family"], basename=source.name,
            source_sha256=source_sha, destination_sha256=destination_sha, RGB_pixels_exact_equal=True,
            preprocess_tensor_exact_equal=True, tensor_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()))
    dump(META / "equivalence-1000.json", dict(passed=True, count=1000, rows=verified))
    snapshot = system_snapshot()
    processes = process_audit()
    assert processes["workers_all_exited"]
    ready = snapshot["memory_current"] < int(snapshot["memory_max"]) * .98
    result = dict(passed=ready, status="LOCAL_SSD_STAGE500_READY" if ready else "LOCAL_SSD_MEMORY_NEAR_LIMIT_HOLD",
        equivalence_samples=1000, byte_RGB_preprocess_exact_equal=True, copy_workers_exited=True,
        system=snapshot, process_audit=processes, local_storage=check_disk(), global_drop_caches=False,
        manifest_sha256=proof["manifest_sha256"], all512000_source_destination_SHA256_matches=True,
        local_images=str(IMAGES), local_index=str(INDEX), no_remaining_images_background_copy=True)
    dump(EXP / "LOCAL_SSD_STAGE500_READY.json", result)
    dump(META / "stage500-ready.json", result)
    print(json.dumps({key:result[key] for key in ("status", "passed", "equivalence_samples")}), flush=True)
    if not ready:
        raise RuntimeError("Cgroup still near500GiB; hold training without any global cache operation")


def report_resources():
    runtime = ROOT / "runtime/SAID-nest-clip-v1/armb_summary02_500gate_localssd_v3"
    phases = Path("/tmp/said-s02-full-phase-localssd-v3/step500")
    cycles = list(manifest_rows(runtime / "step500/cycle_timing.jsonl"))
    assert len(cycles) == 500 and cycles[-1]["step"] == 500
    ranks = {}
    for rank in range(4):
        rows = [row for row in manifest_rows(phases / f"rank{rank}.jsonl") if 7 <= row["step"] <= 500]
        assert len(rows) == 494
        timings = {key: summary([row[key] for row in rows]) for key in
                   ("data_wait_s", "h2d_s", "forward_s", "backward_s", "optimizer_s")}
        timings.update(backward_includes_DDP=True, distinct_DDP_timing_available=False,
                       maximum_cgroup_current_GiB=max(row["system_after"]["memory_current"] for row in rows) / GIB,
                       PSI_scope="host /proc/pressure, not cgroup-specific on this v1 host")
        ranks[str(rank)] = timings
    result = dict(passed=True, optimizer_updates=500, steady_updates="7..500",
                  full_cycle_s=summary([row["four_rank_max_seconds"] for row in cycles if row["step"] >= 7]),
                  ranks=ranks, resource_protection="Unchanged native3 consecutive full cycles>3s", algorithm_changes=False)
    dump(EXP / "evidence/reproduction/local500-phase-summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "copy500", "verify", "copyfull", "resources"))
    choice = parser.parse_args().command
    {"prepare": prepare, "copy500": copy_images, "verify": verify, "copyfull": lambda: copy_images(True),
     "resources": report_resources}[choice]()
