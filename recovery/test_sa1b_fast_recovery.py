import hashlib
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import sa1b_fast_recovery as fast
import sa1b_jpeg_rescue as rescue
import sa1b_recovery as recovery
import sa1b_opendatalab_probe as probe


def test_sources_preserve_partials_and_exclude_quarantine(tmp_path, monkeypatch):
    rows = [dict(filename=f"sa_{position:06d}.tar", archive=str(tmp_path / f"sa_{position:06d}.tar"),
                 extraction_status="pending") for position in range(51)]
    monkeypatch.setattr(fast, "payload_bytes", lambda row: 100 if row["filename"] == "sa_000025.tar" else 0)
    monkeypatch.setattr(fast, "source_path", lambda name: tmp_path / "odl" / name)
    plan = fast.plan_sources(dict(shards=rows))
    assert plan["owners"]["sa_000025.tar"] == "HF_EXISTING_PARTIAL_ONLY"
    assert plan["owners"]["sa_000026.tar"] == "ODL"
    assert all(plan["owners"][f"sa_{position:06d}.tar"] == "RESCUE_ONLY" for position in (14, 16, 17))
    assert plan["initial_hf_bytes"]["sa_000025.tar"] == 100


def test_hf_never_downloads_zero_progress_or_odl_owned():
    row = dict(filename="sa_000041.tar")
    plan = dict(owners={row["filename"]: "ODL"}, initial_hf_bytes={row["filename"]: 0})
    with pytest.raises(ValueError, match="cannot acquire"):
        fast.hf_job({}, row, plan)
    plan["owners"][row["filename"]] = "HF_EXISTING_PARTIAL_ONLY"
    with pytest.raises(ValueError, match="cannot acquire"):
        fast.hf_job({}, row, plan)


def test_odl_never_downloads_hf_owned():
    row = dict(filename="sa_000025.tar")
    with pytest.raises(ValueError, match="HF-owned"):
        fast.odl_job(row, dict(owners={row["filename"]: "HF_EXISTING_PARTIAL_ONLY"}))


def test_rescue_copies_original_bytes_and_rejects_overwrite(tmp_path, monkeypatch):
    destination = tmp_path / "training"
    destination.mkdir()
    monkeypatch.setattr(rescue.recovery, "SAM_ROOT", destination)
    source = tmp_path / "sa_123.jpg"
    Image.new("RGB", (20, 12), "blue").save(source, format="JPEG")
    original = source.read_bytes()
    inventory = {}
    rescue.install(source, source.name, inventory, "test original")
    assert (destination / source.name).read_bytes() == original
    assert inventory[source.name]["sha256"] == hashlib.sha256(original).hexdigest()
    assert inventory[source.name]["width"] == 20
    assert inventory[source.name]["image_transform"] == "NONE"
    Image.new("RGB", (20, 12), "red").save(source, format="JPEG")
    with pytest.raises(ValueError, match="not overwritten"):
        rescue.install(source, source.name, inventory, "different")
    assert (destination / source.name).read_bytes() == original


def test_invalid_jpeg_cannot_be_counted(tmp_path):
    source = tmp_path / "sa_123.jpg"
    source.write_bytes(b"bad JPEG")
    with pytest.raises(Exception):
        rescue.check_jpeg(source)


def test_complete_jpeg_rescue_can_pass_container_exception(tmp_path):
    inventory = tmp_path / "images.jsonl"
    inventory.write_text("original image hashes\n")
    proof = dict(status="TRAINING_IMAGES_FULLY_RECOVERED", training_index_sha256=recovery.RECORDS_SHA,
        required_images=12, recovered_required_images=12, missing_required_image_ids=[], remaining_decode_failures=0,
        duplicate_ids=[], image_transform="NONE", original_archive_modified=False,
        image_inventory=str(inventory), image_inventory_sha256=recovery.digest(inventory))
    row = dict(download_status=recovery.QUARANTINED, required_jpeg_rescue=proof)
    assert recovery.training_shard_ready(row)
    proof["missing_required_image_ids"] = [123]
    assert not recovery.training_shard_ready(row)
    proof["missing_required_image_ids"] = []
    inventory.write_text("tampered\n")
    assert not recovery.training_shard_ready(row)


def test_saved_benchmark_is_not_repeated(tmp_path, monkeypatch):
    path = tmp_path / "benchmark.json"
    saved = dict(status="COMPLETE", selected=dict(concurrency=2))
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(fast, "BENCH", path)
    monkeypatch.setattr(fast, "normalize_benchmark", lambda value: value)
    assert fast.benchmark({}, {}) == saved


def test_transfer_installs_only_original_bytes_and_validates_digest(tmp_path, monkeypatch):
    source_root = tmp_path / "staging"
    target_root = tmp_path / "training"
    source_root.mkdir()
    target_root.mkdir()
    monkeypatch.setattr(rescue.recovery, "SAM_ROOT", target_root)
    source = source_root / "sa_123.jpg"
    Image.new("RGB", (24, 15), "blue").save(source, format="JPEG")
    data = source.read_bytes()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(dict(basename=source.name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)))+"\n")
    output = tmp_path / "transfer.json"
    rescue.transfer_inventory(inventory, source_root, output)
    assert json.loads(output.read_text())["passed"]
    assert source.read_bytes() == (target_root / source.name).read_bytes() == data
    assert not list(target_root.glob(".*.rescue-*"))


def test_normalization_uses_actual_monotonic_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "FOLDER", tmp_path)
    monkeypatch.setattr(fast, "BENCH", tmp_path / "benchmark.json")
    monkeypatch.setattr(fast.recovery, "RECOVERY", tmp_path)
    samples = [dict(cumulative_bytes=fast.MIB*10*interval, total_mib_s=1, elapsed_seconds=10*interval-2) for interval in range(1, 19)]
    row = dict(concurrency=1, seconds=178, samples=samples, per_shard_mib_s={"sa_000041.tar":180/178},
               hf_total_mib_s=0, errors=0, retries=0)
    value = dict(status="COMPLETE", stages=[row])
    normalized = fast.normalize_benchmark(value)
    assert normalized["selected"]["seconds"] == 180
    assert normalized["selected"]["total_mib_s"] == 1
    assert normalized["selected"]["stable"]


def test_complete_sdk_cache_uses_native_assembler_preserves_partial(tmp_path):
    original = b"original archive bytes"
    archive = tmp_path / "sa_000041.tar"
    archive.write_bytes(b"previous partial")
    cleared = []
    downloader = SimpleNamespace(download_dir=str(tmp_path), prefix="", filename=archive.name, LOG=[],
        clear=lambda: cleared.append(True))
    def sew():
        archive.write_bytes(original)
        downloader.clear()
    setattr(downloader, "_BigFileDownloader__get_ranges_from_cache", lambda: [(0, len(original)-1)])
    setattr(downloader, "_BigFileDownloader__sew", sew)
    row = dict(expected_size_bytes=len(original), expected_checklist_checksum=hashlib.md5(original).hexdigest(),
               expected_lfs_sha256=hashlib.sha256(original).hexdigest())
    proof = probe.recover_complete_cache(downloader, row)
    assert proof["cache_preserved"] and not cleared
    assert archive.read_bytes() == original
    assert Path(proof["preserved_preassembly_partial"]).read_bytes() == b"previous partial"


def test_incomplete_or_overlapping_sdk_ranges_are_not_assembled(tmp_path):
    downloader = SimpleNamespace(download_dir=str(tmp_path), prefix="", filename="sa_000041.tar", LOG=[])
    setattr(downloader, "_BigFileDownloader__get_ranges_from_cache", lambda: [(0, 4), (4, 9)])
    assert probe.recover_complete_cache(downloader, dict(expected_size_bytes=10)) is None
    assert not list(tmp_path.iterdir())
