import io
import hashlib
import tarfile
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

import sa1b_opendatalab_split as split


def row(tmp_path, position, status="pending"):
    return dict(filename=f"sa_{position:06d}.tar", archive=str(tmp_path / f"sa_{position:06d}.tar"),
                download_status=status, expected_size_bytes=10, expected_lfs_sha256="digest")


def test_future_assignment_skips_hf_started_and_quarantined(tmp_path):
    active = row(tmp_path, 20, "downloading")
    future = row(tmp_path, 39)
    assert split.eligible(future, [active, future])
    assert not split.eligible(row(tmp_path, 27), [active])
    assert not split.eligible(row(tmp_path, 39, "downloading"), [active])
    assert not split.eligible(row(tmp_path, 39, split.recovery.QUARANTINED), [active])
    assert not split.eligible(future, [])
    (tmp_path / "sa_000039.tar").write_bytes(b"owned")
    assert not split.eligible(future, [active])


def test_local_ssd_target_preserves_existing_partial_and_resumes_its_own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "LOCAL_ODL", tmp_path / "ssd")
    monkeypatch.setattr(split.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 1024**3))
    canonical = tmp_path / "nas"
    local = split.choose_download_target("sa_000030.tar", canonical)
    assert local == tmp_path / "ssd/sa_000030"
    assert (canonical / "OpenDataLab___SA-1B/raw/.cache").is_symlink()
    assert split.choose_download_target("sa_000030.tar", canonical) == local
    partial_target = tmp_path / "existing-nas"
    cache = partial_target / "OpenDataLab___SA-1B/raw/.cache"
    cache.mkdir(parents=True)
    (cache / "old.incomplete").write_bytes(b"retained")
    assert split.choose_download_target("sa_000031.tar", partial_target) == partial_target
    assert (cache / "old.incomplete").read_bytes() == b"retained"
    assert split.choose_download_target("sa_000014.tar", canonical, tmp_path / "forensic") == tmp_path / "forensic"


def test_local_archive_link_requires_actual_pinned_hashes_and_never_overwrites(tmp_path):
    source = tmp_path / "ssd.tar"
    source.write_bytes(b"original archive")
    destination = tmp_path / "canonical.tar"
    metadata = dict(expected_size_bytes=source.stat().st_size,
        expected_checklist_checksum=hashlib.md5(source.read_bytes()).hexdigest(),
        expected_lfs_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    split.publish_local_archive(source, destination, metadata, tmp_path / "proof")
    assert destination.is_symlink() and destination.read_bytes() == source.read_bytes()
    existing = tmp_path / "existing.tar"
    existing.write_bytes(b"different original")
    with pytest.raises(ValueError, match="refusing overwrite"):
        split.publish_local_archive(source, existing, metadata, tmp_path / "proof")
    assert existing.read_bytes() == b"different original"


def test_unverified_handoff_never_creates_destination(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="unverified"):
        split.handoff(source, row(tmp_path, 39), "revision", {})
    assert not (tmp_path / "sa_000039.tar").exists()


def test_full_audit_decodes_original_jpegs_and_matches_required_ids(tmp_path, monkeypatch):
    data = io.BytesIO()
    Image.new("RGB", (12, 9), "red").save(data, format="JPEG")
    archive = tmp_path / "sa_000039.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("./sa_123456.jpg")
        member.size = len(data.getvalue())
        output.addfile(member, io.BytesIO(data.getvalue()))
    monkeypatch.setattr(split.recovery, "verify_archive", lambda *args:
                        dict(observed_sha256="digest", observed_md5="md5", md5_matches_checklist=True))
    monkeypatch.setattr(split.recovery, "requirements", lambda: {"sam": {"sam/images/sa_123456.jpg"}})
    audit = split.full_audit(archive, {}, tmp_path / "evidence")
    assert audit["protocol_passed"]
    assert audit["jpeg_count"] == audit["decoded_jpegs"] == audit["required_images_found"] == 1
    assert audit["sample_byte_hashes"][0]["sha256"] == split.recovery.hashlib.sha256(data.getvalue()).hexdigest()


def test_decode_failure_does_not_pass_protocol(tmp_path, monkeypatch):
    archive = tmp_path / "sa_000039.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("./sa_123456.jpg")
        member.size = 4
        output.addfile(member, io.BytesIO(b"bad!"))
    monkeypatch.setattr(split.recovery, "verify_archive", lambda *args: dict(observed_sha256="digest"))
    monkeypatch.setattr(split.recovery, "requirements", lambda: {"sam": {"sam/images/sa_123456.jpg"}})
    audit = split.full_audit(archive, {}, tmp_path / "evidence")
    assert not audit["protocol_passed"]
    assert len(audit["decode_failures"]) == 1


def test_verified_handoff_retains_original_and_never_overwrites(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "huggingface_hub._local_folder",
                        SimpleNamespace(write_download_metadata=lambda *args: calls.append(args)))
    source = tmp_path / "source.tar"
    source.write_bytes(b"original")
    future = row(tmp_path, 39)
    future["repo_path"] = future["filename"]
    audit = dict(protocol_passed=True, observed_sha256="digest")
    proof = split.handoff(source, future, "revision", audit)
    destination = tmp_path / "sa_000039.tar"
    assert source.read_bytes() == destination.read_bytes() == b"original"
    assert source.stat().st_ino == destination.stat().st_ino
    assert len(calls) == 1
    assert not proof["training_images_installed"]
    with pytest.raises(ValueError, match="HF already owns"):
        split.handoff(source, future, "revision", audit)
    assert destination.read_bytes() == b"original"
