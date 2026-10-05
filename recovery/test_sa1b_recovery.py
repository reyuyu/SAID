import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

import sa1b_recovery as recovery
import validate


def checklist():
    return "\n".join(f"{recovery.EXAMPLES.get(f'sa_{position:06d}.tar', 'f'*32)}  sa_{position:06d}.tar"
                     for position in range(1000))


def make_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_checklist_requires_all_thousand_and_matches_history():
    checksums = recovery.parse_checklist(checklist())
    assert len(checksums) == 1000
    assert checksums["sa_000050.tar"] == recovery.EXAMPLES["sa_000050.tar"]
    with pytest.raises(ValueError, match="1000"):
        recovery.parse_checklist(checklist().splitlines()[0])


def test_quarantine_never_downloads_and_does_not_block_later_shards():
    rows = [dict(download_status=recovery.QUARANTINED, extraction_status="pending"),
            dict(download_status="verified", extraction_status="complete"),
            dict(download_status="pending", extraction_status="pending")]
    assert recovery.pending_shards(rows) == [2]
    assert recovery.download_shard({}, rows[0]) == rows[0]


def test_download_only_forbids_smoke_before_any_work():
    with pytest.raises(ValueError, match="forbids smoke"):
        recovery.run(smoke_on_ready=True, downloads_only=True)


def test_checksum_matching_invalid_tar_is_preserved_without_retry(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    (evidence / "sa1b-shards").mkdir(parents=True)
    monkeypatch.setattr(recovery, "EVIDENCE", evidence)
    archive = tmp_path / "sa_000014.tar"
    content = b"checksum-matching bytes that are not a tar archive"
    archive.write_bytes(content)
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download,
        get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(xet_file_data=None),
        hf_hub_url=lambda *args, **kwargs: "https://example.invalid"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._runtime", SimpleNamespace(is_xet_available=lambda: False))
    row = dict(filename=archive.name, repo_path=archive.name, archive=str(archive), attempts=[],
               expected_size_bytes=len(content), expected_checklist_checksum=hashlib.md5(content).hexdigest(),
               expected_lfs_sha256=hashlib.sha256(content).hexdigest())
    result = recovery.download_shard(dict(revision="fixed", endpoint="https://example.invalid"), row)
    assert result["download_status"] == recovery.QUARANTINED
    assert archive.read_bytes() == content and len(calls) == 1
    assert not result["tar_integrity_passed"]
    assert Path(result["tar_stderr"]).read_text()
    assert result["attempts"][0]["same_object_retry_disabled"]


def test_checklist_rejects_duplicate_or_changed_reference():
    with pytest.raises(ValueError, match="duplicate"):
        recovery.parse_checklist(checklist() + "\n" + checklist().splitlines()[0])
    with pytest.raises(ValueError, match="historical example"):
        recovery.parse_checklist(checklist().replace(recovery.EXAMPLES["sa_000001.tar"], "a" * 32))


def test_revision_search_does_not_assume_main_complete():
    calls = []

    class Api:
        def list_repo_files(self, repo, repo_type, revision):
            calls.append(revision)
            return ["checklist.chk"] + (recovery.NAMES[1:] if revision == "new" else recovery.NAMES)

    history = []
    revision, mapping = recovery.select_revision(Api(), [SimpleNamespace(commit_id="new"), SimpleNamespace(commit_id="old")], history)
    assert revision == "old" and calls == ["new", "old"]
    assert len(mapping) == 52 and history[0]["missing"] == ["sa_000000.tar"]


def test_no_complete_revision_never_selects_partial_source():
    api = SimpleNamespace(list_repo_files=lambda *args, **kwargs: recovery.NAMES[:-1] + ["checklist.chk"])
    assert recovery.select_revision(api, [SimpleNamespace(commit_id="partial")], []) == (None, None)


def test_actual_md5_and_lfs_sha_are_computed_not_guessed(tmp_path):
    archive = tmp_path / "asset.tar"
    content = b"original file content, not a checksum string"
    archive.write_bytes(content)
    row = dict(expected_size_bytes=len(content), expected_checklist_checksum=hashlib.md5(content).hexdigest(),
               expected_lfs_sha256=hashlib.sha256(content).hexdigest())
    result = recovery.verify_archive(archive, row)
    assert result["checksum_algorithm_verified"] and result["md5_matches_checklist"]
    row["expected_checklist_checksum"] = "a" * 32
    with pytest.raises(ValueError, match="Actual-file MD5"):
        recovery.verify_archive(archive, row)


def test_extract_preserves_original_basename_bytes_and_skips_mask_json(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(recovery, "SAM_ROOT", tmp_path / "images")
    archive = tmp_path / "sa_000000.tar"
    original = b"unchanged image payload"
    make_tar(archive, [("original_folder/sa_1.jpg", original), ("original_folder/sa_1.json", b'{}')])
    row = dict(filename=archive.name, archive=str(archive))
    ownership = {}
    recovery.extract_shard(row, ownership)
    assert (recovery.SAM_ROOT / "sa_1.jpg").read_bytes() == original
    assert not (recovery.SAM_ROOT / "sa_1.json").exists()
    assert row["skipped_mask_json"] == 1 and ownership == {"sa_1.jpg": archive.name}
    assert row["extraction_status"] == "complete"


@pytest.mark.parametrize("member", ["../sa_1.jpg", "/sa_1.jpg"])
def test_archive_path_escape_rejected_before_extraction(tmp_path, member):
    archive = tmp_path / "archive.tar"
    make_tar(archive, [(member, b"image")])
    with pytest.raises(ValueError, match="escapes"):
        recovery.audit_structure(archive)


def test_archive_symlink_rejected(tmp_path):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("sa_1.jpg")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        output.addfile(member)
    with pytest.raises(ValueError, match="link/device"):
        recovery.audit_structure(archive)


def test_corrupt_download_deleted_then_retried(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(recovery, "EVIDENCE", evidence)
    monkeypatch.setattr(recovery.time, "sleep", lambda duration: None)
    archive = tmp_path / "sa_000000.tar"
    make_tar(archive, [("sa_1.jpg", b"original")])
    valid = archive.read_bytes()
    archive.write_bytes(b"corrupt")
    calls = []

    def download(*args, **kwargs):
        if calls:
            assert not archive.exists() and kwargs["force_download"]
            archive.write_bytes(valid)
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download,
        get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(xet_file_data=None),
        hf_hub_url=lambda *args, **kwargs: "https://example.invalid"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._runtime", SimpleNamespace(is_xet_available=lambda: False))
    row = dict(filename=archive.name, repo_path=archive.name, archive=str(archive), attempts=[],
               expected_size_bytes=len(valid), expected_checklist_checksum=hashlib.md5(valid).hexdigest(),
               expected_lfs_sha256=hashlib.sha256(valid).hexdigest())
    result = recovery.download_shard(dict(revision="fixed", endpoint="https://example.invalid"), row)
    assert result["download_status"] == "verified" and result["tar_integrity_passed"]
    assert len(calls) == 2 and result["attempts"][0]["corrupt_shard_deleted"]


def test_smoke_rejects_existence_only_without_full_decode_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "EVIDENCE", tmp_path)
    for name in ["git-audit", "environment-audit", "config-audit", "training-audit", "evaluator-audit",
                 "step0-audit", "cpu-tests", "sampling-audit", "construction-audit"]:
        (tmp_path / (name + ".json")).write_text(json.dumps(dict(passed=True)))
    with pytest.raises(RuntimeError, match="training-image-completeness-audit"):
        validate.smoke()


def test_gzip_tar_named_tar_is_supported_without_image_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(recovery, "SAM_ROOT", tmp_path / "images")
    archive = tmp_path / "sa_000000.tar"
    content = b"untouched image content"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("original/sa_1.jpg")
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    row = dict(filename=archive.name, archive=str(archive))
    recovery.extract_shard(row, {})
    assert (recovery.SAM_ROOT / "sa_1.jpg").read_bytes() == content
    assert row["archive_structure"]["compression"] == "gzip"


def test_disk_shortage_stops_before_shard_download(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "EVIDENCE", tmp_path)
    monkeypatch.setattr(recovery.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda command: b"filesystem capacity audit")
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        recovery.disk_check(1000)
    assert not json.loads((tmp_path / "sa1b-disk-audit.json").read_text())["sufficient"]


def test_cross_shard_duplicates_are_reported_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(recovery, "SAM_ROOT", tmp_path / "images")
    archive = tmp_path / "sa_000001.tar"
    make_tar(archive, [("sa_1.jpg", b"image")])
    row = dict(filename=archive.name, archive=str(archive))
    with pytest.raises(ValueError, match="Cross-shard duplicate"):
        recovery.extract_shard(row, {"sa_1.jpg": "sa_000000.tar"})
    assert len(row["duplicate_paths"]) == 1
    assert not (recovery.SAM_ROOT / "sa_1.jpg").exists()
