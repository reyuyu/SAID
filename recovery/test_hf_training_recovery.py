import io
import datetime
import stat
from types import SimpleNamespace
import zipfile

import pytest

import hf_training_recovery as recovery
import sa1b_recovery


def archive_bytes(members):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    payload.seek(0)
    return payload


def test_original_zip_layout_and_count():
    with zipfile.ZipFile(archive_bytes([("train2017/000000000009.jpg", b"original")])) as archive:
        assert len(recovery.audit_zip_structure(archive, "coco", 1)) == 1
        with pytest.raises(ValueError, match="image count"):
            recovery.audit_zip_structure(archive, "coco", 2)


def test_archive_integrity_uses_standard_library_without_unzip(tmp_path, monkeypatch):
    archive = tmp_path / "train2017.zip"
    archive.write_bytes(archive_bytes([("train2017/000000000009.jpg", b"original image")]).getvalue())
    monkeypatch.setattr(recovery.subprocess, "run", lambda *args, **kwargs: pytest.fail("External unzip must never be invoked"))
    assert recovery.test_zip(archive, "coco", 1)["passed"]


def test_archive_integrity_rejects_corrupt_image_crc(tmp_path):
    archive = tmp_path / "train2017.zip"
    payload = archive_bytes([("train2017/000000000009.jpg", b"original image")]).getvalue()
    archive.write_bytes(payload.replace(b"original image", b"corrupt! image"))
    with pytest.raises(ValueError, match="CRC"):
        recovery.test_zip(archive, "coco", 1)


@pytest.mark.parametrize("name", ["../bad.jpg", "/train2017/bad.jpg", "train2017/../../bad.jpg", "parquet/1.parquet"])
def test_unsafe_or_repacked_layout_rejected(name):
    with zipfile.ZipFile(archive_bytes([(name, b"payload")])) as archive:
        with pytest.raises(ValueError):
            recovery.audit_zip_structure(archive, "coco", 1)


def test_duplicate_and_symlink_rejected():
    with pytest.warns(UserWarning):
        payload = archive_bytes([("images/00000/000000010.jpg", b"first"), ("images/00000/000000010.jpg", b"second")])
    with zipfile.ZipFile(payload) as archive:
        with pytest.raises(ValueError, match="Duplicate"):
            recovery.audit_zip_structure(archive, "llava", 2)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        member = zipfile.ZipInfo("images/link.jpg")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "outside.jpg")
    payload.seek(0)
    with zipfile.ZipFile(payload) as archive:
        with pytest.raises(ValueError, match="symlink"):
            recovery.audit_zip_structure(archive, "llava", 1)


@pytest.mark.parametrize("member_path", ["images/00000/000000010.jpg", "00000/000000010.jpg"])
def test_image_extraction_preserves_original_paths_and_bytes(tmp_path, monkeypatch, member_path):
    monkeypatch.setattr(recovery, "extraction_root", lambda asset: tmp_path / "llava_pretrain")
    archive = tmp_path / "images.zip"
    original = b"original compressed image bytes, no processing"
    archive.write_bytes(archive_bytes([(member_path, original)]).getvalue())
    events = []
    assert recovery.extract_images(archive, "llava", 1, lambda **event: events.append(event)) == 1
    assert (tmp_path / "llava_pretrain/images/00000/000000010.jpg").read_bytes() == original
    assert events[0]["phase"] == "extracting"


def test_llava_mixed_layouts_and_normalized_duplicates_rejected():
    with zipfile.ZipFile(archive_bytes([("00000/000000010.jpg", b"first"),
                                      ("images/00000/000000012.jpg", b"second")])) as archive:
        with pytest.raises(ValueError, match="Mixed"):
            recovery.audit_zip_structure(archive, "llava", 2)
    with zipfile.ZipFile(archive_bytes([("00000/000000010.jpg", b"first"),
                                      ("images/00000/000000010.jpg", b"second")])) as archive:
        with pytest.raises(ValueError, match="Duplicate"):
            recovery.audit_zip_structure(archive, "llava", 2)


def test_independent_family_completion_preserves_other_snapshots(tmp_path, monkeypatch):
    import json

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(recovery, "RECOVERY", tmp_path)
    monkeypatch.setattr(recovery, "EVIDENCE", evidence)
    original = dict(checked_utc="2026-10-04T17:00:00+00:00", training_index_sha256=recovery.RECORDS_SHA,
        training_records=1245901, families={family:dict(required_paths=count, recovered_paths=0, missing_paths=count)
        for family,count in dict(sam=569486,coco=118287,llava=558128).items()})
    (tmp_path / "TRAIN_IMAGE_COMPLETENESS.json").write_text(json.dumps(original))
    progress = dict(completed=True, sha256_passed=True, zip_integrity_passed=True, index_missing=0,
                    recovered_required_images=118287, physical_image_count=118287)
    updated = recovery.publish_family_completeness("coco", progress)
    assert updated["families"]["coco"]["recovered_paths"] == 118287
    assert updated["families"]["sam"] == original["families"]["sam"]
    assert updated["family_checked_utc"]["sam"] == original["checked_utc"]
    assert updated["training_index_missing"] == 1127614
    assert not updated["passed"] and not updated["decode_audit"]["executed"]
    with pytest.raises(ValueError, match="not independently"):
        recovery.publish_family_completeness("coco", dict(progress, zip_integrity_passed=False))


def test_ten_minute_low_rate_and_fast_sample_reset():
    assert recovery.slow_source(0, 100, 599, "downloading") == (0, False)
    assert recovery.slow_source(0, 100, 600, "downloading") == (0, True)
    assert recovery.slow_source(0, 1024**2, 500, "downloading") == (500, False)
    assert recovery.slow_source(500, 100, 600, "downloading") == (500, False)
    assert recovery.slow_source(0, 0, 900, "extracting") == (None, False)


def test_completed_worker_not_touched_by_stop():
    class Process:
        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("Completed process must not be killed")

    recovery.stop_owned_process(Process())


def test_callback_factory_is_used_by_sdk_http_and_xet_context():
    events = []

    class Progress:
        def __init__(self, **kwargs):
            self.initial = kwargs.get("initial", 0)

        def update(self, amount):
            return None

        def close(self):
            return None

    sdk = SimpleNamespace(tqdm=Progress)
    recovery.install_progress_callback(sdk, lambda amount, **kwargs: events.append(amount))
    progress = sdk._get_progress_bar_context(log_level=10, initial=100, total=200)
    progress.update(7)
    progress.close()
    assert events == [7, 0]
    assert progress.initial == 100
    assert sdk._get_progress_bar_context(log_level=10, _tqdm_bar=progress).__enter__() is progress


def test_completeness_exposes_requested_missing_count_without_inventing_recovered_images(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(sa1b_recovery, "EVIDENCE", evidence)
    monkeypatch.setattr(sa1b_recovery, "RECOVERY", tmp_path)
    monkeypatch.setattr(sa1b_recovery, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(sa1b_recovery, "STATE", tmp_path / "absent-state.json")
    families = {"sam": {"sam/images/sa_1.jpg"}, "coco": {"coco/train2017/000000000009.jpg"},
                "llava": {"llava/llava_pretrain/images/00000/000000010.jpg"}}
    result = sa1b_recovery.image_completeness(families)
    assert result["training_index_missing"] == result["missing_training_paths"] == 3
    assert result["families"]["coco"]["recovered_paths"] == 0
    assert not result["passed"]


def test_ten_minute_average_cannot_be_evaded_by_one_buffered_burst():
    events = []
    for end in range(30, 601, 30):
        events.append(dict(asset="llava", repo="official/data", revision="fixed",
            checked_utc=datetime.datetime.fromtimestamp(end, datetime.timezone.utc).isoformat(),
            interval_seconds=30, interval_payload_bytes=100*1024**2 if end == 600 else 0))
    assert recovery.window_rate(events, "llava", "official/data", "fixed", 600) == 100*1024**2/600
    assert recovery.window_rate(events, "coco", "official/data", "fixed", 600) == 0


def test_rolling_rate_clips_old_samples_and_preserves_real_wall_time():
    events = [dict(asset="coco", repo="mirror/data", revision="fixed",
        checked_utc=datetime.datetime.fromtimestamp(30, datetime.timezone.utc).isoformat(),
        interval_seconds=30, interval_payload_bytes=60*1024**2)]
    assert recovery.window_rate(events, "coco", "mirror/data", "fixed", 610) == 40*1024**2/600
