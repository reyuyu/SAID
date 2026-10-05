import errno
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

import final_manifest_audit as audit


def image_row(path, family="sam", ordinal=1):
    return dict(ordinal=ordinal, family=family, absolute_path=str(path))


def test_exact_path_decodes_rgb_without_reencoding(tmp_path):
    path = tmp_path / "sa_123.jpg"
    Image.new("RGB", (17, 23), "red").save(path)
    before = path.read_bytes()
    result = audit.check_path(image_row(path), "decode")
    assert result["status"] == "pass"
    assert result["width"] == 17 and result["height"] == 23
    assert path.read_bytes() == before


def test_decode_converts_non_rgb_in_memory_only(tmp_path):
    path = tmp_path / "gray.jpg"
    Image.new("L", (19, 11), 17).save(path)
    before = path.read_bytes()
    result = audit.check_path(image_row(path), "decode")
    assert result["status"] == "pass" and result["rgb_mode"] == "RGB"
    assert path.read_bytes() == before


def test_truncated_jpeg_is_not_accepted(tmp_path):
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (40, 40), "blue").save(path)
    path.write_bytes(path.read_bytes()[:-30])
    result = audit.check_path(image_row(path), "decode")
    assert result["status"] == "corrupt"


def test_io_error_is_not_mislabeled_corruption(tmp_path, monkeypatch):
    def timeout(path):
        raise TimeoutError("temporary NFS timeout")

    monkeypatch.setattr(audit.os, "stat", timeout)
    sleeps = []
    result = audit.check_path(image_row(tmp_path / "any.jpg"), "decode", sleeper=sleeps.append)
    assert result["status"] == "io_error"
    assert result["io_retries"] == 3 and result["timeouts"] == 4
    assert sleeps == [.25, .5, 1]


def test_transient_io_retries_then_passes(tmp_path, monkeypatch):
    path = tmp_path / "valid.jpg"
    Image.new("RGB", (7, 9)).save(path)
    original = audit.os.stat
    attempts = []

    def flaky(filename):
        attempts.append(filename)
        if len(attempts) <= 2:
            raise OSError(errno.EIO, "NFS transient")
        return original(filename)

    monkeypatch.setattr(audit.os, "stat", flaky)
    result = audit.check_path(image_row(path), "existence", sleeper=lambda delay: None)
    assert result["status"] == "pass" and result["io_retries"] == 2


def test_missing_exact_path_is_missing_not_corrupt(tmp_path):
    result = audit.check_path(image_row(tmp_path / "missing.jpg"), "decode")
    assert result["status"] == "missing" and result["io_retries"] == 0


@pytest.mark.parametrize("relative", ["../sam/sa_1.jpg", "/sam/images/sa_1.jpg", "sam/other/sa_1.jpg", "coco/val2017/1.jpg", "unknown/images/1.jpg"])
def test_manifest_rejects_unsafe_or_changed_split_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        audit.describe_path(relative, tmp_path)


def prepare_fixture(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(dict(image=relative))+"\n" for relative in
        ["sam/images/sa_1.jpg", "coco/train2017/000000000001.jpg", "llava/images/sub/1.jpg"]))
    connection = audit.open_database(tmp_path / "checkpoint.sqlite3")
    expected = dict(sam=1,coco=1,llava=1)
    identity = audit.prepare_manifest(connection, tmp_path, records, expected, tmp_path / "assets",
        tmp_path / "published.jsonl", hashlib.sha256(records.read_bytes()).hexdigest())
    return connection,identity,records,expected


def test_manifest_creation_never_enumerates_image_directories(tmp_path, monkeypatch):
    def forbidden(*arguments, **kwargs):
        raise AssertionError("Filesystem directory enumeration is forbidden")

    monkeypatch.setattr(audit.os, "walk", forbidden)
    monkeypatch.setattr(audit.os, "scandir", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    connection,identity,records,expected = prepare_fixture(tmp_path)
    assert identity["required_paths_total"] == 3
    assert identity["expected"] == expected
    rows = [json.loads(line) for line in (tmp_path / "required_training_images.jsonl").read_text().splitlines()]
    assert all({"family","absolute_path","image_id","basename"} <= set(row) for row in rows)
    assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 3
    connection.close()


def test_manifest_resume_is_bound_to_frozen_source(tmp_path):
    connection,identity,records,expected = prepare_fixture(tmp_path)
    resumed = audit.prepare_manifest(connection, tmp_path, records, expected, tmp_path / "assets",
        tmp_path / "published.jsonl", identity["training_index_sha256"])
    assert resumed == identity
    records.write_text(records.read_text()+"\n")
    with pytest.raises(ValueError, match="changed"):
        audit.prepare_manifest(connection, tmp_path, records, expected, tmp_path / "assets",
            tmp_path / "published.jsonl", identity["training_index_sha256"])
    connection.close()


def test_duplicate_benchmark_success_does_not_inflate_coverage(tmp_path):
    connection,identity,records,expected = prepare_fixture(tmp_path)
    counts = audit.family_counts(connection)
    result = dict(ordinal=1,family="sam",stage="existence",status="pass",io_retries=0,error=None)
    assert audit.record_result(connection,result,counts)
    assert not audit.record_result(connection,result,counts)
    assert counts["sam"]["exists"] == 1
    connection.commit()
    assert audit.family_counts(connection)["sam"]["existence_checked"] == 1
    connection.close()


def test_worker_selection_prefers_lowest_stable_near_peak():
    rows = [dict(workers=workers,stable=True,completed=1000,duration_seconds=120,requested_seconds=120,paths_per_second=rate)
        for workers,rate in [(8,90),(16,100),(32,103)]]
    assert audit.choose_workers(rows) == 16
    rows[-1]["stable"] = False
    assert audit.choose_workers(rows) == 16
    for row in rows:
        row["stable"] = False
    with pytest.raises(RuntimeError):
        audit.choose_workers(rows)


def test_later_benchmark_failure_cannot_be_hidden_by_previous_pass(tmp_path):
    connection,identity,records,expected = prepare_fixture(tmp_path)
    counts = audit.family_counts(connection)
    success = dict(ordinal=1,family="sam",stage="decode",status="pass",io_retries=0,error=None)
    failure = dict(success,status="corrupt",error="truncated")
    assert audit.record_result(connection,success,counts)
    assert audit.record_result(connection,failure,counts)
    assert counts["sam"]["decode_checked"] == 1
    assert counts["sam"]["decoded"] == 0 and counts["sam"]["corrupt"] == 1
    connection.close()


def test_missing_existence_evidence_is_exported_without_decode(tmp_path, monkeypatch):
    connection,identity,records,expected = prepare_fixture(tmp_path)
    counts = audit.family_counts(connection)
    failure = dict(ordinal=1,family="sam",stage="existence",status="missing",io_retries=0,error="not found")
    audit.record_result(connection,failure,counts)
    monkeypatch.setattr(audit,"LOCAL",tmp_path)
    audit.write_bad_images(connection)
    rows = [json.loads(line) for line in (tmp_path / "bad-images.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["existence"] == "missing"
    assert rows[0]["decode"] == "pending"
    connection.close()


def test_process_workers_check_only_supplied_paths(tmp_path):
    path = tmp_path / "sa_1.jpg"
    Image.new("RGB", (11,13)).save(path)
    pool = audit.ExactPathPool(2,"decode")
    try:
        assert pool.send(iter([image_row(path)])) == 1
        import time

        deadline = time.monotonic()+10
        results = []
        while pool.busy() and time.monotonic() < deadline:
            results.extend(pool.poll())
            time.sleep(.01)
        assert len(results) == 1 and results[0]["status"] == "pass"
    finally:
        pool.close()


def test_partial_existence_is_not_data_audit_pass(tmp_path):
    connection,identity,records,expected = prepare_fixture(tmp_path)
    result = audit.snapshot(identity,audit.family_counts(connection),"EXISTENCE_AUDIT",identity["started_timestamp"],{})
    assert not result["passed"] and result["decoded_count"] == 0
    assert result["required_paths_total"] == 3
    connection.close()


def test_end_to_end_exact_audit_is_resumable_and_never_launches_training(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    evidence = recovery / "evidence"
    evidence.mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "data_index").mkdir(parents=True)
    image_root = tmp_path / "assets" / "training" / "ShareGPT4V"
    paths = ["sam/images/sa_1.jpg", "coco/train2017/000000000001.jpg", "llava/images/sub/1.jpg"]
    hashes = {}
    for relative in paths:
        path = image_root / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new("RGB", (7,9)).save(path)
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    records = runtime / "data_index/records.jsonl"
    records.write_text("".join(json.dumps(dict(image=relative))+"\n" for relative in paths))
    index_sha = hashlib.sha256(records.read_bytes()).hexdigest()
    expected = dict(sam=1,coco=1,llava=1)
    for name,value in dict(LOCAL=local,RECOVERY=recovery,EVIDENCE=evidence,RUNTIME=runtime,
        ASSETS=tmp_path / "assets",MANIFEST=recovery / "required_training_images.jsonl",
        FINAL=recovery / "FINAL_IMAGE_AUDIT.json",REPORT=recovery / "FINAL_MANIFEST_AUDIT.md",
        TOTAL=3,EXPECTED=expected,RECORDS_SHA=index_sha).items():
        monkeypatch.setattr(audit,name,value)
    (evidence / "training-audit.json").write_text(json.dumps(dict(annotation=True,training_records=3,skip=1000,index_sha256=index_sha)))
    original_prepare = audit.prepare_manifest

    def prepare(connection):
        return original_prepare(connection,local,records,expected,image_root,audit.MANIFEST,index_sha)

    monkeypatch.setattr(audit,"prepare_manifest",prepare)
    monkeypatch.setattr(audit,"validate_recovery_sources",lambda: dict(path_source="test frozen manifest"))
    result = audit.run(seconds=.02)
    assert result["status"] == "DATA_AUDIT_PASS"
    assert result["exists_count"] == result["decoded_count"] == 3
    assert result["corrupt_count"] == result["missing_count"] == result["io_error_count"] == 0
    assert not result["smoke_started"] and not result["formal_training_started"]
    assert audit.run(seconds=.02)["status"] == "DATA_AUDIT_PASS"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==value for path,value in hashes.items())
    assert json.loads((recovery / "TRAIN_IMAGE_COMPLETENESS.json").read_text())["passed"]


def test_legacy_final_decode_entry_routes_to_exact_manifest_without_scan(monkeypatch):
    import sa1b_recovery

    calls = []

    def exact_run():
        calls.append("exact")
        return dict(passed=False)

    monkeypatch.setattr(audit,"run",exact_run)
    monkeypatch.setattr(audit.os,"scandir",lambda *arguments: (_ for _ in ()).throw(AssertionError("Directory scan forbidden")))
    with pytest.raises(RuntimeError,match="did not pass"):
        sa1b_recovery.image_completeness({"sam":set()},decode=True)
    assert calls == ["exact"]
