import gzip
import io
import tarfile
import zlib

import pytest

import sa1b_wallclock_recovery as wallclock
import sa1b_fast_recovery as fast
import sa1b_opendatalab_probe as probe


def make_tar(name, data):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def test_nearly_complete_partial_remains_hf():
    decision = wallclock.migration_decision(11_000_000_000, 10_900_000_000, 4.6, 48)
    assert not decision["migrate"]
    assert decision["odl_effective_per_shard_mib_s"] == 16


def test_slow_partial_migrates_only_after_30_percent_saving():
    decision = wallclock.migration_decision(11_000_000_000, 500_000_000, 4.6, 48)
    assert decision["migrate"]
    assert decision["predicted_time_saving_fraction"] > .30
    with pytest.raises(ValueError):
        wallclock.migration_decision(11, 1, 0, 48)


def test_tolerant_scanner_skips_bad_headers_without_changing_bytes(tmp_path):
    data = b"unchanged original member bytes"
    raw = b"invalid header".ljust(512, b"!")+make_tar("./sa_195618.jpg", data)
    result = wallclock.scan_tar(io.BytesIO(raw), {"sa_195618.jpg"}, tmp_path)
    assert result["returncode"] == 0
    assert result["invalid_header_blocks"] == 1
    assert (tmp_path / "sa_195618.jpg").read_bytes() == data


def test_scanner_rejects_traversal_and_does_not_install_partial(tmp_path):
    raw = make_tar("../sa_195618.jpg", b"malicious")
    result = wallclock.scan_tar(io.BytesIO(raw), {"sa_195618.jpg"}, tmp_path)
    assert result["recovered_members"] == []
    assert not (tmp_path / "sa_195618.jpg").exists()
    raw = make_tar("sa_195618.jpg", b"original bytes")[:516]
    result = wallclock.scan_tar(io.BytesIO(raw), {"sa_195618.jpg"}, tmp_path)
    assert result["returncode"] == 1
    assert not (tmp_path / "sa_195618.jpg").exists()


def test_scanner_supports_gzip_stream(tmp_path):
    raw = gzip.compress(make_tar("sa_156665.jpg", b"original"))
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        result = wallclock.scan_tar(stream, {"sa_156665.jpg"}, tmp_path)
    assert result["recovered_members"] == ["sa_156665.jpg"]


def test_failed_hf_download_never_enters_ingestion(monkeypatch):
    filename = "sa_000021.tar"
    plan = dict(owners={filename: "HF_EXISTING_PARTIAL_ONLY"}, initial_hf_bytes={filename: 100})
    monkeypatch.setattr(fast.recovery, "download_shard", lambda state, row: dict(download_status="failed"))
    with pytest.raises(RuntimeError, match="partial retained"):
        fast.hf_job({}, dict(filename=filename), plan)


def test_forensic_target_is_independent_and_16_is_forbidden():
    assert "forensic/sa_000014/download" in str(wallclock.forensic_target("sa_000014.tar"))
    with pytest.raises(ValueError, match="Only14"):
        wallclock.forensic_job(dict(filename="sa_000016.tar"))


def test_stalled_sdk_watchdog_bounds_failed_range_wait_without_interrupting_assembly():
    assert not probe.payload_stalled(dict(last_payload_monotonic=100), 0, 399)
    assert probe.payload_stalled(dict(last_payload_monotonic=100), 0, 400)
    assert not probe.payload_stalled(dict(last_payload_monotonic=100), 0, 700, archive_recent=True)
    assert not probe.payload_stalled({}, 0, 599)
    assert probe.payload_stalled({}, 0, 600)


def test_deflate_tail_recovers_original_member_after_a_bad_first_block(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in [("junk.bin", b"prefix"*25000), ("./sa_195618.jpg", b"original JPEG member bytes")]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    compressor = zlib.compressobj(level=0, wbits=31)
    data = bytearray(compressor.compress(stream.getvalue())+compressor.flush())
    data[10] = 6
    source = tmp_path / "upstream-invalid.tar"
    source.write_bytes(data)
    result = wallclock.extract_deflate_tail(source, {"sa_195618.jpg"}, tmp_path / "output")
    assert result["status"] == "RECOVERED"
    assert (tmp_path / "output/sa_195618.jpg").read_bytes() == b"original JPEG member bytes"
    assert source.read_bytes() == data
    assert result["member_proofs"][0]["gzip_footer_exact"]


def test_deflate_tail_does_not_invent_a_missing_member(tmp_path):
    source = tmp_path / "archive.gz"
    source.write_bytes(gzip.compress(b"no matching member"))
    result = wallclock.extract_deflate_tail(source, {"sa_195618.jpg"}, tmp_path / "output")
    assert result["status"] == "STILL_MISSING"
    assert not (tmp_path / "output/sa_195618.jpg").exists()


def test_parallel_original_ownership_rejects_duplicates_without_overwriting():
    ownership = {}
    fast.commit_ownership(ownership, ["sa_1.jpg", "sa_2.jpg"], "sa_000031.tar")
    with pytest.raises(ValueError, match="duplicate"):
        fast.commit_ownership(ownership, ["sa_2.jpg", "sa_3.jpg"], "sa_000032.tar")
    assert ownership == {"sa_1.jpg": "sa_000031.tar", "sa_2.jpg": "sa_000031.tar"}
