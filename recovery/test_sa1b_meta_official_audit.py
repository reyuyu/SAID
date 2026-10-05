import json
from urllib.error import HTTPError

import pytest

import sa1b_meta_official_audit as meta


def test_official_routes_exclude_current_mirror_objects():
    assert meta.SHARDS == (14, 16, 17)
    assert all("huggingface" not in url and "openxlab" not in url for url in meta.OFFICIAL_ROUTES)


def test_probe_records_timeout_without_claiming_permission_failure(monkeypatch):
    def timeout(*arguments, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(meta, "urlopen", timeout)
    result = meta.probe_route(meta.DOWNLOAD_URL)
    assert result["error_type"] == "TimeoutError"
    assert not result["reachable"]
    assert "login_required" not in result
    assert not result["permission_bypass_attempted"]


def test_probe_never_logs_signed_query(monkeypatch):
    class Response:
        status = 200
        url = "https://ai.meta.com/download?token=private"
        content = b"<title>Access denied</title>"
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return False

        def read(self):
            return self.content

        def geturl(self):
            return self.url

    monkeypatch.setattr(meta, "urlopen", lambda *arguments, **kwargs: Response())
    result = meta.probe_route("https://ai.meta.com/download?token=private")
    assert "private" not in json.dumps(result)
    assert result["successful_http_response"]


def test_http_denial_is_not_a_successful_download_page(monkeypatch):
    def denied(*arguments, **kwargs):
        raise HTTPError(meta.DOWNLOAD_URL, 403, "Forbidden", {}, None)

    monkeypatch.setattr(meta, "urlopen", denied)
    result = meta.probe_route(meta.DOWNLOAD_URL)
    assert result["status_code"] == 403 and result["reachable"]
    assert not result["successful_http_response"]


def test_historical_comparison_cannot_assert_meta_equivalence(tmp_path, monkeypatch):
    monkeypatch.setattr(meta.recovery, "EVIDENCE", tmp_path)
    rows = [dict(filename=f"sa_{position:06d}.tar", archive=str(tmp_path / "missing.tar"),
        expected_size_bytes=100, expected_checklist_checksum="a"*32, expected_lfs_sha256="b"*64,
        observed_md5="a"*32, observed_sha256="b"*64, tar_integrity_passed=False,
        required_jpeg_rescue=dict(training_index_sha256=meta.recovery.RECORDS_SHA, required_images=2,
            recovered_required_images=1, missing_required_image_ids=[123], remaining_decode_failures=0))
        for position in meta.SHARDS]
    result = meta.historical_comparison(dict(shards=rows))
    assert all(row["bitwise_same"] is None for row in result)
    assert all(row["meta_official"]["gnu_tar_passed"] is None for row in result)
    assert all(row["missing_required_images"] == 1 for row in result)
    rows[0]["required_jpeg_rescue"]["training_index_sha256"] = "wrong"
    with pytest.raises(ValueError, match="frozen index"):
        meta.historical_comparison(dict(shards=rows))
