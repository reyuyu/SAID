import io
from types import SimpleNamespace
import zipfile

import pytest

import llava_copy_audit as audit


def test_approved_copy_requires_identical_archive_not_only_a_similar_repo_name():
    official = dict(expected_sha256=audit.EXPECTED_SHA, expected_size_bytes=audit.EXPECTED_SIZE)
    audit.validate_file_identity(official, dict(official))
    with pytest.raises(ValueError, match="repack"):
        audit.validate_file_identity(official, dict(official, expected_sha256="a"*64))
    with pytest.raises(ValueError, match="previously recorded"):
        audit.validate_file_identity(dict(official, expected_size_bytes=1), official)


def test_range_refuses_full_archive_response_without_reading_body():
    closed = []
    response = SimpleNamespace(status_code=200, close=lambda: closed.append(True))
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="whole 27GB"):
        audit.fetch_range(session, "https://example.invalid/images.zip", 0, 30)
    assert closed == [True]


def test_range_bounds_and_exact_content_range():
    session = SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(status_code=206,
        headers={"Content-Range": "bytes 0-2/100"}, iter_content=lambda **kwargs: [b"abc"], close=lambda: None))
    assert audit.fetch_range(session, "https://example.invalid/images.zip", 0, 3) == b"abc"
    with pytest.raises(ValueError, match="different byte"):
        audit.fetch_range(session, "https://example.invalid/images.zip", 10, 3)
    with pytest.raises(ValueError, match="excessive"):
        audit.fetch_range(session, "https://example.invalid/images.zip", 0, 129*1024**2)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
@pytest.mark.parametrize("member_path", ["images/00000/000000010.jpg", "00000/000000010.jpg"])
def test_image_bytes_read_from_official_partial_with_crc_verification(tmp_path, compression, member_path):
    path = tmp_path / "images.zip.incomplete"
    payload = io.BytesIO()
    original = b"original JPEG encoded bytes, no resize or recompression"*20
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        archive.writestr(member_path, original)
    path.write_bytes(payload.getvalue())
    with zipfile.ZipFile(payload) as archive:
        member = archive.infolist()[0]
        assert audit.read_partial_image(path, member) == original
        path.write_bytes(path.read_bytes()[:40])
        with pytest.raises(ValueError, match="lacks complete"):
            audit.read_partial_image(path, member)


def test_metadata_reader_uses_bounded_seeks_and_ranges(monkeypatch):
    monkeypatch.setattr(audit, "fetch_range", lambda session, url, start, length: b"a"*length)
    reader = audit.RemoteZip(None, "https://example.invalid/images.zip", 100)
    assert reader.seek(-10, 2) == 90
    assert reader.read() == b"a"*10
    assert reader.tell() == 100 and reader.transferred == 10
    with pytest.raises(ValueError, match="Negative"):
        reader.seek(-1)
