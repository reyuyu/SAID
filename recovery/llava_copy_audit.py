"""Gate the explicitly approved 332F copy against official ZIP metadata and bytes."""

import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.parse
import zipfile
import zlib

from audit import ASSETS, EVIDENCE, load
from hf_training_recovery import audit_zip_structure, installed_image_path
from sa1b_recovery import atomic_json, now, requirements


ORIGINAL = "liuhaotian/LLaVA-Pretrain"
REVISION = "70f9d1e5e1a697fe35830875cfc7de1dd590d727"
COPY = "332F/LLaVA-Pretrain"
EXPECTED_SHA = "05459d8cb059bd32322b1c466c1cbd4568b09b1ce1db748425b7977236912660"
EXPECTED_SIZE = 27356108382
RESULT = EVIDENCE / "llava-copy-identity-audit.json"


def validate_file_identity(original, duplicate):
    if original["expected_sha256"] != EXPECTED_SHA or original["expected_size_bytes"] != EXPECTED_SIZE:
        raise ValueError("Official immutable ZIP identity differs from previously recorded metadata")
    if (duplicate["expected_sha256"], duplicate["expected_size_bytes"]) != (EXPECTED_SHA, EXPECTED_SIZE):
        raise ValueError("332F is not a byte-identical original archive; refusing a repack")


def fetch_range(session, url, start, length):
    if start < 0 or length < 0 or length > 128 * 1024**2:
        raise ValueError("Invalid or excessive metadata range")
    if not length:
        return b""
    end = start + length - 1
    last_error = None
    for attempt in range(3):
        response = None
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                   stream=True, timeout=(10, 45))
            if response.status_code != 206:
                raise ValueError("Remote ZIP did not honor Range; refusing to read the whole 27GB archive")
            if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise ValueError("Remote ZIP returned a different byte range")
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=1024**2):
                payload.extend(chunk)
                if len(payload) > length:
                    raise ValueError("Remote range exceeds requested metadata budget")
            if len(payload) != length:
                raise ValueError("Truncated ZIP metadata range")
            return bytes(payload)
        except ValueError:
            raise
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2)
        finally:
            if response is not None:
                response.close()
    raise RuntimeError("ZIP metadata transport failed") from last_error


class RemoteZip(io.RawIOBase):
    def __init__(self, session, url, size):
        self.session = session
        self.url = url
        self.size = size
        self.position = 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError("Invalid ZIP seek mode")
        if position < 0:
            raise ValueError("Negative ZIP seek")
        self.position = position
        return position

    def read(self, amount=-1):
        if amount < 0:
            amount = self.size - self.position
        amount = max(0, min(amount, self.size-self.position))
        if self.transferred+amount > 256 * 1024**2:
            raise ValueError("ZIP metadata-only transfer budget exceeded")
        payload = fetch_range(self.session, self.url, self.position, amount)
        self.position += len(payload)
        self.transferred += len(payload)
        return payload


def metadata_discovery(record):
    import requests
    from huggingface_hub import HfApi, configure_http_backend, get_hf_file_metadata, hf_hub_url

    class BoundedSession(requests.Session):
        def request(self, method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (10, 45)
            if endpoint != "https://huggingface.co" and url.startswith("https://huggingface.co/api/"):
                url = endpoint + url.removeprefix("https://huggingface.co")
            return super().request(method, url, **kwargs)

    cached = load(EVIDENCE / "hf-training-discovery.json")
    official = next(row for row in cached["assets"] if row["asset"] == "llava")
    if official["repo"] != ORIGINAL or official["revision"] != REVISION:
        raise ValueError("Previously pinned official metadata identity missing")
    record["official"] = official
    record["metadata_attempts"] = []
    for cycle in range(3):
        for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
            configure_http_backend(backend_factory=BoundedSession)
            api = HfApi(endpoint=endpoint, token=False)
            try:
                info = api.dataset_info(COPY)
                paths = api.list_repo_files(COPY, repo_type="dataset", revision=info.sha)
                if "images.zip" not in paths:
                    raise ValueError("Approved duplicate has no images.zip")
                entry = api.get_paths_info(COPY, ["images.zip"], repo_type="dataset", revision=info.sha)[0]
                duplicate = dict(asset="llava", repo=COPY, revision=info.sha, filename="images.zip",
                    expected_sha256=entry.lfs.sha256 if entry.lfs else None, expected_size_bytes=entry.size,
                    endpoint=endpoint, files=paths)
                validate_file_identity(official, duplicate)
                original_url = hf_hub_url(ORIGINAL, "images.zip", repo_type="dataset", revision=REVISION, endpoint=endpoint)
                duplicate_url = hf_hub_url(COPY, "images.zip", repo_type="dataset", revision=info.sha, endpoint=endpoint)
                original_head = get_hf_file_metadata(original_url, token=False)
                duplicate_head = get_hf_file_metadata(duplicate_url, token=False)
                if original_head.size != EXPECTED_SIZE or original_head.etag != EXPECTED_SHA:
                    raise ValueError("Live official immutable ZIP HEAD differs from expected identity")
                if duplicate_head.size != EXPECTED_SIZE or duplicate_head.etag != EXPECTED_SHA:
                    raise ValueError("Duplicate immutable ZIP HEAD differs from expected identity")
                duplicate["xet_metadata_available"] = duplicate_head.xet_file_data is not None
                record.update(duplicate=duplicate, byte_identical_lfs_size_and_sha256=True,
                              official_live_metadata_verified=True)
                record["metadata_attempts"].append(dict(cycle=cycle, endpoint=endpoint, passed=True, checked_utc=now()))
                atomic_json(RESULT, record)
                return original_head.location, duplicate_head.location
            except ValueError:
                raise
            except Exception as error:
                record["metadata_attempts"].append(dict(cycle=cycle, endpoint=endpoint, passed=False,
                    error=str(error), checked_utc=now()))
                atomic_json(RESULT, record)
        time.sleep(5)
    raise RuntimeError("Neither official HF nor HF API relay provides reachable immutable metadata")


def image_manifest(archive):
    members = audit_zip_structure(archive, "llava", 558128)
    names = {member.filename for member in members}
    canonical = "\n".join(sorted(names)) + "\n"
    return members, names, hashlib.sha256(canonical.encode()).hexdigest(), canonical


def read_partial_image(partial, member):
    with Path(partial).open("rb") as source:
        source.seek(member.header_offset)
        header = source.read(30)
        if len(header) != 30:
            raise ValueError("Official partial lacks complete local header")
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04":
            raise ValueError("Official partial has no matching original ZIP local header")
        name = source.read(fields[9])
        if len(name) != fields[9]:
            raise ValueError("Official partial lacks complete member name")
        if name.decode("utf-8" if fields[2] & 0x800 else "cp437") != member.filename:
            raise ValueError("Official partial member name differs from immutable archive directory")
        source.seek(fields[10], 1)
        if member.compress_size > 16 * 1024**2:
            raise ValueError("Image sample exceeds bounded transfer size")
        payload = source.read(member.compress_size)
        if len(payload) != member.compress_size:
            raise ValueError("Official partial lacks complete sampled image")
    if member.compress_type == zipfile.ZIP_STORED:
        content = payload
    elif member.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(payload, -15)
    else:
        raise ValueError("Unsupported original image ZIP compression")
    if len(content) != member.file_size or zlib.crc32(content) & 0xffffffff != member.CRC:
        raise ValueError("Official partial sampled image fails size/CRC verification")
    return content


def select_samples(members, partial, count=5):
    available = sorted((member for member in members if member.header_offset+30+len(member.filename.encode())+64+member.compress_size < Path(partial).stat().st_size
                        and member.file_size <= 16 * 1024**2), key=lambda member: member.header_offset)
    if len(available) < count:
        raise ValueError("Official partial has too few comparable original images")
    return [available[position*(len(available)-1)//(count-1)] for position in range(count)]


def main():
    import requests

    record = dict(started_utc=now(), official_repo=ORIGINAL, official_revision=REVISION,
                  approved_copy=COPY, passed=False, phase="metadata_discovery", image_samples=[])
    atomic_json(RESULT, record)
    try:
        original_url, duplicate_url = metadata_discovery(record)
        session = requests.Session()
        original_reader = RemoteZip(session, original_url, EXPECTED_SIZE)
        duplicate_reader = RemoteZip(session, duplicate_url, EXPECTED_SIZE)
        record["phase"] = "zip_directory_comparison"
        atomic_json(RESULT, record)
        with zipfile.ZipFile(original_reader) as official_zip, zipfile.ZipFile(duplicate_reader) as duplicate_zip:
            original_members, original_names, original_sha, canonical = image_manifest(official_zip)
            duplicate_members, duplicate_names, duplicate_sha, unused = image_manifest(duplicate_zip)
            required = {path.removeprefix("llava/llava_pretrain/") for path in requirements()["llava"]}
            installed_names = {str(installed_image_path(name, "llava")) for name in original_names}
            if original_names != duplicate_names or installed_names != required:
                raise ValueError("Official/copy ZIP image paths differ from each other or the frozen 558128-image index")
            manifest_path = EVIDENCE / "llava-original-image-paths.txt"
            manifest_path.write_text(canonical)
            record.update(image_paths_equal=True, required_image_paths_equal=True, image_count=len(original_names),
                          official_path_set_sha256=original_sha, copy_path_set_sha256=duplicate_sha,
                          installation_root=str(ASSETS / "training/ShareGPT4V/llava/llava_pretrain/images"),
                          image_paths_manifest=str(manifest_path), phase="official_partial_byte_samples")
            atomic_json(RESULT, record)
            cache = ASSETS / "downloads/hf_training/llava" / ORIGINAL.replace("/", "--") / REVISION / ".cache/huggingface/download"
            partials = list(cache.glob("*."+EXPECTED_SHA+".incomplete"))
            if not partials:
                raise ValueError("No official revision-pinned SDK partial available for byte comparison")
            partial = max(partials, key=lambda path: path.stat().st_size)
            record["official_partial"] = dict(path=str(partial), size_bytes=partial.stat().st_size)
            for member in select_samples(original_members, partial):
                original_content = read_partial_image(partial, member)
                duplicate_content = duplicate_zip.read(member.filename)
                original_hash = hashlib.sha256(original_content).hexdigest()
                duplicate_hash = hashlib.sha256(duplicate_content).hexdigest()
                if original_hash != duplicate_hash:
                    raise ValueError("332F sampled image differs from original downloaded image bytes")
                record["image_samples"].append(dict(path=member.filename, original_zip_offset=member.header_offset,
                    bytes=len(original_content), original_partial_sha256=original_hash,
                    duplicate_image_sha256=duplicate_hash, matched=True))
                atomic_json(RESULT, record)
        record.update(passed=True, phase="passed", finished_utc=now(),
                      official_metadata_bytes=original_reader.transferred, duplicate_metadata_bytes=duplicate_reader.transferred,
                      transport_hosts=[urllib.parse.urlparse(original_url).hostname, urllib.parse.urlparse(duplicate_url).hostname])
        atomic_json(RESULT, record)
        print(json.dumps({key: record[key] for key in ["passed", "duplicate", "image_count", "image_samples"]}), flush=True)
    except Exception as error:
        record.update(passed=False, phase="failed", finished_utc=now(), error_type=type(error).__name__, error=str(error))
        atomic_json(RESULT, record)
        raise


if __name__ == "__main__":
    main()
