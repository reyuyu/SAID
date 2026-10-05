"""Read only full-image URL parquet columns and fetch bounded audit samples."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit

import sa1b_recovery as recovery


REPO = "Tongyi-DataEngine/SA1B-Paired-Captions-Images"
REVISION = "4859a29a7aa78102d3364852755c7818e7583366"
FOLDER = recovery.EVIDENCE / "modelscope-sa1b"
BASE = f"https://modelscope.cn/api/v1/datasets/{REPO}/repo"
FULL_IMAGE_PREFIX = "https://modelscope.cn-beijing.oss.aliyuncs.com/open_data/sa-1b-cot-qwen/"


class RangeReader(io.RawIOBase):
    def __init__(self, path, size, budget=128 * 1024 * 1024):
        import requests

        self.http = requests
        self.url = BASE + f"?Revision={REVISION}&FilePath={quote(path)}"
        self.size = size
        self.position = 0
        self.bytes_fetched = 0
        self.requests = 0
        self.budget = budget
        self.cache = {}
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= target <= self.size:
            raise ValueError("Parquet range seek is outside the pinned file")
        self.position = target
        return target

    def read(self, length=-1):
        if length < 0 or length > self.budget:
            raise ValueError("Whole-parquet download is forbidden")
        length = min(length, self.size - self.position)
        if length == 0:
            return b""
        start = self.position
        key = (start, length)
        if key in self.cache:
            content = self.cache[key]
        else:
            if self.bytes_fetched + length > self.budget:
                raise ValueError("Metadata-column transfer budget exceeded")
            headers = {"Range": f"bytes={start}-{start+length-1}", "Accept-Encoding": "identity"}
            for attempt in range(3):
                try:
                    with self.session.get(self.url, headers=headers, stream=True, timeout=(10, 30)) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{start+length-1}/{self.size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Server did not honor the exact parquet range")
                        content = response.raw.read(length + 1)
                        if len(content) != length:
                            raise ValueError("Parquet range byte count differs from metadata")
                    break
                except self.http.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            self.bytes_fetched += length
            self.requests += 1
            if len(self.cache) < 128:
                self.cache[key] = content
        self.position += length
        return content

    def close(self):
        self.session.close()
        super().close()


def full_image_id(url):
    if not isinstance(url, str) or not url.startswith(FULL_IMAGE_PREFIX):
        return None
    parsed = urlsplit(url)
    if parsed.query or parsed.fragment:
        return None
    match = re.fullmatch(r"sa_(\d+)\.jpg", Path(parsed.path).name)
    return int(match.group(1)) if match else None


def original_inventory():
    state = recovery.load(recovery.STATE)
    eligible = {}
    for row in state["shards"]:
        if not row.get("tar_integrity_passed") or row.get("extraction_status") != "complete" or row.get("training_images_installed") is False:
            continue
        if row.get("observed_md5") != row["expected_checklist_checksum"] or row.get("observed_sha256") != row["expected_lfs_sha256"]:
            continue
        path = Path(row["image_inventory"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["image_inventory_sha256"]:
            raise ValueError("Normal-shard inventory digest mismatch: " + row["filename"])
        for line in content.splitlines():
            record = json.loads(line)
            match = re.fullmatch(r"sa_(\d+)\.jpg", record["basename"])
            if match:
                position = int(match.group(1))
                if position in eligible:
                    raise ValueError("Duplicate normal-shard original image ID")
                eligible[position] = dict(record, local_path=str(recovery.SAM_ROOT / record["basename"]))
    return eligible


def discover(count=200):
    import pyarrow.parquet as parquet

    eligible = original_inventory()
    files = recovery.load(FOLDER / "pinned-data-tree.json")["Data"]["Files"]
    selected = {}
    scans = []
    for record in files:
        if not record["Path"].endswith(".parquet"):
            continue
        reader = RangeReader(record["Path"], record["Size"])
        scan = dict(path=record["Path"], file_size=record["Size"], sha256=record["Sha256"],
                    revision=REVISION, selected_column="opensource_url", row_groups_scanned=0)
        try:
            table = parquet.ParquetFile(reader)
            scan.update(schema=str(table.schema_arrow), total_rows=table.metadata.num_rows)
            for group in range(table.num_row_groups):
                column = table.read_row_group(group, columns=["opensource_url"]).column("opensource_url")
                scan["row_groups_scanned"] += 1
                for url in column.to_pylist():
                    if isinstance(url, dict):
                        if url.get("bytes") is not None:
                            raise ValueError("Unexpected embedded image bytes; audit this schema before continuing")
                        url = url.get("path")
                    position = full_image_id(url)
                    if position in eligible and position not in selected:
                        selected[position] = dict(eligible[position], image_id=position, source_url=url,
                            parquet_path=record["Path"], parquet_sha256=record["Sha256"], row_group=group,
                            source_field="opensource_url", repo=REPO, revision=REVISION)
                        if len(selected) == count:
                            break
                if len(selected) == count:
                    break
        finally:
            scan.update(range_bytes_fetched=reader.bytes_fetched, range_requests=reader.requests)
            reader.close()
            scans.append(scan)
            recovery.atomic_json(FOLDER / "parquet-scans.json", scans)
            recovery.atomic_json(FOLDER / "audit-samples.json", dict(required_samples=count,
                selected_count=len(selected), complete=len(selected) == count, samples=list(selected.values())))
            print(record["Path"], "selected", len(selected), "range_bytes", reader.bytes_fetched, flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Fewer than200 independently indexed normal-shard images are available")


def download_samples():
    from concurrent.futures import ThreadPoolExecutor
    import requests

    manifest = recovery.load(FOLDER / "audit-samples.json")
    if not manifest["complete"] or manifest["selected_count"] < 200:
        raise ValueError("At least200 source-indexed distinct samples are required")
    folder = FOLDER / "audit-jpegs"
    folder.mkdir(parents=True, exist_ok=True)

    def one(row):
        destination = folder / row["basename"]
        proof_path = folder / (row["basename"] + ".json")
        existing = recovery.load(proof_path, {})
        if destination.is_file() and existing.get("sha256") == recovery.digest(destination):
            return existing
        result = dict(row, source_path=str(destination), checked_utc=recovery.now(), downloaded=False)
        for attempt in range(3):
            try:
                with requests.get(row["source_url"], stream=True, timeout=(10, 30)) as response:
                    result.update(http_status=response.status_code, etag=response.headers.get("ETag"),
                        source_content_type=response.headers.get("Content-Type"), attempts=attempt+1)
                    if response.status_code in (401, 403, 404):
                        break
                    response.raise_for_status()
                    content = response.raw.read(32 * 1024 * 1024 + 1)
                    if len(content) > 32 * 1024 * 1024 or not content.startswith(b"\xff\xd8"):
                        raise ValueError("Source is not a bounded original JPEG")
                temporary = destination.with_suffix(".jpg.download")
                temporary.write_bytes(content)
                temporary.replace(destination)
                result.update(downloaded=True, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
                break
            except requests.RequestException as error:
                result["error_type"] = type(error).__name__
                time.sleep(attempt + 1)
        recovery.atomic_json(proof_path, result)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, manifest["samples"]))
    recovery.atomic_json(FOLDER / "audit-downloads.json", dict(checked_utc=recovery.now(),
        samples=results, downloaded=sum(row["downloaded"] for row in results), requested=len(results),
        status_counts=dict(Counter(str(row.get("http_status")) for row in results))))
    print("audit downloads", sum(row["downloaded"] for row in results), "/", len(results), flush=True)


def locate_id(image_id):
    import pyarrow.parquet as parquet

    matches = []
    scans = []
    for record in recovery.load(FOLDER / "pinned-data-tree.json")["Data"]["Files"]:
        if not record["Path"].endswith(".parquet"):
            continue
        reader = RangeReader(record["Path"], record["Size"])
        try:
            table = parquet.ParquetFile(reader)
            for group in range(table.num_row_groups):
                for value in table.read_row_group(group, columns=["opensource_url"]).column("opensource_url").to_pylist():
                    url = value.get("path") if isinstance(value, dict) else value
                    if isinstance(url, str) and Path(urlsplit(url).path).name == f"sa_{image_id}.jpg":
                        matches.append(dict(url=url, path=record["Path"], sha256=record["Sha256"], row_group=group))
                if matches:
                    break
        finally:
            scans.append(dict(path=record["Path"], range_bytes_fetched=reader.bytes_fetched,
                range_requests=reader.requests))
            reader.close()
            recovery.atomic_json(FOLDER / f"exact-id-{image_id}.json", dict(image_id=image_id,
                matches=matches, scans=scans, selected_column="opensource_url", revision=REVISION))
            print(record["Path"], "ID", image_id, "matches", len(matches), flush=True)
        if matches:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "download-samples", "locate-id"))
    parser.add_argument("--image-id", type=int)
    arguments = parser.parse_args()
    if arguments.command == "discover":
        discover()
    elif arguments.command == "download-samples":
        download_samples()
    else:
        if arguments.image_id is None:
            parser.error("--image-id is required")
        locate_id(arguments.image_id)
