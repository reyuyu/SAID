"""Resumable exact-path existence and full-decode audit; never launch training."""

import argparse
from collections import Counter
import datetime
import errno
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import signal
import sqlite3
import stat
import time

from audit import ASSETS, EVIDENCE, RECOVERY, RECORDS_SHA, ROOT, RUNTIME, digest, load


EXPECTED = {"sam": 569486, "coco": 118287, "llava": 558128}
TOTAL = sum(EXPECTED.values())
LOCAL = Path("/root/.cache/said-recovery/final-manifest-audit")
MANIFEST = RECOVERY / "required_training_images.jsonl"
FINAL = RECOVERY / "FINAL_IMAGE_AUDIT.json"
REPORT = RECOVERY / "FINAL_MANIFEST_AUDIT.md"
BENCHMARK_SECONDS = 120
CHUNK_SIZE = 5000
IO_ERRNOS = {errno.EIO, errno.ETIMEDOUT, errno.EAGAIN, errno.EINTR,
             errno.ESTALE, errno.ECONNRESET, errno.ENETUNREACH, errno.EHOSTUNREACH}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def describe_path(relative, image_root):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError("Unsafe image path in frozen index")
    family = path.parts[0]
    if family not in EXPECTED:
        raise ValueError("Unexpected training-image family")
    if family == "sam" and path.parts[1] != "images":
        raise ValueError("Unexpected SAM path")
    if family == "coco" and path.parts[1] != "train2017":
        raise ValueError("Unexpected COCO split")
    return dict(family=family, absolute_path=str(image_root / path),
                basename=path.name, image_id=path.stem, relative_path=str(path))


def open_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS images (
        ordinal INTEGER PRIMARY KEY, family TEXT NOT NULL,
        absolute_path TEXT NOT NULL UNIQUE, basename TEXT NOT NULL,
        image_id TEXT NOT NULL, relative_path TEXT NOT NULL,
        existence TEXT NOT NULL DEFAULT 'pending', decode TEXT NOT NULL DEFAULT 'pending',
        existence_retries INTEGER NOT NULL DEFAULT 0, decode_retries INTEGER NOT NULL DEFAULT 0,
        existence_error TEXT, decode_error TEXT)""")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return connection


def metadata(connection):
    return {name: json.loads(value) for name, value in connection.execute("SELECT name,value FROM metadata")}


def save_metadata(connection, name, value):
    connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (name, json.dumps(value)))


def publish_manifest_link(local_manifest, manifest):
    if manifest.is_symlink():
        if os.readlink(manifest) != str(local_manifest):
            raise RuntimeError("Existing manifest points to a different source")
    elif manifest.exists():
        if digest(manifest) != digest(local_manifest):
            raise RuntimeError("Refusing to overwrite a different required-path manifest")
    else:
        manifest.symlink_to(local_manifest)


def prepare_manifest(connection, local_folder=LOCAL, records=None, expected=None,
                     image_root=None, manifest=None, expected_index_sha=RECORDS_SHA):
    records = records or RUNTIME / "data_index/records.jsonl"
    expected = expected or EXPECTED
    image_root = image_root or ASSETS / "training/ShareGPT4V"
    manifest = manifest or MANIFEST
    local_manifest = local_folder / "required_training_images.jsonl"
    identity = metadata(connection)
    if identity.get("manifest_complete"):
        if identity["training_index_sha256"] != expected_index_sha:
            raise ValueError("Checkpoint uses a different frozen index")
        if digest(records) != expected_index_sha or digest(local_manifest) != identity["manifest_sha256"]:
            raise ValueError("Frozen index or required manifest changed since checkpoint")
        if identity["expected"] != expected or identity["image_root"] != str(image_root):
            raise ValueError("Checkpoint uses different image roots/counts")
        if connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] != sum(expected.values()):
            raise ValueError("Incomplete checkpoint path table")
        publish_manifest_link(local_manifest, manifest)
        return identity
    connection.execute("DELETE FROM images")
    seen = set()
    counts = Counter()
    records_count = 0
    index_hash = hashlib.sha256()
    manifest_hash = hashlib.sha256()
    temporary = local_manifest.with_suffix(".jsonl.pending")
    with records.open("rb") as source, temporary.open("wb") as output:
        batch = []
        for raw in source:
            index_hash.update(raw)
            record = json.loads(raw)
            records_count += 1
            row = describe_path(record["image"], image_root)
            if row["relative_path"] in seen:
                continue
            seen.add(row["relative_path"])
            row["ordinal"] = len(seen)
            counts[row["family"]] += 1
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            output.write(encoded)
            manifest_hash.update(encoded)
            batch.append(tuple(row[key] for key in ["ordinal", "family", "absolute_path", "basename", "image_id", "relative_path"]))
            if len(batch) == CHUNK_SIZE:
                connection.executemany("INSERT INTO images (ordinal,family,absolute_path,basename,image_id,relative_path) VALUES (?,?,?,?,?,?)", batch)
                connection.commit()
                batch.clear()
                print("manifest", records_count, "records", len(seen), "unique paths", flush=True)
        if batch:
            connection.executemany("INSERT INTO images (ordinal,family,absolute_path,basename,image_id,relative_path) VALUES (?,?,?,?,?,?)", batch)
    if index_hash.hexdigest() != expected_index_sha:
        raise ValueError("Frozen training index SHA256 differs")
    if dict(counts) != expected or records_count != sum(expected.values()):
        raise ValueError("Training records / deduplicated family counts differ from frozen protocol")
    temporary.replace(local_manifest)
    identity = dict(manifest_complete=True, training_index_sha256=expected_index_sha,
        manifest_sha256=manifest_hash.hexdigest(), expected=expected, required_paths_total=len(seen),
        records_count=records_count, duplicate_record_paths=records_count-len(seen),
        image_root=str(image_root), started_utc=now(), started_timestamp=time.time(),
        schema_version=1, recursive_filesystem_scan=False)
    for key, value in identity.items():
        save_metadata(connection, key, value)
    connection.commit()
    publish_manifest_link(local_manifest, manifest)
    atomic_json(local_folder / "manifest-identity.json", identity)
    return identity


def validate_recovery_sources():
    source = RECOVERY / "RECOVERY_MANIFEST.json"
    assets = {row["id"]: row for row in load(source)["assets"]}
    index = assets["data_index"]
    if index["path"] != str(RUNTIME / "data_index/records.jsonl") or index["sha256"] != RECORDS_SHA or index["training_records"] != TOTAL:
        raise ValueError("Recovery manifest does not identify the frozen training index")
    for family, asset_id in {"sam": "sam_training", "coco": "coco_train2017", "llava": "llava_images"}.items():
        row = assets[asset_id]
        path = PurePosixPath(row["path"])
        root = PurePosixPath(str(ASSETS / "training/ShareGPT4V"))
        if not path.is_relative_to(root / family) or row["required_training_paths"] != EXPECTED[family]:
            raise ValueError("Recovery manifest image root/count differs from frozen protocol")
    return dict(recovery_manifest=str(source), recovery_manifest_sha256=digest(source),
        path_source="Frozen training index; family roots/counts cross-checked against recovery manifest")


def classify_error(error, stage):
    if isinstance(error, FileNotFoundError):
        return "missing"
    if isinstance(error, (TimeoutError, PermissionError)):
        return "io_error"
    if isinstance(error, OSError) and error.errno is not None:
        return "io_error"
    return "corrupt" if stage == "decode" else "io_error"


def transient_io(error):
    return isinstance(error, TimeoutError) or (isinstance(error, OSError) and error.errno in IO_ERRNOS)


def timeout_handler(signum, frame):
    raise TimeoutError("Exact-path operation exceeded bounded NFS timeout")


def check_path(row, stage, max_retries=3, timeout_seconds=0, sleeper=time.sleep):
    started = time.monotonic()
    result = dict(ordinal=row["ordinal"], family=row["family"], stage=stage,
                  io_retries=0, io_errors=0, timeouts=0, error=None)
    for attempt in range(max_retries+1):
        try:
            if timeout_seconds:
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            details = os.stat(row["absolute_path"])
            if not stat.S_ISREG(details.st_mode):
                raise OSError(errno.EINVAL, "Required image is not a regular file")
            if details.st_size <= 0:
                raise ValueError("Required image is empty")
            if stage == "decode":
                from PIL import Image, ImageFile

                ImageFile.LOAD_TRUNCATED_IMAGES = False
                with Image.open(row["absolute_path"]) as image:
                    image.load()
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("Empty image dimensions")
                    rgb = image.convert("RGB")
                    rgb.load()
                    result.update(width=image.width, height=image.height, rgb_mode=rgb.mode)
                    rgb.close()
            result["status"] = "pass"
            break
        except Exception as error:
            if timeout_seconds:
                signal.setitimer(signal.ITIMER_REAL, 0)
            kind = classify_error(error, stage)
            result["io_errors"] += int(kind == "io_error")
            result["timeouts"] += int(isinstance(error, TimeoutError))
            if transient_io(error) and attempt < max_retries:
                result["io_retries"] += 1
                sleeper(min(.25 * 2**attempt, 1))
                continue
            result.update(status=kind, error=f"{type(error).__name__}: {error}")
            break
        finally:
            if timeout_seconds:
                signal.setitimer(signal.ITIMER_REAL, 0)
    result["latency_seconds"] = time.monotonic()-started
    return result


def worker_loop(pipe, stage, timeout_seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        while True:
            row = pipe.recv()
            if row is None:
                return
            pipe.send(check_path(row, stage, timeout_seconds=timeout_seconds))
    except (EOFError, BrokenPipeError):
        return
    finally:
        pipe.close()


class BlockedWorker(RuntimeError):
    pass


class ExactPathPool:
    def __init__(self, workers, stage, timeout_seconds=None):
        self.stage = stage
        self.timeout = timeout_seconds or (5 if stage == "existence" else 20)
        self.limit = workers
        self.entries = []
        context = multiprocessing.get_context("fork")
        for position in range(workers):
            parent, child = context.Pipe()
            process = context.Process(target=worker_loop, args=(child, stage, self.timeout), daemon=True)
            process.start()
            child.close()
            self.entries.append(dict(process=process, pipe=parent, row=None, started=None, position=position))

    def send(self, rows):
        count = 0
        for entry in self.entries:
            if entry["row"] is not None or entry["position"] >= self.limit:
                continue
            try:
                row = next(rows)
            except StopIteration:
                break
            entry["pipe"].send(row)
            entry.update(row=row, started=time.monotonic())
            count += 1
        return count

    def poll(self):
        results = []
        for entry in self.entries:
            if entry["row"] is None:
                continue
            if entry["pipe"].poll():
                try:
                    result = entry["pipe"].recv()
                except EOFError as error:
                    raise BlockedWorker("Worker exited without a result; preserve checkpoint") from error
                results.append(result)
                entry.update(row=None, started=None)
            elif not entry["process"].is_alive():
                raise BlockedWorker("Worker died before completing exact-path operation")
            elif time.monotonic()-entry["started"] > 4*self.timeout+15:
                raise BlockedWorker("NFS operation remains blocked beyond retry budget; pause instead of misclassifying an image")
        return results

    def busy(self):
        return any(entry["row"] is not None for entry in self.entries)

    def close(self):
        for entry in self.entries:
            process = entry["process"]
            if process.is_alive():
                if entry["row"] is None:
                    try:
                        entry["pipe"].send(None)
                    except (BrokenPipeError, OSError):
                        pass
                else:
                    process.terminate()
        deadline = time.monotonic()+3
        for entry in self.entries:
            entry["process"].join(max(0, deadline-time.monotonic()))
            if entry["process"].is_alive():
                entry["process"].kill()
            entry["pipe"].close()


def family_counts(connection):
    return {family: dict(required=required, existence_checked=0, exists=0, missing=0,
        existence_io_errors=0, decode_checked=0, decoded=0, corrupt=0,
        decode_missing=0, decode_io_errors=0, io_retry_count=0)
        for family, required in metadata(connection)["expected"].items()} | {
        family: dict(required=count, existence_checked=existing_checked, exists=exists,
            missing=missing, existence_io_errors=existence_io_errors,
            decode_checked=decode_checked, decoded=decoded, corrupt=corrupt,
            decode_missing=decode_missing, decode_io_errors=decode_io_errors, io_retry_count=retries)
        for family, count, existing_checked, exists, missing, existence_io_errors,
            decode_checked, decoded, corrupt, decode_missing, decode_io_errors, retries
        in connection.execute("""SELECT family,COUNT(*),SUM(existence!='pending'),SUM(existence='pass'),
            SUM(existence='missing'),SUM(existence='io_error'),SUM(decode!='pending'),SUM(decode='pass'),
            SUM(decode='corrupt'),SUM(decode='missing'),SUM(decode='io_error'),
            SUM(existence_retries+decode_retries) FROM images GROUP BY family""")}


def record_result(connection, result, counts):
    stage = result["stage"]
    if stage not in {"existence", "decode"}:
        raise ValueError("Invalid audit stage")
    previous = "pending"
    if result["status"] != "pass":
        previous = connection.execute(f"SELECT {stage} FROM images WHERE ordinal=?", (result["ordinal"],)).fetchone()[0]
    cursor = connection.execute(f"UPDATE images SET {stage}=?,{stage}_retries={stage}_retries+?,{stage}_error=? WHERE ordinal=? AND ({stage}='pending' OR ({stage}='pass' AND ?!='pass'))",
        (result["status"], result["io_retries"], result["error"], result["ordinal"], result["status"]))
    if not cursor.rowcount:
        return False
    family = counts[result["family"]]
    family["io_retry_count"] += result["io_retries"]
    if previous == "pass":
        family["exists" if stage == "existence" else "decoded"] -= 1
    else:
        family["existence_checked" if stage == "existence" else "decode_checked"] += 1
    status_key = ({"pass": "exists", "missing": "missing", "io_error": "existence_io_errors"}
        if stage == "existence" else {"pass": "decoded", "missing": "decode_missing",
                                     "io_error": "decode_io_errors", "corrupt": "corrupt"})
    family[status_key[result["status"]]] += 1
    return True


def snapshot(identity, counts, phase, started, benchmarks, workers=None, error=None):
    job = load(LOCAL / "job.json", {})
    if job.get("started_utc"):
        started = min(started, datetime.datetime.fromisoformat(job["started_utc"]).timestamp())
    result = dict(checked_utc=now(), status=phase, training_index_sha256=identity["training_index_sha256"],
        manifest_sha256=identity["manifest_sha256"], required_paths_total=identity["required_paths_total"],
        families=counts, existence_checked=sum(row["existence_checked"] for row in counts.values()),
        exists_count=sum(row["exists"] for row in counts.values()), missing_count=sum(row["missing"] for row in counts.values()),
        decode_checked=sum(row["decode_checked"] for row in counts.values()), decoded_count=sum(row["decoded"] for row in counts.values()),
        corrupt_count=sum(row["corrupt"] for row in counts.values()),
        io_retry_count=sum(row["io_retry_count"] for row in counts.values()),
        io_error_count=sum(row["existence_io_errors"]+row["decode_io_errors"] for row in counts.values()),
        decode_missing_count=sum(row["decode_missing"] for row in counts.values()),
        elapsed_seconds=time.time()-started, active_workers=workers, benchmarks=benchmarks,
        recursive_filesystem_scan=False, image_transform="NONE", rgb_conversion_in_memory_only=True,
        smoke_started=False, formal_training_started=False, error=error, passed=False)
    result["manifest_source"] = str(MANIFEST)
    result["checkpoint"] = str(LOCAL / "checkpoint.sqlite3")
    result["benchmark_io_retry_count"] = sum(row["io_retry_count"] for value in benchmarks.values() for row in value["runs"])
    result["training_index_missing"] = result["missing_count"]+result["decode_missing_count"]
    return result


def write_progress(result, local_folder=LOCAL):
    atomic_json(local_folder / "progress.json", result)
    atomic_json(FINAL, result)
    write_report(result)
    if "DECODE" in result["status"] or result["status"] == "DATA_AUDIT_PASS":
        atomic_json(EVIDENCE / "training-image-decode-progress.json", dict(checked=result["decode_checked"],
            required=result["required_paths_total"], decode_failures=result["corrupt_count"],
            io_errors=result["io_error_count"], families=result["families"], checked_utc=result["checked_utc"]))
    print(result["status"], "existence", f"{result['existence_checked']}/{result['required_paths_total']}",
        "decoded", f"{result['decoded_count']}/{result['required_paths_total']}",
        "corrupt", result["corrupt_count"], "IO retries", result["io_retry_count"], flush=True)


def iter_pending(connection, stage):
    after = 0
    while True:
        rows = connection.execute(f"SELECT ordinal,family,absolute_path FROM images WHERE ordinal>? AND {stage}='pending' ORDER BY ordinal LIMIT ?",
            (after, CHUNK_SIZE)).fetchall()
        if not rows:
            return
        for ordinal, family, absolute_path in rows:
            yield dict(ordinal=ordinal, family=family, absolute_path=absolute_path)
        after = rows[-1][0]


def benchmark_rows(connection, stage):
    groups = [connection.execute(f"SELECT ordinal,family,absolute_path FROM images WHERE family=? AND {stage}='pending' ORDER BY ordinal LIMIT 5000", (family,)).fetchall()
              for family in metadata(connection)["expected"]]
    for position in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if position < len(group):
                ordinal, family, path = group[position]
                yield dict(ordinal=ordinal, family=family, absolute_path=path)


def choose_workers(rows):
    stable = [row for row in rows if row["stable"] and row["completed"] and row["duration_seconds"] >= row["requested_seconds"]*.95]
    if not stable:
        raise RuntimeError("No stable completed concurrency benchmark; preserve audit checkpoint")
    peak = max(row["paths_per_second"] for row in stable)
    return min(row["workers"] for row in stable if row["paths_per_second"] >= peak*.95)


def run_benchmark(connection, stage, workers, counts, identity, benchmarks, seconds=BENCHMARK_SECONDS):
    selected = list(benchmark_rows(connection, stage))
    if not selected:
        return None
    pool = ExactPathPool(workers, stage)
    started = time.monotonic()
    deadline = started+seconds
    completed = 0
    retries = 0
    io_errors = 0
    timeouts = 0
    failures = 0
    unique_added = 0
    repeated_after_complete = False
    latencies = []
    rows = iter(selected)
    last_publish = started
    try:
        while time.monotonic() < deadline or pool.busy():
            results = pool.poll()
            for result in results:
                unique_added += int(record_result(connection, result, counts))
                completed += 1
                retries += result["io_retries"]
                io_errors += result["io_errors"]
                timeouts += result["timeouts"]
                failures += int(result["status"] != "pass")
                latencies.append(result["latency_seconds"])
            if time.monotonic() < deadline:
                if not pool.send(rows) and not pool.busy():
                    pending = list(benchmark_rows(connection, stage))
                    repeated_after_complete = repeated_after_complete or not pending
                    rows = iter(pending or selected)
                    pool.send(rows)
            if time.monotonic()-last_publish >= 5:
                connection.commit()
                progress = snapshot(identity, counts, stage.upper()+"_BENCHMARK", identity["started_timestamp"], benchmarks, workers)
                progress["benchmark_current"] = dict(workers=workers, elapsed_seconds=time.monotonic()-started,
                    requested_seconds=seconds, completed=completed, paths_per_second=completed/max(.001,time.monotonic()-started))
                write_progress(progress)
                last_publish = time.monotonic()
            if completed >= 100 and (timeouts > 3 or io_errors/max(1,completed) > .02):
                break
            time.sleep(.005)
    finally:
        pool.close()
        connection.commit()
    duration = time.monotonic()-started
    ordered = sorted(latencies)
    result = dict(stage=stage, workers=workers, completed=completed, duration_seconds=duration,
        requested_seconds=seconds, paths_per_second=completed/max(duration,.001), io_retry_count=retries,
        io_errors=io_errors, timeouts=timeouts, image_failures=failures,
        mean_nfs_operation_latency_seconds=sum(latencies)/max(1,len(latencies)),
        p95_nfs_operation_latency_seconds=ordered[min(len(ordered)-1,int(len(ordered)*.95))] if ordered else None,
        latency_scope="Exact-path stat" if stage == "existence" else "Exact-path stat/open/full decode and in-memory RGB",
        stable=timeouts == 0 and io_errors/max(1,completed) <= .01, sampled_families=list(counts),
        unique_required_paths_added=unique_added, repeats_only_after_all_required_paths_checked=repeated_after_complete)
    print("benchmark", json.dumps(result), flush=True)
    return result


def full_stage(connection, stage, workers, counts, identity, benchmarks):
    pool = ExactPathPool(workers, stage)
    rows = iter(iter_pending(connection, stage))
    exhausted = False
    last_publish = time.monotonic()
    window_started = last_publish
    recent = []
    completed = 0
    slow_windows = 0
    baseline_rate = next(row["paths_per_second"] for row in benchmarks[stage]["runs"] if row["workers"] == workers)
    try:
        while not exhausted or pool.busy():
            results = pool.poll()
            for result in results:
                record_result(connection, result, counts)
                completed += 1
                recent.append(result)
            if not exhausted:
                sent = pool.send(rows)
                if not sent and not pool.busy():
                    exhausted = True
            if time.monotonic()-window_started >= 15 and len(recent) >= 50:
                io_count = sum(row["io_errors"] for row in recent)
                rate = len(recent)/(time.monotonic()-window_started)
                slow_windows = slow_windows+1 if rate < baseline_rate*.7 else 0
                if io_count/len(recent) > .01 or any(row["timeouts"] for row in recent) or slow_windows >= 2:
                    previous_limit = pool.limit
                    pool.limit = max(1, pool.limit//2)
                    atomic_json(LOCAL / "concurrency-backoff.json", dict(checked_utc=now(), stage=stage,
                        previous_workers=previous_limit, new_workers=pool.limit,
                        reason="NFS IO retries/timeouts or sustained throughput drop", results=len(recent), paths_per_second=rate))
                    baseline_rate = min(baseline_rate, rate)
                    slow_windows = 0
                    print("NFS protection: reduce concurrency to", pool.limit, flush=True)
                recent.clear()
                window_started = time.monotonic()
            if time.monotonic()-last_publish >= 5 or completed >= CHUNK_SIZE:
                connection.commit()
                write_progress(snapshot(identity, counts, stage.upper()+"_AUDIT", identity["started_timestamp"], benchmarks, pool.limit))
                completed = 0
                last_publish = time.monotonic()
            time.sleep(.005)
    finally:
        pool.close()
        connection.commit()
    write_progress(snapshot(identity, counts, stage.upper()+"_AUDIT", identity["started_timestamp"], benchmarks, pool.limit))


def write_report(result):
    lines = ["# Final manifest-driven image audit", "", f"Checked UTC: {result['checked_utc']}.",
        f"Status: **{result['status']}**.", "", "Frozen ShareGPT4V required paths only; no filesystem directory enumeration, recursive glob, find or os.walk.",
        f"Index SHA256: `{result['training_index_sha256']}`.", f"Manifest SHA256: `{result['manifest_sha256']}`.",
        "Manifest: `required_training_images.jsonl`; restart checkpoint: local-SSD SQLite, bound to index/manifest/root/counts.",
        "Timeout/temporary IO errors receive at most3 short-backoff retries, not automatic corruption classification.",
        "PIL full load and RGB conversion are in memory only; encoded image bytes are never changed.", "",
        "| Family | Required | Existence checked | Exists | Missing | Decode checked | Decoded | Corrupt | IO errors | IO retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family, row in result["families"].items():
        lines.append(f"| {family} | {row['required']} | {row['existence_checked']} | {row['exists']} | {row['missing']} | {row['decode_checked']} | {row['decoded']} | {row['corrupt']} | {row['existence_io_errors']+row['decode_io_errors']} | {row['io_retry_count']} |")
    lines.extend(["", f"Required paths total: {result['required_paths_total']}; existence checked: {result['existence_checked']}; missing: {result['missing_count']}.",
        f"Decoded: {result['decoded_count']}; corrupt: {result['corrupt_count']}; persistent IO errors: {result['io_error_count']}; IO retries: {result['io_retry_count']}.",
        f"Elapsed seconds: {result['elapsed_seconds']:.1f}.", "", "## Concurrency benchmarks", "",
        "| Stage | Workers | Seconds | Paths/s | Mean latency ms | P95 latency ms | IO errors | Timeouts | Stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for stage, value in result["benchmarks"].items():
        for row in value["runs"]:
            mean = row["mean_nfs_operation_latency_seconds"]*1000
            p95 = (row["p95_nfs_operation_latency_seconds"] or 0)*1000
            lines.append(f"| {stage} | {row['workers']} | {row['duration_seconds']:.1f} | {row['paths_per_second']:.2f} | {mean:.2f} | {p95:.2f} | {row['io_errors']} | {row['timeouts']} | {row['stable']} |")
        lines.append(f"Selected {stage} workers: {value.get('selected_workers', 'PENDING')}; choose lowest stable concurrency within5% of peak throughput.")
    lines.extend(["", "Full decode starts only after exact-path existence reaches100% with zero missing/persistent IO errors.",
        "DATA_AUDIT_PASS requires all three exact frozen family counts to exist and decode, zero missing/corrupt/IO failures.",
        "No smoke,500-step or4868-step trainer is launched by this program.",
        "If DATA_AUDIT_PASS, the next permitted training action is only the separately authorized four-A100 five-update smoke with common step0 and H4868."])
    if result.get("error"):
        lines.extend(["", "Audit error: "+result["error"]])
    REPORT.write_text("\n".join(lines)+"\n")


def write_bad_images(connection):
    with (LOCAL / "bad-images.jsonl").open("w") as output:
        for ordinal,family,path,existence,decode,exist_error,decode_error in connection.execute("SELECT ordinal,family,absolute_path,existence,decode,existence_error,decode_error FROM images WHERE existence NOT IN ('pending','pass') OR decode NOT IN ('pending','pass')"):
            output.write(json.dumps(dict(ordinal=ordinal,family=family,absolute_path=path,existence=existence,
                decode=decode,existence_error=exist_error,decode_error=decode_error))+"\n")


def publish_pass(result):
    prior = load(EVIDENCE / "training-audit.json", {})
    if not (prior.get("annotation") and prior.get("training_records") == TOTAL and
            prior.get("skip") == 1000 and prior.get("index_sha256") == RECORDS_SHA):
        raise ValueError("Original annotation/index/skip gate must already be verified")
    previous = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
    previous.update(checked_utc=result["checked_utc"], training_records=TOTAL, filtered_records=TOTAL,
        training_index_sha256=result["training_index_sha256"], recovered_training_paths=TOTAL,
        missing_training_paths=0, training_index_missing=0, passed=True,
        update_scope="Frozen-manifest exact-path regular-file and full PIL/RGB decode audit; no directory scan")
    previous["families"] = {family: dict(required_paths=row["required"], recovered_paths=row["exists"],
        missing_paths=row["missing"], decoded_paths=row["decoded"], duplicate_training_index_paths=0,
        audit_evidence=str(FINAL)) for family,row in result["families"].items()}
    previous["decode_audit"] = dict(executed=True, passed=True, checked_images=TOTAL,
        failures=[], io_retry_count=result["io_retry_count"], audit_evidence=str(FINAL))
    previous["required_sam_image_ids"] = dict(count=EXPECTED["sam"], recovered_count=EXPECTED["sam"],
        missing_count=0, source_path_list=str(MANIFEST))
    previous["shard14_forensics"] = dict(status="TRAINING_IMAGES_FULLY_RECOVERED", required_images=11168,
        missing_required_images=0, container_status="QUARANTINED_ARCHIVE_INVALID")
    previous["sam_original_image_files_present"] = EXPECTED["sam"]
    previous["family_checked_utc"] = {family:result["checked_utc"] for family in EXPECTED}
    atomic_json(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", previous)
    atomic_json(EVIDENCE / "training-image-completeness-audit.json", previous)
    prior.update(checked_utc=result["checked_utc"], missing_images=0, missing_examples=[], passed=True,
        image_audit_evidence=str(FINAL), image_manifest_sha256=result["manifest_sha256"])
    atomic_json(EVIDENCE / "training-audit.json", prior)


def run(seconds=BENCHMARK_SECONDS):
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / "audit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    connection = open_database(LOCAL / "checkpoint.sqlite3")
    identity = None
    benchmarks = {}
    counts = {}
    stopping = False
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_handler(signum, frame):
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt("Audit stop requested; preserve manifest and checkpoint")

    signal.signal(signal.SIGTERM, stop_handler)
    try:
        sources = validate_recovery_sources()
        identity = prepare_manifest(connection)
        identity.update(sources)
        for key,value in sources.items():
            save_metadata(connection,key,value)
        connection.commit()
        counts = family_counts(connection)
        benchmarks = metadata(connection).get("benchmarks", {})
        write_progress(snapshot(identity, counts, "MANIFEST_READY", identity["started_timestamp"], benchmarks))
        for stage in ["existence", "decode"]:
            if stage == "decode" and any(row["exists"] != row["required"] for row in counts.values()):
                result = snapshot(identity, counts, "EXISTENCE_AUDIT_FAIL", identity["started_timestamp"], benchmarks)
                write_progress(result)
                write_report(result)
                write_bad_images(connection)
                return result
            remaining = connection.execute(f"SELECT COUNT(*) FROM images WHERE {stage}='pending'").fetchone()[0]
            if not remaining:
                continue
            benchmark = benchmarks.setdefault(stage, dict(runs=[]))
            done = {row["workers"] for row in benchmark["runs"]}
            for workers in [8, 16, 32]:
                if workers not in done:
                    if benchmark["runs"] and not benchmark["runs"][-1]["stable"]:
                        break
                    measured = run_benchmark(connection, stage, workers, counts, identity, benchmarks, seconds)
                    if measured is None:
                        break
                    benchmark["runs"].append(measured)
                    save_metadata(connection, "benchmarks", benchmarks)
                    connection.commit()
                    write_report(snapshot(identity, counts, stage.upper()+"_BENCHMARK", identity["started_timestamp"], benchmarks))
            benchmark["selected_workers"] = choose_workers(benchmark["runs"])
            save_metadata(connection, "benchmarks", benchmarks)
            connection.commit()
            full_stage(connection, stage, benchmark["selected_workers"], counts, identity, benchmarks)
        passed = all(row["exists"] == row["required"] == row["decoded"] and not
            (row["missing"] or row["corrupt"] or row["existence_io_errors"] or row["decode_io_errors"] or row["decode_missing"])
            for row in counts.values())
        result = snapshot(identity, counts, "DATA_AUDIT_PASS" if passed else "DATA_AUDIT_FAIL", identity["started_timestamp"], benchmarks)
        result["passed"] = passed
        write_progress(result)
        write_report(result)
        if passed:
            publish_pass(result)
        write_bad_images(connection)
        return result
    except BaseException as error:
        connection.commit()
        if identity:
            result = snapshot(identity, counts, "AUDIT_PAUSED" if stopping else "AUDIT_ERROR", identity["started_timestamp"], benchmarks,
                error=f"{type(error).__name__}: {error}")
            write_progress(result)
            write_report(result)
        raise
    finally:
        connection.close()
        lock.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-seconds", type=int, default=BENCHMARK_SECONDS)
    arguments = parser.parse_args()
    if arguments.benchmark_seconds < BENCHMARK_SECONDS:
        parser.error("Production benchmarks must run at least120 seconds per worker setting")
    run(arguments.benchmark_seconds)
