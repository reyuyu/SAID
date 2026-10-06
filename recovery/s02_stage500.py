"""CPU-only frozen first500 manifest and resource-guarded local staging.

Run prepare, then stage. Neither command starts training or copies full coverage.
Large ledgers/manifests remain local; the reviewed summary is safe to publish.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time

from recovery import local_ssd_stage as old
from recovery.resource_stall_v2 import system_snapshot, GIB, dump

ROOT = old.ROOT
LOCAL = Path('/root/said_s02_stage500')
RAW = ROOT / 'recovery/evidence/s02-stage500-local'
RESULT = ROOT / 'recovery/S02_LOCAL_STAGE500.json'
CONFIG = ROOT / 'recovery/configs/summary02.json'
CHUNK = 1024 * 1024


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb', buffering=0) as handle:
        while data := handle.read(CHUNK):
            digest.update(data)
        advise(handle.fileno())
    return digest.hexdigest()


def advise(fd):
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def disk(required=0):
    LOCAL.mkdir(parents=True, exist_ok=True)
    mount = subprocess.check_output(['findmnt', '-T', str(LOCAL), '-n', '-o', 'FSTYPE,SOURCE,TARGET'], text=True).strip()
    if any(kind in mount.lower() for kind in ('nfs', 'cifs', 'tmpfs')):
        raise RuntimeError('Local stage must be block-backed: ' + mount)
    free = shutil.disk_usage(LOCAL).free
    if free < required + 64 * GIB:
        raise RuntimeError('Insufficient SSD free space with 64GiB reserve')
    return dict(mount=mount, free_bytes=free, reserve_bytes=64 * GIB,
                local_root=str(LOCAL / 'ShareGPT4V'), lifecycle='pod-local; regenerate after pod rebuild')


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def atomic_copy(source, destination, expected_size, stop=None):
    """Bounded 1MiB stream, fsync/close/rename; hash every source and reread target."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.parent.resolve().is_relative_to(LOCAL.resolve()):
        raise RuntimeError('Unsafe local mirror destination')
    if source.stat().st_size != expected_size:
        raise RuntimeError('Source size changed: ' + str(source))
    if destination.exists():
        if destination.stat().st_size != expected_size or sha(source) != sha(destination):
            raise RuntimeError('Existing local file mismatch: ' + str(destination))
        return dict(size_bytes=expected_size, source_sha256=sha(destination), copied=False)
    temporary = destination.with_name(destination.name + '.incomplete')
    checksum, count = hashlib.sha256(), 0
    with source.open('rb', buffering=0) as incoming:
        before = os.fstat(incoming.fileno())
        with temporary.open('wb', buffering=0) as outgoing:
            while data := incoming.read(CHUNK):
                if stop is not None and stop.is_set():
                    raise InterruptedError('Copy paused; progress preserved')
                outgoing.write(data)
                checksum.update(data)
                count += len(data)
            os.fsync(outgoing.fileno())
            advise(outgoing.fileno())
        after = os.fstat(incoming.fileno())
        advise(incoming.fileno())  # own completed read only; no global cache operation
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or count != expected_size:
        raise RuntimeError('Source changed during copy')
    actual = checksum.hexdigest()
    if sha(temporary) != actual:
        raise RuntimeError('Destination SHA256 mismatch')
    temporary.replace(destination)
    return dict(size_bytes=count, source_sha256=actual, destination_sha256=actual, copied=True)


def prepare():
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    cfg = json.loads(CONFIG.read_text())
    assert all(cfg[k] == v for k, v in dict(seed=0, shuffle_seed=0, sampling_seed=0,
        world_size=4, batch_size=256, epochs=4, accumulation=1,
        sampling_mode='summary_random_detail', view_weights=[1.4, .2, 1.4]).items())
    disk()
    RAW.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for name in ('records.jsonl', 'offsets.npy', 'metadata.json'):
        source = old.ORIGINAL_INDEX / name
        atomic_copy(source, LOCAL / 'data_index' / name, source.stat().st_size)
    meta = json.loads((LOCAL / 'data_index/metadata.json').read_text())
    assert meta['training_records'] == 1245901 and meta['skip'] == 1000
    assert sha(LOCAL / 'data_index/records.jsonl') == meta['records_sha256']
    history = old.historical_rows()
    assert [r['step'] for r in history] == list(range(1, 501))
    before = old.rng_digest()
    streams = {}
    for rank in range(4):
        sampler = DistributedSampler(range(1245901), num_replicas=4, rank=rank, shuffle=True, seed=0, drop_last=False)
        sampler.set_epoch(0)
        streams[rank] = list(itertools.islice(iter(sampler), 128000))
        actual = [s for r in history for h in r['rank_health'] if h['rank'] == rank for s in h['sampling']['sample_ids']]
        assert [i + 1000 for i in streams[rank]] == actual, f'Sample order mismatch on rank {rank}'
    sizes = old.sizes_from_installation()
    offsets = np.load(LOCAL / 'data_index/offsets.npy', mmap_mode='r')
    unique, counts, total = {}, Counter(), 0
    order_digest = hashlib.sha256()
    with (LOCAL / 'data_index/records.jsonl').open('rb') as handle, (RAW / 'ordered-records.jsonl').open('w') as output:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            matched_text = 0
            for step in range(1, 501):
                for rank in range(4):
                    batch = streams[rank][(step - 1) * 256:step * 256]
                    views = []
                    for position, index in enumerate(batch):
                        record = json.loads(data[offsets[index]:offsets[index + 1]])
                        relative = old.safe_relative(record['image'])
                        size = sizes[relative]
                        row = dict(step=step, rank=rank, batch_position=position, sample_id=index + 1000,
                                   record_index=index, relative_path=relative, size_bytes=size)
                        line = json.dumps(row, separators=(',', ':')) + '\n'
                        output.write(line)
                        order_digest.update(line.encode())
                        if relative not in unique:
                            unique[relative] = row
                            counts[relative.split('/')[0]] += 1
                            total += size
                        if step <= 5:
                            views.append(sampled_text_views(record['caption'], 'summary_random_detail', 0, 0, index + 1000))
                    if step <= 5:
                        stream = dict(sample_ids=[i + 1000 for i in batch], views=[v['views'] for v in views],
                            tokens=[[v[k].tolist() for v in views] for k in ('tokens_f', 'tokens_o', 'tokens_e')])
                        actual = hashlib.sha256(json.dumps(stream, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                        expected = next(h['stream_sha256'] for h in history[step - 1]['rank_health'] if h['rank'] == rank)
                        assert actual == expected, 'Frozen F/S/D strings and token stream mismatch'
                        matched_text += 256
    with (RAW / 'unique-images.jsonl').open('w') as handle:
        for name in sorted(unique):
            handle.write(json.dumps(unique[name], separators=(',', ':')) + '\n')
    assert old.rng_digest() == before, 'Replay advanced global RNG'
    proof = dict(status='MANIFEST_READY', seed=0, global_batch=1024, horizon=4868,
        sampling_mode='summary_random_detail', view_weights=[1.4, .2, 1.4], records_count=512000,
        record_order='step, rank, rank-local batch position', unique_images_count=len(unique),
        family_counts=dict(counts), total_bytes=total, staged_GB=total / 1e9,
        historical_sample_ids_matched=512000, historical_FSD_string_token_samples_matched=matched_text,
        offline_global_RNG_unchanged=True, ordered_stream_sha256=order_digest.hexdigest(),
        manifest_sha256=sha(RAW / 'unique-images.jsonl'), historical_source_sha256=sha(old.HISTORY),
        index_records_sha256=meta['records_sha256'], config_sha256=sha(CONFIG),
        local_storage=disk(total), generated_at=utc())
    dump(RAW / 'manifest-proof.json', proof)
    dump(RESULT, proof)
    print(json.dumps(proof), flush=True)


def kernel_tail():
    text = subprocess.check_output(['dmesg', '--raw'], text=True, timeout=10)
    output = []
    for line in text.splitlines():
        try:
            timestamp = float(line.split('[', 1)[1].split(']', 1)[0])
        except (ValueError, IndexError):
            continue
        output.append((timestamp, line.lower()))
    return output


class Guard:
    def __init__(self, baseline, kernel_start, stop):
        self.baseline, self.kernel_start, self.stop = baseline, kernel_start, stop
        self.high_since = None
        self.reason = None
        self.peak = baseline['memory_current']

    def check(self, snapshot, now, kernel=()):
        self.peak = max(self.peak, snapshot['memory_current'])
        if snapshot['memory_current'] > 470 * GIB:
            if self.high_since is None:
                self.high_since = now
            if now - self.high_since >= 300:
                self.reason = 'memory.current >470GiB continuously for 5 minutes'
        else:
            self.high_since = None
        events = snapshot['memory_events']
        for key in ('oom', 'oom_kill'):
            if events.get(key, 0) > self.baseline['memory_events'].get(key, 0):
                self.reason = 'new cgroup ' + key
        if events.get('under_oom', 0):
            self.reason = 'cgroup under_oom'
        for field, increase, floor in (('memory_PSI', 10, 20), ('io_PSI', 20, 50)):
            for kind in ('some', 'full'):
                current = snapshot[field].get(kind, {}).get('avg10', 0)
                base = self.baseline[field].get(kind, {}).get('avg10', 0)
                if current > max(floor, base + increase):
                    self.reason = field + '/' + kind + ' avg10 significantly worsened'
        for stamp, line in kernel:
            if stamp > self.kernel_start and ('not responding' in line and 'nfs' in line or
                any(term in line for term in ('out of memory', 'invoked oom-killer', 'kernel panic', 'hung task', 'blocked for more than'))):
                self.reason = 'new kernel NFS/OOM/panic/hung-task event (raw text retained only locally)'
        if self.reason:
            self.stop.set()
        return self.reason


def stage():
    proof = json.loads((RAW / 'manifest-proof.json').read_text())
    assert sha(RAW / 'unique-images.jsonl') == proof['manifest_sha256']
    disk(proof['total_bytes'])
    baseline = system_snapshot()
    kernel = kernel_tail()
    kernel_start = max((stamp for stamp, _ in kernel), default=0)
    stop = threading.Event()
    guard = Guard(baseline, kernel_start, stop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    state = dict(status='COPYING', started_at=utc(), checked=0, copied=0, bytes=0, workers=2,
                 pid=os.getpid(), anomalies=[], scope='ONLY frozen step1-500 unique manifest', baseline=baseline)
    lock = threading.Lock()
    monitor_done = threading.Event()
    telemetry = RAW / 'resource-telemetry.jsonl'

    def monitor():
        last_t, last_bytes, last_files = started, 0, 0
        last_emit = 0
        while not monitor_done.is_set():
            now = time.monotonic()
            try:
                snapshot = system_snapshot()
                reason = guard.check(snapshot, now, kernel_tail())
                with lock:
                    if reason and reason not in state['anomalies']:
                        state['anomalies'].append(reason)
                    if now - last_emit >= 300 or reason or last_emit == 0:
                        elapsed = max(now - last_t, .001)
                        event = dict(at=utc(), elapsed_s=now - started, workers=state['workers'],
                            checked=state['checked'], copied=state['copied'], bytes=state['bytes'],
                            copy_MiB_s=(state['bytes'] - last_bytes) / elapsed / 2**20,
                            files_s=(state['checked'] - last_files) / elapsed, snapshot=snapshot,
                            peak_memory_current=guard.peak, pause_reason=reason)
                        with telemetry.open('a') as handle:
                            handle.write(json.dumps(event) + '\n')
                        dump(RAW / 'progress.json', event)
                        print(json.dumps({k: event[k] for k in ('at', 'elapsed_s', 'workers', 'checked', 'copy_MiB_s', 'files_s', 'peak_memory_current', 'pause_reason')}), flush=True)
                        last_t, last_bytes, last_files, last_emit = now, state['bytes'], state['checked'], now
            except Exception as error:
                guard.reason = 'Resource monitor failed: ' + type(error).__name__
                state['anomalies'].append(guard.reason)
                stop.set()
            monitor_done.wait(10)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    ledger = RAW / 'hash-ledger.jsonl'
    error = None
    try:
        with ThreadPoolExecutor(max_workers=4) as pool, ledger.open('a') as output:
            iterator, pending = iter(rows(RAW / 'unique-images.jsonl')), {}
            exhausted = False
            while (not exhausted or pending) and not stop.is_set():
                now = time.monotonic()
                if state['workers'] == 2 and now - started >= 1800 and not state['anomalies']:
                    sample = system_snapshot()
                    if sample['memory_current'] < 440 * GIB and sample['memory_current'] - baseline['memory_current'] < 32 * GIB:
                        state['workers'] = 4
                        state['promoted_at'] = utc()
                while not exhausted and len(pending) < state['workers'] and not stop.is_set():
                    row = next(iterator, None)
                    if row is None:
                        exhausted = True
                        break
                    relative = old.safe_relative(row['relative_path'])
                    future = pool.submit(atomic_copy, old.SOURCE / relative, LOCAL / 'ShareGPT4V' / relative, row['size_bytes'], stop)
                    pending[future] = row
                if not pending:
                    break
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    result = future.result()
                    output.write(json.dumps(dict(relative_path=row['relative_path'], **result)) + '\n')
                    with lock:
                        state['checked'] += 1
                        state['copied'] += int(result['copied'])
                        state['bytes'] += result['size_bytes']
                    if state['checked'] % 1000 == 0:
                        output.flush()
                        os.fsync(output.fileno())
                        disk()
                if stop.is_set():
                    for future in pending:
                        future.cancel()
            output.flush()
            os.fsync(output.fileno())
    except Exception as caught:
        error = type(caught).__name__ + ': ' + str(caught)
        stop.set()
    finally:
        monitor_done.set()
        thread.join(timeout=15)
    state.update(elapsed_s=time.monotonic() - started, peak_cgroup_memory=guard.peak,
                 copy_workers_exited=True, finished_at=utc(), error=error, pause_reason=guard.reason)
    complete = not stop.is_set() and error is None and state['checked'] == proof['unique_images_count']
    state['status'] = 'COPY_HASH_PASS' if complete else 'COPY_PAUSED'
    dump(RAW / 'copy-result.json', state)
    dump(RESULT, dict(proof, copy=state, status=state['status'], github_sync_complete=False))
    if not complete:
        raise RuntimeError('Copy paused: ' + str(guard.reason or error or 'operator stop'))
    verify(proof, state, guard)


def verify(proof, copy, guard):
    import torch
    from PIL import Image
    from train.said_cvssl_data import reference_view_a_transform
    torch.set_num_threads(1)
    rng = random.Random(0)
    reservoir = []
    total = 0
    # Stream all local sizes; reservoir sample metadata only, not image payloads.
    for index, row in enumerate(rows(RAW / 'unique-images.jsonl')):
        relative = old.safe_relative(row['relative_path'])
        source, destination = old.SOURCE / relative, LOCAL / 'ShareGPT4V' / relative
        if destination.is_symlink() or not destination.is_file() or destination.stat().st_size != row['size_bytes']:
            raise RuntimeError('Missing/size-mismatched local required image: ' + relative)
        if source.stat().st_size != row['size_bytes']:
            raise RuntimeError('Source size changed at verification: ' + relative)
        total += row['size_bytes']
        if index < 1000:
            reservoir.append(row)
        else:
            slot = rng.randrange(index + 1)
            if slot < 1000:
                reservoir[slot] = row
        if index % 10000 == 0:
            if guard.check(system_snapshot(), time.monotonic(), kernel_tail()):
                raise RuntimeError(guard.reason)
    assert total == proof['total_bytes']
    transform = reference_view_a_transform()
    checks = []
    for row in reservoir:
        relative = row['relative_path']
        source, destination = old.SOURCE / relative, LOCAL / 'ShareGPT4V' / relative
        source_sha, destination_sha = sha(source), sha(destination)
        assert source_sha == destination_sha
        with Image.open(source) as image:
            first = image.convert('RGB')
            pixels, shape, expected = hashlib.sha256(first.tobytes()).hexdigest(), first.size, transform(first)
        with Image.open(destination) as image:
            second = image.convert('RGB')
            actual = transform(second)
            assert shape == second.size and pixels == hashlib.sha256(second.tobytes()).hexdigest()
        assert torch.equal(expected, actual), 'Preprocess tensor differs'
        checks.append(dict(sample_id=row['sample_id'], relative_path=relative, sha256=source_sha,
                           tensor_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()))
        if guard.check(system_snapshot(), time.monotonic(), kernel_tail()):
            raise RuntimeError(guard.reason)
    dump(RAW / 'equivalence-1000.json', dict(passed=True, seed=0, samples=checks))
    print('All sizes/all copy SHA256 and 1000 RGB/preprocess checks passed; observing 5min stability', flush=True)
    stable, start = [], time.monotonic()
    while True:
        snapshot = system_snapshot()
        if guard.check(snapshot, time.monotonic(), kernel_tail()):
            raise RuntimeError(guard.reason)
        if snapshot['memory_current'] > 470 * GIB:
            raise RuntimeError('Resources not admitted for readiness')
        stable.append(snapshot)
        if time.monotonic() - start >= 300:
            break
        time.sleep(10)
    if max(s['memory_current'] for s in stable) - min(s['memory_current'] for s in stable) > 5 * GIB:
        raise RuntimeError('Memory not stable within 5GiB during 5min admission window')
    result = dict(proof, status='LOCAL_STAGE500_READY', staging_ready=True, github_sync_complete=False,
        phase_final_complete=False, training_started=False, copy=copy,
        peak_cgroup_memory=max(guard.peak, *(s['memory_current'] for s in stable)),
        resource_anomalies=copy['anomalies'], verification=dict(all_size_checked=proof['unique_images_count'],
            all_sha256_matched=copy['checked'], random_sha256_samples=1000, RGB_exact_samples=1000,
            preprocess_tensor_exact_samples=1000, sampling_seed=0, passed=True),
        stabilization=dict(duration_s=time.monotonic() - start, snapshots=stable), ready_at=utc(),
        pod_identity=Path('/proc/self/cgroup').read_text().splitlines()[0].split(':')[-1].split('/')[-1],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        prohibited_operations=dict(full_dataset_copy=False, global_drop_caches=False, vm_changes=False, other_process_kills=False))
    dump(LOCAL / 'ready.json', result)
    dump(RESULT, result)
    print(json.dumps(dict(status=result['status'], staged_GB=result['staged_GB'], copy=copy)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'stage'])
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / 'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {'prepare': prepare, 'stage': stage}[args.command]()
        except Exception as error:
            previous = json.loads(RESULT.read_text()) if RESULT.exists() else {}
            dump(RESULT, dict(previous, status='STAGE_PAUSED', staging_ready=False, phase_final_complete=False,
                github_sync_complete=False, error=type(error).__name__ + ': ' + str(error), stopped_at=utc()))
            raise


if __name__ == '__main__':
    main()
