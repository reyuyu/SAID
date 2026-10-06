"""Resume a disposable local mirror; benchmark, verify, audit; never train."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import ctypes
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import signal
import sqlite3
import stat
import subprocess
import threading
import time

from recovery.resource_stall_v2 import system_snapshot, GIB, dump

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'local_assets/training/ShareGPT4V'
LOCAL = Path('/root/said_s02_stage500')
IMAGES = LOCAL / 'ShareGPT4V'
RAW = ROOT / 'recovery/evidence/s02-local-full-data-local'
MANIFEST = ROOT / 'recovery/required_training_images.jsonl'
OLD_LEDGER = ROOT / 'recovery/evidence/s02-stage500-local/hash-ledger.jsonl'
LEDGER = RAW / 'copy-hashes.jsonl'
RESULT = ROOT / 'recovery/S02_LOCAL_FULL_DATA.json'
REPORT = ROOT / 'recovery/S02_LOCAL_FULL_DATA.md'
DB = LOCAL / 'full-stage-progress.sqlite'
EXPECTED = dict(sam=569486, coco=118287, llava=558128)
TOTAL = sum(EXPECTED.values())
INDEX_SHA = '0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8'
MANIFEST_SHA = '60106915fb9ba11ba4686081d7bddb42536d9779b0d32ae3df83166d1e456c2f'
BUFFER = 8 << 20


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(BUFFER), b''):
            digest.update(chunk)
        advise(handle.fileno())
    return digest.hexdigest()


def advise(fd):
    if hasattr(os, 'posix_fadvise'):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass  # Optional hint; never a global cache operation.


def relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\0' in name or not p.parts or p.parts[0] not in EXPECTED:
        raise ValueError('Unsafe training path')
    if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Non-image training path')
    return str(p)


def local_path(name, root=IMAGES):
    p = Path(root) / relative(name)
    if not p.parent.resolve().is_relative_to(Path(root).resolve()) or p.is_symlink():
        raise RuntimeError('Local path symlink/escape; no NFS fallback')
    return p


def rows(path):
    if Path(path).exists():
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def connection():
    db = sqlite3.connect(DB)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS images (ordinal INTEGER PRIMARY KEY, family TEXT, parent TEXT, relative TEXT UNIQUE, size INTEGER, state TEXT DEFAULT "pending", sha TEXT, decoded INTEGER DEFAULT 0)')
    db.execute('CREATE INDEX IF NOT EXISTS queue ON images(state,family,parent,relative)')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
    return db


def meta(db, key, value=None):
    if value is not None:
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key,json.dumps(value)))
        db.commit()
    row = db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else None


def ensure_local_index(source=None,destination=None,records_sha=INDEX_SHA):
    source=Path(source or ROOT/'runtime/SAID-nest-clip-v1/data_index')
    destination=Path(destination or LOCAL/'data_index')
    destination.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError('Training index must be local, not a symlink')
    verified={}
    for name in ('records.jsonl','offsets.npy','metadata.json'):
        incoming,final=source/name,destination/name
        expected=records_sha if name=='records.jsonl' else sha(incoming)
        if final.is_symlink():
            raise RuntimeError('Local training-index symlink forbidden')
        if final.exists():
            if sha(final)!=expected:
                raise RuntimeError('Existing frozen local index changed; preserve and investigate')
        else:
            temp=final.with_suffix(final.suffix+'.part')
            if temp.is_symlink():
                raise RuntimeError('Index partial symlink forbidden')
            checksum=hashlib.sha256()
            with incoming.open('rb') as reader,temp.open('wb') as writer:
                for chunk in iter(lambda:reader.read(BUFFER),b''):
                    checksum.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())  # Only three index artifacts, never per JPEG.
                advise(reader.fileno())
            if checksum.hexdigest()!=expected:
                raise RuntimeError('Canonical training index hash drift')
            temp.replace(final)
        verified[name]=expected
    return verified


def rebuild_manifest():
    target = RAW / 'required_training_images.jsonl'
    if target.exists():
        if sha(target) != MANIFEST_SHA:
            raise RuntimeError('Frozen manifest drift')
    else:
        counts, digest, index_hash = Counter(), hashlib.sha256(), hashlib.sha256()
        tmp = target.with_suffix('.jsonl.pending')
        # Existing local lossless index is preferred; every byte is still hash-checked.
        index = LOCAL / 'data_index/records.jsonl'
        if not index.exists():
            index = ROOT / 'runtime/SAID-nest-clip-v1/data_index/records.jsonl'
        with index.open('rb') as source, tmp.open('wb', buffering=BUFFER) as output:
            for ordinal, raw in enumerate(source, 1):
                index_hash.update(raw)
                p = PurePosixPath(relative(json.loads(raw)['image']))
                row = dict(family=p.parts[0], absolute_path=str(SOURCE / p), basename=p.name,
                    image_id=p.stem, relative_path=str(p), ordinal=ordinal)
                encoded = (json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                output.write(encoded)
                digest.update(encoded)
                counts[p.parts[0]] += 1
        if index_hash.hexdigest() != INDEX_SHA or digest.hexdigest() != MANIFEST_SHA or dict(counts) != EXPECTED:
            raise RuntimeError('Reconstructed index/manifest differs from frozen historical SHA/counts')
        tmp.replace(target)
    if MANIFEST.is_symlink() and not MANIFEST.exists():
        if os.readlink(MANIFEST) != '/root/.cache/said-recovery/final-manifest-audit/required_training_images.jsonl':
            raise RuntimeError('Unexpected dangling manifest link')
        temporary = MANIFEST.with_suffix('.jsonl.pending-link')
        temporary.symlink_to(target)
        temporary.replace(MANIFEST)
    elif not MANIFEST.exists():
        MANIFEST.symlink_to(target)
    elif sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('Canonical frozen manifest changed')
    return dict(path=str(MANIFEST), persistent_target=str(target), sha256=MANIFEST_SHA,
                index_sha256=INDEX_SHA, bytes=target.stat().st_size, reconstruction_byte_exact=True)


def prepare():
    from recovery.local_ssd_stage import sizes_from_installation
    RAW.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(exist_ok=True)
    index_hashes=ensure_local_index()
    manifest = rebuild_manifest()
    db = connection()
    if not meta(db,'manifest_complete'):
        sizes = sizes_from_installation()
        db.execute('DELETE FROM images')
        batch = []
        for row in rows(MANIFEST):
            name = relative(row['relative_path'])
            batch.append((row['ordinal'],row['family'],str(PurePosixPath(name).parent),name,sizes[name]))
            if len(batch) == 10000:
                db.executemany('INSERT INTO images(ordinal,family,parent,relative,size) VALUES (?,?,?,?,?)',batch)
                db.commit()
                batch.clear()
        db.executemany('INSERT INTO images(ordinal,family,parent,relative,size) VALUES (?,?,?,?,?)',batch)
        db.commit()
        meta(db,'manifest_complete',MANIFEST_SHA)
        del sizes
    if meta(db,'manifest_complete') != MANIFEST_SHA:
        raise RuntimeError('Progress DB is from a different frozen manifest')
    counts = dict(db.execute('SELECT family,COUNT(*) FROM images GROUP BY family'))
    if counts != EXPECTED:
        raise RuntimeError('Frozen unique family counts differ')
    # Merge old ledger and any interrupted full-stage ledger without touching image bytes.
    for ledger in (OLD_LEDGER,LEDGER):
        batch = []
        for row in rows(ledger):
            batch.append((row['source_sha256'],relative(row['relative_path']),row['size_bytes']))
            if len(batch) == 10000:
                db.executemany('UPDATE images SET sha=? WHERE relative=? AND size=?',batch)
                db.commit()
                batch.clear()
        db.executemany('UPDATE images SET sha=? WHERE relative=? AND size=?',batch)
        db.commit()
    reused, payload, total, missing, remaining = 0,0,0,0,0
    parents = set()
    batch = []
    for ordinal,name,size,known,previous_state in db.execute('SELECT ordinal,relative,size,sha,state FROM images ORDER BY ordinal'):
        p = IMAGES / name
        if p.parent not in parents:
            if not p.parent.resolve().is_relative_to(IMAGES.resolve()):
                raise RuntimeError('Existing mirror parent escapes cache')
            parents.add(p.parent)
        total += size
        try:
            st = p.lstat()
        except FileNotFoundError:
            state = 'pending'
            missing += 1
            remaining += size
        else:
            if not stat.S_ISREG(st.st_mode) or st.st_size != size:
                raise RuntimeError('Existing local image symlink/size mismatch; preserve and investigate: '+name)
            state = ('copied' if previous_state=='copied' else 'reused') if known else 'unverified'
            reused += 1
            payload += size
        batch.append((state,ordinal))
        if len(batch) == 10000:
            db.executemany('UPDATE images SET state=? WHERE ordinal=?',batch)
            db.commit()
            batch.clear()
    db.executemany('UPDATE images SET state=? WHERE ordinal=?',batch)
    db.commit()
    unverified = db.execute('SELECT COUNT(*) FROM images WHERE state="unverified"').fetchone()[0]
    disk = shutil.disk_usage(LOCAL)
    if disk.free < remaining + 64*GIB:
        raise RuntimeError('Local disk lacks space plus64GiB reserve')
    proof = dict(status='LOCAL_FULL_PREFLIGHT_READY', manifest=manifest, family_counts=counts,total_images=TOTAL,
        local_index_sha256=index_hashes,
        total_bytes=total,total_GB=total/1e9,total_GiB=total/GIB,local_existing_images=reused,local_existing_bytes=payload,
        remaining_images=missing,remaining_bytes=remaining,unledgered_existing_images=unverified,
        destination_free_bytes=disk.free,queue_order='family (COCO,LLaVA,SAM), source parent directory, filename',
        storage=dict(mount=subprocess.check_output(['findmnt','-T',str(LOCAL),'-no','TARGET,SOURCE,FSTYPE'],text=True).strip(),
            lifecycle='ephemeral Docker overlay cache; pod rebuild may discard it',risk_accepted_by_user=True,
            persistent_source=str(SOURCE),source_deletion_forbidden=True),prepared_utc=now())
    previous_proof=meta(db,'preflight') or {}
    proof['initial_reused_images']=previous_proof.get('initial_reused_images',previous_proof.get('local_existing_images',reused))
    proof['initial_reused_bytes']=previous_proof.get('initial_reused_bytes',previous_proof.get('local_existing_bytes',payload))
    meta(db,'preflight',proof)
    db.close()
    dump(RAW/'preflight.json',proof)
    dump(RESULT,dict(proof,training_started=False,github_sync_complete=False,ready=False))
    print(json.dumps(proof),flush=True)
    return proof


def atomic_copy(source, destination, size, stop, root=IMAGES):
    destination = local_path(str(Path(destination).relative_to(root)),root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise RuntimeError('Copy queue must contain missing files only')
    candidates = [destination.with_name(destination.name+s) for s in ('.part','.incomplete')]
    present = [p for p in candidates if p.exists()]
    if len(present)>1:
        raise RuntimeError('Multiple partials require review; preserve both')
    temp = present[0] if present else candidates[0]
    if temp.is_symlink():
        raise RuntimeError('Symlink partial forbidden')
    prefix = temp.stat().st_size if temp.exists() else 0
    if prefix>size:
        raise RuntimeError('Partial exceeds source size')
    digest,count = hashlib.sha256(),0
    with Path(source).open('rb',buffering=BUFFER) as incoming:
        before = os.fstat(incoming.fileno())
        if before.st_size != size:
            raise RuntimeError('Source size differs from inventory')
        with temp.open('r+b' if prefix else 'wb',buffering=BUFFER) as output:
            # Verify retained prefix in the same source pass; never overwrite/re-copy it.
            while count<prefix:
                data = incoming.read(min(BUFFER,prefix-count))
                if not data or output.read(len(data)) != data:
                    raise RuntimeError('Retained partial prefix differs; preserve for review')
                digest.update(data)
                count += len(data)
            output.seek(prefix)
            while data := incoming.read(BUFFER):
                if stop.is_set():
                    raise InterruptedError('Pause requested; partial preserved')
                output.write(data)
                digest.update(data)
                count += len(data)
            output.flush()  # No per-JPEG fsync.
            advise(output.fileno())
        after = os.fstat(incoming.fileno())
        advise(incoming.fileno())
    if count != size or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError('Source changed during copy; partial preserved')
    if stop.is_set():
        raise InterruptedError('Pause requested before atomic rename; partial preserved')
    temp.replace(destination)
    return dict(size_bytes=count,source_sha256=digest.hexdigest(),resumed_prefix_bytes=prefix,
                newly_written_bytes=count-prefix)


def flush_local(paths):
    # Flush only the destination filesystem once per2000 completions, not all host filesystems.
    descriptor = os.open(LOCAL,os.O_RDONLY|os.O_DIRECTORY)
    try:
        library = ctypes.CDLL(None,use_errno=True)
        if library.syncfs(descriptor)!=0:
            raise OSError(ctypes.get_errno(),'destination syncfs failed')
    finally:
        os.close(descriptor)
    for path in paths:
        with Path(path).open('rb') as handle:
            advise(handle.fileno())


class Guard:
    def __init__(self):
        self.since = {}
        self.peak = 0
        self.reason = None
        self.kernel_start = time.monotonic()
        self.last_kernel = 0

    def check(self,s,stamp):
        self.peak = max(self.peak,s['memory_current'])
        events = s['memory_events']
        if events.get('oom_kill',0)>0 or events.get('oom',0)>0 or events.get('under_oom',0):
            self.reason = 'Actual cgroup OOM/oom_kill/under_oom'
        limits = [('memory>300GiB',s['memory_current']>300*GIB,60),
            ('severe memory PSI',s['memory_PSI'].get('full',{}).get('avg10',0)>10,60),
            ('severe IO PSI',s['io_PSI'].get('full',{}).get('avg10',0)>80,120)]
        for key,bad,duration in limits:
            if bad:
                self.since.setdefault(key,stamp)
                if stamp-self.since[key]>=duration:
                    self.reason = key+' sustained'
            else:
                self.since.pop(key,None)
        return self.reason

    def kernel(self,stamp):
        if stamp-self.last_kernel<30:
            return self.reason
        self.last_kernel = stamp
        result = subprocess.run(['dmesg','--raw'],capture_output=True,text=True,timeout=10)
        if result.returncode:
            return self.reason  # Availability recorded; errno/file failures still stop.
        for line in result.stdout.splitlines():
            try:
                when = float(line.split('[',1)[1].split(']',1)[0])
            except (ValueError,IndexError):
                continue
            lower = line.lower()
            if when>self.kernel_start and (('nfs' in lower and 'not responding' in lower) or 'rpc_check_timeout' in lower):
                with (RAW/'kernel-nfs-events.raw.log').open('a') as handle:
                    handle.write(line+'\n')
                self.reason = 'New host-kernel NFS/RPC timeout; attribution unspecified, copy paused'
        return self.reason


def choose_workers(benchmarks):
    selected = None
    for row in sorted(benchmarks,key=lambda r:r['workers']):
        if not row['healthy']:
            continue
        if selected is None or row['MiB_s']>=selected['MiB_s']*1.15:
            selected = row
    if selected is None:
        raise RuntimeError('No healthy benchmark')
    return selected['workers']


class Stage:
    def __init__(self):
        self.db = connection()
        self.proof = meta(self.db,'preflight')
        self.stop = threading.Event()
        self.guard = Guard()
        self.state = meta(self.db,'run_state') or dict(started_utc=now(),elapsed_s=0,newly_copied_images=0,
            newly_copied_bytes=0,resumed_prefix_bytes=0,benchmarks=[],errors=[],pod_anomaly=False)
        self.started = time.monotonic()
        self.base_elapsed = self.state['elapsed_s']
        self.base_images = self.state['newly_copied_images']
        self.base_bytes = self.state['newly_copied_bytes']
        self.history = deque([(self.started,self.base_bytes)])
        self.last_snapshot = 0
        self.last_emit = 0
        self.flush_paths = []
        self.phase = 'COPY'
        self.workers = 0
        self.current_resources=[]
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,self.interrupt)

    def interrupt(self,sig,frame):
        self.state['errors'].append('Supervisor signal '+str(sig))
        self.state['pod_anomaly'] = True
        self.stop.set()

    def observe(self,force=False):
        stamp = time.monotonic()
        if not force and stamp-self.last_snapshot<5:
            return
        s = system_snapshot()
        self.current_resources.append(s)
        self.last_snapshot = stamp
        reason = self.guard.check(s,stamp) or self.guard.kernel(stamp)
        if reason:
            self.stop.set()
        self.history.append((stamp,self.state['newly_copied_bytes']))
        while len(self.history)>2 and stamp-self.history[1][0]>=300:
            self.history.popleft()
        if force or reason or stamp-self.last_emit>=60:
            elapsed = stamp-self.started
            self.state['elapsed_s'] = self.base_elapsed+elapsed
            completed = self.proof.get('initial_reused_images',self.proof['local_existing_images'])+self.state['newly_copied_images']
            payload = self.proof.get('initial_reused_bytes',self.proof['local_existing_bytes'])+self.state['newly_copied_bytes']
            rolling = (self.state['newly_copied_bytes']-self.history[0][1])/max(stamp-self.history[0][0],.001)/2**20
            remaining = max(0,self.proof['total_bytes']-payload)
            event = dict(utc=now(),phase=self.phase,workers=self.workers,completed=completed,total=TOTAL,
                remaining_files=max(0,TOTAL-completed),copied_GB=payload/1e9,rolling5min_MiB_s=rolling,
                ETA_s=remaining/(rolling*2**20) if rolling else None,system=s,peak_cgroup_memory=self.guard.peak,
                pause_reason=reason,elapsed_s=self.state['elapsed_s'])
            with (RAW/'resource-progress.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            dump(RAW/'progress.json',event)
            meta(self.db,'run_state',self.state)
            self.last_emit = stamp
            print(json.dumps({k:event[k] for k in ('utc','phase','workers','completed','copied_GB','rolling5min_MiB_s','ETA_s','peak_cgroup_memory','pause_reason')}),flush=True)
        if self.stop.is_set():
            raise InterruptedError(self.guard.reason or 'Supervisor stopped; progress preserved')

    def copy_run(self,workers,seconds=None):
        self.phase = 'BENCHMARK' if seconds else 'COPY'
        self.workers = workers
        self.current_resources=[]
        self.observe(force=True)
        started = time.monotonic()
        bytes_before,files_before = self.state['newly_copied_bytes'],self.state['newly_copied_images']
        families = Counter()
        iterator = iter(self.db.execute('SELECT ordinal,family,relative,size FROM images WHERE state="pending" ORDER BY family,parent,relative'))
        pending = {}
        exhausted = False
        with ThreadPoolExecutor(max_workers=workers) as pool, LEDGER.open('a',buffering=1<<20) as ledger:
            try:
                while pending or not exhausted:
                    self.observe()
                    deadline = seconds is not None and time.monotonic()-started>=seconds
                    while not deadline and not exhausted and len(pending)<workers:
                        row = next(iterator,None)
                        if row is None:
                            exhausted = True
                            break
                        ordinal,family,name,size = row
                        pending[pool.submit(atomic_copy,SOURCE/name,IMAGES/name,size,self.stop)] = (row,time.monotonic())
                    if not pending:
                        break
                    done,_ = wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        (ordinal,family,name,size),_ = pending.pop(future)
                        result = future.result()
                        ledger.write(json.dumps(dict(relative_path=name,family=family,copied_utc=now(),**result))+'\n')
                        self.db.execute('UPDATE images SET state="copied",sha=? WHERE ordinal=?',(result['source_sha256'],ordinal))
                        self.state['newly_copied_images'] += 1
                        self.state['newly_copied_bytes'] += size
                        self.state['resumed_prefix_bytes'] += result['resumed_prefix_bytes']
                        self.flush_paths.append(IMAGES/name)
                        families[family] += 1
                    if len(self.flush_paths)>=2000:
                        ledger.flush()
                        flush_local(self.flush_paths)
                        os.fsync(ledger.fileno())  # Batched persistent ledger only.
                        self.db.commit()
                        self.flush_paths.clear()
                        if shutil.disk_usage(LOCAL).free<64*GIB:
                            raise RuntimeError('Local reserve below64GiB')
                    if any(time.monotonic()-began>240 for _,began in pending.values()):
                        raise TimeoutError('Source copy exceeded240s; preserve partial and pause')
                    if deadline and not pending:
                        break
            except BaseException:
                self.stop.set()
                for future in pending:
                    future.cancel()
                raise
            finally:
                ledger.flush()
                os.fsync(ledger.fileno())
                self.db.commit()
        elapsed = time.monotonic()-started
        if self.flush_paths:
            flush_local(self.flush_paths)
            self.flush_paths.clear()
        result = dict(workers=workers,requested_seconds=seconds,elapsed_s=elapsed,
            files=self.state['newly_copied_images']-files_before,bytes=self.state['newly_copied_bytes']-bytes_before,
            MiB_s=(self.state['newly_copied_bytes']-bytes_before)/max(elapsed,.001)/2**20,
            files_s=(self.state['newly_copied_images']-files_before)/max(elapsed,.001),family_counts=dict(families),healthy=True,
            source_passes_per_file=1,benchmark_scope='Different real remaining sorted ranges; family composition recorded')
        self.observe(force=True)
        samples=self.current_resources
        result['resources']=dict(samples=len(samples),peak_memory_current=max(s['memory_current'] for s in samples),
            peak_file_cache=max(s['file'] for s in samples),last_memory_current=samples[-1]['memory_current'],
            last_file_cache=samples[-1]['file'],oom=max(s['memory_events'].get('oom',0) for s in samples),
            oom_kill=max(s['memory_events'].get('oom_kill',0) for s in samples),NFS_errors=0,
            IO_PSI_full_avg10_max=max(s['io_PSI'].get('full',{}).get('avg10',0) for s in samples),
            memory_PSI_full_avg10_max=max(s['memory_PSI'].get('full',{}).get('avg10',0) for s in samples),
            PSI_scope='host /proc/pressure; cgroup v1 does not expose cgroup PSI')
        return result

    def verify(self):
        self.phase,self.workers = 'VERIFY',1
        self.observe(force=True)
        from PIL import Image, ImageFile
        import torch
        from train.said_cvssl_data import reference_view_a_transform
        torch.set_num_threads(1)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        transform = reference_view_a_transform()
        counts,total,partials = Counter(),0,0
        rng = random.Random(0)
        reservoirs = {family:[] for family in EXPECTED}
        seen = Counter()
        for ordinal,family,name,size in self.db.execute('SELECT ordinal,family,relative,size FROM images ORDER BY ordinal'):
            p = local_path(name)
            if not p.is_file() or p.stat().st_size!=size:
                raise RuntimeError('Full local size/existence check failed: '+name)
            for suffix in ('.part','.incomplete'):
                partials += int(p.with_name(p.name+suffix).exists())
            counts[family] += 1
            total += size
            seen[family] += 1
            group = reservoirs[family]
            limit = 668 if family=='sam' else 666
            row = (ordinal,name,size)
            if len(group)<limit:
                group.append(row)
            else:
                slot = rng.randrange(seen[family])
                if slot<limit:
                    group[slot] = row
            if ordinal%10000==0:
                self.observe()
        actual_images=0
        stray_partials=[]
        for directory,_,files in os.walk(IMAGES):
            for name in files:
                if name.endswith(('.part','.incomplete')) or '.partial.' in name:
                    stray_partials.append(str(Path(directory)/name))
                if Path(name).suffix.lower() in ('.jpg','.jpeg','.png'):
                    actual_images+=1
            self.observe()
        if dict(counts)!=EXPECTED or total!=self.proof['total_bytes'] or partials or stray_partials or actual_images!=TOTAL:
            raise RuntimeError('Full coverage/count/partial check failed')
        equivalence = []
        for family,group in reservoirs.items():
            for ordinal,name,size in group:
                source,dest = SOURCE/name,local_path(name)
                # Read each selected source only once for SHA and decode/preprocess.
                with source.open('rb') as handle:
                    source_bytes=handle.read()
                    advise(handle.fileno())
                with dest.open('rb') as handle:
                    dest_bytes=handle.read()
                    advise(handle.fileno())
                left,right = hashlib.sha256(source_bytes).hexdigest(),hashlib.sha256(dest_bytes).hexdigest()
                recorded=self.db.execute('SELECT sha FROM images WHERE ordinal=?',(ordinal,)).fetchone()[0]
                if left!=right or (recorded is not None and recorded!=left):
                    raise RuntimeError('Random source/local SHA mismatch: '+name)
                with Image.open(io.BytesIO(source_bytes)) as im:
                    rgb = im.convert('RGB')
                    pixels,shape,tensor = rgb.tobytes(),rgb.size,transform(rgb)
                with Image.open(io.BytesIO(dest_bytes)) as im:
                    rgb = im.convert('RGB')
                    if rgb.size!=shape or rgb.tobytes()!=pixels or not torch.equal(tensor,transform(rgb)):
                        raise RuntimeError('RGB/native preprocessing mismatch: '+name)
                equivalence.append(dict(sample_id=ordinal+999,relative_path=name,family=family,sha256=left))
                self.observe()
        dump(RAW/'equivalence-2000.json',dict(passed=True,samples=equivalence,seed=0))
        # Any final file recovered from an interrupted atomic rename but missing its ledger
        # is verified, never re-copied.
        with LEDGER.open('a') as ledger:
            for ordinal,name,size in self.db.execute('SELECT ordinal,relative,size FROM images WHERE sha IS NULL'):
                source_hash=sha(SOURCE/name)
                if source_hash!=sha(local_path(name)):
                    raise RuntimeError('Unledgered existing file hash mismatch')
                ledger.write(json.dumps(dict(relative_path=name,size_bytes=size,source_sha256=source_hash,reused=True))+'\n')
                self.db.execute('UPDATE images SET sha=?,state="reused" WHERE ordinal=?',(source_hash,ordinal))
        self.db.commit()
        self.phase,self.workers = 'LOCAL_DECODE_AUDIT',16
        def decode(row):
            ordinal,name = row
            with local_path(name).open('rb') as handle:
                with Image.open(handle) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    rgb.load()
                    if min(rgb.size)<=0:
                        raise RuntimeError('Empty local image')
                advise(handle.fileno())
            return ordinal
        iterator = iter(self.db.execute('SELECT ordinal,relative FROM images WHERE decoded=0 ORDER BY family,parent,relative'))
        with ThreadPoolExecutor(max_workers=16) as pool:
            pending,exhausted = {},False
            while pending or not exhausted:
                self.observe()
                while not exhausted and len(pending)<32:
                    row = next(iterator,None)
                    if row is None:
                        exhausted=True
                        break
                    pending[pool.submit(decode,row)] = row
                done,_ = wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    self.db.execute('UPDATE images SET decoded=1 WHERE ordinal=?',(future.result(),))
                self.db.commit()
        audited = self.db.execute('SELECT COUNT(*) FROM images WHERE decoded=1').fetchone()[0]
        if audited!=TOTAL:
            raise RuntimeError('Incomplete local-only decode audit')
        from recovery.s02_full_local_data import frozen_path_proof
        paths = frozen_path_proof(5000)
        dump(RAW/'local-path-proof-5000.json',paths)
        return dict(passed=True,exact_paths=TOTAL,all_sizes_matched=True,symlink_to_NFS=0,partials_remaining=0,
            random_SHA_RGB_preprocess_exact_samples=2000,sample_family_counts={f:len(r) for f,r in reservoirs.items()},
            local_only_decode_checked=audited,local_only_decode_passed=audited,NFS_audit_fallback=False,
            resolved_sample_paths_checked=paths['checked'],resolved_all_local=paths['passed'])

    def run(self):
        status,verification = 'LOCAL_FULL_COPY_PAUSED',None
        try:
            for workers in (2,4,6):
                if not any(b['workers']==workers for b in self.state['benchmarks']):
                    b = self.copy_run(workers,180)
                    self.state['benchmarks'].append(b)
                    meta(self.db,'run_state',self.state)
                    print(json.dumps(dict(event='BENCHMARK_DONE',**b)),flush=True)
            four,six = [next(b for b in self.state['benchmarks'] if b['workers']==w) for w in (4,6)]
            if six['healthy'] and six['MiB_s']>=four['MiB_s']*1.15 and not any(b['workers']==8 for b in self.state['benchmarks']):
                self.state['benchmarks'].append(self.copy_run(8,180))
            self.state['final_workers'] = choose_workers(self.state['benchmarks'])
            self.state['sustained_copy'] = self.copy_run(self.state['final_workers'])
            verification = self.verify()
            status = 'LOCAL_FULL_TRAINING_DATA_READY'
        except Exception as error:
            self.state['errors'].append(type(error).__name__+': '+str(error))
            self.stop.set()
        finally:
            self.state['elapsed_s'] = self.base_elapsed+time.monotonic()-self.started
            meta(self.db,'run_state',self.state)
            result = dict(self.proof,status=status,ready=status=='LOCAL_FULL_TRAINING_DATA_READY',training_started=False,
                github_sync_complete=False,reused_old_files=self.proof.get('initial_reused_images',self.proof['local_existing_images']),
                newly_copied_files=self.state['newly_copied_images'],copy=self.state,verification=verification,
                peak_cgroup_memory=self.guard.peak,pause_reason=self.guard.reason,finished_utc=now(),
                destination_free_bytes=shutil.disk_usage(LOCAL).free,copy_workers_exited=True,
                training_image_root=str(IMAGES),training_index=str(LOCAL/'data_index'),NFS_source_preserved=True,
                raw_evidence_dir=str(RAW),local_hash_ledger=str(LEDGER))
            dump(RESULT,result)
            if result['ready']:
                dump(LOCAL/'full-ready.json',result)
            render(result)
            self.db.close()
            print(json.dumps(dict(event='FINAL',status=status,newly_copied=result['newly_copied_files'],
                peak_cgroup_memory=self.guard.peak,error=self.state['errors'])),flush=True)
        if status!='LOCAL_FULL_TRAINING_DATA_READY':
            raise RuntimeError('Copy paused; retained files/partials/ledgers. '+str(self.state['errors']))


def render(result):
    REPORT.write_text('\n'.join(['# S=0.2 full local training cache','',f"Status: `{result['status']}`.",
        f"Local root: `{IMAGES}`. Ephemeral Docker overlay; risk explicitly accepted. Pod rebuild may discard the cache.",
        f"NFS is the persistent source of truth: `{SOURCE}`. Never delete source images based on local presence.",
        'After cache loss, restore the frozen manifest from NFS and stage missing files again. GitHub backs up code/config/reports.',
        '',json.dumps(result,indent=2),'','No training is started; no global drop_caches, VM changes, archives/masks/eval/cache copying.',
        'Resource PSI is host-scoped on cgroup v1. Benchmarks use different real sorted remaining ranges; family mix is recorded.',
        'Large hash ledgers, manifests, images, partials and raw logs remain local; publication includes only reviewed summaries.'])+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','stage'])
    args=parser.parse_args()
    # Read-only admission; do not kill or interfere with unrelated processes.
    forbidden={'recovery.s02_stage500','recovery.s02_nfs500','recovery.final_manifest_audit',
        'recovery.local_ssd_stage','train.train_nested_semantic_mask'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            arguments=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.decode(errors='replace') in forbidden for arg in arguments):
            raise RuntimeError('Concurrent SAID training/copy/audit detected; no launch')
    RAW.mkdir(parents=True,exist_ok=True)
    with (RAW/'task.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='prepare':
            proof=prepare()
            render(dict(proof,status='LOCAL_FULL_PREFLIGHT_READY'))
        else:
            if not DB.exists() or not (RAW/'preflight.json').exists():
                raise RuntimeError('Prepare exact remaining manifest first')
            Stage().run()


if __name__=='__main__':
    main()
