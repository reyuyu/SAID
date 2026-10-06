"""Fresh canonical-NFS step0->500, strict five-set eval, unconditional stop500."""
import argparse
import datetime
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from recovery.nfs500_policy import POLICY, wait_expired

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runtime/SAID-nest-clip-v1/s02-nfs500-20261006'
OUT = ROOT / 'recovery'
PHASE = Path('/tmp/said-s02-nfs500-20261006')
INDEX = ROOT / 'runtime/SAID-nest-clip-v1/data_index'
IMAGES = ROOT / 'local_assets/training/ShareGPT4V'
CONFIG = OUT / 'configs/summary02_nfs500.json'
STEP0 = ROOT / 'runtime/SAID-nest-clip-v1/shared/step000000.pt'
STEP0_SHA = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
PYTHON = str(ROOT / '.venv/bin/python')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rows(path):
    values = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Only a concurrent incomplete final line; reread on next poll.
    return values


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    if not values:
        return dict(count=0, median=None, p95=None, p99=None, max=None)
    ordered = sorted(values)
    def quantile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return dict(count=len(values), median=statistics.median(values), p95=quantile(.95),
                p99=quantile(.99), max=max(values))


def canonical_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError('Canonical path escaped NFS root; no mirror/fallback permitted')
    return path


def worker():
    import numpy as np
    from train.nested_semantic_data import NestedDataset
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import (
        local_image_dataset, reproduction_train_gate as gate, training_phase_timing as timing)

    class HeartbeatIterator(timing.TimedIterator):
        def __next__(self):
            path = PHASE / f'heartbeat-rank{self.recorder.rank}.json'
            dump(path, dict(state='DATA_WAIT', rank=self.recorder.rank,
                            step=self.recorder.counter + 1, started_monotonic=time.monotonic(), utc=now()))
            try:
                return super().__next__()
            finally:
                dump(path, dict(state='BATCH_RETURNED', rank=self.recorder.rank,
                                step=self.recorder.counter, utc=now()))

    # A top-level dataset is required for spawn workers; only import after torchrun.
    timing.TimedIterator = HeartbeatIterator
    local_image_dataset.LocalImageDataset = CanonicalNFSDataset
    gate.RUN = RUN
    gate.EXP = RUN / 'reviewed'
    gate.EXP.mkdir(exist_ok=True)
    provenance = json.loads((RUN / 'launch-provenance.json').read_text())
    gate.AUTHORIZED_SOURCE_CHANGES = {
        'train/train_nested_semantic_mask.py': (
            '56eaa83f7ee3ea9fa97a754a52288fc61adb28dd41129015ec39b129b82c57e3',
            provenance['source_sha256']['train/train_nested_semantic_mask.py'])}
    # Native loader remains workers8/prefetch2; timeout changes control only.
    original_loader = trainer.DataLoader
    def loader(*args, **kwargs):
        return original_loader(*args, **kwargs, timeout=60)
    trainer.DataLoader = loader
    gate.main()


from train.nested_semantic_data import NestedDataset


class CanonicalNFSDataset(NestedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.image_root.resolve() != IMAGES.resolve() or self.index_dir.resolve() != INDEX.resolve():
            raise RuntimeError('Formal run requires canonical NFS image and index roots')
        self._proofs = 0

    def __getitem__(self, index):
        import numpy as np
        if self._records is None:
            self._file = (self.index_dir / 'records.jsonl').open('rb')
            self._records = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = np.load(self.index_dir / 'offsets.npy', mmap_mode='r')
        record = json.loads(self._records[self._offsets[index]:self._offsets[index + 1]])
        path = canonical_path(self.image_root, record['image'])
        value = super().__getitem__(index)  # Exact native decode/preprocess; errors propagate.
        if self._proofs < 4:
            proof = dict(event='CANONICAL_NFS_IMAGE_READ', rank=int(os.environ['RANK']),
                         worker_pid=os.getpid(), sample_id=value['sample_id'], relative_path=record['image'],
                         actual_path=str(path), local_mirror_used=False, fallback=False)
            with (PHASE / f'image-paths-rank{os.environ["RANK"]}-pid{os.getpid()}.jsonl').open('a') as handle:
                handle.write(json.dumps(proof) + '\n')
            self._proofs += 1
        return value


class Supervisor:
    def __init__(self):
        from recovery.resource_stall_v2 import Nvml
        self.train = RUN / 'step500'
        self.nvml = Nvml()
        self.started = now()
        self.error = None
        self.result = None
        self.active = None
        self.commands = []

    def execute(self, name, command, training=False):
        from recovery.resource_stall_v2 import system_snapshot
        log = RUN / (name + '.log')
        env = dict(os.environ, SAID_FULL_SUPERVISOR_PID=str(os.getpid()), SAID_S02_STAGE='step500',
                   SAID_S02_PHASE_LOCAL=str(PHASE), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        record = dict(name=name, command=command, started_utc=now(), raw_log=str(log))
        self.commands.append(record)
        dump(RUN / 'commands.json', self.commands)
        with log.open('w') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.active = process
            last_resource = 0
            try:
                while process.poll() is None:
                    timestamp = time.monotonic()
                    if timestamp - last_resource >= 5:
                        snapshot = dict(utc=now(), command=name, system=system_snapshot(), gpu=self.nvml.sample())
                        with (RUN / 'resource-telemetry.jsonl').open('a') as stream:
                            stream.write(json.dumps(snapshot) + '\n')
                        last_resource = timestamp
                        if snapshot['system']['memory_events'].get('oom_kill', 0) > 0:
                            raise RuntimeError('HARD_STOP: actual cgroup oom_kill > 0')
                    if training:
                        for rank in range(4):
                            heartbeat = PHASE / f'heartbeat-rank{rank}.json'
                            if heartbeat.exists() and wait_expired(json.loads(heartbeat.read_text()), timestamp):
                                raise RuntimeError(f'HARD_STOP: rank{rank} data_wait >60s still unreturned')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f'{name} process failed, returncode={process.returncode}; see {log}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                self.active = None
                record.update(ended_utc=now(), returncode=process.returncode)
                dump(RUN / 'commands.json', self.commands)

    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        original = json.loads((OUT / 'configs/summary02.json').read_text())
        config = json.loads(CONFIG.read_text())
        assert {k:v for k,v in config.items() if k != 'resource_policy'} == original
        assert config['resource_policy'] == POLICY
        assert sha(STEP0) == STEP0_SHA
        assert subprocess.check_output(['findmnt', '-T', str(IMAGES), '-no', 'FSTYPE'], text=True).strip() == 'nfs'
        assert not system_snapshot()['memory_events'].get('oom_kill', 0)
        assert not RUN.exists(), 'Never overwrite or resume an earlier trajectory'
        RUN.mkdir(parents=True)
        PHASE.mkdir(exist_ok=False)
        sources = ['recovery/s02_nfs500.py', 'recovery/nfs500_policy.py', 'recovery/configs/summary02_nfs500.json',
                   'train/train_nested_semantic_mask.py',
                   'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
                   'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py']
        dump(RUN / 'launch-provenance.json', dict(started_utc=self.started, supervisor_pid=os.getpid(),
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             step0_sha256=STEP0_SHA, source_sha256={p:sha(ROOT / p) for p in sources},
             canonical_image_root=str(IMAGES), canonical_index=str(INDEX), initial_resources=system_snapshot(),
             previous_boot_kernel_evidence='Unavailable in this pod: journalctl reports no journal files',
             local_mirror_used=False, stop_updates=500, horizon=4868, automatic_retry=False))
        command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nnodes=1', '--nproc-per-node=4',
                   '--max-restarts=0', '-m', 'recovery.s02_nfs500', '--worker', '--config', str(CONFIG),
                   '--init-state', str(STEP0), '--index-dir', str(INDEX), '--image-root', str(IMAGES),
                   '--output-dir', str(self.train), '--run-type', 'formal', '--max-updates', '500']
        try:
            self.execute('train500', command, training=True)
            acceptance = json.loads((self.train / 'acceptance.json').read_text())
            assert acceptance['passed'] and all(r['completed_updates'] == 500 and
                r['max_parameter_difference_from_rank0'] == 0 for r in acceptance['ranks'])
            assert json.loads((RUN / 'first-five-gate.json').read_text())['passed']
            import torch
            checkpoint = torch.load(self.train / 'step000500.pt', map_location='cpu', weights_only=False)
            assert checkpoint['global_step'] == checkpoint['completed_steps'] == 500
            assert all(k in checkpoint for k in ('model', 'optimizer', 'scheduler', 'rng_per_rank', 'sampler', 'data_cursor'))
            assert checkpoint['scheduler']['horizon'] == 4868
            assert {int(s['step']) for s in checkpoint['optimizer']['state'].values()} == {500}
            del checkpoint
            # Reuse only strict native evaluation, never its staging/continuation/publish loop.
            pipeline.EXP = RUN / 'reviewed'
            pipeline.RUN = RUN
            self.result = pipeline.Supervisor.evaluate(self, 500)
            historical = json.loads((ROOT / 'experiments/nest_clip_v1/armb_summary02_4epoch_v1/PARENT_500.json').read_text())
            gate = pipeline.reproduction_gate(self.result['scores_percent'], self.result['metrics'], historical['metrics'])
            self.result.update(status=gate['status'], reproduction_gate=gate)
        except Exception as error:
            self.error = type(error).__name__ + ': ' + str(error)
            print(self.error, flush=True)
        finally:
            self.report()
        if self.error:
            raise RuntimeError(self.error)

    def report(self):
        cycles = rows(self.train / 'cycle_timing.jsonl')
        phases = {rank: rows(PHASE / f'rank{rank}.jsonl') for rank in range(4)}
        lookup = {rank:{r['step']:r for r in records} for rank,records in phases.items()}
        resources = rows(RUN / 'resource-telemetry.jsonl')
        all_wait = [r['data_wait_s'] for records in phases.values() for r in records]
        slowest_wait = [max((lookup[rank].get(r['step'], {}).get('data_wait_s', 0) for rank in range(4))) for r in cycles]
        compact = []
        for row in cycles:
            item = dict(step=row['step'], full_cycle_s=row['four_rank_max_seconds'], ranks=[])
            for rank in range(4):
                p = lookup[rank].get(row['step'], {})
                system = p.get('system_after', {})
                item['ranks'].append(dict(rank=rank, data_wait_s=p.get('data_wait_s'), forward_s=p.get('forward_s'),
                    backward_DDP_s=p.get('backward_s'), optimizer_s=p.get('optimizer_s'),
                    full_cycle_s=p.get('wall_cycle_before_checkpoint_s'),
                    memory_current=system.get('memory_current'), file=system.get('file'),
                    io_PSI_some_avg10=system.get('io_PSI', {}).get('some', {}).get('avg10'),
                    memory_PSI_some_avg10=system.get('memory_PSI', {}).get('some', {}).get('avg10')))
            compact.append(item)
        proofs = [r for p in PHASE.glob('image-paths-*.jsonl') for r in rows(p)]
        raw = list(RUN.glob('*.log')) + list(RUN.glob('*.jsonl')) + list(self.train.glob('*.jsonl')) + list(PHASE.glob('*.jsonl'))
        inventory = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p),
                          time_range_utc=[self.started, now()], uploaded=False) for p in sorted(raw)]
        error_text = (RUN / 'train500.log').read_text(errors='replace') if (RUN / 'train500.log').exists() else ''
        import re
        io_matches = sorted(set(re.findall(r'Image failure sample=\d+[^\n]*|NFS[^\n]*not responding[^\n]*|Input/output error[^\n]*|RPC[^\n]*timed out[^\n]*', error_text, re.I)))
        stats = dict(started_utc=self.started, finished_utc=now(), completed_steps=len(cycles),
                     full_cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),
                     data_wait_seconds_slowest_rank=distribution(slowest_wait), data_wait_seconds_all_rank_batches=distribution(all_wait),
                     rank_data_wait_seconds={str(rank):distribution([r['data_wait_s'] for r in records]) for rank,records in phases.items()},
                     steps_gt3s=sum(r['four_rank_max_seconds'] >3 for r in cycles),
                     steps_gt10s=sum(r['four_rank_max_seconds'] >10 for r in cycles),
                     peak_cgroup_memory_bytes=max((r['system']['memory_current'] or 0 for r in resources), default=0),
                     peak_file_cache_bytes=max((r['system']['file'] for r in resources), default=0),
                     last_resources=resources[-1] if resources else None,
                     gpu_utilization_percent={str(rank):distribution([g['gpu_percent'] for r in resources if r['command']=='train500'
                        for g in r['gpu'] if isinstance(g,dict) and g.get('index')==rank and g.get('gpu_percent') is not None]) for rank in range(4)},
                     io_errors=io_matches, io_error_count=len(io_matches),
                     io_error_scope='Training process logs and native image reads; kernel log availability recorded separately',
                     stop_reason=self.error, slow_steps_policy='PERFORMANCE_WARNING >3s; hard stop five consecutive >30s or unreturned wait>60s',
                     PSI_scope='Host /proc/pressure on cgroup v1; not incorrectly labelled as cgroup PSI',
                     canonical_NFS_path_proof=dict(count=len(proofs), ranks=sorted({r['rank'] for r in proofs}),
                        valid=bool(proofs) and all(not r['local_mirror_used'] and canonical_path(IMAGES, r['relative_path']) == Path(r['actual_path']) for r in proofs)),
                     raw_local_artifacts=inventory)
        dump(OUT / 'NFS_RUNTIME_STATS.json', stats)
        dump(OUT / 'NFS_PER_STEP_SUMMARY.json', compact)
        result = self.result or dict(status='INCOMPLETE_HARD_STOP', step=len(cycles), error=self.error, evaluated=False)
        result.update(runtime=str(RUN), canonical_image_root=str(IMAGES), step0_sha256=STEP0_SHA,
                      first_five_gate=json.loads((RUN / 'first-five-gate.json').read_text()) if (RUN / 'first-five-gate.json').exists() else None,
                      git_launch_provenance=json.loads((RUN / 'launch-provenance.json').read_text()),
                      stopped_at_500=len(cycles)==500, checkpoint_uploaded=False)
        dump(OUT / 'STEP500_RESULTS.json', result)
        gate = dict(result.get('reproduction_gate', {}), status=result['status'], continuation_allowed=False,
                    automatic_continuation=False, stop_updates=500, wait_for_next_user_instruction=True,
                    github_sync='PENDING')
        dump(OUT / 'CONTINUATION_GATE.json', gate)
        scores = result.get('scores_percent', {})
        lines = ['# S=0.2 canonical NFS step500 reproduction', '',
                 f"Status: `{result['status']}`. Completed updates: {len(cycles)}/500; scheduler horizon4868.", '',
                 'Fresh common step0 SHA256: `' + STEP0_SHA + '`.',
                 'Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch, four A10080GB, batch256/rank, seed0, workers8; native mathematics/preprocessing unchanged.', '',
                 'SSD staging was operator-stopped before this run; retained images, partials and hash ledger. No copy workers or local mirror used during this run.',
                 f'Canonical images: `{IMAGES}`. Actual worker path proof: `{stats["canonical_NFS_path_proof"]}`.', '',
                 'Full-cycle seconds (all steps, including startup): `' + json.dumps(stats['full_cycle_seconds']) + '`.',
                 'Slowest-rank data-wait seconds: `' + json.dumps(stats['data_wait_seconds_slowest_rank']) + '`.',
                 f">3s steps: {stats['steps_gt3s']}; >10s: {stats['steps_gt10s']}. Logged I/O failures: {len(io_matches)}.",
                 'Peak sampled cgroup bytes: ' + str(stats['peak_cgroup_memory_bytes']) + '. File cache is reported separately and is not RSS.',
                 'Slow steps only warn; no samples skipped/replaced, no workers/batch/math changed. Detailed phase telemetry includes backward/DDP, optimizer, memory, PSI and GPU utilization.', '',
                 'Five-set strict native scores (%): `' + json.dumps(scores) + '`.',
                 'Both directions R@1/5/10 and historical deltas are in STEP500_RESULTS.json; all-step and rank statistics in NFS_RUNTIME_STATS.json and NFS_PER_STEP_SUMMARY.json.', '',
                 'Complete resumable checkpoint remains local: `' + str(self.train / 'step000500.pt') + '`.',
                 'Strict bare export and evaluation leave that checkpoint unchanged. No continuation is authorized; PASS and FAIL both stop500.', '',
                 'Raw logs remain local; reviewed inventory (paths, SHA256, size and UTC scope) is in NFS_RUNTIME_STATS.json. Checkpoints/datasets/images/caches are excluded from GitHub.',
                 'GitHub synchronization: PENDING; phase is not marked finally complete.', '',
                 'Error: ' + str(self.error)]
        (OUT / 'STEP500_NFS_REPRODUCTION.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.worker:
        sys.argv = [sys.argv[0], *remaining]
        worker()
    else:
        if remaining:
            parser.error('Frozen launcher accepts no arbitrary training overrides')
        supervisor = Supervisor()
        def interrupted(signum, frame):
            raise RuntimeError('HARD_STOP: supervisor received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        supervisor.run()


if __name__ == '__main__':
    main()
