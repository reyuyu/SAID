"""Explicit-only future training entry; staging never invokes this module."""
import argparse
import json
import mmap
import os
from pathlib import Path
import signal
import subprocess
import time

from recovery.s02_stage500 import LOCAL, RAW, ROOT, CONFIG, rows, disk, sha
from train.nested_semantic_data import NestedDataset


class StrictLocalDataset(NestedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.image_root.resolve() != (LOCAL / 'ShareGPT4V').resolve():
            raise RuntimeError('Required verified stage500 local root; NFS fallback forbidden')
        self._path_proofs = 0

    def __getitem__(self, index):
        import numpy as np
        # Resolve BEFORE native Image.open, so a symlink cannot redirect a read to NFS.
        if self._records is None:
            self._file = (self.index_dir / 'records.jsonl').open('rb')
            self._records = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = np.load(self.index_dir / 'offsets.npy', mmap_mode='r')
        record = json.loads(self._records[self._offsets[index]:self._offsets[index + 1]])
        path = (self.image_root / record['image']).resolve()
        if not path.is_relative_to(self.image_root.resolve()) or not path.is_file():
            raise RuntimeError('Required local image missing/escaped; NFS fallback forbidden')
        result = super().__getitem__(index)
        if self._path_proofs < 8:
            evidence = dict(event='LOCAL_STAGE500_IMAGE_READ', rank=int(os.environ['RANK']),
                worker_pid=os.getpid(), sample_id=result['sample_id'], actual_path=str(path),
                local_root=str(self.image_root), NFS_fallback=False, native_dataset_semantics=True)
            print(json.dumps(evidence), flush=True)
            directory = Path(os.environ['SAID_S02_PHASE_LOCAL'])
            with (directory / f'image-paths-rank{os.environ["RANK"]}-pid{os.getpid()}.jsonl').open('a') as handle:
                handle.write(json.dumps(evidence) + '\n')
            self._path_proofs += 1
        return result


def admission():
    ready = json.loads((LOCAL / 'ready.json').read_text())
    if ready['status'] != 'LOCAL_STAGE500_READY' or not ready['verification']['passed']:
        raise RuntimeError('Stage500 admission missing')
    if sha(RAW / 'unique-images.jsonl') != ready['manifest_sha256'] or sha(CONFIG) != ready['config_sha256']:
        raise RuntimeError('Manifest/config provenance changed')
    disk()
    count = 0
    for row in rows(RAW / 'unique-images.jsonl'):
        path = LOCAL / 'ShareGPT4V' / row['relative_path']
        if path.is_symlink() or not path.resolve().is_relative_to((LOCAL / 'ShareGPT4V').resolve()) or not path.is_file() or path.stat().st_size != row['size_bytes']:
            raise RuntimeError('Required local file missing/mismatched; no NFS fallback: ' + str(path))
        count += 1
    if count != ready['unique_images_count']:
        raise RuntimeError('Required manifest coverage mismatch')
    if sha(LOCAL / 'data_index/records.jsonl') != ready['index_records_sha256']:
        raise RuntimeError('Training index changed')
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Explicitly launch only when user authorizes training')
    parser.add_argument('--worker', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.worker:
        import sys
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import local_image_dataset, reproduction_train_gate
        local_image_dataset.LocalImageDataset = StrictLocalDataset
        reproduction_train_gate.RUN = ROOT / 'runtime/SAID-nest-clip-v1/s02-stage500-conservative'
        sys.argv = [sys.argv[0], *remaining]
        reproduction_train_gate.main()
    else:
        admission()
        if not args.launch:
            print('LOCAL_STAGE500_ADMISSION_PASS; training not started')
            return
        if remaining:
            parser.error('No arbitrary overrides accepted for frozen reproduction')
        phase = LOCAL / 'training-path-proofs'
        phase.mkdir(exist_ok=False)
        environment = dict(os.environ, SAID_S02_STAGE='step500', SAID_S02_PHASE_LOCAL=str(phase),
                           SAID_FULL_SUPERVISOR_PID=str(os.getpid()))
        command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nnodes=1', '--nproc-per-node=4',
            '--max-restarts=0', '-m', 'recovery.s02_train500', '--worker', '--config', str(CONFIG),
            '--init-state', str(ROOT / 'runtime/SAID-nest-clip-v1/shared/step000000.pt'),
            '--index-dir', str(LOCAL / 'data_index'), '--image-root', str(LOCAL / 'ShareGPT4V'),
            '--output-dir', str(ROOT / 'runtime/SAID-nest-clip-v1/s02-stage500-conservative/step500'),
            '--run-type', 'formal', '--max-updates', '500']
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1.reproduction_full import resource_recurrence
        # Preserve the existing native three-consecutive-full-cycles >3s guard.
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        cycles_path = ROOT / 'runtime/SAID-nest-clip-v1/s02-stage500-conservative/step500/cycle_timing.jsonl'
        try:
            while process.poll() is None:
                cycles = []
                if cycles_path.exists():
                    for line in cycles_path.read_text().splitlines():
                        try:
                            cycles.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # writer may be completing its final line
                if resource_recurrence(cycles)['triggered']:
                    raise RuntimeError('Native three-consecutive-full-cycles >3s guard triggered')
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # only this explicitly launched training group
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    main()
