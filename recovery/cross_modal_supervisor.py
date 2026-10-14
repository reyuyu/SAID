"""Durable, fail-closed supervision for the already authorized X1/X2 stages.

Does not change training math or silently retry failed training. X3 is excluded
because the recorded paired implementation does not implement both terms.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SAID = Path('/opt/data/private/lklk/SAID')
RUNTIME = SAID / 'runtime/said_b16_cross_modal_joint_mask_v1'
REFERENCE = SAID / 'runtime/SAID-nest-clip-multiarms500-v1'
BASE_RESULTS = SAID / 'experiments/nest_clip_v1/said_b16_joint_mask_multiarms500_v1/results'
INIT = SAID / 'runtime/SAID-nest-clip-v1/shared/step000000.pt'
INDEX = SAID / 'runtime/SAID-nest-clip-v1/data_index'
IMAGES = SAID / 'local_assets/training/ShareGPT4V'
PYTHON = SAID / '.venv/bin/python'
TORCHRUN = SAID / '.venv/bin/torchrun'
INIT_SHA = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
PARENT_SHA = {
    'x1': '64f93d7e2a4cf7346db38e491b77f5eea7feb1e4377bc1a1310bb51e955f9e70',
    'x2': '227b76620ba23c96bc8858797b13aeb038b4d3c8a628ed03e4b1e8c79f9bcc4a',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_stream(paths, stop):
    refs = {}
    for name in ('A-Matched.formal500', 'A-Matched.formal500-resume-v1',
                 'A-Matched.formal1217-root', 'A-Matched.formal3651-root'):
        p = REFERENCE / name / 'steps.jsonl'
        if p.exists():
            for item in rows(p):
                assert item['step'] not in refs, 'Reference has duplicate optimizer updates'
                refs[item['step']] = item
    observed = [item for path in paths for item in rows(path)]
    assert [x['step'] for x in observed] == list(range(1, stop + 1)), 'Missing/duplicate/reordered updates'
    samples = 0
    h = hashlib.sha256()
    for item in observed:
        ref = refs[item['step']]
        for key in ('s', 'epoch', 'actual_lrs'):
            assert item[key] == ref[key], (item['step'], key)
        assert len(item['rank_health']) == 4
        assert [x['rank'] for x in item['rank_health']] == list(range(4))
        for a, b in zip(item['rank_health'], ref['rank_health']):
            assert a['updates'] == item['step'] and a['gradients_finite']
            for key in ('rank', 'batch', 'valid', 'stream_sha256', 'sampling', 'reasons'):
                assert a[key] == b[key], (item['step'], a['rank'], key)
            samples += a['batch']
            h.update(json.dumps([item['step'], a['rank'], a['stream_sha256'], a['sampling']],
                                sort_keys=True, allow_nan=False).encode())
    return dict(passed=True, steps=stop, sample_positions=samples,
                compared_fields=['sample_ids', 'stream_sha256', 'sampling', 'reasons', 'cursor', 'actual_lrs'],
                digest=h.hexdigest(), reference=str(REFERENCE),
                limitation='Source summaries are compared; stochastic image pixels and model gradients are not sampling invariants.')


def scores(metrics):
    r1 = {name: [directions[d]['R@1'] * 100 for d in ('I2T', 'T2I')]
          for name, directions in metrics.items()}
    def average(names):
        values = [v for name in names for v in r1[name]]
        return sum(values) / len(values)
    return dict(Score5=average(r1), J_long3=average(['DOCCI', 'Long-DCI', 'Urban-1k']),
                J_long=average(['DOCCI', 'Urban-1k']), Short4=average(['COCO', 'Flickr30k-test1k']),
                Urban_I2T=r1['Urban-1k'][0], Urban_T2I=r1['Urban-1k'][1],
                Urban_Mean=average(['Urban-1k']))


def eligible(candidate, baseline):
    return candidate['Score5'] >= baseline['Score5'] and candidate['J_long3'] >= baseline['J_long3']


class Supervisor:
    def __init__(self, output):
        self.output = output
        self.child = None
        self.state = dict(status='STARTING', updated_at=time.time(), actions=[], decision_rule={
            'stage1217': 'X1 then X2; X3 excluded for paired-formula defect',
            'stage3651': 'At most one candidate; both Score5 and J_long3 >= A-Matched@1217; '
                         'rank by Score5 then J_long3 then arm name; all engineering gates must pass',
            'max_updates': 3651, 'no_retries_or_parameter_changes': True})

    def update(self, **values):
        self.state.update(values, updated_at=time.time())
        atomic_json(self.output / 'STATE.json', self.state)

    def run(self, command, label, gpu=False, training_output=None):
        log = self.output / f'{label}.log'
        assert not log.exists(), 'Refuse to overwrite subprocess evidence'
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' if training_output else '0' if gpu else ''
        self.state['actions'].append(dict(label=label, command=list(map(str, command)), log=str(log)))
        with log.open('x') as f:
            self.child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                          stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            self.update(status='RUNNING', phase=label, child_pid=self.child.pid)
            last = time.monotonic()
            count = 0
            started = last
            while self.child.poll() is None:
                if training_output:
                    p = training_output / 'steps.jsonl'
                    if p.exists():
                        # A partial final line is normal while the trainer writes.
                        lines = p.read_text().splitlines()
                        valid = [json.loads(x) for x in lines[:-1]]
                        try:
                            valid.append(json.loads(lines[-1]))
                        except (IndexError, json.JSONDecodeError):
                            pass
                        if len(valid) > count:
                            count, last = len(valid), time.monotonic()
                            d = valid[-1]
                            assert all(x['gradients_finite'] for x in d['rank_health'])
                            self.update(step=d['step'], phase=label)
                    limit = 3600 if count == 0 else 600
                    assert time.monotonic() - (started if count == 0 else last) < limit, 'Training progress timeout'
                time.sleep(15)
            code = self.child.wait()
            self.child = None
        assert code == 0, f'{label} exited {code}; evidence in {log}'

    def adopt(self, pid, directory):
        start = Path(f'/proc/{pid}/stat').read_text().split()[21]
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        assert 'torchrun' in cmd and str(directory) in cmd
        last_step, last_progress = 500, time.monotonic()
        while Path(f'/proc/{pid}/stat').exists():
            assert Path(f'/proc/{pid}/stat').read_text().split()[21] == start, 'PID reused'
            p = directory / 'steps.jsonl'
            if p.exists():
                lines = p.read_text().splitlines()
                try:
                    d = json.loads(lines[-1])
                except (IndexError, json.JSONDecodeError):
                    time.sleep(15)
                    continue
                assert all(x['gradients_finite'] for x in d['rank_health'])
                if d['step'] > last_step:
                    last_step, last_progress = d['step'], time.monotonic()
                self.update(status='MONITORING_ADOPTED', arm='x1', phase='train1217',
                            step=last_step, adopted_pid=pid)
            if time.monotonic() - last_progress > 600:
                # Stop only the verified adopted launcher. torchrun tears down its workers.
                os.kill(pid, signal.SIGTERM)
                raise RuntimeError('Adopted training progress timeout; stopped own launcher')
            time.sleep(15)

    def validate(self, arm, directory, stop, paths):
        acceptance = read(directory / 'acceptance.json')
        assert acceptance['passed'] and acceptance['resource_failure'] is None
        assert len(acceptance['ranks']) == 4
        assert all(x['completed_updates'] == stop and x['max_parameter_difference_from_rank0'] == 0
                   for x in acceptance['ranks'])
        stream = audit_stream(paths, stop)
        atomic_json(self.output / f'{arm}_{stop}_STREAM_PROOF.json', stream)
        checkpoint = directory / f'step{stop:06d}.pt'
        cfg = read(directory / 'config.json')
        assert cfg['init_sha256'] == INIT_SHA and cfg['horizon'] == 4868
        assert cfg['fusion'] == f'cross_modal_{arm}' and cfg['batch_size'] == 256
        for path, expected in cfg['code_sha256'].items():
            assert sha(ROOT / path) == expected, f'Pinned source changed: {path}'
        full_sha = sha(checkpoint)
        bare = directory / f'bare_student_step{stop:06d}.pt'
        self.run([PYTHON, '-m', 'tools.nest_clip', 'export', '--checkpoint', checkpoint,
                  '--output', bare, '--expect-updates', stop], f'{arm}_{stop}_export')
        self.run([PYTHON, '-m', 'tools.nest_clip', 'verify-export', '--checkpoint', checkpoint,
                  '--bare', bare, '--output', self.output / f'{arm}_{stop}_EXPORT_PROOF.json',
                  '--index-dir', INDEX, '--image-root', IMAGES], f'{arm}_{stop}_verify')
        assert sha(checkpoint) == full_sha, 'Checkpoint changed during export'
        atomic_json(self.output / f'{arm}_{stop}_IDENTITY.json',
                    dict(checkpoint=str(checkpoint), checkpoint_sha256=full_sha,
                         bare=str(bare), bare_sha256=sha(bare), acceptance=acceptance))
        return bare

    def evaluate(self, arm, stop, bare):
        output = self.output / f'{arm}_{stop}_eval'
        output.mkdir(exist_ok=False)
        for dataset, root in [('coco', SAID / 'local_assets/evaluation/coco/val2017'),
                              ('urban', SAID / 'local_assets/evaluation/Urban1k/Urban1k')]:
            self.run([PYTHON, '-m', 'tools.eval_nest_native', '--checkpoint', bare, '--dataset', dataset,
                      '--root', root, '--device', 'cuda:0', '--batch-size', 64,
                      '--output', output / f'{dataset}_native.json'], f'{arm}_{stop}_{dataset}', gpu=True)
        assets = SAID / 'local_assets/retrieval_benchmarks'
        protocols = [f'{name}:{assets}/manifests/{manifest}:{assets}/{images}/images' for name, manifest, images in
                     [('flickr_test1k', 'flickr30k_test1k.jsonl', 'flickr30k'),
                      ('docci', 'docci_test.jsonl', 'docci'),
                      ('long_dci', 'long_dci_reconstructed.jsonl', 'dci')]]
        self.run([PYTHON, '-m', 'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                  '--checkpoint', bare, '--device', 'cuda:0', '--batch-size', 64,
                  '--output-dir', output, *protocols], f'{arm}_{stop}_extended', gpu=True)
        coco, urban = read(output / 'coco_native.json'), read(output / 'urban_native.json')
        metrics = {'COCO': {d: {f'R@{k}': coco[f'{prefix}_R{k}'] for k in (1, 5, 10)}
                            for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]},
                   'Urban-1k': {d: {f'R@{k}': urban[prefix][f'R{k}'] for k in (1, 5, 10)}
                                for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}}
        for name, key in [('Flickr30k-test1k', 'flickr_test1k'), ('DOCCI', 'docci'), ('Long-DCI', 'long_dci')]:
            m = read(output / f'{key}.json')['metrics']
            metrics[name] = {d: {k: v[d] for k, v in m.items()} for d in ('I2T', 'T2I')}
        result = dict(arm=arm, step=stop, metrics=metrics, scores=scores(metrics), exploratory=True)
        atomic_json(output / 'SUMMARY.json', result)
        return result

    def train(self, arm, stop, parent):
        # Refuse to touch any unrelated GPU consumer, including an old launcher.
        lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        assert not lines.strip(), 'BLOCKED_GPU: another GPU process remains'
        output = RUNTIME / f'{arm}_formal{stop}_supervised_v1'
        assert not output.exists(), 'Preserve existing run; no automatic retry'
        assert sha(INIT) == INIT_SHA
        self.run([TORCHRUN, '--standalone', '--nnodes=1', '--nproc-per-node=4', '-m',
                  'train.train_nested_semantic_mask', '--config', f'configs/said_b16_cross_modal_{arm}_500.json',
                  '--init-state', INIT, '--index-dir', INDEX, '--image-root', IMAGES,
                  '--output-dir', output, '--run-type', 'formal', '--max-updates', stop,
                  '--resume', parent], f'{arm}_{stop}_train', training_output=output)
        return output

    def main(self, pid):
        assert sha(INIT) == INIT_SHA
        for arm, expected in PARENT_SHA.items():
            assert sha(RUNTIME / f'{arm}_formal500/step000500.pt') == expected
        self.run([PYTHON, '-m', 'unittest', 'tests.test_cross_modal_supervisor', '-v'], 'cpu_regression')
        directory = RUNTIME / 'x1_formal1217_retry'
        self.adopt(pid, directory)
        candidates = []
        paths_by_arm = {}
        directories = {}
        for arm in ('x1', 'x2'):
            if arm == 'x2':
                directory = self.train(arm, 1217, RUNTIME / f'{arm}_formal500/step000500.pt')
            paths = [RUNTIME / f'{arm}_formal500/steps.jsonl', directory / 'steps.jsonl']
            bare = self.validate(arm, directory, 1217, paths)
            result = self.evaluate(arm, 1217, bare)
            candidates.append(result)
            directories[arm], paths_by_arm[arm] = directory, paths
            self.update(status='STAGE1217_COMPLETE', arm=arm, scores=result['scores'])
        baseline = read(BASE_RESULTS / 'A-Matched-1217/SUMMARY.json')['scores']
        approved = [r for r in candidates if eligible(r['scores'], baseline)]
        approved.sort(key=lambda r: (-r['scores']['Score5'], -r['scores']['J_long3'], r['arm']))
        atomic_json(self.output / 'PROMOTION_DECISION.json',
                    dict(baseline=baseline, candidates=candidates,
                         winner=approved[0]['arm'] if approved else None, rule=self.state['decision_rule']))
        if approved:
            arm = approved[0]['arm']
            directory = self.train(arm, 3651, directories[arm] / 'step001217.pt')
            paths = paths_by_arm[arm] + [directory / 'steps.jsonl']
            bare = self.validate(arm, directory, 3651, paths)
            result = self.evaluate(arm, 3651, bare)
            self.update(final_result=result)
            # Flickr-full is a required final artifact. A missing frozen protocol
            # is reported explicitly instead of substituting the 1k test split.
            manifest = SAID / 'local_assets/retrieval_benchmarks/manifests/flickr30k_full.jsonl'
            assert manifest.exists(), 'Flickr-full frozen manifest missing: final delivery incomplete'
            self.run([PYTHON, '-m', 'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                      '--checkpoint', bare, '--device', 'cuda:0', '--batch-size', 64,
                      '--output-dir', self.output / f'{arm}_3651_flickr_full',
                      f'flickr_full:{manifest}:{SAID}/local_assets/retrieval_benchmarks/flickr30k/images'],
                     f'{arm}_3651_flickr_full', gpu=True)
        for arm, expected in PARENT_SHA.items():
            assert sha(RUNTIME / f'{arm}_formal500/step000500.pt') == expected
        assert sha(INIT) == INIT_SHA
        self.update(status='STAGES_COMPLETE', requires='Report, source/initialization audit and GitHub synchronization remain; no more training')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adopt-pid', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (RUNTIME / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Supervisor(args.output)
        try:
            runner.main(args.adopt_pid)
        except BaseException as exc:
            if runner.child is not None and runner.child.poll() is None:
                os.killpg(runner.child.pid, signal.SIGTERM)
                try:
                    runner.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(runner.child.pid, signal.SIGKILL)
                    runner.child.wait()
            runner.update(status='STOPPED_FAILURE', error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
