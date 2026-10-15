"""Read-only collection, stream checks, and serial-per-GPU native evaluation.

Training uses the unmodified historical trainers in their pinned worktrees.
This controller never changes model state or optimizer/sampling mathematics.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
MAIN = Path('/opt/data/private/lklk/SAID')
L14 = Path('/opt/data/private/lklk/said-e2-worktrees/hns-v1-l14-full5epoch6085-fromscratch-v1')
RUNTIME = MAIN / 'runtime/SAID-nest-clip-v1'
RUN = RUNTIME / 'best-checkpoint-sweep-v1'
OUTPUT = REPO / 'experiments/nest_clip_v1/best_checkpoint_sweep_v1'
PYTHON = MAIN / '.venv/bin/python'
ASSETS = MAIN / 'local_assets'
L14_FORMAL = RUNTIME / 'hns-v1-l14-full5epoch6085-fromscratch-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resource_snapshot():
    result = {'utc': utc()}
    for name in ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt',
                 'memory.oom_control', 'memory.stat'):
        p = Path('/sys/fs/cgroup/memory') / name
        if p.exists():
            result[name] = p.read_text().strip()
    result['gpus'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()
    return result


def normalize_metrics(obj):
    x = obj.get('metrics', obj)
    if 'image2text_R1' in x:
        return {d: {f'R@{k}': x[f'{prefix}_R{k}'] * 100 for k in (1, 5, 10)}
                for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}
    if 'image2text' in x:
        return {d: {f'R@{k}': x[prefix][f'R{k}'] * 100 for k in (1, 5, 10)}
                for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}
    if 'R@1' in x:
        return {d: {f'R@{k}': x[f'R@{k}'][d] * 100 for k in (1, 5, 10)}
                for d in ('I2T', 'T2I')}
    return {d: {f'R@{k}': x[d].get(f'R@{k}', x[d].get(f'R{k}')) * 100
                for k in (1, 5, 10)} for d in ('I2T', 'T2I')}


def scores(metrics):
    def mean(names):
        return sum(metrics[n][d]['R@1'] for n in names for d in ('I2T', 'T2I')) / (2 * len(names))
    return {'Score5': mean(list(metrics)),
            'J_long3': mean(['Urban-1k', 'DOCCI', 'Long-DCI']),
            'J_long': mean(['Urban-1k', 'DOCCI']),
            'Short4': mean(['COCO', 'Flickr30k-Test1K']),
            'Urban_I2T': metrics['Urban-1k']['I2T']['R@1'],
            'Urban_T2I': metrics['Urban-1k']['T2I']['R@1'],
            'Urban_Mean': mean(['Urban-1k'])}


def evaluate_l14(step, directory=None):
    """Start four independent jobs together; Urban uses the first free GPU."""
    directory = Path(directory or RUN / f'L14_{step}_exact')
    bare = directory / f'student_step{step}.pt'
    report = read(directory / 'export-check.json')
    assert report['passed'] and sha(bare) == report['bare_sha256']
    output = directory / 'eval'
    output.mkdir(exist_ok=True)
    manifests = ASSETS / 'retrieval_benchmarks/manifests'
    args = {
        'coco': ['--dataset', 'coco', '--root', str(ASSETS / 'evaluation/coco/val2017')],
        'flickr_test1k': ['--dataset', 'manifest', '--manifest', str(manifests / 'flickr30k_test1k.jsonl'),
                          '--image-root', str(ASSETS / 'retrieval_benchmarks/flickr30k/images')],
        'docci': ['--dataset', 'manifest', '--manifest', str(manifests / 'docci_test.jsonl'),
                  '--image-root', str(ASSETS / 'retrieval_benchmarks/docci/images')],
        'long_dci': ['--dataset', 'manifest', '--manifest', str(manifests / 'long_dci_reconstructed.jsonl'),
                     '--image-root', str(ASSETS / 'retrieval_benchmarks/dci/images')],
        'urban': ['--dataset', 'urban', '--root', str(ASSETS / 'evaluation/Urban1k/Urban1k')],
    }
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip(), 'GPU busy'
    state = {'started_utc': utc(), 'bare_sha256': report['bare_sha256'],
             'evaluator_sha256': sha(L14 / 'recovery/eval_l14_native.py'), 'jobs': [],
             'resource_before': resource_snapshot()}
    active = {}
    queued = [(n, i) for i, n in enumerate(('coco', 'flickr_test1k', 'docci', 'long_dci'))
                 if not (output / f'{n}.json').exists()]
    urban_pending = not (output / 'urban.json').exists()
    def launch(name, gpu):
        command = [str(PYTHON), '-u', str(L14 / 'recovery/eval_l14_native.py'), '--checkpoint', str(bare),
                   *args[name], '--output', str(output / f'{name}.json'), '--device', 'cuda:0', '--batch-size', '64']
        rec = {'name': name, 'physical_gpu': gpu, 'command': command, 'started_utc': utc(),
               'log': str(output / f'{name}.controller.log')}
        state['jobs'].append(rec)
        env = dict(os.environ, PYTHONPATH=str(L14), CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        log = open(rec['log'], 'x')
        proc = subprocess.Popen(command, cwd=L14, env=env, stdout=log, stderr=subprocess.STDOUT)
        rec['pid'] = proc.pid
        active[gpu] = (proc, log, rec)
        write(output / 'execution.json', state)
    for name, gpu in queued:
        launch(name, gpu)
    if urban_pending and len(active) < 4:
        launch('urban', min(set(range(4)) - set(active)))
        urban_pending = False
    failed = False
    while active:
        for gpu, (proc, log, rec) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            rec.update(ended_utc=utc(), exit_code=code)
            del active[gpu]
            failed |= code != 0
            if urban_pending:
                launch('urban', gpu)
                urban_pending = False
        write(output / 'execution.json', state)
        if active:
            time.sleep(5)
    state.update(ended_utc=utc(), resource_after=resource_snapshot())
    for n in args:
        p = output / f'{n}.json'
        assert p.exists(), (n, 'missing result')
        assert read(p)['checkpoint_sha256'] == report['bare_sha256']
    state['passed'] = not failed
    write(output / 'execution.json', state)
    assert not failed, 'evaluation task failed; see logs'


def audit_stream(directory, reference):
    """Compare full stream digests, all sampling dictionaries, LR and health."""
    directory = Path(directory)
    config = read(directory / 'config.json')
    rows = [json.loads(s) for s in (directory / 'steps.jsonl').read_text().splitlines()]
    wanted = {r['step'] for r in rows}
    refs = {}
    for line in Path(reference).open():
        row = json.loads(line)
        if row['step'] in wanted:
            refs[row['step']] = row
    assert len(refs) == len(rows), ('missing reference steps', len(refs), len(rows))
    h = config['horizon']
    first = config['parent_completed_updates'] + 1
    assert [r['step'] for r in rows] == list(range(first, config['max_updates'] + 1))
    mask_peak = .001
    for row in rows:
        ref = refs[row['step']]
        assert row['actual_lrs'] == ref['actual_lrs'], ('LR drift', row['step'])
        s = row['step'] - 1
        expected_bb = 1e-6 * (s + 1) / 200 if s < 200 else .5e-6 * (1 + math.cos(math.pi * (s - 200) / (h - 200)))
        factor = .5 * (1 + math.cos(math.pi * s / h))
        assert row['actual_lrs'] == {'backbone': expected_bb, 'text_mask_and_shared_pool': mask_peak * factor,
                                     'visual_mask': mask_peak * factor, 'fusion_adapter': .0002 * factor}
        assert math.isfinite(row['loss'])
        for a, b in zip(row['rank_health'], ref['rank_health'], strict=True):
            assert a['rank'] == b['rank']
            assert a['stream_sha256'] == b['stream_sha256'], ('stream', row['step'], a['rank'])
            assert a['sampling'] == b['sampling'], ('sampling', row['step'], a['rank'])
            assert a['batch'] == b['batch'] and a['valid'] == b['valid']
            assert a['gradients_finite']
    return {'passed': True, 'start_update': first, 'end_update': rows[-1]['step'],
            'updates': len(rows), 'rank_records': len(rows) * 4,
            'sample_positions': sum(a['batch'] for r in rows for a in r['rank_health']),
            'reference_path': str(reference), 'reference_sha256': sha(reference),
            'steps_sha256': sha(directory / 'steps.jsonl'),
            'full_stream_hashes_equal': True, 'full_sampling_dictionaries_equal': True,
            'actual_lrs_equal': True, 'analytic_lrs_equal': True,
            'all_gradients_finite': True}


def collect():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = read(MAIN / 'experiments/nest_clip_v1/epoch_curve_eval_v1/RESULTS.json')['models']['HNS_v1']
    results = {'status': 'IN_PROGRESS', 'units': 'percent', 'selection': 'exploratory public benchmark selection',
               'backbones': {'ViT-B/16': {}, 'ViT-L/14': {}}}
    for step in (3651, 4868):
        rec = history[str(step)]
        metrics = {('Flickr30k-Test1K' if n.lower() == 'flickr30k-test1k' else n):
                   {d: {k: v * 100 for k, v in values.items()} for d, values in m.items()}
                   for n, m in rec['metrics'].items()}
        results['backbones']['ViT-B/16'][str(step)] = {
            'step': step, 'scheduler_horizon': 4868, 'reused': True,
            'checkpoint_sha256': rec['checkpoint_sha256'], 'bare_sha256': rec['bare_sha256'],
            'checkpoint': rec['export']['checkpoint'], 'bare': rec['export']['bare'],
            'metrics': metrics, 'summary_percent': scores(metrics),
            'source': str(MAIN / 'experiments/nest_clip_v1/epoch_curve_eval_v1/RESULTS.json')}
    for backbone, steps in [('ViT-B/16', [4151, 4651]), ('ViT-L/14', [3651, 4151, 4651, 5151, 5651])]:
        for step in steps:
            prefix = 'B16' if backbone == 'ViT-B/16' else 'L14'
            path = RUN / (f'B16_{step}' if prefix == 'B16' else f'L14_{step}_exact')
            export = path / 'export-check.json'
            if not export.exists():
                continue
            e = read(export)
            files = {'COCO': 'coco/coco.json', 'Flickr30k-Test1K': 'flickr/flickr_test1k.json',
                     'DOCCI': 'docci/docci.json', 'Long-DCI': 'long_dci/long_dci.json', 'Urban-1k': 'urban/urban.json'}
            if prefix == 'L14':
                files = dict(zip(files, ['coco.json', 'flickr_test1k.json', 'docci.json', 'long_dci.json', 'urban.json']))
            if not all((path / 'eval' / f).exists() for f in files.values()):
                continue
            raw = {n: read(path / 'eval' / f) for n, f in files.items()}
            assert all(r['checkpoint_sha256'] == e['bare_sha256'] for r in raw.values())
            metrics = {n: normalize_metrics(r) for n, r in raw.items()}
            results['backbones'][backbone][str(step)] = {
                'step': step, 'scheduler_horizon': 4868 if prefix == 'B16' else 6085,
                'reused': False, 'checkpoint': str(path / f'step{step:06d}.pt'),
                'bare': str(path / f'student_step{step}.pt'),
                'checkpoint_sha256': e['checkpoint_sha256'], 'bare_sha256': e['bare_sha256'],
                'metrics': metrics, 'summary_percent': scores(metrics),
                'strict_export': e, 'source': str(path / 'eval'),
                'result_file_sha256': {f: sha(path / 'eval' / f) for f in files.values()}}
    for step in (4868, 6085):
        path = L14 / f'experiments/nest_clip_v1/hns_v1_l14_full5epoch6085_fromscratch_v1/RESULTS_{step}.json'
        rec = read(path)
        metrics = {('Long-DCI' if n == 'Long-DCI-reconstructed' else n): m for n, m in rec['metrics'].items()}
        results['backbones']['ViT-L/14'][str(step)] = {
            'step': step, 'scheduler_horizon': 6085, 'reused': True,
            'checkpoint': str(L14_FORMAL / 'formal6085_retry' / f'step{step:06d}.pt'),
            'bare': str(L14_FORMAL / 'evaluation_retry' / f'bare_student_step{step:06d}.pt'),
            'checkpoint_sha256': rec['checkpoint_sha256'], 'bare_sha256': rec['bare_student_sha256'],
            'metrics': metrics, 'summary_percent': scores(metrics), 'source': str(path),
            'source_sha256': sha(path)}
    for backbone, rows in results['backbones'].items():
        baseline = rows['3651' if backbone == 'ViT-B/16' else '6085']['summary_percent']
        for r in rows.values():
            r['delta_vs_paper_pp'] = {k: v - baseline[k] for k, v in r['summary_percent'].items()}
    results['updated_utc'] = utc()
    write(OUTPUT / 'CHECKPOINT_SWEEP_RESULTS.json', results)
    for b, rows in results['backbones'].items():
        print(b)
        for step, r in sorted(rows.items(), key=lambda x: int(x[0])):
            print(step, json.dumps(r['summary_percent']))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'eval-l14', 'audit-stream', 'snapshot'])
    parser.add_argument('--step', type=int)
    parser.add_argument('--directory')
    parser.add_argument('--reference')
    a = parser.parse_args()
    if a.command == 'collect':
        collect()
    elif a.command == 'eval-l14':
        evaluate_l14(a.step, a.directory)
    elif a.command == 'audit-stream':
        result = audit_stream(a.directory, a.reference)
        write(Path(a.directory) / 'sweep-stream-audit.json', result)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(resource_snapshot(), indent=2))
