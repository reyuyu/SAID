"""One authorized Summary/Detail experiment: smoke5, fresh500, frozen native eval."""
import argparse
import ast
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DistributedSampler

from train.nested_semantic_data import file_sha
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_summary_detail_500_v1')
PYTHON = '/root/miniconda3/envs/said-repro/bin/python'
TORCHRUN = '/root/miniconda3/envs/said-repro/bin/torchrun'
SHARED = Path('/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt')
INDEX = Path('/root/lk_projects/SAID-nest-clip-v1/data_index')
ASSETS = Path('/root/lk_projects/SAID-assets')
BRANCH = 'codex/nest-balanced-summary-detail-500-v1'
INIT_SHA = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def load(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=2)+'\n')


def records(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def stream_check(row, reference):
    assert row['step'] == reference['step']
    for a, b in zip(row['rank_health'], reference['rank_health']):
        assert a['rank'] == b['rank']
        for field in ['sample_id_sha256', 'full_view_sha256', 'fixed_first_reference_stream_sha256']:
            assert a['sampling'][field] == b['sampling'][field], (row['step'], a['rank'], field)


def preflight():
    b = load(EXP/'BASELINE.json'); c = load(EXP/'config.json')
    assert b['checkpoint_sha256'] == 'f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8'
    assert b['bare_sha256'] == '90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e'
    assert file_sha(b['checkpoint']) == b['checkpoint_sha256']
    assert file_sha(b['student']) == b['bare_sha256']
    assert file_sha(SHARED) == INIT_SHA == b['config']['init_sha256']
    assert c['sampling_mode'] == 'summary_detail'
    for key in c:
        if key not in ['sampling_mode', 'experiment_name']:
            assert c[key] == b['config'][key], key
    paths = ['model/nested_semantic_mask.py', 'model/nested_vcp_mask.py', 'model/nested_fusion_mask.py',
             'model/balanced_hparam_search.py', 'model/model_longclip.py', 'model/longclip.py',
             'model/said_cls_cvssl.py', 'train/said_cvssl_data.py', 'eval/retrieval/coco_retrieval.py',
             'tools/urban1k_retrieval.py']
    unchanged = {}
    for path in paths:
        expected = subprocess.check_output(['git', 'show', '14653c9:'+path], cwd=ROOT)
        assert (ROOT/path).read_bytes() == expected, path
        unchanged[path] = hashlib.sha256(expected).hexdigest()
    old_trainer = subprocess.check_output(['git','show','14653c9:train/train_nested_semantic_mask.py'], cwd=ROOT, text=True)
    new_trainer = (ROOT/'train/train_nested_semantic_mask.py').read_text()
    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    old_functions, new_functions = functions(old_trainer), functions(new_trainer)
    frozen_functions = ['build_optimizer','learning_rates','optimizer_learning_rates','training_horizon','seed_all','rng_state','restore_rng_state']
    assert all(old_functions[n] == new_functions[n] for n in frozen_functions)
    assert load(INDEX/'metadata.json') == b['config']['data']
    assert file_sha(INDEX/'records.jsonl') == b['config']['data']['records_sha256']
    audit = load(EXP/'SAMPLING_AUDIT.json')
    assert audit['status'] == 'COMPLETE' and audit['F_equivalence']['passed']
    assert audit['F_equivalence']['F_tokens_bitwise_equal_count'] == 1000
    baseline = records(Path(b['root'])/'steps.jsonl')
    assert len(baseline) == 500
    for rank in range(4):
        sampler = DistributedSampler(range(1245901), num_replicas=4, rank=rank,
                                     shuffle=True, seed=0, drop_last=False)
        sampler.set_epoch(0)
        ids = [i+1000 for i in itertools.islice(iter(sampler), 500*256)]
        for step, row in enumerate(baseline):
            group = ids[step*256:(step+1)*256]
            h = hashlib.sha256(json.dumps(group, separators=(',', ':')).encode()).hexdigest()
            assert h == row['rank_health'][rank]['sampling']['sample_id_sha256'], (rank, step)
    initial = torch.load(SHARED, map_location='cpu', weights_only=False)
    assert initial['completed_steps'] == 0 and not initial['optimizer']['state']
    assert initial['provenance']['source'] == 'OpenAI CLIP + original random MaskNetwork'
    result = {'passed': True, 'reference_commit': '14653c92c6da9d552a2b624ab169eaaa275cdde8',
        'common_step0_sha256': INIT_SHA, 'baseline_retrained': False,
        'baseline_full_sha256_verified': b['checkpoint_sha256'], 'baseline_bare_sha256_verified': b['bare_sha256'],
        'unchanged_model_optimizer_scheduler_evaluator_sources': unchanged,
        'unchanged_trainer_functions_AST': frozen_functions,
        'indexed_training_data_SHA256_verified': b['config']['data']['records_sha256'],
        'all_500x4_sampler_id_hashes_match': True, 'F_1000_equivalence': audit['F_equivalence'],
        'only_training_behavior_difference': 'summary_detail text construction and its specified raw-sentence validity rule',
        'no_view_relation_or_sentence_drop': True, 'stop_updates': 500, 'scheduler_horizon': 4868,
        'checked_at': now()}
    dump(EXP/'evidence/preflight.json', result)
    dump(RUN/'state.json', {'status': 'READY', 'stage': 'preflight passed', 'branch': BRANCH,
        'started_at': now(), 'baseline_scores': b['scores'], 'stop_updates': 500, 'horizon': 4868})
    print(json.dumps({'preflight': 'PASS', 'sampler_hashes_compared': 2000}), flush=True)


def execute(name, argv, gpu=False, training_root=None):
    command_dir = EXP/'commands'; command_dir.mkdir(exist_ok=True)
    execution = RUN/'execution'; execution.mkdir(exist_ok=True)
    log = execution/(name+'.console.txt')
    assert not log.exists(), 'Preserve interrupted/failed stage evidence; do not overwrite it'
    if gpu:
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        assert not occupied, occupied
    state = load(RUN/'state.json'); state.update(status='RUNNING', stage=name, stage_started_at=now())
    dump(RUN/'state.json', state)
    record = {'argv': argv, 'cwd': str(ROOT), 'env': {'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
              'started_at': now(), 'console': str(log)}
    dump(command_dir/(name+'.json'), record)
    started = time.monotonic()
    baseline = records(Path(load(EXP/'BASELINE.json')['root'])/'steps.jsonl') if training_root else None
    compared = 0
    with log.open('w') as out:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            env={**os.environ, **record['env']}, start_new_session=True)
        state['stage_pid'] = process.pid; dump(RUN/'state.json', state)
        try:
            while process.poll() is None:
                if training_root and (training_root/'steps.jsonl').exists():
                    lines = (training_root/'steps.jsonl').read_text().splitlines()
                    for line in lines[compared:]:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            break
                        stream_check(row, baseline[compared]); compared += 1
                        if compared in [1, 5, 100, 200, 500]:
                            print(json.dumps({'stage': name, 'matched_steps': compared, 'loss': row['loss']}), flush=True)
                time.sleep(2)
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            raise
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-started, finished_at=now())
    dump(command_dir/(name+'.json'), record)
    assert code == 0, f'{name} failed; see {log}'
    if training_root:
        rows = records(training_root/'steps.jsonl')
        for row, ref in zip(rows, baseline):
            stream_check(row, ref)
        dump(EXP/'evidence'/(name+'-stream-match.json'), {'passed': True, 'steps': len(rows),
             'four_ranks': True, 'sample_id_and_full_and_reference_hashes_equal': True,
             'local_views_expected_to_differ': True})
    print(json.dumps({'stage': name, 'exit_code': code, 'elapsed_seconds': record['elapsed_seconds']}), flush=True)


def train(kind):
    assert load(EXP/'evidence/preflight.json')['passed']
    c = load(EXP/'config.json')
    stop = 5 if kind == 'smoke' else 500
    if kind == 'formal':
        assert load(RUN/'smoke/acceptance.json')['passed']
        assert load(EXP/'evidence/smoke-5-stream-match.json')['passed']
    root = RUN/('smoke' if kind == 'smoke' else 'step500')
    argv = [TORCHRUN, '--standalone', '--nnodes=1', '--nproc-per-node=4', '--max-restarts=0',
        '-m', 'train.train_nested_semantic_mask', '--config', str(EXP/'config.json'),
        '--init-state', str(SHARED), '--index-dir', str(INDEX), '--image-root', str(ASSETS/'training/ShareGPT4V'),
        '--output-dir', str(root), '--run-type', kind, '--max-updates', str(stop)]
    execute(kind+'-'+str(stop), argv, gpu=True, training_root=root)
    acceptance = load(root/'acceptance.json'); assert acceptance['passed']
    assert all(r['completed_updates'] == stop and r['max_parameter_difference_from_rank0'] == 0 for r in acceptance['ranks'])
    actual = load(root/'config.json')
    assert actual['horizon'] == 4868 and actual['start_updates'] == 0 and actual['resume'] is None
    assert actual['init_sha256'] == INIT_SHA
    for f in ['config.json', 'acceptance.json', 'text_examples.json']:
        dump(EXP/'evidence'/(kind+'-'+str(stop)+'-'+f), load(root/f))
    cycles = records(root/'cycle_timing.jsonl')
    normal = [r['four_rank_max_seconds'] for r in cycles if not r['warmup']]
    if kind == 'formal':
        assert len(cycles) == 500 and len(normal) == 495 and statistics.fmean(normal) <= 3
        assert max(r['peak_allocated_gib'] for r in acceptance['ranks']) <= 65
    state = load(RUN/'state.json'); state.update(status='RUNNING', stage=kind+' completed',
        smoke_passed=True if kind=='smoke' else state.get('smoke_passed'),
        formal_updates=500 if kind=='formal' else 0)
    dump(RUN/'state.json', state)


def evaluate():
    root = RUN/'step500'; ckpt = root/'step000500.pt'; bare = root/'student_step500.pt'
    assert load(root/'acceptance.json')['passed']
    execute('export', [PYTHON, '-m', 'tools.nest_clip', 'export', '--checkpoint', str(ckpt),
        '--expect-updates', '500', '--output', str(bare)])
    execute('verify-export', [PYTHON, '-m', 'tools.nest_clip', 'verify-export', '--checkpoint', str(ckpt),
        '--bare', str(bare), '--output', str(root/'export-check.json'), '--index-dir', str(INDEX),
        '--image-root', str(ASSETS/'training/ShareGPT4V')])
    for name, path in [('coco', ASSETS/'evaluation/coco/val2017'), ('urban', ASSETS/'evaluation/Urban1k/Urban1k')]:
        execute('eval-'+name, [PYTHON, '-m', 'tools.eval_nest_native', '--checkpoint', str(bare),
            '--dataset', name, '--root', str(path), '--device', 'cuda:0', '--batch-size', '64',
            '--output', str(root/(name+'_native.json'))], gpu=True)
    bench = ASSETS/'retrieval_benchmarks'
    for name, manifest, folder in [('flickr_test1k','flickr30k_test1k.jsonl','flickr30k'),
                                  ('docci','docci_test.jsonl','docci'),
                                  ('long_dci','long_dci_reconstructed.jsonl','dci')]:
        execute('eval-'+name, [PYTHON, '-m', 'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
            '--checkpoint', str(bare), '--device', 'cuda:0', '--batch-size', '64', '--output-dir', str(root/name),
            f'{name}:{bench/"manifests"/manifest}:{bench/folder/"images"}'], gpu=True)
    m, raw, sources = native_metrics(root)
    result = {'status': 'EVALUATED', 'checkpoint': str(ckpt), 'checkpoint_sha256': file_sha(ckpt),
        'bare_student': str(bare), 'bare_sha256': file_sha(bare), 'metrics': m, 'scores': scores(m),
        'stop_updates': 500, 'horizon': 4868, 'strict_export': load(root/'export-check.json'),
        'baseline': load(EXP/'BASELINE.json'), 'config': load(root/'config.json')}
    dump(RUN/'result.json', result)
    for name, value in raw.items():
        dump(EXP/'raw'/f'{name}.json', value)
    dump(EXP/'evidence/strict-export.json', result['strict_export'])
    print(json.dumps({'scores_percent': {k: v*100 for k, v in result['scores'].items()}}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['preflight','smoke','formal','evaluate'])
    stage = p.parse_args().stage
    if stage == 'preflight':
        preflight()
    elif stage in ('smoke','formal'):
        train(stage)
    else:
        evaluate()


if __name__ == '__main__':
    main()
