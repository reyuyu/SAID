"""Read-only verification and aggregation of reviewed NEST-CLIP formal outputs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

ROOT = Path('/root/lk_projects/SAID-nest-clip-v1')
EVIDENCE = ROOT / 'formal500-evidence'
REPO = Path('/root/lk_projects/SAID')
REVISION = 'a5742401395f381642bb5ad88ba2d952128a436f'
EXPECTED_INIT = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
MILESTONES = (1, 100, 200, 201, 300, 400, 500)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def read_rows(arm):
    rows = [json.loads(line) for line in (ROOT / f'formal/{arm}/steps.jsonl').read_text().splitlines()]
    for row in rows:
        row['common_loss'] = row['loss'] - row['inc_weight'] * row['inc']
        for view in ('F', 'O', 'E'):
            row[f'{view}_align'] = (row.get(f'{view}_i2t', 0.) + row.get(f'{view}_t2i', 0.))
    return rows


def audit_arm(arm):
    execution = read(EVIDENCE / f'train500-{arm}.execution.json')
    assert execution['exit_code'] == 0 and execution['git_head'] == REVISION
    config = read(ROOT / f'formal/{arm}/config.json')
    acceptance = read(ROOT / f'formal/{arm}/acceptance.json')
    assert config['git_head'] == REVISION
    assert config['init_sha256'] == EXPECTED_INIT and config['resume'] is None
    assert config['horizon'] == 3651 and config['max_updates'] == 500
    assert config['batch_size'] == 256 and config['world_size'] == 4 and config['accumulation'] == 1
    assert config['seed'] == 0 and config['epochs'] == 3 and config['workers'] == 8
    assert config['training_records'] == 1245901 and config['batches_per_epoch'] == 1217
    assert config['data'] == read(EVIDENCE / 'preflight.json')['data']
    assert len({x['uuid'] for x in config['ranks']}) == 4
    assert acceptance['passed'] and len(acceptance['ranks']) == 4
    for rank in acceptance['ranks']:
        assert rank['completed_updates'] == rank['updates_this_run'] == 500
        assert rank['max_parameter_difference_from_rank0'] == 0
    for path, expected in config['code_sha256'].items():
        assert sha(REPO / path) == expected, path
    rows = read_rows(arm)
    assert [r['step'] for r in rows] == list(range(1, 501))
    assert [r['s'] for r in rows] == list(range(500))
    reasons = Counter()
    duplicate_ids = set()
    duplicate_steps = []
    for row in rows:
        assert row['nonfinite'] == 0
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        assert [x['rank'] for x in row['rank_health']] == list(range(4))
        for rank in row['rank_health']:
            assert rank['batch'] == 256 and rank['updates'] == row['step']
            assert rank['gradients_finite'] and all(math.isfinite(v) for v in rank['gradient_norms'].values())
            reasons.update(rank['reasons'])
        assert sum(x['valid'] for x in row['rank_health']) == row['valid_global']
        assert row['F_candidates'] == 1024
        assert row['O_candidates'] == row['E_candidates'] == (row['valid_global'] if row['valid_global'] >= 2 else 0)
        expected_inc = min(1., row['s']/200.) if arm == 'A3' and row['valid_global'] >= 2 else 0.
        assert row['inc_weight'] == expected_inc
        s = row['s']
        expected_backbone = 1e-6*(s+1)/200 if s < 200 else .5e-6*(1+math.cos(math.pi*(s-200)/(3651-200)))
        assert math.isclose(row['lr_backbone'], expected_backbone, rel_tol=1e-12)
        assert math.isclose(row['lr_mask'], .5e-3*(1+math.cos(math.pi*s/3651)), rel_tol=1e-12)
        if row['duplicate_image_ids']:
            duplicate_steps.append(row['step'])
            duplicate_ids.update(row['duplicate_image_ids'])
    assert sum(reasons.values()) == 512000
    checkpoints = []
    assert sorted(p.name for p in (ROOT / f'formal/{arm}').glob('step*.pt')) == [f'step{s:06d}.pt' for s in (0,100,200,300,400,500)]
    for step in (0, 100, 200, 300, 400, 500):
        path = ROOT / f'formal/{arm}/step{step:06d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['completed_steps'] == step
        assert payload['scheduler_horizon'] == 3651 and payload['stop_updates'] == 500
        assert payload['config']['init_sha256'] == EXPECTED_INIT
        assert len(payload['rng_per_rank']) == 4
        assert all(int(state['step']) == step for state in payload['optimizer']['state'].values())
        if step == 0:
            initial = torch.load(ROOT / 'shared/step000000.pt', map_location='cpu', weights_only=False)
            assert payload['model'].keys() == initial['model'].keys()
            assert all(torch.equal(payload['model'][k], v) for k, v in initial['model'].items())
            assert payload['optimizer'] == initial['optimizer']
            del initial
        del payload
        checkpoints.append(dict(step=step, server_path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    result = dict(passed=True, arm=arm, execution=execution, config=config, acceptance=acceptance,
                  continuous_steps=True, losses_and_gradients_finite=True, common_step0_verified=True,
                  checkpoint_metadata_verified=True, checkpoints=checkpoints, samples_seen=512000,
                  view_reason_counts=dict(reasons), valid_local_ratio=reasons['valid']/512000,
                  f_only_steps=sum(r['valid_global'] < 2 for r in rows),
                  global_valid_min=min(r['valid_global'] for r in rows),
                  global_valid_max=max(r['valid_global'] for r in rows),
                  duplicate_steps=duplicate_steps, duplicate_ids=sorted(duplicate_ids),
                  sum_max_step_seconds=sum(max(x['seconds'] for x in r['rank_health']) for r in rows))
    write(EVIDENCE / f'{arm}-training-audit.json', result)
    print(json.dumps(dict(arm=arm, passed=True, last_step=500, last_loss=rows[-1]['loss'],
                          checkpoints=len(checkpoints), reasons=dict(reasons))), flush=True)


def metrics(row):
    keys = ['loss','common_loss','inc','inc_weight','hard_inclusion_violation','oe_iou','valid_global',
            'lr_backbone','lr_mask']
    keys += [f'{v}_{k}' for v in ('F','O','E') for k in ('i2t','t2i','align','keep_ratio','all_open','all_closed','candidates')]
    return {k:row.get(k, 0.) for k in keys}


def audit_pair():
    audits = {arm:read(EVIDENCE / f'{arm}-training-audit.json') for arm in ('A2','A3')}
    rows = {arm:read_rows(arm) for arm in audits}
    differences = [k for k in audits['A2']['config'] if audits['A2']['config'][k] != audits['A3']['config'][k]]
    assert set(differences) <= {'arm','config','output_dir','ranks'}
    for a,b in zip(rows['A2'], rows['A3']):
        assert a['step'] == b['step']
        assert [x['stream_sha256'] for x in a['rank_health']] == [x['stream_sha256'] for x in b['rank_health']]
    initials = [torch.load(ROOT/f'formal/{arm}/step000000.pt', map_location='cpu', weights_only=False) for arm in ('A2','A3')]
    for a,b in zip(initials[0]['rng_per_rank'], initials[1]['rng_per_rank']):
        assert torch.equal(a['cpu'],b['cpu']) and torch.equal(a['cuda'],b['cuda'])
        assert a['python'] == b['python']
        assert a['numpy'][0] == b['numpy'][0] and (a['numpy'][1] == b['numpy'][1]).all() and a['numpy'][2:] == b['numpy'][2:]
    result = dict(passed=True, compared_steps=500, compared_rank_streams=2000,
                  streams_identical=True, step0_rngs_identical=True,
                  config_difference_keys=differences, substantive_config_difference=['arm'])
    write(EVIDENCE/'paired-training-audit.json',result)
    print(json.dumps(result), flush=True)


def aggregate():
    assert read(EVIDENCE/'paired-training-audit.json')['passed']
    audits = {arm:read(EVIDENCE / f'{arm}-training-audit.json') for arm in ('A2','A3')}
    rows = {arm:read_rows(arm) for arm in audits}
    differences = [k for k in audits['A2']['config'] if audits['A2']['config'][k] != audits['A3']['config'][k]]
    assert set(differences) <= {'arm','config','output_dir','ranks'}
    for a,b in zip(rows['A2'], rows['A3']):
        assert a['step'] == b['step']
        assert [x['stream_sha256'] for x in a['rank_health']] == [x['stream_sha256'] for x in b['rank_health']]
    init_a = torch.load(ROOT/'formal/A2/step000000.pt', map_location='cpu', weights_only=False)
    init_b = torch.load(ROOT/'formal/A3/step000000.pt', map_location='cpu', weights_only=False)
    for a,b in zip(init_a['rng_per_rank'], init_b['rng_per_rank']):
        assert torch.equal(a['cpu'],b['cpu']) and torch.equal(a['cuda'],b['cuda'])
        assert a['python'] == b['python']
        assert a['numpy'][0] == b['numpy'][0] and (a['numpy'][1] == b['numpy'][1]).all() and a['numpy'][2:] == b['numpy'][2:]
    del init_a, init_b
    mechanism = {}
    evaluations = {}
    executions = {}
    for arm in audits:
        mechanism[arm] = dict(milestones={str(s):metrics(rows[arm][s-1]) for s in MILESTONES},
                              last50_mean={k:statistics.fmean(metrics(r)[k] for r in rows[arm][-50:]) for k in metrics(rows[arm][-1])})
        check = read(EVIDENCE / f'{arm}-export-check.json')
        assert check['passed'] and check['strict_load'] and check['optimizer_steps'] == [500]
        assert check['image_max_abs'] == check['text_max_abs'] == 0
        audits[arm]['export_verification'] = check
        evaluations[arm] = {}
        for dataset in ('coco','urban'):
            evaluations[arm][dataset] = read(ROOT / f'formal/{arm}/{dataset}_native.json')
            assert evaluations[arm][dataset]['checkpoint_sha256'] == check['bare_sha256']
        for name in (f'train500-{arm}', f'export500-{arm}', f'verify-export-{arm}', f'coco-{arm}', f'urban-{arm}'):
            execution = read(EVIDENCE / (name+'.execution.json'))
            assert execution['exit_code'] == 0 and execution['git_head'] == REVISION
            executions[name] = execution
    comparison = []
    for dataset in ('coco', 'urban'):
        for direction,key in (('I2T','image2text'),('T2I','text2image')):
            for k in (1,5,10):
                values = {arm:(evaluations[arm][dataset][f'{key}_R{k}'] if dataset == 'coco'
                               else evaluations[arm][dataset][key][f'R{k}']) for arm in audits}
                assert all(0 <= x <= 1 for x in values.values())
                comparison.append(dict(dataset=dataset, direction=direction, metric=f'R@{k}',
                                       A2=100*values['A2'], A3=100*values['A3'],
                                       delta_pp=100*(values['A3']-values['A2']), raw_recall=values))
    result = dict(reviewed_revision=REVISION, training_logic_unchanged=True, seed=0, updates=500,
                  evaluation_protocol=dict(checkpoint_step=500, device='cuda:0', image_batch_size=64,
                      context_length=248, coco_images=5000, coco_texts=25000, coco_similarity_chunk=512,
                      urban_images=1000, urban_texts=1000, text_encoding='unchanged reviewed evaluator',
                      scoring='normalized native image/text embeddings, dot product',
                      mask=False, fusion=False, rerank=False, raw_recall_range=[0,1],
                      displayed_recall='100 * raw', difference='100 * (A3 - A2), percentage points'),
                  preflight=read(EVIDENCE/'preflight.json'), audits=audits, executions=executions,
                  streams_match_all_500_steps_4_ranks=True, step0_rngs_match=True,
                  configuration_difference_keys=differences, raw_evaluations=evaluations,
                  recall_comparison_percent=comparison, mechanism=mechanism,
                  common_loss_definition='total_loss - inc_weight * inc',
                  limits=['single seed=0; 500 updates; no significance claim',
                          'training-batch mechanism logs only; no additional mechanism evaluation',
                          'step500 only; no checkpoint selection'])
    write(REPO/'experiments/nest_clip_v1/formal500_results.json', result)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit-arm','audit-pair','aggregate'])
    parser.add_argument('--arm', choices=['A2','A3'])
    args = parser.parse_args()
    if args.action == 'audit-arm':
        audit_arm(args.arm)
    elif args.action == 'audit-pair':
        audit_pair()
    else:
        aggregate()
