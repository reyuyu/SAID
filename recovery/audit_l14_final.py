"""Read-only checkpoint, full stream, LR and resource audit of the single L14 run."""
import gc
import hashlib
import json
import math
from pathlib import Path
import torch
from tools.retrieval_bounded import sha, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-v1-l14-pure3epoch3651-v2')
EXP = ROOT / 'experiments/nest_clip_v1/hns_v1_l14_pure3epoch3651_v2'
INIT = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-v1-l14-pure3epoch3651-v1/shared/step000000.pt')


def rows(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN / 'formal3651/config.json').read_text())
    assert sha(INIT) == '8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912'
    checkpoints = []
    for step in (0, 1217, 2434, 3651):
        path = RUN / 'formal3651' / f'step{step:06d}.pt'
        before = sha(path)
        x = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        counters = sorted({int(s['step']) for s in x['optimizer']['state'].values()})
        check = dict(path=str(path), sha256=before, completed_steps=x['completed_steps'],
                     scheduler_horizon=x['scheduler_horizon'], next_epoch=x['next_epoch'],
                     next_batch=x['next_batch'], optimizer_steps=counters,
                     optimizer_state_count=len(x['optimizer']['state']),
                     optimizer_groups=[g['name'] for g in x['optimizer']['param_groups']],
                     rank_rng_keys=[sorted(s) for s in x['rng_per_rank']],
                     adapter_present=bool(x['adapter']), source_manifest=x['config']['code_sha256'])
        check['passed'] = (x['completed_steps'] == step and x['scheduler_horizon'] == 3651
                          and x['next_epoch'] == step // 1217 and x['next_batch'] == step % 1217
                          and counters == ([step] if step else []) and len(x['rng_per_rank']) == 4
                          and all(set(('cpu','cuda','python','numpy','loader_generator')) <= set(s)
                                  for s in x['rng_per_rank'])
                          and len(x['optimizer']['param_groups']) == 4 and bool(x['adapter']))
        check['source_manifest_matches_current'] = all(sha(ROOT / p) == h for p, h in check['source_manifest'].items())
        check['sha256_after'] = sha(path)
        check['immutable'] = before == check['sha256_after']
        checkpoints.append(check)
        del x
        gc.collect()
    atomic_json(EXP / 'CHECKPOINT_AUDIT.json', {'checkpoints': checkpoints,
        'step500_checkpoint': 'NOT_SAVED: inherited interval1217; cannot retrospectively reconstruct without retraining',
        'passed_saved_checkpoints': all(c['passed'] and c['immutable'] and c['source_manifest_matches_current'] for c in checkpoints)}, exclusive=True)
    r = rows(RUN / 'formal3651/steps.jsonl')
    ref = rows('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-v1-pure3epoch3651-v1/step3651/steps.jsonl')
    assert len(r) == len(ref) == 3651
    mismatches = []
    positions = 0
    aggregate = hashlib.sha256()
    for a, b in zip(r, ref):
        assert a['step'] == b['step']
        expected_epoch = (a['step'] - 1) // 1217
        if a['epoch'] != expected_epoch:
            mismatches.append([a['step'], 'epoch'])
        for h, other in zip(a['rank_health'], b['rank_health']):
            positions += h['batch']
            expected_batch = 180 if a['step'] % 1217 == 0 else 256
            # The L14 v2 recorder adds lowest_selected_sentence_indices_sha256;
            # compare the frozen common sampling schema and retain the additive
            # field in the receipt rather than treating telemetry enrichment as
            # data-stream drift.
            # Compare the intersection of the two schemas.  The original
            # reference has one optional telemetry digest that the v2 recorder
            # does not emit; treating that additive field as a data mismatch
            # would invalidate an otherwise identical stream.  All shared
            # fields remain compared exactly, including the complete nested
            # sampling summaries.
            common_keys = sorted(set(h['sampling']) & set(other['sampling']))
            common_sampling = {k: h['sampling'][k] for k in common_keys}
            other_common = {k: other['sampling'][k] for k in common_keys}
            if h['batch'] != expected_batch or h['stream_sha256'] != other['stream_sha256'] or digest(common_sampling) != digest(other_common):
                mismatches.append([a['step'], h['rank'], 'stream/sampling/batch'])
            aggregate.update(h['stream_sha256'].encode())
        s = a['step'] - 1
        base = 1e-6*(s+1)/200 if s < 200 else .5e-6*(1+math.cos(math.pi*(s-200)/(3651-200)))
        factor = .5*(1+math.cos(math.pi*s/3651))
        lr = dict(backbone=base, text_mask_and_shared_pool=1e-3*factor, visual_mask=1e-3*factor, fusion_adapter=2e-4*factor)
        if a['actual_lrs'] != lr:
            mismatches.append([a['step'], 'LR', a['actual_lrs'], lr])
    proof = dict(updates=3651, contiguous_steps=[a['step'] for a in r] == list(range(1,3652)),
                 sample_positions=positions, expected_sample_positions=1245904*3,
                 reference='HNS-v1 B16 Pure3 real logged stream; model-related outputs excluded',
                 full_rank_stream_sha256=aggregate.hexdigest(), mismatches=mismatches,
                 evidence='sampleIDs/F/Dall/D3 text/token hashes plus entire K/detail/token sampling dictionary on every update/rank',
                 passed=not mismatches and positions == 1245904*3)
    atomic_json(EXP / 'DATA_STREAM_AND_LR_AUDIT.json', proof, exclusive=True)
    timing = rows(RUN / 'formal3651/cycle_timing.jsonl')
    vals = torch.tensor([x['four_rank_max_seconds'] for x in timing], dtype=torch.float64)
    phases = rows(RUN / 'formal3651/phase_timing.jsonl')
    gate = json.loads((RUN / 'resource_probe/acceptance.json').read_text())
    smoke = json.loads((RUN / 'smoke5/acceptance.json').read_text())
    final = json.loads((RUN / 'formal3651/acceptance.json').read_text())
    resource = dict(gate=gate, smoke=smoke, formal=final,
        cycle_mean=float(vals.mean()), cycle_p95=float(torch.quantile(vals,.95)), cycle_max=float(vals.max()),
        phase_means={k:sum(x[k] for x in phases)/len(phases) for k in ('data_wait_s','update_compute_s','post_update_logging_s','barrier_sync_s','monitor_reduce_s')},
        timing_limit_scope='5+30 resource gate; formal inherited three-consecutive>6s stop rule',
        numerical_finite=all(a.get('nonfinite',0)==0 and all(h['gradients_finite'] for h in a['rank_health']) for a in r),
        last50={k:sum(a[k] for a in r[-50:] if isinstance(a.get(k),(float,int))) /
                sum(1 for a in r[-50:] if isinstance(a.get(k),(float,int)))
                for k,v in r[-1].items() if isinstance(v,(float,int)) and
                all(isinstance(a.get(k),(float,int)) for a in r[-50:])},
        limitations=['update_compute_s includes transfer, GPU forward/backward/optimizer, DDP and stream serialization; no CUDA-event or NCCL-profiler decomposition',
                     'v1 step9 has no phase timers; its exact GPU/DataLoader/NCCL/monitor attribution is INCONCLUSIVE; v2 timings do not retrospectively prove v1 cause'])
    atomic_json(EXP / 'RESOURCE_AND_TRAINING_AUDIT.json', resource, exclusive=True)
    assert all(c['passed'] and c['immutable'] and c['source_manifest_matches_current'] for c in checkpoints)
    assert proof['passed'] and gate['passed'] and smoke['passed'] and final['passed'] and resource['numerical_finite']
    print(json.dumps({'checkpoint_audit':True,'stream_lr_audit':proof['passed'],'positions':positions,'resources':True}))


if __name__ == '__main__':
    main()
