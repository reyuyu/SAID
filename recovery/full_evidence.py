"""Review full continuation locally; publish only bounded diagnostic summaries."""
import json
import math
from pathlib import Path
import re
import subprocess

from recovery.s02_local_full import ROOT, OUT, RUN, REVIEW, PHASE, FINAL, PARENT, PARENT_SHA, IMAGES, INDEX, now, dump, rows, sha, distribution

RANDOMK = dict(Score5=72.768147,J_long3=77.070244,J_long=85.485003,Short4=66.315)
RANDOMK_R1 = {'COCO':(61.70,42.22),'Urban-1k':(92.10,91.10),'Flickr30k-test1k':(88.90,72.44),
    'DOCCI':(78.76,79.98),'Long-DCI':(59.668508,60.812944)}


def ordered_samples(steps):
    import torch
    from torch.utils.data import DistributedSampler
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(INDEX,IMAGES,'summary_random_detail',0)
    count=0
    for epoch in range(4):
        records=[row for row in steps if row['epoch']==epoch]
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
            sampler.set_epoch(epoch)
            indices=list(sampler)
            for row in records:
                batch=(row['step']-1)%1217
                expected=[i+1000 for i in indices[batch*256:(batch+1)*256]]
                health=next(h for h in row['rank_health'] if h['rank']==rank)
                assert health['sampling']['sample_ids']==expected, f'Frozen sample drift step{row["step"]} rank{rank}'
                count+=len(expected)
    return dict(passed=True,records_checked=count,first_update=501,last_update=steps[-1]['step'] if steps else None,
        frozen_sampler='DistributedSampler seed0,epoch0..3,batch256/rank,world4; original ordered index')


def report():
    saved=json.loads((RUN/'full-supervisor-result.json').read_text())
    result=saved['result'] or dict(status='INCOMPLETE_HARD_STOP',error=saved['error'])
    steps=rows(FINAL/'steps.jsonl')
    cycles=rows(FINAL/'cycle_timing.jsonl')
    phases={rank:rows(PHASE/f'rank{rank}.jsonl') for rank in range(4)}
    lookup={r:{p['step']:p for p in records} for r,records in phases.items()}
    telemetry=[s for s in rows(RUN/'full-resource-telemetry.jsonl') if s['command']=='train4868']
    systems=[s['system'] for s in telemetry]+[p['system_after'] for records in phases.values() for p in records]
    waits=[max(lookup[r].get(c['step'],{}).get('data_wait_s',0) for r in range(4)) for c in cycles]
    text=(RUN/'train4868.log').read_text(errors='replace')
    errors=sorted(set(re.findall(r'Image failure sample=\d+[^\n]*|Input/output error[^\n]*|Missing local sample=\d+[^\n]*',text)))
    path_proofs=[r for p in PHASE.glob('image-paths-*.jsonl') for r in rows(p)]
    path_valid=bool(path_proofs) and all(Path(p['actual_path']).is_relative_to(IMAGES) and p['NFS_fallback'] is False for p in path_proofs)
    stats=dict(scope='Continuation updates501..4868; replay and export/evaluation excluded from cycle distributions',
        started_utc=saved['started_utc'],finished_utc=saved['ended_utc'],completed_updates=len(cycles),
        full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        data_wait_seconds_slowest_rank=distribution(waits),
        rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in records]) for r,records in phases.items()},
        steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        peak_cgroup_memory_bytes=max((s['memory_current'] for s in systems),default=0),
        peak_file_cache_bytes=max((s['file'] for s in systems),default=0),
        oom_kill=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0),
        GPU_peak_memory={str(r):dict(allocated_GiB=max((h['peak_allocated_gib'] for s in steps for h in s['rank_health'] if h['rank']==r),default=0),
            reserved_GiB=max((h['peak_reserved_gib'] for s in steps for h in s['rank_health'] if h['rank']==r),default=0)) for r in range(4)},
        io_error_count=len(errors),io_errors=errors,stop_reason=saved['error'],
        pod_supervisor_anomaly_observed=saved['error'] is not None,
        PSI_scope='Host /proc/pressure on cgroupv1, not per-cgroup PSI',
        local_runtime_path_proof=dict(passed=path_valid,count=len(path_proofs),ranks=sorted({p['rank'] for p in path_proofs}),NFS_fallback=False),
        checkpoint_timing=rows(FINAL/'checkpoint_timing.jsonl'))
    for kind in ('io_PSI','memory_PSI'):
        for pressure in ('some','full'):
            stats[f'{kind}_{pressure}_avg10']=distribution([s[kind][pressure]['avg10'] for s in systems if pressure in s[kind]])
    stats['GPU_utilization_percent']={str(r):distribution([g['gpu_percent'] for s in telemetry for g in s['gpu']
        if isinstance(g,dict) and g.get('index')==r and g.get('gpu_percent') is not None]) for r in range(4)}
    first500=rows(RUN/'step500/cycle_timing.jsonl')
    stats['whole_trajectory_full_cycle_seconds']=distribution([c['four_rank_max_seconds'] for c in first500+cycles])
    stats['whole_trajectory_steps_gt3s']=sum(c['four_rank_max_seconds']>3 for c in first500+cycles)
    stats['whole_trajectory_steps_gt10s']=sum(c['four_rank_max_seconds']>10 for c in first500+cycles)
    inventories=list(RUN.glob('*4868*.log'))+[RUN/'full-resource-telemetry.jsonl']+list(FINAL.glob('*.jsonl'))+list(PHASE.glob('*.jsonl'))+[
        REVIEW/'resume-stream-reference.json',REVIEW/'prelaunch-local-path-proof-5000.json']
    stats['raw_local_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),time_range_utc=[saved['started_utc'],saved['ended_utc']],uploaded=False) for p in inventories if p.exists()]
    provenance=json.loads((OUT/'RESUME_PROVENANCE.json').read_text())
    provenance.update(launch=json.loads((RUN/'full-launch-provenance.json').read_text()),
        restoration_gate=json.loads((REVIEW/'STEP500_RESUME_AUDIT.json').read_text()) if (REVIEW/'STEP500_RESUME_AUDIT.json').exists() else None,
        stream_gates=[json.loads(p.read_text()) for p in sorted(REVIEW.glob('resume-batch-*.json'))],
        LR_gates=[json.loads(p.read_text()) for p in sorted(REVIEW.glob('resume-lr-*.json'))])
    complete=bool(saved['result']) and saved['error'] is None
    if complete:
        assert [s['step'] for s in steps]==list(range(501,4869))
        assert [c['step'] for c in cycles]==list(range(501,4869))
        assert all(len(phases[r])==4368 for r in range(4))
        assert len(provenance['stream_gates'])==len(provenance['LR_gates'])==20
        assert provenance['restoration_gate']['passed'] and path_valid and not errors and stats['oom_kill']==0
        assert all(math.isfinite(s['loss']) and all(h['gradients_finite'] and h['batch']==256 for h in s['rank_health']) for s in steps)
        provenance['full_frozen_sample_stream']=ordered_samples(steps)
        assert sha(PARENT)==PARENT_SHA, 'Parent checkpoint modified'
        assert all(s['actual_lrs']==dict(zip([g['name'] for g in provenance['optimizer_groups']],
            __import__('recovery.s02_local_full',fromlist=['expected_lrs']).expected_lrs(s['step']-1,json.loads((OUT/'configs/summary02_local500.json').read_text())))) for s in steps)
        result['delta_vs_RandomK_pp']={k:result['scores_percent'][k]-value for k,value in RANDOMK.items()}
        result['dataset_R1_delta_vs_RandomK_pp']={name:{d:result['metrics'][name][d]['R@1']*100-baseline[j] for j,d in enumerate(('I2T','T2I'))} for name,baseline in RANDOMK_R1.items()}
    result.update(completed_4868=complete,completed_updates_this_continuation=len(cycles),
        acceptance=saved['acceptance'],final_checkpoint_proof=saved['final_checkpoint_proof'],
        local_image_root=str(IMAGES),NFS_fallback=False,runtime=str(RUN),checkpoint_uploaded=False,
        exact_stop_updates=4868,automatic_new_experiments=False,github_sync_complete=False)
    slow=[]
    for c in cycles:
        if c['four_rank_max_seconds']<=3:
            continue
        ranks=[]
        for r in range(4):
            p=lookup[r].get(c['step'],{})
            s=p.get('system_after',{})
            ranks.append(dict(rank=r,**{k:v for k,v in p.items() if k in ('data_wait_s','forward_s','backward_s','optimizer_s','ddp_sync_s')},
                memory_current=s.get('memory_current'),file_cache=s.get('file'),
                IO_PSI_full_avg10=s.get('io_PSI',{}).get('full',{}).get('avg10'),memory_PSI_full_avg10=s.get('memory_PSI',{}).get('full',{}).get('avg10')))
        slow.append(dict(step=c['step'],full_cycle_s=c['four_rank_max_seconds'],ranks=ranks))
    numeric_keys=[key for key in steps[0] if isinstance(steps[0][key],(float,int)) and key not in ('step','s','epoch')] if steps else []
    diagnostics=dict(slow_steps=slow,epoch_scalar_metrics={str(e):{k:distribution([s[k] for s in steps if s['epoch']==e and isinstance(s.get(k),(float,int))]) for k in numeric_keys} for e in range(4)},
        all_step_raw_metrics=str(FINAL/'steps.jsonl'),scope='F/S/D CE, alignment,sparsity,inclusion,keep ratio,gate stats,LR recorded natively every update; summarized by epoch')
    dump(OUT/'FULL_DIAGNOSTICS.json',diagnostics)
    dump(OUT/'FULL_RUNTIME_STATS.json',stats)
    dump(OUT/'RESUME_PROVENANCE.json',provenance)
    dump(OUT/'FULL_RESULTS.json',result)
    lines=['# S=0.2 full trajectory continuation','',f'Status: `{result["status"]}`; completed4868: `{complete}`.',
        f'Resumed exact evaluated checkpoint500 SHA256 `{PARENT_SHA}`; continuation501..4868, horizon4868.',
        'Full model/adapter/AdamW, per-rank Python/NumPy/CPU/CUDA RNG and loader generator restored. Native replay of consumed batches skips optimizer updates; step501..505 ordered IDs/FSD/token/LR gates pass.',
        'Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch,4 A10080GB,256/rank,global1024,accum1,seed0; native optimizer/sparsity/inclusion/ramp/preprocess unchanged.',
        f'Local-only training root `{IMAGES}`; missing/escaped/symlink paths fail-fast, no NFS fallback. No copy or full audit launched.',
        'Cache is ephemeral Docker overlay; persistent NFS original images remain source of truth.',
        '',f'Continuation full-cycle seconds: `{json.dumps(stats["full_cycle_seconds"])}`.',
        f'Slowest-rank data_wait seconds: `{json.dumps(stats["data_wait_seconds_slowest_rank"])}`.',
        f'>3s: {stats["steps_gt3s"]}; >10s: {stats["steps_gt10s"]}. I/O errors: {len(errors)}; oom_kill: {stats["oom_kill"]}.',
        f'Peak cgroup: {stats["peak_cgroup_memory_bytes"]/2**30:.3f}GiB; peak file cache: {stats["peak_file_cache_bytes"]/2**30:.3f}GiB. GPU/rank and PSI distributions: FULL_RUNTIME_STATS.json.',
        'Ordinary>3s warnings continue; hard stops only actual image/CUDA/DDP/finite-state/OOM failure, active step>60s or supervisor anomaly.',
        'Checkpoints saved persistently at1217/2434/3651/4868, runtime checkpoint timing retained.',
        '', '|Dataset|I2T R@1/5/10 (%)|T2I R@1/5/10 (%)|R1 delta vs RandomK pp|','|---|---|---|---|']
    for name,m in result.get('metrics',{}).items():
        values=[' / '.join(f'{m[d][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        delta=result['dataset_R1_delta_vs_RandomK_pp'][name]
        lines.append(f'|{name}|{values[0]}|{values[1]}|{delta["I2T"]:+.6f} / {delta["T2I"]:+.6f}|')
    lines+=['',f'Scores (%): `{json.dumps(result.get("scores_percent"))}`.',
        f'RandomK deltas (pp): `{json.dumps(result.get("delta_vs_RandomK_pp"))}`.',
        'Classification fixed before evaluation: Score5>72.768147 and J_long3>=76.870244 => FULL_POSITIVE; improved Score5 with failed long guard => SHORT_LONG_TRADEOFF; otherwise EARLY_SIGNAL_DID_NOT_SCALE.',
        'Strict bare native evaluation: normalized native image embedding @ normalized native full-caption text embedding.T; no mask/gate/rerank/ensemble/Summary/Detail inference.',
        f'Final complete checkpoint `{FINAL/"step004868.pt"}` SHA256 `{result.get("checkpoint_sha256")}`.',
        f'Bare student `{FINAL/"student_step4868.pt"}` SHA256 `{result.get("strict_export",{}).get("bare_sha256")}`.',
        'Raw logs path/size/SHA/UTC inventory: FULL_RUNTIME_STATS.json; raw logs, checkpoints, mirror and full hash ledgers stay local.',
        'CPU checks:23 tests passed before continuation launch. GitHub synchronization receipt follows. No automatic further training or experiments.',
        f'Error: {saved["error"]}']
    (OUT/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],completed_4868=complete,scores=result.get('scores_percent'),runtime=stats['full_cycle_seconds'])))


if __name__=='__main__':
    report()
