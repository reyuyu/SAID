"""Independent full-run evidence checks; small reports, all large assets local."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from recovery.nested_detail_d3_balanced_full import (
    ROOT,OUT,EXP,RUN,PHASE,TRAIN,CONFIG,REFERENCE_EXP,REFERENCE_RUN,INDEX,IMAGES,BASELINE,DEBIAS,
    STEP0_SHA,dump,rows,sha,now,frozen_config)
from recovery.s02_nfs500 import distribution

SCORE_KEYS=('Score5','J_long3','J_long','Short4')
MASK_KEYS=('inc','inc_weight','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation')


def decision(scores,urban,baseline):
    old=baseline['scores_percent'];bm=baseline['metrics']['Urban-1k']
    delta={k:scores[k]-old[k] for k in SCORE_KEYS}
    du={d:urban[d]-round(100*bm[d]['R@1'],3) for d in ('I2T','T2I')}
    strong=delta['Score5']>1e-6 and delta['J_long3']>1e-6 and du['T2I']>1e-6 and delta['Short4']>=-.2-1e-6
    positive=(delta['Score5']>=.1-1e-6 or delta['J_long3']>=.1-1e-6) and min(du.values())>=-.2-1e-6 and min(delta.values())>=-.2-1e-6
    tradeoff=(delta['J_long3']>1e-6 or max(du.values())>1e-6) and (delta['Score5']<-.2-1e-6 or delta['Short4']<-.2-1e-6)
    negative=delta['Score5']< -1e-6 and delta['J_long3']<=1e-6 and max(du.values())<=1e-6
    status=('FULL_STRONG_POSITIVE' if strong else 'FULL_POSITIVE' if positive else 'FULL_TRADEOFF' if tradeoff else
            'FULL_NEGATIVE' if negative else 'FULL_INCONCLUSIVE')
    return dict(status=status,strong=strong,positive=positive,tradeoff=tradeoff,negative=negative,
        scores_delta_pp=delta,Urban_delta_pp=du,
        declared_before_training=True,clear_gain_pp=.1,material_decline_pp=.2,
        automatic_new_experiments=False,automatic_continuation=False)


def checkpoint_proof(payload,step):
    import torch
    assert payload['completed_steps']==payload['global_step']==payload['scheduler']['completed_steps']==step
    assert payload['scheduler_horizon']==payload['scheduler']['horizon']==4868
    assert payload['data_cursor']==dict(next_epoch=step//1217,next_batch=step%1217)
    assert payload['trajectory_root']==str(RUN) and len(payload['rng_per_rank'])==4
    c=payload['config'];frozen_config(c)
    assert c['resume'] is None and c['start_updates']==0 and c['init_sha256']==STEP0_SHA
    assert c['max_updates']==4868
    for rng in payload['rng_per_rank']:
        assert all(k in rng for k in ('python','numpy','cpu','cuda','loader_generator'))
        assert all(torch.is_tensor(rng[k]) and rng[k].numel()>0 for k in ('cpu','cuda','loader_generator'))
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={step}
    assert payload['adapter'] is not None and payload['model'] and payload['optimizer']['state']
    assert all(torch.isfinite(v).all() for state in (payload['model'],payload['adapter']) for v in state.values())
    assert all(torch.isfinite(v).all() for s in payload['optimizer']['state'].values() for v in s.values())
    return dict(passed=True,completed_steps=step,global_step=step,horizon=4868,cursor=payload['data_cursor'],
        optimizer_counters=[step],model_adapter_optimizer=True,rng_ranks=4,loader_rng_present=True,
        fresh_common0=True,uploaded=False)


def verify_stream(steps):
    """All IDs and actual selected-index digests; sampled strings/tokens separately audited."""
    import torch
    from torch.utils.data import DistributedSampler
    from recovery.s02_full_local_data import FullLocalDataset
    from train.nested_semantic_data import sample_partial_detail_indices
    from recovery.nested_detail_d3_balanced500 import digest_indices,matched_rows
    from recovery.s02_local_full import expected_lrs
    assert [r['step'] for r in steps]==list(range(1,4869))
    before=torch.get_rng_state().clone()
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    cfg=json.loads(CONFIG.read_text())
    count=0;strict=0;degenerate=0;tail=0
    for epoch in range(4):
        epoch_rows=steps[epoch*1217:(epoch+1)*1217]
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
            sampler.set_epoch(epoch);indices=list(sampler)
            for batch,row in enumerate(epoch_rows):
                assert row['epoch']==epoch and row['s']==row['step']-1
                assert math.isfinite(row['loss']) and row['nonfinite']==0
                expected=[i+1000 for i in indices[batch*256:(batch+1)*256]]
                h=next(h for h in row['rank_health'] if h['rank']==rank);s=h['sampling']
                assert h['batch']==len(expected)==(180 if batch==1216 else 256)
                assert h['updates']==row['step'] and h['gradients_finite'] and s['sample_ids']==expected
                assert s['nested_d3_exact']
                choices=[sample_partial_detail_indices(n,0,epoch,sid) for n,sid in zip(s['n'],expected)]
                assert s['D3_selected_sentence_indices_sha256']==digest_indices(expected,choices)
                assert list(row['actual_lrs'].values())==expected_lrs(row['step']-1,cfg)
                count+=len(expected)
                strict+=sum(n>2 for n in s['n']);degenerate+=sum(n<=2 for n in s['n'])
                tail+=int(batch==1216)
    matched=matched_rows(steps[:500],rows(REFERENCE_RUN/'step500/steps.jsonl'))
    assert matched==512000 and count==4*(1245901+3) and torch.equal(before,torch.get_rng_state())
    return dict(passed=True,all_steps=4868,records_checked=count,epochs=[0,1,2,3],
        actual_D3_selected_index_digests_verified=True,all_LR_exact=True,all_gradients_finite=True,
        first500_records_exact_reference=matched,first500_F_Dall_D3_strings_tokens_indices_exact=True,
        strict_subset_samples=strict,degenerate_samples=degenerate,strict_subset_ratio=strict/count,
        epoch_tail_rank_batches=tail,epoch_tail_batch_per_rank=180,global_RNG_untouched=True,
        source='Frozen DistributedSampler, immutable original D3 sampler, actual rank-health n/IDs/epoch')


def view_means(records):
    result={}
    for view,prefix,weight in [('F','F',1.35),('Dall','O',1.35),('D3','E',.3)]:
        means={k:statistics.fmean(r[prefix+'_'+k] for r in records)
            for k in ('i2t','t2i','keep_ratio','g_mean','g_variance','g_saturation')}
        means['combined_CE']=means['i2t']+means['t2i']
        means['weighted_CE']=10/3*weight*means['combined_CE'];result[view]=means
    total=sum(v['weighted_CE'] for v in result.values())
    for v in result.values():v['alignment_share_percent']=100*v['weighted_CE']/total
    return result


def hierarchy(records):
    values={k:statistics.fmean(r[k] for r in records) for k in MASK_KEYS}
    return {k.replace('Ds','D3'):v for k,v in values.items()}


def runtime(steps,commands):
    cycles=rows(TRAIN/'cycle_timing.jsonl')
    phases={r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
    assert [c['step'] for c in cycles]==list(range(1,4869))
    assert all([p['step'] for p in records]==list(range(1,4869)) for records in phases.values())
    lookup={r:{p['step']:p for p in records} for r,records in phases.items()}
    telemetry=[s for s in rows(RUN/'resource-telemetry.jsonl') if s['command']=='train4868']
    systems=[s['system'] for s in telemetry]+[p['system_after'] for pp in phases.values() for p in pp]
    waits=[max(lookup[r][c['step']]['data_wait_s'] for r in range(4)) for c in cycles]
    paths=[r for p in PHASE.glob('image-paths-*.jsonl') for r in rows(p)]
    assert paths and all(Path(p['actual_path']).is_relative_to(IMAGES) and p['NFS_fallback'] is False for p in paths)
    text=(RUN/'train4868.log').read_text(errors='replace')
    errors=sorted(set(re.findall(r'Image failure sample=\d+[^\n]*|Input/output error[^\n]*|Missing local sample=\d+[^\n]*',text)))
    oom=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0)
    assert not errors and oom==0
    inventory=list(RUN.glob('*.log'))+[RUN/'resource-telemetry.jsonl',RUN/'sampling-audit-4000.json',RUN/'prelaunch-local-path-proof-5000.json']
    inventory+=list(TRAIN.glob('*.jsonl'))+list(PHASE.glob('*.jsonl'))
    intervals={c['raw_log']:[c['started_utc'],c['ended_utc']] for c in commands}
    span=[commands[0]['started_utc'],commands[-1]['ended_utc']]
    result=dict(completed_updates=4868,full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        steady_steps7_plus_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles if c['step']>=7]),
        data_wait_seconds_slowest_rank=distribution(waits),
        rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in pp]) for r,pp in phases.items()},
        steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),peak_file_cache_bytes=max(s['file'] for s in systems),
        peak_anon_bytes=max(s['anon'] for s in systems),oom_kill=oom,true_training_io_error_count=len(errors),
        pod_or_supervisor_anomaly_observed=False,local_only_proof=dict(passed=True,count=len(paths),
            ranks=sorted({p['rank'] for p in paths}),NFS_fallback=False),
        GPU_peak_allocated_GiB={str(r):max(h['peak_allocated_gib'] for row in steps for h in row['rank_health'] if h['rank']==r) for r in range(4)},
        checkpoint_timing=rows(TRAIN/'checkpoint_timing.jsonl'),training_wall_seconds=(json.loads((TRAIN/'acceptance.json').read_text())['ranks'][0]['seconds']),
        PSI_scope='Host /proc/pressure, cgroupv1; not per-cgroup PSI',
        memory_interpretation='Includes file/page cache, not RSS; actual oom_kill verified separately',
        local_raw_artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            time_range_utc=intervals.get(str(p),span),uploaded=False) for p in inventory if p.exists()])
    for kind in ('io_PSI','memory_PSI'):
        result[kind]={v:distribution([s[kind][v]['avg10'] for s in systems if v in s[kind]]) for v in ('some','full')}
    return result


def comparisons(result,baseline,debias):
    result['comparison_baselines']=dict(S02_full=baseline,DeBias=debias)
    result['delta_vs_baselines_pp']={name:{k:result['scores_percent'][k]-v for k,v in old['scores_percent'].items() if k in SCORE_KEYS}
        for name,old in result['comparison_baselines'].items()}
    result['dataset_delta_vs_baselines_pp']={name:{dataset:{direction:{metric:100*(result['metrics'][dataset][direction][metric]-value)
        for metric,value in metrics.items()} for direction,metrics in values.items()}
        for dataset,values in old.get('metrics',{}).items() if dataset in result['metrics']}
        for name,old in result['comparison_baselines'].items()}


def write_report(result,diag,masks,stats):
    lines=['# Nested D3 Balanced full4868','',
        f'Status: `{result["status"]}`. Fresh common step0, exactly4868 optimizer updates; no further experiment.',
        'Input config identical to reviewed500 version; only CLI stop500->4868. Frozen F/Dall/D3 [1.35,1.35,.30], native summed directional CE, alignment10/3, unchanged detached-child chain/ramp200/max1 and sparsity(F+2Dall+2D3)/3.',
        'ViT-B/16 Balanced-Stack-Patch, pair-conditioned Hard-ST mask/shared pool/balanced gate,4 A10080GB,256/rank,accum1,seed0,workers8,horizon4868; native optimizer/LR/AMP/preprocess unchanged. Original epoch tail180/rank retained.',
        f'Local-only training root `{IMAGES}`; missing/symlink/escape fails, no NFS fallback. Ephemeral Docker overlay; NFS originals retained. No copy/full decode audit started.',
        '', '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
    for name,v in result['metrics'].items():
        values=[' / '.join(f'{100*v[d][k]:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        lines.append(f'| {name} | {values[0]} | {values[1]} |')
    lines+=['','| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in [*result['comparison_baselines'].items(),('D3 Balanced full',result)]:
        p=r['scores_percent'];u=r.get('metrics',{}).get('Urban-1k',{})
        scores=' | '.join(f'{p[k]:.6f}' if k in p else 'missing' for k in SCORE_KEYS)
        urban=' | '.join(f'{100*u[d]["R@1"]:.3f}' if d in u else 'missing' for d in ('I2T','T2I'))
        lines.append(f'| {name} | {scores} | {urban} |')
    for name,old in result['comparison_baselines'].items():
        lines+=['',f'## Deltas vs {name}','',
            'Aggregate deltas(pp): `'+json.dumps(result['delta_vs_baselines_pp'][name])+'`.',
            '| Dataset | Baseline I2T/T2I R@1 | D3 Balanced I2T/T2I R@1 | Delta I2T/T2I(pp) |','|---|---|---|---|']
        for dataset,v in result['metrics'].items():
            prev=old.get('metrics',{}).get(dataset,{})
            a=' / '.join(f'{100*prev[d]["R@1"]:.6f}' if d in prev else 'missing' for d in ('I2T','T2I'))
            b=' / '.join(f'{100*v[d]["R@1"]:.6f}' for d in ('I2T','T2I'))
            delta=result['dataset_delta_vs_baselines_pp'][name].get(dataset,{})
            dd=' / '.join(f'{delta[d]["R@1"]:+.6f}' if d in delta else 'missing' for d in ('I2T','T2I'))
            lines.append(f'| {dataset} | {a} | {b} | {dd} |')
        lines+=['','All available R@1/5/10 deltas are in RESULTS.json. Reference limitations: '+old.get('reference_limitations','None')+'.']
    lines+=['','## Final last50 diagnostics','','| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment share | Keep ratio |','|---|---:|---:|---:|---:|---:|---:|']
    for v,s in diag['last50_views'].items():
        lines.append(f'| {v} | {s["i2t"]:.6f} | {s["t2i"]:.6f} | {s["combined_CE"]:.6f} | {s["weighted_CE"]:.6f} | {s["alignment_share_percent"]:.3f}% | {s["keep_ratio"]:.6f} |')
    lines+=['','Mask hierarchy: `'+json.dumps(masks['last50'])+'`.',
        'Mask delta vs500: `'+json.dumps(masks['delta_vs500'])+'`. Soft inclusion does not guarantee zero hard violations; no projection added.',
        'Four-epoch scalar curves, gate mean/variance/saturation, LR and inclusion ramp:TRAINING_DIAGNOSTICS.json.',
        '', 'Full-cycle(s): `'+json.dumps(stats['full_cycle_seconds'])+'`.',
        'Slowest-rank data_wait(s): `'+json.dumps(stats['data_wait_seconds_slowest_rank'])+'`.',
        f'>3s:{stats["steps_gt3s"]}; >10s:{stats["steps_gt10s"]}; I/O errors:{stats["true_training_io_error_count"]}; oom_kill:{stats["oom_kill"]}; Pod/supervisor anomaly:{stats["pod_or_supervisor_anomaly_observed"]}.',
        'GPU peak GiB/rank: `'+json.dumps(stats['GPU_peak_allocated_GiB'])+'`.',
        f'Peak cgroup:{stats["peak_cgroup_memory_bytes"]/2**30:.3f}GiB; file cache:{stats["peak_file_cache_bytes"]/2**30:.3f}GiB; anon:{stats["peak_anon_bytes"]/2**30:.3f}GiB. Cache is not process RSS/OOM proof. PSI is host-scoped.',
        '', 'Strict native inference:normalized native image embedding @ normalized native full-caption text embedding.T. No mask/gate/Dall/D3/rerank/ensemble/TTA.',
        f'Common0 SHA256:`{STEP0_SHA}`.',f'Final full checkpoint SHA256:`{result["checkpoint_sha256"]}`.',
        f'Bare SHA256:`{result["strict_export"]["bare_sha256"]}`.',
        'Exact export embeddings/checkpoint immutability:EXPORT_AUDIT.json; persistent checkpoints500/1217/2434/3651/4868 and raw log inventories:RUNTIME_STATS.json.',
        'Sample audit:SAMPLING_AUDIT.json,4000 offline strings/tokens/indices across4 epochs; all runtime IDs/indices/LR checked and first500 text stream matched independently.',
        'Decision and thresholds declared before training: `'+json.dumps(result['decision'])+'`.',
        'Git branch:experiment/nested-detail-d3-balanced-full-v1. Checkpoints/bare/datasets/local mirror/cache/raw logs not uploaded. No automatic new experiment.']
    (EXP/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    import torch
    torch.set_num_threads(4)
    saved=json.loads((RUN/'supervisor-result.json').read_text())
    assert saved['error'] is None and saved['result']
    result=saved['result'];commands=saved['commands']
    assert all(c['returncode']==0 for c in commands)
    steps=rows(TRAIN/'steps.jsonl');launch=json.loads((RUN/'launch-provenance.json').read_text())
    for p,expected in launch['source_sha256'].items():
        assert sha(ROOT/p)==expected
        blob=subprocess.check_output(['git','show',launch['git_head']+':'+p],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==expected,('Launch source not archived',p)
    assert all(r['completed_updates']==r['updates_this_run']==4868 and r['max_parameter_difference_from_rank0']==0
        for r in saved['acceptance']['ranks'])
    checkpoint_inventory=[]
    for step in (5,500,1217,2434,3651,4868):
        path=TRAIN/f'step{step:06d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        proof=checkpoint_proof(payload,step);del payload
        checkpoint_inventory.append(dict(**proof,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    assert checkpoint_inventory[-1]['sha256']==result['checkpoint_sha256']==result['strict_export']['checkpoint_sha256']
    bare=TRAIN/'student_step4868.pt';assert sha(bare)==result['strict_export']['bare_sha256']
    assert result['evaluation_checkpoint_immutable'] and result['strict_export']['passed']
    assert result['strict_export']['image_max_abs']==result['strict_export']['text_max_abs']==0
    proof=verify_stream(steps)
    path_proof=json.loads((RUN/'prelaunch-local-path-proof-5000.json').read_text());assert path_proof['passed'] and path_proof['count']==5000
    diag=dict(last50_steps=[r['step'] for r in steps[-50:]],last50_views=view_means(steps[-50:]),
        epochs={str(e):dict(all_updates_views=view_means(steps[e*1217:(e+1)*1217]),
            last50_views=view_means(steps[(e+1)*1217-50:(e+1)*1217]),
            last50_hierarchy=hierarchy(steps[(e+1)*1217-50:(e+1)*1217]),
            losses=distribution([r['loss'] for r in steps[e*1217:(e+1)*1217]])) for e in range(4)},
        selected_steps={str(r['step']):dict(views=view_means([r]),hierarchy=hierarchy([r]),LR=r['actual_lrs'])
            for r in steps if r['step'] in (1,5,100,200,500,1217,2434,3651,4868)},
        full_stream_proof=proof,formula='10/3*(1.35*L_F+1.35*L_Dall+.30*L_D3), directional CE summed')
    old_mask=json.loads((REFERENCE_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())['last50']
    h=hierarchy(steps[-50:])
    masks=dict(last50=h,keep_ratio={v:s['keep_ratio'] for v,s in diag['last50_views'].items()},
        delta_vs500={k:h[k]-old_mask[k] for k in h},
        detached_child_chain_only=True,ramp_updates=200,inclusion_max=1,hard_projection=False)
    stats=runtime(steps,commands);stats['local_large_binary_assets']=checkpoint_inventory+[
        dict(path=str(bare),bytes=bare.stat().st_size,sha256=sha(bare),uploaded=False)]
    sampling=json.loads((EXP/'SAMPLING_AUDIT.json').read_text());sampling['runtime_proof']=proof
    stats['sampling_raw_evidence']=sampling['raw_evidence']
    baseline=json.loads(BASELINE.read_text())
    baseline={k:baseline[k] for k in ('metrics','scores_percent')}
    debias=json.loads(DEBIAS.read_text())
    comparisons(result,baseline,debias)
    urban={d:round(100*result['metrics']['Urban-1k'][d]['R@1'],3) for d in ('I2T','T2I')}
    result['decision']=decision(result['scores_percent'],urban,baseline);result['status']=result['decision']['status']
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    _,raw,sources=native_metrics(TRAIN)
    prior=json.loads((REFERENCE_EXP/'RESULTS.json').read_text())['native_evaluation_provenance']
    for dataset,data in raw.items():
        assert data['checkpoint_sha256']==result['strict_export']['bare_sha256']
        assert data.get('native_only',data.get('native_student_only')) is True
        for k in ('protocol','n_images','n_captions','manifest_sha256'):
            if k in prior[dataset]:assert data[k]==prior[dataset][k]
    result.update(completed_steps=4868,stopped_at_4868=True,initialized_from_common0=True,resume=None,
        local_image_root=str(IMAGES),NFS_fallback=False,configuration=json.loads(CONFIG.read_text()),
        launch_provenance=launch,full_stream_proof=proof,checkpoint_path=str(TRAIN/'step004868.pt'),
        native_evaluation_provenance={name:dict(result_path=str(p),sha256=sha(p)) for name,p in sources.items()},
        automatic_new_experiments=False,checkpoint_uploaded=False,bare_uploaded=False,commands=commands)
    dump(EXP/'RESULTS.json',result);dump(EXP/'TRAINING_DIAGNOSTICS.json',diag)
    dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks);dump(EXP/'RUNTIME_STATS.json',stats)
    dump(EXP/'SAMPLING_AUDIT.json',sampling);dump(EXP/'EXPORT_AUDIT.json',result['strict_export'])
    dump(EXP/'VALIDATION.json',dict(passed=True,reviewed_at_utc=now(),all_4868_steps_and4_rank_states_verified=True,
        immutable_native_export=True,all_source_SHA256_archived=True,full_optimizer_RNG_cursor_checkpoints_verified=True,
        related_CPU_tests=launch['CPU_tests'],local_only_5000_prelaunch_and_runtime_paths_verified=True))
    write_report(result,diag,masks,stats)
    print(json.dumps(dict(status=result['status'],review_passed=True,scores=result['scores_percent'])),flush=True)


if __name__=='__main__':main()
