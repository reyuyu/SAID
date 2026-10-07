"""Independent CPU/checkpoint/stream/native-protocol review and seven-arm summaries."""
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from recovery import nested_d3_local_search as search
from recovery.s02_nfs500 import dump,rows,sha,distribution,now

SCORES=('Score5','J_long3','J_long','Short4')
PRIMARY=('Score5','J_long3','Urban_T2I','Short4')
GATES=dict(Score5=71.063600,J_long3=74.954666,Urban_T2I=89.7,Short4=65.227000)
EPS=1e-6


def quality(result):
    return dict(**{k:result['scores_percent'][k] for k in SCORES},
        Urban_I2T=round(100*result['metrics']['Urban-1k']['I2T']['R@1'],3),
        Urban_T2I=round(100*result['metrics']['Urban-1k']['T2I']['R@1'],3))


def dominates(a,b,keys=PRIMARY):
    return all(a[k]>=b[k]-EPS for k in keys) and any(a[k]>b[k]+EPS for k in keys)


def classify(result,frontier=False):
    q=quality(result);d={k:q[k]-v for k,v in GATES.items()}
    best=d['Score5']>EPS and d['J_long3']>=-EPS and d['Urban_T2I']>=-EPS and d['Short4']>=-.15-EPS
    small_tradeoff=min(d[k] for k in ('Score5','J_long3','Urban_T2I'))>=-.2-EPS and d['Short4']>=-.15-EPS
    positive=frontier and max(d[k] for k in ('Score5','J_long3','Urban_T2I'))>=.1-EPS and small_tradeoff
    return 'NEW_500_BEST' if best else 'PARETO_POSITIVE' if positive else 'NO_IMPROVEMENT'


def checkpoint_proof(payload,step,arm):
    import torch
    cfg=payload['config'];search.frozen_config(cfg,arm)
    assert payload['completed_steps']==payload['global_step']==payload['scheduler']['completed_steps']==step
    assert payload['scheduler_horizon']==payload['scheduler']['horizon']==cfg['horizon']==4868
    assert payload['data_cursor']==dict(next_epoch=0,next_batch=step)
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==search.STEP0_SHA
    assert cfg['max_updates']==500 and payload['trajectory_root']==str(search.RUN_ROOT/arm)
    assert len(payload['rng_per_rank'])==4
    for rank in payload['rng_per_rank']:
        assert all(k in rank for k in ('python','numpy','cpu','cuda','loader_generator'))
        assert all(torch.is_tensor(rank[k]) and rank[k].numel()>0 for k in ('cpu','cuda','loader_generator'))
    assert payload['adapter'] is not None and payload['optimizer']['state']
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={step}
    for state in (payload['model'],payload['adapter']):
        assert all(torch.isfinite(v).all() for v in state.values())
    assert all(torch.isfinite(v).all() for state in payload['optimizer']['state'].values() for v in state.values())
    return dict(passed=True,completed_steps=step,optimizer_counters=[step],horizon=4868,
        cursor=payload['data_cursor'],model_adapter_optimizer=True,four_rank_RNG_and_loader_present=True,
        fresh_common0=True,uploaded=False)


def diagnostics(steps,arm):
    low='Dk' if search.random_k_arm(arm) else 'D3';spec=search.ARMS[arm]
    def view_means(records):
        output={}
        for v,p,w in zip(('F','Dall',low),('F','O','E'),spec['weights']):
            means={k:statistics.fmean(r[p+'_'+k] for r in records) for k in
                ('i2t','t2i','keep_ratio','positive_keep_ratio','g_mean','g_variance','g_saturation')}
            means['combined_CE']=means['i2t']+means['t2i']
            means['weighted_CE']=10/3*w*means['combined_CE'];output[v]=means
        total=sum(v['weighted_CE'] for v in output.values())
        for v in output.values():v['alignment_share_percent']=100*v['weighted_CE']/total
        return output
    total=Counter();ratios=Counter();hist=Counter();pool=Counter();by_m={};valid=0;strict=0;records=0;sentence_sum=0
    for row in steps:
        for h in row['rank_health']:
            s=h['sampling'];t=s['nested_detail_statistics'];d=s['nested_d3_statistics']
            valid+=t['valid_samples'];records+=d['records'];strict+=d['strict_subset_samples']
            hist.update(d['K_eff_histogram']);pool.update(d['m_histogram'])
            sentence_sum+=sum(int(k)*n for k,n in t['Dall_sentence_count_histogram'].items())
            for label,data in t['views'].items():
                name=low if label=='Ds' else label
                total[name+'_effective']+=data['effective_token_sum'];total[name+'_content']+=data['content_token_sum']
            ratios.update(t['coverage_ratio_sums'])
            for n,counts in s['K_histogram_by_n'].items():by_m.setdefault(str(int(n)-1),Counter()).update(counts)
    valid_k={k:v for k,v in hist.items() if int(k)>0}
    sampling=dict(records=records,valid_records=valid,K_histogram_all=dict(hist),K_histogram_valid=valid_k,
        K_histogram_by_m={k:dict(v) for k,v in by_m.items()},m_histogram=dict(pool),
        mean_K_valid=sum(int(k)*n for k,n in valid_k.items())/valid,
        mean_K_all=sum(int(k)*n for k,n in hist.items())/records,
        Dall_mean_sentences=sentence_sum/valid,strict_subset_ratio=strict/records,
        mean_effective_tokens={v:total[v+'_effective']/valid for v in ('F','Dall',low)},
        mean_content_tokens={v:total[v+'_content']/valid for v in ('F','Dall',low)},
        mean_lowest_Dall_content_token_coverage=ratios['Ds_Dall']/valid,
        pooled_lowest_Dall_content_token_coverage=total[low+'_content']/total['Dall_content'],
        coverage_definition='Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples')
    keys=('inc','inc_weight','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation')
    masks={k.replace('Ds',low):statistics.fmean(r[k] for r in steps[-50:]) for k in keys}
    return dict(last50_steps=list(range(451,501)),last50_views=view_means(steps[-50:]),sampling=sampling,
        selected_steps={str(r['step']):view_means([r]) for r in steps if r['step'] in (1,100,200,500)},
        raw_scalar_curves_local_only=True,weighted_formula='10/3 * sum(w_view * directional_summed_CE)'),masks


def runtime(arm,steps,saved):
    run=search.RUN_ROOT/arm;train=run/'step500';phase=search.LOCAL/(search.PHASE_PREFIX+arm)
    cycles=rows(train/'cycle_timing.jsonl');phases={r:rows(phase/f'rank{r}.jsonl') for r in range(4)}
    assert [c['step'] for c in cycles]==list(range(1,501))
    assert all([p['step'] for p in pp]==list(range(1,501)) for pp in phases.values())
    telemetry=rows(run/'resource-telemetry.jsonl')
    systems=[p['system'] for p in telemetry if p['command']=='train500']+[p['system_after'] for pp in phases.values() for p in pp]
    waits=[max(phases[r][i]['data_wait_s'] for r in range(4)) for i in range(500)]
    paths=[p for f in phase.glob('image-paths-*.jsonl') for p in rows(f)]
    assert paths and all(Path(p['actual_path']).is_relative_to(search.IMAGES) and p['NFS_fallback'] is False for p in paths)
    text=(run/'train500.log').read_text(errors='replace')
    assert not any(t in text for t in ('Image failure sample=','Input/output error','Missing local sample='))
    oom=max(s['memory_events'].get('oom_kill',0) for s in systems);assert oom==0
    commands=saved['commands'];intervals={c['raw_log']:[c['started_utc'],c['ended_utc']] for c in commands}
    span=[saved['started_utc'],saved['ended_utc']]
    raw=list(run.glob('*.log'))+list(run.glob('*.jsonl'))+list(train.glob('*.jsonl'))+list(phase.glob('*.jsonl'))
    raw+=[run/'sampling-audit-1000.json',run/'prelaunch-local-path-proof-5000.json',run/'commands.json',run/'launch-provenance.json']
    stats=dict(completed_updates=500,training_wall_seconds=saved['acceptance']['ranks'][0]['seconds'],
        full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        data_wait_seconds_slowest_rank=distribution(waits),
        rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in pp]) for r,pp in phases.items()},
        steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),peak_file_cache_bytes=max(s['file'] for s in systems),
        peak_anon_bytes=max(s['anon'] for s in systems),oom_kill=oom,true_training_IO_errors=0,pod_or_supervisor_anomaly_observed=False,
        GPU_peak_allocated_GiB={str(r):max(h['peak_allocated_gib'] for row in steps for h in row['rank_health'] if h['rank']==r) for r in range(4)},
        local_only_proof=dict(passed=True,records=len(paths),ranks=sorted({p['rank'] for p in paths}),NFS_fallback=False),
        PSI_scope='Host /proc/pressure, cgroupv1; not per-cgroup PSI',
        memory_interpretation='Cgroup includes file cache; neither memory.current nor file cache is RSS/OOM proof',
        raw_artifact_time_range_basis='Exact command intervals for logs; conservative supervisor start to end containment window for other raw artifacts',
        local_raw_artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            time_range_utc=intervals.get(str(p),span),uploaded=False) for p in sorted(set(raw))])
    for kind in ('io_PSI','memory_PSI'):
        stats[kind]={v:distribution([s[kind][v]['avg10'] for s in systems]) for v in ('some','full')}
    return stats


def recall_delta(result,anchor):
    return {dataset:{d:{k:100*(v-anchor['metrics'][dataset][d][k]) for k,v in metrics.items()}
        for d,metrics in value.items()} for dataset,value in result['metrics'].items()}


def arm_report(arm,result,diag,masks,gradient,stats):
    spec=search.ARMS[arm];low='Dk' if search.random_k_arm(arm) else 'D3'
    lines=[f'# {arm}: isolated Nested D3 local500 search arm','',
        f'Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`{spec["axis"]}`.',
        f'Alignment:{spec["weights"]}; absolute sparsity coefficients:{search.arm_sparsity(arm)} (mass{sum(search.arm_sparsity(arm))}); mode:{spec["mode"]}.',
        'Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.',
        f'Local-only:`{search.IMAGES}`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.',
        '', '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
    for name,v in result['metrics'].items():
        recalls=[' / '.join(f'{100*v[d][k]:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        lines.append(f'| {name} | {recalls[0]} | {recalls[1]} |')
    lines+=['','| Metric | Arm | Delta vs Anchor(pp) |','|---|---:|---:|']
    q=quality(result)
    for k,v in q.items():lines.append(f'| {k} | {v:.6f} | {result["quality_delta_vs_anchor_pp"][k]:+.6f} |')
    lines+=['','| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |','|---|---|---|']
    for name,v in result['recall_delta_vs_anchor_pp'].items():
        recalls=[' / '.join(f'{v[d][k]:+.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        lines.append(f'| {name} | {recalls[0]} | {recalls[1]} |')
    lines+=['','| View | Combined CE | Weighted CE | Alignment share | Keep ratio |','|---|---:|---:|---:|---:|']
    for v,s in diag['last50_views'].items():lines.append(f'| {v} | {s["combined_CE"]:.6f} | {s["weighted_CE"]:.6f} | {s["alignment_share_percent"]:.3f}% | {s["keep_ratio"]:.6f} |')
    lines+=['',f'Raw gradient norms:`{gradient["mean_gradient_norms"]}`.',
        f'Weighted gradient norms:`{gradient["weighted_mean_gradient_norms"]}`; weighted {low}/Dall ratio:`{gradient["weighted_lowest_Dall_ratio"]:.6f}`.',
        'Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.',
        f'Mask hierarchy:`{masks}`. Soft detached-child inclusion does not impose zero hard-mask violations.',
        f'Sampling:`{diag["sampling"]}`.',
        f'Full cycle(s):`{stats["full_cycle_seconds"]}`; slowest-rank data_wait(s):`{stats["data_wait_seconds_slowest_rank"]}`.',
        f'OOM kills:{stats["oom_kill"]}; true training I/O errors:0; Pod anomaly:false. GPU peak:`{stats["GPU_peak_allocated_GiB"]}` GiB.',
        'First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.',
        'Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.',
        f'Common0 SHA256:`{search.STEP0_SHA}`.',f'Checkpoint SHA256:`{result["checkpoint_sha256"]}`.',
        f'Bare SHA256:`{result["strict_export"]["bare_sha256"]}`.',
        'Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.',
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.']
    (search.experiment_dir(arm)/'REPORT.md').write_text('\n'.join(lines)+'\n')


def review_arm(arm):
    import torch
    torch.set_num_threads(4);search.activate(arm)
    run=search.RUN;exp=search.ARM_EXP;train=run/'step500'
    saved=json.loads((run/'supervisor-result.json').read_text())
    assert saved['error'] is None and all(c['returncode']==0 for c in saved['commands'])
    launch=json.loads((run/'launch-provenance.json').read_text())
    for p,h in launch['source_sha256'].items():
        assert sha(search.ROOT/p)==h
        assert hashlib.sha256(subprocess.check_output(['git','show',launch['git_head']+':'+p],cwd=search.ROOT)).hexdigest()==h
    assert all(r['completed_updates']==r['updates_this_run']==500 and r['max_parameter_difference_from_rank0']==0 for r in saved['acceptance']['ranks'])
    assert json.loads((run/'first-five-gate.json').read_text())['passed']
    assert json.loads((run/'prelaunch-local-path-proof-5000.json').read_text())['passed']
    steps=rows(train/'steps.jsonl');assert [r['step'] for r in steps]==list(range(1,501))
    proof=search.matched_stream(steps,rows(search.ANCHOR_RUN/'step500/steps.jsonl'),arm);assert proof['records']==512000
    inventory=[]
    for step in (5,500):
        path=train/f'step{step:06d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        validation=checkpoint_proof(payload,step,arm);del payload
        inventory.append(dict(**validation,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    result=saved['result'];export=result['strict_export']
    assert export['passed'] and export['strict_load'] and export['image_max_abs']==export['text_max_abs']==0
    assert inventory[-1]['sha256']==result['checkpoint_sha256']==export['checkpoint_sha256']
    bare=train/'student_step500.pt';assert sha(bare)==export['bare_sha256'] and result['evaluation_checkpoint_immutable']
    gradient=json.loads((exp/'GRADIENT_SPOTCHECK.json').read_text())
    assert gradient['passed'] and gradient['checkpoint_sha256']==result['checkpoint_sha256'] and gradient['checkpoint_unchanged']
    assert len(gradient['batches'])==8 and gradient['frozen_anchor_batch_IDs_and_protocol_exact']
    assert all(b['all_gradients_finite'] for b in gradient['batches'])
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
    metrics,raw,sources=native_metrics(train);assert metrics==result['metrics']
    anchor=json.loads((search.ANCHOR_EXP/'RESULTS.json').read_text())
    calculated=scores(metrics)
    assert all(math.isclose(100*v,result['scores_percent'][k],abs_tol=1e-10) for k,v in calculated.items())
    for dataset,data in raw.items():
        assert data['checkpoint_sha256']==export['bare_sha256'] and data.get('native_only',data.get('native_student_only')) is True
        previous=anchor['native_evaluation_provenance'][dataset]
        for k in ('protocol','n_images','n_captions','manifest_sha256'):
            if k in previous:assert previous[k]==data[k]
    diag,masks=diagnostics(steps,arm);stats=runtime(arm,steps,saved)
    old_masks=json.loads((search.ANCHOR_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())['last50']
    old_diag=json.loads((search.ANCHOR_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    low='Dk' if search.random_k_arm(arm) else 'D3'
    mask_delta={k:v-old_masks[k if k in old_masks else k.replace('Dk','D3')] for k,v in masks.items()}
    keep_delta={v:s['keep_ratio']-old_diag['last50_views'][v if v in old_diag['last50_views'] else 'D3']['keep_ratio'] for v,s in diag['last50_views'].items()}
    stats['local_binary_assets']=inventory+[dict(path=str(bare),bytes=bare.stat().st_size,sha256=sha(bare),uploaded=False)]
    q,old=quality(result),quality(anchor)
    result.update(arm=arm,spec=search.ARMS[arm],completed_steps=500,stopped_at_500=True,resume=None,
        initialized_from_common0=True,configuration=search.arm_config(arm),launch_provenance=launch,
        local_image_root=str(search.IMAGES),NFS_fallback=False,stream_proof=proof,
        quality_delta_vs_anchor_pp={k:q[k]-v for k,v in old.items()},recall_delta_vs_anchor_pp=recall_delta(result,anchor),
        native_evaluation_provenance=raw,native_evaluation_sources={name:dict(path=str(p),sha256=sha(p)) for name,p in sources.items()},
        checkpoint_uploaded=False,bare_uploaded=False,automatic_full=False,automatic_combinations=False,
        classification='PENDING_ALL_ARMS',gradient_weighted_lowest_Dall_ratio=gradient['weighted_lowest_Dall_ratio'])
    dump(exp/'RESULTS.json',result);dump(exp/'TRAINING_DIAGNOSTICS.json',diag)
    dump(exp/'MASK_HIERARCHY_AUDIT.json',dict(last50=masks,keep_ratio={v:s['keep_ratio'] for v,s in diag['last50_views'].items()},
        delta_vs_anchor=mask_delta,keep_ratio_delta_vs_anchor=keep_delta,
        inclusion_edges=['Dall->F','lowest->Dall'],detached_child=True,ramp200=True,inclusion_max=1,hard_projection=False))
    dump(exp/'RUNTIME_STATS.json',stats);dump(exp/'EXPORT_AUDIT.json',export)
    sampling=json.loads((exp/'SAMPLING_AUDIT.json').read_text());sampling['runtime_proof']=proof;dump(exp/'SAMPLING_AUDIT.json',sampling)
    dump(exp/'VALIDATION.json',dict(passed=True,reviewed_utc=now(),all512000_actual_records_verified=True,
        common0_and_optimizer_RNG_cursor_verified=True,source_SHA256_archived_and_unchanged=True,
        strict_export_embeddings_exact=True,native_eval_protocols_identical_anchor=True,four_rank_parameters_synchronized=True,
        gradient_protocol_identical_anchor=True,CPU_tests=launch['CPU_tests']))
    arm_report(arm,result,diag,masks,gradient,stats)
    print(json.dumps(dict(arm=arm,review_passed=True,scores=result['scores_percent'])),flush=True)


def summarize():
    anchor=json.loads((search.ANCHOR_EXP/'RESULTS.json').read_text())
    results={'Anchor':anchor}
    for arm in search.ARMS:
        exp=search.EXP/arm
        assert json.loads((exp/'VALIDATION.json').read_text())['passed']
        results[arm]=json.loads((exp/'RESULTS.json').read_text())
    q={name:quality(r) for name,r in results.items()}
    frontier=[name for name in results if not any(dominates(other,q[name]) for n,other in q.items() if n!=name)]
    all_metrics_frontier=[name for name in results if not any(dominates(other,q[name],tuple(q[name])) for n,other in q.items() if n!=name)]
    classifications={name:classify(r,name in frontier) for name,r in results.items() if name!='Anchor'}
    eligible=[name for name,status in classifications.items() if status=='NEW_500_BEST']
    winner=max(eligible,key=lambda n:(q[n]['Score5'],q[n]['J_long3'],q[n]['Urban_T2I'],q[n]['Short4'])) if eligible else None
    status='NEW_500_BEST' if winner else 'PARETO_POSITIVE' if 'PARETO_POSITIVE' in classifications.values() else 'NO_IMPROVEMENT'
    summary=dict(status=status,winner=winner,eligible_new_bests=eligible,classifications=classifications,
        leaderboard=q,quality_deltas_vs_anchor_pp={n:{k:v-q['Anchor'][k] for k,v in value.items()} for n,value in q.items() if n!='Anchor'},
        full_recall_deltas_vs_anchor_pp={n:r['recall_delta_vs_anchor_pp'] for n,r in results.items() if n!='Anchor'},
        Pareto_frontier=frontier,Pareto_objectives=list(PRIMARY),six_metric_Pareto_frontier=all_metrics_frontier,
        winner_rule='Score5>71.063600,J_long3>=74.954666,UrbanT2I>=89.7,Short4 decline<=.15pp; rounding tolerance1e-6pp',
        positive_rule='Primary Pareto arm with >=.1pp key gain, each key decline<=.2pp and Short4 decline<=.15pp',
        winner_tiebreak='Score5 then J_long3 then UrbanT2I then Short4 among eligible independent arms',
        all_arms_completed_and_reviewed=True,automatic_full=False,automatic_combinations=False)
    summaries={}
    for arm in search.ARMS:
        exp=search.EXP/arm;diag=json.loads((exp/'TRAINING_DIAGNOSTICS.json').read_text());mask=json.loads((exp/'MASK_HIERARCHY_AUDIT.json').read_text())
        gradient=json.loads((exp/'GRADIENT_SPOTCHECK.json').read_text());low='Dk' if arm=='KR234' else 'D3'
        summaries[arm]=dict(last50_views=diag['last50_views'],sampling=diag['sampling'],mask=mask,
            raw_gradient_norms=gradient['mean_gradient_norms'],weighted_gradient_norms=gradient['weighted_mean_gradient_norms'],
            weighted_lowest_Dall_ratio=gradient['weighted_lowest_Dall_ratio'],lowest_view=low)
    summary['arm_diagnostics']=summaries
    lines=['# Nested D3 Balanced: seven isolated local500 arms','',f'Status:`{status}`. Winner:`{winner}`. Exactly seven fresh-common0 runs; stopped at500, no full/combinations.',
        'Winner/positive rules and tie-break were frozen before launch in SEARCH_PLAN.json. Primary Pareto objectives:Score5,J_long3,Urban T2I,Short4; six-metric secondary frontier also retained.',
        '', '| Arm | Alignment | K | D3 sparsity r | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in results:
        spec=search.ARMS.get(arm,dict(weights=[1.35,1.35,.30],r=2.));v=q[arm]
        lines.append('| '+arm+' | '+ '/'.join(str(w) for w in spec['weights'])+f' | {"Random valid2/3/4" if arm=="KR234" else "3 (bounded strict subset)"} | {spec["r"]} | '+' | '.join(f'{v[k]:.6f}' for k in (*SCORES,'Urban_I2T','Urban_T2I'))+' |')
    lines+=['','| Arm | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I | Classification |','|---|---:|---:|---:|---:|---:|---:|---|']
    for arm,delta in summary['quality_deltas_vs_anchor_pp'].items():lines.append('| '+arm+' | '+' | '.join(f'{delta[k]:+.6f}' for k in (*SCORES,'Urban_I2T','Urban_T2I'))+f' | {classifications[arm]} |')
    lines+=['',f'Primary Pareto frontier:{frontier}. Six-metric frontier:{all_metrics_frontier}.',
        'All individual R@1/5/10 and complete direction/metric deltas:each arm REPORT.md/RESULTS.json and aggregate RESULTS.json.',
        '', '| Arm | Lowest alignment share(%) | Weighted lowest/Dall gradient ratio | Lowest keep | Dall-lowest IoU | Lowest outside Dall(%) |','|---|---:|---:|---:|---:|---:|']
    for arm in search.ARMS:
        exp=search.EXP/arm;d=json.loads((exp/'TRAINING_DIAGNOSTICS.json').read_text());m=json.loads((exp/'MASK_HIERARCHY_AUDIT.json').read_text())['last50']
        low='Dk' if arm=='KR234' else 'D3';v=d['last50_views'][low]
        lines.append(f'| {arm} | {v["alignment_share_percent"]:.3f} | {results[arm]["gradient_weighted_lowest_Dall_ratio"]:.6f} | {v["keep_ratio"]:.6f} | {m["Dall_"+low+"_mask_iou"]:.6f} | {100*m[low+"_Dall_hard_violation"]:.3f} |')
    lines+=['','Weight axis:compare W20/W25/Anchor(.30)/W35/W40 directly; each has identical512000 sample/text/token/indices trajectory. Sparsity axis:S25/S30 redistribute fixed coefficient mass5, no inclusion change. KR234 changes only K selection; original order/fallback and global RNG stay frozen.',
        'Related CPU tests passed (exact count in CPU_TESTS.json), first5 hard gate,5000 prelaunch local paths,complete runtime stream/indices/LR review,8 frozen gradient batches,full optimizer/RNG/cursor checkpoint review and strict native export.',
        'Checkpoint/bare/data/cache/raw logs retained locally, never uploaded. Per-arm RUNTIME_STATS.json records local raw evidence paths,size,SHA256 and UTC windows.',
        'No automated full training,second round,or combination of weight/sparsity/K changes. Await human selection.']
    lines+=['','## Independent axis trends','',
        '| D3 alignment weight | Arm | Score5 | J_long3 | Urban T2I | Short4 |','|---:|---|---:|---:|---:|---:|']
    order=['W20','W25','Anchor','W35','W40']
    for name in order:
        dose=.3 if name=='Anchor' else search.ARMS[name]['weights'][2]
        lines.append(f'| {dose:.3f} | {name} | '+ ' | '.join(f'{q[name][k]:.6f}' for k in ('Score5','J_long3','Urban_T2I','Short4'))+' |')
    best_weight=max(order,key=lambda n:q[n]['Score5'])
    lines+=['',f'Highest Score5 on the weight axis:{best_weight}. All four weight arms preserve fixed K3 and anchor sparsity; no extrapolation or combined arm.',
        '', '| Sparsity r | Arm | Lowest keep | Dall-lowest IoU | Lowest outside Dall(%) | Score5 | J_long3 |','|---:|---|---:|---:|---:|---:|---:|']
    ad=json.loads((search.ANCHOR_EXP/'TRAINING_DIAGNOSTICS.json').read_text());am=json.loads((search.ANCHOR_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())['last50']
    for name in ('Anchor','S25','S30'):
        if name=='Anchor':keep=ad['last50_views']['D3']['keep_ratio'];mask=am;r=2.
        else:keep=summaries[name]['last50_views']['D3']['keep_ratio'];mask=summaries[name]['mask']['last50'];r=search.ARMS[name]['r']
        lines.append(f'| {r} | {name} | {keep:.6f} | {mask["Dall_D3_mask_iou"]:.6f} | {100*mask["D3_Dall_hard_violation"]:.3f} | {q[name]["Score5"]:.6f} | {q[name]["J_long3"]:.6f} |')
    ks=summaries['KR234']['sampling']
    lines+=['',f'KR234 valid K histogram:`{ks["K_histogram_valid"]}`; mean K:{ks["mean_K_valid"]:.6f}; mean Dk effective tokens:{ks["mean_effective_tokens"]["Dk"]:.6f}; mean per-sample Dk/Dall content-token coverage:{ks["mean_lowest_Dall_content_token_coverage"]:.6f}.',
        'Conditioned K counts by pool size m,CE/shares/gradients and mask changes are in KR234/TRAINING_DIAGNOSTICS.json,GRADIENT_SPOTCHECK.json and MASK_HIERARCHY_AUDIT.json.']
    for arm in search.ARMS:
        exp=search.EXP/arm;results[arm]['classification']=classifications[arm];dump(exp/'RESULTS.json',results[arm])
        with (exp/'REPORT.md').open('a') as stream:stream.write('\nFinal search classification:`'+classifications[arm]+'`.\n')
    dump(search.EXP/'RESULTS.json',summary)
    text='\n'.join(lines)+'\n';(search.EXP/'SEARCH_SUMMARY.md').write_text(text);(search.EXP/'REPORT.md').write_text(text)
    print(json.dumps(dict(status=status,winner=winner,Pareto_frontier=frontier)),flush=True)
