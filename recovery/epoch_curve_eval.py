"""Read-only epoch replay: unchanged strict exports and four-GPU native eval."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from tools import eval_five_parallel as ev

ROOT=ev.ROOT
EXP=ROOT/'experiments/nest_clip_v1/epoch_curve_eval_v1'
RUN=ROOT/'runtime/SAID-nest-clip-v1/epoch-curve-eval-v1'
BRANCH='analysis/d3-hns-epoch-curve-eval-v1'
STEPS=(1217,2434,3651,4868)
MODELS=('D3_Balanced','HNS_v1')
KEYS=('Score5','J_long3','J_long','Short4')
SOURCES={
    'D3_Balanced':ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1',
    'HNS_v1':ROOT/'experiments/nest_clip_v1/nested_d3_hns_full_v1',
}
RESULT_NAMES={'D3_Balanced':'RESULTS.json','HNS_v1':'FULL_RESULTS.json'}
STATS_NAMES={'D3_Balanced':'RUNTIME_STATS.json','HNS_v1':'FULL_RUNTIME_STATS.json'}
INDEX=Path('/root/said_s02_stage500/data_index')
IMAGES=Path('/root/said_s02_stage500/ShareGPT4V')
HALF=ROOT/'experiments/nest_clip_v1/nested_d3_hns_half1217_v1/comparison1217.json'


def read(path):return json.loads(Path(path).read_text())


def save(name,value):
    path=EXP/name;path.parent.mkdir(parents=True,exist_ok=True);ev.save(path,value)


def state(status,**kw):
    save('STATE.json',dict(status=status,updated_utc=ev.utc(),pid=os.getpid(),training=False,**kw))


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def queue():return [(model,step) for step in STEPS for model in MODELS]


def references():return {m:read(p/RESULT_NAMES[m]) for m,p in SOURCES.items()}


def source_assets(model):
    stats=read(SOURCES[model]/STATS_NAMES[model])
    return stats['local_large_binary_assets'] if model=='D3_Balanced' else stats['local_assets_not_uploaded']


def checkpoint_metadata(payload,step,model):
    assert payload['completed_steps']==payload['global_step']==payload['scheduler']['completed_steps']==step
    assert payload['scheduler_horizon']==payload['scheduler']['horizon']==4868
    assert payload['data_cursor']==dict(next_epoch=step//1217,next_batch=step%1217)
    counters=sorted({int(s['step']) for s in payload['optimizer']['state'].values()})
    assert counters==[step] and len(payload['rng_per_rank'])==4
    cfg=payload['config'];assert cfg['base_model']=='ViT-B/16'
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg['view_weights']==[1.35,1.35,.3]
    assert bool(cfg.get('hns_enabled',False))==(model=='HNS_v1')
    assert cfg['inclusion_max']==(0 if model=='HNS_v1' else 1)
    assert cfg['init_sha256']=='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
    return dict(update=step,optimizer_steps=counters,scheduler=payload['scheduler'],data_cursor=payload['data_cursor'],
        trajectory_root=payload['trajectory_root'],rng_ranks=4,model_architecture=cfg['base_model'],
        configuration=cfg,configuration_sources=cfg['code_sha256'],
        optimizer_group_parameter_counts=[len(g['params']) for g in payload['optimizer']['param_groups']],
        model_shapes={k:list(payload['model'][k].shape) for k in ('visual.conv1.weight','token_embedding.weight','positional_embedding')},
        model_adapter_present=bool(payload['model']) and payload['adapter'] is not None)


def valid_export(check,checkpoint_sha,bare_sha,step):
    return (check.get('passed') is True and check.get('strict_load') is True and
        check.get('optimizer_steps')==[step] and check.get('image_max_abs')==0 and check.get('text_max_abs')==0 and
        check.get('checkpoint_sha256')==checkpoint_sha and check.get('bare_sha256')==bare_sha)


def prepare():
    import torch
    torch.set_num_threads(4)
    assert git('branch','--show-current')==BRANCH
    assert not RUN.exists(),'Never overwrite an existing replay'
    ev.require_gpu_idle({0,1,2,3})
    validation=read(ROOT/'engineering/parallel_five_eval_v1/COMPARISON.json')
    scheduler=read(ROOT/'engineering/parallel_five_eval_v1/PARALLEL_RUN.json')
    assert validation['passed'] and scheduler['scheduler_source_sha256']==ev.sha(ROOT/'tools/eval_five_parallel.py')
    frozen=dict(validation['unchanged_evaluator_source_sha256'])
    frozen['tools/eval_five_parallel.py']=scheduler['scheduler_source_sha256']
    for p,h in frozen.items():assert ev.sha(ROOT/p)==h
    refs=references();half=read(HALF)['models'];inventory={};missing=[];sources={}
    for model,source in SOURCES.items():
        assets=source_assets(model)
        proofs=assets if model=='D3_Balanced' else refs[model]['checkpoint_proofs']
        rows={p.get('completed_steps',p.get('step')):p for p in proofs if p['path'].endswith('.pt')}
        inventory[model]={}
        for step in STEPS:
            proof=rows.get(step)
            if proof is None or not Path(proof['path']).is_file():
                missing.append(dict(model=model,step=step,path=proof.get('path') if proof else None));continue
            path=Path(proof['path']);digest=ev.sha(path)
            assert digest==proof['sha256'] and path.stat().st_size==proof['bytes']
            payload=torch.load(path,map_location='cpu',weights_only=False)
            meta=checkpoint_metadata(payload,step,model);del payload
            record=dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size,**meta)
            candidates=[]
            if step==1217:
                old=half['D3 Balanced' if model=='D3_Balanced' else 'HNS-v1']
                if old['checkpoint_sha256']==digest:
                    candidates.append((Path(old['bare']),old['strict_export']))
            if step==4868:
                assert digest==refs[model]['checkpoint_sha256']
                bare_sha=refs[model]['strict_export']['bare_sha256']
                candidates += [(Path(a['path']),refs[model]['strict_export']) for a in assets if a.get('sha256')==bare_sha]
            for bare,check in candidates:
                if bare.is_file() and valid_export(check,digest,ev.sha(bare),step):
                    record['reuse_export']=dict(bare=str(bare),bare_sha256=ev.sha(bare),strict_export=check,
                        evidence=str(HALF if step==1217 else source/RESULT_NAMES[model]));break
            inventory[model][str(step)]=record
        sources[model]={str(p):ev.sha(p) for p in (source/RESULT_NAMES[model],source/STATS_NAMES[model],
            source/'config.json',source/'FULL_RESULTS.md')}
        for name in ('EXPORT_AUDIT.json','RESUME_PROVENANCE.json','PLAN.json','VALIDATION.json'):
            p=source/name
            if p.exists():sources[model][str(p)]=ev.sha(p)
    save('CHECKPOINT_INVENTORY.json',dict(models=inventory,missing=missing,source_reports=sources,
        frozen_evaluation_sources=frozen,checked_utc=ev.utc(),no_training=True,
        optional_step500='Not replayed: epoch1-4 is the mandatory scope; optional early references never replace epoch checkpoints'))
    assert not missing,'Missing checkpoint(s); no training reconstruction or substitutions'
    RUN.mkdir(parents=True)
    state('READY',queue=queue(),gpu_mapping=ev.gpu_mapping(),missing_checkpoints=[])


def command(key,cmd):
    log=RUN/(key+'.log');assert not log.exists()
    record=dict(key=key,command=cmd,started_utc=ev.utc(),log=str(log),evaluator_git_commit=git('rev-parse','HEAD'))
    start=time.monotonic()
    with log.open('xb') as handle:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'),start_new_session=True)
        record['pid']=p.pid
        try:rc=p.wait()
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    record.update(returncode=rc,ended_utc=ev.utc(),wall_seconds=time.monotonic()-start)
    commands=read(EXP/'EVAL_COMMANDS.json') if (EXP/'EVAL_COMMANDS.json').exists() else []
    commands.append(record);save('EVAL_COMMANDS.json',commands)
    if rc:raise RuntimeError(key+' failed; see '+str(log))


def frozen_sources(inventory):
    for path,h in inventory['frozen_evaluation_sources'].items():assert ev.sha(ROOT/path)==h
    for reports in inventory['source_reports'].values():
        for p,h in reports.items():assert ev.sha(p)==h


def evaluate(model,step,entry):
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short
    ev.require_gpu_idle({0,1,2,3})
    checkpoint=Path(entry['path']);assert ev.sha(checkpoint)==entry['sha256']
    out=RUN/model/f'step{step}';out.mkdir(parents=True,exist_ok=False)
    prefix=model+'-'+str(step)
    if 'reuse_export' in entry:
        export=entry['reuse_export'];bare=Path(export['bare'])
        assert valid_export(export['strict_export'],entry['sha256'],ev.sha(bare),step)
        ev.save(out/'export-check.json',export['strict_export'])
        audit=dict(reused=True,source_evidence=export['evidence'])
    else:
        bare=out/f'student_step{step}.pt'
        command(prefix+'-export',[sys.executable,'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),
            '--output',str(bare),'--expect-updates',str(step)])
        command(prefix+'-verify',[sys.executable,'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),
            '--bare',str(bare),'--output',str(out/'export-check.json'),'--index-dir',str(INDEX),'--image-root',str(IMAGES)])
        audit=dict(reused=False)
    check=read(out/'export-check.json');bare_sha=ev.sha(bare)
    assert valid_export(check,entry['sha256'],bare_sha,step)
    audit.update(checkpoint=str(checkpoint),bare=str(bare),bare_size_bytes=bare.stat().st_size,strict_export=check)
    save(f'evidence/{model}/step{step}/export-check.json',audit)
    command(prefix+'-eval',[sys.executable,'-m','tools.eval_five_parallel','--checkpoint',str(bare),
        '--training-checkpoint',str(checkpoint),'--output-dir',str(out)])
    m,raw,paths=native_metrics(out)
    aggregates=with_short(dict(metrics=m,scores=scores(m)))
    percent={k:100*aggregates[{'Score5':'Score5_R1','Short4':'Short4_R1'}.get(k,k)] for k in KEYS}
    assert ev.sha(checkpoint)==entry['sha256'] and ev.sha(bare)==bare_sha
    receipt=read(out/'EVAL_PARALLEL_RUN.json');assert receipt['status']=='COMPLETED'
    for ds,p in paths.items():save(f'evaluations/{model}/step{step}/{ds}.json',read(p))
    save(f'evidence/{model}/step{step}/EVAL_PARALLEL_RUN.json',receipt)
    result=dict(model=model,update=step,epoch=step//1217,metrics=m,scores_percent=percent,export=audit,
        checkpoint_sha256=entry['sha256'],bare_sha256=bare_sha,checkpoint_unchanged=True,
        evaluation_wall_seconds=receipt['wall_seconds'],gpu_mapping=receipt['gpu_mapping'],output_dir=str(out))
    if step==4868:
        previous=references()[model]
        result['final_reproduction']=dict(metrics_exact=m==previous['metrics'],
            aggregates_exact=all(percent[k]==previous['scores_percent'][k] for k in KEYS),
            max_recall_abs_error=max(abs(m[d][dr][k]-previous['metrics'][d][dr][k]) for d in m for dr in m[d] for k in m[d][dr]))
        if not result['final_reproduction']['metrics_exact'] or not result['final_reproduction']['aggregates_exact']:
            save('FINAL_REPRODUCTION_MISMATCH.json',dict(model=model,new=result,original=previous))
            raise RuntimeError('FINAL_REPRODUCTION_MISMATCH: stop trajectory interpretation')
    return result


def crossings(points):
    """Discrete observations only; ties do not imply a known crossing step."""
    return [dict(between_epochs=[a['epoch'],b['epoch']],delta_before=a['delta'],delta_after=b['delta'])
        for a,b in zip(points,points[1:]) if a['delta']*b['delta']<0]


def curves(results):
    aggregate={k:[] for k in KEYS};directional={};gains=[]
    for step in STEPS:
        a=results['D3_Balanced'][str(step)];b=results['HNS_v1'][str(step)]
        for key in KEYS:
            av,bv=a['scores_percent'][key],b['scores_percent'][key]
            aggregate[key].append(dict(update=step,epoch=step//1217,Balanced=av,HNS=bv,delta=bv-av))
        for ds in a['metrics']:
            for dr in ('I2T','T2I'):
                av,bv=100*a['metrics'][ds][dr]['R@1'],100*b['metrics'][ds][dr]['R@1']
                directional.setdefault(ds+'/'+dr,[]).append(dict(update=step,epoch=step//1217,Balanced=av,HNS=bv,delta=bv-av))
    for i,(s,t) in enumerate(zip(STEPS,STEPS[1:])):
        values={}
        for key,points in aggregate.items():
            a,b=points[i],points[i+1];ga=b['Balanced']-a['Balanced'];gb=b['HNS']-a['HNS']
            values[key]=dict(Balanced=ga,HNS=gb,HNS_minus_Balanced_growth=gb-ga)
        gains.append(dict(segment=f'E{i+1}->E{i+2}',updates=[s,t],gains_pp=values))
    return aggregate,directional,gains


def training_diagnostics(inventory):
    result={}
    for model in MODELS:
        assets=source_assets(model)
        if model=='D3_Balanced':assets=read(SOURCES[model]/STATS_NAMES[model])['local_raw_artifacts']
        paths=[Path(a['path']) for a in assets if a['path'].endswith('/steps.jsonl')]
        assert len(paths)==1 and paths[0].is_file()
        path=paths[0];windows={s:[] for s in STEPS}
        with path.open() as handle:
            for line in handle:
                r=json.loads(line)
                for s in STEPS:
                    if s-49<=r['step']<=s:windows[s].append(r)
        result[model]={}
        for step,records in windows.items():
            assert [r['step'] for r in records]==list(range(step-49,step+1))
            views={}
            for view,prefix,weight in (('F','F',1.35),('Dall','O',1.35),('D3','E',.30)):
                ce=statistics.fmean(r[prefix+'_i2t']+r[prefix+'_t2i'] for r in records)
                views[view]=dict(keep=statistics.fmean(r[prefix+'_keep_ratio'] for r in records),
                    i2t_CE=statistics.fmean(r[prefix+'_i2t'] for r in records),
                    t2i_CE=statistics.fmean(r[prefix+'_t2i'] for r in records),combined_CE=ce,
                    weighted_CE=10/3*weight*ce)
            total=sum(v['weighted_CE'] for v in views.values())
            for v in views.values():v['alignment_share_percent']=100*v['weighted_CE']/total
            telemetry={}
            for key in ('F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation'):
                pairs=[(r['step'],r[key]) for r in records if isinstance(r.get(key),(int,float)) and math.isfinite(r[key])]
                telemetry[key.replace('Ds','D3')]=dict(mean=statistics.fmean(v for _,v in pairs) if pairs else None,
                    observations=len(pairs),steps=[s for s,_ in pairs],scope='epoch-last50' if len(pairs)==50 else 'selected-step')
            result[model][str(step)]=dict(scope='epoch-last50',steps=[r['step'] for r in records],views=views,telemetry=telemetry)
        result[model]['source']=dict(path=str(path),bytes=path.stat().st_size,sha256=ev.sha(path),uploaded=False)
    save('TRAINING_DIAGNOSTICS_COMPARISON.json',result)
    return result


def report(results,inventory,start):
    assert all(results[m]['4868']['final_reproduction']['metrics_exact'] and
        results[m]['4868']['final_reproduction']['aggregates_exact'] for m in MODELS)
    aggregate,directions,gains=curves(results);diagnostics=training_diagnostics(inventory)
    save('EPOCH_CURVES.json',dict(aggregates=aggregate,epoch_to_epoch_gains=gains,
        crossings={k:crossings(v) for k,v in aggregate.items()}))
    save('DIRECTIONAL_R1_CURVES.json',dict(curves=directions,crossings={k:crossings(v) for k,v in directions.items()}))
    save('EXPORT_AUDIT_SUMMARY.json',{m:{s:r['export'] for s,r in rr.items()} for m,rr in results.items()})
    runtime=dict(total_replay_wall_seconds=time.monotonic()-start,
        five_set_evaluation_wall_seconds=sum(r['evaluation_wall_seconds'] for rr in results.values() for r in rr.values()),
        durations={m:{s:r['evaluation_wall_seconds'] for s,r in rr.items()} for m,rr in results.items()},
        checkpoint_order=queue(),gpu_mapping=ev.gpu_mapping(),checkpoints_parallel=False,training=False,
        evaluator_math_changed=False,completed_checkpoints=8)
    save('RUNTIME_STATS.json',runtime)
    score_cross=crossings(aggregate['Score5'])
    earliest=[g for g in gains if g['gains_pp']['Score5']['HNS_minus_Balanced_growth']<0]
    recommendation='YES' if earliest and earliest[0]['segment']=='E1->E2' else 'NO'
    conclusions=dict(Score5_crossings=score_cross,
        first_segment_with_slower_HNS_Score5_growth=earliest[0]['segment'] if earliest else None,
        aggregate_first_negative_epoch={k:next((p['epoch'] for p in v if p['delta']<0),None) for k,v in aggregate.items()},
        final_directional_gap_pp={k:v[-1]['delta'] for k,v in directions.items()},
        HALF_CONTINUE_TO_2434=recommendation,
        recommendation_rule='YES if HNS first loses relative Score5 growth over E1->E2; otherwise cautious NO based on epoch curve alone',
        causal_claim=False,training_performed=False,full_final_exactly_reproduced=True,
        interpretation='One seed, four discrete epoch observations. Slower relative growth is not necessarily an absolute retrieval decline. Alignment loss share is not a gradient norm.')
    save('RESULTS.json',dict(status='COMPLETE',models=results,conclusions=conclusions,training=False))
    with (EXP/'aggregate_curve.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','update','metric','Balanced','HNS','delta_HNS_minus_Balanced_pp'])
        for key,points in aggregate.items():
            for p in points:w.writerow([p['epoch'],p['update'],key,p['Balanced'],p['HNS'],p['delta']])
    with (EXP/'directional_r1_curve.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','update','direction','Balanced_R1','HNS_R1','delta_pp'])
        for key,points in directions.items():
            for p in points:w.writerow([p['epoch'],p['update'],key,p['Balanced'],p['HNS'],p['delta']])
    lines=['# D3 Balanced vs HNS-v1 epoch replay','',
        'Pure retrospective evaluation. No training, optimizer update, HNS-Half continuation, model/loss change or new experiment.',
        'Eight existing checkpoints sequentially replayed with the validated four-GPU scheduler. Both finals reproduce every original directional R@1/5/10 and all four aggregates exactly.',
        'Existing native evaluators and aggregation functions are reused unchanged. No mask, auxiliary inference, rerank, ensemble, TTA or checkpoint averaging.',
        '', '| Model / Score5 | E1 / 1217 | E2 / 2434 | E3 / 3651 | E4 / 4868 |','|---|---:|---:|---:|---:|']
    for model,label in [('D3_Balanced','D3 Balanced'),('HNS_v1','HNS-v1')]:
        lines.append('| '+label+' | '+' | '.join(f'{results[model][str(s)]["scores_percent"]["Score5"]:.6f}' for s in STEPS)+' |')
    lines.append('| HNS minus Balanced | '+' | '.join(f'{p["delta"]:+.6f}' for p in aggregate['Score5'])+' |')
    for key,points in aggregate.items():
        lines += ['',f'## {key} (%)','','| Epoch | Balanced | HNS | HNS minus Balanced(pp) |','|---|---:|---:|---:|']
        lines += [f'| {p["epoch"]} | {p["Balanced"]:.6f} | {p["HNS"]:.6f} | {p["delta"]:+.6f} |' for p in points]
    lines += ['','## Epoch-to-epoch gains','','| Segment | Metric | Balanced gain(pp) | HNS gain(pp) | Relative HNS growth(pp) |','|---|---|---:|---:|---:|']
    for g in gains:
        for key,v in g['gains_pp'].items():lines.append(f'| {g["segment"]} | {key} | {v["Balanced"]:+.6f} | {v["HNS"]:+.6f} | {v["HNS_minus_Balanced_growth"]:+.6f} |')
    lines += ['','## Directional R@1 trajectory','','| Dataset / direction | E1 B/H | E2 B/H | E3 B/H | E4 B/H | Final HNS minus Balanced(pp) |','|---|---|---|---|---|---:|']
    for key,v in directions.items():
        lines.append('| '+key+' | '+' | '.join(f'{p["Balanced"]:.6f}/{p["HNS"]:.6f}' for p in v)+f' | {v[-1]["delta"]:+.6f} |')
    lines += ['','## Matched training diagnostics','',
        'Extracted from immutable existing scalar training logs. CE, weighted CE, shares and keep are epoch-last50 means. Hierarchy telemetry retains its actual observation count and step list; partial observations are labeled selected-step, never treated as 50 full observations.',
        '', '| Model | Epoch | F keep | Dall keep | D3 keep | F-Dall IoU | Dall-D3 IoU | Dall outside F(%) | D3 outside Dall(%) | D3 alignment share(%) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        for s in STEPS:
            d=diagnostics[m][str(s)];v=d['views'];t=d['telemetry']
            vals=[v[x]['keep'] for x in ('F','Dall','D3')]+[t[x]['mean'] for x in ('F_Dall_mask_iou','Dall_D3_mask_iou')]+[
                100*t[x]['mean'] if t[x]['mean'] is not None else None for x in ('Dall_F_hard_violation','D3_Dall_hard_violation')]+[v['D3']['alignment_share_percent']]
            lines.append(f'| {m} | {s//1217} | '+' | '.join('missing' if x is None else f'{x:.6f}' for x in vals)+' |')
    lines += ['','## Interpretation','',
        'Observed Score5 crossover intervals: `'+json.dumps(score_cross)+'`. No exact crossing update is inferred.',
        'First epoch with negative HNS-minus-Balanced gap by aggregate: `'+json.dumps(conclusions['aggregate_first_negative_epoch'])+'`.',
        'First segment with slower relative HNS Score5 growth: `'+str(conclusions['first_segment_with_slower_HNS_Score5_growth'])+'`.',
        'Final directional gaps(pp): `'+json.dumps(conclusions['final_directional_gap_pp'])+'`.',
        f'`HALF_CONTINUE_TO_2434 = {recommendation}`. This is a recommendation only; no Half training starts.',
        'Keep/violation changes and retrieval are observational at discrete epoch points; temporal correspondence alone establishes no causal mechanism. CE share does not establish gradient dominance.',
        '',f'Total replay wall time: {runtime["total_replay_wall_seconds"]:.3f}s; sum of five-set evaluation wall times: {runtime["five_set_evaluation_wall_seconds"]:.3f}s.',
        'GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. No simultaneous checkpoints.',
        'Checkpoint/bare weights, data and raw logs remain local. CHECKPOINT_INVENTORY.json and EXPORT_AUDIT_SUMMARY.json preserve exact paths, hashes, sizes, update/optimizer/cursor/config provenance and strict embedding equality.',
        'All 30 directional recalls per checkpoint are in RESULTS.json and evaluations/. EVAL_COMMANDS.json and evidence/ preserve exact subprocess commands, GPU mapping, UTC, return codes and durations. Optional step500 was not replayed, so it cannot obstruct epoch1-4.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def publish():
    from recovery import check_stage500_publish as review
    assert git('branch','--show-current')==BRANCH and not git('diff','--cached','--name-only')
    paths=[ROOT/'recovery/epoch_curve_eval.py',ROOT/'tests/test_epoch_curve_eval.py']
    names=('REPORT.md','RESULTS.json','CHECKPOINT_INVENTORY.json','EVAL_COMMANDS.json','EXPORT_AUDIT_SUMMARY.json',
        'EPOCH_CURVES.json','DIRECTIONAL_R1_CURVES.json','TRAINING_DIAGNOSTICS_COMPARISON.json','RUNTIME_STATS.json',
        'STATE.json','CPU_TESTS.json','aggregate_curve.csv','directional_r1_curve.csv','PROGRESS.json')
    paths += [EXP/n for n in names]
    paths += [EXP/f'evaluations/{m}/step{s}/{d}.json' for m,s in queue()
              for d in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')]
    paths += [EXP/f'evidence/{m}/step{s}/{n}.json' for m,s in queue() for n in ('export-check','EVAL_PARALLEL_RUN')]
    relative=[str(p.relative_to(ROOT)) for p in paths];assert all(p.is_file() for p in paths)
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
    review.ALLOWED=set(relative);checked=review.inspect();assert checked['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Report unified D3 Balanced and HNS-v1 epoch replay'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    remote=git('rev-parse','origin/'+BRANCH);assert remote==head==git('rev-parse','FETCH_HEAD')
    ev.save(RUN/'GITHUB_RECEIPT.json',dict(branch=BRANCH,commit=head,remote_HEAD=remote,push_success=True,
        remote_HEAD_matches_local=True,checked_utc=ev.utc(),publication_check=checked))


def run():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Evaluation runner interrupted; preserve outputs')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();inventory=read(EXP/'CHECKPOINT_INVENTORY.json');results={m:{} for m in MODELS}
        try:
            assert read(EXP/'CPU_TESTS.json')['passed']
            for model,step in queue():
                frozen_sources(inventory)
                state('EVALUATING',model=model,update=step,completed=sum(len(x) for x in results.values()),queue=queue())
                results[model][str(step)]=evaluate(model,step,inventory['models'][model][str(step)])
                save('PROGRESS.json',dict(models=results,completed=sum(len(x) for x in results.values()),analysis_pending_final_gate=True))
            frozen_sources(inventory)
            for entries in inventory['models'].values():
                for entry in entries.values():assert ev.sha(entry['path'])==entry['sha256']
            state('REPORTING',completed=8)
            report(results,inventory,start);ev.require_gpu_idle({0,1,2,3})
            state('EVALUATION_AND_ANALYSIS_COMPLETE',completed=8,finals_exact=True)
            publish()
            ev.save(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',finished_utc=ev.utc(),training=False,
                github=read(RUN/'GITHUB_RECEIPT.json')))
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',error=type(error).__name__+': '+str(error),training_reconstruction=False,automatic_retry=False)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--detach',action='store_true');args=p.parse_args()
    if args.prepare:prepare();return
    if not args.detach:run();return
    assert read(EXP/'STATE.json')['status']=='READY' and not (RUN/'runner.json').exists()
    with (RUN/'runner.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','recovery.epoch_curve_eval'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    ev.save(RUN/'runner.json',dict(pid=child.pid,session=child.pid,log=str(RUN/'runner.log'),queue=queue(),training=False))
    print(child.pid,flush=True)


if __name__=='__main__':main()
