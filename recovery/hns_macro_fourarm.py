"""Four independent fresh-common0 HNS macro arms, smoke/eval/report sequentially."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from recovery.s02_nfs500 import ROOT,dump,rows,sha,now,distribution,STEP0,STEP0_SHA
from recovery import nested_d3_local_search as search,s02_local500 as local

PROJECT=Path('/opt/data/private/lklk/SAID')
BRANCH='experiment/hns-macro-loss-fourarm500-v1'
BASE_BRANCH='experiment/nested-d3-hard-nested-sparsity500-v1'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1'
BASE_RUN=PROJECT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS'
EXP=ROOT/'experiments/nest_clip_v1/hns_macro_loss_fourarm500_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-macro-fourarm500-v1'
ENTRY='recovery.hns_macro_fourarm'
MACRO_KEYS=('lambda_align','lambda_sparse','lambda_hierarchy')
GATES=('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json',
       'DEFAULT_GRADIENT_AUDIT.json','FULL_REFERENCE_PROVENANCE.json')
ARMS={'E1-A12':(12.,1.,1.),'E2-S08':(10.,.8,1.),'E3-H075':(10.,1.,.75),'E4-H125':(10.,1.,1.25)}
CODE=('model/balanced_hparam_search.py','recovery/hns_macro_fourarm.py','recovery/hns_macro_equivalence.py',
      'recovery/hns_macro_gradient.py','tests/test_hns_macro.py','tests/test_hns_macro_runner.py',
      'tools/eval_five_parallel.py','tests/test_eval_five_parallel.py')
OLD_MATCH=search.matched_stream


def read(path):return json.loads(Path(path).read_text())


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=120).strip()


def base_config():return read(BASE_EXP/'config.json')


def config(arm):return dict(base_config(),**dict(zip(MACRO_KEYS,ARMS[arm])))


def frozen(cfg,arm):
    assert all(cfg.get(k)==v for k,v in config(arm).items()),'Frozen arm drift'
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['sparsity_scale']==1
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg.get('view_sparsity_weights',[1,2,2])==[1,2,2]
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg['workers']==8
    assert not cfg.get('hns_half_after500',False) and not cfg.get('hns_detach_child',False)
    assert cfg.get('hns_beta',[2,2])==[2,2]


def activate(arm,smoke=False):
    search.ARMS={n:dict(axis='macro',weights=[1.35,1.35,.3],r=2.,mode='nested_detail_d3',
        experiment_dir=str(EXP/n)) for n in ARMS}
    search.RUN_ROOT=RUN;search.EXP=EXP;search.ANCHOR_EXP=BASE_EXP;search.ANCHOR_RUN=BASE_RUN
    search.PHASE_PREFIX='hns-macro-fourarm500-v1-';search.EDITED={'model/balanced_hparam_search.py'}
    search.arm_config=config;search.frozen_config=frozen;search.matched_stream=matched_stream
    search.checkpoint_invariants=checkpoint_first5
    search.activate(arm)
    if smoke:
        local.RUN=RUN/(arm+'.smoke5');local.PHASE=local.IMAGES.parent/('hns-macro-fourarm500-v1-'+arm+'-smoke5')


def matched_stream(actual,reference,arm):
    proof=OLD_MATCH(actual,reference,arm)
    for row in actual:
        assert row['HNS_enabled'] and row['inc_weight']==row['inclusion_loss']==0
        assert row['lambda_h']==min(1.,(row['step']-1)/200.)
        la,ls,lh=ARMS[arm]
        assert [row['macro_'+k] for k in MACRO_KEYS]==[la,ls,lh]
        expected=[la*row['macro_raw_align'],ls*row['macro_raw_sparse'],lh*row['lambda_h']*row['macro_raw_hierarchy']]
        weighted=[row['macro_weighted_'+n] for n in ('align','sparse','hierarchy')]
        for a,b in zip(expected,weighted):assert math.isclose(a,b,rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(row['loss'],sum(weighted),rel_tol=4e-6,abs_tol=2e-6)
    proof.update(full_HNS_IDs_paths_text_tokens_indices_LR_exact=True,only_macro_scales_changed=True)
    return proof


def checkpoint_first5(payload,reference):
    import torch
    arm=search.ARM;frozen(payload['config'],arm)
    cfg=payload['config'];old=reference['config']
    assert payload['completed_steps']==5 and payload['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size',
              'accumulation','seed','sampling_seed','shuffle_seed','workers','optimizer_groups'):
        if k in old:assert cfg[k]==old[k],k
    runtime=copy.deepcopy(cfg['runtime_model'])
    for k in MACRO_KEYS:runtime['search_hparams'].pop(k,None)
    assert runtime==old['runtime_model']
    assert set(cfg['code_sha256'])==set(old['code_sha256'])
    changed={k for k,v in old['code_sha256'].items() if cfg['code_sha256'][k]!=v}
    assert changed=={'model/balanced_hparam_search.py'}
    assert payload['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert payload['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={5}
    pairs=[(payload['model'],reference['model']),(payload['adapter'],reference['adapter'])]
    pairs.extend((v,reference['optimizer']['state'][k]) for k,v in payload['optimizer']['state'].items())
    for new,previous in pairs:
        assert new.keys()==previous.keys()
        for k,v in new.items():assert v.shape==previous[k].shape and v.dtype==previous[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,fresh_common0=True,initialization_exact=True,optimizer_groups_order_exact=True,
        optimizer_steps=[5],macro_scales=dict(zip(MACRO_KEYS,ARMS[arm])),authorized_sources=sorted(changed))


def identity(path,step,arm,run_type='formal'):
    import torch
    p=torch.load(path,map_location='cpu',weights_only=False);cfg=p['config'];frozen(cfg,arm)
    assert p['completed_steps']==p['global_step']==p['scheduler']['completed_steps']==step
    assert p['scheduler_horizon']==p['scheduler']['horizon']==4868
    assert cfg['max_updates']==step and cfg['run_type']==run_type
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==STEP0_SHA
    assert p['data_cursor']==dict(next_epoch=0,next_batch=step)
    assert len(p['rng_per_rank'])==4 and all('loader_generator' in r for r in p['rng_per_rank'])
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={step}
    for state in (p['model'],p['adapter'],*[s for s in p['optimizer']['state'].values()]):
        assert all(torch.isfinite(v).all() for v in state.values())
    result=dict(passed=True,path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size,completed_steps=step,
        optimizer_counters=[step],scheduler=p['scheduler'],scaler=p['scaler'],sampler=p['sampler'],cursor=p['data_cursor'],
        rng_ranks=4,loader_generator_restorable=True,configuration=cfg,uploaded=False)
    del p;return result


def evaluator_proof():
    check=read(PROJECT/'engineering/parallel_five_eval_v1/COMPARISON.json')
    receipt=read(PROJECT/'engineering/parallel_five_eval_v1/PARALLEL_RUN.json')
    assert check['passed'] and check['metrics_exact'] and check['scores_exact']
    frozen=dict(check['unchanged_evaluator_source_sha256']);frozen['tools/eval_five_parallel.py']=receipt['scheduler_source_sha256']
    for name,digest in frozen.items():assert sha(ROOT/name)==digest,name
    return frozen


def verify_full_reference():
    branch='experiment/nested-d3-hns-full-v1'
    path='experiments/nest_clip_v1/nested_d3_hns_full_v1/FULL_RESULTS.json'
    blob=subprocess.check_output(['git','show','origin/'+branch+':'+path],cwd=ROOT)
    assert blob==(PROJECT/path).read_bytes()
    result=json.loads(blob)
    checkpoint=PROJECT/'runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1/step4868/step004868.pt'
    assert result['completed_steps']==4868 and result['evaluation_checkpoint_immutable']
    assert sha(checkpoint)==result['checkpoint_sha256']
    dump(EXP/'FULL_REFERENCE_PROVENANCE.json',dict(passed=True,branch=branch,
        commit=git('rev-parse','origin/'+branch),raw_JSON_sha256=hashlib.sha256(blob).hexdigest(),
        checkpoint=dict(path=str(checkpoint),sha256=result['checkpoint_sha256'],uploaded=False),
        raw_results=result,not_used_for500_ranking=True,not_used_for_resume=True))


def state(status,arm=None,**extra):
    dump(EXP/'QUEUE_STATE.json',dict(status=status,active_arm=arm,order=list(ARMS),pid=os.getpid(),session=os.getsid(0),
        updated_utc=now(),stop_per_arm=500,automatic_full=False,automatic_fifth_arm=False,**extra))


def prepare():
    import torch
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    torch.set_num_threads(4);require_gpu_idle({0,1,2,3})
    assert git('branch','--show-current')==BRANCH
    mother=git('rev-parse','origin/'+BASE_BRANCH)
    assert git('merge-base',mother,'HEAD')==mother
    assert not EXP.exists() and not RUN.exists(),'No overwrite or duplicate preparation'
    baseline=read(BASE_EXP/'RESULTS.json');official={}
    for name in ('config.json','RESULTS.json','TRAINING_DIAGNOSTICS.json','GRADIENT_AUDIT.json','MASK_HIERARCHY_AUDIT.json','VALIDATION.json'):
        path='experiments/nest_clip_v1/nested_d3_hns500_v1/'+name
        blob=subprocess.check_output(['git','show','origin/'+BASE_BRANCH+':'+path],cwd=ROOT)
        assert blob==(PROJECT/path).read_bytes()==(ROOT/path).read_bytes(),name
        official[name]=hashlib.sha256(blob).hexdigest()
    assert baseline['evaluation_checkpoint_immutable'] and baseline['completed_steps']==500
    assert sha(BASE_RUN/'step500/step000500.pt')==baseline['checkpoint_sha256']
    assert sha(STEP0)==STEP0_SHA
    original=torch.load(BASE_RUN/'step500/step000500.pt',map_location='cpu',weights_only=False)
    assert original['completed_steps']==500 and original['scheduler_horizon']==4868
    assert {int(s['step']) for s in original['optimizer']['state'].values()}=={500}
    protocol=original['config'];del original
    production=code_manifest();changed=[]
    for path,digest in production.items():
        previous=hashlib.sha256(subprocess.check_output(['git','show',mother+':'+path],cwd=ROOT)).hexdigest()
        if previous!=digest:changed.append(path)
    assert changed==['model/balanced_hparam_search.py'],changed
    ready=read(local.IMAGES.parent/'full-ready.json');assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    eval_sources=evaluator_proof()
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'DEFAULT_HNS_CONFIG.json',base_config())
    for arm in ARMS:
        (EXP/arm).mkdir();dump(EXP/arm/'config.json',config(arm));frozen(config(arm),arm)
    dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,mother_branch=BASE_BRANCH,mother_commit=mother,
        original_raw_results=baseline,source_JSON_sha256=official,
        original_checkpoint=dict(path=str(BASE_RUN/'step500/step000500.pt'),sha256=baseline['checkpoint_sha256'],uploaded=False),
        original_training_protocol=protocol,common0_sha256=STEP0_SHA,production_sources=production,
        only_modified_production_source=changed,evaluator_sources=eval_sources,local_only=True,
        original_full_branch='experiment/nested-d3-hns-full-v1',
        original_full_commit=git('rev-parse','origin/experiment/nested-d3-hns-full-v1')))
    proof=local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],NFS_fallback=False,
        image_root=str(local.IMAGES),raw_path=str(RUN/'local-path-proof-5000.json'),sha256=sha(RUN/'local-path-proof-5000.json')))
    activate(list(ARMS)[0]);(RUN/list(ARMS)[0]).mkdir()
    search.sampling_audit()
    dump(EXP/'PLAN.json',dict(arms={n:dict(zip(MACRO_KEYS,s)) for n,s in ARMS.items()},order=list(ARMS),
        common0_sha256=STEP0_SHA,resume=None,independent_smoke5_and_formal500=True,max_updates=500,horizon=4868,
        alignment_internal=[1.35,1.35,.3],sparsity_internal=[1,2,2],beta=[2,2],no_SG=True,old_soft_inclusion=0,
        ramp='min(1,completed BEFORE update /200)',batch_per_rank=256,world=4,workers=8,seed=0,
        evaluation='Validated native four-GPU scheduler, batch64',GPU_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        score_ranking='Score5 primary; <0.05pp single-seed differences are small, unconfirmed',
        balanced_selection='Highest Score5 with J_long3/Short4/UrbanT2I deltas >= -0.2/-0.2/-0.3pp; baseline eligible',
        stop_after_all_four=True,automatic_1217_2434_3651_4868=False,automatic_combination=False,fifth_arm=False,second_seed=False))
    state('PREPARED')
    verify_full_reference()


def worker(arm,smoke):
    from train import train_nested_semantic_mask as trainer
    activate(arm,smoke);frozen(read(EXP/arm/'config.json'),arm)
    assert '--resume' not in sys.argv
    trainer.sampling_diagnostics=search.observe_selection
    if smoke:local.worker()
    else:search.worker()


def training_command(arm,smoke=False):
    run=RUN/(arm+'.smoke5' if smoke else arm)
    return [str(PROJECT/'.venv/bin/torchrun'),'--standalone','--nproc-per-node=4','--max-restarts=0','-m',ENTRY,
        '--arm',arm,'--smoke-worker' if smoke else '--worker','--config',str(EXP/arm/'config.json'),
        '--init-state',str(STEP0),'--index-dir',str(local.INDEX),'--image-root',str(local.IMAGES),
        '--output-dir',str(run/'step500'),'--run-type','smoke' if smoke else 'formal','--max-updates','5' if smoke else '500']


def torchrun(module,*args):
    return [str(PROJECT/'.venv/bin/torchrun'),'--standalone','--nproc-per-node=4','--max-restarts=0','-m',module,*map(str,args)]


def evaluate(supervisor,arm):
    from tools.eval_five_parallel import require_gpu_idle
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short
    require_gpu_idle({0,1,2,3});evaluator_proof()
    runtime=RUN/arm;checkpoint=runtime/'step500/step000500.pt';proof=identity(checkpoint,500,arm)
    out=runtime/'evaluations';out.mkdir(exist_ok=False);bare=out/'student_step500.pt';python=str(PROJECT/'.venv/bin/python')
    supervisor.execute('export500',[python,'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),'--output',str(bare),'--expect-updates','500'])
    supervisor.execute('verify500',[python,'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),'--bare',str(bare),
        '--output',str(out/'export-check.json'),'--index-dir',str(local.INDEX),'--image-root',str(local.IMAGES)])
    check=read(out/'export-check.json');assert check['passed'] and check['strict_load'] and check['optimizer_steps']==[500]
    assert check['image_max_abs']==check['text_max_abs']==0 and check['checkpoint_sha256']==proof['sha256']
    state('EVALUATING',arm)
    supervisor.execute('five-eval500',[python,'-m','tools.eval_five_parallel','--checkpoint',str(bare),
        '--training-checkpoint',str(checkpoint),'--output-dir',str(out)])
    metrics,raw,outputs=native_metrics(out);aggregate=with_short(dict(metrics=metrics,scores=scores(metrics)))
    percent={k:100*aggregate[{'Score5':'Score5_R1','Short4':'Short4_R1'}.get(k,k)] for k in ('Score5','J_long3','J_long','Short4')}
    assert sha(checkpoint)==proof['sha256'] and sha(bare)==check['bare_sha256']
    for dataset,value in raw.items():dump(EXP/arm/'evaluations'/f'{dataset}.json',value)
    receipt=read(out/'EVAL_PARALLEL_RUN.json');assert receipt['status']=='COMPLETED'
    dump(EXP/arm/'evaluations/EVAL_PARALLEL_RUN.json',receipt)
    return dict(completed_steps=500,metrics=metrics,scores_percent=percent,checkpoint=proof,
        strict_export=check,bare=dict(path=str(bare),sha256=check['bare_sha256'],uploaded=False),
        checkpoint_unchanged=True,evaluation_wall_seconds=receipt['wall_seconds'])


def quality(result):
    q={k:result['scores_percent'][k] for k in ('Score5','J_long3','J_long','Short4')}
    q.update({f'Urban_{dr}':100*result['metrics']['Urban-1k'][dr]['R@1'] for dr in ('I2T','T2I')});return q


def compare(result,baseline):
    return dict(quality_delta_pp={k:v-quality(baseline)[k] for k,v in quality(result).items()},
        recall_delta_pp={ds:{dr:{k:100*(v-baseline['metrics'][ds][dr][k]) for k,v in rec.items()}
            for dr,rec in dirs.items()} for ds,dirs in result['metrics'].items()})


def report_arm(arm,result,supervisor):
    from recovery.nested_d3_local_search_evidence import diagnostics
    activate(arm);exp=EXP/arm;runtime=RUN/arm;steps=rows(runtime/'step500/steps.jsonl')
    assert len(steps)==500
    proof=matched_stream(steps,rows(BASE_RUN/'step500/steps.jsonl'),arm);assert proof['records']==512000
    dump(exp/'SAMPLING_PROOF.json',proof)
    diag,masks=diagnostics(steps,arm)
    # Existing diagnostics use10 as the historical alignment outer scale.
    for point in [diag['last50_views'],*diag['selected_steps'].values()]:
        for view in point.values():view['weighted_CE']*=ARMS[arm][0]/10
    keys=[k for k in steps[0] if k.startswith(('HNS_','macro_')) or k in ('V_DF_hard','V_3D_hard','lambda_h','loss')]
    means={k:statistics.fmean(float(r[k]) for r in steps[-50:]) for k in keys}
    diag['macro_and_HNS_last50']=means
    diag['macro_selected_steps']={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,500)}
    masks.update(HNS_last50=means,no_mask_to_mask_SG=True,beta=[2,2],telemetry_population='Same valid-pair population across all four arms')
    dump(exp/'TRAINING_DIAGNOSTICS.json',diag);dump(exp/'MASK_HIERARCHY_AUDIT.json',masks)
    baseline=read(BASE_EXP/'RESULTS.json');result.update(arm=arm,macro_scales=dict(zip(MACRO_KEYS,ARMS[arm])),**compare(result,baseline))
    gradient=read(exp/'GRADIENT_AUDIT.json');assert gradient['passed']
    old=read(BASE_EXP/'GRADIENT_AUDIT.json');assert old['sample_ids_sha256']==gradient['sample_ids_sha256']
    group_mapping=dict(native_visual_backbone='shared_visual_backbone',native_text_backbone='shared_text_backbone',
        text_mask_shared_pool='shared_text_mask_and_pool',visual_mask='shared_visual_mask',fusion_adapter_gate='fusion_shared_module')
    reference_gradients={}
    for group,previous_group in group_mapping.items():
        before=dict(weighted_alignment_norm=old['components']['alignment']['group_norms'][previous_group]['raw'],
            weighted_sparsity_norm=old['components']['original_sparsity']['group_norms'][previous_group]['raw'],
            weighted_hierarchy_norm=old['group_cosines'][previous_group]['hierarchy_norm'],
            total_norm=old['components']['total_training']['group_norms'][previous_group]['raw'])
        current=gradient['group_diagnostics'][group]
        reference_gradients[group]=dict(original_HNS_v1=before,delta={k:current[k]-v for k,v in before.items()},
            ratio={k:None if v==0 else current[k]/v for k,v in before.items()})
    gradient['matched_HNS_v1_gradient_comparison']=reference_gradients
    dump(exp/'GRADIENT_AUDIT.json',gradient)
    result['gradient_audit_matched_original_1024_cohort']=True
    dump(exp/'RESULTS.json',result);dump(exp/'VALIDATION.json',dict(passed=True,stream=proof,
        first5=read(runtime/'first-five-gate.json'),smoke=read(exp/'SMOKE_EVIDENCE.json'),
        acceptance=read(runtime/'step500/acceptance.json'),strict_export=result['strict_export'],gradient_audit_passed=True))
    cycles=rows(runtime/'step500/cycle_timing.jsonl');waits={}
    for rank in range(4):
        for row in rows(local.PHASE/f'rank{rank}.jsonl'):waits[row['step']]=max(waits.get(row['step'],0),row['data_wait_s'])
    systems=[r['system'] for r in rows(runtime/'resource-telemetry.jsonl')]
    stats=dict(full_cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),data_wait_seconds=distribution(list(waits.values())),
        steps_gt3s=sum(r['four_rank_max_seconds']>3 for r in cycles),steps_gt10s=sum(r['four_rank_max_seconds']>10 for r in cycles),
        peak_cgroup_memory_bytes=max(r['memory_current'] for r in systems),peak_file_cache_bytes=max(r['file'] for r in systems),
        oom_kill=max(r['memory_events'].get('oom_kill',0) for r in systems),
        GPU_peak_GiB={str(rank):max(h['peak_allocated_gib'] for r in steps for h in r['rank_health'] if h['rank']==rank) for rank in range(4)},
        commands=supervisor.commands,raw_assets=[dict(path=str(p),bytes=p.stat().st_size,uploaded=False) for p in runtime.rglob('*') if p.is_file()])
    for kind in ('io_PSI','memory_PSI'):stats[kind]={v:distribution([s[kind][v]['avg10'] for s in systems]) for v in ('some','full')}
    text=(runtime/'train500.log').read_text();errors=[t for t in ('Image failure sample=','Input/output error','Missing local sample=','CUDA out of memory','Traceback','HARD_STOP') if t in text]
    assert not errors and stats['oom_kill']==0;stats.update(true_IO_errors=errors,pod_supervisor_anomaly=False)
    dump(exp/'RUNTIME_STATS.json',stats);dump(exp/'EXPORT_VERIFICATION.json',result['strict_export'])
    dump(exp/'COMMANDS.json',supervisor.commands)
    lines=[f'# {arm}: HNS macro-loss500','',f'Macro scales: `{result["macro_scales"]}`. Independent smoke5 and fresh common0 formal500. Horizon4868; local-only.',
        'Internal alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, no SG, ramp200, fixedK3 unchanged.',
        '', '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|']
    for ds,m in result['metrics'].items():lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    lines+=['','Quality: `'+json.dumps(quality(result))+'`.','Deltas vs original HNS-v1(pp): `'+json.dumps(result['quality_delta_pp'])+'`.',
        'All30 recall deltas: RESULTS.json. Actual component gradients/cosines: GRADIENT_AUDIT.json, matched immutable first global1024 batch atstep500.',
        'Last50 CE/share/keep/IoU/violations/equality: TRAINING_DIAGNOSTICS.json and MASK_HIERARCHY_AUDIT.json.',
        'No inference mask/gate/rerank/ensemble/TTA. No long continuation or combination. Single seed500 only; <0.05pp gains unconfirmed.',
        '/root cache is ephemeral; persistent NFS originals retained. Weights/images/cache/raw large logs never uploaded.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')


def champion(qualities):
    baseline=qualities['HNS-v1'];best=max(qualities,key=lambda n:qualities[n]['Score5'])
    long=max(qualities,key=lambda n:(qualities[n]['J_long3'],qualities[n]['J_long']))
    eligible=[n for n,q in qualities.items() if q['J_long3']>=baseline['J_long3']-.2 and q['Short4']>=baseline['Short4']-.2 and q['Urban_T2I']>=baseline['Urban_T2I']-.3]
    balanced=max(eligible,key=lambda n:qualities[n]['Score5'])
    return dict(BEST_SCORE5_ARM=best,BEST_LONG_ARM=long,BEST_BALANCED_ARM=balanced,
        decision='KEEP_HNS_V1' if best=='HNS-v1' else 'CANDIDATE_FOR_HUMAN_E1_E2_VALIDATION',
        gain_is_small_unconfirmed=0<qualities[best]['Score5']-baseline['Score5']<.05,
        recommendation=best,automatic_continuation=False)


def combined():
    results={'HNS-v1':read(BASE_EXP/'RESULTS.json'),**{arm:read(EXP/arm/'RESULTS.json') for arm in ARMS}}
    qualities={n:quality(v) for n,v in results.items()};choice=champion(qualities)
    structures={'HNS-v1':read(BASE_EXP/'TRAINING_DIAGNOSTICS.json')['HNS']['last50'],
        **{arm:read(EXP/arm/'TRAINING_DIAGNOSTICS.json')['macro_and_HNS_last50'] for arm in ARMS}}
    contrasts={arm:compare(results[arm],results['HNS-v1']) for arm in ARMS}
    questions={
        'Q1_alignment':contrasts['E1-A12'],
        'Q2_sparsity':dict(retrieval=contrasts['E2-S08'],keep_changes={v:structures['E2-S08']['HNS_'+v+'_keep']-structures['HNS-v1']['HNS_'+v+'_keep'] for v in ('F','Dall','D3')},full_conclusion=False),
        'Q3_hierarchy_dose':{n:dict(scale=1 if n=='HNS-v1' else ARMS[n][2],quality=qualities[n],structure=structures[n]) for n in ('E3-H075','HNS-v1','E4-H125')},
        'Q4_metric_winners':{k:max(qualities,key=lambda n:qualities[n][k]) for k in ('Score5','J_long3','Short4')},
        'Q5_masks':{arm:dict(telemetry=structures[arm],keep_delta_vs_HNS={v:structures[arm]['HNS_'+v+'_keep']-structures['HNS-v1']['HNS_'+v+'_keep'] for v in ('F','Dall','D3')},
            common_inflation_ge5pp=all(structures[arm]['HNS_'+v+'_keep']-structures['HNS-v1']['HNS_'+v+'_keep']>=.05 for v in ('F','Dall','D3')),
            lowest_shrink_ge5pp=structures[arm]['HNS_D3_keep']-structures['HNS-v1']['HNS_D3_keep']<=-.05,
            equality_rise_ge15pp=any(structures[arm][k]-structures['HNS-v1'][k]>=.15 for k in ('HNS_DF_exact_equality_ratio','HNS_3D_exact_equality_ratio','HNS_triple_exact_equality_ratio')),
            both_coverage_gaps_le_point002=structures[arm]['HNS_gap_F_D']<=.002 and structures[arm]['HNS_gap_D_D3']<=.002,
            note='Descriptive flags, not stop conditions or proof of harmful over-regularization') for arm in ARMS},
        'Q6_gradient_evidence':{arm:read(EXP/arm/'GRADIENT_AUDIT.json')['group_diagnostics'] for arm in ARMS},
        'limitation':'One seed at500 updates; effects are early measured contrasts, not causal or full/E3 claims. Gradient norms and cosines are measured, never inferred from scalar loss size.'}
    value=dict(status='FOUR_ARMS_COMPLETED',qualities=qualities,comparisons=contrasts,models=results,
        scientific_questions=questions,**choice,all_four_completed500=True,extra_experiments=False,GPU_idle=True)
    dump(EXP/'RESULTS.json',value)
    lines=['# HNS macro-loss four-arm search500','', '| Model | lambdaA | lambdaS | lambdaH | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for n,q in qualities.items():
        scales=(10,1,1) if n=='HNS-v1' else ARMS[n]
        lines.append('| '+n+' | '+' | '.join(str(v) for v in scales)+' | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {q["Urban_I2T"]:.6f} / {q["Urban_T2I"]:.6f} |')
    lines+=['','| Model | COCO I/T | Urban I/T | Flickr I/T | DOCCI I/T | Long I/T |','|---|---|---|---|---|---|']
    for n,r in results.items():lines.append('| '+n+' | '+' | '.join(' / '.join(f'{100*r["metrics"][ds][dr]["R@1"]:.6f}' for dr in ('I2T','T2I')) for ds in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI'))+' |')
    lines+=['','Selection: `'+json.dumps(choice)+'`.','', '## Measured scientific contrasts','']
    da=contrasts['E1-A12']['quality_delta_pp']['Score5'];ds=contrasts['E2-S08']['quality_delta_pp']['Score5']
    lines+=['',f'Alignment12 changes Score5 by {da:+.6f}pp; sparsity0.8 changes it by {ds:+.6f}pp. The larger observed contrast is '+
        ('alignment' if da>ds else 'sparsity' if ds>da else 'tied')+'. A larger contrast can still be negative; differences below0.05pp are unconfirmed.',
        'Hierarchy0.75/1/1.25 Score5: '+str([qualities[n]['Score5'] for n in ('E3-H075','HNS-v1','E4-H125')])+'. Retrieval and violations are evaluated separately; no causal conclusion follows from scalar losses.',
        'A candidate is only recommended for human E1/E2 validation. No continuation is launched.']
    for q,data in questions.items():lines+=['',q+': `'+json.dumps(data)+'`.']
    lines+=['','Every arm delta and30 recalls are in RESULTS.json. A gain below0.05pp is small and unconfirmed. These500 rankings are not full-training claims.',
        'All four arms stopped500. No fifth arm, second seed, coefficient combination or1217/2434/3651/4868 training. GPU idle.',
        'Only macro coefficients changed; actual fetched default-objective equivalence, per-parameter gradients, gloo DDP and real BF16/NCCL gates passed.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    from recovery import check_stage500_publish as checker
    assert git('branch','--show-current')==BRANCH
    files=[ROOT/p for p in CODE]+[p for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.md','.json')]
    relative=[str(p.relative_to(ROOT)) for p in files]
    assert set(git('diff','--cached','--name-only').splitlines())<=set(relative)
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
    if git('diff','--cached','--name-only'):
        previous=Path.cwd();os.chdir(ROOT)
        try:checker.ALLOWED=set(relative);assert checker.inspect()['passed']
        finally:os.chdir(previous)
        subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' HNS macro-loss four-arm500'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==git('rev-parse','origin/'+BRANCH)==git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,branch=BRANCH,commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def run():
    from tools.eval_five_parallel import require_gpu_idle
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Supervisor interrupted; preserve progress')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED','No implicit retry'
        for name in GATES:assert read(EXP/name)['passed']
        assert sha(STEP0)==STEP0_SHA
        completed=[];arm=None
        try:
            for arm in ARMS:
                require_gpu_idle({0,1,2,3});activate(arm)
                from train.train_nested_semantic_mask import code_manifest
                assert code_manifest()==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
                evaluator_proof()
                runtime=RUN/arm;runtime.mkdir(exist_ok=True)
                if arm!=list(ARMS)[0]:search.sampling_audit()
                activate(arm,True);local.RUN.mkdir(exist_ok=False);local.PHASE.mkdir(exist_ok=False)
                sources=read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
                provenance=dict(source_sha256=sources,common0_sha256=STEP0_SHA,resume=None,arm=arm,
                    image_root=str(local.IMAGES),NFS_fallback=False,macro_scales=dict(zip(MACRO_KEYS,ARMS[arm])),git_head=git('rev-parse','HEAD'))
                dump(local.RUN/'launch-provenance.json',dict(provenance,run_type='smoke',stop=5))
                supervisor=local.Supervisor();state('SMOKE5_RUNNING',arm,completed_arms=completed)
                supervisor.execute('smoke5',training_command(arm,True),training=True)
                smoke_identity=identity(local.RUN/'step500/step000005.pt',5,arm,'smoke')
                accept=read(local.RUN/'step500/acceptance.json');assert accept['passed']
                stream=matched_stream(rows(local.RUN/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],arm)
                dump(EXP/arm/'SMOKE_EVIDENCE.json',dict(passed=True,independent_common0=True,stream=stream,checkpoint=smoke_identity,acceptance=accept))
                # New process, empty AdamW and common0 again; never resume smoke.
                require_gpu_idle({0,1,2,3});activate(arm);local.PHASE.mkdir(exist_ok=False)
                dump(local.RUN/'launch-provenance.json',dict(provenance,run_type='formal',stop=500))
                dump(EXP/arm/'FORMAL_PROVENANCE.json',dict(provenance,run_type='formal',stop=500))
                supervisor=local.Supervisor();state('FORMAL500_RUNNING',arm,completed_arms=completed)
                supervisor.execute('train500',training_command(arm),training=True)
                acceptance=read(runtime/'step500/acceptance.json')
                assert acceptance['passed'] and all(r['completed_updates']==r['updates_this_run']==500 and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
                assert read(runtime/'first-five-gate.json')['passed']
                proof=matched_stream(rows(runtime/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl'),arm)
                assert proof['records']==512000;dump(EXP/arm/'SAMPLING_PROOF.json',proof)
                state('GRADIENT_AUDIT',arm,completed_arms=completed)
                supervisor.execute('gradient-audit500',torchrun('recovery.hns_macro_gradient',
                    '--checkpoint',runtime/'step500/step000500.pt','--output',EXP/arm/'GRADIENT_AUDIT.json'))
                result=evaluate(supervisor,arm);report_arm(arm,result,supervisor)
                completed.append(arm);state('ARM_COMPLETED',arm,completed_arms=completed)
                if len(completed)==4:require_gpu_idle({0,1,2,3});combined();state('FOUR_COMPLETED_GPU_IDLE',arm,completed_arms=completed)
                publish()
            dump(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',arms=completed,stop=500,GPU_idle=True,finished_utc=now()))
        except BaseException as error:state('STOPPED_WITH_EVIDENCE',arm,completed_arms=completed,error=repr(error),automatic_retry=False);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),default=list(ARMS)[0])
    for flag in ('prepare','worker','smoke-worker','run','publish-setup','launch'):p.add_argument('--'+flag,action='store_true')
    args,remaining=p.parse_known_args()
    if args.worker or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.arm,args.smoke_worker)
    elif args.prepare:prepare()
    elif args.publish_setup:publish(True)
    elif args.run:run()
    elif args.launch:
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED' and not (RUN/'DETACHED_LAUNCH.json').exists()
        assert read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==git('rev-parse','HEAD')
        for name in GATES:assert read(EXP/name)['passed']
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,session=child.pid,order=list(ARMS),durable=True,log=str(RUN/'runner.log'),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',launch);print(json.dumps(launch),flush=True)
    else:p.error('Choose action')


if __name__=='__main__':main()
