"""Sequential HNS-A8 / Balanced-A8,S12,H125; fresh smoke5 then fresh500."""
import copy
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from recovery import hns_macro_fourarm as runner
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,dump,rows,sha,now,distribution

PROJECT=runner.PROJECT
BRANCH='experiment/hns-balanced-macro-fourarm500-v1'
MOTHER='origin/experiment/hns-macro-loss-fourarm500-v1'
EXP=ROOT/'experiments/nest_clip_v1/hns_balanced_macro_fourarm500_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-balanced-macro-fourarm500-v1'
ENTRY='recovery.hns_balanced_macro_fourarm'
ARMS={'E1-HNS-A8':(8.,1.,1.),'E2-Balanced-A8':(8.,1.,1.),
      'E3-Balanced-S12':(10.,1.2,1.),'E4-Balanced-H125':(10.,1.,1.25)}
METHODS={n:('HNS' if n.startswith('E1') else 'Balanced') for n in ARMS}
REFERENCES={
    'HNS':dict(branch='origin/experiment/nested-d3-hard-nested-sparsity500-v1',
        exp=ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1',
        run=PROJECT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS'),
    'Balanced':dict(branch='origin/experiment/nested-detail-d3-balanced-full-v1',
        exp=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1',
        run=PROJECT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007')}
MACRO_KEYS=runner.MACRO_KEYS
ORIGINAL_STREAM=runner.OLD_MATCH
GATES=('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json',
       'DEFAULT_HNS_GRADIENT_AUDIT.json','DEFAULT_BALANCED_GRADIENT_AUDIT.json')
CODE=runner.CODE+('recovery/hns_balanced_macro_fourarm.py','recovery/hns_balanced_macro_equivalence.py',
                  'tests/test_hns_balanced_macro.py','tests/test_hns_balanced_macro_runner.py')


def config(arm):
    return dict(runner.read(REFERENCES[METHODS[arm]]['exp']/'config.json'),**dict(zip(MACRO_KEYS,ARMS[arm])))


def frozen(cfg,arm):
    assert all(cfg.get(k)==v for k,v in config(arm).items()),'Frozen arm drift'
    assert cfg['sparsity_scale']==1 and cfg['view_weights']==[1.35,1.35,.3]
    assert cfg.get('view_sparsity_weights',[1,2,2])==[1,2,2]
    assert cfg.get('hns_enabled',False)==(METHODS[arm]=='HNS')
    assert cfg['inclusion_max']==(0 if METHODS[arm]=='HNS' else 1)
    assert cfg['inclusion_hierarchy']=='detail_chain' and cfg['sampling_mode']=='nested_detail_d3'
    assert cfg['workers']==8 and cfg['batch_size']==256 and cfg['world_size']==4
    assert cfg.get('hns_beta',[2,2])==[2,2] and not cfg.get('hns_detach_child',False)
    assert not cfg.get('hns_half_after500',False)


def activate(arm,smoke=False):
    reference=REFERENCES[METHODS[arm]]
    runner.BASE_EXP,runner.BASE_RUN=reference['exp'],reference['run']
    search=runner.search
    search.ARMS={n:dict(axis='macro',weights=[1.35,1.35,.3],r=2.,mode='nested_detail_d3',
        experiment_dir=str(EXP/n)) for n in ARMS}
    search.RUN_ROOT=RUN;search.EXP=EXP;search.ANCHOR_EXP=runner.BASE_EXP;search.ANCHOR_RUN=runner.BASE_RUN
    search.PHASE_PREFIX='hns-balanced-macro-fourarm500-v1-'
    search.EDITED={'model/balanced_hparam_search.py'}
    search.arm_config=config;search.frozen_config=frozen;search.matched_stream=matched_stream
    search.checkpoint_invariants=checkpoint_first5
    search.activate(arm)
    if smoke:
        runner.local.RUN=RUN/(arm+'.smoke5')
        runner.local.PHASE=runner.local.IMAGES.parent/('hns-balanced-macro-fourarm500-v1-'+arm+'-smoke5')


def matched_stream(actual,reference,arm):
    proof=ORIGINAL_STREAM(actual,reference,arm)
    for row in actual:
        ramp=min(1.,(row['step']-1)/200.)
        if METHODS[arm]=='HNS':
            assert row['HNS_enabled'] and row['inc_weight']==row['inclusion_loss']==0
            assert row['lambda_h']==ramp
        else:
            assert not row.get('HNS_enabled',False) and 'HNS_surcharge' not in row
            assert row['inclusion_enabled'] and row['inc_weight']==ramp
            assert math.isclose(row['macro_raw_hierarchy'],row['inc'],rel_tol=4e-6,abs_tol=2e-6)
        a,s,h=ARMS[arm]
        assert tuple(row['macro_'+k] for k in MACRO_KEYS)==(a,s,h)
        expected=(a*row['macro_raw_align'],s*row['macro_raw_sparse'],h*ramp*row['macro_raw_hierarchy'])
        weighted=[row['macro_weighted_'+n] for n in ('align','sparse','hierarchy')]
        for x,y in zip(expected,weighted):assert math.isclose(x,y,rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(row['loss'],sum(weighted),rel_tol=4e-6,abs_tol=2e-6)
    proof.update(hierarchy_method=METHODS[arm],only_macro_scales_changed=True,
        IDs_F_Dall_D3_text_tokens_indices_and_LR_exact=True)
    return proof


def checkpoint_first5(payload,reference):
    import torch
    arm=runner.search.ARM;cfg=payload['config'];old=reference['config'];frozen(cfg,arm)
    assert payload['completed_steps']==5 and payload['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size',
              'accumulation','seed','sampling_seed','shuffle_seed','workers','optimizer_groups'):
        if k in old:assert cfg[k]==old[k],k
    runtime=copy.deepcopy(cfg['runtime_model'])
    for k in MACRO_KEYS:runtime['search_hparams'].pop(k,None)
    assert runtime==old['runtime_model'],(runtime,old['runtime_model'])
    assert cfg['code_sha256']==runner.read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    assert payload['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert payload['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={5}
    pairs=[(payload['model'],reference['model']),(payload['adapter'],reference['adapter'])]
    pairs.extend((v,reference['optimizer']['state'][k]) for k,v in payload['optimizer']['state'].items())
    for new,previous in pairs:
        assert new.keys()==previous.keys()
        for k,v in new.items():assert v.shape==previous[k].shape and v.dtype==previous[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,fresh_common0=True,optimizer_groups_order_exact=True,initialization_exact=True,
        optimizer_steps=[5],hierarchy_method=METHODS[arm],macro_scales=dict(zip(MACRO_KEYS,ARMS[arm])))


def prepare():
    import torch
    from train.train_nested_semantic_mask import code_manifest
    from tools.eval_five_parallel import require_gpu_idle
    torch.set_num_threads(4);require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'Refuse duplicate/overwrite'
    assert sha(STEP0)==STEP0_SHA
    production=code_manifest();mother=runner.git('rev-parse',MOTHER)
    changed=[p for p,d in production.items() if d!=hashlib.sha256(subprocess.check_output(['git','show',mother+':'+p],cwd=ROOT)).hexdigest()]
    assert changed==['model/balanced_hparam_search.py'],changed
    baselines={}
    for method,ref in REFERENCES.items():
        official={}
        for name in ('config.json','RESULTS.json','TRAINING_DIAGNOSTICS.json','MASK_HIERARCHY_AUDIT.json'):
            p=ref['exp']/name;relative=str(p.relative_to(ROOT))
            blob=subprocess.check_output(['git','show',ref['branch']+':'+relative],cwd=ROOT)
            assert blob==p.read_bytes()==(PROJECT/relative).read_bytes(),(method,name)
            official[name]=hashlib.sha256(blob).hexdigest()
        result=runner.read(ref['exp']/'RESULTS.json');checkpoint=ref['run']/'step500/step000500.pt'
        assert result['completed_steps']==500 and result['evaluation_checkpoint_immutable']
        assert sha(checkpoint)==result['checkpoint_sha256']
        baselines[method]=dict(branch=ref['branch'],commit=runner.git('rev-parse',ref['branch']),
            original_raw_results=result,source_JSON_sha256=official,
            checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),uploaded=False))
    historical=ROOT/'experiments/nest_clip_v1/hns_macro_loss_fourarm500_v1/RESULTS.json'
    blob=subprocess.check_output(['git','show',MOTHER+':'+str(historical.relative_to(ROOT))],cwd=ROOT)
    assert blob==historical.read_bytes()
    ready=runner.read(runner.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator=runner.evaluator_proof()
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,mother_commit=mother,baselines=baselines,
        production_sources=production,only_modified_production_source=changed,evaluator_sources=evaluator,
        previous_macro_search=runner.read(historical),previous_macro_search_sha256=sha(historical),
        common0_sha256=STEP0_SHA,local_only=True))
    for method,ref in REFERENCES.items():
        cfg=runner.read(ref['exp']/'config.json');cfg.update(lambda_align=10.,lambda_sparse=1.,lambda_hierarchy=1.)
        dump(EXP/('DEFAULT_'+method.upper()+'_CONFIG.json'),cfg)
    for arm in ARMS:
        (EXP/arm).mkdir();dump(EXP/arm/'config.json',config(arm));frozen(config(arm),arm)
    proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],NFS_fallback=False,
        image_root=str(runner.local.IMAGES),raw_path=str(RUN/'local-path-proof-5000.json'),sha256=sha(RUN/'local-path-proof-5000.json')))
    activate(next(iter(ARMS)));(RUN/next(iter(ARMS))).mkdir();runner.search.sampling_audit()
    dump(EXP/'PLAN.json',dict(order=list(ARMS),arms={n:dict(method=METHODS[n],**dict(zip(MACRO_KEYS,v))) for n,v in ARMS.items()},
        fresh_common0=True,resume=None,independent_smoke5_and_formal500=True,max_updates=500,horizon=4868,
        view_weights=[1.35,1.35,.3],sparsity_weights=[1,2,2],fixed_K=3,ramp='min(1,completed_before_update/200)',
        HNS='Hard-ST ReLU child-parent; beta2/2; no-SG; old soft inclusion0',
        Balanced='Soft detached-child detail-chain inclusion; no HNS; inclusion_max1 scaled once',
        local_only=True,NFS_fallback=False,world=4,batch_per_rank=256,workers=8,seed=0,
        GPU_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        stop_after_four=True,automatic_full=False,fifth_arm=False,automatic_combination=False,second_seed=False))
    runner.state('PREPARED')


def report_arm(arm,result,supervisor):
    from recovery.nested_d3_local_search_evidence import diagnostics
    activate(arm);exp=EXP/arm;runtime=RUN/arm;steps=rows(runtime/'step500/steps.jsonl');assert len(steps)==500
    proof=matched_stream(steps,rows(runner.BASE_RUN/'step500/steps.jsonl'),arm);assert proof['records']==512000
    diag,masks=diagnostics(steps,arm)
    for point in [diag['last50_views'],*diag['selected_steps'].values()]:
        for view in point.values():view['weighted_CE']*=ARMS[arm][0]/10
    diag['weighted_formula']='lambdaA/3 * sum(w_view * directional_summed_CE)'
    keys=[k for k in steps[0] if k.startswith(('HNS_','macro_')) or k in ('inc','inc_weight','inclusion_loss','lambda_h','loss')]
    means={k:statistics.fmean(float(r[k]) for r in steps[-50:]) for k in keys}
    diag['macro_last50']=means
    diag['macro_selected_steps']={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,500)}
    masks.update(hierarchy_method=METHODS[arm],no_mask_to_mask_SG=METHODS[arm]=='HNS',
        detached_child=METHODS[arm]=='Balanced',macro_last50=means,
        keep_ratios={v:diag['last50_views'][v]['keep_ratio'] for v in ('F','Dall','D3')})
    old_masks=runner.read(runner.BASE_EXP/'MASK_HIERARCHY_AUDIT.json')['last50']
    old_views=runner.read(runner.BASE_EXP/'TRAINING_DIAGNOSTICS.json')['last50_views']
    masks['delta_vs_own_baseline']={k:masks[k]-v for k,v in old_masks.items() if k in masks}
    masks['keep_delta_vs_own_baseline']={v:masks['keep_ratios'][v]-old_views[v]['keep_ratio'] for v in old_views}
    diag['baseline_last50_views']=old_views
    baseline=runner.read(runner.BASE_EXP/'RESULTS.json');hns=runner.read(REFERENCES['HNS']['exp']/'RESULTS.json')
    result.update(arm=arm,hierarchy_method=METHODS[arm],macro_scales=dict(zip(MACRO_KEYS,ARMS[arm])),
        **runner.compare(result,baseline),comparison_vs_HNS_v1=runner.compare(result,hns))
    gradient=runner.read(exp/'GRADIENT_AUDIT.json');assert gradient['passed']
    old=runner.read(EXP/('DEFAULT_'+METHODS[arm].upper()+'_GRADIENT_AUDIT.json'))
    assert old['sample_ids_sha256']==gradient['sample_ids_sha256']
    gradient['matched_baseline_comparison']={g:dict(baseline=old['group_diagnostics'][g],
        current=gradient['group_diagnostics'][g],delta={k:gradient['group_diagnostics'][g][k]-v
            for k,v in old['group_diagnostics'][g].items() if isinstance(v,(float,int))}) for g in old['group_diagnostics']}
    gradient['baseline_view_gradients']=old['view_gradients']
    result['gradient_audit_matched_baseline_1024_cohort']=True
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('GRADIENT_AUDIT',gradient),('SAMPLING_PROOF',proof),('EXPORT_AUDIT',result['strict_export'])]:
        dump(exp/(name+'.json'),value)
    dump(exp/'VALIDATION.json',dict(passed=True,stream=proof,first5=runner.read(runtime/'first-five-gate.json'),
        smoke=runner.read(exp/'SMOKE_EVIDENCE.json'),acceptance=runner.read(runtime/'step500/acceptance.json'),
        strict_export=result['strict_export'],gradient_audit_passed=True))
    cycles=rows(runtime/'step500/cycle_timing.jsonl');waits={}
    for rank in range(4):
        for row in rows(runner.local.PHASE/f'rank{rank}.jsonl'):waits[row['step']]=max(waits.get(row['step'],0),row['data_wait_s'])
    systems=[r['system'] for r in rows(runtime/'resource-telemetry.jsonl')]
    stats=dict(full_cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),
        data_wait_seconds=distribution(list(waits.values())),
        steps_gt3s=sum(r['four_rank_max_seconds']>3 for r in cycles),steps_gt10s=sum(r['four_rank_max_seconds']>10 for r in cycles),
        peak_cgroup_memory_bytes=max(r['memory_current'] for r in systems),peak_file_cache_bytes=max(r['file'] for r in systems),
        oom_kill=max(r['memory_events'].get('oom_kill',0) for r in systems),
        GPU_peak_GiB={str(rank):max(h['peak_allocated_gib'] for r in steps for h in r['rank_health'] if h['rank']==rank) for rank in range(4)},
        commands=supervisor.commands,raw_assets=[dict(path=str(p),bytes=p.stat().st_size,uploaded=False) for p in runtime.rglob('*') if p.is_file()])
    for kind in ('io_PSI','memory_PSI'):stats[kind]={v:distribution([s[kind][v]['avg10'] for s in systems]) for v in ('some','full')}
    text=(runtime/'train500.log').read_text();errors=[t for t in ('Image failure sample=','Input/output error','Missing local sample=','CUDA out of memory','Traceback','HARD_STOP') if t in text]
    assert not errors and stats['oom_kill']==0;stats.update(true_IO_errors=errors,pod_supervisor_anomaly=False)
    dump(exp/'RUNTIME_STATS.json',stats);dump(exp/'COMMANDS.json',supervisor.commands)
    lines=[f'# {arm}: {METHODS[arm]} macro-loss @500','',
        f'Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: {ARMS[arm]}.',
        'View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.',
        'Hierarchy: '+('Hard-ST hard support, beta2/2, no-SG, soft inclusion0.' if METHODS[arm]=='HNS' else 'Soft probability detached-child adjacent chain; HNS disabled; inclusion_max1 scaled once.'),
        '', '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for ds,m in result['metrics'].items():lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    lines+=['','Scores: `'+json.dumps(runner.quality(result))+'`.',
        'Delta vs own baseline(pp): `'+json.dumps(result['quality_delta_pp'])+'`.',
        'Delta vs HNS-v1(pp): `'+json.dumps(result['comparison_vs_HNS_v1']['quality_delta_pp'])+'`.',
        'All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.',
        'Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.',
        'Single seed500; gains below0.05pp are weak signals. No automatic full, fifth arm, combination or additional seed.',
        '/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')


def selection(qualities):
    hns=max(('HNS-v1','E1-HNS-A8'),key=lambda n:qualities[n]['Score5'])
    balanced=max(('D3 Balanced','E2-Balanced-A8','E3-Balanced-S12','E4-Balanced-H125'),key=lambda n:qualities[n]['Score5'])
    best=max(qualities,key=lambda n:qualities[n]['Score5'])
    gain=qualities[best]['Score5']-qualities['HNS-v1']['Score5']
    return dict(BEST_HNS_ARM=hns,BEST_BALANCED_ARM=balanced,BEST_OVERALL_500=best,
        RECOMMENDED_NEXT_VALIDATION=best if gain>0 else 'KEEP_HNS_V1',
        overall_gain_pp=gain,weak_single_seed_signal=0<gain<.05,automatic_continuation=False)


def combined():
    results={'HNS-v1':runner.read(REFERENCES['HNS']['exp']/'RESULTS.json'),
        'D3 Balanced':runner.read(REFERENCES['Balanced']['exp']/'RESULTS.json'),
        **{arm:runner.read(EXP/arm/'RESULTS.json') for arm in ARMS}}
    q={n:runner.quality(v) for n,v in results.items()};choice=selection(q)
    comparisons={arm:dict(own_baseline=runner.compare(results[arm],results['HNS-v1' if METHODS[arm]=='HNS' else 'D3 Balanced']),
        HNS_v1=runner.compare(results[arm],results['HNS-v1']),
        weak_own_baseline_signal=0<results[arm]['quality_delta_pp']['Score5']<.05) for arm in ARMS}
    value=dict(status='FOUR_ARMS_COMPLETED',all_four_completed500=True,models=results,qualities=q,
        comparisons=comparisons,**choice,GPU_idle=True,extra_experiments=False,
        diagnostics={arm:runner.read(EXP/arm/'GRADIENT_AUDIT.json')['group_diagnostics'] for arm in ARMS})
    dump(EXP/'RESULTS.json',value)
    lines=['# HNS + D3 Balanced macro-loss four-arm search @500','',
        '| Model | lambdaA | lambdaS | lambdaH | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    order=['HNS-v1','E1-HNS-A8','D3 Balanced','E2-Balanced-A8','E3-Balanced-S12','E4-Balanced-H125']
    for n in order:
        v=q[n];scales=ARMS.get(n,(10,1,1));lines.append('| '+n+' | '+' | '.join(str(s) for s in scales)+' | '+
            ' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {v["Urban_I2T"]:.3f} / {v["Urban_T2I"]:.3f} |')
    lines+=['','| Arm | Baseline | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T/T2I | Delta vs HNS Score5 |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for arm in ARMS:
        d=comparisons[arm]['own_baseline']['quality_delta_pp'];h=comparisons[arm]['HNS_v1']['quality_delta_pp']
        lines.append('| '+arm+' | '+METHODS[arm]+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {d["Urban_I2T"]:+.3f} / {d["Urban_T2I"]:+.3f} | {h["Score5"]:+.6f} |')
    lines+=['','Selection: `'+json.dumps(choice)+'`.',
        'Every dataset/direction R@1/5/10 and all deltas are in per-arm RESULTS.json and the aggregate models/comparisons.',
        'Single seed; improvement below0.05pp is a weak signal.500 rankings need not scale toE3/full training.',
        'All arms stopped500; GPUs idle. No fifth arm, combinations, seed sweep or1217/2434/3651/4868 continuation.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    from recovery import check_stage500_publish as checker
    import os
    assert runner.git('branch','--show-current')==BRANCH
    files=[ROOT/p for p in CODE]+[p for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.md','.json')]
    relative=[str(p.relative_to(ROOT)) for p in files]
    assert set(runner.git('diff','--cached','--name-only').splitlines())<=set(relative)
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
    if runner.git('diff','--cached','--name-only'):
        previous=Path.cwd();os.chdir(ROOT)
        try:checker.ALLOWED=set(relative);assert checker.inspect()['passed']
        finally:os.chdir(previous)
        subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' HNS and Balanced macro-loss four-arm500'],cwd=ROOT,check=True)
    head=runner.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==runner.git('rev-parse','origin/'+BRANCH)==runner.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,branch=BRANCH,
        commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def configure():
    runner.BRANCH=BRANCH;runner.EXP=EXP;runner.RUN=RUN;runner.ENTRY=ENTRY;runner.ARMS=ARMS
    runner.CODE=CODE;runner.GATES=GATES;runner.config=config;runner.frozen=frozen
    runner.activate=activate;runner.matched_stream=matched_stream;runner.checkpoint_first5=checkpoint_first5
    runner.prepare=prepare;runner.report_arm=report_arm;runner.combined=combined;runner.publish=publish


def main():
    configure();runner.main()


if __name__=='__main__':main()
