"""HNS-S12 sparse-ratio two-arm500, sequential and fresh from common step0."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess

from recovery import hns_balanced_macro_fourarm as shared
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,dump,rows,sha,now

runner=shared.runner
PROJECT=shared.PROJECT
BRANCH='experiment/hns-s12-sparse-ratio-twoarm500-v1'
MOTHER='origin/experiment/hns-s12-full4868-v1'
EXP=ROOT/'experiments/nest_clip_v1/hns_s12_sparse_ratio_twoarm500_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-sparse-ratio-twoarm500-v1'
ENTRY='recovery.hns_s12_sparse_ratio_twoarm500'
ARMS={
    'E1-AlignMatched':dict(axis='sparsity',weights=[1.35,1.35,.30],mode='nested_detail_d3',
                           sparsity_weights=[2.25,2.25,.50],ratio='4.5:4.5:1'),
    'E2-Uniform':dict(axis='sparsity',weights=[1.35,1.35,.30],mode='nested_detail_d3',
                      sparsity_weights=[5./3.,5./3.,5./3.],ratio='1:1:1'),
}
REF=dict(branch='experiment/hns-static-strength-twoarm500-v1',
         exp=ROOT/'experiments/nest_clip_v1/hns_static_strength_twoarm500_v1/E1-HNS-S12',
         run=PROJECT/'runtime/SAID-nest-clip-v1/hns-static-strength-twoarm500-v1/E1-HNS-S12')
CODE=shared.CODE+('recovery/hns_s12_sparse_ratio_twoarm500.py',
                  'recovery/hns_sparse_ratio_equivalence.py','tests/test_hns_s12_sparse_ratio_twoarm500.py')
GATES=('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json','SPARSE_RATIO_EQUIVALENCE.json')
ORIGINAL_WORKER=runner.worker
ORIGINAL_REPORT=shared.report_arm
ORIGINAL_MATCH=__import__('recovery.nested_d3_local_search',fromlist=['matched_stream']).matched_stream


def config(arm):
    spec=ARMS[arm]
    cfg=runner.read(REF['exp']/'config.json')
    cfg.update(view_sparsity_weights=list(spec['sparsity_weights']),
               lambda_align=10.,lambda_sparse=1.2,lambda_hierarchy=1.)
    return cfg


def frozen(cfg,arm):
    expected=config(arm)
    assert all(cfg.get(k)==v for k,v in expected.items()), 'Frozen sparse-ratio config drift'
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['sparsity_scale']==1
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg['view_sparsity_weights']==ARMS[arm]['sparsity_weights']
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg.get('hns_beta',[2.,2.])==[2.,2.]
    assert not cfg.get('hns_detach_child',False) and cfg['workers']==8 and cfg['batch_size']==256
    assert abs(sum(cfg['view_sparsity_weights'])-5.)<1e-12


def activate(arm,smoke=False):
    assert arm in ARMS
    runner.BASE_EXP,runner.BASE_RUN=REF['exp'],REF['run']
    search=runner.search
    search.ARMS={n:dict(axis=spec['axis'],weights=spec['weights'],r=2.,mode=spec['mode'],
                         sparsity_weights=spec['sparsity_weights'],experiment_dir=str(EXP/n))
                 for n,spec in ARMS.items()}
    search.RUN_ROOT=RUN;search.EXP=EXP;search.ANCHOR_EXP=REF['exp'];search.ANCHOR_RUN=REF['run']
    search.PHASE_PREFIX='hns-s12-sparse-ratio-twoarm500-v1-'
    search.EDITED={'model/balanced_hparam_search.py'}
    search.arm_config=config;search.frozen_config=frozen;search.matched_stream=matched_stream
    search.checkpoint_invariants=checkpoint_invariants
    search.activate(arm)
    if smoke:
        runner.local.RUN=RUN/(arm+'.smoke5')
        runner.local.PHASE=runner.local.IMAGES.parent/('hns-s12-sparse-ratio-twoarm500-v1-'+arm+'-smoke5')


def checkpoint_invariants(current,reference):
    """Common-step5 state must match; only sparse coefficients/config paths differ."""
    import copy,torch
    arm=runner.search.ARM;cfg=current['config'];old=reference['config'];frozen(cfg,arm)
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for key in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size',
                'accumulation','seed','sampling_seed','shuffle_seed','workers','optimizer_groups'):
        if key in old: assert cfg[key]==old[key],key
    assert cfg['view_sparsity_weights']==ARMS[arm]['sparsity_weights']
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs=[(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs.extend((v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items())
    for new,previous in pairs:
        assert new.keys()==previous.keys()
        for key,value in new.items():
            assert value.shape==previous[key].shape and value.dtype==previous[key].dtype
            torch.testing.assert_close(value,previous[key],atol=0,rtol=0)
    return dict(passed=True,fresh_common0=True,initialization_exact=True,optimizer_groups_order_exact=True,
                optimizer_steps=[5],sparsity_coefficients_exact=True,authorized_source='model/balanced_hparam_search.py')


def matched_stream(actual,reference,arm):
    """Reuse the frozen HNS sample/LR proof and add coefficient checks."""
    proof=ORIGINAL_MATCH(actual,reference,arm)
    spec=ARMS[arm];coeff=spec['sparsity_weights']
    for row in actual:
        assert row['HNS_enabled'] is True and row['inc_weight']==row['inclusion_loss']==0
        assert row['macro_lambda_align']==10. and row['macro_lambda_sparse']==1.2 and row['macro_lambda_hierarchy']==1.
        assert row['HNS_sparse_coeff_F']==coeff[0]
        assert row['HNS_sparse_coeff_Dall']==coeff[1]
        assert row['HNS_sparse_coeff_D3']==coeff[2]
        expected=1.2*sum(c*row[f'HNS_sparse_raw_{v}']/3 for c,v in zip(coeff,('F','Dall','D3')))
        assert math.isclose(row['macro_weighted_sparse'],expected,rel_tol=5e-6,abs_tol=2e-6)
        assert math.isclose(sum(row[f'HNS_sparse_weighted_{v}'] for v in ('F','Dall','D3')),
                           row['macro_weighted_sparse'],rel_tol=5e-6,abs_tol=2e-6)
    proof.update(sparsity_coefficients_exact=True,raw_ratio_sum=5.,effective_lambda_sparse=1.2,
                 all_sample_ids_F_Dall_D3_text_tokens_indices_and_LR_exact=True)
    return proof


def prepare():
    import torch
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    torch.set_num_threads(4);require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'No overwrite or automatic retry'
    assert sha(STEP0)==STEP0_SHA
    production=code_manifest();mother=runner.git('rev-parse',MOTHER)
    changed=[]
    for path,digest in production.items():
        previous=hashlib.sha256(subprocess.check_output(['git','show',mother+':'+path],cwd=ROOT)).hexdigest()
        if digest!=previous:changed.append(path)
    assert changed==['model/balanced_hparam_search.py'],changed
    official={}
    for name in ('config.json','RESULTS.json','TRAINING_DIAGNOSTICS.json','MASK_HIERARCHY_AUDIT.json'):
        path=REF['exp']/name;relative=str(path.relative_to(ROOT))
        blob=subprocess.check_output(['git','show',REF['branch']+':'+relative],cwd=ROOT)
        assert blob==path.read_bytes(),name
        official[name]=hashlib.sha256(blob).hexdigest()
    baseline=runner.read(REF['exp']/'RESULTS.json');checkpoint=REF['run']/'step500/step000500.pt'
    assert baseline['completed_steps']==500 and baseline['checkpoint_unchanged']
    assert sha(checkpoint)==baseline['checkpoint']['sha256']
    ready=runner.read(runner.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator=runner.evaluator_proof()
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    # Run the fixed-state coefficient isolation before any GPU training and
    # carry forward only the previously passed infrastructure receipts.
    subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','recovery.hns_sparse_ratio_equivalence',
                    '--output',str(EXP/'SPARSE_RATIO_EQUIVALENCE.json')],cwd=ROOT,check=True)
    for name in ('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json'):
        source=EXP.parent.parent/'hns_static_strength_twoarm500_v1'/name
        if not source.exists(): source=REF['exp'].parent/name
        value=runner.read(source);assert value.get('passed') is True
        dump(EXP/name,value)
    dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,mother_commit=mother,production_sources=production,
        production_source_changes=[],original_HNS_branch=REF['branch'],original_HNS_commit=runner.git('rev-parse',REF['branch']),
        original_raw_results=baseline,original_JSON_sha256=official,common0_sha256=STEP0_SHA,
        original_checkpoint=dict(path=str(checkpoint),sha256=baseline['checkpoint']['sha256'],uploaded=False),
        evaluator_sources=evaluator,local_only=True))
    cfg=runner.read(REF['exp']/'config.json');cfg.update(lambda_align=10.,lambda_sparse=1.2,lambda_hierarchy=1.)
    dump(EXP/'DEFAULT_HNS_CONFIG.json',cfg)
    for arm in ARMS:
        (EXP/arm).mkdir();dump(EXP/arm/'config.json',config(arm));frozen(config(arm),arm)
    proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],image_root=str(runner.local.IMAGES),
        NFS_fallback=False,raw_path=str(RUN/'local-path-proof-5000.json'),sha256=sha(RUN/'local-path-proof-5000.json')))
    activate(next(iter(ARMS)));(RUN/next(iter(ARMS))).mkdir();runner.search.sampling_audit()
    dump(EXP/'PLAN.json',dict(order=list(ARMS),arms={n:dict(ratio=s['ratio'],raw_ratio=[1.35,1.35,.30] if n.startswith('E1') else [1,1,1],
        normalized_weights=s['sparsity_weights'],effective_coefficients=[1.2*x for x in s['sparsity_weights']]) for n,s in ARMS.items()},
        fresh_common0=True,resume=None,independent_smoke5_formal500=True,max_updates=500,horizon=4868,
        view_weights=[1.35,1.35,.3],baseline_sparsity_weights=[1,2,2],lambda_sparse=1.2,beta=[2,2],no_SG=True,soft_inclusion=0,
        ramp='min(1,completed BEFORE update/200)',fixed_K=3,seed=0,world=4,batch_per_rank=256,workers=8,
        local_only=True,NFS_fallback=False,coefficient_control='Same original HNS@500 parameters, same1024 inputs and one shared raw loss graph; no parameter updates',
        severe_mask_stop='All three valid-population Hard-ST supports entirely empty OR entirely full for five consecutive updates; ordinary density/equality changes are diagnostics only',
        automatic_full=False,third_arm=False,combinations=False,additional_seed=False,coefficient_mass=5.,
        exact_stream_records=512000,training_data='/root/said_s02_stage500/ShareGPT4V'))
    runner.state('PREPARED',arm_count=2,automatic_third_arm=False)


def endpoint_streak(logs,previous,count):
    if 'HNS_F_keep' not in logs:return None,0
    keep=[float(logs['HNS_'+v+'_keep']) for v in ('F','Dall','D3')]
    assert all(math.isfinite(v) and 0<=v<=1 for v in keep),'Nonfinite mask telemetry'
    endpoint='empty' if all(v==0 for v in keep) else 'full' if all(v==1 for v in keep) else None
    return endpoint,(count+1 if endpoint is not None and endpoint==previous else 1 if endpoint is not None else 0)


def worker(arm,smoke):
    from model.balanced_hparam_search import BalancedSearch
    original=BalancedSearch.forward;previous=None;count=0
    def guard(module,*args,**kwargs):
        nonlocal previous,count
        loss,logs=original(module,*args,**kwargs)
        previous,count=endpoint_streak(logs,previous,count)
        if count>=5:
            evidence=dict(status='TECHNICAL_HARD_MASK_DEGENERATION',endpoint=previous,consecutive_updates=count,
                arm=arm,macro_scales=ARMS[arm],updated_utc=now(),coefficients_not_adjusted=True)
            if int(os.environ.get('RANK','0'))==0:dump(runner.local.RUN/'MASK_FAILURE.json',evidence)
            raise RuntimeError('HARD_STOP: all three supports '+previous+' for five updates; preserve evidence')
        return loss,logs
    BalancedSearch.forward=guard
    ORIGINAL_WORKER(arm,smoke)


def validate_controls():
    proof=runner.read(EXP/'SPARSE_RATIO_EQUIVALENCE.json')
    assert proof['passed'] and proof['no_parameter_updates']
    assert proof['coefficient_mass']==5. and proof['effective_lambda_sparse']==1.2
    assert proof['default_hns_behavior_preserved'] and proof['alignment_hierarchy_unchanged']
    assert set(proof['arms'])==set(ARMS)
    return proof


def report_arm(arm,result,supervisor):
    from recovery.nested_d3_local_search_evidence import diagnostics
    activate(arm);exp=EXP/arm;runtime=RUN/arm
    steps=rows(runtime/'step500/steps.jsonl');assert len(steps)==500
    proof=matched_stream(steps,rows(REF['run']/'step500/steps.jsonl'),arm);assert proof['records']==512000
    dump(exp/'SAMPLING_PROOF.json',proof)
    diag,masks=diagnostics(steps,arm)
    keys=[k for k in steps[0] if k.startswith(('HNS_','macro_')) or k in ('V_DF_hard','V_3D_hard','lambda_h','loss')]
    means={k:sum(float(r[k]) for r in steps[-50:])/50 for k in keys}
    diag['macro_and_HNS_last50']=means
    diag['macro_selected_steps']={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,500)}
    masks.update(HNS_last50=means,no_mask_to_mask_SG=True,beta=[2,2],
                 sparsity_ratio=ARMS[arm]['ratio'],normalized_weights=ARMS[arm]['sparsity_weights'],
                 effective_coefficients=[1.2*x for x in ARMS[arm]['sparsity_weights']],
                 mask_density_order=all(masks.get('keep_ratios',{}).get(a,0)>=masks.get('keep_ratios',{}).get(b,0)
                                        for a,b in (('F','Dall'),('Dall','D3'))))
    base=runner.read(REF['exp']/'RESULTS.json')
    result.update(arm=arm,sparsity_ratio=ARMS[arm]['ratio'],normalized_weights=ARMS[arm]['sparsity_weights'],
                  effective_coefficients=[1.2*x for x in ARMS[arm]['sparsity_weights']],
                  **runner.compare(result,base))
    gradient=runner.read(exp/'GRADIENT_AUDIT.json');assert gradient['passed']
    fixed=validate_controls();gradient['sparse_ratio_equivalence']=fixed
    gradient['state_vs_coefficient_warning']='Trained-state gradients are reported separately from fixed-state coefficient equivalence.'
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),('GRADIENT_AUDIT',gradient)]:dump(exp/(name+'.json'),value)
    dump(exp/'VALIDATION.json',dict(passed=True,stream=proof,first5=runner.read(runtime/'first-five-gate.json'),
        smoke=runner.read(exp/'SMOKE_EVIDENCE.json'),acceptance=runner.read(runtime/'step500/acceptance.json'),
        strict_export=result['strict_export'],gradient_audit_passed=True))
    dump(exp/'RUNTIME_STATS.json',dict(commands=supervisor.commands,uploaded=False,
        source='sequential fresh-common0 local-only training'))
    lines=[f'# {arm}: HNS-S12 sparse-ratio @500','',
        f'Fresh common0, smoke5 and formal500. Normalized ratio `{ARMS[arm]["ratio"]}`; weights `{ARMS[arm]["sparsity_weights"]}`; effective coefficients `{[1.2*x for x in ARMS[arm]["sparsity_weights"]]}`.',
        'HNS beta2/2, Hard-ST, no-SG, K3, lambda_align10, lambda_sparse1.2, lambda_hierarchy1, original ramp200.',
        '', '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for ds,m in result['metrics'].items():lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    lines+=['','Scores: `'+json.dumps(runner.quality(result))+'`.',
        'Delta vs HNS-S12 baseline(pp): `'+json.dumps(result['quality_delta_pp'])+'`.',
        'Weighted per-view sparsity contribution and HNS hierarchy telemetry are in TRAINING_DIAGNOSTICS.json; gradients/cosines are in GRADIENT_AUDIT.json.',
        'All512000 sample IDs, F/Dall/D3 text/tokens/indices and LR matched. Native full-caption inference only.',
        'Single seed500 exploratory result; no continuation, third arm, coefficient combination or other K.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')


def selection(qualities):
    best=max(qualities,key=lambda n:qualities[n]['Score5'])
    gain=qualities[best]['Score5']-qualities['HNS-S12']['Score5']
    return dict(BEST_OVERALL_500=best,RECOMMENDED_NEXT_VALIDATION=best if gain>0 else 'KEEP_HNS_S12',
        gain_pp=gain,weak_single_seed_signal=0<gain<.05,automatic_continuation=False)


def combined():
    results={'HNS-S12':runner.read(REF['exp']/'RESULTS.json'),**{n:runner.read(EXP/n/'RESULTS.json') for n in ARMS}}
    q={n:runner.quality(v) for n,v in results.items()};choice=selection(q)
    masks={n:runner.read(EXP/n/'MASK_HIERARCHY_AUDIT.json') for n in ARMS}
    controls=validate_controls();grads={n:runner.read(EXP/n/'GRADIENT_AUDIT.json') for n in ARMS}
    questions=dict(
        Q1_alignment_matched=dict(Score5_delta_pp=results['E1-AlignMatched']['quality_delta_pp']['Score5'],structure=masks['E1-AlignMatched']),
        Q2_uniform=dict(Score5_delta_pp=results['E2-Uniform']['quality_delta_pp']['Score5'],structure=masks['E2-Uniform']),
        Q3_structure_vs_retrieval=dict(retrieval_delta={n:results[n]['quality_delta_pp'] for n in ARMS},structure=masks,lower_violation_does_not_prove_retrieval_gain=True),
        Q4_short_long={n:{k:results[n]['quality_delta_pp'][k] for k in ('Short4','J_long3','J_long')} for n in ARMS},
        Q5_gradient_evidence=dict(pure_coefficient_controls=controls,actual_trained_states={n:grads[n].get('group_diagnostics',{}) for n in ARMS},
            warning='Coefficient magnitude and learned-state gradient changes must not be conflated.'),
        Q6_future_candidate=choice)
    comparisons={n:runner.compare(results[n],results['HNS-S12']) for n in ARMS}
    dump(EXP/'RESULTS.json',dict(status='TWO_ARMS_COMPLETED',all_two_completed500=True,models=results,qualities=q,
        comparisons=comparisons,scientific_questions=questions,**choice,GPU_idle=True,extra_experiments=False))
    dump(EXP/'SCIENTIFIC_DIAGNOSTICS.json',questions)
    lines=['# HNS-S12 sparse-ratio two-arm @500','',
        '| Model | Sparsity ratio | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for n,v in q.items():
        ratio='1:2:2' if n=='HNS-S12' else ARMS[n]['ratio']
        lines.append('| '+n+' | '+ratio+' | '+' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {v["Urban_I2T"]:.3f} / {v["Urban_T2I"]:.3f} | {(v["Urban_I2T"]+v["Urban_T2I"])/2:.3f} |')
    lines+=['','| Arm | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T/T2I |','|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        d=comparisons[arm]['quality_delta_pp'];lines.append('| '+arm+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {d["Urban_I2T"]:+.3f} / {d["Urban_T2I"]:+.3f} |')
    lines+=['','Selection: `'+json.dumps(choice)+'`.','',
        'Q1: Alignment-matched retrieval/structure: '+json.dumps(questions['Q1_alignment_matched'])+'.',
        'Q2: Uniform retrieval/structure: '+json.dumps(questions['Q2_uniform'])+'.',
        'Q3: Retrieval versus structure: '+json.dumps(questions['Q3_structure_vs_retrieval'])+'. Lower violations alone do not imply retrieval gains.',
        'Q4: Short4/long deltas '+json.dumps(questions['Q4_short_long'])+'.',
        'Q5: Gradient norms/cosines and fixed-state coefficient controls are in SCIENTIFIC_DIAGNOSTICS.json and GRADIENT_AUDIT.json. Learned-state gradients are separate.',
        'Q6: '+choice['RECOMMENDED_NEXT_VALIDATION']+'. Single-seed improvement below0.05pp is a weak signal, not full/E3 evidence.',
        'All30 per-arm recalls and deltas are in RESULTS.json. Both arms fresh-common0, exactly500 updates, local-only. GPU idle.',
        'No third arm, H2/H8, coefficient combinations, epoch-switching, new seed, new loss or1217/2434/3651/4868 continuation.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    from recovery import check_stage500_publish as checker
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
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' HNS static-strength two-arm500'],cwd=ROOT,check=True)
    head=runner.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==runner.git('rev-parse','origin/'+BRANCH)==runner.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,branch=BRANCH,commit=head,
        remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def run():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Supervisor interrupted; preserve progress')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert runner.read(EXP/'QUEUE_STATE.json')['status']=='PREPARED','No implicit retry'
        for name in GATES:assert runner.read(EXP/name)['passed']
        validate_controls();assert sha(STEP0)==STEP0_SHA
        completed=[];arm=None
        try:
            for arm in ARMS:
                require_gpu_idle({0,1,2,3});activate(arm)
                assert code_manifest()==runner.read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
                runner.evaluator_proof();runtime=RUN/arm;runtime.mkdir(exist_ok=True)
                if arm!=next(iter(ARMS)):runner.search.sampling_audit()
                activate(arm,True);runner.local.RUN.mkdir(exist_ok=False);runner.local.PHASE.mkdir(exist_ok=False)
                provenance=dict(source_sha256=code_manifest(),common0_sha256=STEP0_SHA,resume=None,arm=arm,
                    image_root=str(runner.local.IMAGES),NFS_fallback=False,ratio=ARMS[arm]['ratio'],
                    normalized_weights=ARMS[arm]['sparsity_weights'],effective_coefficients=[1.2*x for x in ARMS[arm]['sparsity_weights']],
                    git_head=runner.git('rev-parse','HEAD'))
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='smoke',stop=5))
                supervisor=runner.local.Supervisor();runner.state('SMOKE5_RUNNING',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                supervisor.execute('smoke5',runner.training_command(arm,True),training=True)
                identity=runner.identity(runner.local.RUN/'step500/step000005.pt',5,arm,'smoke')
                accept=runner.read(runner.local.RUN/'step500/acceptance.json');assert accept['passed']
                stream=matched_stream(rows(runner.local.RUN/'step500/steps.jsonl'),rows(REF['run']/'step500/steps.jsonl')[:5],arm)
                dump(EXP/arm/'SMOKE_EVIDENCE.json',dict(passed=True,independent_common0=True,checkpoint=identity,acceptance=accept,stream=stream,coefficients_unchanged=True))
                require_gpu_idle({0,1,2,3});activate(arm);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='formal',stop=500))
                dump(EXP/arm/'FORMAL_PROVENANCE.json',dict(provenance,run_type='formal',stop=500))
                supervisor=runner.local.Supervisor();runner.state('FORMAL500_RUNNING',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                supervisor.execute('train500',runner.training_command(arm),training=True)
                accept=runner.read(runtime/'step500/acceptance.json')
                assert accept['passed'] and all(r['completed_updates']==r['updates_this_run']==500 and r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
                assert runner.read(runtime/'first-five-gate.json')['passed']
                proof=matched_stream(rows(runtime/'step500/steps.jsonl'),rows(REF['run']/'step500/steps.jsonl'),arm)
                assert proof['records']==512000;dump(EXP/arm/'SAMPLING_PROOF.json',proof)
                runner.state('GRADIENT_AUDIT',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                supervisor.execute('gradient-audit500',runner.torchrun('recovery.hns_macro_gradient','--checkpoint',runtime/'step500/step000500.pt','--output',EXP/arm/'GRADIENT_AUDIT.json'))
                result=runner.evaluate(supervisor,arm);report_arm(arm,result,supervisor)
                completed.append(arm);runner.state('ARM_COMPLETED',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                if len(completed)==len(ARMS):
                    assert len(completed)==2;require_gpu_idle({0,1,2,3});combined()
                    runner.state('TWO_COMPLETED_GPU_IDLE',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                publish()
            dump(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',arms=completed,stop=500,GPU_idle=True,finished_utc=now(),third_arm=False))
        except BaseException as error:
            runner.state('STOPPED_WITH_EVIDENCE',arm,completed_arms=completed,error=repr(error),automatic_retry=False,automatic_third_arm=False)
            raise


def configure():
    shared.BRANCH=BRANCH;shared.EXP=EXP;shared.RUN=RUN;shared.ENTRY=ENTRY;shared.ARMS=ARMS
    shared.METHODS={n:'HNS' for n in ARMS};shared.REFERENCES={'HNS':REF};shared.CODE=CODE;shared.GATES=GATES
    shared.activate=activate;shared.config=config;shared.configure()
    runner.config=config;runner.frozen=frozen;runner.activate=activate;runner.matched_stream=matched_stream
    runner.prepare=prepare;runner.run=run;runner.worker=worker;runner.report_arm=report_arm;runner.combined=combined;runner.publish=publish


def main():
    configure();runner.main()


if __name__=='__main__':main()
