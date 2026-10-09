"""Exactly two independent HNS500 arms. No inherited four-arm controller."""
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
BRANCH='experiment/hns-static-strength-twoarm500-v1'
MOTHER='origin/experiment/hns-balanced-macro-fourarm500-v1'
EXP=ROOT/'experiments/nest_clip_v1/hns_static_strength_twoarm500_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-static-strength-twoarm500-v1'
ENTRY='recovery.hns_static_strength_twoarm'
ARMS={'E1-HNS-S12':(10.,1.2,1.),'E2-HNS-H4':(10.,1.,4.)}
REF=shared.REFERENCES['HNS']
CODE=shared.CODE+('recovery/hns_static_strength_twoarm.py','recovery/hns_static_strength_equivalence.py',
                  'tests/test_hns_static_strength_twoarm.py')
GATES=('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json','DEFAULT_HNS_GRADIENT_AUDIT.json')
ORIGINAL_WORKER=runner.worker
ORIGINAL_REPORT=shared.report_arm


def config(arm):
    return dict(runner.read(REF['exp']/'config.json'),**dict(zip(runner.MACRO_KEYS,ARMS[arm])))


def activate(arm,smoke=False):
    assert arm in ARMS
    runner.BASE_EXP,runner.BASE_RUN=REF['exp'],REF['run']
    search=runner.search
    search.ARMS={n:dict(axis='macro',weights=[1.35,1.35,.3],r=2.,mode='nested_detail_d3',experiment_dir=str(EXP/n)) for n in ARMS}
    search.RUN_ROOT=RUN;search.EXP=EXP;search.ANCHOR_EXP=REF['exp'];search.ANCHOR_RUN=REF['run']
    search.PHASE_PREFIX='hns-static-strength-twoarm500-v1-'
    search.arm_config=shared.config;search.frozen_config=shared.frozen;search.matched_stream=shared.matched_stream
    search.checkpoint_invariants=shared.checkpoint_first5
    search.activate(arm)
    if smoke:
        runner.local.RUN=RUN/(arm+'.smoke5')
        runner.local.PHASE=runner.local.IMAGES.parent/('hns-static-strength-twoarm500-v1-'+arm+'-smoke5')


def prepare():
    import torch
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    torch.set_num_threads(4);require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'No overwrite or automatic retry'
    assert sha(STEP0)==STEP0_SHA
    production=code_manifest();mother=runner.git('rev-parse',MOTHER)
    for path,digest in production.items():
        assert digest==hashlib.sha256(subprocess.check_output(['git','show',mother+':'+path],cwd=ROOT)).hexdigest(),path
    official={}
    for name in ('config.json','RESULTS.json','TRAINING_DIAGNOSTICS.json','MASK_HIERARCHY_AUDIT.json'):
        path=REF['exp']/name;relative=str(path.relative_to(ROOT))
        blob=subprocess.check_output(['git','show',REF['branch']+':'+relative],cwd=ROOT)
        assert blob==path.read_bytes()==(PROJECT/relative).read_bytes(),name
        official[name]=hashlib.sha256(blob).hexdigest()
    baseline=runner.read(REF['exp']/'RESULTS.json');checkpoint=REF['run']/'step500/step000500.pt'
    assert baseline['completed_steps']==500 and baseline['evaluation_checkpoint_immutable']
    assert sha(checkpoint)==baseline['checkpoint_sha256']
    ready=runner.read(runner.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator=runner.evaluator_proof()
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,mother_commit=mother,production_sources=production,
        production_source_changes=[],original_HNS_branch=REF['branch'],original_HNS_commit=runner.git('rev-parse',REF['branch']),
        original_raw_results=baseline,original_JSON_sha256=official,common0_sha256=STEP0_SHA,
        original_checkpoint=dict(path=str(checkpoint),sha256=baseline['checkpoint_sha256'],uploaded=False),
        evaluator_sources=evaluator,local_only=True))
    cfg=runner.read(REF['exp']/'config.json');cfg.update(lambda_align=10.,lambda_sparse=1.,lambda_hierarchy=1.)
    dump(EXP/'DEFAULT_HNS_CONFIG.json',cfg)
    for arm in ARMS:
        (EXP/arm).mkdir();dump(EXP/arm/'config.json',shared.config(arm));shared.frozen(shared.config(arm),arm)
    proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],image_root=str(runner.local.IMAGES),
        NFS_fallback=False,raw_path=str(RUN/'local-path-proof-5000.json'),sha256=sha(RUN/'local-path-proof-5000.json')))
    activate(next(iter(ARMS)));(RUN/next(iter(ARMS))).mkdir();runner.search.sampling_audit()
    dump(EXP/'PLAN.json',dict(order=list(ARMS),arms={n:dict(zip(runner.MACRO_KEYS,v)) for n,v in ARMS.items()},
        fresh_common0=True,resume=None,independent_smoke5_formal500=True,max_updates=500,horizon=4868,
        view_weights=[1.35,1.35,.3],sparsity_weights=[1,2,2],beta=[2,2],no_SG=True,soft_inclusion=0,
        ramp='min(1,completed BEFORE update/200)',fixed_K=3,seed=0,world=4,batch_per_rank=256,workers=8,
        local_only=True,NFS_fallback=False,coefficient_control='Same original HNS@500 parameters, same1024 inputs and one shared raw loss graph; no parameter updates',
        severe_mask_stop='All three valid-population Hard-ST supports entirely empty OR entirely full for five consecutive updates; ordinary density/equality changes are diagnostics only',
        automatic_full=False,third_arm=False,combinations=False,additional_seed=False))
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
    proof=runner.read(EXP/'DEFAULT_HNS_GRADIENT_AUDIT.json');assert proof['passed'] and proof['checkpoint']['unchanged']
    assert proof['records']==1024 and proof['no_parameter_updates']
    assert proof['state_digest_before']==proof['state_digest_after']
    assert set(proof['fixed_state_coefficient_controls'])==set(ARMS)
    for arm,value in proof['fixed_state_coefficient_controls'].items():
        assert tuple(value['macro_scales'][k] for k in runner.MACRO_KEYS)==ARMS[arm]
        assert value['same_raw_graph'] and value['same_parameters'] and value['same_inputs']
    return proof


def report_arm(arm,result,supervisor):
    ORIGINAL_REPORT(arm,result,supervisor)
    fixed=validate_controls();actual=runner.read(EXP/arm/'GRADIENT_AUDIT.json')
    assert actual['sample_ids_sha256']==fixed['sample_ids_sha256']
    actual['fixed_original_HNS_coefficient_control']=fixed['fixed_state_coefficient_controls'][arm]
    actual['state_vs_coefficient_warning']='Actual trained500 gradients include learned-state changes; pure coefficient effects use the shared original checkpoint graph control separately.'
    dump(EXP/arm/'GRADIENT_AUDIT.json',actual)
    path=EXP/arm/'REPORT.md';text=path.read_text().replace('fifth arm','third arm')
    text+='\nPure coefficient control and trained-state audit are reported separately. H4 was fixed4 throughout; no intermediate retrieval-based adjustment.\n'
    path.write_text(text)


def selection(qualities):
    best=max(qualities,key=lambda n:qualities[n]['Score5'])
    gain=qualities[best]['Score5']-qualities['HNS-v1']['Score5']
    return dict(BEST_OVERALL_500=best,RECOMMENDED_NEXT_VALIDATION=best if gain>0 else 'KEEP_HNS_V1',
        gain_pp=gain,weak_single_seed_signal=0<gain<.05,automatic_continuation=False)


def combined():
    results={'HNS-v1':runner.read(REF['exp']/'RESULTS.json'),**{n:runner.read(EXP/n/'RESULTS.json') for n in ARMS}}
    q={n:runner.quality(v) for n,v in results.items()};choice=selection(q)
    base_masks=runner.read(REF['exp']/'MASK_HIERARCHY_AUDIT.json')
    masks={n:runner.read(EXP/n/'MASK_HIERARCHY_AUDIT.json') for n in ARMS}
    controls=validate_controls();grads={n:runner.read(EXP/n/'GRADIENT_AUDIT.json') for n in ARMS}
    questions=dict(
        Q1_sparsity=dict(Score5_delta_pp=results['E1-HNS-S12']['quality_delta_pp']['Score5'],keep_changes=masks['E1-HNS-S12']['keep_delta_vs_own_baseline']),
        Q2_H4_violations={k:v for k,v in masks['E2-HNS-H4']['delta_vs_own_baseline'].items() if 'violation' in k},
        Q3_structure_vs_retrieval=dict(retrieval_delta=results['E2-HNS-H4']['quality_delta_pp'],structure=masks['E2-HNS-H4'],lower_violation_does_not_prove_retrieval_gain=True),
        Q4_short_long={n:{k:results[n]['quality_delta_pp'][k] for k in ('Short4','J_long3','J_long')} for n in ARMS},
        Q5_conflict=dict(fixed_checkpoint_default=controls['group_diagnostics'],pure_coefficient_controls=controls['fixed_state_coefficient_controls'],
            actual_trained_states={n:grads[n]['group_diagnostics'] for n in ARMS},warning='Coefficient magnitude and learned-state cosine changes must not be conflated.'),
        Q6_future_candidate=choice)
    comparisons={n:runner.compare(results[n],results['HNS-v1']) for n in ARMS}
    dump(EXP/'RESULTS.json',dict(status='TWO_ARMS_COMPLETED',all_two_completed500=True,models=results,qualities=q,
        comparisons=comparisons,scientific_questions=questions,**choice,GPU_idle=True,extra_experiments=False))
    dump(EXP/'SCIENTIFIC_DIAGNOSTICS.json',questions)
    lines=['# HNS static-strength two-arm @500','',
        '| Model | lambdaA | lambdaS | lambdaH | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for n,v in q.items():
        scales=ARMS.get(n,(10,1,1));lines.append('| '+n+' | '+' | '.join(str(s) for s in scales)+' | '+
            ' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {v["Urban_I2T"]:.3f} / {v["Urban_T2I"]:.3f} |')
    lines+=['','| Arm | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T/T2I |','|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        d=comparisons[arm]['quality_delta_pp'];lines.append('| '+arm+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {d["Urban_I2T"]:+.3f} / {d["Urban_T2I"]:+.3f} |')
    lines+=['','Selection: `'+json.dumps(choice)+'`.','',
        'Q1: S12 Score5 delta '+str(questions['Q1_sparsity']['Score5_delta_pp'])+'pp; keep changes '+json.dumps(questions['Q1_sparsity']['keep_changes'])+'.',
        'Q2: H4 hard violation deltas '+json.dumps(questions['Q2_H4_violations'])+'.',
        'Q3: H4 retrieval deltas '+json.dumps(questions['Q3_structure_vs_retrieval']['retrieval_delta'])+'. Lower violations alone do not imply retrieval gains.',
        'Q4: Short4/long deltas '+json.dumps(questions['Q4_short_long'])+'.',
        'Q5: Gradient norms/cosines for every parameter group are in SCIENTIFIC_DIAGNOSTICS.json and GRADIENT_AUDIT.json. Pure coefficient controls share the original HNS@500 loss graph; learned-state gradients are separate.',
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
                    image_root=str(runner.local.IMAGES),NFS_fallback=False,macro_scales=dict(zip(runner.MACRO_KEYS,ARMS[arm])),git_head=runner.git('rev-parse','HEAD'))
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='smoke',stop=5))
                supervisor=runner.local.Supervisor();runner.state('SMOKE5_RUNNING',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                supervisor.execute('smoke5',runner.training_command(arm,True),training=True)
                identity=runner.identity(runner.local.RUN/'step500/step000005.pt',5,arm,'smoke')
                accept=runner.read(runner.local.RUN/'step500/acceptance.json');assert accept['passed']
                stream=shared.matched_stream(rows(runner.local.RUN/'step500/steps.jsonl'),rows(REF['run']/'step500/steps.jsonl')[:5],arm)
                dump(EXP/arm/'SMOKE_EVIDENCE.json',dict(passed=True,independent_common0=True,checkpoint=identity,acceptance=accept,stream=stream,coefficients_unchanged=True))
                require_gpu_idle({0,1,2,3});activate(arm);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='formal',stop=500))
                dump(EXP/arm/'FORMAL_PROVENANCE.json',dict(provenance,run_type='formal',stop=500))
                supervisor=runner.local.Supervisor();runner.state('FORMAL500_RUNNING',arm,completed_arms=completed,arm_count=2,automatic_third_arm=False)
                supervisor.execute('train500',runner.training_command(arm),training=True)
                accept=runner.read(runtime/'step500/acceptance.json')
                assert accept['passed'] and all(r['completed_updates']==r['updates_this_run']==500 and r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
                assert runner.read(runtime/'first-five-gate.json')['passed']
                proof=shared.matched_stream(rows(runtime/'step500/steps.jsonl'),rows(REF['run']/'step500/steps.jsonl'),arm)
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
    runner.prepare=prepare;runner.run=run;runner.worker=worker;runner.report_arm=report_arm;runner.combined=combined;runner.publish=publish


def main():
    configure();runner.main()


if __name__=='__main__':main()
