"""One frozen hierarchy090 arm; common0 smoke5, fresh formal500, audit/eval/stop."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from recovery import said_e2_early_lr_threearm500 as base
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now
from recovery.s02_local_full import expected_lrs

runner = base.runner
PROJECT, PARENT_COMMIT = base.PROJECT, base.PARENT_COMMIT
BRANCH = 'experiment/said-e2-hierarchy090-500-v1'
ENTRY = 'recovery.said_e2_hierarchy090_500'
EXP = ROOT/'experiments/nest_clip_v1/said_e2_hierarchy090_500_v1'
RUN = PROJECT/'runtime/SAID-nest-clip-v1/said-e2-hierarchy090-500-v1'
ARM = 'E2-Hierarchy090'
ARMS = {ARM: (1.,1.,1.,1.)}
BASE_EXP, BASE_RUN, BASE_SHA, GROUPS = base.BASE_EXP, base.BASE_RUN, base.BASE_SHA, base.GROUPS
SOURCE_FILES = ('recovery/said_e2_hierarchy090_500.py',
                'recovery/said_e2_hierarchy090_precheck.py',
                'recovery/HIERARCHY090_CPU_TEST_IMPLEMENTATION_DIAGNOSIS.md',
                'tests/test_said_e2_hierarchy090.py')
ORIGINAL_ACTIVATE, ORIGINAL_REPORT = base.activate, base.report_arm


def read(path): return json.loads(Path(path).read_text())


def config(arm):
    from model.balanced_hparam_search import trial_id
    assert arm == ARM
    cfg = dict(read(BASE_EXP/'config.json'), lambda_hierarchy=.9)
    cfg['trial_id'] = trial_id(cfg)
    return cfg


def frozen(cfg, arm):
    assert arm == ARM and all(cfg.get(k)==v for k,v in config(arm).items()), 'Frozen hierarchy090 drift'
    assert [cfg[k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')]==[10.,1.2,.9]
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg['view_sparsity_weights']==[5/3]*3
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['sparsity_scale']==1
    assert cfg.get('hns_beta',[2,2])==[2,2] and not cfg.get('hns_detach_child',False)
    assert cfg.get('summary_t2i_weight',1.)==1. and cfg['epochs']==4
    assert cfg['workers']==8 and cfg['batch_size']==256 and cfg['accumulation']==1


def activate(arm, smoke=False):
    ORIGINAL_ACTIVATE(arm,smoke)
    runner.search.PHASE_PREFIX='said-e2-hierarchy090-500-v1-'
    runner.search.PHASE=runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm)
    runner.local.PHASE=runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm+('-smoke5' if smoke else ''))


def state(status, arm=None, **extra):
    dump(EXP/'QUEUE_STATE.json',dict(status=status,active_arm=arm,order=[ARM],pid=os.getpid(),
        updated_utc=now(),stop_per_arm=500,horizon=4868,second_arm=False,
        automatic_continuation=False,automatic_retry=False,**extra))


def matched_stream(actual, reference, arm):
    frozen(config(arm),arm)
    assert len(actual)==len(reference) and 0<len(actual)<=500
    assert [r['step'] for r in actual]==[r['step'] for r in reference]==list(range(1,len(actual)+1))
    count, manifest, lrs = 0, [], []
    for a,b in zip(actual,reference):
        assert a['epoch']==b['epoch']==0 and a['s']==a['step']-1
        assert a['nonfinite']==0 and math.isfinite(a['loss']) and a['HNS_enabled']
        assert a['inclusion_loss']==a['inc_weight']==0 and a['lambda_h']==min(1.,a['s']/200.)
        assert [a['macro_'+k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')]==[10.,1.2,.9]
        assert [a['HNS_sparse_coeff_'+v] for v in ('F','Dall','D3')]==[5/3]*3
        assert a['macro_weighted_align']==10*a['macro_raw_align'] or math.isclose(a['macro_weighted_align'],10*a['macro_raw_align'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['macro_weighted_sparse'],1.2*a['macro_raw_sparse'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['macro_weighted_hierarchy'],.9*a['lambda_h']*a['macro_raw_hierarchy'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['loss'],sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy')),rel_tol=4e-6,abs_tol=2e-6)
        assert a['actual_lrs']==b['actual_lrs']==dict(zip(GROUPS,expected_lrs(a['s'],config(arm))))
        old={h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            ref=old[h['rank']]
            assert h['batch']==ref['batch']==256 and h['updates']==a['step'] and h['gradients_finite']
            assert h['stream_sha256']==ref['stream_sha256'] and h['sampling']==ref['sampling']
            manifest.append([a['step'],h['rank'],h['stream_sha256'],h['sampling']]);count+=256
        lrs.append([a['step'],a['actual_lrs']])
    assert count==1024*len(actual)
    return dict(passed=True,optimizer_updates=len(actual),records=count,horizon=4868,
        all_sample_ids_F_Dall_D3_text_tokens_K_indices_exact=True,all_native_LR_exact=True,
        no_missing_or_duplicate_updates=True,lambda_hierarchy=.9,stream_sha256=base.digest(manifest),
        LR_sha256=base.digest(lrs),reference_log_sha256=sha(BASE_RUN/'step500/steps.jsonl'))


def checkpoint_first5(current, reference):
    import torch
    cfg,old=current['config'],reference['config'];frozen(cfg,ARM)
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for k in ('component_initialization','adapter_initialization','data','parameter_counts','horizon',
              'batch_size','world_size','accumulation','seed','sampling_seed','shuffle_seed','workers','code_sha256'):
        assert cfg[k]==old[k],k
    a,b=copy.deepcopy(cfg['runtime_model']),copy.deepcopy(old['runtime_model'])
    assert a['search_hparams'].pop('lambda_hierarchy')==.9
    assert b['search_hparams'].pop('lambda_hierarchy')==1.
    assert a==b and current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    assert all(torch.isfinite(v).all() for s in (current['model'],current['adapter'],*current['optimizer']['state'].values()) for v in s.values())
    assert current['hierarchy090_experiment']['trial_id']==cfg['trial_id']
    return dict(passed=True,fresh_common0=True,initialization_and_adapter_exact=True,
        optimizer_steps=[5],production_sources_exact=True,only_lambda_hierarchy_changed=True)


def identity(path, step, arm, run_type='formal'):
    import torch
    result=base.ORIGINAL_IDENTITY(path,step,arm,run_type)
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['hierarchy090_experiment']['lambda_hierarchy']==.9
    assert p['hierarchy090_experiment']['trial_id']==p['config']['trial_id']
    assert p['config']['code_sha256']==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    assert p['config']['component_initialization']==read(BASE_RUN/'step500/config.json')['component_initialization']
    result['hierarchy090_experiment']=p['hierarchy090_experiment']
    return result


def worker(arm, smoke):
    import torch
    import torch.distributed as dist
    from model.balanced_hparam_search import BalancedSearch
    from train import train_nested_semantic_mask as trainer
    activate(arm,smoke);cfg=config(arm);frozen(cfg,arm)
    assert '--resume' not in sys.argv
    trainer.sampling_diagnostics=runner.search.observe_selection
    old_rates,old_save,old_optimizer=trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer
    old_forward=BalancedSearch.forward;context={}
    spec=dict(arm=ARM,lambda_hierarchy=.9,baseline_lambda_hierarchy=1.,trial_id=cfg['trial_id'],
        first_update=1,last_update=500,horizon=4868,LR_override=False,production_source_changes=[],
        controller_sha256=sha(Path(__file__)))
    reference=rows(BASE_RUN/'step500/steps.jsonl')[0]
    def rates(module, completed, horizon):
        assert horizon==4868 and 0<=completed<(5 if smoke else 500)
        context['step']=completed+1
        assert module.macro_hparams==dict(lambda_align=10.,lambda_sparse=1.2,lambda_hierarchy=.9)
        if completed==0:
            recorded=read(runner.local.RUN/'step500/config.json');frozen(recorded,arm)
            old=read(BASE_RUN/'step500/config.json')
            for k in ('component_initialization','adapter_initialization','data','parameter_counts','code_sha256'):
                assert recorded[k]==old[k],k
        values=old_rates(module,completed,horizon)
        assert list(values)==expected_lrs(completed,cfg)
        return values
    def forward(module,*args,**kwargs):
        loss,logs=old_forward(module,*args,**kwargs)
        if context['step']==1:context['loss']=loss.detach().clone()
        keep=[float(logs['HNS_'+v+'_keep']) for v in ('F','Dall','D3')]
        assert all(math.isfinite(v) and 0<=v<=1 for v in keep)
        endpoint='empty' if all(v==0 for v in keep) else 'full' if all(v==1 for v in keep) else None
        context['streak']=context.get('streak',0)+1 if endpoint and endpoint==context.get('endpoint') else 1 if endpoint else 0
        context['endpoint']=endpoint
        if context['streak']>=5:
            dump(runner.local.RUN/f'MASK_FAILURE-rank{dist.get_rank()}.json',dict(step=context['step'],endpoint=endpoint))
            raise RuntimeError('HARD_STOP: complete mask degeneration five consecutive updates')
        return loss,logs
    def optimizer(module):
        opt=old_optimizer(module)
        assert tuple(g['name'] for g in opt.param_groups)==GROUPS and not opt.state
        def pre(opt,args,kwargs):
            if context['step']!=1:return
            loss=context['loss'].clone();dist.all_reduce(loss);loss/=4
            norms={g['name']:float(trainer.gradient_norm(g['params'])) for g in opt.param_groups}
            ref=next(h for h in reference['rank_health'] if h['rank']==dist.get_rank())
            difference=trainer.parameter_agreement(module)
            proof=dict(rank=dist.get_rank(),no_update_yet=True,loss=float(loss),baseline_loss=reference['loss'],
                gradient_norms=norms,baseline_gradient_norms=ref['gradient_norms'],parameter_difference=difference,
                optimizer_state_empty=not opt.state,actual_LR={g['name']:g['lr'] for g in opt.param_groups},
                hierarchy_ramp_at_first_step=0,declared_configuration=spec)
            proof['passed']=(abs(float(loss)-reference['loss'])<=1e-5 and
                all(math.isclose(norms[k],ref['gradient_norms'][k],rel_tol=1e-5,abs_tol=1e-4) for k in norms)
                and difference==0 and not opt.state)
            dump(runner.local.RUN/f'PREUPDATE-rank{dist.get_rank()}.json',proof)
            verdict=torch.tensor(int(proof['passed']),device='cuda');dist.all_reduce(verdict,op=dist.ReduceOp.MIN)
            assert verdict.item(),'First loss/gradient/init mismatch; receipt saved before update'
        opt.register_step_pre_hook(pre);return opt
    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            payload['hierarchy090_experiment']=spec;payload['config']['hierarchy090_experiment']=spec
        return old_save(payload,path)
    trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer=rates,save,optimizer
    BalancedSearch.forward=forward
    if smoke:runner.local.worker()
    else:runner.search.worker()


def historical_audit():
    """Inspect stored experiment configurations and completed-node receipts, not new candidates."""
    roots=[PROJECT/'experiments',PROJECT.parent/'said-e2-worktrees']
    hits=[];seen=set();count=0
    for root in roots:
        paths=root.rglob('config.json') if root.name=='experiments' else (
            p for w in root.iterdir() if w.is_dir() and (w/'experiments').is_dir()
            for p in (w/'experiments').rglob('config.json'))
        for p in paths:
            if p.is_symlink() or p.resolve() in seen:continue
            seen.add(p.resolve())
            try:d=read(p)
            except (ValueError,OSError):continue
            count+=1
            if d.get('lambda_hierarchy')!=.9 or not d.get('hns_enabled'):continue
            keys=('view_weights','view_sparsity_weights','lambda_align','lambda_sparse','sampling_mode',
                  'seed','batch_size','world_size','epochs','fusion','visual','condition_mode')
            if all(d.get(k)==config(ARM).get(k) for k in keys):
                completed=any((p.parent/n).exists() for n in ('RESULTS.json','HIERARCHY090_NATIVE_RESULTS.json'))
                hits.append(dict(path=str(p),completed_receipt_exists=completed))
    assert not hits, 'Prior same-condition hierarchy090 configuration found; no duplicate launch'
    return dict(passed=True,configurations_inspected=count,matches=hits,
        scope='Stored config.json files in main experiments and all local worktrees; Git branch/commit search also performed',
        limitation='No claim about experiments outside available repository/runtime evidence')


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current')==BRANCH
    assert runner.git('merge-base',PARENT_COMMIT,'HEAD')==PARENT_COMMIT
    assert not EXP.exists() and not RUN.exists(), 'Never overwrite or retry an experiment'
    history=historical_audit()
    EXP.mkdir(parents=True);RUN.mkdir(parents=True);state('PRECHECK_RUNNING')
    try:
        assert sha(STEP0)==STEP0_SHA and sha(BASE_RUN/'step500/step000500.pt')==BASE_SHA
        production=code_manifest();old=read(BASE_RUN/'step500/config.json')
        assert old['code_sha256']==production
        assert all(sha(ROOT/p)==__import__('hashlib').sha256(subprocess.check_output(['git','show',PARENT_COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in production)
        baseline=read(BASE_EXP/'RESULTS.json');q=runner.quality(baseline)
        assert baseline['completed_steps']==500 and baseline['checkpoint_unchanged']
        expected=dict(Score5=71.31437098984591,J_long3=75.12061831640985,J_long=84.1950027179718,Short4=65.605)
        assert all(abs(q[k]-v)<1e-9 for k,v in expected.items())
        assert round(q['Urban_I2T'],3)==91.1 and round(q['Urban_T2I'],3)==90.
        for p in BASE_EXP.rglob('*'):
            if p.is_file():assert subprocess.check_output(['git','show',PARENT_COMMIT+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
        ready=read(runner.local.IMAGES.parent/'full-ready.json')
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        eval_sources=runner.evaluator_proof()
        dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,parent_commit=PARENT_COMMIT,
            production_sources=production,evaluator_sources=eval_sources,protected_files_sha256=base.protected(),
            common0_path=str(STEP0.resolve()),common0_sha256=STEP0_SHA,baseline_checkpoint_sha256=BASE_SHA,
            baseline_results=baseline,production_source_changes=[]))
        dump(EXP/'HISTORICAL_DEDUP_AUDIT.json',history)
        dump(EXP/ARM/'config.json',config(ARM));frozen(config(ARM),ARM)
        dump(EXP/'HIERARCHY090_PLAN.json',dict(arm=ARM,trial_id=config(ARM)['trial_id'],
            only_variable=dict(lambda_hierarchy=dict(baseline=1.,candidate=.9)),
            common0_path=str(STEP0.resolve()),common0_sha256=STEP0_SHA,resume=None,
            independent_smoke5_then_fresh_formal500=True,stop_updates=500,horizon=4868,
            world=4,batch_per_rank=256,samples=512000,second_arm=False,automatic_continuation=False,
            production_source_changes=[],dedup=history,
            evaluation_mapping=dict(COCO=0,DOCCI=1,Long_DCI=2,Flickr_then_Urban=3)))
        cmd=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q',SOURCE_FILES[-1],
             'tests/test_hns_s12_sparse_ratio_twoarm500.py','tests/test_hns_macro.py']
        with (RUN/'cpu-tests.log').open('x') as log:
            checked=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(EXP/'CPU_TESTS.json',dict(passed=checked.returncode==0,command=cmd,output=(RUN/'cpu-tests.log').read_text()))
        assert checked.returncode==0,'CPU regression failed'
        proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
        assert proof['passed']
        dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],
            image_root=str(runner.local.IMAGES),index=str(runner.local.INDEX),NFS_fallback=False))
        with (RUN/'precheck.log').open('x') as log:
            checked=subprocess.run(runner.torchrun('recovery.said_e2_hierarchy090_precheck'),cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        assert checked.returncode==0 and read(EXP/'LOSS_AND_GRADIENT_EQUIVALENCE.json')['passed']
        assert sha(STEP0)==STEP0_SHA
        state('PREPARED')
    except BaseException as error:state('STOPPED_WITH_EVIDENCE',error=repr(error));raise


def report_arm(arm, result, supervisor):
    ORIGINAL_REPORT(arm,result,supervisor)
    r=read(EXP/arm/'RESULTS.json');r.pop('lr_multipliers',None)
    r.update(only_variable='lambda_hierarchy',lambda_hierarchy=.9,native_LR_unchanged=True)
    dump(EXP/arm/'RESULTS.json',r)
    d=read(EXP/arm/'TRAINING_DIAGNOSTICS.json')
    d.update(original_HNS_formula='ramp(t)*(2*V_DF+2*V_3D)/3',
        weighted_HNS_formula='0.9*ramp(t)*(2*V_DF+2*V_3D)/3',beta=[2,2],ramp_steps=200,
        no_mask_to_mask_SG=True,original_visual_text_detach=True)
    dump(EXP/arm/'TRAINING_DIAGNOSTICS.json',d)


def combined(completed):
    assert completed==[ARM]
    r=read(EXP/ARM/'RESULTS.json');b=read(BASE_EXP/'RESULTS.json')
    q={a:dict(runner.quality(x),Urban_Mean=(runner.quality(x)['Urban_I2T']+runner.quality(x)['Urban_T2I'])/2)
       for a,x in {'E2-Uniform':b,ARM:r}.items()}
    d=read(EXP/ARM/'TRAINING_DIAGNOSTICS.json');g=read(EXP/ARM/'GRADIENT_AUDIT.json')
    masks=read(EXP/ARM/'MASK_HIERARCHY_AUDIT.json');proof=read(EXP/ARM/'FULL500_STREAM_AND_LR_PROOF.json')
    dump(EXP/'FULL500_STREAM_PROOF.json',proof)
    dump(EXP/'HIERARCHY090_NATIVE_RESULTS.json',dict(models={'E2-Uniform':b,ARM:r},qualities=q,
        delta_pp=r['quality_delta_pp'],recall_delta_pp=r['recall_delta_pp'],scientific_status='EXPLORATORY'))
    dump(EXP/'HIERARCHY090_TRAINING_DIAGNOSTICS.json',dict(training=d,masks=masks,gradient=g))
    lines=['# E2-Hierarchy090 @500','',
        'Only lambda_hierarchy changes from1.0 to0.9 through native macro_terms; production sources are unchanged. '
        'Independent common0 smoke5 precedes fresh common0 formal500; native4868-horizon LR and all512000 sample positions match E2.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---:|---:|---:|---:|---|---:|']
    for a,v in q.items():lines.append('| '+a+' | '+' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {v["Urban_I2T"]:.3f}/{v["Urban_T2I"]:.3f} | {v["Urban_Mean"]:.3f} |')
    delta=r['quality_delta_pp'];lines+=['','Delta in percentage points: `'+json.dumps(delta)+'`.','',
        '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 | ΔI2T R1/R5/R10 | ΔT2I R1/R5/R10 |',
        '|---|---|---|---|---|']
    for ds,m in r['metrics'].items():
        vals=[' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
        vals+=[' / '.join(f'{r["recall_delta_pp"][ds][dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
        lines.append('| '+ds+' | '+' | '.join(vals)+' |')
    means=d['last50_macro_HNS'];diff=d['last50_delta_vs_E2']
    lines+=['','## Measured research questions','',
        f'Q1: ΔScore5={delta["Score5"]:+.6f}pp; '+('higher' if delta['Score5']>0 else 'not higher')+' than E2.',
        f'Q2: Urban ΔI2T/ΔT2I={delta["Urban_I2T"]:+.3f}/{delta["Urban_T2I"]:+.3f}pp. Both directions improve: '+str(delta['Urban_I2T']>0 and delta['Urban_T2I']>0)+'.',
        'Q3: Last50 hard violations Dall→F/D3→Dall='+f'{100*means["V_DF_hard"]:.6f}/{100*means["V_3D_hard"]:.6f}%, differences='+f'{100*diff["V_DF_hard"]:+.6f}/{100*diff["V_3D_hard"]:+.6f}pp. These means do not establish statistical significance.',
        f'Q4: ΔJ_long3={delta["J_long3"]:+.6f}, ΔJ_long={delta["J_long"]:+.6f}, ΔShort4={delta["Short4"]:+.6f}pp. See all dataset/direction differences above.',
        'Q5: '+('A possible exploratory replication candidate because both Score5 and Urban Mean rise; full4-Epoch benefit remains unverified.' if delta['Score5']>0 and delta['Urban_Mean']>0 else 'No joint improvement of Score5 and Urban Mean; these data do not justify prioritizing full4-Epoch continuation as an overall winner.'),
        '', 'Last50 original/weighted losses, keep ratios, HNS violations/IoU and baseline differences: `'+json.dumps(dict(means=means,delta=diff))+'`.',
        'Gradient norms/cosines by parameter group: `'+json.dumps(g['group_diagnostics'])+'`.',
        'Matched probe gradient differences: `'+json.dumps(d['gradient_group_delta_vs_E2'])+'`.',
        'Checkpoint SHA256: `'+r['checkpoint']['sha256']+'`; bare SHA256: `'+r['bare']['sha256']+'`.',
        '', 'All30 recalls and deltas, complete restore-state identity, strict export, source manifest, full stream/LR, last50 curves and resource receipts are adjacent JSON files. '
        'Four-rank parameter difference is zero at acceptance. Checkpoints and original E2 artifacts remain immutable.',
        'Exploratory single-seed500 result on repeatedly observed public benchmarks. Independent validation is NOT_ESTABLISHED. '
        'Urban0.1pp represents one net correct query; no paired significance or unbiased SOTA claim. Density or a gradient cosine alone is not evidence of semantic benefit.',
        'No other arm, optimizer parameter change, inference mask/rerank/ensemble/TTA or continuation beyond500 was launched.']
    (EXP/'HIERARCHY090_REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    files=list(SOURCE_FILES)+[str(p.relative_to(ROOT)) for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
    assert set(runner.git('diff','--cached','--name-only').splitlines())<=set(files)
    assert all((ROOT/p).stat().st_size<5*1024*1024 for p in files)
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if runner.git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' E2 Hierarchy090 single-arm500'],cwd=ROOT,check=True)
    head=runner.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==runner.git('rev-parse','origin/'+BRANCH)==runner.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,
        branch=BRANCH,commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def configure():
    base.BRANCH,base.EXP,base.RUN,base.ENTRY=BRANCH,EXP,RUN,ENTRY
    base.ARMS,base.SOURCE_FILES=ARMS,SOURCE_FILES
    base.config,base.frozen,base.activate=config,frozen,activate
    base.worker,base.identity,base.checkpoint_first5=worker,identity,checkpoint_first5
    base.matched_stream,base.state=matched_stream,state
    base.prepare,base.report_arm,base.combined,base.publish=prepare,report_arm,combined,publish
    base.configure()


def main():
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=[ARM])
    for flag in ('prepare','publish-setup','launch','run','worker','smoke-worker'):parser.add_argument('--'+flag,action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.arm,args.smoke_worker);return
    assert not remaining
    if args.prepare:prepare()
    elif args.publish_setup:publish(True)
    elif args.run:base.run()
    elif args.launch:
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED' and not (RUN/'DETACHED_LAUNCH.json').exists()
        assert read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==runner.git('rev-parse','HEAD')
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,durable=True,arm=ARM,log=str(RUN/'runner.log'),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',launch);print(json.dumps(launch),flush=True)
    else:parser.error('Choose action')


if __name__=='__main__':main()
