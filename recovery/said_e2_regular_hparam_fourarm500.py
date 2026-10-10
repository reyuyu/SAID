"""Four fixed conventional scalars; common0 smoke5 and fresh formal500.

Reuse the audited sequential queue/export/evaluation infrastructure, without
changing native optimizer rates, symmetric CE, data or inference protocol.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from recovery import said_e2_early_lr_threearm500 as base
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,dump,rows,sha,now,distribution
from recovery.s02_local_full import expected_lrs

runner=base.runner
PROJECT=base.PROJECT
PARENT_COMMIT=base.PARENT_COMMIT
BRANCH='experiment/said-e2-regular-hparam-fourarm500-v1'
ENTRY='recovery.said_e2_regular_hparam_fourarm500'
EXP=ROOT/'experiments/nest_clip_v1/said_e2_regular_hparam_fourarm500_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/said-e2-regular-hparam-fourarm500-v1'
BASE_EXP,BASE_RUN,BASE_SHA=base.BASE_EXP,base.BASE_RUN,base.BASE_SHA
GROUPS=base.GROUPS
ARMS={'P1-Temp95':(95.,1.2),'P2-Temp105':(105.,1.2),
      'P3-Sparse110':(100.,1.1),'P4-Sparse130':(100.,1.3)}
SOURCE_FILES=('model/nested_fusion_mask.py','recovery/said_e2_regular_hparam_fourarm500.py',
    'recovery/said_e2_regular_hparam_precheck.py','recovery/said_e2_regular_hparam_gradient.py',
    'tests/test_said_e2_regular_hparam.py')
ORIGINAL_ACTIVATE,ORIGINAL_REPORT=base.activate,base.report_arm
ORIGINAL_TORCHRUN=runner.torchrun


def read(path):return json.loads(Path(path).read_text())


def config(arm):
    assert arm in ARMS
    scale,sparse=ARMS[arm]
    return dict(read(BASE_EXP/'config.json'),contrastive_logit_scale=scale,lambda_sparse=sparse)


def frozen(cfg,arm):
    assert all(cfg.get(k)==v for k,v in config(arm).items()), 'Frozen one-scalar configuration drift'
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg['view_sparsity_weights']==[5/3]*3
    assert cfg['lambda_align']==10. and cfg['lambda_hierarchy']==1. and cfg['sparsity_scale']==1.
    assert cfg['inclusion_max']==0 and cfg['hns_enabled'] and cfg.get('hns_beta',[2,2])==[2,2]
    assert not cfg.get('hns_detach_child',False) and cfg.get('summary_t2i_weight',1.)==1.
    assert cfg['epochs']==4 and cfg['batch_size']==256 and cfg['workers']==8


def state(status,arm=None,**extra):
    dump(EXP/'QUEUE_STATE.json',dict(status=status,active_arm=arm,order=list(ARMS),pid=os.getpid(),
        updated_utc=now(),stop_per_arm=500,horizon=4868,automatic_continuation=False,
        fifth_arm=False,automatic_retry=False,**extra))


def activate(arm,smoke=False):
    ORIGINAL_ACTIVATE(arm,smoke)
    runner.search.PHASE_PREFIX='said-e2-regular-hparam-fourarm500-v1-'
    runner.search.PHASE=runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm)
    runner.local.PHASE=runner.search.PHASE
    if smoke:runner.local.PHASE=runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm+'-smoke5')


def matched_stream(actual,reference,arm):
    assert len(actual)==len(reference) and 0<len(actual)<=500
    assert [r['step'] for r in actual]==[r['step'] for r in reference]==list(range(1,len(actual)+1))
    count,manifest,LRs=0,[],[]
    for a,b in zip(actual,reference):
        assert a['epoch']==b['epoch']==0 and a['s']==a['step']-1
        assert a['nonfinite']==0 and math.isfinite(a['loss'])
        assert a['HNS_enabled'] and a['inc_weight']==a['inclusion_loss']==0
        assert a['lambda_h']==min(1.,a['s']/200.)
        scale,sparse=ARMS[arm]
        assert a['contrastive_logit_scale']==scale
        assert [a['macro_'+k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')]==[10.,sparse,1.]
        assert [a['HNS_sparse_coeff_'+v] for v in ('F','Dall','D3')]==[5/3]*3
        assert math.isclose(a['macro_weighted_sparse'],sparse*a['macro_raw_sparse'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['loss'],sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy')),rel_tol=4e-6,abs_tol=2e-6)
        native=dict(zip(GROUPS,expected_lrs(a['s'],config(arm))))
        assert a['actual_lrs']==b['actual_lrs']==native, 'Native LR changed'
        old={h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            ref=old[h['rank']]
            assert h['batch']==ref['batch']==256 and h['updates']==a['step'] and h['gradients_finite']
            assert h['stream_sha256']==ref['stream_sha256'] and h['sampling']==ref['sampling']
            count+=h['batch'];manifest.append([a['step'],h['rank'],h['stream_sha256'],h['sampling']])
        LRs.append([a['step'],a['actual_lrs']])
    assert count==len(actual)*1024
    return dict(passed=True,optimizer_updates=len(actual),records=count,no_missing_or_duplicate_updates=True,
        all_sample_ids_F_Dall_D3_text_tokens_K_indices_exact=True,all_native_LR_exact=True,horizon=4868,
        only_declared_scalar_changed=dict(contrastive_logit_scale=ARMS[arm][0],lambda_sparse=ARMS[arm][1]),
        stream_sha256=base.digest(manifest),LR_sha256=base.digest(LRs),reference_log_sha256=sha(BASE_RUN/'step500/steps.jsonl'))


def checkpoint_first5(current,reference):
    import torch
    arm=base.CURRENT_ARM;cfg=current['config'];old=reference['config'];frozen(cfg,arm)
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for k in ('component_initialization','adapter_initialization','data','parameter_counts','horizon',
              'batch_size','world_size','accumulation','seed','sampling_seed','shuffle_seed','workers'):
        assert cfg[k]==old[k],k
    runtime,oldruntime=copy.deepcopy(cfg['runtime_model']),copy.deepcopy(old['runtime_model'])
    assert runtime['search_hparams'].pop('lambda_sparse')==ARMS[arm][1]
    assert oldruntime['search_hparams'].pop('lambda_sparse')==1.2
    assert runtime==oldruntime and cfg['code_sha256']==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    assert all(torch.isfinite(v).all() for s in (current['model'],current['adapter'],*current['optimizer']['state'].values()) for v in s.values())
    assert current['regular_hparam_experiment']['contrastive_logit_scale']==ARMS[arm][0]
    assert current['regular_hparam_experiment']['lambda_sparse']==ARMS[arm][1]
    return dict(passed=True,fresh_common0=True,initialization_and_adapter_exact=True,
        optimizer_groups_and_native_LR_exact=True,optimizer_steps=[5],post_update_parameters_expected_to_differ=True)


def identity(path,step,arm,run_type='formal'):
    import torch
    proof=base.ORIGINAL_IDENTITY(path,step,arm,run_type)
    p=torch.load(path,map_location='cpu',weights_only=False)
    cfg=p['config']
    assert cfg['component_initialization']==read(BASE_RUN/'step500/config.json')['component_initialization']
    assert cfg['code_sha256']==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    spec=p['regular_hparam_experiment']
    assert (spec['contrastive_logit_scale'],spec['lambda_sparse'])==ARMS[arm]
    proof['regular_hparam_experiment']=spec
    return proof


def worker(arm,smoke):
    import torch
    import torch.distributed as dist
    from model.balanced_hparam_search import BalancedSearch
    from train import train_nested_semantic_mask as trainer
    activate(arm,smoke);cfg=config(arm);frozen(cfg,arm)
    assert '--resume' not in sys.argv
    trainer.sampling_diagnostics=runner.search.observe_selection
    old_init,old_forward=BalancedSearch.__init__,BalancedSearch.forward
    old_rates,old_save,old_optimizer=trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer
    context={};spec=dict(arm=arm,contrastive_logit_scale=ARMS[arm][0],lambda_sparse=ARMS[arm][1],
        first_update=1,last_update=500,horizon=4868,controller_sha256=sha(Path(__file__)),LR_override=False)
    def init(module,*args,**kwargs):
        assert 'contrastive_logit_scale' not in kwargs
        return old_init(module,*args,contrastive_logit_scale=cfg['contrastive_logit_scale'],**kwargs)
    def rates(module,completed,horizon):
        assert horizon==4868 and 0<=completed<(5 if smoke else 500)
        context['step']=completed+1
        assert module.contrastive_logit_scale==cfg['contrastive_logit_scale'] and module.macro_hparams['lambda_sparse']==cfg['lambda_sparse']
        if completed==0:
            recorded=read(runner.local.RUN/'step500/config.json');frozen(recorded,arm)
            old=read(BASE_RUN/'step500/config.json')
            for k in ('component_initialization','adapter_initialization','data','parameter_counts'):
                assert recorded[k]==old[k],k
        values=old_rates(module,completed,horizon)
        assert list(values)==expected_lrs(completed,cfg)
        return values
    def forward(module,*args,**kwargs):
        loss,logs=old_forward(module,*args,**kwargs)
        logs['contrastive_logit_scale']=module.contrastive_logit_scale
        if context['step']==1:context['loss']=logs['loss'].detach().clone()
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
            norms={g['name']:float(trainer.gradient_norm(g['params'])) for g in opt.param_groups}
            expected=read(EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json')['ranks'][dist.get_rank()]['arms'][arm]
            loss=float(context['loss']);difference=trainer.parameter_agreement(module)
            proof=dict(rank=dist.get_rank(),no_update_yet=True,loss=loss,expected_loss=expected['global_loss'],
                gradient_norms=norms,expected_gradient_norms=expected['optimizer_group_gradient_norms'],
                parameter_difference=difference,optimizer_empty=not opt.state,
                declared_scalars=spec,actual_LR={g['name']:g['lr'] for g in opt.param_groups})
            proof['passed']=(math.isclose(loss,expected['global_loss'],rel_tol=1e-6,abs_tol=1e-5)
                and all(math.isclose(norms[k],expected['optimizer_group_gradient_norms'][k],rel_tol=1e-5,abs_tol=1e-5) for k in norms)
                and difference==0 and not opt.state)
            dump(runner.local.RUN/f'PREUPDATE-rank{dist.get_rank()}.json',proof)
            verdict=torch.tensor(int(proof['passed']),device='cuda');dist.all_reduce(verdict,op=dist.ReduceOp.MIN)
            assert verdict.item(),'First-update expected scalar/gradient/initialization mismatch'
        opt.register_step_pre_hook(pre);return opt
    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            payload['regular_hparam_experiment']=spec
            payload['config']['regular_hparam_experiment']=spec
        return old_save(payload,path)
    BalancedSearch.__init__,BalancedSearch.forward=init,forward
    trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer=rates,save,optimizer
    if smoke:runner.local.worker()
    else:runner.search.worker()


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current')==BRANCH
    assert runner.git('merge-base',PARENT_COMMIT,'HEAD')==PARENT_COMMIT
    assert not EXP.exists() and not RUN.exists(),'No overwrite or automatic retry'
    EXP.mkdir(parents=True);RUN.mkdir(parents=True);state('PRECHECK_RUNNING')
    try:
        assert sha(STEP0)==STEP0_SHA and sha(BASE_RUN/'step500/step000500.pt')==BASE_SHA
        old=read(BASE_RUN/'step500/config.json');production=code_manifest()
        changed=[p for p,h in production.items() if old['code_sha256'][p]!=h]
        assert changed==['model/nested_fusion_mask.py']
        baseline=read(BASE_EXP/'RESULTS.json');assert baseline['completed_steps']==500 and baseline['checkpoint_unchanged']
        assert abs(runner.quality(baseline)['Score5']-71.31437098984591)<1e-9
        for p in BASE_EXP.rglob('*'):
            if p.is_file():
                assert subprocess.check_output(['git','show',PARENT_COMMIT+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
        ready=read(runner.local.IMAGES.parent/'full-ready.json')
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,parent_commit=PARENT_COMMIT,
            inherited_queue_commit='815f5ab6a3e8045ad491fa4219e50913feae2b44',production_sources=production,
            authorized_production_changes=changed,evaluator_sources=runner.evaluator_proof(),
            protected_files_sha256=base.protected(),common0_sha256=STEP0_SHA,baseline_results=baseline))
        for arm in ARMS:dump(EXP/arm/'config.json',config(arm));frozen(config(arm),arm)
        dump(EXP/'FOUR_ARM_PLAN.json',dict(order=list(ARMS),arms={a:dict(contrastive_logit_scale=s,lambda_sparse=w) for a,(s,w) in ARMS.items()},
            fresh_common0=True,independent_smoke5_then_formal500=True,horizon=4868,stop=500,
            world=4,batch_per_rank=256,samples_per_arm=512000,original_native_LR=True,symmetric_CE=True,
            fifth_arm=False,automatic_continuation=False,parameters_frozen_before_test_results=True))
        tests=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q',SOURCE_FILES[-1],
               'tests/test_hns_macro.py','tests/test_hns_s12_sparse_ratio_twoarm500.py']
        with (RUN/'cpu-tests.log').open('x') as log:checked=subprocess.run(tests,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(EXP/'CPU_TESTS.json',dict(passed=checked.returncode==0,command=tests,output=(RUN/'cpu-tests.log').read_text()))
        assert checked.returncode==0
        proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
        dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=proof['passed'],count=proof['count'],image_root=str(runner.local.IMAGES),NFS_fallback=False))
        screen=read(ROOT/'experiments/nest_clip_v1/said_e2_early_lr_threearm500_v1/INDEPENDENT_VALIDATION_SCREEN.json')
        meta=read(runner.local.INDEX/'metadata.json')
        assert screen['annotation_sha256']==meta['annotation_sha256'] and screen['training_index_sha256']==meta['records_sha256']
        dump(EXP/'INDEPENDENT_VALIDATION_SCREEN.json',dict(screen,current_protocol='NOT_ESTABLISHED',public_results='EXPLORATORY'))
        cmd=ORIGINAL_TORCHRUN('recovery.said_e2_regular_hparam_precheck')
        with (RUN/'preupdate-equivalence.log').open('x') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        assert result.returncode==0 and read(EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json')['passed']
        state('PREPARED')
    except BaseException as error:state('STOPPED_WITH_EVIDENCE',error=repr(error));raise


def report_arm(arm,result,supervisor):
    ORIGINAL_REPORT(arm,result,supervisor)
    r=read(EXP/arm/'RESULTS.json');r['declared_scalars']=dict(contrastive_logit_scale=ARMS[arm][0],lambda_sparse=ARMS[arm][1])
    r['native_LR_unchanged']=True;r.pop('lr_multipliers',None);dump(EXP/arm/'RESULTS.json',r)


def combined(completed):
    models={'E2-Uniform':read(BASE_EXP/'RESULTS.json'),**{a:read(EXP/a/'RESULTS.json') for a in completed}}
    q={a:dict(runner.quality(r),Urban_Mean=(runner.quality(r)['Urban_I2T']+runner.quality(r)['Urban_T2I'])/2) for a,r in models.items()}
    dump(EXP/'FOUR_ARM_NATIVE_RESULTS.json',dict(models=models,qualities=q,completed_arms=completed,all_four_completed=len(completed)==4,scientific_status='EXPLORATORY'))
    dump(EXP/'FOUR_ARM_FULL_STREAM_PROOF.json',{a:read(EXP/a/'FULL500_STREAM_AND_LR_PROOF.json') for a in completed})
    dump(EXP/'FOUR_ARM_GRADIENT_MASK_DIAGNOSTICS.json',{a:dict(training=read(EXP/a/'TRAINING_DIAGNOSTICS.json'),masks=read(EXP/a/'MASK_HIERARCHY_AUDIT.json'),gradient=read(EXP/a/'GRADIENT_AUDIT.json')) for a in completed})
    lines=['# E2 conventional hyperparameter four-arm @500','',
        'Only the declared training contrastive logit scale or macro lambda_sparse varies; original symmetric directional CE, F/Dall/D3 weights, Hard-ST, HNS, detach, AdamW/native LR and inference protocol remain unchanged.',
        'All arms independently use common0 smoke5, then restart common0 formal500 with horizon4868.',
        '', '| Model | Logit scale | lambda_sparse | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for a,v in q.items():
        s,w=ARMS.get(a,(100,1.2))
        lines.append(f'| {a} | {s} | {w} | '+' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {v["Urban_I2T"]:.3f}/{v["Urban_T2I"]:.3f} | {v["Urban_Mean"]:.3f} |')
    lines+=['','| Arm | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T/T2I | ΔUrban Mean |','|---|---:|---:|---:|---:|---|---:|']
    for a in completed:
        d=models[a]['quality_delta_pp']
        lines.append('| '+a+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {d["Urban_I2T"]:+.3f}/{d["Urban_T2I"]:+.3f} | {d["Urban_Mean"]:+.3f} |')
    for a in ARMS:
        lines+=['',f'## {a}','','Status: '+('COMPLETED' if a in completed else 'PENDING')+'.']
        if a not in completed:continue
        r=models[a];diag=read(EXP/a/'TRAINING_DIAGNOSTICS.json');grad=read(EXP/a/'GRADIENT_AUDIT.json')
        lines+=['Checkpoint SHA256: `'+r['checkpoint']['sha256']+'`.','Bare SHA256: `'+r['bare']['sha256']+'`.','',
            '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 | ΔI2T R1/R5/R10 | ΔT2I R1/R5/R10 |','|---|---|---|---|---|']
        for ds,m in r['metrics'].items():
            values=[' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            values+=[' / '.join(f'{r["recall_delta_pp"][ds][dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append('| '+ds+' | '+' | '.join(values)+' |')
        lines+=['','Last50 CE/share and delta: `'+json.dumps({k:diag[k] for k in ('last50_views','last50_view_delta_vs_E2')})+'`.',
            'Last50 loss/mask/HNS/IoU and delta: `'+json.dumps({k:diag[k] for k in ('last50_macro_HNS','last50_delta_vs_E2')})+'`.',
            'Actual four-group LR/gradient norms: `'+json.dumps({k:diag[k] for k in ('actual_group_LR','group_gradient_norms')})+'`.',
            'Matched-state gradient component norms/cosines and differences: `'+json.dumps(dict(current=grad['group_diagnostics'],delta_vs_E2=diag['gradient_group_delta_vs_E2']))+'`.',
            'These trained-state mechanisms are descriptive, not proof of causation. Fixed-state isolation was checked separately before training.']
    if len(completed)==4:
        lines+=['','Arms improving both Score5 and Urban bidirectional mean: '+str([a for a in completed if all(models[a]['quality_delta_pp'][k]>0 for k in ('Score5','Urban_Mean'))])+'.',
            'All positive and negative results retained; no combination, rerun, extra seed or continuation selected automatically.']
    lines+=['','All scores are percent and differences percentage points. Urban0.1pp is one query; small single-seed changes are not stable-performance evidence.',
        'No independent retrieval validation protocol established. Public benchmarks, especially Urban, were repeatedly observed; this is exploratory evidence, not unbiased SOTA or independent generalization.',
        'Original baseline checkpoints/results stay immutable. Default100 source-versus-legacy forward/loss and gradients were checked on a real BF16 four-rank batch; only score_block CE logits are temperature-scaled. Masks/sigmoid/ST and both directional definitions remain unchanged.',
        'Each full stage verifies512000 sample positions, all IDs/text/tokens/K/indices, unchanged native LR, four-rank agreement and complete restorable checkpoint/export.',
        'Hard stop500 for each of four arms. No fifth arm, combined scalar,1217/4868 continuation, rerank or ensemble.']
    (EXP/'REGULAR_HPARAM_FOURARM500_REPORT.md').write_text('\n'.join(lines)+'\n')


def configure():
    base.BRANCH,base.EXP,base.RUN,base.ENTRY=BRANCH,EXP,RUN,ENTRY
    base.ARMS={a:(1.,1.,1.,1.) for a in ARMS}
    base.SOURCE_FILES=SOURCE_FILES
    base.config,base.frozen,base.activate,base.checkpoint_first5=config,frozen,activate,checkpoint_first5
    base.matched_stream,base.identity,base.state=matched_stream,identity,state
    base.prepare,base.worker,base.report_arm,base.combined=prepare,worker,report_arm,combined
    base.configure()
    def torchrun(module,*args):
        if module=='recovery.hns_macro_gradient':module='recovery.said_e2_regular_hparam_gradient'
        return ORIGINAL_TORCHRUN(module,*args)
    runner.torchrun=torchrun


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(ARMS))
    for flag in ('prepare','publish-setup','launch','run','worker','smoke-worker'):parser.add_argument('--'+flag,action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.arm,args.smoke_worker);return
    assert not remaining
    if args.prepare:prepare()
    elif args.publish_setup:base.publish(True)
    elif args.run:base.run()
    elif args.launch:
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED' and not (RUN/'DETACHED_LAUNCH.json').exists()
        assert read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==runner.git('rev-parse','HEAD')
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,durable=True,order=list(ARMS),log=str(RUN/'runner.log'),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',launch);print(json.dumps(launch),flush=True)
    else:parser.error('Choose action')


if __name__=='__main__':main()
