"""Four frozen E2 alignment arms; independent common0 smoke5/formal500."""
import argparse
import ast
import copy
import hashlib
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys

from recovery import said_e2_early_lr_threearm500 as base
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now
from recovery.s02_local_full import expected_lrs

runner = base.runner
PROJECT, PARENT_COMMIT = base.PROJECT, base.PARENT_COMMIT
BRANCH = 'experiment/said-e2-align-weight-fourarm500-v1'
ENTRY = 'recovery.said_e2_align_weight_fourarm500'
EXP = ROOT/'experiments/nest_clip_v1/said_e2_align_weight_fourarm500_v1'
RUN = PROJECT/'runtime/SAID-nest-clip-v1/said-e2-align-weight-fourarm500-v1'
BASE_EXP, BASE_RUN, BASE_SHA, GROUPS = base.BASE_EXP, base.BASE_RUN, base.BASE_SHA, base.GROUPS
ARMS = {'A1-View-DallPlus': ([1.30,1.40,.30],10.),
        'A2-View-FPlus': ([1.40,1.30,.30],10.),
        'A3-Align95': ([1.35,1.35,.30],9.5),
        'A4-Align105': ([1.35,1.35,.30],10.5)}
PRODUCTION_CHANGES = ['model/balanced_hparam_search.py','train/train_nested_semantic_mask.py']
SOURCE_FILES = (*PRODUCTION_CHANGES,'recovery/said_e2_align_weight_fourarm500.py',
                'recovery/said_e2_align_weight_precheck.py','tests/test_said_e2_align_weight.py')
ORIGINAL_ACTIVATE, ORIGINAL_REPORT = base.activate, base.report_arm


def read(path): return json.loads(Path(path).read_text())


def config(arm):
    from model.balanced_hparam_search import trial_id
    weights, align = ARMS[arm]
    cfg=dict(read(BASE_EXP/'config.json'),view_weights=list(weights),lambda_align=align)
    cfg['trial_id']=trial_id(cfg)
    return cfg


def frozen(cfg,arm):
    assert all(cfg.get(k)==v for k,v in config(arm).items()), 'Frozen alignment arm drift'
    assert math.isclose(sum(cfg['view_weights']),3.,abs_tol=1e-12)
    assert cfg['view_weights'][2]==.3
    assert cfg['view_sparsity_weights']==[5/3]*3
    assert cfg['lambda_sparse']==1.2 and cfg['lambda_hierarchy']==1.
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['sparsity_scale']==1.
    assert cfg.get('hns_beta',[2,2])==[2,2] and not cfg.get('hns_detach_child',False)
    assert cfg.get('summary_t2i_weight',1.)==1.
    assert cfg['epochs']==4 and cfg['workers']==8 and cfg['batch_size']==256


def state(status,arm=None,**extra):
    dump(EXP/'QUEUE_STATE.json',dict(status=status,active_arm=arm,order=list(ARMS),pid=os.getpid(),
        updated_utc=now(),stop_per_arm=500,horizon=4868,automatic_continuation=False,
        fifth_arm=False,automatic_retry=False,**extra))


def activate(arm,smoke=False):
    ORIGINAL_ACTIVATE(arm,smoke)
    for name in ARMS:
        runner.search.ARMS[name].update(axis='alignment',weights=list(ARMS[name][0]))
    runner.search.PHASE_PREFIX='said-e2-align-weight-fourarm500-v1-'
    runner.search.PHASE=runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm)
    runner.local.PHASE=(runner.search.LOCAL/(runner.search.PHASE_PREFIX+arm+'-smoke5')
                        if smoke else runner.search.PHASE)


def matched_stream(actual,reference,arm):
    assert len(actual)==len(reference) and 0<len(actual)<=500
    assert [r['step'] for r in actual]==[r['step'] for r in reference]==list(range(1,len(actual)+1))
    count,manifest,lrs=0,[],[]
    for a,b in zip(actual,reference):
        assert a['epoch']==b['epoch']==0 and a['s']==a['step']-1
        assert a['nonfinite']==0 and math.isfinite(a['loss'])
        assert a['HNS_enabled'] and a['inc_weight']==a['inclusion_loss']==0
        assert a['lambda_h']==min(1.,a['s']/200.)
        assert a['alignment_view_weights']==ARMS[arm][0]
        assert [a['macro_'+k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')]==[ARMS[arm][1],1.2,1.]
        assert [a['HNS_sparse_coeff_'+v] for v in ('F','Dall','D3')]==[5/3]*3
        assert math.isclose(a['macro_weighted_align'],ARMS[arm][1]*a['macro_raw_align'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['macro_weighted_sparse'],1.2*a['macro_raw_sparse'],rel_tol=4e-6,abs_tol=2e-6)
        assert math.isclose(a['loss'],sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy')),rel_tol=4e-6,abs_tol=2e-6)
        native=dict(zip(GROUPS,expected_lrs(a['s'],config(arm))))
        assert a['actual_lrs']==b['actual_lrs']==native, 'Native LR changed'
        old={h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            ref=old[h['rank']]
            assert h['batch']==ref['batch']==256 and h['updates']==a['step'] and h['gradients_finite']
            assert h['stream_sha256']==ref['stream_sha256'] and h['sampling']==ref['sampling']
            count+=256;manifest.append([a['step'],h['rank'],h['stream_sha256'],h['sampling']])
        lrs.append([a['step'],a['actual_lrs']])
    assert count==len(actual)*1024
    return dict(passed=True,optimizer_updates=len(actual),records=count,horizon=4868,
        no_missing_or_duplicate_updates=True,all_sample_ids_F_Dall_D3_text_tokens_K_indices_exact=True,
        all_native_LR_exact=True,view_weights=ARMS[arm][0],lambda_align=ARMS[arm][1],
        stream_sha256=base.digest(manifest),LR_sha256=base.digest(lrs),
        reference_log_sha256=sha(BASE_RUN/'step500/steps.jsonl'))


def checkpoint_first5(current,reference):
    import torch
    arm=base.CURRENT_ARM;cfg=current['config'];old=reference['config'];frozen(cfg,arm)
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for key in ('component_initialization','adapter_initialization','data','parameter_counts','horizon',
                'batch_size','world_size','accumulation','seed','sampling_seed','shuffle_seed','workers'):
        assert cfg[key]==old[key],key
    runtime,oldruntime=copy.deepcopy(cfg['runtime_model']),copy.deepcopy(old['runtime_model'])
    for key in ('view_weights','lambda_align'):
        assert runtime['search_hparams'].pop(key)==config(arm)[key]
        oldruntime['search_hparams'].pop(key)
    assert runtime==oldruntime
    assert cfg['code_sha256']==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    assert all(torch.isfinite(v).all() for s in (current['model'],current['adapter'],*current['optimizer']['state'].values()) for v in s.values())
    assert current['alignment_weight_experiment']['trial_id']==cfg['trial_id']
    return dict(passed=True,fresh_common0=True,initialization_and_adapter_exact=True,
        authorized_whitelist_source_changes=PRODUCTION_CHANGES,optimizer_steps=[5],
        native_LR_and_groups_exact=True,post_update_parameters_expected_to_differ=True)


def identity(path,step,arm,run_type='formal'):
    import torch
    proof=base.ORIGINAL_IDENTITY(path,step,arm,run_type)
    p=torch.load(path,map_location='cpu',weights_only=False);cfg=p['config']
    assert cfg['component_initialization']==read(BASE_RUN/'step500/config.json')['component_initialization']
    assert cfg['code_sha256']==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    spec=p['alignment_weight_experiment']
    assert spec['view_weights']==ARMS[arm][0] and spec['lambda_align']==ARMS[arm][1]
    assert spec['trial_id']==cfg['trial_id']
    proof['alignment_weight_experiment']=spec
    return proof


def worker(arm,smoke):
    import torch
    import torch.distributed as dist
    from model.balanced_hparam_search import BalancedSearch,trial_id
    from train import train_nested_semantic_mask as trainer
    activate(arm,smoke);cfg=config(arm);frozen(cfg,arm)
    assert '--resume' not in sys.argv
    trainer.sampling_diagnostics=runner.search.observe_selection
    old_rates,old_save,old_optimizer=trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer
    old_forward=BalancedSearch.forward;context={}
    spec=dict(arm=arm,view_weights=cfg['view_weights'],lambda_align=cfg['lambda_align'],
        trial_id=trial_id(cfg),first_update=1,last_update=500,horizon=4868,
        controller_sha256=sha(Path(__file__)),LR_override=False,
        authorized_whitelist_changes=PRODUCTION_CHANGES)
    def rates(module,completed,horizon):
        assert horizon==4868 and 0<=completed<(5 if smoke else 500)
        context['step']=completed+1
        assert module.search_hparams['view_weights']==cfg['view_weights']
        assert module.macro_hparams==dict(lambda_align=cfg['lambda_align'],lambda_sparse=1.2,lambda_hierarchy=1.)
        if completed==0:
            recorded=read(runner.local.RUN/'step500/config.json');frozen(recorded,arm)
            old=read(BASE_RUN/'step500/config.json')
            for key in ('component_initialization','adapter_initialization','data','parameter_counts'):
                assert recorded[key]==old[key],key
        values=old_rates(module,completed,horizon)
        assert list(values)==expected_lrs(completed,cfg)
        return values
    def forward(module,*args,**kwargs):
        loss,logs=old_forward(module,*args,**kwargs)
        logs['alignment_view_weights']=list(module.search_hparams['view_weights'])
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
            expected=read(EXP/'CONFIG_AND_LOSS_EQUIVALENCE.json')['ranks'][dist.get_rank()]['arms'][arm]
            loss=float(context['loss']);difference=trainer.parameter_agreement(module)
            proof=dict(rank=dist.get_rank(),no_update_yet=True,loss=loss,expected_loss=expected['global_loss'],
                gradient_norms=norms,expected_gradient_norms=expected['optimizer_group_gradient_norms'],
                parameter_difference=difference,optimizer_empty=not opt.state,declared_configuration=spec,
                actual_LR={g['name']:g['lr'] for g in opt.param_groups})
            proof['passed']=(math.isclose(loss,expected['global_loss'],rel_tol=1e-6,abs_tol=1e-5)
                and all(math.isclose(norms[k],expected['optimizer_group_gradient_norms'][k],rel_tol=1e-5,abs_tol=1e-5) for k in norms)
                and difference==0 and not opt.state)
            dump(runner.local.RUN/f'PREUPDATE-rank{dist.get_rank()}.json',proof)
            verdict=torch.tensor(int(proof['passed']),device='cuda');dist.all_reduce(verdict,op=dist.ReduceOp.MIN)
            assert verdict.item(),'First update initialization/loss/gradient mismatch'
        opt.register_step_pre_hook(pre);return opt
    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            payload['alignment_weight_experiment']=spec;payload['config']['alignment_weight_experiment']=spec
        return old_save(payload,path)
    trainer.optimizer_learning_rates,trainer.atomic_save,trainer.build_optimizer=rates,save,optimizer
    BalancedSearch.forward=forward
    if smoke:runner.local.worker()
    else:runner.search.worker()


def source_scope():
    """Prove exactly two reviewed assertion expansions; every other AST node is unchanged."""
    checks={}
    for name in PRODUCTION_CHANGES:
        old=subprocess.check_output(['git','show',PARENT_COMMIT+':'+name],cwd=ROOT,text=True)
        new=(ROOT/name).read_text()
        if name.startswith('model/'):
            before="assert self.search_hparams['view_weights'] == [1.35,1.35,.30]"
            after="assert self.search_hparams['view_weights'] in (\n                [1.35,1.35,.30], [1.30,1.40,.30], [1.40,1.30,.30]\n            ), 'Unreviewed HNS alignment allocation'"
            # Only the HNS assertion changes; the Balanced guard stays strict.
            expected=old.rsplit(before,1);assert len(expected)==2
            expected=after.join(expected)
        else:
            before='[1.325,1.325,.35], [1.3,1.3,.40])'
            after='[1.325,1.325,.35], [1.3,1.3,.40], [1.30,1.40,.30], [1.40,1.30,.30])'
            assert old.count(before)==1;expected=old.replace(before,after)
        assert new==expected, 'Unapproved source diff: '+name
        a,b=ast.parse(old),ast.parse(new)
        if name.startswith('model/'):
            cls=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BalancedSearch')
            f=lambda tree:next(n for n in cls(tree).body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            assert ast.dump(f(a))==ast.dump(f(b))
            macro=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='macro_terms')
            assert ast.dump(macro(a))==ast.dump(macro(b))
        checks[name]=dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_file_sha256=sha(ROOT/name),
                         exact_reviewed_whitelist_extension=True)
    return checks


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    from model.balanced_hparam_search import trial_id
    require_gpu_idle({0,1,2,3});assert runner.git('branch','--show-current')==BRANCH
    assert runner.git('merge-base',PARENT_COMMIT,'HEAD')==PARENT_COMMIT
    assert not EXP.exists() and not RUN.exists(), 'No overwrite or automatic retry'
    EXP.mkdir(parents=True);RUN.mkdir(parents=True);state('PRECHECK_RUNNING')
    try:
        scope=source_scope();old=read(BASE_RUN/'step500/config.json')
        original_init=Path(old['init_state']).resolve(strict=True)
        assert original_init==STEP0.resolve(strict=True) and old['init_sha256']==STEP0_SHA
        assert sha(STEP0)==STEP0_SHA and sha(BASE_RUN/'step500/step000500.pt')==BASE_SHA
        production=code_manifest()
        assert sorted(k for k,v in production.items() if old['code_sha256'][k]!=v)==PRODUCTION_CHANGES
        baseline=read(BASE_EXP/'RESULTS.json')
        assert baseline['completed_steps']==500 and baseline['checkpoint_unchanged']
        assert abs(runner.quality(baseline)['Score5']-71.31437098984591)<1e-9
        for p in BASE_EXP.rglob('*'):
            if p.is_file():assert subprocess.check_output(['git','show',PARENT_COMMIT+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
        ready=read(runner.local.IMAGES.parent/'full-ready.json')
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,parent_commit=PARENT_COMMIT,
            inherited_queue_commit='815f5ab6a3e8045ad491fa4219e50913feae2b44',production_sources=production,
            authorized_production_changes=PRODUCTION_CHANGES,source_scope=scope,
            evaluator_sources=runner.evaluator_proof(),protected_files_sha256=base.protected(),
            original_common0_provenance_path=old['init_state'],resolved_common0_path=str(original_init),
            common0_sha256=STEP0_SHA,baseline_results=baseline))
        specs={}
        for arm,(weights,align) in ARMS.items():
            cfg=config(arm);frozen(cfg,arm);dump(EXP/arm/'config.json',cfg)
            specs[arm]=dict(view_weights=weights,lambda_align=align,lambda_sparse=1.2,lambda_hierarchy=1.,trial_id=trial_id(cfg))
        assert len({v['trial_id'] for v in specs.values()})==4
        dump(EXP/'FOUR_ARM_PLAN.json',dict(order=list(ARMS),arms=specs,fresh_common0=True,
            independent_smoke5_then_formal500=True,stop=500,horizon=4868,world=4,batch_per_rank=256,
            samples_per_arm=512000,original_native_LR=True,symmetric_CE=True,
            fifth_arm=False,automatic_continuation=False,parameters_frozen_before_test_results=True))
        dump(EXP/'CPU_TEST_INITIAL_IMPLEMENTATION_DIAGNOSIS.json',dict(
            initial_adhoc_suite=dict(passed=80,failed=1,seconds=21.29),
            failing_test='test_hns_s12_sparse_ratio_twoarm500.py::test_commands_are_fresh_common0_and_local_only',
            observed='Expected recovery.hns_s12_sparse_ratio_twoarm500, observed recovery.said_e2_align_weight_fourarm500',
            root_cause='New command test called configure in the shared pytest process; mutable helper bindings leaked into a later historical test',
            fix='Run new controller command checks in an isolated subprocess',
            production_math_or_experiment_config_changed=False,GPU_tasks_started_before_fix=False))
        (EXP/'HISTORICAL_DEDUP_AUDIT.md').write_text(
            '# Historical deduplication scope\n\nThe user states historical deduplication is completed and authorizes exactly A1–A4. '
            'This runner does not broaden that search or rerun the E2 baseline.\n\n'
            'Candidates: A1 [1.30,1.40,0.30]/10; A2 [1.40,1.30,0.30]/10; '
            'A3 [1.35,1.35,0.30]/9.5; A4 [1.35,1.35,0.30]/10.5. '
            'All preserve Uniform sparsity and original symmetric CE.\n\n'
            'Earlier HNS macro scales8/12, early/late LR arms, temperature95/105 and sparsity1.1/1.3 '
            'are distinct interventions and are not repeated. Four unique native trial IDs are recorded in FOUR_ARM_PLAN.json. '
            'User-provided deduplication is an input authorization, not a claim that this script independently audited every historical run.\n')
        command=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q','tests/test_said_e2_align_weight.py',
                 'tests/test_hns_macro.py','tests/test_hns_s12_sparse_ratio_twoarm500.py']
        with (RUN/'cpu-tests.log').open('x') as log:
            tested=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(EXP/'CPU_TESTS.json',dict(passed=tested.returncode==0,command=command,output=(RUN/'cpu-tests.log').read_text()))
        assert tested.returncode==0
        proof=runner.local.path_proof();dump(RUN/'local-path-proof-5000.json',proof)
        dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=proof['passed'],count=proof['count'],image_root=str(runner.local.IMAGES),NFS_fallback=False))
        screen=read(ROOT/'experiments/nest_clip_v1/said_e2_early_lr_threearm500_v1/INDEPENDENT_VALIDATION_SCREEN.json')
        meta=read(runner.local.INDEX/'metadata.json')
        assert screen['annotation_sha256']==meta['annotation_sha256'] and screen['training_index_sha256']==meta['records_sha256']
        dump(EXP/'INDEPENDENT_VALIDATION_SCREEN.json',dict(screen,current_protocol='NOT_ESTABLISHED',public_results='EXPLORATORY'))
        with (RUN/'preupdate-equivalence.log').open('x') as log:
            checked=subprocess.run(runner.torchrun('recovery.said_e2_align_weight_precheck'),cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        assert checked.returncode==0 and read(EXP/'CONFIG_AND_LOSS_EQUIVALENCE.json')['passed']
        state('PREPARED')
    except BaseException as error:state('STOPPED_WITH_EVIDENCE',error=repr(error));raise


def report_arm(arm,result,supervisor):
    ORIGINAL_REPORT(arm,result,supervisor)
    r=read(EXP/arm/'RESULTS.json');r.pop('lr_multipliers',None)
    r['declared_alignment_configuration']=dict(view_weights=ARMS[arm][0],lambda_align=ARMS[arm][1])
    r['native_LR_unchanged']=True;dump(EXP/arm/'RESULTS.json',r)
    d=read(EXP/arm/'TRAINING_DIAGNOSTICS.json')
    for point in [d['last50_views'],*d['selected_steps'].values()]:
        for v in point.values():v['weighted_CE']*=ARMS[arm][1]/10.
    oldrows=rows(BASE_RUN/'step500/steps.jsonl')
    weights0=[1.35,1.35,.3]
    for view,prefix,w0 in zip(('F','Dall','D3'),('F','O','E'),weights0):
        old_ce=statistics.fmean(x[prefix+'_i2t']+x[prefix+'_t2i'] for x in oldrows[-50:])
        old_weighted=10/3*w0*old_ce
        old_share=100*old_weighted/statistics.fmean(x['macro_weighted_align'] for x in oldrows[-50:])
        d['last50_view_delta_vs_E2'][view]['weighted_CE']=d['last50_views'][view]['weighted_CE']-old_weighted
        d['last50_view_delta_vs_E2'][view]['alignment_share_percent']=d['last50_views'][view]['alignment_share_percent']-old_share
    d['weighted_formula']='Original 10/sum(view_weights)*sum(w_i*CE_i), followed by native macro_terms lambda_align/10'
    gradient=read(EXP/arm/'GRADIENT_AUDIT.json');baseline=read(BASE_EXP/'GRADIENT_AUDIT.json')
    d['view_gradient_delta_vs_E2']={g:{view:{k:value-baseline['view_gradients'][g][view][k]
        for k,value in gradient['view_gradients'][g][view].items()} for view in ('F','Dall','D3')}
        for g in gradient['view_gradients']}
    dump(EXP/arm/'TRAINING_DIAGNOSTICS.json',d)


def combined(completed):
    models={'E2-Uniform':read(BASE_EXP/'RESULTS.json'),**{a:read(EXP/a/'RESULTS.json') for a in completed}}
    q={a:dict(runner.quality(r),Urban_Mean=(runner.quality(r)['Urban_I2T']+runner.quality(r)['Urban_T2I'])/2) for a,r in models.items()}
    dump(EXP/'FOUR_ARM_NATIVE_RESULTS.json',dict(models=models,qualities=q,completed_arms=completed,
        all_four_completed=len(completed)==4,scientific_status='EXPLORATORY'))
    dump(EXP/'FULL500_STREAM_AND_LR_PROOF.json',{a:read(EXP/a/'FULL500_STREAM_AND_LR_PROOF.json') for a in completed})
    dump(EXP/'GRADIENT_AND_MASK_DIAGNOSTICS.json',{a:dict(training=read(EXP/a/'TRAINING_DIAGNOSTICS.json'),
        masks=read(EXP/a/'MASK_HIERARCHY_AUDIT.json'),gradient=read(EXP/a/'GRADIENT_AUDIT.json')) for a in completed})
    lines=['# E2-Uniform alignment-weight four-arm @500','',
        'Four predeclared arms; original symmetric CE, HNS, Uniform sparsity, detach, AdamW/native LR and native inference are frozen. '
        'Production source changes are only two explicit whitelist additions, recorded in the checkpoint manifest.',
        'Each arm independently runs common0 smoke5, then restarts common0 formal500; scheduler horizon4868.',
        '', '| Model | View weights | lambda_align | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for a,v in q.items():
        w,l=ARMS.get(a,([1.35,1.35,.3],10.))
        lines.append(f'| {a} | {w} | {l} | '+' | '.join(f'{v[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {v["Urban_I2T"]:.3f}/{v["Urban_T2I"]:.3f} | {v["Urban_Mean"]:.3f} |')
    lines+=['','| Arm | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T/T2I | ΔUrban Mean |','|---|---:|---:|---:|---:|---|---:|']
    for a in completed:
        d=models[a]['quality_delta_pp']
        lines.append('| '+a+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {d["Urban_I2T"]:+.3f}/{d["Urban_T2I"]:+.3f} | {d["Urban_Mean"]:+.3f} |')
    for a in ARMS:
        lines+=['',f'## {a}','','Status: '+('COMPLETED' if a in completed else 'PENDING')+'.']
        if a not in completed:continue
        r=models[a];diag=read(EXP/a/'TRAINING_DIAGNOSTICS.json');grad=read(EXP/a/'GRADIENT_AUDIT.json')
        lines+=['Checkpoint SHA256: `'+r['checkpoint']['sha256']+'`.','Bare SHA256: `'+r['bare']['sha256']+'`.',
            'Trial ID: `'+r['checkpoint']['configuration']['trial_id']+'`.','',
            '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 | ΔI2T R1/R5/R10 | ΔT2I R1/R5/R10 |','|---|---|---|---|---|']
        for ds,m in r['metrics'].items():
            precision=3 if ds=='Urban-1k' else 6
            vals=[' / '.join(f'{100*m[dr][k]:.{precision}f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            vals+=[' / '.join(f'{r["recall_delta_pp"][ds][dr][k]:+.{precision}f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append('| '+ds+' | '+' | '.join(vals)+' |')
        lines+=['','Last50 raw CE, actual weighted CE/share and baseline deltas: `'+json.dumps({k:diag[k] for k in ('last50_views','last50_view_delta_vs_E2')})+'`.',
            'Last50 loss/masks/violations/IoU and baseline deltas: `'+json.dumps({k:diag[k] for k in ('last50_macro_HNS','last50_delta_vs_E2')})+'`.',
            'Four-group actual LR and gradient norms: `'+json.dumps({k:diag[k] for k in ('actual_group_LR','group_gradient_norms')})+'`.',
            'Three-view actual gradients and delta: `'+json.dumps(dict(current=grad['view_gradients'],delta=diag['view_gradient_delta_vs_E2']))+'`.',
            'Component gradient norms/cosines: `'+json.dumps(grad['group_diagnostics'])+'`.']
    if len(completed)==4:
        joint=[a for a in completed if all(models[a]['quality_delta_pp'][k]>0 for k in ('Score5','J_long3'))]
        both=[a for a in completed if all(models[a]['quality_delta_pp'][k]>0 for k in ('Urban_I2T','Urban_T2I'))]
        lines+=['','## Predeclared research questions','',
            'Q1: View reallocation results and all directional tradeoffs are given in A1/A2 recall/delta tables; no directional CE reweighting was introduced.',
            'Q2: Macro-only alignment results are A3/A4; ΔScore5='+str({a:models[a]['quality_delta_pp']['Score5'] for a in ('A3-Align95','A4-Align105')})+'.',
            'Q3: Arms jointly improving Score5 and J_long3: '+str(joint)+'.',
            'Q4: Arms improving both Urban directions: '+str(both)+'. Every opposing-direction change and regression is retained.',
            'Q5: These single-seed500 results can only nominate exploratory validation candidates. '
            'They cannot establish full4-Epoch benefits; any continuation requires a new user decision. No continuation was launched.']
    lines+=['','All values are percentages and differences percentage points. Below0.05pp single-seed gains are weak signals; Urban0.1pp is one query.',
        'Independent retrieval validation is NOT_ESTABLISHED. The public five tests have been repeatedly observed; these are exploratory results, not unbiased SOTA or independent generalization.',
        'Mechanism diagnostics describe trained states on a matched probe batch; density, IoU or a single gradient cosine is not evidence of retrieval causation.',
        'All formal streams verify512000 positions and every ID/text/token/K/index plus native LR. All checkpoints retain optimizer, scheduler, four-rank RNG/loader, sampler and cursor.',
        'Original E2 weights/results remain immutable. No fifth arm, combination, new weights, seed,1217/4868 continuation, mask inference, rerank, ensemble or TTA.']
    (EXP/'ALIGN_WEIGHT_FOURARM500_REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    files=list(SOURCE_FILES)+[str(p.relative_to(ROOT)) for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
    assert set(runner.git('diff','--cached','--name-only').splitlines())<=set(files)
    assert all((ROOT/p).stat().st_size<5*1024*1024 for p in files)
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if runner.git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' E2 alignment-weight four-arm500'],cwd=ROOT,check=True)
    head=runner.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==runner.git('rev-parse','origin/'+BRANCH)==runner.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,branch=BRANCH,
        commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def cpu_review():
    """Final metadata regression before any smoke/formal launch; no GPU computation."""
    assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED'
    previous=read(EXP/'CPU_TESTS.json')
    for arm in ARMS:
        assert config(arm)==read(EXP/arm/'config.json')
        assert config(arm)['trial_id']==read(EXP/'FOUR_ARM_PLAN.json')['arms'][arm]['trial_id']
    dump(EXP/'CPU_TESTS_BEFORE_TRIAL_METADATA_REVIEW.json',previous)
    with (RUN/'cpu-tests-final-config.log').open('x') as log:
        checked=subprocess.run(previous['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    dump(EXP/'CPU_TESTS.json',dict(passed=checked.returncode==0,command=previous['command'],
        output=(RUN/'cpu-tests-final-config.log').read_text(),
        final_native_trial_ids_verified=True,no_model_loss_LR_or_sampling_changes=True,
        controller_sha256=sha(Path(__file__)),tests_sha256=sha(ROOT/'tests/test_said_e2_align_weight.py')))
    assert checked.returncode==0


def run():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Supervisor interrupted; no automatic retry')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED'
        assert read(EXP/'CPU_TESTS.json')['passed'] and read(EXP/'CONFIG_AND_LOSS_EQUIVALENCE.json')['passed']
        completed=[];arm=None
        try:
            for arm in ARMS:
                require_gpu_idle({0,1,2,3})
                assert base.protected()==read(EXP/'BASELINE_PROVENANCE.json')['protected_files_sha256']
                assert code_manifest()==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
                runner.evaluator_proof();activate(arm)
                runtime=RUN/arm;runtime.mkdir(exist_ok=False)
                # The user's per-arm CPU acceptance precedes each independent smoke.
                command=read(EXP/'CPU_TESTS.json')['command']
                with (runtime/'cpu-tests.log').open('x') as log:
                    checked=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                dump(EXP/arm/'CPU_TESTS.json',dict(passed=checked.returncode==0,command=command,
                    output=(runtime/'cpu-tests.log').read_text()))
                assert checked.returncode==0
                runner.search.sampling_audit()
                provenance=dict(source_sha256=code_manifest(),authorized_whitelist_changes=PRODUCTION_CHANGES,
                    step0_sha256=STEP0_SHA,resume=None,arm=arm,
                    alignment_configuration=dict(view_weights=ARMS[arm][0],lambda_align=ARMS[arm][1]),
                    native_LR_unchanged=True,image_root=str(runner.local.IMAGES),NFS_fallback=False,
                    git_head=runner.git('rev-parse','HEAD'))
                activate(arm,True);runner.local.RUN.mkdir(exist_ok=False);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='smoke',stop=5))
                supervisor=runner.local.Supervisor();state('SMOKE5_RUNNING',arm,completed_arms=completed)
                supervisor.execute('smoke5',base.training_command(arm,True),training=True)
                smokeproof=identity(runner.local.RUN/'step500/step000005.pt',5,arm,'smoke')
                acceptance=read(runner.local.RUN/'step500/acceptance.json');assert acceptance['passed']
                stream=matched_stream(rows(runner.local.RUN/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],arm)
                pre=[read(runner.local.RUN/f'PREUPDATE-rank{rank}.json') for rank in range(4)]
                assert all(v['passed'] for v in pre)
                dump(EXP/arm/'SMOKE_EVIDENCE.json',dict(passed=True,independent_common0=True,
                    checkpoint=smokeproof,acceptance=acceptance,stream=stream,preupdate=pre))
                require_gpu_idle({0,1,2,3});activate(arm);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runtime/'launch-provenance.json',dict(provenance,run_type='formal',stop=500))
                dump(EXP/arm/'FORMAL_PROVENANCE.json',dict(provenance,run_type='formal',stop=500))
                supervisor=runner.local.Supervisor();state('FORMAL500_RUNNING',arm,completed_arms=completed)
                supervisor.execute('train500',base.training_command(arm),training=True)
                assert read(runtime/'step500/acceptance.json')['passed'] and read(runtime/'first-five-gate.json')['passed']
                proof=matched_stream(rows(runtime/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl'),arm)
                assert proof['records']==512000;dump(EXP/arm/'FULL500_STREAM_AND_LR_PROOF.json',proof)
                state('GRADIENT_AUDIT',arm,completed_arms=completed)
                supervisor.execute('gradient-audit500',runner.torchrun('recovery.hns_macro_gradient',
                    '--checkpoint',runtime/'step500/step000500.pt','--output',EXP/arm/'GRADIENT_AUDIT.json'))
                result=runner.evaluate(supervisor,arm);report_arm(arm,result,supervisor);completed.append(arm)
                require_gpu_idle({0,1,2,3});combined(completed)
                state('ARM_COMPLETED',arm,completed_arms=completed);publish()
            assert completed==list(ARMS);require_gpu_idle({0,1,2,3})
            assert base.protected()==read(EXP/'BASELINE_PROVENANCE.json')['protected_files_sha256']
            assert code_manifest()==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
            dump(EXP/'FINAL_REVIEW.json',dict(passed=True,arms=completed,each_exact500_updates=True,
                protected_baseline_unchanged=True,approved_source_manifest_unchanged_during_queue=True,
                source_changes_relative_to_E2=PRODUCTION_CHANGES,GPU_idle=True,no_extra_experiments=True,finished_utc=now()))
            state('COMPLETED_GPU_IDLE',completed_arms=completed);publish()
            dump(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',arms=completed,stop=500,GPU_idle=True))
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',arm,completed_arms=completed,error=repr(error));raise


def configure():
    base.BRANCH,base.EXP,base.RUN,base.ENTRY=BRANCH,EXP,RUN,ENTRY
    base.ARMS={a:(1.,1.,1.,1.) for a in ARMS};base.SOURCE_FILES=SOURCE_FILES
    base.config,base.frozen,base.activate,base.checkpoint_first5=config,frozen,activate,checkpoint_first5
    base.matched_stream,base.identity,base.state=matched_stream,identity,state
    base.prepare,base.worker,base.report_arm,base.combined,base.publish=prepare,worker,report_arm,combined,publish
    base.configure()


def main():
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(ARMS))
    for flag in ('prepare','cpu-review','publish-setup','launch','run','worker','smoke-worker'):parser.add_argument('--'+flag,action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.arm,args.smoke_worker);return
    assert not remaining
    if args.prepare:prepare()
    elif args.cpu_review:cpu_review()
    elif args.publish_setup:publish(True)
    elif args.run:run()
    elif args.launch:
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED' and not (RUN/'DETACHED_LAUNCH.json').exists()
        assert read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==runner.git('rev-parse','HEAD')
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        evidence=dict(pid=child.pid,durable=True,order=list(ARMS),log=str(RUN/'runner.log'),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',evidence);print(json.dumps(evidence),flush=True)
    else:parser.error('Choose action')


if __name__=='__main__':main()
