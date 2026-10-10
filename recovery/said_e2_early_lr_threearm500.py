"""Three frozen early LR arms, independent common0 smoke5/formal500.

Native trainer, model, optimizer groups, loss and evaluator are unchanged.
The isolated worker scales only the native optimizer_learning_rates return.
"""
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

from recovery import hns_macro_fourarm as runner
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now, distribution
from recovery.s02_local_full import expected_lrs

PROJECT = Path('/opt/data/private/lklk/SAID')
BRANCH = 'experiment/said-e2-early-lr-threearm500-v1'
PARENT_COMMIT = '6dcae270f8e754323512648e1e09ed903ab698fc'
ENTRY = 'recovery.said_e2_early_lr_threearm500'
EXP = ROOT / 'experiments/nest_clip_v1/said_e2_early_lr_threearm500_v1'
RUN = PROJECT / 'runtime/SAID-nest-clip-v1/said-e2-early-lr-threearm500-v1'
BASE_EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_sparse_ratio_twoarm500_v1/E2-Uniform'
BASE_RUN = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform'
BASE_SHA = '74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a'
GROUPS = ('backbone', 'text_mask_and_shared_pool', 'visual_mask', 'fusion_adapter')
ARMS = {'C1-Fusion085': (1., 1., 1., .85), 'C2-Fusion115': (1., 1., 1., 1.15),
        'C3-MaskBalance': (1., 1.15, .85, 1.)}
SOURCE_FILES = ('recovery/said_e2_early_lr_threearm500.py',
                'recovery/said_e2_early_lr_precheck.py',
                'tests/test_said_e2_early_lr_threearm500.py')
ORIGINAL_IDENTITY = runner.identity
CURRENT_ARM = None


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scaled_rates(values, arm):
    assert arm in ARMS and len(values) == len(GROUPS)
    return tuple(value * factor for value, factor in zip(values, ARMS[arm]))


def config(arm):
    assert arm in ARMS
    return read(BASE_EXP / 'config.json')


def frozen(cfg, arm):
    assert all(cfg.get(k) == v for k, v in config(arm).items()), 'Frozen E2 config drift'
    assert cfg['view_weights'] == [1.35, 1.35, .3]
    assert cfg['view_sparsity_weights'] == [5/3]*3
    assert [cfg[k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')] == [10.,1.2,1.]
    assert cfg['hns_enabled'] and cfg['inclusion_max'] == 0
    assert cfg.get('hns_beta', [2,2]) == [2,2] and not cfg.get('hns_detach_child', False)
    assert cfg['epochs'] == 4 and cfg['workers'] == 8 and cfg['batch_size'] == 256


def state(status, arm=None, **extra):
    dump(EXP / 'QUEUE_STATE.json', dict(status=status, active_arm=arm, order=list(ARMS),
        pid=os.getpid(), updated_utc=now(), stop_per_arm=500, horizon=4868,
        automatic_continuation=False, fourth_arm=False, automatic_retry=False, **extra))


def activate(arm, smoke=False):
    global CURRENT_ARM
    CURRENT_ARM = arm
    search = runner.search
    search.ARMS = {n:dict(axis='early_lr', weights=[1.35,1.35,.3], r=2.,
        mode='nested_detail_d3', sparsity_weights=[5/3]*3, experiment_dir=str(EXP/n)) for n in ARMS}
    search.RUN_ROOT, search.EXP = RUN, EXP
    search.ANCHOR_EXP, search.ANCHOR_RUN = BASE_EXP, BASE_RUN
    search.PHASE_PREFIX = 'said-e2-early-lr-threearm500-v1-'
    search.EDITED = set()
    search.arm_config, search.frozen_config = config, frozen
    search.matched_stream, search.checkpoint_invariants = matched_stream, checkpoint_first5
    search.activate(arm)
    if smoke:
        runner.local.RUN = RUN / (arm + '.smoke5')
        runner.local.PHASE = search.LOCAL / (search.PHASE_PREFIX + arm + '-smoke5')


def matched_stream(actual, reference, arm):
    assert len(actual) == len(reference) and 0 < len(actual) <= 500
    assert [r['step'] for r in actual] == [r['step'] for r in reference] == list(range(1,len(actual)+1))
    count, manifest, lr_manifest = 0, [], []
    for a, b in zip(actual, reference):
        assert a['epoch'] == b['epoch'] == 0 and a['s'] == a['step']-1
        assert a['nonfinite'] == 0 and math.isfinite(a['loss'])
        assert a['HNS_enabled'] and a['inc_weight'] == a['inclusion_loss'] == 0
        assert a['lambda_h'] == min(1.,a['s']/200.)
        assert [a['macro_'+k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')] == [10.,1.2,1.]
        assert [a['HNS_sparse_coeff_'+v] for v in ('F','Dall','D3')] == [5/3]*3
        assert math.isclose(a['loss'],sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy')),rel_tol=4e-6,abs_tol=2e-6)
        native = expected_lrs(a['s'], config(arm))
        assert b['actual_lrs'] == dict(zip(GROUPS,native)), 'Reference native LR drift'
        assert a['actual_lrs'] == dict(zip(GROUPS,scaled_rates(native,arm))), 'Declared LR multiplier mismatch'
        old = {h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']} == set(old) == {0,1,2,3}
        for h in a['rank_health']:
            ref = old[h['rank']]
            assert h['batch'] == ref['batch'] == 256 and h['updates'] == a['step']
            assert h['gradients_finite'] and h['stream_sha256'] == ref['stream_sha256']
            assert h['sampling'] == ref['sampling'], 'IDs/text/tokens/K/detail indices drift'
            count += h['batch']
            manifest.append([a['step'],h['rank'],h['stream_sha256'],h['sampling']])
        lr_manifest.append([a['step'],a['actual_lrs']])
    assert count == len(actual)*1024
    return dict(passed=True, optimizer_updates=len(actual), records=count, first_update=1,
        last_update=len(actual), all_sample_ids_F_Dall_D3_text_tokens_K_indices_exact=True,
        no_missing_or_duplicate_updates=True, native_schedule_unchanged=True, horizon=4868,
        expected_scaled_LR_exact=True, multipliers=dict(zip(GROUPS,ARMS[arm])),
        stream_sha256=digest(manifest), LR_sha256=digest(lr_manifest),
        reference_log_sha256=sha(BASE_RUN/'step500/steps.jsonl'))


def checkpoint_first5(current, reference):
    import torch
    arm = CURRENT_ARM
    cfg, old = current['config'], reference['config']
    frozen(cfg,arm)
    assert current['completed_steps'] == 5 and current['scheduler_horizon'] == 4868
    assert cfg['start_updates'] == 0 and cfg['resume'] is None and cfg['init_sha256'] == STEP0_SHA
    assert cfg['max_updates'] == 500 and cfg['run_type'] == 'formal'
    for key in ('component_initialization','adapter_initialization','data','parameter_counts',
                'runtime_model','horizon','batch_size','world_size','accumulation','seed',
                'sampling_seed','shuffle_seed','workers','code_sha256'):
        assert cfg[key] == old[key], key
    for a, b in zip(current['optimizer']['param_groups'],reference['optimizer']['param_groups']):
        assert {k:v for k,v in a.items() if k!='lr'} == {k:v for k,v in b.items() if k!='lr'}
    assert current['optimizer']['state'].keys() == reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()} == {5}
    assert all(torch.isfinite(v).all() for s in (current['model'],current['adapter'],*current['optimizer']['state'].values()) for v in s.values())
    assert current['early_lr_override']['multipliers'] == dict(zip(GROUPS,ARMS[arm]))
    return dict(passed=True, fresh_common0=True, initialization_and_adapter_exact=True,
        native_production_sources_exact=True, optimizer_groups_and_counters_valid=True,
        post_update_parameter_difference_expected=True)


def identity(path, step, arm, run_type='formal'):
    import torch
    result = ORIGINAL_IDENTITY(path,step,arm,run_type)
    p = torch.load(path,map_location='cpu',weights_only=False)
    assert p['early_lr_override']['multipliers'] == dict(zip(GROUPS,ARMS[arm]))
    assert p['scheduler']['lr_override'] == p['early_lr_override']
    assert p['config']['component_initialization'] == read(BASE_RUN/'step500/config.json')['component_initialization']
    assert p['config']['code_sha256'] == read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
    result['lr_override'] = p['early_lr_override']
    return result


def worker(arm, smoke):
    import torch
    import torch.distributed as dist
    from model.balanced_hparam_search import BalancedSearch
    from train import train_nested_semantic_mask as trainer
    activate(arm,smoke)
    assert '--resume' not in sys.argv
    trainer.sampling_diagnostics = runner.search.observe_selection
    old_rates, old_save, old_optimizer = trainer.optimizer_learning_rates, trainer.atomic_save, trainer.build_optimizer
    old_forward = BalancedSearch.forward
    context = {}
    override = dict(arm=arm, first_update=1, last_update=500, scheduler_horizon=4868,
        multipliers=dict(zip(GROUPS,ARMS[arm])), applied_after_native_function=True,
        controller_sha256=sha(Path(__file__)))
    reference = rows(BASE_RUN/'step500/steps.jsonl')[0]
    def rates(module, completed, horizon):
        assert horizon == 4868 and 0 <= completed < (5 if smoke else 500)
        context['step'] = completed+1
        if completed == 0:
            cfg = read(runner.local.RUN/'step500/config.json')
            frozen(cfg,arm)
            old = read(BASE_RUN/'step500/config.json')
            for key in ('component_initialization','adapter_initialization','runtime_model','data','code_sha256','parameter_counts'):
                assert cfg[key] == old[key], key
        values = old_rates(module,completed,horizon)
        assert list(values) == expected_lrs(completed,config(arm))
        return scaled_rates(values,arm)
    def forward(module,*args,**kwargs):
        loss, logs = old_forward(module,*args,**kwargs)
        if context['step'] == 1:
            context['loss'] = loss.detach().clone()
            context['logs'] = {k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()}
        keep = [float(logs['HNS_'+v+'_keep']) for v in ('F','Dall','D3')]
        assert all(math.isfinite(v) and 0<=v<=1 for v in keep)
        endpoint = 'empty' if all(v==0 for v in keep) else 'full' if all(v==1 for v in keep) else None
        context['streak'] = context.get('streak',0)+1 if endpoint and endpoint==context.get('endpoint') else 1 if endpoint else 0
        context['endpoint'] = endpoint
        if context['streak'] >= 5:
            dump(runner.local.RUN/f'MASK_FAILURE-rank{dist.get_rank()}.json',dict(endpoint=endpoint,step=context['step']))
            raise RuntimeError('HARD_STOP: complete mask degeneration for five updates')
        return loss, logs
    def optimizer(module):
        opt = old_optimizer(module)
        assert tuple(g['name'] for g in opt.param_groups) == GROUPS and not opt.state
        def before_step(opt,args,kwargs):
            if context['step'] != 1:
                return
            loss = context['loss'].clone(); dist.all_reduce(loss); loss /= 4
            ref = next(h for h in reference['rank_health'] if h['rank']==dist.get_rank())
            norms = {g['name']:float(trainer.gradient_norm(g['params'])) for g in opt.param_groups}
            difference = trainer.parameter_agreement(module)
            proof = dict(rank=dist.get_rank(), first_update=1, no_update_yet=True,
                loss=float(loss), baseline_loss=reference['loss'], gradient_norms=norms,
                baseline_gradient_norms=ref['gradient_norms'], parameter_difference=difference,
                optimizer_state_empty=not opt.state, actual_LR={g['name']:g['lr'] for g in opt.param_groups},
                initialization=read(runner.local.RUN/'step500/config.json')['component_initialization'])
            proof['passed'] = (abs(float(loss)-reference['loss']) <= 1e-5 and
                all(abs(norms[k]-ref['gradient_norms'][k]) <= 1e-4 for k in norms) and
                difference==0 and not opt.state)
            dump(runner.local.RUN/f'PREUPDATE-rank{dist.get_rank()}.json',proof)
            verdict = torch.tensor(int(proof['passed']),device='cuda');dist.all_reduce(verdict,op=dist.ReduceOp.MIN)
            assert verdict.item(), 'First loss/gradient/init equivalence failed; receipt saved before update'
        opt.register_step_pre_hook(before_step)
        return opt
    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            payload['early_lr_override'] = override
            payload['config']['early_lr_override'] = override
            payload['scheduler']['lr_override'] = override
        return old_save(payload,path)
    trainer.optimizer_learning_rates, trainer.atomic_save, trainer.build_optimizer = rates, save, optimizer
    BalancedSearch.forward = forward
    if smoke:
        runner.local.worker()
    else:
        runner.search.worker()


def training_command(arm, smoke=False):
    runtime = RUN / (arm+'.smoke5' if smoke else arm)
    return runner.torchrun(ENTRY,'--arm',arm,'--smoke-worker' if smoke else '--worker',
        '--config',EXP/arm/'config.json','--init-state',STEP0,
        '--index-dir',runner.local.INDEX,'--image-root',runner.local.IMAGES,
        '--output-dir',runtime/'step500','--run-type','smoke' if smoke else 'formal',
        '--max-updates',5 if smoke else 500)


def protected():
    paths = [STEP0, BASE_RUN/'step500/step000500.pt',BASE_RUN/'step500/step000005.pt',
             BASE_RUN/'step500/config.json',BASE_RUN/'step500/steps.jsonl']
    paths += [p for p in BASE_EXP.rglob('*') if p.is_file()]
    result = read(BASE_EXP/'RESULTS.json')
    paths.append(Path(result['bare']['path']))
    paths += [p for p in (BASE_RUN/'evaluations').rglob('*') if p.is_file() and p.suffix=='.json']
    return {str(p):sha(p) for p in paths}


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    require_gpu_idle({0,1,2,3})
    assert runner.git('branch','--show-current') == BRANCH
    assert runner.git('merge-base',PARENT_COMMIT,'HEAD') == PARENT_COMMIT
    assert not EXP.exists() and not RUN.exists(), 'Never overwrite or retry an experiment'
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    state('PRECHECK_RUNNING')
    try:
        assert sha(STEP0)==STEP0_SHA and sha(BASE_RUN/'step500/step000500.pt')==BASE_SHA
        cfg = read(BASE_RUN/'step500/config.json')
        production = code_manifest()
        assert cfg['code_sha256'] == production
        assert all(sha(ROOT/p)==hashlib.sha256(subprocess.check_output(['git','show',PARENT_COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in production)
        baseline = read(BASE_EXP/'RESULTS.json')
        assert baseline['completed_steps']==500 and baseline['checkpoint_unchanged']
        q=runner.quality(baseline)
        assert abs(q['Score5']-71.31437098984591)<1e-9
        assert round(q['Urban_I2T'],3)==91.1 and round(q['Urban_T2I'],3)==90.
        for p in BASE_EXP.rglob('*'):
            if p.is_file():
                rel=str(p.relative_to(ROOT))
                assert subprocess.check_output(['git','show',PARENT_COMMIT+':'+rel],cwd=ROOT)==p.read_bytes()
        ready=read(runner.local.IMAGES.parent/'full-ready.json')
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        eval_sources=runner.evaluator_proof()
        hashes=protected()
        dump(EXP/'BASELINE_PROVENANCE.json',dict(passed=True,parent_commit=PARENT_COMMIT,
            production_sources=production,evaluator_sources=eval_sources,protected_files_sha256=hashes,
            common0_sha256=STEP0_SHA,baseline_checkpoint_sha256=BASE_SHA,baseline_results=baseline))
        for arm in ARMS:
            dump(EXP/arm/'config.json',config(arm));frozen(config(arm),arm)
        dump(EXP/'THREE_ARM_PLAN.json',dict(order=list(ARMS),multipliers={a:dict(zip(GROUPS,m)) for a,m in ARMS.items()},
            common0_sha256=STEP0_SHA,resume=None,independent_smoke5_then_fresh_formal500=True,
            stop_updates=500,horizon=4868,world=4,batch_per_rank=256,samples_per_arm=512000,
            evaluation_mapping=dict(COCO=0,DOCCI=1,Long_DCI=2,Flickr_then_Urban=3),
            model_loss_optimizer_data_evaluation_math_unchanged=True,fourth_arm=False,automatic_continuation=False))
        cmd=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q',SOURCE_FILES[-1],
             'tests/test_hns_s12_sparse_ratio_twoarm500.py','tests/test_hns_macro.py']
        with (RUN/'cpu-tests.log').open('x') as log:
            checked=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(EXP/'CPU_TESTS.json',dict(passed=checked.returncode==0,returncode=checked.returncode,
            command=cmd,output=(RUN/'cpu-tests.log').read_text()))
        assert checked.returncode==0, 'CPU test failure'
        proof=runner.local.path_proof()
        dump(RUN/'local-path-proof-5000.json',proof)
        dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=proof['passed'],count=proof['count'],
            image_root=str(runner.local.IMAGES),index=str(runner.local.INDEX),NFS_fallback=False))
        # Previous exact leakage screen is read-only evidence, with input fingerprints rechecked.
        screen_path=PROJECT/'../said-e2-worktrees/late-lr-fourarm4868/experiments/nest_clip_v1/said_e2_late_lr_fourarm4868_v1/INDEPENDENT_VALIDATION_SCREEN.json'
        screen=read(screen_path)
        meta=read(runner.local.INDEX/'metadata.json')
        assert screen['annotation_sha256']==meta['annotation_sha256'] and screen['training_index_sha256']==meta['records_sha256']
        assert sha(runner.local.INDEX/'records.jsonl')==meta['records_sha256']
        assert sha(Path(meta['annotation']))==meta['annotation_sha256']
        dump(EXP/'INDEPENDENT_VALIDATION_SCREEN.json',dict(screen,evidence_reused=True,
            evidence_source=str(screen_path.resolve()),evidence_sha256=sha(screen_path),
            input_fingerprints_reverified=True,selection_independent_of_test_results=False))
        command=runner.torchrun('recovery.said_e2_early_lr_precheck')
        with (RUN/'preupdate-equivalence.log').open('x') as log:
            checked=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        assert checked.returncode==0, 'Real BF16 DDP preupdate failed; receipt/log saved'
        assert read(EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json')['passed']
        state('PREPARED')
    except BaseException as error:
        state('STOPPED_WITH_EVIDENCE',error=repr(error));raise


def report_arm(arm, result, supervisor):
    from recovery.nested_d3_local_search_evidence import diagnostics
    activate(arm)
    runtime, exp = RUN/arm, EXP/arm
    actual=rows(runtime/'step500/steps.jsonl');reference=rows(BASE_RUN/'step500/steps.jsonl')
    proof=matched_stream(actual,reference,arm);assert proof['records']==512000
    diag,masks=diagnostics(actual,arm)
    baseline_diag,_=diagnostics(reference,arm)
    diag['last50_view_delta_vs_E2']={v:{k:value-baseline_diag['last50_views'][v][k]
        for k,value in measures.items()} for v,measures in diag['last50_views'].items()}
    keys=[k for k in actual[0] if k.startswith(('HNS_','macro_')) or k in ('V_DF_hard','V_3D_hard','lambda_h','loss')]
    means={k:statistics.fmean(float(r[k]) for r in actual[-50:]) for k in keys}
    oldmeans={k:statistics.fmean(float(r[k]) for r in reference[-50:]) for k in keys}
    diag.update(last50_macro_HNS=means,last50_delta_vs_E2={k:v-oldmeans[k] for k,v in means.items()},
        actual_group_LR={g:dict(first=actual[0]['actual_lrs'][g],last=actual[-1]['actual_lrs'][g]) for g in GROUPS},
        group_gradient_norms={g:dict(min=min(h['gradient_norms'][g] for r in actual for h in r['rank_health']),
            max=max(h['gradient_norms'][g] for r in actual for h in r['rank_health']),
            last50_mean=statistics.fmean(h['gradient_norms'][g] for r in actual[-50:] for h in r['rank_health']),
            baseline_last50_mean=statistics.fmean(h['gradient_norms'][g] for r in reference[-50:] for h in r['rank_health'])) for g in GROUPS})
    dump(exp/'LAST50_TRAINING_CURVE.json',[{k:r[k] for k in ['step','actual_lrs',*keys]} for r in actual[-50:]])
    masks.update(HNS_last50=means,delta_vs_E2={k:means[k]-oldmeans[k] for k in means if k.startswith('HNS_')},
        density_order=means['HNS_F_keep']>=means['HNS_Dall_keep']>=means['HNS_D3_keep'])
    gradient=read(exp/'GRADIENT_AUDIT.json');assert gradient['passed'] and gradient['no_parameter_updates']
    base_grad=read(BASE_EXP/'GRADIENT_AUDIT.json')
    assert gradient['sample_ids_sha256']==base_grad['sample_ids_sha256']
    diag['gradient_group_delta_vs_E2']={g:{k:None if v is None or base_grad['group_diagnostics'][g][k] is None else v-base_grad['group_diagnostics'][g][k]
        for k,v in values.items()} for g,values in gradient['group_diagnostics'].items()}
    result.update(arm=arm,lr_multipliers=dict(zip(GROUPS,ARMS[arm])),**runner.compare(result,read(BASE_EXP/'RESULTS.json')),
        scientific_status='EXPLORATORY_REPEATED_PUBLIC_BENCHMARKS',fresh_common0=True)
    result['quality_delta_pp']['Urban_Mean']=(result['quality_delta_pp']['Urban_I2T']+result['quality_delta_pp']['Urban_T2I'])/2
    cycles=rows(runtime/'step500/cycle_timing.jsonl')
    assert [r['step'] for r in cycles]==list(range(1,501))
    stats=dict(cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),
        peak_allocated_GiB={str(rank):max(h['peak_allocated_gib'] for r in actual for h in r['rank_health'] if h['rank']==rank) for rank in range(4)},
        commands=supervisor.commands,source='actual four-rank global1024 formal500')
    for name,value in [('RESULTS',result),('FULL500_STREAM_AND_LR_PROOF',proof),('TRAINING_DIAGNOSTICS',diag),
        ('MASK_HIERARCHY_AUDIT',masks),('RUNTIME_STATS',stats),('EXPORT_AUDIT',result['strict_export'])]:
        dump(exp/(name+'.json'),value)
    acceptance=read(runtime/'step500/acceptance.json')
    assert acceptance['passed'] and all(r['completed_updates']==r['updates_this_run']==500 and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
    preupdate=[read(runtime/f'PREUPDATE-rank{rank}.json') for rank in range(4)]
    assert all(r['passed'] for r in preupdate)
    dump(exp/'VALIDATION.json',dict(passed=True,first_preupdate=preupdate,
        formal_first5=read(runtime/'first-five-gate.json'),smoke=read(exp/'SMOKE_EVIDENCE.json'),
        acceptance=acceptance,full_stream=proof,strict_export=result['strict_export'],gradient_audit_passed=True))


def combined(completed):
    results={'E2-Uniform':read(BASE_EXP/'RESULTS.json'),**{arm:read(EXP/arm/'RESULTS.json') for arm in completed}}
    qualities={n:dict(runner.quality(v),Urban_Mean=(runner.quality(v)['Urban_I2T']+runner.quality(v)['Urban_T2I'])/2) for n,v in results.items()}
    dump(EXP/'THREE_ARM_NATIVE_RESULTS.json',dict(completed_arms=completed,models=results,qualities=qualities,
        scientific_status='EXPLORATORY_REPEATED_PUBLIC_BENCHMARKS',all_declared_arms_reported=len(completed)==3))
    dump(EXP/'THREE_ARM_FULL_STREAM_PROOF.json',{arm:read(EXP/arm/'FULL500_STREAM_AND_LR_PROOF.json') for arm in completed})
    dump(EXP/'THREE_ARM_GRADIENT_MASK_DIAGNOSTICS.json',{arm:dict(training=read(EXP/arm/'TRAINING_DIAGNOSTICS.json'),
        masks=read(EXP/arm/'MASK_HIERARCHY_AUDIT.json'),gradient=read(EXP/arm/'GRADIENT_AUDIT.json')) for arm in completed})
    lines=['# E2-Uniform early LR three-arm @500','',
        'Each arm uses independent common0 smoke5, then fresh common0 formal500. Only native LR return values are multiplied. Horizon4868 stays unchanged.',
        '', '| Model | BB/Text mask/Visual mask/Fusion multipliers | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for n,q in qualities.items():
        factors=ARMS.get(n,(1,1,1,1))
        lines.append('| '+n+' | '+str(factors)+' | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {q["Urban_I2T"]:.3f}/{q["Urban_T2I"]:.3f} | {q["Urban_Mean"]:.3f} |')
    lines += ['', 'All values are percentages; deltas below are percentage points against E2@500.', '',
        '| Arm | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T/T2I | ΔUrban Mean |',
        '|---|---:|---:|---:|---:|---|---:|']
    for arm in completed:
        d=results[arm]['quality_delta_pp']
        lines.append('| '+arm+' | '+' | '.join(f'{d[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {d["Urban_I2T"]:+.3f}/{d["Urban_T2I"]:+.3f} | {d["Urban_Mean"]:+.3f} |')
    for arm in ARMS:
        lines += ['', f'## {arm}', '', 'Status: '+('COMPLETED' if arm in completed else 'PENDING')+'.']
        if arm not in completed:continue
        r=results[arm]
        lines += ['Checkpoint SHA256: `'+r['checkpoint']['sha256']+'`.',
            'Bare student SHA256: `'+r['bare']['sha256']+'`.', '',
            '| Dataset | I2T R@1/5/10 | T2I R@1/5/10 | ΔI2T R@1/5/10 | ΔT2I R@1/5/10 |',
            '|---|---|---|---|---|']
        for ds,m in r['metrics'].items():
            vals=[' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            vals += [' / '.join(f'{r["recall_delta_pp"][ds][dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append('| '+ds+' | '+' | '.join(vals)+' |')
        diag=read(EXP/arm/'TRAINING_DIAGNOSTICS.json');g=read(EXP/arm/'GRADIENT_AUDIT.json')
        lines += ['', 'Last50 per-view CE/share: `'+json.dumps(diag['last50_views'])+'`.',
            'Last50 per-view delta versus E2: `'+json.dumps(diag['last50_view_delta_vs_E2'])+'`.',
            'Last50 mask/loss and baseline deltas: `'+json.dumps({k:diag[k] for k in ('last50_macro_HNS','last50_delta_vs_E2')})+'`.',
            'Four group LR/gradient norms: `'+json.dumps({k:diag[k] for k in ('actual_group_LR','group_gradient_norms')})+'`.',
            'Read-only matched-batch component gradients/cosines: `'+json.dumps(g['group_diagnostics'])+'`.',
            'Component-gradient delta versus E2: `'+json.dumps(diag['gradient_group_delta_vs_E2'])+'`.']
    if len(completed)==3:
        joint=[a for a in completed if all(results[a]['quality_delta_pp'][k]>0 for k in ('Score5','Urban_I2T','Urban_T2I'))]
        lines += ['', 'Arms jointly improving Score5 and both Urban directions: '+str(joint)+'.',
            'All dataset/direction deltas, including regressions, are retained. No formal model selection or continuation is performed.']
    lines += ['', 'Public tests, including Urban, have been repeatedly observed: these single-seed500 results are exploratory, not independent generalization or unbiased SOTA evidence.',
        'No trusted independent retrieval validation protocol established. The previously skipped1000 candidate has an exact caption duplicate in training; content/perceptual and semantic leakage remain unverified. Fingerprints were rechecked and evidence reused without claiming a new screen.',
        'Urban0.1pp equals one query per direction. Small gains are not statistically established. Lower density, higher IoU or favorable probe cosines alone do not establish better semantic selection.',
        'Production model/loss/trainer/data/evaluator files are unchanged. Every500-step stream covers512000 sample positions with exact text/token/K/index matching and declared scaled LR.',
        'The last50 curve, full30 recalls/deltas, restoreable checkpoint audit, resource/gradient/mask evidence and individual success statuses are in adjacent JSONs.',
        'The initial read-only probe failed before optimizer creation because in-memory integer keys/tuples were compared directly against JSON string keys/lists. CPU canonicalization proved the full sampling summary exact. The initial receipt/log were preserved; only the isolated comparator was corrected. PREUPDATE_EQUIVALENCE_VERIFIED.json records the subsequent full-gradient acceptance.',
        'Strict stop500: no fourth arm, new multiplier, seed or1217/2434/3651/4868 training.']
    (EXP/'EARLY_LR_THREEARM500_REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    files=list(SOURCE_FILES)+[str(p.relative_to(ROOT)) for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
    assert set(runner.git('diff','--cached','--name-only').splitlines()) <= set(files)
    assert all((ROOT/p).stat().st_size<5*1024*1024 for p in files), 'Large artifacts stay local'
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if runner.git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' E2 early LR three-arm500'],cwd=ROOT,check=True)
    head=runner.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==runner.git('rev-parse','origin/'+BRANCH)==runner.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,
        branch=BRANCH,commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def run():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Supervisor interrupted; no automatic retry')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED'
        assert read(EXP/'CPU_TESTS.json')['passed']
        assert read(EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json')['passed']
        completed=[];arm=None
        try:
            for arm in ARMS:
                require_gpu_idle({0,1,2,3})
                assert protected()==read(EXP/'BASELINE_PROVENANCE.json')['protected_files_sha256']
                assert code_manifest()==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
                runner.evaluator_proof();activate(arm)
                runtime=RUN/arm;runtime.mkdir(exist_ok=False)
                runner.search.sampling_audit()
                provenance=dict(source_sha256=code_manifest(),step0_sha256=STEP0_SHA,resume=None,
                    arm=arm,LR_multipliers=dict(zip(GROUPS,ARMS[arm])),image_root=str(runner.local.IMAGES),
                    NFS_fallback=False,git_head=runner.git('rev-parse','HEAD'))
                activate(arm,True);runner.local.RUN.mkdir(exist_ok=False);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runner.local.RUN/'launch-provenance.json',dict(provenance,run_type='smoke',stop=5))
                supervisor=runner.local.Supervisor();state('SMOKE5_RUNNING',arm,completed_arms=completed)
                supervisor.execute('smoke5',training_command(arm,True),training=True)
                smokeproof=identity(runner.local.RUN/'step500/step000005.pt',5,arm,'smoke')
                acceptance=read(runner.local.RUN/'step500/acceptance.json');assert acceptance['passed']
                stream=matched_stream(rows(runner.local.RUN/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],arm)
                pre=[read(runner.local.RUN/f'PREUPDATE-rank{rank}.json') for rank in range(4)]
                assert all(r['passed'] for r in pre)
                dump(EXP/arm/'SMOKE_EVIDENCE.json',dict(passed=True,independent_common0=True,
                    checkpoint=smokeproof,acceptance=acceptance,stream=stream,preupdate=pre))
                require_gpu_idle({0,1,2,3});activate(arm);runner.local.PHASE.mkdir(exist_ok=False)
                dump(runtime/'launch-provenance.json',dict(provenance,run_type='formal',stop=500))
                dump(EXP/arm/'FORMAL_PROVENANCE.json',dict(provenance,run_type='formal',stop=500))
                supervisor=runner.local.Supervisor();state('FORMAL500_RUNNING',arm,completed_arms=completed)
                supervisor.execute('train500',training_command(arm),training=True)
                assert read(runtime/'step500/acceptance.json')['passed']
                assert read(runtime/'first-five-gate.json')['passed']
                proof=matched_stream(rows(runtime/'step500/steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl'),arm)
                assert proof['records']==512000
                dump(EXP/arm/'FULL500_STREAM_AND_LR_PROOF.json',proof)
                state('GRADIENT_AUDIT',arm,completed_arms=completed)
                supervisor.execute('gradient-audit500',runner.torchrun('recovery.hns_macro_gradient',
                    '--checkpoint',runtime/'step500/step000500.pt','--output',EXP/arm/'GRADIENT_AUDIT.json'))
                result=runner.evaluate(supervisor,arm)
                report_arm(arm,result,supervisor);completed.append(arm)
                require_gpu_idle({0,1,2,3});combined(completed)
                state('ARM_COMPLETED',arm,completed_arms=completed);publish()
            assert completed==list(ARMS)
            require_gpu_idle({0,1,2,3})
            assert protected()==read(EXP/'BASELINE_PROVENANCE.json')['protected_files_sha256']
            assert code_manifest()==read(EXP/'BASELINE_PROVENANCE.json')['production_sources']
            dump(EXP/'FINAL_REVIEW.json',dict(passed=True,arms=completed,each_exact500_updates=True,
                protected_baseline_unchanged=True,production_sources_unchanged=True,GPU_idle=True,
                no_extra_experiments=True,finished_utc=now()))
            state('COMPLETED_GPU_IDLE',completed_arms=completed);publish()
            dump(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',arms=completed,stop=500,GPU_idle=True))
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',arm,completed_arms=completed,error=repr(error))
            raise


def configure():
    runner.BRANCH,runner.EXP,runner.RUN,runner.ENTRY = BRANCH,EXP,RUN,ENTRY
    runner.ARMS,runner.BASE_EXP,runner.BASE_RUN = ARMS,BASE_EXP,BASE_RUN
    runner.config,runner.frozen,runner.activate,runner.identity = config,frozen,activate,identity
    runner.matched_stream,runner.training_command = matched_stream,training_command
    runner.state = state


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list(ARMS))
    for name in ('prepare','publish-setup','launch','run','worker','smoke-worker','complete-precheck'):
        parser.add_argument('--'+name,action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.arm,args.smoke_worker);return
    assert not remaining
    if args.prepare:prepare()
    elif args.complete_precheck:
        assert read(EXP/'QUEUE_STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
        assert read(EXP/'PRECHECK_IMPLEMENTATION_DIAGNOSIS.json')['JSON_canonicalized_full_sampling_exact']
        assert read(EXP/'CPU_TESTS.json')['passed']
        assert not (RUN/'preupdate-equivalence-verified.log').exists()
        from tools.eval_five_parallel import require_gpu_idle
        require_gpu_idle({0,1,2,3})
        state('CORRECTED_READONLY_PRECHECK_RUNNING')
        try:
            command=runner.torchrun('recovery.said_e2_early_lr_precheck')
            with (RUN/'preupdate-equivalence-verified.log').open('x') as log:
                checked=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
            assert checked.returncode==0 and read(EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json')['passed']
            state('PREPARED')
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',error=repr(error));raise
    elif args.publish_setup:publish(True)
    elif args.run:run()
    elif args.launch:
        assert read(EXP/'QUEUE_STATE.json')['status']=='PREPARED' and not (RUN/'DETACHED_LAUNCH.json').exists()
        assert read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==runner.git('rev-parse','HEAD')
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,durable=True,log=str(RUN/'runner.log'),order=list(ARMS),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',launch);print(json.dumps(launch),flush=True)
    else:parser.error('Choose action')


if __name__=='__main__':main()
