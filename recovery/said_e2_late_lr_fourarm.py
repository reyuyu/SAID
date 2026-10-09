"""Four preregistered E2 epoch4 LR arms; native trainer and math unchanged.

Only the return value of optimizer_learning_rates is scaled, after the native
4868-horizon function. Every arm resumes the same immutable step3651 payload.
"""
import argparse
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

from recovery import hns_s12_2434_validation as protocol
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now
from recovery.s02_local_full import expected_lrs as native_expected_lrs

PROJECT = Path('/opt/data/private/lklk/SAID')
BRANCH = 'experiment/said-e2-late-lr-fourarm4868-v1'
PARENT_COMMIT = '6dcae270f8e754323512648e1e09ed903ab698fc'
PARENT_RUN = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-uniform-full4868-v1'
PARENT = PARENT_RUN / 'step3651/training/step003651.pt'
PARENT_SHA = '4e74f7e44128756f8398e26c97bc7c8f905aa8e0898baced56a266e5ed1b2156'
BASE_EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1'
EXP = ROOT / 'experiments/nest_clip_v1/said_e2_late_lr_fourarm4868_v1'
RUN = PROJECT / 'runtime/SAID-nest-clip-v1/said-e2-late-lr-fourarm4868-v1'
ENTRY = 'recovery.said_e2_late_lr_fourarm'
GROUPS = ('backbone', 'text_mask_and_shared_pool', 'visual_mask', 'fusion_adapter')
ARMS = {'B1-BB085': (0.85, 1., 1., 1.), 'B2-BB115': (1.15, 1., 1., 1.),
        'B3-MASK085': (1., 0.85, 0.85, 1.), 'B4-MASK115': (1., 1.15, 1.15, 1.)}
START, STOP = 3651, 4868
ORIGINAL_IDENTITY = protocol.identity
CURRENT_ARM = None


def read(path):
    return json.loads(Path(path).read_text())


def scaled_rates(values, arm, completed):
    assert arm in ARMS and len(values) == 4
    return tuple(v * m if completed >= START else v for v, m in zip(values, ARMS[arm]))


def config():
    return read(BASE_EXP / 'config.json')


def frozen(cfg):
    assert all(cfg.get(k) == v for k, v in config().items()), 'Frozen E2 configuration drift'
    assert [cfg[k] for k in ('lambda_align', 'lambda_sparse', 'lambda_hierarchy')] == [10., 1.2, 1.]
    assert cfg['view_weights'] == [1.35, 1.35, .3] and cfg['view_sparsity_weights'] == [5/3]*3
    assert cfg['hns_enabled'] and cfg['inclusion_max'] == 0 and not cfg.get('hns_detach_child', False)


def arm_run(arm):
    assert arm in ARMS
    return RUN / arm


def arm_exp(arm):
    assert arm in ARMS
    return EXP / arm / 'step4868'


def phase(arm):
    return protocol.local.IMAGES.parent / ('formal-said-e2-late-lr-' + arm)


def identity(path, completed):
    protocol.frozen = frozen
    proof = ORIGINAL_IDENTITY(path, completed)
    if Path(path) == PARENT:
        assert completed == START and proof['sha256'] == PARENT_SHA
    else:
        import torch
        p = torch.load(path, map_location='cpu', weights_only=False)
        assert completed == STOP and p['config']['start_updates'] == START
        assert p['config']['resume'] == str(PARENT) and p['resume_lineage']['sha256'] == PARENT_SHA
        assert p['late_lr_override']['multipliers'] == dict(zip(GROUPS, ARMS[CURRENT_ARM]))
        assert p['scheduler']['lr_override'] == p['late_lr_override']
        assert p['config']['updates_planned_this_run'] == STOP-START
        proof.update(late_lr_override=p['late_lr_override'], restored_from_sha256=PARENT_SHA,
                     updates_this_run=STOP-START)
    return proof


def activate(arm):
    global CURRENT_ARM
    CURRENT_ARM = arm
    protocol.BRANCH, protocol.EXP, protocol.RUN = BRANCH, EXP / arm, RUN
    protocol.PARENT, protocol.PARENT_SHA = PARENT, PARENT_SHA
    protocol.frozen, protocol.identity = frozen, identity
    protocol.segment = lambda target: arm_run(arm)
    protocol.phase = lambda target: phase(arm)
    protocol.predecessor = lambda target: (PARENT, START)
    def configure(target):
        assert target == STOP
        protocol.local.RUN = arm_run(arm)
        protocol.local.PHASE = phase(arm)
        protocol.local.CONFIG = EXP / 'config.json'
    protocol.configure = configure
    protocol.expected_lrs = lambda s, cfg: list(scaled_rates(native_expected_lrs(s, cfg), arm, s))
    configure(STOP)


def state(status, **extra):
    dump(EXP / 'STATE.json', dict(status=status, updated_utc=now(), pid=os.getpid(),
        order=list(ARMS), start=START, stop=STOP, automatic_fifth_arm=False,
        automatic_fifth_epoch=False, **extra))


def training_command(arm):
    return protocol.common.torchrun(ENTRY, '--worker', '--arm', arm,
        '--config', EXP / 'config.json', '--init-state', STEP0, '--resume', PARENT,
        '--index-dir', protocol.local.INDEX, '--image-root', protocol.local.IMAGES,
        '--output-dir', arm_run(arm) / 'training', '--run-type', 'formal', '--max-updates', STOP)


def worker(arm):
    """Reuse native resume audits, adding LR metadata and first5 update receipts."""
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from recovery import nested_d3_local_search as search
    activate(arm)
    original_rates, original_save = trainer.optimizer_learning_rates, trainer.atomic_save
    original_optimizer, original_observe = trainer.build_optimizer, search.observe_selection
    original_validate = trainer.validate_resume_payload
    context = {}
    override = dict(arm=arm, first_update=START+1, last_update=STOP,
        multipliers=dict(zip(GROUPS, ARMS[arm])), applied_after_native_function=True,
        scheduler_horizon=STOP, controller_sha256=sha(Path(__file__)))
    def validate(payload, cfg, *args):
        cfg['late_lr_override'] = override
        return original_validate(payload, cfg, *args)
    def optimizer(module):
        context['optimizer'] = original_optimizer(module)
        assert tuple(g['name'] for g in context['optimizer'].param_groups) == GROUPS
        return context['optimizer']
    def rates(module, completed, horizon):
        context.update(module=module, step=completed+1)
        values = original_rates(module, completed, horizon)
        return scaled_rates(values, arm, completed)
    def save(payload, path):
        if 'model' in payload and 'optimizer' in payload:
            payload['late_lr_override'] = override
            payload['config']['late_lr_override'] = override
            payload['scheduler']['lr_override'] = override
        return original_save(payload, path)
    def observe(batch):
        step = context['step']
        if START < step <= START+5:
            opt, module = context['optimizer'], context['module']
            assert {int(s['step']) for s in opt.state.values()} == {step}
            assert all(torch.isfinite(p).all() for p in module.parameters())
            diff = trainer.parameter_agreement(module)
            assert diff == 0
            dump(arm_run(arm) / f'post-{step}-rank{dist.get_rank()}.json', dict(passed=True,
                step=step, optimizer_counters=[step], parameter_difference=diff,
                actual_lrs={g['name']: g['lr'] for g in opt.param_groups}))
        return original_observe(batch)
    trainer.build_optimizer, trainer.optimizer_learning_rates = optimizer, rates
    trainer.atomic_save, search.observe_selection = save, observe
    trainer.validate_resume_payload = validate
    protocol.worker(STOP)


def full_stream(arm):
    actual = rows(arm_run(arm) / 'training/steps.jsonl')
    reference_path = PARENT_RUN / 'step4868/training/steps.jsonl'
    reference = rows(reference_path)
    assert [r['step'] for r in actual] == [r['step'] for r in reference] == list(range(START+1, STOP+1))
    records, manifests, lr_rows = 0, [], []
    cfg = config()
    for a, b in zip(actual, reference):
        assert a['epoch'] == b['epoch'] == 3 and a['s'] == a['step']-1
        assert a['nonfinite'] == 0 and a['HNS_enabled'] and a['lambda_h'] == 1.
        assert a['inc_weight'] == a['inclusion_loss'] == 0
        assert [a['macro_'+k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')] == [10.,1.2,1.]
        expected = dict(zip(GROUPS, scaled_rates(native_expected_lrs(a['s'], cfg), arm, a['s'])))
        assert a['actual_lrs'] == expected
        assert b['actual_lrs'] == dict(zip(GROUPS, native_expected_lrs(a['s'], cfg)))
        assert math.isclose(a['loss'], sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy')), rel_tol=4e-6, abs_tol=2e-6)
        old = {h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']} == set(old) == {0,1,2,3}
        for h in a['rank_health']:
            ref = old[h['rank']]
            assert h['updates'] == a['step']-START and h['batch'] == ref['batch']
            assert h['gradients_finite'] and h['stream_sha256'] == ref['stream_sha256']
            assert h['sampling'] == ref['sampling'], 'Full text/token/K/detail index mismatch'
            records += h['batch']
            manifests.append([a['step'], h['rank'], h['stream_sha256'], h['sampling']])
        lr_rows.append([a['step'], expected])
    assert records == 1245904 and all(h['batch'] == 180 for h in actual[-1]['rank_health'])
    def digest(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    proof = dict(passed=True, arm=arm, optimizer_updates=1217, first_update=3652, last_update=4868,
        samples_all_ranks=records, last_batch_per_rank=180, no_missing_or_duplicate_steps=True,
        all_sample_ids_F_Dall_D3_text_tokens_K_indices_exact=True, full_stage_stream_sha256=digest(manifests),
        expected_LR_exact=True, lr_rows_sha256=digest(lr_rows), multipliers=dict(zip(GROUPS, ARMS[arm])),
        reference=dict(path=str(reference_path), sha256=sha(reference_path)),
        actual=dict(path=str(arm_run(arm)/'training/steps.jsonl'),sha256=sha(arm_run(arm)/'training/steps.jsonl')))
    dump(arm_exp(arm) / 'FULL_STREAM_AND_LR_PROOF.json', proof)
    return actual, proof


def report_arm(arm, supervisor, result, proof):
    records = rows(arm_run(arm) / 'training/steps.jsonl')
    keys = [k for k in records[0] if k.startswith(('HNS_', 'macro_')) or k in
            ('loss','F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i')]
    last = {k:statistics.fmean(float(r[k]) for r in records[-50:]) for k in keys}
    grad_norms = {g:dict(min=min(h['gradient_norms'][g] for r in records for h in r['rank_health']),
        max=max(h['gradient_norms'][g] for r in records for h in r['rank_health']),
        last50_mean=statistics.fmean(h['gradient_norms'][g] for r in records[-50:] for h in r['rank_health'])) for g in GROUPS}
    audit = read(arm_exp(arm)/'GRADIENT_AUDIT.json')
    assert audit['passed'] and audit['no_parameter_updates']
    acceptance = read(arm_run(arm)/'training/acceptance.json')
    dump(arm_exp(arm)/'TRAINING_DIAGNOSTICS.json', dict(last50=last,
        per_step_group_gradient_health=grad_norms, gradient_probe=audit,
        density_order=last['HNS_F_keep']>=last['HNS_Dall_keep']>=last['HNS_D3_keep'],
        acceptance=acceptance, commands=read(arm_run(arm)/'commands.json'),
        full_stream_proof=proof, no_model_or_loss_math_change=True))
    dump(arm_exp(arm)/'RESULTS.json', dict(result, arm=arm, lr_multipliers=dict(zip(GROUPS,ARMS[arm])),
        scientific_status='EXPLORATORY_REPEATED_PUBLIC_BENCHMARKS', fresh_from_shared3651=True))
    q = protocol.common.quality(result)
    lines = [f'# {arm} epoch4 LR experiment', '', f'LR multipliers: {dict(zip(GROUPS,ARMS[arm]))}', '',
        f'All 1217 optimizer updates and {proof["samples_all_ranks"]} sample positions verified.',
        f'Checkpoint SHA256: {result["checkpoint"]["sha256"]}',
        f'Scores: {json.dumps(q)}', f'Delta pp: {json.dumps(result["comparisons"])}', '',
        'All recalls and diagnostics are saved in JSON. Public benchmark results are exploratory.',
        'No changes to loss, model, detach, data, batch or evaluator mathematics.']
    (arm_exp(arm)/'REPORT.md').write_text('\n'.join(lines)+'\n')


def combined(completed):
    baseline = read(BASE_EXP/'step4868/RESULTS.json')
    models = {'E2-Uniform':baseline, **{arm:read(arm_exp(arm)/'RESULTS.json') for arm in completed}}
    dump(EXP/'FOUR_ARM_NATIVE_RESULTS.json', dict(status='COMPLETED' if len(completed)==4 else 'PARTIAL',
        models=models, complete_order=completed, exploratory=True, independent_validation_verified=False,
        no_unbiased_SOTA_claim=True, selection_using_test_results_only=True))
    dump(EXP/'CHECKPOINT_RESTORE_AND_LR_PROOF.json', dict(parent=read(EXP/'PARENT_IDENTITY.json'),
        preupdate=read(EXP/'PREUPDATE_EQUIVALENCE.json'), arms={a:read(arm_exp(a)/'RESUME_GATE.json') for a in completed}))
    dump(EXP/'FOUR_ARM_FULL_STREAM_PROOF.json', dict(arms={a:read(arm_exp(a)/'FULL_STREAM_AND_LR_PROOF.json') for a in completed}))
    dump(EXP/'FOUR_ARM_GRADIENT_MASK_DIAGNOSTICS.json', dict(arms={a:read(arm_exp(a)/'TRAINING_DIAGNOSTICS.json') for a in completed}))
    lines = ['# E2-Uniform late LR four-arm epoch4 experiment','',
        'All four multipliers were fixed before training. Every arm independently resumes the same E2@3651.',
        'Results are exploratory: public benchmarks were repeatedly observed; no trustworthy independent, deduplicated validation protocol was available.','',
        '| Model | BB | Text mask | Visual mask | Adapter | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for name,result in models.items():
        q = protocol.common.quality(result)
        mult = [1.]*4 if name=='E2-Uniform' else ARMS[name]
        lines.append('| '+name+' | '+' | '.join(str(v) for v in mult)+' | '+
            ' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} | {(q["Urban_I2T"]+q["Urban_T2I"])/2:.3f} |')
    for name in completed:
        result = models[name];d=protocol.common.compare(result,baseline)['quality_delta_pp']
        lines += ['',f'## {name}', '',f'Delta vs E2 (pp): `{json.dumps(d)}`',
            f'Checkpoint SHA256: `{result["checkpoint"]["sha256"]}`.',
            '| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |',
            '|---|---|---|---|---|']
        for ds,dirs in result['metrics'].items():
            delta=protocol.common.compare(result,baseline)['recall_delta_pp'][ds]
            values=[' / '.join(f'{100*dirs[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            values+=[' / '.join(f'{delta[dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append('| '+ds+' | '+' | '.join(values)+' |')
    lines+=['','Urban 0.1pp is one query; single-seed differences are not established statistical gains.',
        'Lower mask density/violation alone does not prove better evidence selection.',
        'No fifth arm, new seed, new multiplier, checkpoint fusion or epoch5 is authorized.']
    (EXP/'FOUR_ARM_4868_REPORT.md').write_text('\n'.join(lines)+'\n')


def baseline_snapshot():
    baseline=read(BASE_EXP/'step4868/RESULTS.json')
    blob=subprocess.check_output(['git','show',PARENT_COMMIT+':'+str((BASE_EXP/'step4868/RESULTS.json').relative_to(ROOT))],cwd=ROOT)
    assert blob==(BASE_EXP/'step4868/RESULTS.json').read_bytes()
    dump(EXP/'BASELINE_IMMUTABILITY.json',dict(results_sha256=hashlib.sha256(blob).hexdigest(),
        baseline_checkpoint=baseline['checkpoint'],baseline_bare=baseline['bare'],
        original_eval_files={str(p):sha(p) for p in (PARENT_RUN/'step4868/evaluations').rglob('*.json')}))


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0,1,2,3})
    assert not EXP.exists() and not RUN.exists(), 'No overwrite or duplicate run'
    assert protocol.common.git('branch','--show-current')==BRANCH
    assert protocol.common.git('merge-base',PARENT_COMMIT,'HEAD')==PARENT_COMMIT
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/experiment/hns-s12-uniform-full4868-v1'],cwd=ROOT,text=True).split()[0]==PARENT_COMMIT
    activate('B1-BB085')
    proof=identity(PARENT,START)
    assert sha(STEP0)==STEP0_SHA
    baseline=read(BASE_EXP/'step4868/RESULTS.json')
    baseline_blob=subprocess.check_output(['git','show',PARENT_COMMIT+':'+str((BASE_EXP/'step4868/RESULTS.json').relative_to(ROOT))],cwd=ROOT)
    assert baseline_blob==(BASE_EXP/'step4868/RESULTS.json').read_bytes()
    assert baseline['completed_steps']==STOP and baseline['evaluation_checkpoint_immutable']
    expected=dict(Score5=73.812186642,J_long3=78.193644403,J_long=87.055002253,
        Short4=67.240,Urban_I2T=93.9,Urban_T2I=92.9)
    assert all(abs(protocol.common.quality(baseline)[k]-v)<1e-5 for k,v in expected.items())
    assert sha(Path(baseline['checkpoint']['path']))==baseline['checkpoint']['sha256']
    assert sha(Path(baseline['bare']['path']))==baseline['bare']['sha256']
    evaluator=protocol.common.evaluator_proof()
    ready=read(protocol.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'config.json',config());dump(EXP/'PARENT_IDENTITY.json',proof)
    dump(EXP/'FOUR_ARM_PLAN.json',dict(order=list(ARMS),arms={a:dict(zip(GROUPS,m)) for a,m in ARMS.items()},
        common_checkpoint=str(PARENT),sha256=PARENT_SHA,parent_commit=PARENT_COMMIT,
        start_updates=START,stop_updates=STOP,updates_per_arm=1217,horizon=4868,
        production_changes=[],production_manifest=proof['sources'],evaluator_manifest=evaluator,
        batch_per_rank=256,world_size=4,seed=0,local_only=True,
        multipliers_applied_after_native_LR=True,independent_validation='UNVERIFIED',
        validation_limitations=['No frozen independent retrieval protocol for ShareGPT4V skipped1000',
            'No complete image-content deduplication or public-test-overlap certificate'],
        exploratory=True,automatic_fifth_arm=False,automatic_epoch5=False))
    baseline_snapshot()
    reference=protocol.resume_reference(START)
    for arm in ARMS:
        arm_exp(arm).mkdir(parents=True);arm_run(arm).mkdir()
        dump(EXP/arm/'config.json',config())
        dump(EXP/arm/'REFERENCES.json',dict(models={'E2_Uniform':{'4868':baseline}}))
        dump(arm_run(arm)/'parent-identity.json',proof)
        dump(arm_run(arm)/'resume-reference.json',reference)
        dump(arm_run(arm)/'launch-provenance.json',dict(source_sha256=proof['sources'],parent=str(PARENT),
            parent_sha256=PARENT_SHA,lr_multipliers=dict(zip(GROUPS,ARMS[arm])),stop=STOP))
    paths=protocol.local.path_proof()
    dump(RUN/'local-path-proof.json',paths)
    dump(EXP/'LOCAL_ONLY_PROOF.json',{k:v for k,v in paths.items() if k!='rows'})
    state('PREPARED',completed_arms=[])


def publish(message):
    subprocess.run(['git','add','--',str(EXP.relative_to(ROOT)), 'recovery/said_e2_late_lr_fourarm.py',
        'recovery/said_e2_late_lr_precheck.py','recovery/said_e2_late_lr_validation_split.py',
        'tests/test_said_e2_late_lr_fourarm.py'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    staged=protocol.common.git('diff','--cached','--name-only').splitlines()
    assert all(not p.endswith(('.pt','.log','.gz','.bin')) for p in staged)
    assert all((ROOT/p).stat().st_size<2_000_000 for p in staged)
    if staged:subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    head=protocol.common.git('rev-parse','HEAD')
    assert head==protocol.common.git('rev-parse','origin/'+BRANCH)==protocol.common.git('rev-parse','FETCH_HEAD')
    dump(RUN/'GITHUB_RECEIPT.json',dict(local_HEAD=head,remote_HEAD=head,matched=True,checked_utc=now()))


def run():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(EXP/'STATE.json')['status']=='PREPARED'
        assert read(EXP/'CPU_TESTS.json')['passed'] and read(EXP/'PREUPDATE_EQUIVALENCE.json')['passed']
        completed=[]
        try:
            for arm in ARMS:
                require_gpu_idle({0,1,2,3});activate(arm)
                assert sha(PARENT)==PARENT_SHA and code_manifest()==read(EXP/'PARENT_IDENTITY.json')['sources']
                phase(arm).mkdir(exist_ok=False)
                supervisor=protocol.local.Supervisor()
                state('TRAINING',active_arm=arm,completed_arms=completed)
                supervisor.execute('train',training_command(arm),training=True)
                acceptance=read(arm_run(arm)/'training/acceptance.json')
                assert acceptance['passed'] and all(r['completed_updates']==STOP and
                    r['updates_this_run']==1217 and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
                gate=protocol.resume_gate(STOP)
                checks=[read(arm_run(arm)/f'post-{s}-rank{r}.json') for s in range(START+1,START+6) for r in range(4)]
                assert all(c['passed'] for c in checks)
                dump(arm_exp(arm)/'RESUME_GATE.json',dict(gate,post_update_checks=checks,multipliers=dict(zip(GROUPS,ARMS[arm]))))
                _,proof=full_stream(arm)
                identity(arm_run(arm)/'training/step004868.pt',STOP)
                require_gpu_idle({0,1,2,3})
                state('GRADIENT_AUDIT',active_arm=arm,completed_arms=completed)
                supervisor.execute('gradient',protocol.common.torchrun('recovery.hns_macro_gradient',
                    '--checkpoint',arm_run(arm)/'training/step004868.pt','--expect-updates',STOP,
                    '--output',arm_exp(arm)/'GRADIENT_AUDIT.json'))
                result=protocol.evaluate(supervisor,STOP)
                report_arm(arm,supervisor,result,proof)
                completed.append(arm);combined(completed)
                require_gpu_idle({0,1,2,3})
                state('COMPLETED_GPU_IDLE' if len(completed)==4 else 'ARM_COMPLETED',active_arm=arm,completed_arms=completed)
                publish('Report E2 late LR '+arm+' epoch4 results')
            assert sha(PARENT)==PARENT_SHA
            dump(RUN/'completed.json',dict(completed_arms=completed,stop=4868,GPU_idle=True,utc=now()))
        except BaseException as exc:
            state('HARD_STOP_WITH_EVIDENCE',active_arm=arm,completed_arms=completed,error=repr(exc),automatic_retry=False)
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args,rest=parser.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*rest];worker(args.arm)
    elif args.prepare:prepare()
    elif args.run:run()
    else:parser.error('choose prepare, run or worker')


if __name__=='__main__':main()
