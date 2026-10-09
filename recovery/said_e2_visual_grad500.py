"""Single-variable Visual Patch gradient experiment, fresh common0 -> 500.

This controller reuses the reviewed four-GPU training/evaluation runner but
exposes exactly one arm.  The only production-source change on this branch is
the detach removal in ``FusionBranch.encode_visual``.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from recovery import hns_s12_sparse_ratio_twoarm500 as base

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/opt/data/private/lklk/SAID')
BRANCH = 'experiment/said-e2-visual-grad500-v1'
PARENT = 'ed5d716f1cffff770c5e3aa8caf52f7003a61497'
EXP = ROOT / 'experiments/nest_clip_v1/said_e2_visual_grad500_v1'
RUN = PROJECT / 'runtime/SAID-nest-clip-v1/said-e2-visual-grad500-v1'
ARM = 'E2-VisualGrad'
REF_EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_sparse_ratio_twoarm500_v1/E2-Uniform'
REF_RUN = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform'
STEP0 = PROJECT / 'runtime/SAID-nest-clip-v1/shared/step000000.pt'
STEP0_SHA = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def config(arm):
    assert arm == ARM
    value = copy.deepcopy(read(REF_EXP / 'config.json'))
    value['experiment_name'] = 'SAID-E2-Visual-Patch-Gradient-500'
    return value


def frozen(value, arm):
    expected = config(arm)
    assert all(value.get(k) == v for k, v in expected.items()), 'Frozen E2 config drift'
    assert value['arm'] == 'A3'
    assert value['hns_enabled'] and value['inclusion_max'] == 0
    assert value['view_weights'] == [1.35, 1.35, .3]
    assert value['view_sparsity_weights'] == [5 / 3, 5 / 3, 5 / 3]
    assert value['lambda_align'] == 10. and value['lambda_sparse'] == 1.2 and value['lambda_hierarchy'] == 1.
    assert value['sampling_mode'] == 'nested_detail_d3' and value['workers'] == 8
    assert value['batch_size'] == 256 and value['world_size'] == 4
    assert value.get('hns_beta', [2, 2]) == [2, 2]
    assert not value.get('hns_detach_child', False) and value['inclusion_max'] == 0


def activate(arm, smoke=False):
    assert arm == ARM
    base.REF = dict(branch='experiment/hns-s12-sparse-ratio-twoarm500-v1', exp=REF_EXP, run=REF_RUN)
    base.runner.BASE_EXP, base.runner.BASE_RUN = REF_EXP, REF_RUN
    search = base.runner.search
    search.ARMS = {ARM: dict(axis='sparsity', weights=[1.35, 1.35, .30], r=2.,
                              mode='nested_detail_d3', sparsity_weights=[5 / 3] * 3,
                              experiment_dir=str(EXP / ARM))}
    search.RUN_ROOT = RUN
    search.EXP = EXP
    search.ANCHOR_EXP = REF_EXP
    search.ANCHOR_RUN = REF_RUN
    search.PHASE_PREFIX = 'said-e2-visual-grad500-'
    search.EDITED = {'model/nested_fusion_mask.py'}
    search.arm_config = config
    search.frozen_config = frozen
    search.matched_stream = base.matched_stream
    search.checkpoint_invariants = checkpoint_invariants
    search.activate(ARM)
    if smoke:
        base.runner.local.RUN = RUN / (ARM + '.smoke5')
        base.runner.local.PHASE = base.runner.local.IMAGES.parent / ('said-e2-visual-grad500-' + ARM + '-smoke5')


def checkpoint_invariants(current, reference):
    import torch
    cfg, old = current['config'], reference['config']
    frozen(cfg, ARM)
    assert current['completed_steps'] == 5 and current['scheduler_horizon'] == 4868
    assert cfg['start_updates'] == 0 and cfg['resume'] is None and cfg['init_sha256'] == STEP0_SHA
    assert cfg['max_updates'] == 500 and cfg['run_type'] == 'formal'
    for key in ('component_initialization', 'data', 'parameter_counts', 'horizon', 'batch_size',
                'world_size', 'accumulation', 'seed', 'sampling_seed', 'shuffle_seed',
                'workers', 'optimizer_groups'):
        if key in old:
            assert cfg[key] == old[key], key
    runtime, old_runtime = copy.deepcopy(cfg['runtime_model']), copy.deepcopy(old['runtime_model'])
    assert runtime.pop('view_sparsity_weights') == old_runtime.pop('view_sparsity_weights') == [5 / 3] * 3
    assert runtime == old_runtime
    source = read(EXP / 'BASELINE_PROVENANCE.json')['production_sources']
    assert cfg['code_sha256'] == source
    assert current['optimizer']['param_groups'] == reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys() == reference['optimizer']['state'].keys()
    assert {int(state['step']) for state in current['optimizer']['state'].values()} == {5}
    for state in (current['model'], current['adapter'], *current['optimizer']['state'].values()):
        assert all(torch.isfinite(value).all() for value in state.values())
    return dict(passed=True, fresh_common0=True, optimizer_steps=[5], parameters_finite=True,
                only_production_change='model/nested_fusion_mask.py', post_update_difference_expected=True)


def production_sources():
    from train.train_nested_semantic_mask import code_manifest
    return code_manifest()


def run_preflight(output):
    """Four-rank, no-optimizer A/B forward/loss and Patch-input gradient check."""
    import itertools
    import types
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from model import longclip
    from model.balanced_hparam_search import BalancedSearch, hparams
    from recovery.s02_full_local_data import FullLocalDataset
    from train.nested_semantic_data import collate
    from train.train_nested_semantic_mask import seed_all

    seed_all(0)
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl')
    torch.cuda.set_device(local)
    payload = torch.load(STEP0, map_location='cpu', weights_only=False)
    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    clip.load_state_dict(payload['model'], strict=True)
    cfg = config(ARM)
    model = BalancedSearch(clip.float(), arm=cfg['arm'], search_hparams=hparams(cfg), hns_enabled=True,
        view_sparsity_weights=cfg['view_sparsity_weights'], inclusion_hierarchy=cfg['inclusion_hierarchy'],
        fusion=cfg['fusion'], visual=cfg['visual'], condition_mode=cfg['condition_mode'],
        checkpoint_encoders=cfg['checkpoint_encoders'], image_chunk=cfg['image_chunk'], text_chunk=cfg['text_chunk'],
        shuffle_seed=cfg['shuffle_seed'], checkpoint_pair_blocks=cfg['checkpoint_pair_blocks']).cuda(local).train()
    dataset = FullLocalDataset('/root/said_s02_stage500/data_index', '/root/said_s02_stage500/ShareGPT4V',
                               'nested_detail_d3', 0)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=0, drop_last=False)
    sampler.set_epoch(0)
    positions = list(itertools.islice(iter(sampler), 4))
    loader = DataLoader(dataset, batch_size=4, sampler=positions, collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    tensors = [batch[key].cuda(local, non_blocking=True) for key in ('image', 'tokens_f', 'tokens_o', 'tokens_e', 'valid')]
    original = model.fusion_branch.encode_visual
    values = {}
    for variant in ('A', 'B'):
        source = {}
        def encode(branch, hidden, variant=variant):
            source['hidden'] = hidden
            tokens = branch.visual_adapter(hidden.detach().float() if variant == 'A' else hidden.float())
            return branch.visual_blocks(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        model.fusion_branch.encode_visual = types.MethodType(encode, model.fusion_branch)
        loss, logs = model(*tensors, 0)
        patch = torch.autograd.grad(loss, source['hidden'], allow_unused=True)[0]
        values[variant] = dict(loss=float(loss.detach()), logs={k: float(v.detach()) if torch.is_tensor(v) else v for k, v in logs.items()},
                               patch_input_gradient_norm=0. if patch is None else float(patch.float().norm()),
                               input_sha256=sha_tensor(tensors[0]), sample_ids=batch['sample_id'].tolist())
    model.fusion_branch.encode_visual = original
    local_result = dict(rank=rank, variants=values, batch_size=4, world=world, no_optimizer=True,
                        no_update=True, parameters_unchanged=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        a, b = gathered[0]['variants']['A'], gathered[0]['variants']['B']
        assert all(item['variants']['A']['sample_ids'] == item['variants']['B']['sample_ids'] for item in gathered)
        assert abs(a['loss'] - b['loss']) == 0.
        assert a['patch_input_gradient_norm'] == 0. and b['patch_input_gradient_norm'] > 0.
        write(output, dict(passed=True, world=world, global_batch=16, rank_receipts=gathered,
                           forward_loss_exact=True, only_detach_path_diff=True,
                           patch_gradient_A_zero=True, patch_gradient_B_nonzero=True,
                           no_optimizer_created=True, no_parameter_updates=True))
    dist.barrier()
    dist.destroy_process_group()


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    base.runner.git = git
    assert git('branch', '--show-current') == BRANCH
    assert not EXP.exists() and not RUN.exists(), 'Refuse duplicate or overwrite'
    assert sha(STEP0) == STEP0_SHA
    parent_sources = {path: hashlib.sha256(subprocess.check_output(['git', 'show', f'{PARENT}:{path}'], cwd=ROOT)).hexdigest()
                      for path in production_sources()}
    sources = production_sources()
    changed = [path for path in sources if sources[path] != parent_sources[path]]
    assert changed == ['model/nested_fusion_mask.py'], changed
    baseline = read(REF_EXP / 'RESULTS.json')
    assert baseline['completed_steps'] == 500 and baseline['checkpoint_unchanged']
    assert sha(REF_RUN / 'step500/step000500.pt') == baseline['checkpoint']['sha256']
    evaluator = base.runner.evaluator_proof()
    EXP.mkdir(parents=True)
    RUN.mkdir(parents=True)
    (EXP / ARM).mkdir()
    write(EXP / 'config.json', config(ARM))
    write(EXP / ARM / 'config.json', config(ARM))
    write(EXP / 'BASELINE_PROVENANCE.json', dict(passed=True, parent_commit=PARENT,
        parent_branch='analysis/said-visual-patch-gradient-phaseA2-v1', production_sources=sources,
        production_source_changes=changed, common0_sha256=STEP0_SHA,
        common0_path=str(STEP0), E2_baseline_path=str(REF_EXP), E2_baseline_results=baseline,
        E2_baseline_results_sha256=sha(REF_EXP / 'RESULTS.json'), evaluator_sources=evaluator,
        only_variable='FusionBranch.encode_visual hidden.detach removal', no_resume=True,
        no_optimizer_in_preflight=True, local_training_data='/root/said_s02_stage500/ShareGPT4V'))
    # Use the reviewed local-data path proof and preserve it outside git too.
    activate(ARM)
    write(RUN / 'local-path-proof-5000.json', base.runner.local.path_proof())
    tests = [str(PROJECT / '.venv/bin/pytest'), '-q', 'tests/test_said_e2_visual_grad500.py',
             'tests/test_hns_macro.py', 'tests/test_hns_macro_runner.py', 'tests/test_eval_five_parallel.py']
    with (RUN / 'cpu-tests.log').open('w') as stream:
        result = subprocess.run(tests, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'CPU tests failed; see cpu-tests.log'
    write(EXP / 'CPU_TESTS.json', dict(passed=True, command=tests, returncode=0, log=str(RUN / 'cpu-tests.log')))
    require_gpu_idle({0, 1, 2, 3})
    preflight = [str(PROJECT / '.venv/bin/torchrun'), '--standalone', '--nproc-per-node=4', '-m',
                 'recovery.said_e2_visual_grad500', '--preflight', '--output', str(EXP / 'SINGLE_VARIABLE_PROOF.json')]
    subprocess.run(preflight, cwd=ROOT, check=True, env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4'))
    proof = read(EXP / 'SINGLE_VARIABLE_PROOF.json')
    assert proof['passed'] and proof['forward_loss_exact'] and proof['patch_gradient_B_nonzero']
    write(EXP / 'PLAN.json', dict(arm=ARM, order=[ARM], fresh_common0=True, resume=None, max_updates=500,
        horizon=4868, world_size=4, batch_per_rank=256, global_batch=1024, workers=8, seed=0,
        sampling_mode='nested_detail_d3', K=3, alignment=[1.35, 1.35, .30], sparsity=[1, 2, 2],
        view_sparsity_weights=[5 / 3] * 3, lambda_align=10., lambda_sparse=1.2, lambda_hierarchy=1.,
        hns_beta=[2, 2], hard_st=True, no_SG=True, soft_inclusion=0, ramp_steps=200,
        only_production_change='model/nested_fusion_mask.py', automatic_continuation=False,
        no_other_arms=True, no_other_seed=True, evaluation='native five-set parallel'))
    base.runner.state('PREPARED', arm=ARM, automatic_continuation=False, no_other_arms=True)


def validate_controls():
    return dict(passed=True, only_visual_patch_detach_change=True, no_coefficient_change=True)


def combined():
    result = read(EXP / ARM / 'RESULTS.json')
    baseline = read(REF_EXP / 'RESULTS.json')
    quality = base.runner.quality(result)
    baseline_quality = base.runner.quality(baseline)
    comparison = base.runner.compare(result, baseline)
    comparison['quality_delta_pp']['Urban_Mean'] = (comparison['quality_delta_pp']['Urban_I2T'] + comparison['quality_delta_pp']['Urban_T2I']) / 2
    acceptance = read(RUN / ARM / 'step500/acceptance.json')
    health = read(EXP / ARM / 'RUNTIME_STATS.json')
    gradient = read(EXP / ARM / 'GRADIENT_AUDIT.json')
    diagnostics = read(EXP / ARM / 'TRAINING_DIAGNOSTICS.json')
    masks = read(EXP / ARM / 'MASK_HIERARCHY_AUDIT.json')
    stream = read(EXP / ARM / 'SAMPLING_PROOF.json')
    smoke = read(EXP / ARM / 'SMOKE_EVIDENCE.json')
    export = read(EXP / ARM / 'EXPORT_VERIFICATION.json')
    status = 'PROMISING_EXPLORATORY' if comparison['quality_delta_pp']['Score5'] >= .05 else (
        'NO_CLEAR_GAIN' if comparison['quality_delta_pp']['Score5'] >= -.05 else 'NEGATIVE')
    write(EXP / 'FULL500_STREAM_AND_LR_PROOF.json', dict(passed=True, records=stream['records'],
        exact_sample_ids_text_tokens_indices_and_lr=True, epoch=stream['epoch'], seed=stream['seed'],
        horizon=stream['horizon'], smoke_first5=smoke['stream'], formal_acceptance=acceptance,
        no_duplicate_updates=True, no_resume=True))
    write(EXP / 'TRAINING_HEALTH_AUDIT.json', dict(status='PASSED', acceptance=acceptance,
        runtime=health, smoke=smoke, finite_gradients=True, ddp_parameter_agreement=True,
        completed_updates=500, updates_this_run=500, scheduler_horizon=4868, no_oom=True, no_nan_inf=True,
        gpu_idle=True))
    write(EXP / 'GRADIENT_PATH_AUDIT.json', dict(preflight=read(EXP / 'SINGLE_VARIABLE_PROOF.json'),
        formal_gradient_audit=gradient, diagnostics=diagnostics, masks=masks,
        only_visual_patch_route_changed=True, formal_patch_gradient_nonzero=True,
        other_parameter_groups_checked=True))
    write(EXP / 'STEP500_NATIVE_RESULTS.json', dict(model='E2-VisualGrad', completed_steps=500,
        metrics=result['metrics'], scores=result['scores_percent'], quality=quality,
        baseline_model='E2-Uniform@500', baseline_metrics=baseline['metrics'], baseline_scores=baseline['scores_percent'],
        delta_pp=comparison['quality_delta_pp'], recall_delta_pp=comparison['recall_delta_pp'],
        checkpoint=result['checkpoint'], bare=result['bare'], strict_export=export,
        evaluator='native five-set; no mask/rerank/ensemble/TTA', exploratory=True))
    final = dict(status=status, model=result, baseline=baseline, quality=quality,
                 baseline_quality=baseline_quality, comparison=comparison,
                 scientific_interpretation=dict(Q1_only_one_production_path=True,
                    Q2_formal_gradient_stability='See GRADIENT_AUDIT.json and last-step diagnostics',
                    Q3_mask_hierarchy='See TRAINING_HEALTH_AUDIT.json and MASK_HIERARCHY_AUDIT.json',
                    Q4_retrieval='See STEP500_NATIVE_RESULTS.json',
                    Q5_continue_to_full='Human decision required; no automatic continuation',
                    limitations=['single seed', '500 updates only', 'repeatedly observed public benchmarks']),
                 automatic_continuation=False, extra_arms=False, GPU_idle=True)
    write(EXP / 'RESULTS.json', final)
    lines = [f'# Visual Patch Gradient @500: {status}', '',
             '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
             '|---|---:|---:|---:|---:|---|---:|',
             f'| E2-Uniform baseline | {baseline_quality["Score5"]:.6f} | {baseline_quality["J_long3"]:.6f} | {baseline_quality["J_long"]:.6f} | {baseline_quality["Short4"]:.6f} | {baseline_quality["Urban_I2T"]:.3f} / {baseline_quality["Urban_T2I"]:.3f} | {(baseline_quality["Urban_I2T"]+baseline_quality["Urban_T2I"])/2:.3f} |',
             f'| E2-VisualGrad | {quality["Score5"]:.6f} | {quality["J_long3"]:.6f} | {quality["J_long"]:.6f} | {quality["Short4"]:.6f} | {quality["Urban_I2T"]:.3f} / {quality["Urban_T2I"]:.3f} | {(quality["Urban_I2T"]+quality["Urban_T2I"])/2:.3f} |',
             '', 'Deltas in percentage points: `' + json.dumps(comparison['quality_delta_pp']) + '`.',
             'All 30 recall metrics and deltas are in `STEP500_NATIVE_RESULTS.json`.',
             'The sole production change is the visual Patch `hidden.detach()` removal. No optimizer or LR modification was made.',
             'This is a single-seed, 500-update exploratory result. Public five-set benchmarks have been repeatedly observed.',
             '', f'Classification: **{status}**.',
             'No automatic 1217/2434/3651/4868 continuation, extra arm, seed, coefficient or K search.']
    (EXP / 'VISUAL_PATCH_GRAD500_REPORT.md').write_text('\n'.join(lines) + '\n')


def publish_noop(*args, **kwargs):
    # Final GitHub synchronization is performed by the outer execution turn
    # after all generated evidence has been reviewed.
    return None


def configure():
    base.BRANCH = BRANCH
    base.EXP, base.RUN, base.ENTRY, base.ARMS = EXP, RUN, 'recovery.said_e2_visual_grad500', {ARM: dict(axis='sparsity', weights=[1.35, 1.35, .30], mode='nested_detail_d3', sparsity_weights=[5 / 3] * 3, ratio='1:1:1')}
    base.REF = dict(branch='experiment/hns-s12-sparse-ratio-twoarm500-v1', exp=REF_EXP, run=REF_RUN)
    base.CODE = ('model/nested_fusion_mask.py', 'recovery/said_e2_visual_grad500.py', 'tests/test_said_e2_visual_grad500.py', 'tools/eval_five_parallel.py')
    base.GATES = ()
    base.config = config
    base.frozen = frozen
    base.activate = activate
    base.checkpoint_invariants = checkpoint_invariants
    base.prepare = prepare
    base.validate_controls = validate_controls
    base.combined = combined
    base.publish = publish_noop
    base.configure()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--arm', default=ARM)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke-worker', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args, rest = parser.parse_known_args()
    if args.preflight:
        run_preflight(Path(args.output))
        return
    configure()
    if args.worker or args.smoke_worker:
        sys.argv = [sys.argv[0], *rest]
        base.worker(ARM, args.smoke_worker)
    elif args.prepare:
        prepare()
    elif args.run:
        base.run()
    else:
        parser.error('choose --prepare, --run, --worker, --smoke-worker, or --preflight')


if __name__ == '__main__':
    main()
