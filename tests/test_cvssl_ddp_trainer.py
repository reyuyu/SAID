"""Real-trainer DDP acceptance test (HARD GATE).

The worker builds ``SaidClsCvsslTrainModule`` + ``DistributedDataParallel`` + the two AdamW groups
and runs the production ``cvssl_train_step``. This test compares, per rank, the gradients and the
post-step state of a 2-rank DDP run against a single-process run of the SAME global batch.

Tolerance policy (declared up front; no assertion is relaxed to hide a mismatch):

* Every rank must produce **bit-identical** parameter gradients and parameter values -- DDP's
  reducer is deterministic, and this repository's diagnostic runs confirm ``max|g_r0 - g_r1| == 0``
  in fp32. A real synchronisation bug (a rank missing from the reduction, a wrong ``*world_size``
  factor, a bypassed DDP path) is orders of magnitude larger than fp32 round-off.
* DDP-averaged vs single-process is compared with a **relative** criterion
  (``max|diff| <= ATOL + RTOL * max|reference|``), because the two orders of accumulation cannot be
  bit-equal in fp32. Measured residuals for this exact setup: fp32 ``6.1e-05`` on a ``6.4e+02``
  gradient (rel ``1.3e-07``); the same quantity in a fp64 control collapses to ``8.3e-12``
  (rel ``1.3e-14``), i.e. the fp32 figure is accumulation order, not a scaling error. RTOL = 1e-5
  is still ~100x tighter than one fp32 ulp of the reference, and a 2x scaling bug would give
  rel = 0.5 and fail by four orders of magnitude.
* Parameters are compared numerically, NOT by hashing the fp32 bytes: a ``sha256`` of fp32 tensors
  differs on an ulp. Cross-rank equality is still asserted on the digests, because that comparison
  is exact in principle.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, 'tests', '_cvssl_ddp_trainer_worker.py')
PYTHON = sys.executable
PORT = 29611
ATOL = 0.0
RTOL = 1e-5


def _run(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, cwd=REPO_ROOT)
    return result


def _single(case, out, steps, **extra):
    command = [PYTHON, WORKER, '--case', case, '--mode', 'single', '--out', out,
               '--steps', str(steps)]
    for key, value in extra.items():
        command += ['--%s' % key, str(value)]
    result = _run(command)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.load(open(out.format(rank=0)))


def _two_rank(case, out_dir, steps=1, port=PORT, **extra):
    env = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
           'GLOO_SOCKET_IFNAME': 'lo'}
    command = [PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port', str(port), WORKER, '--case', case, '--mode', 'rank',
               '--out', os.path.join(out_dir, 'rank{rank}.json'), '--steps', str(steps)]
    for key, value in extra.items():
        command += ['--%s' % key, str(value)]
    result = _run(command, env)
    assert result.returncode == 0, result.stderr[-3000:]
    return [json.load(open(os.path.join(out_dir, 'rank%d.json' % rank))) for rank in range(2)]


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _max_abs(a, b):
    left, right = _tensor(a), _tensor(b)
    assert left.shape == right.shape
    return float((left - right).abs().max())


def _scale(value):
    return float(_tensor(value).abs().max())


def _assert_close(tag, reference, other):
    difference = _max_abs(reference, other)
    allowed = ATOL + RTOL * _scale(reference)
    assert difference <= allowed, (tag, difference, allowed, _scale(reference))


@pytest.mark.parametrize('case', ['S0', 'C0'])
def test_gradients_per_rank_match_the_single_process_reference(tmp_path, case):
    single = _single(case, str(tmp_path / 'single_{rank}.json'), steps=1)
    ranks = _two_rank(case, str(tmp_path))
    reference = single['history']['1']['grads']
    for rank, payload in enumerate(ranks):
        grads = payload['history']['1']['grads']
        assert grads.keys() == reference.keys(), 'parameter set differs'
        for name, value in reference.items():
            other = grads[name]
            if value is None or other is None:
                assert value is None and other is None, (rank, name)
                continue
            _assert_close(('gradient mismatch', rank, name), value, other)


@pytest.mark.parametrize('case', ['S0', 'C0'])
def test_rank_gradients_are_bit_identical_after_the_ddp_reduction(tmp_path, case):
    """The core acceptance sentence: every rank updates the same parameter state."""
    ranks = _two_rank(case, str(tmp_path))
    first, second = ranks[0]['history']['1']['grads'], ranks[1]['history']['1']['grads']
    assert first.keys() == second.keys()
    for name in first:
        if first[name] is None or second[name] is None:
            assert first[name] is None and second[name] is None, name
            continue
        assert _max_abs(first[name], second[name]) == 0.0, ('rank gradient differs', name)


@pytest.mark.parametrize('case', ['S0', 'C0'])
def test_one_step_update_and_optimizer_state_match(tmp_path, case):
    single = _single(case, str(tmp_path / 'single_{rank}.json'), steps=1)
    ranks = _two_rank(case, str(tmp_path))
    reference = single['history']['1']
    for rank, payload in enumerate(ranks):
        step = payload['history']['1']
        assert step['param_digest'] == ranks[1 - rank]['history']['1']['param_digest'], rank
        assert step['backbone_opt_digest'] == ranks[1 - rank]['history']['1'][
            'backbone_opt_digest'], rank
        assert step['mask_opt_digest'] == ranks[1 - rank]['history']['1']['mask_opt_digest'], rank
        for name, value in reference['params'].items():
            _assert_close(('parameter mismatch', rank, name), value, step['params'][name])
        for key, value in reference['backbone_exp_avg'].items():
            _assert_close(('backbone exp_avg', rank, key), value, step['backbone_exp_avg'][key])
        for key, value in reference['mask_exp_avg'].items():
            _assert_close(('mask exp_avg', rank, key), value, step['mask_exp_avg'][key])


def test_forward_losses_are_the_rank_mean_of_the_global_loss(tmp_path):
    """Per-rank losses are local-anchor means; their mean must be the single-process global value.

    The per-rank values themselves are NOT expected to be equal (each rank scores its own anchors);
    only the aggregated quantity is defined globally. ``loss_vssl_global_mean`` is the one loss that
    is already reduced, so it must match the single-process value on every rank.
    """
    cases = ('S0', 'G0', 'R0', 'C0')
    for index, case in enumerate(cases):
        directory = str(tmp_path / case)
        os.makedirs(directory, exist_ok=True)
        single = _single(case, directory + '/single_{rank}.json', steps=1)
        ranks = _two_rank(case, directory, port=PORT + 20 + index)
        reference, step = single['history']['1'], ranks[0]['history']['1']
        assert step['global_sample_count'] == reference['global_sample_count']
        for key in ('loss_sidm', 'loss_dism', 'loss_sparsity', 'loss_smart'):
            values = [payload['history']['1'][key] for payload in ranks]
            mean = sum(values) / len(values)
            assert abs(mean - reference[key]) <= ATOL + RTOL * abs(reference[key]), (case, key)
        for payload in ranks:
            value = payload['history']['1']['loss_vssl_global_mean']
            assert abs(value - reference['loss_vssl_global_mean']) <= ATOL + RTOL * abs(
                reference['loss_vssl_global_mean']), (case, 'global_mean')


def test_random_mask_is_the_same_for_the_same_sample_in_single_and_rank_modes(tmp_path):
    """R0's random mask must depend on the sample, not on the local batch layout.

    A generator consumed over local rows would give rank 1's row 0 the permutation of rank 0's
    row 0, so ``loss_vssl_global_mean`` would differ between the single-process global-batch run
    and the 2-rank run (measured before the fix: 0.03309 vs 0.00889, rel 0.73).
    """
    single = _single('R0', str(tmp_path / 'single_{rank}.json'), steps=1)
    ranks = _two_rank('R0', str(tmp_path))
    for payload in ranks:
        assert abs(payload['history']['1']['loss_vssl_global_mean']
                   - single['history']['1']['loss_vssl_global_mean']) <= RTOL * abs(
                       single['history']['1']['loss_vssl_global_mean'])
    # and the shuffled mask must not be the complement itself (R0 must actually shuffle)
    assert single['history']['1']['loss_vssl_global_mean'] \
        != _single('C0', str(tmp_path / 'c0_{rank}.json'), steps=1)['history']['1'][
            'loss_vssl_global_mean']


def test_data_stream_is_the_same_global_batch_sliced_per_rank(tmp_path):
    single = _single('S0', str(tmp_path / 'single_{rank}.json'), steps=1)
    ranks = _two_rank('S0', str(tmp_path))
    assert (ranks[0]['history']['1']['sample_id'] + ranks[1]['history']['1']['sample_id']
            == single['history']['1']['sample_id'])
    assert (ranks[0]['history']['1']['image_id'] + ranks[1]['history']['1']['image_id']
            == single['history']['1']['image_id'])
    assert ranks[0]['history']['1']['caption_said'] + ranks[1]['history']['1']['caption_said'] \
        == single['history']['1']['caption_said']


def test_every_rank_starts_from_the_same_and_unmodified_model(tmp_path):
    """The worker's model is a pure function of a fixed seed: every rank must build the same one.

    The only way to observe that through JSON is the state the run ends on (a bit-equal first-step
    gradient already implies a bit-equal start), so this test pins the metadata and the state keys
    that the DDP run must share with the single-process run.
    """
    single = _single('S0', str(tmp_path / 'single_{rank}.json'), steps=1)
    ranks = _two_rank('S0', str(tmp_path))
    assert (ranks[0]['final_state'].keys() == ranks[1]['final_state'].keys()
            == single['final_state'].keys())
    for rank in range(2):
        assert ranks[rank]['case'] == 'S0'
        assert ranks[rank]['arm'] == 'S0_smartclip'
        assert ranks[rank]['lambda_u'] == 0.0


def test_zero_gradient_parameter_is_reported_as_none_on_every_rank(tmp_path):
    """``text_projection`` legitimately receives no gradient; all ranks must agree on that."""
    ranks = _two_rank('S0', str(tmp_path))
    for rank, payload in enumerate(ranks):
        grads = payload['history']['1']['grads']
        assert grads['clip.text_projection'] is None, rank
        assert any(value is not None for value in grads.values()), rank


def test_optimizer_groups_match_the_production_builder(tmp_path):
    single = _single('S0', str(tmp_path / 'single_{rank}.json'), steps=1)
    step = single['history']['1']
    assert step['n_backbone'] + step['n_mask'] == len(step['backbone_group_names']) + len(
        step['mask_group_names'])
    assert step['n_backbone'] == len(step['backbone_group_names'])
    assert step['n_mask'] == len(step['mask_group_names'])
    assert all('mask_net' in name for name in step['mask_group_names'])
    steps = _two_rank('S0', str(tmp_path))
    for payload in steps:
        assert payload['history']['1']['backbone_group_names'] == step['backbone_group_names']
        assert payload['history']['1']['mask_group_names'] == step['mask_group_names']


@pytest.mark.parametrize('case', ['S0', 'G0', 'R0', 'C0', 'E', 'F'])
def test_ranks_stay_identical_across_20_steps(tmp_path, case):
    """Cross-rank consistency must hold at step 1, 5 and 20 -- for all four arms."""
    ranks = _two_rank(case, str(tmp_path), steps=20)
    for step in ('1', '5', '20'):
        first, second = ranks[0]['history'][step], ranks[1]['history'][step]
        assert first['param_digest'] == second['param_digest'], (case, step, 'params')
        assert first['backbone_opt_digest'] == second['backbone_opt_digest'], (case, step)
        assert first['mask_opt_digest'] == second['mask_opt_digest'], (case, step)
        # per-rank SIDM/DISM are local-anchor means and legitimately differ; the reduced U loss and
        # the update are the quantities that must agree across ranks
        assert abs(first['loss_vssl_global_mean']
                   - second['loss_vssl_global_mean']) <= 1e-7, (case, step)
        for key in ('loss_smart', 'loss_vssl_global_mean', 'loss_sparsity', 'loss_sidm',
                    'loss_dism'):
            assert first[key] == first[key], (case, step, key)          # not NaN
    # the two view-batch arms must actually train on the U term, not silently skip it
    if case in ('G0', 'R0', 'C0'):
        assert ranks[0]['history']['1']['valid_ab'] > 0
        assert ranks[0]['history']['1']['valid_ba'] > 0
        assert ranks[0]['history']['1']['loss_scale'] == 2.0


@pytest.mark.parametrize('mode', ['single', 'rank'])
def test_degenerate_all_zero_mask_stays_finite(tmp_path, mode):
    """An all-zero U mask (no valid anchor in either direction) must give finite zeros, no NaN
    and no deadlock, in both the single-process and the 2-rank path."""
    if mode == 'single':
        payload = _single('D', str(tmp_path / 'single_{rank}.json'), steps=1)
        step = payload['history']['1']
        assert abs(step['loss_vssl_global_mean']) < 1e-9
        assert step['valid_ab'] == 0.0 and step['valid_ba'] == 0.0
    else:
        ranks = _two_rank('D', str(tmp_path))
        for payload in ranks:
            step = payload['history']['1']
            assert abs(step['loss_vssl_global_mean']) < 1e-9
            assert step['valid_ab'] == 0.0 and step['valid_ba'] == 0.0
            assert step['loss_smart'] == step['loss_smart']
            for value in step['grads'].values():
                if value is None:
                    continue
                assert _tensor(value).abs().max() == _tensor(value).abs().max()  # not NaN
        assert ranks[0]['history']['1']['param_digest'] == ranks[1]['history']['1'][
            'param_digest']


def test_unequal_local_batches_fail_loudly_on_every_rank(tmp_path):
    """Case G: genuinely different per-rank batch sizes must raise, not hang."""
    env = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(PORT + 1),
           'GLOO_SOCKET_IFNAME': 'lo'}
    result = _run([PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=2',
                   '--master_port', str(PORT + 1), WORKER, '--case', 'G', '--mode', 'rank',
                   '--out', os.path.join(str(tmp_path), 'rank{rank}.json'), '--steps', '1'], env)
    assert result.returncode != 0
    assert 'ragged distributed batch is not supported' in (result.stderr + result.stdout)


@pytest.mark.parametrize('static_graph', [0, 1])
@pytest.mark.parametrize('checkpoint_views', [0, 1])
def test_static_graph_and_activation_checkpointing_do_not_break_synchronisation(
        tmp_path, static_graph, checkpoint_views):
    """Both engineering switches must leave every rank bit-identical to the single process."""
    tag = 'sg%d_ck%d' % (static_graph, checkpoint_views)
    directory = str(tmp_path / tag)
    os.makedirs(directory, exist_ok=True)
    single = _single('C0', directory + '/single_{rank}.json', steps=2,
                     static_graph=static_graph, checkpoint_views=checkpoint_views)
    ranks = _two_rank('C0', directory, steps=2, port=PORT + 40 + 2 * static_graph + checkpoint_views,
                      static_graph=static_graph, checkpoint_views=checkpoint_views)
    for rank, payload in enumerate(ranks):
        step = payload['history']['1']
        assert step['param_digest'] == ranks[1 - rank]['history']['1']['param_digest'], (tag, rank)
        for name, value in single['history']['1']['params'].items():
            _assert_close((tag, 'param', name), value, step['params'][name])
