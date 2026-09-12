"""S0-TriMask-HS v0.2 acceptance tests (task section 10 A-G).

A gate type / B gradients / C three-path reuse and responsibilities / D forward+backward against an
independent per-pair reference / E boundaries / F real two-rank DDP with the text sparsity term /
G v0.1 regression, mode mismatch, export keys.

The stub models and helpers are imported from ``tests/test_said_trimask.py`` so the two suites
exercise the same objects; no stub is duplicated and the v0.1 suite is not modified.
"""
import argparse
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.said_trimask import (ARM, ARM_HS, HARD_GATE, LAMBDA_SPARSE_T_HS,  # noqa: E402
                                OBJECTIVE, OBJECTIVE_HS, SOFT_GATE, TEXT_GATE_EPS,
                                TextMaskHead, TriMaskTrainModule, trimask_terms)
from train_said_trimask import (build_trimask_optimizers,  # noqa: E402
                                check_checkpoint_compatibility, gradients_are_finite,
                                trimask_train_step)

from test_said_trimask import (BATCH, DIM, TEXT_WIDTH, StubClip, _clip, _dist,  # noqa: E402
                               _grad_map, _group_max_grad, _images, _tokens)

EPS = 1e-6
EXPORT_TOOL = os.path.join(REPO_ROOT, 'tools', 'diag', 'export_trimask_student.py')
TRAINER = os.path.join(REPO_ROOT, 'train', 'train_said_trimask.py')


def _hard_module(clip, lambda_1=10.0, lambda_2=1.0, lambda_3=1.0, lambda_sparse_i=2.0,
                 lambda_sparse_t=LAMBDA_SPARSE_T_HS, gate_std=1.0, seed=0):
    """A hard-gate module with a deliberately non-trivial mask (both 0 and 1 coordinates)."""
    module = TriMaskTrainModule(clip, rank=0, lambda_1=lambda_1, lambda_2=lambda_2,
                                lambda_3=lambda_3, lambda_sparse_i=lambda_sparse_i,
                                lambda_sparse_t=lambda_sparse_t, text_mask_width=TEXT_WIDTH,
                                text_mask_layers=1, text_mask_heads=8, text_mask_seed=seed,
                                text_gate_mode=HARD_GATE)
    if gate_std:
        torch.manual_seed(7)
        nn.init.normal_(module.text_mask_net.text_gate_projection.weight, std=gate_std)
    return module


# ---------------------------------------------------------------- A. gate type
def test_hard_gate_forward_is_strictly_binary_and_initially_open():
    _dist()
    clip = _clip()
    hidden = clip.encode_text(_tokens(), return_full=True)[1]

    fresh = TextMaskHead(width=TEXT_WIDTH, heads=8, seed=0, gate_mode=HARD_GATE)
    details = fresh.forward_with_details(hidden)
    assert torch.allclose(details['pT'], torch.full_like(details['pT'], 8.0 / 9.0), atol=1e-6)
    assert torch.equal(details['hT'], torch.ones_like(details['hT']))
    assert torch.equal(details['mT'], torch.ones_like(details['mT']))
    # the v0.2 gate has no 0.1 floor: the minimum value is 0, not TEXT_GATE_EPS
    assert float(details['mT'].min()) == 1.0

    module = _hard_module(clip, gate_std=1.0)
    mask = module.text_mask_net(hidden)
    assert torch.equal(mask, (mask >= 0.5).float()), 'the hard forward must be exactly 0/1'
    assert float(mask.min()) == 0.0 and float(mask.max()) == 1.0
    assert 0.0 < float(mask.mean()) < 1.0, 'the test fixture must not be all-open'
    assert float((mask == 0).float().mean()) > 0.0


def test_controlled_logits_produce_the_requested_keep_count_without_topk():
    head = TextMaskHead(width=TEXT_WIDTH, heads=8, seed=0, gate_mode=HARD_GATE)
    logits = torch.tensor([[-3.0, -0.1, 0.1, 3.0, 0.0]])
    details = head.gate_details(logits)
    assert details['hT'].tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0]], details['hT']
    # a different logit vector keeps a different number of coordinates: the count is a function of
    # the logits, not of a top-k or a fixed ratio
    other = head.gate_details(torch.tensor([[-3.0, -2.0, -1.0, -0.1, 0.1]]))
    assert float(other['hT'].sum()) == 1.0
    assert float(details['hT'].sum()) != float(other['hT'].sum())
    # exactly at the threshold the coordinate is kept (>= , not >)
    assert float(head.gate_details(torch.tensor([[0.0]]))['hT'][0, 0]) == 1.0


def test_soft_mode_has_the_floor_and_hard_mode_does_not():
    logits = torch.tensor([[-50.0, 50.0]])
    soft = TextMaskHead(width=TEXT_WIDTH, heads=8, gate_mode=SOFT_GATE).gate_details(logits)['mT']
    hard = TextMaskHead(width=TEXT_WIDTH, heads=8, gate_mode=HARD_GATE).gate_details(logits)['mT']
    assert float(soft.min()) == pytest.approx(TEXT_GATE_EPS, abs=1e-6)
    assert float(soft.max()) == pytest.approx(1.0, abs=1e-6)
    assert hard.tolist() == [[0.0, 1.0]]


# ---------------------------------------------------------------- B. gradients
def test_hard_st_gradient_is_the_sigmoid_proxy():
    head = TextMaskHead(width=TEXT_WIDTH, heads=8, gate_mode=HARD_GATE)
    logits = torch.tensor([[-5.0, -0.5, 0.5, 5.0]], requires_grad=True)
    details = head.gate_details(logits)
    loss = details['mT'].sum()
    grad = torch.autograd.grad(loss, logits)[0]
    # d mT / d aT == sigmoid'(aT) on every coordinate, open and closed alike: the threshold itself
    # contributes nothing, which is exactly the straight-through convention
    expected = torch.sigmoid(logits.detach()) * (1.0 - torch.sigmoid(logits.detach()))
    assert torch.allclose(grad, expected, atol=1e-6), (grad, expected)
    # and it is NOT a finite difference of the hard forward (that would be 0 almost everywhere)
    assert float(grad.abs().min()) > 0.0


def test_text_sparsity_uses_abs_with_the_straight_through_graph():
    head = TextMaskHead(width=TEXT_WIDTH, heads=8, gate_mode=HARD_GATE)
    logits = torch.tensor([[-5.0, -0.5, 0.5, 5.0]], requires_grad=True)
    mask = head.gate_details(logits)['mT']
    assert mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    loss = torch.mean(torch.abs(mask))
    assert float(loss) == pytest.approx(0.5, abs=1e-6)      # the forward value is the keep ratio
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    probability = torch.sigmoid(logits.detach())
    # manual reference: sign(m)*sigmoid'(*) / 4, with sign(0) = 0, so closed coordinates carry no
    # sparsity gradient at all -- this asymmetry is the adopted definition, not an accident
    manual = torch.sign(mask.detach()) * probability * (1 - probability) / 4.0
    assert torch.allclose(grad, manual, atol=1e-7), (grad, manual)
    assert float(grad[0, 0]) == 0.0 and float(grad[0, 1]) == 0.0
    assert float(grad[0, 2]) > 0.0

    # mean(mT) would give a different (non-zero) gradient on the closed coordinates; the test pins
    # the abs version so a silent switch to the soft sparsity cannot pass
    alternative = torch.autograd.grad(mask.mean(), logits, retain_graph=True)[0]
    assert not torch.allclose(alternative, manual, atol=1e-9)
    assert float(alternative[0, 0]) > 0.0


# ---------------------------------------------------------------- C. reuse and responsibilities
def test_masks_are_generated_once_and_shared_by_the_three_paths():
    _dist()
    clip = _clip()
    module = _hard_module(clip)
    images, tokens = _images(), _tokens()
    counts = {'visual': 0, 'stem': 0}
    clip.mask_net.register_forward_hook(lambda *_: counts.__setitem__('visual',
                                                                     counts['visual'] + 1))
    module.text_mask_net.text_stem.register_forward_hook(
        lambda *_: counts.__setitem__('stem', counts['stem'] + 1))
    out = module(images, tokens)
    assert counts == {'visual': 1, 'stem': 1}, counts
    from model.said_trimask import trimask_pairwise_scores
    q1, q2, q3, _ = trimask_pairwise_scores(out['v_a'], out['t_raw'], out['m_i'], out['m_t'])
    assert torch.equal(q1, out['q1']) and torch.equal(q2, out['q2'])
    assert torch.equal(q3, out['q3'])
    # the probability/hard/mask tensors returned for logging are the ones the gate actually used
    assert torch.equal(out['text_gate_hard'], out['m_t'])
    assert torch.equal(out['m_t'], (out['text_gate_probability'] >= 0.5).float())


def _responsibility(module, images, tokens):
    out = module(images, tokens)
    grads = _grad_map(out['loss_total'], module)
    return (_group_max_grad(grads, 'clip.mask_net'),
            _group_max_grad(grads, 'text_mask_net.text_gate_projection'),
            _group_max_grad(grads, 'text_mask_net.text_stem'))


def test_path_and_sparsity_responsibilities():
    _dist()
    images, tokens = _images(), _tokens()

    # path 1 alone: the text branch is untouched when both text-side terms are off
    visual, gate, stem = _responsibility(
        _hard_module(_clip(), lambda_1=10.0, lambda_2=0.0, lambda_3=0.0, lambda_sparse_i=0.0,
                     lambda_sparse_t=0.0), images, tokens)
    assert visual > 0.0 and gate == 0.0 and stem == 0.0

    # path 2 alone: the visual mask net is untouched
    visual, gate, stem = _responsibility(
        _hard_module(_clip(), lambda_1=0.0, lambda_2=10.0, lambda_3=0.0, lambda_sparse_i=0.0,
                     lambda_sparse_t=0.0), images, tokens)
    assert visual == 0.0 and gate > 0.0 and stem > 0.0

    # path 3 trains both sides
    visual, gate, stem = _responsibility(
        _hard_module(_clip(), lambda_1=0.0, lambda_2=0.0, lambda_3=10.0, lambda_sparse_i=0.0,
                     lambda_sparse_t=0.0), images, tokens)
    assert visual > 0.0 and gate > 0.0 and stem > 0.0

    # the visual sparsity term alone trains only the visual mask net
    visual, gate, stem = _responsibility(
        _hard_module(_clip(), lambda_1=0.0, lambda_2=0.0, lambda_3=0.0, lambda_sparse_i=2.0,
                     lambda_sparse_t=0.0), images, tokens)
    assert visual > 0.0 and gate == 0.0 and stem == 0.0

    # the text sparsity term alone trains the text branch, and nothing else
    visual, gate, stem = _responsibility(
        _hard_module(_clip(), lambda_1=0.0, lambda_2=0.0, lambda_3=0.0, lambda_sparse_i=0.0,
                     lambda_sparse_t=0.2), images, tokens)
    assert visual == 0.0, 'the text sparsity term must not touch the visual mask net'
    assert gate > 0.0, 'the text sparsity term must reach the gate projection'


# ---------------------------------------------------------------- D. reference equivalence
def _explicit_hard_reference(v, t_raw, mask_i, mask_t, eps=EPS):
    batch = v.shape[0]
    scores = [torch.zeros(batch, batch) for _ in range(3)]
    for i in range(batch):
        v_hat = F.normalize(v[i], dim=-1, eps=eps)
        for j in range(batch):
            t_hat = F.normalize(t_raw[j], dim=-1, eps=eps)
            r_t = F.normalize(t_raw[j] * mask_t[j], dim=-1, eps=eps)
            masked = F.normalize(v[i] * mask_i[j], dim=-1, eps=eps)
            scores[0][i, j] = 100.0 * torch.dot(masked, t_hat)
            scores[1][i, j] = 100.0 * torch.dot(v_hat, r_t)
            scores[2][i, j] = 100.0 * torch.dot(masked, r_t)
    targets = torch.arange(batch)
    i2t = [F.cross_entropy(score, targets) for score in scores]
    t2i = [F.cross_entropy(score.t(), targets) for score in scores]
    total = (10.0 * (i2t[0] + t2i[0]) + 1.0 * (i2t[1] + t2i[1]) + 1.0 * (i2t[2] + t2i[2])
             + 2.0 * torch.mean(torch.abs(mask_i)) + 0.2 * torch.mean(torch.abs(mask_t)))
    return total, scores


def test_hard_gate_matrix_form_matches_the_explicit_reference_with_gradients():
    _dist()
    generator = torch.Generator().manual_seed(5)
    v = torch.randn(BATCH, DIM, generator=generator)
    t_raw = torch.randn(BATCH, DIM, generator=generator)
    t_raw[0] = -0.7 * v[0]
    logits_i = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(6)) * 2.0
    logits_t = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(7)) * 2.0
    logits_i.requires_grad_(True)
    logits_t.requires_grad_(True)
    soft_i = torch.sigmoid(logits_i)
    mask_i = (soft_i >= 0.5).float() - soft_i.detach() + soft_i
    soft_t = torch.sigmoid(logits_t)
    mask_t = (soft_t >= 0.5).float() - soft_t.detach() + soft_t

    out = trimask_terms(v, t_raw, mask_i, mask_t, 0, lambda_sparse_t=LAMBDA_SPARSE_T_HS,
                        gate_mode=HARD_GATE)
    reference_total, reference_scores = _explicit_hard_reference(v, t_raw, mask_i, mask_t)
    assert float(out['loss_total']) == pytest.approx(float(reference_total), rel=1e-5)
    for index, key in enumerate(('q1', 'q2', 'q3')):
        assert torch.allclose(out[key], reference_scores[index], atol=1e-4, rtol=1e-5), key

    ours = torch.autograd.grad(out['loss_total'], [logits_i, logits_t], retain_graph=True)
    theirs = torch.autograd.grad(reference_total, [logits_i, logits_t])
    for left, right in zip(ours, theirs):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-4)
    # the text gate really is a mix of open and closed coordinates in this fixture
    assert 0.0 < float(mask_t.mean()) < 1.0


# ---------------------------------------------------------------- E. boundaries
def _finite_under(weights):
    _dist()
    generator = torch.Generator().manual_seed(3)
    v = torch.randn(2, DIM, generator=generator)
    t_raw = torch.randn(2, DIM, generator=generator)
    logits_i = torch.full((2, DIM), weights[0], requires_grad=True)
    logits_t = torch.full((2, DIM), weights[1], requires_grad=True)
    soft_i = torch.sigmoid(logits_i)
    mask_i = (soft_i >= 0.5).float() - soft_i.detach() + soft_i
    soft_t = torch.sigmoid(logits_t)
    mask_t = (soft_t >= 0.5).float() - soft_t.detach() + soft_t
    out = trimask_terms(v, t_raw, mask_i, mask_t, 0, lambda_sparse_t=LAMBDA_SPARSE_T_HS,
                        gate_mode=HARD_GATE)
    grads = torch.autograd.grad(out['loss_total'], [logits_i, logits_t])
    return out, grads, mask_i, mask_t


def test_all_open_all_closed_and_empty_intersection_stay_finite():
    # both gates fully open
    out, grads, mask_i, mask_t = _finite_under((20.0, 20.0))
    assert torch.isfinite(out['loss_total'])
    assert float(out['mask_intersection_intersection_count_mean']) == pytest.approx(DIM, abs=1e-6)
    assert float(out['mask_intersection_jaccard_mean']) == pytest.approx(1.0, abs=1e-6)
    assert all(torch.isfinite(g).all() for g in grads)

    # both gates fully closed: empty masked vectors, empty union, undefined Jaccard reported as
    # null together with the count of undefined rows -- never a fabricated 0
    out, grads, mask_i, mask_t = _finite_under((-20.0, -20.0))
    assert torch.isfinite(out['loss_total'])
    assert float(out['mask_i_empty_fraction']) == 1.0
    assert float(out['mask_t_empty_fraction']) == 1.0
    assert float(out['mask_intersection_union_empty_fraction']) == 1.0
    assert float(out['mask_intersection_jaccard_defined_count']) == 0.0
    assert out['mask_intersection_jaccard_mean'] is None
    assert json.loads(json.dumps({'jaccard': out['mask_intersection_jaccard_mean']}))['jaccard'] is None
    assert all(torch.isfinite(g).all() for g in grads)
    # degeneracy is recorded, never auto-repaired: a fully closed gate stays closed
    assert float(mask_t.abs().mean()) == 0.0
    assert float(out['mask_intersection_intersection_empty_fraction']) == 1.0

    # visual closed, text open: both non-empty is false, the intersection is empty, no NaN
    out, grads, mask_i, mask_t = _finite_under((-20.0, 20.0))
    assert torch.isfinite(out['loss_total'])
    assert float(out['mask_intersection_intersection_empty_fraction']) == 1.0
    assert float(out['mask_intersection_both_nonempty_but_intersection_empty_fraction']) == 0.0
    assert all(torch.isfinite(g).all() for g in grads)


def test_nonfinite_gradients_are_detected_and_the_update_is_refused(monkeypatch):
    _dist()
    clip = _clip()
    clip.logit_scale.requires_grad_(False)
    module = _hard_module(clip)
    images, tokens = _images(), _tokens()
    batch = {'image_a': images, 'caption_said': ['a photo of a cat .'] * BATCH}
    optimizer, mask_optimizer, _, _ = build_trimask_optimizers(
        clip, module.text_mask_net, argparse.Namespace(lr=1e-3, mask_lr=1e-2,
                                                       weight_decay=1e-2))

    def run():
        return trimask_train_step(module, batch, optimizer, mask_optimizer, torch.device('cpu'),
                                  torch.float32, amp_enabled=False, world_size=1)

    first = run()
    assert float(first['grads_finite']) == 1.0
    before = [p.detach().clone() for p in module.parameters()]

    import train_said_trimask as trainer
    monkeypatch.setattr(trainer, 'gradients_are_finite', lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        run()
    after = [p.detach().clone() for p in module.parameters()]
    for left, right in zip(before, after):
        assert torch.equal(left, right), 'a refused update must not change any parameter'

    monkeypatch.undo()
    # and the predicate itself really looks at the gradients
    for parameter in module.parameters():
        parameter.grad = None
    assert gradients_are_finite(module, torch.device('cpu'), 1) is True
    module.clip.mask_net.out.weight.grad = torch.full_like(module.clip.mask_net.out.weight,
                                                           float('inf'))
    assert gradients_are_finite(module, torch.device('cpu'), 1) is False


# ---------------------------------------------------------------- F. two-rank DDP
WORKER = os.path.join(REPO_ROOT, 'tests', '_trimask_hs_ddp_worker.py')
GRAD_RTOL = 1e-3
GRAD_FLOOR = 1e-4
PARAM_RTOL = 1e-5


def _peak(values):
    return max(float(torch.tensor(value, dtype=torch.float64).abs().max())
               for value in values if value is not None)


def _run_worker(mode, rows, out_path, port):
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port', str(port), WORKER, '--mode', mode, '--rows', str(rows),
               '--out', out_path]
    env = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
           'GLOO_SOCKET_IFNAME': 'lo'}
    return subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def test_two_rank_hard_gate_step_with_text_sparsity_matches_single_process(tmp_path):
    out = str(tmp_path / 'hs_{rank}.json')
    single = _run_worker('single', 2, out, 29705)
    assert single.returncode == 0, single.stderr[-3000:]
    single_payload = json.load(open(out.format(rank=0)))
    assert single_payload['rows'] == 4
    assert single_payload['text_zero_fraction'] > 0.0 and \
        single_payload['text_zero_fraction'] < 1.0, 'the fixture mask must be non-trivial'

    ranks_result = _run_worker('rank', 2, out, 29706)
    assert ranks_result.returncode == 0, ranks_result.stderr[-3000:]
    ranks = [json.load(open(out.format(rank=rank))) for rank in range(2)]
    assert [payload['rows'] for payload in ranks] == [2, 2]

    for label in ('loss_total', 'loss_1', 'loss_2', 'loss_3', 'loss_sparse_i', 'loss_sparse_t'):
        averaged = 0.5 * (ranks[0][label] + ranks[1][label])
        assert averaged == pytest.approx(single_payload[label], rel=1e-5), (label, averaged)

    reference = single_payload['grads']
    global_peak = _peak(reference.values())
    for rank, payload in enumerate(ranks):
        for name, value in reference.items():
            other = payload['grads'][name]
            if value is None or other is None:
                assert value is None and other is None, (rank, name)
                continue
            left = torch.tensor(value, dtype=torch.float64)
            right = torch.tensor(other, dtype=torch.float64)
            difference = float((left - right).abs().max())
            allowed = GRAD_RTOL * max(float(left.abs().max()), GRAD_FLOOR * global_peak)
            assert difference <= allowed, (rank, name, difference, allowed)
    for name in reference:
        if reference[name] is None:
            continue
        assert ranks[0]['grads'][name] == ranks[1]['grads'][name], name
    assert ranks[0]['param_digest'] == ranks[1]['param_digest']
    for name, value in single_payload['param_sample'].items():
        left = torch.tensor(value, dtype=torch.float64)
        right = torch.tensor(ranks[0]['param_sample'][name], dtype=torch.float64)
        difference = float((left - right).abs().max())
        allowed = PARAM_RTOL * float(left.abs().max())
        assert difference <= allowed, (name, difference, allowed)


# ---------------------------------------------------------------- G. regression
def test_soft_mode_default_is_unchanged():
    head = TextMaskHead(width=TEXT_WIDTH, heads=8, seed=0)
    assert head.gate_mode == SOFT_GATE
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    expected = TEXT_GATE_EPS + (1.0 - TEXT_GATE_EPS) * torch.sigmoid(logits)
    assert torch.allclose(head.gate_details(logits)['mT'], expected, atol=1e-7)
    assert torch.allclose(head.gate_details(logits)['hT'], torch.ones_like(logits))


def test_lambda_sparse_t_reaches_the_loss_and_the_gradients():
    _dist()
    clip = _clip()
    images, tokens = _images(), _tokens()
    zero = _hard_module(clip, lambda_sparse_t=0.0)
    out_zero = zero(images, tokens)
    assert float(out_zero['weighted_loss_sparse_t']) == 0.0
    clip2 = _clip()
    nonzero = _hard_module(clip2, lambda_sparse_t=LAMBDA_SPARSE_T_HS)
    out = nonzero(images, tokens)
    assert float(out['weighted_loss_sparse_t']) == pytest.approx(
        LAMBDA_SPARSE_T_HS * float(out['loss_sparse_t']), rel=1e-6)
    expected_total = (10.0 * float(out['loss_1']) + float(out['loss_2']) + float(out['loss_3'])
                      + 2.0 * float(out['loss_sparse_i'])
                      + LAMBDA_SPARSE_T_HS * float(out['loss_sparse_t']))
    assert float(out['loss_total']) == pytest.approx(expected_total, rel=1e-4)
    assert float(out['lambda_sparse_t']) == pytest.approx(LAMBDA_SPARSE_T_HS, rel=1e-6)
    # the term is not a constant: it must change the gradient the gate sees
    grads_a = _grad_map(out['loss_total'], nonzero)
    grads_b = _grad_map(out_zero['loss_total'], zero)
    name = 'text_mask_net.text_gate_projection.weight'
    assert not torch.allclose(grads_a[name], grads_b[name])


def test_checkpoint_mode_mismatch_is_rejected():
    soft = {'objective': OBJECTIVE, 'arm': ARM, 'text_gate_mode': SOFT_GATE, 'lambda_sparse_t': 0.0}
    hard = {'objective': OBJECTIVE_HS, 'arm': ARM_HS, 'text_gate_mode': HARD_GATE,
            'lambda_sparse_t': LAMBDA_SPARSE_T_HS}
    legacy = {'objective': OBJECTIVE, 'arm': ARM}          # a real v0.1 checkpoint payload
    check_checkpoint_compatibility(soft, ARM, OBJECTIVE, SOFT_GATE, 0.0)
    check_checkpoint_compatibility(hard, ARM_HS, OBJECTIVE_HS, HARD_GATE, LAMBDA_SPARSE_T_HS)
    check_checkpoint_compatibility(legacy, ARM, OBJECTIVE, SOFT_GATE, 0.0)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(soft, ARM_HS, OBJECTIVE_HS, HARD_GATE, LAMBDA_SPARSE_T_HS)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(hard, ARM, OBJECTIVE, SOFT_GATE, 0.0)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(legacy, ARM_HS, OBJECTIVE_HS, HARD_GATE, LAMBDA_SPARSE_T_HS)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility({**hard, 'lambda_sparse_t': 0.5}, ARM_HS, OBJECTIVE_HS,
                                       HARD_GATE, LAMBDA_SPARSE_T_HS)


def test_cli_rejects_contradictory_mode_and_sparsity_flags():
    base = [sys.executable, TRAINER, '--init_state', '/nonexistent', '--output_dir', '/tmp/x']
    contradictory = subprocess.run(base + ['--text_gate_mode', SOFT_GATE, '--lambda_sparse_t', '0.2'],
                                   capture_output=True, text=True, cwd=REPO_ROOT)
    assert contradictory.returncode != 0
    assert 'lambda_sparse_t' in (contradictory.stderr + contradictory.stdout)
    wrong_arm = subprocess.run(base + ['--text_gate_mode', HARD_GATE, '--arm', ARM],
                               capture_output=True, text=True, cwd=REPO_ROOT)
    assert wrong_arm.returncode != 0
    assert 'contradicts' in (wrong_arm.stderr + wrong_arm.stdout)


def test_export_tool_accepts_both_modes_and_rejects_a_mode_mismatch(tmp_path):
    help_text = subprocess.run([sys.executable, EXPORT_TOOL, '--help'], capture_output=True,
                               text=True, cwd=REPO_ROOT)
    assert help_text.returncode == 0
    for flag in ('--checkpoint', '--out', '--expect-steps', '--expect-gate-mode'):
        assert flag in help_text.stdout, flag
    assert SOFT_GATE in help_text.stdout and HARD_GATE in help_text.stdout

    # a v0.2 payload must not be exported under the v0.1 expectation, and vice versa: both checks
    # fire before any model is built, so they are exact and fast
    module = _hard_module(_clip())
    payload_path = tmp_path / 'hard.pt'
    torch.save({'model': module.clip.state_dict(),
                'text_mask_net': module.text_mask_net.state_dict(),
                'completed_steps': 500, 'objective': OBJECTIVE_HS, 'arm': ARM_HS,
                'text_gate_mode': HARD_GATE, 'lambda_sparse_t': LAMBDA_SPARSE_T_HS},
               payload_path)
    out = str(tmp_path / 'student.pt')
    mismatched = subprocess.run([sys.executable, EXPORT_TOOL, '--checkpoint', str(payload_path),
                                 '--out', out, '--expect-gate-mode', SOFT_GATE],
                                capture_output=True, text=True, cwd=REPO_ROOT)
    assert mismatched.returncode != 0
    assert 'text_gate_mode' in (mismatched.stderr + mismatched.stdout)
    wrong_steps = subprocess.run([sys.executable, EXPORT_TOOL, '--checkpoint', str(payload_path),
                                  '--out', out, '--expect-steps', '499'],
                                 capture_output=True, text=True, cwd=REPO_ROOT)
    assert wrong_steps.returncode != 0
    assert 'completed_steps' in (wrong_steps.stderr + wrong_steps.stdout)
    assert not os.path.exists(out)


def test_head_construction_preserves_the_cpu_and_the_current_cuda_rng():
    """The isolation claim covers CPU *and* CUDA, not just the generator that is easy to check."""
    torch.manual_seed(11)
    cpu_before = torch.get_rng_state()
    cuda_before = None
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        cuda_before = torch.cuda.get_rng_state(0)
    TextMaskHead(width=TEXT_WIDTH, heads=8, seed=0, gate_mode=HARD_GATE)
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_before)
    # reproducibility from the seed alone, independently of the ambient generators
    first = TextMaskHead(width=TEXT_WIDTH, heads=8, seed=3, gate_mode=HARD_GATE)
    second = TextMaskHead(width=TEXT_WIDTH, heads=8, seed=3, gate_mode=HARD_GATE)
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, second.state_dict()[name]), name


def test_bare_student_export_keys_exclude_the_text_branch():
    from model.said_trimask import student_state_from_checkpoint
    module = _hard_module(_clip())
    state = module.clip.state_dict()
    assert student_state_from_checkpoint(state) == state
    polluted = dict(state)
    polluted['text_mask_net.text_gate_projection.weight'] = torch.zeros(TEXT_WIDTH, TEXT_WIDTH)
    with pytest.raises(ValueError):
        student_state_from_checkpoint(polluted)
