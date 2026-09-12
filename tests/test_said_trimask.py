"""S0-TriMask v0.1 acceptance tests.

Coverage required by the task, in order: path-1 degeneration to the reference SmartCLIP step,
text-gate initialisation equivalence, single mask generation and reuse, per-path gradient
responsibilities, an independent per-pair mathematical reference (forward *and* gradients),
cross-entropy basic properties, a real two-rank DDP step against the single-process reference, and
the CLI / bare-student interface. The stubs mirror ``tests/test_said_cls_c1.py`` so the two
objectives are exercised the same way.
"""
import argparse
import json
import math
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

from model.said_cls_cvssl import compute_smartclip_terms, said_mask_from_hidden  # noqa: E402
from model.said_trimask import (LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_SPARSE_I,  # noqa: E402
                                OBJECTIVE, TEXT_GATE_EPS, TriMaskTrainModule,
                                _path_terms, student_state_from_checkpoint,
                                trimask_pairwise_scores, trimask_terms)
from train_said_trimask import (build_trimask_optimizers,  # noqa: E402
                                trimask_train_step)

DIM = 32
TEXT_WIDTH = 32
TOKENS = 5
BATCH = 4
EPS = 1e-6
ATOL = 0.0
GRAD_RTOL = 1e-4


class StubVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubMaskNet(nn.Module):
    """Behaves like the repository ``MaskNetwork``: ``[B, L, W] -> [B, DIM]``."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, 12)
        self.out = nn.Linear(12, DIM)
        self.position = nn.Parameter(torch.randn(TOKENS, 12) * 0.1)

    def forward(self, hidden):
        features = torch.tanh(self.proj(hidden)) + self.position[:hidden.shape[1]]
        return self.out(features.mean(dim=1))


class StubClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = nn.Parameter(torch.randn(TEXT_WIDTH, DIM) * 0.1)
        self.text_offset = nn.Parameter(torch.randn(TOKENS, TEXT_WIDTH) * 0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        self.visual = StubVisual()
        self.mask_net = StubMaskNet()

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens, return_full=False, return_pool=False):
        base = tokens.float().mean(dim=1, keepdim=True)
        ramp = torch.arange(TOKENS, dtype=torch.float32).view(1, TOKENS)
        hidden = (base + ramp).unsqueeze(-1).repeat(1, 1, TEXT_WIDTH) * 0.05
        hidden = hidden + self.text_offset
        pooled = hidden.mean(dim=1) @ self.text_projection
        return (pooled, hidden) if return_full else pooled


def _dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29695')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    return dist


def _clip(seed=1234):
    torch.manual_seed(seed)
    return StubClip()


def _tokens(seed=3, batch=BATCH):
    return torch.randint(0, 100, (batch, TOKENS),
                         generator=torch.Generator().manual_seed(seed))


def _images(seed=11, batch=BATCH):
    return torch.randn(batch, 3, 4, 4, generator=torch.Generator().manual_seed(seed)) * 0.4


def _module(clip, lambda_1=LAMBDA_1, lambda_2=LAMBDA_2, lambda_3=LAMBDA_3,
            lambda_sparse_i=LAMBDA_SPARSE_I, seed=0):
    return TriMaskTrainModule(clip, rank=0, lambda_1=lambda_1, lambda_2=lambda_2,
                              lambda_3=lambda_3, lambda_sparse_i=lambda_sparse_i,
                              text_mask_width=TEXT_WIDTH, text_mask_layers=1,
                              text_mask_heads=8, text_mask_seed=seed)


def _open_gate(module, std=0.05):
    """Leave the zero-output-layer special case: only then does the stem see a gradient."""
    torch.manual_seed(7)
    nn.init.normal_(module.text_mask_net.text_gate_projection.weight, std=std)


def _grad_map(loss, module):
    items = [(name, parameter) for name, parameter in module.named_parameters()
             if parameter.requires_grad]
    grads = torch.autograd.grad(loss, [parameter for _, parameter in items],
                                retain_graph=True, allow_unused=True)
    return {name: grad for (name, _), grad in zip(items, grads)}


def _group_max_grad(grads, prefix):
    worst = 0.0
    for name, grad in grads.items():
        if not name.startswith(prefix) or grad is None:
            continue
        worst = max(worst, float(grad.detach().abs().max()))
    return worst


def _assert_grad_maps_close(left, right, rtol=GRAD_RTOL):
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        difference = float((a.double() - b.double()).abs().max())
        allowed = ATOL + rtol * float(a.double().abs().max())
        assert difference <= allowed, (name, difference, allowed)


def _explicit_reference(v, t_raw, mask_i, mask_t, eps=EPS):
    """Per-pair loops: no matrix algebra, no broadcasting, no shared denominator."""
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
    sparse = torch.mean(torch.abs(mask_i))
    total = (10.0 * (i2t[0] + t2i[0]) + 1.0 * (i2t[1] + t2i[1]) + 1.0 * (i2t[2] + t2i[2])
             + 2.0 * sparse)
    return total, scores


# ---------------------------------------------------------------- 1. path-1 degeneration
def test_path1_degenerates_to_the_reference_smartclip_step():
    """lambda_2 = lambda_3 = 0: loss and shared-parameter gradients are the reference S0 step."""
    _dist()
    clip = _clip()
    module = _module(clip, lambda_2=0.0, lambda_3=0.0)
    images, tokens = _images(), _tokens()

    v = clip.encode_image(images)
    t_raw, hidden = clip.encode_text(tokens, return_full=True)
    mask_s, _, _ = said_mask_from_hidden(clip.mask_net, hidden, soft_mask=False)
    reference = compute_smartclip_terms(v, t_raw, mask_s, 0, lambda_align=10.0,
                                        lambda_sparse=2.0)

    out = module(images, tokens)
    assert float(out['loss_1']) == pytest.approx(float(reference['loss_sidm']
                                                       + reference['loss_dism']), rel=1e-5)
    assert float(out['loss_sparse_i']) == pytest.approx(float(reference['loss_sparsity']),
                                                        rel=1e-6)
    assert float(out['loss_total']) == pytest.approx(float(reference['loss_smart']), rel=1e-5)

    clip_grads = _grad_map(out['loss_total'], clip)
    reference_grads = _grad_map(reference['loss_smart'], clip)
    _assert_grad_maps_close(clip_grads, reference_grads)


# ---------------------------------------------------------------- 2. text gate initialisation
def test_text_gate_initialisation_preserves_the_text_direction():
    _dist()
    clip = _clip()
    module = _module(clip)
    images, tokens = _images(), _tokens()
    t_raw, hidden = clip.encode_text(tokens, return_full=True)

    gate = module.text_mask_net(hidden)
    assert torch.allclose(gate, torch.full_like(gate, 0.9), atol=1e-6)
    assert torch.allclose(F.normalize(t_raw * gate, dim=-1, eps=EPS),
                          F.normalize(t_raw, dim=-1, eps=EPS), atol=1e-6)

    out = module(images, tokens)
    v = clip.encode_image(images)
    unmasked = 100.0 * (F.normalize(v, dim=-1, eps=EPS)
                        @ F.normalize(t_raw, dim=-1, eps=EPS).t())
    assert torch.allclose(out['q2'].detach(), unmasked.detach(), atol=1e-4)
    assert torch.allclose(out['q3'].detach(), out['q1'].detach(), atol=1e-4)
    # note: the *default* total loss is still not the S0 loss, because paths 2 and 3 carry
    # weight 1 each; only lambda_2 = lambda_3 = 0 degenerates to S0 (previous test)


# ---------------------------------------------------------------- 3. one mask, reused
def test_one_mask_per_caption_is_generated_once_and_reused_by_all_paths():
    _dist()
    clip = _clip()
    module = _module(clip)
    # the gate is opened on purpose: with the zeroed output layer the text gate is the constant 0.9
    # for every caption, so "another caption changes the text mask" would be untestable
    _open_gate(module)
    images, tokens = _images(), _tokens()

    counts = {'visual_mask': 0, 'text_stem': 0}

    def count_visual(_module, _inputs, _output):
        counts['visual_mask'] += 1

    def count_text(_module, _inputs, _output):
        counts['text_stem'] += 1

    clip.mask_net.register_forward_hook(count_visual)
    module.text_mask_net.text_stem.register_forward_hook(count_text)

    out = module(images, tokens)
    assert counts == {'visual_mask': 1, 'text_stem': 1}, counts

    # the score matrices really are functions of the two masks the module reported: rebuilding
    # them from out['m_i']/out['m_t'] reproduces q1/q2/q3 bit for bit
    v = out['v_a']
    t_raw = out['t_raw']
    q1, q2, q3, _ = trimask_pairwise_scores(v, t_raw, out['m_i'], out['m_t'])
    assert torch.equal(q1, out['q1'])
    assert torch.equal(q2, out['q2'])
    assert torch.equal(q3, out['q3'])

    # masks depend on the text only: swapping the images changes nothing about them
    permutation = torch.tensor([2, 0, 3, 1])
    swapped = module(images[permutation], tokens)
    assert torch.equal(out['m_i'], swapped['m_i'])
    assert torch.equal(out['m_t'], swapped['m_t'])

    # and a different caption elsewhere in the batch cannot change this caption's masks
    other_tokens = tokens.clone()
    other_tokens[1] = torch.tensor([5, 6, 7, 8, 9])
    other_tokens[3] = torch.tensor([11, 12, 13, 14, 15])
    changed = module(images, other_tokens)
    assert torch.equal(out['m_i'][0], changed['m_i'][0])
    assert torch.equal(out['m_t'][0], changed['m_t'][0])
    assert torch.equal(out['mask_logits_i'][0], changed['mask_logits_i'][0])
    # the caption that did change has a different text gate and different visual logits
    assert not torch.equal(out['m_t'][1], changed['m_t'][1])
    assert not torch.equal(out['mask_logits_i'][1], changed['mask_logits_i'][1])

    # determinism: the same input twice gives the same masks and the same scores
    repeat = module(images, tokens)
    assert torch.equal(out['m_i'], repeat['m_i'])
    assert torch.equal(out['m_t'], repeat['m_t'])
    assert torch.equal(out['q1'], repeat['q1'])


# ---------------------------------------------------------------- 4. gradient responsibilities
def _backward_groups(module, images, tokens):
    out = module(images, tokens)
    grads = _grad_map(out['loss_total'], module)
    return {
        'visual_mask_net': _group_max_grad(grads, 'clip.mask_net'),
        'text_stem': _group_max_grad(grads, 'text_mask_net.text_stem'),
        'text_gate': _group_max_grad(grads, 'text_mask_net.text_gate_projection'),
    }


def test_gradient_responsibilities_of_the_three_paths():
    _dist()
    images, tokens = _images(), _tokens()

    # path 1 only (sparse term switched off to isolate the path, see the note below)
    module = _module(_clip(), lambda_1=10.0, lambda_2=0.0, lambda_3=0.0, lambda_sparse_i=0.0)
    _open_gate(module)
    groups = _backward_groups(module, images, tokens)
    assert groups['visual_mask_net'] > 0.0, groups
    assert groups['text_stem'] == 0.0 and groups['text_gate'] == 0.0, groups

    # path 2 only: the visual mask net is untouched by the path, ...
    module = _module(_clip(), lambda_1=0.0, lambda_2=10.0, lambda_3=0.0, lambda_sparse_i=0.0)
    _open_gate(module)
    groups = _backward_groups(module, images, tokens)
    assert groups['text_stem'] > 0.0 and groups['text_gate'] > 0.0, groups
    assert groups['visual_mask_net'] == 0.0, groups

    # ... but the sparse term of the *real* objective keeps training the visual mask anyway:
    # L_sparse_I is defined on the real matched pairs and is not a property of any single path
    module = _module(_clip(), lambda_1=0.0, lambda_2=10.0, lambda_3=0.0,
                     lambda_sparse_i=LAMBDA_SPARSE_I)
    _open_gate(module)
    groups = _backward_groups(module, images, tokens)
    assert groups['visual_mask_net'] > 0.0, groups

    # path 3 trains both branches
    module = _module(_clip(), lambda_1=0.0, lambda_2=0.0, lambda_3=10.0, lambda_sparse_i=0.0)
    _open_gate(module)
    groups = _backward_groups(module, images, tokens)
    assert groups['visual_mask_net'] > 0.0, groups
    assert groups['text_stem'] > 0.0 and groups['text_gate'] > 0.0, groups


def test_text_branch_gate_zero_at_initialisation_but_the_stem_is_live_afterwards():
    """W=0 hides the stem from the first update only; after one AdamW step it must see gradient."""
    torch.manual_seed(11)
    clip = StubClip()
    _dist()
    module = _module(clip, lambda_1=0.0, lambda_2=10.0, lambda_3=0.0, lambda_sparse_i=0.0)
    images, tokens = _images(), _tokens()
    clip.logit_scale.requires_grad_(False)
    optimizer, mask_optimizer, _, _ = build_trimask_optimizers(
        clip, module.text_mask_net, argparse.Namespace(lr=1e-3, mask_lr=1e-2,
                                                       weight_decay=1e-2))
    first = module(images, tokens)
    grads = _grad_map(first['loss_total'], module)
    assert _group_max_grad(grads, 'text_mask_net.text_gate_projection') > 0.0
    assert _group_max_grad(grads, 'text_mask_net.text_stem') == 0.0

    first['loss_total'].backward()
    mask_optimizer.step()
    mask_optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)

    second = module(images, tokens)
    grads = _grad_map(second['loss_total'], module)
    assert _group_max_grad(grads, 'text_mask_net.text_stem') > 0.0, 'the stem stayed dead'


# ---------------------------------------------------------------- 5. independent reference
def test_matrix_form_matches_the_explicit_pairwise_reference():
    _dist()
    generator = torch.Generator().manual_seed(5)
    v = torch.randn(BATCH, DIM, generator=generator)
    t_raw = torch.randn(BATCH, DIM, generator=generator)
    t_raw[0] = -0.7 * v[0]                      # guarantee both signs of the cosine
    t_raw[2] = 0.2 * v[2]
    logits_i = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(6)) * 2.0
    logits_t = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(7))
    logits_i.requires_grad_(True)
    logits_t.requires_grad_(True)
    soft_i = torch.sigmoid(logits_i)
    mask_i = (soft_i >= 0.5).float() - soft_i.detach() + soft_i
    mask_t = TEXT_GATE_EPS + (1.0 - TEXT_GATE_EPS) * torch.sigmoid(logits_t)

    out = trimask_terms(v, t_raw, mask_i, mask_t, 0)
    reference_total, reference_scores = _explicit_reference(v, t_raw, mask_i, mask_t)
    assert float(out['loss_total']) == pytest.approx(float(reference_total), rel=1e-5)
    for index, key in enumerate(('q1', 'q2', 'q3')):
        assert torch.allclose(out[key], reference_scores[index], atol=1e-4, rtol=1e-5), key

    module_grads = torch.autograd.grad(out['loss_total'], [logits_i, logits_t], retain_graph=True)
    reference_grads = torch.autograd.grad(reference_total, [logits_i, logits_t])
    for left, right in zip(module_grads, reference_grads):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-4)


def test_zero_mask_norm_uses_the_declared_lower_bound_without_nan():
    _dist()
    generator = torch.Generator().manual_seed(9)
    v = torch.randn(2, DIM, generator=generator)
    t_raw = torch.randn(2, DIM, generator=generator)
    logits_i = torch.full((2, DIM), -20.0, requires_grad=True)   # every coordinate stays closed
    soft_i = torch.sigmoid(logits_i)
    mask_i = (soft_i >= 0.5).float() - soft_i.detach() + soft_i
    mask_t = torch.full((2, DIM), 0.9)
    out = trimask_terms(v, t_raw, mask_i, mask_t, 0)
    assert torch.isfinite(out['loss_total'])
    diagonal = out['q1'].diagonal()
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-9)
    grads = torch.autograd.grad(out['loss_total'], logits_i)[0]
    assert torch.isfinite(grads).all()


# ---------------------------------------------------------------- 6. cross-entropy properties
def test_cross_entropy_directional_properties():
    _dist()
    batch = 4
    targets = torch.arange(batch)
    uniform = torch.zeros(batch, batch)
    i2t, t2i, _ = _path_terms(uniform, 0, batch, targets)
    assert float(i2t) == pytest.approx(math.log(batch), rel=1e-6)
    assert float(t2i) == pytest.approx(math.log(batch), rel=1e-6)

    # shifting one direction's whole logit row set leaves that direction's cross-entropy alone
    shifted = uniform + 5.0
    i2t_shifted, _, _ = _path_terms(shifted, 0, batch, targets)
    assert float(i2t_shifted) == pytest.approx(float(i2t), rel=1e-6)

    # a better positive and a worse negative must improve the same direction
    improved = uniform.clone()
    improved[range(batch), range(batch)] += 3.0
    i2t_better, _, _ = _path_terms(improved, 0, batch, targets)
    assert float(i2t_better) < float(i2t)

    worse_negatives = uniform.clone()
    for i in range(batch):
        for j in range(batch):
            if i != j:
                worse_negatives[i, j] -= 3.0
    i2t_worse_neg, _, _ = _path_terms(worse_negatives, 0, batch, targets)
    assert float(i2t_worse_neg) < float(i2t)


# ---------------------------------------------------------------- 7. real DDP step
WORKER = os.path.join(REPO_ROOT, 'tests', '_trimask_ddp_worker.py')
# Measured on this exact setup: the gradients of the dominant parameters agree to ~1e-6 relative
# between the single-process and the two-rank run, and gradients that are essentially zero (they
# cancel out of much larger intermediate terms) show only absolute noise. The tolerance therefore
# scales with the *global* gradient peak, with a floor for those near-zero entries; any real
# scaling mistake (a stray world_size factor, a doubled path weight) moves the dominant entries by
# ~100%, far outside this band.
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


def test_two_rank_trimask_step_matches_the_single_process_reference(tmp_path):
    out = str(tmp_path / 'trimask_{rank}.json')
    single = _run_worker('single', 2, out, 29697)
    assert single.returncode == 0, single.stderr[-3000:]
    single_payload = json.load(open(out.format(rank=0)))
    assert single_payload['rows'] == 4

    ranks_result = _run_worker('rank', 2, out, 29698)
    assert ranks_result.returncode == 0, ranks_result.stderr[-3000:]
    ranks = [json.load(open(out.format(rank=rank))) for rank in range(2)]
    assert [payload['rows'] for payload in ranks] == [2, 2]

    # each rank holds its own anchors, so its local means differ: the convention is verified by
    # averaging the two rank losses back into the single-process global mean
    for label in ('loss_total', 'loss_1', 'loss_2', 'loss_3', 'loss_sparse_i'):
        averaged = 0.5 * (ranks[0][label] + ranks[1][label])
        assert averaged == pytest.approx(single_payload[label], rel=1e-5), (label, averaged)

    reference = single_payload['grads']
    global_peak = _peak(reference.values())
    for rank, payload in enumerate(ranks):
        assert payload['grads'].keys() == reference.keys()
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
    # ranks must be bit-identical to each other (same all-reduce, same tensors)
    for name in reference:
        if reference[name] is None:
            continue
        assert ranks[0]['grads'][name] == ranks[1]['grads'][name], name
    assert ranks[0]['param_digest'] == ranks[1]['param_digest']
    assert ranks[0]['text_branch_digest'] == ranks[1]['text_branch_digest']
    # ... and one real AdamW update through DDP lands on the same parameters as one single-process
    # update of the whole global batch
    for name, value in single_payload['param_sample'].items():
        left = torch.tensor(value, dtype=torch.float64)
        right = torch.tensor(ranks[0]['param_sample'][name], dtype=torch.float64)
        difference = float((left - right).abs().max())
        allowed = PARAM_RTOL * float(left.abs().max())
        assert difference <= allowed, (name, difference, allowed)


# ---------------------------------------------------------------- 8. interface / CLI
def test_student_export_rejects_the_new_text_branch_keys():
    module = _module(_clip())
    state = module.clip.state_dict()
    assert student_state_from_checkpoint(state) == state
    polluted = dict(state)
    polluted['text_mask_net.text_gate_projection.weight'] = torch.zeros(TEXT_WIDTH, TEXT_WIDTH)
    with pytest.raises(ValueError):
        student_state_from_checkpoint(polluted)


def test_optimizer_groups_cover_everything_once_and_exclude_logit_scale():
    clip = _clip()
    module = _module(clip)
    clip.logit_scale.requires_grad_(False)
    optimizer, mask_optimizer, n_backbone, n_mask = build_trimask_optimizers(
        clip, module.text_mask_net, argparse.Namespace(lr=1e-6, mask_lr=1e-3,
                                                       weight_decay=1e-2))
    backbone_ids = {id(p) for p in optimizer.param_groups[0]['params']}
    mask_ids = {id(p) for p in mask_optimizer.param_groups[0]['params']}
    visual_mask_params = len(list(clip.mask_net.parameters()))
    text_params = len(list(module.text_mask_net.parameters()))
    assert not backbone_ids & mask_ids
    assert id(clip.logit_scale) not in backbone_ids | mask_ids
    assert n_mask == visual_mask_params + text_params, (n_mask, visual_mask_params, text_params)
    assert len(mask_ids) == n_mask
    assert n_backbone + n_mask == len(
        [p for p in list(clip.parameters()) + list(module.text_mask_net.parameters())
         if p.requires_grad])
    assert optimizer.param_groups[0]['lr'] == 1e-6
    assert mask_optimizer.param_groups[0]['lr'] == 1e-3
    assert mask_optimizer.param_groups[0]['weight_decay'] == 0.0


def test_module_construction_does_not_consume_the_ambient_rng():
    clip = _clip()
    _dist()
    torch.manual_seed(99)
    expected = torch.randn(6)
    torch.manual_seed(99)
    module = TriMaskTrainModule(clip, rank=0, text_mask_width=TEXT_WIDTH, text_mask_heads=8,
                                text_mask_seed=0)
    actual = torch.randn(6)
    assert torch.equal(expected, actual), 'building the text branch consumed the ambient RNG'
    # and the branch is reproducible from its seed alone
    other = TriMaskTrainModule(_clip(), rank=0, text_mask_width=TEXT_WIDTH, text_mask_heads=8,
                               text_mask_seed=0)
    for name, parameter in module.text_mask_net.named_parameters():
        assert torch.equal(parameter, other.text_mask_net.state_dict()[name]), name


def test_cli_entry_point_is_importable_and_documented():
    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'train',
                                                          'train_said_trimask.py'), '--help'],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    for flag in ('--lambda_1', '--lambda_2', '--lambda_3', '--lambda_sparse_i',
                 '--text_mask_width', '--text_mask_seed', '--init_state', '--output_dir',
                 '--max_steps', '--save_completed_steps', '--log_every'):
        assert flag in result.stdout, flag
    assert OBJECTIVE == 'smartclip_trimask'
