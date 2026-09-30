"""Focused numerical and integration tests for VCP-Mask and TI-noInc."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from model.nested_semantic_mask import NestedSemanticMask
from model.nested_vcp_mask import (
    NestedVCPMask,
    VCPQueryAdapter,
    vcp_pair_mask,
    vcp_pair_scores,
    vcp_positive_masks,
)
from tests.test_nested_jointmask import TinyJointCLIP
from train.train_nested_semantic_mask import build_optimizer


def transformed(mask_net, hidden):
    return mask_net.resblocks(hidden.permute(1, 0, 2)).permute(1, 0, 2)


def test_u_zero_exactly_matches_original_attention_pool_for_any_image_hidden():
    torch.manual_seed(401)
    clip = TinyJointCLIP()
    adapter = VCPQueryAdapter(8, 4, 8)
    hidden = torch.randn(3, 7, 8)
    image_hidden = torch.randn(3, 8)
    z = transformed(clip.mask_net, hidden)
    pool = clip.mask_net.attn_pool.attention
    queries = adapter(image_hidden, pool.weight.squeeze(0))
    _, probabilities = vcp_positive_masks(queries, z, pool.bias.squeeze(0))
    expected = torch.sigmoid(clip.mask_net(hidden))
    torch.testing.assert_close(probabilities, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(queries, pool.weight.expand_as(queries), atol=0, rtol=0)


def test_nonzero_u_changes_pooling_and_visual_hidden_is_detached():
    torch.manual_seed(403)
    adapter = VCPQueryAdapter(8, 4, 8)
    with torch.no_grad():
        adapter.output.weight.normal_(std=.2)
    pool_weight = nn.Parameter(torch.randn(8))
    hidden = torch.randn(2, 8, requires_grad=True)
    queries = adapter(hidden, pool_weight)
    assert not torch.equal(queries[0], queries[1])
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    masks, probabilities = vcp_positive_masks(queries, tokens, torch.tensor(0.))
    assert not torch.equal(probabilities[0], probabilities[1])
    masks.sum().backward()
    assert hidden.grad is None
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert pool_weight.grad is not None and pool_weight.grad.abs().sum() > 0
    assert adapter.output.weight.grad is not None and adapter.output.weight.grad.abs().sum() > 0
    assert adapter.visual.weight.grad is not None and adapter.visual.weight.grad.abs().sum() > 0


def test_labels_never_enter_vcp_masks_or_scores():
    torch.manual_seed(409)
    images = torch.randn(4, 8)
    texts = torch.randn(4, 8)
    queries = torch.randn(4, 8)
    tokens = torch.randn(4, 6, 8)
    scores, _ = vcp_pair_scores(images, texts, queries, tokens, torch.tensor(.2), 2, 3)
    mask_before, probability_before = vcp_pair_mask(queries, tokens, torch.tensor(.2))
    first = F.cross_entropy(scores, torch.arange(4))
    second = F.cross_entropy(scores, torch.tensor([1, 0, 3, 2]))
    scores_after, _ = vcp_pair_scores(images, texts, queries, tokens, torch.tensor(.2), 3, 2)
    mask_after, probability_after = vcp_pair_mask(queries, tokens, torch.tensor(.2))
    assert first != second
    torch.testing.assert_close(scores_after, scores, atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(mask_after, mask_before, atol=0, rtol=0)
    torch.testing.assert_close(probability_after, probability_before, atol=0, rtol=0)


def test_vcp_runs_resblocks_once_per_text_view_and_registers_auxiliary_once():
    torch.manual_seed(419)
    model = NestedVCPMask(TinyJointCLIP(), checkpoint_encoders=False,
                          image_chunk=2, text_chunk=2)
    calls = 0
    original = model.clip.mask_net.resblocks.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.clip.mask_net.resblocks.forward = counted
    images = torch.randn(4, 8)
    tokens = [torch.randint(0, 31, (4, 6)) for _ in range(3)]
    valid = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    loss, logs = model(images, *tokens, valid, 37)
    assert calls == 3
    assert torch.isfinite(loss) and logs['condition_mode'] == 'vcp_mask'
    loss.backward()
    assert model.vcp_query.output.weight.grad.abs().sum() > 0
    optimizer = build_optimizer(model)
    assert [group['name'] for group in optimizer.param_groups] == [
        'backbone', 'shared_mask', 'joint_adapter']
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    assert len(ids) == len(set(ids))


def test_ti_noinc_differs_only_by_weighted_inclusion_term():
    torch.manual_seed(421)
    a3 = NestedSemanticMask(TinyJointCLIP(), arm='A3', checkpoint_encoders=False,
                            condition_mode='joint_image', image_chunk=2, text_chunk=2,
                            checkpoint_pair_blocks=False)
    with torch.no_grad():
        a3.joint_adapter.output.weight.normal_(std=.08)
    noinc = copy.deepcopy(a3)
    noinc.arm = 'A2'
    images = torch.randn(4, 8)
    tokens = [torch.randint(0, 31, (4, 6)) for _ in range(3)]
    valid = torch.ones(4, dtype=torch.bool)
    completed = 100
    loss_a3, logs_a3 = a3(images, *tokens, valid, completed)
    loss_noinc, logs_noinc = noinc(images, *tokens, valid, completed)
    expected = logs_a3['inc_weight'] * logs_a3['inc']
    torch.testing.assert_close(loss_a3 - loss_noinc, expected, atol=2e-5, rtol=2e-6)
    assert logs_noinc['inc_weight'] == 0
    for key in logs_a3:
        if key not in ('loss', 'inc_weight'):
            left, right = logs_a3[key], logs_noinc[key]
            if torch.is_tensor(left):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            else:
                assert left == right


def test_vcp_adamw_step_preserves_named_gradient_none_states():
    torch.manual_seed(431)
    left = NestedVCPMask(TinyJointCLIP(), checkpoint_encoders=False,
                         image_chunk=2, text_chunk=3)
    right = copy.deepcopy(left)
    with torch.no_grad():
        left.vcp_query.output.weight.normal_(std=.03)
        right.load_state_dict(left.state_dict())
    images = torch.randn(4, 8)
    tokens = [torch.randint(0, 31, (4, 6)) for _ in range(3)]
    valid = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    left_loss, _ = left(images, *tokens, valid, 51)
    right_loss, _ = right(images, *tokens, valid, 51)
    left_loss.backward(); right_loss.backward()
    left_named, right_named = dict(left.named_parameters()), dict(right.named_parameters())
    assert left_named.keys() == right_named.keys()
    for name in left_named:
        assert (left_named[name].grad is None) == (right_named[name].grad is None), name
        if left_named[name].grad is not None:
            torch.testing.assert_close(left_named[name].grad, right_named[name].grad,
                                       atol=0, rtol=0, msg=name)
    left_opt, right_opt = build_optimizer(left), build_optimizer(right)
    left_opt.step(); right_opt.step()
    for name in left_named:
        torch.testing.assert_close(left_named[name], right_named[name], atol=0, rtol=0, msg=name)
