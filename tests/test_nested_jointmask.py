"""Focused tests for text-only and pairwise image-conditioned NEST masks."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.model_longclip import MaskNetwork, VisionTransformer
from model.nested_semantic_mask import (JointMaskAdapter, NestedSemanticMask,
                                        deterministic_shuffle_indices, masked_scores,
                                        pair_scores, pair_view_terms, positive_masks)
from train.train_nested_semantic_mask import build_optimizer, learning_rates


class TinyVisual(nn.Module):
    def __init__(self, hidden=8, output=8):
        super().__init__()
        self.hidden = nn.Linear(8, hidden)
        self.proj = nn.Parameter(torch.randn(hidden, output) * .1)

    def forward(self, images, return_hidden=False):
        hidden = self.hidden(images)
        embedding = hidden @ self.proj
        return (embedding, hidden) if return_hidden else embedding


class TinyJointCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.token_embedding = nn.Embedding(31, 8)
        self.text_projection = nn.Parameter(torch.randn(8, 8) * .1)
        self.mask_net = MaskNetwork(8, layers=1, heads=2)

    def encode_image(self, images, return_hidden=False):
        return self.visual(images, return_hidden=return_hidden)

    def encode_text(self, tokens, return_full=False):
        hidden = self.token_embedding(tokens)
        eot = tokens.argmax(-1)
        pooled = hidden[torch.arange(len(tokens)), eot]
        embedding = pooled @ self.text_projection
        return (embedding, hidden) if return_full else embedding


def test_visual_hidden_interface_preserves_native_output_and_state_dict():
    torch.manual_seed(2)
    visual = VisionTransformer(8, 4, 8, 1, 1, 4).eval()
    images = torch.randn(2, 3, 8, 8)
    before = tuple(visual.state_dict())
    native = visual(images)
    projected, hidden = visual(images, return_hidden=True)
    torch.testing.assert_close(projected, native, atol=0, rtol=0)
    torch.testing.assert_close(projected, hidden @ visual.proj, atol=0, rtol=0)
    assert hidden.shape == (2, 8) and tuple(visual.state_dict()) == before


def test_zero_delta_pair_scores_and_regularizers_match_original():
    torch.manual_seed(3)
    z = torch.randn(5, 8, requires_grad=True)
    text = torch.randn(5, 8, requires_grad=True)
    logits = torch.randn(5, 8, requires_grad=True)
    probability = torch.sigmoid(logits)
    mask = (probability >= .5).float() - probability.detach() + probability
    zeros = torch.zeros(5, 4)
    paired, _ = pair_scores(z, text, zeros, logits, zeros, None, 2, 3)
    torch.testing.assert_close(paired, masked_scores(z, text, mask, 2), atol=8e-5, rtol=3e-6)
    valid = torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool)
    old_align, old_sparse, _ = __import__(
        'model.nested_semantic_mask', fromlist=['view_terms']).view_terms(
            z, text, mask, valid, valid)
    new_align, new_sparse, positive, soft, _ = pair_view_terms(
        z, text, logits, zeros, zeros, valid, valid, None,
        image_chunk=2, text_chunk=3)
    torch.testing.assert_close(new_align, old_align, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(new_sparse, old_sparse, atol=0, rtol=0)
    torch.testing.assert_close(positive, mask, atol=0, rtol=0)
    torch.testing.assert_close(soft, probability, atol=0, rtol=0)


def test_labels_cannot_change_pair_masks_or_scores():
    torch.manual_seed(5)
    z, text = torch.randn(4, 8), torch.randn(4, 8)
    logits, a, b = torch.randn(4, 8), torch.randn(4, 4), torch.randn(4, 4)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    nn.init.normal_(adapter.output.weight)
    scores, _ = pair_scores(z, text, a, logits, b, adapter, 2, 2)
    masks, _, _ = positive_masks(logits, a, b, adapter)
    first = F.cross_entropy(scores, torch.arange(4))
    second = F.cross_entropy(scores, torch.tensor([1, 0, 3, 2]))
    scores_again, _ = pair_scores(z, text, a, logits, b, adapter, 3, 3)
    masks_again, _, _ = positive_masks(logits, a, b, adapter)
    assert first != second
    torch.testing.assert_close(scores_again, scores, atol=8e-5, rtol=3e-6)
    torch.testing.assert_close(masks_again, masks, atol=0, rtol=0)


def test_image_condition_changes_mask_only_after_output_becomes_nonzero():
    torch.manual_seed(7)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    logits, b = torch.randn(1, 8), torch.randn(1, 4)
    a = torch.stack((torch.ones(4), -torch.ones(4)))
    initial, _, delta = positive_masks(logits.expand(2, -1), a, b.expand(2, -1), adapter)
    assert delta.count_nonzero() == 0
    torch.testing.assert_close(initial[0], initial[1], atol=0, rtol=0)
    with torch.no_grad():
        adapter.output.weight.copy_(torch.eye(8, 4))
    changed, probabilities, delta = positive_masks(logits.expand(2, -1), a, b.expand(2, -1), adapter)
    assert not torch.equal(probabilities[0], probabilities[1])
    assert not torch.equal(delta[0], delta[1])
    assert changed.shape == (2, 8)


def test_condition_detach_and_adapter_gradient_schedule():
    torch.manual_seed(11)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    image_hidden = torch.randn(3, 8, requires_grad=True)
    text_hidden = torch.randn(3, 8, requires_grad=True)
    a, b = adapter.image_condition(image_hidden), adapter.text_condition(text_hidden)
    logits = torch.randn(3, 8, requires_grad=True)
    masks, _, _ = positive_masks(logits, a, b, adapter)
    masks.sum().backward()
    assert image_hidden.grad is None and text_hidden.grad is None
    assert adapter.output.weight.grad.abs().sum() > 0
    assert adapter.image.weight.grad.abs().sum() == 0
    assert adapter.text.weight.grad.abs().sum() == 0
    adapter.zero_grad(set_to_none=True)
    with torch.no_grad():
        adapter.output.weight.normal_(std=.1)
    positive_masks(logits.detach(), adapter.image_condition(image_hidden),
                   adapter.text_condition(text_hidden), adapter)[0].sum().backward()
    assert adapter.image.weight.grad.abs().sum() > 0
    assert adapter.text.weight.grad.abs().sum() > 0


def test_shuffle_is_reproducible_fixed_point_free_and_rng_neutral():
    random.seed(13); np.random.seed(13); torch.manual_seed(13)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    first = deterministic_shuffle_indices(17, 0, 31)
    second = deterministic_shuffle_indices(17, 0, 31)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert torch.equal(first, second)
    assert torch.equal(first.sort().values, torch.arange(17))
    assert not torch.any(first == torch.arange(17))
    assert before[0] == after[0] and before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_adapter_optimizer_groups_initialization_and_rng_replay():
    torch.manual_seed(17)
    base = TinyJointCLIP()
    state = torch.get_rng_state().clone()
    first = NestedSemanticMask(copy.deepcopy(base), checkpoint_encoders=False,
                               condition_mode='joint_image')
    torch.set_rng_state(state)
    second = NestedSemanticMask(copy.deepcopy(base), checkpoint_encoders=False,
                                condition_mode='joint_shuffled_image')
    for left, right in zip(first.joint_adapter.state_dict().values(),
                           second.joint_adapter.state_dict().values()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert first.joint_adapter.output.weight.count_nonzero() == 0
    assert first.joint_adapter.image.weight.count_nonzero() > 0
    optimizer = build_optimizer(first)
    assert [group['name'] for group in optimizer.param_groups] == [
        'backbone', 'shared_mask', 'joint_adapter']
    assert not optimizer.state
    assert learning_rates(0, 3651, True) == (5e-9, 1e-3, 1e-4)


def test_tiny_full_forward_runs_all_modes_without_native_loss():
    torch.manual_seed(23)
    images, tokens = torch.randn(4, 8), [torch.randint(0, 31, (4, 6)) for _ in range(3)]
    valid = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    for mode in ('text_only', 'joint_image', 'joint_shuffled_image'):
        model = NestedSemanticMask(TinyJointCLIP(), checkpoint_encoders=False,
                                   condition_mode=mode, image_chunk=2, text_chunk=2,
                                   checkpoint_pair_blocks=False)
        loss, logs = model(images, *tokens, valid, 3)
        assert torch.isfinite(loss) and logs['condition_mode'] == mode
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.grad is not None)
