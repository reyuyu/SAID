"""Correctness tests for CLS/Patch x All/Text input-level joint masks."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.model_longclip import MaskNetwork, VisionTransformer
from model.nested_joint_input import (NestedJointInputMask, VisualInputProjection,
                                      joint_logits, joint_pair_scores,
                                      positive_joint_masks)


class TinyVisual(nn.Module):
    def __init__(self, width=8, output=8, patches=3):
        super().__init__()
        self.hidden = nn.Linear(8, width)
        self.patch_offsets = nn.Parameter(torch.randn(patches, width) * .1)
        self.proj = nn.Parameter(torch.randn(width, output) * .1)

    def forward(self, images, return_joint_tokens=False):
        cls = self.hidden(images)
        patches = cls[:, None, :] + self.patch_offsets[None]
        z = cls @ self.proj
        return (z, cls, patches) if return_joint_tokens else z


class TinyJointInputCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.token_embedding = nn.Embedding(31, 8)
        self.text_projection = nn.Parameter(torch.randn(8, 8) * .1)
        self.mask_net = MaskNetwork(8, layers=1, heads=2)

    def encode_image(self, images):
        return self.visual(images)

    def encode_image_with_joint_tokens(self, images):
        return self.visual(images, return_joint_tokens=True)

    def encode_text(self, tokens, return_full=False):
        hidden = self.token_embedding(tokens)
        eot = tokens.argmax(-1)
        pooled = hidden[torch.arange(len(tokens)), eot]
        embedding = pooled @ self.text_projection
        return (embedding, hidden) if return_full else embedding


def test_visual_joint_interface_is_native_exact_and_state_compatible():
    torch.manual_seed(2)
    visual = VisionTransformer(8, 4, 8, 1, 1, 4).eval()
    images = torch.randn(2, 3, 8, 8)
    before = copy.deepcopy(visual.state_dict())
    native = visual(images)
    z, cls, patch = visual(images, return_joint_tokens=True)
    torch.testing.assert_close(z, native, atol=0, rtol=0)
    torch.testing.assert_close(z, cls @ visual.proj, atol=0, rtol=0)
    assert cls.shape == (2, 8) and patch.shape == (2, 4, 8)
    reloaded = VisionTransformer(8, 4, 8, 1, 1, 4).eval()
    reloaded.load_state_dict(before, strict=True)
    assert tuple(visual.state_dict()) == tuple(before)


def test_joint_lengths_and_readout_only_changes_pool_scope():
    torch.manual_seed(3)
    mask = MaskNetwork(8, layers=1, heads=2)
    text = torch.randn(2, 6, 8)
    for visual_count in (1, 3):
        visual = torch.randn(2, visual_count, 8)
        all_logits, all_weights = joint_logits(mask, visual, text, "all", return_pool_weights=True)
        text_logits, text_weights = joint_logits(mask, visual, text, "text", return_pool_weights=True)
        assert all_logits.shape == text_logits.shape == (2, 8)
        assert all_weights.shape == (2, visual_count + 6, 1)
        assert text_weights.shape == (2, 6, 1)
        torch.testing.assert_close(all_weights.sum(1), torch.ones(2, 1))
        torch.testing.assert_close(text_weights.sum(1), torch.ones(2, 1))


def test_text_readout_uses_post_interaction_text_outputs():
    torch.manual_seed(5)
    mask = MaskNetwork(8, layers=1, heads=2)
    text = torch.randn(1, 6, 8).expand(2, -1, -1).clone()
    visual = torch.stack((torch.arange(8, dtype=torch.float32)[None],
                          torch.tensor([[0., 2., -1., 4., 3., -2., 1., 5.]])))
    logits = joint_logits(mask, visual, text, "text")
    assert not torch.equal(logits[0], logits[1])


def test_pair_diagonal_matches_positive_definition_and_labels_do_not_enter_masks():
    torch.manual_seed(7)
    n, width = 4, 8
    z, text = torch.randn(n, width), torch.randn(n, width)
    visual, hidden = torch.randn(n, 1, width), torch.randn(n, 6, width)
    mask = MaskNetwork(width, layers=1, heads=2)
    scores, _ = joint_pair_scores(z, text, visual, hidden, mask, "text", 3,
                                  checkpoint_block=False)
    positive, probabilities = positive_joint_masks(visual, hidden, mask, "text", False)
    expected = 100 * (F.normalize(z * positive, dim=-1) * F.normalize(text, dim=-1)).sum(-1)
    torch.testing.assert_close(scores.diag(), expected, atol=1e-5, rtol=1e-5)
    first = F.cross_entropy(scores, torch.arange(n))
    second = F.cross_entropy(scores, torch.tensor([1, 0, 3, 2]))
    assert first != second
    again = positive_joint_masks(visual, hidden, mask, "text", False)[1]
    torch.testing.assert_close(probabilities, again, atol=0, rtol=0)


def test_hidden_detach_projection_and_mask_gradients():
    torch.manual_seed(11)
    projection = VisualInputProjection(8, 8)
    mask = MaskNetwork(8, layers=1, heads=2)
    raw = torch.randn(2, 1, 8, requires_grad=True)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    visual = projection(raw)
    logits = joint_logits(mask, visual, hidden.detach(), "all")
    logits.sum().backward()
    assert raw.grad is None and hidden.grad is None
    assert projection.projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in mask.parameters())


def test_candidate_initialization_is_shared_and_rng_replayed():
    random.seed(13); np.random.seed(13); torch.manual_seed(13)
    clip = TinyJointInputCLIP()
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    modules = []
    for visual, readout in (("cls", "all"), ("cls", "text"),
                            ("patch", "all"), ("patch", "text")):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        modules.append(NestedJointInputMask(copy.deepcopy(clip), visual, readout,
                                            checkpoint_encoders=False))
    for other in modules[1:]:
        torch.testing.assert_close(
            modules[0].visual_input_projection.projection.weight,
            other.visual_input_projection.projection.weight, atol=0, rtol=0)
        left = modules[0].clip.mask_net.state_dict()
        right = other.clip.mask_net.state_dict()
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_tiny_full_forward_all_candidates_and_fallback():
    torch.manual_seed(17)
    images = torch.randn(3, 8)
    tokens = [torch.randint(0, 31, (3, 6)) for _ in range(3)]
    for visual, readout in (("cls", "all"), ("cls", "text"),
                            ("patch", "all"), ("patch", "text")):
        model = NestedJointInputMask(TinyJointInputCLIP(), visual, readout,
                                     pair_microbatch=2, checkpoint_pair_block=False,
                                     checkpoint_encoders=False)
        valid = torch.tensor([1, 1, 0], dtype=torch.bool)
        loss, logs = model(images, *tokens, valid, 3)
        assert torch.isfinite(loss) and logs["visual_mode"] == visual
        assert logs["readout_mode"] == readout
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.grad is not None)
        model.zero_grad(set_to_none=True)
        fallback, fallback_logs = model(images, *tokens, torch.zeros(3, dtype=torch.bool), 3)
        assert torch.isfinite(fallback) and fallback_logs["O_candidates"] == 0


def test_optimizer_groups_and_schedule():
    from train.train_nested_joint_input import build_optimizer, learning_rates
    torch.manual_seed(19)
    model = NestedJointInputMask(TinyJointInputCLIP(), "cls", "text",
                                 checkpoint_encoders=False)
    optimizer = build_optimizer(model)
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone", "shared_mask", "visual_input_projection"]
    assert not optimizer.state
    assert learning_rates(0, 3651) == (5e-9, 1e-3, 1e-4)
