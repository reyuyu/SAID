"""Numerical acceptance for the JointMask scoring performance patch."""
import copy

import torch

from model.nested_semantic_mask import (JointMaskAdapter, _pair_score_block,
                                        pair_scores)


def dynamic_reference(z, t, image_condition, base_logits, text_condition,
                      adapter, active_pairs):
    from model.nested_semantic_mask import pair_mask
    masks, probabilities, delta = pair_mask(
        base_logits, image_condition, text_condition, adapter)
    scores = 100 * (torch.nn.functional.normalize(
        z[:, None].float() * masks, dim=-1, eps=1e-6) *
        torch.nn.functional.normalize(t.float(), dim=-1, eps=1e-6)[None]).sum(-1)
    selected = (probabilities.detach() >= .5)[active_pairs]
    selected_delta = delta.detach()[active_pairs]
    if selected.numel():
        summary = torch.stack((selected.float().sum(),
                               selected.new_tensor(selected.numel(), dtype=torch.float32),
                               selected.all(-1).float().sum(),
                               (~selected).all(-1).float().sum(),
                               selected_delta.abs().sum(),
                               selected_delta.new_tensor(selected_delta.numel(),
                                                         dtype=torch.float32)))
    else:
        summary = scores.detach().new_zeros(6)
    return scores, summary


def test_fixed_shape_statistics_match_dynamic_boolean_selection():
    torch.manual_seed(31)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    adapter.output.weight.data.normal_(std=.2)
    args = (torch.randn(5, 8), torch.randn(7, 8), torch.randn(5, 4),
            torch.randn(7, 8), torch.randn(7, 4), adapter)
    for active in (torch.rand(5, 7) > .35, torch.zeros(5, 7, dtype=torch.bool),
                   torch.ones(5, 7, dtype=torch.bool)):
        old_scores, old_summary = dynamic_reference(*args, active)
        new_scores, new_summary = _pair_score_block(*args, active)
        torch.testing.assert_close(new_scores, old_scores, atol=0, rtol=0)
        torch.testing.assert_close(new_summary[:4], old_summary[:4], atol=0, rtol=0)
        torch.testing.assert_close(new_summary[4:], old_summary[4:], atol=2e-6, rtol=2e-7)


def run_variant(state, image_chunk, text_chunk, checkpoint_blocks, collect_stats):
    tensors = [value.detach().clone().requires_grad_(True) for value in state[:5]]
    adapter = copy.deepcopy(state[5])
    scores, summary = pair_scores(*tensors, adapter, image_chunk, text_chunk,
                                  state[6], state[7], checkpoint_blocks,
                                  collect_stats=collect_stats)
    loss = scores.square().mean()
    loss.backward()
    gradients = [value.grad.detach().clone() for value in tensors]
    gradients.extend(parameter.grad.detach().clone() if parameter.grad is not None else None
                     for parameter in adapter.parameters())
    optimizer = torch.optim.SGD([*tensors, *adapter.parameters()], lr=1e-4)
    optimizer.step()
    updated = [value.detach().clone() for value in tensors]
    updated.extend(parameter.detach().clone() for parameter in adapter.parameters())
    return loss.detach(), scores.detach(), summary, gradients, updated


def test_chunks_checkpoint_and_discarded_statistics_preserve_math():
    torch.manual_seed(37)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    adapter.output.weight.data.normal_(std=.15)
    row_valid = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
    column_valid = torch.tensor([1, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
    state = (torch.randn(5, 8), torch.randn(7, 8), torch.randn(5, 4),
             torch.randn(7, 8), torch.randn(7, 4), adapter, row_valid, column_valid)
    reference = run_variant(state, 2, 3, False, True)
    for image_chunk, text_chunk, checkpoint_blocks, collect_stats in (
            (4, 4, False, True), (8, 8, False, True),
            (2, 3, True, True), (8, 8, True, True),
            (8, 8, False, False), (8, 8, True, False)):
        candidate = run_variant(state, image_chunk, text_chunk,
                                checkpoint_blocks, collect_stats)
        torch.testing.assert_close(candidate[0], reference[0], atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(candidate[1], reference[1], atol=2e-5, rtol=2e-6)
        if collect_stats:
            torch.testing.assert_close(candidate[2], reference[2], atol=2e-6, rtol=2e-7)
        else:
            assert candidate[2] is None
        for actual, expected in zip(candidate[3], reference[3]):
            assert (actual is None) == (expected is None)
            if actual is not None:
                torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-5)
        for actual, expected in zip(candidate[4], reference[4]):
            torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-6)


def test_partial_tail_tiles_preserve_scores_and_statistics():
    """Exercise production-sized candidate geometry with incomplete final tiles."""
    torch.manual_seed(41)
    adapter = JointMaskAdapter(8, 8, 4, 8)
    adapter.output.weight.data.normal_(std=.1)
    images, texts = 180, 720
    args = (torch.randn(images, 8), torch.randn(texts, 8),
            torch.randn(images, 4), torch.randn(texts, 8),
            torch.randn(texts, 4), adapter)
    row_valid = torch.rand(images) > .2
    column_valid = torch.rand(texts) > .2
    reference = pair_scores(*args, 32, 64, row_valid, column_valid, False)
    candidate = pair_scores(*args, 128, 128, row_valid, column_valid, False)
    torch.testing.assert_close(candidate[0], reference[0], atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(candidate[1], reference[1], atol=2e-5, rtol=2e-6)
