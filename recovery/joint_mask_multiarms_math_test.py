"""CPU-only mathematical and dependency tests for the Joint Mask arms."""
import math

import torch
from torch import nn

from model.nested_fusion_mask import FusionBranch


def main():
    torch.manual_seed(7)
    width, batch, tokens = 8, 3, 5
    branch = FusionBranch(nn.ModuleList([]), width, width, 'balanced_stack', 1)
    with torch.no_grad():
        branch.gate.weight.copy_(torch.randn_like(branch.gate.weight) * 0.2)
    text = torch.randn(batch, width, requires_grad=True)
    visual = torch.randn(batch, width, requires_grad=True)
    wt, wv = branch.gate.weight[:, :width], branch.gate.weight[:, width:]
    a, b = nn.functional.linear(text, wt), nn.functional.linear(visual, wv)
    old_gate = torch.sigmoid(a + b)
    old_logits = old_gate * text + (1 - old_gate) * visual
    matched = branch.joint_mask_logits(visual, text, paired=True)
    assert torch.allclose(matched, old_logits, atol=1e-7, rtol=1e-7)

    expected = {
        'J1': torch.sigmoid(a + b + math.log(3.0)) * text +
              (1 - torch.sigmoid(a + b + math.log(3.0))) * visual,
        'J2': a + 0.5 * b,
        'J3': a + 0.5 * torch.tanh(b),
        'J4': a + 0.5 * b - 0.25 * (a - b).abs(),
    }
    for mode, reference in expected.items():
        branch.joint_mask_mode = mode
        actual = branch.joint_mask_logits(visual, text, paired=True)
        assert torch.allclose(actual, reference, atol=1e-7, rtol=1e-7), mode
        grad_text, grad_visual = torch.autograd.grad(actual.square().sum(), (text, visual), retain_graph=True)
        assert torch.isfinite(grad_text).all() and torch.isfinite(grad_visual).all()
        assert grad_text.abs().sum() > 0 and grad_visual.abs().sum() > 0, mode
        replaced = branch.joint_mask_logits(visual.roll(1, 0), text, paired=True)
        assert (actual - replaced).abs().sum() > 0, mode

    # Unpaired score construction must preserve text-by-image broadcasting.
    branch.joint_mask_mode = 'J2'
    unpaired = branch.joint_mask_logits(visual, text, paired=False)
    assert unpaired.shape == (batch, batch, width)
    assert torch.allclose(unpaired[0, 1],
                          nn.functional.linear(text[0], wt) + 0.5 * nn.functional.linear(visual[1], wv),
                          atol=1e-7, rtol=1e-7)
    print('joint mask CPU math/dependency tests: PASS')


if __name__ == '__main__':
    main()
