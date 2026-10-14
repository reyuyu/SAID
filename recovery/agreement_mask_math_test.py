"""CPU mathematical and gradient tests for the Agreement-Mask rule."""
import torch
from torch import nn

from model.nested_fusion_mask import FusionBranch, hard_st


def main():
    torch.manual_seed(13)
    width = 16
    branch = FusionBranch(nn.ModuleList([]), width, width, 'balanced_stack', 196,
                          joint_mask_mode='agreement')
    visual = torch.randn(4, width, requires_grad=True)
    text = torch.randn(4, width, requires_grad=True)
    raw_v = torch.tanh(visual / (visual.square().mean(-1, keepdim=True).add(1e-6).sqrt()))
    raw_t = torch.tanh(text / (text.square().mean(-1, keepdim=True).add(1e-6).sqrt()))
    expected_raw = raw_v * raw_t + 0.2 * torch.tanh(branch.agreement_theta)
    actual_scaled = branch.joint_mask_logits(visual, text, paired=True)
    torch.testing.assert_close(actual_scaled, 5 * expected_raw, atol=1e-6, rtol=1e-6)
    probability = actual_scaled.sigmoid()
    hard = hard_st(probability)
    assert torch.equal(hard, (expected_raw > 0).float())
    assert torch.all(branch.agreement_theta == 0)
    raw_loss = hard.sum() + probability.sum()
    raw_loss.backward()
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert text.grad is not None and text.grad.abs().sum() > 0
    assert branch.agreement_theta.grad is not None and branch.agreement_theta.grad.abs().sum() > 0
    unpaired = branch.joint_mask_logits(visual, text, paired=False)
    assert unpaired.shape == (4, 4, width)
    changed = branch.joint_mask_logits(visual.roll(1, 0), text.detach(), paired=True)
    assert not torch.equal(actual_scaled.detach(), changed.detach())
    with torch.no_grad():
        bounded = 0.2 * torch.tanh(branch.agreement_theta)
        assert float(bounded.abs().max()) < 0.2
    print('agreement-mask CPU math/dependency tests: PASS')


if __name__ == '__main__':
    main()
