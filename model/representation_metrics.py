"""Detached batch diagnostics; never part of the training objective."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def batch_representation_gaps(z_full, z_said, text, eps=1e-8):
    # Disable autocast as well as autograd: diagnostics accumulate in float32.
    with torch.autocast(device_type=text.device.type, enabled=False):
        full, said, target = [F.normalize(x.detach().float(), dim=-1)
                              for x in (z_full, z_said, text)]
        full_gap = (1 - (full * target).sum(-1)).mean()
        said_gap = (1 - (said * target).sum(-1)).mean()
        gain = full_gap - said_gap
        return {'pair_gap_full': full_gap, 'pair_gap_said': said_gap,
                'balancing_gain': gain,
                'relative_balancing_gain': gain / full_gap.clamp_min(eps)}
