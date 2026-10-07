"""Joint parent/child V/G feasible-region regularization on soft probabilities."""
import math

import torch

EDGES = ('Dall_F', 'D3_Dall')


def validate_regions(regions):
    assert isinstance(regions, dict) and set(regions) == set(EDGES)
    output = {}
    for edge in EDGES:
        band = regions[edge]
        assert set(band) == {'eps_v', 'gamma_low', 'gamma_high'}
        values = {key: float(value) for key, value in band.items()}
        assert all(math.isfinite(value) for value in values.values())
        assert 0 <= values['eps_v'] <= 1
        assert 0 < values['gamma_low'] < values['gamma_high'] <= 1
        output[edge] = values
    return output


def vg_ratios(child, parent):
    # Both numerators AND denominators retain both endpoints' autograd paths.
    violation = torch.relu(child - parent).sum(-1) / (child.sum(-1) + 1e-6)
    gap = torch.relu(parent - child).sum(-1) / (parent.sum(-1) + 1e-6)
    return violation, gap


def joint_vg_edge(child, parent, region):
    violation, gap = vg_ratios(child, parent)
    upward = torch.relu(violation - region['eps_v']).square()
    lower = torch.relu(region['gamma_low'] - gap).square()
    upper = torch.relu(gap - region['gamma_high']).square()
    v_ok = violation <= region['eps_v']
    g_ok = (gap >= region['gamma_low']) & (gap <= region['gamma_high'])
    return dict(V=violation, G=gap, V_penalty=upward, G_lower_penalty=lower,
                G_upper_penalty=upper, V_legal=v_ok.float(), G_legal=g_ok.float(),
                zero_edge=(v_ok & g_ok).float())


def joint_vg_regularizer(omega_f, pf, pd, p3, valid, regions, world, count):
    """DDP world/count sums over valid pairs; only F has global sparsity."""
    means = {}
    edge_total = omega_f.new_zeros(())
    for name, child, parent in [('Dall_F', pd, pf), ('D3_Dall', p3, pd)]:
        edge = joint_vg_edge(child, parent, regions[name])
        local = {key: world / count * value[valid].sum() for key, value in edge.items()}
        penalty = local['V_penalty'] + local['G_lower_penalty'] + local['G_upper_penalty']
        edge_total = edge_total + penalty
        means.update({name + '_' + key: value for key, value in local.items()})
        means[name + '_edge_loss'] = penalty
    for name, probability in [('F', pf), ('Dall', pd), ('D3', p3)]:
        means[name + '_mean_soft_support'] = world / count * probability.mean(-1)[valid].sum()
    total = omega_f / 3 + edge_total
    means.update(Omega_F=omega_f, total_vg_edges=edge_total, total_nested_regularizer=total)
    return total, means
