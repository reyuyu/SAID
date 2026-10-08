"""Hard-ST illegal-support surcharge; joint gradients and read-only telemetry."""
import math
import torch


def hierarchy_weight(completed):
    assert isinstance(completed, int) and completed >= 0
    return min(1., completed / 200.)


def hard_violation(child, parent, detach_child=False):
    # ReLU, not child*(1-parent): no surrogate gradient at equal zero masks.
    # Both Hard-ST inputs retain their original graph.
    return torch.relu((child.detach() if detach_child else child) - parent).mean(-1)


def hns_terms(mf, md, m3, valid, completed, world, valid_count, beta=(2., 2.), detach_child=False):
    assert len(beta) == 2 and all(math.isfinite(v) and v >= 0 for v in beta)
    df = world / valid_count * hard_violation(md, mf, detach_child)[valid].sum()
    d3 = world / valid_count * hard_violation(m3, md, detach_child)[valid].sum()
    weight = hierarchy_weight(completed)
    return weight * (beta[0] * df + beta[1] * d3) / 3, dict(V_DF=df, V_3D=d3, lambda_h=weight)


@torch.no_grad()
def hard_telemetry(mf, md, m3, valid):
    """Local sums; reduce once, then normalize with global valid_count."""
    f, d, third = (m[valid] >= .5 for m in (mf, md, m3))
    output = {}
    width = mf.shape[-1]
    for name, child, parent in [('DF', d, f), ('3D', third, d)]:
        child_only = child & ~parent
        parent_only = parent & ~child
        intersection = child & parent
        union = child | parent
        output.update({name + '_violation_count': child_only.sum(dtype=torch.float32),
            name + '_child_only_count': child_only.sum(dtype=torch.float32),
            name + '_parent_only_count': parent_only.sum(dtype=torch.float32),
            name + '_intersection_count': intersection.sum(dtype=torch.float32),
            name + '_IoU_sum': (intersection.sum(-1) / union.sum(-1).clamp_min(1)).sum(),
            name + '_exact_equal_samples': (child == parent).all(-1).sum(dtype=torch.float32),
            name + '_equal_coordinates': (child == parent).sum(dtype=torch.float32)})
    for name, mask in [('F', f), ('Dall', d), ('D3', third)]:
        output[name + '_selected_count'] = mask.sum(dtype=torch.float32)
    output['triple_equal_samples'] = ((f == d) & (d == third)).all(-1).sum(dtype=torch.float32)
    output['triple_equal_coordinates'] = ((f == d) & (d == third)).sum(dtype=torch.float32)
    return output, width
