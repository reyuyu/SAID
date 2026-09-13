"""Production DDP total objective versus an independent full-batch reference.

The independent reference builds the complete global batch in one graph and never calls a collective,
so it is an outside check of both the S0 term, the suffix alignment term, the valid-positive U-gate
sparsity term and the DDP reduction that combines them.

Run with ``--lambda-suffix 10 --lambda-u-sparse 2`` to check the Dual-Mask-Full v0.1 objective; the
defaults (1.0 / 0.0) reproduce the accepted Clean v0.1 objective.
"""
import argparse
import copy
import json
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, 'train')):
    if path not in sys.path:
        sys.path.insert(0, path)
from model.dual_mask_suffix import DualMaskSuffixTrainModule
from model.said_cls_cvssl import said_mask_from_hidden
from train_dual_mask_suffix import build_optimizers

# Fixed before running these probes; absolute and relative errors stay separate.
GRAD_ATOL, GRAD_RTOL = 2e-4, 2e-5
VALUE_ATOL, VALUE_RTOL = 2e-4, 2e-5
UPDATE_ATOL, UPDATE_RTOL = 2e-6, 2e-5


class ToyClip(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.image = nn.Linear(6, 4)
        self.tokens = nn.Embedding(32, 4)
        self.text = nn.Linear(4, 4)
        self.mask_net = ToyMaskNet()

    def encode_image(self, images):
        return self.image(images.flatten(1))

    def encode_text(self, tokens, return_full=False):
        hidden = self.tokens(tokens)
        pooled = self.text(hidden.mean(dim=1))
        return (pooled, hidden) if return_full else pooled


class ToyMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor([2.0, 0.1, -0.2, 0.3]))
            self.linear.weight[0].zero_()

    def forward(self, hidden):
        return self.linear(hidden.mean(dim=1))


def independent_global_objective(module, images, prefix, suffix, valid,
                                 lambda_suffix=1.0, lambda_u_sparse=0.0):
    """No collectives: the complete global batch already lives in this graph.

    The U-gate sparsity of the reference is built from the same diagonal positive pairs the
    production tile loop extracts -- row ``i`` against candidate column ``i`` -- averaged over the
    feature dimension once and then over the global valid count ``V``.
    """
    g = module.clip.encode_image(images)
    p, hidden = module.clip.encode_text(prefix, return_full=True)
    m_s, _, _ = said_mask_from_hidden(module.clip.mask_net, hidden)
    p = p / p.norm(dim=-1, keepdim=True)
    masked = g[:, None, :] * m_s[None, :, :]
    masked = masked / masked.norm(dim=-1, keepdim=True)
    s0_scores = 100 * (masked * p[None, :, :]).sum(-1)
    labels = torch.arange(len(g), device=g.device)
    s0 = 10 * (F.cross_entropy(s0_scores, labels) + F.cross_entropy(s0_scores.T, labels)) + 2 * m_s.abs().mean()
    if int(valid.sum()) < 2:
        zero = g.sum() * 0.0
        return s0 + zero, zero, zero, zero, zero
    t = F.normalize(module.clip.encode_text(suffix).float(), dim=-1, eps=1e-6)
    g_norm = F.normalize(g.float(), dim=-1, eps=1e-6)
    rows = []
    diagonal = []
    for i in range(len(g)):
        cells = []
        for j in range(len(t)):
            x = torch.cat((g_norm[i].detach(), g_norm[i].detach() * m_s[j].detach()))
            probability = torch.sigmoid(module.suffix_gate(x))
            mask = (probability >= .5).to(probability.dtype) + (probability - probability.detach())
            if i == j:
                diagonal.append(mask)
            u = F.normalize(g_norm[i] * mask, dim=-1, eps=1e-6)
            cells.append(100 * (u * t[j]).sum())
        rows.append(torch.stack(cells))
    q = torch.stack(rows)
    selected = q[valid][:, valid]
    valid_labels = torch.arange(int(valid.sum()), device=g.device)
    i2t, t2i = F.cross_entropy(selected, valid_labels), F.cross_entropy(selected.T, valid_labels)
    diagonal = torch.stack(diagonal)
    s_u = (diagonal.abs().mean(dim=-1) * valid.float()).sum() / int(valid.sum())
    align = lambda_suffix * (i2t + t2i)
    return s0 + align + lambda_u_sparse * s_u, i2t + t2i, i2t, t2i, s_u


def error_metrics(actual, reference):
    denominator = float(reference.detach().abs().max())
    maximum = float((actual.detach() - reference.detach()).abs().max())
    return {'max_abs_error': maximum, 'relative_error': maximum / max(denominator, 1e-12),
            'relative_denominator': denominator}


def parameter_group(name):
    if name.startswith('suffix_gate.'):
        return 'suffix_gate'
    if name.startswith('clip.mask_net.'):
        return 's0_mask'
    return 'visual' if name.startswith('clip.image.') else 'text'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--backend', choices=('gloo', 'nccl'), default='gloo')
    parser.add_argument('--lambda-suffix', type=float, default=1.0)
    parser.add_argument('--lambda-u-sparse', type=float, default=0.0)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if args.backend == 'nccl' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(args.backend)
    torch.manual_seed(123)
    model = DualMaskSuffixTrainModule(ToyClip(), 'masked', rank=rank, feature_dim=4,
                                      image_chunk=2, text_chunk=3,
                                      lambda_suffix=args.lambda_suffix,
                                      lambda_u_sparse=args.lambda_u_sparse).to(device)
    # Test-only non-open gate. Production initialization is not changed.
    with torch.no_grad():
        model.suffix_gate[2].weight.normal_(0, .12)
        model.suffix_gate[2].weight[0].zero_()
        model.suffix_gate[2].bias.copy_(torch.tensor([1.0, -.25, .1, -.1], device=device))
    reference = copy.deepcopy(model)
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True,
             static_graph=False, device_ids=[device.index] if device.type == 'cuda' else None)
    opt_args = argparse.Namespace(lr=1e-6, mask_lr=1e-3, suffix_lr=1e-4, weight_decay=1e-2)
    optimizers, ref_optimizers = build_optimizers(model, opt_args), build_optimizers(reference, opt_args)
    cases = [[1,0,0,0,1,1,1,0], [0,0,0,0,1,0,1,1], [0]*8,
             [1,0,0,0,0,0,0,0], [1,0,1,0,0,1,0,1]]
    results = []
    for index, valid_values in enumerate(cases):
        torch.manual_seed(777 + index)
        images = torch.randn(8, 1, 6, device=device)
        prefix = torch.randint(1, 31, (8, 5), device=device)
        suffix = torch.randint(1, 31, (8, 5), device=device)
        valid = torch.tensor(valid_values, dtype=torch.bool, device=device)
        sl = slice(rank * 4, (rank + 1) * 4)
        for optimizer in (*optimizers, *ref_optimizers):
            optimizer.zero_grad(set_to_none=True)
        out = ddp(images[sl], prefix[sl], suffix[sl], valid[sl], torch.arange(4, device=device) + rank * 4)
        ref_total, ref_suffix, ref_i2t, ref_t2i, ref_s_u = independent_global_objective(
            reference, images, prefix, suffix, valid,
            lambda_suffix=args.lambda_suffix, lambda_u_sparse=args.lambda_u_sparse)
        total = out['loss_total'].detach().clone()
        dist.all_reduce(total); total /= world
        torch.testing.assert_close(total, ref_total.detach(), atol=VALUE_ATOL, rtol=VALUE_RTOL)
        torch.testing.assert_close(out['loss_suffix_global'], ref_suffix.detach(), atol=VALUE_ATOL, rtol=VALUE_RTOL)
        directions = torch.stack((out['loss_suffix_i2t_sum'], out['loss_suffix_t2i_sum']))
        dist.all_reduce(directions); directions /= max(int(valid.sum()), 1)
        torch.testing.assert_close(directions, torch.stack((ref_i2t, ref_t2i)).detach(), atol=VALUE_ATOL, rtol=VALUE_RTOL)
        # The locally backpropagated sparsity must be (W / V) * local_sum, so that DDP averaging
        # reproduces (1 / V) * global_sum exactly.
        if int(valid.sum()) >= 2 and args.lambda_u_sparse != 0.0:
            sparse_backward = out['loss_u_sparse_backward'].detach().clone()
            dist.all_reduce(sparse_backward); sparse_backward /= world
            torch.testing.assert_close(args.lambda_u_sparse * sparse_backward, 
                                       (args.lambda_u_sparse * ref_s_u).detach(),
                                       atol=VALUE_ATOL, rtol=VALUE_RTOL)
            torch.testing.assert_close(float(out['m_u_positive_count_local']),
                                       float(valid[sl].sum()), atol=0.0, rtol=0.0)
        sparse_error = error_metrics(args.lambda_u_sparse * out['loss_u_sparse_global'],
                                     (args.lambda_u_sparse * ref_s_u))
        out['loss_total'].backward()
        ref_total.backward()
        gradients, updates, groups = {}, {}, {}
        for (name, parameter), (ref_name, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == ref_name
            assert (parameter.grad is None) == (ref_parameter.grad is None), name
            if parameter.grad is None:
                gradients[name] = {'grad_state': 'None'}
                continue
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL,
                                       msg=lambda msg: name + '\n' + msg)
            gradients[name] = error_metrics(parameter.grad, ref_parameter.grad)
            group = groups.setdefault(parameter_group(name), {'max_abs_error': 0., 'reference_max_abs': 0.})
            group['max_abs_error'] = max(group['max_abs_error'], gradients[name]['max_abs_error'])
            group['reference_max_abs'] = max(group['reference_max_abs'], gradients[name]['relative_denominator'])
        for group in groups.values():
            group['relative_error'] = group['max_abs_error'] / max(group['reference_max_abs'], 1e-12)
        if int(valid.sum()) >= 2:
            assert model.suffix_gate[0].weight.grad.abs().max() > 0
        for optimizer in (*optimizers, *ref_optimizers):
            optimizer.step()
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(parameter, ref_parameter, atol=UPDATE_ATOL, rtol=UPDATE_RTOL,
                                       msg=lambda msg: name + '\n' + msg)
            updates[name] = error_metrics(parameter, ref_parameter)
        vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        root_vector = vector.clone()
        dist.broadcast(root_vector, src=0)
        torch.testing.assert_close(vector, root_vector, atol=UPDATE_ATOL, rtol=UPDATE_RTOL)
        results.append({'case': index, 'global_valid': int(valid.sum()),
            'valid_per_rank': [sum(valid_values[:4]), sum(valid_values[4:])],
            'loss_total_global': float(total), 'loss_suffix_global': float(out['loss_suffix_global']),
            'u_sparse_global': float(out['loss_u_sparse_global']),
            'u_sparse_locally_backpropagated': float(out['loss_u_sparse_backward']),
            'u_sparse_reference': float(ref_s_u),
            'u_sparse_weighted_error': sparse_error,
            'u_sparse_positive_count_local': (None if out['m_u_positive_count_local'] is None
                                              else float(out['m_u_positive_count_local'])),
            'loss_total_error': error_metrics(total, ref_total),
            'suffix_direction_errors': error_metrics(directions, torch.stack((ref_i2t, ref_t2i))),
            'gradients': gradients, 'parameter_groups': groups, 'updates': updates,
            'rank_parameter_error': error_metrics(vector, root_vector), 'status': 'within_fixed_tolerance'})
    if rank == 0:
        report = {'torch_version': torch.__version__, 'backend': args.backend, 'world_size': world,
            'lambda_suffix': args.lambda_suffix, 'lambda_u_sparse': args.lambda_u_sparse,
            'objective': '10*L_S + 2*S_S + %g*L_U + %g*S_U' % (args.lambda_suffix, args.lambda_u_sparse),
            'communication': 'suffix_flat_contiguous_all_gather',
            'relative_definition': 'max_abs(actual-reference) / max(max_abs(reference), 1e-12)',
            'tolerances': {'gradient_atol': GRAD_ATOL, 'gradient_rtol': GRAD_RTOL,
                           'value_atol': VALUE_ATOL, 'value_rtol': VALUE_RTOL,
                           'update_atol': UPDATE_ATOL, 'update_rtol': UPDATE_RTOL},
            'score_shape': [4, 8], 'transposed_score_stride': [1, 8],
            'cases': results, 'status': 'passed'}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
