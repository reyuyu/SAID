"""Run with torchrun --standalone --nproc_per_node=2 -m tests.l14_ddp_worker."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_fusion_mask import fusion_scores
from model.nested_semantic_mask import gather, hard_st
from tests.fusion_ddp_worker import fp32_encoders, theoretical_zero_mask
from tests.test_balanced_hparams import weighted_reference
from tests.test_balanced_l14 import l14_inputs, make_l14_model
from tests.test_nested_fusion import explicit_inputs, explicit_logits, reference_loss
from train.train_nested_semantic_mask import build_optimizer, optimizer_learning_rates


CASES = [('all_valid', 2, [1, 1, 1, 1]),
         ('rank1_zero_valid', 2, [1, 1, 0, 0]),
         ('global_valid_0', 2, [0, 0, 0, 0]),
         ('global_valid_1', 2, [1, 0, 0, 0]),
         ('tail', 1, [1, 1])]


def run_case(local, rank, index, case, batch, validity, search_hparams):
    torch.manual_seed(1451 + index)
    module = make_l14_model(search_hparams=search_hparams, image_chunk=3, text_chunk=2).cuda()
    assert module.fusion_branch.visual_tokens == 256
    assert module.fusion_branch.visual_adapter.weight.shape == (768, 1024)
    for blocks in (module.clip.mask_net.resblocks, module.fusion_branch.visual_blocks):
        assert len(blocks) == 1 and blocks[0].attn.num_heads == 12
    reference = copy.deepcopy(module)
    # Same FP32 reference mode as fusion_ddp_worker: isolate distributed correctness.
    fp32_encoders(module)
    fp32_encoders(reference)
    ddp = DDP(module, device_ids=[local], output_device=local,
              find_unused_parameters=True, static_graph=False)
    images, views = l14_inputs(batch * 2, device='cuda')
    valid = torch.tensor(validity, device='cuda', dtype=torch.bool)
    sl = slice(rank * batch, (rank + 1) * batch)
    loss, logs = ddp(images[sl], *[v[sl] for v in views], valid[sl], 61)
    expected = (weighted_reference if search_hparams else reference_loss)(
        reference, images, views, valid, 61)
    score_errors = {}
    with torch.no_grad():
        z, vv = module.encode_visual(images[sl])
        zg = gather(z, False)
        vg = tuple(gather(x, False) for x in vv)
        for label, tokens in zip(('F', 'O', 'E'), views if sum(validity) >= 2 else views[:1]):
            text, tt = module.encode_view(tokens[sl])
            score, _, _ = fusion_scores(module, zg, text, vg, tt,
                                         torch.ones_like(valid), torch.ones_like(valid[sl]))
            score = gather(score, False)
            rz, rt, rv, rtt = explicit_inputs(reference, images, tokens)
            mask = hard_st(explicit_logits(reference, rv, rtt).sigmoid())
            target = 100 * (F.normalize(rz[None] * mask, dim=-1, eps=1e-6) *
                            F.normalize(rt, dim=-1, eps=1e-6)[:, None]).sum(-1)
            torch.testing.assert_close(score, target, atol=8e-5, rtol=3e-6, msg=label)
            score_errors[label] = float((score - target).abs().max())
    loss.backward()
    expected.backward()
    actual_named = dict(module.named_parameters())
    reference_named = dict(reference.named_parameters())
    assert actual_named.keys() == reference_named.keys()
    gradients = []
    for name, actual in actual_named.items():
        target = reference_named[name]
        assert (actual.grad is None) == (target.grad is None), name
        if actual.grad is None:
            gradients.append(dict(name=name, none=True, max_abs=0))
            continue
        assert torch.isfinite(actual.grad).all() and torch.isfinite(target.grad).all(), name
        diff = (actual.grad - target.grad).abs()
        idx = tuple(int(v) for v in torch.unravel_index(diff.argmax(), diff.shape))
        gradients.append(dict(name=name, none=False, max_abs=float(diff.max()), index=idx,
                              actual_norm=float(actual.grad.norm()), reference_norm=float(target.grad.norm()),
                              actual_at_worst=float(actual.grad[idx]), reference_at_worst=float(target.grad[idx])))
        try:
            torch.testing.assert_close(actual.grad, target.grad, atol=1.5e-3, rtol=1e-4, msg=name)
        except AssertionError:
            print(json.dumps(dict(case=case, rank=rank, gradients=gradients), indent=2), flush=True)
            raise
    averaged_loss = loss.detach().clone()
    dist.all_reduce(averaged_loss)
    averaged_loss /= 2
    torch.testing.assert_close(averaged_loss, expected.detach(), atol=3e-4, rtol=3e-5)
    torch.testing.assert_close(logs['loss'], expected.detach(), atol=3e-4, rtol=3e-5)
    assert logs['valid_global'] == sum(validity)
    assert logs['O_candidates'] == logs['E_candidates'] == (sum(validity) if sum(validity) >= 2 else 0)
    optimizers = [build_optimizer(module), build_optimizer(reference)]
    for optimizer in optimizers:
        for group, rate in zip(optimizer.param_groups, optimizer_learning_rates(module, 61, 4868)):
            group['lr'] = rate
        optimizer.step()
    updates = []
    for name, actual in actual_named.items():
        target = reference_named[name]
        diff = (actual - target).abs()
        exempt = torch.zeros_like(diff, dtype=torch.bool)
        theoretical = torch.zeros_like(diff, dtype=torch.bool)
        if actual.grad is not None:
            theoretical = theoretical_zero_mask(name, actual.grad, 'patch')
            group = next(g for g in optimizers[0].param_groups if any(p is actual for p in g['params']))
            ga, gb = actual.grad, target.grad
            predicted = (group['lr'] * (ga / (ga.abs() + 1e-8) - gb / (gb.abs() + 1e-8))).abs()
            rounding = 2 * torch.finfo(actual.dtype).eps * torch.maximum(
                actual.abs(), target.abs()).clamp_min(.01)
            exempt = ((ga.abs() < 2e-5) & (gb.abs() < 2e-5) &
                      ((predicted - diff).abs() <= rounding))
        unexplained = diff.masked_fill(exempt, 0)
        flat = int(diff.argmax())
        idx = tuple(int(v) for v in torch.unravel_index(diff.argmax(), diff.shape))
        updates.append(dict(name=name, max_abs=float(diff.max()), unexplained_max=float(unexplained.max()),
                            index=idx, exempt_count=int(exempt.sum()),
                            exempt_over_tolerance_count=int((exempt & (diff > 2e-6)).sum()),
                            theoretical_null_count=int(theoretical.sum()),
                            near_zero_gradient=bool(exempt.flatten()[flat]),
                            theoretical_null_direction=bool(theoretical.flatten()[flat]),
                            actual_gradient=None if actual.grad is None else float(actual.grad.flatten()[flat]),
                            reference_gradient=None if target.grad is None else float(target.grad.flatten()[flat])))
        if float(unexplained.max()) > 2e-6:
            print(json.dumps(dict(case=case, rank=rank, gradients=gradients, updates=updates),
                             indent=2), flush=True)
            raise AssertionError(f'Unexplained AdamW difference {name}: {float(unexplained.max())}')
    result = dict(case=case, rank=rank, local_batch=batch, validity=validity,
                  visual_hidden=1024, text_hidden=768, native_width=768, patches=256,
                  mask_layers=1, mask_heads=12, image_chunk=3, text_chunk=2,
                  search_hparams=module.search_hparams, encoder_precision='fp32',
                  loss_error=float((logs['loss'] - expected.detach()).abs()),
                  score_errors=score_errors, gradients=gradients, updates=updates)
    dist.barrier()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--search-arms', action='store_true',
                        help='Also exercise the nondefault coefficients used by fusion_ddp_worker.')
    args = parser.parse_args()
    local, rank, world = [int(os.environ[k]) for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')]
    assert world == 2, 'This worker requires exactly two CUDA ranks.'
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    try:
        arms = [None]
        if args.search_arms:
            arms.append(dict(fusion_lr=2e-4, visual_mask_lr_scale=.5, view_weights=[2, 1, 1],
                             sparsity_scale=.75, inclusion_max=1.5))
        results = []
        for hp in arms:
            for index, (case, batch, validity) in enumerate(CASES):
                if rank == 0:
                    print(json.dumps(dict(testing='balanced_l14', case=case, search_hparams=hp)), flush=True)
                result = run_case(local, rank, index, case, batch, validity, hp)
                per_rank = [None] * world
                dist.all_gather_object(per_rank, result)
                if rank == 0:
                    results.extend(per_rank)
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(passed=True, world_size=world,
                                                       cases=results), indent=2) + '\n')
            print(json.dumps(dict(passed=True, cases=len(results) // world)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
