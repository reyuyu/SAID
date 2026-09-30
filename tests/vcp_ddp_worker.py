"""Two-rank NCCL reference for VCP global scores, gradients, and AdamW update."""
import argparse
import copy
import json
import os
import types

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_semantic_mask import inclusion, inclusion_weight
from model.nested_vcp_mask import NestedVCPMask, vcp_pair_scores, vcp_positive_masks
from tests.test_nested_jointmask import TinyJointCLIP
from train.train_nested_semantic_mask import build_optimizer, learning_rates


def force_fp32(module):
    clip = module.clip

    def encode_image(_, images, return_hidden=False):
        with torch.autocast('cuda', enabled=False):
            return clip.visual(images.float(), return_hidden=return_hidden)

    def encode_view(self, tokens):
        with torch.autocast('cuda', enabled=False):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
            transformed = self.clip.mask_net.resblocks(
                hidden.detach().float().permute(1, 0, 2)).permute(1, 0, 2)
        return text.float(), transformed

    clip.encode_image = types.MethodType(encode_image, clip)
    module.encode_view = types.MethodType(encode_view, module)


def reference_loss(module, images, token_views, valid, completed):
    z, image_hidden = module.clip.encode_image(images, return_hidden=True)
    pool = module.clip.mask_net.attn_pool.attention
    queries = module.vcp_query(image_hidden, pool.weight.squeeze(0))
    values, positives = [], []
    enabled_views = token_views if int(valid.sum()) >= 2 else token_views[:1]
    for index, tokens in enumerate(enabled_views):
        text, transformed = module.encode_view(tokens)
        selected = torch.ones_like(valid) if index == 0 else valid
        scores, _ = vcp_pair_scores(
            z, text, queries, transformed, pool.bias.squeeze(0),
            module.image_chunk, module.text_chunk, selected, selected,
            checkpoint_blocks=False)
        labels = torch.arange(len(z), device=z.device)[selected]
        image_logits = scores.T[selected].masked_fill(~selected[None], -torch.inf)
        text_logits = scores[selected].masked_fill(~selected[None], -torch.inf)
        align = (F.cross_entropy(image_logits, labels) +
                 F.cross_entropy(text_logits, labels))
        positive, probability = vcp_positive_masks(
            queries, transformed, pool.bias.squeeze(0))
        sparse = positive[selected].abs().mean()
        values.append((align, sparse))
        positives.append(probability)
    if int(valid.sum()) < 2:
        return 10 * values[0][0] + values[0][1]
    inc = inclusion(*positives)[valid].mean()
    return ((10 / 3) * sum(value[0] for value in values) +
            (values[0][1] + 2 * values[1][1] + 2 * values[2][1]) / 3 +
            inclusion_weight(module.arm, completed) * inc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    local, rank, world = [int(os.environ[key]) for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')]
    assert world == 2
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl')
    cases = [
        ('all_valid', 2, [1, 1, 1, 1]),
        ('rank1_zero_valid', 2, [1, 1, 0, 0]),
        ('global_v1', 2, [1, 0, 0, 0]),
        ('global_v0', 2, [0, 0, 0, 0]),
        ('tail_batch', 1, [1, 1]),
    ]
    results = []
    for case_index, (name, local_batch, validity) in enumerate(cases):
        torch.manual_seed(500 + case_index)
        module = NestedVCPMask(TinyJointCLIP().cuda(), checkpoint_encoders=False,
                               image_chunk=2, text_chunk=3,
                               checkpoint_pair_blocks=False).cuda()
        with torch.no_grad():
            module.vcp_query.output.weight.normal_(std=.04)
        reference = copy.deepcopy(module)
        force_fp32(module); force_fp32(reference)
        ddp = DDP(module, device_ids=[local], output_device=local,
                  find_unused_parameters=True, static_graph=False)
        total = local_batch * world
        images = torch.randn(total, 8, device='cuda')
        tokens = [torch.randint(0, 31, (total, 6), device='cuda') for _ in range(3)]
        valid = torch.tensor(validity, dtype=torch.bool, device='cuda')
        section = slice(rank * local_batch, (rank + 1) * local_batch)
        completed = 61
        loss, logs = ddp(images[section], *[view[section] for view in tokens],
                         valid[section], completed)
        expected = reference_loss(reference, images, tokens, valid, completed)
        loss.backward(); expected.backward()
        actual_named = dict(module.named_parameters())
        expected_named = dict(reference.named_parameters())
        assert actual_named.keys() == expected_named.keys()
        diagnostics = []
        for parameter_name in actual_named:
            actual, target = actual_named[parameter_name], expected_named[parameter_name]
            assert (actual.grad is None) == (target.grad is None), parameter_name
            if actual.grad is None:
                diagnostics.append(dict(name=parameter_name, none=True, max_abs=0.))
                continue
            difference = (actual.grad - target.grad).abs()
            flat = int(difference.argmax())
            index = tuple(int(x) for x in torch.unravel_index(
                torch.tensor(flat, device=difference.device), difference.shape))
            diagnostics.append(dict(name=parameter_name, none=False, index=index,
                                    max_abs=float(difference.flatten()[flat]),
                                    actual_norm=float(actual.grad.norm()),
                                    expected_norm=float(target.grad.norm())))
            torch.testing.assert_close(actual.grad, target.grad, atol=1.5e-3, rtol=1e-4,
                                       msg=parameter_name)
        torch.testing.assert_close(logs['loss'], expected.detach(), atol=3e-4, rtol=3e-5)

        left, right = build_optimizer(module), build_optimizer(reference)
        rates = learning_rates(completed, 3651, True)
        for optimizer in (left, right):
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
        left.step(); right.step()
        updates = []
        for parameter_name in actual_named:
            difference = (actual_named[parameter_name] - expected_named[parameter_name]).abs()
            flat = int(difference.argmax())
            index = tuple(int(x) for x in torch.unravel_index(
                torch.tensor(flat, device=difference.device), difference.shape))
            actual_gradient = actual_named[parameter_name].grad
            expected_gradient = expected_named[parameter_name].grad
            actual_at_index = (None if actual_gradient is None else
                               float(actual_gradient.flatten()[flat]))
            expected_at_index = (None if expected_gradient is None else
                                 float(expected_gradient.flatten()[flat]))
            near_zero_gradient = (actual_at_index is not None and
                                  abs(actual_at_index) < 2e-5 and
                                  abs(expected_at_index) < 2e-5)
            updates.append(dict(name=parameter_name, index=index,
                                max_abs=float(difference.flatten()[flat]),
                                actual_gradient_at_index=actual_at_index,
                                expected_gradient_at_index=expected_at_index,
                                near_zero_gradient=near_zero_gradient))
        worst_gradient = max(diagnostics, key=lambda row: row['max_abs'])
        worst_update = max(updates, key=lambda row: row['max_abs'])
        unexplained = [row for row in updates
                       if row['max_abs'] > 2e-6 and not row['near_zero_gradient']]
        if unexplained:
            print(json.dumps(dict(rank=rank, case=name,
                                  worst_gradients=sorted(diagnostics, key=lambda x:x['max_abs'], reverse=True)[:10],
                                  worst_updates=sorted(updates, key=lambda x:x['max_abs'], reverse=True)[:10]),
                             indent=2), flush=True)
            raise AssertionError(unexplained[0])
        results.append(dict(case=name, local_batch=local_batch, validity=validity,
                            loss_error=float((logs['loss'] - expected.detach()).abs()),
                            worst_gradient=worst_gradient, worst_adamw_update=worst_update,
                            near_zero_adamw_updates=[row for row in updates
                                                     if row['max_abs'] > 2e-6]))
        dist.barrier()
    if rank == 0:
        payload = dict(passed=True, world_size=world, cases=results)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
