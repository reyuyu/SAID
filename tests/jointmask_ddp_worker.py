"""Two-rank NCCL reference for pair masks, uneven validity and shuffled conditions."""
import copy
import json
import os
import types

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_semantic_mask import (NestedSemanticMask, deterministic_shuffle_indices,
                                        inclusion, inclusion_weight, pair_scores,
                                        positive_masks)
from tests.test_nested_jointmask import TinyJointCLIP


def force_fp32_encoder_paths(module):
    """Remove batch-shape-dependent BF16 GEMM noise from the DDP math test."""
    clip = module.clip

    def encode_image(_, images, return_hidden=False):
        with torch.autocast('cuda', enabled=False):
            return clip.visual(images, return_hidden=return_hidden)

    def encode_view(self, tokens):
        with torch.autocast('cuda', enabled=False):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
        eot = tokens.argmax(-1)
        pooled = hidden[torch.arange(len(tokens), device=tokens.device), eot]
        base = self.clip.mask_net(hidden.detach())
        condition = (self.joint_adapter.text_condition(pooled)
                     if self.joint_adapter is not None else
                     pooled.new_zeros((len(pooled), 64)))
        return text, base, condition

    clip.encode_image = types.MethodType(encode_image, clip)
    module.encode_view = types.MethodType(encode_view, module)


def reference_loss(module, images, token_views, valid, completed):
    with torch.autocast('cuda', enabled=False):
        z, image_hidden = module.clip.encode_image(images, return_hidden=True)
    z, image_hidden = z.float(), image_hidden.float()
    if module.joint_adapter is None:
        condition = z.new_zeros((len(z), 64))
    else:
        condition = module.joint_adapter.image_condition(image_hidden)
    if module.condition_mode == 'joint_shuffled_image':
        condition = condition[deterministic_shuffle_indices(
            len(condition), module.shuffle_seed, completed, condition.device)]
    values, positives = [], []
    enabled_views = token_views if int(valid.sum()) >= 2 else token_views[:1]
    for index, tokens in enumerate(enabled_views):
        with torch.autocast('cuda', enabled=False):
            text, hidden = module.clip.encode_text(tokens, return_full=True)
        text, hidden = text.float(), hidden.float()
        eot = tokens.argmax(-1)
        pooled = hidden[torch.arange(len(tokens), device=tokens.device), eot]
        base = module.clip.mask_net(hidden.detach())
        text_condition = (module.joint_adapter.text_condition(pooled)
                          if module.joint_adapter is not None else
                          pooled.new_zeros((len(pooled), 64)))
        selected = torch.ones_like(valid) if index == 0 else valid
        scores, _ = pair_scores(z, text, condition, base, text_condition,
                                module.joint_adapter, module.image_chunk,
                                module.text_chunk, selected if index else None,
                                selected if index else None, False)
        labels = torch.arange(len(z), device=z.device)[selected]
        candidate = torch.ones_like(valid) if index == 0 else valid
        image_logits = scores[selected].masked_fill(~candidate[None], -torch.inf)
        text_logits = scores.T[selected].masked_fill(~candidate[None], -torch.inf)
        align = (F.cross_entropy(image_logits, labels) +
                 F.cross_entropy(text_logits, labels))
        diagonal = torch.arange(len(z), device=z.device)
        positive_mask, positive_probability, _ = positive_masks(
            base, condition, text_condition, module.joint_adapter)
        sparse = positive_mask[selected].abs().mean()
        values.append((align, sparse))
        positives.append(positive_probability)
    if int(valid.sum()) < 2:
        return 10 * values[0][0] + values[0][1]
    inc = inclusion(*positives)[valid].mean()
    return (10 / 3) * sum(value[0] for value in values) + (
        values[0][1] + 2 * values[1][1] + 2 * values[2][1]) / 3 + (
        inclusion_weight('A3', completed) * inc)


def main():
    local, rank, world = [int(os.environ[key]) for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')]
    assert world == 2
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl')
    cases = [
        ('text_only', [1, 1, 1, 1], 2, 2, 2, False),
        ('joint_image', [1, 1, 1, 0, 0, 0], 3, 2, 2, False),
        ('joint_image', [0, 0, 0, 0], 2, 2, 2, False),
        ('joint_image', [1, 0, 0, 0], 2, 2, 2, False),
        ('joint_shuffled_image', [1, 1, 1, 1], 2, 2, 2, False),
        ('joint_image', [1, 1, 1, 0, 0, 0], 3, 8, 8, True),
        ('joint_image', [1, 1, 1, 0, 0, 0], 3, 8, 8, False),
    ]
    results = []
    for case_index, (mode, validity, local_batch, image_chunk,
                     text_chunk, pair_checkpoint) in enumerate(cases):
        torch.manual_seed(101 + case_index)
        module = NestedSemanticMask(TinyJointCLIP().cuda(), arm='A3',
                                    checkpoint_encoders=False, condition_mode=mode,
                                    image_chunk=image_chunk, text_chunk=text_chunk,
                                    checkpoint_pair_blocks=pair_checkpoint).cuda()
        reference = copy.deepcopy(module)
        force_fp32_encoder_paths(module)
        force_fp32_encoder_paths(reference)
        ddp = DDP(module, device_ids=[local], output_device=local,
                  find_unused_parameters=True, static_graph=False)
        total = local_batch * world
        images = torch.randn(total, 8, device='cuda')
        tokens = [torch.randint(0, 31, (total, 6), device='cuda') for _ in range(3)]
        valid = torch.tensor(validity, dtype=torch.bool, device='cuda')
        section = slice(rank * local_batch, (rank + 1) * local_batch)
        completed = 37
        loss, logs = ddp(images[section], *[view[section] for view in tokens],
                         valid[section], completed)
        expected = reference_loss(reference, images, tokens, valid, completed)
        loss.backward()
        expected.backward()
        gradient_errors = []
        gradient_diagnostics = []
        for (name, parameter), (_, target) in zip(module.named_parameters(),
                                                    reference.named_parameters()):
            assert (parameter.grad is None) == (target.grad is None), name
            if parameter.grad is not None:
                error = float((parameter.grad - target.grad).abs().max())
                gradient_errors.append(error)
                gradient_diagnostics.append(dict(
                    name=name, error=error,
                    actual_norm=float(parameter.grad.norm()),
                    expected_norm=float(target.grad.norm())))
        try:
            for diagnostic, (_, parameter), (_, target) in zip(
                    gradient_diagnostics, module.named_parameters(), reference.named_parameters()):
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, target.grad,
                                               atol=1.2e-3, rtol=8e-5,
                                               msg=diagnostic['name'])
        except AssertionError:
            print(json.dumps(dict(rank=rank, case=case_index, mode=mode,
                                  loss=float(loss.detach()), expected=float(expected.detach()),
                                  gradients=sorted(gradient_diagnostics,
                                                   key=lambda item: item['error'], reverse=True)),
                             indent=2), flush=True)
            raise
        torch.testing.assert_close(logs['loss'], expected.detach(), atol=2e-4, rtol=2e-5)
        left = torch.optim.SGD(module.parameters(), lr=1e-4)
        right = torch.optim.SGD(reference.parameters(), lr=1e-4)
        left.step(); right.step()
        update_error = max(float((a - b).abs().max())
                           for a, b in zip(module.parameters(), reference.parameters()))
        for a, b in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=6e-6, rtol=4e-5)
        results.append(dict(mode=mode, validity=validity, local_batch=local_batch,
                            image_chunk=image_chunk, text_chunk=text_chunk,
                            pair_checkpoint=pair_checkpoint,
                            loss_error=float((logs['loss'] - expected.detach()).abs()),
                            max_gradient_error=max(gradient_errors),
                            max_update_error=update_error,
                            shuffle_shift=int(logs['shuffle_shift'])))
        dist.barrier()
    if rank == 0:
        print(json.dumps(dict(passed=True, world_size=world, backend='nccl',
                              cases=results), indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
