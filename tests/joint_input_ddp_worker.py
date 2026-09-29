"""Two-rank NCCL reference for input-level JointInput masks and AdamW updates."""
import copy
import json
import os
import types

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_joint_input import (NestedJointInputMask, joint_pair_scores,
                                      positive_joint_masks)
from model.nested_semantic_mask import inclusion, inclusion_weight
from tests.test_nested_joint_input import TinyJointInputCLIP
from train.train_nested_joint_input import build_optimizer


def force_fp32(module):
    def encode_visual(self, images):
        z, cls, patch = self.clip.encode_image_with_joint_tokens(images)
        raw = cls[:, None] if self.visual_mode == 'cls' else patch
        return z.float(), raw.detach().float()

    def encode_text(self, tokens):
        text, hidden = self.clip.encode_text(tokens, return_full=True)
        return text.float(), hidden.detach().float()

    module.encode_visual = types.MethodType(encode_visual, module)
    module.encode_text = types.MethodType(encode_text, module)


def reference_loss(module, images, token_views, valid, completed):
    z, raw = module.encode_visual(images)
    visual = module.visual_input_projection(raw)
    values, positives = [], []
    enabled_views = token_views if int(valid.sum()) >= 2 else token_views[:1]
    for index, tokens in enumerate(enabled_views):
        text, hidden = module.encode_text(tokens)
        selected = torch.ones_like(valid) if index == 0 else valid
        scores, _ = joint_pair_scores(
            z, text, visual, hidden, module.clip.mask_net, module.readout_mode,
            module.pair_microbatch, selected, selected,
            module.checkpoint_pair_block, True)
        labels = torch.arange(len(z), device=z.device)[selected]
        logits_i = scores[selected].masked_fill(~selected[None], -torch.inf)
        logits_t = scores.T[selected].masked_fill(~selected[None], -torch.inf)
        align = F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)
        positive, probability = positive_joint_masks(
            visual, hidden, module.clip.mask_net, module.readout_mode,
            module.checkpoint_pair_block)
        values.append((align, positive[selected].abs().mean()))
        positives.append(probability)
    if int(valid.sum()) < 2:
        return 10 * values[0][0] + values[0][1]
    inc = inclusion(*positives)[valid].mean()
    return ((10 / 3) * sum(value[0] for value in values) +
            (values[0][1] + 2 * values[1][1] + 2 * values[2][1]) / 3 +
            inclusion_weight('A3', completed) * inc)


def diagnostic(actual, expected, gradient=True):
    entries = []
    for name, parameter in actual.items():
        target = expected[name]
        left = parameter.grad if gradient else parameter
        right = target.grad if gradient else target
        assert (left is None) == (right is None), name
        if left is None:
            continue
        difference = (left - right).abs()
        flat = int(difference.argmax())
        index = tuple(int(v) for v in torch.unravel_index(
            torch.tensor(flat, device=difference.device), difference.shape))
        entries.append(dict(name=name, index=index,
                            error=float(difference.flatten()[flat]),
                            actual_max=float(left.abs().max()),
                            expected_max=float(right.abs().max())))
    return max(entries, key=lambda item: item['error'])


def main():
    local, rank, world = [int(os.environ[k]) for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE')]
    assert world == 2
    torch.cuda.set_device(local); torch.set_num_threads(2)
    dist.init_process_group('nccl')
    cases = [
        ('cls', 'all', [1, 1, 1, 1], 2, False),
        ('cls', 'text', [0, 0, 1, 1], 2, True),
        ('patch', 'all', [0, 0, 0, 0], 2, True),
        ('patch', 'text', [1, 0, 0, 0], 2, True),
        ('patch', 'text', [1, 1], 1, False),
    ]
    results = []
    for case_index, (visual_mode, readout_mode, validity, local_batch, checkpoint_block) in enumerate(cases):
        torch.manual_seed(401 + case_index)
        module = NestedJointInputMask(
            TinyJointInputCLIP().cuda(), visual_mode, readout_mode,
            pair_microbatch=2, checkpoint_pair_block=checkpoint_block,
            checkpoint_encoders=False).cuda()
        reference = copy.deepcopy(module)
        force_fp32(module); force_fp32(reference)
        ddp = DDP(module, device_ids=[local], output_device=local,
                  find_unused_parameters=True, static_graph=False)
        total = local_batch * world
        images = torch.randn(total, 8, device='cuda')
        tokens = [torch.randint(0, 31, (total, 6), device='cuda') for _ in range(3)]
        valid = torch.tensor(validity, dtype=torch.bool, device='cuda')
        section = slice(rank * local_batch, (rank + 1) * local_batch)
        completed = 37
        actual_loss, logs = ddp(images[section], *[view[section] for view in tokens],
                                valid[section], completed)
        expected_loss = reference_loss(reference, images, tokens, valid, completed)
        actual_loss.backward(); expected_loss.backward()
        actual_parameters = dict(module.named_parameters())
        expected_parameters = dict(reference.named_parameters())
        assert actual_parameters.keys() == expected_parameters.keys()
        worst_gradient = diagnostic(actual_parameters, expected_parameters, True)
        for name, parameter in actual_parameters.items():
            target = expected_parameters[name]
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, target.grad,
                                           atol=2e-3, rtol=2e-4, msg=name)
        left, right = build_optimizer(module), build_optimizer(reference)
        left.step(); right.step()
        worst_update = diagnostic(actual_parameters, expected_parameters, False)
        null_direction_exception = False
        if worst_update['error'] > 3.1e-4:
            name, index = worst_update['name'], worst_update['index']
            parameter, target = actual_parameters[name], expected_parameters[name]
            if name.endswith('attn.in_proj_bias'):
                width = parameter.numel() // 3
                offset = index[0]
                null_direction_exception = (width <= offset < 2 * width and
                    abs(float(parameter.grad[index])) < 1e-4 and
                    abs(float(target.grad[index])) < 1e-4 and
                    worst_update['error'] <= 2.01e-3)
            elif name == 'clip.mask_net.attn_pool.attention.bias':
                null_direction_exception = (
                    abs(float(parameter.grad[index])) < 1e-4 and
                    abs(float(target.grad[index])) < 1e-4 and
                    worst_update['error'] <= 2.01e-3)
            if not null_direction_exception:
                raise AssertionError(worst_update)
        torch.testing.assert_close(logs['loss'], expected_loss.detach(), atol=3e-4, rtol=3e-5)
        results.append(dict(
            visual_mode=visual_mode, readout_mode=readout_mode,
            validity=validity, local_batch=local_batch,
            checkpoint_block=checkpoint_block,
            loss_error=float((logs['loss'] - expected_loss.detach()).abs()),
            max_gradient_error=worst_gradient,
            max_adamw_update_error=worst_update,
            adamw_null_direction_exception=null_direction_exception))
        dist.barrier()
    if rank == 0:
        print(json.dumps(dict(passed=True, world_size=2, backend='nccl', cases=results), indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
