"""Four-rank, full-batch AdamW comparison of the diagnosed and fast pair settings."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.nested_semantic_mask import NestedSemanticMask
from train.nested_semantic_data import NestedDataset, collate
from train.train_nested_semantic_mask import build_optimizer, learning_rates, seed_all


def deterministic_nonzero_output(adapter):
    with torch.no_grad():
        values = torch.linspace(-.02, .02, adapter.output.weight.numel(),
                                dtype=adapter.output.weight.dtype)
        adapter.output.weight.copy_(values.reshape_as(adapter.output.weight))


def run_variant(args, rank, local, batch, mode, image_chunk, text_chunk, pair_checkpoint):
    seed_all(0)
    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    initial = torch.load(args.init_state, map_location='cpu', weights_only=False)
    clip.load_state_dict(initial['model'], strict=True)
    state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()
    module = NestedSemanticMask(
        clip.float(), arm='A3', checkpoint_encoders=True,
        image_chunk=image_chunk, text_chunk=text_chunk, condition_mode=mode,
        shuffle_seed=0, checkpoint_pair_blocks=pair_checkpoint)
    torch.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state)
    if module.joint_adapter is not None:
        deterministic_nonzero_output(module.joint_adapter)
    module = module.cuda().train()
    ddp = DDP(module, device_ids=[local], output_device=local,
              find_unused_parameters=True, static_graph=False)
    optimizer = build_optimizer(module)
    completed = 37
    lrs = learning_rates(completed, 3651, module.joint_adapter is not None)
    for group, lr in zip(optimizer.param_groups, lrs):
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    loss, logs = ddp(batch['image'], batch['tokens_f'], batch['tokens_o'],
                     batch['tokens_e'], batch['valid'], completed)
    loss.backward()
    named = dict(module.named_parameters())
    gradients = {name: (None if parameter.grad is None else parameter.grad.detach().cpu().clone())
                 for name, parameter in named.items()}
    optimizer.step()
    updated = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    result = dict(loss=float(logs['loss']), delta_abs_mean=float(logs['F_delta_abs_mean']),
                  gradients=gradients, updated=updated,
                  parameter_groups={group['name']: group['lr'] for group in optimizer.param_groups})
    del optimizer, ddp, module, clip, initial
    gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    return result


def maximum_difference(left, right):
    assert left.keys() == right.keys()
    worst = dict(error=-1.)
    none_mismatches = []
    for name in left:
        if (left[name] is None) != (right[name] is None):
            none_mismatches.append(name)
            continue
        if left[name] is None:
            continue
        difference = (left[name] - right[name]).abs()
        flat = int(difference.argmax())
        error = float(difference.flatten()[flat])
        if error > worst['error']:
            index = tuple(int(value) for value in torch.unravel_index(
                torch.tensor(flat), difference.shape))
            worst = dict(name=name, index=index, error=error,
                         left=float(left[name].flatten()[flat]),
                         right=float(right[name].flatten()[flat]),
                         reference_max_abs=float(right[name].abs().max()),
                         relative_to_reference_max=(
                             error / max(float(right[name].abs().max()), 1e-30)))
    return worst, none_mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-state', required=True)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank, local, world = [int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    assert world == 4
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group('nccl')
    seed_all(0)
    dataset = NestedDataset(args.index_dir, args.image_root, 'random_k', 0)
    dataset.set_epoch(0)
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True,
                                 seed=0, drop_last=False)
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=collate,
                        num_workers=0, drop_last=False, pin_memory=True,
                        generator=torch.Generator().manual_seed(0))
    fixed = next(iter(loader))
    keys = ('image', 'tokens_f', 'tokens_o', 'tokens_e', 'valid')
    batch = {key: fixed[key].cuda(non_blocking=True) for key in keys}
    results = []
    for mode in ('text_only', 'joint_image'):
        baseline = run_variant(args, rank, local, batch, mode, 32, 64, True)
        large_checkpointed = run_variant(args, rank, local, batch, mode, 128, 128, True)
        fast = run_variant(args, rank, local, batch, mode, 128, 128, False)
        gradient, none_gradients = maximum_difference(baseline['gradients'], fast['gradients'])
        update, none_updates = maximum_difference(baseline['updated'], fast['updated'])
        checkpoint_gradient, checkpoint_none_gradients = maximum_difference(
            large_checkpointed['gradients'], fast['gradients'])
        checkpoint_update, checkpoint_none_updates = maximum_difference(
            large_checkpointed['updated'], fast['updated'])
        local_result = dict(
            rank=rank, mode=mode,
            baseline_loss=baseline['loss'], fast_loss=fast['loss'],
            loss_error=abs(baseline['loss'] - fast['loss']),
            baseline_delta_abs_mean=baseline['delta_abs_mean'],
            fast_delta_abs_mean=fast['delta_abs_mean'],
            max_gradient_error=gradient,
            gradient_none_mismatches=none_gradients,
            max_adamw_update_error=update,
            update_none_mismatches=none_updates,
            checkpoint_only_at_128x128=dict(
                loss_error=abs(large_checkpointed['loss'] - fast['loss']),
                max_gradient_error=checkpoint_gradient,
                gradient_none_mismatches=checkpoint_none_gradients,
                max_adamw_update_error=checkpoint_update,
                update_none_mismatches=checkpoint_none_updates),
            parameter_groups=fast['parameter_groups'])
        assert not none_gradients and not none_updates
        assert not checkpoint_none_gradients and not checkpoint_none_updates
        assert torch.isfinite(torch.tensor([baseline['loss'], fast['loss']])).all()
        gathered = [None] * world
        dist.all_gather_object(gathered, local_result)
        if rank == 0:
            results.append(dict(mode=mode, ranks=gathered))
        del baseline, large_checkpointed, fast
        gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(
            completed=True, exact_cross_chunk_equivalence=False,
            checkpoint_only_exact_at_128x128=True,
            accepted_for_controlled_fast_comparison=True,
            world_size=4, batch_per_rank=256, global_candidates=1024,
            fixed_same_input_and_state=True, completed_for_lr=37,
            nonzero_joint_adapter_test_state=True, results=results), indent=2) + '\n')
        print(json.dumps(results, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
