"""Four-rank fixed-batch timing/profiling for NEST JointMask pair scoring."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import time
import types

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
import model.nested_semantic_mask as nsm
from model.nested_semantic_mask import NestedSemanticMask, hard_st
from train.nested_semantic_data import NestedDataset, collate
from train.train_nested_semantic_mask import build_optimizer, seed_all


class Tracker:
    def __init__(self, profile_labels=False):
        self.active = False
        self.profile_labels = profile_labels
        self.events = defaultdict(list)
        self.block_calls = 0

    def reset(self):
        self.events.clear()
        self.block_calls = 0

    def call(self, name, function, *args, **kwargs):
        if not self.active:
            return function(*args, **kwargs)
        context = torch.profiler.record_function('jointmask::' + name) if self.profile_labels else nullcontext()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with context:
            start.record()
            result = function(*args, **kwargs)
            end.record()
        self.events[name].append((start, end))
        return result

    def milliseconds(self):
        return {name: sum(start.elapsed_time(end) for start, end in events)
                for name, events in self.events.items()}


def install_baseline(tracker):
    """Exact 82e06b2 pair block, including both dynamic boolean selections."""
    def block(z, t, image_condition, base_logits, text_condition, adapter, active_pairs):
        tracker.block_calls += 1
        masks, probabilities, delta = nsm.pair_mask(
            base_logits, image_condition, text_condition, adapter)
        scores = 100 * (torch.nn.functional.normalize(
            z[:, None].float() * masks, dim=-1, eps=1e-6) *
            torch.nn.functional.normalize(t.float(), dim=-1, eps=1e-6)[None]).sum(-1)
        selected = (probabilities.detach() >= .5)[active_pairs]
        selected_delta = delta.detach()[active_pairs]
        if selected.numel():
            summary = torch.stack((selected.float().sum(),
                                   selected.new_tensor(selected.numel(), dtype=torch.float32),
                                   selected.all(-1).float().sum(),
                                   (~selected).all(-1).float().sum(),
                                   selected_delta.abs().sum(),
                                   selected_delta.new_tensor(selected_delta.numel(),
                                                             dtype=torch.float32)))
        else:
            summary = scores.detach().new_zeros(6)
        return scores, summary.detach()

    def pair_scores(images, texts, image_condition, base_logits, text_condition, adapter=None,
                    image_chunk=32, text_chunk=64, row_valid=None, column_valid=None,
                    checkpoint_blocks=False, collect_stats=True):
        row_valid = (torch.ones(len(images), dtype=torch.bool, device=images.device)
                     if row_valid is None else row_valid)
        column_valid = (torch.ones(len(texts), dtype=torch.bool, device=texts.device)
                        if column_valid is None else column_valid)
        rows, summaries = [], []
        for image_start in range(0, len(images), image_chunk):
            image_stop = min(image_start + image_chunk, len(images))
            block_row = []
            for text_start in range(0, len(texts), text_chunk):
                text_stop = min(text_start + text_chunk, len(texts))
                active = (row_valid[image_start:image_stop, None] &
                          column_valid[None, text_start:text_stop])
                args = (images[image_start:image_stop], texts[text_start:text_stop],
                        image_condition[image_start:image_stop], base_logits[text_start:text_stop],
                        text_condition[text_start:text_stop])
                function = lambda *values, active=active: block(
                    *values, adapter=adapter, active_pairs=active)
                if checkpoint_blocks and torch.is_grad_enabled():
                    scores, summary = checkpoint(function, *args, use_reentrant=False)
                else:
                    scores, summary = function(*args)
                block_row.append(scores)
                summaries.append(summary)
            rows.append(torch.cat(block_row, dim=1))
        return torch.cat(rows), torch.stack(summaries).sum(0)

    nsm.pair_scores = pair_scores


def install_optimized_counter(tracker):
    original = nsm._pair_score_block

    def counted(*args, **kwargs):
        tracker.block_calls += 1
        return original(*args, **kwargs)

    nsm._pair_score_block = counted


def install_matrix_reference():
    """Text-only numerical/performance reference using the original matrix identity."""
    def matrix_terms(z, t, base_logits, text_condition, image_condition, valid,
                     valid_global, adapter=None, z_global=None, image_condition_global=None,
                     image_chunk=32, text_chunk=64, checkpoint_blocks=False):
        assert adapter is None
        probabilities = torch.sigmoid(base_logits)
        masks = hard_st(probabilities)
        align, sparse, base_logs = nsm.view_terms(
            z, t, masks, valid, valid_global, z_global, score_chunk=text_chunk)
        count = int(valid_global.sum())
        keep = masks.detach()[valid]
        local = torch.stack((keep.sum(), keep.new_tensor(keep.numel()),
                             keep.all(-1).float().sum(),
                             (~keep.bool()).all(-1).float().sum()))
        total = nsm.global_sum(local)
        ratio = total[0] / total[1].clamp_min(1)
        logs = dict(i2t=base_logs['i2t'], t2i=base_logs['t2i'], sparse=base_logs['sparse'],
                    keep_ratio=base_logs['sparse'], all_open=total[2] / count,
                    all_closed=total[3] / count, candidates=count,
                    positive_keep_ratio=ratio, negative_keep_ratio=ratio,
                    positive_all_open=total[2] / count,
                    positive_all_closed=total[3] / count,
                    negative_all_open=total[2] / count,
                    negative_all_closed=total[3] / count,
                    delta_abs_mean=0., positive_delta_abs_mean=0.,
                    negative_delta_abs_mean=0.)
        return align, sparse, masks, probabilities, logs

    nsm.pair_view_terms = matrix_terms


def wrap_timed_paths(module, tracker):
    original_image = module.clip.encode_image
    module.clip.encode_image = lambda *args, **kwargs: tracker.call(
        'image_encoder', original_image, *args, **kwargs)
    original_view = module.encode_view
    module.encode_view = types.MethodType(
        lambda self, *args, **kwargs: tracker.call(
            'text_encoder_mask', original_view, *args, **kwargs), module)
    original_pair = nsm.pair_scores
    nsm.pair_scores = lambda *args, **kwargs: tracker.call(
        'pair_forward', original_pair, *args, **kwargs)
    original_gather, original_sum = nsm.gather, nsm.global_sum
    nsm.gather = lambda *args, **kwargs: tracker.call(
        'forward_communication', original_gather, *args, **kwargs)
    nsm.global_sum = lambda *args, **kwargs: tracker.call(
        'forward_communication', original_sum, *args, **kwargs)


def event_call(function):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    result = function()
    end.record()
    return result, start, end


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['T', 'TI'], required=True)
    parser.add_argument('--implementation', choices=['baseline', 'optimized', 'matrix'], required=True)
    parser.add_argument('--image-chunk', type=int, default=32)
    parser.add_argument('--text-chunk', type=int, default=64)
    parser.add_argument('--pair-checkpoint', type=int, choices=[0, 1], default=1)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--init-state', required=True)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--image-root', required=True)
    args = parser.parse_args()
    rank, local, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    assert world == 4 and not (args.implementation == 'matrix' and args.mode != 'T')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(0)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    dataset = NestedDataset(args.index_dir, args.image_root, 'random_k', 0)
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True,
                                 seed=0, drop_last=False)
    dataset.set_epoch(0); sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler, collate_fn=collate,
                        num_workers=0, drop_last=False, pin_memory=True,
                        generator=torch.Generator().manual_seed(0))
    iterator = iter(loader)
    waits, fixed = [], None
    started = time.perf_counter()
    fixed = next(iterator)
    waits.append(1000 * (time.perf_counter() - started))
    del iterator, loader

    tracker = Tracker(profile_labels=args.profile)
    if args.implementation == 'baseline':
        install_baseline(tracker)
    elif args.implementation == 'optimized':
        install_optimized_counter(tracker)
    else:
        install_matrix_reference()

    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    initial = torch.load(args.init_state, map_location='cpu', weights_only=False)
    clip.load_state_dict(initial['model'], strict=True)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    module = NestedSemanticMask(
        clip.float(), arm='A3', checkpoint_encoders=True,
        image_chunk=args.image_chunk, text_chunk=args.text_chunk,
        condition_mode='text_only' if args.mode == 'T' else 'joint_image',
        shuffle_seed=0, checkpoint_pair_blocks=bool(args.pair_checkpoint))
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
    module = module.cuda().train()
    wrap_timed_paths(module, tracker)
    ddp = DDP(module, device_ids=[local], output_device=local,
              find_unused_parameters=True, static_graph=False)
    optimizer = build_optimizer(module)
    for group in optimizer.param_groups:
        group['lr'] = {'backbone': 5e-9, 'shared_mask': 1e-3,
                       'joint_adapter': 1e-4}[group['name']]
    identity = dict(rank=rank, local_rank=local, pid=os.getpid(),
                    hostname=socket.gethostname(),
                    uuid=str(torch.cuda.get_device_properties(local).uuid))
    del initial

    keys = ('image', 'tokens_f', 'tokens_o', 'tokens_e', 'valid')
    records = []
    total_iterations = args.warmup + args.steps
    profile_context = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=args.warmup, warmup=0,
                                         active=args.steps, repeat=1),
        record_shapes=False, profile_memory=True, with_stack=False)
        if args.profile else nullcontext())
    torch.cuda.reset_peak_memory_stats()
    with profile_context as profiler:
        for iteration in range(total_iterations):
            measured = iteration >= args.warmup
            tracker.active = measured
            tracker.reset()
            cpu_started = time.perf_counter()
            device_batch, data_start, data_end = event_call(
                lambda: {key: fixed[key].cuda(non_blocking=True) for key in keys})
            optimizer.zero_grad(set_to_none=True)
            (forward_result, forward_start, forward_end) = event_call(
                lambda: ddp(device_batch['image'], device_batch['tokens_f'],
                            device_batch['tokens_o'], device_batch['tokens_e'],
                            device_batch['valid'], iteration))
            loss, logs = forward_result
            (_, backward_start, backward_end) = event_call(loss.backward)
            (_, optimizer_start, optimizer_end) = event_call(optimizer.step)
            torch.cuda.synchronize()
            internal = tracker.milliseconds()
            health = dict(rank=rank, iteration=iteration,
                          loss=float(logs['loss']), block_calls=tracker.block_calls,
                          data_h2d_ms=data_start.elapsed_time(data_end),
                          forward_ms=forward_start.elapsed_time(forward_end),
                          encoder_ms=internal.get('image_encoder', 0.) +
                                     internal.get('text_encoder_mask', 0.),
                          pair_forward_ms=internal.get('pair_forward', 0.),
                          forward_communication_ms=internal.get('forward_communication', 0.),
                          backward_recompute_ms=backward_start.elapsed_time(backward_end),
                          optimizer_ms=optimizer_start.elapsed_time(optimizer_end),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            log_started = time.perf_counter()
            peers = [None] * world
            dist.all_gather_object(peers, dict(rank=rank, loss=health['loss'],
                                               block_calls=health['block_calls']))
            ids = nsm.gather(fixed['image_id'].cuda(), False)
            ids.unique(return_counts=True)
            health['logging_ms'] = 1000 * (time.perf_counter() - log_started)
            torch.cuda.synchronize()
            health['step_ms'] = 1000 * (time.perf_counter() - cpu_started)
            if measured:
                records.append(health)
            if args.profile:
                profiler.step()
    tracker.active = False

    local_result = dict(identity=identity, data_wait_ms=waits, steps=records,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    gathered = [None] * world
    dist.all_gather_object(gathered, local_result)
    if args.profile:
        trace = output / f'trace-rank{rank}.json'
        profiler.export_chrome_trace(str(trace))
        averages = []
        for event in profiler.key_averages():
            averages.append(dict(key=event.key, count=event.count,
                                 self_cpu_time_total=event.self_cpu_time_total,
                                 cpu_time_total=event.cpu_time_total,
                                 self_cuda_time_total=getattr(event, 'self_cuda_time_total', 0.),
                                 cuda_time_total=getattr(event, 'cuda_time_total', 0.)))
        (output / f'profiler-rank{rank}.json').write_text(json.dumps(averages))
    if rank == 0:
        metric_names = ('step_ms', 'data_h2d_ms', 'forward_ms', 'encoder_ms',
                        'pair_forward_ms', 'forward_communication_ms',
                        'backward_recompute_ms', 'optimizer_ms', 'logging_ms')
        maximum_by_step = {name: [max(rank_result['steps'][step][name]
                                           for rank_result in gathered)
                                  for step in range(args.steps)]
                           for name in metric_names}
        summary = {name: dict(median_ms=percentile(values, 50),
                              p90_ms=percentile(values, 90),
                              values_ms=values)
                   for name, values in maximum_by_step.items()}
        payload = dict(passed=True, args=vars(args), ranks=gathered,
                       max_rank_summary=summary,
                       data_wait=dict(startup_max_ms=max(result['data_wait_ms'][0]
                                                         for result in gathered),
                                      steady_max_values_ms=[]),
                       block_calls_per_rank=[step['block_calls']
                                             for step in gathered[0]['steps']],
                       losses=[step['loss'] for step in gathered[0]['steps']])
        (output / 'result.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(dict(output=str(output),
                              median_step_ms=summary['step_ms']['median_ms'],
                              p90_step_ms=summary['step_ms']['p90_ms'],
                              peaks=[result['peak_allocated_gib'] for result in gathered],
                              blocks=payload['block_calls_per_rank'])), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
