"""Actual ShareGPT4V training Dataset/DataLoader, without model construction or optimization."""
import argparse
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from clip.clip import _transform

from train.sharegpt4v import share4v_train_dataset
from tools.data.audit_sharegpt4v import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--json', default='share-captioner_coco_lcs_sam_1246k_1107.json')
    parser.add_argument('--audit', type=Path, default=Path('outputs/data_audit/sharegpt4v_full_audit.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    random.seed(2702 + rank)
    torch.manual_seed(2702 + rank)
    torch.set_num_threads(2)
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    try:
        dataset = share4v_train_dataset(str(args.root), args.json, str(args.root),
                                       strict_manifest=args.audit, preprocess=_transform(224))
        count = 1000 if world == 1 else world * 20 * 16
        indices = random.Random(2702).sample(range(len(dataset)), count)
        subset = Subset(dataset, indices)
        sampler = DistributedSampler(subset, num_replicas=world, rank=rank, shuffle=True, seed=2702,
                                     drop_last=False) if world > 1 else None
        if sampler:
            sampler.set_epoch(0)
        selected = list(iter(sampler)) if sampler else list(range(count))
        # num_workers=0 for the single-process requirement; worker decode in 4-GPU mode.
        loader = DataLoader(subset, batch_size=16, sampler=sampler, shuffle=False,
                            num_workers=2 if world > 1 else 0,
                            **({'multiprocessing_context': 'spawn'} if world > 1 else {}))
        seen = batches = sam_count = 0
        shapes = set()
        with torch.no_grad():
            for images, captions in loader:
                assert images.ndim == 4 and tuple(images.shape[1:]) == (3, 224, 224)
                assert len(captions) == len(images) and all(isinstance(c, str) and c.strip() for c in captions)
                assert torch.isfinite(images).all()
                if world > 1:
                    images = images.to(local_rank)
                    assert images.device.index == local_rank and torch.isfinite(images).all()
                shapes.add(tuple(images.shape))
                batches += 1
                seen += len(captions)
        global_indices = [indices[i] for i in selected]
        sam_count = sum(dataset.json_data[i]['image'].startswith('sam/images/') for i in global_indices)
        result = {'rank': rank, 'records_read': seen, 'batches': batches, 'shapes': sorted(shapes),
                  'sam_records': sam_count, 'errors': 0, 'dataset_indices': global_indices}
        results = [None] * world
        if world > 1:
            dist.all_gather_object(results, result)
        else:
            results = [result]
        if rank == 0:
            all_indices = [i for item in results for i in item['dataset_indices']]
            assert len(all_indices) == len(set(all_indices)) == count
            assert all(item['batches'] == 20 for item in results) if world > 1 else seen == 1000
            write_json(args.output, {'world_size': world, 'records_read': count,
                       'sampler_duplicates': 0, 'errors': 0, 'backward_calls': 0, 'optimizer_steps': 0,
                       'per_rank': [{k: v for k, v in item.items() if k != 'dataset_indices'} for item in results]})
            print('Loader smoke PASS: %s' % args.output, flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
