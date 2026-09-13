#!/usr/bin/env python
"""Independent rebuild of the epoch-0 rank0 sample stream for the S0_GLOBAL_ONLY run.

The stream is rebuilt from the manifest and ``DistributedSampler(seed=0)`` alone -- no image is
decoded and no trainer code path is reused beyond the dataset's ``image_id_from_path`` -- and then
compared with the digests the trainer logged.

Both digest conventions are evaluated on purpose:

* ``int64`` raw bytes -- what a naive implementation would use;
* ``float32`` bytes -- what the project's ``tensor_digest`` actually uses.

Reporting both is the point: image ids are ~2.6e18 integers, which float32 cannot represent, so the
logged per-step digest is an agreement test *under a stated convention*, not a byte-exact one. The
script prints which convention reproduces the log instead of asserting that any digest matches.

Usage:  python3 tools/diag/rebuild_stream_digests.py [--train-root DIR] [--run-dir DIR]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch


def parse_log(path):
    rows = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row['completed_steps'])] = row
    return rows


def digest(values, dtype):
    array = np.asarray(values).astype(dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', default='/root/SAID-globalonly-v01')
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--manifest', default='share-captioner_coco_lcs_sam_1246k_1107.json')
    parser.add_argument('--dataset-root', default='/root/datasets/ShareGPT4V')
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch-size-local', type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.train_root, 'train'))
    sys.path.insert(0, args.train_root)
    from said_cvssl_data import Share4VCvsslDataset, image_id_from_path

    run_dir = args.run_dir or os.path.join(args.train_root, 'runs_salu/s0_global_only_v01')
    rows = parse_log(os.path.join(run_dir, 'salu_log.jsonl'))
    steps = sorted(rows)

    dataset = Share4VCvsslDataset(root=args.dataset_root, json_file=args.manifest,
                                  img_root=args.dataset_root, seed=0, augment_view_b=False,
                                  strict_manifest=None)
    all_ids = [image_id_from_path(record['image']) for record in dataset.json_data]
    print('  manifest records: %d | total_len offset: %d' % (len(all_ids), dataset.total_len))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=0)
    sampler.set_epoch(0)
    order = list(iter(sampler))
    local = args.batch_size_local

    for label, dtype in (('int64 bytes', np.int64),
                         ('float32 bytes (trainer tensor_digest)', np.float32)):
        checked = agreed = 0
        examples = []
        for step in steps:
            start = (step - 1) * local
            if start + local > len(order):
                break
            expected = [all_ids[index] for index in order[start:start + local]]
            checked += 1
            got = digest(expected, dtype)
            if got == rows[step]['batch_image_id_sha256']:
                agreed += 1
            elif len(examples) < 2:
                examples.append((step, got, rows[step]['batch_image_id_sha256']))
        print('  image_id, convention %-40s compared %d | identical %d' % (label, checked, agreed))
        if examples:
            print('    first mismatches (step, rebuilt, logged):', examples)

    checked = agreed = 0
    for step in steps:
        start = (step - 1) * local
        if start + local > len(order):
            break
        expected = [index + dataset.total_len for index in order[start:start + local]]
        checked += 1
        if digest(expected, np.float32) == rows[step]['batch_sample_id_sha256']:
            agreed += 1
    print('  sample_id (index + total_len, float32)        compared %d | identical %d' % (checked, agreed))
    print('  scope: rank%d only (the trainer logs rank0 rows)' % args.rank)


if __name__ == '__main__':
    main()
