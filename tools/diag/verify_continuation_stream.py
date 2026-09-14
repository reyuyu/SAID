"""Independently verify that the continuation continues the original data stream exactly.

Three checks, each reported with what it does and does not prove:

1. **positional rebuild** -- rebuild the epoch-0 rank-0 sampler order from the manifest and
   ``DistributedSampler(seed=0)`` and check that the continuation's first logged step sits on global
   batch index ``completed_steps`` of that order (i.e. the batch the original run had not reached);
2. **replay anchor** -- replay the loader with the original configuration and hash the rank-0 payloads
   of the first ``completed_steps`` batches; this must reproduce the cumulative stream digest the
   original run recorded. The replay harness is therefore proven equivalent to the trainer's stream;
3. **continuation content** -- extend the same replay to the continuation's first steps and compare
   the per-step digests with what the continuation logged.

Check 3 is only meaningful because check 2 passed; the script reports both and never treats a failed
anchor as a pass. Image decoding is unavoidable here because the prefix length K is drawn from the
dataset's process-wide ``random`` stream, which depends on the stream position and worker assignment.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = '/root/SAID-s0-dualmask-full-v01'
for _path in (REPO, os.path.join(REPO, 'train')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from said_cvssl_data import image_id_from_path                        # noqa: E402
from train_dual_mask_suffix import (                                 # noqa: E402
    DualMaskSuffixDataset, _seed_everything, batch_stream_payload, dual_mask_suffix_collate,
)

try:                                                                  # optional, only for --dry
    from train_dual_mask_suffix import ResumePositionSampler
except ImportError:                                                   # pragma: no cover
    ResumePositionSampler = None


def read_log(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-run', default=os.path.join(
        REPO, 'runs_salu/dualmask_full_masked_formal500_v01'))
    parser.add_argument('--continuation-run', default=os.path.join(
        REPO, 'runs_salu/dualmask_full_masked_3epoch_v01'))
    parser.add_argument('--steps', type=int, default=10,
                        help='how many continuation steps to compare')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total-len', type=int, default=1000)
    parser.add_argument('--positional-only', action='store_true',
                        help='only run the batch-order check (no replay); this is the part that can '
                             'be verified independently of DataLoader worker scheduling')
    parser.add_argument('--dry', dest='dry', action='store_true', default=True,
                        help='skip the image decode (the digest fields are still the real ones); '
                             'the equivalence of the dry draw is asserted by '
                             'tests/test_dual_mask_suffix_resume.py')
    parser.add_argument('--decode-images', dest='dry', action='store_false',
                        help='decode every image instead (slow, same expected result)')
    args = parser.parse_args()

    original_config = json.load(open(os.path.join(args.original_run, 'config.json')))
    completed = int(original_config['formal_optimizer_updates'])
    anchor = {int(entry['rank']): entry['stream_sha256']
              for entry in original_config['stream_summary']}
    continuation = read_log(os.path.join(args.continuation_run, 'salu_log.jsonl'))
    logged = {row['completed_steps']: row for row in continuation}
    first_steps = sorted(logged)[:args.steps]
    print('original run: %d updates, rank-0 anchor digest %s' % (completed, anchor[0]))
    print('continuation: %d logged steps %d..%d, skipped_batches=%s, stream scope=%s'
          % (len(continuation), min(logged), max(logged), logged[min(logged)].get('skipped_batches'),
             logged[min(logged)].get('stream_digest_scope')))

    # 1. positional rebuild (no image decoding, no RNG): the batch order alone
    _seed_everything(args.seed)
    dataset = DualMaskSuffixDataset(seed=args.seed, total_len=args.total_len)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=4, rank=0, shuffle=True, seed=args.seed)
    sampler.set_epoch(0)
    order = list(iter(sampler))
    # the last batch of an epoch is ragged (drop_last=False), so the count is a ceiling
    batches_per_epoch = -(-len(order) // args.batch_size)
    print('  sampler order: %d indices -> %d batches per epoch' % (len(order), batches_per_epoch))
    position_ok = None
    if batches_per_epoch != 1217:
        position_ok = False
        print('  WARNING: batches per epoch %d != 1217' % batches_per_epoch)
    else:
        expected_epoch, expected_step = divmod(completed, batches_per_epoch)
        row = logged[min(logged)]
        position_ok = (row['epoch'] == expected_epoch and row['step_in_epoch'] == expected_step)
        print('  positional rebuild: original consumed %d batches -> next batch is epoch %d index %d; '
              'continuation started at epoch %d index %d -> %s'
              % (completed, expected_epoch, expected_step, row['epoch'], row['step_in_epoch'],
                 'OK' if position_ok else 'MISMATCH'))

    # 2/3. replay the loader with the original configuration
    if args.positional_only:
        print('  replay skipped (--positional-only)')
        print('STREAM_POSITION_VERDICT %s' % ('VERIFIED' if position_ok else 'PROBLEM'))
        return
    if args.dry and ResumePositionSampler is None:
        raise SystemExit('--dry needs the optional ResumePositionSampler; use --decode-images')
    # the caption stream comes from the process-wide ``random`` state that the workers inherit when
    # they are forked, so the replay must start from the same seeded state as the trainer
    _seed_everything(args.seed)
    replay_sampler = sampler
    if args.dry:
        # every position is marked dry: the caption draw still happens, images are placeholders
        replay_sampler = ResumePositionSampler(sampler, 1217, args.batch_size, 10 ** 9)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=replay_sampler,
                        shuffle=False, num_workers=args.num_workers, pin_memory=False,
                        collate_fn=dual_mask_suffix_collate, drop_last=False)
    print('  replay mode: %s' % ('dry (no image decode)' if args.dry else 'full image decode'))
    digest = hashlib.sha256()
    replay_anchor = None
    per_step = {}
    wanted = set(first_steps)
    for index, batch in enumerate(loader):
        payload = batch_stream_payload(batch)
        digest.update(payload)
        step = index + 1
        if step == completed:
            replay_anchor = digest.hexdigest()
        if step in wanted:
            per_step[step] = hashlib.sha256(payload).hexdigest()
        if step >= max(first_steps):
            break
    anchor_ok = replay_anchor == anchor[0]
    print('  replay anchor: rebuilt cumulative digest after %d batches %s recorded 0x%s -> %s'
          % (completed, replay_anchor, anchor[0], 'MATCH' if anchor_ok else 'MISMATCH'))
    matches, mismatches = [], []
    for step in first_steps:
        got, want = per_step.get(step), logged[step].get('batch_stream_sha256')
        (matches if got == want else mismatches).append(step)
    print('  continuation content: %d/%d per-step digests match (steps %s)'
          % (len(matches), len(first_steps), first_steps))
    if mismatches:
        print('    mismatching steps: %s' % mismatches)
        for step in mismatches[:2]:
            print('      step %d rebuilt %s logged %s' % (step, per_step.get(step),
                                                          logged[step].get('batch_stream_sha256')))
    print('  scope: rank 0 only (the trainers log rank-0 batch digests)')
    verdict = position_ok and anchor_ok and not mismatches
    print('STREAM_CONTINUATION_VERDICT %s' % ('VERIFIED' if verdict else 'NOT_VERIFIED'))
    print('caveat: check 3 only proves content continuity because check 2 reproduced the original '
          'run\'s recorded digest with the same harness')


if __name__ == '__main__':
    main()
