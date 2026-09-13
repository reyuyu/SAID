#!/usr/bin/env python
"""Compare the S0_GLOBAL_ONLY data stream against the logs of the other 500-step arms.

The arms do not all log the same digest fields, so this script reports, per candidate log, exactly
which fields exist and how many of the overlapping steps agree -- never a bare "identical" claim.
It also states when a comparison is impossible (missing field, no overlapping steps, unparseable
log file) instead of counting that as agreement.

Usage:  python3 tools/diag/compare_arm_streams.py [--run-dir RUN] [--glob '/root/SAID*/runs_salu/*/*']
"""
import argparse
import glob
import json
import os

DEFAULT_RUN = '/root/SAID-globalonly-v01/runs_salu/s0_global_only_v01'
PER_STEP_FIELDS = ('batch_image_id_sha256', 'batch_caption_sha256',
                   'batch_prefix_k_sha256', 'batch_sample_id_sha256')
# the cumulative stream digests use different key names in different trainers
CUMULATIVE_MAP = {'caption': ('caption', 'caption_stream_sha256'),
                  'image_id': ('image_id', 'image_id_stream_sha256'),
                  'prefix_k': ('prefix_k', 'prefix_k_stream_sha256'),
                  'sample': ('sample', 'sample_stream_sha256')}


def parse_log(path):
    """Return (line_count, rows, bad_line_count) -- bad lines are counted, never silently dropped."""
    rows, bad, total = [], 0, 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return total, rows, bad


def by_step(rows):
    return {row['completed_steps']: row for row in rows if 'completed_steps' in row}


def cumulative(row, arm_key):
    """Read one cumulative stream digest, tolerating the two naming conventions."""
    block = row.get('rank_local_stream_digests') or {}
    return block.get(arm_key)


def compare(reference, candidate_path, reference_path):
    total, rows, bad = parse_log(candidate_path)
    if not rows:
        print('  %s' % candidate_path)
        print('    lines=%d parsed=0 bad=%d -> NOT COMPARABLE (no parseable row)' % (total, bad))
        return
    cand = by_step(rows)
    overlap = sorted(set(reference) & set(cand))
    print('  %s' % candidate_path)
    print('    lines=%d parsed=%d bad=%d steps=%s..%s' % (total, len(rows), bad, min(cand), max(cand)))
    if not overlap:
        print('    overlap with the reference run: 0 steps -> NOT COMPARABLE')
        return
    for field in PER_STEP_FIELDS:
        if field not in rows[0]:
            print('    %-24s field absent in this log -> NOT COMPARABLE' % field)
            continue
        same = sum(1 for step in overlap if reference[step].get(field) == cand[step].get(field))
        print('    %-24s identical %d/%d' % (field, same, len(overlap)))
    for name, keys in CUMULATIVE_MAP.items():
        same = 0
        comparable = False
        for step in overlap:
            ref_value = cumulative(reference[step], keys[0])
            cand_value = cumulative(cand[step], keys[1])
            if ref_value is None or cand_value is None:
                continue
            comparable = True
            same += int(ref_value == cand_value)
        if comparable:
            print('    cumulative %-12s identical %d/%d (key %r vs %r)'
                  % (name, same, len(overlap), keys[0], keys[1]))
        else:
            print('    cumulative %-12s NOT COMPARABLE (candidate log has no cumulative digest)' % name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=DEFAULT_RUN)
    parser.add_argument('--glob', default='/root/SAID*/runs_salu/*/salu_log.jsonl,/root/SAID*/runs_salu/*/*/salu_log.jsonl,/root/SAID*/runs_salu/*/*/*/salu_log.jsonl')
    args = parser.parse_args()

    reference_path = os.path.join(args.run_dir, 'salu_log.jsonl')
    total, rows, bad = parse_log(reference_path)
    reference = by_step(rows)
    print('reference: %s' % reference_path)
    print('  lines=%d parsed=%d bad=%d steps=%d..%d'
          % (total, len(rows), bad, min(reference), max(reference)))
    print('  scope: rank0 only (the trainers log rank0 rows)')
    print()
    candidates = []
    for pattern in args.glob.split(','):
        candidates.extend(glob.glob(pattern))
    for path in sorted(set(candidates)):
        if os.path.abspath(path) == os.path.abspath(reference_path):
            continue
        compare(reference, path, reference_path)


if __name__ == '__main__':
    main()
