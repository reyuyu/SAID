"""Complete training-corpus statistics and matched sampling correctness evidence."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import mmap
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import torch

from model import longclip
from train.nested_semantic_data import sampled_text_views, file_sha

INDEX = Path('/root/lk_projects/SAID-nest-clip-v1/data_index')
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_summary_detail_500_v1')
EXP = Path(__file__).resolve().parents[1]/'experiments/nest_clip_v1/balanced_summary_detail_500_v1'


def distribution(counter, quantiles):
    items = sorted((int(k), v) for k, v in counter.items())
    total = sum(v for _, v in items)
    if not total:
        return {'count': 0}
    def value_at(rank):
        cumulative = 0
        for value, count in items:
            cumulative += count
            if rank < cumulative:
                return value
        raise AssertionError(rank)
    result = {'count': total, 'mean': sum(k*v for k, v in items)/total}
    for p in sorted(set(quantiles+[50])):
        q = (total-1)*p/100
        lower, upper = math.floor(q), math.ceil(q)
        a, b = value_at(lower), value_at(upper)
        result['p'+str(p)] = a+(b-a)*(q-lower)
    result['median'] = result['p50']
    return result


def read_records(start, stop):
    offsets = np.load(INDEX/'offsets.npy', mmap_mode='r')
    with (INDEX/'records.jsonl').open('rb') as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for index in range(start, stop):
                yield index, json.loads(data[offsets[index]:offsets[index+1]])


def whole_chunk(bounds):
    result = {k: Counter() for k in ('sentences', 'summary_tokens', 'detail_tokens', 'full_raw_tokens')}
    valid, detail_truncated, summary_truncated, first_overlong, empty = 0, 0, 0, 0, 0
    for _, row in read_records(*bounds):
        caption = row['caption'].replace('\n', ' ').strip()
        parts = [x.strip() for x in caption.split('. ') if x.strip()]
        if not parts:
            empty += 1
            continue
        n = len(parts)
        result['sentences'][n] += 1
        full = len(longclip._tokenizer.encode('. '.join(parts)))+2
        summary = len(longclip._tokenizer.encode(parts[0]))+2
        result['full_raw_tokens'][full] += 1
        result['summary_tokens'][summary] += 1
        first_overlong += summary > 248
        if n >= 2:
            valid += 1
            detail = len(longclip._tokenizer.encode('. '.join(parts[1:])))+2
            result['detail_tokens'][detail] += 1
            detail_truncated += detail > 248
            summary_truncated += summary > 248
    return {'histograms': result, 'valid': valid, 'detail_truncated': detail_truncated,
            'summary_truncated': summary_truncated, 'first_overlong': first_overlong,
            'empty': empty, 'bounds': bounds}


def subset_chunk(indices):
    offsets = np.load(INDEX/'offsets.npy', mmap_mode='r')
    hist = {k: Counter() for k in ('old_prefix_tokens', 'old_remainder_tokens',
                                 'old_prefix_sentences', 'old_remainder_sentences',
                                 'actual_full_tokens', 'new_summary_tokens', 'new_detail_tokens',
                                 'new_summary_sentences', 'new_detail_sentences')}
    old_valid = raw_valid = 0
    old_coverage_sum = new_coverage_sum = new_effective_coverage_sum = 0.
    detail_extends_visible_full = visible_valid_diff = 0
    with (INDEX/'records.jsonl').open('rb') as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for index in indices:
                row = json.loads(data[offsets[index]:offsets[index+1]])
                old = sampled_text_views(row['caption'], 'random_k', 0, 0, index+1000)
                new = sampled_text_views(row['caption'], 'summary_detail', 0, 0, index+1000)
                assert torch.equal(old['tokens_f'], new['tokens_f'])
                assert old['views'][0] == new['views'][0]
                hist['actual_full_tokens'][int(new['tokens_f'].argmax())+1] += 1
                old_valid += old['valid']; raw_valid += new['valid']
                visible_valid_diff += old['valid'] != new['valid']
                if new['valid']:
                    hist['new_summary_sentences'][1] += 1
                    hist['new_detail_sentences'][new['n']-1] += 1
                    denominator = new['untruncated_lengths'][2]-2
                    assert denominator > 0
                    old_coverage_sum += (old['untruncated_lengths'][2]-2)/denominator if old['valid'] else 0.
                    new_coverage_sum += 1.
                    new_effective_coverage_sum += (int(new['tokens_e'].argmax())-1)/denominator
                    hist['new_summary_tokens'][new['untruncated_lengths'][1]] += 1
                    hist['new_detail_tokens'][new['untruncated_lengths'][2]] += 1
                    detail_extends_visible_full += len(new['views'][2]) > len('. '.join(new['views'][0].split('. ')[1:]))
                if old['valid']:
                    hist['old_prefix_tokens'][old['untruncated_lengths'][1]] += 1
                    hist['old_remainder_tokens'][old['untruncated_lengths'][2]] += 1
                    hist['old_prefix_sentences'][old['K']] += 1
                    hist['old_remainder_sentences'][old['n']-old['K']] += 1
    return {'histograms': hist, 'old_valid': old_valid, 'raw_valid': raw_valid,
            'old_coverage_sum': old_coverage_sum, 'new_coverage_sum': new_coverage_sum,
            'new_effective_coverage_sum': new_effective_coverage_sum,
            'detail_extends_visible_full': detail_extends_visible_full,
            'visible_valid_diff': visible_valid_diff, 'count': len(indices)}


def merge(records):
    result = {'histograms': {}}
    for record in records:
        for name, counter in record['histograms'].items():
            result['histograms'].setdefault(name, Counter()).update(counter)
        for k, v in record.items():
            if k not in ('histograms', 'bounds'):
                result[k] = result.get(k, 0)+v
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True); EXP.mkdir(parents=True, exist_ok=True)
    meta = json.loads((INDEX/'metadata.json').read_text()); n = meta['training_records']
    assert n == 1245901
    started = time.time()
    # A separate audit process with local sampling RNG; never changes training RNG.
    subset = sorted(random.Random(0).sample(range(n), 100000))
    fixed1000 = random.Random(0).sample(range(n), 1000)
    offsets = np.load(INDEX/'offsets.npy', mmap_mode='r')
    digest = hashlib.sha256()
    with (INDEX/'records.jsonl').open('rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for i in fixed1000:
                row = json.loads(data[offsets[i]:offsets[i+1]])
                a = sampled_text_views(row['caption'], 'random_k', 0, 0, i+1000)
                b = sampled_text_views(row['caption'], 'summary_detail', 0, 0, i+1000)
                assert a['views'][0] == b['views'][0] and torch.equal(a['tokens_f'], b['tokens_f'])
                digest.update(str(i+1000).encode()); digest.update(b['tokens_f'].numpy().tobytes())
    check = {'passed': True, 'samples': 1000, 'F_raw_equal_count': 1000,
             'F_tokens_bitwise_equal_count': 1000, 'seed': 0, 'digest': digest.hexdigest()}
    (EXP/'evidence/F_1000_EQUIVALENCE.json').write_text(json.dumps(check, indent=2)+'\n')
    print(json.dumps({'stage': 'F-equivalence', **check}), flush=True)
    context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        chunks = [(start, min(start+10000, n)) for start in range(0, n, 10000)]
        futures = [pool.submit(whole_chunk, c) for c in chunks]
        pieces = []
        for future in as_completed(futures):
            pieces.append(future.result())
            if len(pieces) % 10 == 0 or len(pieces) == len(chunks):
                print(json.dumps({'stage': 'whole-corpus', 'chunks_done': len(pieces),
                                  'chunks_total': len(chunks), 'elapsed_seconds': time.time()-started}), flush=True)
        whole = merge(pieces)
        assert sum(whole['histograms']['sentences'].values()) == n and whole['empty'] == 0
        (RUN/'whole-corpus-histograms.json').write_text(json.dumps(whole, indent=2)+'\n')
        futures = [pool.submit(subset_chunk, subset[s:s+1000]) for s in range(0, len(subset), 1000)]
        pieces = []
        for future in as_completed(futures):
            pieces.append(future.result())
            if len(pieces) % 10 == 0:
                print(json.dumps({'stage': 'matched-100k', 'chunks_done': len(pieces),
                                  'chunks_total': len(futures)}), flush=True)
        sample = merge(pieces)
    result = {'status': 'COMPLETE', 'corpus': meta, 'training_records': n, 'parser': 'unchanged literal . split and cleanup',
        'token_length_definition': 'Actual CLIP BPE, SOT/EOT included; coverage excludes SOT/EOT',
        'Full_definition': 'Exactly baseline visible-prefix F; whole-corpus raw-F length is separately labeled',
        'local_view_definition': 'Summary=first raw cleaned segment; Detail=all subsequent raw segments compactly retokenized with248 truncation',
        'sentence_count': distribution(whole['histograms']['sentences'], [10,25,50,75,90,95]),
        'single_sentence_fraction': whole['histograms']['sentences'].get(1,0)/n,
        'two_sentence_fraction': whole['histograms']['sentences'].get(2,0)/n,
        'three_plus_sentence_fraction': sum(v for k,v in whole['histograms']['sentences'].items() if k>=3)/n,
        'summary_tokens_before_truncation': distribution(whole['histograms']['summary_tokens'], [90,95,99]),
        'detail_tokens_before_truncation_valid_only': distribution(whole['histograms']['detail_tokens'], [90,95,99]),
        'full_raw_tokens_before_packing': distribution(whole['histograms']['full_raw_tokens'], [90,95,99]),
        'summary_truncated_fraction_valid': whole['summary_truncated']/whole['valid'],
        'detail_truncated_fraction_valid': whole['detail_truncated']/whole['valid'],
        'raw_valid_fraction': whole['valid']/n,
        'F_equivalence': check, 'subset_count': len(subset),
        'subset_sha256': hashlib.sha256(json.dumps(subset, separators=(',', ':')).encode()).hexdigest(),
        'subset_epoch': 0, 'sampling_seed': 0,
        'matched_subset': {k: distribution(v, [90,95,99]) for k,v in sample['histograms'].items()},
        'random_k_valid_fraction_subset': sample['old_valid']/sample['count'],
        'summary_detail_valid_fraction_subset': sample['raw_valid']/sample['count'],
        'old_R_non_summary_detail_coverage_mean': sample['old_coverage_sum']/sample['raw_valid'],
        'new_D_detail_coverage_before_truncation_mean': sample['new_coverage_sum']/sample['raw_valid'],
        'new_D_effective_detail_coverage_after_truncation_mean': sample['new_effective_coverage_sum']/sample['raw_valid'],
        'detail_extends_visible_F_fraction_subset': sample['detail_extends_visible_full']/sample['raw_valid'],
        'validity_difference_count_subset': sample['visible_valid_diff'],
        'elapsed_seconds': time.time()-started, 'audit_workers': args.workers}
    (EXP/'SAMPLING_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    (RUN/'sampling-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'COMPLETE', 'old_R_coverage': result['old_R_non_summary_detail_coverage_mean'],
                      'new_D_coverage': result['new_D_effective_detail_coverage_after_truncation_mean'],
                      'detail_truncated_fraction': result['detail_truncated_fraction_valid']}), flush=True)


if __name__ == '__main__':
    main()
