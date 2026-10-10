"""Retain the original COCO numerical protocol and expose streaming GPU differences."""
import argparse
import json
from pathlib import Path
import torch
from eval.retrieval.coco_retrieval import _top_indices
from tools.eval_hyfl_native import rows_and_images
from tools.retrieval_bounded import ShardCache, atomic_json, sha


def run(runtime, output):
    runtime, output = Path(runtime), Path(output); torch.set_num_threads(4)
    result_path = runtime / 'COCO/attempt1/RESULT.json'
    receipt = json.loads(result_path.read_text())
    job = json.loads((result_path.parent / 'JOB.json').read_text())
    rows, images = rows_and_images(job['manifest'])
    cache = ShardCache(runtime / 'cache/COCO', receipt['identity'])
    banks = []
    for kind, records in [('image', images), ('text', rows)]:
        xs = []
        for start in range(0, len(records), 64):
            ids = [r[0] for r in records[start:start + 64]] if kind == 'image' else [r['caption_id'] for r in records[start:start + 64]]
            x = cache.read(kind, start, ids)
            if x is None:
                raise ValueError('missing actual feature shard')
            xs.append(x)
        banks.append(torch.nn.functional.normalize(torch.cat(xs), dim=-1))
    im, tx = banks; mapping = {r[0]: i for i, r in enumerate(images)}
    image_truth = [set() for _ in images]
    for j, r in enumerate(rows):
        image_truth[mapping[r['positive_image_id']]].add(j)
    text_truth = [{mapping[r['positive_image_id']]} for r in rows]
    streamed = torch.load(result_path.with_suffix('.queries.pt'), weights_only=True)
    metrics, cases = {}, {}
    for direction, q, g, truth in [('I2T', im, tx, image_truth), ('T2I', tx, im, text_truth)]:
        all_top, all_scores = [], []
        for start in range(0, len(q), 512):
            scores = q[start:start + 512] @ g.T
            top = _top_indices(scores, 11)
            all_top.append(top); all_scores.append(scores.gather(1, top))
        top = torch.cat(all_top); scores = torch.cat(all_scores)
        counts = {}; cases[direction] = {}
        for k in (1, 5, 10):
            hits = [bool(set(r[:k].tolist()) & truth[i]) for i, r in enumerate(top)]
            counts[str(k)] = sum(hits)
            other = streamed[direction]['hits'][str(k)]
            diff = [i for i, (a, b) in enumerate(zip(hits, other)) if a != b]
            cases[direction][str(k)] = [{
                'query_index': i, 'query_id': images[i][0] if direction == 'I2T' else rows[i]['caption_id'],
                'canonical_hit': hits[i], 'streaming_gpu_hit': other[i],
                'positive_indices': sorted(truth[i]), 'canonical_top11': top[i].tolist(),
                'canonical_scores': scores[i].tolist(), 'gpu_top11': streamed[direction]['top_indices'][i].tolist(),
                'gpu_scores': streamed[direction]['top_scores'][i].tolist()} for i in diff]
        assert counts == receipt['legacy_regression']['actual_counts'][direction] == receipt['legacy_regression']['expected_counts'][direction]
        metrics[direction] = {'query_count': len(q), 'candidate_count': len(g), 'correct': counts,
                              'recall_percent': {k: 100 * v / len(q) for k, v in counts.items()}}
    proof = {'status': 'PASS', 'feature_source': str(result_path), 'feature_source_sha256': sha(result_path),
             'checkpoint_sha256': receipt['checkpoint_sha256'], 'manifest_sha256': receipt['identity']['manifest_sha256'],
             'protocol': 'original native COCO CPU FP32 query chunk512, full gallery, per-row1D argsort',
             'why_primary': 'User3.4/5.3 retain previously verified native COCO mathematics/numerical protocol. Not best-score selection.',
             'model_inference_repeated': False, 'historical_scores_copied': False,
             'all_six_canonical_counts_equal_history': True, 'metrics': metrics,
             'streaming_gpu_diagnostic_metrics': receipt['metrics'], 'query_differences': cases,
             'canonical_evaluator_sha256': sha(Path(__file__).resolve().parents[1] / 'eval/retrieval/coco_retrieval.py')}
    atomic_json(output / 'CANONICAL_COCO_REPROOF.json', proof, exclusive=True)
    print(json.dumps({'metrics': metrics, 'differences': cases}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runtime', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.runtime, a.output)
