#!/usr/bin/env python3
"""Pair-conditioned mask-space retrieval for TI-fast checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from model import longclip
from model.nested_semantic_mask import JointMaskAdapter, pair_mask, pair_scores, positive_masks


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized_metrics(scores, image_ids, positive_ids):
    """Match the frozen extended evaluator while computing top-10 only once."""
    n_images, n_texts = scores.shape
    image_index = {str(value): index for index, value in enumerate(image_ids)}
    positives_by_image = {str(value): [] for value in image_ids}
    for text_index, positive in enumerate(positive_ids):
        positives_by_image.setdefault(str(positive), []).append(text_index)
    top_i2t = scores.topk(min(10, n_texts), dim=1, largest=True, sorted=True).indices.cpu()
    top_t2i = scores.t().topk(min(10, n_images), dim=1, largest=True, sorted=True).indices.cpu()
    result = {}
    for k in (1, 5, 10):
        i2t = sum(any(int(candidate) in positives_by_image[str(image_ids[i])]
                      for candidate in top_i2t[i, :min(k, n_texts)])
                  for i in range(n_images)) / n_images
        t2i = sum(image_index.get(str(positive_ids[j]), -1) in
                  set(map(int, top_t2i[j, :min(k, n_images)]))
                  for j in range(n_texts)) / n_texts
        result[f'R@{k}'] = {'I2T': i2t, 'T2I': t2i}
    return result


def masked_score_tile(images, texts_normalized, image_condition, base_logits,
                      text_condition, adapter):
    masks, _, _ = pair_mask(base_logits, image_condition, text_condition, adapter)
    masked_images = F.normalize(images[:, None].float() * masks, dim=-1, eps=1e-6)
    return 100 * (masked_images * texts_normalized[None]).sum(-1)


def score_diagnostics(scores, image_ids, positive_ids):
    """Score/margin summaries in the same 100*cosine units as masked training."""
    device = scores.device
    image_index = {str(value): index for index, value in enumerate(image_ids)}
    positive_image = torch.tensor([image_index[str(value)] for value in positive_ids],
                                  dtype=torch.long, device=device)
    text_index = torch.arange(len(positive_ids), device=device)
    positive_scores = scores[positive_image, text_index]
    by_image = {str(value): [] for value in image_ids}
    for index, value in enumerate(positive_ids):
        by_image[str(value)].append(index)
    negative_i2t = scores.clone()
    for image_i, value in enumerate(image_ids):
        negative_i2t[image_i, by_image[str(value)]] = -torch.inf
    hardest_i2t = negative_i2t.max(dim=1).values
    positive_i2t = torch.stack([
        scores[i, by_image[str(value)]].max() for i, value in enumerate(image_ids)])
    negative_t2i = scores.clone()
    negative_t2i[positive_image, text_index] = -torch.inf
    hardest_t2i = negative_t2i.max(dim=0).values
    positive_t2i = positive_scores

    def summary(values):
        values = values.float()
        quantiles = torch.quantile(values, torch.tensor([.05, .5, .95], device=device))
        return {'mean': float(values.mean()), 'std': float(values.std(unbiased=False)),
                'p05': float(quantiles[0]), 'median': float(quantiles[1]),
                'p95': float(quantiles[2]), 'min': float(values.min()),
                'max': float(values.max())}

    return {
        'positive_pair_scores': summary(positive_scores),
        'i2t_hardest_negative_scores': summary(hardest_i2t),
        'i2t_positive_minus_hardest_negative': summary(positive_i2t - hardest_i2t),
        't2i_hardest_negative_scores': summary(hardest_t2i),
        't2i_positive_minus_hardest_negative': summary(positive_t2i - hardest_t2i),
    }


def self_test():
    torch.manual_seed(31)
    adapter = JointMaskAdapter(6, 8, 3, 8)
    images, texts = torch.randn(4, 8), torch.randn(5, 8)
    image_condition = adapter.image_condition(torch.randn(4, 6))
    base_logits = torch.randn(5, 8)
    text_condition = adapter.text_condition(torch.randn(5, 8))
    candidate = masked_score_tile(images, F.normalize(texts, dim=-1), image_condition,
                                  base_logits, text_condition, adapter)
    reference, _ = pair_scores(images, texts, image_condition, base_logits, text_condition,
                               adapter, image_chunk=2, text_chunk=3, collect_stats=False)
    torch.testing.assert_close(candidate, reference, atol=1e-5, rtol=1e-5)
    ids = [str(i) for i in range(4)]
    metric = normalized_metrics(torch.eye(4), ids, ids)
    assert all(metric[k][direction] == 1 for k in metric for direction in ('I2T', 'T2I'))
    print(json.dumps({'passed': True, 'max_abs': float((candidate-reference).abs().max())}))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest')
    parser.add_argument('--image-root')
    parser.add_argument('--output')
    parser.add_argument('--native-reference')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-chunk', type=int, default=128)
    parser.add_argument('--text-chunk', type=int, default=128)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    for name in ('checkpoint', 'manifest', 'image_root', 'output'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    images, seen = [], set()
    for row in rows:
        if row['image_id'] not in seen:
            seen.add(row['image_id']); images.append((row['image_id'], row['image_path']))
    image_ids = [item[0] for item in images]
    positive_ids = [row['positive_image_id'] for row in rows]

    started = time.perf_counter()
    model, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu',
                                                args=argparse.Namespace())
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['completed_steps'] == 3651
    model.load_state_dict(checkpoint['model'], strict=True)
    adapter = JointMaskAdapter(int(model.visual.proj.shape[0]),
                               int(model.text_projection.shape[0]), 64,
                               int(model.text_projection.shape[1]))
    adapter.load_state_dict(checkpoint['adapter'], strict=True)
    model = model.float().to(args.device).eval()
    adapter = adapter.float().to(args.device).eval()
    loaded = time.perf_counter()

    text_features, base_logits, text_conditions = [], [], []
    for start in range(0, len(rows), args.batch_size):
        tokens = longclip.tokenize([row['caption'] for row in rows[start:start+args.batch_size]],
                                   truncate=True).to(args.device)
        text, hidden = model.encode_text(tokens, return_full=True)
        eot = tokens.argmax(dim=-1)
        eot_hidden = hidden[torch.arange(len(tokens), device=args.device), eot].float()
        text_features.append(text.float())
        base_logits.append(model.mask_net(hidden.float().detach()))
        text_conditions.append(adapter.text_condition(eot_hidden))
    text_features = torch.cat(text_features)
    base_logits = torch.cat(base_logits)
    text_conditions = torch.cat(text_conditions)
    text_normalized = F.normalize(text_features, dim=-1, eps=1e-6)
    text_done = time.perf_counter()

    image_features, image_conditions = [], []
    for start in range(0, len(images), args.batch_size):
        batch = torch.stack([preprocess(Image.open(Path(args.image_root)/path).convert('RGB'))
                             for _, path in images[start:start+args.batch_size]]).to(args.device)
        image, hidden = model.encode_image(batch, return_hidden=True)
        image_features.append(image.float())
        image_conditions.append(adapter.image_condition(hidden.float()))
    image_features = torch.cat(image_features)
    image_conditions = torch.cat(image_conditions)
    image_done = time.perf_counter()

    native_scores = 100 * F.normalize(image_features, dim=-1, eps=1e-6) @ text_normalized.t()
    native_metrics = normalized_metrics(native_scores, image_ids, positive_ids)
    if args.native_reference:
        reference = json.loads(Path(args.native_reference).read_text())['metrics']
        for metric in ('R@1', 'R@5', 'R@10'):
            for direction in ('I2T', 'T2I'):
                assert abs(native_metrics[metric][direction] - reference[metric][direction]) < 1e-12
    native_diagnostics = score_diagnostics(native_scores, image_ids, positive_ids)
    native_done = time.perf_counter()

    masked_scores = torch.empty((len(images), len(rows)), dtype=torch.float32,
                                device=args.device)
    for image_start in range(0, len(images), args.image_chunk):
        image_stop = min(image_start + args.image_chunk, len(images))
        for text_start in range(0, len(rows), args.text_chunk):
            text_stop = min(text_start + args.text_chunk, len(rows))
            masked_scores[image_start:image_stop, text_start:text_stop] = masked_score_tile(
                image_features[image_start:image_stop], text_normalized[text_start:text_stop],
                image_conditions[image_start:image_stop], base_logits[text_start:text_stop],
                text_conditions[text_start:text_stop], adapter)
    masked_metrics = normalized_metrics(masked_scores, image_ids, positive_ids)
    masked_diagnostics = score_diagnostics(masked_scores, image_ids, positive_ids)
    masked_done = time.perf_counter()

    image_index = {str(value): index for index, value in enumerate(image_ids)}
    positive_image = torch.tensor([image_index[str(value)] for value in positive_ids],
                                  dtype=torch.long, device=args.device)
    positive, probabilities, delta = positive_masks(
        base_logits, image_conditions[positive_image], text_conditions, adapter)
    mask_stats = {
        'positive_keep_ratio': float(positive.abs().mean()),
        'positive_all_open_fraction': float((positive >= .5).all(-1).float().mean()),
        'positive_all_closed_fraction': float((positive < .5).all(-1).float().mean()),
        'positive_probability_mean': float(probabilities.mean()),
        'positive_delta_abs_mean': float(delta.abs().mean()),
    }
    result = {
        'protocol': 'long_dci_pair_conditioned_mask_space_v1',
        'definition': '100*cos(normalize(z_i * hard_st(sigmoid(base_j + delta_ij))), normalize(t_j))',
        'n_images': len(images), 'n_captions': len(rows),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_sha(args.checkpoint),
        'manifest_sha256': file_sha(args.manifest),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'chunks': {'image': args.image_chunk, 'text': args.text_chunk},
        'native_metrics_recomputed': native_metrics,
        'masked_metrics': masked_metrics,
        'delta_pp_masked_minus_native': {
            metric: {direction: 100*(masked_metrics[metric][direction]-native_metrics[metric][direction])
                     for direction in ('I2T', 'T2I')}
            for metric in ('R@1', 'R@5', 'R@10')},
        'native_score_diagnostics_100x_cosine': native_diagnostics,
        'masked_score_diagnostics_100x_cosine': masked_diagnostics,
        'positive_mask_diagnostics': mask_stats,
        'timing_seconds': {'load': loaded-started, 'text_encoding_and_gate': text_done-loaded,
                           'image_encoding': image_done-text_done,
                           'native_scoring_and_metrics': native_done-image_done,
                           'masked_pair_scoring_and_metrics': masked_done-native_done,
                           'total': masked_done-started},
        'device': args.device, 'full_pair_matrix': True, 'native_student_only': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
