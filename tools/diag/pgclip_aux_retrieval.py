"""Auxiliary-branch retrieval evaluation for PG-CLIP v0.1 (read-only diagnostic).

The frozen promotion protocol uses **only** ``Normalize(encode_image(I))`` and
``Normalize(encode_text(C))`` (native CLS/EOS, no gate, no fusion, no reranking). This tool adds a
second, explicitly diagnostic protocol that scores retrieval with the **pre-projection conditional
path**:

    native    QG[i, j] = 100 * <Norm(h_i @ W),            t̂_j>
    auxiliary QP[i, j] = 100 * <Norm((h_i * mask_j) @ W), t̂_j>

``mask_j`` is the hard straight-through gate of candidate caption ``j``, so the rule is the training
one: for I2T every candidate text uses its own mask, and for T2I a fixed text keeps its single mask
(the two directions read the same matrix, exactly as the objective does). Both paths are computed
from the *same* encoded features, in one pass, so the comparison is paired.

Read-only by construction: no optimizer, no parameter update (the CLIP and gate digests are checked
before and after), one small JSON written into the run directory, and the output marked
``not_a_gate_candidate`` so it can never be mistaken for the frozen evaluation.

    python tools/diag/pgclip_aux_retrieval.py --checkpoint <pgclip_..._step000500.pt> \
        --out-dir <run dir> --dataset both
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (os.path.join(REPO, 'train'), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.pgclip import (FIXED_SCALE, NORM_EPS, OBJECTIVE, PreProjectionGate,  # noqa: E402
                          file_sha256, mask_statistics, native_scores, state_digest)
from said_cvssl_data import reference_view_a_transform                        # noqa: E402

COCO_ROOT = os.environ.get('COCO_DATA_ROOT', '/root/datasets/coco')
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
COCO_CAPTIONS_PER_IMAGE = 5
RANK_KS = (1, 5, 10)
URBAN_HELPER = '/root/SAID-gap-completion/tools/urban1k_retrieval.py'


# --------------------------------------------------------------------------- scores
def auxiliary_scores(h: torch.Tensor, W: torch.Tensor, t_unit: torch.Tensor, masks: torch.Tensor,
                     image_chunk: int = 1024, text_chunk: int = 256, progress=None) -> torch.Tensor:
    """``QP[i, j] = 100 * <Norm((h_i * mask_j) @ W), t̂_j>`` in exact fp32 blocks.

    Rows are images, columns are candidate captions; the mask of a column belongs to that caption
    only. No ``[B, B, 768]`` activation is ever resident: the masked product is built per
    (text block, image block) pair. The whole score computation runs under ``no_grad`` -- this is a
    read-only evaluation, and building an autograd graph per block would keep every intermediate
    alive (the first version of this tool leaked ~160 MB per block and ran the GPU out of memory).
    """
    for tensor, name in ((h, 'h'), (W, 'W'), (t_unit, 't_unit'), (masks, 'masks')):
        if tensor.dtype != torch.float32:
            raise TypeError('%s must be fp32, got %s' % (name, tensor.dtype))
    images, texts = h.shape[0], t_unit.shape[0]
    if masks.shape[0] != texts:
        raise ValueError('one mask per candidate text is required (%d masks, %d texts)'
                         % (masks.shape[0], texts))
    out = torch.empty(images, texts, dtype=torch.float32, device=h.device)
    with torch.no_grad(), torch.autocast(device_type=h.device.type, enabled=False):
        blocks = 0
        for j0 in range(0, texts, text_chunk):
            mask_block = masks[j0:j0 + text_chunk]
            t_block = t_unit[j0:j0 + text_chunk]
            for i0 in range(0, images, image_chunk):
                h_block = h[i0:i0 + image_chunk]
                masked = h_block.unsqueeze(0) * mask_block.unsqueeze(1)       # [bj, bi, 768]
                u = F.normalize(masked @ W, dim=-1, eps=NORM_EPS)             # [bj, bi, 512]
                block = FIXED_SCALE * (u * t_block.unsqueeze(1)).sum(-1)      # [bj, bi]
                out[i0:i0 + image_chunk, j0:j0 + text_chunk] = block.t()
                blocks += 1
            if progress is not None:
                progress(j0 + text_chunk, texts, blocks)
    return out


def naive_auxiliary_scores(h, W, t_unit, masks):
    """Pair-by-pair reference (tests only): the same formula with no blocking at all."""
    rows = []
    for index in range(h.shape[0]):
        values = []
        for column in range(t_unit.shape[0]):
            vector = F.normalize((h[index] * masks[column]) @ W, dim=-1, eps=NORM_EPS)
            values.append(FIXED_SCALE * (vector * t_unit[column]).sum())
        rows.append(torch.stack(values))
    return torch.stack(rows)


# --------------------------------------------------------------------------- metrics
def ranks_with_positive_sets(scores: torch.Tensor,
                             positive_columns: torch.Tensor) -> torch.Tensor:
    """Rank of the best positive per row, with the repository tie rule.

    ``positive_columns`` is [rows, P] and may be padded with -1 for rows that have fewer positives
    (COCO gives every image five captions, Urban-1k one). A row's rank is

        rank = 1 + #{j : s_j > max_positive} + #{j < min_positive_index : s_j == max_positive}

    i.e. ties are broken by the candidate index, exactly like the repository convention
    (``id_j < id_pos`` ranks ahead). The multi-positive set is the historical COCO convention: any of
    an image's five captions counts as a hit.
    """
    if scores.dim() != 2:
        raise ValueError('scores must be a matrix, got shape %s' % (tuple(scores.shape),))
    rows, columns = scores.shape
    valid = positive_columns >= 0
    if positive_columns.shape[0] != rows:
        raise ValueError('one positive set per row is required (%d sets, %d rows)'
                         % (positive_columns.shape[0], rows))
    safe = positive_columns.clamp_min(0)
    gathered = scores.gather(1, safe).masked_fill(~valid, float('-inf'))
    positive_max = gathered.max(dim=1).values
    better = (scores > positive_max.unsqueeze(1)).sum(dim=1)
    first_positive = safe.masked_fill(~valid, columns).min(dim=1).values
    index = torch.arange(columns, device=scores.device).unsqueeze(0)
    tie_before = ((scores == positive_max.unsqueeze(1)) & (index < first_positive.unsqueeze(1))
                  ).sum(dim=1)
    return 1 + better + tie_before


def diagonal_positive_columns(size: int, device=None) -> torch.Tensor:
    return torch.arange(size, device=device).unsqueeze(1)


def retrieval_metrics(scores: torch.Tensor, i2t_positives: torch.Tensor = None,
                      t2i_positives: torch.Tensor = None, rank_ks=RANK_KS) -> dict:
    """I2T and T2I recall at k for a [images, texts] score matrix.

    ``i2t_positives`` is [images, P] (the candidate columns that count as that image's captions) and
    ``t2i_positives`` is [texts, P] (the images that count for that caption). For a square matrix
    both default to the diagonal.
    """
    images, texts = scores.shape
    if i2t_positives is None:
        if images != texts:
            raise ValueError('non-square scores need explicit positives: %s'
                             % ((images, texts),))
        i2t_positives = diagonal_positive_columns(images, scores.device)
    if t2i_positives is None:
        if images != texts:
            raise ValueError('non-square scores need explicit positives: %s'
                             % ((images, texts),))
        t2i_positives = diagonal_positive_columns(texts, scores.device)
    i2t = ranks_with_positive_sets(scores, i2t_positives)
    t2i = ranks_with_positive_sets(scores.t().contiguous(), t2i_positives)
    metrics = {'n_images': int(images), 'n_texts': int(texts)}
    for k in rank_ks:
        metrics['i2t_r%d' % k] = float((i2t <= k).float().mean())
        metrics['t2i_r%d' % k] = float((t2i <= k).float().mean())
    metrics['i2t_mrr'] = float((1.0 / i2t.float()).mean())
    metrics['t2i_mrr'] = float((1.0 / t2i.float()).mean())
    metrics['i2t_rank_mean'] = float(i2t.float().mean())
    metrics['t2i_rank_mean'] = float(t2i.float().mean())
    return metrics


def paired_outcome_from_ranks(native_ranks: torch.Tensor, auxiliary_ranks: torch.Tensor,
                              rank_ks=RANK_KS) -> dict:
    change = auxiliary_ranks - native_ranks
    block = {'rank_improved': int((change < 0).sum()),
             'rank_unchanged': int((change == 0).sum()),
             'rank_worsened': int((change > 0).sum()),
             'max_rank_worsening': int(change.max()) if change.numel() else 0,
             'max_rank_improvement': int(-change.min()) if change.numel() else 0,
             'mean_rank_change': float(change.float().mean())}
    for k in rank_ks:
        block['hits_gained_r%d' % k] = int(((native_ranks > k) & (auxiliary_ranks <= k)).sum())
        block['hits_lost_r%d' % k] = int(((native_ranks <= k) & (auxiliary_ranks > k)).sum())
        block['hit_count_change_r%d' % k] = (block['hits_gained_r%d' % k]
                                             - block['hits_lost_r%d' % k])
    return block


def paired_outcome(native_scores: torch.Tensor, auxiliary_scores: torch.Tensor,
                   i2t_positives: torch.Tensor = None, t2i_positives: torch.Tensor = None,
                   rank_ks=RANK_KS) -> dict:
    """Discrete paired evidence per direction (hits gained/lost, rank improved/same/worse)."""
    images, texts = native_scores.shape
    if i2t_positives is None and images == texts:
        i2t_positives = diagonal_positive_columns(images, native_scores.device)
    if t2i_positives is None and images == texts:
        t2i_positives = diagonal_positive_columns(texts, native_scores.device)
    if i2t_positives is None or t2i_positives is None:
        raise ValueError('non-square scores need explicit positives')
    return {
        'I2T': paired_outcome_from_ranks(
            ranks_with_positive_sets(native_scores, i2t_positives),
            ranks_with_positive_sets(auxiliary_scores, i2t_positives), rank_ks),
        'T2I': paired_outcome_from_ranks(
            ranks_with_positive_sets(native_scores.t().contiguous(), t2i_positives),
            ranks_with_positive_sets(auxiliary_scores.t().contiguous(), t2i_positives), rank_ks),
    }


# --------------------------------------------------------------------------- data
def coco_pairs(limit_images=None):
    """``(image_paths, captions, i2t_positives, t2i_positives)`` in the frozen canonical order.

    The paths are rebuilt from the annotation index instead of holding thousands of open PIL images
    in memory; the order is the sorted image-id order that ``CocoCaptions`` itself uses. Every image
    owns its first five caption columns, so I2T has five positives per row and T2I has one image per
    caption -- the historical COCO convention.
    """
    from torchvision.datasets import CocoCaptions
    root = os.path.join(COCO_ROOT, 'val2017')
    annotations = os.path.join(COCO_ROOT, 'annotations', 'captions_val2017.json')
    if not os.path.isdir(root) or not os.path.isfile(annotations):
        raise SystemExit('COCO val2017 not found under %s' % COCO_ROOT)
    dataset = CocoCaptions(root=root, annFile=annotations, transform=None)
    paths, captions, per_image = [], [], []
    for index, image_id in enumerate(sorted(dataset.coco.imgs.keys())):
        if limit_images is not None and index >= limit_images:
            break
        paths.append(os.path.join(dataset.root, dataset.coco.imgs[image_id]['file_name']))
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        entries = dataset.coco.loadAnns(ann_ids)[:COCO_CAPTIONS_PER_IMAGE]
        captions.extend([entry['caption'] for entry in entries])
        per_image.append(len(entries))
    i2t_positives = []
    column = 0
    for count in per_image:
        i2t_positives.append(list(range(column, column + count)))
        column += count
    t2i_positives = []
    for image_index, count in enumerate(per_image):
        t2i_positives.extend([[image_index] for _ in range(count)])
    return paths, captions, i2t_positives, t2i_positives


def urban_pairs(limit_images=None):
    """``(image_paths, captions, i2t_positives, t2i_positives)`` from the upstream Urban-1k layout."""
    import importlib.util
    if not os.path.isfile(URBAN_HELPER):
        raise SystemExit('the Urban-1k helper is missing at %s' % URBAN_HELPER)
    spec = importlib.util.spec_from_file_location('urban1k_retrieval', URBAN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = module.image_caption_pairs(URBAN_ROOT)
    if limit_images is not None:
        pairs = pairs[:limit_images]
    captions = module.read_captions(pairs)
    paths = [image for image, _caption in pairs]
    i2t_positives = [[index] for index in range(len(paths))]
    t2i_positives = [[index] for index in range(len(captions))]
    return paths, list(captions), i2t_positives, t2i_positives


# --------------------------------------------------------------------------- encoding
def encode_images(model, preprocess, paths, device, batch_size, limit=None):
    from PIL import Image
    hidden, native = [], []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = torch.stack([
                preprocess(item.convert('RGB') if hasattr(item, 'convert')
                           else Image.open(item).convert('RGB'))
                for item in batch_paths]).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                h_pre, v_raw = model.encode_image_with_preprojection(tensors)
            with torch.autocast(device_type='cuda', enabled=False):
                hidden.append(h_pre.float().cpu())
                native.append(v_raw.float().cpu())
    return torch.cat(hidden), torch.cat(native)


def encode_texts(model, gate, captions, device, batch_size):
    """``(t_unit, masks, statistics)`` from ONE text pass per batch and ONE gate evaluation."""
    features, masks, probabilities = [], [], []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            batch = captions[start:start + batch_size]
            tokens = longclip.tokenize(batch, truncate=True).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                hidden = model.encode_text_final_hidden(tokens)
            with torch.autocast(device_type='cuda', enabled=False):
                hidden = hidden.float()
                eot = model.eot_indices(tokens)[0]
                raw = hidden[torch.arange(hidden.shape[0], device=device), eot] \
                    @ model.text_projection.float()
                features.append(F.normalize(raw, dim=-1, eps=NORM_EPS).cpu())
                batch_masks, batch_probabilities = gate(hidden)
                masks.append(batch_masks.cpu())
                probabilities.append(batch_probabilities.float().cpu())
    stacked_masks = torch.cat(masks)
    stacked_probabilities = torch.cat(probabilities)
    return torch.cat(features), stacked_masks, mask_statistics(stacked_masks.to(device),
                                                               stacked_probabilities.to(device))


def evaluate_split(model, gate, preprocess, name, paths, captions, device, image_batch,
                   text_batch, image_chunk, text_chunk, i2t_positives, t2i_positives):
    started = time.time()
    h, native_features = encode_images(model, preprocess, paths, device, image_batch)
    t_unit, masks, mask_stats = encode_texts(model, gate, captions, device, text_batch)
    h = h.to(device)
    native_features = F.normalize(native_features.to(device), dim=-1, eps=NORM_EPS)
    t_unit = t_unit.to(device)
    masks = masks.to(device)
    i2t_tensor = _padded_positive_columns(i2t_positives, device)
    t2i_tensor = _padded_positive_columns(t2i_positives, device)

    def progress(done, total, blocks):
        print('[aux] %s: %d/%d texts (%d blocks, %.2f GiB allocated)'
              % (name, done, total, blocks, torch.cuda.memory_allocated() / 2 ** 30), flush=True)

    with torch.no_grad():
        native = native_scores(h, model.visual.proj.float(), t_unit)
        auxiliary = auxiliary_scores(h, model.visual.proj.float(), t_unit, masks,
                                     image_chunk=image_chunk, text_chunk=text_chunk,
                                     progress=progress if name == 'coco_val2017' else None)
    native_metrics = retrieval_metrics(native, i2t_tensor, t2i_tensor)
    auxiliary_metrics = retrieval_metrics(auxiliary, i2t_tensor, t2i_tensor)
    deltas = {key: auxiliary_metrics[key] - native_metrics[key]
              for key in native_metrics if key.startswith(('i2t_r', 't2i_r'))}
    return {
        'dataset': name, 'images': len(paths), 'texts': len(captions),
        'native': native_metrics, 'auxiliary': auxiliary_metrics, 'delta': deltas,
        'paired_native_vs_auxiliary': paired_outcome(native, auxiliary, i2t_tensor, t2i_tensor),
        'mask_statistics_over_evaluated_texts': mask_stats,
        'wall_seconds': time.time() - started,
    }


def _padded_positive_columns(positive_lists, device):
    """``[rows, max(P)]`` index tensor padded with -1 (rows without a positive are impossible here)."""
    width = max(len(entry) for entry in positive_lists)
    padded = torch.full((len(positive_lists), width), -1, dtype=torch.long)
    for index, entry in enumerate(positive_lists):
        padded[index, :len(entry)] = torch.tensor(entry, dtype=torch.long)
    return padded.to(device)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dataset', default='both', choices=['coco', 'urban1k', 'both'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--image_batch', type=int, default=128)
    parser.add_argument('--text_batch', type=int, default=256)
    parser.add_argument('--image_chunk', type=int, default=1024)
    parser.add_argument('--text_chunk', type=int, default=256)
    parser.add_argument('--limit_images', type=int, default=None,
                        help='diagnostic subset only: the pool sizes are always reported, a subset '
                             'is never presented as the frozen protocol')
    parser.add_argument('--expect_steps', type=int, default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('objective') != OBJECTIVE:
        raise SystemExit('not a PG-CLIP checkpoint: objective=%r' % payload.get('objective'))
    if args.expect_steps is not None and int(payload.get('completed_steps', -1)) != args.expect_steps:
        raise SystemExit('completed_steps=%r, expected %d' % (payload.get('completed_steps'),
                                                              args.expect_steps))
    gate_state = payload.get('gate_state')
    if not isinstance(gate_state, dict) or not gate_state:
        raise SystemExit('this checkpoint carries no gate tensors (%s), so the auxiliary branch '
                         'cannot be evaluated: with the initialisation gate every mask is exactly 1 '
                         'and the auxiliary scores are provably identical to the native ones. '
                         'Checkpoints written before the gate_state fix cannot be used here.'
                         % ('gate' if 'gate' in payload else 'no gate key'))
    device = torch.device(args.device)
    model, preprocess = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                                args=argparse.Namespace())
    model.load_state_dict(payload['clip'], strict=True)
    model = model.to(device).eval()
    gate = PreProjectionGate(device=device).to(device)
    gate.load_state_dict(gate_state)
    gate.eval()

    clip_digest_before = state_digest(model.state_dict())
    gate_digest_before = state_digest(gate.state_dict())
    splits = []
    if args.dataset in ('coco', 'both'):
        splits.append(('coco_val2017',) + coco_pairs(args.limit_images))
    if args.dataset in ('urban1k', 'both'):
        splits.append(('urban1k',) + urban_pairs(args.limit_images))

    results = {}
    for name, paths, captions, i2t_positives, t2i_positives in splits:
        print('[aux] %s: %d images x %d texts' % (name, len(paths), len(captions)), flush=True)
        results[name] = evaluate_split(model, gate, preprocess, name, paths, captions, device,
                                       args.image_batch, args.text_batch, args.image_chunk,
                                       args.text_chunk, i2t_positives, t2i_positives)
        summary = results[name]
        print('[aux] %s native I2T R@1 %.4f T2I R@1 %.4f | auxiliary I2T R@1 %.4f T2I R@1 %.4f'
              % (name, summary['native']['i2t_r1'], summary['native']['t2i_r1'],
                 summary['auxiliary']['i2t_r1'], summary['auxiliary']['t2i_r1']), flush=True)

    clip_digest_after = state_digest(model.state_dict())
    gate_digest_after = state_digest(gate.state_dict())
    if clip_digest_before != clip_digest_after or gate_digest_before != gate_digest_after:
        raise SystemExit('the parameter digest changed: this tool must stay read-only')

    report = {
        'probe': 'pgclip_aux_retrieval', 'read_only': True, 'new_optimizer_updates': 0,
        'not_a_gate_candidate': True,
        'note': ('diagnostic only: the frozen promotion protocol uses native CLS/EOS without the '
                 'gate, without path fusion and without reranking. This file reports the '
                 'pre-projection conditional path on the same features and must never be read as '
                 'the frozen evaluation.'),
        'protocol': {
            'native': 'QG[i,j] = 100 * <Norm(h_i @ W), t_j>',
            'auxiliary': 'QP[i,j] = 100 * <Norm((h_i * mask_j) @ W), t_j>',
            'mask_source': 'the hard straight-through gate of candidate caption j (I2T); one mask '
                           'per fixed text for T2I',
            'precision': 'bf16 autocast for the trunks, fp32 for the pre-projection core '
                         '(same convention as training)',
            'tie_rule': 'rank = 1 + #{j : score_j > score_pos} + #{j < pos : score_j == score_pos}',
            'pool': {name: {'images': block['images'], 'texts': block['texts']}
                     for name, block in results.items()},
            'subset': args.limit_images is not None,
            'limit_images': args.limit_images,
            'canonical_reference_protocol': {'coco': '5000 images x 25000 captions',
                                             'urban1k': '1000 images x 1000 captions'},
        },
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'completed_steps': payload.get('completed_steps'),
        'arm': payload.get('arm'), 'objective': payload.get('objective'),
        'implementation_sha': payload.get('git_head'),
        'clip_state_digest': clip_digest_after,
        'gate_state_digest': gate_digest_after,
        'results': results,
        'not_run': ['any training or optimizer update',
                    'any change to the frozen promotion verdict',
                    'path fusion, reranking, or using the auxiliary scores as the primary metric'],
    }
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pgclip_aux_retrieval.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('WROTE %s' % path)
    for name, block in results.items():
        print('AUX_RETRIEVAL %s %s' % (name, json.dumps({
            'images': block['images'], 'texts': block['texts'],
            'native_i2t_r1': block['native']['i2t_r1'],
            'native_t2i_r1': block['native']['t2i_r1'],
            'auxiliary_i2t_r1': block['auxiliary']['i2t_r1'],
            'auxiliary_t2i_r1': block['auxiliary']['t2i_r1'],
            'delta_i2t_r1_points': block['delta']['i2t_r1'] * 100.0,
            'delta_t2i_r1_points': block['delta']['t2i_r1'] * 100.0,
            'mask_kept_mean': block['mask_statistics_over_evaluated_texts']['mask_kept_mean'],
        }, sort_keys=True)))


if __name__ == '__main__':
    main()
