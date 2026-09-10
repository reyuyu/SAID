"""Standard, deterministic image<->text retrieval evaluation.

Recall@K definition is identical to the legacy SmartCLIP ``train_utils.eval_coco``:
each image is scored against every text (and vice versa) with L2-normalised
features, and a hit means any of the image's own captions (or the image that owns
a caption) appears in the top-K.

Scoring protocol
----------------
* ``similarity_chunk = 512`` is the **canonical protocol** for this project: the
  full ``num_images x num_texts`` matrix is never materialised, and every model,
  dataset and checkpoint must be evaluated with this same chunk so numbers are
  directly comparable.
* Chunking changes only which similarity rows are computed together. It is *not*
  claimed to be bitwise identical to the full-matrix computation: for a handful of
  near-tied rows (measured: 1 of 5,000 on COCO val2017 i2t R@5) the FP32 GEMM
  block shape can change the last bits of a similarity and therefore the ordering
  at the K boundary. Candidate selection reproduces the legacy rule
  (per-row 1-D ``argsort()[-k:]``) so ties are broken the same way, but a
  near-tie whose value itself shifts by 1 ulp can still flip.
* Fairness therefore comes from fixing the evaluator and the chunk size, not from
  assuming floating-point equality with a different GEMM shape.
"""
import json
import os

import torch
import torch.nn.functional as F

K_VALUES = (1, 5, 10)
DEFAULT_SIMILARITY_CHUNK = 512


def _top_indices(block, k):
    """Top-k indices per row with the legacy ``argsort()[-k:]`` semantics.

    ``train_utils.eval_coco`` sorts **each similarity row on its own** with a 1-D
    ``argsort()``. A batched 2-D ``argsort`` (or ``topk``) can order equal values
    differently, which flips rows whose k-th and (k+1)-th similarities are tied —
    measured on COCO val2017 that is a 1-image difference in i2t R@5. Sorting row
    by row inside the similarity chunk reproduces the legacy selection exactly
    while still never materialising the full similarity matrix.
    """
    width = block.shape[1]
    k = min(k, width)
    rows = []
    for index in range(block.shape[0]):
        order = torch.argsort(block[index])
        rows.append(order[width - k:].flip(0))
    return torch.stack(rows)


def _recall_counts(top_indices, truth_of_row, k_values=K_VALUES):
    """Count hits for each K given pre-computed top indices per row."""
    hits = {k: 0 for k in k_values}
    for row in range(top_indices.shape[0]):
        candidates = top_indices[row].tolist()
        truth = truth_of_row(row)
        for k in k_values:
            if any(candidate in candidates[:k] for candidate in truth):
                hits[k] += 1
    return hits


def _image_to_text_counts(images, texts, captions_per_image, chunk, k_values):
    total = images.shape[0]
    top_k = min(max(k_values), texts.shape[0])
    hits = {k: 0 for k in k_values}
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = images[start:stop] @ texts.t()  # [chunk, num_texts]
        top_indices = _top_indices(block, top_k)
        partial = _recall_counts(
            top_indices,
            lambda row: range((start + row) * captions_per_image, (start + row + 1) * captions_per_image),
            k_values,
        )
        for k in k_values:
            hits[k] += partial[k]
    return hits


def _text_to_image_counts(images, texts, captions_per_image, chunk, k_values):
    total = texts.shape[0]
    top_k = min(max(k_values), images.shape[0])
    hits = {k: 0 for k in k_values}
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = texts[start:stop] @ images.t()  # [chunk, num_images]
        top_indices = _top_indices(block, top_k)
        partial = _recall_counts(
            top_indices,
            lambda row: ((start + row) // captions_per_image,),
            k_values,
        )
        for k in k_values:
            hits[k] += partial[k]
    return hits


def retrieval_metrics(image_features, text_features, captions_per_image=5,
                      similarity_chunk=DEFAULT_SIMILARITY_CHUNK, k_values=K_VALUES):
    """Exact Recall@K for both directions using bounded similarity blocks.

    Args:
        image_features: [N, D] features (normalised internally).
        text_features: [N * captions_per_image, D] features in consecutive caption order.
        captions_per_image: captions per image.
        similarity_chunk: number of similarity rows computed per block
            (``None`` or 0 = a single block). Affects peak memory only.
        k_values: recall cut-offs to report.

    Returns:
        dict with ``image2text_R{k}`` and ``text2image_R{k}`` in [0, 1].
    """
    if captions_per_image < 1:
        raise ValueError('captions_per_image must be >= 1, got %r' % (captions_per_image,))
    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError('features must be 2-D, got %r and %r'
                         % (tuple(image_features.shape), tuple(text_features.shape)))
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError('feature dim mismatch: %d vs %d'
                         % (image_features.shape[1], text_features.shape[1]))

    images = F.normalize(image_features.float(), dim=-1)
    texts = F.normalize(text_features.float(), dim=-1)
    n_images, n_texts = images.shape[0], texts.shape[0]
    if n_texts != n_images * captions_per_image:
        raise ValueError('text features %d != %d images x %d captions'
                         % (n_texts, n_images, captions_per_image))

    chunk = n_images if not similarity_chunk else min(int(similarity_chunk), n_images)
    text_chunk = n_texts if not similarity_chunk else min(int(similarity_chunk), n_texts)

    out = {}
    for k, value in _image_to_text_counts(images, texts, captions_per_image, chunk, k_values).items():
        out['image2text_R%d' % k] = value / float(n_images)
    for k, value in _text_to_image_counts(images, texts, captions_per_image, text_chunk, k_values).items():
        out['text2image_R%d' % k] = value / float(n_texts)
    return out


@torch.inference_mode()
def evaluate_coco(model, preprocess, root=None, ann_file=None, batch_size=64,
                  similarity_chunk=DEFAULT_SIMILARITY_CHUNK, device=None):
    """COCO val2017 retrieval with the standard 5-caption protocol."""
    from torchvision.datasets import CocoCaptions
    from model import longclip

    root = root or os.path.join(os.environ.get('COCO_DATA_ROOT', '../../datasets/coco'), 'val2017')
    ann_file = ann_file or os.path.join(os.path.dirname(root), 'annotations', 'captions_val2017.json')
    dataset = CocoCaptions(root=root, annFile=ann_file, transform=preprocess)

    core = model.module if hasattr(model, 'module') else model
    device = torch.device(device or next(core.parameters()).device)

    image_features = []
    text_features = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, stop)]
        images = torch.stack([item[0] for item in batch]).to(device)
        captions = [caption for _, caps in batch for caption in caps[:5]]
        image_features.append(core.encode_image(images).detach().cpu().float())
        tokens = longclip.tokenize(captions, truncate=True).to(device)
        text_features.append(core.encode_text(tokens).detach().cpu().float())

    return retrieval_metrics(
        torch.cat(image_features),
        torch.cat(text_features),
        captions_per_image=5,
        similarity_chunk=similarity_chunk,
    )


def evaluate_features(image_features, text_features, captions_per_image=5,
                      similarity_chunk=DEFAULT_SIMILARITY_CHUNK):
    """Metrics from pre-computed features (same chunked protocol)."""
    return retrieval_metrics(image_features, text_features,
                             captions_per_image=captions_per_image,
                             similarity_chunk=similarity_chunk)


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
