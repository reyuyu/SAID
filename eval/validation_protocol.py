"""Canonical ShareGPT4V-1K validation protocol (three fixed caption variants).

Protocol
--------
* **Cohort**: the first 1,000 records of the dataset JSON (the audited held-out
  split), one caption per image -> a 1,000-way retrieval.
* **Captions** are generated once and frozen in an immutable manifest
  (``outputs/validation/sharegpt4v1k_manifest.json``); they are never resampled:
    - ``first_sentence`` : the first sentence
    - ``fixed_sparse``   : the training prefix rule (uniform prefix over the
      ``'. '`` split) with a *private* RNG seeded by ``seed + json_index``
    - ``full_dense``     : the full caption
* **Standard retrieval** (I2T / T2I R@1/5/10) uses only ``encode_image`` /
  ``encode_text``. Said features never enter the retrieval ranking; they appear
  only in the representation diagnostics (pair gap, RMG, balancing gain,
  conditioning margin).
* **Canonical similarity chunk**: ``similarity_chunk = 512`` for every model and
  every dataset, so numbers from different runs are comparable.

The manifest is deterministic: rebuilding it from the same JSON and seed yields
the same records, and a stored manifest whose digest/seed/size do not match is
rejected instead of being overwritten.
"""
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval.retrieval.coco_retrieval import DEFAULT_SIMILARITY_CHUNK, retrieval_metrics
from eval.salu.representation_balance_metrics import cyclic_indices, gap_comparison
from eval.salu.representation_probe import file_sha256, training_like_caption, write_json
from model import longclip

VARIANTS = ('first_sentence', 'fixed_sparse', 'full_dense')
MANIFEST_SEED = 26
N_IMAGES = 1000
CANONICAL_SIMILARITY_CHUNK = DEFAULT_SIMILARITY_CHUNK  # 512
PROTOCOL_NAME = 'sharegpt4v1k-fixed-captions-v1'


def first_sentence(dense: str) -> str:
    sentences = [part.strip() for part in dense.replace('\n', ' ').split('. ') if part.strip()]
    if not sentences:
        raise ValueError('empty caption')
    return sentences[0]


def build_manifest(records: List[dict], dataset_sha256: str, n: int = N_IMAGES,
                   seed: int = MANIFEST_SEED) -> dict:
    """Deterministic manifest for the first ``n`` held-out images."""
    if n < 2:
        raise ValueError('cohort must have at least 2 images')
    samples = []
    for index, record in enumerate(records[:n]):
        image_id = record['image']
        if Path(image_id).is_absolute() or '..' in Path(image_id).parts:
            raise ValueError('image IDs must be relative dataset paths')
        dense = record['conversations'][1]['value'].replace('\n', ' ')
        samples.append({
            'json_index': index,
            'image_path': image_id,
            'captions': {
                'first_sentence': first_sentence(dense),
                'fixed_sparse': training_like_caption(dense, seed + index)['caption'],
                'full_dense': dense,
            },
        })
    if len(samples) != n:
        raise ValueError('not enough records: %d < %d' % (len(samples), n))
    return {
        'schema_version': 1,
        'protocol': PROTOCOL_NAME,
        'n': n,
        'seed': seed,
        'variants': list(VARIANTS),
        'dataset_json_sha256': dataset_sha256,
        'similarity_chunk': CANONICAL_SIMILARITY_CHUNK,
        'caption_rules': {
            'first_sentence': 'first sentence of the newline-flattened caption',
            'fixed_sparse': 'training prefix rule with a private RNG seeded by seed + json_index',
            'full_dense': 'complete caption',
        },
        'samples': samples,
    }


def load_or_create_manifest(path, dataset_json, n: int = N_IMAGES, seed: int = MANIFEST_SEED) -> dict:
    """Load the frozen manifest, or build and store it once (never overwrite)."""
    path = Path(path)
    dataset_digest = file_sha256(dataset_json)
    expected = build_manifest(json.loads(Path(dataset_json).read_text(encoding='utf-8')),
                              dataset_digest, n, seed)
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError('existing validation manifest differs; refusing to resample or overwrite')
        return actual
    write_json(path, expected)
    return expected


@contextlib.contextmanager
def rng_guard():
    """Restore every RNG state on exit so validation cannot steer training."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def rng_snapshot() -> dict:
    """Compact digest of every RNG state (for RNG-safety assertions)."""
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode('utf-8'))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode('utf-8'))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(str(numpy_state[2:]).encode('utf-8'))
    digest.update(torch.get_rng_state().numpy().tobytes())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    for state in cuda:
        digest.update(state.numpy().tobytes())
    return {'sha256': digest.hexdigest(), 'cuda_devices': len(cuda)}


@torch.inference_mode()
def evaluate_variant(model, samples: List[dict], image_root, variant: str, preprocess,
                     batch_size: int = 64, similarity_chunk: int = CANONICAL_SIMILARITY_CHUNK,
                     device=None) -> Dict:
    """Standard retrieval + representation diagnostics for one caption variant."""
    if variant not in VARIANTS:
        raise ValueError('unknown caption variant %r' % (variant,))
    device = torch.device(device or next(model.parameters()).device)
    core = model.module if hasattr(model, 'module') else model
    image_root = Path(image_root)

    standard_images, router_images, patches, texts = [], [], [], []
    equivalence = 0.0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        tensors = []
        for sample in chunk:
            with Image.open(image_root / sample['image_path']) as image:
                tensors.append(preprocess(image.convert('RGB')))
        tensor = torch.stack(tensors).to(device)
        standard = core.encode_image(tensor)
        full, patch = core.encode_router_input(tensor)
        if start == 0:
            equivalence = float((standard - full).abs().max())
            if equivalence > 1e-6:
                raise RuntimeError('encode_router_input does not match encode_image (%.3e)' % equivalence)
        standard_images.append(F.normalize(standard.float(), dim=-1).cpu())
        router_images.append(F.normalize(full.float(), dim=-1).cpu())
        patches.append(patch.float().cpu())
        captions = [sample['captions'][variant] for sample in chunk]
        tokens = longclip.tokenize(captions, truncate=True).to(device)
        texts.append(F.normalize(core.encode_text(tokens).float(), dim=-1).cpu())

    z_standard = torch.cat(standard_images)
    z_full = torch.cat(router_images)
    patch_bank = torch.cat(patches)
    text = torch.cat(texts)

    _, said = core.said_router(text.to(device), patch_bank.to(device))
    said = F.normalize(said.float(), dim=-1).cpu()
    shuffled_text = text[cyclic_indices(len(samples))]
    _, said_shuffled = core.said_router(shuffled_text.to(device), patch_bank.to(device))
    said_shuffled = F.normalize(said_shuffled.float(), dim=-1).cpu()

    retrieval = retrieval_metrics(z_standard, text, captions_per_image=1,
                                 similarity_chunk=similarity_chunk)
    comparison = gap_comparison(z_full.numpy(), z_full.numpy(), said.numpy(),
                                text.numpy(), said_shuffled.numpy())
    diagnostics = {
        'full_pair_gap': comparison['full']['pair_gap'],
        'full_rmg': comparison['full']['rmg'],
        'said_pair_gap': comparison['said']['pair_gap'],
        'said_rmg': comparison['said']['rmg'],
        'balancing_gain': comparison['balancing_gain'],
        'relative_balancing_gain': comparison['relative_balancing_gain'],
        'conditioning_margin': comparison['conditioning_gap_margin'],
        'said_own_gap': comparison['said_own_gap'],
        'said_shuffle_gap': comparison['said_shuffle_gap'],
        'l2m_full': comparison['full']['l2m'],
        'l2m_said': comparison['said']['l2m'],
    }
    return {
        'retrieval': retrieval,
        'diagnostics': diagnostics,
        'n': len(samples),
        'caption_variant': variant,
        'similarity_chunk': similarity_chunk,
        'z_full_vs_encode_image_max_abs_diff': equivalence,
        'note': 'Said features are used for diagnostics only and never enter the retrieval ranking',
    }


def evaluate_all_variants(model, samples, image_root, preprocess, variants=VARIANTS,
                          batch_size: int = 64, similarity_chunk: int = CANONICAL_SIMILARITY_CHUNK,
                          device=None) -> Dict[str, Dict]:
    """Run every caption variant inside one RNG guard."""
    results = {}
    with rng_guard():
        for variant in variants:
            started = time.time()
            result = evaluate_variant(model, samples, image_root, variant, preprocess,
                                      batch_size=batch_size, similarity_chunk=similarity_chunk,
                                      device=device)
            result['wall_sec'] = time.time() - started
            results[variant] = result
    return results
