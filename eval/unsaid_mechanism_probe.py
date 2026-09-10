"""Phase 2.8B: offline Unsaid mechanism & reachability probe (read-only).

This script only *evaluates* checkpoints. It never trains, never writes a checkpoint,
never calls ``backward`` into model parameters, and never changes the frozen Minimal
Unsaid definition. The only thing it optimizes is a free per-sample logit vector
``alpha`` of the offline softmax-mixture oracle, whose gradients stop at the detached
patch matrix.

Frozen math being measured (Phase 2.8A, unchanged)::

    s_i = q^T k_i,  A_s = softmax(s / tau_said),  A_u = softmax(-s / tau_unsaid)
    z_s = normalize(sum_i A_s_i h_i),  z_u = normalize(sum_i A_u_i h_i)
    g = sg(normalize(z_G)),  s = sg(normalize(z_S))
    target_u = sg(normalize(g - (g^T s) s)),  valid = ||g - (g^T s) s|| > eps
    L_u = mean_valid[1 - cos(z_u, target_u)]

Reachability diangostics (per valid sample, fp32 by default so bf16 noise cannot
dominate the geometry):

* ``C_current``       cos(z_u, target_u) -- what the trained router currently reaches
* ``C_patch_max``     max_i cos(normalize(h_i), target_u) -- is one patch already aligned
* ``C_span``          ||V_r^T V_r target_u|| for the row space V_r of H (fp32 SVD,
                      rank from a stable singular-value tolerance). Loose upper bound
                      for *arbitrary signed* linear combinations only; it is not what
                      the positive softmax pooling can be assumed to reach.
* ``C_softmax_oracle`` approximate softmax-mixture oracle: optimize only a free logit
                      vector ``alpha`` (Adam, fixed steps/lr, deterministic init from
                      the current Unsaid logits) to maximise cos(z_oracle, target_u).
                      Explicitly an *approximate* oracle, not a theoretical bound.
* ``optimization_gap = C_softmax_oracle - C_current``
* ``positive_mixture_gap = C_span - C_softmax_oracle``

Usage::

    python eval/unsaid_mechanism_probe.py \
      --arm SU \
      --checkpoints "initial:runs_salu/phase28b_arm_SU/salu_initial.pt,step500:runs_salu/phase28b_arm_SU/salu_said_only_last.pt" \
      --output_dir outputs/unsaid_mechanism
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.salu.representation_probe import file_sha256, write_json  # noqa: E402
from eval.validation_protocol import (  # noqa: E402
    CANONICAL_SIMILARITY_CHUNK,
    PROTOCOL_NAME as SHAREGPT4V1K_PROTOCOL,
    VARIANTS,
    load_or_create_manifest,
)
from model import longclip  # noqa: E402
from model import unsaid_core  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402

DEFAULT_NUM_IMAGES = 256
DEFAULT_ORACLE_SAMPLES = 64
DEFAULT_ORACLE_STEPS = 100
DEFAULT_ORACLE_LR = 0.1
PRIMARY_VARIANT = 'first_sentence'
SPAN_TOL_RATIO = 1e-6      # singular values below max(S) * max(N, D) * eps * ratio are dropped


# --------------------------------------------------------------------------- #
# pure reachability helpers
# --------------------------------------------------------------------------- #
def distribution(values) -> Dict[str, Optional[float]]:
    """mean / median / P10 / P90 of a 1-D sequence (``None`` when it is empty)."""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'n': int(array.size),
            'mean': float(array.mean()),
            'median': float(np.median(array)),
            'p10': float(np.percentile(array, 10)),
            'p90': float(np.percentile(array, 90))}


def best_patch_cosine(patches: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``max_i cos(normalize(h_i), target)`` per sample: [B]."""
    if patches.dim() != 3 or target.dim() != 2:
        raise ValueError('expected patches [B, N, D] and target [B, D], got %r and %r'
                         % (tuple(patches.shape), tuple(target.shape)))
    unit_patches = F.normalize(patches.float(), dim=-1)
    unit_target = F.normalize(target.float(), dim=-1).unsqueeze(1)
    return (unit_patches * unit_target).sum(dim=-1).max(dim=-1).values


def span_reachability(patches: torch.Tensor, target: torch.Tensor,
                      tol_ratio: float = SPAN_TOL_RATIO) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projection of ``target`` onto the row span of ``H`` (fp32 SVD), per sample.

    Returns:
        c_span: [B] in [0, 1] (clamped for fp error) -- how much of a unit target lies
            in the *signed* linear span of the patch rows.
        rank: [B] int64 -- number of singular values above the stable tolerance.
    """
    if patches.dim() != 3 or target.dim() != 2:
        raise ValueError('expected patches [B, N, D] and target [B, D], got %r and %r'
                         % (tuple(patches.shape), tuple(target.shape)))
    spans, ranks = [], []
    for index in range(patches.shape[0]):
        matrix = patches[index].float()
        unit_target = F.normalize(target[index].float(), dim=-1)
        # right singular vectors span the row space of H
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        if singular.numel() == 0 or float(singular[0]) == 0.0:
            spans.append(torch.zeros((), dtype=torch.float32))
            ranks.append(0)
            continue
        tol = float(singular[0]) * max(matrix.shape) * torch.finfo(torch.float32).eps * tol_ratio
        rank = int((singular > tol).sum())
        if rank == 0:
            spans.append(torch.zeros((), dtype=torch.float32))
            ranks.append(0)
            continue
        basis = vh[:rank]                                   # [rank, D], orthonormal rows
        projection = basis.t() @ (basis @ unit_target)
        spans.append(projection.norm().clamp(0.0, 1.0))
        ranks.append(rank)
    return torch.stack(spans), torch.tensor(ranks, dtype=torch.int64)


def softmax_mixture_oracle(patches: torch.Tensor, target: torch.Tensor,
                           init_logits: Optional[torch.Tensor] = None,
                           steps: int = DEFAULT_ORACLE_STEPS,
                           lr: float = DEFAULT_ORACLE_LR) -> Dict[str, torch.Tensor]:
    """Approximate softmax-mixture oracle: free logits, fixed patches and target.

    Only ``alpha`` (``A_oracle = softmax(alpha)``) is optimized, with Adam, a fixed
    step count and a deterministic initialization (the current Unsaid logits by
    default). Not a global optimum and not a theoretical upper bound -- an
    *approximate* oracle for how far positive pooling could plausibly go from the
    current point.
    """
    if patches.dim() != 3 or target.dim() != 2:
        raise ValueError('expected patches [B, N, D] and target [B, D], got %r and %r'
                         % (tuple(patches.shape), tuple(target.shape)))
    if steps <= 0 or lr <= 0:
        raise ValueError('oracle steps and lr must be positive, got %r / %r' % (steps, lr))
    batch, n_patches, _ = patches.shape
    patches = patches.detach().float()
    unit_target = F.normalize(target.detach().float(), dim=-1)

    def _cosine(logits):
        attention = torch.softmax(logits.float(), dim=-1)
        pooled = torch.einsum('bn,bnd->bd', attention, patches)
        return (F.normalize(pooled, dim=-1) * unit_target).sum(dim=-1)

    with torch.no_grad():
        if init_logits is None:
            init = torch.zeros(batch, n_patches, dtype=torch.float32)
        else:
            init = init_logits.detach().float().clone()
            if init.shape != (batch, n_patches):
                raise ValueError('init_logits must be [B, N], got %r' % (tuple(init.shape),))
        cos_init = _cosine(init)

    alpha = init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([alpha], lr=lr)
    best = cos_init.clone()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        cosine = _cosine(alpha)
        (-cosine.mean()).backward()
        optimizer.step()
        with torch.no_grad():
            best = torch.maximum(best, _cosine(alpha))
    with torch.no_grad():
        cos_final = _cosine(alpha)
    return {'cos_init': cos_init, 'cos_best': best, 'cos_final': cos_final,
            'steps': int(steps), 'lr': float(lr)}


# --------------------------------------------------------------------------- #
# per-checkpoint probe
# --------------------------------------------------------------------------- #
def parse_checkpoints(spec: str) -> List[Tuple[str, str]]:
    """``"tag:path,tag:path"`` -> ``[(tag, path), ...]`` (order preserved)."""
    checkpoints = []
    for entry in str(spec).split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError('checkpoint entries must be "tag:path", got %r' % (entry,))
        tag, path = entry.split(':', 1)
        checkpoints.append((tag.strip(), path.strip()))
    if not checkpoints:
        raise ValueError('at least one checkpoint is required')
    return checkpoints


@torch.no_grad()
def collect_features(model, samples, image_root, variant, preprocess, batch_size, device):
    """Standard global features, router patch features and captions of the cohort."""
    core = model.module if hasattr(model, 'module') else model
    image_root = str(image_root)
    globals_, patches, texts = [], [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        tensors = []
        for sample in chunk:
            with Image.open(os.path.join(image_root, sample['image_path'])) as image:
                tensors.append(preprocess(image.convert('RGB')))
        tensor = torch.stack(tensors).to(device)
        z_global, patch = core.encode_router_input(tensor)
        globals_.append(z_global.detach().float().cpu())
        patches.append(patch.detach().float().cpu())
        captions = [sample['captions'][variant] for sample in chunk]
        tokens = longclip.tokenize(captions, truncate=True).to(device)
        texts.append(core.encode_text(tokens).detach().float().cpu())
    return torch.cat(globals_), torch.cat(patches), torch.cat(texts)


def probe_variant(model, samples, image_root, variant, preprocess, device='cpu',
                  batch_size: int = 64, tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
                  unsaid_residual_eps: float = unsaid_core.DEFAULT_RESIDUAL_EPS,
                  oracle_samples: int = DEFAULT_ORACLE_SAMPLES,
                  oracle_steps: int = DEFAULT_ORACLE_STEPS,
                  oracle_lr: float = DEFAULT_ORACLE_LR) -> Dict:
    """One checkpoint x one caption variant: mechanism metrics + reachability.

    Everything is computed in fp32 (the model is cast before the probe runs), with
    ``no_grad`` everywhere except the oracle's free logits. The model is left exactly
    as it was: no parameter is updated, no gradient is accumulated.
    """
    if variant not in VARIANTS:
        raise ValueError('unknown caption variant %r' % (variant,))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            z_global, patches, text = collect_features(model, samples, image_root, variant,
                                                       preprocess, batch_size, device)
            core = model.module if hasattr(model, 'module') else model
            router = core.said_router
            details = router.forward_with_details(text.to(device), patches.to(device))
            scores = details['scores'].detach().float().cpu()
            A_said = details['attention'].detach().float().cpu()
            z_said = details['said'].detach().float().cpu()
            A_unsaid = unsaid_core.complement_attention(scores, tau_unsaid).detach().float().cpu()
            z_unsaid = unsaid_core.complement_representation(A_unsaid, patches).detach().float().cpu()
            target, residual_norm, valid = unsaid_core.complement_target(
                z_global, z_said, unsaid_residual_eps)
            target = target.float().cpu()
            residual_norm = residual_norm.float().cpu()
            valid = valid.cpu()
            loss_eval = float(unsaid_core.complement_loss(z_unsaid, target, valid))
    finally:
        if was_training:
            model.train()

    cos_current = (F.normalize(z_unsaid, dim=-1) * target).sum(dim=-1)
    cos_said_target = (F.normalize(z_said, dim=-1) * target).sum(dim=-1)
    cos_said_unsaid = (F.normalize(z_said, dim=-1) * F.normalize(z_unsaid, dim=-1)).sum(dim=-1)
    patch_max = best_patch_cosine(patches, target)
    span, span_rank = span_reachability(patches, target)

    valid_index = torch.nonzero(valid, as_tuple=False).flatten()
    oracle_count = min(int(oracle_samples), int(valid_index.numel()))
    oracle = {'samples': oracle_count, 'steps': int(oracle_steps), 'lr': float(oracle_lr),
              'init': 'current_unsaid_logits'}
    if oracle_count > 0:
        selected = valid_index[:oracle_count]
        result = softmax_mixture_oracle(
            patches[selected], target[selected],
            init_logits=-scores[selected] / float(tau_unsaid),
            steps=oracle_steps, lr=oracle_lr)
        cos_oracle = result['cos_best'].detach().cpu()
        cos_oracle_init = result['cos_init'].detach().cpu()
        oracle.update({'cos_best': distribution(cos_oracle),
                       'cos_init': distribution(cos_oracle_init),
                       'cos_best_values': [round(float(v), 6) for v in cos_oracle]})
    else:
        cos_oracle = torch.zeros(0)
        cos_oracle_init = torch.zeros(0)
        oracle.update({'cos_best': distribution([]), 'cos_init': distribution([]),
                       'cos_best_values': []})

    entropy_said = unsaid_core.attention_entropy(A_said)
    entropy_unsaid = unsaid_core.attention_entropy(A_unsaid)
    overlap = unsaid_core.attention_overlap(A_said, A_unsaid)
    jsd = unsaid_core.attention_jsd(A_said, A_unsaid)
    patch_count = int(patches.shape[1])
    metrics = {
        'loss_unsaid_eval': loss_eval,
        'unsaid_valid_ratio': float(valid.float().mean()),
        'unsaid_residual_norm_mean': distribution(residual_norm[valid]) if valid.any() else distribution([]),
        'unsaid_target_cosine': distribution(cos_current[valid]) if valid.any() else distribution([]),
        'said_unsaid_cosine': distribution(cos_said_unsaid[valid]) if valid.any() else distribution([]),
        'said_target_cosine': distribution(cos_said_target[valid]) if valid.any() else distribution([]),
        'said_attention_entropy': distribution(entropy_said),
        'unsaid_attention_entropy': distribution(entropy_unsaid),
        'said_effective_patch_count': distribution(entropy_said.exp()),
        'unsaid_effective_patch_count': distribution(entropy_unsaid.exp()),
        'said_unsaid_attention_overlap': distribution(overlap),
        'said_unsaid_attention_jsd': distribution(jsd),
        'said_attention_max': distribution(A_said.max(dim=-1).values),
        'unsaid_attention_max': distribution(A_unsaid.max(dim=-1).values),
        'reachability': {
            'c_current': distribution(cos_current[valid]) if valid.any() else distribution([]),
            'c_patch_max': distribution(patch_max[valid]) if valid.any() else distribution([]),
            'c_span': distribution(span[valid]) if valid.any() else distribution([]),
            'c_softmax_oracle': oracle['cos_best'],
            'c_softmax_oracle_init': oracle['cos_init'],
            'patch_span_rank': distribution(span_rank[valid].float()) if valid.any() else distribution([]),
            'optimization_gap': (distribution(cos_oracle)['mean'] - distribution(cos_current[valid])['mean'])
                                if (oracle_count > 0 and valid.any()) else None,
            'positive_mixture_gap': (distribution(span[valid])['mean'] - distribution(cos_oracle)['mean'])
                                    if (oracle_count > 0 and valid.any()) else None,
        },
    }
    return {
        'variant': variant,
        'n_images': int(len(samples)),
        'n_valid': int(valid_index.numel()),
        'n_patches': patch_count,
        'tau_unsaid': float(tau_unsaid),
        'unsaid_residual_eps': float(unsaid_residual_eps),
        'precision': 'fp32',
        'oracle': oracle,
        'metrics': metrics,
        'samples_detail': {
            'valid': [bool(v) for v in valid],
            'c_current': [round(float(v), 6) for v in cos_current],
            'c_patch_max': [round(float(v), 6) for v in patch_max],
            'c_span': [round(float(v), 6) for v in span],
            'residual_norm': [round(float(v), 6) for v in residual_norm],
            'said_unsaid_cosine': [round(float(v), 6) for v in cos_said_unsaid],
            'span_rank': [int(v) for v in span_rank],
        },
    }


def probe_checkpoint(model, samples, image_root, variants, preprocess, tag, path,
                     device='cpu', **kwargs) -> Dict:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    incompatible = model.load_state_dict(checkpoint['model'], strict=False)
    if list(incompatible.missing_keys) or list(incompatible.unexpected_keys):
        raise RuntimeError('checkpoint %s does not match the model: missing=%r unexpected=%r'
                           % (path, list(incompatible.missing_keys),
                              list(incompatible.unexpected_keys)))
    digest_before = _state_digest(model)
    started = time.time()
    variants_out = {}
    for variant in variants:
        variants_out[variant] = probe_variant(model, samples, image_root, variant, preprocess,
                                              device=device, **kwargs)
    return {
        'tag': tag,
        'checkpoint': path,
        'checkpoint_sha256': file_sha256(path),
        'checkpoint_step': int(checkpoint.get('step', 0)),
        'checkpoint_epoch': int(checkpoint.get('epoch', 0)),
        'wall_sec': time.time() - started,
        'state_digest_before': digest_before,
        'state_digest_after': _state_digest(model),
        'variants': variants_out,
    }


def _state_digest(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Offline Unsaid mechanism & reachability probe')
    parser.add_argument('--arm', required=True, help='label of the arm being probed, e.g. S or SU')
    parser.add_argument('--checkpoints', required=True,
                        help='comma-separated "tag:path" checkpoint list, evaluated in order')
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'unsaid_mechanism'))
    parser.add_argument('--manifest', default=os.path.join('outputs', 'validation',
                                                           'sharegpt4v1k_manifest.json'))
    parser.add_argument('--data_root', default=os.environ.get('SHARE4V_DATA_ROOT',
                                                              '../datasets/ShareGPT4V'))
    parser.add_argument('--dataset_json', default=os.environ.get(
        'SHARE4V_JSON', 'share-captioner_coco_lcs_sam_1246k_1107.json'))
    parser.add_argument('--num_images', type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--oracle_samples', type=int, default=DEFAULT_ORACLE_SAMPLES)
    parser.add_argument('--oracle_steps', type=int, default=DEFAULT_ORACLE_STEPS)
    parser.add_argument('--oracle_lr', type=float, default=DEFAULT_ORACLE_LR)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tau_unsaid', type=float, default=unsaid_core.DEFAULT_TAU_UNSAID)
    parser.add_argument('--unsaid_residual_eps', type=float,
                        default=unsaid_core.DEFAULT_RESIDUAL_EPS)
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--said_feature_source', default='residual')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    variants = [name.strip() for name in args.variants.split(',') if name.strip()]
    dataset_json = args.dataset_json if os.path.isabs(args.dataset_json) \
        else os.path.join(args.data_root, args.dataset_json)
    manifest = load_or_create_manifest(args.manifest, dataset_json)
    samples = manifest['samples'][:args.num_images]
    write_json(os.path.join(args.output_dir, 'manifest.json'), {
        'arm': args.arm,
        'probe': 'unsaid_mechanism_probe',
        'protocol': SHAREGPT4V1K_PROTOCOL,
        'num_images': len(samples),
        'image_indices': [sample['json_index'] for sample in samples],
        'manifest_path': args.manifest,
        'manifest_sha256': file_sha256(args.manifest),
        'dataset_json_sha256': manifest['dataset_json_sha256'],
        'variants': variants,
        'primary_variant': PRIMARY_VARIANT,
        'similarity_chunk_not_used_for_probe': CANONICAL_SIMILARITY_CHUNK,
        'checkpoints': [{'tag': tag, 'path': path} for tag, path in checkpoints],
        'tau_unsaid': args.tau_unsaid,
        'unsaid_residual_eps': args.unsaid_residual_eps,
        'said_feature_source': args.said_feature_source,
        'precision': 'fp32',
        'oracle': {'samples': args.oracle_samples, 'steps': args.oracle_steps, 'lr': args.oracle_lr,
                   'init': 'current_unsaid_logits', 'kind': 'approximate softmax-mixture oracle'},
        'batch_size': args.batch_size,
        'device': args.device,
    })

    clip_model, preprocess = longclip.load_from_clip(args.base_model, device='cpu', args=args)
    model = SALUModel(clip_model, said_feature_source=args.said_feature_source)
    model.float()                       # fp32 geometry: no bf16 noise in the probe
    model = model.to(args.device)
    model.eval()

    out_path = os.path.join(args.output_dir, 'arm_%s.jsonl' % args.arm.lower())
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    with open(out_path, 'w', encoding='utf-8') as stream:
        for tag, path in checkpoints:
            record = probe_checkpoint(
                model, samples, args.data_root, variants, preprocess, tag, path,
                device=args.device, batch_size=args.batch_size, tau_unsaid=args.tau_unsaid,
                unsaid_residual_eps=args.unsaid_residual_eps,
                oracle_samples=args.oracle_samples, oracle_steps=args.oracle_steps,
                oracle_lr=args.oracle_lr)
            record['arm'] = args.arm
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')
            stream.flush()
            print('PROBE %s %s step=%d wall=%.1fs digest_stable=%s'
                  % (args.arm, tag, record['checkpoint_step'], record['wall_sec'],
                     record['state_digest_before'] == record['state_digest_after']), flush=True)
    print('PROBE_DONE %s wall=%.1fs records=%d output=%s'
          % (args.arm, time.time() - started, len(checkpoints), out_path), flush=True)


if __name__ == '__main__':
    main()
