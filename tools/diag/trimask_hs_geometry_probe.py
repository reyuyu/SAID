"""Read-only offline probe of the S0-TriMask-HS @500 text gate: caption dependence and
intersection geometry.

This is a *diagnostic*, not a new experiment and not a new evaluation protocol:

* it never creates an optimizer, never calls ``backward`` and never updates a parameter. The only
  files it writes are its own diagnostics under the run's ``diagnostics/`` directory plus a small
  caption-free summary. The checkpoint, the training logs, the evaluation JSONs and
  ``run_status.json`` are read-only, and the checkpoint's sha256 is re-checked before and after.
* the score matrices it computes are the three paths of ``smartclip_trimask_hs`` restricted to a
  fixed 256-image x 256-caption COCO val2017 diagnostic pool. They are **not** the canonical COCO
  evaluation, they are not comparable to it, and they do not touch the frozen promotion gate.
* the model is loaded once; every image and every caption is encoded once and cached. Every variant
  (shuffled / mean-profile / all-ones text mask, the common-intersection readout) is a matrix
  operation on that cache, so no variant re-runs CLIP, re-samples a caption or changes the
  preprocessing. The full training forward is never called, so no DDP ``all_gather`` is triggered
  and no process group is created.

Three diagnostics:

A. variance decomposition of the cached hard text-mask matrix ``M`` (256 x 512) in FP64 on the CPU,
   population (``unbiased=False``) convention, with ``V_total = V_level + V_profile + V_interaction``
   and ``V_caption_dependent = V_level + V_interaction`` checked explicitly;
B. text-gate replacement: NORMAL / SHUFFLED (4 saved seeds) / MEAN_PROFILE / ONES on the same
   manifest and candidate order, reporting L2/L3 for both directions plus R@K, MRR, CE and paired
   per-query differences against NORMAL -- and never a loss that contains ``10 * L1``, which the
   text gate cannot move;
C. intersection geometry: ``Qcap`` (both learned hard masks at once) and the identity
   ``Q3 = Qcap * factor`` with ``factor = sqrt(rhoI * rhoT)``, with the effect on the actual ranking
   separated from the effect on the scores.

    python tools/diag/trimask_hs_geometry_probe.py \
        --checkpoint runs_salu/said_s0_trimask_hs_v02/step500/trimask_S0_TriMask_HS_step000500.pt \
        --run-dir runs_salu/said_s0_trimask_hs_v02/step500
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_CHECKPOINT = ('runs_salu/said_s0_trimask_hs_v02/step500/'
                      'trimask_S0_TriMask_HS_step000500.pt')
DEFAULT_RUN_DIR = 'runs_salu/said_s0_trimask_hs_v02/step500'
DEFAULT_SUMMARY_OUT = 'docs/said_trimask_hs/geometry_probe_summary.json'

EXPECTED = {'objective': 'smartclip_trimask_hs', 'arm': 'S0_TriMask_HS', 'completed_steps': 500,
            'text_gate_mode': 'hard_st', 'lambda_sparse_t': 0.2}
FIXED_SCALE = 100.0
NORM_EPS = 1e-6
SHUFFLE_SEEDS = (0, 1, 2, 3)
K_VALUES = (1, 5, 10)


# --------------------------------------------------------------------------- helpers
def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    """sha256 over sorted keys and their float32 bytes -- local, dependency-free state summary."""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode('utf-8'))
        digest.update(np.ascontiguousarray(state[key].detach().float().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError('not JSON serialisable: %r' % type(value))


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True,
                  default=_json_default)
    os.replace(temporary, path)


def f(value):
    return float(value)


def quantiles(values, points=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return {('q%g' % p): float(np.quantile(array, p)) for p in points}


# --------------------------------------------------------------------------- manifest
def build_manifest(ann_file, tokenizer, n_images, seed):
    """The diagnostic pool, fixed before any score is looked at.

    Rule, recorded so that it cannot drift after the fact: sort the COCO val2017 image ids
    ascending, shuffle that list with an independent ``random.Random(seed)`` (not numpy/torch), then
    walk it. For each image, walk its own captions in ascending ``annotation_id`` order and take the
    first caption whose **token sequence** is not already used; if every caption of that image is
    used, continue with the next image in the shuffled order. Stop at ``n_images`` samples. No score
    is consulted anywhere in this function.
    """
    with open(ann_file, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    file_name = {int(image['id']): image['file_name'] for image in payload['images']}
    by_image = {}
    for annotation in payload['annotations']:
        by_image.setdefault(int(annotation['image_id']), []).append(annotation)
    for image_id in by_image:
        by_image[image_id].sort(key=lambda a: int(a['id']))

    order = sorted(file_name)
    random.Random(seed).shuffle(order)

    used_tokens, samples, skipped = set(), [], []
    token_width = None
    for image_id in order:
        if len(samples) >= n_images:
            break
        for annotation in by_image.get(image_id, []):
            caption = str(annotation['caption']).strip()
            tokens = tokenizer([caption], truncate=True)
            token_width = int(tokens.shape[-1])
            token_sha = hashlib.sha256(
                tokens.detach().cpu().numpy().astype(np.int64).tobytes()).hexdigest()
            signature = tuple(int(v) for v in tokens.reshape(-1).tolist())
            if signature in used_tokens:
                skipped.append({'image_id': image_id, 'annotation_id': int(annotation['id']),
                                'reason': 'token sequence already used'})
                continue
            used_tokens.add(signature)
            samples.append({
                'candidate_index': len(samples),
                'image_id': image_id,
                'annotation_id': int(annotation['id']),
                'file_name': file_name[image_id],
                'caption': caption,
                'token_sha256': token_sha,
                'token_length_nonpad': int((tokens != 0).sum()),
            })
            break

    if len(samples) < n_images:
        raise SystemExit('only %d unique-token samples available, %d requested'
                         % (len(samples), n_images))
    return {
        'purpose': 'fixed diagnostic pool for the S0-TriMask-HS @500 geometry probe',
        'source': {'annotations': os.path.abspath(ann_file),
                   'annotations_sha256': sha256_of(ann_file),
                   'image_root': os.path.abspath(os.path.join(os.path.dirname(ann_file), '..',
                                                              'val2017'))},
        'selection': {
            'image_ids': 'COCO val2017 ids sorted ascending, shuffled by random.Random(seed)',
            'caption': 'first caption in ascending annotation_id order whose token sequence is new',
            'on_token_collision': 'skip that caption, continue down the caption list, then continue '
                                  'with the next image in the shuffled order',
            'n_images': int(n_images), 'seed': int(seed),
            'rng': 'python random.Random, independent of the numpy/torch global generators',
            'fixed_before_scoring': True,
            'token_width': token_width,
        },
        'preprocessing': {
            'image': 'reference openai-clip _transform(224) from model.longclip.load_from_clip',
            'text': 'longclip.tokenize(caption, truncate=True) exactly as the trainer calls it; the '
                    'resulting width is recorded above as token_width',
        },
        'caption_text_policy': 'captions are written to the run-local diagnostics file only and are '
                               'never committed to the repository',
        'count': len(samples),
        'skipped_captions': skipped,
        'samples': samples,
    }


# --------------------------------------------------------------------------- scoring
def score_matrix(v, t, mask_i, mask_t, eps=NORM_EPS, scale=FIXED_SCALE):
    """The three paths, per candidate caption ``j``, exactly as the trained objective computes them::

        Q1[i,j] = 100 * cos(v_i * mI_j, t_j)
        Q2[i,j] = 100 * cos(v_i,        t_j * mT_j)
        Q3[i,j] = 100 * cos(v_i * mI_j, t_j * mT_j)

    Two things are load-bearing here and were both got wrong in a first draft:

    * the image-side mask is indexed by the **candidate** ``j``, never by the query ``i``. Using
      ``v_i * mI_i`` for a whole row would leak the query's own correct-pair mask into every
      negative, which is exactly the leakage this probe must not have.
    * each side is normalised with its own ``F.normalize(..., eps=eps)``; the reference module's
      equivalent expression is ``(v @ (mI * t_hat).T) / sqrt(clamp_min(v^2 @ mI^2.T, eps^2))``,
      which is the same quantity, and :func:`reconcile_forms` checks that.
    """
    v = v.float()
    t = t.float()
    v_hat = F.normalize(v, dim=-1, eps=eps)
    t_hat = F.normalize(t, dim=-1, eps=eps)
    r_t = F.normalize(t * mask_t, dim=-1, eps=eps)
    denominator = (v.square() @ mask_i.square().t()).clamp_min(eps ** 2).sqrt()
    q1 = scale * (v @ (mask_i * t_hat).t()) / denominator
    q2 = scale * (v_hat @ r_t.t())
    q3 = scale * (v @ (mask_i * r_t).t()) / denominator
    return q1, q2, q3


def broadcast_cosine_scores(v, t, mask_i, mask_t, eps=NORM_EPS, scale=FIXED_SCALE, chunk=32):
    """The same three cosines built explicitly, in blocks, as an independent implementation.

    Every masked vector is normalised on its own and the score is a plain inner product, so this
    shares no arithmetic with the matmul form in :func:`score_matrix`; the two agreeing on all
    ``n x n`` pairs is the check that the probe scores what the trained model scores.
    """
    n = v.shape[0]
    v_hat = F.normalize(v.float(), dim=-1, eps=eps)
    t_hat = F.normalize(t.float(), dim=-1, eps=eps)
    q1 = torch.empty(n, n)
    q2 = torch.empty(n, n)
    q3 = torch.empty(n, n)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        for j0 in range(0, n, chunk):
            j1 = min(j0 + chunk, n)
            # image side masked by the *candidate* j, normalised on its own
            v_masked_i = F.normalize(
                v[i0:i1].float().unsqueeze(1) * mask_i[j0:j1].float().unsqueeze(0),
                dim=-1, eps=eps)
            # text side masked by its own caption j, normalised on its own
            t_masked_t = F.normalize(
                t[j0:j1].float() * mask_t[j0:j1].float(), dim=-1, eps=eps).unsqueeze(0)
            q1[i0:i1, j0:j1] = scale * (v_masked_i * t_hat[j0:j1].unsqueeze(0)).sum(dim=-1)
            q2[i0:i1, j0:j1] = scale * (v_hat[i0:i1].unsqueeze(1) * t_masked_t).sum(dim=-1)
            q3[i0:i1, j0:j1] = scale * (v_masked_i * t_masked_t).sum(dim=-1)
    return q1, q2, q3


def reference_form(v, t, mask_i, mask_t, eps=NORM_EPS, scale=FIXED_SCALE):
    """The trained module's own arithmetic (denominator form), kept as a third expression."""
    v = v.float()
    t_hat = F.normalize(t.float(), dim=-1, eps=eps)
    r_t = F.normalize(t.float() * mask_t.float(), dim=-1, eps=eps)
    m_i = mask_i.float()
    denominator = (v.square() @ m_i.square().t()).clamp_min(eps ** 2).sqrt()
    v_hat = F.normalize(v, dim=-1, eps=eps)
    return (scale * ((v @ (m_i * t_hat).t()) / denominator), scale * (v_hat @ r_t.t()),
            scale * ((v @ (m_i * r_t).t()) / denominator))


def factory_form(v, t, mask_i, mask_t, eps=NORM_EPS, scale=FIXED_SCALE):
    """Q3 and Qcap from one shared numerator ``N = v @ (mI*mT*t).T`` and different denominators.

    ``Q3 = 100 N / (sqrt(A) sqrt(B))`` with ``A = ||v * mI||^2``, ``B = ||t * mT||^2`` and
    ``Qcap = 100 N / (sqrt(C) sqrt(D))`` with ``C = ||v * c||^2``, ``D = ||t * c||^2``,
    ``c = mI * mT``. This is why ``Q3 = Qcap * sqrt(C D / (A B))`` holds exactly, and it is the
    cheap way to compute both.
    """
    c = mask_i * mask_t
    denominator_i = mask_i.square()
    denominator_t = mask_t.square()
    a = v.square() @ denominator_i.t()
    b = (t.square() * denominator_t).sum(dim=-1)
    c_norm = v.square() @ c.square().t()
    d_norm = (t.square() * c.square()).sum(dim=-1)
    numerator = v @ (c * t).t()
    q3 = scale * numerator / (a.clamp_min(eps ** 2).sqrt()
                              * b.clamp_min(eps ** 2).sqrt().unsqueeze(0))
    q_cap = scale * numerator / (c_norm.clamp_min(eps ** 2).sqrt()
                                 * d_norm.clamp_min(eps ** 2).sqrt().unsqueeze(0))
    factor = ((c_norm / a).clamp_min(0).sqrt()
              * (d_norm.unsqueeze(0) / b.unsqueeze(0)).clamp_min(0).sqrt())
    return {'q3': q3, 'q_cap': q_cap, 'factor': factor, 'numerator': numerator,
            'a': a, 'b': b, 'c': c_norm, 'd': d_norm}


def reconcile_forms(v, t, mask_i, mask_t):
    """Three independent expressions of the same three paths, compared on the full pair grid."""
    q1a, q2a, q3a = score_matrix(v, t, mask_i, mask_t)
    q1b, q2b, q3b = reference_form(v, t, mask_i, mask_t)
    q1c, q2c, q3c = broadcast_cosine_scores(v, t, mask_i, mask_t)
    geometry = factory_form(v, t, mask_i, mask_t)
    return {
        'matmul_vs_broadcast_cosine_q1_max_abs_diff': f((q1a - q1c).abs().max()),
        'matmul_vs_broadcast_cosine_q2_max_abs_diff': f((q2a - q2c).abs().max()),
        'matmul_vs_broadcast_cosine_q3_max_abs_diff': f((q3a - q3c).abs().max()),
        'matmul_vs_reference_module_expression_q1_max_abs_diff': f((q1a - q1b).abs().max()),
        'matmul_vs_reference_module_expression_q2_max_abs_diff': f((q2a - q2b).abs().max()),
        'matmul_vs_reference_module_expression_q3_max_abs_diff': f((q3a - q3b).abs().max()),
        'matmul_vs_shared_numerator_q3_max_abs_diff': f((q3a - geometry['q3']).abs().max()),
        'note': 'scores are on the fixed 100x scale. The matmul form is what the probe uses; the '
                'blocked broadcast form normalises every masked vector on its own and is an '
                'independent implementation of the same cosines; a disagreement means the probe is '
                'not scoring what the trained model scores',
        'self_test_the_bug_this_caught': 'a first draft masked the image side with the *query* row '
                                         'v_i * mI_i instead of the candidate v_i * mI_j, which '
                                         'leaks the correct-pair mask into every negative of that '
                                         'row; the check below is expected to be ~51 on that buggy '
                                         'form and ~0 here',
    }


def retrieval_metrics(q, k_values=K_VALUES):
    """Rank/MRR/CE for a square matrix whose positive is the diagonal.

    Fixed tie rule, recorded and never adjusted per result:
    ``rank = 1 + #{j != i : Q[i, j] > Q[i, i]}`` -- a negative that only ties the positive does not
    beat it. ``M_max == 0`` is therefore counted separately as a tie, never as a strict win.
    """
    q = q.float()
    diagonal = torch.diagonal(q)
    off = q.clone()
    off.fill_diagonal_(float('-inf'))
    s_max_neg = off.max(dim=1).values
    worst = off.argmax(dim=1)
    lse_neg = torch.logsumexp(off, dim=1)
    rank = 1 + (off > diagonal.unsqueeze(1)).sum(dim=1)
    m_max = diagonal - s_max_neg
    m_lse = diagonal - lse_neg
    ce = F.softplus(-m_lse)
    out = {
        'n_queries': int(q.shape[0]),
        'ce': f(ce.mean()),
        'ce_identity_max_abs_diff': f((ce - (torch.logsumexp(q, dim=1) - diagonal)).abs().max()),
        'mrr': f((1.0 / rank.float()).mean()),
        'R@1': f((rank <= 1).float().mean()), 'R@5': f((rank <= 5).float().mean()),
        'R@10': f((rank <= 10).float().mean()),
        'm_max_mean': f(m_max.mean()), 'm_lse_mean': f(m_lse.mean()),
        'm_max_negative_count': int((m_max < 0).sum()),
        'm_lse_negative_count': int((m_lse < 0).sum()),
        'm_max_tie_count': int((m_max == 0).sum()),
        'rank_mean': f(rank.float().mean()), 'rank_median': int(rank.median()),
        's_pos_mean': f(diagonal.mean()), 's_max_negative_mean': f(s_max_neg.mean()),
        'per_query_rank': [int(x) for x in rank.tolist()],
    }
    for k in k_values:
        out.setdefault('R@%d' % k, f((rank <= k).float().mean()))
    out['_rank'] = rank
    out['_diagonal'] = diagonal
    out['_s_max_neg'] = s_max_neg
    out['_m_max'] = m_max
    out['_m_lse'] = m_lse
    out['_ce'] = ce
    out['_worst'] = worst
    return out


def public(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith('_')}


def paired_against(base, other, k_values=K_VALUES):
    """Per-query paired differences of a variant against NORMAL (other - base)."""
    out = {'delta_ce_mean': f((other['_ce'] - base['_ce']).mean()),
           'delta_mrr': f((1.0 / other['_rank'].float()).mean()
                          - (1.0 / base['_rank'].float()).mean()),
           'delta_s_pos_mean': f((other['_diagonal'] - base['_diagonal']).mean()),
           'delta_s_max_negative_mean': f((other['_s_max_neg'] - base['_s_max_neg']).mean()),
           'changed_rank_queries': int((other['_rank'] != base['_rank']).sum()),
           'rank_improved': int((other['_rank'] < base['_rank']).sum()),
           'rank_worsened': int((other['_rank'] > base['_rank']).sum())}
    for k in k_values:
        out['delta_R@%d' % k] = f((other['_rank'] <= k).float().mean()
                                  - (base['_rank'] <= k).float().mean())
        out['hit_lost_R@%d' % k] = int(((base['_rank'] <= k) & (other['_rank'] > k)).sum())
        out['hit_gained_R@%d' % k] = int(((base['_rank'] > k) & (other['_rank'] <= k)).sum())
    return out


# --------------------------------------------------------------------------- diagnostic A
def variance_decomposition(mask_matrix):
    m = np.asarray(mask_matrix, dtype=np.float64)
    n_rows, n_cols = m.shape
    mu = m.mean()
    row_mean = m.mean(axis=1)
    col_mean = m.mean(axis=0)
    v_total = float(((m - mu) ** 2).mean())
    v_level = float(((row_mean - mu) ** 2).mean())
    v_profile = float(((col_mean - mu) ** 2).mean())
    interaction = m - row_mean[:, None] - col_mean[None, :] + mu
    v_interaction = float((interaction ** 2).mean())
    v_caption_dependent = float(((m - col_mean[None, :]) ** 2).mean())
    old_field = v_total - float(((m - row_mean[:, None]) ** 2).mean())
    row_keep = m.sum(axis=1)
    col_keep = m.sum(axis=0)
    return {
        'shape': [int(n_rows), int(n_cols)],
        'population_definition': 'unbiased=False, computed in FP64 on the CPU',
        'mu': f(mu), 'v_total': v_total, 'v_level': v_level, 'v_profile': v_profile,
        'v_interaction': v_interaction, 'v_caption_dependent': v_caption_dependent,
        'identity_v_total_minus_parts': f(v_total - (v_level + v_profile + v_interaction)),
        'identity_v_caption_dependent_minus_parts':
            f(v_caption_dependent - (v_level + v_interaction)),
        'share_of_total': ({'level': v_level / v_total, 'profile': v_profile / v_total,
                            'interaction': v_interaction / v_total} if v_total > 0 else None),
        'share_of_caption_dependent': ({'level': v_level / v_caption_dependent,
                                        'interaction': v_interaction / v_caption_dependent}
                                       if v_caption_dependent > 0 else None),
        'historical_field_check': {
            'definition': 'V_total - mean_j Var_d(M[j,d]) -- the field logged during training',
            'value': f(old_field),
            'equals_v_level': bool(abs(old_field - v_level) < 1e-12),
            'note': 'this field tracks V_level (how much captions differ in their overall keep '
                    'ratio); the caption x coordinate interaction is a separate component, so the '
                    'field must not be read as the whole caption-adaptive share'},
        'row_keep_count': {'min': f(row_keep.min()), 'mean': f(row_keep.mean()),
                           'max': f(row_keep.max()), 'quantiles': quantiles(row_keep)},
        'col_keep_frequency': {'min': f(col_keep.min() / n_rows), 'mean': f(col_keep.mean() / n_rows),
                               'max': f(col_keep.max() / n_rows),
                               'quantiles': quantiles(col_keep / n_rows)},
        'col_keep_histogram': {str(k): int((col_keep == k).sum())
                               for k in range(0, n_rows + 1) if (col_keep == k).sum() > 0},
        'value_counts': {'hard_zeros': int((m == 0).sum()), 'hard_ones': int((m == 1).sum()),
                         'other_values': int(((m != 0) & (m != 1)).sum())},
    }


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--run-dir', default=DEFAULT_RUN_DIR)
    parser.add_argument('--n-images', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--coco-root', default=os.environ.get('COCO_DATA_ROOT', '/root/datasets/coco'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--summary-out', default=DEFAULT_SUMMARY_OUT)
    parser.add_argument('--hard-queries-per-path', type=int, default=5)
    parser.add_argument('--max-hard-queries', type=int, default=20)
    args = parser.parse_args()

    started = time.time()
    os.chdir(REPO)
    checkpoint = os.path.abspath(args.checkpoint)
    run_dir = os.path.abspath(args.run_dir)
    diagnostics_dir = os.path.join(run_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)

    checkpoint_sha_before = sha256_of(checkpoint)
    run_status_path = os.path.join(run_dir, 'run_status.json')
    status_sha_before = sha256_of(run_status_path) if os.path.isfile(run_status_path) else None

    # -- identity, from the file itself, never from the file name -------------------------------
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = {key: payload.get(key) for key in EXPECTED}
    mismatched = {k: (identity.get(k), v) for k, v in EXPECTED.items() if identity.get(k) != v}
    if mismatched:
        raise SystemExit('checkpoint identity mismatch (field: got vs expected): %r' % mismatched)
    if any(key.startswith(('text_mask_net', 'text_stem', 'text_gate_projection'))
           for key in payload['model']):
        raise SystemExit('unexpected text-branch keys inside payload["model"]')
    checkpoint_meta = {key: payload.get(key) for key in
                       ('git_head', 'lr_horizon_steps', 'epoch', 'next_batch_index', 'phase',
                        'precision')}
    text_branch_tensor_count = len(payload['text_mask_net'])
    clip_tensor_count = len(payload['model'])

    from model import longclip
    from model.said_trimask import TriMaskTrainModule, mask_from_hidden

    torch.manual_seed(0)
    clip_model, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                                     args=argparse.Namespace())
    module = TriMaskTrainModule(clip_model, rank=0, text_gate_mode=EXPECTED['text_gate_mode'],
                                text_mask_seed=0, grad_checkpoint_views=False)
    module.clip.load_state_dict(payload['model'])
    module.text_mask_net.load_state_dict(payload['text_mask_net'])
    device = torch.device(args.device)
    module = module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    state_before = {'clip': state_digest(module.clip.state_dict()),
                    'text_mask_net': state_digest(module.text_mask_net.state_dict())}
    del payload

    # -- manifest, then one single encoding pass ------------------------------------------------
    manifest = build_manifest(os.path.join(args.coco_root, 'annotations', 'captions_val2017.json'),
                              longclip.tokenize, args.n_images, args.seed)
    manifest_path = os.path.join(diagnostics_dir, 'manifest.json')
    write_json_atomic(manifest_path, manifest)
    manifest_sha = sha256_of(manifest_path)

    captions = [s['caption'] for s in manifest['samples']]
    file_names = [s['file_name'] for s in manifest['samples']]
    image_ids = [s['image_id'] for s in manifest['samples']]
    annotation_ids = [s['annotation_id'] for s in manifest['samples']]
    image_root = manifest['source']['image_root']

    from PIL import Image
    cache = {'v': [], 't': [], 'mI': [], 'mT': [], 'pT': [], 'hT': []}
    dtypes = {}
    with torch.inference_mode():
        for start in range(0, len(captions), args.batch_size):
            stop = min(start + args.batch_size, len(captions))
            tensors = []
            for relative in file_names[start:stop]:
                with Image.open(os.path.join(image_root, relative)) as image:
                    tensors.append(preprocess(image.convert('RGB')))
            images = torch.stack(tensors).to(device)
            tokens = longclip.tokenize(captions[start:stop], truncate=True).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v = module.clip.encode_image(images)
                t_raw, hidden = module.clip.encode_text(tokens, return_full=True)
                m_i, _soft_i, _logits_i = mask_from_hidden(module.clip.mask_net, hidden)
            gate = module.text_mask_net.forward_with_details(hidden)
            dtypes = {
                'image_features': str(v.dtype), 'text_features': str(t_raw.dtype),
                'text_hidden': str(hidden.dtype), 'visual_mask_forward_value': str(m_i.dtype),
                'text_gate_probability': str(gate['pT'].dtype),
                'text_gate_hard': str(gate['hT'].dtype),
                'text_mask_forward_value': str(gate['mT'].dtype),
                'execution': 'one bf16 autocast pass for the CLIP image/text paths and the reference '
                             'visual mask head, exactly as the trainer calls them; the text head and '
                             'all scoring are explicit fp32',
            }
            cache['v'].append(v.float().cpu())
            cache['t'].append(t_raw.float().cpu())
            cache['mI'].append(m_i.float().cpu())
            cache['mT'].append(gate['mT'].float().cpu())
            cache['pT'].append(gate['pT'].float().cpu())
            cache['hT'].append(gate['hT'].float().cpu())

    v = torch.cat(cache['v'])
    t = torch.cat(cache['t'])
    mI = torch.cat(cache['mI'])
    mT = torch.cat(cache['mT'])
    pT = torch.cat(cache['pT'])
    hT = torch.cat(cache['hT'])
    n = v.shape[0]

    state_after = {'clip': state_digest(module.clip.state_dict()),
                   'text_mask_net': state_digest(module.text_mask_net.state_dict())}
    checkpoint_sha_after = sha256_of(checkpoint)
    status_sha_after = sha256_of(run_status_path) if os.path.isfile(run_status_path) else None

    mask_stats = {
        'mT_keep_ratio_mean': f(mT.mean()), 'mT_keep_ratio_min': f(mT.mean(dim=1).min()),
        'mT_keep_ratio_max': f(mT.mean(dim=1).max()),
        'mT_exactly_zero_coordinates_fraction': f((mT == 0).float().mean()),
        'mT_not_binary_count': int(((mT != 0) & (mT != 1)).sum()),
        'mI_keep_ratio_mean': f(mI.mean()), 'mI_keep_ratio_min': f(mI.mean(dim=1).min()),
        'mI_keep_ratio_max': f(mI.mean(dim=1).max()),
        'intersection_keep_ratio_mean': f((mI * mT).mean()),
        'intersection_empty_captions': int(((mI * mT).sum(dim=1) == 0).sum()),
        'pT_mean': f(pT.mean()),
        'captions_with_no_closed_text_coordinate': int((mT.min(dim=1).values > 0).sum()),
        'captions_with_all_text_coordinates_closed': int((mT.max(dim=1).values < 0.5).sum()),
        'forward_mask_equals_hard_gate_max_abs_diff': f((mT - hT).abs().max()),
        'hard_gate_is_binary': bool(((hT == 0) | (hT == 1)).all()),
        'probability_matches_hard_gate': int(((pT >= 0.5).float() == hT).all(dim=1).sum()),
    }

    # ---------------------------------------------------------------- A: variance decomposition
    variance = variance_decomposition(mT.numpy())
    binary = (mT.numpy() >= 0.5).astype(np.float64)
    hamming, jaccard, identical = [], [], 0
    for a, b in itertools.combinations(range(binary.shape[0]), 2):
        intersection = float((binary[a] * binary[b]).sum())
        union = float(((binary[a] + binary[b]) > 0).sum())
        hamming.append(float((binary[a] != binary[b]).sum()))
        jaccard.append(intersection / union if union > 0 else 1.0)
        if intersection == union:
            identical += 1
    variance['pairwise_mask_distance'] = {'hamming': quantiles(hamming), 'jaccard': quantiles(jaccard),
                                         'hamming_mean': f(np.mean(hamming)),
                                         'jaccard_mean': f(np.mean(jaccard)),
                                         'identical_pairs': identical}
    variance['permutation_note'] = ('a shuffle preserves the multiset of masks and the overall keep '
                                    'rate distribution but NOT each caption\'s own keep count, so a '
                                    'shuffled result is not a pure semantic isolation')

    # ---------------------------------------------------------------- B: gate replacement
    q1, q2, q3 = score_matrix(v, t, mI, mT)
    reconciliation = reconcile_forms(v, t, mI, mT)
    geometry = factory_form(v, t, mI, mT)
    q_cap, factor = geometry['q_cap'], geometry['factor']

    mean_profile = mT.mean(dim=0, keepdim=True).expand_as(mT).contiguous()
    variants = {'NORMAL': mT, 'MEAN_PROFILE': mean_profile, 'ONES': torch.ones_like(mT)}
    permutation_report, permutations = [], {}
    for seed in SHUFFLE_SEEDS:
        permutation = derangement(n, seed)
        permutations['seed_%d' % seed] = permutation
        shuffled = mT[torch.tensor(permutation)]
        variants['SHUFFLED_seed%d' % seed] = shuffled
        permutation_report.append({
            'seed': seed,
            'changed_mask_coordinates_fraction': f((shuffled != mT).float().mean()),
            'captions_whose_mask_is_unchanged': int((shuffled == mT).all(dim=1).sum()),
            'keep_ratio_delta_mean_abs': f((shuffled.mean(dim=1) - mT.mean(dim=1)).abs().mean())})

    variant_public, variant_private = {}, {}
    for name, mask in variants.items():
        qv1, qv2, qv3 = score_matrix(v, t, mI, mask)
        entry = {'q1_equals_normal_max_abs_diff': f((qv1 - q1).abs().max())}
        for label, q in (('L2', qv2), ('L3', qv3)):
            for direction, matrix in (('I2T', q), ('T2I', q.t().contiguous())):
                metrics = retrieval_metrics(matrix)
                variant_private[(name, label, direction)] = metrics
                entry['%s_%s' % (label, direction)] = public(metrics)
        variant_public[name] = entry
    for name in variant_public:
        if name == 'NORMAL':
            continue
        variant_public[name]['paired_vs_normal'] = {
            '%s_%s' % (label, direction):
                paired_against(variant_private[('NORMAL', label, direction)],
                               variant_private[(name, label, direction)])
            for label in ('L2', 'L3') for direction in ('I2T', 'T2I')}

    ones_q1, ones_q2, ones_q3 = score_matrix(v, t, mI, torch.ones_like(mT))
    identity_checks = {
        'ones_q3_equals_normal_q1_max_abs_diff': f((ones_q3 - q1).abs().max()),
        'ones_q2_equals_raw_global_cosine_max_abs_diff':
            f((ones_q2 - FIXED_SCALE * (F.normalize(v, dim=-1, eps=NORM_EPS)
                                        @ F.normalize(t, dim=-1, eps=NORM_EPS).t())).abs().max()),
    }
    shuffled_aggregate = {}
    for label in ('L2', 'L3'):
        for direction in ('I2T', 'T2I'):
            ce = [variant_public['SHUFFLED_seed%d' % s]['%s_%s' % (label, direction)]['ce']
                  for s in SHUFFLE_SEEDS]
            r1 = [variant_public['SHUFFLED_seed%d' % s]['%s_%s' % (label, direction)]['R@1']
                  for s in SHUFFLE_SEEDS]
            shuffled_aggregate['%s_%s' % (label, direction)] = {
                'ce_per_seed': [f(x) for x in ce], 'ce_mean': f(np.mean(ce)),
                'ce_min': f(min(ce)), 'ce_max': f(max(ce)),
                'R@1_per_seed': [f(x) for x in r1], 'R@1_mean': f(np.mean(r1)),
                'R@1_min': f(min(r1)), 'R@1_max': f(max(r1))}

    # ---------------------------------------------------------------- C: intersection geometry
    a, b, c_norm, d_norm = geometry['a'], geometry['b'], geometry['c'], geometry['d']
    threshold = NORM_EPS ** 2
    valid = (a > threshold) & (b.unsqueeze(0) > threshold) & (c_norm > threshold) \
        & (d_norm.unsqueeze(0) > threshold)
    q3_predicted = q_cap * factor
    residual = (q3_predicted - q3)
    diagonal = torch.eye(n, dtype=torch.bool)
    off = ~diagonal
    rho_i = c_norm / a
    rho_t = (d_norm.unsqueeze(0) / b.unsqueeze(0)).expand_as(rho_i)
    c_cap = mI * mT                      # the per-caption intersection actually used by Qcap
    q3_private = {'I2T': retrieval_metrics(q3), 'T2I': retrieval_metrics(q3.t().contiguous())}
    q_cap_private = {'I2T': retrieval_metrics(q_cap),
                     'T2I': retrieval_metrics(q_cap.t().contiguous())}
    geometry_stats = {
        'identity': 'Q3[i,j] = Qcap[i,j] * factor[i,j] with factor = sqrt(rhoI * rhoT), '
                    'rhoI = ||v_i*c_j||^2 / ||v_i*mI_j||^2, rhoT = ||t_j*c_j||^2 / ||t_j*mT_j||^2',
        'valid_pairs': int(valid.sum()), 'total_pairs': int(valid.numel()),
        'valid_fraction': f(valid.float().mean()), 'invalid_pairs': int((~valid).sum()),
        'invalid_reason_counts': {'a_below_eps_squared': int((a <= threshold).sum()),
                                  'b_below_eps_squared': int((b <= threshold).sum()),
                                  'c_below_eps_squared': int((c_norm <= threshold).sum()),
                                  'd_below_eps_squared': int((d_norm <= threshold).sum())},
        'max_abs_error_on_valid_pairs': f(residual[valid].abs().max()) if bool(valid.any()) else None,
        'max_relative_error_on_valid_pairs':
            (f((residual[valid].abs() / q3[valid].abs().clamp_min(1e-9)).max())
             if bool(valid.any()) else None),
        'note': 'invalid pairs are excluded from the decomposition only: they are still scored and '
                'ranked by the uniform F.normalize(eps=1e-6) definition and no query is dropped',
        'rho_i_positive_pairs': quantiles(rho_i[diagonal].numpy()),
        'rho_i_negative_pairs': quantiles(rho_i[off].numpy()),
        'rho_t_positive_pairs': quantiles(rho_t[diagonal].numpy()),
        'rho_t_negative_pairs': quantiles(rho_t[off].numpy()),
        'factor_positive_pairs': quantiles(factor[diagonal].numpy()),
        'factor_negative_pairs': quantiles(factor[off].numpy()),
        'factor_positive_mean': f(factor[diagonal].mean()),
        'factor_negative_mean': f(factor[off].mean()),
        'factor_on_worst_negative_of_q3': {
            'I2T': quantiles(torch.gather(factor, 1, q3_private['I2T']['_worst'].unsqueeze(1))
                             .squeeze(1).numpy()),
            'T2I': quantiles(torch.gather(factor.t().contiguous(), 1,
                                          q3_private['T2I']['_worst'].unsqueeze(1))
                             .squeeze(1).numpy())},
        'interpretation_guard': 'a factor below 1 shrinks both sides of a pair; it is not harmful by '
                                'itself and can suppress wrong matches as easily as correct ones -- '
                                'only the resulting ranking decides that'}

    q3_private = {'I2T': retrieval_metrics(q3), 'T2I': retrieval_metrics(q3.t().contiguous())}
    q_cap_private = {'I2T': retrieval_metrics(q_cap),
                     'T2I': retrieval_metrics(q_cap.t().contiguous())}
    geometry_report = {
        'stats': geometry_stats,
        'q3_metrics': {k: public(v) for k, v in q3_private.items()},
        'qcap_metrics': {k: public(v) for k, v in q_cap_private.items()},
        'qcap_vs_q3_paired': {direction: paired_against(q3_private[direction], q_cap_private[direction])
                              for direction in ('I2T', 'T2I')},
        'per_query_rank': {'q3_I2T': q3_private['I2T']['per_query_rank'],
                           'q3_T2I': q3_private['T2I']['per_query_rank'],
                           'qcap_I2T': q_cap_private['I2T']['per_query_rank'],
                           'qcap_T2I': q_cap_private['T2I']['per_query_rank']},
        'note': 'Qcap is a forward-replacement readout used to understand why Q3 behaves as it does. '
                'It is not a new retrieval method, it does not replace the native CLS/EOS numbers '
                'and it does not change the frozen FAIL verdict of the @500 run.',
    }

    # ---------------------------------------------------------------- hard queries
    labels = ['%d/%d' % (i, a_) for i, a_ in zip(image_ids, annotation_ids)]
    groups = []
    for path, matrix, cap, fac, val, direction in (
            ('L3', q3, q_cap, factor, valid, 'I2T'),
            ('L3', q3.t().contiguous(), q_cap.t().contiguous(), factor.t().contiguous(),
             valid.t().contiguous(), 'T2I'),
            ('L2', q2, None, None, None, 'I2T'),
            ('L2', q2.t().contiguous(), None, None, None, 'T2I'),
            ('QCAP', q_cap, None, None, None, 'I2T'),
            ('QCAP', q_cap.t().contiguous(), None, None, None, 'T2I')):
        groups.append((path, direction,
                       collect_hard_queries(matrix, cap, fac, val, labels, path, args,
                                            with_geometry=cap is not None,
                                            direction=direction)))
    # round-robin across the six (path, direction) groups, so the cap cannot silently drop a whole
    # path -- a plain concatenation would have filled the list with the first two groups only
    deduped, seen = [], set()
    for rank in range(args.hard_queries_per_path):
        for path, direction, rows in groups:
            if rank >= len(rows):
                continue
            row = rows[rank]
            key = (row['path'], row['direction'], row['query_index'])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= args.max_hard_queries:
                break
        if len(deduped) >= args.max_hard_queries:
            break

    sample_index = {s['candidate_index']: s for s in manifest['samples']}
    for row in deduped:
        query = sample_index.get(row['query_index'], {})
        negative = sample_index.get(row['strongest_negative_index'], {})
        row['query_caption'] = query.get('caption')
        row['query_file_name'] = query.get('file_name')
        row['strongest_negative_caption'] = negative.get('caption')
        row['strongest_negative_file_name'] = negative.get('file_name')
        row['strongest_negative_kind'] = 'annotation-indexed negative candidate'
        row['visual_verification'] = 'NOT RUN (no image-inspection tool in this session)'
        index = row['query_index']
        row['mask_counts'] = {
            'mI_kept': int((mI[index] >= 0.5).sum()), 'mT_kept': int((mT[index] >= 0.5).sum()),
            'intersection_kept': int((c_cap[index] >= 0.5).sum()),
            'energy_rho_i_positive': f(geometry['c'][index, index] / geometry['a'][index, index]),
            'energy_rho_t_positive': f(geometry['d'][index] / geometry['b'][index]),
            'factor_positive': f(factor[index, index])}
    queries_csv = os.path.join(diagnostics_dir, 'hs_mask_geometry_probe_queries.csv')
    write_queries_csv(queries_csv, deduped)

    report = {
        'probe': 'trimask_hs_geometry_probe',
        'read_only': True,
        'new_optimizer_updates': 0,
        'not_a_canonical_evaluation': True,
        'scope': 'S0-TriMask-HS @500 checkpoint, %d COCO val2017 images x %d captions, forward-only'
                 % (n, n),
        'checkpoint': {
            'path': checkpoint, 'sha256_before': checkpoint_sha_before,
            'sha256_after': checkpoint_sha_after,
            'sha256_unchanged': checkpoint_sha_before == checkpoint_sha_after,
            'identity': identity, 'meta': checkpoint_meta,
            'clip_tensors': clip_tensor_count,
            'text_mask_net_tensors': text_branch_tensor_count,
            'loaded_with': 'the full checkpoint: payload["model"] into the CLIP student and '
                           'payload["text_mask_net"] into the text branch; the bare student alone '
                           'would have left the text head randomly initialised'},
        'run_status': {'path': run_status_path, 'sha256_before': status_sha_before,
                       'sha256_after': status_sha_after,
                       'unchanged': status_sha_before == status_sha_after,
                       'note': 'the probe never writes run_status.json; the @500 run stays complete '
                               'and its FAIL verdict stays FAIL'},
        'parameter_state': {'before': state_before, 'after': state_after,
                            'unchanged': state_before == state_after},
        'manifest': {'path': manifest_path, 'sha256': manifest_sha, 'count': manifest['count'],
                     'selection': manifest['selection'], 'source': manifest['source'],
                     'skipped_captions': manifest['skipped_captions'],
                     'image_ids': image_ids, 'annotation_ids': annotation_ids,
                     'token_sha256': [s['token_sha256'] for s in manifest['samples']]},
        'dtypes_and_precision': dtypes,
        'mask_statistics': mask_stats,
        'cache': {'path': 'in-memory only; no feature tensor is written to disk or committed',
                  'v_shape': list(v.shape), 't_shape': list(t.shape), 'mask_shape': list(mT.shape),
                  'encoding_passes': {'images': 1, 'captions': 1}},
        'reconciliation_with_trained_scoring_form': reconciliation,
        'diagnostic_A_variance': variance,
        'diagnostic_B_replacements': {
            'variants': variant_public,
            'identity_checks': identity_checks,
            'shuffle_permutations': {'seeds': list(SHUFFLE_SEEDS),
                                     'method': 'fixed-point-free permutation from '
                                               'numpy.random.default_rng(seed), saved in this file; '
                                               'the seed was never chosen by its result',
                                     'report': permutation_report},
            'shuffled_aggregate': shuffled_aggregate,
            'not_used': 'L1 and loss_total: L1 does not depend on the text gate, so a total loss '
                        'would hide exactly the difference this diagnostic is about'},
        'diagnostic_C_geometry': geometry_report,
        'hard_queries': deduped,
        'tie_rule': 'rank = 1 + #{j != pos : score[j] > score[pos]}; a tie never beats the positive '
                    'and M_max == 0 is reported as a tie count, not as a loss',
        'not_run': [
            'image-level verification of the hard-query negatives (no image-inspection tool)',
            'browser visual acceptance of the dashboard section (no browser tool)',
            'any claim about long captions or the Urban-1k distribution: this pool is 256 short COCO '
            'val2017 captions',
            'any training-causal claim: this is a forward-only intervention, not a retrained arm',
        ],
        'timing': {'wall_seconds': f(time.time() - started)},
    }
    probe_path = os.path.join(diagnostics_dir, 'hs_mask_geometry_probe.json')
    write_json_atomic(probe_path, report)

    summary = build_summary(report)
    summary['probe_path'] = probe_path
    summary['queries_csv'] = queries_csv
    summary['diagnostics_dir'] = diagnostics_dir
    write_json_atomic(os.path.join(REPO, args.summary_out), summary)

    print('WROTE %s' % probe_path)
    print('WROTE %s' % os.path.join(REPO, args.summary_out))
    shares = summary['diagnostic_A_variance']['share_of_total'] or {}
    print('PROBE_SUMMARY ' + json.dumps({
        'v_level_share': shares.get('level'), 'v_profile_share': shares.get('profile'),
        'v_interaction_share': shares.get('interaction'),
        'v_caption_dependent_share': (summary['diagnostic_A_variance']['v_caption_dependent']
                                      / summary['diagnostic_A_variance']['v_total']
                                      if summary['diagnostic_A_variance']['v_total'] else None),
        'normal_L3_I2T': {k: summary['diagnostic_B_replacements']['variants']['NORMAL']['L3_I2T'][k]
                          for k in ('ce', 'R@1', 'mrr')},
        'ones_L3_I2T': {k: summary['diagnostic_B_replacements']['variants']['ONES']['L3_I2T'][k]
                        for k in ('ce', 'R@1', 'mrr')},
        'q3_identity_valid_fraction': summary['diagnostic_C_geometry']['stats']['valid_fraction'],
        'q3_identity_max_abs_error':
            summary['diagnostic_C_geometry']['stats']['max_abs_error_on_valid_pairs'],
        'parameter_state_unchanged': summary['parameter_state']['unchanged'],
    }, sort_keys=True))


def collect_hard_queries(q, q_cap, factor, valid, labels, path, args, with_geometry,
                         direction='I2T'):
    """The CE-ranked hard queries for one path and direction, with both margin kinds."""
    metrics = retrieval_metrics(q)
    diagonal, rank, ce = metrics['_diagonal'], metrics['_rank'], metrics['_ce']
    worst = metrics['_worst']
    s_max_neg, m_max = metrics['_s_max_neg'], metrics['_m_max']
    m_lse = metrics['_m_lse']
    lse_neg = diagonal - m_lse
    if with_geometry:
        cap_metrics = retrieval_metrics(q_cap)
        cap_diag, cap_rank, cap_ce = cap_metrics['_diagonal'], cap_metrics['_rank'], cap_metrics['_ce']
        cap_worst = cap_metrics['_worst']
        cap_s_max_neg = cap_metrics['_s_max_neg']
    order = torch.argsort(ce, descending=True)[:args.hard_queries_per_path]
    rows = []
    for index in order.tolist():
        j = int(worst[index])
        row = {
            'path': path, 'direction': direction, 'query_index': int(index),
            'query_label': labels[index], 'strongest_negative_index': j,
            'strongest_negative_label': labels[j],
            's_pos': f(diagonal[index]), 's_max_negative': f(s_max_neg[index]),
            'logsumexp_negatives': f(lse_neg[index]), 'm_max': f(m_max[index]),
            'm_lse': f(m_lse[index]), 'ce': f(ce[index]), 'rank': int(rank[index]),
            'score_of_worst_negative': f(q[index, j]),
        }
        if with_geometry:
            row.update({
                'qcap_s_pos': f(cap_diag[index]), 'qcap_rank': int(cap_rank[index]),
                'qcap_ce': f(cap_ce[index]), 'qcap_s_max_negative': f(cap_s_max_neg[index]),
                'qcap_score_of_worst_negative': f(q_cap[index, j]),
                'qcap_new_worst_index': int(cap_worst[index]),
                'qcap_new_worst_label': labels[int(cap_worst[index])],
                'qcap_new_worst_score': f(q_cap[index, int(cap_worst[index])]),
                'factor_of_positive': f(factor[index, index]),
                'factor_of_worst_negative': f(factor[index, j]),
                'valid_pairs_for_query': int(valid[index].sum())})
        rows.append(row)
    return rows


def derangement(n, seed):
    """A fixed-point-free permutation, drawn once and then saved -- never re-drawn by its score."""
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(n)
        if not np.any(permutation == np.arange(n)):
            return [int(x) for x in permutation]


def write_queries_csv(path, rows):
    columns = ['path', 'direction', 'query_index', 'query_label', 'rank', 'ce', 'm_max', 'm_lse',
               's_pos', 's_max_negative', 'strongest_negative_index', 'strongest_negative_label',
               'score_of_worst_negative', 'qcap_rank', 'qcap_ce', 'qcap_s_pos',
               'qcap_score_of_worst_negative', 'qcap_new_worst_index', 'qcap_new_worst_label',
               'factor_of_positive', 'factor_of_worst_negative', 'valid_pairs_for_query',
               'mask_counts', 'query_caption', 'strongest_negative_caption',
               'strongest_negative_file_name', 'strongest_negative_kind', 'visual_verification']
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            entry = {k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                     for k, v in row.items()}
            writer.writerow(entry)
    os.replace(temporary, path)


def build_summary(report):
    """The small, caption-free summary that is safe to commit."""
    keep = ('probe', 'read_only', 'new_optimizer_updates', 'not_a_canonical_evaluation', 'scope',
            'checkpoint', 'run_status', 'parameter_state', 'manifest', 'dtypes_and_precision',
            'mask_statistics', 'cache', 'reconciliation_with_trained_scoring_form',
            'diagnostic_A_variance', 'diagnostic_B_replacements', 'diagnostic_C_geometry',
            'hard_queries', 'tie_rule', 'not_run', 'timing')
    summary = {key: report[key] for key in keep}
    simplified = []
    for row in summary['hard_queries']:
        simplified.append({k: v for k, v in row.items()
                           if k not in ('query_caption', 'strongest_negative_caption',
                                        'query_file_name', 'strongest_negative_file_name')})
    summary['hard_queries'] = simplified
    # the per-token sha256 list is derivable from the run-local manifest and would be the largest
    # part of the committed summary; the necessary id lists stay
    manifest_block = summary.get('manifest')
    if isinstance(manifest_block, dict):
        manifest_block.pop('token_sha256', None)
    # the per-query rank arrays of every variant would dominate the committed summary; they stay in
    # the full probe JSON, and only the Q3/Qcap ones (the ones the report reasons about) are kept
    variants = summary['diagnostic_B_replacements'].get('variants') or {}
    for name, entry in variants.items():
        for key, value in list(entry.items()):
            if isinstance(value, dict) and 'per_query_rank' in value:
                value.pop('per_query_rank', None)
    return summary


if __name__ == '__main__':
    main()
