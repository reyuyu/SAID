"""Read-only CLIP-512 functional probe: which representation changes actually move retrieval, do
image-object deletions form a compatible cross-modal direction, and does SmartCLIP's coordinate mask
select the visually relevant content?

Two models only, both verified by their own payload and file hash at run time:

    A  shared_init   the project's common initialisation (248-token CLIP + visual mask_net)
    B  s0_500        S0_smartclip @ 500 updates (objective said_cls_cvssl, arm S0_smartclip)

Zero new optimizer updates: no optimizer, no backward, no parameter write, no resumed training, no
loss/threshold/sparsity change, no canonical re-evaluation and no follow-up method search. Checkpoint
sha256 and the parameter-state digest are compared before/after, and ``run_status.json`` is untouched.

Phase A (text functionality, always delivered) and phase B (controlled crops) run in one pass; if the
crop stage cannot build enough scenes it records why and phase A still lands.

Corrections relative to the earlier text-nuisance probe, all of which this file implements:

* the paraphrase of "A brown horse is running in a field." no longer changes the colour;
* I2T uses ``Q`` and T2I uses ``Q.T`` — both directions are ranked separately, never one row
  direction reported as "both directions";
* ties are broken by candidate id, never in favour of the correct label, and tie counts are reported;
* the paired test is on the discrete per-query outcome (hit / rank change), and effect sizes plus
  change counts are the headline, not a wall of exploratory p-values;
* Jaccard is intersection/union; the intersection count at rank k is called ``overlap_at_k``;
* an L2 distance between normalised vectors is never called "a percentage of semantic change";
* effective token length uses the repository's EOT position (``argmax``), never ``token_id != 0``,
  and every appended sentence is checked to have really entered the encoder.

Literature is kept separate from measurement (see the report): the modality-gap / object-bias /
information-imbalance findings of "Two Effects, One Trigger" (ICLR 2025) and the alignment /
uniformity definitions of Wang & Isola (ICML 2020) are cited as *external* claims; this probe only
measures a simple mean/centroid gap and never claims to reproduce RMG or a causal test.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SHARED_INIT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt'
S0_500 = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/ddpfix_step500_S0_smartclip/'
          'cvssl_S0_smartclip_step000500.pt')
HS_RUN = os.path.join(REPO, 'runs_salu', 'said_s0_trimask_hs_v02', 'step500')
MANIFEST = os.path.join(HS_RUN, 'diagnostics', 'manifest.json')
OUT_DIR = os.path.join(HS_RUN, 'diagnostics', 'clip512_functional_probe')
DEFAULT_SUMMARY = 'docs/clip512_functional_probe/summary.json'
DEFAULT_REPORT_DIR = 'docs/clip512_functional_probe'

FIXED_SCALE = 100.0
NORM_EPS = 1e-6
TOKEN_WIDTH = 248
K_VALUES = (1, 5, 10)
MAX_TEXTS_PER_MODEL = 1200
MAX_VIEWS_PER_MODEL = 300
POOL_IMAGES = 128
MAX_CROP_SCENES = 32
MAX_SCENES_PER_CATEGORY_PAIR = 4
SCALE_CONTROL = (0.5, 1.0, 2.0)

MODEL_SPECS = [
    {'key': 'shared_init', 'label': 'A · 项目共同初始化 CLIP（248-token）', 'path': SHARED_INIT,
     'expected_sha256': 'c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8'},
    {'key': 's0_500', 'label': 'B · S0_smartclip @500（视觉 mask，无文本门）', 'path': S0_500,
     'expected_sha256': '758dcdd2a9112d54ccb8784e211d329297209c5d7be26cd7a89e68185867c743'},
]

BASE_CAPTIONS = [
    'A dog is sitting on the grass.',
    'A black cat is lying on a red mat.',
    'A red car is parked beside a tree.',
    'Two people are riding bicycles.',
    'A child is holding a yellow balloon.',
    'A bird is standing on a wooden fence.',
    'A blue boat is floating on a lake.',
    'A bowl of fruit is on a table.',
    'A brown horse is running in a field.',
    'A person is carrying a green bag.',
    'A train is passing a station.',
    'A cup is beside a plate.',
]
# semantics-preserving paraphrases: same object, colour, count and spatial relation. The previous
# version paraphrased the brown horse as "A white horse ...", which changed a colour; fixed here.
PARAPHRASES = [
    'There is a dog seated on the lawn.',
    'A cat that is black rests on a mat that is red.',
    'Next to a tree stands a parked red car.',
    'A pair of people ride bicycles.',
    'A yellow balloon is being held by a child.',
    'On a fence made of wood stands a bird.',
    'A boat that is blue floats on a lake.',
    'On a table sits a bowl filled with fruit.',
    'In a field, a horse that is brown is running.',
    'A green bag is being carried by a person.',
    'A station is being passed by a train.',
    'Next to a plate there is a cup.',
]
VISUAL_CHANGES = [
    'A cat is sitting on the grass.',
    'A white cat is lying on a red mat.',
    'A red car is parked behind a tree.',
    'Three people are riding bicycles.',
    'A child is holding a blue balloon.',
    'A bird is standing on a metal fence.',
    'A blue boat is floating on a river.',
    'A bowl of fruit is on a chair.',
    'A brown horse is walking in a field.',
    'A person is carrying a black bag.',
    'A bus is passing a station.',
    'A cup is behind a plate.',
]
CHANGE_DESCRIPTIONS = [
    'dog -> cat', 'colour black -> white', 'spatial beside -> behind', 'count two -> three',
    'colour yellow -> blue', 'material wooden -> metal', 'lake -> river', 'table -> chair',
    'running -> walking', 'colour green -> black', 'train -> bus', 'spatial beside -> behind',
]
# the paraphrase fix is checked mechanically: every locked concept must survive. Each entry is a list
# of synonym groups -- one member of each group must appear in the paraphrase, so "grass" -> "lawn"
# is allowed while "brown" -> "white" is not.
SEMANTIC_LOCK = [
    [['dog'], ['grass', 'lawn']],
    [['black'], ['cat'], ['red'], ['mat']],
    [['red'], ['car'], ['tree']],
    [['two', 'pair'], ['people'], ['bicycle', 'bicycles']],
    [['yellow'], ['balloon'], ['child']],
    [['wooden', 'wood'], ['fence'], ['bird']],
    [['blue'], ['boat'], ['lake']],
    [['fruit'], ['table'], ['bowl']],
    [['brown'], ['horse'], ['field']],
    [['green'], ['bag'], ['person']],
    [['train'], ['station']],
    [['cup'], ['plate']],
]
R_SUFFIXES = {
    'R1': 'I personally like this picture.',
    'R2': 'I personally dislike this picture.',
    'R3': 'Thank you for sharing this picture.',
    'R4': 'This is just my personal opinion.',
}
R_ORDER = ('R1', 'R2', 'R3', 'R4')
COLOUR_NUMBERS = ('black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'orange', 'grey',
                  'gray', 'pink', 'purple', 'two', 'three', 'four', 'five')


# --------------------------------------------------------------------------- small helpers
def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode('utf-8'))
        digest.update(np.ascontiguousarray(state[key].detach().float().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
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


def unit_rows(matrix, eps=NORM_EPS):
    return F.normalize(matrix.float(), dim=-1, eps=eps)


def token_sha(tokenizer, text):
    tokens = tokenizer([text], context_length=TOKEN_WIDTH)
    ids = np.asarray(tokens[0].tolist(), dtype=np.int64)
    return {'sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
            'eot_index': int(np.argmax(ids)), 'effective_length': int(np.argmax(ids)) + 1,
            'ids': ids.tolist()}


def suffix_entered(base_ids, suffix_ids, combined_ids, tolerance=1):
    content = [int(t) for t in suffix_ids[1:suffix_ids.index(49407)]]
    if not content:
        return True
    start = base_ids.index(49407)
    end = combined_ids.index(49407)
    tail = combined_ids[start:end]
    return sum(1 for token in content if token in tail) >= len(content) - tolerance


def jaccard(set_a, set_b):
    """True Jaccard: intersection over union (the earlier probe labelled intersection/k as one)."""
    union = set(set_a) | set(set_b)
    if not union:
        return 0.0
    return len(set(set_a) & set(set_b)) / float(len(union))


def overlap_at_k(order_a, order_b, k):
    """Intersection count at a fixed rank budget -- deliberately not called Jaccard."""
    return len(set(order_a[:k]) & set(order_b[:k]))


def ranked_metrics(scores, k_values=K_VALUES):
    """R@K / MRR / CE / entropy / margins for a matrix whose positive is the diagonal.

    ``rank = 1 + #{j : score_j > score_pos or (score_j == score_pos and id_j < id_pos)}``: ties are
    broken by candidate id (never in favour of the correct label), and ties are counted separately.
    """
    scores = scores.float()
    n = scores.shape[0]
    ids = torch.arange(n).unsqueeze(0).expand(n, n)
    diagonal = torch.diagonal(scores)
    off = scores.clone()
    off.fill_diagonal_(float('-inf'))
    s_max_negative = off.max(dim=1).values
    lse_negatives = torch.logsumexp(off, dim=1)
    tie_matrix = (off == diagonal.unsqueeze(1))
    better = (off > diagonal.unsqueeze(1))
    tie_better = tie_matrix & (ids < ids.diagonal().unsqueeze(1))
    rank = 1 + better.sum(dim=1) + tie_better.sum(dim=1)
    m_lse = diagonal - lse_negatives
    ce = F.softplus(-m_lse)
    probabilities = torch.softmax(scores, dim=1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
    out = {
        'n_queries': int(n), 'R@1': f((rank <= 1).float().mean()),
        'R@5': f((rank <= 5).float().mean()), 'R@10': f((rank <= 10).float().mean()),
        'mrr': f((1.0 / rank.float()).mean()), 'ce': f(ce.mean()),
        'entropy': f(entropy.mean()), 'rank_mean': f(rank.float().mean()),
        'm_max_mean': f((diagonal - s_max_negative).mean()), 'm_lse_mean': f(m_lse.mean()),
        'tie_pairs': int(tie_matrix.sum().item()), 'tie_queries': int((tie_matrix.sum(dim=1) > 0).sum().item()),
        'ce_identity_max_abs_diff': f((ce - (torch.logsumexp(scores, dim=1) - diagonal)).abs().max()),
        'per_query_rank': [int(x) for x in rank.tolist()],
        'per_query_ce': [f(x) for x in ce.tolist()],
        'per_query_entropy': [f(x) for x in entropy.tolist()],
        'per_query_m_max': [f(x) for x in (diagonal - s_max_negative).tolist()],
        'per_query_m_lse': [f(x) for x in m_lse.tolist()],
    }
    out['_rank'] = rank
    out['_diagonal'] = diagonal
    out['_worst'] = off.argmax(dim=1)
    return out


def public(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith('_')}


def paired_outcome(base_ranks, other_ranks, base_hits, other_hits):
    """Discrete per-query outcome, the honest paired summary (no exploratory p-value wall)."""
    base_ranks = np.asarray(base_ranks)
    other_ranks = np.asarray(other_ranks)
    base_hits = np.asarray(base_hits)
    other_hits = np.asarray(other_hits)
    return {
        'R@1_hits_lost': int((base_hits & ~other_hits).sum()),
        'R@1_hits_gained': int((~base_hits & other_hits).sum()),
        'R@1_hit_count_change': int(other_hits.sum() - base_hits.sum()),
        'rank_worsened': int((other_ranks > base_ranks).sum()),
        'rank_improved': int((other_ranks < base_ranks).sum()),
        'rank_unchanged': int((other_ranks == base_ranks).sum()),
        'median_rank_change': float(np.median(other_ranks - base_ranks)),
        'max_rank_worsening': int(max(0, (other_ranks - base_ranks).max())),
    }


def common_offset_invariance(delta_q, delta_t, images_unit, suffix):
    """I2T: removing the per-row constant ``100 v_i . mu_R`` must not change ranking or CE.

    The comparison uses ``ranked_metrics``, which masks the diagonal to ``-inf``: a per-row shift is
    then exactly invariant in real arithmetic (both the argmax order and the log-sum-exp margin), and
    only floating-point rounding is left. Masking the diagonal with ``0`` instead would move the
    masked entry relative to the shifted negatives and fake a reordering, so that is not done here.
    """
    mu = delta_t.mean(dim=0)
    constant = FIXED_SCALE * (images_unit @ mu)
    adjusted = delta_q - constant.unsqueeze(1)
    base = ranked_metrics(delta_q)
    shifted = ranked_metrics(adjusted)
    ce_delta = max(abs(a - b) for a, b in zip(base['per_query_ce'], shifted['per_query_ce']))
    entropy_delta = max(abs(a - b) for a, b in zip(base['per_query_entropy'],
                                                  shifted['per_query_entropy']))
    return {'mu_norm': f(mu.norm()), 'row_constant_mean': f(constant.mean()),
            'per_query_rank_identical': base['per_query_rank'] == shifted['per_query_rank'],
            'argmax_negative_identical': base['_worst'].tolist() == shifted['_worst'].tolist(),
            'ce_max_abs_change': f(ce_delta), 'entropy_max_abs_change': f(entropy_delta),
            'note': 'this is the I2T statement only: for T2I the image side differs per query, so the '
                    'same constant is not shared and the invariance does not follow'}


def candidate_covariance(images_unit, delta_t):
    """``DeltaT^T C_V DeltaT`` must equal the variance of the un-scaled score change over candidates."""
    centered = images_unit - images_unit.mean(dim=0, keepdim=True)
    covariance = (centered.t() @ centered) / images_unit.shape[0]
    quadratic = (delta_t @ covariance * delta_t).sum(dim=1)
    score_change = images_unit @ delta_t.t()                 # [n_candidates, n_texts]
    variance = score_change.var(dim=0, unbiased=False)
    return {'max_abs_diff': f((quadratic - variance).abs().max()),
            'quadratic_quantiles': quantiles(quadratic.numpy()),
            'note': 'a sensitivity of score separation against the current candidate pool, not an '
                    'amount of useful semantic information, and it says nothing about directions '
                    'outside this pool'}


def coordinate_vs_subspace(deltas, half_labels, k_values=(4, 8, 16)):
    """Same rank budget for a coordinate subset and for SVD directions, both fitted on the first half.

    Centered and uncentered variants are reported separately; the second half only *evaluates* the
    orderings and the mean/directions fixed on the first half (no refitting, no re-selection).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    labels = np.asarray(half_labels)
    first, second = deltas[labels == 0], deltas[labels == 1]
    out = {}
    for mode in ('uncentered', 'centered'):
        first_used = first
        second_used = second
        if mode == 'centered':
            mean_first = first.mean(axis=0)
            first_used = first - mean_first
            second_used = second - mean_first          # the first half's mean, applied to both
        energy_first = (first_used ** 2).mean(axis=0)
        total_second = float((second_used ** 2).sum())
        order = np.argsort(-energy_first)
        _, _, right = np.linalg.svd(first_used, full_matrices=False)
        components = right[:max(k_values)]
        coordinate_capture, subspace_capture = {}, {}
        for k in k_values:
            coordinate_capture['k%d' % k] = float(
                (second_used[:, order[:k]] ** 2).sum() / total_second) if total_second else None
            projection = second_used @ components[:k].T
            subspace_capture['k%d' % k] = float(
                (projection ** 2).sum() / total_second) if total_second else None
        out[mode] = {
            'coordinate_topk_capture': coordinate_capture,
            'subspace_topk_capture': subspace_capture,
            'coordinate_order_top16': [int(i) for i in order[:16]],
            'same_rank_budget': True,
            'n_pairs_first': int(first.shape[0]), 'n_pairs_second': int(second.shape[0]),
        }
    return out


def distribution_statistics(images_unit, texts_unit):
    """Centroid gap, paired alignment and per-modality uniformity (stable log-mean-exp form)."""
    def uniformity(x):
        distances = torch.cdist(x, x, p=2) ** 2
        n = x.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        values = -2.0 * distances[mask]
        return f(torch.logsumexp(values, dim=0) - math.log(values.numel()))

    return {
        'centroid_gap': f((images_unit.mean(dim=0) - texts_unit.mean(dim=0)).norm()),
        'paired_alignment': f(((images_unit - texts_unit) ** 2).sum(dim=1).mean()),
        'image_uniformity': uniformity(images_unit),
        'text_uniformity': uniformity(texts_unit),
        'note': 'uniformity is a set statistic, not a semantic label for any coordinate; nothing here '
                'is optimised by dropping dimensions',
    }


# --------------------------------------------------------------------------- view geometry
def canvas_transform(preprocess):
    """The official preprocessing minus ToTensor and Normalize: the un-normalised 224x224 canvas.

    Only the *geometric* steps are kept (Resize + CenterCrop). Leaving ``ToTensor`` in would make the
    "canvas" a tensor and the later Resize would fail, which is exactly the kind of double-transform
    bug this helper exists to avoid.
    """
    from torchvision import transforms
    steps = [step for step in preprocess.transforms
             if not isinstance(step, (transforms.Normalize, transforms.ToTensor))]
    return transforms.Compose(steps), steps


def tensor_transform(preprocess):
    """Resize (to an exact square) + ToTensor + Normalize, applied exactly once.

    ``Resize(224)`` with an int only scales the *shorter* edge, so a 224x226 crop would silently stay
    224x226; an explicit ``(224, 224)`` forces the square output every view must have.
    """
    from torchvision import transforms
    normalise = [step for step in preprocess.transforms if isinstance(step, transforms.Normalize)]
    return transforms.Compose([transforms.Resize((224, 224),
                                                 interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.ToTensor()] + normalise)


def unify_view(image, preprocess):
    """``(canvas, scale, offset)`` for the model's real field of view.

    The canvas is the official Resize+CenterCrop result *without* Normalize, the offsets follow
    torchvision's own ``center_crop`` rounding, and the caller cross-checks the analytic mapping
    against the pipeline output so a box can never be placed with an assumed transform.
    """
    from torchvision import transforms
    resize = next(step for step in preprocess.transforms if isinstance(step, transforms.Resize))
    resized = resize(image.convert('RGB'))
    canvas = transforms.CenterCrop(224)(resized)
    offset = (int(round((resized.size[0] - 224) / 2.0)), int(round((resized.size[1] - 224) / 2.0)))
    scale = resized.size[1] / float(image.size[1])          # height ratio == width ratio
    return canvas, scale, offset, resized.size


def map_box_to_canvas(box, scale, offset):
    x1, y1, x2, y2 = [float(v) for v in box]
    mapped = (x1 * scale - offset[0], y1 * scale - offset[1],
              x2 * scale - offset[0], y2 * scale - offset[1])
    clipped = (max(0.0, min(224.0, mapped[0])), max(0.0, min(224.0, mapped[1])),
               max(0.0, min(224.0, mapped[2])), max(0.0, min(224.0, mapped[3])))
    return mapped, clipped


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return box_area((x1, y1, x2, y2))


def square_from_box(box, canvas=224, context=0.10):
    """A square crop covering the box plus ``context``, shifted (never shrunk) to stay in canvas."""
    width = (box[2] - box[0]) * (1.0 + 2 * context)
    height = (box[3] - box[1]) * (1.0 + 2 * context)
    side = min(float(canvas), max(width, height, 8.0))
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    x1 = min(max(0.0, center_x - side / 2.0), canvas - side)
    y1 = min(max(0.0, center_y - side / 2.0), canvas - side)
    return (x1, y1, x1 + side, y1 + side)


def grid_control_crop(keep_boxes, canvas=224, side_ratio=0.90, steps=8):
    """The first square in a fixed grid that keeps every listed box at least 95% inside."""
    side = canvas * side_ratio
    for iy in range(steps + 1):
        for ix in range(steps + 1):
            x1 = (canvas - side) * ix / float(steps)
            y1 = (canvas - side) * iy / float(steps)
            rect = (x1, y1, x1 + side, y1 + side)
            if all(intersection_area(rect, box) >= 0.95 * box_area(box) for box in keep_boxes):
                return rect
    return None


def retained_fraction(crop, box):
    area = box_area(box)
    if area <= 0:
        return 0.0
    return intersection_area(crop, box) / area


def segmentation_point_retention(segmentation, crop, scale=None, offset=None):
    """Secondary, point-based check on the polygon annotation.

    The polygon coordinates live in the *original image* frame, so they are mapped through the same
    scale/offset the boxes use before being tested against a canvas-space crop; comparing raw polygon
    coordinates with a canvas crop would report 0% for every scene and prove nothing.
    """
    points = []
    if not segmentation:
        return None
    for polygon in segmentation:
        for index in range(0, len(polygon) - 1, 2):
            x, y = polygon[index], polygon[index + 1]
            if scale is not None and offset is not None:
                x, y = x * scale - offset[0], y * scale - offset[1]
            points.append((x, y))
    if not points:
        return None
    inside = sum(1 for x, y in points if crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3])
    return inside / float(len(points))


def select_crop_scenes(instances_path, prefer_ids, preprocess, max_scenes=MAX_CROP_SCENES,
                       max_per_pair=MAX_SCENES_PER_CATEGORY_PAIR, min_canvas_share=0.02,
                       seed=0):
    """Deterministic scene selection, all geometry fixed before any model output is inspected."""
    payload = json.load(open(instances_path, 'r', encoding='utf-8'))
    file_name = {int(image['id']): image['file_name'] for image in payload['images']}
    size = {int(image['id']): (int(image['width']), int(image['height']))
            for image in payload['images']}
    categories = {int(cat['id']): cat['name'] for cat in payload['categories']}
    by_image = defaultdict(list)
    for annotation in payload['annotations']:
        if int(annotation.get('iscrowd', 0)) == 1:
            continue
        by_image[int(annotation['image_id'])].append(annotation)
    for key in by_image:
        by_image[key].sort(key=lambda a: int(a['id']))
    order = [int(i) for i in prefer_ids]
    order += [int(i) for i in sorted(file_name) if int(i) not in set(order)]
    pair_counts = defaultdict(int)
    scenes, rejected = [], []
    reason_counts = defaultdict(int)

    def reject(image_id, reason):
        reason_counts[reason] += 1
        if len(rejected) < 60:
            rejected.append({'image_id': image_id, 'reason': reason})

    for image_id in order:
        if len(scenes) >= max_scenes:
            break
        if image_id not in file_name:
            continue
        record = {'image_id': image_id, 'file_name': file_name[image_id],
                  'original_size': size[image_id]}
        build = build_scene_views(image_id, by_image[image_id], categories, file_name,
                                 size[image_id], record, min_canvas_share, preprocess)
        if build.get('scene') is None:
            reject(image_id, build.get('reason'))
            continue
        scene = build['scene']
        pair = tuple(sorted((scene['A']['category'], scene['B']['category'])))
        if pair_counts[pair] >= max_per_pair:
            reject(image_id, 'category pair %s already used %d times'
                   % ('+'.join(pair), pair_counts[pair]))
            continue
        pair_counts[pair] += 1
        scene['category_pair'] = '+'.join(pair)
        scenes.append(scene)
    return scenes, rejected, dict(reason_counts), categories, file_name


def build_scene_views(image_id, annotations, categories, file_name, original_size, record,
                      min_canvas_share, preprocess):
    """Unify the field of view, then accept only scenes meeting every fixed geometric rule."""
    from PIL import Image
    path = os.path.join('/root/datasets/coco/val2017', file_name[image_id])
    with Image.open(path) as handle:
        image = handle.convert('RGB')
        canvas, scale, offset, resized = unify_view(image, preprocess)
        official = canvas_transform(preprocess)[0](image)
    record['official_view'] = {'resized_size': list(resized), 'scale': scale,
                               'center_crop_offset': list(offset), 'canvas_size': [224, 224],
                               'analytic_mapping_matches_pipeline':
                                   bool(np.array_equal(np.asarray(canvas), np.asarray(official)))}
    boxes = []
    for annotation in annotations:
        mapped, clipped = map_box_to_canvas(annotation['bbox'], scale, offset)
        area = box_area(clipped)
        if area <= 0:
            continue
        boxes.append({'annotation_id': int(annotation['id']),
                      'category_id': int(annotation['category_id']),
                      'category': categories.get(int(annotation['category_id']), 'unknown'),
                      'bbox_canvas': [round(v, 2) for v in clipped],
                      'area_canvas': area,
                      'share_of_canvas': area / float(224 * 224),
                      'segmentation': annotation.get('segmentation'),
                      'mapped_bbox': [round(v, 2) for v in mapped]})
    keep = [box for box in boxes if box['share_of_canvas'] >= min_canvas_share]
    if len(keep) < 2:
        return {'scene': None, 'reason': 'fewer than two boxes above %.0f%% of the canvas'
                                         % (100 * min_canvas_share)}
    by_category = defaultdict(list)
    for box in keep:
        by_category[box['category']].append(box)
    names = sorted(by_category)
    for first_index in range(len(names)):
        for second_index in range(first_index + 1, len(names)):
            for box_a in by_category[names[first_index]]:
                for box_b in by_category[names[second_index]]:
                    a, b = (box_a, box_b) if box_a['area_canvas'] >= box_b['area_canvas'] \
                        else (box_b, box_a)
                    if intersection_area(a['bbox_canvas'], b['bbox_canvas']) > 0:
                        continue
                    crop_a = square_from_box(a['bbox_canvas'])
                    crop_b = square_from_box(b['bbox_canvas'])
                    control = grid_control_crop([a['bbox_canvas'], b['bbox_canvas']])
                    if control is None:
                        continue
                    if retained_fraction(crop_a, b['bbox_canvas']) > 0.01:
                        continue
                    if retained_fraction(crop_b, a['bbox_canvas']) > 0.01:
                        continue
                    # no other instance of A's category may survive in I_A, and vice versa
                    others_a = [o for o in by_category[a['category']]
                                if o['annotation_id'] != a['annotation_id']]
                    others_b = [o for o in by_category[b['category']]
                                if o['annotation_id'] != b['annotation_id']]
                    if any(retained_fraction(crop_a, o['bbox_canvas']) > 0.01 for o in others_a):
                        continue
                    if any(retained_fraction(crop_b, o['bbox_canvas']) > 0.01 for o in others_b):
                        continue
                    scene = {
                        'image_id': image_id, 'file_name': file_name[image_id],
                        'original_size': list(original_size),
                        'official_view': record['official_view'],
                        'A': {**{k: v for k, v in a.items() if k != 'segmentation'},
                              'retained_in_AB': retained_fraction((0, 0, 224, 224), a['bbox_canvas']),
                              'retained_in_A_crop': retained_fraction(crop_a, a['bbox_canvas']),
                              'retained_in_B_crop': retained_fraction(crop_b, a['bbox_canvas']),
                              'retained_in_control': retained_fraction(control, a['bbox_canvas']),
                              'segmentation_point_retention_in_A_crop':
                                  segmentation_point_retention(a.get('segmentation'), crop_a,
                                                               scale, offset)},
                        'B': {**{k: v for k, v in b.items() if k != 'segmentation'},
                              'retained_in_AB': retained_fraction((0, 0, 224, 224), b['bbox_canvas']),
                              'retained_in_A_crop': retained_fraction(crop_a, b['bbox_canvas']),
                              'retained_in_B_crop': retained_fraction(crop_b, b['bbox_canvas']),
                              'retained_in_control': retained_fraction(control, b['bbox_canvas']),
                              'segmentation_point_retention_in_B_crop':
                                  segmentation_point_retention(b.get('segmentation'), crop_b,
                                                               scale, offset)},
                        'crops': {'I_AB': [0.0, 0.0, 224.0, 224.0],
                                  'I_A': [round(v, 2) for v in crop_a],
                                  'I_B': [round(v, 2) for v in crop_b],
                                  'I_control': [round(v, 2) for v in control]},
                        'isolation': ('bbox_and_segmentation_points' if (a.get('segmentation')
                                                                         and b.get('segmentation'))
                                      else 'bbox_controlled'),
                        'canvas_image': canvas,
                    }
                    return {'scene': scene, 'reason': None}
    return {'scene': None, 'reason': 'no spatially separable pair with a valid control crop'}


# --------------------------------------------------------------------------- model handling
def load_clip(spec, device):
    from model import longclip
    clip, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                               args=argparse.Namespace())
    payload = torch.load(spec['path'], map_location='cpu', weights_only=False)
    clip.load_state_dict(payload['model'], strict=True)
    clip = clip.float().to(device).eval()
    return clip, preprocess, payload


def encode_texts(clip, texts, tokenizer, device, batch=64):
    pooled, hidden = {}, {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            tokens = tokenizer(chunk, context_length=TOKEN_WIDTH).to(device)
            features, states = clip.encode_text(tokens, return_full=True)
            features = features.float().cpu()
            states = states.float().cpu()
            for index, text in enumerate(chunk):
                pooled[text] = features[index].clone()
                hidden[text] = states[index].clone()
    return pooled, hidden


def encode_views(clip, tensors, device, batch=32):
    out = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch):
            stacked = torch.stack(tensors[start:start + batch]).to(device)
            out.append(clip.encode_image(stacked).float().cpu())
    return torch.cat(out)


def view_tensor(image, preprocess):
    """Official view: Resize+CenterCrop canvas, then Resize/ToTensor/Normalize exactly once."""
    canvas = canvas_transform(preprocess)[0](image)
    return canvas, tensor_transform(preprocess)(canvas)


def crop_tensor(canvas, crop, preprocess):
    """A square integer crop of the canvas, resized to the model's 224x224 input exactly once."""
    x1 = int(round(crop[0]))
    y1 = int(round(crop[1]))
    side = int(round(min(crop[2] - crop[0], crop[3] - crop[1])))
    side = max(4, min(side, 224))
    x1 = max(0, min(x1, 224 - side))
    y1 = max(0, min(y1, 224 - side))
    region = canvas.crop((x1, y1, x1 + side, y1 + side))
    return tensor_transform(preprocess)(region), region


def text_hard_mask(clip, hidden, device=None):
    """The reference visual mask for one caption's hidden state.

    ``said_mask_from_hidden`` expects the full ``[batch, tokens, width]`` hidden state, so the cached
    single-caption tensor is given a batch dimension and moved to the mask network's own device.
    """
    from model.said_cls_cvssl import said_mask_from_hidden
    target = device or next(clip.mask_net.parameters()).device
    states = hidden.unsqueeze(0).to(target).float()
    mask, soft, _logits = said_mask_from_hidden(clip.mask_net, states, soft_mask=False)
    return mask.float().cpu()[0], soft.float().cpu()[0]


def template_for(category):
    article = 'an' if category[:1].lower() in 'aeiou' else 'a'
    return 'A photo of %s %s.' % (article, category)


def template_pair_for(category_a, category_b):
    article_a = 'an' if category_a[:1].lower() in 'aeiou' else 'a'
    article_b = 'an' if category_b[:1].lower() in 'aeiou' else 'a'
    return 'A photo of %s %s and %s %s.' % (article_a, category_a, article_b, category_b)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='shared_init,s0_500')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--summary-out', default=DEFAULT_SUMMARY)
    parser.add_argument('--instances',
                        default='/root/datasets/coco/annotations/instances_val2017.json')
    parser.add_argument('--max-scenes', type=int, default=MAX_CROP_SCENES)
    parser.add_argument('--skip-phase-b', action='store_true')
    args = parser.parse_args()

    started = time.time()
    os.chdir(REPO)
    device = torch.device(args.device)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, 'status.json')

    def status(**fields):
        payload = {'probe': 'clip512_functional_probe', 'read_only': True,
                   'new_optimizer_updates': 0, 'updated_at': time.time(),
                   'updated_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S'), 'phase_a': 'pending',
                   'phase_b': 'pending'}
        if os.path.exists(status_path):
            try:
                payload.update(json.load(open(status_path, 'r', encoding='utf-8')))
            except (ValueError, OSError):
                pass
        payload.update(fields)
        payload['updated_at'] = time.time()
        payload['updated_at_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        write_json_atomic(status_path, payload)

    status(phase_a='running')
    run_status = os.path.join(HS_RUN, 'run_status.json')
    run_status_before = sha256_of(run_status) if os.path.isfile(run_status) else None

    from model import longclip

    # ---- template audit: the paraphrase must not change colour / count / object / relation ----
    audit = []
    for index, base in enumerate(BASE_CAPTIONS):
        groups = SEMANTIC_LOCK[index]
        base_lower = base.lower()
        paraphrase_lower = PARAPHRASES[index].lower()
        change_lower = VISUAL_CHANGES[index].lower()
        missing = [group[0] for group in groups
                   if not any(word in paraphrase_lower for word in group)]
        introduced = [word for word in COLOUR_NUMBERS
                      if word in paraphrase_lower and word not in base_lower]
        differing_groups = [group[0] for group in groups
                            if not any(word in change_lower for word in group)]
        audit.append({'index': index, 'base': base, 'paraphrase': PARAPHRASES[index],
                      'visual_change': VISUAL_CHANGES[index],
                      'change_description': CHANGE_DESCRIPTIONS[index],
                      'locked_groups': [list(group) for group in groups],
                      'paraphrase_missing_locked_concepts': missing,
                      'paraphrase_introduced_colour_or_count': introduced,
                      'visual_change_differs_in_concepts': differing_groups,
                      'paraphrase_ok': not missing and not introduced})
    paraphrase_audit_ok = all(item['paraphrase_ok'] for item in audit)
    if not paraphrase_audit_ok:
        status(phase_a='failed', failure_reason='paraphrase audit failed; a paraphrase changes '
                                                'colour, count, object or relation')

    # ---- the fixed 128-image pool, in manifest order -------------------------------------------
    manifest = json.load(open(args.manifest, 'r', encoding='utf-8'))
    image_root = manifest['source']['image_root']
    pool = manifest['samples'][:POOL_IMAGES]
    if len(pool) < POOL_IMAGES:
        raise SystemExit('the manifest holds only %d samples' % len(pool))
    captions = [sample['caption'] for sample in pool]
    conditions = ['BASE'] + list(R_ORDER) + ['REPEAT']
    variant_text = {condition: [] for condition in conditions}
    for sample in pool:
        variant_text['BASE'].append(sample['caption'])
        variant_text['REPEAT'].append('%s %s' % (sample['caption'], sample['caption']))
        for key in R_ORDER:
            variant_text[key].append('%s %s' % (sample['caption'], R_SUFFIXES[key]))
    suffix_tokens = {key: token_sha(longclip.tokenize, R_SUFFIXES[key]) for key in R_ORDER}
    length_checks = []
    for index, sample in enumerate(pool):
        base_tokens = token_sha(longclip.tokenize, variant_text['BASE'][index])['ids']
        for key in R_ORDER:
            combined = token_sha(longclip.tokenize, variant_text[key][index])
            length_checks.append({
                'candidate_index': index, 'condition': key,
                'effective_length': combined['effective_length'],
                'suffix_entered': suffix_entered(base_tokens, suffix_tokens[key]['ids'],
                                                 combined['ids'])})
    truncation_report = {
        'max_effective_length': int(max(item['effective_length'] for item in length_checks)),
        'over_capacity': [item for item in length_checks if item['effective_length'] > TOKEN_WIDTH],
        'suffix_not_entered': [item for item in length_checks if not item['suffix_entered']],
        'eot_convention': 'effective length = argmax(token ids) + 1 (repository EOT rule), never a '
                          'count of token_id != 0',
    }

    hand_texts = []
    for item in audit:
        hand_texts += [item['base'], item['paraphrase'], item['visual_change']]
        hand_texts += ['%s %s' % (item['base'], R_SUFFIXES[key]) for key in R_ORDER]
    hand_texts += [R_SUFFIXES[key] for key in R_ORDER]

    models_report, unified_tables, coordinate_tables = {}, {}, {}
    for spec in MODEL_SPECS:
        if spec['key'] not in args.models.split(','):
            continue
        clip, preprocess, payload = load_clip(spec, device)
        digest_before = state_digest(clip.state_dict())

        # one encoding pass per distinct string and per image; full-precision cache saved locally
        distinct_texts = list(dict.fromkeys([text for condition in conditions
                                             for text in variant_text[condition]] + hand_texts))
        if len(distinct_texts) > MAX_TEXTS_PER_MODEL:
            raise SystemExit('text budget exceeded: %d > %d' % (len(distinct_texts),
                                                                MAX_TEXTS_PER_MODEL))
        pooled, hidden = encode_texts(clip, distinct_texts, longclip.tokenize, device)
        from PIL import Image
        canvases, view_tensors = [], []
        for sample in pool:
            with Image.open(os.path.join(image_root, sample['file_name'])) as image:
                canvas, tensor = view_tensor(image.convert('RGB'), preprocess)
            canvases.append(canvas)
            view_tensors.append(tensor)
        images = encode_views(clip, view_tensors, device)
        images_unit = unit_rows(images)
        text_matrix = {condition: unit_rows(torch.stack([pooled[text] for text in
                                                        variant_text[condition]]))
                       for condition in conditions}
        q = {condition: FIXED_SCALE * (images_unit @ text_matrix[condition].t())
             for condition in conditions}

        cache = {'checkpoint_sha256': spec['expected_sha256'],
                 'dtype': 'float32', 'context_length': TOKEN_WIDTH,
                 'preprocessing': 'openai-clip _transform(224) (Resize+CenterCrop+ToTensor+'
                                  'Normalize), autocast off',
                 'text_keys': distinct_texts,
                 'text_token_sha256': [token_sha(longclip.tokenize, text)['sha256']
                                       for text in distinct_texts],
                 'image_keys': [sample['file_name'] for sample in pool],
                 'image_sha256': [sha256_of(os.path.join(image_root, sample['file_name']))
                                  for sample in pool],
                 'text_features': torch.stack([pooled[text] for text in distinct_texts]),
                 'image_features': images}
        torch.save(cache, os.path.join(out_dir, 'cache_%s_fp32.pt' % spec['key']))

        # ---- per-condition retrieval, both directions ------------------------------------------
        per_condition = {}
        for condition in conditions:
            i2t = ranked_metrics(q[condition])
            t2i = ranked_metrics(q[condition].t().contiguous())
            per_condition[condition] = {'I2T': public(i2t), 'T2I': public(t2i)}
            per_condition[condition]['_i2t'] = i2t
            per_condition[condition]['_t2i'] = t2i
        delta_t = {condition: text_matrix[condition] - text_matrix['BASE']
                   for condition in conditions if condition != 'BASE'}
        delta_q = {condition: FIXED_SCALE * (images_unit @ delta_t[condition].t())
                   for condition in delta_t}
        identity = {condition: {'Q_R_minus_Q_max_abs_diff':
                                f((q[condition] - q['BASE'] - delta_q[condition]).abs().max())}
                    for condition in delta_t}

        # ---- closed forms ----
        offset_checks, covariance_checks = {}, {}
        for condition in delta_t:
            offset_checks[condition] = common_offset_invariance(
                delta_q[condition], delta_t[condition], images_unit, condition)
            covariance_checks[condition] = candidate_covariance(images_unit, delta_t[condition])

        # ---- per-coordinate margin contribution at a fixed negative candidate ------------------
        dimension_c = np.zeros(512)
        dimension_abs = np.zeros(512)
        contribution_records = {}
        for condition in R_ORDER:
            base_ranks = per_condition['BASE']['_i2t']['_rank']
            other_ranks = per_condition[condition]['_i2t']['_rank']
            base_hit = base_ranks <= 1
            other_hit = other_ranks <= 1
            fixed_rows = []
            for index in range(len(pool)):
                base_negative = int(per_condition['BASE']['_i2t']['_worst'][index])
                new_negative = int(per_condition[condition]['_i2t']['_worst'][index])
                for label, negative in (('base_strongest_negative', base_negative),
                                        ('new_strongest_negative', new_negative)):
                    if negative == index:
                        continue
                    contribution = FIXED_SCALE * images_unit[index] * (
                        delta_t[condition][index] - delta_t[condition][negative])
                    actual = float((q[condition][index, index] - q[condition][index, negative])
                                   - (q['BASE'][index, index] - q['BASE'][index, negative]))
                    fixed_rows.append({
                        'query_index': index, 'negative_index': negative, 'which': label,
                        'negative_switched': bool(base_negative != new_negative),
                        'contribution_sum': f(contribution.sum()),
                        'actual_margin_change': actual,
                        'identity_error': f(contribution.sum() - actual),
                        'rank_worsened': bool(other_ranks[index] > base_ranks[index]),
                        'rank_improved': bool(other_ranks[index] < base_ranks[index]),
                        'hit_lost': bool(base_hit[index] and not other_hit[index]),
                        'hit_gained': bool(not base_hit[index] and other_hit[index]),
                        'contribution': contribution.numpy()})
            # dimension aggregation, split by query outcome, signed and absolute kept separate
            for group, predicate in (('all', lambda row: True),
                                     ('hit_lost', lambda row: row['hit_lost']),
                                     ('hit_gained', lambda row: row['hit_gained']),
                                     ('rank_worsened', lambda row: row['rank_worsened']),
                                     ('rank_unchanged', lambda row: not row['rank_worsened']
                                      and not row['rank_improved'])):
                rows = [row for row in fixed_rows if predicate(row)]
                if not rows:
                    continue
                stack = np.stack([row['contribution'] for row in rows])
                dimension_c += stack.mean(axis=0)
                dimension_abs += np.abs(stack).mean(axis=0)
            contribution_records[condition] = {
                'rows': [{k: v for k, v in row.items() if k != 'contribution'}
                         for row in fixed_rows],
                'identity_error_max_abs': f(max(abs(row['identity_error']) for row in fixed_rows)),
                'negative_switched_queries': int(sum(1 for row in fixed_rows
                                                     if row['negative_switched']))}
        dimension_c /= len(R_ORDER)
        dimension_abs /= len(R_ORDER)

        # ---- raw change energy per coordinate and the coordinate/subspace comparison ----
        delta_stack = np.stack([delta_t[condition].numpy() for condition in R_ORDER])   # [4,128,512]
        raw_energy = (delta_stack ** 2).mean(axis=(0, 1))
        top_raw = np.argsort(-raw_energy)[:32]
        top_margin = np.argsort(-dimension_abs)[:32]
        half_labels = np.concatenate([[0 if index < len(pool) // 2 else 1
                                       for index in range(len(pool))] for _ in R_ORDER])
        coordinate_tables[spec['key']] = coordinate_vs_subspace(
            delta_stack.reshape(-1, 512), half_labels)

        # ---- charts for the page: energy vs margin contribution ----
        chart = {'raw_delta_energy': [float(x) for x in raw_energy],
                 'abs_margin_contribution': [float(x) for x in dimension_abs],
                 'signed_margin_contribution': [float(x) for x in dimension_c],
                 'top32_raw_energy': [int(i) for i in top_raw],
                 'top32_abs_margin': [int(i) for i in top_margin],
                 'top32_overlap_at_32': overlap_at_k(list(top_raw), list(top_margin), 32),
                 'top32_jaccard': jaccard(top_raw, top_margin)}
        worst = sorted(range(len(pool)),
                       key=lambda index: (per_condition['R1']['_i2t']['_rank'][index]
                                          - per_condition['BASE']['_i2t']['_rank'][index]),
                       reverse=True)[:5]
        best = sorted(range(len(pool)),
                      key=lambda index: (per_condition['R1']['_i2t']['_rank'][index]
                                         - per_condition['BASE']['_i2t']['_rank'][index]))[:5]
        chart['worst_queries'] = [{
            'query_index': index, 'image_id': pool[index]['image_id'],
            'annotation_id': pool[index]['annotation_id'],
            'base_caption': pool[index]['caption'],
            'suffixed_caption': variant_text['R1'][index],
            'base_rank': int(per_condition['BASE']['_i2t']['_rank'][index]),
            'r1_rank': int(per_condition['R1']['_i2t']['_rank'][index]),
            'base_strongest_negative_index': int(per_condition['BASE']['_i2t']['_worst'][index]),
            'r1_strongest_negative_index': int(per_condition['R1']['_i2t']['_worst'][index]),
            'base_m_max': per_condition['BASE']['_i2t']['per_query_m_max'][index],
            'r1_m_max': per_condition['R1']['_i2t']['per_query_m_max'][index],
            'base_m_lse': per_condition['BASE']['_i2t']['per_query_m_lse'][index],
            'r1_m_lse': per_condition['R1']['_i2t']['per_query_m_lse'][index],
            'top_abs_contribution_coordinates': [int(i) for i in
                                                 np.argsort(-np.abs(FIXED_SCALE
                                                                    * images_unit[index].numpy()
                                                                    * (delta_t['R1'][index].numpy()
                                                                       - delta_t['R1'][int(per_condition['R1']['_i2t']['_worst'][index])].numpy())))[:8]],
        } for index in worst]
        chart['best_queries'] = [{
            'query_index': index, 'image_id': pool[index]['image_id'],
            'annotation_id': pool[index]['annotation_id'],
            'base_rank': int(per_condition['BASE']['_i2t']['_rank'][index]),
            'r1_rank': int(per_condition['R1']['_i2t']['_rank'][index]),
        } for index in best]

        # ---- paired outcomes for the page (discrete, not a p-value wall) ----
        paired = {condition: {
            'I2T': paired_outcome(per_condition['BASE']['_i2t']['per_query_rank'],
                                  per_condition[condition]['_i2t']['per_query_rank'],
                                  [r <= 1 for r in per_condition['BASE']['_i2t']['per_query_rank']],
                                  [r <= 1 for r in per_condition[condition]['_i2t']['per_query_rank']]),
            'T2I': paired_outcome(per_condition['BASE']['_t2i']['per_query_rank'],
                                  per_condition[condition]['_t2i']['per_query_rank'],
                                  [r <= 1 for r in per_condition['BASE']['_t2i']['per_query_rank']],
                                  [r <= 1 for r in per_condition[condition]['_t2i']['per_query_rank']]),
        } for condition in delta_t}

        # ---- scale control: ranking must be preserved while CE/entropy may move ----
        scale_control = {}
        for factor in SCALE_CONTROL:
            scaled = ranked_metrics(q['BASE'] * factor)
            scale_control['x%g' % factor] = {
                'ce': scaled['ce'], 'entropy': scaled['entropy'],
                'ranking_identical_to_x1': bool((scaled['_rank']
                                                 == per_condition['BASE']['_i2t']['_rank']).all())}

        # ---- distribution metrics on the 128 pool (set statistics, no per-dimension labels) ----
        distribution = distribution_statistics(images_unit, text_matrix['BASE'])

        # ---- the visual mask of this model, measured on the pool captions ----------------------
        mask_keep_frequency = None
        if hasattr(clip, 'mask_net'):
            kept = torch.zeros(512)
            for index in range(len(pool)):
                mask, _soft = text_hard_mask(clip, hidden[variant_text['BASE'][index]])
                kept += (mask >= 0.5).float()
            mask_keep_frequency = (kept / len(pool)).tolist()

        # ---- the unified 512-row table ---------------------------------------------------------
        hand_visual_energy = np.zeros(512)
        hand_nonvis_energy = np.zeros(512)
        for item in audit:
            base_unit = unit_rows(pooled[item['base']].unsqueeze(0))[0]
            change = unit_rows(pooled[item['visual_change']].unsqueeze(0))[0]
            hand_visual_energy += ((change - base_unit) ** 2).numpy()
            for key in R_ORDER:
                suffixed = unit_rows(pooled['%s %s' % (item['base'], R_SUFFIXES[key])]
                                     .unsqueeze(0))[0]
                hand_nonvis_energy += ((suffixed - base_unit) ** 2).numpy()
        hand_visual_energy /= len(audit)
        hand_nonvis_energy /= len(audit) * len(R_ORDER)
        pooled_mean_image = images_unit.mean(dim=0)
        pooled_mean_text = text_matrix['BASE'].mean(dim=0)
        rows = []
        for dimension in range(512):
            rows.append({
                'dimension': dimension,
                'mean_image': f(pooled_mean_image[dimension]),
                'mean_text': f(pooled_mean_text[dimension]),
                'var_image': f(images_unit[:, dimension].var(unbiased=False)),
                'var_text': f(text_matrix['BASE'][:, dimension].var(unbiased=False)),
                'mean_gap_squared': f((pooled_mean_image[dimension]
                                       - pooled_mean_text[dimension]) ** 2),
                'nonvisual_text_delta_energy_pool': f(raw_energy[dimension]),
                'nonvisual_text_delta_energy_handwritten': f(hand_nonvis_energy[dimension]),
                'handwritten_visual_text_delta_energy': f(hand_visual_energy[dimension]),
                'margin_contribution_signed_mean': f(dimension_c[dimension]),
                'margin_contribution_abs_mean': f(dimension_abs[dimension]),
                'mask_keep_frequency': (f(mask_keep_frequency[dimension])
                                        if mask_keep_frequency else None),
                'crop_object_delta_energy': None,
                'crop_control_delta_energy': None,
                'object_pair_K': None,
            })
        unified_tables[spec['key']] = rows

        digest_after = state_digest(clip.state_dict())
        models_report[spec['key']] = {
            'label': spec['label'], 'path': spec['path'],
            'sha256': sha256_of(spec['path']),
            'sha256_matches_expected': sha256_of(spec['path']) == spec['expected_sha256'],
            'identity': {k: payload.get(k) for k in ('objective', 'phase', 'completed_steps')},
            'config_arm': (payload.get('config') or {}).get('arm'),
            'config_lambda_U': (payload.get('config') or {}).get('lambda_U'),
            'parameter_state_digest_before': digest_before,
            'parameter_state_digest_after': digest_after,
            'parameter_state_unchanged': digest_before == digest_after,
            'dtype': 'float32', 'autocast': 'disabled',
            'texts_encoded': len(distinct_texts), 'views_encoded': len(pool),
            'repeat_check': repeat_check(clip, distinct_texts[:4], longclip.tokenize, device),
            'per_condition': {condition: {'I2T': public(per_condition[condition]['_i2t']),
                                          'T2I': public(per_condition[condition]['_t2i'])}
                              for condition in conditions},
            'deltaQ_identity': identity,
            'common_offset': offset_checks,
            'candidate_covariance': covariance_checks,
            'margin_contribution': contribution_records,
            'coordinate_vs_subspace': coordinate_tables[spec['key']],
            'chart': chart,
            'paired_outcome': paired,
            'scale_control': scale_control,
            'distribution': distribution,
            'repeat_error_note': 'the repeat error is a numerical reference for one forward pass, not '
                                 'a claim that every floating-point path is error free',
        }
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status(phase_a='complete' if spec is MODEL_SPECS[-1] else 'running',
               phase_a_models_done=sorted(models_report))
    status(phase_a='complete', unified_table_dimensions=512,
           paraphrase_audit_ok=paraphrase_audit_ok)

    # ------------------------------------------------------------------ phase B
    phase_b = {'status': 'not_run', 'reason': None, 'scenes': [], 'rejected': []}
    if args.skip_phase_b:
        phase_b['reason'] = 'disabled by --skip-phase-b'
    elif not os.path.isfile(args.instances):
        phase_b['reason'] = 'instances annotation file missing: %s' % args.instances
    else:
        status(phase_b='running')
        try:
            preprocess = load_clip(MODEL_SPECS[0], 'cpu')[1]
            prefer = [sample['image_id'] for sample in pool]
            scenes, rejected, reason_counts, categories, file_name = select_crop_scenes(
                args.instances, prefer, preprocess, max_scenes=args.max_scenes)
            phase_b.update({'status': 'complete' if scenes else 'not_run',
                            'reason': None if scenes else 'no scene satisfied the fixed geometry '
                                                          'rules',
                            'rejected': rejected[:40],
                            'rejection_reason_counts': reason_counts,
                            'n_scenes': len(scenes),
                            'yield_note': 'the yield is what the fixed geometric rules allow; the '
                                          'scene count is never padded by relaxing them',
                            'coverage_note': 'annotation-constrained only; human visual '
                                             'verification NOT RUN (no image-inspection tool)'})
            sheets = os.path.join(out_dir, 'contact_sheets')
            os.makedirs(sheets, exist_ok=True)
            scene_records = []
            all_view_tensors = {key: [] for key in ('I_AB', 'I_A', 'I_B', 'I_control')}
            for index, scene in enumerate(scenes):
                canvas = scene.pop('canvas_image')
                views = {}
                for key, crop in scene['crops'].items():
                    if key == 'I_AB':
                        tensor = tensor_transform(preprocess)(canvas)
                        region = canvas
                    else:
                        tensor, region = crop_tensor(canvas, crop, preprocess)
                    views[key] = {'tensor': tensor, 'region': region}
                    all_view_tensors[key].append(tensor)
                scene['view_shas'] = {key: hashlib.sha256(
                    np.asarray(views[key]['tensor'].numpy(), dtype=np.float32).tobytes()).hexdigest()
                    for key in views}
                strip = Image.new('RGB', (224 * 4, 224))
                for position, key in enumerate(('I_AB', 'I_A', 'I_B', 'I_control')):
                    strip.paste(views[key]['region'].resize((224, 224)), (224 * position, 0))
                strip.save(os.path.join(sheets, 'scene_%02d.png' % index))
                scene['templates'] = {
                    'T_A': template_for(scene['A']['category']),
                    'T_B': template_for(scene['B']['category']),
                    'T_AB': template_pair_for(scene['A']['category'], scene['B']['category'])}
                scene['_views'] = views
                scene_records.append(scene)
            for spec in MODEL_SPECS:
                if spec['key'] not in args.models.split(','):
                    continue
                clip, preprocess_model, payload = load_clip(spec, device)
                digest_before = state_digest(clip.state_dict())
                view_features = {key: encode_views(clip, all_view_tensors[key], device)
                                 for key in all_view_tensors}
                template_texts = sorted({text for scene in scene_records
                                         for text in scene['templates'].values()})
                pooled_b, hidden_b = encode_texts(clip, template_texts, longclip.tokenize, device)
                text_unit_b = {key: unit_rows(pooled_b[key].unsqueeze(0))[0] for key in template_texts}
                entry = {'texts_encoded': len(template_texts),
                         'views_encoded': sum(len(v) for v in all_view_tensors.values()),
                         'per_scene': []}
                for index, scene in enumerate(scene_records):
                    view_unit = {key: unit_rows(view_features[key][index].unsqueeze(0))[0]
                                 for key in view_features}
                    scores = {view: {name: f(FIXED_SCALE * (view_unit[view] @ text_unit_b[text]))
                                     for name, text in scene['templates'].items()}
                              for view in view_unit}
                    dv = view_unit['I_A'] - view_unit['I_B']
                    dt = text_unit_b[scene['templates']['T_A']] - text_unit_b[scene['templates']['T_B']]
                    k_value = f(FIXED_SCALE * (dv @ dt))
                    k_identity = (scores['I_A']['T_A'] + scores['I_B']['T_B']
                                  - scores['I_A']['T_B'] - scores['I_B']['T_A'])
                    k_dim = (FIXED_SCALE * dv * dt).numpy()
                    record = {
                        'image_id': scene['image_id'], 'category_pair': scene['category_pair'],
                        'A': scene['A']['category'], 'B': scene['B']['category'],
                        'retention': {'A_in_A_crop': scene['A']['retained_in_A_crop'],
                                      'B_in_A_crop': scene['A']['retained_in_B_crop'],
                                      'A_in_B_crop': scene['B']['retained_in_A_crop'],
                                      'B_in_B_crop': scene['B']['retained_in_B_crop'],
                                      'A_in_control': scene['A']['retained_in_control'],
                                      'B_in_control': scene['B']['retained_in_control']},
                        'isolation': scene['isolation'],
                        'scores': scores,
                        'four_score_identity': {'K': k_value, 'K_from_four_scores': k_identity,
                                                'max_abs_diff': f(abs(k_value - k_identity))},
                        'cos_dv_dt': f(F.cosine_similarity(dv.unsqueeze(0), dt.unsqueeze(0))),
                        'A_preferred_on_A_view': bool(scores['I_A']['T_A'] > scores['I_B']['T_A']),
                        'B_preferred_on_B_view': bool(scores['I_B']['T_B'] > scores['I_A']['T_B']),
                        'A_margin': f(scores['I_A']['T_A'] - scores['I_B']['T_A']),
                        'B_margin': f(scores['I_B']['T_B'] - scores['I_A']['T_B']),
                        'K_dim_positive_energy': f(float(k_dim[k_dim > 0].sum())),
                        'K_dim_negative_energy': f(float(k_dim[k_dim < 0].sum())),
                        'K_top_coordinates': [int(i) for i in np.argsort(-np.abs(k_dim))[:8]],
                    }
                    if hasattr(clip, 'mask_net'):
                        masked = {}
                        for name, text in scene['templates'].items():
                            mask, _soft = text_hard_mask(clip, hidden_b[text])
                            masked[name] = {'mask_kept': int((mask >= 0.5).sum().item()),
                                            'mask_norm': f(mask.norm()),
                                            'views': {}}
                            for view in view_unit:
                                z = F.normalize((view_features[view][index] * mask).unsqueeze(0),
                                                dim=-1, eps=NORM_EPS)[0]
                                masked[name]['views'][view] = {
                                    'score': f(FIXED_SCALE * (z @ text_unit_b[text])),
                                    'norm': f(z.norm())}
                        record['smartclip_masked'] = masked
                        record['smartclip_selection'] = {
                            'same_mask_used_for_all_views': True,
                            'T_A_delete_unsaid_B_distance':
                                f((masked['T_A']['views']['I_AB']['score']
                                   - masked['T_A']['views']['I_A']['score'])),
                            'T_A_delete_said_A_distance':
                                f((masked['T_A']['views']['I_AB']['score']
                                   - masked['T_A']['views']['I_B']['score'])),
                            'T_A_control_distance':
                                f((masked['T_A']['views']['I_AB']['score']
                                   - masked['T_A']['views']['I_control']['score'])),
                            'T_B_delete_unsaid_A_distance':
                                f((masked['T_B']['views']['I_AB']['score']
                                   - masked['T_B']['views']['I_B']['score'])),
                            'T_B_delete_said_B_distance':
                                f((masked['T_B']['views']['I_AB']['score']
                                   - masked['T_B']['views']['I_A']['score'])),
                            'native_T_A_delete_unsaid_B': f(scores['I_AB']['T_A']
                                                            - scores['I_A']['T_A']),
                            'native_T_A_delete_said_A': f(scores['I_AB']['T_A']
                                                          - scores['I_B']['T_A']),
                            'masked_prefers_A_view_for_T_A': bool(
                                masked['T_A']['views']['I_A']['score']
                                > masked['T_A']['views']['I_B']['score']),
                            'native_prefers_A_view_for_T_A': bool(scores['I_A']['T_A']
                                                                  > scores['I_B']['T_A']),
                            'note': 'a caption whose target was deleted must not keep its response; '
                                    'the four-score comparison stays valid because every view uses '
                                    'the same text and its single mask, which is why no single '
                                    'dv.dot(dt) identity is claimed for the masked readout'}
                    entry['per_scene'].append(record)
                entry['parameter_state_unchanged'] = (
                    digest_before == state_digest(clip.state_dict()))
                entry['sha256'] = sha256_of(spec['path'])
                # write the image-side columns back into the unified dimension table
                delete_energy = np.zeros(512)
                control_energy = np.zeros(512)
                k_energy = np.zeros(512)
                for index, scene in enumerate(scene_records):
                    view_unit = {key: unit_rows(view_features[key][index].unsqueeze(0))[0]
                                 for key in view_features}
                    delete_energy += (((view_unit['I_AB'] - view_unit['I_A']) ** 2
                                       + (view_unit['I_AB'] - view_unit['I_B']) ** 2) / 2).numpy()
                    control_energy += ((view_unit['I_AB'] - view_unit['I_control']) ** 2).numpy()
                    dt_scene = (text_unit_b[scene['templates']['T_A']]
                                - text_unit_b[scene['templates']['T_B']])
                    k_energy += (FIXED_SCALE * (view_unit['I_A'] - view_unit['I_B'])
                                 * dt_scene).numpy()
                count = max(1, len(scene_records))
                for dimension in range(512):
                    unified_tables[spec['key']][dimension]['crop_object_delta_energy'] = f(
                        delete_energy[dimension] / count)
                    unified_tables[spec['key']][dimension]['crop_control_delta_energy'] = f(
                        control_energy[dimension] / count)
                    unified_tables[spec['key']][dimension]['object_pair_K'] = f(
                        k_energy[dimension] / count)
                phase_b.setdefault('models', {})[spec['key']] = entry
                del clip
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for scene in scene_records:
                scene.pop('_views', None)
            phase_b['scenes'] = [{k: v for k, v in scene.items() if not k.startswith('_')}
                                 for scene in scene_records]
            status(phase_b=phase_b['status'], phase_b_scenes=len(scene_records))
        except Exception as error:                      # phase A must still land
            import traceback
            phase_b.update({'status': 'failed',
                            'reason': '%s: %s' % (type(error).__name__, error),
                            'traceback': traceback.format_exc()[-1500:]})
            status(phase_b='failed', phase_b_reason=phase_b['reason'])

    run_status_after = sha256_of(run_status) if os.path.isfile(run_status) else None
    report = {
        'probe': 'clip512_functional_probe', 'read_only': True, 'new_optimizer_updates': 0,
        'scope': 'which 512-d representation changes actually move retrieval, do object deletions '
                 'form a compatible cross-modal direction, and does the SmartCLIP coordinate mask '
                 'select the relevant content',
        'models_used': [spec['key'] for spec in MODEL_SPECS if spec['key'] in args.models.split(',')],
        'pool': {'manifest': os.path.abspath(args.manifest),
                 'manifest_sha256': sha256_of(args.manifest), 'images': len(pool),
                 'description': 'the first 128 samples of the frozen 256-sample manifest, in order',
                 'image_ids': [sample['image_id'] for sample in pool],
                 'annotation_ids': [sample['annotation_id'] for sample in pool],
                 'conditions': conditions},
        'template_audit': audit, 'paraphrase_audit_ok': paraphrase_audit_ok,
        'truncation': truncation_report,
        'lengths': {'pool_max_effective_length': truncation_report['max_effective_length'],
                    'suffix_tokens': {key: {'effective_length': suffix_tokens[key]['effective_length'],
                                            'sha256': suffix_tokens[key]['sha256']}
                                      for key in R_ORDER}},
        'models': models_report,
        'phase_b': phase_b,
        'unified_dimension_table': unified_tables,
        'sources': {
            'pool_statistics': '128 COCO val2017 images x 128 captions (frozen manifest order)',
            'handwritten_statistics': '12 handwritten base captions x {paraphrase, one-visual-change, '
                                      '4 appended suffixes}',
            'crop_statistics': 'up to %d annotation-constrained two-object scenes' % args.max_scenes,
            'note': 'columns come from different sample counts and different populations; they are '
                    'never presented as one exact decomposition',
        },
        'not_run': [
            'human visual verification of the contact sheets (no image-inspection tool in this '
            'session): the crops are annotation-constrained, not pixel-exact isolations',
            'browser visual acceptance of the dashboard area (no browser tool)',
            'any training, projection head, dimension deletion or token-mask method search',
            'any causal claim from a single forward intervention',
        ],
        'timing': {'wall_seconds': f(time.time() - started)},
        'run_status': {'path': run_status, 'sha256_before': run_status_before,
                       'sha256_after': run_status_after,
                       'unchanged': run_status_before == run_status_after},
    }
    probe_path = os.path.join(out_dir, 'clip512_functional_probe.json')
    write_json_atomic(probe_path, report)
    summary = build_summary(report)
    summary['probe_path'] = probe_path
    write_json_atomic(os.path.join(REPO, args.summary_out), summary)
    write_dimension_csv(os.path.join(out_dir, 'clip512_dimension_table.csv'), unified_tables)
    status(phase_a='complete', phase_b=phase_b['status'], finished=True,
           summary_written=os.path.join(REPO, args.summary_out))
    print('WROTE %s' % probe_path)
    print('WROTE %s' % os.path.join(REPO, args.summary_out))
    print('PHASE_A %s | PHASE_B %s' % ('complete', phase_b['status']))
    print('HEADLINE ' + json.dumps(summary['headline'], sort_keys=True))


def repeat_check(clip, texts, tokenizer, device, repeats=3):
    runs = [encode_texts(clip, texts, tokenizer, device)[0] for _ in range(repeats)]
    error = max(f((run[text] - runs[0][text]).abs().max()) for run in runs[1:] for text in texts)
    cosine = min(f(F.cosine_similarity(run[text], runs[0][text], dim=0))
                 for run in runs[1:] for text in texts)
    return {'forward_error_max_abs': error, 'cosine_min': cosine, 'repeats': repeats}


def build_summary(report):
    """The compact, caption-free summary that is safe to commit."""
    keep = ('probe', 'read_only', 'new_optimizer_updates', 'scope', 'models_used', 'template_audit',
            'paraphrase_audit_ok', 'truncation', 'lengths', 'not_run', 'timing', 'run_status',
            'sources')
    summary = {key: report[key] for key in keep}
    summary['pool'] = {k: v for k, v in report['pool'].items()
                       if k in ('manifest', 'manifest_sha256', 'images', 'description', 'image_ids',
                                'annotation_ids', 'conditions')}
    models, headline = {}, {}
    for key, entry in report['models'].items():
        models[key] = {
            'label': entry['label'], 'path': entry['path'], 'sha256': entry['sha256'],
            'sha256_matches_expected': entry['sha256_matches_expected'],
            'identity': entry['identity'], 'config_arm': entry['config_arm'],
            'config_lambda_U': entry['config_lambda_U'],
            'parameter_state_unchanged': entry['parameter_state_unchanged'],
            'dtype': entry['dtype'], 'autocast': entry['autocast'],
            'texts_encoded': entry['texts_encoded'], 'views_encoded': entry['views_encoded'],
            'repeat_check': entry['repeat_check'],
            'per_condition': {condition: {direction: {k: v for k, v in metrics.items()
                                                      if not k.startswith('per_query')}
                                          for direction, metrics in body.items()}
                              for condition, body in entry['per_condition'].items()},
            'deltaQ_identity': entry['deltaQ_identity'],
            'common_offset': entry['common_offset'],
            'candidate_covariance': {condition: {'max_abs_diff': body['max_abs_diff'],
                                                 'quadratic_quantiles': body['quadratic_quantiles']}
                                     for condition, body in entry['candidate_covariance'].items()},
            'margin_contribution': {
                condition: {'identity_error_max_abs': body['identity_error_max_abs'],
                            'negative_switched_queries': body['negative_switched_queries']}
                for condition, body in entry['margin_contribution'].items()},
            'coordinate_vs_subspace': entry['coordinate_vs_subspace'],
            'chart': {k: v for k, v in entry['chart'].items()
                      if k in ('top32_overlap_at_32', 'top32_jaccard', 'top32_raw_energy',
                               'top32_abs_margin')},
            'paired_outcome': entry['paired_outcome'],
            'scale_control': entry['scale_control'],
            'distribution': entry['distribution'],
        }
        i2t_base = entry['per_condition']['BASE']['I2T']
        i2t_r1 = entry['per_condition']['R1']['I2T']
        t2i_base = entry['per_condition']['BASE']['T2I']
        t2i_r1 = entry['per_condition']['R1']['T2I']
        headline[key] = {
            'I2T_BASE_R@1': i2t_base['R@1'], 'I2T_R1_R@1': i2t_r1['R@1'],
            'I2T_R1_delta_R@1': i2t_r1['R@1'] - i2t_base['R@1'],
            'T2I_BASE_R@1': t2i_base['R@1'], 'T2I_R1_R@1': t2i_r1['R@1'],
            'T2I_R1_delta_R@1': t2i_r1['R@1'] - t2i_base['R@1'],
            'deltaQ_identity_max_abs_diff': max(body['Q_R_minus_Q_max_abs_diff']
                                                for body in entry['deltaQ_identity'].values()),
            'margin_identity_max_abs_error': max(body['identity_error_max_abs']
                                                 for body in entry['margin_contribution'].values()),
            'common_offset_ranking_identical': all(body['per_query_rank_identical']
                                                   for body in entry['common_offset'].values()),
            'top32_overlap_at_32': entry['chart']['top32_overlap_at_32'],
            'top32_jaccard': entry['chart']['top32_jaccard'],
            'centroid_gap': entry['distribution']['centroid_gap'],
            'paired_alignment': entry['distribution']['paired_alignment'],
            'image_uniformity': entry['distribution']['image_uniformity'],
            'text_uniformity': entry['distribution']['text_uniformity'],
            'repeat_error_max_abs': entry['repeat_check']['forward_error_max_abs'],
        }
    summary['models'] = models
    phase_b = report['phase_b']
    summary['phase_b'] = {
        'status': phase_b['status'], 'reason': phase_b.get('reason'),
        'n_scenes': phase_b.get('n_scenes'), 'coverage_note': phase_b.get('coverage_note'),
        'rejected_sample': (phase_b.get('rejected') or [])[:20],
        'scenes': [{k: v for k, v in scene.items() if k != 'canvas_image'}
                   for scene in (phase_b.get('scenes') or [])],
        'models': {key: {'texts_encoded': body['texts_encoded'],
                         'views_encoded': body['views_encoded'],
                         'parameter_state_unchanged': body['parameter_state_unchanged'],
                         'per_scene': body['per_scene']}
                   for key, body in (phase_b.get('models') or {}).items()},
    }
    if phase_b.get('models'):
        for key, body in phase_b['models'].items():
            records = body['per_scene']
            if not records:
                continue
            headline.setdefault(key, {}).update({
                'K_identity_max_abs_diff': max(r['four_score_identity']['max_abs_diff']
                                               for r in records),
                'A_preferred_on_A_view_fraction': float(np.mean(
                    [r['A_preferred_on_A_view'] for r in records])),
                'B_preferred_on_B_view_fraction': float(np.mean(
                    [r['B_preferred_on_B_view'] for r in records])),
                'mean_cos_dv_dt': float(np.mean([r['cos_dv_dt'] for r in records])),
            })
    summary['headline'] = headline
    return summary


def write_dimension_csv(path, unified_tables):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = ['model', 'dimension', 'mean_image', 'mean_text', 'var_image', 'var_text',
               'mean_gap_squared', 'nonvisual_text_delta_energy_pool',
               'nonvisual_text_delta_energy_handwritten', 'handwritten_visual_text_delta_energy',
               'margin_contribution_signed_mean', 'margin_contribution_abs_mean',
               'mask_keep_frequency', 'crop_object_delta_energy', 'crop_control_delta_energy',
               'object_pair_K']
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, rows in sorted(unified_tables.items()):
            for row in rows:
                entry = dict(row)
                entry['model'] = model
                writer.writerow(entry)
    os.replace(temporary, path)


if __name__ == '__main__':
    main()


