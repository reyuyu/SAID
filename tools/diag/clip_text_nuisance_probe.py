"""Read-only probe: does a non-visual appended sentence enter the 512-d text representation, is the
effect concentrated or distributed, does it damage image-text matching, and does the
S0-TriMask-HS text gate reduce it without losing visual content sensitivity?

What this is *not*: it creates no optimizer, calls no ``backward``, updates no parameter, resumes no
training, changes no loss/threshold/sparsity coefficient, re-runs no canonical COCO/Urban evaluation
and starts no follow-up training. Checkpoints, ``run_status.json`` and the canonical evaluation files
are read-only; each checkpoint's sha256 is re-checked after the run and the parameter-state digest
must be unchanged.

Method (cached: one encoding per distinct string and one per image, per model):

* twelve hand-written visual base captions ``C``, each in six forms -- ``C``, ``C`` plus one of four
  non-visual author-stance/meta sentences ``R1..R4``, a paraphrase keeping the scene meaning, and a
  change of exactly one visual element. The four ``R`` and the empty string are encoded separately
  for description only. All of it was written before any score was looked at.
* a fixed 128-image COCO val2017 pool taken in manifest order from the existing 256-sample
  geometry-probe manifest, with six caption variants per image (``C``, ``C+R1..C+R4``, ``C``
  repeated). Each ``R`` forms its own complete candidate pool, the images and the correct pairing
  are identical in every condition, and no variant is mixed into another pool.
* NATIVE readout ``Q = 100 * cos(v_i, t_j)`` for every model, plus a TEXT_MASKED readout
  ``z = normalize(t_raw * mT)`` for the HS model only, where ``mT`` is regenerated from the hidden
  state of the actual input -- never reused from the clean caption.

No coordinate is ever named a sentiment dimension or an unused dimension; raw coordinates are shown
by index only, and every delta is the effect of the whole text intervention, not a pure sentiment
vector.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SHARED_INIT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt')
HS_500 = os.path.join(REPO, 'runs_salu', 'said_s0_trimask_hs_v02', 'step500',
                      'trimask_S0_TriMask_HS_step000500.pt')
S0_500 = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/ddpfix_step500_S0_smartclip/'
          'cvssl_S0_smartclip_step000500.pt')
HS_RUN = os.path.join(REPO, 'runs_salu', 'said_s0_trimask_hs_v02', 'step500')
MANIFEST = os.path.join(HS_RUN, 'diagnostics', 'manifest.json')
DEFAULT_OUT_DIR = os.path.join(HS_RUN, 'diagnostics', 'text_nuisance')
DEFAULT_SUMMARY = 'docs/said_trimask_hs/text_nuisance_summary.json'

FIXED_SCALE = 100.0
NORM_EPS = 1e-6
K_VALUES = (1, 5, 10)
TOP_K = (1, 4, 8, 16, 32, 64)
TOKEN_WIDTH = 248
MAX_TEXTS_PER_MODEL = 900
MAX_IMAGES = 128
EMPTY_TEXT = ''

# ---------------------------------------------------------------- the hand-written text sets
# Written by hand before any model output was seen, then frozen. Nothing is picked, edited or
# dropped afterwards because of a score.
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
PARAPHRASES = [
    'There is a dog seated on the lawn.',
    'A cat that is black rests on a mat that is red.',
    'Next to a tree stands a parked red car.',
    'A pair of people ride bicycles.',
    'A yellow balloon is being held by a child.',
    'On a fence made of wood stands a bird.',
    'A boat that is blue floats on a lake.',
    'On a table sits a bowl filled with fruit.',
    'A white horse is running in a field.',
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
R_SUFFIXES = {
    'R1': 'I personally like this picture.',
    'R2': 'I personally dislike this picture.',
    'R3': 'Thank you for sharing this picture.',
    'R4': 'This is just my personal opinion.',
}
R_ORDER = ('R1', 'R2', 'R3', 'R4')
LENGTH_MATCHED_PAIRS = (('R1', 'R2'),)

MODEL_SPECS = [
    {'key': 'shared_init', 'label': 'A · 项目共同初始化 CLIP（248-token）', 'path': SHARED_INIT,
     'kind': 'clip_only',
     'expected_sha256': 'c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8'},
    {'key': 'hs_500', 'label': 'B · S0-TriMask-HS @500（完整 checkpoint，含文本门）',
     'path': HS_500, 'kind': 'trimask_hs',
     'expected_sha256': '46f6d9c7e31ce3298d08e87a32b4228ca3bf5e216a674079deb5b0fa9fc0c4ef'},
    {'key': 's0_500', 'label': 'C · S0_smartclip @500（视觉 mask，无文本门）', 'path': S0_500,
     'kind': 'clip_only',
     'expected_sha256': '758dcdd2a9112d54ccb8784e211d329297209c5d7be26cd7a89e68185867c743'},
]


# ---------------------------------------------------------------- helpers
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


def token_report(tokenizer, text):
    """EOT position, effective length and ids, using the repository's own EOT convention.

    ``encode_text`` picks the EOT embedding with ``text.argmax(dim=-1)`` (the EOT id is the largest),
    so the effective length is ``argmax + 1``. Counting ``token_id != 0`` would be wrong here because
    a legal content token can be id 0.
    """
    tokens = tokenizer([text], context_length=TOKEN_WIDTH)
    ids = [int(v) for v in tokens[0].tolist()]
    eot_index = int(np.argmax(np.asarray(ids)))
    return {'token_ids': ids, 'eot_index': eot_index, 'effective_length': eot_index + 1,
            'width': len(ids),
            'sha256': hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()}


def suffix_check(base_report, suffix_report, combined_report, tolerance=1):
    """Whether the appended sentence really reached the encoder.

    Primary test is the length test done by the caller (a sequence wider than the model's 248
    positions is truncated and the sample is dropped *before* any scoring). This helper adds a content
    test on the tokens *between* the special tokens: appending text replaces the base's EOT, so the
    suffix content starts exactly at the base's EOT index, and the combined sequence must carry all of
    the suffix's content tokens up to ``tolerance`` (one BPE merge across the join, e.g. ". I").
    """
    suffix_content = suffix_report['token_ids'][1:suffix_report['eot_index']]
    if not suffix_content:
        return {'shared_content_tokens': 0, 'suffix_content_tokens': 0, 'entered': True}
    tail = combined_report['token_ids'][base_report['eot_index']:
                                       combined_report['eot_index']]
    shared = sum(1 for token in suffix_content if token in tail)
    return {'shared_content_tokens': int(shared), 'suffix_content_tokens': len(suffix_content),
            'entered': bool(shared >= len(suffix_content) - tolerance)}


def sign_test(worse, better):
    """Exact two-sided sign test on the discordant paired rank changes.

    With 128 queries a few queries moving is not a result, so the count of queries whose rank got
    worse versus better is turned into an exact binomial p-value instead of being reported as a
    percentage. Ties (unchanged queries) carry no information and are excluded, as the test requires.
    """
    discordant = worse + better
    if discordant == 0:
        return {'worse': worse, 'better': better, 'discordant': 0, 'p_value_two_sided': 1.0}
    smaller = min(worse, better)
    tail = sum(math.comb(discordant, i) for i in range(0, smaller + 1)) / float(2 ** discordant)
    return {'worse': worse, 'better': better, 'discordant': discordant,
            'p_value_two_sided': float(min(1.0, 2.0 * tail))}


def ranking_metrics(scores, k_values=K_VALUES):
    """R@K / MRR / CE / margins with a fixed, recorded tie rule.

    ``rank = 1 + #{j != pos : score[j] > score[pos]}``: a negative that merely ties the positive does
    not beat it, and ties are counted and reported instead of being hidden.
    """
    scores = scores.float()
    diagonal = torch.diagonal(scores)
    off = scores.clone()
    off.fill_diagonal_(float('-inf'))
    s_max_negative = off.max(dim=1).values
    lse_negatives = torch.logsumexp(off, dim=1)
    rank = 1 + (off > diagonal.unsqueeze(1)).sum(dim=1)
    m_lse = diagonal - lse_negatives
    ce = F.softplus(-m_lse)
    ties = (off == diagonal.unsqueeze(1))
    out = {
        'n_queries': int(scores.shape[0]),
        'R@1': f((rank <= 1).float().mean()), 'R@5': f((rank <= 5).float().mean()),
        'R@10': f((rank <= 10).float().mean()), 'mrr': f((1.0 / rank.float()).mean()),
        'ce': f(ce.mean()), 'rank_mean': f(rank.float().mean()),
        'm_max_mean': f((diagonal - s_max_negative).mean()), 'm_lse_mean': f(m_lse.mean()),
        's_pos_mean': f(diagonal.mean()), 's_max_negative_mean': f(s_max_negative.mean()),
        'tie_pairs_total': int(ties.sum().item()),
        'tie_queries': int((ties.sum(dim=1) > 0).sum().item()),
        'ce_identity_max_abs_diff': f((ce - (torch.logsumexp(scores, dim=1)
                                             - diagonal)).abs().max()),
        'per_query_rank': [int(x) for x in rank.tolist()],
    }
    out['_rank'] = rank
    out['_diagonal'] = diagonal
    return out


def public(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith('_')}


def coordinate_report(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    energy = (deltas ** 2).mean(axis=0)
    total = float(energy.sum())
    order = np.argsort(-energy)
    return {'n_pairs': int(deltas.shape[0]), 'dimension': int(deltas.shape[1]),
            'energy_total': total,
            'topk_share': {('top%d_share' % k):
                           (float(energy[order[:k]].sum() / total) if total > 0 else None)
                           for k in TOP_K},
            'top_coordinates_by_index': [int(i) for i in order[:32]],
            'per_coordinate_energy': [float(x) for x in energy]}


def split_half_report(deltas, half_labels, k_values=TOP_K):
    """Fix the coordinate ordering on the first half, then evaluate it on the second half."""
    deltas = np.asarray(deltas, dtype=np.float64)
    labels = np.asarray(half_labels)
    first, second = deltas[labels == 0], deltas[labels == 1]
    energy_first = (first ** 2).mean(axis=0)
    energy_second = (second ** 2).mean(axis=0)
    order = np.argsort(-energy_first)
    total_second = float(energy_second.sum())
    fixed = ({('top%d_share' % k): float(energy_second[order[:k]].sum() / total_second)
              for k in k_values} if total_second > 0 else {})
    best_second = np.argsort(-energy_second)
    overlap = {('top%d_jaccard' % k):
               float(len(set(order[:k].tolist()) & set(best_second[:k].tolist())) / float(k))
               for k in (1, 4) + k_values}
    correlation = (float(np.corrcoef(energy_first, energy_second)[0, 1])
                   if energy_first.std() > 0 and energy_second.std() > 0 else None)
    return {'n_pairs_first': int(first.shape[0]), 'n_pairs_second': int(second.shape[0]),
            'coordinate_order_from_first_half': [int(i) for i in order[:32]],
            'second_half_topk_share_using_first_half_order': fixed,
            'topk_jaccard_between_halves': overlap,
            'energy_profile_correlation': correlation,
            'note': 'the second half only evaluates the ordering fixed on the first half; it never '
                    'selects its own best coordinates'}


def svd_report(deltas, components=(1, 4, 8, 16)):
    deltas = np.asarray(deltas, dtype=np.float64)
    singular = np.linalg.svd(deltas, full_matrices=False, compute_uv=False)
    squared = singular ** 2
    total = float(squared.sum())
    mean_delta = deltas.mean(axis=0)
    mean_energy = float((mean_delta ** 2).sum())
    mean_pair_energy = float((deltas ** 2).sum(axis=1).mean())
    return {'squared_singular_values': [float(x) for x in squared[:32]],
            'component_share': ({('pc%d' % c): float(squared[:c].sum() / total)
                                 for c in components} if total > 0 else {}),
            'mean_delta_energy_share':
                (mean_energy / mean_pair_energy if mean_pair_energy > 0 else None),
            'note': 'uncentered SVD: the leading directions include the common offset of all deltas, '
                    'so they are not a pure semantic subspace'}


# ---------------------------------------------------------------- model handling
def clip_of(model):
    return model.clip if hasattr(model, 'clip') else model


def build_model(spec, device):
    from model import longclip
    from model.said_trimask import TriMaskTrainModule

    payload = torch.load(spec['path'], map_location='cpu', weights_only=False)
    actual_sha = sha256_of(spec['path'])
    clip, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                               args=argparse.Namespace())
    info = {'key': spec['key'], 'label': spec['label'], 'path': spec['path'], 'kind': spec['kind'],
            'sha256': actual_sha, 'sha256_matches_expected': actual_sha == spec['expected_sha256']}
    if spec['kind'] == 'trimask_hs':
        module = TriMaskTrainModule(clip, rank=0, text_gate_mode='hard_st', text_mask_seed=0,
                                    grad_checkpoint_views=False)
        module.clip.load_state_dict(payload['model'])
        module.text_mask_net.load_state_dict(payload['text_mask_net'])
        module = module.float().to(device).eval()
        info.update({
            'identity': {k: payload.get(k) for k in
                         ('objective', 'arm', 'completed_steps', 'text_gate_mode',
                          'lambda_sparse_t', 'loss_profile')},
            'has_text_gate': True, 'text_mask_net_tensors': len(payload['text_mask_net']),
            'loaded_with': 'the full checkpoint: payload["model"] into the CLIP student AND '
                           'payload["text_mask_net"] into the text branch (never a bare student '
                           'with a randomly initialised text gate)'})
        return module, info, preprocess
    clip.load_state_dict(payload['model'], strict=True)
    clip = clip.float().to(device).eval()
    info.update({'identity': {k: payload.get(k) for k in
                              ('objective', 'arm', 'completed_steps', 'phase')},
                 'has_text_gate': False,
                 'loaded_with': 'payload["model"] (CLIP + reference visual mask_net) loaded '
                                'strictly; this checkpoint has no text branch'})
    return clip, info, preprocess


def encode_texts(model, texts, tokenizer, device, batch_size=64, autocast=False):
    """Raw pooled features and full hidden states under FP32 with autocast off, keyed by string."""
    features, hiddens, eot = {}, {}, {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            tokens = tokenizer(chunk, context_length=TOKEN_WIDTH).to(device)
            if autocast:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pooled, hidden = clip_of(model).encode_text(tokens, return_full=True)
            else:
                pooled, hidden = clip_of(model).encode_text(tokens, return_full=True)
            pooled = pooled.float().cpu()
            hidden = hidden.float().cpu()
            positions = tokens.argmax(dim=-1).cpu()
            for index, text in enumerate(chunk):
                features[text] = pooled[index].clone()
                hiddens[text] = hidden[index].clone()
                eot[text] = int(positions[index])
    return features, hiddens, eot


def encode_images(model, paths, preprocess, device, batch_size=32):
    from PIL import Image
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert('RGB')))
            batch = torch.stack(tensors).to(device)
            features.append(clip_of(model).encode_image(batch).float().cpu())
    return torch.cat(features)


def text_masked_readout(model, texts, hiddens, eot, native, device):
    """``z(C) = normalize(t_raw(C) * mT(C))`` with mT regenerated from that input's own hidden state.

    The pooled feature is rebuilt from the cached hidden state with the same EOT convention the
    encoder uses, and the rebuilt value is compared against the native pooled feature as an internal
    consistency check. The mask is never taken from the clean caption.
    """
    out, masks, raw_error = {}, {}, 0.0
    with torch.inference_mode():
        for text in texts:
            hidden = hiddens[text].to(device).unsqueeze(0)
            gate = model.text_mask_net.forward_with_details(hidden)
            mask = gate['mT'].float().cpu()[0]
            index = torch.tensor([eot[text]], device=device)
            raw = (hidden[0, index[0]] @ clip_of(model).text_projection).float().cpu()
            raw_error = max(raw_error, f((raw - native[text]).abs().max()))
            out[text] = F.normalize((raw * mask).unsqueeze(0), dim=-1, eps=NORM_EPS)[0]
            masks[text] = mask
    return out, masks, raw_error


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='shared_init,hs_500,s0_500')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-images', type=int, default=MAX_IMAGES)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--summary-out', default=DEFAULT_SUMMARY)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--repeat-check', type=int, default=3)
    args = parser.parse_args()

    started = time.time()
    os.chdir(REPO)
    device = torch.device(args.device)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_status = os.path.join(HS_RUN, 'run_status.json')
    run_status_before = sha256_of(run_status) if os.path.isfile(run_status) else None

    from model import longclip

    # ---- fixed text sets, built before any model output is seen --------------------------------
    hand_pairs = []
    for index, base in enumerate(BASE_CAPTIONS):
        hand_pairs.append({
            'index': index, 'base': base, 'paraphrase': PARAPHRASES[index],
            'visual_change': VISUAL_CHANGES[index],
            'change_description': CHANGE_DESCRIPTIONS[index],
            'with_suffix': {key: '%s %s' % (base, R_SUFFIXES[key]) for key in R_ORDER}})
    hand_texts = []
    for entry in hand_pairs:
        hand_texts += [entry['base'], entry['paraphrase'], entry['visual_change']]
        hand_texts += [entry['with_suffix'][key] for key in R_ORDER]
    hand_texts += [R_SUFFIXES[key] for key in R_ORDER] + [EMPTY_TEXT]
    hand_texts = list(dict.fromkeys(hand_texts))

    suffix_reports = {key: token_report(longclip.tokenize, R_SUFFIXES[key]) for key in R_ORDER}

    # ---- the fixed 128-image pool, in manifest order -------------------------------------------
    manifest = json.load(open(args.manifest, 'r', encoding='utf-8'))
    image_root = manifest['source']['image_root']
    pool, skipped = [], []
    for sample in manifest['samples']:
        if len(pool) >= args.n_images:
            break
        caption = sample['caption']
        variants = {'BASE': caption, 'REPEAT': '%s %s' % (caption, caption)}
        variants.update({key: '%s %s' % (caption, R_SUFFIXES[key]) for key in R_ORDER})
        reports = {key: token_report(longclip.tokenize, text) for key, text in variants.items()}
        too_long = [key for key, item in reports.items() if item['effective_length'] > TOKEN_WIDTH]
        checks = {key: suffix_check(reports['BASE'], suffix_reports[key], reports[key])
                  for key in R_ORDER}
        if too_long or not all(item['entered'] for item in checks.values()):
            skipped.append({'image_id': sample['image_id'], 'annotation_id': sample['annotation_id'],
                            'reason': 'length or suffix-content check failed', 'too_long': too_long,
                            'suffix_check': checks})
            continue
        pool.append({'candidate_index': len(pool), 'image_id': sample['image_id'],
                     'annotation_id': sample['annotation_id'], 'file_name': sample['file_name'],
                     'caption': caption, 'variants': variants, 'tokens': reports,
                     'suffix_check': checks})
    if len(pool) < args.n_images:
        raise SystemExit('only %d qualifying samples for %d requested' % (len(pool), args.n_images))

    pool_texts = []
    for entry in pool:
        pool_texts += list(entry['variants'].values())
    pool_texts = list(dict.fromkeys(pool_texts))
    distinct_texts = list(dict.fromkeys(hand_texts + pool_texts))
    if len(distinct_texts) > MAX_TEXTS_PER_MODEL:
        raise SystemExit('budget exceeded: %d distinct texts > %d'
                         % (len(distinct_texts), MAX_TEXTS_PER_MODEL))

    # ---- length report -------------------------------------------------------------------------
    length_groups = {}
    for entry in pool:
        for key, report in entry['tokens'].items():
            length_groups.setdefault('real_pool:' + key, []).append(report['effective_length'])
    for entry in hand_pairs:
        for name in ('base', 'paraphrase', 'visual_change'):
            length_groups.setdefault('hand:' + name, []).append(
                token_report(longclip.tokenize, entry[name])['effective_length'])
        for key in R_ORDER:
            length_groups.setdefault('hand:' + key, []).append(
                token_report(longclip.tokenize, entry['with_suffix'][key])['effective_length'])
    for key in R_ORDER:
        length_groups.setdefault('suffix_alone:' + key, []).append(
            suffix_reports[key]['effective_length'])
    length_groups.setdefault('empty_string', []).append(
        token_report(longclip.tokenize, EMPTY_TEXT)['effective_length'])
    length_report = {group: {'n': len(values), 'min': int(np.min(values)),
                             'mean': float(np.mean(values)), 'max': int(np.max(values))}
                     for group, values in sorted(length_groups.items())}

    image_paths = [os.path.join(image_root, entry['file_name']) for entry in pool]
    models_report, coordinates_all, ui_payload = {}, {}, {}
    for spec in MODEL_SPECS:
        if spec['key'] not in args.models.split(','):
            continue
        model, info, preprocess = build_model(spec, device)
        info['parameter_state_digest_before'] = state_digest(clip_of(model).state_dict())

        native, hiddens, eot = encode_texts(model, distinct_texts, longclip.tokenize, device)
        images = encode_images(model, image_paths, preprocess, device)

        # numerical references: the same input re-encoded, and the training path's bf16 autocast
        repeat_texts = BASE_CAPTIONS[:4]
        repeat_runs = []
        for _ in range(max(2, args.repeat_check)):
            repeat_runs.append(encode_texts(model, repeat_texts, longclip.tokenize, device)[0])
        repeat_error = max(f((run[text] - repeat_runs[0][text]).abs().max())
                           for run in repeat_runs[1:] for text in repeat_texts)
        repeat_cosine = min(f(F.cosine_similarity(run[text], repeat_runs[0][text], dim=0))
                            for run in repeat_runs[1:] for text in repeat_texts)
        bf16_features = encode_texts(model, repeat_texts, longclip.tokenize, device,
                                     autocast=True)[0]
        bf16_error = max(f((bf16_features[text].float() - native[text].float()).abs().max())
                         for text in repeat_texts)
        bf16_cosine = min(f(F.cosine_similarity(bf16_features[text].float(),
                                                native[text].float(), dim=0))
                          for text in repeat_texts)

        readouts = {'NATIVE': native}
        if info['has_text_gate']:
            masked, masks, raw_error = text_masked_readout(model, distinct_texts, hiddens, eot,
                                                           native, device)
            readouts['TEXT_MASKED'] = masked
            mask_rows = {key: [] for key in R_ORDER}
            for entry in pool:
                base_mask = masks[entry['variants']['BASE']]
                for key in R_ORDER:
                    variant_mask = masks[entry['variants'][key]]
                    mask_rows[key].append({
                        'hamming': int((base_mask != variant_mask).sum().item()),
                        'kept_base': int((base_mask >= 0.5).sum().item()),
                        'kept_variant': int((variant_mask >= 0.5).sum().item())})
            info['mask_change_vs_clean'] = {
                key: {'n': len(rows),
                      'hamming_mean': float(np.mean([r['hamming'] for r in rows])),
                      'kept_base_mean': float(np.mean([r['kept_base'] for r in rows])),
                      'kept_variant_mean': float(np.mean([r['kept_variant'] for r in rows]))}
                for key, rows in sorted(mask_rows.items())}
            info['masked_readout_pooling_reconstruction_max_abs_diff'] = raw_error

        def vector(readout, text):
            return readouts[readout][text].float()

        def unit(readout, text):
            return F.normalize(vector(readout, text).unsqueeze(0), dim=-1, eps=NORM_EPS)[0]

        # ---- hand-written pairs -----------------------------------------------------------------
        hand_report, hand_groups = {}, {}
        for readout in sorted(readouts):
            per_pair = {}
            for entry in hand_pairs:
                base_unit = unit(readout, entry['base'])
                row = {'base_raw_norm': f(vector(readout, entry['base']).norm())}
                for name, text in (('base', entry['base']), ('paraphrase', entry['paraphrase']),
                                   ('visual_change', entry['visual_change'])):
                    other = unit(readout, text)
                    row[name] = {'cos': f(F.cosine_similarity(base_unit, other, dim=0)),
                                 'l2': f((base_unit - other).norm()),
                                 'raw_norm': f(vector(readout, text).norm())}
                for key in R_ORDER:
                    other = unit(readout, entry['with_suffix'][key])
                    row[key] = {'cos': f(F.cosine_similarity(base_unit, other, dim=0)),
                                'l2': f((base_unit - other).norm()),
                                'raw_norm': f(vector(readout, entry['with_suffix'][key]).norm())}
                per_pair[str(entry['index'])] = row
            hand_report[readout] = per_pair
            # vectors for the read-only page (run-local file, never committed): the twelve
            # hand-written pairs, each as its base and six variants of normalised 512-d vectors
            ui_payload.setdefault(spec['key'], {})[readout] = {
                'hand_vectors': {str(entry['index']): {
                    name: [round(x, 3) for x in unit(readout, text).tolist()]
                    for name, text in [('base', entry['base']),
                                       ('paraphrase', entry['paraphrase']),
                                       ('visual_change', entry['visual_change'])]
                    + [(key, entry['with_suffix'][key]) for key in R_ORDER]}
                    for entry in hand_pairs},
                'hand_labels': {'base': BASE_CAPTIONS, 'paraphrase': PARAPHRASES,
                                'visual_change': VISUAL_CHANGES,
                                'change_description': CHANGE_DESCRIPTIONS,
                                'R_suffixes': R_SUFFIXES, 'R_order': list(R_ORDER)},
                'pool_delta_energy': None}

            groups = {}
            for name, keys in (('appended_suffix', R_ORDER), ('paraphrase', ('paraphrase',)),
                               ('visual_change', ('visual_change',))):
                cosines, l2 = [], []
                for entry in hand_pairs:
                    base_unit = unit(readout, entry['base'])
                    for key in keys:
                        text = entry['with_suffix'][key] if key in R_ORDER else entry[key]
                        other = unit(readout, text)
                        cosines.append(f(F.cosine_similarity(base_unit, other, dim=0)))
                        l2.append(f((base_unit - other).norm()))
                groups[name] = {'n': len(cosines), 'cos': quantiles(cosines), 'l2': quantiles(l2)}
            groups['length_matched_pairs'] = {}
            for left, right in LENGTH_MATCHED_PAIRS:
                pairs = [{'pair': str(entry['index']),
                          'cos': f(F.cosine_similarity(unit(readout, entry['with_suffix'][left]),
                                                       unit(readout, entry['with_suffix'][right]),
                                                       dim=0))} for entry in hand_pairs]
                groups['length_matched_pairs']['%s_vs_%s' % (left, right)] = {
                    'length_delta': suffix_reports[right]['effective_length']
                                    - suffix_reports[left]['effective_length'],
                    'cos': quantiles([row['cos'] for row in pairs]), 'per_pair': pairs}
            hand_groups[readout] = groups

        # masked vs native: is the reduction specific to R or global over all three groups?
        if 'TEXT_MASKED' in readouts:
            specificity = {}
            for name in ('appended_suffix', 'paraphrase', 'visual_change'):
                native_median = hand_groups['NATIVE'][name]['l2']['q0.5']
                masked_median = hand_groups['TEXT_MASKED'][name]['l2']['q0.5']
                specificity[name] = {
                    'native_l2_median': native_median, 'masked_l2_median': masked_median,
                    'ratio_masked_over_native': (masked_median / native_median
                                                 if native_median else None)}
            info['masked_vs_native_sensitivity'] = specificity

        # ---- coordinate concentration on the real pool ------------------------------------------
        per_readout = {}
        for readout in sorted(readouts):
            deltas, half_labels, suffix_labels = [], [], []
            for entry in pool:
                base_unit = unit(readout, entry['variants']['BASE'])
                for key in R_ORDER:
                    deltas.append((unit(readout, entry['variants'][key]) - base_unit).numpy())
                    half_labels.append(0 if entry['candidate_index'] < len(pool) // 2 else 1)
                    suffix_labels.append(key)
            deltas = np.asarray(deltas)
            suffix_labels = np.asarray(suffix_labels)
            all_pairs = coordinate_report(deltas)
            ui_payload[spec['key']][readout]['pool_delta_energy'] = all_pairs['per_coordinate_energy']
            visual_deltas = np.asarray([(unit(readout, entry['visual_change'])
                                         - unit(readout, entry['base'])).numpy()
                                        for entry in hand_pairs])
            visual_report = coordinate_report(visual_deltas)
            top_r = set(all_pairs['top_coordinates_by_index'][:32])
            top_v = set(visual_report['top_coordinates_by_index'][:32])
            visual_report['overlap_with_R_sensitive_top32'] = {
                'jaccard': len(top_r & top_v) / float(len(top_r | top_v)),
                'shared': sorted(top_r & top_v),
                'note': 'descriptive only: no coordinate is deleted and none is named semantically'}
            per_readout[readout] = {
                'real_pool_all_pairs': all_pairs,
                'real_pool_split_half': split_half_report(deltas, half_labels),
                'real_pool_uncentered_svd': svd_report(deltas),
                'per_suffix': {key: coordinate_report(deltas[suffix_labels == key])
                               for key in R_ORDER},
                'hand_visual_change': visual_report}
        coordinates_all[spec['key']] = per_readout

        # ---- retrieval on the fixed pool --------------------------------------------------------
        image_unit = F.normalize(images, dim=-1, eps=NORM_EPS)
        pool_report = {}
        for readout in sorted(readouts):
            conditions = {}
            for condition in ('BASE',) + R_ORDER + ('REPEAT',):
                text_unit = torch.stack([unit(readout, entry['variants'][condition])
                                         for entry in pool])
                metrics = ranking_metrics(FIXED_SCALE * (image_unit @ text_unit.t()))
                conditions[condition] = metrics
            base_metrics = conditions['BASE']
            comparisons = {}
            for condition in R_ORDER + ('REPEAT',):
                metrics = conditions[condition]
                worsened = int((metrics['_rank'] > base_metrics['_rank']).sum())
                improved = int((metrics['_rank'] < base_metrics['_rank']).sum())
                comparisons[condition] = {
                    'delta_R@1': metrics['R@1'] - base_metrics['R@1'],
                    'delta_R@5': metrics['R@5'] - base_metrics['R@5'],
                    'delta_R@10': metrics['R@10'] - base_metrics['R@10'],
                    'delta_mrr': metrics['mrr'] - base_metrics['mrr'],
                    'delta_ce': metrics['ce'] - base_metrics['ce'],
                    'delta_s_pos_mean': metrics['s_pos_mean'] - base_metrics['s_pos_mean'],
                    'delta_s_max_negative_mean':
                        metrics['s_max_negative_mean'] - base_metrics['s_max_negative_mean'],
                    'rank_unchanged_queries': int((metrics['_rank'] == base_metrics['_rank']).sum()),
                    'rank_worsened': worsened, 'rank_improved': improved,
                    'ranks_worsened_by_more_than_one':
                        int((metrics['_rank'] > base_metrics['_rank'] + 1).sum()),
                    'paired_sign_test': sign_test(worsened, improved),
                    'per_query_rank': metrics['per_query_rank'],
                    'base_per_query_rank': base_metrics['per_query_rank']}
            r1 = np.asarray(conditions['R1']['per_query_rank'])
            r2 = np.asarray(conditions['R2']['per_query_rank'])
            pool_report[readout] = {
                'conditions': {k: public(v) for k, v in conditions.items()},
                'vs_base': comparisons,
                'like_vs_dislike': {
                    'delta_R@1': conditions['R2']['R@1'] - conditions['R1']['R@1'],
                    'delta_ce': conditions['R2']['ce'] - conditions['R1']['ce'],
                    'rank_identical_queries': int((r1 == r2).sum()),
                    'rank_differs_by_more_than_one': int((np.abs(r1 - r2) > 1).sum()),
                    'top1_agreement_queries': int(((r1 == 1) & (r2 == 1)).sum()),
                    'top1_r1_only': int(((r1 == 1) & (r2 != 1)).sum()),
                    'top1_r2_only': int(((r2 == 1) & (r1 != 1)).sum())}}

        info['parameter_state_digest_after'] = state_digest(clip_of(model).state_dict())
        info['parameter_state_unchanged'] = (info['parameter_state_digest_before']
                                             == info['parameter_state_digest_after'])
        info['sha256_after'] = sha256_of(spec['path'])
        info['sha256_unchanged'] = info['sha256'] == info['sha256_after']
        info['numerics'] = {
            'repeat_check_forward_error_max_abs': repeat_error,
            'repeat_check_cosine_min': repeat_cosine,
            'bf16_autocast_vs_fp32_error_max_abs': bf16_error,
            'bf16_autocast_vs_fp32_cosine_min': bf16_cosine,
            'note': 'this probe encodes in FP32 with autocast off; the training path runs the CLIP '
                    'image/text paths under bf16 autocast, and the bf16-vs-fp32 numbers are the scale '
                    'of that difference. A coordinate delta smaller than either reference cannot be '
                    'read as a semantic change'}
        models_report[spec['key']] = {
            'info': info, 'hand_pairs': hand_report, 'hand_groups': hand_groups,
            'pool': pool_report,
            'text_counts': {'distinct_texts': len(distinct_texts), 'hand_texts': len(hand_texts),
                            'pool_texts': len(pool_texts)}}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_status_after = sha256_of(run_status) if os.path.isfile(run_status) else None
    not_run = [
        'no image-level verification of the negatives: negatives are defined by annotation index and '
        'are not claimed to be semantically wrong captions',
        'no browser visual acceptance of the dashboard section (no browser tool in this session)',
        'no claim that any coordinate or direction is an "unused" or "sentiment" dimension',
        'no follow-up training was started; nothing here is a training-causal claim',
        'the hand-written visual-change group has no paired images, so it is used only for text-side '
        'sensitivity and is never claimed to be a wrong caption of a real image',
    ]
    if 'hs_500' not in models_report:
        not_run.append('the HS text-mask readout: the HS model was not selected in this run')
    report = {
        'probe': 'clip_text_nuisance_probe',
        'read_only': True,
        'new_optimizer_updates': 0,
        'scope': 'non-visual appended sentence vs the 512-d text representation and the 128x128 '
                 'image-text candidate ranking; a diagnostic pool, not the canonical evaluation',
        'budget': {'models': args.models.split(','), 'max_texts_per_model': MAX_TEXTS_PER_MODEL,
                   'distinct_texts_per_model': len(distinct_texts), 'images': len(pool),
                   'new_optimizer_updates': 0},
        'numerics_policy': {
            'encoding': 'eval() + torch.inference_mode(), FP32 parameters and FP32 forward, autocast '
                        'disabled',
            'training_path_difference': 'the trainers run the CLIP image/text paths under bf16 '
                                        'autocast; the per-model bf16-vs-fp32 numbers give the scale',
            'repeat_error': 'the same input is encoded repeatedly and the maximum absolute difference '
                            'is reported per model as the numerical reference'},
        'hand_written_texts': {
            'base_captions': BASE_CAPTIONS, 'paraphrases': PARAPHRASES,
            'visual_changes': VISUAL_CHANGES, 'change_descriptions': CHANGE_DESCRIPTIONS,
            'R_suffixes': R_SUFFIXES, 'R_order': list(R_ORDER), 'empty_string': EMPTY_TEXT,
            'terminology': 'the R sentences are non-visual author-stance/meta text: they add no '
                           'described scene fact. They are not claimed to be meaningless for every '
                           'visual task, and a large single coordinate is never named a sentiment '
                           'dimension.',
            'fixed_before_scoring': True},
        'pool': {'manifest': os.path.abspath(args.manifest),
                 'manifest_sha256': sha256_of(args.manifest), 'requested': args.n_images,
                 'accepted': len(pool), 'skipped': skipped,
                 'image_ids': [entry['image_id'] for entry in pool],
                 'annotation_ids': [entry['annotation_id'] for entry in pool],
                 'suffix_check_summary': {key: {'entered_all': bool(all(
                     entry['suffix_check'][key]['entered'] for entry in pool)),
                     'min_shared_content_tokens': int(min(
                         entry['suffix_check'][key]['shared_content_tokens'] for entry in pool))}
                     for key in R_ORDER},
                 'note': 'candidate order is the manifest order; every condition uses the same images '
                         'and the same correct pairing, and no variant is mixed into another pool'},
        'lengths': {'groups': length_report, 'suffix_alone': suffix_reports,
                    'eot_convention': 'effective length = argmax(token ids) + 1 (the repository EOT '
                                      'rule); counting token_id != 0 would be wrong because a legal '
                                      'content token can be id 0',
                    'truncation_note': 'every input is checked before scoring: an input wider than '
                                       'the 248 positions is dropped, and the appended suffix must '
                                       'also pass a content-token test, so a cut-off suffix can '
                                       'never be scored as "the model ignored the suffix"'},
        'models': models_report,
        'coordinates': coordinates_all,
        'not_run': not_run,
        'timing': {'wall_seconds': f(time.time() - started)},
        'run_status': {'path': run_status, 'sha256_before': run_status_before,
                       'sha256_after': run_status_after,
                       'unchanged': run_status_before == run_status_after},
    }
    probe_path = os.path.join(out_dir, 'clip_text_nuisance_probe.json')
    write_json_atomic(probe_path, report)
    summary = build_summary(report)
    summary['probe_path'] = probe_path
    write_json_atomic(os.path.join(REPO, args.summary_out), summary)
    write_hand_pairs_csv(os.path.join(out_dir, 'clip_text_nuisance_hand_pairs.csv'), models_report)
    ui_path = os.path.join(out_dir, 'clip_text_nuisance_ui.json')
    write_json_atomic(ui_path, {
        'probe': 'clip_text_nuisance_probe', 'read_only': True,
        'new_optimizer_updates': 0,
        'note': 'run-local file for the read-only page: the twelve hand-written pairs as normalised '
                '512-d vectors (4 decimals) plus the per-coordinate delta energy of the real pool. '
                'Never committed; the page only reads it, a refresh never triggers a forward pass.',
        'model_labels': {spec['key']: spec['label'] for spec in MODEL_SPECS},
        'coordinates': ui_payload,
        'pool': {'image_ids': [entry['image_id'] for entry in pool],
                 'annotation_ids': [entry['annotation_id'] for entry in pool]},
        'headline': summary['headline'],
        'pool_metrics': {key: {readout: {name: {k: v for k, v in metrics.items()
                                                if k != 'per_query_rank'}
                                         for name, metrics in body['conditions'].items()}
                               for readout, body in entry['pool'].items()}
                         for key, entry in models_report.items()},
        'pool_vs_base': {key: {readout: body['vs_base'] for readout, body in entry['pool'].items()}
                         for key, entry in models_report.items()},
        'hand_groups': {key: entry['hand_groups'] for key, entry in models_report.items()},
        'coordinate_summary': {key: {readout: {
            'topk_share': body['real_pool_all_pairs']['topk_share'],
            'top_coordinates_by_index': body['real_pool_all_pairs']['top_coordinates_by_index'],
            'split_half': body['real_pool_split_half'],
            'svd': body['real_pool_uncentered_svd'],
            'hand_visual_change': {k: v for k, v in body['hand_visual_change'].items()
                                   if k != 'per_coordinate_energy'}}
            for readout, body in per_readout.items()}
            for key, per_readout in coordinates_all.items()},
        'not_run': not_run})
    print('WROTE %s' % probe_path)
    print('WROTE %s' % ui_path)
    print('WROTE %s' % os.path.join(REPO, args.summary_out))
    print('NUISANCE_SUMMARY ' + json.dumps(summary['headline'], sort_keys=True))


def build_summary(report):
    keep = ('probe', 'read_only', 'new_optimizer_updates', 'scope', 'budget', 'numerics_policy',
            'hand_written_texts', 'lengths', 'not_run', 'timing', 'run_status')
    summary = {key: report[key] for key in keep}
    summary['pool'] = {k: v for k, v in report['pool'].items()
                       if k in ('manifest', 'manifest_sha256', 'requested', 'accepted', 'skipped',
                                'image_ids', 'annotation_ids', 'suffix_check_summary', 'note')}
    # the committed summary keeps the length statistics but not the 248-wide id arrays, which are
    # reproducible from the tokenizer and would be its largest single block
    summary['lengths'] = dict(report['lengths'])
    summary['lengths']['suffix_alone'] = {
        key: {k: v for k, v in body.items() if k != 'token_ids'}
        for key, body in report['lengths']['suffix_alone'].items()}
    models, headline = {}, {}
    for key, entry in report['models'].items():
        keep_info = {k: v for k, v in entry['info'].items() if k != 'loaded_with'}
        models[key] = {'info': keep_info, 'hand_pairs': entry['hand_pairs'],
                       'hand_groups': entry['hand_groups'], 'text_counts': entry['text_counts'],
                       'pool': {}}
        for readout, body in entry['pool'].items():
            models[key]['pool'][readout] = {
                'conditions': {name: {k: v for k, v in metrics.items()
                                      if k != 'per_query_rank'}
                               for name, metrics in body['conditions'].items()},
                'vs_base': {name: {k: v for k, v in item.items()
                                   if k not in ('per_query_rank', 'base_per_query_rank')}
                            for name, item in body['vs_base'].items()},
                'like_vs_dislike': body['like_vs_dislike']}
        native_base = ((entry['pool'].get('NATIVE') or {}).get('conditions') or {}).get('BASE') or {}
        native_r1 = ((entry['pool'].get('NATIVE') or {}).get('conditions') or {}).get('R1') or {}
        native_vs = ((entry['pool'].get('NATIVE') or {}).get('vs_base') or {}).get('R1') or {}
        native_groups = (entry['hand_groups'] or {}).get('NATIVE') or {}
        masked_base = ((entry['pool'].get('TEXT_MASKED') or {}).get('conditions') or {}).get('BASE') or {}
        masked_vs = ((entry['pool'].get('TEXT_MASKED') or {}).get('vs_base') or {}).get('R1') or {}
        headline[key] = {
            'native_base_R@1': native_base.get('R@1'), 'native_R1_R@1': native_r1.get('R@1'),
            'native_R1_delta_R@1': native_vs.get('delta_R@1'),
            'native_R1_delta_ce': native_vs.get('delta_ce'),
            'masked_base_R@1': masked_base.get('R@1'),
            'masked_R1_delta_R@1': masked_vs.get('delta_R@1'),
            'hand_suffix_l2_median': (native_groups.get('appended_suffix') or {}).get('l2', {}).get('q0.5'),
            'hand_paraphrase_l2_median': (native_groups.get('paraphrase') or {}).get('l2', {}).get('q0.5'),
            'hand_visual_change_l2_median': (native_groups.get('visual_change') or {}).get('l2', {}).get('q0.5'),
            'repeat_error_max_abs': (entry['info'].get('numerics') or {}).get(
                'repeat_check_forward_error_max_abs'),
            'bf16_vs_fp32_error_max_abs': (entry['info'].get('numerics') or {}).get(
                'bf16_autocast_vs_fp32_error_max_abs')}
    summary['models'] = models
    coordinates = {}
    for model_key, per_readout in report['coordinates'].items():
        coordinates[model_key] = {}
        for readout, body in per_readout.items():
            coordinates[model_key][readout] = {
                'real_pool_all_pairs': {k: v for k, v in body['real_pool_all_pairs'].items()
                                        if k != 'per_coordinate_energy'},
                'real_pool_split_half': body['real_pool_split_half'],
                'real_pool_uncentered_svd': body['real_pool_uncentered_svd'],
                'per_suffix': {name: {k: v for k, v in item.items()
                                      if k != 'per_coordinate_energy'}
                               for name, item in body['per_suffix'].items()},
                'hand_visual_change': {k: v for k, v in body['hand_visual_change'].items()
                                       if k != 'per_coordinate_energy'}}
    summary['coordinates'] = coordinates
    summary['headline'] = headline
    return summary


def write_hand_pairs_csv(path, models_report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = ['model', 'readout', 'pair_index', 'variant', 'cos', 'l2', 'raw_norm']
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, entry in sorted(models_report.items()):
            for readout, pairs in sorted(entry['hand_pairs'].items()):
                for pair_index, row in sorted(pairs.items(), key=lambda kv: int(kv[0])):
                    for variant in ('base', 'paraphrase', 'visual_change') + R_ORDER:
                        body = row.get(variant) or {}
                        if not isinstance(body, dict):
                            continue
                        writer.writerow({'model': model, 'readout': readout,
                                         'pair_index': pair_index, 'variant': variant,
                                         'cos': body.get('cos'), 'l2': body.get('l2'),
                                         'raw_norm': body.get('raw_norm')})
    os.replace(temporary, path)


if __name__ == '__main__':
    main()
