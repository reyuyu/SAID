"""USR -- Unsaid Semantic Retrieval, protocol ``sharegpt4v1k-usr-v1`` (fixed evaluation).

Task: given an image and its *observed* prefix ``C_said``, retrieve the image's true
withheld suffix sentence ``U_i`` out of the fixed candidate pool ``{U_1..U_Q}``.

The cohort comes from the frozen Phase 2.7B manifest (``sharegpt4v1k-fixed-captions-v1``):
``C_full = captions['full_dense']`` and ``C_said = captions['fixed_sparse']``, so no prefix
is ever re-sampled. ``K`` is recovered exactly (``C_said`` must equal the first ``K``
sentences of ``C_full`` under the training split rule); samples with ``K == N`` have no
suffix and are excluded. The target is ``J = sample_unsaid_index(json_index, N, K, 26)``
with the same stateless function the trainer uses, ``U_i = S_J``.

Four scorers are evaluated on the same queries and the same candidate pool:

* ``global``      -- ``scale * cos(normalize(g_i), normalize(E_T(U_j)))`` (prefix unused)
* ``anti_said``   -- legacy ``softmax(-s_said / tau_unsaid)`` pooling, candidate-independent
* ``raw``         -- hidden-semantic attention with ``beta = 0``
* ``debiased``    -- the main method, ``beta = 1`` (shared ``score_unsaid_candidates`` API)

Scoring is chunked (image batches x candidate chunks); only the four ``[Q, Q]`` CPU fp32
score matrices are kept, never a ``[Q, Q, P]`` GPU tensor. Duplicate suffix sentences are
**not** handled in this protocol (candidate ``j`` is always the target of query ``j``).
"""
import argparse
import json
import os
import sys
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
from eval.validation_protocol import CANONICAL_SIMILARITY_CHUNK, rng_guard  # noqa: E402
from model import longclip, unsaid_core  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import sample_unsaid_index  # noqa: E402

PROTOCOL_NAME = 'sharegpt4v1k-usr-v1'
PROTOCOL_VERSION = 1
USR_SUFFIX_SEED = 26
SOURCE_PROTOCOL = 'sharegpt4v1k-fixed-captions-v1'
SENTENCE_SPLIT_RULE = 'caption.replace("\\n", " ").split(". ")'
TAU_UNSAID = unsaid_core.DEFAULT_TAU_UNSAID
GATE_FLOOR = unsaid_core.DEFAULT_GATE_FLOOR
GATE_TEMPERATURE = unsaid_core.DEFAULT_GATE_TEMPERATURE
SCORERS = ('global', 'anti_said', 'raw', 'debiased')


def split_sentences(caption: str) -> List[str]:
    """Training sentence convention (identical to ``sharegpt4v.sample_unsaid_index``)."""
    return caption.replace('\n', ' ').split('. ')


def recover_prefix_k(caption_full: str, caption_said: str) -> Tuple[int, int]:
    """Exact ``(K, N)`` recovery; ambiguous or impossible matches are protocol errors."""
    sentences = split_sentences(caption_full)
    candidates = [k for k in range(1, len(sentences) + 1)
                  if '. '.join(sentences[:k]) == caption_said]
    if len(candidates) != 1:
        raise ValueError('protocol error: C_said does not match a unique prefix of C_full '
                         '(%d matches)' % len(candidates))
    return candidates[0], len(sentences)


def build_manifest(source_manifest: dict, source_manifest_path: str,
                   suffix_seed: int = USR_SUFFIX_SEED) -> Dict:
    """Fixed USR cohort: one primary target per valid query, candidates = the targets."""
    queries, excluded = [], 0
    for sample in source_manifest['samples']:
        captions = sample['captions']
        k, n = recover_prefix_k(captions['full_dense'], captions['fixed_sparse'])
        if k >= n:
            excluded += 1
            continue
        json_index = int(sample['json_index'])
        j = sample_unsaid_index(json_index, n, k, suffix_seed)
        if j is None or not (k < j <= n):
            raise ValueError('protocol error: invalid suffix index J=%r for json_index=%d'
                             % (j, json_index))
        sentences = split_sentences(captions['full_dense'])
        queries.append({
            'query_id': len(queries),
            'json_index': json_index,
            'image_path': sample['image_path'],
            'N': n, 'K': k, 'J': int(j),
            'caption_full': captions['full_dense'],
            'caption_said': captions['fixed_sparse'],
            'target_text': sentences[j - 1],
        })
    return {
        'protocol_name': PROTOCOL_NAME,
        'protocol_version': PROTOCOL_VERSION,
        'source_protocol': source_manifest.get('protocol', SOURCE_PROTOCOL),
        'source_validation_manifest_path': source_manifest_path,
        'source_validation_manifest_sha256': file_sha256(source_manifest_path),
        'dataset_json_sha256': source_manifest['dataset_json_sha256'],
        'suffix_seed': int(suffix_seed),
        'sentence_split_rule': SENTENCE_SPLIT_RULE,
        'canonical_similarity_chunk': CANONICAL_SIMILARITY_CHUNK,
        'tau_unsaid': TAU_UNSAID,
        'gate_floor': GATE_FLOOR,
        'gate_temperature': GATE_TEMPERATURE,
        'excluded_no_suffix': excluded,
        'query_count': len(queries),
        'candidate_count': len(queries),
        'queries': queries,
    }


def load_or_create_usr_manifest(path: str, source_manifest_path: str,
                                suffix_seed: int = USR_SUFFIX_SEED) -> Dict:
    """Load the frozen USR manifest or build it once; never overwrite a different one."""
    source = json.loads(open(source_manifest_path, encoding='utf-8').read())
    expected = build_manifest(source, source_manifest_path, suffix_seed)
    if os.path.exists(path):
        actual = json.loads(open(path, encoding='utf-8').read())
        if actual != expected:
            raise ValueError('existing USR manifest differs; refusing to resample or overwrite')
        return actual
    write_json(path, expected)
    return expected


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rank_of(scores: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """1-based rank of each row's correct candidate (ties broken deterministically by index).

    ``scores`` may be rectangular ``[N_query, Q_candidate]``; the candidate pool is then the
    *full* pool and ``labels[row]`` gives the correct candidate column (defaults to the
    diagonal for a square matrix).
    """
    if scores.dim() != 2:
        raise ValueError('scores must be 2-D, got %r' % (tuple(scores.shape),))
    n_query, n_candidate = scores.shape
    if labels is None:
        if n_query != n_candidate:
            raise ValueError('labels are required for rectangular scores %r'
                             % (tuple(scores.shape),))
        labels = torch.arange(n_query)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    position = (order == labels.unsqueeze(1)).float().argmax(dim=1)
    return position + 1


def retrieval_report(scores: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """R@1/5/10, MRR, mean/median rank and positive-vs-negative margins.

    Subgroups evaluate ``scores[subset, :]`` with the *original* labels, so the candidate
    pool always stays the full ``Q`` (only query rows are ever selected).
    """
    if scores.dim() != 2:
        raise ValueError('scores must be 2-D, got %r' % (tuple(scores.shape),))
    n_query, n_candidate = scores.shape
    if labels is None:
        labels = torch.arange(n_query)
    labels = labels.long()
    ranks = rank_of(scores, labels)
    report = {}
    for k in (1, 5, 10):
        report['R@%d' % k] = float((ranks <= min(k, n_candidate)).float().mean())
    report['MRR'] = float((1.0 / ranks.float()).mean())
    report['mean_rank'] = float(ranks.float().mean())
    report['median_rank'] = float(ranks.float().median())
    report['candidate_pool'] = int(n_candidate)
    report['query_count'] = int(n_query)
    positive = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    negatives = scores[mask].view(n_query, n_candidate - 1) if n_candidate > 1 \
        else torch.zeros(n_query, 0)
    report['positive_score_mean'] = float(positive.mean())
    report['positive_minus_mean_negative'] = (float((positive - negatives.mean(dim=1)).mean())
                                              if n_candidate > 1 else 0.0)
    report['positive_minus_best_negative'] = (float((positive - negatives.max(dim=1).values).mean())
                                              if n_candidate > 1 else 0.0)
    return report


def random_chance(query_count: int) -> Dict[str, float]:
    return {'R@1': 1.0 / query_count,
            'R@5': min(5.0 / query_count, 1.0),
            'R@10': min(10.0 / query_count, 1.0)}


def delta_report(raw_scores: torch.Tensor, debiased_scores: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Debiased - Raw on the same query rows, same *full* candidate pool (same labels)."""
    raw = retrieval_report(raw_scores, labels)
    debiased = retrieval_report(debiased_scores, labels)
    delta = {('Delta ' + key): debiased[key] - raw[key] for key in ('R@1', 'R@5', 'R@10', 'MRR')}
    rank_delta = (rank_of(raw_scores, labels).float() - rank_of(debiased_scores, labels).float())
    improved = float((rank_delta > 0).float().mean())
    worsened = float((rank_delta < 0).float().mean())
    delta.update({'improved_fraction': improved,
                  'unchanged_fraction': 1.0 - improved - worsened,
                  'worsened_fraction': worsened,
                  'mean_delta_rank': float(rank_delta.mean()),
                  'median_delta_rank': float(rank_delta.median())})
    return delta


def distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'n': int(array.size), 'mean': float(array.mean()), 'median': float(np.median(array)),
            'p10': float(np.percentile(array, 10)), 'p90': float(np.percentile(array, 90))}


def rank_bins(values: Sequence[float], groups: int) -> List[List[int]]:
    """Deterministic equal-count bins by value (ties broken by index), lowest first."""
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    bins = [[] for _ in range(groups)]
    for position, index in enumerate(order):
        bins[min(groups - 1, position * groups // max(1, len(order)))].append(index)
    return bins


def stratified_report(scores: Dict[str, torch.Tensor], values: Sequence[float], groups: int,
                      label: str) -> List[Dict]:
    """Raw vs Debiased retrieval inside deterministic value bins.

    Subgroups only select *query rows* (``scores[name][subset, :]``); the candidate pool
    always stays the full ``Q`` with the original labels.
    """
    output = []
    for position, index in enumerate(rank_bins(values, groups)):
        if not index:
            continue
        subset = torch.tensor(index).long()
        raw = retrieval_report(scores['raw'][subset, :], subset)
        debiased = retrieval_report(scores['debiased'][subset, :], subset)
        delta = delta_report(scores['raw'][subset, :], scores['debiased'][subset, :], subset)
        output.append({'bin': position, 'label': label, 'count': len(index),
                       'value_mean': float(np.mean([values[i] for i in index])),
                       'raw': raw, 'debiased': debiased, 'delta': delta})
    return output


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_images(model, queries, image_root, preprocess, batch_size, device):
    core = model.module if hasattr(model, 'module') else model
    globals_, patches = [], []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        tensors = []
        for query in chunk:
            with Image.open(os.path.join(image_root, query['image_path'])) as image:
                tensors.append(preprocess(image.convert('RGB')))
        tensor = torch.stack(tensors).to(device)
        z_global, patch = core.encode_router_input(tensor)
        globals_.append(z_global.detach().float().cpu())
        patches.append(patch.detach().float().cpu())
    return torch.cat(globals_), torch.cat(patches)


@torch.no_grad()
def encode_texts(model, texts, device):
    core = model.module if hasattr(model, 'module') else model
    tokens = longclip.tokenize(list(texts), truncate=True).to(device)
    return core.encode_text(tokens).detach().float().cpu()


def score_usr(model, queries, image_root, preprocess, device='cpu', image_batch_size: int = 32,
              candidate_chunk_size: int = 64) -> Dict:
    """All four USR score matrices plus positive-attention and novelty diagnostics."""
    query_count = len(queries)
    target_texts = [query['target_text'] for query in queries]
    prefix_texts = [query['caption_said'] for query in queries]
    z_global, patches = encode_images(model, queries, image_root, preprocess,
                                      image_batch_size, device)
    t_target = encode_texts(model, target_texts, device)
    t_prefix = encode_texts(model, prefix_texts, device)
    scale = float(model.clip.logit_scale.exp().clamp(max=100))
    unit_target = F.normalize(t_target, dim=-1)

    scores = {name: torch.zeros(query_count, query_count) for name in SCORERS}
    positive_attention = {name: [] for name in ('raw', 'debiased')}
    gate_all, coverage = [], {name: [] for name in ('raw', 'debiased')}
    core = model.module if hasattr(model, 'module') else model
    for start in range(0, query_count, image_batch_size):
        stop = min(start + image_batch_size, query_count)
        slack = slice(start, stop)
        batch_patches = patches[slack].to(device)
        batch_prefix = t_prefix[slack].to(device)
        # A. global: prefix unused
        scores['global'][slack] = scale * F.normalize(z_global[slack], dim=-1) @ unit_target.t()
        # B. legacy anti-Said: candidate-independent pooling of the prefix attention
        details = core.said_router.forward_with_details(F.normalize(batch_prefix, dim=-1),
                                                       batch_patches)
        anti = unsaid_core.complement_attention(details['scores'], TAU_UNSAID)
        z_anti = F.normalize(torch.einsum('bp,bpd->bd', anti, batch_patches), dim=-1)
        scores['anti_said'][slack] = scale * z_anti.cpu() @ unit_target.t()
        # C/D. raw (beta=0) and debiased (beta=1) hidden-semantic scoring, chunked
        for name, beta in (('raw', 0.0), ('debiased', 1.0)):
            matrix = model.score_unsaid_candidates(batch_patches, batch_prefix,
                                                   unit_target.to(device), beta=beta,
                                                   tau_unsaid=TAU_UNSAID, gate_floor=GATE_FLOOR,
                                                   gate_temperature=GATE_TEMPERATURE,
                                                   candidate_chunk_size=candidate_chunk_size
                                                   if candidate_chunk_size else 0)
            scores[name][slack] = matrix.detach().float().cpu()
        # positive-query attention diagnostics (own target only; B == C identity)
        for name, beta in (('raw', 0.0), ('debiased', 1.0)):
            own = model.score_unsaid_candidates(batch_patches, batch_prefix, unit_target[slack].to(device),
                                                beta=beta, tau_unsaid=TAU_UNSAID,
                                                gate_floor=GATE_FLOOR,
                                                gate_temperature=GATE_TEMPERATURE,
                                                candidate_chunk_size=0, return_details=True)
            gate = unsaid_core.said_suppression_gate(details['scores'], GATE_FLOOR,
                                                     GATE_TEMPERATURE)
            attention = own['attention'][torch.arange(stop - start), torch.arange(stop - start)]
            positive_attention[name].append(attention.detach().float().cpu())
            coverage[name].extend(unsaid_core.said_coverage(attention, gate).tolist())
            if name == 'raw':
                gate_all.append(gate.detach().float().cpu())
    novelty = (1.0 - (F.normalize(t_prefix, dim=-1) * F.normalize(t_target, dim=-1)).sum(dim=-1))
    return {
        'scores': scores,
        'raw_said_coverage': coverage['raw'],
        'debiased_said_coverage': coverage['debiased'],
        'novelty': [float(value) for value in novelty],
        'gate': torch.cat(gate_all) if gate_all else torch.zeros(0),
    }


def evaluate(model, queries, image_root, preprocess, device='cpu', image_batch_size: int = 32,
             candidate_chunk_size: int = 64) -> Dict:
    """Full USR evaluation with safety checks (RNG, state digest, eval mode)."""
    was_training = model.training
    digest_before = state_digest(model)
    with rng_guard():
        model.eval()
        try:
            payload = score_usr(model, queries, image_root, preprocess, device=device,
                                image_batch_size=image_batch_size,
                                candidate_chunk_size=candidate_chunk_size)
        finally:
            if was_training:
                model.train()
    payload['state_digest_before'] = digest_before
    payload['state_digest_after'] = state_digest(model)
    payload['parameter_grads'] = sum(1 for parameter in model.parameters()
                                     if parameter.grad is not None)
    payload['reports'] = {name: retrieval_report(matrix) for name, matrix in payload['scores'].items()}
    payload['random_chance'] = random_chance(len(queries))
    payload['delta_debiased_vs_raw'] = delta_report(payload['scores']['raw'],
                                                    payload['scores']['debiased'])
    payload['positive_attention'] = {
        'raw_said_coverage': distribution(payload['raw_said_coverage']),
        'debiased_said_coverage': distribution(payload['debiased_said_coverage']),
        'gate': distribution(payload['gate'].reshape(-1).tolist()),   # flattened gate values
    }
    payload['coverage_quartiles'] = stratified_report(payload['scores'],
                                                      payload['raw_said_coverage'], 4, 'coverage')
    first = [i for i, q in enumerate(queries) if q['J'] == q['K'] + 1]
    later = [i for i, q in enumerate(queries) if q['J'] >= q['K'] + 2]
    payload['position_split'] = {
        'first_suffix': _subset_report(payload['scores'], first),
        'later_suffix': _subset_report(payload['scores'], later),
    }
    payload['relative_position_terciles'] = stratified_report(
        payload['scores'], [q['J'] / float(q['N']) for q in queries], 3, 'J/N')
    payload['novelty_terciles'] = stratified_report(payload['scores'], payload['novelty'], 3,
                                                    'novelty')
    payload['novelty_distribution'] = distribution(payload['novelty'])
    return payload


def _subset_report(scores: Dict[str, torch.Tensor], index: Sequence[int]) -> Dict:
    """Query-row subgroup with the full candidate pool and the original labels."""
    if not index:
        return {'count': 0}
    subset = torch.tensor(list(index)).long()
    raw = retrieval_report(scores['raw'][subset, :], subset)
    debiased = retrieval_report(scores['debiased'][subset, :], subset)
    return {'count': len(index), 'raw': raw, 'debiased': debiased,
            'delta': delta_report(scores['raw'][subset, :], scores['debiased'][subset, :], subset)}


def state_digest(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def parse_checkpoints(spec: str) -> List[Tuple[str, str]]:
    entries = []
    for entry in str(spec).split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError('checkpoint entries must be "tag:path", got %r' % (entry,))
        tag, path = entry.split(':', 1)
        entries.append((tag.strip(), path.strip()))
    if not entries:
        raise ValueError('at least one checkpoint is required')
    return entries


def main():
    parser = argparse.ArgumentParser(description='USR: fixed unsaid semantic retrieval')
    parser.add_argument('--checkpoints', required=True, help='comma-separated "tag:path"')
    parser.add_argument('--manifest', default=os.path.join('outputs', 'validation',
                                                           'sharegpt4v1k_usr_manifest.json'))
    parser.add_argument('--source_manifest', default=os.path.join('outputs', 'validation',
                                                                  'sharegpt4v1k_manifest.json'))
    parser.add_argument('--data_root', default=os.environ.get('SHARE4V_DATA_ROOT',
                                                              '../datasets/ShareGPT4V'))
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'usr'))
    parser.add_argument('--image_batch_size', type=int, default=32)
    parser.add_argument('--candidate_chunk_size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    manifest = load_or_create_usr_manifest(args.manifest, args.source_manifest)
    queries = manifest['queries'][:args.limit or None]
    clip_model, preprocess = longclip.load_from_clip(args.base_model, device='cpu', args=args)
    model = SALUModel(clip_model, said_feature_source='residual').float().to(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    for tag, path in parse_checkpoints(args.checkpoints):
        if path in ('', '-'):                       # no checkpoint: NOT an official baseline
            checkpoint = {'step': 0, 'epoch': 0}
            checkpoint_sha256 = None
            tag = tag + '_fresh_untracked'      # never named "initial" in the results
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint_sha256 = file_sha256(path)
            incompatible = model.load_state_dict(checkpoint['model'], strict=False)
            if list(incompatible.missing_keys) or list(incompatible.unexpected_keys):
                raise RuntimeError('checkpoint %s does not match the model' % path)
        payload = evaluate(model, queries, args.data_root, preprocess, device=args.device,
                           image_batch_size=args.image_batch_size,
                           candidate_chunk_size=args.candidate_chunk_size)
        payload.update({'tag': tag, 'checkpoint': path or 'initial (no checkpoint loaded)',
                        'checkpoint_sha256': checkpoint_sha256,
                        'checkpoint_step': int(checkpoint.get('step', 0)),
                        'protocol': PROTOCOL_NAME, 'query_count': len(queries),
                        'manifest_sha256': file_sha256(args.manifest)})
        payload['scores'] = {name: matrix.tolist() for name, matrix in payload['scores'].items()}
        payload['gate'] = payload['gate'].tolist()
        write_json(os.path.join(args.output_dir, 'usr_%s.json' % tag), payload)
        summary = payload['reports']
        print('USR %-8s ' % tag + ' '.join('%s:R1=%.4f MRR=%.4f' % (name, summary[name]['R@1'],
                                                                   summary[name]['MRR'])
                                           for name in SCORERS), flush=True)
        print('USR %-8s delta_debiased_vs_raw %s' % (tag, {k: round(v, 4) for k, v in
                                                           payload['delta_debiased_vs_raw'].items()}),
              flush=True)
    print('USR_DONE output=%s' % args.output_dir, flush=True)


if __name__ == '__main__':
    main()
