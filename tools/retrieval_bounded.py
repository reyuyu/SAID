"""FP32 retrieval with explicit positives, bounded scoring and verified shards."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch

KS = (1, 5, 10)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.pending')
    with tmp.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    if exclusive:
        try:
            os.link(tmp, path)
        finally:
            tmp.unlink()
    else:
        os.replace(tmp, path)


def validate_ids(image_ids, caption_ids, positives):
    if not image_ids or not caption_ids:
        raise ValueError('empty gallery/query')
    if len(set(image_ids)) != len(image_ids) or len(set(caption_ids)) != len(caption_ids):
        raise ValueError('duplicate image/caption ID')
    if len(caption_ids) != len(positives):
        raise ValueError('positive length mismatch')
    if not set(positives) <= set(image_ids) or set(positives) != set(image_ids):
        raise ValueError('missing positive/image without captions')


def validate_features(x):
    if x.ndim != 2 or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('features must be finite 2D FP32')
    if (x.norm(dim=1) == 0).any():
        raise ValueError('zero feature')


def select(scores, indices, k=11):
    """Descending FP32 score, exact ties by ascending manifest candidate index.

    Near ties are NOT quantized. Their margins are recorded by the caller.
    Sorting indices first preserves the tie rule across arbitrary merge blocks.
    """
    order = indices.argsort(dim=1, stable=True)
    indices = indices.gather(1, order)
    scores = scores.gather(1, order)
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), indices.gather(1, order)


def stream_topk(query, gallery, query_chunk=256, gallery_chunk=4096):
    validate_features(query); validate_features(gallery)
    if query.shape[1] != gallery.shape[1] or min(query_chunk, gallery_chunk) < 1:
        raise ValueError('invalid dimension/chunk')
    all_indices, all_scores = [], []
    for start in range(0, len(query), query_chunk):
        q = query[start:start + query_chunk]
        best_s = q.new_empty((len(q), 0))
        best_i = torch.empty((len(q), 0), dtype=torch.long, device=q.device)
        for g in range(0, len(gallery), gallery_chunk):
            block = q @ gallery[g:g + gallery_chunk].T
            if not torch.isfinite(block).all():
                raise ValueError('nonfinite similarities')
            idx = torch.arange(g, g + block.shape[1], device=q.device).expand(len(q), -1)
            s, i = select(block, idx)
            best_s, best_i = select(torch.cat((best_s, s), 1), torch.cat((best_i, i), 1))
        all_indices.append(best_i.cpu()); all_scores.append(best_s.cpu())
    return torch.cat(all_indices), torch.cat(all_scores)


def summarize(indices, scores, truth):
    hits = {str(k): [bool(set(row[:k].tolist()) & truth[i]) for i, row in enumerate(indices)] for k in KS}
    margins = {}
    for k in KS:
        if scores.shape[1] > k:
            d = scores[:, k - 1] - scores[:, k]
            margins[str(k)] = {'exact_ties': int((d == 0).sum()),
                               'near_ties_le_1e-6': int((d <= 1e-6).sum()),
                               'minimum_margin': float(d.min())}
    return {'query_count': len(truth), 'candidate_count': None,
            'correct': {k: sum(v) for k, v in hits.items()},
            'recall_percent': {k: 100 * sum(v) / len(v) for k, v in hits.items()},
            'boundary_audit': margins}, hits


def retrieval(images, texts, image_ids, caption_ids, positives, **chunks):
    validate_ids(image_ids, caption_ids, positives)
    if len(images) != len(image_ids) or len(texts) != len(caption_ids):
        raise ValueError('feature/ID count mismatch')
    pos_i = {x: i for i, x in enumerate(image_ids)}
    it = [set() for _ in image_ids]
    for j, p in enumerate(positives):
        it[pos_i[p]].add(j)
    tt = [{pos_i[p]} for p in positives]
    out, details = {}, {}
    for name, q, g, truth in [('I2T', images, texts, it), ('T2I', texts, images, tt)]:
        idx, scores = stream_topk(q, g, **chunks)
        out[name], hits = summarize(idx, scores, truth)
        out[name]['candidate_count'] = len(g)
        details[name] = {'top_indices': idx, 'top_scores': scores, 'hits': hits}
    return out, details


class ShardCache:
    """Each receipt binds identity, IDs, tensor shape/dtype and file SHA.

    Orphan tensor/receipt or corrupt shard fails closed. Missing both means
    an unstarted shard and can be encoded on explicit task recovery.
    """
    def __init__(self, root, identity):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        p = self.root / 'IDENTITY.json'
        if p.exists():
            if json.loads(p.read_text()) != identity:
                raise ValueError('cache identity mismatch')
        elif any(self.root.iterdir()):
            raise ValueError('cache missing identity')
        else:
            atomic_json(p, identity, exclusive=True)

    def paths(self, kind, start):
        p = self.root / f'{kind}_{start:08d}.pt'
        return p, p.with_suffix('.json')

    def read(self, kind, start, ids):
        p, r = self.paths(kind, start)
        if not p.exists() and not r.exists():
            return None
        if not p.is_file() or not r.is_file():
            raise ValueError('incomplete cache shard')
        meta = json.loads(r.read_text())
        if meta['identity_sha'] != digest(self.identity) or meta['ids'] != ids or meta['sha256'] != sha(p):
            raise ValueError('cache shard SHA/IDs/identity mismatch')
        x = torch.load(p, map_location='cpu', weights_only=True)
        validate_features(x)
        if list(x.shape) != meta['shape'] or len(x) != len(ids):
            raise ValueError('cache shard shape mismatch')
        return x

    def write(self, kind, start, ids, x):
        validate_features(x)
        p, r = self.paths(kind, start)
        if p.exists() or r.exists():
            raise FileExistsError(p)
        tmp = p.with_suffix('.pending')
        torch.save(x.cpu(), tmp)
        os.link(tmp, p); tmp.unlink()
        atomic_json(r, {'identity_sha': digest(self.identity), 'ids': ids,
                        'sha256': sha(p), 'shape': list(x.shape)}, exclusive=True)
