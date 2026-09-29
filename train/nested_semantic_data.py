"""Literal sentence proxy and spawn-safe, full-size indexed ShareGPT4V dataset."""
import hashlib
import json
import mmap
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import longclip
from train.said_cvssl_data import reference_view_a_transform, image_id_from_path

TEXT_RULE = 'newline->space; strip; literal . split; nonempty; longest consecutive prefix; independent F/O/E <=248 incl SOT/EOT'


def text_views(caption):
    caption = caption.replace('\n', ' ').strip()
    parts = [x.strip() for x in caption.split('. ') if x.strip()]
    if not caption or not parts:
        raise ValueError('empty full caption')
    token_len = lambda x: len(longclip._tokenizer.encode(x)) + 2
    if token_len(parts[0]) > 248:
        tf = longclip.tokenize([caption], context_length=248, truncate=True)[0]
        empty = longclip.tokenize([''], context_length=248)[0]
        return dict(tokens_f=tf, tokens_o=empty, tokens_e=empty.clone(), valid=False,
                    reason='first_segment_overlong', views=[caption, '', ''],
                    untruncated_lengths=[token_len(caption), 2, 2])
    best = (parts[0], '', '', [token_len(parts[0]), 2, 2])
    for k in range(2, len(parts) + 1):
        f, o, e = '. '.join(parts[:k]), parts[0], '. '.join(parts[1:k])
        lengths = [token_len(x) for x in (f, o, e)]
        if max(lengths) <= 248:
            best = f, o, e, lengths
        # Check every prefix: do not assume tokenizer lengths strictly increase.
    f, o, e, lengths = best
    tokens = longclip.tokenize([f, o, e], context_length=248, truncate=False)
    return dict(tokens_f=tokens[0], tokens_o=tokens[1], tokens_e=tokens[2],
                valid=bool(o and e), reason='valid' if o and e else 'single_visible_segment',
                views=[f, o, e], untruncated_lengths=lengths)


def sample_split_k(n, sampling_seed, epoch, sample_id):
    """Uniform local draw, independent of every process-wide RNG and rank."""
    if n < 2:
        raise ValueError('random_k requires at least two visible segments')
    material = f'{int(sampling_seed)}:{int(epoch)}:{int(sample_id)}'.encode('utf-8')
    local_seed = int.from_bytes(hashlib.sha256(material).digest(), 'big')
    return random.Random(local_seed).randrange(1, n)


def sampled_text_views(caption, sampling_mode='fixed_first', sampling_seed=0,
                       epoch=0, sample_id=0):
    """Pack F with the unchanged old rule, then optionally resplit ONLY that F.

    tokens_o/e carry prefix/remainder in random_k. The old fields below are
    retained only on CPU for a replay digest against the original fixed A3 logs.
    n=0, K=0 marks the overlong-first-segment fallback (no complete visible
    segment count); n=1, K=0 marks a single visible segment.
    """
    if sampling_mode not in ('fixed_first', 'random_k'):
        raise ValueError(f'Unknown sampling_mode: {sampling_mode}')
    original = text_views(caption)
    result = dict(original, reference_views=original['views'],
                  reference_tokens_o=original['tokens_o'],
                  reference_tokens_e=original['tokens_e'])
    if not original['valid']:
        result.update(n=1 if original['reason'] == 'single_visible_segment' else 0, K=0)
        return result
    parts = original['views'][0].split('. ')
    n = len(parts)
    assert n >= 2 and all(parts)
    k = 1 if sampling_mode == 'fixed_first' else sample_split_k(n, sampling_seed, epoch, sample_id)
    if not 1 <= k < n:
        raise ValueError(f'Invalid split: sample_id={sample_id}, epoch={epoch}, n={n}, K={k}')
    result.update(n=n, K=k)
    prefix, remainder = '. '.join(parts[:k]), '. '.join(parts[k:])
    assert prefix and remainder and '. '.join((prefix, remainder)) == original['views'][0]
    lengths = [len(longclip._tokenizer.encode(s)) + 2 for s in (prefix, remainder)]
    if max(lengths) > 248:
        raise ValueError(f'RandomK token boundary overflow: sample_id={sample_id}, epoch={epoch}, '
                         f'n={n}, K={k}, P/R_lengths={lengths}, F={original["views"][0]!r}, '
                         f'P={prefix!r}, R={remainder!r}')
    if k != 1:
        local_tokens = longclip.tokenize([prefix, remainder], context_length=248, truncate=False)
        result.update(tokens_o=local_tokens[0], tokens_e=local_tokens[1],
                      views=[original['views'][0], prefix, remainder],
                      untruncated_lengths=[original['untruncated_lengths'][0], *lengths])
    # k=1 preserves all old fields bit-for-bit, including token tensors.
    return result


def sampling_diagnostics(batch):
    """CPU metadata/digests only; never enters the model's objective graph."""
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                        separators=(',', ':')).encode()).hexdigest()
    ids = batch['sample_id'].tolist()
    tokens = [batch[k].cpu().tolist() for k in ('tokens_f', 'tokens_o', 'tokens_e')]
    ns, ks = batch['n'].tolist(), batch['K'].tolist()
    valid = batch['valid'].cpu().tolist()
    by_n = {}
    p_lengths, r_lengths, p_counts, r_counts = [], [], [], []
    for n, k, enabled, lengths in zip(ns, ks, valid, batch['untruncated_lengths']):
        if enabled:
            by_n.setdefault(str(n), Counter())[str(k)] += 1
            p_lengths.append(lengths[1]); r_lengths.append(lengths[2])
            p_counts.append(k); r_counts.append(n-k)
    histogram = lambda values: dict(sorted(Counter(values).items()))
    reference = dict(sample_ids=ids, views=batch['reference_views'],
                     tokens=[tokens[0], batch['reference_tokens_o'].tolist(),
                             batch['reference_tokens_e'].tolist()])
    return dict(sample_ids=ids, n=ns, K=ks, valid_count=sum(valid),
                k1_count=sum(k == 1 and enabled for k, enabled in zip(ks, valid)),
                K_histogram_by_n={n: dict(v) for n, v in by_n.items()},
                prefix_segments=histogram(p_counts), remainder_segments=histogram(r_counts),
                prefix_token_lengths=histogram(p_lengths), remainder_token_lengths=histogram(r_lengths),
                sample_id_sha256=digest(ids),
                full_view_sha256=digest(dict(sample_ids=ids, F=[v[0] for v in batch['views']], tokens_f=tokens[0])),
                local_views_sha256=digest(dict(sample_ids=ids, PR=[v[1:] for v in batch['views']], tokens_pr=tokens[1:])),
                split_sha256=digest(dict(sample_ids=ids, n=ns, K=ks)),
                fixed_first_reference_stream_sha256=digest(reference))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prepare_index(annotation, output):
    """A lossless order-preserving IO index, NOT a training subset or token cache."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads(Path(annotation).read_text())
    offsets = [0]
    with (output / 'records.jsonl').open('wb') as f:
        for record in records[1000:]:
            raw = json.dumps({'image': record['image'], 'caption': record['conversations'][1]['value']},
                             ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
            f.write(raw)
            offsets.append(offsets[-1] + len(raw))
    np.save(output / 'offsets.npy', np.asarray(offsets, dtype=np.int64))
    meta = dict(annotation=str(Path(annotation).resolve()), annotation_sha256=file_sha(annotation),
                original_records=len(records), training_records=len(offsets)-1, skip=1000,
                records_sha256=file_sha(output / 'records.jsonl'), text_rule=TEXT_RULE)
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return meta


class NestedDataset(Dataset):
    def __init__(self, index_dir, image_root, sampling_mode='fixed_first', sampling_seed=0):
        self.index_dir, self.image_root = Path(index_dir), Path(image_root)
        self.metadata = json.loads((self.index_dir / 'metadata.json').read_text())
        self.transform = reference_view_a_transform()
        if sampling_mode not in ('fixed_first', 'random_k'):
            raise ValueError(sampling_mode)
        self.sampling_mode, self.sampling_seed, self.epoch = sampling_mode, int(sampling_seed), 0
        self._records = self._offsets = self._file = None

    def __len__(self):
        return self.metadata['training_records']

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        if self._records is None:
            self._file = (self.index_dir / 'records.jsonl').open('rb')
            self._records = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = np.load(self.index_dir / 'offsets.npy', mmap_mode='r')
        record = json.loads(self._records[self._offsets[index]:self._offsets[index+1]])
        views = sampled_text_views(record['caption'], self.sampling_mode, self.sampling_seed,
                                   self.epoch, index+1000)
        path = self.image_root / record['image']
        try:
            with Image.open(path) as im:
                image = self.transform(im.convert('RGB'))
        except Exception as exc:
            raise RuntimeError(f'Image failure sample={index+1000} path={path}') from exc
        return dict(image=image, image_id=image_id_from_path(record['image']),
                    sample_id=index+1000, **views)


def collate(samples):
    result = {k: torch.stack([s[k] for s in samples]) for k in ('image', 'tokens_f', 'tokens_o', 'tokens_e')}
    result.update({k: torch.tensor([s[k] for s in samples]) for k in ('valid', 'sample_id', 'image_id')})
    for k in ('reason', 'views', 'untruncated_lengths'):
        result[k] = [s[k] for s in samples]
    if 'n' in samples[0]:
        result.update({k: torch.tensor([s[k] for s in samples]) for k in ('n', 'K')})
        result['reference_views'] = [s['reference_views'] for s in samples]
        for k in ('reference_tokens_o', 'reference_tokens_e'):
            result[k] = torch.stack([s[k] for s in samples])
    return result
