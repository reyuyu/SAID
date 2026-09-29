"""Literal sentence proxy and spawn-safe, full-size indexed ShareGPT4V dataset."""
import hashlib
import json
import mmap
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
    def __init__(self, index_dir, image_root):
        self.index_dir, self.image_root = Path(index_dir), Path(image_root)
        self.metadata = json.loads((self.index_dir / 'metadata.json').read_text())
        self.transform = reference_view_a_transform()
        self._records = self._offsets = self._file = None

    def __len__(self):
        return self.metadata['training_records']

    def __getitem__(self, index):
        if self._records is None:
            self._file = (self.index_dir / 'records.jsonl').open('rb')
            self._records = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = np.load(self.index_dir / 'offsets.npy', mmap_mode='r')
        record = json.loads(self._records[self._offsets[index]:self._offsets[index+1]])
        views = text_views(record['caption'])
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
    return result
