"""Immutable held-out probe. This module is never imported by training."""
import hashlib
import json
import math
import random
from pathlib import Path

DETAIL_LEVELS = ('first_sentence', '25%', '50%', '75%', '100%')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def caption_ladder(dense):
    sentences = [part.strip() for part in dense.replace('\n', ' ').split('. ')
                 if part.strip()]
    if not sentences:
        raise ValueError('empty dense caption')
    counts = [1] + [max(1, math.ceil(len(sentences) * fraction))
                    for fraction in (.25, .5, .75, 1.)]
    variants, aliases, seen = [], {}, {}
    for level, count in zip(DETAIL_LEVELS, counts):
        caption = '. '.join(sentences[:count])
        if caption not in seen:
            seen[caption] = len(variants)
            variants.append({'caption': caption, 'sentence_count': count})
        aliases[level] = seen[caption]
    return {'sentence_count': len(sentences), 'variants': variants, 'levels': aliases}


def training_like_caption(dense, seed):
    # Exactly the existing ShareGPT4V prefix construction, with a PRIVATE RNG.
    # This fixed validation caption does not modify the production RNG stream.
    sentences = dense.replace('\n', ' ').split('. ')
    count = random.Random(seed).randint(1, len(sentences))
    return {'caption': '. '.join(sentences[:count]), 'prefix_sentences': count,
            'total_sentences': len(sentences), 'seed': seed,
            'construction': 'uniform integer prefix of newline-flattened .-space split'}


def build_manifest(records, dataset_sha256, n=512, seed=26, heldout_count=1000):
    samples, seen = [], set()
    for index, record in enumerate(records[:heldout_count]):
        image_id = record['image']
        if image_id in seen:
            continue
        if Path(image_id).is_absolute() or '..' in Path(image_id).parts:
            raise ValueError('probe image IDs must be relative dataset paths')
        dense = record['conversations'][1]['value']
        seen.add(image_id)
        samples.append({'order': len(samples), 'image_id': image_id,
                        'dataset_index': index, 'dense_caption': dense,
                        'training_caption': training_like_caption(dense, seed + index),
                        'caption_detail': caption_ladder(dense)})
        if len(samples) == n:
            break
    if len(samples) != n or n < 2:
        raise ValueError('not enough unique held-out images (N must be >= 2)')
    return {'schema_version': 1, 'name': '固定表征平衡验证集', 'n': n, 'seed': seed,
            'dataset_sha256': dataset_sha256, 'heldout_count': heldout_count,
            'selection': 'first N unique image IDs from held-out prefix, source order',
            'training_excludes_dataset_indices_below': heldout_count,
            'detail_scope': 'validation only; aliases share inference, all levels retain N images',
            'samples': samples}


def load_or_create_manifest(path, dataset_json, n=512, seed=26):
    path = Path(path)
    dataset_digest = file_sha256(dataset_json)
    expected = build_manifest(json.loads(Path(dataset_json).read_text(encoding='utf-8')),
                              dataset_digest, n, seed)
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError('existing probe manifest differs; refusing resampling or overwrite')
        return actual
    write_json(path, expected)
    return expected
