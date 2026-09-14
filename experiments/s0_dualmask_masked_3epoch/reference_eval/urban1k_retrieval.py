"""Urban-1k retrieval evaluation, faithful to the upstream SmartCLIP protocol.

Upstream reference: ``eval/retrieval/Urban1k.py`` in Mid-Push/SmartCLIP (unchanged since the
repository's initial commit). That script does exactly this:

    model, preprocess = longclip.load(checkpoint)
    text_feature = model.encode_text(longclip.tokenize(captions, truncate=True))
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    image_embeds = stack(model.encode_image(preprocess(image)))
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    T2I R@1 = mean(argmax(text_i @ image_embeds.T) == i)
    I2T R@1 = mean(argmax(image_i @ text_feature.T) == i)

This module reproduces that metric definition exactly (normalised native CLS embeddings, plain inner
product, full 1000-item pool, diagonal as the positive, no reranking, no masked/decoder features,
same 248-token truncating tokenizer) and additionally reports R@5/R@10, which the upstream script
does not print. R@1 is therefore directly comparable with upstream numbers; R@5/R@10 are a superset
added for this project's reporting convention.

The dataset (`BeichenZhang/Urban1k`, revision pinned in ``REVISION.txt``) is 1000 images with 1000
long captions (median 106 words), i.e. a long-caption benchmark that stresses the 248-token budget.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_ROOT = '/root/datasets/Urban1k/Urban1k'


def image_caption_pairs(root: str = DEFAULT_ROOT) -> List[Tuple[str, str]]:
    """``[(image_path, caption_path)]`` sorted by stem, exactly as the upstream script pairs them."""
    image_root = os.path.join(root, 'image')
    caption_root = os.path.join(root, 'caption')
    if not os.path.isdir(image_root) or not os.path.isdir(caption_root):
        raise FileNotFoundError('Urban1k layout not found under %s' % root)
    stems = sorted({os.path.splitext(name)[0] for name in os.listdir(image_root)})
    caption_stems = {os.path.splitext(name)[0] for name in os.listdir(caption_root)}
    if set(stems) != caption_stems:
        missing = sorted(set(stems) - caption_stems)[:5]
        extra = sorted(caption_stems - set(stems))[:5]
        raise ValueError('image/caption stems do not match (missing %s, extra %s)'
                         % (missing, extra))
    pairs = []
    for stem in stems:
        image_path = os.path.join(image_root, stem + '.jpg')
        caption_path = os.path.join(caption_root, stem + '.txt')
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        pairs.append((image_path, caption_path))
    return pairs


def read_captions(pairs) -> List[str]:
    """First line of each caption, like upstream's ``f.readlines()[0]`` (verified single-line)."""
    captions = []
    for _, caption_path in pairs:
        with open(caption_path, encoding='utf-8') as handle:
            captions.append(handle.readline().strip())
    return captions


def _recall(similarity: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """Rank-based recall with the diagonal as the positive (upstream metric definition)."""
    targets = torch.arange(similarity.shape[0], device=similarity.device)
    out = {}
    for k in ks:
        topk = similarity.topk(min(k, similarity.shape[1]), dim=1).indices
        out['R%d' % k] = float((topk == targets[:, None]).any(dim=1).float().mean())
    return out


@torch.no_grad()
def evaluate_urban1k(model, preprocess, root: str = DEFAULT_ROOT, batch_size: int = 64,
                     device: str = 'cuda', tokenizer=None,
                     checkpoint_sha256: Optional[str] = None) -> Dict:
    """Native-CLS retrieval on Urban-1k: ``normalize(encode_image)`` vs ``normalize(encode_text)``."""
    # the repository's LongCLIP (248 tokens) lives in the ``model`` package; the bare top-level
    # ``longclip`` module uses relative imports and only works from inside it
    from model import longclip as _longclip
    tokenizer = tokenizer or _longclip.tokenize

    pairs = image_caption_pairs(root)
    captions = read_captions(pairs)
    text_tokens = tokenizer(captions, truncate=True).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        images = torch.stack([preprocess(Image.open(path).convert('RGB')) for path, _ in chunk])
        images = images.to(device)
        features = model.encode_image(images)
        image_features.append(features)
    image_features = torch.cat(image_features, dim=0)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    image2text = image_features @ text_features.t()
    text2image = text_features @ image_features.t()
    result = {
        'protocol': 'urban1k-native-cls-v1',
        'root': root,
        'n_images': len(pairs),
        'n_captions': len(captions),
        'image_representation': 'legacy_cls',
        'global_pool': 'n/a',
        'similarity': 'normalized features, plain inner product, full 1000-item pool',
        'positive': 'diagonal',
        'image2text': _recall(image2text),
        'text2image': _recall(text2image),
        'checkpoint_sha256': checkpoint_sha256,
        'upstream_reference': 'eval/retrieval/Urban1k.py (Mid-Push/SmartCLIP)',
        'upstream_metrics_reproduced': ['T2I R@1', 'I2T R@1'],
        'added_here': ['R@5', 'R@10'],
    }
    return result


def dataset_fingerprint(root: str = DEFAULT_ROOT) -> Dict[str, str]:
    """Provenance for the dataset: revision, archive hash, counts, caption digest."""
    digest = hashlib.sha256()
    pairs = image_caption_pairs(root)
    for image_path, caption_path in pairs:
        digest.update(os.path.basename(image_path).encode())
        with open(caption_path, 'rb') as handle:
            digest.update(handle.read())
    revision_file = os.path.join(os.path.dirname(root.rstrip('/')), 'REVISION.txt')
    revision = None
    if os.path.exists(revision_file):
        with open(revision_file) as handle:
            revision = handle.read().strip()
    return {'root': root, 'n_pairs': len(pairs), 'revision': revision,
            'pair_caption_sha256': digest.hexdigest()}
