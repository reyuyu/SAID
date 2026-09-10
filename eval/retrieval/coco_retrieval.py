"""Standard, deterministic COCO 5-caption retrieval evaluation."""
import json
import os
from typing import Iterable

import torch


def retrieval_metrics(image_features, text_features, captions_per_image=5):
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    sim = image_features @ text_features.T
    n = image_features.shape[0]
    out = {}
    for direction, scores, total, truth in (
        ('image2text', sim, n, lambda i: range(i * captions_per_image, (i + 1) * captions_per_image)),
        ('text2image', sim.T, n * captions_per_image, lambda i: (i // captions_per_image,)),
    ):
        for k in (1, 5, 10):
            hits = 0
            for i in range(total):
                top = scores[i].topk(min(k, scores.shape[1])).indices.tolist()
                if any(j in top for j in truth(i)):
                    hits += 1
            out[f'{direction}_R{k}'] = hits / float(total)
    return out


@torch.inference_mode()
def evaluate_coco(model, preprocess, root=None, ann_file=None, batch_size=64,
                  similarity_chunk=1024, device=None):
    from torchvision.datasets import CocoCaptions
    from model import longclip
    root = root or os.path.join(os.environ.get('COCO_DATA_ROOT', '../../datasets/coco'), 'val2017')
    ann_file = ann_file or os.path.join(os.path.dirname(root), 'annotations', 'captions_val2017.json')
    ds = CocoCaptions(root=root, annFile=ann_file, transform=preprocess)
    device = device or next(model.parameters()).device
    core = model.module if hasattr(model, 'module') else model
    ims, txt = [], []
    for start in range(0, len(ds), batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        ims.extend([x[0] for x in batch])
        caps = [c[:5] for _, c in batch]
        flat = [c for row in caps for c in row]
        ims_t = torch.stack(ims[-len(batch):]).to(device)
        ims_f = core.encode_image(ims_t).detach().cpu().float()
        tx = longclip.tokenize(flat, truncate=True).to(device)
        txt_f = core.encode_text(tx).detach().cpu().float()
        # Keep CPU fp32 features; chunking is an explicit protocol parameter.
        ims[-len(batch):] = []
        if 'image_features' not in locals(): image_features = []
        image_features.append(ims_f)
        if 'text_features' not in locals(): text_features = []
        text_features.append(txt_f)
    return evaluate_features(torch.cat(image_features), torch.cat(text_features), captions_per_image=5,
                             similarity_chunk=similarity_chunk)


def evaluate_features(image_features, text_features, captions_per_image=5, similarity_chunk=1024):
    # Chunk argument is retained for protocol reproducibility; metrics are exact.
    return retrieval_metrics(image_features, text_features, captions_per_image)


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
