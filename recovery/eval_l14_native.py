"""Native ViT-L/14 retrieval evaluator for the HNS-v1 Pure3 run.

This is an isolated read-only wrapper around the repository's canonical
retrieval definitions.  It accepts a bare CLIP state dict and never loads the
training wrapper or applies a mask at inference time.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model import longclip
from eval.retrieval.coco_retrieval import evaluate_coco
from tools.urban1k_retrieval import evaluate_urban1k


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metric(sim, image_ids, positives):
    """Return canonical full-pool manifest metrics for arbitrary positive IDs."""
    image_ids = [str(x) for x in image_ids]
    positives = [str(x) for x in positives]
    id_to_i = {x: i for i, x in enumerate(image_ids)}
    out = {}
    for k in (1, 5, 10):
        i2t = 0
        for i, image_id in enumerate(image_ids):
            order = torch.argsort(sim[i], descending=True)[:k].tolist()
            if any(positives[j] == image_id for j in order):
                i2t += 1
        t2i = 0
        for j, positive in enumerate(positives):
            order = torch.argsort(sim[:, j], descending=True)[:k].tolist()
            if id_to_i.get(positive, -1) in order:
                t2i += 1
        out[f'R@{k}'] = {'I2T': i2t / len(image_ids), 'T2I': t2i / len(positives)}
    return out


@torch.inference_mode()
def evaluate_manifest(model, preprocess, manifest, image_root, device, batch_size):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    images = []
    seen = set()
    for row in rows:
        image_id = str(row['image_id'])
        if image_id not in seen:
            seen.add(image_id)
            images.append((image_id, row['image_path']))
    text_features = []
    for start in range(0, len(rows), batch_size):
        tokens = longclip.tokenize([r['caption'] for r in rows[start:start + batch_size]], truncate=True).to(device)
        text_features.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    text_features = torch.cat(text_features)
    image_features = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensor = torch.stack([preprocess(Image.open(Path(image_root) / path).convert('RGB')) for _, path in batch]).to(device)
        image_features.append(F.normalize(model.encode_image(tensor).float(), dim=-1).cpu())
    image_features = torch.cat(image_features)
    return {'n_images': len(images), 'n_captions': len(rows),
            'metrics': metric(image_features @ text_features.T,
                              [x[0] for x in images],
                              [str(r['positive_image_id']) for r in rows])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', choices=('coco', 'urban', 'manifest'), required=True)
    p.add_argument('--root')
    p.add_argument('--manifest')
    p.add_argument('--image-root')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--model', default='ViT-L/14')
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    model, preprocess = longclip.load_from_clip(args.model, device='cpu', args=argparse.Namespace())
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.float().to(args.device).eval()
    started = time.time()
    if args.dataset == 'coco':
        ann = str(Path(args.root).parent / 'annotations' / 'captions_val2017.json')
        result = {'metrics': evaluate_coco(model, preprocess, root=args.root, ann_file=ann,
                                           batch_size=args.batch_size, device=args.device,
                                           similarity_chunk=512, image_representation='legacy_cls')}
    elif args.dataset == 'urban':
        if not args.root:
            raise ValueError('--root is required for urban')
        result = evaluate_urban1k(model, preprocess, root=args.root, batch_size=args.batch_size,
                                  device=args.device)
        result['metrics'] = {'I2T': result.pop('image2text'), 'T2I': result.pop('text2image')}
    else:
        if not args.manifest or not args.image_root:
            raise ValueError('--manifest and --image-root are required for manifest datasets')
        result = evaluate_manifest(model, preprocess, args.manifest, args.image_root,
                                   args.device, args.batch_size)
    result.update(protocol=args.dataset, checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  model=args.model, native_student_only=True, elapsed_seconds=time.time() - started)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
