"""CLI: native-CLS retrieval on Urban-1k (and optionally COCO val2017) for one checkpoint.

    python tools/eval_urban1k_cls.py --checkpoint <path> --label S0@500 [--coco] [--out <json>]

Strict loading rules (no silent skipping):
  * a checkpoint whose payload contains ``completed_steps`` is checked against ``--expect-steps``
    when given, and its state must load with ``strict=True`` (missing/unexpected must be empty);
  * both the full training checkpoints (S0/C0, which carry optimizer state) and the exported bare
    student checkpoints (C1) are accepted;
  * the image representation is exactly ``normalize(model.encode_image(x))`` and the text
    representation ``normalize(model.encode_text(t))``.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys

import torch

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'eval', 'retrieval'))
sys.path.insert(0, os.path.join(REPO, 'tools'))

# ``from model import longclip`` -- the repository's LongCLIP; the bare top-level module uses
# relative imports and only resolves from inside the package
from model import longclip  # noqa: E402

from tools.urban1k_retrieval import (DEFAULT_ROOT, dataset_fingerprint,  # noqa: E402
                                     evaluate_urban1k)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_student(path, base_model, device, expect_steps=None):
    """Build the model and load the student state with strict=True.

    Accepted layouts: a training payload with ``model`` (S0/C0/C1 checkpoints), a bare state dict, a
    bare state dict carrying a ``clip.`` prefix (the historic SALU/SmartCLIP layout), and a raw DDP
    ``module.`` prefix. The prefix is stripped only when the remaining keys are a subset of the
    model's own keys, so a wrong checkpoint can never silently "load".
    """
    model, preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                               args=argparse.Namespace())
    target = set(model.state_dict())

    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        state = payload['model']
        meta = {'completed_steps': payload.get('completed_steps'),
                'objective': payload.get('objective'),
                'git_head': payload.get('git_head'),
                'config': payload.get('config')}
    elif isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        state, meta = dict(payload), {}
    else:
        raise ValueError('unsupported checkpoint layout at %s' % path)

    if not set(state) <= target:
        stripped = None
        for prefix in ('module.', 'clip.', 'model.'):
            candidate = {key[len(prefix):]: value for key, value in state.items()
                         if key.startswith(prefix)}
            if candidate and set(candidate) <= target:
                stripped = prefix
                state = candidate
                break
        if stripped is None:
            extra = sorted(set(state) - target)[:5]
            raise SystemExit('%s does not match the model: unexpected keys %s' % (path, extra))
        meta['key_prefix_stripped'] = stripped

    if expect_steps is not None:
        got = meta.get('completed_steps')
        if got is None:
            raise SystemExit('%s carries no completed_steps; cannot assert step %d'
                             % (path, expect_steps))
        if int(got) != int(expect_steps):
            raise SystemExit('%s has completed_steps=%s, expected %d' % (path, got, expect_steps))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if list(missing) or list(unexpected):
        raise SystemExit('strict load failed: missing=%s unexpected=%s' % (missing, unexpected))
    model = model.to(device).eval()
    meta['loaded_tensors'] = len(state)
    meta['missing_keys'] = list(missing)
    meta['unexpected_keys'] = list(unexpected)
    return model, preprocess, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--urban_root', default=DEFAULT_ROOT)
    parser.add_argument('--coco', action='store_true', help='also run the canonical COCO val2017')
    parser.add_argument('--coco_root', default=os.environ.get('COCO_DATA_ROOT'))
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()

    device = parsed.device
    model, preprocess, meta = load_student(parsed.checkpoint, parsed.base_model, device,
                                           parsed.expect_steps)
    output = {
        'label': parsed.label,
        'checkpoint': parsed.checkpoint,
        'checkpoint_sha256': sha256_of(parsed.checkpoint),
        'completed_steps': meta.get('completed_steps'),
        'objective': meta.get('objective'),
        'git_head': meta.get('git_head'),
        'loaded_tensors': meta.get('loaded_tensors'),
        'missing_keys': meta.get('missing_keys'),
        'unexpected_keys': meta.get('unexpected_keys'),
        'image_representation': 'legacy_cls',
        'preprocessing': 'repo reference openai-clip _transform(224) (tensor-identical to LongCLIP)',
        'tokenizer': 'longclip.tokenize(context_length=248, truncate=True)',
        'urban1k_dataset': dataset_fingerprint(parsed.urban_root),
    }
    print('[%s] running Urban-1k' % parsed.label, flush=True)
    output['urban1k'] = evaluate_urban1k(model, preprocess, root=parsed.urban_root,
                                         batch_size=parsed.batch_size, device=device)
    if parsed.coco:
        print('[%s] running COCO val2017 with the canonical evaluator' % parsed.label, flush=True)
        coco_module = importlib.import_module('coco_retrieval')
        # legacy_cls only: the patch_global column is SAID-ExGAP's z_G and needs
        # ``encode_router_input``, which a plain CLIP/SmartCLIP checkpoint does not have. The
        # native-CLS column is exactly what C1 must be judged on anyway.
        by_representation = coco_module.evaluate_coco_representations(
            model, preprocess, root=parsed.coco_root, batch_size=parsed.batch_size,
            device=device, global_pool='mean',
            representations=('legacy_cls',))
        output['coco_val2017'] = by_representation['legacy_cls']
        output['coco_val2017_by_representation'] = by_representation
    os.makedirs(os.path.dirname(os.path.abspath(parsed.out)), exist_ok=True)
    with open(parsed.out, 'w') as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print('WROTE %s' % parsed.out)
    print('URBAN1K ' + json.dumps({k: v for k, v in output['urban1k'].items()
                                  if k in ('image2text', 'text2image')}, sort_keys=True))


if __name__ == '__main__':
    main()
