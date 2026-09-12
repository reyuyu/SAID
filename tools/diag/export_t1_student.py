"""Export a bare student CLIP state dict from a SAID-Token v1 checkpoint, with strict verification.

The training checkpoint deliberately contains auxiliary modules (the two aggregators, the router, the
decoder, three optimizer states, the frozen-reference digest). The canonical evaluator must load ONLY
the student -- the native CLS is the judge -- so this tool writes the student on its own and then
proves it loads with ``strict=True`` against a freshly built model. No silent ``strict=False`` that
could skip every tensor and still "succeed".
"""
import argparse
import hashlib
import os
import sys

import torch

REPO = '/root/SAID-token-v1'
sys.path.insert(0, REPO)

from model import longclip  # noqa: E402

AUXILIARY_PREFIXES = ('image_aggregator', 'text_aggregator', 'router', 'decoder',
                      'reference_visual')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--expect-version', default=None)
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    if parsed.expect_steps is not None:
        got = int(payload.get('completed_steps', -1))
        if got != parsed.expect_steps:
            raise ValueError('completed_steps %d != %d' % (got, parsed.expect_steps))
    if parsed.expect_version is not None:
        got = payload.get('config', {}).get('implementation_version')
        if got != parsed.expect_version:
            raise ValueError('implementation_version %r != %r' % (got, parsed.expect_version))
    state = payload['model']
    forbidden = [key for key in state if key.startswith(AUXILIARY_PREFIXES)]
    if forbidden:
        raise SystemExit('the student state contains auxiliary keys: %r' % forbidden[:5])

    model, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    missing, unexpected = model.load_state_dict(state, strict=True)
    print('STRICT_LOAD_OK missing=%s unexpected=%s' % (missing, unexpected))

    os.makedirs(os.path.dirname(os.path.abspath(parsed.out)), exist_ok=True)
    torch.save({'model': state,
                'completed_steps': payload.get('completed_steps'),
                'objective': payload.get('objective'),
                'arm': payload.get('arm'),
                'config': payload.get('config'),
                'git_head': payload.get('git_head'),
                'reference_visual_state_sha256': payload.get('reference_visual_state_sha256')},
               parsed.out)
    with open(parsed.out, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print('EXPORTED %s' % parsed.out)
    print('EXPORT_SHA256 %s' % digest)
    print('STUDENT_TENSORS %d' % len(state))


if __name__ == '__main__':
    main()
