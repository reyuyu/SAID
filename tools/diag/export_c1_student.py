"""Export a bare student CLIP state dict from a C1 training checkpoint, with strict verification.

The training checkpoint deliberately contains auxiliary modules (``reconstruction_decoder``, the
three optimizer states, the frozen-reference digest). The canonical evaluator must load ONLY the
student, so this tool writes the student on its own and then proves it loads with
``strict=True`` against a freshly built model -- no silent ``strict=False`` that could skip every
tensor and still "succeed".
"""
import argparse
import hashlib
import os
import sys

import torch

REPO = '/root/SAID-c1-tcr'
sys.path.insert(0, REPO)

from model import longclip  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    if parsed.expect_steps is not None:
        got = int(payload.get('completed_steps', -1))
        assert got == parsed.expect_steps, 'completed_steps %d != %d' % (got, parsed.expect_steps)
    state = payload['model']
    forbidden = [key for key in state if key.startswith('reference_visual')
                 or key.startswith('decoder')]
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
