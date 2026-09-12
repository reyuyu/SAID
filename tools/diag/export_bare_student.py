"""Bare-student export with strict verification for any SAID checkpoint that stores ``model``.

Used by the S0 continuation: the canonical and Urban evaluators must load ONLY the student, so the
export writes ``payload['model']`` on its own, proves it loads with ``strict=True`` (no
``strict=False`` that could skip every tensor) and proves the native ``encode_image`` /
``encode_text`` outputs are bit-identical to the checkpoint it came from.
"""
import argparse
import hashlib
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from model import longclip  # noqa: E402

TOKENS = 248


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_outputs(model, device='cpu'):
    generator = torch.Generator().manual_seed(1234)
    images = torch.randn(2, 3, 224, 224, generator=generator)
    tokens = longclip.tokenize(['a photo of a cat .', 'a red bus downtown .'], truncate=True)
    model.eval()
    with torch.no_grad():
        v = model.encode_image(images.to(device).type(model.dtype))
        t, hidden = model.encode_text(tokens.to(device), return_full=True)
    return {'image': v.float().cpu(), 'text': t.float().cpu(), 'hidden': hidden.float().cpu()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--expect-objective', default=None)
    parser.add_argument('--expect-arm', default=None)
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    if 'model' not in payload:
        raise SystemExit('the checkpoint does not store a model state dict')
    objective = payload.get('objective')
    arm = (payload.get('config') or {}).get('arm') or payload.get('arm')
    if parsed.expect_objective and objective != parsed.expect_objective:
        raise SystemExit('objective %r != expected %r' % (objective, parsed.expect_objective))
    if parsed.expect_arm and arm != parsed.expect_arm:
        raise SystemExit('arm %r != expected %r' % (arm, parsed.expect_arm))
    got = int(payload.get('completed_steps', -1))
    if parsed.expect_steps is not None and got != parsed.expect_steps:
        raise SystemExit('completed_steps %d != %d' % (got, parsed.expect_steps))
    state = payload['model']
    forbidden = [key for key in state if key.startswith('decoder')
                 or key.startswith('reference_visual') or key.startswith('said_router')
                 or key.startswith('exgap')]
    if forbidden:
        raise SystemExit('the student state contains auxiliary keys: %r' % forbidden[:5])

    model, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    missing, unexpected = model.load_state_dict(state, strict=True)
    print('STRICT_LOAD_OK missing=%s unexpected=%s' % (missing, unexpected))
    if missing or unexpected:
        raise SystemExit('strict load reported missing/unexpected tensors')

    reference, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                           args=argparse.Namespace())
    reference.load_state_dict(state, strict=True)
    before = native_outputs(reference)
    after = native_outputs(model)
    del reference
    for key in before:
        difference = float((before[key] - after[key]).abs().max())
        print('NATIVE_OUTPUT_DELTA %s %.3e shape=%r' % (key, difference, list(after[key].shape)))
        if difference != 0.0:
            raise SystemExit('native %s output changed across the export' % key)
    if list(after['hidden'].shape[1:]) != [TOKENS, 512]:
        raise SystemExit('unexpected text hidden shape %r' % (list(after['hidden'].shape),))

    os.makedirs(os.path.dirname(os.path.abspath(parsed.out)), exist_ok=True)
    torch.save({'model': state,
                'completed_steps': got,
                'objective': objective,
                'arm': arm,
                'config': payload.get('config'),
                'git_head': payload.get('git_head'),
                'initial_state_sha256': payload.get('initial_state_sha256'),
                'resume_parent': payload.get('resume_parent')}, parsed.out)
    print('EXPORTED %s' % parsed.out)
    print('SOURCE_CHECKPOINT_SHA256 %s' % file_sha256(parsed.checkpoint))
    print('EXPORT_SHA256 %s' % file_sha256(parsed.out))
    print('STUDENT_TENSORS %d' % len(state))


if __name__ == '__main__':
    main()
