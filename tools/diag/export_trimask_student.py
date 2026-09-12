"""Export the bare student CLIP state from an S0-TriMask checkpoint, with strict verification.

The training checkpoint deliberately carries the new text branch and the two optimizer states. The
canonical evaluator must load ONLY the student, so this tool writes the student on its own, proves
it loads with ``strict=True`` against a freshly built model (no ``strict=False`` that could skip
every tensor), and additionally proves that the exported student produces bit-identical native
``encode_image`` / ``encode_text`` outputs to the checkpoint it came from.
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import longclip  # noqa: E402
from model.said_trimask import ARM, OBJECTIVE, student_state_from_checkpoint  # noqa: E402

TOKENS = 248


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_outputs(model, device='cpu'):
    """Native CLS/EOS outputs on a fixed dummy input, in fp32, no grad."""
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
    parser.add_argument('--expect-steps', type=int, default=500)
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('objective') != OBJECTIVE or payload.get('arm') != ARM:
        raise SystemExit('not an S0-TriMask checkpoint: objective=%r arm=%r'
                         % (payload.get('objective'), payload.get('arm')))
    got = int(payload.get('completed_steps', -1))
    if parsed.expect_steps is not None and got != parsed.expect_steps:
        raise SystemExit('completed_steps %d != %d' % (got, parsed.expect_steps))
    state = student_state_from_checkpoint(payload['model'])
    forbidden = [key for key in state if key.startswith('text_mask_net')
                 or key.startswith('text_gate_projection') or key.startswith('decoder')
                 or key.startswith('reference_visual')]
    if forbidden:
        raise SystemExit('the student state contains auxiliary keys: %r' % forbidden[:5])
    if 'text_mask_net' not in payload:
        raise SystemExit('the training checkpoint is missing the text branch state')

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
                'completed_steps': payload.get('completed_steps'),
                'objective': payload.get('objective'),
                'arm': ARM,
                'config': payload.get('config'),
                'git_head': payload.get('git_head'),
                'initial_state_sha256': payload.get('initial_state_sha256')},
               parsed.out)
    print('EXPORTED %s' % parsed.out)
    print('SOURCE_CHECKPOINT_SHA256 %s' % file_sha256(parsed.checkpoint))
    print('EXPORT_SHA256 %s' % file_sha256(parsed.out))
    print('STUDENT_TENSORS %d' % len(state))


if __name__ == '__main__':
    main()
