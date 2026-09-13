"""Export the bare PG-CLIP student (``clip.state_dict()``) from a training checkpoint.

The training checkpoint also holds the new 768-d ``PreProjectionGate`` and both optimizer states; the
canonical evaluator must see only the CLIP student, so the gate is *not* written to the student and
its presence is asserted absent. The export is proved lossless at the same precision and through the
same native interfaces: the same fixed inputs are pushed through the training state and through a
freshly built model loaded from the exported file, and the outputs are compared exactly.
"""
import argparse
import hashlib
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import longclip                                                    # noqa: E402
from model.pgclip import check_resume_compatible, config_dict                 # noqa: E402


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_probe(model):
    """Deterministic inputs for the lossless-export comparison (same dtype, same interfaces)."""
    generator = torch.Generator().manual_seed(20260913)
    images = torch.randn(2, 3, 224, 224, generator=generator, dtype=torch.float32)
    texts = longclip.tokenize(['a photo of a cat .', 'a dog on the grass .'], truncate=True)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    return image_features, text_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--expect-objective', default=None)
    parser.add_argument('--expect-arm', default=None)
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    problems = []
    if parsed.expect_steps is not None and int(payload.get('completed_steps', -1)) != parsed.expect_steps:
        problems.append('completed_steps=%r expected %d' % (payload.get('completed_steps'),
                                                            parsed.expect_steps))
    if parsed.expect_objective is not None and payload.get('objective') != parsed.expect_objective:
        problems.append('objective=%r expected %r' % (payload.get('objective'),
                                                      parsed.expect_objective))
    if parsed.expect_arm is not None and payload.get('arm') != parsed.expect_arm:
        problems.append('arm=%r expected %r' % (payload.get('arm'), parsed.expect_arm))
    try:
        check_resume_compatible(payload, {})
    except SystemExit as error:
        problems.append(str(error))
    if problems:
        raise SystemExit('checkpoint verification failed: %s' % problems)
    if 'gate' not in payload or 'clip' not in payload:
        raise SystemExit('checkpoint layout is not a PG-CLIP payload (needs clip + gate)')

    state = payload['clip']
    gate_in_student = [key for key in state
                       if key.startswith('gate.') or key.startswith('preproj_gate.')]
    if gate_in_student:
        raise SystemExit('the student state must not contain the gate: %r' % gate_in_student[:5])

    model, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    missing, unexpected = model.load_state_dict(state, strict=True)
    print('STRICT_LOAD_OK missing=%s unexpected=%s' % (list(missing), list(unexpected)))
    native_keys = set(model.state_dict())
    print('STUDENT_TENSORS %d NATIVE_KEY_SET_MATCH %s'
          % (len(state), set(state) == native_keys))
    if set(state) != native_keys:
        raise SystemExit('the student key set is not the native CLIP key set: %r'
                         % sorted(set(state) ^ native_keys)[:5])

    model.eval()
    before_image, before_text = fixed_probe(model)

    os.makedirs(os.path.dirname(os.path.abspath(parsed.out)), exist_ok=True)
    torch.save({'model': state,
                'completed_steps': payload.get('completed_steps'),
                'objective': payload.get('objective'),
                'arm': payload.get('arm'),
                'phase': payload.get('phase'),
                'loss_weights': payload.get('loss_weights'),
                'gate': payload.get('gate'),
                'config': payload.get('config'),
                'git_head': payload.get('git_head'),
                'digests': payload.get('digests')}, parsed.out)
    print('EXPORTED %s' % parsed.out)

    reloaded, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                          args=argparse.Namespace())
    reloaded.load_state_dict(torch.load(parsed.out, map_location='cpu',
                                        weights_only=False)['model'], strict=True)
    reloaded.eval()
    after_image, after_text = fixed_probe(reloaded)
    image_diff = float((before_image - after_image).abs().max())
    text_diff = float((before_text - after_text).abs().max())
    print('LOSSLESS_EXPORT image_max_abs_diff=%r text_max_abs_diff=%r' % (image_diff, text_diff))
    if image_diff != 0.0 or text_diff != 0.0:
        raise SystemExit('the exported student does not reproduce the training state outputs')
    print('EXPORT_SHA256 %s' % sha256_of(parsed.out))


if __name__ == '__main__':
    main()
