"""Export a bare student CLIP state from an S0-TriMask / S0-TriMask-HS checkpoint.

The training checkpoint deliberately carries the new text branch and the two optimizer states. The
canonical evaluator must load ONLY the student, so this tool writes the student on its own, proves
it loads with ``strict=True`` against a freshly built model (no ``strict=False`` that could skip
every tensor), and additionally proves that the exported student produces bit-identical native
``encode_image`` / ``encode_text`` outputs to the checkpoint it came from.

Both objectives are accepted explicitly. A v0.2 (hard-gate) checkpoint is *not* relabelled as the
v0.1 objective to satisfy an older exporter: the allowed set and the mode assertion are extended
here instead, and ``--expect-gate-mode`` pins which mode is expected. Older v0.1 checkpoints carry no
``text_gate_mode`` at all, which only the soft mode may accept.
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import longclip  # noqa: E402
from model.said_trimask import (ARM, HARD_GATE, LOSS_PROFILES, LOSS_PROFILE_DEFAULT,  # noqa: E402
                                OBJECTIVE, SOFT_GATE, TEXT_GATE_MODES, all_profile_names,
                                profile_names, student_state_from_checkpoint)

TOKENS = 248
# built from the profile table instead of a hand-written list: every (profile, gate mode) pair the
# trainers can produce is accepted, and nothing else is. A new arm therefore cannot be rejected by
# this tool merely for being new -- but a checkpoint whose arm/objective pair is unknown still is.
EXPECTED_MODES = {}
ALLOWED_OBJECTIVES = []
_OBJECTIVE_TO_ARM = {}
for _gate_mode in TEXT_GATE_MODES:
    for _profile in LOSS_PROFILES:
        try:
            _arm, _objective, _phase = profile_names(_profile, _gate_mode)
        except ValueError:                    # a profile that is not defined for this mode
            continue
        EXPECTED_MODES[_arm] = _gate_mode
        ALLOWED_OBJECTIVES.append(_objective)
        _OBJECTIVE_TO_ARM[_objective] = _arm
ALLOWED_OBJECTIVES = tuple(sorted(set(ALLOWED_OBJECTIVES)))


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
    parser.add_argument('--expect-gate-mode', default=None, choices=list(TEXT_GATE_MODES),
                        help='fail unless the checkpoint records this text-gate mode')
    parser.add_argument('--expect-loss-profile', dest='expect_loss_profile',
                        default=None, choices=list(LOSS_PROFILES),
                        help='fail unless the checkpoint records this loss-weight profile')
    parser.add_argument('--base_model', default='ViT-B/16')
    parsed = parser.parse_args()

    payload = torch.load(parsed.checkpoint, map_location='cpu', weights_only=False)
    objective = payload.get('objective')
    if objective not in ALLOWED_OBJECTIVES:
        raise SystemExit('not an S0-TriMask(-HS) checkpoint: objective=%r (allowed %r)'
                         % (objective, list(ALLOWED_OBJECTIVES)))
    gate_mode = payload.get('text_gate_mode')
    if gate_mode is not None and gate_mode not in TEXT_GATE_MODES:
        raise SystemExit('unknown text_gate_mode %r' % (gate_mode,))
    if gate_mode is None:
        # a v0.1 checkpoint with no recorded mode can only be the soft one
        if objective != OBJECTIVE:
            raise SystemExit('objective %r must record text_gate_mode' % (objective,))
        gate_mode = SOFT_GATE
    recorded_arm = payload.get('arm') or _OBJECTIVE_TO_ARM.get(objective)
    if recorded_arm is None:
        raise SystemExit('the checkpoint records neither arm nor a decidable objective')
    if EXPECTED_MODES.get(recorded_arm) not in (None, gate_mode):
        raise SystemExit('arm %r disagrees with text_gate_mode %r' % (recorded_arm, gate_mode))
    if parsed.expect_gate_mode is not None and gate_mode != parsed.expect_gate_mode:
        raise SystemExit('text_gate_mode %r != expected %r' % (gate_mode, parsed.expect_gate_mode))
    loss_profile = payload.get('loss_profile') or LOSS_PROFILE_DEFAULT
    if loss_profile not in LOSS_PROFILES:
        raise SystemExit('unknown loss_profile %r' % (loss_profile,))
    if (parsed.expect_loss_profile is not None
            and loss_profile != parsed.expect_loss_profile):
        raise SystemExit('loss_profile %r != expected %r'
                         % (loss_profile, parsed.expect_loss_profile))
    got = int(payload.get('completed_steps', payload.get('step', -1)))
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
                'completed_steps': payload.get('completed_steps', payload.get('step')),
                'objective': objective,
                'arm': recorded_arm,
                'text_gate_mode': gate_mode,
                'loss_profile': loss_profile,
                'lambda_sparse_t': payload.get('lambda_sparse_t'),
                'config': payload.get('config'),
                'git_head': payload.get('git_head'),
                'initial_state_sha256': payload.get('initial_state_sha256')},
               parsed.out)
    print('EXPORTED %s' % parsed.out)
    print('GATE_MODE %s' % gate_mode)
    print('LOSS_PROFILE %s' % loss_profile)
    print('SOURCE_CHECKPOINT_SHA256 %s' % file_sha256(parsed.checkpoint))
    print('EXPORT_SHA256 %s' % file_sha256(parsed.out))
    print('STUDENT_TENSORS %d' % len(state))


if __name__ == '__main__':
    main()
