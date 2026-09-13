"""Export the bare native CG-CLIP v0.1 student from a training checkpoint.

CG-CLIP keeps the native CLIP encoder and computes the LAST visual block's CLS row twice: the
``native`` path is the untouched native computation, and the ``conditional`` path re-uses the same
native CLS query, Keys, Values, ``out_proj``, residuals, ``ln_2``/MLP, ``ln_post`` and
``visual.proj`` while renormalising the native CLS attention weights with a 196-d patch gate produced
by ``model/cgclip.py:CaptionGate``. The conditional path is **training only**: it changes the shared
trunk while training and does not exist at inference, so the canonical evaluator must see only the
CLIP student. The caption gate's tensors (``A``, ``B`` and the scalar bias) are therefore *not*
written to the student, and their absence is asserted.

The export is proved lossless at the same precision and through the same native interfaces: one
fixed probe is pushed through the training state and through a freshly built model loaded from the
exported file, and the two outputs are compared exactly (no tolerance).

Two files are produced inside ``--output_dir``:

* ``cgclip_v01_student.pt`` -- ``{'model': clip_state_dict}`` only;
* ``cgclip_v01_student_metadata.json`` -- the provenance record of that student.

The export is deterministic and idempotent: an existing student file is never overwritten silently.
If the new bytes are identical to the existing bytes the export reports ``ALREADY_IDENTICAL`` and
leaves the file alone; if they differ the tool refuses and asks for an explicit decision.

    python tools/diag/export_cgclip_student.py \
        --checkpoint runs_salu/cgclip_v01/CG_CLIP_V01_step000500.pt \
        --output_dir runs_salu/cgclip_v01
"""
import argparse
import datetime
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import longclip                                                    # noqa: E402
from model.cgclip import (ARM, FIXED_SCALE, GATE_BIAS_INIT, GATE_KIND, GATE_KEY_DIM,  # noqa: E402
                          GATE_SEED, OBJECTIVE, PATCH_TOKENS, check_resume_compatible,
                          config_dict, file_sha256, git_head, state_digest, visual_spec)

STUDENT_NAME = 'cgclip_v01_student.pt'
METADATA_NAME = 'cgclip_v01_student_metadata.json'
DEFAULT_BASE_MODEL = 'ViT-B/16'
DEFAULT_EXPECT_STEPS = 500
PROBE_SEED = 20260913
# the training checkpoint keys this exporter relies on (see train/train_cgclip.py:save_checkpoint)
REQUIRED_PAYLOAD_KEYS = ('clip', 'gate_state', 'optimizer_clip', 'optimizer_gate', 'completed_steps',
                         'gate_config', 'gate_state_digest', 'clip_state_digest')
# names that must never appear in the exported student: the conditional path is training only
FORBIDDEN_STUDENT_PREFIXES = ('gate.', 'caption_gate.', 'gate_state.', 'preproj_gate.', 'mask_gate.')
ARM_ALIASES = {'B16': 'ViT-B/16', 'B32': 'ViT-B/32', 'L14': 'ViT-L/14'}
# the caption-gate description, rebuilt here from the module constants so that the exporter does not
# have to instantiate the training-only gate at all
GATE_DESCRIPTOR = {
    'gate_kind': GATE_KIND, 'gate_key_dim': GATE_KEY_DIM, 'gate_seed': GATE_SEED,
    'gate_bias_init': GATE_BIAS_INIT,
    'gate_query_input': 'Normalize(t_raw) detached',
    'gate_key_input': 'U = last.ln_1(X11) patches, detached',
    'gate_forward': 'hard straight-through: hard = (p >= 0.5), m = hard + (p - p.detach())',
    'gate_patches': PATCH_TOKENS,
    'cls_self_gate': 'fixed 1, not trainable, not in the sparse term',
    'gate_query_init': 'A.weight = 0 (no bias)',
    'gate_key_init': 'B.weight = xavier_uniform (no bias)',
    'gate_bias': 'single trainable scalar initialised to log(8)',
    'shared_over_heads': True, 'top_k': None, 'soft_floor': None,
}


def sha256_of(path):
    """sha256 of a whole file, streamed (same rule as ``model/cgclip.py:file_sha256``)."""
    return file_sha256(path)


def read_bytes(path):
    with open(path, 'rb') as handle:
        return handle.read()


def fixed_probe(model):
    """Deterministic inputs for the lossless-export comparison (same dtype, native interfaces).

    Only ``encode_image`` / ``encode_text`` are used: those are the inference interfaces that the
    evaluator calls, and they are the same for both the training state and the reloaded student.
    """
    generator = torch.Generator().manual_seed(PROBE_SEED)
    images = torch.randn(2, 3, 224, 224, generator=generator, dtype=torch.float32)
    texts = longclip.tokenize(['a photo of a cat .', 'a dog on the grass .'], truncate=True)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    return image_features, text_features


def verify_payload(payload, checkpoint, expect_steps, expect_arm):
    """Refuse anything that is not a complete, finished CG-CLIP v0.1 checkpoint."""
    if not isinstance(payload, dict):
        raise SystemExit('checkpoint %s is not a dict payload (got %s)'
                         % (checkpoint, type(payload).__name__))
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise SystemExit('checkpoint %s is not a CG-CLIP payload: missing keys %r'
                         % (checkpoint, missing))
    problems = []
    if payload.get('objective') != OBJECTIVE:
        problems.append('objective=%r expected %r' % (payload.get('objective'), OBJECTIVE))
    if expect_arm is not None and payload.get('arm') != expect_arm:
        problems.append('arm=%r expected %r' % (payload.get('arm'), expect_arm))
    if payload.get('gate_kind') != GATE_KIND:
        problems.append('gate_kind=%r expected %r' % (payload.get('gate_kind'), GATE_KIND))
    steps = payload.get('completed_steps')
    if expect_steps is not None and int(steps) != int(expect_steps):
        problems.append('completed_steps=%r expected %d (the single CG-CLIP v0.1 configuration runs '
                        'exactly %d optimizer updates)' % (steps, int(expect_steps),
                                                           int(expect_steps)))
    try:
        check_resume_compatible(payload, {})
    except SystemExit as error:
        problems.append(str(error))
    if problems:
        raise SystemExit('checkpoint verification failed for %s: %s' % (checkpoint, problems))
    if not isinstance(payload['clip'], dict) or not payload['clip']:
        raise SystemExit('checkpoint %s carries an empty clip state' % checkpoint)
    if not isinstance(payload['gate_state'], dict) or not payload['gate_state']:
        raise SystemExit('checkpoint %s carries an empty gate_state (the caption gate tensors went '
                         'missing; the training-only gate cannot be reproduced from it)' % checkpoint)


def verify_state(payload, checkpoint):
    """Check the state dict completeness and the recomputed clip digest of the source state."""
    state = payload['clip']
    forbidden = [key for key in state if key.startswith(FORBIDDEN_STUDENT_PREFIXES)
                 or 'gate' in key.lower()]
    if forbidden:
        raise SystemExit('the CG-CLIP student state must not contain gate tensors: %r'
                         % sorted(forbidden)[:5])
    recomputed = state_digest(state)
    stored = payload.get('clip_state_digest')
    if stored != recomputed:
        raise SystemExit('clip_state_digest mismatch for %s: stored=%r recomputed=%r (the state was '
                         'modified, partially written or written by another code path)'
                         % (checkpoint, stored, recomputed))
    return recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='CG-CLIP v0.1 training checkpoint (clip + gate_state + both optimizers)')
    parser.add_argument('--output_dir', required=True,
                        help='directory that receives %s and %s' % (STUDENT_NAME, METADATA_NAME))
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL,
                        help='native CLIP architecture to rebuild: %s (the trainer also accepts the '
                             'short alias B16)' % DEFAULT_BASE_MODEL)
    parser.add_argument('--expect-steps', type=int, default=DEFAULT_EXPECT_STEPS,
                        help='required completed_steps value; the single CG-CLIP v0.1 configuration '
                             'is %d optimizer updates' % DEFAULT_EXPECT_STEPS)
    parser.add_argument('--expect-arm', default=ARM, help='required arm value')
    parser.add_argument('--device', default='cpu',
                        help='device used to build and probe the student; the exported tensors are '
                             'always the checkpoint tensors, untouched')
    parsed = parser.parse_args()

    checkpoint = os.path.abspath(parsed.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit('checkpoint not found: %s' % checkpoint)
    output_dir = os.path.abspath(parsed.output_dir)
    base_model = ARM_ALIASES.get(parsed.base_model, parsed.base_model)
    output_path = os.path.join(output_dir, STUDENT_NAME)
    metadata_path = os.path.join(output_dir, METADATA_NAME)

    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verify_payload(payload, checkpoint, parsed.expect_steps, parsed.expect_arm)
    state = payload['clip']
    clip_digest = verify_state(payload, checkpoint)
    source_sha256 = sha256_of(checkpoint)

    model, _preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                args=argparse.Namespace())
    native_keys = set(model.state_dict())
    missing_keys = sorted(set(state) - native_keys)
    absent_keys = sorted(native_keys - set(state))
    if missing_keys or absent_keys:
        raise SystemExit('the checkpoint clip state is not the native CLIP state: keys missing from '
                         'the model %r, model tensors missing from the state %r'
                         % (missing_keys[:5], absent_keys[:5]))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if list(missing) or list(unexpected):
        raise SystemExit('strict load reported missing=%r unexpected=%r'
                         % (list(missing), list(unexpected)))
    spec = visual_spec(model)
    gate_descriptor = dict(GATE_DESCRIPTOR)
    print('STRICT_LOAD_OK missing=%s unexpected=%s' % (list(missing), list(unexpected)))
    print('STUDENT_TENSORS %d NATIVE_KEY_SET_MATCH %s'
          % (len(state), set(state) == native_keys))
    print('CLIP_STATE_DIGEST %s GATE_STATE_DIGEST %s'
          % (clip_digest, payload.get('gate_state_digest')))
    print('GATE_ABSENT_FROM_STUDENT %s (training-only conditional path)'
          % (not [key for key in state if 'gate' in key.lower()]))

    device = torch.device(parsed.device)
    model = model.to(device)
    model.eval()
    before_image, before_text = fixed_probe(model)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path + '.tmp', 'wb') as handle:
        torch.save({'model': state}, handle)
    new_bytes = read_bytes(output_path + '.tmp')
    if os.path.exists(output_path):
        old_bytes = read_bytes(output_path)
        if old_bytes == new_bytes:
            os.remove(output_path + '.tmp')
            print('ALREADY_IDENTICAL %s (byte-identical export, the existing file is left untouched)'
                  % output_path)
        else:
            os.remove(output_path + '.tmp')
            raise SystemExit('refusing to overwrite %s: the existing file is not byte-identical to '
                             'the export of %s (remove it explicitly to re-export)'
                             % (output_path, checkpoint))
    else:
        os.replace(output_path + '.tmp', output_path)
        print('EXPORTED %s' % output_path)
    student_sha256 = sha256_of(output_path)

    reloaded, _preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                    args=argparse.Namespace())
    reloaded.load_state_dict(torch.load(output_path, map_location='cpu',
                                        weights_only=False)['model'], strict=True)
    reloaded = reloaded.to(device).eval()
    after_image, after_text = fixed_probe(reloaded)
    image_diff = float((before_image - after_image).abs().max())
    text_diff = float((before_text - after_text).abs().max())
    print('LOSSLESS_EXPORT image_max_abs_diff=%r text_max_abs_diff=%r' % (image_diff, text_diff))
    if image_diff != 0.0 or text_diff != 0.0:
        raise SystemExit('the exported student does not reproduce the training state outputs')

    metadata = {
        'student_file': os.path.basename(output_path),
        'student_sha256': student_sha256,
        'student_tensor_count': len(state),
        'student_key': 'model',
        'student_scope': 'visual + text CLIP weights only (native inference path)',
        'arm': ARM,
        'objective': OBJECTIVE,
        'gate_kind': GATE_KIND,
        'base_model': base_model,
        'source_checkpoint': checkpoint,
        'source_checkpoint_sha256': source_sha256,
        'source_completed_steps': int(payload['completed_steps']),
        'clip_state_digest': clip_digest,
        'gate_state_digest': payload.get('gate_state_digest'),
        'source_phase': payload.get('phase'),
        'source_rank': payload.get('rank'),
        'source_world_size': payload.get('world_size'),
        'source_batch_size_per_gpu': payload.get('batch_size_per_gpu'),
        'source_loss_weights': payload.get('loss_weights'),
        'source_git_head': payload.get('git_head'),
        'export_git_head': git_head(REPO),
        'exported_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'export_probe': {
            'kind': 'fixed deterministic probe through encode_image / encode_text',
            'seed': PROBE_SEED,
            'image_max_abs_diff': image_diff,
            'text_max_abs_diff': text_diff,
            'identical': bool(image_diff == 0.0 and text_diff == 0.0),
            'scope': 'proves the exported student reproduces the training state through the native '
                     'inference interfaces, at the same precision, with no tolerance',
        },
        'logit_scale': {
            'present': 'logit_scale' in state,
            'used': False,
            'note': 'the CG-CLIP v0.1 score scale is the fixed %.1f; logit_scale is present in the '
                    'native CLIP key set but is never read by the objective or by the evaluator'
                    % FIXED_SCALE,
        },
        'mask_net': {
            'present': sorted(key for key in state if key.startswith('mask_net.')) != [],
            'used': False,
            'note': 'the native CLIP module carries a legacy mask_net whose tensors are part of the '
                    'native state dict; it is frozen (excluded from both optimizer groups) and is '
                    'never called by encode_image / encode_text, so it is carried unchanged for key '
                    'set fidelity and has no effect at inference',
        },
        'conditional_path': {
            'present_in_student': False,
            'reason': 'the caption-gated conditional CLS path is TRAINING ONLY: it changes the shared '
                      'trunk while training and does not exist at inference, so the caption gate '
                      'tensors (A, B and the scalar bias) are intentionally absent from the student',
            'gate_config': gate_descriptor,
            'gate_state_digest_not_in_student': payload.get('gate_state_digest'),
        },
        'visual_spec': spec,
        'config_identity': config_dict(),
        'not_run': ['the conditional (caption-gated) CLS path', 'any optimizer update',
                    'any evaluation or retrieval metric'],
    }
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print('EXPORT_CGCLIP_STUDENT ' + json.dumps({
        'arm': ARM,
        'student_file': output_path,
        'student_sha256': student_sha256,
        'student_tensor_count': len(state),
        'metadata_file': metadata_path,
        'source_checkpoint': checkpoint,
        'source_checkpoint_sha256': source_sha256,
        'completed_steps': int(payload['completed_steps']),
        'clip_state_digest': clip_digest,
        'gate_state_digest': payload.get('gate_state_digest'),
        'gate_in_student': False,
        'conditional_path_in_student': False,
        'lossless_image_max_abs_diff': image_diff,
        'lossless_text_max_abs_diff': text_diff,
        'logit_scale_present_unused': 'logit_scale' in state,
        'new_optimizer_updates': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
