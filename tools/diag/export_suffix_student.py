"""Export the bare native student from an S0-Suffix v0.1 training checkpoint.

S0-Suffix v0.1 keeps the native CLIP encoder as the whole inference path and adds a *training only*
suffix-alignment branch (``model/...SuffixMask``: ``layer1`` 1024->512, ``layer2`` 512->512, with
biases). That branch changes the shared trunk while training, exactly like the CG-CLIP caption gate,
and it does not exist at inference. The frozen evaluators therefore must see ONLY the native CLIP
state:

* the native CLIP tensors are carried unchanged, including the pre-existing ``mask_net`` tensors,
  which are part of the native CLIP key set and are never called by ``encode_image`` /
  ``encode_text``;
* the NEW suffix mask tensors are ABSENT. Their absence is asserted by prefix and by the exact key
  set, so a suffix tensor can never leak into a student;
* ``completed_steps`` travels with the student because the frozen evaluators assert the step count
  of the checkpoint they are handed.

The export is proved lossless at the same precision and through the same native interfaces: one
fixed deterministic probe is pushed through the training clip state and through a freshly built
model loaded from the exported file, and the two outputs are compared exactly (no tolerance).

The schema of the training checkpoint is pinned here (this file never imports the training code, so a
half-written trainer cannot break the export):

    clip_state / clip              non-empty dict of tensors, the native CLIP state (written to the student)
    suffix_mask_state              suffix-mask tensor dict for S0_SUFFIX_MASK, explicitly None for
                                   S0_SUFFIX_NATIVE
    suffix_mask_config             non-empty DESCRIPTOR dict of the suffix mask module; a tensors dict
                                   or the same key as ``suffix_mask_state`` is refused
    clip_state_digest              sha256 of ``clip_state`` (sha256 over sorted keys)
    suffix_mask_state_digest       sha256 of ``suffix_mask_state``
    completed_steps                must equal ``--expect-steps`` (500)
    objective                      must equal ``said_prefix_suffix_v01``
    arm                            must equal ``--expect-arm`` when that flag is given

Two files are produced inside ``--output_dir``:

* ``s0_suffix_student_step000500.pt`` -- ``{'model': clip_state, 'completed_steps': 500}``;
* ``s0_suffix_student_step000500_metadata.json`` -- the provenance record of that student.

The export is deterministic and idempotent: an existing student file is never overwritten silently.
If the new bytes are identical to the existing bytes the export reports ``ALREADY_IDENTICAL`` and
leaves the file alone; if they differ the tool refuses and asks for an explicit decision.

    python tools/diag/export_suffix_student.py \
        --checkpoint /root/SAID-s0-suffix-v01/runs_salu/s0_suffix_mask/S0_SUFFIX_MASK_step000500.pt \
        --output_dir /root/SAID-s0-suffix-v01/runs_salu/s0_suffix_mask/student_export
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import longclip                                                    # noqa: E402

# --------------------------------------------------------------------------- experiment identity
OBJECTIVE = 'said_prefix_suffix_v01'
ARM_NATIVE = 'S0_SUFFIX_NATIVE'
ARM_MASK = 'S0_SUFFIX_MASK'
ARMS = (ARM_NATIVE, ARM_MASK)

STUDENT_NAME = 's0_suffix_student_step000500.pt'
METADATA_NAME = 's0_suffix_student_step000500_metadata.json'
DEFAULT_BASE_MODEL = 'ViT-B/16'
DEFAULT_EXPECT_STEPS = 500
PROBE_SEED = 20260913

# the checkpoint key names this exporter reads, in resolution order; the first name that is present
# wins and the name actually used is recorded in the metadata and in the JSON line
CLIP_STATE_KEYS = ('clip_state', 'clip', 'native_clip_state', 'clip_state_dict')
SUFFIX_MASK_STATE_KEYS = ('suffix_mask_state', 'suffix_state', 'suffix_mask_tensors')
SUFFIX_MASK_CONFIG_KEYS = ('suffix_mask_config', 'suffix_config', 'suffix_mask_descriptor')
CLIP_STATE_DIGEST_KEYS = ('clip_state_digest', 'clip_digest')
SUFFIX_MASK_DIGEST_KEYS = ('suffix_mask_state_digest', 'suffix_mask_digest')
# keys a checkpoint must carry for this exporter to trust it as a finished S0-Suffix v0.1 checkpoint
REQUIRED_PAYLOAD_KEYS = ('completed_steps', 'objective', 'arm', 'suffix_mask_config',
                         'optimizer_clip', 'optimizer_mask')

# the suffix mask module of S0-Suffix v0.1: two linear layers, 1024 -> 512 -> 512
SUFFIX_MASK_TENSOR_SHAPES = {'layer1.weight': (512, 1024), 'layer1.bias': (512,),
                             'layer2.weight': (512, 512), 'layer2.bias': (512,)}
SUFFIX_MASK_TENSOR_NAMES = tuple(sorted(SUFFIX_MASK_TENSOR_SHAPES))
# any student key carrying one of these prefixes would mean training-only tensors leaked into the
# student; the student key set is additionally compared against the native CLIP key set exactly
FORBIDDEN_STUDENT_PREFIXES = ('suffix_mask.', 'suffix_mask', 'suffix.', 'suffix_', 'prefix_suffix.',
                              'suffix_branch.', 'suffix_projection.')
# every top-level key the student file may carry: nothing else is written, nothing else is accepted
STUDENT_PAYLOAD_KEYS = ('model', 'completed_steps')
ARM_ALIASES = {'B16': 'ViT-B/16', 'B32': 'ViT-B/32', 'L14': 'ViT-L/14'}


# --------------------------------------------------------------------------- small helpers
def sha256_of(path, block=1 << 20):
    """sha256 of a whole file, streamed."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state_dict):
    """sha256 over sorted keys: key bytes then ``value.detach().float().cpu().numpy().tobytes()``.

    This is the same rule the trainer uses and the same rule the runner's verify stage recomputes, so
    the exporter proves it wrote the tensors it read.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('state_digest needs a non-empty dict')
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise ValueError('state_digest entry %r is not a tensor (%s)'
                             % (key, type(value).__name__))
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def git_head(repo=None):
    """The repository HEAD, or ``'unknown'``: never fatal, only a provenance field."""
    try:
        result = subprocess.run(['git', '-C', repo or REPO, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return result.stdout.strip() or 'unknown'


def resolve_key(payload, candidates, what):
    """The first present key of ``candidates``; a hard refusal when none is present."""
    for name in candidates:
        if name in payload:
            return name, payload[name]
    raise SystemExit('the checkpoint carries none of the %s key(s) %r; keys present: %r'
                     % (what, list(candidates), sorted(str(key) for key in payload)))


def describe_config(value):
    """A description dict must be a non-empty dict and must not smuggle tensors in."""
    if not isinstance(value, dict) or not value:
        return 'it is empty or not a dict (%s)' % type(value).__name__
    tensor_keys = sorted(str(key) for key, item in value.items() if torch.is_tensor(item))
    if tensor_keys:
        return ('it carries tensor entries %r: the description dict and the tensor dict are separate '
                'keys and must never be merged into one' % tensor_keys[:5])
    return None


def scalar_subset(value):
    """The JSON-safe subset of a description dict (nested structures are described, not copied)."""
    subset = {}
    for key in sorted(value):
        item = value[key]
        if isinstance(item, (str, int, float, bool)) or item is None:
            subset[key] = item
        else:
            subset[key] = '<%s, not copied>' % type(item).__name__
    return subset


# --------------------------------------------------------------------------- verification
def verify_payload(payload, checkpoint, expect_steps, expect_arm):
    """Refuse anything that is not a complete, finished S0-Suffix v0.1 checkpoint.

    Returns the resolved key names so the caller never guesses twice.
    """
    if not isinstance(payload, dict):
        raise SystemExit('checkpoint %s is not a dict payload (got %s)'
                         % (checkpoint, type(payload).__name__))
    problems = []
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        problems.append('missing top-level key(s) %r' % missing)
    if payload.get('objective') != OBJECTIVE:
        problems.append('objective=%r expected %r' % (payload.get('objective'), OBJECTIVE))
    if expect_arm is not None and payload.get('arm') != expect_arm:
        problems.append('arm=%r expected %r' % (payload.get('arm'), expect_arm))
    elif expect_arm is None and payload.get('arm') not in ARMS:
        problems.append('arm=%r is not one of %r' % (payload.get('arm'), list(ARMS)))
    try:
        completed = int(payload.get('completed_steps'))
    except (TypeError, ValueError):
        completed = None
    if completed is None:
        problems.append('completed_steps=%r is not an integer' % (payload.get('completed_steps'),))
    elif expect_steps is not None and completed != int(expect_steps):
        problems.append('completed_steps=%r expected %d: S0-Suffix v0.1 runs exactly %d optimizer '
                        'updates and only a finished checkpoint may be exported'
                        % (payload.get('completed_steps'), int(expect_steps), int(expect_steps)))
    if problems:
        raise SystemExit('checkpoint verification failed for %s: %s' % (checkpoint, problems))
    return completed


def verify_state(payload, checkpoint, arm, clip_key, mask_key):
    """Check the two state dicts, prove they are separate keys, and recompute both digests."""
    if clip_key == mask_key:
        raise SystemExit('the clip state and the suffix mask state are the SAME checkpoint key (%r): '
                         'the tensors and the description must live under distinct keys, a key '
                         'collision of exactly this kind already destroyed one experiment'
                         % clip_key)
    state = payload[clip_key]
    if not isinstance(state, dict) or not state:
        raise SystemExit('checkpoint %s carries an empty clip state (%s)'
                         % (checkpoint, type(state).__name__))
    non_tensors = sorted(str(key) for key, value in state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise SystemExit('the clip state of %s carries non-tensor entries %r'
                         % (checkpoint, non_tensors[:6]))
    leaked = sorted(key for key in state if key.startswith(FORBIDDEN_STUDENT_PREFIXES))
    if leaked:
        raise SystemExit('the clip state of %s already carries suffix-alignment tensors %r: the '
                         'suffix branch is training only and must be stored under %r, never inside '
                         'the native CLIP state' % (checkpoint, leaked[:6], mask_key))

    mask_state = payload.get(mask_key)
    if arm == ARM_NATIVE:
        if mask_state is not None:
            raise SystemExit('arm %s must carry suffix_mask_state=None; found %s instead (a native '
                             'arm has no suffix module, and a populated mask state here would mean '
                             'the checkpoint is not the arm it claims to be)'
                             % (ARM_NATIVE, type(mask_state).__name__))
    else:
        if not isinstance(mask_state, dict) or not mask_state:
            raise SystemExit('arm %s must carry a non-empty suffix mask state under %r; found %s'
                             % (arm, mask_key, type(mask_state).__name__))
        non_tensors = sorted(str(key) for key, value in mask_state.items()
                             if not torch.is_tensor(value))
        if non_tensors:
            raise SystemExit('the suffix mask state carries non-tensor entries %r' % non_tensors[:6])
        shapes = {str(key).split('.', 1)[-1]: tuple(value.shape)
                  for key, value in mask_state.items()}
        wrong = {name: list(shape) for name, shape in sorted(shapes.items())
                 if SUFFIX_MASK_TENSOR_SHAPES.get(name) != shape}
        absent = [name for name in SUFFIX_MASK_TENSOR_NAMES if name not in shapes]
        if wrong or absent:
            raise SystemExit('the suffix mask state does not match the suffix mask module: absent %r, '
                             'wrong shapes %r (expected %r)'
                             % (absent, wrong, {name: list(shape) for name, shape in
                                                sorted(SUFFIX_MASK_TENSOR_SHAPES.items())}))

    clip_digest = state_digest(state)
    recorded_clip = payload.get('clip_state_digest')
    if recorded_clip is None:
        _name, recorded_clip = resolve_key(payload, CLIP_STATE_DIGEST_KEYS,
                                           'clip-state digest')
    if not isinstance(recorded_clip, str) or recorded_clip.lower() != clip_digest.lower():
        raise SystemExit('clip_state_digest mismatch for %s: stored=%r recomputed=%r (the state was '
                         'modified, partially written or written by another code path)'
                         % (checkpoint, recorded_clip, clip_digest))

    mask_digest = None
    if arm == ARM_NATIVE:
        # the native arm stores no suffix tensors, so there is nothing to digest; a recorded digest
        # for a state that is explicitly None would be a contradiction, so it is reported as such
        _name, recorded_mask = resolve_key(payload, SUFFIX_MASK_DIGEST_KEYS, 'suffix-mask digest')
        if recorded_mask is not None:
            raise SystemExit('arm %s records %s=%r while suffix_mask_state is None; the flat arm '
                             'must not claim a suffix digest' % (ARM_NATIVE, _name, recorded_mask))
    else:
        mask_digest = state_digest(mask_state)
        _name, recorded_mask = resolve_key(payload, SUFFIX_MASK_DIGEST_KEYS, 'suffix-mask digest')
        if not isinstance(recorded_mask, str) or recorded_mask.lower() != mask_digest.lower():
            raise SystemExit('suffix_mask_state_digest mismatch for %s: stored=%r recomputed=%r'
                             % (checkpoint, recorded_mask, mask_digest))
    return clip_digest, mask_digest


# --------------------------------------------------------------------------- lossless probe
def load_native_model(base_model, state, device):
    """A freshly built native CLIP with ``state`` loaded strictly (no ``strict=False`` anywhere)."""
    model, preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                args=argparse.Namespace())
    native_keys = set(model.state_dict())
    extra = sorted(set(state) - native_keys)
    absent = sorted(native_keys - set(state))
    if extra or absent:
        raise SystemExit('the checkpoint clip state is not the native CLIP state: keys missing from '
                         'the model %r, model tensors missing from the state %r'
                         % (extra[:5], absent[:5]))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if list(missing) or list(unexpected):
        raise SystemExit('strict load reported missing=%r unexpected=%r'
                         % (list(missing), list(unexpected)))
    return model.to(device).eval(), preprocess, native_keys


def fixed_probe(model):
    """Deterministic inputs for the lossless-export comparison (same dtype, native interfaces).

    Only ``encode_image`` / ``encode_text`` are used: those are the inference interfaces the frozen
    evaluators call, and they are identical for the training state and the reloaded student.
    """
    generator = torch.Generator().manual_seed(PROBE_SEED)
    images = torch.randn(2, 3, 224, 224, generator=generator, dtype=torch.float32)
    texts = longclip.tokenize(['a photo of a cat .', 'a dog on the grass .'], truncate=True)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    return image_features, text_features


def read_bytes(path):
    with open(path, 'rb') as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='S0-Suffix v0.1 training checkpoint (native CLIP state + suffix mask '
                             'state + optimizers + completed_steps)')
    parser.add_argument('--output_dir', required=True,
                        help='directory that receives %s and %s' % (STUDENT_NAME, METADATA_NAME))
    parser.add_argument('--student-name', dest='student_name', default=STUDENT_NAME,
                        help='student file name inside --output_dir (default %s)' % STUDENT_NAME)
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL,
                        help='native CLIP architecture to rebuild: %s (the trainer also accepts the '
                             'short alias B16)' % DEFAULT_BASE_MODEL)
    parser.add_argument('--expect-steps', type=int, default=DEFAULT_EXPECT_STEPS,
                        help='required completed_steps value; S0-Suffix v0.1 is %d optimizer updates'
                             % DEFAULT_EXPECT_STEPS)
    parser.add_argument('--expect-arm', default=None,
                        help='required arm value; the default reads the arm from the checkpoint and '
                             'accepts only %s' % '/'.join(ARMS))
    parser.add_argument('--device', default='cpu',
                        help='device used to build and probe the student; the exported tensors are '
                             'always the checkpoint tensors, untouched')
    parsed = parser.parse_args()

    checkpoint = os.path.abspath(parsed.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit('checkpoint not found: %s' % checkpoint)
    output_dir = os.path.abspath(parsed.output_dir)
    base_model = ARM_ALIASES.get(parsed.base_model, parsed.base_model)
    output_path = os.path.join(output_dir, parsed.student_name)
    metadata_name = (METADATA_NAME if parsed.student_name == STUDENT_NAME
                     else os.path.splitext(parsed.student_name)[0] + '_metadata.json')
    metadata_path = os.path.join(output_dir, metadata_name)

    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = verify_payload(payload, checkpoint, parsed.expect_steps, parsed.expect_arm)
    arm = payload.get('arm')
    clip_key, clip_state = resolve_key(payload, CLIP_STATE_KEYS, 'clip state')
    mask_key, _mask_value = resolve_key(payload, SUFFIX_MASK_STATE_KEYS, 'suffix mask state')
    config_key, mask_config = resolve_key(payload, SUFFIX_MASK_CONFIG_KEYS, 'suffix mask config')
    if config_key == mask_key or config_key == clip_key:
        raise SystemExit('the suffix mask DESCRIPTION is stored under %r, which collides with a '
                         'tensor key (%r/%r): the description and the tensors must be separate keys'
                         % (config_key, clip_key, mask_key))
    config_problem = describe_config(mask_config)
    if config_problem:
        raise SystemExit('%s is not a usable description of the suffix mask module: %s'
                         % (config_key, config_problem))
    clip_digest, mask_digest = verify_state(payload, checkpoint, arm, clip_key, mask_key)
    source_sha256 = sha256_of(checkpoint)

    # ---- the student is the native CLIP state, and nothing else ------------------------------
    model, preprocess, native_keys = load_native_model(base_model, clip_state,
                                                       torch.device(parsed.device))
    leaked = sorted(key for key in clip_state if key.startswith(FORBIDDEN_STUDENT_PREFIXES))
    print('STRICT_LOAD_OK native_key_set_match=%s tensors=%d'
          % (set(clip_state) == native_keys, len(clip_state)))
    print('CLIP_STATE_DIGEST %s SUFFIX_MASK_STATE_DIGEST %s' % (clip_digest, mask_digest))
    print('SUFFIX_MASK_ABSENT_FROM_STUDENT %s (training-only branch, keys=%s)'
          % (not leaked, SUFFIX_MASK_TENSOR_NAMES))
    before_image, before_text = fixed_probe(model)
    del model

    student_payload = {'model': clip_state, 'completed_steps': completed}
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path + '.tmp', 'wb') as handle:
        torch.save(student_payload, handle)
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

    # ---- prove the written file reproduces the training state exactly ------------------------
    written = torch.load(output_path, map_location='cpu', weights_only=False)
    if not isinstance(written, dict):
        raise SystemExit('%s did not load back as a dict (%s)'
                         % (output_path, type(written).__name__))
    unexpected_keys = sorted(str(key) for key in written if key not in STUDENT_PAYLOAD_KEYS)
    if unexpected_keys:
        raise SystemExit('the student carries unexpected top-level keys %r; the allowed keys are %r'
                         % (unexpected_keys, list(STUDENT_PAYLOAD_KEYS)))
    if set(written.get('model') or {}) != set(clip_state):
        raise SystemExit('the reloaded student key set differs from the checkpoint clip state')
    if int(written.get('completed_steps', -1)) != completed:
        raise SystemExit('the reloaded student carries completed_steps=%r expected %d'
                         % (written.get('completed_steps'), completed))
    reloaded, _preprocess, _keys = load_native_model(base_model, written['model'],
                                                     torch.device(parsed.device))
    after_image, after_text = fixed_probe(reloaded)
    image_diff = float((before_image - after_image).abs().max())
    text_diff = float((before_text - after_text).abs().max())
    print('LOSSLESS_EXPORT image_max_abs_diff=%r text_max_abs_diff=%r' % (image_diff, text_diff))
    if image_diff != 0.0 or text_diff != 0.0:
        raise SystemExit('the exported student does not reproduce the training state outputs')
    del reloaded

    metadata = {
        'student_file': os.path.basename(output_path),
        'student_path': output_path,
        'student_sha256': student_sha256,
        'student_tensor_count': len(clip_state),
        'student_key': 'model',
        'student_payload_keys': sorted(STUDENT_PAYLOAD_KEYS),
        'student_scope': 'visual + text native CLIP weights only (the inference path)',
        'completed_steps': completed,
        'arm': arm,
        'objective': payload.get('objective'),
        'base_model': base_model,
        'source_checkpoint': checkpoint,
        'source_checkpoint_sha256': source_sha256,
        'source_clip_state_key': clip_key,
        'source_suffix_mask_state_key': mask_key,
        'source_suffix_mask_config_key': config_key,
        'clip_state_digest': clip_digest,
        'suffix_mask_state_digest': mask_digest,
        'source_git_head': payload.get('git_head'),
        'export_git_head': git_head(REPO),
        'exported_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'export_probe': {
            'kind': 'fixed deterministic probe through encode_image / encode_text',
            'seed': PROBE_SEED,
            'image_max_abs_diff': image_diff,
            'text_max_abs_diff': text_diff,
            'identical': bool(image_diff == 0.0 and text_diff == 0.0),
            'scope': 'proves the exported student reproduces the training clip state through the '
                     'native inference interfaces, at the same precision, with no tolerance',
        },
        'suffix_branch': {
            'present_in_student': False,
            'tensor_names_expected_in_checkpoint': list(SUFFIX_MASK_TENSOR_NAMES),
            'tensor_shapes': {name: list(shape)
                              for name, shape in sorted(SUFFIX_MASK_TENSOR_SHAPES.items())},
            'state_present_in_checkpoint': payload.get(mask_key) is not None,
            'state_digest_not_in_student': mask_digest,
            'reason': 'the suffix-alignment branch is TRAINING ONLY: it changes the shared trunk '
                      'while training and does not exist at inference, so its tensors are '
                      'intentionally absent from the student',
        },
        'mask_net': {
            'present': sorted(key for key in clip_state if key.startswith('mask_net.')) != [],
            'used': False,
            'note': 'the native CLIP module carries a legacy mask_net whose tensors are part of the '
                    'native state dict; it is never called by encode_image / encode_text, so it is '
                    'carried unchanged for key-set fidelity and has no effect at inference',
        },
        'config_descriptor': scalar_subset(mask_config),
        'not_run': ['the suffix-alignment branch', 'any optimizer update',
                    'any evaluation or retrieval metric'],
    }
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    print('EXPORT_SUFFIX_STUDENT ' + json.dumps({
        'objective': payload.get('objective'),
        'arm': arm,
        'student_file': output_path,
        'student_sha256': student_sha256,
        'student_tensor_count': len(clip_state),
        'metadata_file': metadata_path,
        'source_checkpoint': checkpoint,
        'source_checkpoint_sha256': source_sha256,
        'completed_steps': completed,
        'clip_state_digest': clip_digest,
        'suffix_mask_state_digest': mask_digest,
        'suffix_mask_in_student': False,
        'suffix_mask_state_was_none': payload.get(mask_key) is None,
        'lossless_image_max_abs_diff': image_diff,
        'lossless_text_max_abs_diff': text_diff,
        'new_optimizer_updates': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
