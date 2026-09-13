"""Export the bare native student for S0-GlobalOnly v0.1.

The arm trains the native towers only, so the student is simply the trained CLIP state dict: no mask
network, no extra module, nothing to strip. The tool refuses to run unless the checkpoint is a real
500-update checkpoint and the state digest matches, and it proves the export is lossless by running
a fixed probe through the native encode_image / encode_text interfaces before and after.
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (REPO, os.path.join(REPO, 'train')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
import train_global_only as trainer                                             # noqa: E402

ARM = trainer.ARM
EXPECTED_STEPS = 500


def build_model(device, base_model='ViT-B/16'):
    model, _ = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    return model.to(device)


def probe(model, device, seed=20260913):
    """A fixed deterministic probe through the native inference interfaces."""
    import torch.nn.functional as F
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(2, 3, 224, 224, generator=generator).to(device)
    text = longclip.tokenize(['a photo of a cat', 'a dog on grass'], truncate=True).to(device)
    with torch.no_grad():
        return {'image': model.encode_image(images).float().cpu(),
                'text': model.encode_text(text).float().cpu()}


def main():
    parser = argparse.ArgumentParser(description='export the %s bare student' % ARM)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--expect_steps', type=int, default=EXPECTED_STEPS)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    problems = []
    if not isinstance(payload, dict):
        raise SystemExit('the checkpoint is not a dict: %r' % type(payload))
    steps = int(payload.get('completed_steps', -1))
    if steps != args.expect_steps:
        problems.append('completed_steps=%r expected %d' % (payload.get('completed_steps'),
                                                            args.expect_steps))
    if payload.get('arm') != ARM:
        problems.append('arm=%r expected %r' % (payload.get('arm'), ARM))
    state = payload.get('clip') or payload.get('model')
    if not isinstance(state, dict) or not state:
        problems.append('no clip/model state dict in the checkpoint')
    if not isinstance(payload.get('optimizer'), dict) or not payload['optimizer'].get('state'):
        problems.append('the optimizer state is missing or empty')
    digest = payload.get('clip_state_digest')
    if digest and state and digest != trainer.state_digest(state):
        problems.append('clip_state_digest does not describe the stored state')
    if problems:
        raise SystemExit('checkpoint verification failed for %s: %s' % (args.checkpoint, problems))

    device = torch.device(args.device)
    model = build_model(device)
    before = probe(model, device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:                                    # strict=True already raises
        raise SystemExit('strict load reported missing=%r unexpected=%r' % (missing, unexpected))
    after = probe(model, device)
    differences = {key: float((before[key] - after[key]).abs().max()) for key in before}

    os.makedirs(args.output_dir, exist_ok=True)
    student_path = os.path.join(args.output_dir, 's0_global_only_student.pt')
    if os.path.isfile(student_path):
        raise SystemExit('refusing to overwrite %s' % student_path)
    torch.save({'model': state, 'completed_steps': steps, 'arm': ARM,
                'objective': trainer.OBJECTIVE, 'lambda_align': trainer.LAMBDA_ALIGN,
                'fixed_scale': trainer.FIXED_SCALE}, student_path)
    metadata = {
        'arm': ARM, 'objective': trainer.OBJECTIVE, 'phase': trainer.PHASE,
        'source_checkpoint': os.path.abspath(args.checkpoint),
        'source_checkpoint_sha256': trainer.file_sha256(args.checkpoint),
        'completed_steps': steps, 'tensor_count': len(state),
        'clip_state_digest': trainer.state_digest(state),
        'student_file': student_path, 'student_sha256': trainer.file_sha256(student_path),
        'student_key': 'model',
        'mask_network_in_student': any(key.startswith('mask_net') for key in state),
        'lossless_probe': differences,
        'lossless_note': ('max abs difference of encode_image / encode_text on a fixed probe before '
                          'and after loading the trained state; this arm adds no inference-time '
                          'module, so the student is the native CLIP state dict exactly'),
        'git_head': trainer.git_head(REPO),
    }
    metadata_path = os.path.join(args.output_dir, 's0_global_only_student_metadata.json')
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print('EXPORTED ' + student_path, flush=True)
    print('LOSSLESS image_max_abs_diff=%r text_max_abs_diff=%r'
          % (differences['image'], differences['text']), flush=True)
    print('EXPORT_GLOBAL_ONLY ' + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
