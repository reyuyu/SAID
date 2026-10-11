"""Strict ViT-L/14 bare-student export for HNS-v1 Pure3."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from model import longclip


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expect-updates', required=True, type=int)
    args = p.parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('completed_steps') != args.expect_updates:
        raise ValueError((payload.get('completed_steps'), args.expect_updates))
    model, _ = longclip.load_from_clip('ViT-L/14', device='cpu', args=argparse.Namespace())
    model.load_state_dict(payload['model'], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    student, _ = longclip.load_from_clip('ViT-L/14', device='cpu', args=argparse.Namespace())
    student.load_state_dict(torch.load(output, map_location='cpu', weights_only=True), strict=True)
    # A deterministic CPU round-trip proves that the exported state is the same
    # native image/text encoder as the checkpoint's CLIP submodule.
    tokens = longclip.tokenize(['a small scene with a person and a vehicle'], truncate=True)
    images = torch.zeros(1, 3, 224, 224)
    model.eval(); student.eval()
    with torch.inference_mode():
        image_delta = (model.encode_image(images) - student.encode_image(images)).abs().max().item()
        text_delta = (model.encode_text(tokens) - student.encode_text(tokens)).abs().max().item()
    if image_delta != 0.0 or text_delta != 0.0:
        raise AssertionError((image_delta, text_delta))
    report = {
        'passed': True, 'strict_load': True, 'completed_steps': args.expect_updates,
        'checkpoint_sha256': sha256(checkpoint), 'bare_sha256': sha256(output),
        'image_max_abs': image_delta, 'text_max_abs': text_delta,
        'model': 'ViT-L/14', 'native_student_only': True,
    }
    output.with_name('export-check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
