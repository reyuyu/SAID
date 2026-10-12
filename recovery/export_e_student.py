"""Strict bare-student export and native embedding equivalence audit for E."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from model import longclip


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def optimizer_steps(optimizer):
    values = []
    for state in optimizer.get('state', {}).values():
        step = state.get('step')
        if step is not None:
            values.append(int(step.item() if torch.is_tensor(step) else step))
    return sorted(set(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bare-output', required=True)
    p.add_argument('--receipt', required=True)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['completed_steps'] == 3651
    assert checkpoint['scheduler_horizon'] == 4868
    assert checkpoint['stop_updates'] == 3651
    assert optimizer_steps(checkpoint['optimizer']) == [3651]
    state = checkpoint['model']
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=SimpleNamespace())
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.manual_seed(0)
    image = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 49408, (2, 248), dtype=torch.long)
    with torch.no_grad():
        image_before = model.encode_image(image).float()
        text_before = model.encode_text(text).float()
    bare = {name: value.detach().cpu().clone() for name, value in state.items()}
    Path(args.bare_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bare, args.bare_output)
    reloaded, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=SimpleNamespace())
    reloaded.load_state_dict(torch.load(args.bare_output, map_location='cpu', weights_only=False), strict=True)
    reloaded.eval()
    with torch.no_grad():
        image_after = reloaded.encode_image(image).float()
        text_after = reloaded.encode_text(text).float()
    receipt = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'completed_steps': checkpoint['completed_steps'],
        'scheduler_horizon': checkpoint['scheduler_horizon'],
        'optimizer_steps': optimizer_steps(checkpoint['optimizer']),
        'bare_student': str(Path(args.bare_output).resolve()),
        'bare_student_sha256': sha256(args.bare_output),
        'state_tensor_count': len(bare),
        'strict_load': True,
        'image_max_abs': float((image_before - image_after).abs().max()),
        'text_max_abs': float((text_before - text_after).abs().max()),
    }
    receipt['passed'] = receipt['image_max_abs'] == 0.0 and receipt['text_max_abs'] == 0.0
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
    assert receipt['passed'], receipt


if __name__ == '__main__':
    main()
