"""Strictly export the native student of a completed 3/0, 500-update checkpoint."""
import argparse
import json
from pathlib import Path
import sys

from common import sha, tensor_digest, verify_repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error('Choose new output and report paths')
    revision = verify_repo(args.repo)
    source_sha = sha(args.checkpoint)
    if args.expected_sha256 and source_sha != args.expected_sha256:
        raise RuntimeError('Source checkpoint SHA256 mismatch')
    sys.path[:0] = [str(args.repo.resolve()), str(args.repo.resolve() / 'train')]
    import torch
    from model import longclip
    from model.dual_mask_suffix import DualMaskSuffixTrainModule
    # The source is an owned training artifact; an expected SHA can bind a transferred copy.
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    required = {'suffix_mode': 'masked', 'suffix_lambda': 3.0, 'u_sparsity': 0.0,
                'formal_optimizer_updates': 500, 'world_size': 4, 'batch_size_per_gpu': 256,
                'epochs': 3, 'loader_batches': 1217, 'lr_horizon_steps': 3651, 'seed': 0,
                'training_sha': revision}
    if payload['completed_steps'] != 500:
        raise RuntimeError('Expected exactly 500 completed updates')
    for key, value in required.items():
        if config.get(key) != value:
            raise RuntimeError('Checkpoint configuration mismatch: ' + key)
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    module = DualMaskSuffixTrainModule(model, 'masked', lambda_suffix=3, lambda_u_sparse=0)
    state = payload['clip_state']
    module.clip.load_state_dict(state, strict=True)
    module.suffix_gate.load_state_dict(payload['suffix_gate_state'], strict=True)
    if len(state) != 317 or tensor_digest(state) != payload['provenance']['state_digest']:
        raise RuntimeError('Student identity mismatch')
    for mapping in [state, payload['suffix_gate_state']]:
        if not all(torch.isfinite(v).all().item() for v in mapping.values()):
            raise RuntimeError('Non-finite model state')
    counters = {}
    for name in ['clip', 'mask', 'suffix']:
        optimizer = payload['optimizer_states'][name]
        steps = [int(x['step']) for x in optimizer['state'].values() if 'step' in x]
        if not steps or min(steps) != 500 or max(steps) != 500:
            raise RuntimeError('Optimizer step mismatch: ' + name)
        for fields in optimizer['state'].values():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all().item() for v in fields.values()):
                raise RuntimeError('Non-finite optimizer state: ' + name)
        counters[name] = {'min': min(steps), 'max': max(steps), 'count': len(steps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    restored = torch.load(args.output, map_location='cpu', weights_only=True)
    model.load_state_dict(restored, strict=True)
    if not all(torch.equal(v, restored[k]) for k, v in state.items()):
        raise RuntimeError('Bare student export is not lossless')
    report = {'status': 'PASS', 'completed_steps': 500, 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': source_sha, 'bare_student': str(args.output.resolve()),
              'bare_sha256': sha(args.output), 'tensor_count': 317, 'tensor_digest': tensor_digest(state),
              'strict_clip': True, 'strict_suffix_gate': True, 'strict_bare_student': True,
              'roundtrip_equal': True, 'optimizer_steps': counters, 'training_sha': revision}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
