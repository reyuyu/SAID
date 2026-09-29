"""Rebuild the historical 317-tensor initial state; never claim the original container SHA."""
import argparse
import json
from pathlib import Path
import random
import sys

from common import BASE_SHA, INITIAL_DIGEST, sha, tensor_digest, verify_repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--clip-base', type=Path, default=Path.home() / '.cache/clip/ViT-B-16.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error('Choose new output and report paths')
    revision = verify_repo(args.repo)
    if sha(args.clip_base) != BASE_SHA:
        raise RuntimeError('CLIP base SHA256 mismatch')
    sys.path[:0] = [str(args.repo.resolve()), str(args.repo.resolve() / 'train')]
    import numpy as np
    import torch
    from model import longclip
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model, _ = longclip.load_from_clip(str(args.clip_base.resolve()), device='cpu',
                                     args=argparse.Namespace(base_model='ViT-B/16'))
    state = model.state_dict()
    digest = tensor_digest(state)
    if len(state) != 317 or digest != INITIAL_DIGEST:
        raise RuntimeError('Historical initial tensor identity did not match: ' + digest)
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise RuntimeError('Non-finite initialization')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': state, 'sha256': digest, 'reconstruction': True,
                'source': 'verified CLIP base + pinned LongCLIP constructor, seed=0'}, args.output)
    restored = torch.load(args.output, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(restored, strict=True)
    if tensor_digest(restored) != INITIAL_DIGEST:
        raise RuntimeError('Saved initialization failed roundtrip')
    report = {'status': 'PASS', 'path': str(args.output.resolve()), 'file_sha256': sha(args.output),
              'tensor_digest': digest, 'tensor_count': len(state), 'base_clip_sha256': BASE_SHA,
              'code_revision': revision, 'historical_tensor_identity_match': True,
              'original_container_recovered': False, 'torch_version': torch.__version__}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
