"""Identity checks shared by the portable 3/0 reproduction utilities."""
import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
INITIAL_DIGEST = 'caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf'
BASE_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode('utf-8'))
        digest.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_repo(repo):
    manifest = json.loads((BUNDLE / 'manifests/code_sha256.json').read_text())
    for relative, expected in manifest['files'].items():
        path = Path(repo) / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Fixed training code mismatch: ' + str(path))
    return manifest['training_sha']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--init', type=Path)
    args = parser.parse_args()
    revision = verify_repo(args.repo)
    report = {'status': 'PASS', 'training_sha': revision}
    if args.init:
        import torch
        payload = torch.load(args.init, map_location='cpu', weights_only=True)
        state = payload.get('model', payload)
        if len(state) != 317 or tensor_digest(state) != INITIAL_DIGEST:
            raise RuntimeError('Initial tensor identity mismatch')
        report.update(init_sha256=sha(args.init), init_tensor_digest=INITIAL_DIGEST)
    print(json.dumps(report))
