"""Pre-flight: does the shared init still produce the digest the arms will log?

``SaidClsCvsslTrainModule`` reports the digest of the *loaded* state, so the question that matters
is not the file's sha256 but ``state_digest(model.state_dict())`` after ``load_init_state``. This
script answers exactly that, plus the parameter dtype counts (real dtype, not metadata).
"""
import hashlib
import sys

import torch

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/train')

from train_said_cls_cvssl import load_init_state, state_digest  # noqa: E402


class TinyClip(torch.nn.Module):
    """Only needs the same state-dict key layout as the real model for the digest question."""

    def __init__(self):
        super().__init__()
        self.text_projection = torch.nn.Parameter(torch.zeros(1))
        self.mask_net = torch.nn.Linear(2, 2)


def main():
    path = sys.argv[1]
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        state = payload['model']
    else:
        state = payload
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode('utf-8'))
        if torch.is_tensor(value):
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode('utf-8'))
    dtypes = {}
    for value in state.values():
        if torch.is_tensor(value):
            dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
    print('INIT_FILE %s' % path)
    print('INIT_TENSORS %d' % len(state))
    print('INIT_DTYPE_COUNTS %s' % sorted(dtypes.items()))
    print('INIT_STATE_SHA256 %s' % digest.hexdigest())
    print('HISTORICAL_REFERENCE caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf')
    print('MATCHES_HISTORICAL %s'
          % (digest.hexdigest()
             == 'caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf'))


if __name__ == '__main__':
    main()
