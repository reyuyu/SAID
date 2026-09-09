"""Explicit RNG control and auditable data streams for paired SALU experiments."""
import hashlib
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def state_digest(state):
    digest=hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def update_caption_digest(digest, captions):
    for caption in captions:
        encoded=caption.encode('utf-8')
        digest.update(len(encoded).to_bytes(8,'little'))
        digest.update(encoded)
