"""Two-rank DDP worker for the PG-CLIP acceptance tests (F).

Runs one real ``pgclip_train_step`` per rank with a **non-trivial** gate (its output weight is
perturbed, so the hard masks are genuinely mixed and not the all-open degenerate case) and writes the
post-step state plus the pre-step (post-backward, post-all-reduce) gradients, so the parent test can
compare a two-rank update against a single-process update of the same global batch.

    torchrun --nproc_per_node=2 tests/_pgclip_ddp_worker.py --out-dir <dir> --fixture <fixture.pt>
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, 'train'), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.pgclip import PreProjectionGate, build_optimizers, state_digest    # noqa: E402
from train_pgclip import (PgClipTrainModule, pgclip_train_step,                # noqa: E402
                          setup_distributed)


def grad_summary(grad):
    if grad is None:
        return None
    return {'norm': float(grad.float().norm()), 'max_abs': float(grad.float().abs().max()),
            'mean': float(grad.float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parsed = parser.parse_args()

    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    fixture = torch.load(parsed.fixture, map_location='cpu', weights_only=False)

    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.load_state_dict(fixture['clip_state'], strict=True)
    model = model.to(device)
    gate = PreProjectionGate(device=device).to(device)
    gate.load_state_dict(fixture['gate_state'])

    module = PgClipTrainModule(model, gate, rank=rank, image_chunk=8, text_chunk=8,
                               qp_checkpoint=False).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    ddp_model._set_static_graph()
    optimizers = build_optimizers(module.clip, module.gate, clip_lr=1e-4, gate_lr=1e-3)

    offset = rank * parsed.batch_size
    batch = {'image_a': fixture['images'][offset:offset + parsed.batch_size],
             'caption_said': fixture['captions'][offset:offset + parsed.batch_size]}
    text_ids = fixture['text_ids'][offset:offset + parsed.batch_size].to(device)
    state_pre = {key: value.detach().cpu().clone() for key, value in module.clip.state_dict().items()}

    out = pgclip_train_step(ddp_model, batch, optimizers, device, torch.bfloat16,
                            amp_enabled=False, completed_steps=0, text_ids=text_ids,
                            capture_grads=True)
    grads = out['_grads']
    record = {
        'rank': rank, 'world_size': world,
        'loss_global': float(out['loss_global']), 'loss_preproj': float(out['loss_preproj']),
        'loss_sparse': float(out['loss_sparse']), 'loss_total': float(out['loss_total']),
        'mask_kept_mean': float(out['encoded']['masks'].detach().sum(dim=1).float().mean()),
        'clip_state': {key: value.detach().cpu() for key, value in module.clip.state_dict().items()},
        'clip_state_pre': state_pre,
        'gate_state': {key: value.detach().cpu() for key, value in module.gate.state_dict().items()},
        'clip_grads': {(key[len('clip.'):] if key.startswith('clip.') else key): value.detach().cpu()
                       for key, value in grads.items()
                       if (key[len('clip.'):] if key.startswith('clip.') else key)
                       in fixture['grad_subset'] or key in fixture['grad_subset']},
        'grad_summaries': {key: grad_summary(value) for key, value in grads.items()},
        'grad_health': out['_grad_health'],
    }
    record['clip_digest_after'] = state_digest(module.clip.state_dict())
    record['gate_digest_after'] = state_digest(module.gate.state_dict())
    os.makedirs(parsed.out_dir, exist_ok=True)
    torch.save(record, os.path.join(parsed.out_dir, 'rank%d.pt' % rank))
    with open(os.path.join(parsed.out_dir, 'rank%d.json' % rank), 'w') as handle:
        json.dump({key: value for key, value in record.items()
                   if key not in ('clip_state', 'clip_state_pre', 'gate_state', 'clip_grads')},
                  handle, indent=2, sort_keys=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
