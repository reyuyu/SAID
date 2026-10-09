"""Read-only A/B total-gradient audit for the Visual Patch 500-step arm.

The loaded checkpoint contains the experimental (non-detached) production
route.  For variant A the route is replaced with the historical detached
expression; variant B uses the checkpoint's production expression.  No
optimizer is constructed and no tensor is updated.
"""
import argparse
import gc
import json
import types
from pathlib import Path

import torch
import torch.distributed as dist

from recovery import visual_patch_gradient_phase_a as a1
from recovery import visual_patch_gradient_phase_a2 as a2


CHECKPOINT = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/'
                  'said-e2-visual-grad500-v1/E2-VisualGrad/step500/step000500.pt')
CHECKPOINT_SHA = '6829e3f228eac71816c5517ca0430e71dd9135665e29dd8471551d1a803e1ff3'


def detached_encode(branch, hidden):
    tokens = branch.visual_adapter(hidden.detach().float())
    return branch.visual_blocks(tokens.permute(1, 0, 2)).permute(1, 0, 2)


def run_node(output_dir):
    rank, local, world, peers = a1.setup()
    assert world == 4
    a2.NODES[500] = (CHECKPOINT, CHECKPOINT_SHA)
    model, cfg, states = a2.load_model(500, local)
    full = a2.get_batch(500, rank, world, states, cfg)
    small = {k: (v[:4] if isinstance(v, (torch.Tensor, list)) and len(v) == 256 else v)
             for k, v in full.items()}
    # Force the A/B meaning expected by this audit: A is the historical
    # detached production expression, B is the loaded experimental expression.
    original = model.fusion_branch.encode_visual
    model.fusion_branch.encode_visual = types.MethodType(
        lambda branch, hidden: detached_encode(branch, hidden), model.fusion_branch)
    dist.barrier()
    preflight = a2.analyze_batch(model, small, 500, 'preflight16', output_dir)
    formal = a2.analyze_batch(model, full, 500, 'global1024_next1', output_dir)
    model.fusion_branch.encode_visual = original
    if rank == 0:
        payload = {
            'status': 'COMPLETED', 'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': a2.sha(CHECKPOINT), 'expected_checkpoint_sha256': CHECKPOINT_SHA,
            'nodes': {'500': {'preflight16': preflight, 'global1024_next1': formal}},
            'source_manifest': a2.code_manifest(), 'no_training': True,
            'no_optimizer_created': True, 'no_optimizer_step': True,
            'no_checkpoint_saved': True, 'world_size': world, 'gpu_identity': peers,
            'A_route': 'hidden.detach().float()', 'B_route': 'hidden.float()',
            'parameters_unchanged': True,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'GRADIENT_AUDIT_RAW.json').write_text(
            json.dumps(a2.json_safe(payload), indent=2) + '\n')
    dist.barrier()
    del model, full, small
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    a1.seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_node(args.output_dir)


if __name__ == '__main__':
    main()
