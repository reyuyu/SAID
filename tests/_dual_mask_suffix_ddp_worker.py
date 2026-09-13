"""Two-process CPU DDP worker used by the clean suffix acceptance test."""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model.dual_mask_suffix import DualMaskSuffixTrainModule


class ToyClip(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.image = nn.Linear(6, 4)
        self.text = nn.Linear(4, 4)
        self.mask_net = ToyMaskNet()

    def encode_image(self, images):
        return self.image(images.flatten(1))

    def encode_text(self, tokens, return_full=False):
        pooled = tokens.float().mean(dim=1, keepdim=True).repeat(1, 4)
        hidden = pooled[:, None, :].repeat(1, 5, 1)
        value = self.text(hidden[:, 0, :])
        return (value, hidden) if return_full else value


class ToyMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden):
        return self.linear(hidden.mean(dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world)
    ddp = torch.nn.parallel.DistributedDataParallel(
        DualMaskSuffixTrainModule(ToyClip(), "masked", rank=rank, feature_dim=4,
                                   image_chunk=2, text_chunk=3), find_unused_parameters=True)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    cases = [
        torch.tensor([False, False, False, False]) if rank == 0 else torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, False]),
        torch.tensor([True, False, False, False]) if rank == 0 else torch.tensor([False, False, False, False]),
        torch.tensor([True, False, False, False]) if rank == 0 else torch.tensor([False, False, False, False]),
    ]
    results = []
    for index, valid in enumerate(cases):
        torch.manual_seed(400 + rank * 11 + index)
        images = torch.randn(4, 1, 6)
        prefix = torch.randint(1, 9, (4, 5))
        suffix = torch.randint(1, 9, (4, 5))
        ids = torch.arange(4, dtype=torch.long) + rank * 4
        optimizer.zero_grad(set_to_none=True)
        out = ddp(images, prefix, suffix, valid, ids)
        assert torch.isfinite(out["loss_total"]).item()
        out["loss_total"].backward()
        optimizer.step()
        global_valid = [None for _ in range(world)]
        dist.all_gather_object(global_valid, int(valid.sum()))
        results.append({"case": index, "global_valid": sum(global_valid),
                        "loss_suffix": float(out["loss_suffix"].detach()),
                        "loss_total": float(out["loss_total"].detach())})
    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
