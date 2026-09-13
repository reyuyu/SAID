"""Two-process CPU DDP worker used by the clean suffix acceptance test."""

import argparse
import copy
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
from model.dual_mask_suffix import pairwise_masked_scores
from model.said_cls_cvssl import said_mask_from_hidden


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
    # One non-degenerate 1-vs-3 case against an explicit full-batch reference.
    torch.manual_seed(777)
    all_images = torch.randn(8, 1, 6)
    all_prefix = torch.randint(1, 9, (8, 5))
    all_suffix = torch.randint(1, 9, (8, 5))
    all_valid = torch.tensor([True, False, False, False, True, True, True, False])
    sl = slice(rank * 4, (rank + 1) * 4)
    optimizer.zero_grad(set_to_none=True)
    check = ddp(all_images[sl], all_prefix[sl], all_suffix[sl], all_valid[sl],
                torch.arange(4) + rank * 4)
    check["loss_suffix"].backward()
    prod_grads = [ddp.module.clip.image.weight.grad.detach().clone(),
                  ddp.module.clip.text.weight.grad.detach().clone(),
                  ddp.module.suffix_gate[0].weight.grad.detach().clone()]
    ref_clip = ToyClip(); ref_clip.load_state_dict(ddp.module.clip.state_dict())
    ref_gate = copy.deepcopy(ddp.module.suffix_gate)
    g = ref_clip.encode_image(all_images)
    p_raw, p_hidden = ref_clip.encode_text(all_prefix, return_full=True)
    m_s, _, _ = said_mask_from_hidden(ref_clip.mask_net, p_hidden)
    t = ref_clip.encode_text(all_suffix)
    q, _ = pairwise_masked_scores(g, m_s, t, ref_gate, image_chunk=3, text_chunk=4)
    valid_idx = all_valid.nonzero(as_tuple=False).flatten()
    ref_loss = (nn.functional.cross_entropy(q[valid_idx][:, valid_idx], torch.arange(valid_idx.numel()), reduction='sum') +
                nn.functional.cross_entropy(q.t()[valid_idx][:, valid_idx], torch.arange(valid_idx.numel()), reduction='sum')) / valid_idx.numel()
    ref_loss.backward()
    ref_grads = [ref_clip.image.weight.grad, ref_clip.text.weight.grad, ref_gate[0].weight.grad]
    errors = [max(float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-12)))
              for a, b in zip(prod_grads, ref_grads)]
    norms = [[float(a.norm()), float(b.norm())] for a, b in zip(prod_grads, ref_grads)]
    optimizer.step()
    ref_opt = torch.optim.AdamW(list(ref_clip.parameters()) + list(ref_gate.parameters()), lr=1e-3)
    ref_opt.step()
    update_error = float((ddp.module.clip.image.weight - ref_clip.image.weight).abs().max())
    if rank == 0:
        gradient_report = {"max_abs_or_rel": max(errors), "per_tensor_max_abs_or_rel": errors,
                           "norms_production_reference": norms,
                           "loss_global": float(check["loss_suffix_global"]),
                           "reference_loss": float(ref_loss.detach()), "update_max_abs": update_error}
    else:
        gradient_report = None
    dist.barrier()
    cases = [
        torch.tensor([False, False, False, False]) if rank == 0 else torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, False]),
        torch.tensor([True, False, False, False]) if rank == 0 else torch.tensor([False, False, False, False]),
        torch.tensor([True, False, False, False]) if rank == 0 else torch.tensor([False, False, False, False]),
        torch.tensor([True, False, True, False]) if rank == 0 else torch.tensor([False, True, False, True]),
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
            json.dump({"cases": results, "gradient_reference": gradient_report}, handle,
                      indent=2, sort_keys=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
