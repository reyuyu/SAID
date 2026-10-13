#!/usr/bin/env python3
"""Read-only per-sample mask and BalancedStack diagnostics for the DOCCI probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from analysis.modality_gap_docci_b16_v2_first_sentence.extract_features import (
    IMAGE_ROOT, LONGCLIP_PATH, OURS_PATH, build_ours, first_sentence, longclip_preprocess_from_clip,
    prefix_ids, sentence_ids, sha256,
)
from model import longclip
from model.nested_fusion_mask import pair_logits
from model.nested_semantic_mask import hard_st


SETTINGS = ("full", "prefix50", "first_sentence")


def summarize(x: torch.Tensor, prefix: str, out: dict[str, list[float]]):
    x = x.float()
    out[prefix + "_mean"].extend(x.mean(-1).detach().cpu().tolist())
    out[prefix + "_std"].extend(x.std(-1, unbiased=False).detach().cpu().tolist())
    out[prefix + "_min"].extend(x.min(-1).values.detach().cpu().tolist())
    out[prefix + "_max"].extend(x.max(-1).values.detach().cpu().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to reuse diagnostic output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    samples = json.loads((args.input_dir / "samples.json").read_text())
    features = np.load(args.input_dir / "features.npz")
    if len(samples) != 1000:
        raise RuntimeError(f"expected 1000 samples, found {len(samples)}")
    if sha256(LONGCLIP_PATH) != "ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894":
        raise RuntimeError("Long-CLIP checkpoint hash mismatch")
    if sha256(OURS_PATH) != "83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07":
        raise RuntimeError("Ours checkpoint hash mismatch")

    token_sets = {s: [] for s in SETTINGS}
    for row in samples:
        text = row["caption"]
        token_sets["full"].append(prefix_ids(text, None)[0])
        token_sets["prefix50"].append(prefix_ids(text, .5)[0])
        token_sets["first_sentence"].append(sentence_ids(text)[0])

    # Load the same production checkpoint and the same preprocess used by the
    # v2 extractor. No optimizer or model mutation is created.
    ours, ours_cfg = build_ours(device)
    ours_preprocess = longclip_preprocess_from_clip()
    stats: dict[str, list[float]] = {}
    for setting in SETTINGS:
        for field in ("mask", "prob", "logit", "gate", "text_component", "visual_component"):
            for suffix in ("mean", "std", "min", "max"):
                stats[f"{setting}_{field}_{suffix}" if suffix else f"{setting}_{field}"] = []
        for field in ("text_cond_norm", "visual_cond_norm", "raw_z_norm", "masked_pre_norm", "masked_norm", "paired_cosine"):
            stats[f"{setting}_{field}"] = []
        stats[f"{setting}_saturated_fraction"] = []
        stats[f"{setting}_selected_count"] = []
        stats[f"{setting}_all_open"] = []
        stats[f"{setting}_all_closed"] = []

    masks, probabilities, logits = {}, {}, {}
    gates = {}
    components = {}
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            stop = min(start + args.batch_size, len(samples))
            images = [Image.open(IMAGE_ROOT / row["image_path"]).convert("RGB") for row in samples[start:stop]]
            image_tensor = torch.stack([ours_preprocess(im) for im in images]).to(device)
            z, visual = ours.encode_visual(image_tensor)
            # BalancedStack returns pooled visual/text conditions at index 1.
            visual_cond = visual[1]
            for setting in SETTINGS:
                tokens = torch.stack(token_sets[setting][start:stop]).to(device)
                text, condition = ours.encode_view(tokens)
                text_cond = condition[1]
                gate = ours.fusion_branch.balanced_gate(visual_cond, text_cond, paired=True)
                logit = pair_logits(ours, visual, condition, paired=True)
                prob = logit.sigmoid()
                hard = (prob >= .5).float()
                masked_pre = z.float() * hard
                masked = torch.nn.functional.normalize(masked_pre, dim=-1, eps=1e-6)
                text_part = gate * text_cond
                visual_part = (1. - gate) * visual_cond
                # Exact feature-space diagnostics used by the existing probe.
                pair_cos = (masked * torch.nn.functional.normalize(text.float(), dim=-1, eps=1e-6)).sum(-1)
                for name, tensor in (("mask", hard), ("prob", prob), ("logit", logit), ("gate", gate),
                                     ("text_component", text_part), ("visual_component", visual_part),
                                     ("text_cond_norm", text_cond.norm(dim=-1)), ("visual_cond_norm", visual_cond.norm(dim=-1))):
                    if name in ("text_cond_norm", "visual_cond_norm"):
                        stats[f"{setting}_{name}"].extend(tensor.detach().cpu().tolist())
                    else:
                        summarize(tensor, f"{setting}_{name}", stats)
                stats[f"{setting}_raw_z_norm"].extend(z.float().norm(dim=-1).detach().cpu().tolist())
                stats[f"{setting}_masked_pre_norm"].extend(masked_pre.norm(dim=-1).detach().cpu().tolist())
                stats[f"{setting}_masked_norm"].extend(masked.norm(dim=-1).detach().cpu().tolist())
                stats[f"{setting}_paired_cosine"].extend(pair_cos.detach().cpu().tolist())
                stats[f"{setting}_saturated_fraction"].extend((((prob < .01) | (prob > .99)).float().mean(-1)).detach().cpu().tolist())
                stats[f"{setting}_selected_count"].extend(hard.sum(-1).detach().cpu().tolist())
                stats[f"{setting}_all_open"].extend(hard.all(-1).detach().cpu().tolist())
                stats[f"{setting}_all_closed"].extend((1 - hard.any(-1).float()).detach().cpu().tolist())
                masks.setdefault(setting, []).append(hard.cpu().numpy().astype(np.uint8))
                probabilities.setdefault(setting, []).append(prob.cpu().numpy().astype(np.float32))
                logits.setdefault(setting, []).append(logit.cpu().numpy().astype(np.float32))
                gates.setdefault(setting, []).append(gate.cpu().numpy().astype(np.float32))
                components.setdefault(setting, []).append(torch.stack((text_part.norm(dim=-1), visual_part.norm(dim=-1)), dim=1).cpu().numpy().astype(np.float32))

    packed = {f"{kind}_{setting}": np.concatenate(parts, axis=0) for kind, source in (("mask", masks), ("prob", probabilities), ("logit", logits), ("gate", gates), ("components", components)) for setting, parts in source.items()}
    np.savez_compressed(args.output_dir / "mask_arrays.npz", **packed)
    (args.output_dir / "per_sample_mask_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    provenance = {
        "input_features": str(args.input_dir / "features.npz"),
        "input_samples": str(args.input_dir / "samples.json"),
        "input_feature_sha256": sha256(args.input_dir / "features.npz"),
        "input_samples_sha256": sha256(args.input_dir / "samples.json"),
        "sample_count": len(samples), "settings": list(SETTINGS),
        "longclip_checkpoint": str(LONGCLIP_PATH), "longclip_checkpoint_sha256": sha256(LONGCLIP_PATH),
        "ours_checkpoint": str(OURS_PATH), "ours_checkpoint_sha256": sha256(OURS_PATH),
        "ours_config": ours_cfg,
        "mask_formula": "pair_logits(BalancedStack, paired=True) -> sigmoid -> hard_st; masked image = normalize(z * hard_mask)",
        "balanced_stack_formula": "gate=sigmoid(W_text(text_condition)+W_visual(visual_condition)); logits=gate*text_condition+(1-gate)*visual_condition",
        "no_training": True, "optimizer_created": False,
    }
    (args.output_dir / "diagnostic_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "settings": list(SETTINGS), "mask_arrays": str(args.output_dir / 'mask_arrays.npz')}, indent=2))


if __name__ == "__main__":
    main()
