#!/usr/bin/env python3
"""GPU inference for the new random caption views only.

Existing Full/Prefix50/First-Sentence arrays are never recomputed here; this
script writes one compact feature file per random view and records the exact
sample subset used by NoFirst-Matched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from model.nested_fusion_mask import pair_logits
from model.nested_semantic_mask import hard_st

LONGCLIP_PATH = Path("/opt/data/private/lklk/SAID/runtime/third_party_models/longclip_b16/longclip-B.pt")
OURS_PATH = Path("/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1/step4868/step003651.pt")
EXPECTED_LONGCLIP_SHA = "ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894"
EXPECTED_OURS_SHA = "83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07"
CONTEXT = 248


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pack(text: str) -> torch.Tensor:
    tok = list(longclip._tokenizer.encode(text))[: CONTEXT - 2]
    ids = [longclip._tokenizer.encoder["<|startoftext|>"], *tok, longclip._tokenizer.encoder["<|endoftext|>"]]
    out = torch.zeros(CONTEXT, dtype=torch.long); out[:len(ids)] = torch.tensor(ids, dtype=torch.long); return out


def normalize(x):
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6)


def build_ours(device):
    payload = torch.load(OURS_PATH, map_location="cpu", weights_only=False); cfg = payload["config"]
    clip, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    module = BalancedSearch(clip.float(), arm=cfg["arm"], checkpoint_encoders=False, image_chunk=cfg.get("image_chunk", 128), text_chunk=cfg.get("text_chunk", 128), condition_mode=cfg.get("condition_mode", "dual_branch"), shuffle_seed=cfg.get("shuffle_seed", 0), checkpoint_pair_blocks=cfg.get("checkpoint_pair_blocks", False), fusion=cfg["fusion"], visual=cfg["visual"], search_hparams=hparams(cfg), inclusion_hierarchy=cfg.get("inclusion_hierarchy", "detail_chain"), view_sparsity_weights=cfg.get("view_sparsity_weights", [1.0, 2.0, 2.0]), hns_enabled=cfg.get("hns_enabled", False), hns_beta=cfg.get("hns_beta", [2.0, 2.0]))
    module.clip.load_state_dict(payload["model"], strict=True); module.fusion_branch.load_state_dict(payload["adapter"], strict=True); module.to(device).eval(); return module, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--views", type=Path, required=True); ap.add_argument("--image-root", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=False); device = torch.device(args.device); started = time.perf_counter()
    for path, expected in ((LONGCLIP_PATH, EXPECTED_LONGCLIP_SHA), (OURS_PATH, EXPECTED_OURS_SHA)):
        actual = sha256(path)
        if actual != expected: raise RuntimeError(f"Checkpoint SHA mismatch: {path} {actual} != {expected}")
    payload = json.loads(args.views.read_text()); records = payload["records"]
    view_names = list(records[0]["views"])
    valid_indices = {name: [i for i, row in enumerate(records) if row["views"][name].get("valid", False)] for name in view_names}
    for row in records:
        if not (args.image_root / row["image_path"]).is_file(): raise RuntimeError(f"Missing image: {row['image_path']}")
    long_model, long_preprocess = longclip.load(str(LONGCLIP_PATH), device=device); ours, ours_cfg = build_ours(device); _, ours_preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    before = {"longclip": digest(long_model), "sema_clip": digest(ours)}
    arrays = {name: {"longclip_image": [], "longclip_text": [], "ours_native_image": [], "ours_native_text": [], "ours_masked_image": [], "ours_masked_text": [], "mask": []} for name in view_names}
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records)); rows = records[start:stop]
            images = [Image.open(args.image_root / row["image_path"]).convert("RGB") for row in rows]
            long_images = torch.stack([long_preprocess(im) for im in images]).to(device); ours_images = torch.stack([ours_preprocess(im) for im in images]).to(device)
            long_img = normalize(long_model.encode_image(long_images)); ours_z, ours_visual = ours.encode_visual(ours_images); native_img = normalize(ours_z)
            for name in view_names:
                tokens = torch.stack([pack(row["views"][name].get("text", "")) for row in rows]).to(device)
                long_txt = normalize(long_model.encode_text(tokens)); ours_txt, cond = ours.encode_view(tokens); ours_txt = normalize(ours_txt)
                logits = pair_logits(ours, ours_visual, cond, paired=True); mask = hard_st(logits.sigmoid()).float(); masked_img = normalize(ours_z * mask)
                local = [i - start for i in valid_indices[name] if start <= i < stop]
                if local:
                    idx = torch.tensor(local, device=device, dtype=torch.long)
                    arrays[name]["longclip_image"].append(long_img[idx].cpu().numpy()); arrays[name]["longclip_text"].append(long_txt[idx].cpu().numpy()); arrays[name]["ours_native_image"].append(native_img[idx].cpu().numpy()); arrays[name]["ours_native_text"].append(ours_txt[idx].cpu().numpy()); arrays[name]["ours_masked_image"].append(masked_img[idx].cpu().numpy()); arrays[name]["ours_masked_text"].append(ours_txt[idx].cpu().numpy()); arrays[name]["mask"].append(mask[idx].cpu().numpy().astype(np.uint8))
            if stop % 160 == 0 or stop == len(records): print(json.dumps({"dataset": args.dataset, "completed_pairs": stop, "elapsed_s": time.perf_counter() - started}), flush=True)
    manifest = {"dataset": args.dataset, "sampling_manifest": str(args.views), "sampling_manifest_sha256": sha256(args.views), "view_names": view_names, "valid_indices": valid_indices, "files": {}, "checkpoint_sha256": {"longclip": sha256(LONGCLIP_PATH), "sema_clip": sha256(OURS_PATH)}, "checkpoint_paths": {"longclip": str(LONGCLIP_PATH), "sema_clip": str(OURS_PATH)}, "model_state_sha256_before": before, "precision": {"longclip": sorted({str(p.dtype) for p in long_model.parameters()}), "sema_clip": sorted({str(p.dtype) for p in ours.parameters()})}, "autocast": False, "ours_config": ours_cfg, "elapsed_s": time.perf_counter() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30}
    for name in view_names:
        path = args.output_dir / f"{name}.npz"; valid = valid_indices[name]
        packed = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays[name].items()}; np.savez_compressed(path, **packed); manifest["files"][name] = {"path": str(path), "sha256": sha256(path), "valid_count": len(valid), "feature_shapes": {k: list(v.shape) for k,v in packed.items()}}
    after = {"longclip": digest(long_model), "sema_clip": digest(ours)}; assert before == after; manifest["model_state_sha256_after"] = after
    for path, expected in ((LONGCLIP_PATH, EXPECTED_LONGCLIP_SHA), (OURS_PATH, EXPECTED_OURS_SHA)): assert sha256(path) == expected
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "views": len(view_names), "feature_manifest": str(args.output_dir/'feature_manifest.json'), "elapsed_s": manifest['elapsed_s']}, indent=2))


if __name__ == "__main__": main()
