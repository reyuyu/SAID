"""Extract matched DOCCI modality-gap features for Long-CLIP and HNS-v1.

This is an inference-only script. It samples the existing official DOCCI test
manifest once, uses the same BPE-prefix token IDs for every model, and saves
normalized 512-D features plus provenance for the plotting/reporting stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
MANIFEST = Path("/opt/data/private/lklk/SAID/local_assets/retrieval_benchmarks/manifests/docci_test.jsonl")
IMAGE_ROOT = Path("/opt/data/private/lklk/SAID/local_assets/retrieval_benchmarks/docci/images")
EXPECTED_LONGCLIP_SHA = "ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894"
EXPECTED_OURS_SHA = "83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07"
CONTEXT = 248
CONTENT_LIMIT = CONTEXT - 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prefix_ids(text: str, fraction: float | None) -> tuple[torch.Tensor, int, int]:
    """Return SOT/content/EOT IDs using the official Long-CLIP BPE tokenizer."""
    sot = longclip._tokenizer.encoder["<|startoftext|>"]
    eot = longclip._tokenizer.encoder["<|endoftext|>"]
    content = list(longclip._tokenizer.encode(text))
    original = len(content)
    if fraction is not None:
        # Prefix sizes are defined in BPE-token space; retain at least one
        # content token when the caption has content.
        keep = max(1, int(math.ceil(original * fraction))) if original else 0
        content = content[:keep]
    content = content[:CONTENT_LIMIT]
    ids = [sot, *content, eot]
    result = torch.zeros(CONTEXT, dtype=torch.long)
    result[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return result, original, len(content)


def load_records(n: int, seed: int) -> list[dict]:
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    valid = [r for r in records if (IMAGE_ROOT / r["image_path"]).is_file()]
    if len(valid) != len(records):
        raise RuntimeError(f"DOCCI manifest has missing images: {len(records)-len(valid)}")
    rng = random.Random(seed)
    chosen = rng.sample(range(len(valid)), n)
    return [valid[i] for i in chosen]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6)


def finite_report(x: np.ndarray) -> dict:
    finite = np.isfinite(x).all(axis=1)
    norms = np.linalg.norm(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    return {
        "rows": int(x.shape[0]),
        "dims": int(x.shape[1]),
        "nonfinite_rows": int((~finite).sum()),
        "zero_rows": int((norms <= 1e-8).sum()),
        "finite": bool(finite.all()),
    }


def build_ours(device: torch.device):
    payload = torch.load(OURS_PATH, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    clip, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    module = BalancedSearch(
        clip.float(),
        arm=cfg["arm"],
        checkpoint_encoders=False,
        image_chunk=cfg.get("image_chunk", 128),
        text_chunk=cfg.get("text_chunk", 128),
        condition_mode=cfg.get("condition_mode", "dual_branch"),
        shuffle_seed=cfg.get("shuffle_seed", 0),
        checkpoint_pair_blocks=cfg.get("checkpoint_pair_blocks", False),
        fusion=cfg["fusion"],
        visual=cfg["visual"],
        search_hparams=hparams(cfg),
        inclusion_hierarchy=cfg.get("inclusion_hierarchy", "detail_chain"),
        view_sparsity_weights=cfg.get("view_sparsity_weights", [1.0, 2.0, 2.0]),
        hns_enabled=cfg.get("hns_enabled", False),
        hns_beta=cfg.get("hns_beta", [2.0, 2.0]),
    )
    module.clip.load_state_dict(payload["model"], strict=True)
    module.fusion_branch.load_state_dict(payload["adapter"], strict=True)
    module.to(device).eval()
    return module, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authorized inference run")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "samples").mkdir()

    for path, expected in ((LONGCLIP_PATH, EXPECTED_LONGCLIP_SHA), (OURS_PATH, EXPECTED_OURS_SHA)):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    records = load_records(args.n, args.seed)
    sample_rows = []
    token_sets: dict[str, list[torch.Tensor]] = {"full": [], "prefix50": [], "prefix30": []}
    length_rows = []
    for row in records:
        captions = row["caption"]
        versions = {}
        for name, frac in (("full", None), ("prefix50", 0.50), ("prefix30", 0.30)):
            ids, original_len, kept_len = prefix_ids(captions, frac)
            token_sets[name].append(ids)
            versions[name] = {"original_bpe_tokens": original_len, "kept_bpe_tokens": kept_len}
        sample_rows.append({**row, "versions": versions})
        length_rows.append({"image_id": row["image_id"], **versions})
    (args.output_dir / "samples.json").write_text(json.dumps(sample_rows, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "caption_lengths.json").write_text(json.dumps(length_rows, indent=2) + "\n")

    long_model, long_preprocess = longclip_load(LONGCLIP_PATH, device)
    ours, ours_cfg = build_ours(device)
    ours_preprocess = longclip_preprocess_from_clip()
    # The SAID native preprocess is the repository Long-CLIP transform. Keep
    # official Long-CLIP's transform separately and record both fingerprints.
    _, ours_preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())

    arrays: dict[str, list[np.ndarray]] = {}
    keep_ratios: dict[str, list[float]] = {k: [] for k in token_sets}
    names = ("longclip", "ours_masked", "ours_native")
    for model_name in names:
        for setting in token_sets:
            arrays[f"{model_name}_{setting}_image"] = []
            arrays[f"{model_name}_{setting}_text"] = []

    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records))
            pil_images = [Image.open(IMAGE_ROOT / r["image_path"]).convert("RGB") for r in records[start:stop]]
            long_images = torch.stack([long_preprocess(im) for im in pil_images]).to(device)
            ours_images = torch.stack([ours_preprocess(im) for im in pil_images]).to(device)
            long_img = normalize(long_model.encode_image(long_images))
            ours_z, ours_visual = ours.encode_visual(ours_images)
            ours_native_img = normalize(ours_z)
            for setting, token_list in token_sets.items():
                tokens = torch.stack(token_list[start:stop]).to(device)
                long_txt = normalize(long_model.encode_text(tokens))
                ours_txt, ours_condition = ours.encode_view(tokens)
                ours_txt = normalize(ours_txt)
                logits = pair_logits(ours, ours_visual, ours_condition, paired=True)
                probability = logits.sigmoid()
                mask = hard_st(probability).float()
                ours_masked_img = normalize(ours_z * mask)
                arrays[f"longclip_{setting}_image"].append(long_img.cpu().numpy())
                arrays[f"longclip_{setting}_text"].append(long_txt.cpu().numpy())
                arrays[f"ours_masked_{setting}_image"].append(ours_masked_img.cpu().numpy())
                arrays[f"ours_masked_{setting}_text"].append(ours_txt.cpu().numpy())
                arrays[f"ours_native_{setting}_image"].append(ours_native_img.cpu().numpy())
                arrays[f"ours_native_{setting}_text"].append(ours_txt.cpu().numpy())
                keep_ratios[setting].extend(mask.mean(-1).detach().cpu().tolist())

    packed = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}
    np.savez_compressed(args.output_dir / "features.npz", **packed)
    feature_checks = {key: finite_report(value) for key, value in packed.items()}
    provenance = {
        "seed": args.seed,
        "sample_count": len(records),
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "image_root": str(IMAGE_ROOT),
        "longclip_checkpoint": str(LONGCLIP_PATH),
        "longclip_checkpoint_sha256": sha256(LONGCLIP_PATH),
        "ours_checkpoint": str(OURS_PATH),
        "ours_checkpoint_sha256": sha256(OURS_PATH),
        "ours_checkpoint_completed_steps": int(torch.load(OURS_PATH, map_location="cpu", weights_only=False)["completed_steps"]),
        "ours_config": ours_cfg,
        "caption_tokenization": "Long-CLIP BPE; SOT/EOT; prefix ceil fraction; content capped at 246 tokens",
        "feature_dim": 512,
        "models": {"longclip": "official model.longclip.load", "ours": "BalancedSearch + NestedFusionMask paired hard-ST mask"},
        "preprocess": {"longclip": repr(long_preprocess), "ours": repr(ours_preprocess)},
        "feature_checks": feature_checks,
        "mask_keep_ratio": {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in keep_ratios.items()},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "features": len(packed), "sample_count": len(records), "mask_keep_ratio": provenance["mask_keep_ratio"]}, indent=2))


def longclip_load(path: Path, device: torch.device):
    return longclip.load(str(path), device=device)


def longclip_preprocess_from_clip():
    _, preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    return preprocess


if __name__ == "__main__":
    main()
