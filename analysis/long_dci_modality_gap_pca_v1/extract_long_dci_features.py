#!/usr/bin/env python3
"""Extract fixed Long-DCI modality-gap features for PCA diagnostics.

Inference only.  The source manifest, checkpoints, tokenizer, balanced-stack
mask path, and sample order are recorded in the output provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
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
MANIFEST = Path("/opt/data/private/lklk/SAID/local_assets/retrieval_benchmarks/manifests/long_dci_reconstructed.jsonl")
IMAGE_ROOT = Path("/opt/data/private/lklk/SAID/local_assets/retrieval_benchmarks/dci/images")
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


def tokenizer_ids(text: str) -> list[int]:
    return list(longclip._tokenizer.encode(text))


def pack_content(content: list[int]) -> torch.Tensor:
    sot = longclip._tokenizer.encoder["<|startoftext|>"]
    eot = longclip._tokenizer.encoder["<|endoftext|>"]
    content = content[:CONTENT_LIMIT]
    ids = [sot, *content, eot]
    out = torch.zeros(CONTEXT, dtype=torch.long)
    out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


ABBREVIATIONS = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc"}


def first_sentence(text: str) -> tuple[str, dict]:
    """Return first natural sentence and explicit boundary diagnostics."""
    candidates = []
    skipped = []
    for match in re.finditer(r'''[.!?]+["”’)\]]*(?=\s|$)''', text):
        punct = match.group(0)[0]
        remainder = text[match.end():].lstrip()
        if punct == ".":
            before = text[: match.start()].rstrip()
            word = re.split(r"\s+", before)[-1].lower().strip("(\"'[]{}")
            if word in ABBREVIATIONS and remainder:
                skipped.append({"position": match.start(), "reason": "known_abbreviation", "word": word})
                continue
            if re.fullmatch(r"(?:[a-z]\.)+[a-z]", word) and remainder:
                skipped.append({"position": match.start(), "reason": "initialism", "word": word})
                continue
            if re.fullmatch(r"[a-z]", word) and remainder and remainder[0].isupper():
                skipped.append({"position": match.start(), "reason": "single_initial", "word": word})
                continue
        if punct in ("!", "?") and remainder and remainder[0].islower():
            skipped.append({"position": match.start(), "reason": "internal_punctuation_before_lowercase", "punctuation": punct})
            continue
        candidates.append(match)
    if candidates:
        end = candidates[0].end()
        sentence = text[:end].strip()
        return sentence, {"boundary_found": True, "fallback_full_caption": False, "boundary": candidates[0].group(0), "skipped_candidates": skipped}
    return text.strip(), {"boundary_found": False, "fallback_full_caption": True, "boundary": None, "skipped_candidates": skipped}


def make_caption_versions(caption: str) -> tuple[dict[str, torch.Tensor], dict]:
    original = tokenizer_ids(caption)
    full = original[:CONTENT_LIMIT]
    prefix = full[: max(1, int(math.ceil(len(full) * 0.50)))] if full else []
    sentence_text, sentence_meta = first_sentence(caption)
    sentence_tokens = tokenizer_ids(sentence_text)[:CONTENT_LIMIT]
    return (
        {"full": pack_content(full), "prefix50": pack_content(prefix), "first_sentence": pack_content(sentence_tokens)},
        {
            "original_bpe_tokens": len(original),
            "full_kept_bpe_tokens": len(full),
            "prefix50_kept_bpe_tokens": len(prefix),
            "first_sentence_bpe_tokens": len(sentence_tokens),
            "first_sentence_text": sentence_text,
            "first_sentence_boundary": sentence_meta,
        },
    )


def load_records(n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if any(row.get("protocol_id") != "long_dci_reconstructed" for row in rows):
        raise RuntimeError("Manifest contains a protocol other than long_dci_reconstructed")
    valid = [row for row in rows if (IMAGE_ROOT / row["image_path"]).is_file()]
    if len(valid) != len(rows):
        raise RuntimeError(f"Missing Long-DCI images: {len(rows)-len(valid)}")
    if len({row["image_id"] for row in valid}) != len(valid):
        raise RuntimeError("Long-DCI image IDs are not unique")
    rng = random.Random(seed)
    chosen = rng.sample(range(len(valid)), n)
    return [valid[i] for i in chosen]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6)


def finite_report(x: np.ndarray) -> dict:
    finite = np.isfinite(x).all(axis=1)
    norms = np.linalg.norm(x, axis=1)
    return {"rows": int(x.shape[0]), "dims": int(x.shape[1]), "nonfinite_rows": int((~finite).sum()), "finite": bool(finite.all()), "zero_rows": int((norms <= 1e-8).sum()), "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def state_digest(module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_ours(device: torch.device):
    payload = torch.load(OURS_PATH, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    clip, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    module = BalancedSearch(
        clip.float(), arm=cfg["arm"], checkpoint_encoders=False,
        image_chunk=cfg.get("image_chunk", 128), text_chunk=cfg.get("text_chunk", 128),
        condition_mode=cfg.get("condition_mode", "dual_branch"), shuffle_seed=cfg.get("shuffle_seed", 0),
        checkpoint_pair_blocks=cfg.get("checkpoint_pair_blocks", False), fusion=cfg["fusion"], visual=cfg["visual"],
        search_hparams=hparams(cfg), inclusion_hierarchy=cfg.get("inclusion_hierarchy", "detail_chain"),
        view_sparsity_weights=cfg.get("view_sparsity_weights", [1.0, 2.0, 2.0]),
        hns_enabled=cfg.get("hns_enabled", False), hns_beta=cfg.get("hns_beta", [2.0, 2.0]),
    )
    module.clip.load_state_dict(payload["model"], strict=True)
    module.fusion_branch.load_state_dict(payload["adapter"], strict=True)
    module.to(device).eval()
    return module, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this authorized inference run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    for path, expected in ((LONGCLIP_PATH, EXPECTED_LONGCLIP_SHA), (OURS_PATH, EXPECTED_OURS_SHA)):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    records = load_records(args.n, args.seed)
    token_sets = {"full": [], "prefix50": [], "first_sentence": []}
    sample_rows, length_rows = [], []
    for row in records:
        versions, lengths = make_caption_versions(row["caption"])
        for name in token_sets:
            token_sets[name].append(versions[name])
        sample_rows.append({**row, "versions": lengths})
        length_rows.append({"image_id": row["image_id"], **lengths})
    (args.output_dir / "samples.json").write_text(json.dumps(sample_rows, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "caption_lengths.json").write_text(json.dumps(length_rows, indent=2, ensure_ascii=False) + "\n")

    long_model, long_preprocess = longclip.load(str(LONGCLIP_PATH), device=device)
    ours, ours_cfg = build_ours(device)
    initial_digests = {"longclip": state_digest(long_model), "sema_clip": state_digest(ours)}
    precision = {"longclip": sorted({str(p.dtype) for p in long_model.parameters()}), "sema_clip": sorted({str(p.dtype) for p in ours.parameters()})}
    _, ours_preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", args=SimpleNamespace())
    arrays = {f"{model}_{setting}_{modality}": [] for model in ("longclip", "ours_masked", "ours_native") for setting in token_sets for modality in ("image", "text")}
    keep_ratios = {setting: [] for setting in token_sets}
    masks = {setting: [] for setting in token_sets}
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records))
            images = [Image.open(IMAGE_ROOT / row["image_path"]).convert("RGB") for row in records[start:stop]]
            long_images = torch.stack([long_preprocess(im) for im in images]).to(device)
            ours_images = torch.stack([ours_preprocess(im) for im in images]).to(device)
            long_img = normalize(long_model.encode_image(long_images))
            ours_z, ours_visual = ours.encode_visual(ours_images)
            ours_native_img = normalize(ours_z)
            for setting in token_sets:
                tokens = torch.stack(token_sets[setting][start:stop]).to(device)
                long_txt = normalize(long_model.encode_text(tokens))
                ours_txt, ours_condition = ours.encode_view(tokens)
                ours_txt = normalize(ours_txt)
                logits = pair_logits(ours, ours_visual, ours_condition, paired=True)
                mask = hard_st(logits.sigmoid()).float()
                ours_masked_img = normalize(ours_z * mask)
                arrays[f"longclip_{setting}_image"].append(long_img.cpu().numpy())
                arrays[f"longclip_{setting}_text"].append(long_txt.cpu().numpy())
                arrays[f"ours_masked_{setting}_image"].append(ours_masked_img.cpu().numpy())
                arrays[f"ours_masked_{setting}_text"].append(ours_txt.cpu().numpy())
                arrays[f"ours_native_{setting}_image"].append(ours_native_img.cpu().numpy())
                arrays[f"ours_native_{setting}_text"].append(ours_txt.cpu().numpy())
                keep_ratios[setting].extend(mask.mean(-1).cpu().tolist())
                masks[setting].append(mask.cpu().numpy().astype(np.uint8))
            if stop % 160 == 0 or stop == len(records):
                print(json.dumps({"completed_pairs": stop, "elapsed_s": time.perf_counter() - started}), flush=True)

    packed = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}
    np.savez_compressed(args.output_dir / "features.npz", **packed)
    np.savez_compressed(args.output_dir / "mask_arrays.npz", **{f"mask_{k}": np.concatenate(v, axis=0) for k, v in masks.items()})
    final_digests = {"longclip": state_digest(long_model), "sema_clip": state_digest(ours)}
    assert final_digests == initial_digests, "Inference unexpectedly changed model state"
    for path, expected in ((LONGCLIP_PATH, EXPECTED_LONGCLIP_SHA), (OURS_PATH, EXPECTED_OURS_SHA)):
        assert sha256(path) == expected, "Checkpoint changed during inference"
    assert all(finite_report(value)["finite"] and not finite_report(value)["zero_rows"] for value in packed.values())
    first_fallback = sum(bool(row["versions"]["first_sentence_boundary"]["fallback_full_caption"]) for row in sample_rows)
    provenance = {
        "task": "Long-DCI Modality Gap PCA",
        "seed": args.seed, "sample_count": len(records), "sample_selection": "random.Random(seed).sample over valid manifest rows",
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "manifest_total_rows": 7602,
        "image_root": str(IMAGE_ROOT), "unique_image_ids": len({r["image_id"] for r in records}),
        "protocol_id": "long_dci_reconstructed", "checkpoint_sha256": {"longclip": sha256(LONGCLIP_PATH), "sema_clip": sha256(OURS_PATH)},
        "checkpoint_paths": {"longclip": str(LONGCLIP_PATH), "sema_clip": str(OURS_PATH)},
        "caption_rule": {"context": 248, "content_limit": CONTENT_LIMIT, "full": "BPE tokens truncated to 246 content tokens", "prefix50": "ceil(0.5 * effective Full content tokens)", "first_sentence": "natural punctuation boundary; abbreviation and decimal checks; fallback explicitly recorded"},
        "caption_lengths": {"full_mean": float(np.mean([r["versions"]["full_kept_bpe_tokens"] for r in sample_rows])), "prefix50_mean": float(np.mean([r["versions"]["prefix50_kept_bpe_tokens"] for r in sample_rows])), "first_sentence_mean": float(np.mean([r["versions"]["first_sentence_bpe_tokens"] for r in sample_rows])), "first_sentence_fallback_count": first_fallback, "first_sentence_fallback_fraction": first_fallback / len(sample_rows)},
        "models": {"longclip": "official Long-CLIP native image/text", "ours_masked": "BalancedSearch encode_visual + paired hard-ST BalancedStack mask", "ours_native": "Sema-CLIP unmasked normalized image/text"},
        "ours_config": ours_cfg,
        "precision": precision, "autocast": False,
        "preprocess": {"longclip": repr(long_preprocess), "sema_clip": repr(ours_preprocess)},
        "model_state_sha256_before": initial_digests, "model_state_sha256_after": final_digests,
        "elapsed_s": time.perf_counter() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "feature_checks": {key: finite_report(value) for key, value in packed.items()},
        "mask_keep_ratio": {key: {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} for key, vals in keep_ratios.items()},
        "features_sha256": sha256(args.output_dir / "features.npz"),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample_count": len(records), "features_sha256": provenance["features_sha256"], "mask_keep_ratio": provenance["mask_keep_ratio"], "first_sentence_fallback_count": first_fallback}, indent=2))


if __name__ == "__main__":
    main()
