#!/usr/bin/env python3
"""Strictly verify the L14 release files and checkpoint metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.backbone import load_native_state
from model import longclip

EXPECTED = {
    "student": "f360465fc02a5465d312c672d5179dcd8991d9c98ffd14325831b9150a8f7da1",
    "training": "8a036746e44a789e69d40987cf6852ee9603b6881e30ef30ab88c8463bcec538",
    "step0": "8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--training")
    p.add_argument("--step0")
    p.add_argument("--output")
    a = p.parse_args()
    torch.set_num_threads(4)
    paths = {"student": Path(a.student), "training": Path(a.training) if a.training else None,
             "step0": Path(a.step0) if a.step0 else None}
    result = {"passed": True, "files": {}}
    for name, path in paths.items():
        if path is None:
            continue
        got = sha256(path)
        if got != EXPECTED[name]:
            raise SystemExit(f"{name} SHA256 mismatch: {got} != {EXPECTED[name]}")
        result["files"][name] = {"path": str(path), "sha256": got}
    state = torch.load(paths["student"], map_location="cpu", weights_only=True)
    model, _, spec = load_native_state(state, base_model="ViT-L/14", device="cpu")
    tokens = longclip.tokenize(["A red cat sits on a chair. " * 50], context_length=248, truncate=True)
    assert tokens.shape == (1, 248) and tokens[0, -1] == longclip._tokenizer.encoder["<|endoftext|>"]
    with torch.no_grad():
        features = model.encode_text(tokens).float()
    assert features.shape == (1, 768) and torch.isfinite(features).all()
    result["student"] = {"strict_load": True, "backbone": spec["base_model"],
                          "embedding_dimension": int(model.text_projection.shape[1]),
                          "context_length": int(model.context_length),
                          "visual_patch_count": int(model.visual.positional_embedding.shape[0] - 1),
                          "long_text_token_shape": list(tokens.shape),
                          "long_text_embedding_shape": list(features.shape),
                          "long_text_content_tokens": 246}
    if paths["training"]:
        payload = torch.load(paths["training"], map_location="cpu", weights_only=False)
        if int(payload.get("completed_steps", -1)) != 6085:
            raise SystemExit(f"training completed_steps={payload.get('completed_steps')}, expected 6085")
        if "optimizer" not in payload or "scheduler_horizon" not in payload:
            raise SystemExit("training checkpoint lacks optimizer or scheduler horizon state")
        assert payload["scheduler_horizon"] == 6085
        steps = sorted({int(v["step"]) for v in payload["optimizer"]["state"].values() if "step" in v})
        assert steps == [6085], steps
        assert "rng_per_rank" in payload and "next_epoch" in payload and "next_batch" in payload
        result["training"] = {"completed_steps": payload["completed_steps"],
                               "has_optimizer": True,
                               "scheduler_horizon": payload["scheduler_horizon"],
                               "optimizer_step_counters": steps,
                               "has_rng_and_cursor": True,
                               "has_serialized_scheduler_object": "scheduler" in payload,
                               "top_level_keys": sorted(payload.keys())}
    if paths["step0"]:
        payload = torch.load(paths["step0"], map_location="cpu", weights_only=False)
        result["step0"] = {"completed_steps": payload.get("completed_steps", 0),
                            "top_level_keys": sorted(payload.keys())}
    text = json.dumps(result, indent=2, sort_keys=True)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
