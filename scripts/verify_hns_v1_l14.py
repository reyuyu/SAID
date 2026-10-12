#!/usr/bin/env python3
"""Strictly verify the L14 release files and checkpoint metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from model.backbone import load_native_state

EXPECTED = {
    "student": "4a1e5ed69b7aa728d7189518b9f285052e8293290963e079d886ddcab5fe819e",
    "training": "60325144f586394ca75f7cc8fcd3432f777bc8a17324e1e76d84f7120f34d8bb",
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
    result["student"] = {"strict_load": True, "backbone": spec["base_model"],
                          "embedding_dimension": int(model.text_projection.shape[1]),
                          "context_length": int(model.context_length),
                          "visual_patch_count": int(model.visual.positional_embedding.shape[0] - 1)}
    if paths["training"]:
        payload = torch.load(paths["training"], map_location="cpu", weights_only=False)
        if int(payload.get("completed_steps", -1)) != 4868:
            raise SystemExit(f"training completed_steps={payload.get('completed_steps')}, expected 4868")
        if "optimizer" not in payload or "scheduler" not in payload:
            raise SystemExit("training checkpoint lacks optimizer or scheduler state")
        result["training"] = {"completed_steps": payload["completed_steps"],
                               "has_optimizer": True, "has_scheduler": True,
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
