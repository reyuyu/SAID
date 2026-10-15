"""Strictly extract a ViT-L/14 bare student state from a full checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from model import longclip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expect-updates", type=int, required=True)
    args = p.parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["completed_steps"] == args.expect_updates
    model, _ = longclip.load_from_clip("ViT-L/14", device="cpu", args=argparse.Namespace())
    model.load_state_dict(payload["model"], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(json.dumps({"checkpoint": str(checkpoint), "output": str(output),
                      "checkpoint_sha256": sha256(checkpoint),
                      "bare_sha256": sha256(output), "strict_load": True}, indent=2))


if __name__ == "__main__":
    main()
