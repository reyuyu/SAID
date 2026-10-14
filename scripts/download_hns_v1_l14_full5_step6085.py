#!/usr/bin/env python3
"""Download and verify the private SAID HNS-v1 ViT-L/14 release."""
import argparse
import hashlib
import subprocess
from pathlib import Path

REPO = "hongchenyang/SAID-HNS-v1-ViT-L14-Full5-Step6085"
FILES = {
    "inference/student_step6085.pt": "f360465fc02a5465d312c672d5179dcd8991d9c98ffd14325831b9150a8f7da1",
    "training/hns_v1_l14_step6085_full.pt": "8a036746e44a789e69d40987cf6852ee9603b6881e30ef30ab88c8463bcec538",
    "training/common_step000000.pt": "8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="checkpoints/hns_v1_l14")
    p.add_argument("--revision", default=None)
    p.add_argument("--student-only", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    wanted = {k: v for k, v in FILES.items() if not args.student_only or k.startswith("inference/")}
    for remote, expected in wanted.items():
        target = out / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ms-hub", "download", REPO, remote, "--local-dir", str(out)]
        if args.revision:
            cmd += ["--revision", args.revision]
        subprocess.run(cmd, check=True)
        if not target.is_file():
            raise SystemExit(f"download did not create {target}")
        got = sha256(target)
        if got != expected:
            raise SystemExit(f"SHA256 mismatch for {remote}: {got} != {expected}")
        print(f"verified {remote}: {got}")

if __name__ == "__main__":
    main()
