#!/usr/bin/env python3
"""Download and verify the private SAID HNS-v1 ViT-L/14 release."""
import argparse
import hashlib
import subprocess
from pathlib import Path

REPO = "hongchenyang/SAID-HNS-v1-ViT-L14-Full4-Step4868"
FILES = {
    "inference/student_step4868.pt": "4a1e5ed69b7aa728d7189518b9f285052e8293290963e079d886ddcab5fe819e",
    "training/hns_v1_l14_step4868_full.pt": "60325144f586394ca75f7cc8fcd3432f777bc8a17324e1e76d84f7120f34d8bb",
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
