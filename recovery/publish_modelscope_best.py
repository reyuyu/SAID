"""Publish and download-verify the two private ModelScope best checkpoints.

The token is read only from MODELSCOPE_API_TOKEN. This script refuses to use a
CLI-persisted token or a token supplied in source/configuration files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "docs/modelscope/best_checkpoint_release_v1"
RUNTIME = Path("/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1")


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, *, capture=False):
    env = dict(os.environ)
    token = env.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError("MODELSCOPE_API_TOKEN is not configured")
    env.pop("MS_TOKEN", None)
    return subprocess.run(cmd, env=env, text=True, check=True, capture_output=capture)


def whoami() -> tuple[str, str]:
    p = run(["modelscope", "whoami"], capture=True)
    text = (p.stdout or "") + (p.stderr or "")
    if "hongchenyang0814" not in text:
        raise RuntimeError("whoami did not verify the requested account display name")
    # Current CLI formats include username/name fields; accept JSON or a simple field.
    try:
        obj = json.loads(p.stdout.strip())
        for key in ("username", "user_name", "namespace", "owner", "name"):
            if isinstance(obj.get(key), str) and obj[key]:
                return obj[key], text
    except Exception:
        pass
    m = re.search(r"(?:username|user[_ ]?name|namespace|owner)\s*[:=]\s*([A-Za-z0-9_.-]+)", text, re.I)
    if not m:
        raise RuntimeError("could not parse writable namespace from whoami output")
    return m.group(1), text


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--verify-dir", default="/opt/data/private/lklk/modelscope_verify_best"); args = p.parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit("BLOCKED_AUTH: set MODELSCOPE_API_TOKEN; no token was read from chat, files, or persisted CLI state")
    namespace, who = whoami()
    manifest = json.loads((RELEASE / "RELEASE_MANIFEST.json").read_text())
    verify_root = Path(args.verify_dir); verify_root.mkdir(parents=True, exist_ok=True)
    receipts = {"namespace": namespace, "whoami": who, "token_embedded": False, "repositories": {}}
    for backbone, model in manifest["models"].items():
        short = "b16" if backbone == "ViT-B/16" else "l14"
        repo = f"{namespace}/{manifest['repositories'][backbone]}"
        small = RELEASE / short
        run(["modelscope", "create", "--repo-type", "model", "--visibility", "private", "--exist-ok", repo])
        run(["modelscope", "upload", repo, str(small), ".", "--repo-type", "model", "--use-cache", "--commit-message", "Add audited release metadata"])
        for local, remote in ((model["bare_student_path"], "bare_student.pt"), (model["checkpoint_path"], "training_checkpoint.pt")):
            run(["modelscope", "upload", repo, local, remote, "--repo-type", "model", "--use-cache", "--commit-message", f"Add {remote}"])
        downloaded = verify_root / short
        if downloaded.exists(): shutil.rmtree(downloaded)
        run(["modelscope", "download", repo, "--repo-type", "model", "--local-dir", str(downloaded), "--force"])
        checks = {"bare_student.pt": model["bare_student_sha256"], "training_checkpoint.pt": model["checkpoint_sha256"]}
        actual = {name: sha(downloaded / name) for name in checks}
        if actual != checks:
            raise RuntimeError(f"SHA mismatch for {repo}: expected={checks} actual={actual}")
        receipts["repositories"][backbone] = {"repo_id": repo, "url": f"https://modelscope.cn/models/{repo}", "sha256": actual, "download_dir": str(downloaded)}
    (RELEASE / "MODELSCOPE_UPLOAD_VERIFY.json").write_text(json.dumps(receipts, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipts, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
