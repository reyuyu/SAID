"""Prepare small, non-secret ModelScope release manifests and README files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments/nest_clip_v1/best_checkpoint_sweep_v1"
OUT = ROOT / "docs/modelscope/best_checkpoint_release_v1"
RUNTIME = Path("/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1")


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def model_record(backbone: str, step: int) -> dict:
    sweep = read_json(SWEEP / "CHECKPOINT_SWEEP_RESULTS.json")
    row = sweep["backbones"][backbone][str(step)]
    return {
        "backbone": backbone,
        "step": step,
        "scheduler_horizon": row["scheduler_horizon"],
        "checkpoint_path": row["checkpoint"],
        "bare_student_path": row["bare"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "bare_student_sha256": row["bare_sha256"],
        "source_branch": "experiment/nested-d3-hns-full-v1" if backbone == "ViT-B/16" else "experiment/hns-v1-l14-full5epoch6085-fromscratch-v1",
        "source_commit": "45468e1" if backbone == "ViT-B/16" else "8a0b6e8",
        "summary_percent": row["summary_percent"],
        "flickr30k_full": sweep["flickr30k_full"]["results"][f"{backbone}@{step}"],
    }


def readme(rec: dict, repo_id: str) -> str:
    b = rec["backbone"]
    config_name = "ViT-B/16" if b == "ViT-B/16" else "ViT-L/14"
    s = rec["summary_percent"]
    f = rec["flickr30k_full"]
    return f"""# SAID HNS-v1 {config_name} best checkpoint

This private ModelScope repository contains the exploratory best checkpoint selected from the verified SAID HNS-v1 checkpoint sweep.

- Backbone: `{config_name}`
- Training step: `{rec['step']}`
- Scheduler horizon: `{rec['scheduler_horizon']}`
- Source branch: `{rec['source_branch']}`
- Source commit: `{rec['source_commit']}`
- Training data: ShareGPT4V-1M, with the audited nested-detail D3 sampling protocol
- Text context: 248 tokens, native SAID tokenizer

The repository contains both the strict bare CLIP student for inference and the full training checkpoint for exact recovery. The bare student is the file to use for native retrieval inference.

## Inference

Inference uses native CLIP image and text embeddings, L2 normalization, and cosine similarity over the candidate pool. **Mask, Fusion, reranking, ensemble, and TTA are not required at inference time.**

```python
import torch
from model import longclip

model, preprocess = longclip.load_from_clip("{config_name}", device="cuda")
state = torch.load("bare_student.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

Use the corresponding SAID repository at commit `{rec['source_commit']}` for the tokenizer and preprocessing implementation.

## Audited retrieval results (percent)

| Metric | Value |
|---|---:|
| Score5 | {s['Score5']:.6f} |
| J_long3 | {s['J_long3']:.6f} |
| J_long | {s['J_long']:.6f} |
| Short4 | {s['Short4']:.6f} |
| Urban I2T R@1 | {s['Urban_I2T']:.6f} |
| Urban T2I R@1 | {s['Urban_T2I']:.6f} |
| Flickr30k-Full I2T R@1/R@5/R@10 | {' / '.join(f'{x:.6f}' for x in f['I2T'])} |
| Flickr30k-Full T2I R@1/R@5/R@10 | {' / '.join(f'{x:.6f}' for x in f['T2I'])} |

The public benchmark selection is exploratory and is not an independent validation-set model choice.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = {
        "ViT-B/16": model_record("ViT-B/16", 3651),
        "ViT-L/14": model_record("ViT-L/14", 4651),
    }
    repo_names = {
        "ViT-B/16": "SAID-HNS-v1-ViT-B16-BestStep3651",
        "ViT-L/14": "SAID-HNS-v1-ViT-L14-BestStep4651",
    }
    for backbone, rec in records.items():
        d = OUT / ("b16" if backbone == "ViT-B/16" else "l14")
        d.mkdir(exist_ok=True)
        (d / "README.md").write_text(readme(rec, repo_names[backbone]))
        (d / "MODEL_CONFIG.json").write_text(json.dumps({
            "model": backbone, "context_length": 248, "native_inference": True,
            "requires_mask_at_inference": False, "requires_fusion_at_inference": False,
            "source_commit": rec["source_commit"], "scheduler_horizon": rec["scheduler_horizon"],
            "step": rec["step"],
        }, indent=2) + "\n")
        (d / "SOURCE_MANIFEST.json").write_text(json.dumps(rec, indent=2) + "\n")
        (d / "SHA256SUMS.txt").write_text(
            f"{rec['bare_student_sha256']}  bare_student.pt\n{rec['checkpoint_sha256']}  training_checkpoint.pt\n"
        )
    report = {"status": "PREPARED_AUTH_REQUIRED", "account_display": "hongchenyang0814", "repositories": repo_names,
              "models": records, "token_source": "MODELSCOPE_API_TOKEN", "token_embedded": False,
              "flickr30k_full_used_for_selection": True}
    (OUT / "RELEASE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "RELEASE_REPORT.md").write_text("""# SAID best-checkpoint ModelScope release\n\n""" +
        "ModelScope publication is prepared for two private repositories. Authentication must come from `MODELSCOPE_API_TOKEN`; no token is stored in this repository. The selected nodes use the completed Flickr30k-Full results in the checkpoint sweep.\n\n" +
        "| Backbone | Step | Score5 | Checkpoint SHA256 | Bare SHA256 | Repository |\n|---|---:|---:|---|---|---|\n" +
        "\n".join(f"| {b} | {r['step']} | {r['summary_percent']['Score5']:.6f} | `{r['checkpoint_sha256']}` | `{r['bare_student_sha256']}` | `{repo_names[b]}` |" for b, r in records.items()) + "\n")


if __name__ == "__main__":
    main()
