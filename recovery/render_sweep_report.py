"""Render the completed checkpoint sweep report from frozen result receipts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/nest_clip_v1/best_checkpoint_sweep_v1"
RUNTIME = Path("/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/best-checkpoint-sweep-v1")
L14 = Path("/opt/data/private/lklk/said-e2-worktrees/hns-v1-l14-full5epoch6085-fromscratch-v1")
L14_RELEASE = Path("/opt/data/private/lklk/said-release-l14")


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(x: float) -> str:
    return f"{x:.6f}"


def add_full_results(results: dict) -> None:
    full = {}
    old = L14 / "experiments/nest_clip_v1/hns_v1_l14_full5epoch6085_fromscratch_v1/FLICKR_FULL_RESULTS.json"
    if not old.exists():
        old = L14_RELEASE / "experiments/nest_clip_v1/hns_v1_l14_full5epoch6085_fromscratch_v1/FLICKR_FULL_RESULTS.json"
    if old.exists():
        raw = json.loads(old.read_text())
    else:
        raw = json.loads(subprocess.check_output([
            "git", "-C", str(L14), "show",
            "8a0b6e8:experiments/nest_clip_v1/hns_v1_l14_full5epoch6085_fromscratch_v1/FLICKR_FULL_RESULTS.json",
        ], text=True))
    if raw:
        model_rows = raw.get("models", raw)
        for step in (4868, 6085):
            key = f"step{step}"
            if key in model_rows:
                rec = model_rows[key]
                if "metrics" in rec:
                    rec = {"checkpoint_sha256": rec.get("checkpoint_sha256"),
                           "I2T": [rec["metrics"]["I2T"][f"R@{k}"] for k in (1, 5, 10)],
                           "T2I": [rec["metrics"]["T2I"][f"R@{k}"] for k in (1, 5, 10)],
                           "source": "git:8a0b6e8"}
                full[f"ViT-L/14@{step}"] = rec
    runtime = RUNTIME / "L14_4651_flickr_full/RESULT.json"
    if runtime.exists():
        raw = json.loads(runtime.read_text())
        m = raw["metrics"]
        full["ViT-L/14@4651"] = {
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "I2T": [raw["metrics"]["I2T"]["recall_percent"][str(k)] for k in (1, 5, 10)],
            "T2I": [raw["metrics"]["T2I"]["recall_percent"][str(k)] for k in (1, 5, 10)],
            "source": str(runtime),
        }
    b16_runtime = RUNTIME / "B16_3651_flickr_full/RESULT.json"
    if b16_runtime.exists():
        raw = json.loads(b16_runtime.read_text())
        full["ViT-B/16@3651"] = {
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "I2T": [raw["metrics"]["I2T"]["recall_percent"][str(k)] for k in (1, 5, 10)],
            "T2I": [raw["metrics"]["T2I"]["recall_percent"][str(k)] for k in (1, 5, 10)],
            "source": str(b16_runtime),
        }
    results["flickr30k_full"] = {
        "protocol": "Flickr30k-Full-official-token-31783x158915",
        "manifest_sha256": "bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc",
        "image_content_sha256": "e4fef70dd3f73034970b1ec0e20d81a4b421d1871e350e3266e41b744d3ab159",
        "results": full,
        "B16_3651_status": "COMPLETED_WITH_SWEEP_NATIVE_B16_EVALUATOR",
    }


def summary_table(rows: dict) -> str:
    lines = ["| Step | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in sorted(rows, key=int):
        s = rows[step]["summary_percent"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            step, *(pct(s[k]) for k in ("Score5", "J_long3", "J_long", "Short4", "Urban_I2T", "Urban_T2I", "Urban_Mean"))))
    return "\n".join(lines)


def dataset_tables(rows: dict) -> str:
    out = []
    for step in sorted(rows, key=int):
        out.append(f"### Step {step}\n")
        out.append("| Dataset | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in rows[step]["metrics"].items():
            out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                name, *(pct(m[d][f"R@{k}"]) for d, k in (("I2T", 1), ("I2T", 5), ("I2T", 10), ("T2I", 1), ("T2I", 5), ("T2I", 10)))))
        out.append("")
    return "\n".join(out)


def main() -> None:
    results = json.loads((OUT / "CHECKPOINT_SWEEP_RESULTS.json").read_text())
    add_full_results(results)
    results["status"] = "COMPLETED"
    results["selection"] = "exploratory public benchmark selection; no independent validation set was used"
    (OUT / "CHECKPOINT_SWEEP_RESULTS.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    b = results["backbones"]["ViT-B/16"]
    l = results["backbones"]["ViT-L/14"]
    report = [
        "# SAID HNS-v1 checkpoint sweep",
        "",
        "This report compares checkpoints from the verified B/16 (scheduler horizon 4868) and L/14 (scheduler horizon 6085) trajectories. All retrieval numbers are native image/text embedding results on the fixed five-set protocols. Checkpoint selection is exploratory public-test selection; no independent validation set was used, so the selected score is not an unbiased generalization estimate.",
        "",
        "## ViT-B/16 summary",
        "",
        summary_table(b),
        "",
        "The best B/16 Score5, J_long3, J_long, Short4, and Urban mean are all at step 3651. Relative to the paper checkpoint (step 3651), later nodes decline in Score5 by 0.078779 pp (4151), 0.131818 pp (4651), and 0.136334 pp (4868).",
        "",
        "## ViT-L/14 summary",
        "",
        summary_table(l),
        "",
        "The best L/14 Score5 is step 4651 (78.003728%). The best J_long3 is also step 4651 (82.427547%), while the best J_long is step 6085 (90.775002%). Step 5651 is second on J_long (90.765002%). The Score5 winner is therefore the recommended exploratory checkpoint; step 6085 remains the long-text specialist.",
        "",
        "## Complete per-dataset tables",
        "",
        "The following tables include all R@1/R@5/R@10 values used to compute the summaries.",
        "",
        "## B/16",
        "",
        dataset_tables(b),
        "## L/14",
        "",
        dataset_tables(l),
        "## Flickr30k-Full",
        "",
        "The fixed 31,783-image/158,915-caption protocol uses manifest SHA256 `bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc`. Existing verified results are included for L/14@4868 and @6085; L/14@4651 and B/16@3651 were evaluated in this sweep with native bare-student evaluators.",
        "",
        "| Checkpoint | I2T R@1/5/10 | T2I R@1/5/10 |",
        "|---|---|---|",
    ]
    for key, rec in results["flickr30k_full"]["results"].items():
        report.append(f"| {key} | {' / '.join(pct(x) for x in rec['I2T'])} | {' / '.join(pct(x) for x in rec['T2I'])} |")
    report += [
        "",
        "## Audit and limitations",
        "",
        "All newly generated continuation segments passed four-rank parameter synchronization, NCCL all-reduce, finite-gradient, and checkpoint acceptance checks. L/14 stream audits for 3651→4151 and 4151→4651 passed with 500 updates and 512,000 sample positions per segment. B/16 stream digests and all shared sampling fields matched the historical observer; the historical observer contains one additive `lowest_selected_sentence_indices_sha256` field absent from the newer observer schema, so B/16 is reported as shared-field/digest matched rather than byte-identical observer dictionaries.",
        "",
        "No training or evaluation process remains after completion. Public benchmark selection and repeated Urban/Long-DCI observation limit the scientific interpretation; an independent held-out validation protocol is still needed before claiming a final model choice.",
        "",
        "## Recommended checkpoints",
        "",
        "- B/16: step 3651, checkpoint SHA256 `83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07`.",
        "- L/14: step 4651 by Score5/J_long3, checkpoint SHA256 `9a773bfb54e3dde33ae724e4e3a13a0614381050ac5b9aa29939d54198c1639b`.",
        "- L/14 long-text specialist: step 6085, checkpoint SHA256 `8a036746e44a789e69d40987cf6852ee9603b6881e30ef30ab88c8463bcec538`.",
    ]
    (OUT / "CHECKPOINT_SWEEP_REPORT.md").write_text("\n".join(report) + "\n")

    best = {
        "selection_policy": "Score5 primary; J_long3, J_long, Urban, Flickr30k-Full, and health as secondary evidence",
        "exploratory_public_test_selection": True,
        "ViT-B/16": {"score5": {"step": 3651, "checkpoint_sha256": b["3651"]["checkpoint_sha256"], "score5": b["3651"]["summary_percent"]["Score5"]},
                      "long_text": {"step": 3651, "J_long3": b["3651"]["summary_percent"]["J_long3"], "J_long": b["3651"]["summary_percent"]["J_long"]},
                      "recommended": {"step": 3651, "reason": "best Score5 and all secondary summary metrics"}},
        "ViT-L/14": {"score5": {"step": 4651, "checkpoint_sha256": l["4651"]["checkpoint_sha256"], "score5": l["4651"]["summary_percent"]["Score5"]},
                      "long_text": {"step": 6085, "J_long3": l["6085"]["summary_percent"]["J_long3"], "J_long": l["6085"]["summary_percent"]["J_long"]},
                      "recommended": {"step": 4651, "reason": "best Score5 and J_long3; step 6085 retained as J_long specialist"}},
    }
    best_md = [
        "# Recommended checkpoints", "",
        "| Backbone | Score5 checkpoint | Score5 | Long-text checkpoint | J_long3 | J_long | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| ViT-B/16 | 3651 | {pct(b['3651']['summary_percent']['Score5'])} | 3651 | {pct(b['3651']['summary_percent']['J_long3'])} | {pct(b['3651']['summary_percent']['J_long'])} | step3651 |",
        f"| ViT-L/14 | 4651 | {pct(l['4651']['summary_percent']['Score5'])} | 6085 | {pct(l['6085']['summary_percent']['J_long3'])} | {pct(l['6085']['summary_percent']['J_long'])} | step4651 for Score5/J_long3; step6085 for J_long |",
        "",
        "The selection is exploratory and uses repeatedly observed public benchmarks; it is not a held-out validation choice.",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
    ]
    (OUT / "BEST_CHECKPOINTS.md").write_text("\n".join(best_md) + "\n")

    provenance = {
        "branch": "experiment/said-hns-best-checkpoint-sweep-v1",
        "source_trajectories": {
            "ViT-B/16": {"branch": "experiment/nested-d3-hns-full-v1", "commit": "45468e1", "scheduler_horizon": 4868},
            "ViT-L/14": {"branch": "experiment/hns-v1-l14-full5epoch6085-fromscratch-v1", "commit": "8a0b6e8", "scheduler_horizon": 6085},
        },
        "checkpoint_records": {bname: {step: {k: row.get(k) for k in ("checkpoint_sha256", "bare_sha256", "checkpoint", "bare", "scheduler_horizon", "source")} for step, row in rows.items()} for bname, rows in results["backbones"].items()},
        "runtime_root": str(RUNTIME),
        "stream_audits": {
            "L14_3651_to_4151": str(RUNTIME / "L14_4151_exact/sweep-stream-audit.json"),
            "L14_4151_to_4651": str(RUNTIME / "L14_4651_exact/sweep-stream-audit.json"),
            "B16": {"status": "shared_fields_and_stream_digest_match", "schema_note": "historical observer has additive lowest_selected_sentence_indices_sha256"},
        },
        "acceptance_receipts": {
            name: {
                "path": str(RUNTIME / name / "acceptance.json"),
                "passed": json.loads((RUNTIME / name / "acceptance.json").read_text())["passed"],
            }
            for name in ("B16_4151", "B16_4651", "L14_4151_exact", "L14_4651_exact", "L14_5151_exact", "L14_5651_exact")
        },
        "evaluators": {
            "L14_native_sha256": sha(L14 / "recovery/eval_l14_native.py"),
            "L14_flickr_full_sha256": sha(Path("/opt/data/private/lklk/said-l14-dci-full-eval-v1/recovery/eval_l14_bounded.py")),
            "B16_flickr_full_sha256": sha(ROOT / "recovery/eval_b16_flickr_full.py"),
        },
        "flickr30k_full": results["flickr30k_full"],
        "training_processes_remaining": False,
        "gpu_idle_after_completion": True,
    }
    (OUT / "CHECKPOINT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
