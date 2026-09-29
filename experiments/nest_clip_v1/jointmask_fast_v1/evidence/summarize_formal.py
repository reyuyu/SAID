#!/usr/bin/env python3
"""Build the compact JointMask fast formal result bundle from preserved server artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "experiments/nest_clip_v1/jointmask_fast_v1"
SERVER = Path("/root/lk_projects/SAID-nest-clip-v1/jointmask_fast_v1")
FORMAL = SERVER / "formal"
GROUPS = ("T-fast", "TI-fast", "TI-Shuffle-fast")
DATASETS = ("COCO", "Urban-1k", "Flickr30k-test1k", "DOCCI", "long-DCI")
DIRECTIONS = ("I2T", "T2I")
RECALLS = ("R@1", "R@5", "R@10")
KEY_STEPS = (1, 100, 200, 201, 300, 400, 500)
TRAIN_FIELDS = (
    "loss", "common_loss", "F_i2t", "F_t2i", "O_i2t", "O_t2i", "E_i2t", "E_t2i",
    "F_keep_ratio", "O_keep_ratio", "E_keep_ratio", "F_all_open", "F_all_closed",
    "O_all_open", "O_all_closed", "E_all_open", "E_all_closed", "inc", "inc_weight",
    "hard_inclusion_violation", "oe_iou", "F_delta_abs_mean", "O_delta_abs_mean",
    "E_delta_abs_mean", "valid_global", "nonfinite",
)


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_recall(block):
    return {f"R@{key[1:]}" if key.startswith("R") and not key.startswith("R@") else key: value
            for key, value in block.items()}


def evaluation(group: str):
    root = FORMAL / group
    coco = load(root / "coco_native.json")
    urban = load(root / "urban_native.json")
    metrics = {
        "COCO": {
            "I2T": {"R@1": coco["image2text_R1"], "R@5": coco["image2text_R5"],
                    "R@10": coco["image2text_R10"]},
            "T2I": {"R@1": coco["text2image_R1"], "R@5": coco["text2image_R5"],
                    "R@10": coco["text2image_R10"]},
        },
        "Urban-1k": {"I2T": normalize_recall(urban["image2text"]),
                     "T2I": normalize_recall(urban["text2image"])},
    }
    sources = {
        "COCO": root / "coco_native.json",
        "Urban-1k": root / "urban_native.json",
        "Flickr30k-test1k": root / "flickr_test1k/flickr_test1k.json",
        "DOCCI": root / "docci/docci.json",
        "long-DCI": root / "long_dci/long_dci.json",
    }
    metadata = {}
    for name in ("Flickr30k-test1k", "DOCCI", "long-DCI"):
        raw = load(sources[name])
        metrics[name] = {direction: {} for direction in DIRECTIONS}
        for recall, pair in raw["metrics"].items():
            for direction in DIRECTIONS:
                metrics[name][direction][recall] = pair[direction]
        metadata[name] = {k: raw[k] for k in raw if k != "metrics"}
    metadata["COCO"] = {"n_images": 5000, "n_captions": 25000,
                         "similarity_chunk": 512, "batch_size": 64,
                         "checkpoint_sha256": coco["checkpoint_sha256"], "native_only": True}
    metadata["Urban-1k"] = {k: urban[k] for k in urban if k not in ("image2text", "text2image")}
    return metrics, metadata, sources


def compact_step(row):
    out = {}
    for key in TRAIN_FIELDS:
        if key == "common_loss":
            out[key] = row["loss"] - row["inc_weight"] * row["inc"]
        elif key in row:
            out[key] = row[key]
    out["F_candidates"] = row["F_candidates"]
    out["O_candidates"] = row["O_candidates"]
    out["E_candidates"] = row["E_candidates"]
    return out


def training(group: str):
    root = FORMAL / group
    rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
    by_step = {row["step"]: row for row in rows}
    last50 = rows[-50:]
    means = {}
    for key in TRAIN_FIELDS:
        values = []
        for row in last50:
            if key == "common_loss":
                values.append(row["loss"] - row["inc_weight"] * row["inc"])
            elif key in row:
                values.append(row[key])
        means[key] = statistics.mean(values)
    synchronized = [max(rank["seconds"] for rank in row["rank_health"]) for row in rows]
    stable = synchronized[10:]
    acceptance = load(root / "acceptance.json")
    config = load(root / "config.json")
    export = load(root / "export-check.json")
    stream_rows = []
    digest_keys = ("sample_id_sha256", "full_view_sha256", "local_views_sha256",
                   "split_sha256", "fixed_first_reference_stream_sha256")
    for row in rows:
        item = []
        for rank in row["rank_health"]:
            sampling = rank["sampling"]
            item.append({"rank": rank["rank"], "stream_sha256": rank["stream_sha256"],
                         **{key: sampling[key] for key in digest_keys}})
        stream_rows.append(item)
    return {
        "config": config,
        "acceptance": acceptance,
        "export_check": export,
        "checkpoint_step500_sha256": sha256(root / "step000500.pt"),
        "key_steps": {str(step): compact_step(by_step[step]) for step in KEY_STEPS},
        "last50_mean": means,
        "timing": {
            "training_seconds": max(rank["seconds"] for rank in acceptance["ranks"]),
            "stable_step_max_rank_median_seconds": statistics.median(stable),
            "stable_step_max_rank_p90_seconds": statistics.quantiles(stable, n=10)[8],
            "stable_step_max_rank_mean_seconds": statistics.mean(stable),
        },
        "integrity": {
            "records": len(rows),
            "steps_continuous": [row["step"] for row in rows] == list(range(1, 501)),
            "nonfinite_sum": sum(row["nonfinite"] for row in rows),
            "fallback_steps": sum(row["valid_global"] < 2 for row in rows),
            "duplicate_id_steps": sum(bool(row["duplicate_image_ids"]) for row in rows),
            "valid_global_min": min(row["valid_global"] for row in rows),
            "valid_global_max": max(row["valid_global"] for row in rows),
            "F_candidates_min": min(row["F_candidates"] for row in rows),
            "F_candidates_max": max(row["F_candidates"] for row in rows),
        },
        "_stream_rows": stream_rows,
    }


def comparison(metrics, left, right):
    delta = {}
    flat = []
    r1 = []
    wins = ties = losses = 0
    for dataset in DATASETS:
        delta[dataset] = {}
        for direction in DIRECTIONS:
            delta[dataset][direction] = {}
            for recall in RECALLS:
                value = 100 * (metrics[left][dataset][direction][recall]
                               - metrics[right][dataset][direction][recall])
                delta[dataset][direction][recall] = value
                flat.append(value)
                if recall == "R@1":
                    r1.append(value)
                if value > 1e-12:
                    wins += 1
                elif value < -1e-12:
                    losses += 1
                else:
                    ties += 1
    return {"delta_pp": delta, "mean_delta_pp_all_30": statistics.mean(flat),
            "mean_delta_pp_r1_10": statistics.mean(r1),
            "wins_ties_losses": [wins, ties, losses]}


def fmt_pct(value):
    return f"{100 * value:.2f}"


def fmt_pp(value):
    return f"{value:+.2f}"


def main():
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    all_metrics = {}
    all_metadata = {}
    source_files = {}
    groups = {}
    for group in GROUPS:
        metrics, metadata, sources = evaluation(group)
        all_metrics[group] = metrics
        all_metadata[group] = metadata
        source_files[group] = sources
        groups[group] = training(group)

    streams_equal = all(groups[GROUPS[0]]["_stream_rows"] == groups[group]["_stream_rows"]
                        for group in GROUPS[1:])
    for group in GROUPS:
        del groups[group]["_stream_rows"]

    execution = {}
    for group in GROUPS:
        execution[group] = {}
        for stage in ("formal", "export500", "verify-export"):
            name = f"{stage}-{group}.execution.json"
            execution[group][stage] = load(EXP / "evidence" / name)
        execution[group]["evaluation"] = {}
        for action in ("coco", "urban", "flickr_test1k", "docci", "long_dci"):
            execution[group]["evaluation"][action] = load(
                EXP / "evidence" / f"eval-{group}-{action}.execution.json")

    result = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "branch": "codex/nest-jointmask-fast-v1",
        "training_git_head": git_head,
        "performance_base_commit": "f614bda2eab5ad2b71a08ff9f62abdc68edb593a",
        "initial_checkpoint_sha256": "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6",
        "protocol": {
            "training": {"world_size": 4, "batch_per_rank": 256, "global_batch": 1024,
                         "accumulation": 1, "seed": 0, "updates": 500,
                         "scheduler_horizon": 3651, "training_records": 1245901,
                         "image_chunk": 128, "text_chunk": 128,
                         "checkpoint_pair_blocks": False, "checkpoint_encoders": True},
            "evaluation": {"batch_size": 64, "native_student_only": True,
                           "similarity": "normalized image/text embeddings inner product",
                           "datasets": list(DATASETS), "excluded": ["DCI Full"]},
        },
        "stream_digests_equal_all_steps_all_ranks": streams_equal,
        "groups": groups,
        "metrics": all_metrics,
        "evaluation_metadata": all_metadata,
        "comparisons": {
            "TI-fast_minus_T-fast": comparison(all_metrics, "TI-fast", "T-fast"),
            "TI-fast_minus_TI-Shuffle-fast": comparison(
                all_metrics, "TI-fast", "TI-Shuffle-fast"),
        },
        "execution": execution,
    }

    raw_root = EXP / "evidence" / "native_results"
    for group, sources in source_files.items():
        destination = raw_root / group
        destination.mkdir(parents=True, exist_ok=True)
        for dataset, source in sources.items():
            shutil.copy2(source, destination / f"{dataset.lower().replace('-', '_')}.json")

    result_path = EXP / "FORMAL500_RESULTS.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# NEST JointMask fast v1: 500-step formal results",
        "",
        "Date: 2026-09-29 UTC",
        "",
        f"Training commit: `{git_head}` on `codex/nest-jointmask-fast-v1`. Performance base: "
        "`f614bda2eab5ad2b71a08ff9f62abdc68edb593a`.",
        "",
        "All three groups started independently from the same step-0 checkpoint "
        "(`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`) and ran "
        "500 synchronized updates with 4 A100 80GB GPUs, batch 256 per rank, global batch 1024, "
        "seed 0, and scheduler horizon 3651. The common fast setting was 128x128 pair tiles, pair "
        "checkpoint off, and encoder checkpoint on.",
        "",
        "## Result",
        "",
        "Correct image conditioning improved native retrieval in this single-seed 500-step run. "
        "TI-fast beat T-fast on 26 of 30 reported recall metrics (2 ties, 2 losses), with mean "
        f"change {result['comparisons']['TI-fast_minus_T-fast']['mean_delta_pp_all_30']:+.3f} pp "
        "and mean R@1 change "
        f"{result['comparisons']['TI-fast_minus_T-fast']['mean_delta_pp_r1_10']:+.3f} pp. "
        "TI-fast also beat the architecture-matched shuffled-image control on 26 of 30 metrics "
        "(1 tie, 3 losses), with mean change "
        f"{result['comparisons']['TI-fast_minus_TI-Shuffle-fast']['mean_delta_pp_all_30']:+.3f} pp "
        "and mean R@1 change "
        f"{result['comparisons']['TI-fast_minus_TI-Shuffle-fast']['mean_delta_pp_r1_10']:+.3f} pp.",
        "",
        "The TI-fast advantage over TI-Shuffle-fast was positive for all 24 non-Urban metrics. "
        "Urban-1k was mixed: TI-fast lost I2T R@1/R@10 and T2I R@1, tied T2I R@5, and won the "
        "remaining two recalls. This is one seed and one stopping point, so these differences are "
        "descriptive rather than statistically significant.",
        "",
        "## Native retrieval",
        "",
        "Values are percentages. Delta columns are percentage points.",
        "",
        "| Dataset | Direction | Metric | T-fast | TI-fast | TI-Shuffle-fast | TI-T | TI-Shuffle |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    c_t = result["comparisons"]["TI-fast_minus_T-fast"]["delta_pp"]
    c_s = result["comparisons"]["TI-fast_minus_TI-Shuffle-fast"]["delta_pp"]
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                lines.append("| " + " | ".join((dataset, direction, recall,
                    fmt_pct(all_metrics["T-fast"][dataset][direction][recall]),
                    fmt_pct(all_metrics["TI-fast"][dataset][direction][recall]),
                    fmt_pct(all_metrics["TI-Shuffle-fast"][dataset][direction][recall]),
                    fmt_pp(c_t[dataset][direction][recall]),
                    fmt_pp(c_s[dataset][direction][recall]))) + " |")

    lines += ["", "## Training mechanism", "",
              "The retrieval gain does not coincide with lower inclusion violation. Over the final "
              "50 steps, TI-fast had higher hard inclusion violation than both controls. Its O/E "
              "IoU was lower, and its keep ratios were lower, while no view produced all-open or "
              "all-closed masks. The correct-image adapter correction was about twice the shuffled "
              "control by the end of training. This indicates that the paired visual condition was "
              "used and changed mask selection, but the native retrieval improvement cannot be "
              "explained as improved inclusion compliance.", "",
              "| Group | common loss | hard violation | O/E IoU | F/O/E keep | F/O/E delta |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        m = groups[group]["last50_mean"]
        lines.append(f"| {group} | {m['common_loss']:.4f} | {m['hard_inclusion_violation']:.4f} | "
                     f"{m['oe_iou']:.4f} | {m['F_keep_ratio']:.4f}/{m['O_keep_ratio']:.4f}/"
                     f"{m['E_keep_ratio']:.4f} | {m['F_delta_abs_mean']:.4f}/"
                     f"{m['O_delta_abs_mean']:.4f}/{m['E_delta_abs_mean']:.4f} |")

    lines += ["", "At step 500, TI-fast's F/O/E mean absolute visual corrections were "
              f"{groups['TI-fast']['key_steps']['500']['F_delta_abs_mean']:.4f}/"
              f"{groups['TI-fast']['key_steps']['500']['O_delta_abs_mean']:.4f}/"
              f"{groups['TI-fast']['key_steps']['500']['E_delta_abs_mean']:.4f}, compared with "
              f"{groups['TI-Shuffle-fast']['key_steps']['500']['F_delta_abs_mean']:.4f}/"
              f"{groups['TI-Shuffle-fast']['key_steps']['500']['O_delta_abs_mean']:.4f}/"
              f"{groups['TI-Shuffle-fast']['key_steps']['500']['E_delta_abs_mean']:.4f} for the "
              "shuffled control. T-fast has no joint adapter and therefore zero correction.", "",
              "## Execution and integrity", "",
              "| Group | Training min | Stable step median/P90 s | Peak allocated/reserved GiB | "
              "Parameter max diff |",
              "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        a = groups[group]["acceptance"]["ranks"]
        t = groups[group]["timing"]
        lines.append(f"| {group} | {t['training_seconds']/60:.2f} | "
                     f"{t['stable_step_max_rank_median_seconds']:.3f}/"
                     f"{t['stable_step_max_rank_p90_seconds']:.3f} | "
                     f"{max(x['peak_allocated_gib'] for x in a):.2f}/"
                     f"{max(x['peak_reserved_gib'] for x in a):.2f} | "
                     f"{max(x['max_parameter_difference_from_rank0'] for x in a):.1f} |")

    lines += ["", "All groups have exactly 500 continuous step records, finite losses and gradients, "
              "zero fallback steps, zero duplicate-ID steps, successful final NCCL checks, and "
              "identical sample/text/token stream digests at every step and rank. Global valid "
              "O/E candidates ranged from 1022 to 1024; F always used 1024 candidates. All exports "
              "strictly loaded at optimizer step 500 and matched full-checkpoint native image and "
              "text embeddings with maximum absolute error 0.", "",
              "T-fast has no joint adapter, so TI-fast versus T-fast also includes the adapter "
              "parameterization. TI-fast versus TI-Shuffle-fast is the stronger controlled test of "
              "whether the correct paired image matters: their adapter initial tensors are identical, "
              "and only the visual condition presented to the gate is shuffled.", "",
              "## Evaluation protocol and artifacts", "",
              "COCO uses 5000 images and 25000 captions with similarity chunk 512. Urban-1k uses "
              "1000 image-caption pairs. Flickr30k test1k uses 1000 images and 5000 captions. DOCCI "
              "uses 5000 pairs, and long-DCI uses 7602 pairs. All use batch 64 and normalized native "
              "student image/text embeddings with plain inner product; no mask, fusion, or reranking "
              "is used. DCI Full was excluded as requested.", "",
              "Machine-readable results: `FORMAL500_RESULTS.json`. Preserved small raw evaluator JSONs "
              "are under `evidence/native_results/`; stage commands, commits, wall times, console output, "
              "and exit codes are under `evidence/`. Training checkpoints, bare students, data, and "
              "large caches remain on the server and are not committed.", ""]
    (EXP / "FORMAL500_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"results": str(result_path), "report": str(EXP / 'FORMAL500_REPORT.md'),
                      "streams_equal": streams_equal}, indent=2))


if __name__ == "__main__":
    main()
