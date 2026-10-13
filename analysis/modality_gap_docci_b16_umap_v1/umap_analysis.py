#!/usr/bin/env python3
"""Fixed-parameter UMAP audit for the existing DOCCI modality-gap features."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap


CAPTIONS = [
    ("full", "Full"),
    ("prefix50", "50% Prefix"),
    ("first_sentence", "First Sentence"),
]
MODELS = [
    ("longclip", "Long-CLIP"),
    ("ours_masked", "Ours Masked"),
    ("ours_native", "Ours Native"),
]
BLUE = "#2166ac"
ORANGE = "#d95f02"
BLUE_DARK = "#08306b"
ORANGE_DARK = "#993404"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-15)


def paired_metrics(image: np.ndarray, text: np.ndarray) -> dict:
    image = l2_normalize(image)
    text = l2_normalize(text)
    cosine = np.sum(image * text, axis=1)
    l2 = np.linalg.norm(image - text, axis=1)
    return {
        "centroid_gap": float(np.linalg.norm(image.mean(axis=0) - text.mean(axis=0))),
        "paired_cosine": float(cosine.mean()),
        "paired_l2": float(l2.mean()),
        "image_centroid_norm": float(np.linalg.norm(image.mean(axis=0))),
        "text_centroid_norm": float(np.linalg.norm(text.mean(axis=0))),
        "paired_cosine_values": cosine,
        "paired_l2_values": l2,
        "image_normalized": image,
        "text_normalized": text,
    }


def bootstrap_ci(image: np.ndarray, text: np.ndarray, seed: int, reps: int) -> dict:
    """Paired bootstrap CIs in the original 512-D space."""
    image = l2_normalize(image)
    text = l2_normalize(text)
    n = image.shape[0]
    rng = np.random.default_rng(seed)
    estimates = np.empty((reps, 3), dtype=np.float64)
    chunk = 100
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        ids = rng.integers(0, n, size=(stop - start, n))
        im = image[ids]
        tx = text[ids]
        estimates[start:stop, 0] = np.linalg.norm(im.mean(axis=1) - tx.mean(axis=1), axis=1)
        estimates[start:stop, 1] = np.mean(np.sum(im * tx, axis=2), axis=1)
        estimates[start:stop, 2] = np.mean(np.linalg.norm(im - tx, axis=2), axis=1)
    lo, hi = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {
        "centroid_gap_ci_low": float(lo[0]),
        "centroid_gap_ci_high": float(hi[0]),
        "paired_cosine_ci_low": float(lo[1]),
        "paired_cosine_ci_high": float(hi[1]),
        "paired_l2_ci_low": float(lo[2]),
        "paired_l2_ci_high": float(hi[2]),
    }


def bootstrap_mask_ci(mask: np.ndarray, seed: int, reps: int) -> tuple[float, float, float]:
    values = np.asarray(mask, dtype=np.float64).mean(axis=1)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, values.shape[0], size=(reps, values.shape[0]))
    means = values[ids].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def fit_model_umap(z: np.lib.npyio.NpzFile, model: str) -> tuple[object, dict]:
    arrays = []
    for cap, _ in CAPTIONS:
        arrays += [l2_normalize(z[f"{model}_{cap}_image"]).astype(np.float32),
                   l2_normalize(z[f"{model}_{cap}_text"]).astype(np.float32)]
    fit = np.concatenate(arrays, axis=0)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        metric="cosine",
        random_state=0,
        transform_seed=0,
        n_jobs=1,
        init="spectral",
        verbose=False,
    )
    reducer.fit(fit)
    projected = {}
    for cap, _ in CAPTIONS:
        projected[f"{model}_{cap}_image"] = reducer.transform(l2_normalize(z[f"{model}_{cap}_image"]).astype(np.float32))
        projected[f"{model}_{cap}_text"] = reducer.transform(l2_normalize(z[f"{model}_{cap}_text"]).astype(np.float32))
    return reducer, {"projected": projected, "fit_count": int(fit.shape[0])}


def row_limits(projected: dict, model: str) -> tuple[tuple[float, float], tuple[float, float]]:
    all_xy = np.concatenate([
        projected[f"{model}_{cap}_{kind}"]
        for cap, _ in CAPTIONS for kind in ("image", "text")
    ], axis=0)
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return (float(lo[0] - .06 * span[0]), float(hi[0] + .06 * span[0])), (float(lo[1] - .06 * span[1]), float(hi[1] + .06 * span[1]))


def draw_panel(ax, image, text, pair_ids, title, xlim, ylim, annotation):
    pair_image, pair_text = image[pair_ids], text[pair_ids]
    for p, q in zip(pair_image, pair_text):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#9ecae1", alpha=.36, lw=.55, zorder=1)
    ax.scatter(image[:, 0], image[:, 1], s=8, alpha=.30, c=BLUE, linewidths=0, zorder=2)
    ax.scatter(text[:, 0], text[:, 1], s=8, alpha=.30, c=ORANGE, linewidths=0, zorder=2)
    ci, ct = image.mean(0), text.mean(0)
    ax.scatter([ci[0]], [ci[1]], marker="X", s=112, c=BLUE_DARK, edgecolors="white", linewidths=.8, zorder=3)
    ax.scatter([ct[0]], [ct[1]], marker="X", s=112, c=ORANGE_DARK, edgecolors="white", linewidths=.8, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=.15, linewidth=.55)
    ax.text(.02, .98, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#bdbdbd", alpha=.88))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--mask-input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    args = ap.parse_args()
    inp, mask_in, out = Path(args.input_dir), Path(args.mask_input), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(inp / "features.npz")
    mask_z = np.load(mask_in / "mask_arrays.npz")
    provenance = json.loads((inp / "provenance.json").read_text())
    sample_ids = np.sort(np.random.default_rng(0).choice(1000, size=50, replace=False)).astype(int)

    reducers, projections = {}, {}
    for model, _ in MODELS:
        reducer, info = fit_model_umap(z, model)
        reducers[model] = reducer
        projections[model] = info

    # Main 2x3: Long-CLIP and Ours Masked. Each row has its own shared UMAP axes.
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), squeeze=False, facecolor="white")
    for row, (model, label) in enumerate(MODELS[:2]):
        xlim, ylim = row_limits(projections[model]["projected"], model)
        for col, (cap, cap_label) in enumerate(CAPTIONS):
            im = projections[model]["projected"][f"{model}_{cap}_image"]
            tx = projections[model]["projected"][f"{model}_{cap}_text"]
            vals = paired_metrics(z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"])
            draw_panel(axes[row, col], im, tx, sample_ids, f"{label} — {cap_label}", xlim, ylim,
                       f"Centroid Gap {vals['centroid_gap']:.3f}\nPaired Cosine {vals['paired_cosine']:.3f}")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="none", alpha=.55, label="Image"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="none", alpha=.55, label="Text"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=BLUE_DARK, markeredgecolor="white", label="Image centroid"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=ORANGE_DARK, markeredgecolor="white", label="Text centroid"),
        Line2D([0], [0], color="#9ecae1", lw=.8, label="50 paired links"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("DOCCI modality gap — model-specific shared UMAP fits", y=1.045, fontsize=15)
    fig.text(.5, .005, "Each model row uses one UMAP fit over all three caption settings; rows are not on a shared UMAP.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .97))
    fig.savefig(out / "umap_main_2x3.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "umap_main_2x3.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Ours Native 1x3.
    model = "ours_native"
    xlim, ylim = row_limits(projections[model]["projected"], model)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), squeeze=False, facecolor="white")
    for col, (cap, cap_label) in enumerate(CAPTIONS):
        im = projections[model]["projected"][f"{model}_{cap}_image"]
        tx = projections[model]["projected"][f"{model}_{cap}_text"]
        vals = paired_metrics(z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"])
        draw_panel(axes[0, col], im, tx, sample_ids, f"Ours Native — {cap_label}", xlim, ylim,
                   f"Centroid Gap {vals['centroid_gap']:.3f}\nPaired Cosine {vals['paired_cosine']:.3f}")
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.suptitle("DOCCI modality gap — Ours Native", y=1.09, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out / "umap_ours_native_1x3.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "umap_ours_native_1x3.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for model, model_label in MODELS:
        for cap, cap_label in CAPTIONS:
            vals = paired_metrics(z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"])
            ci = bootstrap_ci(z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"], seed=0, reps=args.bootstrap_reps)
            mask_mean = mask_low = mask_high = ""
            if model == "ours_masked":
                mask_mean, mask_low, mask_high = bootstrap_mask_ci(mask_z[f"mask_{cap}"], seed=0, reps=args.bootstrap_reps)
            im2d = projections[model]["projected"][f"{model}_{cap}_image"][sample_ids]
            tx2d = projections[model]["projected"][f"{model}_{cap}_text"][sample_ids]
            umap_length = float(np.linalg.norm(im2d - tx2d, axis=1).mean())
            rows.append({
                "model": model, "model_label": model_label, "caption_setting": cap, "caption_label": cap_label,
                "sample_count": 1000, "centroid_gap": vals["centroid_gap"],
                "centroid_gap_ci_low": ci["centroid_gap_ci_low"], "centroid_gap_ci_high": ci["centroid_gap_ci_high"],
                "paired_cosine": vals["paired_cosine"], "paired_cosine_ci_low": ci["paired_cosine_ci_low"], "paired_cosine_ci_high": ci["paired_cosine_ci_high"],
                "paired_l2": vals["paired_l2"], "paired_l2_ci_low": ci["paired_l2_ci_low"], "paired_l2_ci_high": ci["paired_l2_ci_high"],
                "mask_keep_ratio": mask_mean, "mask_keep_ci_low": mask_low, "mask_keep_ci_high": mask_high,
                "paired_link_count": 50, "mean_umap_pair_length": umap_length,
                "image_centroid_norm": vals["image_centroid_norm"], "text_centroid_norm": vals["text_centroid_norm"],
            })
    fields = list(rows[0].keys())
    with (out / "umap_quantitative_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    params = {
        "umap": {"n_neighbors": 30, "min_dist": 0.15, "metric": "cosine", "random_state": 0, "transform_seed": 0, "n_jobs": 1, "fit_scope": "one fit per model over image/text features for Full, 50% Prefix, First Sentence"},
        "bootstrap": {"reps": args.bootstrap_reps, "seed": 0, "ci": "percentile 2.5/97.5", "unit": "paired sample index"},
        "sample_count": 1000, "paired_link_count": 50, "paired_sample_indices": sample_ids.tolist(),
        "feature_source": {"path": str(inp / "features.npz"), "sha256": sha256(inp / "features.npz"), "provenance_sha256": sha256(inp / "provenance.json"), "samples_sha256": sha256(inp / "samples.json")},
        "mask_source": {"path": str(mask_in / "mask_arrays.npz"), "sha256": sha256(mask_in / "mask_arrays.npz"), "identity": "mask means match v2 provenance exactly for all three caption settings"},
        "feature_dim": 512, "models": [m for m, _ in MODELS], "captions": [c for c, _ in CAPTIONS],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "umap_learn": getattr(umap, "__version__", "unknown"), "matplotlib": matplotlib.__version__},
    }
    (out / "UMAP_PARAMS_AND_SAMPLES.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "UMAP_PROVENANCE.json").write_text(json.dumps({"input_provenance": provenance, "parameters": params, "outputs": ["umap_main_2x3.png", "umap_main_2x3.pdf", "umap_ours_native_1x3.png", "umap_ours_native_1x3.pdf", "umap_quantitative_metrics.csv"]}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "feature_sha256": params["feature_source"]["sha256"], "sample_indices": sample_ids.tolist()}, indent=2))


if __name__ == "__main__":
    main()
