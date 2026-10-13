#!/usr/bin/env python3
"""Paper-style PCA views for the fixed DOCCI V2 First Sentence features.

This script is deliberately feature-only: it never loads a checkpoint and
never extracts features.  It creates shared-PCA and per-model-PCA views for
Long-CLIP and Ours Masked under the three fixed caption conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


CAPTIONS = [
    ("full", "Full"),
    ("prefix50", "50% Prefix"),
    ("first_sentence", "First Sentence"),
]
MODELS = [("longclip", "Long-CLIP"), ("ours_masked", "Ours Masked")]
BLUE = "#2166ac"
ORANGE = "#d95f02"
BLUE_DARK = "#08306b"
ORANGE_DARK = "#993404"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-15)


def high_dim_metrics(image: np.ndarray, text: np.ndarray) -> dict[str, float]:
    image = normalized(image)
    text = normalized(text)
    im_centroid = image.mean(axis=0)
    tx_centroid = text.mean(axis=0)
    return {
        "centroid_gap": float(np.linalg.norm(im_centroid - tx_centroid)),
        "paired_cosine": float(np.sum(image * text, axis=1).mean()),
        "paired_l2": float(np.linalg.norm(image - text, axis=1).mean()),
    }


def kde_contour(ax, points: np.ndarray, color: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Draw light KDE contours, with a safe fallback for singular clouds."""
    grid_x, grid_y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 90),
        np.linspace(ylim[0], ylim[1], 90),
    )
    xy = np.vstack([points[:, 0], points[:, 1]])
    try:
        density = gaussian_kde(xy)(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        sample_density = gaussian_kde(xy)(xy)
        levels = np.unique(np.quantile(sample_density, [0.25, 0.55, 0.80]))
        levels = levels[np.isfinite(levels)]
        if len(levels) >= 2 and np.nanmax(density) > np.nanmin(density):
            ax.contour(grid_x, grid_y, density, levels=levels, colors=[color], linewidths=0.7, alpha=0.42)
    except (np.linalg.LinAlgError, ValueError):
        # A contour is supplementary; do not alter points or silently invent
        # a distribution when a covariance estimate is singular.
        return


def bounds(projected: dict[str, np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate(list(projected.values()), axis=0)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    return (
        (float(lo[0] - 0.07 * span[0]), float(hi[0] + 0.07 * span[0])),
        (float(lo[1] - 0.07 * span[1]), float(hi[1] + 0.07 * span[1])),
    )


def draw_figure(
    projected: dict[str, np.ndarray],
    rows: list[dict],
    out_path: Path,
    title: str,
    component_labels: tuple[str, str],
) -> None:
    all_proj = {k: v for k, v in projected.items()}
    xlim, ylim = bounds(all_proj)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.1), squeeze=False)
    for row, (model, model_label) in enumerate(MODELS):
        for col, (caption, caption_label) in enumerate(CAPTIONS):
            ax = axes[row, col]
            im = projected[f"{model}_{caption}_image"]
            tx = projected[f"{model}_{caption}_text"]
            ax.scatter(im[:, 0], im[:, 1], s=3, alpha=0.12, c=BLUE, edgecolors="none", rasterized=True)
            ax.scatter(tx[:, 0], tx[:, 1], s=3, alpha=0.12, c=ORANGE, edgecolors="none", rasterized=True)
            kde_contour(ax, im, BLUE, xlim, ylim)
            kde_contour(ax, tx, ORANGE, xlim, ylim)
            ci, ct = im.mean(axis=0), tx.mean(axis=0)
            ax.scatter([ci[0]], [ci[1]], marker="X", s=88, c=BLUE_DARK, edgecolors="white", linewidths=0.7, zorder=5)
            ax.scatter([ct[0]], [ct[1]], marker="X", s=88, c=ORANGE_DARK, edgecolors="white", linewidths=0.7, zorder=5)
            metric = next(r for r in rows if r["model"] == model and r["caption"] == caption)
            ax.text(
                0.03,
                0.97,
                f"Centroid Gap = {metric['centroid_gap']:.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
            )
            ax.set_title(caption_label, fontsize=11)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(component_labels[0])
            ax.set_ylabel(component_labels[1])
            ax.grid(alpha=0.14, linewidth=0.5)
            if col == 0:
                ax.text(-0.23, 0.5, model_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=BLUE, alpha=0.55, label="Image"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=ORANGE, alpha=0.55, label="Text"),
        plt.Line2D([], [], marker="X", linestyle="", markersize=8, color=BLUE_DARK, markeredgecolor="white", label="Image centroid"),
        plt.Line2D([], [], marker="X", linestyle="", markersize=8, color=ORANGE_DARK, markeredgecolor="white", label="Text centroid"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.suptitle(title, y=1.035, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_projection(z: dict[str, np.ndarray], model_keys: list[str], components: tuple[int, int]) -> tuple[dict[str, np.ndarray], PCA]:
    fit_data = np.concatenate([normalized(z[k]) for k in model_keys], axis=0)
    pca = PCA(n_components=3, svd_solver="full", random_state=0)
    pca.fit(fit_data)
    projected = {k: pca.transform(normalized(v))[:, components] for k, v in z.items()}
    return projected, pca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.input_dir / "features.npz"
    provenance_path = args.input_dir / "provenance.json"
    z_npz = np.load(feature_path)
    z = {k: z_npz[k] for k in z_npz.files}
    expected = [f"{m}_{c}_{mod}" for m, _ in MODELS for c, _ in CAPTIONS for mod in ("image", "text")]
    missing = sorted(set(expected) - set(z))
    if missing:
        raise KeyError(f"Missing required feature arrays: {missing}")

    rows = []
    for model, _ in MODELS:
        for caption, _ in CAPTIONS:
            row = {"model": model, "caption": caption, "sample_count": int(z[f"{model}_{caption}_image"].shape[0])}
            row.update(high_dim_metrics(z[f"{model}_{caption}_image"], z[f"{model}_{caption}_text"]))
            rows.append(row)

    main_keys = [f"{model}_{caption}_{mod}" for model, _ in MODELS for caption, _ in CAPTIONS for mod in ("image", "text")]
    fits = {}
    for scheme, fit_keys in (
        ("shared", main_keys),
        ("per_model_longclip", [k for k in main_keys if k.startswith("longclip_")]),
        ("per_model_ours_masked", [k for k in main_keys if k.startswith("ours_masked_")]),
    ):
        for projection_name, components, labels in (
            ("pc1_pc2", (0, 1), ("PC1", "PC2")),
            ("pc2_pc3", (1, 2), ("PC2", "PC3")),
        ):
            projected, pca = fit_projection(z, fit_keys, components)
            fits[(scheme, projection_name)] = {"projected": projected, "pca": pca, "components": components, "labels": labels, "fit_keys": fit_keys}

    for scheme, proj_name, filename in (
        ("shared", "pc1_pc2", "pca_shared_main_2x3"),
        ("shared", "pc2_pc3", "pca_shared_pc2_pc3_2x3"),
        ("per_model_longclip", "pc1_pc2", "pca_per_model_main_2x3"),
        ("per_model_longclip", "pc2_pc3", "pca_per_model_pc2_pc3_2x3"),
    ):
        # The per-model main figure uses a common panel layout, but each row's
        # projection comes from its own fit.  Combine the two row-specific fits.
        if scheme == "per_model_longclip":
            base = fits[(scheme, proj_name)]
            other = fits[("per_model_ours_masked", proj_name)]
            projected = dict(base["projected"])
            projected.update({k: v for k, v in other["projected"].items() if k.startswith("ours_masked_")})
            pca_text = "Per-model PCA (Long-CLIP / Ours Masked rows fitted independently)"
            labels = base["labels"]
            i0, i1 = base["components"]
            component_text = " / ".join(
                f"{name.replace('per_model_', '')}: {labels[0]}={fits[(name, proj_name)]['pca'].explained_variance_ratio_[i0]*100:.2f}%, {labels[1]}={fits[(name, proj_name)]['pca'].explained_variance_ratio_[i1]*100:.2f}%"
                for name in ("per_model_longclip", "per_model_ours_masked")
            )
            var_text = component_text
            title = f"DOCCI modality gap — {pca_text}\n{var_text}"
        else:
            fit = fits[(scheme, proj_name)]
            projected, labels = fit["projected"], fit["labels"]
            pca = fit["pca"]
            i0, i1 = fit["components"]
            title = f"DOCCI modality gap — Shared PCA\n{labels[0]}={pca.explained_variance_ratio_[i0]*100:.2f}%, {labels[1]}={pca.explained_variance_ratio_[i1]*100:.2f}%"
        draw_figure(projected, rows, args.output_dir / filename, title, labels)

    with (args.output_dir / "pca_quantitative_metrics.csv").open("w", newline="") as f:
        fields = ["model", "caption", "sample_count", "centroid_gap", "paired_cosine", "paired_l2"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    variance = {}
    for key, fit in fits.items():
        variance[f"{key[0]}__{key[1]}"] = {
            "components": list(fit["components"]),
            "explained_variance_ratio": fit["pca"].explained_variance_ratio_.tolist(),
            "fit_point_count": int(sum(z[k].shape[0] for k in fit["fit_keys"])),
            "fit_arrays": fit["fit_keys"],
        }
    params = {
        "input_features": str(feature_path),
        "input_features_sha256": sha256(feature_path),
        "input_provenance": str(provenance_path),
        "input_provenance_sha256": sha256(provenance_path),
        "sample_count": 1000,
        "models": [m for m, _ in MODELS],
        "captions": [c for c, _ in CAPTIONS],
        "pca": {"solver": "full", "random_state": 0, "input_normalization": "per-row L2 normalization in original 512-D space", "shared_fit": "Long-CLIP and Ours Masked, all three captions, image/text; 12000 points", "per_model_fit": "one fit for all three captions and image/text within each row model; 6000 points per model", "sensitivity_projection": "PC2-PC3; no result hidden"},
        "explained_variance": variance,
        "plot": {"scatter_size": 3, "scatter_alpha": 0.12, "kde": "scipy gaussian_kde over all 1000 image or text points per panel", "colors": {"image": BLUE, "text": ORANGE}, "background": "white", "centroids": "mean of all projected points; high-dimensional gap label uses original 512-D normalized features"},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "outputs": [],
    }
    lines = [
        "# DOCCI V2 First Sentence — PCA Visualization",
        "",
        "This is a feature-only, fixed-seed visualization using the existing DOCCI V2 First Sentence `features.npz`; no checkpoint was loaded and no features were re-extracted.",
        "",
        "The main 2×3 layout has Long-CLIP in row 1 and Ours Masked in row 2, with Full, 50% Prefix, and First Sentence in columns. Image points are blue and text points orange (`s=3`, `alpha=0.12`); light KDE contours and both centroids are overlaid.",
        "",
        "## PCA schemes",
        "",
        "- Shared PCA: one fit over all 12,000 Long-CLIP and Ours Masked image/text points across the three caption settings.",
        "- Per-model PCA: Long-CLIP and Ours Masked each have an independent fit over their own 6,000 points. Each row shares its own projection across columns; row coordinates must not be compared across models.",
        "- PC2–PC3 versions are included as a fixed sensitivity view, with explained variance recorded in `PCA_PARAMETERS_AND_VARIANCE.json`.",
        "",
        "## Explained variance",
        "",
        "| Fit | PC1 | PC2 | PC3 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("shared__pc1_pc2", "per_model_longclip__pc1_pc2", "per_model_ours_masked__pc1_pc2"):
        label = {"shared__pc1_pc2": "Shared", "per_model_longclip__pc1_pc2": "Long-CLIP per-model", "per_model_ours_masked__pc1_pc2": "Ours Masked per-model"}[key]
        vals = variance[key]["explained_variance_ratio"]
        lines.append(f"| {label} | {vals[0]*100:.2f}% | {vals[1]*100:.2f}% | {vals[2]*100:.2f}% |")
    lines += [
        "",
        "## 512-D quantitative metrics",
        "",
        "| Model | Caption | Centroid Gap | Paired Cosine | Paired L2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['caption']} | {row['centroid_gap']:.6f} | {row['paired_cosine']:.6f} | {row['paired_l2']:.6f} |")
    lines += [
        "",
        "## Visual comparison",
        "",
        "The shared PCA preserves one global coordinate system and is the appropriate reference for comparing the two rows. Per-model PCA usually makes within-model local structure easier to inspect, but its two rows have independently rotated/scaled PCA axes and their cross-row distances are not scientifically comparable. The PC2–PC3 views are a sensitivity check rather than a replacement for the primary PC1–PC2 views.",
        "",
        "The image/text double-cluster structure remains visible under both PCA fitting schemes. KDE contours make local density and shape easier to see, while the original 512-D Centroid Gap remains the quantitative modality-gap measure. Points were not filtered, shifted, rescaled by modality, or manually cleaned.",
        "",
        "PCA is better suited for the paper main figure when the question is global cluster separation and cross-row comparison. Per-model PCA and PC2–PC3 are useful supplementary diagnostics for local structure and sensitivity.",
    ]
    (args.output_dir / "PCA_VISUALIZATION_REPORT.md").write_text("\n".join(lines) + "\n")
    params["outputs"] = sorted(p.name for p in args.output_dir.iterdir() if p.is_file())
    (args.output_dir / "PCA_PARAMETERS_AND_VARIANCE.json").write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "feature_sha256": params["input_features_sha256"], "files": params["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
