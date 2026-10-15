from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

SEED = 20261013
BOOT_REPS = 200
CAPTIONS = ["full", "prefix50", "first_sentence"]
CAPTION_LABELS = {"full": "Full", "prefix50": "50% Prefix", "first_sentence": "First Sentence"}
MODELS = ["longclip", "ours_native", "ours_masked"]
MODEL_LABELS = {"longclip": "Long-CLIP", "ours_native": "Ours Native", "ours_masked": "Ours Masked"}
MODEL_COLORS = {"longclip": "#2364AA", "ours_native": "#7A5195", "ours_masked": "#2A9D8F"}
IMAGE_COLOR = "#2F80ED"
TEXT_COLOR = "#E76F51"
GRID_COLOR = "#D9DEE7"
DARK = "#1F2937"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    }
)


def load_data():
    z = np.load(INPUT / "features.npz", allow_pickle=True)
    features = {k: z[k].astype(np.float64) for k in z.files}
    raw = pd.read_csv(INPUT / "modality_gap_metrics.csv")
    return features, raw


def key(model: str, caption: str, modality: str) -> str:
    return f"{model}_{caption}_{modality}"


def savefig(fig: plt.Figure, name: str):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax):
    ax.grid(axis="y", color=GRID_COLOR, lw=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#9AA4B2")
    ax.spines["bottom"].set_color("#9AA4B2")


def add_modality_legend(fig, extra=None, loc="upper center", bbox=(0.5, 0.02)):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=IMAGE_COLOR, markeredgecolor="none", markersize=7, label="Image"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TEXT_COLOR, markeredgecolor="none", markersize=7, label="Text"),
    ]
    if extra:
        handles.extend(extra)
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, frameon=False, ncol=min(len(handles), 5), handletextpad=0.35, columnspacing=1.0)


def fit_pca_arrays(models, components=(0, 1)):
    all_arrays = []
    for model in models:
        for caption in CAPTIONS:
            all_arrays.extend([features[key(model, caption, "image")], features[key(model, caption, "text")]])
    pca = PCA(n_components=max(components) + 1, svd_solver="full", random_state=SEED)
    pca.fit(np.vstack(all_arrays))
    return pca


def plot_pca_shared(models, name_prefix, components=(0, 1), title=None):
    pca = fit_pca_arrays(models, components)
    nrows = len(models)
    fig, axes = plt.subplots(nrows, 3, figsize=(10.4, 3.5 * nrows), squeeze=False, sharex=False, sharey=False)
    plotted = []
    for r, model in enumerate(models):
        for c, caption in enumerate(CAPTIONS):
            ax = axes[r, c]
            im = pca.transform(features[key(model, caption, "image")])[:, list(components)]
            tx = pca.transform(features[key(model, caption, "text")])[:, list(components)]
            ax.scatter(im[:, 0], im[:, 1], s=6, alpha=0.18, color=IMAGE_COLOR, linewidths=0, rasterized=True)
            ax.scatter(tx[:, 0], tx[:, 1], s=6, alpha=0.18, color=TEXT_COLOR, linewidths=0, rasterized=True, marker="s")
            ax.scatter(im[:, 0].mean(), im[:, 1].mean(), s=62, color=IMAGE_COLOR, edgecolor="white", linewidth=0.8, zorder=5)
            ax.scatter(tx[:, 0].mean(), tx[:, 1].mean(), s=62, color=TEXT_COLOR, edgecolor="white", linewidth=0.8, marker="s", zorder=5)
            ax.plot([im[:, 0].mean(), tx[:, 0].mean()], [im[:, 1].mean(), tx[:, 1].mean()], color=DARK, lw=1.1, alpha=0.85, zorder=4)
            ax.set_title(CAPTION_LABELS[caption])
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nPC{components[1]+1}")
            else:
                ax.set_ylabel(f"PC{components[1]+1}")
            ax.set_xlabel(f"PC{components[0]+1}")
            clean_axes(ax)
            plotted.append(np.vstack([im, tx]))
    # consistent limits for direct visual comparison
    flat = np.vstack(plotted)
    for ax in axes.ravel():
        xlo, xhi = np.percentile(flat[:, 0], [0.5, 99.5]); ylo, yhi = np.percentile(flat[:, 1], [0.5, 99.5])
        padx, pady = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
        ax.set_xlim(xlo - padx, xhi + padx); ax.set_ylim(ylo - pady, yhi + pady)
    fig.suptitle(title or f"Shared PCA: {' + '.join(MODEL_LABELS[m] for m in models)}", y=1.01, fontsize=13, fontweight="bold")
    add_modality_legend(fig, extra=[Line2D([0], [0], color=DARK, lw=1.1, label="Centroid gap")], bbox=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig(fig, name_prefix)
    return pca


def plot_pca_per_model(model, components=(0, 1), name_prefix="pca_per_model"):
    # A separate fit per model makes the representation-specific geometry explicit.
    all_arrays = []
    for caption in CAPTIONS:
        all_arrays.extend([features[key(model, caption, "image")], features[key(model, caption, "text")]])
    pca = PCA(n_components=max(components) + 1, svd_solver="full", random_state=SEED).fit(np.vstack(all_arrays))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), squeeze=False)
    all_pts = []
    for c, caption in enumerate(CAPTIONS):
        ax = axes[0, c]
        im = pca.transform(features[key(model, caption, "image")])[:, list(components)]
        tx = pca.transform(features[key(model, caption, "text")])[:, list(components)]
        ax.scatter(im[:, 0], im[:, 1], s=6, alpha=0.18, color=IMAGE_COLOR, linewidths=0, rasterized=True)
        ax.scatter(tx[:, 0], tx[:, 1], s=6, alpha=0.18, color=TEXT_COLOR, linewidths=0, marker="s", rasterized=True)
        ax.scatter(im[:, 0].mean(), im[:, 1].mean(), s=62, color=IMAGE_COLOR, edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter(tx[:, 0].mean(), tx[:, 1].mean(), s=62, color=TEXT_COLOR, edgecolor="white", linewidth=0.8, marker="s", zorder=5)
        ax.plot([im[:, 0].mean(), tx[:, 0].mean()], [im[:, 1].mean(), tx[:, 1].mean()], color=DARK, lw=1.1)
        ax.set_title(CAPTION_LABELS[caption]); ax.set_xlabel(f"PC{components[0]+1}"); ax.set_ylabel(f"PC{components[1]+1}")
        clean_axes(ax); all_pts.extend([im, tx])
    flat = np.vstack(all_pts); xlo, xhi = np.percentile(flat[:, 0], [0.5, 99.5]); ylo, yhi = np.percentile(flat[:, 1], [0.5, 99.5])
    for ax in axes.ravel():
        ax.set_xlim(xlo - 0.06*(xhi-xlo), xhi + 0.06*(xhi-xlo)); ax.set_ylim(ylo - 0.06*(yhi-ylo), yhi + 0.06*(yhi-ylo))
    fig.suptitle(f"Per-model PCA: {MODEL_LABELS[model]}", y=1.01, fontsize=13, fontweight="bold")
    add_modality_legend(fig, extra=[Line2D([0], [0], color=DARK, lw=1.1, label="Centroid gap")], bbox=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); savefig(fig, name_prefix)
    return pca


def kde_line(ax, x, color, lw=1.5, alpha=1.0, label=None, linestyle="-"):
    if np.std(x) < 1e-8:
        return
    grid = np.linspace(np.min(x), np.max(x), 300)
    try:
        kde = gaussian_kde(x, bw_method=0.3)
        ax.plot(grid, kde(grid), color=color, lw=lw, alpha=alpha, label=label, linestyle=linestyle)
    except np.linalg.LinAlgError:
        pass


def full_gap_axes():
    axes = {}
    for model in MODELS:
        im = features[key(model, "full", "image")].mean(0)
        tx = features[key(model, "full", "text")].mean(0)
        d = im - tx
        axes[model] = d / np.linalg.norm(d)
    return axes


def plot_gap_axis():
    gap_axes = full_gap_axes()
    all_proj = []
    for model in MODELS:
        for caption in CAPTIONS:
            all_proj.extend([features[key(model, caption, "image")] @ gap_axes[model], features[key(model, caption, "text")] @ gap_axes[model]])
    lim = max(abs(np.percentile(np.concatenate(all_proj), [0.2, 99.8]))) * 1.08
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.1), sharex=True, sharey=True)
    for r, model in enumerate(MODELS):
        for c, caption in enumerate(CAPTIONS):
            ax = axes[r, c]
            im = features[key(model, caption, "image")] @ gap_axes[model]
            tx = features[key(model, caption, "text")] @ gap_axes[model]
            bins = np.linspace(-lim, lim, 46)
            ax.hist(im, bins=bins, density=True, color=IMAGE_COLOR, alpha=0.30, edgecolor="none")
            ax.hist(tx, bins=bins, density=True, color=TEXT_COLOR, alpha=0.27, edgecolor="none")
            kde_line(ax, im, IMAGE_COLOR, label="Image KDE")
            kde_line(ax, tx, TEXT_COLOR, label="Text KDE")
            mi, mt = im.mean(), tx.mean()
            ax.axvline(mi, color=IMAGE_COLOR, lw=1.3, ls="--")
            ax.axvline(mt, color=TEXT_COLOR, lw=1.3, ls="--")
            ax.annotate(f"Δμ_axis={abs(mi-mt):.3f}", xy=((mi+mt)/2, ax.get_ylim()[1]*0.85), ha="center", va="top", fontsize=8, color=DARK, fontweight="bold")
            if r == 0: ax.set_title(CAPTION_LABELS[caption])
            if c == 0: ax.set_ylabel(f"{MODEL_LABELS[model]}\nDensity")
            if r == 2: ax.set_xlabel("Projection on Full-caption gap axis")
            clean_axes(ax); ax.set_xlim(-lim, lim)
    fig.suptitle("1D modality-gap axis distributions (axis fixed from Full captions)", y=0.995, fontsize=13, fontweight="bold")
    add_modality_legend(fig, extra=[Line2D([0], [0], color=IMAGE_COLOR, lw=1.3, ls="--", label="Image centroid"), Line2D([0], [0], color=TEXT_COLOR, lw=1.3, ls="--", label="Text centroid")], bbox=(0.5, -0.002))
    fig.tight_layout(rect=(0, 0.04, 1, 0.975)); savefig(fig, "gap_axis_hist_kde_3x3")

    # A clean per-model variant is useful for appendix readability.
    for model in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.3), sharey=True)
        for c, caption in enumerate(CAPTIONS):
            ax = axes[c]
            im = features[key(model, caption, "image")] @ gap_axes[model]; tx = features[key(model, caption, "text")] @ gap_axes[model]
            bins = np.linspace(-lim, lim, 46)
            ax.hist(im, bins=bins, density=True, color=IMAGE_COLOR, alpha=0.3, edgecolor="none")
            ax.hist(tx, bins=bins, density=True, color=TEXT_COLOR, alpha=0.26, edgecolor="none")
            kde_line(ax, im, IMAGE_COLOR); kde_line(ax, tx, TEXT_COLOR)
            ax.axvline(im.mean(), color=IMAGE_COLOR, lw=1.3, ls="--"); ax.axvline(tx.mean(), color=TEXT_COLOR, lw=1.3, ls="--")
            ax.set_title(f"{CAPTION_LABELS[caption]}\nΔμ_axis={abs(im.mean()-tx.mean()):.3f}")
            ax.set_xlabel("Gap-axis projection"); clean_axes(ax); ax.set_xlim(-lim, lim)
        axes[0].set_ylabel("Density"); fig.suptitle(f"Gap-axis distribution: {MODEL_LABELS[model]}", y=1.02, fontsize=13, fontweight="bold")
        add_modality_legend(fig, bbox=(0.5, -0.05)); fig.tight_layout(rect=(0, 0.06, 1, 0.94)); savefig(fig, f"gap_axis_{model}_1x3")


def plot_similarity():
    rng = np.random.default_rng(SEED)
    shift = 137
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.1), sharex=True, sharey=True)
    for r, model in enumerate(MODELS):
        for c, caption in enumerate(CAPTIONS):
            ax = axes[r, c]
            im = features[key(model, caption, "image")]; tx = features[key(model, caption, "text")]
            pos = np.sum(im * tx, axis=1)
            neg = np.sum(im * np.roll(tx, shift, axis=0), axis=1)
            bins = np.linspace(-1, 1, 45)
            ax.hist(pos, bins=bins, density=True, color=MODEL_COLORS[model], alpha=0.31, edgecolor="none")
            ax.hist(neg, bins=bins, density=True, color="#9AA4B2", alpha=0.20, edgecolor="none")
            kde_line(ax, pos, MODEL_COLORS[model], lw=1.7, label="Positive KDE")
            kde_line(ax, neg, "#6B7280", lw=1.25, linestyle="--", label="Negative KDE")
            ax.axvline(pos.mean(), color=MODEL_COLORS[model], lw=1.2, ls="-")
            ax.axvline(neg.mean(), color="#6B7280", lw=1.0, ls="--")
            ax.text(0.03, 0.94, f"μ+={pos.mean():.3f}\nμ−={neg.mean():.3f}", transform=ax.transAxes, va="top", fontsize=8, color=DARK)
            if r == 0: ax.set_title(CAPTION_LABELS[caption])
            if c == 0: ax.set_ylabel(f"{MODEL_LABELS[model]}\nDensity")
            if r == 2: ax.set_xlabel("Paired cosine similarity")
            clean_axes(ax); ax.set_xlim(-1, 1)
    fig.suptitle("Paired vs. shifted-negative cosine distributions", y=0.995, fontsize=13, fontweight="bold")
    add_modality_legend(fig, extra=[Patch(facecolor="#9AA4B2", alpha=0.35, label="Shifted negative")], bbox=(0.5, -0.002))
    fig.tight_layout(rect=(0, 0.04, 1, 0.975)); savefig(fig, "similarity_positive_negative_3x3")


def plot_mask_displacement():
    gap_axis = full_gap_axes()["ours_masked"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    vals = []
    for caption in CAPTIONS:
        native = features[key("ours_native", caption, "image")]
        masked = features[key("ours_masked", caption, "image")]
        vals.append((masked - native) @ gap_axis)
    parts = ax.violinplot(vals, positions=np.arange(3), showmeans=False, showmedians=True, showextrema=False, widths=0.78)
    for body in parts["bodies"]:
        body.set_facecolor(MODEL_COLORS["ours_masked"]); body.set_edgecolor("none"); body.set_alpha(0.45)
    parts["cmedians"].set_color(MODEL_COLORS["ours_masked"]); parts["cmedians"].set_linewidth(1.8)
    bp = ax.boxplot(vals, positions=np.arange(3), widths=0.16, showfliers=False, patch_artist=True, boxprops=dict(facecolor="white", edgecolor=DARK, linewidth=0.9), medianprops=dict(color=DARK, linewidth=1.4), whiskerprops=dict(color=DARK, linewidth=0.8), capprops=dict(color=DARK, linewidth=0.8))
    means = [v.mean() for v in vals]
    ax.scatter(np.arange(3), means, s=46, color=MODEL_COLORS["ours_masked"], edgecolor="white", linewidth=0.7, zorder=5, label="Mean displacement")
    for i, (v, m) in enumerate(zip(vals, means)):
        toward_text = np.mean(v < 0) * 100
        ax.text(i, ax.get_ylim()[1] * 0.92, f"mean={m:+.3f}\n{toward_text:.0f}% textward", ha="center", va="top", fontsize=8, color=DARK)
    ax.axhline(0, color="#6B7280", lw=1.1, ls="--")
    ax.set_xticks(np.arange(3), [CAPTION_LABELS[c] for c in CAPTIONS]); ax.set_ylabel("Projection of (Masked − Native) image feature\non Ours Full gap axis (negative = toward text)")
    ax.set_title("Mask-induced image-feature displacement", fontsize=13, fontweight="bold", pad=12)
    clean_axes(ax); ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); savefig(fig, "mask_displacement_gap_axis_violin")

    # Centroid movement view: paired native/masked positions on the same fixed axis.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(3)
    native_pos = []; masked_pos = []; text_pos = []
    for caption in CAPTIONS:
        native_pos.append((features[key("ours_native", caption, "image")].mean(0) @ gap_axis))
        masked_pos.append((features[key("ours_masked", caption, "image")].mean(0) @ gap_axis))
        text_pos.append((features[key("ours_masked", caption, "text")].mean(0) @ gap_axis))
    ax.plot(x, native_pos, color=MODEL_COLORS["ours_native"], marker="o", lw=2, label="Native image centroid")
    ax.plot(x, masked_pos, color=MODEL_COLORS["ours_masked"], marker="o", lw=2, label="Masked image centroid")
    ax.plot(x, text_pos, color=TEXT_COLOR, marker="s", lw=1.5, ls="--", label="Text centroid")
    for i in range(3): ax.annotate("", xy=(i, masked_pos[i]), xytext=(i, native_pos[i]), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
    ax.set_xticks(x, [CAPTION_LABELS[c] for c in CAPTIONS]); ax.set_ylabel("Centroid projection on Ours Full gap axis"); ax.set_title("Centroid movement after masking", fontsize=13, fontweight="bold", pad=12)
    clean_axes(ax); ax.legend(frameon=False, ncol=3, loc="best"); fig.tight_layout(); savefig(fig, "mask_centroid_shift_gap_axis")


def bootstrap_metrics(img, tx, reps=BOOT_REPS):
    rng = np.random.default_rng(SEED)
    n = len(img); out = np.zeros((reps, 3), dtype=np.float64)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        ib = img[idx]; tb = tx[idx]
        out[r, 0] = np.linalg.norm(ib.mean(0) - tb.mean(0))
        out[r, 1] = np.mean(np.sum(ib * tb, axis=1))
        out[r, 2] = np.mean(np.linalg.norm(ib - tb, axis=1))
    return np.percentile(out, [2.5, 50, 97.5], axis=0).T


def compute_metrics(raw):
    rows = []
    consistency = []
    for model in MODELS:
        for caption in CAPTIONS:
            im = features[key(model, caption, "image")]; tx = features[key(model, caption, "text")]
            centroid_gap = np.linalg.norm(im.mean(0) - tx.mean(0))
            cosine = np.mean(np.sum(im * tx, axis=1))
            l2 = np.mean(np.linalg.norm(im - tx, axis=1))
            im_norm = np.linalg.norm(im.mean(0)); tx_norm = np.linalg.norm(tx.mean(0))
            raw_row = raw[(raw.model == model) & (raw.caption_setting == caption)].iloc[0]
            ci = bootstrap_metrics(im, tx)
            rows.append({"model": model, "model_label": MODEL_LABELS[model], "caption_setting": caption, "caption_label": CAPTION_LABELS[caption], "centroid_gap": centroid_gap, "centroid_gap_ci_low": ci[0,0], "centroid_gap_ci_high": ci[0,2], "paired_cosine": cosine, "paired_cosine_ci_low": ci[1,0], "paired_cosine_ci_high": ci[1,2], "paired_l2": l2, "paired_l2_ci_low": ci[2,0], "paired_l2_ci_high": ci[2,2], "image_centroid_norm": im_norm, "text_centroid_norm": tx_norm, "mask_keep_ratio": raw_row.mask_keep_ratio})
            consistency.append({"model": model, "caption_setting": caption, "metric": "centroid_gap", "recomputed": centroid_gap, "reported": raw_row.centroid_gap, "absolute_error": abs(centroid_gap-raw_row.centroid_gap)})
            consistency.append({"model": model, "caption_setting": caption, "metric": "paired_cosine", "recomputed": cosine, "reported": raw_row.mean_paired_cosine, "absolute_error": abs(cosine-raw_row.mean_paired_cosine)})
            consistency.append({"model": model, "caption_setting": caption, "metric": "paired_l2", "recomputed": l2, "reported": raw_row.mean_paired_l2, "absolute_error": abs(l2-raw_row.mean_paired_l2)})
    metrics = pd.DataFrame(rows); pd.DataFrame(consistency).to_csv(TAB / "metrics_consistency_check.csv", index=False); metrics.to_csv(TAB / "metrics_with_bootstrap_ci.csv", index=False)
    return metrics


def plot_trends(metrics):
    specs = [("centroid_gap", "Centroid gap (L2 in 512D)", "centroid_gap_trend"), ("paired_cosine", "Paired cosine similarity", "paired_cosine_trend"), ("paired_l2", "Paired L2 distance", "paired_l2_trend")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    xpos = np.arange(3)
    for ax, (metric, ylabel, _) in zip(axes, specs):
        for model in MODELS:
            sub = metrics[metrics.model == model].set_index("caption_setting").loc[CAPTIONS]
            y = sub[metric].values; lo = sub[f"{metric}_ci_low"].values; hi = sub[f"{metric}_ci_high"].values
            ax.plot(xpos, y, marker="o", lw=2.1, ms=5.5, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.fill_between(xpos, lo, hi, color=MODEL_COLORS[model], alpha=0.11, linewidth=0)
        ax.set_xticks(xpos, ["Full", "50%\nPrefix", "First\nSentence"]); ax.set_ylabel(ylabel); clean_axes(ax)
        ax.set_xlim(-0.1, 2.1)
    axes[0].legend(frameon=False, loc="best", fontsize=8.3)
    fig.suptitle("Caption completeness changes the modality gap and paired alignment", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); savefig(fig, "caption_completeness_gap_trends_1x3")


def write_params(metrics):
    params = {
        "seed": SEED,
        "input_branch": "analysis/modality-gap-docci-b16-v1",
        "input_commit": "124e89574d6ec586f5eaf4f5ff63f9679b3c89ff",
        "reference_commits": {"pca_v2": "ad5fc58c60a8c5ff5eb3a0788a7c3f832ec61e34", "umap_v1": "69f60d92174f22b2c91396cc5acd18abe84037a1"},
        "input_directory": str(INPUT.relative_to(ROOT)),
        "feature_dimension": 512,
        "sample_count_per_condition": 1000,
        "caption_order": CAPTIONS,
        "models": MODELS,
        "pca": {"solver": "full", "shared_fit": "all image/text arrays for selected models and captions", "component_pairs": [[0,1],[1,2]], "random_state": SEED, "point_alpha": 0.18, "point_size": 6},
        "gap_axis": {"axis_definition": "normalized Full-caption (image centroid - text centroid) per model", "hist_bins": 45, "density": True, "kde_bw_method": 0.3, "centroid_lines": True},
        "similarity": {"positive": "paired image/text dot product", "negative": "text shifted by 137 samples (fixed, all 1000 samples)", "hist_bins": 44, "kde_bw_method": 0.3},
        "mask_displacement": {"vector": "Ours Masked image - Ours Native image", "axis": "Ours Full-caption gap axis", "violin": True, "box_overlay": True},
        "bootstrap": {"enabled": True, "reps": BOOT_REPS, "ci": 0.95, "seed": SEED},
        "raw_metrics_reference": "input/modality_gap_metrics.csv",
        "max_metric_consistency_error": float(pd.read_csv(TAB / "metrics_consistency_check.csv")["absolute_error"].max()),
    }
    with open(ROOT / "PLOT_PARAMS.json", "w", encoding="utf-8") as f: json.dump(params, f, indent=2, ensure_ascii=False)
    sweep = pd.DataFrame([
        {"family": "PCA", "variant": "shared", "components": "PC1-PC2", "models": "Long-CLIP + Ours Masked", "selected_for_main": True},
        {"family": "PCA", "variant": "shared", "components": "PC2-PC3", "models": "Long-CLIP + Ours Masked", "selected_for_main": False},
        {"family": "PCA", "variant": "per_model", "components": "PC1-PC2", "models": "each model", "selected_for_main": False},
        {"family": "PCA", "variant": "per_model", "components": "PC2-PC3", "models": "each model", "selected_for_main": False},
        {"family": "Gap-axis", "variant": "hist+KDE", "components": "1D", "models": "all", "selected_for_main": True},
        {"family": "Similarity", "variant": "positive+negative", "components": "1D", "models": "all", "selected_for_main": False},
        {"family": "Mask displacement", "variant": "violin+box", "components": "1D", "models": "Ours", "selected_for_main": True},
    ])
    sweep.to_csv(ROOT / "PARAMETER_SWEEP_SUMMARY.csv", index=False)


def main():
    global features
    features, raw = load_data()
    # All requested PCA variants are retained.
    plot_pca_shared(["longclip", "ours_masked"], "pca_shared_longclip_masked_pc1_pc2_2x3", (0, 1), "Shared PCA (PC1–PC2): Long-CLIP + Ours Masked")
    plot_pca_shared(["longclip", "ours_masked"], "pca_shared_longclip_masked_pc2_pc3_2x3", (1, 2), "Shared PCA (PC2–PC3): Long-CLIP + Ours Masked")
    plot_pca_shared(["ours_native"], "pca_shared_ours_native_pc1_pc2_1x3", (0, 1), "Shared PCA (PC1–PC2): Ours Native")
    plot_pca_shared(["ours_native"], "pca_shared_ours_native_pc2_pc3_1x3", (1, 2), "Shared PCA (PC2–PC3): Ours Native")
    for model in MODELS:
        label = model
        plot_pca_per_model(model, (0, 1), f"pca_per_model_{label}_pc1_pc2_1x3")
        plot_pca_per_model(model, (1, 2), f"pca_per_model_{label}_pc2_pc3_1x3")
    plot_gap_axis(); plot_similarity(); plot_mask_displacement()
    metrics = compute_metrics(raw); plot_trends(metrics); write_params(metrics)
    # A compact machine-readable manifest of all generated figures.
    manifest = sorted(str(p.relative_to(ROOT)) for p in FIG.rglob("*") if p.is_file())
    with open(ROOT / "FIGURE_MANIFEST.json", "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} figure files in {FIG}")


if __name__ == "__main__":
    main()
