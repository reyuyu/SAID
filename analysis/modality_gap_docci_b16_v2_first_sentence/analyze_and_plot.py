#!/usr/bin/env python3
"""Compute modality-gap metrics and shared-PCA figures for the DOCCI probe."""
import argparse, csv, json, os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAPTIONS = [("full", "Full"), ("prefix50", "50% prefix"), ("first_sentence", "First sentence")]
MODELS = [("longclip", "Long-CLIP"), ("ours_masked", "Ours masked"), ("ours_native", "Ours native")]


def metrics(image, text):
    # Features are normalized at extraction time; renormalize defensively for
    # this audit so all reported quantities are in the original 512-D space.
    image = image.astype(np.float64)
    text = text.astype(np.float64)
    image /= np.linalg.norm(image, axis=1, keepdims=True).clip(min=1e-15)
    text /= np.linalg.norm(text, axis=1, keepdims=True).clip(min=1e-15)
    centroid_gap = float(np.linalg.norm(image.mean(axis=0) - text.mean(axis=0)))
    cosine = np.sum(image * text, axis=1)
    l2 = np.linalg.norm(image - text, axis=1)
    return {
        "centroid_gap": centroid_gap,
        "mean_paired_cosine": float(cosine.mean()),
        "mean_paired_l2": float(l2.mean()),
        "image_centroid_norm": float(np.linalg.norm(image.mean(axis=0))),
        "text_centroid_norm": float(np.linalg.norm(text.mean(axis=0))),
    }


def write_plot(path, panels, title, xlim, ylim, pca_var):
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4), squeeze=False)
    axes = axes[0]
    for ax, (label, im, tx) in zip(axes, panels):
        ax.scatter(im[:, 0], im[:, 1], s=8, alpha=0.30, c="#2166ac", label="Image")
        ax.scatter(tx[:, 0], tx[:, 1], s=8, alpha=0.30, c="#d95f02", label="Text")
        ci, ct = im.mean(0), tx.mean(0)
        ax.scatter([ci[0]], [ci[1]], marker="X", s=105, c="#08306b", edgecolors="white", linewidths=0.8, label="Image centroid")
        ax.scatter([ct[0]], [ct[1]], marker="X", s=105, c="#993404", edgecolors="white", linewidths=0.8, label="Text centroid")
        ax.set_title(label)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title + f"\nShared PCA: PC1={pca_var[0]*100:.2f}%, PC2={pca_var[1]*100:.2f}%", y=1.07, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(inp / "features.npz")
    provenance = json.loads((inp / "provenance.json").read_text())
    lengths = json.loads((inp / "caption_lengths.json").read_text())

    # One PCA fit for all six main panels. Native features are projected with
    # that same transform so their geometry remains comparable.
    main_features = []
    for model in ("longclip", "ours_masked"):
        for cap, _ in CAPTIONS:
            main_features += [z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"]]
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    pca.fit(np.concatenate(main_features, axis=0))
    projected = {k: pca.transform(z[k]) for k in z.files}
    all_main = np.concatenate([projected[k] for k in projected if k.startswith("longclip_") or k.startswith("ours_masked_")], axis=0)
    # Symmetric margins and a single range for all panels.
    lo, hi = all_main.min(axis=0), all_main.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    xlim = (float(lo[0] - 0.06*span[0]), float(hi[0] + 0.06*span[0]))
    ylim = (float(lo[1] - 0.06*span[1]), float(hi[1] + 0.06*span[1]))

    main_panels = []
    for model, model_label in (("longclip", "Long-CLIP"), ("ours_masked", "Ours (masked)")):
        for cap, cap_label in CAPTIONS:
            main_panels.append((f"{model_label} — {cap_label}", projected[f"{model}_{cap}_image"], projected[f"{model}_{cap}_text"]))
    # Make a 2x3 figure explicitly (the helper is 1-row, so use a dedicated layout).
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    for row, (model, model_label) in enumerate((("longclip", "Long-CLIP"), ("ours_masked", "Ours (masked)"))):
        for col, (cap, cap_label) in enumerate(CAPTIONS):
            ax = axes[row, col]
            im, tx = projected[f"{model}_{cap}_image"], projected[f"{model}_{cap}_text"]
            ax.scatter(im[:,0], im[:,1], s=7, alpha=.28, c="#2166ac", label="Image")
            ax.scatter(tx[:,0], tx[:,1], s=7, alpha=.28, c="#d95f02", label="Text")
            ci, ct = im.mean(0), tx.mean(0)
            ax.scatter([ci[0]], [ci[1]], marker="X", s=100, c="#08306b", edgecolors="white", linewidths=.8)
            ax.scatter([ct[0]], [ct[1]], marker="X", s=100, c="#993404", edgecolors="white", linewidths=.8)
            ax.set_title(f"{model_label} — {cap_label}")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.18, linewidth=.6)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    handles = [plt.Line2D([], [], marker='o', linestyle='', color="#2166ac", alpha=.6, label="Image"), plt.Line2D([], [], marker='o', linestyle='', color="#d95f02", alpha=.6, label="Text"), plt.Line2D([], [], marker='X', linestyle='', color="#08306b", label="Image centroid"), plt.Line2D([], [], marker='X', linestyle='', color="#993404", label="Text centroid")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle(f"DOCCI modality gap — shared PCA (PC1={pca.explained_variance_ratio_[0]*100:.2f}%, PC2={pca.explained_variance_ratio_[1]*100:.2f}%)", y=1.045, fontsize=15)
    fig.tight_layout()
    fig.savefig(out / "modality_gap_main_2x3.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "modality_gap_main_2x3.pdf", bbox_inches="tight")
    plt.close(fig)

    native_panels = [(f"Ours native — {cap_label}", projected[f"ours_native_{cap}_image"], projected[f"ours_native_{cap}_text"]) for cap, cap_label in CAPTIONS]
    write_plot(out / "modality_gap_ours_native_1x3.png", native_panels, "DOCCI Ours native modality gap", xlim, ylim, pca.explained_variance_ratio_)
    write_plot(out / "modality_gap_ours_native_1x3.pdf", native_panels, "DOCCI Ours native modality gap", xlim, ylim, pca.explained_variance_ratio_)

    rows = []
    keep = provenance.get("mask_keep_ratio", {})
    for model, model_label in MODELS:
        for cap, cap_label in CAPTIONS:
            m = metrics(z[f"{model}_{cap}_image"], z[f"{model}_{cap}_text"])
            m.update({"model": model, "model_label": model_label, "caption_setting": cap, "caption_label": cap_label, "sample_count": int(z[f"{model}_{cap}_image"].shape[0])})
            if model == "ours_masked":
                m["mask_keep_ratio"] = float(keep[cap]["mean"])
            else:
                m["mask_keep_ratio"] = ""
            rows.append(m)
    fields = ["model", "model_label", "caption_setting", "caption_label", "sample_count", "centroid_gap", "mean_paired_cosine", "mean_paired_l2", "mask_keep_ratio", "image_centroid_norm", "text_centroid_norm"]
    with (out / "modality_gap_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out / "pca_explained_variance.json").write_text(json.dumps({"fit_feature_count": int(np.concatenate(main_features).shape[0]), "components": 2, "explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "xlim": xlim, "ylim": ylim}, indent=2))
    length_summary = {cap: {"mean_tokens": float(np.mean([x[cap]["kept_bpe_tokens"] for x in lengths])), "min_tokens": int(np.min([x[cap]["kept_bpe_tokens"] for x in lengths])), "max_tokens": int(np.max([x[cap]["kept_bpe_tokens"] for x in lengths]))} for cap, _ in CAPTIONS}
    (out / "caption_length_summary.json").write_text(json.dumps(length_summary, indent=2))
    (out / "provenance_analysis.json").write_text(json.dumps({"extract_provenance": provenance, "pca_fit": "all Long-CLIP and Ours masked image/text features", "normalization": "L2 normalization in 512-D before metrics and PCA input", "output_files": ["modality_gap_main_2x3.png", "modality_gap_main_2x3.pdf", "modality_gap_ours_native_1x3.png", "modality_gap_ours_native_1x3.pdf"]}, indent=2))

    def find(model, cap): return next(r for r in rows if r["model"] == model and r["caption_setting"] == cap)
    lines = ["# DOCCI Modality Gap — Long-CLIP vs Sema-CLIP", "", "This is a fixed-seed (0), 1,000-pair DOCCI exploratory analysis. Metrics are computed in the original 512-D L2-normalized feature space; plots use one shared PCA fit for all six main panels.", "", f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.6f}, PC2={pca.explained_variance_ratio_[1]:.6f}.", "", "## Caption lengths (BPE tokens including SOT/EOT)", "", "| Setting | Mean | Min | Max |", "|---|---:|---:|---:|"]
    for cap, label in CAPTIONS:
        s = json.loads((out / "caption_length_summary.json").read_text())[cap]; lines.append(f"| {label} | {s['mean_tokens']:.2f} | {s['min_tokens']} | {s['max_tokens']} |")
    lines += ["", "## Metrics", "", "| Model | Caption | Centroid gap | Paired cosine | Paired L2 | Mask keep |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        k = r['mask_keep_ratio']; k = f"{k:.4f}" if isinstance(k, float) else "—"
        lines.append(f"| {r['model_label']} | {r['caption_label']} | {r['centroid_gap']:.6f} | {r['mean_paired_cosine']:.6f} | {r['mean_paired_l2']:.6f} | {k} |")
    lines += ["", "## Trend summary", ""]
    for model, label in MODELS:
        vals = [find(model, c)['centroid_gap'] for c, _ in CAPTIONS]
        lines.append(f"- {label} centroid gap Full→50%→50% prefix→First sentence: " + " → ".join(f"{v:.6f}" for v in vals))
    lines += ["", "The plot is descriptive and does not establish retrieval gains or causality. The Ours masked image feature follows the training-time caption-conditioned mask path; Ours native is an explicit no-mask control."]
    (out / "MODALITY_GAP_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "metrics_csv": str(out / 'modality_gap_metrics.csv')}, indent=2))


if __name__ == "__main__":
    main()
