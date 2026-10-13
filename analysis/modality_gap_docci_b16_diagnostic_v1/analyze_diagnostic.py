#!/usr/bin/env python3
"""Diagnose DOCCI Ours-masked image double-cluster behavior."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SETTINGS = ("full", "prefix50", "first_sentence")
SETTING_LABELS = {"full": "Full", "prefix50": "50% prefix", "first_sentence": "First sentence"}
MODELS = ("longclip", "ours_masked", "ours_native")
MODEL_LABELS = {"longclip": "Long-CLIP", "ours_masked": "Ours masked", "ours_native": "Ours native"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def kmeans_info(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    km = KMeans(n_clusters=2, n_init=50, random_state=0, algorithm="lloyd").fit(x)
    labels = km.labels_.astype(np.int8)
    counts = np.bincount(labels, minlength=2)
    order = np.argsort(-counts)
    major, minor = int(order[0]), int(order[1])
    sil = float(silhouette_score(x, labels, metric="euclidean"))
    pca2 = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x)
    sil_pca2 = float(silhouette_score(pca2, labels, metric="euclidean"))
    stability = []
    for seed in range(10):
        other = KMeans(n_clusters=2, n_init=20, random_state=seed, algorithm="lloyd").fit(x).labels_
        stability.append(float(adjusted_rand_score(labels, other)))
    centers = km.cluster_centers_
    return {
        "labels": labels,
        "counts": counts.tolist(),
        "major_label": major,
        "minor_label": minor,
        "major_count": int(counts[major]),
        "minor_count": int(counts[minor]),
        "minor_fraction": float(counts[minor] / len(labels)),
        "silhouette_k2": sil,
        "silhouette_pca2_with_512d_labels": sil_pca2,
        "kmeans_seed_stability_ari_min": float(min(stability)),
        "kmeans_seed_stability_ari_mean": float(np.mean(stability)),
        "inertia_k2": float(km.inertia_),
        "centroid_distance": float(np.linalg.norm(centers[0] - centers[1])),
        "centers": centers.tolist(),
    }


def pairwise_transition(a: np.ndarray, b: np.ndarray) -> dict:
    # Canonicalize each partition independently: 0=majority, 1=minority.
    # KMeans label numbers are arbitrary across settings/models.
    a_counts, b_counts = np.bincount(a, minlength=2), np.bincount(b, minlength=2)
    a = (a == int(np.argmin(a_counts))).astype(np.int8)
    b = (b == int(np.argmin(b_counts))).astype(np.int8)
    out = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            out[i, j] = int(((a == i) & (b == j)).sum())
    return {"matrix": out.tolist(), "ari": float(adjusted_rand_score(a, b)), "minority_jaccard": float(((a == 1) & (b == 1)).sum() / max(1, ((a == 1) | (b == 1)).sum()))}


def cluster_mask_stats(labels, mask, prob, logit, gate, components, pair_cos, raw_norm, pre_norm, masked_norm, text_cond_norm, visual_cond_norm):
    result = {}
    for c in (0, 1):
        idx = labels == c
        def mean_std(v):
            v = np.asarray(v)[idx]
            return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
        def selected_summary(v):
            v = np.asarray(v)
            return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
        binary = mask[idx]
        flat_centroid = binary.mean(0)
        result[str(c)] = {
            "count": int(idx.sum()),
            "mask_keep_ratio": selected_summary(binary.mean(1)),
            "selected_channels": selected_summary(binary.sum(1)),
            "probability_mean": selected_summary(prob[idx].mean(1)),
            "probability_std": selected_summary(prob[idx].std(1)),
            "logit_mean": selected_summary(logit[idx].mean(1)),
            "logit_std": selected_summary(logit[idx].std(1)),
            "saturated_fraction": selected_summary(((prob[idx] < .01) | (prob[idx] > .99)).mean(1)),
            "gate_mean": selected_summary(gate[idx].mean(1)),
            "text_component_norm": selected_summary(components[idx, 0]),
            "visual_component_norm": selected_summary(components[idx, 1]),
            "text_condition_norm": selected_summary(text_cond_norm[idx]),
            "visual_condition_norm": selected_summary(visual_cond_norm[idx]),
            "raw_image_norm": selected_summary(raw_norm[idx]),
            "masked_pre_normalization_norm": selected_summary(pre_norm[idx]),
            "masked_post_normalization_norm": selected_summary(masked_norm[idx]),
            "paired_cosine": selected_summary(pair_cos[idx]),
            "mask_channel_frequency_mean": float(flat_centroid.mean()),
            "mask_channel_frequency_std": float(flat_centroid.std()),
            "mask_channel_frequency_min": float(flat_centroid.min()),
            "mask_channel_frequency_max": float(flat_centroid.max()),
        }
    c0, c1 = mask[labels == 0].mean(0), mask[labels == 1].mean(0)
    result["between_cluster_mask_centroid_l1"] = float(np.abs(c0 - c1).mean())
    result["between_cluster_mask_centroid_hamming"] = float((np.rint(c0) != np.rint(c1)).mean())
    result["unique_mask_rows"] = int(np.unique(mask, axis=0).shape[0])
    counts = np.unique(mask, axis=0, return_counts=True)[1]
    result["max_duplicate_mask_fraction"] = float(counts.max() / len(mask))
    return result


def plot_cluster_panels(out, projected, labels, title, row_names, xlim=None, ylim=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    colors = np.where(labels == 0, "#2166ac", "#7b3294")
    for r, row in enumerate(row_names):
        for c, setting in enumerate(SETTINGS):
            ax = axes[r, c]
            pts = projected[row][setting]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=.42, c=colors, linewidths=0)
            cent = np.vstack([pts[labels == k].mean(0) for k in (0, 1)])
            ax.scatter(cent[:, 0], cent[:, 1], marker="X", s=110, c=["#08306b", "#762a83"], edgecolors="white", linewidths=.8)
            ax.set_title(f"{row} — {SETTING_LABELS[setting]}")
            if xlim: ax.set_xlim(*xlim)
            if ylim: ax.set_ylim(*ylim)
            ax.grid(alpha=.18); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle(title, y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.input_dir / "features.npz")
    masks = np.load(args.mask_dir / "mask_arrays.npz")
    mask_stats = json.loads((args.mask_dir / "per_sample_mask_stats.json").read_text())
    samples = json.loads((args.input_dir / "samples.json").read_text())
    caption_lengths = json.loads((args.input_dir / "caption_lengths.json").read_text())
    if len(samples) != 1000:
        raise RuntimeError("sample count mismatch")
    image_ids = [x["image_id"] for x in samples]
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("duplicate image IDs in fixed sample list")
    for model in MODELS:
        for setting in SETTINGS:
            if z[f"{model}_{setting}_image"].shape != (1000, 512):
                raise RuntimeError(f"bad feature shape for {model}_{setting}")

    # KMeans is run directly on the original 512-D normalized image features.
    model_clusters = {}
    for model in MODELS:
        model_clusters[model] = {setting: kmeans_info(z[f"{model}_{setting}_image"]) for setting in SETTINGS}
    ours_labels = {s: model_clusters["ours_masked"][s]["labels"] for s in SETTINGS}

    cluster_rows = []
    for model in MODELS:
        for setting in SETTINGS:
            info = model_clusters[model][setting]
            row = {"model": model, "setting": setting, "major_count": info["major_count"], "minor_count": info["minor_count"], "minor_fraction": info["minor_fraction"], "silhouette_k2": info["silhouette_k2"], "silhouette_pca2_with_512d_labels": info["silhouette_pca2_with_512d_labels"], "kmeans_seed_stability_ari_min": info["kmeans_seed_stability_ari_min"], "kmeans_seed_stability_ari_mean": info["kmeans_seed_stability_ari_mean"], "centroid_distance_512d": info["centroid_distance"]}
            if model == "ours_masked":
                s = setting
                ms = {k: np.asarray(v) for k, v in mask_stats.items() if k.startswith(s + "_")}
                labels = info["labels"]
                mask = masks[f"mask_{s}"]; prob = masks[f"prob_{s}"]; logit = masks[f"logit_{s}"]; gate = masks[f"gate_{s}"]; comp = masks[f"components_{s}"]
                raw = np.asarray(mask_stats[f"{s}_raw_z_norm"]); pre = np.asarray(mask_stats[f"{s}_masked_pre_norm"]); post = np.asarray(mask_stats[f"{s}_masked_norm"]); tc = np.asarray(mask_stats[f"{s}_text_cond_norm"]); vc = np.asarray(mask_stats[f"{s}_visual_cond_norm"]); cos = np.asarray(mask_stats[f"{s}_paired_cosine"])
                per = cluster_mask_stats(labels, mask, prob, logit, gate, comp, cos, raw, pre, post, tc, vc)
                row.update({"major_mask_keep": per[str(info["major_label"])] ["mask_keep_ratio"]["mean"], "minor_mask_keep": per[str(info["minor_label"])] ["mask_keep_ratio"]["mean"], "major_paired_cosine": per[str(info["major_label"])] ["paired_cosine"]["mean"], "minor_paired_cosine": per[str(info["minor_label"])] ["paired_cosine"]["mean"], "mask_centroid_l1": per["between_cluster_mask_centroid_l1"], "mask_centroid_hamming": per["between_cluster_mask_centroid_hamming"], "unique_mask_rows": per["unique_mask_rows"], "max_duplicate_mask_fraction": per["max_duplicate_mask_fraction"]})
                model_clusters[model][setting]["mask_stats"] = per
            cluster_rows.append(row)

    transitions = []
    for a, b in (("full", "prefix50"), ("full", "first_sentence"), ("prefix50", "first_sentence")):
        t = pairwise_transition(ours_labels[a], ours_labels[b])
        t.update({"source": a, "target": b})
        transitions.append(t)
    for setting in SETTINGS:
        for other_model in ("ours_native", "longclip"):
            t = pairwise_transition(ours_labels[setting], model_clusters[other_model][setting]["labels"])
            t.update({"source": f"ours_masked_{setting}", "target": f"{other_model}_{setting}"})
            transitions.append(t)

    # Representative IDs: closest samples to each 512-D KMeans centroid.
    representatives = {}
    for setting in SETTINGS:
        x = z[f"ours_masked_{setting}_image"]
        info = model_clusters["ours_masked"][setting]
        reps = {}
        for c in (info["major_label"], info["minor_label"]):
            idx = np.flatnonzero(info["labels"] == c)
            d = np.linalg.norm(x[idx] - np.asarray(info["centers"])[c], axis=1)
            chosen = idx[np.argsort(d)[:10]]
            reps["major" if c == info["major_label"] else "minor"] = [{"index": int(i), "image_id": samples[i]["image_id"], "image_path": samples[i]["image_path"], "caption": samples[i]["caption"], "first_sentence": samples[i]["versions"]["first_sentence"].get("text", ""), "mask_keep_ratio": float(masks[f"mask_{setting}"][i].mean()), "paired_cosine": float(np.dot(z[f"ours_masked_{setting}_image"][i], z[f"ours_masked_{setting}_text"][i]))} for i in chosen]
        representatives[setting] = reps

    # Shared PCA matching the original main-figure protocol.
    shared_fit = np.concatenate([z[f"{m}_{s}_{kind}"] for m in ("longclip", "ours_masked") for s in SETTINGS for kind in ("image", "text")])
    pca_shared = PCA(n_components=2, svd_solver="full", random_state=0).fit(shared_fit)
    ours_fit = np.concatenate([z[f"ours_masked_{s}_{kind}"] for s in SETTINGS for kind in ("image", "text")])
    pca_ours = PCA(n_components=2, svd_solver="full", random_state=0).fit(ours_fit)
    native_fit = np.concatenate([z[f"ours_{m}_{s}_{kind}"] for m in ("masked", "native") for s in SETTINGS for kind in ("image", "text")])
    pca_native = PCA(n_components=2, svd_solver="full", random_state=0).fit(native_fit)
    projected_shared = {"Ours masked": {s: pca_shared.transform(z[f"ours_masked_{s}_image"]) for s in SETTINGS}}
    projected_ours = {"Ours masked": {s: pca_ours.transform(z[f"ours_masked_{s}_image"]) for s in SETTINGS}}
    projected_native = {"Ours masked": {s: pca_native.transform(z[f"ours_masked_{s}_image"]) for s in SETTINGS}, "Ours native": {s: pca_native.transform(z[f"ours_native_{s}_image"]) for s in SETTINGS}}
    def limits(projected):
        q = np.concatenate([x for row in projected.values() for x in row.values()]); lo, hi = q.min(0), q.max(0); span = np.maximum(hi-lo, 1e-8); return (float(lo[0]-.06*span[0]), float(hi[0]+.06*span[0])), (float(lo[1]-.06*span[1]), float(hi[1]+.06*span[1]))
    xlim, ylim = limits(projected_shared)
    plot_cluster_panels(args.output_dir / "shared_pca_cluster_diagnostic", projected_shared, ours_labels["full"], "Ours masked clusters in shared PCA (Full labels)", ("Ours masked", "Ours masked"), xlim, ylim)
    xlim2, ylim2 = limits(projected_ours)
    plot_cluster_panels(args.output_dir / "ours_only_pca_cluster_diagnostic", projected_ours, ours_labels["full"], "Ours masked clusters in Ours-only PCA (Full labels)", ("Ours masked", "Ours masked"), xlim2, ylim2)
    # Native and masked use one PCA and the same Full-derived cluster labels.
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), squeeze=False); xlim3, ylim3 = limits(projected_native)
    for r, s in enumerate(SETTINGS):
        for c, model in enumerate(("Ours masked", "Ours native")):
            ax = axes[r, c]; pts = projected_native[model][s]; colors = np.where(ours_labels[s] == 0, "#2166ac", "#7b3294"); ax.scatter(pts[:,0], pts[:,1], s=8, alpha=.42, c=colors); ax.set_title(f"{model} — {SETTING_LABELS[s]}"); ax.set_xlim(*xlim3); ax.set_ylim(*ylim3); ax.grid(alpha=.18); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Ours masked vs native in one shared Ours PCA", y=1.0); fig.tight_layout(); fig.savefig(args.output_dir / "native_vs_masked_pca.png", dpi=220, bbox_inches="tight"); fig.savefig(args.output_dir / "native_vs_masked_pca.pdf", bbox_inches="tight"); plt.close(fig)

    # Save compact JSON without serializing 512-D labels/centers repeatedly.
    serial_clusters = {}
    for model in MODELS:
        serial_clusters[model] = {}
        for s in SETTINGS:
            x = dict(model_clusters[model][s]); x.pop("labels", None); x["centers"] = np.asarray(x["centers"]).tolist()
            if "mask_stats" in x: x["mask_stats"] = x["mask_stats"]
            serial_clusters[model][s] = x
    (args.output_dir / "cluster_statistics.json").write_text(json.dumps(serial_clusters, indent=2) + "\n")
    (args.output_dir / "cluster_migration.json").write_text(json.dumps(transitions, indent=2) + "\n")
    (args.output_dir / "representative_samples.json").write_text(json.dumps(representatives, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "pca_diagnostic.json").write_text(json.dumps({"shared": pca_shared.explained_variance_ratio_.tolist(), "ours_only": pca_ours.explained_variance_ratio_.tolist(), "ours_native_masked_shared": pca_native.explained_variance_ratio_.tolist(), "shared_fit_rows": int(shared_fit.shape[0]), "ours_only_fit_rows": int(ours_fit.shape[0]), "native_fit_rows": int(native_fit.shape[0]), "shared_limits": [xlim, ylim], "ours_only_limits": [xlim2, ylim2], "native_limits": [xlim3, ylim3]}, indent=2) + "\n")
    with (args.output_dir / "cluster_statistics.csv").open("w", newline="") as f:
        fields = sorted({k for row in cluster_rows for k in row}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cluster_rows)
    # Detailed per-cluster mechanism table (the nested JSON retains the full
    # distributions; this CSV exposes the directly comparable means/stds).
    mechanism_rows = []
    association_rows = []
    native_distance_rows = []
    for s in SETTINGS:
        info = model_clusters["ours_masked"][s]
        per = info["mask_stats"]
        for c in (info["major_label"], info["minor_label"]):
            x = per[str(c)]
            row = {"setting": s, "cluster_role": "major" if c == info["major_label"] else "minor", "cluster_label": c, "count": x["count"]}
            for name in ("mask_keep_ratio", "selected_channels", "probability_mean", "probability_std", "logit_mean", "logit_std", "saturated_fraction", "gate_mean", "text_component_norm", "visual_component_norm", "text_condition_norm", "visual_condition_norm", "raw_image_norm", "masked_pre_normalization_norm", "masked_post_normalization_norm", "paired_cosine"):
                row[name + "_mean"] = x[name]["mean"]; row[name + "_std"] = x[name]["std"]
            mechanism_rows.append(row)
        feature_labels = info["labels"]
        for name in ("mask", "prob", "logit", "gate"):
            ml = KMeans(n_clusters=2, n_init=50, random_state=0, algorithm="lloyd").fit(masks[f"{name}_{s}"]).labels_
            association_rows.append({"setting": s, "source": name, "feature_cluster_ari": float(adjusted_rand_score(feature_labels, ml)), "source_cluster_silhouette": float(silhouette_score(masks[f"{name}_{s}"], ml)), "source_cluster_0": int((ml == 0).sum()), "source_cluster_1": int((ml == 1).sum())})
        masked = z[f"ours_masked_{s}_image"]; native = z[f"ours_native_{s}_image"]; dist = np.linalg.norm(masked - native, axis=1); cos = np.sum(masked * native, axis=1)
        for c in (info["major_label"], info["minor_label"]):
            idx = info["labels"] == c
            native_distance_rows.append({"setting": s, "cluster_role": "major" if c == info["major_label"] else "minor", "count": int(idx.sum()), "masked_native_l2_mean": float(dist[idx].mean()), "masked_native_l2_std": float(dist[idx].std()), "masked_native_cosine_mean": float(cos[idx].mean()), "masked_native_cosine_std": float(cos[idx].std())})
    with (args.output_dir / "mask_cluster_comparison.csv").open("w", newline="") as f:
        fields = sorted({k for row in mechanism_rows for k in row}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(mechanism_rows)
    with (args.output_dir / "mask_feature_cluster_association.csv").open("w", newline="") as f:
        fields = sorted({k for row in association_rows for k in row}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(association_rows)
    with (args.output_dir / "native_masked_cluster_comparison.csv").open("w", newline="") as f:
        fields = sorted({k for row in native_distance_rows for k in row}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(native_distance_rows)
    caption_rows = []
    for s in SETTINGS:
        labels = model_clusters["ours_masked"][s]["labels"]
        vals = np.asarray([x[s]["kept_bpe_tokens"] for x in caption_lengths])
        full_vals = np.asarray([x["full"]["kept_bpe_tokens"] for x in caption_lengths])
        info = model_clusters["ours_masked"][s]
        for c in (info["major_label"], info["minor_label"]):
            idx = labels == c
            caption_rows.append({"setting": s, "cluster_role": "major" if c == info["major_label"] else "minor", "count": int(idx.sum()), "setting_tokens_mean": float(vals[idx].mean()), "setting_tokens_std": float(vals[idx].std()), "full_tokens_mean": float(full_vals[idx].mean()), "full_tokens_std": float(full_vals[idx].std())})
    with (args.output_dir / "caption_cluster_stats.csv").open("w", newline="") as f:
        fields = sorted({k for row in caption_rows for k in row}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(caption_rows)
    with (args.output_dir / "cluster_migration.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "target", "matrix_00", "matrix_01", "matrix_10", "matrix_11", "ARI", "minority_jaccard"])
        for t in transitions: w.writerow([t["source"], t["target"], *np.asarray(t["matrix"]).reshape(-1).tolist(), t["ari"], t["minority_jaccard"]])
    (args.output_dir / "diagnostic_manifest.json").write_text(json.dumps({"input_features_sha256": sha256(args.input_dir / "features.npz"), "input_samples_sha256": sha256(args.input_dir / "samples.json"), "mask_arrays_sha256": sha256(args.mask_dir / "mask_arrays.npz"), "sample_count": len(samples), "settings": list(SETTINGS), "cluster_method": "KMeans k=2, n_init=50, random_state=0 on original 512-D image features", "no_training": True}, indent=2) + "\n")
    print(json.dumps({"rows": len(cluster_rows), "transitions": transitions, "pca": {"shared": pca_shared.explained_variance_ratio_.tolist(), "ours_only": pca_ours.explained_variance_ratio_.tolist(), "native": pca_native.explained_variance_ratio_.tolist()}}, indent=2))


if __name__ == "__main__":
    main()
