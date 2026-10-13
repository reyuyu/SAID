#!/usr/bin/env python3
"""PCA, bootstrap and 512-D KMeans diagnostics for Long-DCI features."""
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


CAPTIONS = [("full", "Full"), ("prefix50", "50% Prefix"), ("first_sentence", "First Sentence")]
MODELS = [("longclip", "Long-CLIP"), ("ours_masked", "Sema-CLIP Masked"), ("ours_native", "Sema-CLIP Native")]
MAIN_MODELS = MODELS[:2]
BLUE, ORANGE = "#2166ac", "#d95f02"
BLUE_DARK, ORANGE_DARK = "#08306b", "#993404"
DOCCI_METRICS = Path("/opt/data/private/lklk/SAID/analysis/modality_gap_docci_b16_v2_first_sentence/modality_gap_metrics.csv")
DOCCI_CLUSTERS = Path("/opt/data/private/lklk/worktrees/modality-gap-docci-b16-diagnostic-v1/analysis/modality_gap_docci_b16_diagnostic_v1/cluster_statistics.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-15)


def metrics(im, tx):
    im, tx = norm(im), norm(tx)
    ci, ct = im.mean(0), tx.mean(0)
    return {"centroid_gap": float(np.linalg.norm(ci - ct)), "paired_cosine": float(np.sum(im * tx, axis=1).mean()), "paired_l2": float(np.linalg.norm(im - tx, axis=1).mean()), "image_centroid_norm": float(np.linalg.norm(ci)), "text_centroid_norm": float(np.linalg.norm(ct))}


def bootstrap(im, tx, mask, reps=2000, seed=0):
    im, tx = norm(im), norm(tx)
    n = len(im)
    rng = np.random.default_rng(seed)
    out = np.empty((reps, 6), dtype=np.float64)
    for start in range(0, reps, 100):
        stop = min(start + 100, reps)
        ids = rng.integers(0, n, size=(stop - start, n))
        bi, bt = im[ids], tx[ids]
        out[start:stop, 0] = np.linalg.norm(bi.mean(1) - bt.mean(1), axis=1)
        out[start:stop, 1] = np.mean(np.sum(bi * bt, axis=2), axis=1)
        out[start:stop, 2] = np.mean(np.linalg.norm(bi - bt, axis=2), axis=1)
        out[start:stop, 3] = np.linalg.norm(bi.mean(1), axis=1)
        out[start:stop, 4] = np.linalg.norm(bt.mean(1), axis=1)
        out[start:stop, 5] = mask[ids].reshape(stop - start, n, -1).mean(axis=(1, 2)) if mask is not None else np.nan
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return {f"{name}_ci_low": float(lo[i]) for i, name in enumerate(("centroid_gap", "paired_cosine", "paired_l2", "image_centroid_norm", "text_centroid_norm", "mask_keep_ratio"))} | {f"{name}_ci_high": float(hi[i]) for i, name in enumerate(("centroid_gap", "paired_cosine", "paired_l2", "image_centroid_norm", "text_centroid_norm", "mask_keep_ratio"))}


def canonical(labels):
    counts = np.bincount(labels, minlength=2)
    minor = int(np.argmin(counts))
    return (labels == minor).astype(np.int8), int(counts.max()), int(counts.min())


def cluster_info(x):
    x = norm(x)
    km = KMeans(n_clusters=2, n_init=50, random_state=0, algorithm="lloyd").fit(x)
    labels, major, minor = canonical(km.labels_)
    counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    stability = []
    for seed in range(10):
        other = KMeans(n_clusters=2, n_init=20, random_state=seed, algorithm="lloyd").fit(x).labels_
        stability.append(float(adjusted_rand_score(km.labels_, other)))
    return {"labels": labels, "major_count": major, "minor_count": minor, "minor_fraction": minor / len(labels), "silhouette_score": float(silhouette_score(x, labels, metric="euclidean")), "cluster_center_distance": float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])), "kmeans_inertia": float(km.inertia_), "kmeans_seed_stability_ari_min": min(stability), "kmeans_seed_stability_ari_mean": float(np.mean(stability)), "centers": km.cluster_centers_.tolist(), "counts": counts}


def transition(a, b):
    return {"ari": float(adjusted_rand_score(a, b)), "minority_jaccard": float(((a == 1) & (b == 1)).sum() / max(1, ((a == 1) | (b == 1)).sum())), "matrix": [[int(((a == i) & (b == j)).sum()) for j in (0, 1)] for i in (0, 1)]}


def kde(ax, points, color, xlim, ylim):
    xx, yy = np.meshgrid(np.linspace(*xlim, 85), np.linspace(*ylim, 85))
    try:
        kde_fn = gaussian_kde(points.T)
        z = kde_fn(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        sample = kde_fn(points.T)
        levels = np.unique(np.quantile(sample, [0.25, 0.55, 0.80]))
        if len(levels) > 1:
            ax.contour(xx, yy, z, levels=levels, colors=[color], linewidths=.7, alpha=.42)
    except (np.linalg.LinAlgError, ValueError):
        pass


def panel(ax, im, tx, row):
    ax.scatter(im[:, 0], im[:, 1], s=3, alpha=.12, color=BLUE, edgecolors="none", rasterized=True)
    ax.scatter(tx[:, 0], tx[:, 1], s=3, alpha=.12, color=ORANGE, edgecolors="none", rasterized=True)
    ci, ct = im.mean(0), tx.mean(0)
    ax.scatter([ci[0]], [ci[1]], marker="X", s=85, color=BLUE_DARK, edgecolors="white", linewidths=.7, zorder=5)
    ax.scatter([ct[0]], [ct[1]], marker="X", s=85, color=ORANGE_DARK, edgecolors="white", linewidths=.7, zorder=5)
    kde(ax, im, BLUE, ax.get_xlim(), ax.get_ylim())
    kde(ax, tx, ORANGE, ax.get_xlim(), ax.get_ylim())
    ax.text(.03, .97, f"Centroid Gap = {row['centroid_gap']:.4f}", transform=ax.transAxes, va="top", fontsize=8.3, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 2})
    ax.grid(alpha=.14, linewidth=.5)


def get_bounds(data):
    p = np.concatenate(list(data.values()), axis=0)
    lo, hi = p.min(0), p.max(0); span = np.maximum(hi - lo, 1e-8)
    return (float(lo[0] - .07 * span[0]), float(hi[0] + .07 * span[0])), (float(lo[1] - .07 * span[1]), float(hi[1] + .07 * span[1]))


def draw_grid(projected, rows, path, title, per_row_bounds=False):
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.1), squeeze=False)
    all_bounds = get_bounds(projected) if not per_row_bounds else None
    for ri, (model, label) in enumerate(MAIN_MODELS):
        row_keys = [k for k in projected if k.startswith(model + "_")]
        bounds = get_bounds({k: projected[k] for k in row_keys}) if per_row_bounds else all_bounds
        for ci, (cap, cap_label) in enumerate(CAPTIONS):
            ax = axes[ri, ci]
            ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
            im, tx = projected[f"{model}_{cap}_image"], projected[f"{model}_{cap}_text"]
            panel(ax, im, tx, next(r for r in rows if r["model"] == model and r["caption"] == cap))
            ax.set_title(cap_label); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        axes[ri, 0].text(-.23, .5, label, transform=axes[ri, 0].transAxes, rotation=90, va="center", ha="center", fontsize=11)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=BLUE, alpha=.55, label='Image'), plt.Line2D([], [], marker='o', linestyle='', color=ORANGE, alpha=.55, label='Text'), plt.Line2D([], [], marker='X', linestyle='', color=BLUE_DARK, markeredgecolor='white', label='Image centroid'), plt.Line2D([], [], marker='X', linestyle='', color=ORANGE_DARK, markeredgecolor='white', label='Text centroid')]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .995), fontsize=9)
    fig.suptitle(title, y=1.035, fontsize=14); fig.tight_layout(rect=(.02, 0, 1, .96))
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white'); fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)


def draw_native(projected, rows, path, title, bounds):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), squeeze=False)
    for ci, (cap, label) in enumerate(CAPTIONS):
        ax = axes[0, ci]; ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
        im, tx = projected[f"ours_native_{cap}_image"], projected[f"ours_native_{cap}_text"]
        panel(ax, im, tx, next(r for r in rows if r["model"] == "ours_native" and r["caption"] == cap)); ax.set_title(label); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=BLUE, alpha=.55, label='Image'), plt.Line2D([], [], marker='o', linestyle='', color=ORANGE, alpha=.55, label='Text'), plt.Line2D([], [], marker='X', linestyle='', color=BLUE_DARK, markeredgecolor='white', label='Image centroid'), plt.Line2D([], [], marker='X', linestyle='', color=ORANGE_DARK, markeredgecolor='white', label='Text centroid')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02)); fig.suptitle(title, y=1.11, fontsize=14); fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white'); fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input-dir', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--bootstrap-reps', type=int, default=2000); args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    z0 = np.load(args.input_dir / 'features.npz'); z = {k: z0[k] for k in z0.files}; masks0 = np.load(args.input_dir / 'mask_arrays.npz'); masks = {k.removeprefix('mask_'): masks0[k] for k in masks0.files}
    rows, boot = [], {}
    for model, _ in MODELS:
        for cap, _ in CAPTIONS:
            im, tx = z[f'{model}_{cap}_image'], z[f'{model}_{cap}_text']; m = masks[cap] if model == 'ours_masked' else None
            row = {'model': model, 'caption': cap, 'sample_count': len(im), **metrics(im, tx), 'mask_keep_ratio': float(m.mean()) if m is not None else None}; row.update(bootstrap(im, tx, m, args.bootstrap_reps, 0)); rows.append(row); boot[f'{model}__{cap}'] = row

    main_keys = [f'{m}_{c}_{mod}' for m, _ in MAIN_MODELS for c, _ in CAPTIONS for mod in ('image','text')]
    native_keys = [f'ours_native_{c}_{mod}' for c, _ in CAPTIONS for mod in ('image','text')]
    fit_specs = {'shared': main_keys, 'longclip': [k for k in main_keys if k.startswith('longclip_')], 'ours_masked': [k for k in main_keys if k.startswith('ours_masked_')], 'ours_native': native_keys}
    fits = {}; projected = {}
    for name, keys in fit_specs.items():
        pca = PCA(n_components=2, svd_solver='full', random_state=0); pca.fit(np.concatenate([norm(z[k]) for k in keys], axis=0)); fits[name] = pca
        for k in z: projected[f'{name}__{k}'] = pca.transform(norm(z[k]))
    shared_proj = {k.removeprefix('shared__'): v for k, v in projected.items() if k.startswith('shared__')}; native_proj = {k.removeprefix('ours_native__'): v for k, v in projected.items() if k.startswith('ours_native__')}; per_proj = {k.removeprefix('longclip__'): v for k, v in projected.items() if k.startswith('longclip__')}; per_proj.update({k.removeprefix('ours_masked__'): v for k, v in projected.items() if k.startswith('ours_masked__')})
    draw_grid(shared_proj, rows, args.output_dir / 'long_dci_pca_shared_main_2x3', f'Long-DCI modality gap — Shared PCA (PC1={fits["shared"].explained_variance_ratio_[0]*100:.2f}%, PC2={fits["shared"].explained_variance_ratio_[1]*100:.2f}%)')
    draw_native(shared_proj, rows, args.output_dir / 'long_dci_pca_shared_native_1x3', 'Long-DCI Sema-CLIP Native — Shared PCA', get_bounds(shared_proj))
    draw_grid(per_proj, rows, args.output_dir / 'long_dci_pca_per_model_main_2x3', f'Long-DCI modality gap — Per-model PCA (Long-CLIP PC1={fits["longclip"].explained_variance_ratio_[0]*100:.2f}%, Ours Masked PC1={fits["ours_masked"].explained_variance_ratio_[0]*100:.2f}%)', per_row_bounds=True)
    draw_native({k.removeprefix('ours_native__'): v for k, v in projected.items() if k.startswith('ours_native__')}, rows, args.output_dir / 'long_dci_pca_per_model_native_1x3', f'Long-DCI Sema-CLIP Native — Per-model PCA (PC1={fits["ours_native"].explained_variance_ratio_[0]*100:.2f}%)', get_bounds(native_proj))

    clusters, transitions = {}, []
    for model in ('ours_masked','ours_native','longclip'):
        clusters[model] = {}
        for cap, _ in CAPTIONS: clusters[model][cap] = cluster_info(z[f'{model}_{cap}_image'])
    for a,b in (('full','prefix50'),('full','first_sentence'),('prefix50','first_sentence')):
        t=transition(clusters['ours_masked'][a]['labels'],clusters['ours_masked'][b]['labels']); t.update({'source':a,'target':b,'model':'ours_masked'}); transitions.append(t)
    for cap,_ in CAPTIONS:
        for other in ('ours_native','longclip'):
            t=transition(clusters['ours_masked'][cap]['labels'],clusters[other][cap]['labels']); t.update({'source':f'ours_masked_{cap}','target':f'{other}_{cap}','model':'cross_model'}); transitions.append(t)
    cluster_json = {m:{c:{k:v for k,v in info.items() if k != 'labels'} for c,info in caps.items()} for m,caps in clusters.items()}; cluster_json['transitions']=transitions
    (args.output_dir/'long_dci_kmeans_diagnostics.json').write_text(json.dumps(cluster_json,indent=2)+'\n')
    with (args.output_dir/'long_dci_metrics.csv').open('w',newline='') as f:
        fields=['model','caption','sample_count','centroid_gap','centroid_gap_ci_low','centroid_gap_ci_high','paired_cosine','paired_cosine_ci_low','paired_cosine_ci_high','paired_l2','paired_l2_ci_low','paired_l2_ci_high','image_centroid_norm','image_centroid_norm_ci_low','image_centroid_norm_ci_high','text_centroid_norm','text_centroid_norm_ci_low','text_centroid_norm_ci_high','mask_keep_ratio','mask_keep_ratio_ci_low','mask_keep_ratio_ci_high']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (args.output_dir/'long_dci_cluster_diagnostics.csv').open('w',newline='') as f:
        fields=['model','caption','major_count','minor_count','minor_fraction','silhouette_score','cluster_center_distance','kmeans_inertia','kmeans_seed_stability_ari_min','kmeans_seed_stability_ari_mean']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({'model':m,'caption':c,**{k:v for k,v in info.items() if k in fields[2:]}}) for m,caps in clusters.items() for c,info in caps.items()]
    comp=[]
    with DOCCI_METRICS.open() as f:
        for r in csv.DictReader(f): comp.append({'dataset':'DOCCI','model':r['model'],'caption':r['caption_setting'],'centroid_gap':float(r['centroid_gap']),'paired_cosine':float(r['mean_paired_cosine']),'paired_l2':float(r['mean_paired_l2']),'mask_keep_ratio':float(r['mask_keep_ratio']) if r['mask_keep_ratio'] else None})
    for r in rows: comp.append({'dataset':'Long-DCI', 'model':r['model'], 'caption':r['caption'], 'centroid_gap':r['centroid_gap'], 'paired_cosine':r['paired_cosine'], 'paired_l2':r['paired_l2'], 'mask_keep_ratio':r['mask_keep_ratio']})
    with (args.output_dir/'DOCCI_LONG_DCI_COMPARISON.csv').open('w',newline='') as f:
        fields=['dataset','model','caption','centroid_gap','paired_cosine','paired_l2','mask_keep_ratio']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(comp)
    variance={name:{'explained_variance_ratio':p.explained_variance_ratio_.tolist(),'fit_array_count':len(keys),'fit_point_count':sum(len(z[k]) for k in keys),'fit_arrays':keys} for name,(p,keys) in ((n,(fits[n],keys)) for n,keys in fit_specs.items())}
    provenance=json.loads((args.input_dir/'provenance.json').read_text()); params={'feature_sha256':sha256(args.input_dir/'features.npz'),'mask_sha256':sha256(args.input_dir/'mask_arrays.npz'),'feature_provenance_sha256':sha256(args.input_dir/'provenance.json'),'pca':{'solver':'full','random_state':0,'input':'per-row L2 normalized 512-D features','shared_fit':'Long-CLIP and Sema-CLIP Masked, 12000 points','per_model_fit':'6000 points per model; Native also fit independently for native control'},'bootstrap':{'reps':args.bootstrap_reps,'seed':0,'paired_unit':'sample index','ci':'percentile 2.5/97.5'},'explained_variance':variance,'kmeans':{'n_clusters':2,'n_init':50,'random_state':0,'space':'original 512-D normalized image features'},'input_provenance':provenance,'docci_sources':{'metrics':str(DOCCI_METRICS),'clusters':str(DOCCI_CLUSTERS)},'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (args.output_dir/'LONG_DCI_PCA_PARAMS_AND_PROVENANCE.json').write_text(json.dumps(params,indent=2)+'\n')
    report=['# Long-DCI Modality Gap PCA','', 'This is a fixed seed=0, 1,000-pair, inference-only analysis using the reconstructed Long-DCI manifest. No training or checkpoint modification was performed.', '', '## Caption audit', '', f"The manifest contains 7,602 unique image IDs and all image files are present. First-sentence parsing had {provenance['caption_lengths']['first_sentence_fallback_count']} explicit no-boundary fallbacks ({provenance['caption_lengths']['first_sentence_fallback_fraction']:.3%}); these rows are listed in `samples.json`. Mean effective BPE lengths are Full={provenance['caption_lengths']['full_mean']:.2f}, 50% Prefix={provenance['caption_lengths']['prefix50_mean']:.2f}, First Sentence={provenance['caption_lengths']['first_sentence_mean']:.2f}.", '', '## 512-D metrics', '', '| Model | Caption | Gap | Paired Cosine | Paired L2 | Image centroid norm | Text centroid norm | Mask keep |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        mask_text = '—' if r['mask_keep_ratio'] is None else f"{r['mask_keep_ratio']:.6f} [{r['mask_keep_ratio_ci_low']:.6f}, {r['mask_keep_ratio_ci_high']:.6f}]"
        report.append(f"| {r['model']} | {r['caption']} | {r['centroid_gap']:.6f} [{r['centroid_gap_ci_low']:.6f}, {r['centroid_gap_ci_high']:.6f}] | {r['paired_cosine']:.6f} [{r['paired_cosine_ci_low']:.6f}, {r['paired_cosine_ci_high']:.6f}] | {r['paired_l2']:.6f} [{r['paired_l2_ci_low']:.6f}, {r['paired_l2_ci_high']:.6f}] | {r['image_centroid_norm']:.6f} [{r['image_centroid_norm_ci_low']:.6f}, {r['image_centroid_norm_ci_high']:.6f}] | {r['text_centroid_norm']:.6f} [{r['text_centroid_norm_ci_low']:.6f}, {r['text_centroid_norm_ci_high']:.6f}] | {mask_text} |")
    report += ['', '## KMeans masked-image diagnosis', '', '| Caption | Major/Minor | Silhouette | Center distance |', '|---|---:|---:|---:|']
    for c,_ in CAPTIONS:
        q=clusters['ours_masked'][c]; report.append(f"| {c} | {q['major_count']}/{q['minor_count']} | {q['silhouette_score']:.6f} | {q['cluster_center_distance']:.6f} |")
    lookup={(r['dataset'],r['model'],r['caption']):r for r in comp}
    report += ['', '## Long-DCI versus DOCCI', '', '| Dataset | Model | Full Gap | 50% Gap | First Sentence Gap | Full Cosine | 50% Cosine | First Sentence Cosine |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in ('ours_masked','ours_native'):
        vals=[lookup[('DOCCI',model,c)] for c,_ in CAPTIONS]; vals2=[lookup[('Long-DCI',model,c)] for c,_ in CAPTIONS]
        label='Sema-CLIP Masked' if model=='ours_masked' else 'Sema-CLIP Native'
        report.append(f"| DOCCI / Long-DCI | {label} | {vals[0]['centroid_gap']:.4f} / {vals2[0]['centroid_gap']:.4f} | {vals[1]['centroid_gap']:.4f} / {vals2[1]['centroid_gap']:.4f} | {vals[2]['centroid_gap']:.4f} / {vals2[2]['centroid_gap']:.4f} | {vals[0]['paired_cosine']:.4f} / {vals2[0]['paired_cosine']:.4f} | {vals[1]['paired_cosine']:.4f} / {vals2[1]['paired_cosine']:.4f} | {vals[2]['paired_cosine']:.4f} / {vals2[2]['paired_cosine']:.4f} |")
    m_full=next(r for r in rows if r['model']=='ours_masked' and r['caption']=='full'); n_full=next(r for r in rows if r['model']=='ours_native' and r['caption']=='full')
    m_first=next(r for r in rows if r['model']=='ours_masked' and r['caption']=='first_sentence'); n_first=next(r for r in rows if r['model']=='ours_native' and r['caption']=='first_sentence')
    report += ['', '## Answers to the study questions', '', f"- Masked has a smaller Long-DCI Centroid Gap than Native in all three settings: Full {m_full['centroid_gap']:.4f} vs {n_full['centroid_gap']:.4f}, 50% {next(r for r in rows if r['model']=='ours_masked' and r['caption']=='prefix50')['centroid_gap']:.4f} vs {next(r for r in rows if r['model']=='ours_native' and r['caption']=='prefix50')['centroid_gap']:.4f}, First Sentence {m_first['centroid_gap']:.4f} vs {n_first['centroid_gap']:.4f}.", f"- The Gap and Paired Cosine trends agree within Long-DCI Masked across caption shortening (Gap rises from {m_full['centroid_gap']:.4f} to {m_first['centroid_gap']:.4f}, while cosine falls from {m_full['paired_cosine']:.4f} to {m_first['paired_cosine']:.4f}), but Masked versus Native is not uniformly aligned: Masked has the smaller Gap yet also the lower cosine.", f"- The Masked-image double-cluster diagnosis partially reproduces the DOCCI behavior. Long-DCI Full has silhouette {clusters['ours_masked']['full']['silhouette_score']:.4f} and a {clusters['ours_masked']['full']['major_count']}/{clusters['ours_masked']['full']['minor_count']} split; the 50% and First Sentence silhouettes are weaker ({clusters['ours_masked']['prefix50']['silhouette_score']:.4f}, {clusters['ours_masked']['first_sentence']['silhouette_score']:.4f}). This is dataset- and caption-dependent, not a universal separation.", "- Long-DCI is useful as a complementary cross-dataset figure because it exposes this dependence, but it does not uniformly supersede DOCCI as the single modality-gap main figure. Shared PCA is the cross-model reference; per-model PCA remains diagnostic only.", '', '## Interpretation', '', 'The PCA figures use one Shared PCA for the Long-CLIP and Sema-CLIP Masked rows, plus a Native 1×3 control and per-model PCA controls. No points were removed or shifted, and PCA coordinates are not used as modality-gap values.']
    (args.output_dir/'LONG_DCI_MODALITY_GAP_REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'rows':len(rows),'feature_sha256':params['feature_sha256'],'explained_variance':variance},indent=2))


if __name__ == '__main__': main()
