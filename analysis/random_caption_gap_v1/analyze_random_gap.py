#!/usr/bin/env python3
"""Aggregate fixed Caption Completeness views and create paper figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEEDS = (0, 1, 2, 3, 4)
MODELS = ("longclip", "ours_native", "ours_masked")
MODEL_LABELS = {"longclip": "Long-CLIP", "ours_native": "Sema-CLIP Native", "ours_masked": "Sema-CLIP Masked"}
BASE = ("full", "prefix50", "first_sentence")
PRIMARY = ("full", "prefix50", "random_distributed_random50", "random_distributed_random10", "first_sentence", "nofirst_matched")
COLORS = {"longclip": "#1f77b4", "ours_native": "#2ca02c", "ours_masked": "#d62728"}
BLUE, ORANGE = "#2166ac", "#d95f02"
DOCCI_MASK = Path("/opt/data/private/lklk/SAID/analysis/modality_gap_docci_b16_diagnostic_v1_runtime/mask_arrays.npz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-15)


def metric_values(im, tx, mask=None):
    im, tx = norm(im), norm(tx); ci, ct = im.mean(0), tx.mean(0); ri, rt = np.linalg.norm(ci), np.linalg.norm(ct); c = float(np.dot(ci, ct) / max(1e-15, ri * rt)); angular = 1.0 - c; gap = float(np.linalg.norm(ci - ct)); norm_term = float((ri - rt) ** 2); angle_term = float(2 * ri * rt * angular)
    return {"centroid_gap": gap, "angular_centroid_cosine": c, "angular_centroid_gap": angular, "image_centroid_norm": float(ri), "text_centroid_norm": float(rt), "paired_cosine": float(np.sum(im * tx, axis=1).mean()), "paired_l2": float(np.linalg.norm(im - tx, axis=1).mean()), "norm_difference_squared": norm_term, "angular_term_squared": angle_term, "norm_term_fraction": norm_term / max(gap * gap, 1e-15), "angular_term_fraction": angle_term / max(gap * gap, 1e-15), "mask_keep_ratio": float(mask.mean()) if mask is not None else np.nan}


def bootstrap_one(im, tx, mask, reps=2000, seed=0):
    im, tx = norm(im), norm(tx); n = len(im); rng = np.random.default_rng(seed); out = np.empty((reps, 9), dtype=np.float64)
    pair_cos = np.sum(im * tx, axis=1)
    pair_l2 = np.linalg.norm(im - tx, axis=1)
    mask_flat = np.asarray(mask, dtype=np.float64).reshape(n, -1).mean(axis=1) if mask is not None else None
    # Multinomial counts are exactly equivalent to paired index resampling,
    # while avoiding a large (reps x n x 512) temporary tensor.
    for start in range(0, reps, 500):
        stop = min(start + 500, reps); counts = rng.multinomial(n, np.full(n, 1.0 / n), size=stop-start).astype(np.float64)
        ci, ct = (counts @ im) / n, (counts @ tx) / n
        ri, rt = np.linalg.norm(ci, axis=1), np.linalg.norm(ct, axis=1); c = np.sum(ci * ct, axis=1) / np.maximum(1e-15, ri * rt); gap = np.linalg.norm(ci - ct, axis=1); nt = (ri - rt) ** 2; at = 2 * ri * rt * (1 - c)
        pc, pl = (counts @ pair_cos) / n, (counts @ pair_l2) / n
        mr = (counts @ mask_flat) / n if mask_flat is not None else np.full(stop-start,np.nan)
        out[start:stop] = np.column_stack((gap, 1-c, ri, rt, pc, pl, nt, at, mr))
    lo, hi = np.nanpercentile(out, [2.5,97.5], axis=0); names=("centroid_gap","angular_centroid_gap","image_centroid_norm","text_centroid_norm","paired_cosine","paired_l2","norm_difference_squared","angular_term_squared","mask_keep_ratio")
    return {f"{n}_ci_low": float(lo[i]) for i,n in enumerate(names)} | {f"{n}_ci_high": float(hi[i]) for i,n in enumerate(names)}


def cluster(x):
    x = norm(x); km=KMeans(n_clusters=2,n_init=50,random_state=0,algorithm="lloyd").fit(x); counts=np.bincount(km.labels_,minlength=2); minor=int(np.argmin(counts)); labels=(km.labels_==minor).astype(np.int8); return {"labels":labels,"major_count":int(counts.max()),"minor_count":int(counts.min()),"minor_fraction":float(counts.min()/len(x)),"silhouette":float(silhouette_score(x,labels,metric="euclidean")),"center_distance":float(np.linalg.norm(km.cluster_centers_[0]-km.cluster_centers_[1])),"inertia":float(km.inertia_)}


def load_source(dataset, root):
    if dataset == "docci":
        old_dir = root / "analysis/modality_gap_docci_b16_v2_first_sentence"; old_mask = DOCCI_MASK
    else:
        old_dir = root / "analysis/long_dci_modality_gap_pca_v1/extracted"; old_mask = old_dir / "mask_arrays.npz"
    return {"features": np.load(old_dir/"features.npz"), "samples": json.loads((old_dir/"samples.json").read_text()), "provenance": json.loads((old_dir/"provenance.json").read_text()), "mask": np.load(old_mask), "old_dir": old_dir}


def view_key(view, seed=None):
    if view in BASE: return view
    return f"{view}_seed{seed}"


def new_view_file(feature_dir, view, seed): return feature_dir / f"{view}_seed{seed}.npz"


def valid_indices(feature_dir, view, seed):
    """Return the exact sample indices retained by a reconstructed random view."""
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    return np.asarray(manifest["valid_indices"][f"{view}_seed{seed}"], dtype=np.int64)


def fetch(dataset, view_json, feature_dir, source, model, view, seed=None, subset=None):
    if view in BASE:
        im=source["features"][f"{model}_{view}_image"]; tx=source["features"][f"{model}_{view}_text"]
        mask=source["mask"][f"mask_{view}"] if model=="ours_masked" and f"mask_{view}" in source["mask"].files else None
        if subset is not None: im,tx=im[subset],tx[subset]; mask=mask[subset] if mask is not None else None
        return im,tx,mask
    path=new_view_file(feature_dir,view,seed); z=np.load(path)
    im,tx,mask=z[f"{model}_image"],z[f"{model}_text"],z["mask"] if model=="ours_masked" else None
    return im,tx,mask


def aggregate_rows(rows):
    fields = [k for k in rows[0] if k not in {"dataset","model","view","seed","strategy","valid_count","subset"} and isinstance(rows[0][k],(int,float))]
    out=[]
    for key in sorted({(r['dataset'],r['model'],r['view'],r['strategy']) for r in rows if r['seed'] != 'base'}):
        rr=[r for r in rows if (r['dataset'],r['model'],r['view'],r['strategy'])==key and r['seed']!='base']; q={"dataset":key[0],"model":key[1],"view":key[2],"strategy":key[3],"seed":"mean5","valid_count":int(round(np.mean([r['valid_count'] for r in rr]))),"subset":"seed_mean"}
        for f in fields: q[f]=float(np.mean([r[f] for r in rr])); q[f"{f}_seed_std"]=float(np.std([r[f] for r in rr],ddof=1))
        out.append(q)
    return out


def kde_band(ax, values, color, label):
    values=np.asarray(values); grid=np.linspace(values.min()-0.03,values.max()+0.03,220); curves=[]
    for v in values if values.ndim==2 else [values]:
        if np.ptp(v)<1e-8: curves.append(np.zeros_like(grid)); continue
        curves.append(gaussian_kde(v)(grid))
    curves=np.asarray(curves); ax.plot(grid,curves.mean(0),color=color,label=label); ax.fill_between(grid,curves.min(0),curves.max(0),color=color,alpha=.12)


def savefig(fig,path): fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white'); fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white'); plt.close(fig)


def plot_trend(summary, dataset, out):
    labels=["Full","Prefix50","Random50","Random10","First Sentence","NoFirst-Matched"]; keys=["full","prefix50","random_distributed_random50","random_distributed_random10","first_sentence","nofirst_matched"]; fig,ax=plt.subplots(figsize=(10,5.5)); x=np.arange(len(keys))
    for model in MODELS:
        vals=[]; err=[]
        for k in keys:
            r=next(r for r in summary if r['dataset']==dataset and r['model']==model and r['view']==k and r['seed']=='mean5') if k.startswith('random') or k=='nofirst_matched' else next(r for r in summary if r['dataset']==dataset and r['model']==model and r['view']==k and r['seed']=='base')
            vals.append(r['centroid_gap']); err.append(r.get('centroid_gap_seed_std',0))
        ax.errorbar(x,vals,yerr=err,marker='o',capsize=3,color=COLORS[model],label=MODEL_LABELS[model])
    ax.set_xticks(x,labels,rotation=20,ha='right'); ax.set_ylabel('512D Euclidean Centroid Gap'); ax.set_title(f'{dataset.upper()} — Caption completeness and modality gap'); ax.grid(alpha=.2); ax.legend(frameon=False); savefig(fig,out/f'FIGURE_A_{dataset.upper()}_CAPTION_GAP')


def plot_norm_angular(summary,dataset,out):
    labels=["Full","Prefix50","Random50","Random10","First Sentence","NoFirst-Matched"]; keys=["full","prefix50","random_distributed_random50","random_distributed_random10","first_sentence","nofirst_matched"]; fig,axes=plt.subplots(1,2,figsize=(13,4.8)); x=np.arange(len(keys))
    for model in MODELS:
        for ax,metric,ylabel in ((axes[0],'text_centroid_norm','Text centroid norm'),(axes[1],'angular_centroid_gap','Angular centroid gap (1 − cosine)')):
            vals=[]; err=[]
            for k in keys:
                r=next(r for r in summary if r['dataset']==dataset and r['model']==model and r['view']==k and r['seed']=='mean5') if k.startswith('random') or k=='nofirst_matched' else next(r for r in summary if r['dataset']==dataset and r['model']==model and r['view']==k and r['seed']=='base')
                vals.append(r[metric]); err.append(r.get(metric+'_seed_std',0))
            ax.errorbar(x,vals,yerr=err,marker='o',capsize=3,color=COLORS[model],label=MODEL_LABELS[model])
            ax.set_xticks(x,labels,rotation=25,ha='right'); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0].set_title('Text centroid concentration'); axes[1].set_title('Centroid direction gap'); axes[0].legend(frameon=False); fig.suptitle(f'{dataset.upper()} — Norm and angular decomposition'); fig.tight_layout(); savefig(fig,out/f'FIGURE_B_{dataset.upper()}_NORM_ANGULAR')


def plot_pca(dataset, out, source, feature_dir, views, second=False):
    selected=[("first_sentence",0),("nofirst_matched",0)] if second else [("full",None),("prefix50",None),("random_distributed_random50",0),("random_distributed_random10",0)]
    data={}; fit=[]
    for view,seed in selected:
        for model in ('longclip','ours_masked'):
            im,tx,_=fetch(dataset,views,feature_dir,source,model,view,seed)
            # First Sentence and NoFirst-Matched must be compared on the same
            # valid sample IDs.  The random feature file is already compacted;
            # apply its manifest indices to the historical First Sentence view.
            if second and view == 'first_sentence':
                idx = valid_indices(feature_dir, 'nofirst_matched', seed)
                im, tx = im[idx], tx[idx]
            data[(model,view,'image')]=norm(im); data[(model,view,'text')]=norm(tx); fit += [data[(model,view,'image')],data[(model,view,'text')]]
    pca=PCA(n_components=2,svd_solver='full',random_state=0).fit(np.concatenate(fit)); projected={k:pca.transform(v) for k,v in data.items()}; bounds=np.concatenate(list(projected.values())); lo,hi=bounds.min(0),bounds.max(0); span=np.maximum(hi-lo,1e-8); xlim=(lo[0]-.07*span[0],hi[0]+.07*span[0]); ylim=(lo[1]-.07*span[1],hi[1]+.07*span[1]); fig,axes=plt.subplots(2,len(selected),figsize=(5.1*len(selected),8.3),squeeze=False)
    labels={"full":"Full","prefix50":"Prefix50","random_distributed_random50":"Random50","random_distributed_random10":"Random10","first_sentence":"First Sentence","nofirst_matched":"NoFirst-Matched"}
    for ri,model in enumerate(('longclip','ours_masked')):
        for ci,(view,seed) in enumerate(selected):
            ax=axes[ri,ci]; im=projected[(model,view,'image')]; tx=projected[(model,view,'text')]; ax.scatter(im[:,0],im[:,1],s=3,alpha=.12,color=BLUE,rasterized=True); ax.scatter(tx[:,0],tx[:,1],s=3,alpha=.12,color=ORANGE,rasterized=True); ax.scatter([im[:,0].mean()],[im[:,1].mean()],marker='X',s=80,color='#08306b',edgecolors='white',label='Image centroid'); ax.scatter([tx[:,0].mean()],[tx[:,1].mean()],marker='X',s=80,color='#993404',edgecolors='white',label='Text centroid'); ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(labels[view]);ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.grid(alpha=.15)
        axes[ri,0].text(-.22,.5,'Long-CLIP' if model=='longclip' else 'Sema-CLIP Masked',transform=axes[ri,0].transAxes,rotation=90,va='center',ha='center')
    fig.suptitle(f'{dataset.upper()} — Shared PCA ({"First Sentence / NoFirst-Matched" if second else "Full / Prefix50 / Random50 / Random10"}; PC1={pca.explained_variance_ratio_[0]*100:.2f}%, PC2={pca.explained_variance_ratio_[1]*100:.2f}%)',y=1.02); fig.tight_layout(); savefig(fig,out/f'FIGURE_C_{dataset.upper()}_PCA_{"FIRST_NOFIRST" if second else "PRIMARY"}')


def plot_cosine(dataset,out,source,feature_dir,views):
    pairs=[('prefix50','random_distributed_random50','Prefix50 vs Random50'),('first_sentence','nofirst_matched','First Sentence vs NoFirst-Matched'),('random_distributed_random50','random_distributed_random10','Random50 vs Random10')]; fig,axes=plt.subplots(1,3,figsize=(15,4.5),squeeze=False); axes=axes[0]
    for ai,(a,b,title) in enumerate(pairs):
        ax=axes[ai]
        for model in MODELS:
            for view,style in ((a,'--'),(b,'-')):
                seedvals=[]
                # For First Sentence vs NoFirst-Matched use the five common
                # subsets, so the historical First Sentence features are
                # indexed by the same per-seed valid IDs as NoFirst.
                if a == 'first_sentence' and b == 'nofirst_matched':
                    seeds=[0,1,2,3,4]
                else:
                    seeds=[None] if view in BASE else [0,1,2,3,4]
                for seed in seeds:
                    im,tx,_=fetch(dataset,views,feature_dir,source,model,view,seed)
                    if view == 'first_sentence' and a == 'first_sentence' and b == 'nofirst_matched':
                        idx = valid_indices(feature_dir, 'nofirst_matched', seed)
                        im, tx = im[idx], tx[idx]
                    vals=np.sum(norm(im)*norm(tx),axis=1); seedvals.append(vals)
                grid=np.linspace(min(v.min() for v in seedvals)-.02,max(v.max() for v in seedvals)+.02,200); curves=np.array([gaussian_kde(v)(grid) for v in seedvals]); ax.plot(grid,curves.mean(0),linestyle=style,color=COLORS[model],label=f'{MODEL_LABELS[model]} {view}'); ax.fill_between(grid,curves.min(0),curves.max(0),color=COLORS[model],alpha=.05)
        ax.set_title(title);ax.set_xlabel('Paired cosine');ax.set_ylabel('Density');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7,frameon=False); fig.suptitle(f'{dataset.upper()} — Paired cosine distributions (seed range shown as bands)'); fig.tight_layout(); savefig(fig,out/f'FIGURE_D_{dataset.upper()}_PAIRED_COSINE')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/opt/data/private/lklk/SAID')); ap.add_argument('--work-root',type=Path,default=Path('analysis/random_caption_gap_v1')); ap.add_argument('--bootstrap-reps',type=int,default=2000); args=ap.parse_args(); out=args.work_root/'results'; out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; cluster_rows=[]; transitions=[]; summary=[]; source_audit={}
    for dataset in ('docci','long_dci'):
        source=load_source(dataset,args.root); views_path=args.work_root/f'{dataset}_views.json'; feature_dir=args.work_root/f'features_{dataset}'; view_payload=json.loads(views_path.read_text()); source_audit[dataset]={'old_feature_path':str(source['old_dir']/"features.npz"),'old_feature_sha256':sha256(source['old_dir']/"features.npz"),'old_samples_path':str(source['old_dir']/"samples.json"),'old_samples_sha256':sha256(source['old_dir']/"samples.json"),'random_views_sha256':sha256(views_path),'random_feature_manifest_sha256':sha256(feature_dir/'feature_manifest.json')}
        valid_nofirst={s: np.array(json.loads((feature_dir/'feature_manifest.json').read_text())['valid_indices'][f'nofirst_matched_seed{s}'],dtype=int) for s in SEEDS}
        first_all=source['features']['ours_masked_first_sentence_image']
        # Base views, plus First Sentence restricted to the NoFirst common subset for the paired test.
        for model in MODELS:
            for view in BASE:
                im,tx,m=fetch(dataset,views_path,feature_dir,source,model,view); row={'dataset':dataset,'model':model,'view':view,'strategy':'historical','seed':'base','valid_count':len(im),'subset':'all'}; row.update(metric_values(im,tx,m)); row.update(bootstrap_one(im,tx,m,args.bootstrap_reps,0)); all_rows.append(row)
        for seed in SEEDS:
            for model in MODELS:
                idx=valid_nofirst[seed]; im,tx,m=fetch(dataset,views_path,feature_dir,source,model,'first_sentence',None); im,tx=im[idx],tx[idx]; m=m[idx] if m is not None else None; row={'dataset':dataset,'model':model,'view':'first_sentence_common_nofirst','strategy':'historical_common_subset','seed':seed,'valid_count':len(idx),'subset':'nofirst_valid'}; row.update(metric_values(im,tx,m)); row.update(bootstrap_one(im,tx,m,args.bootstrap_reps,seed)); all_rows.append(row)
            for view in ('random_distributed_random50','random_distributed_random10','random_contiguous_random50','random_contiguous_random10','nofirst_matched'):
                for model in MODELS:
                    im,tx,m=fetch(dataset,views_path,feature_dir,source,model,view,seed); row={'dataset':dataset,'model':model,'view':view,'strategy':view.rsplit('_',1)[0],'seed':seed,'valid_count':len(im),'subset':'valid_view'}; row.update(metric_values(im,tx,m)); row.update(bootstrap_one(im,tx,m,args.bootstrap_reps,seed)); all_rows.append(row)
        # Masked-image KMeans diagnostics cover historical and reconstructed
        # views.  Base views use the full fixed 1000-pair sample; reconstructed
        # views use their explicit valid subset.
        for view in BASE:
            q=cluster(fetch(dataset,views_path,feature_dir,source,'ours_masked',view)[0])
            cluster_rows.append({'dataset':dataset,'view':view,'seed':'base',**{k:v for k,v in q.items() if k!='labels'}})
        # cluster transitions for masked views across seeds
        labels_by={}
        for view in BASE:
            labels_by[(view,'base')]=cluster(fetch(dataset,views_path,feature_dir,source,'ours_masked',view)[0])['labels']
        for seed in SEEDS:
            idx=valid_indices(feature_dir,'nofirst_matched',seed)
            im,tx,m=fetch(dataset,views_path,feature_dir,source,'ours_masked','first_sentence')
            q=cluster(im[idx]); labels_by[('first_sentence_common_nofirst',seed)]=q['labels']
            cluster_rows.append({'dataset':dataset,'view':'first_sentence_common_nofirst','seed':seed,**{k:v for k,v in q.items() if k!='labels'}})
        for view in ('random_distributed_random50','random_distributed_random10','random_contiguous_random50','random_contiguous_random10','nofirst_matched'):
            for seed in SEEDS:
                q=cluster(fetch(dataset,views_path,feature_dir,source,'ours_masked',view,seed)[0]); labels_by[(view,seed)]=q['labels']
                cluster_rows.append({'dataset':dataset,'view':view,'seed':seed,**{k:v for k,v in q.items() if k!='labels'}})
        for view in labels_by:
            for other in labels_by:
                if view < other and len(labels_by[view])==len(labels_by[other]): transitions.append({'dataset':dataset,'source':f'{view[0]}_seed{view[1]}','target':f'{other[0]}_seed{other[1]}','ari':float(adjusted_rand_score(labels_by[view],labels_by[other]))})
    # Aggregate random seed point estimates. Base rows remain single fixed views.
    seed_mean_rows=aggregate_rows(all_rows); summary=all_rows+seed_mean_rows
    fields=sorted({k for r in summary for k in r.keys()});
    with (out/'RANDOM_CAPTION_METRICS.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    with (out/'RANDOM_CAPTION_CLUSTER_DIAGNOSTICS.csv').open('w',newline='') as f: fields2=sorted({k for r in cluster_rows for k in r});w=csv.DictWriter(f,fieldnames=fields2);w.writeheader();w.writerows(cluster_rows)
    (out/'RANDOM_CAPTION_CLUSTER_TRANSITIONS.json').write_text(json.dumps(transitions,indent=2)+'\n')
    # Plot and summarize each dataset. The views manifest is passed by path for reproducibility.
    for dataset in ('docci','long_dci'):
        source=load_source(dataset,args.root); views=args.work_root/f'{dataset}_views.json'; feature_dir=args.work_root/f'features_{dataset}'; plot_trend(summary,dataset,out); plot_norm_angular(summary,dataset,out); plot_pca(dataset,out,source,feature_dir,views,False); plot_pca(dataset,out,source,feature_dir,views,True); plot_cosine(dataset,out,source,feature_dir,views)
    params={'datasets':source_audit,'seeds':list(SEEDS),'bootstrap':{'reps':args.bootstrap_reps,'seed_per_view':'seed for random view; bootstrap RNG seed=0 for base and seed for random view','unit':'paired sample index','ci':'percentile 2.5/97.5'},'views':{'primary_trend':['full','prefix50','random_distributed_random50','random_distributed_random10','first_sentence','nofirst_matched'],'random_strategies':['random_distributed','random_contiguous'],'nofirst_rule':'random contiguous crop after first sentence, matched to First Sentence BPE length, invalid rows excluded explicitly'},'plotting':{'pca':'Shared PCA per dataset and figure; per-model or cross-dataset coordinate comparison not used','scatter':'blue image/orange text, alpha=0.12','seed_visual':'seed0 for PCA layouts; seed range retained in metrics'},'model_checkpoints':{'longclip':'ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894','sema_clip':'83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07'}}
    (out/'RANDOM_CAPTION_PROVENANCE.json').write_text(json.dumps(params,indent=2)+'\n')
    # A compact report is generated after all rows are available.
    lines=['# Random Caption Completeness & First-Sentence Dependency','', 'This is an inference-only experiment using fixed DOCCI and reconstructed Long-DCI samples. Historical Full/Prefix50/First-Sentence features are reused; only random views were re-inferred.', '', '## Reproducibility', '', 'Random Distributed and Random Contiguous Crop use seeds 0–4. NoFirst-Matched is valid only when a post-first-sentence segment can be sampled at the First Sentence BPE length. Invalid rows are excluded, never zero-filled.', '', '## Main findings']
    for dataset in ('docci','long_dci'):
        get=lambda m,v,s: next(r for r in summary if r['dataset']==dataset and r['model']==m and r['view']==v and r['seed']==s)
        mask50=get('ours_masked','random_distributed_random50','mean5'); mask10=get('ours_masked','random_distributed_random10','mean5'); long50=get('longclip','random_distributed_random50','mean5'); long10=get('longclip','random_distributed_random10','mean5'); nofirst=get('ours_masked','nofirst_matched','mean5')
        lines += [f"### {dataset.upper()}", '', f"- Long-CLIP Random50 Gap={long50['centroid_gap']:.4f}, Random10 Gap={long10['centroid_gap']:.4f}; Paired Cosine={long50['paired_cosine']:.4f}/{long10['paired_cosine']:.4f}.", f"- Sema-CLIP Masked Random50 Gap={mask50['centroid_gap']:.4f}, Random10 Gap={mask10['centroid_gap']:.4f}; Paired Cosine={mask50['paired_cosine']:.4f}/{mask10['paired_cosine']:.4f}.", f"- Sema-CLIP Masked NoFirst-Matched Gap={nofirst['centroid_gap']:.4f}; this is evaluated only on {nofirst['valid_count']} valid samples.", '']
    lines += ['The CSV contains per-seed values, five-seed mean and standard deviation, paired bootstrap intervals, norm/angular decomposition and mask ratios. PCA figures use seed 0 for the random layouts while retaining all seeds in the quantitative records.', '']
    (out/'RANDOM_CAPTION_GAP_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(summary),'cluster_rows':len(cluster_rows),'output_dir':str(out)},indent=2))


if __name__ == '__main__': main()
