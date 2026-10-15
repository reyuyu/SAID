"""Replot existing DOCCI features. No training, extraction or sample selection."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / 'input'
STEM = 'modality_gap_main_2x3_fullcaption'
SEED = 20261013
MODELS = ['longclip', 'ours_native', 'ours_masked']
LABELS = ['Long-CLIP', 'Sema-CLIP (Global)', 'Sema-CLIP (Masked)']
BLUE, ORANGE = '#2F5D8A', '#D9822B'
DARK_BLUE, DARK_ORANGE = '#173A5E', '#9A4E00'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    samples = json.loads((INPUT/'samples.json').read_text(encoding='utf-8'))
    lengths = json.loads((INPUT/'caption_lengths.json').read_text(encoding='utf-8'))
    provenance = json.loads((INPUT/'provenance.json').read_text(encoding='utf-8'))
    assert len(samples) == len(lengths) == provenance['sample_count'] == 1000
    ids = [s['image_id'] for s in samples]
    assert len(set(ids)) == 1000
    assert ids == [s['image_id'] for s in lengths]
    raw = pd.read_csv(INPUT/'modality_gap_metrics.csv')
    features = np.load(INPUT/'features.npz', allow_pickle=False)
    assert np.array_equal(features['ours_native_full_text'], features['ours_masked_full_text'])
    data, rows, pca_meta = {}, [], {}
    for model in MODELS:
        image = features[f'{model}_full_image'].astype('float64')
        text = features[f'{model}_full_text'].astype('float64')
        assert image.shape == text.shape == (1000, 512)
        assert np.isfinite(image).all() and np.isfinite(text).all()
        assert np.allclose(np.linalg.norm(image, axis=1), 1, atol=1e-6)
        assert np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-6)
        ci, ct = image.mean(0), text.mean(0)
        gap = float(np.linalg.norm(ci-ct))
        cosine = float(np.mean(np.sum(image*text, axis=1)))
        l2 = float(np.mean(np.linalg.norm(image-text, axis=1)))
        ref = raw[(raw.model == model)&(raw.caption_setting == 'full')].iloc[0]
        assert abs(gap-ref.centroid_gap) < 1e-7
        assert abs(cosine-ref.mean_paired_cosine) < 1e-7
        assert abs(l2-ref.mean_paired_l2) < 1e-7
        axis = (ci-ct)/gap
        ip, tp = image@axis, text@axis
        assert abs((ip.mean()-tp.mean())-gap) < 1e-12
        pca = PCA(n_components=2, svd_solver='full', random_state=SEED)
        xy = pca.fit_transform(np.vstack((image,text)))
        data[model] = (xy[:1000], xy[1000:], ip, tp)
        pca_meta[model] = {'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
                           'projected_centroid_gap':float(np.linalg.norm(xy[:1000].mean(0)-xy[1000:].mean(0)))}
        rows.append(dict(model=model, display_label=LABELS[MODELS.index(model)], sample_count=1000,
                         centroid_gap_512d=gap, paired_cosine_512d=cosine, paired_l2_512d=l2,
                         image_centroid_norm_512d=float(np.linalg.norm(ci)),text_centroid_norm_512d=float(np.linalg.norm(ct)),
                         mask_keep_ratio=ref.mask_keep_ratio, image_gap_axis_centroid=float(ip.mean()),
                         text_gap_axis_centroid=float(tp.mean())))
    # All observations determine the limits; no percentile clipping or point filtering.
    all_xy = np.vstack([a for d in data.values() for a in d[:2]])
    low, high = all_xy.min(0), all_xy.max(0)
    pad = .07*(high-low)
    all_proj = np.concatenate([a for d in data.values() for a in d[2:]])
    xmin,xmax = all_proj.min(), all_proj.max()
    xpad=.07*(xmax-xmin)
    xmin-=xpad; xmax+=xpad
    bins=np.linspace(xmin,xmax,51)
    grid=np.linspace(xmin,xmax,1000)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                         'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
    fig, axs=plt.subplots(2,3,figsize=(11.7,6.5))
    fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.14,wspace=.29,hspace=.35)
    for col, model in enumerate(MODELS):
        xy_i,xy_t,ip,tp=data[model]; row=rows[col]
        ax=axs[0,col]
        ax.scatter(xy_i[:,0],xy_i[:,1],s=12,alpha=.38,c=BLUE,marker='o',edgecolors='none',rasterized=True,zorder=2)
        ax.scatter(xy_t[:,0],xy_t[:,1],s=12,alpha=.38,c=ORANGE,marker='s',edgecolors='none',rasterized=True,zorder=2)
        mui,mut=xy_i.mean(0),xy_t.mean(0)
        ax.plot([mui[0],mut[0]],[mui[1],mut[1]],color='#66717E',lw=1.25,zorder=3)
        ax.scatter(*mui,s=88,marker='D',color=DARK_BLUE,edgecolor='white',linewidth=1,zorder=4)
        ax.scatter(*mut,s=108,marker='P',color=DARK_ORANGE,edgecolor='white',linewidth=1,zorder=4)
        ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),xlabel='PC1',ylabel='PC2')
        ax.set_aspect('equal',adjustable='box')
        ax.set_title(LABELS[col],pad=11,fontweight='semibold')
        ax=axs[1,col]
        ax.hist(ip,bins=bins,density=True,color=BLUE,alpha=.34,edgecolor='none',zorder=2)
        ax.hist(tp,bins=bins,density=True,color=ORANGE,alpha=.30,edgecolor='none',zorder=2)
        ik=gaussian_kde(ip,bw_method='scott'); tk=gaussian_kde(tp,bw_method='scott')
        ax.plot(grid,ik(grid),color=BLUE,lw=1.85,zorder=3)
        ax.plot(grid,tk(grid),color=ORANGE,lw=1.85,zorder=3)
        ax.axvline(ip.mean(),color=DARK_BLUE,ls=(0,(4,3)),lw=1.25,zorder=4)
        ax.axvline(tp.mean(),color=DARK_ORANGE,ls=(0,(4,3)),lw=1.25,zorder=4)
        top=max(ax.get_ylim()[1],float(ik(grid).max()),float(tk(grid).max()))*1.22
        ax.set_ylim(0,top)
        ax.annotate('',xy=(tp.mean(),top*.85),xytext=(ip.mean(),top*.85),arrowprops=dict(arrowstyle='<->',color='#4B5563',lw=1.1))
        ax.text((ip.mean()+tp.mean())/2,top*.90,f"Gap = {row['centroid_gap_512d']:.3f}",ha='center',fontsize=10,color='#252B33')
        ax.set(xlim=(xmin,xmax),xlabel='Gap-axis projection',ylabel='Density')
    for ax in axs.flat:
        ax.grid(color='#E5E7EB',lw=.55,alpha=.9)
        ax.set_axisbelow(True)
        ax.spines['bottom'].set_color('#9CA3AF'); ax.spines['left'].set_color('#9CA3AF')
        ax.tick_params(length=3,color='#9CA3AF')
    handles=[Line2D([],[],marker='o',ls='none',color=BLUE,markersize=6,label='Image features'),
             Line2D([],[],marker='s',ls='none',color=ORANGE,markersize=6,label='Text features'),
             Line2D([],[],marker='D',ls='none',color=DARK_BLUE,markersize=6,label='Image centroid'),
             Line2D([],[],marker='P',ls='none',color=DARK_ORANGE,markersize=7,label='Text centroid')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=10,
               columnspacing=1.8,handletextpad=.4)
    fig.savefig(OUT/f'{STEM}.png',dpi=400,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{STEM}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/f'{STEM}_metrics.csv',index=False)
    params=dict(dataset='DOCCI test',caption='Full',sample_count=1000,feature_dimension=512,
        source_feature_file=str(INPUT/'features.npz'),source_sha256=sha(INPUT/'features.npz'),
        source_commit='124e89574d6ec586f5eaf4f5ff63f9679b3c89ff',sample_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        sample_order_verification='One shared samples.json and source provenance; unchanged row order; caption_lengths IDs match; Native/Masked text arrays identical',
        seed=SEED,pca=dict(fit='Full image+text concatenation, independently per model',components=['PC1','PC2'],svd_solver='full',details=pca_meta),
        gap_axis='(mean(image)-mean(text))/centroid_gap; raw dot-product projection, no translation',
        histogram=dict(bins=50,shared_bin_edges=bins.tolist(),density=True,alpha_image=.34,alpha_text=.30),
        kde=dict(bandwidth_rule='Scott',evaluation_points=1000),
        limits=dict(pca='shared global min/max + 7% padding, equal aspect, all points included',gap_axis='shared global min/max + 7% padding',density='per-panel ymax to show distribution shape; compare separation along common x scale'),
        scatter=dict(size=12,alpha=.38),colors=dict(image=BLUE,text=ORANGE,image_centroid=DARK_BLUE,text_centroid=DARK_ORANGE),
        figure_inches=[11.7,6.5],png_dpi=400,annotation_space='original 512D; Full gap-axis centroid separation equals 512D gap analytically')
    (OUT/f'{STEM}_params.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    notes=f'''# Full Caption 主文图说明

- 数据集：DOCCI test，固定的 1000 个样本，仅 Full Caption。
- 输入：`../input/features.npz`、`../input/modality_gap_metrics.csv`；来自 GitHub 提交 `124e89574d6ec586f5eaf4f5ff63f9679b3c89ff`，没有训练或重新提特征。
- 展示映射：`longclip_full_*` → Long-CLIP；`ours_native_full_*` → Sema-CLIP (Global)；`ours_masked_full_*` → Sema-CLIP (Masked)。后两个仅修改图中名称，特征内容保持不变。
- 样本顺序：复用单一导出的 shared samples manifest；1000 个唯一 image IDs 与 caption_lengths.json 的顺序一致，Native/Masked 的 Full text 数组逐元素一致。NPZ 没有逐模型独立的 ID 数组，三组的对应关系依据共享导出 provenance；未重排或筛选。输入 hash 和 ID 序列 hash 见参数文件。
- 第一行：每个模型单独用 Full image+text 共 2000 个向量拟合 PCA，PC1–PC2，full SVD，seed={SEED}。三列坐标范围一致、纵横等比例。所有点可见，s=12，alpha=0.38；线段仅连接投影 centroid。
- 第二行：每模型轴 `u=(μ_image−μ_text)/||μ_image−μ_text||`，投影为 `x·u`；没有平移分布。共同横坐标范围与 50 个等宽 bins，density=True；KDE 使用 Scott 带宽、1000 点网格。虚线为两个 projected centroid，双向箭头长度为 gap。纵轴密度分别自动缩放，比较间距应看共同的横轴，不应比较分布峰值高度。
- 标注：Gap 只显示在第二行；Gap 在原始归一化 512D 空间计算，并与源 CSV 校验，误差 <1e-7。由于 Full gap-axis 正是对应 centroid 差方向，该行投影 centroid gap 与 512D gap 数学上相等。PCA 连线长度不能作为完整 512D gap。Paired cosine、Paired L2、centroid norms 和 mask keep ratio 保存在配套 metrics CSV；原有 bootstrap CI 在 `../tables/metrics_with_bootstrap_ci.csv` 保留。
- 风格：白底浅灰网格，Image {BLUE}，Text {ORANGE}，centroids {DARK_BLUE}/{DARK_ORANGE}；PNG 400 dpi，PDF 嵌入字体和矢量文字/KDE，散点栅格化。

| 表示 | Gap (512D) |
|---|---:|
'''
    for label,row in zip(LABELS,rows):
        notes+=f"| {label} | {row['centroid_gap_512d']:.6f} |\n"
    notes+='''
该图适合作为双栏主文图。第二行是更直接的 gap 证据，第一行提供分布形态；PCA 为逐模型独立拟合，不能用跨列点的相对位置解释模型间的语义对应关系。Masked 减小 global centroid gap，但 Native 的 paired cosine 更高，应避免把 global gap 改善写成逐样本匹配也全面改善。
'''
    (OUT/f'{STEM}_notes.md').write_text(notes,encoding='utf-8')
    print(pd.DataFrame(rows)[['display_label','centroid_gap_512d','paired_cosine_512d']].to_string(index=False))
    print('Saved PNG, PDF, notes, metrics, and parameters:',OUT)

if __name__=='__main__':
    main()
