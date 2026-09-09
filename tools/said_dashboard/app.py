"""Said attention visualization dashboard (Streamlit, read-only artifacts).

The dashboard NEVER loads a checkpoint or runs the model: it only reads the
precomputed artifacts produced by ``eval/salu/export_dashboard_artifacts.py``
(JSON / NPY / PNG under ``outputs/salu_dashboard/``), so switching checkpoints or
samples is instant.

Run (server, loopback only)::

    cd /root/SAID
    streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501

Then tunnel from your laptop::

    ssh -L 8501:127.0.0.1:8501 <server>

and open http://127.0.0.1:8501
"""
import os
import sys

import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(DASHBOARD_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from data import (  # noqa: E402
    ArtifactError,
    VARIANTS,
    attention_stats,
    caption_for,
    checkpoint_metrics,
    checkpoints,
    difference_map,
    heatmap_rgb,
    js_divergence,
    load_attention,
    load_image,
    load_manifest,
    load_metrics,
    load_sample_metrics,
    metric_series,
    overlay_rgb,
    sample_meta,
    samples,
)

st.set_page_config(page_title='Said attention dashboard', layout='wide')


@st.cache_data(show_spinner=False)
def _manifest(root):
    return load_manifest(root)


@st.cache_data(show_spinner=False)
def _metrics(root):
    return load_metrics(root)


@st.cache_data(show_spinner=False)
def _attn(root, tag, index, variant):
    return load_attention(root, tag, index, variant)


@st.cache_data(show_spinner=False)
def _image(root, tag, index):
    return np.asarray(load_image(root, tag, index))


@st.cache_data(show_spinner=False)
def _sample_metrics(root, tag):
    return load_sample_metrics(root, tag)


def _render(attn, image, scale_max, mode):
    if mode == 'heatmap':
        return heatmap_rgb(attn, scale_max=scale_max)
    return overlay_rgb(Image.fromarray(image), attn, scale_max=scale_max)


def main():
    page = st.sidebar.radio('Page', ['Said attention', 'Semantic Grounding Audit'])
    if page == 'Semantic Grounding Audit':
        from grounding_page import main as grounding_main
        grounding_main()
        return
    st.title('Said attention dashboard')
    st.caption('Read-only view of precomputed artifacts. The dashboard never loads a checkpoint or runs the model.')

    root = st.sidebar.text_input('artifact root', os.environ.get('SAID_DASHBOARD_ROOT', 'outputs/salu_dashboard'))
    try:
        manifest = _manifest(root)
        metrics = _metrics(root)
    except ArtifactError as exc:
        st.error(str(exc))
        st.info('Run: python eval/salu/export_dashboard_artifacts.py --checkpoints ... --output_dir %s' % root)
        return

    tags = checkpoints(manifest)
    sample_entries = samples(manifest)
    indices = [entry['index'] for entry in sample_entries]

    st.sidebar.header('Selection')
    tag = st.sidebar.selectbox('checkpoint', tags, index=len(tags) - 1)
    index = st.sidebar.selectbox(
        'sample', indices,
        format_func=lambda i: 'sample %02d  (%s)' % (i, sample_meta(manifest, i).get('image_id', '?')),
    )
    variant_a = st.sidebar.selectbox('caption A', VARIANTS, index=0)
    variant_b = st.sidebar.selectbox('caption B', VARIANTS, index=1)
    mode = st.sidebar.radio('display mode', ('heatmap', 'overlay'), index=1)
    shared_scale = st.sidebar.checkbox('shared colour scale (recommended)', value=True)
    st.sidebar.header('Evolution')
    evolution_variant = st.sidebar.selectbox('evolution caption', VARIANTS, index=0)
    evolution_tags = st.sidebar.multiselect('evolution checkpoints', tags, default=tags)

    meta = sample_meta(manifest, index)
    image = _image(root, tag, index)
    attn_a = _attn(root, tag, index, variant_a)
    attn_b = _attn(root, tag, index, variant_b)

    st.subheader('Image and captions')
    col_img, col_caps = st.columns([1, 2])
    with col_img:
        st.image(image, caption='%s (%s)' % (meta.get('image_id', ''), tag), width=320)
    with col_caps:
        for variant in VARIANTS:
            st.markdown('**%s** — %s' % (variant, caption_for(manifest, index, variant)))

    st.subheader('Same image / different caption')
    scale = max(attn_a.max(), attn_b.max()) if shared_scale else None
    cols = st.columns(4)
    cols[0].image(image, caption='original', width=320)
    cols[1].image(_render(attn_a, image, scale, mode), caption='A: %s' % variant_a, width=320)
    cols[2].image(_render(attn_b, image, scale, mode), caption='B: %s' % variant_b, width=320)
    diff = difference_map(attn_a, attn_b)
    cols[3].image(heatmap_rgb(diff, scale_max=diff.max() if diff.max() > 0 else 1.0),
                  caption='|A - B|', width=320)

    entry = checkpoint_metrics(metrics, tag)
    try:
        per_sample = _sample_metrics(root, tag).get(str(index), {})
    except ArtifactError:
        per_sample = {}
    zs_a = np.asarray(per_sample.get('zs', {}).get(variant_a, []), dtype=np.float64)
    zs_b = np.asarray(per_sample.get('zs', {}).get(variant_b, []), dtype=np.float64)
    zs_cos = float(zs_a @ zs_b) if zs_a.size and zs_b.size else float('nan')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('JSD(A, B)', '%.6f' % js_divergence(attn_a, attn_b))
    m2.metric('mean |A - B|', '%.6f' % float(diff.mean()))
    m3.metric('z_s cosine(A, B)', '%.5f' % zs_cos)
    m4.metric('caption-conditioning ratio', '%.2f' % entry['caption_conditioning_ratio'])

    st.subheader('Attention evolution (shared colour scale)')
    evo_tags = evolution_tags or tags
    evo_maps = {t: _attn(root, t, index, evolution_variant) for t in evo_tags}
    evo_scale = max(m.max() for m in evo_maps.values())
    evo_cols = st.columns(len(evo_tags))
    for col, t in zip(evo_cols, evo_tags):
        col.image(_render(evo_maps[t], _image(root, t, index), evo_scale, mode),
                  caption='%s (%s)' % (t, evolution_variant), width=280)
        stats = attention_stats(evo_maps[t])
        col.caption('entropy %.3f | eff patches %.1f | max %.4f' % (
            stats['entropy'], stats['effective_patch_count'], stats['max']))

    st.subheader('Metric evolution')
    series = metric_series(metrics)
    frame = pd.DataFrame(series).set_index('checkpoint')
    left, right = st.columns(2)
    left.line_chart(frame[['attention_entropy', 'effective_patch_count']])
    left.caption('attention sharpening (entropy down, effective patch count down)')
    right.line_chart(frame[['caption_shuffle_jsd', 'precision_noise_jsd']])
    right.caption('caption-induced JSD vs bf16/fp32 precision-noise JSD')
    left2, right2 = st.columns(2)
    left2.line_chart(frame[['route_top1_acc', 'evidence_top1_acc']])
    left2.caption('route / evidence identification top-1 accuracy (64-way)')
    right2.line_chart(frame[['conditioning_ratio', 'zs_cosine']])
    right2.caption('caption-conditioning ratio and own-vs-shuffled z_s cosine')

    st.subheader('Precision noise panel')
    n1, n2, n3 = st.columns(3)
    n1.metric('caption change JSD', '%.6f' % entry['caption_shuffle']['mean_js_divergence'])
    n2.metric('precision noise JSD', '%.6f' % entry['precision_noise']['mean_js_divergence'])
    n3.metric('ratio', '%.2f' % entry['caption_conditioning_ratio'])
    st.caption('A ratio above 1 (ideally above 2) means the attention change caused by swapping the caption '
               'is larger than the bf16/fp32 numerical noise floor.')

    st.subheader('Route identification')
    r1, r2, r3, r4 = st.columns(4)
    r1.metric('route top-1 acc', '%.4f' % entry['route_identification']['top1_acc'])
    r2.metric('route margin', '%.4f' % entry['route_identification']['margin'])
    r3.metric('evidence top-1 acc', '%.4f' % entry['evidence_identification']['top1_acc'])
    r4.metric('evidence margin', '%.4f' % entry['evidence_identification']['margin'])
    st.caption('chance = 1 / %d = %.4f' % (manifest.get('num_samples', 0),
                                           1.0 / max(1, manifest.get('num_samples', 1))))

    with st.expander('per-checkpoint metrics (raw json)'):
        st.json(entry)


if __name__ == '__main__':
    try:
        main()
    except ArtifactError as exc:
        st.error(str(exc))
        st.info('The selected sample or checkpoint is incomplete. Select another one or re-export its artifacts.')
