"""Semantic Grounding Audit page in the existing Streamlit dashboard."""
import os

import numpy as np
import pandas as pd
import streamlit as st

from data import ArtifactError, heatmap_rgb
from grounding_data import (LABELS, annotated_image, load_bundle, load_crop,
                            load_map, select_phrases, switching_metrics)
from eval.salu.grounding_metrics import union_iou


@st.cache_data(show_spinner=False)
def _bundle(root):
    return load_bundle(root)


@st.cache_data(show_spinner=False)
def _map(root, variant, key):
    return load_map(root, variant, key)


def main():
    st.title('Semantic Grounding Audit')
    st.caption('Phrase queries with annotated entities. Green: target GT; cyan: other target; red cross: peak patch center.')
    root = st.sidebar.text_input('grounding artifact root', os.environ.get('SEMANTIC_GROUNDING_ROOT', 'outputs/semantic_grounding'))
    try:
        manifest, summary, phrases = _bundle(root)
    except ArtifactError as exc:
        st.error(str(exc))
        st.info('Select a completed grounding audit export. See docs/phase23_grounding_audit.md.')
        return
    st.caption('%s / %s | %d images | %d phrase mentions | %s' %
               (manifest['dataset'], manifest['split'], manifest['num_images'], manifest['num_phrases'], manifest['precision']))
    rows = []
    for variant in manifest['variants']:
        s = summary[variant]
        rows.append({'model': LABELS[variant], **{k: s[k]['mean'] for k in
                    ['pointing_correct', 'gt_mass', 'gt_area_fraction', 'mass_gain',
                     'target_gt_distractor', 'localization_margin', 'switch_margin']}})
    st.subheader('Overall grounding')
    st.dataframe(pd.DataFrame(rows).set_index('model'))
    model = st.sidebar.selectbox('ranking model', manifest['variants'], index=2, format_func=LABELS.get)
    view = st.sidebar.selectbox('case filter', ['All', 'Correct pointing', 'Wrong pointing',
                                               'Negative localization margin', 'Highest failures', 'Highest successes'])
    sort = st.sidebar.selectbox('sort metric', ['gt_mass', 'mass_gain', 'localization_margin'], index=1)
    descending = st.sidebar.checkbox('descending', value=False)
    selected = select_phrases(phrases, model, view, sort, descending)
    if not selected:
        st.info('No phrases match this filter.')
        return
    image_ids = list(dict.fromkeys(p['image_id'] for p in selected))
    image_id = st.sidebar.selectbox('grounding image', image_ids)
    candidates = [p['id'] for p in selected if p['image_id'] == image_id]
    key = st.sidebar.selectbox('target phrase', candidates, format_func=lambda key: phrases[key]['phrase']+' ['+key+']')
    p = phrases[key]
    image = load_crop(root, image_id)
    st.subheader('Image and annotated phrase')
    left, right = st.columns([1, 2])
    left.image(annotated_image(image, p['boxes']), width='stretch')
    right.markdown('**Query:** '+p['phrase'])
    right.write('Sentence (context only, never the model query): '+p['sentence'])
    right.write('Official categories: '+', '.join(p['categories']))
    right.write('Visible GT area retained after crop: %.1f%%' % (100*p.get('visible_area_retention', 1)))
    st.subheader('Four-model comparison (shared colour scale)')
    shared = st.checkbox('shared grounding colour scale', value=True)
    maps = {v: _map(root, v, key) for v in manifest['variants']}
    vmax = max(a.max() for a in maps.values()) if shared else None
    for col, variant in zip(st.columns(4), manifest['variants']):
        stats = p['metrics'][variant]
        col.image(annotated_image(image, p['boxes'], maps[variant], vmax, stats['peak_xy']),
                  caption=LABELS[variant], width='stretch')
        col.write('Pointing correct: '+('yes' if stats['pointing_correct'] else 'no'))
        for label in ['gt_mass', 'gt_area_fraction', 'mass_gain', 'localization_margin']:
            value = stats[label]
            col.caption('%s: %s' % (label, 'N/A (no distractor)' if value is None else '%.4f' % value))
        col.caption('peak row/col: %d/%d (%s)' % (stats['peak_row'], stats['peak_col'], stats['peak_location']))
    diff = np.abs(maps['phase22_router']-maps['phase22_direct'])
    st.image(heatmap_rgb(diff), caption='|Phase 2.2 Router - same-backbone direct CLIP| (difference scale)', width=224)
    st.subheader('Same-image phrase switching')
    groups = manifest['switching_groups']
    if not groups:
        st.info('No spatially separated entity pairs in this export.')
    else:
        gid = st.selectbox('switching image', [g['image_id'] for g in groups],
                           index=next((i for i, g in enumerate(groups) if g['image_id'] == image_id), 0))
        group = next(g for g in groups if g['image_id'] == gid)
        fmt = lambda key: phrases[key]['phrase']+' ['+key+']'
        aid = st.selectbox('Phrase A', group['phrase_ids'], format_func=fmt)
        bid = st.selectbox('Phrase B', [k for k in group['phrase_ids'] if k != aid], format_func=fmt)
        variant = st.selectbox('switching model', manifest['variants'], index=2, format_func=LABELS.get)
        pa, pb = phrases[aid], phrases[bid]
        aa, ab = _map(root, variant, aid), _map(root, variant, bid)
        switch_image = load_crop(root, gid)
        scale = max(aa.max(), ab.max())
        cols = st.columns(2)
        cols[0].image(annotated_image(switch_image, pa['boxes'], aa, scale, pa['metrics'][variant]['peak_xy'], pb['boxes']),
                      caption='A: '+pa['phrase'], width='stretch')
        cols[1].image(annotated_image(switch_image, pb['boxes'], ab, scale, pb['metrics'][variant]['peak_xy'], pa['boxes']),
                      caption='B: '+pb['phrase'], width='stretch')
        metrics = switching_metrics(root, variant, pa, pb)
        st.dataframe(pd.DataFrame([{k: metrics[k] for k in ['M_aa', 'M_ab', 'M_bb', 'M_ba', 'switch_margin', 'jsd']}]))
        st.write('Both queries prefer their own target: '+str(metrics['both_prefer_target']))
        st.caption('GT union IoU = %.4f. JSD measures change; switch margin measures whether change favours the named target.' % union_iou(pa['boxes'], pb['boxes']))
    with st.expander('Category breakdown and peak locations'):
        s = summary[model]
        st.json(s['peak_location'])
        st.caption('Background means outside all annotated boxes; unannotated objects may be present.')
        st.dataframe(pd.DataFrame([{'category': cat, 'phrases': stats['phrases'],
                                   'pointing': stats['pointing_correct']['mean'],
                                   'mass_gain': stats['mass_gain']['mean']}
                                  for cat, stats in s.get('categories', {}).items()]))
