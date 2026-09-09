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
    st.title('历史：语义定位审计')
    st.caption('历史机制分析：带标注实体的短语查询。绿色为目标标注，青色为其他目标，红十字为峰值 patch 中心。')
    root = st.sidebar.text_input('历史定位审计目录', os.environ.get('SEMANTIC_GROUNDING_ROOT', 'outputs/semantic_grounding'))
    try:
        manifest, summary, phrases = _bundle(root)
    except ArtifactError as exc:
        st.error('历史诊断文件不可用：' + str(exc))
        st.info('请选择已完成的历史定位审计导出，说明见 docs/phase23_grounding_audit.md。')
        return
    st.caption('%s / %s | %d images | %d phrase mentions | %s' %
               (manifest['dataset'], manifest['split'], manifest['num_images'], manifest['num_phrases'], manifest['precision']))
    rows = []
    for variant in manifest['variants']:
        s = summary[variant]
        rows.append({'model': LABELS[variant], **{k: s[k]['mean'] for k in
                    ['pointing_correct', 'gt_mass', 'gt_area_fraction', 'mass_gain',
                     'target_gt_distractor', 'localization_margin', 'switch_margin']}})
    st.subheader('历史定位指标汇总')
    st.dataframe(pd.DataFrame(rows).set_index('model'))
    model = st.sidebar.selectbox('排序模型', manifest['variants'], index=2, format_func=LABELS.get)
    view = st.sidebar.selectbox('样本筛选', ['All', 'Correct pointing', 'Wrong pointing',
                                               'Negative localization margin', 'Highest failures', 'Highest successes'])
    sort = st.sidebar.selectbox('排序指标', ['gt_mass', 'mass_gain', 'localization_margin'], index=1)
    descending = st.sidebar.checkbox('降序', value=False)
    selected = select_phrases(phrases, model, view, sort, descending)
    if not selected:
        st.info('没有符合筛选条件的短语。')
        return
    image_ids = list(dict.fromkeys(p['image_id'] for p in selected))
    image_id = st.sidebar.selectbox('审计图像', image_ids)
    candidates = [p['id'] for p in selected if p['image_id'] == image_id]
    key = st.sidebar.selectbox('目标短语', candidates, format_func=lambda key: phrases[key]['phrase']+' ['+key+']')
    p = phrases[key]
    image = load_crop(root, image_id)
    st.subheader('图像与标注短语')
    left, right = st.columns([1, 2])
    left.image(annotated_image(image, p['boxes']), width='stretch')
    right.markdown('**查询短语：** '+p['phrase'])
    right.write('原句（仅作上下文，不作为模型查询）：'+p['sentence'])
    right.write('官方类别：'+', '.join(p['categories']))
    right.write('Visible GT area retained after crop: %.1f%%' % (100*p.get('visible_area_retention', 1)))
    st.subheader('四模型对比（共享颜色范围）')
    shared = st.checkbox('共享颜色范围', value=True)
    maps = {v: _map(root, v, key) for v in manifest['variants']}
    vmax = max(a.max() for a in maps.values()) if shared else None
    for col, variant in zip(st.columns(4), manifest['variants']):
        stats = p['metrics'][variant]
        col.image(annotated_image(image, p['boxes'], maps[variant], vmax, stats['peak_xy']),
                  caption=LABELS[variant], width='stretch')
        col.write('峰值指向正确：'+('yes' if stats['pointing_correct'] else 'no'))
        for label in ['gt_mass', 'gt_area_fraction', 'mass_gain', 'localization_margin']:
            value = stats[label]
            col.caption('%s: %s' % (label, '不适用（无干扰目标）' if value is None else '%.4f' % value))
        col.caption('peak row/col: %d/%d (%s)' % (stats['peak_row'], stats['peak_col'], stats['peak_location']))
    diff = np.abs(maps['phase22_router']-maps['phase22_direct'])
    st.image(heatmap_rgb(diff), caption='|Phase 2.2 Router - same-backbone direct CLIP| (difference scale)', width=224)
    st.subheader('同图像短语切换')
    groups = manifest['switching_groups']
    if not groups:
        st.info('本次导出没有空间分离的实体对。')
    else:
        gid = st.selectbox('切换实验图像', [g['image_id'] for g in groups],
                           index=next((i for i, g in enumerate(groups) if g['image_id'] == image_id), 0))
        group = next(g for g in groups if g['image_id'] == gid)
        fmt = lambda key: phrases[key]['phrase']+' ['+key+']'
        aid = st.selectbox('短语 A', group['phrase_ids'], format_func=fmt)
        bid = st.selectbox('短语 B', [k for k in group['phrase_ids'] if k != aid], format_func=fmt)
        variant = st.selectbox('切换实验模型', manifest['variants'], index=2, format_func=LABELS.get)
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
        st.write('两个查询均偏向自身目标：'+str(metrics['both_prefer_target']))
        st.caption('GT union IoU = %.4f. JSD measures change; switch margin measures whether change favours the named target.' % union_iou(pa['boxes'], pb['boxes']))
    with st.expander('类别统计与峰值位置'):
        s = summary[model]
        st.json(s['peak_location'])
        st.caption('背景指所有标注框以外的区域，其中可能存在未标注物体。')
        st.dataframe(pd.DataFrame([{'category': cat, 'phrases': stats['phrases'],
                                   'pointing': stats['pointing_correct']['mean'],
                                   'mass_gain': stats['mass_gain']['mean']}
                                  for cat, stats in s.get('categories', {}).items()]))
