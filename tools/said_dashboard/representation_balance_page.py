"""Chinese artifact-only monitoring. No checkpoint or model imports."""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

FONT = 'Microsoft YaHei, PingFang SC, Noto Sans CJK SC, Source Han Sans SC, Arial Unicode MS, sans-serif'
COCO_MISSING = '当前 checkpoint 尚未运行 COCO Retrieval 评估。'
TAGS = {'initial': '初始模型（未训练）', 'step100': '第 100 步', 'step200': '第 200 步',
        'step400': '第 400 步', 'final': '训练结束'}
DETAIL = {'first_sentence': '第一句', '25%': '25%', '50%': '50%', '75%': '75%', '100%': '完整文本'}
REPRESENTATIONS = {'base': '初始 CLIP', 'full': '当前完整表征', 'said': '当前 Said 表征'}
TRAIN_LABELS = {'loss_global': '全局对齐损失', 'loss_said': 'Said 损失',
                'loss_route': '路由识别损失', 'loss_evidence': '证据识别损失',
                'route_top1_acc': '路由 Top-1', 'evidence_top1_acc': '证据 Top-1',
                'route_margin': '路由 Margin', 'evidence_margin': '证据 Margin',
                'backbone_lr': '骨干网络学习率', 'head_lr': 'Router 学习率',
                'said_attention_entropy': '注意力熵', 'said_effective_patch_count': '有效 patch 数'}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError('诊断文件尚未生成：' + Path(path).name) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('诊断文件无法读取：' + Path(path).name) from exc


def load_bundle(root, tag):
    root = Path(root)
    return {name: read_json(root / tag / (name + '.json'))
            for name in ('gap_metrics', 'caption_detail', 'pca')}


def style():
    st.markdown('<style>html, body, .stApp, .stMarkdown, .stTextInput input, '
                '.stSelectbox, .stRadio, .stMetric, .stCaption {font-family:' + FONT + ';}'
                '[data-testid="stMetricLabel"] {white-space:normal; height:auto;}'
                '[data-testid="stSidebar"] label {white-space:normal;}'
                '</style>', unsafe_allow_html=True)


def show_chart(chart, height=250):
    st.altair_chart(chart.properties(height=height).configure(font=FONT).configure_axis(
        labelFont=FONT, titleFont=FONT).configure_legend(labelFont=FONT, titleFont=FONT,
                                                       orient='bottom'),
        width='stretch')


def cards(items):
    for start in range(0, len(items), 3):
        for column, (label, value, help_text) in zip(st.columns(3), items[start:start + 3]):
            column.metric(label, '未记录' if value is None else '%.5f' % value, help=help_text)


def pca_chart(pca, name):
    points, centers, boundaries = pca['points'], pca['centroids'], pca['ellipses']
    rows = []
    for group, label in [(name, '视觉表征'), ('text', '文本表征')]:
        rows.extend({'横坐标': x, '纵坐标': y, '模态': label, '固定样本序号': i}
                    for i, (x, y) in enumerate(points[group]))
    axes = {'x': alt.X('横坐标:Q', title='主成分 1', scale=alt.Scale(domain=pca['axis_limits']['x'], nice=False)),
            'y': alt.Y('纵坐标:Q', title='主成分 2', scale=alt.Scale(domain=pca['axis_limits']['y'], nice=False))}
    color = alt.Color('模态:N', scale=alt.Scale(domain=['视觉表征', '文本表征'], range=['#3677bc', '#e68a32']),
                      title=None, legend=alt.Legend(orient='bottom'))
    dots = alt.Chart(pd.DataFrame(rows)).mark_point(filled=True, opacity=.42, size=19).encode(
        **axes, color=color,
        shape=alt.Shape('模态:N', scale=alt.Scale(domain=['视觉表征', '文本表征'], range=['circle', 'triangle-up']),
                        title=None, legend=alt.Legend(orient='bottom')),
        tooltip=['模态:N', '固定样本序号:Q'])
    ellipse_rows = []
    center_rows = []
    for group, label, center_label in [(name, '视觉表征', '视觉中心'), ('text', '文本表征', '文本中心')]:
        ellipse_rows.extend({'横坐标': x, '纵坐标': y, '模态': label, '顺序': i}
                            for i, (x, y) in enumerate(boundaries[group]))
        center_rows.append({'横坐标': centers[group][0], '纵坐标': centers[group][1],
                            '模态': label, '中心': center_label})
    ellipses = alt.Chart(pd.DataFrame(ellipse_rows)).mark_line(strokeDash=[5, 3], opacity=.8).encode(
        **axes, color=color, order='顺序:Q')
    centroids = alt.Chart(pd.DataFrame(center_rows)).mark_point(shape='cross', size=150, strokeWidth=2).encode(
        **axes, color=color, tooltip=['中心:N'],
        shape=alt.Shape('中心:N', title=None, scale=alt.Scale(domain=['视觉中心', '文本中心'], range=['cross', 'cross']),
                        legend=alt.Legend(orient='bottom')))
    pairs = [{'横坐标': points[name][i][0], '纵坐标': points[name][i][1],
              '文本横坐标': points['text'][i][0], '文本纵坐标': points['text'][i][1]}
             for i in pca['paired_line_indices']]
    lines = alt.Chart(pd.DataFrame(pairs)).mark_rule(color='#8c96a0', opacity=.18).encode(
        **axes, x2='文本横坐标:Q', y2='文本纵坐标:Q')
    return (lines + dots + ellipses + centroids).resolve_scale(shape='independent')


def line_chart(rows, x, y, series, order=None):
    return alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
        x=alt.X(x + (':O' if order else ':Q'), sort=order, title=x),
        y=alt.Y(y + ':Q', scale=alt.Scale(zero=False), title=y),
        color=alt.Color(series + ':N', title=None, legend=alt.Legend(orient='bottom')),
        tooltip=[x, series, alt.Tooltip(y + ':Q', format='.5f')])


def render_coco(coco):
    if not coco:
        st.info(COCO_MISSING)
        return
    st.caption(coco.get('source', '标准 CLIP 全局检索评估'))
    rows = [{'检索方向': label, **{k: '%.2f%%' % (100 * v) for k, v in coco['metrics'][key].items()}}
            for key, label in [('I2T', '图像 → 文本'), ('T2I', '文本 → 图像')]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def main():
    style()
    st.title('表征平衡监控')
    st.caption('固定 Probe 指标：检验 Caption 条件化表征是否更匹配当前文本。所有正式距离在原始高维空间计算。')
    root = st.sidebar.text_input('表征诊断目录', os.environ.get('REPRESENTATION_BALANCE_ROOT', 'outputs/representation_balance'))
    try:
        summary = read_json(Path(root) / 'summary.json')
        tags = list(summary['checkpoints'])
        if not tags:
            st.info('固定 Probe 正在导出，首个 checkpoint 完成后即可查看。')
            return
        tag = st.sidebar.selectbox('选择模型阶段', tags, index=len(tags) - 1,
                                    format_func=lambda x: TAGS.get(x, x))
        bundle = load_bundle(root, tag)
        render(summary, tag, bundle)
    except (ValueError, KeyError, TypeError) as exc:
        st.warning('表征诊断暂不可用，请检查离线导出结果。' + (str(exc) if isinstance(exc, ValueError) else '文件结构不完整。'))
        st.info('请先完成固定 Probe 导出；此页面不依赖注意力图文件。')


def render(summary, tag, bundle):
    entry = summary['checkpoints'][tag]
    metrics, detail, pca = [bundle[k] for k in ('gap_metrics', 'caption_detail', 'pca')]
    train = entry['training']
    st.subheader('训练概览')
    st.write('当前 checkpoint：%s；已完成 %d 步；epoch：%d' %
              (TAGS.get(tag, tag), entry['step'], entry['epoch']))
    st.caption('复用已完成的 Phase 2.5 residual 训练轨迹。下列损失与路由统计来自对应训练 Batch；初始模型尚未训练，旧日志未记录的新字段不回填。')
    st.dataframe(pd.DataFrame([{'训练指标': label, '数值': '未记录' if train.get(key) is None else '%.6g' % train[key]}
                              for key, label in TRAIN_LABELS.items()]), hide_index=True, width='stretch')

    st.subheader('表征差距')
    st.caption('固定 Probe：%d 张图像，五个阶段共用同一 manifest。初始 CLIP 图像表征固定，文本表征随当前模型变化。' % summary['n'])
    help_gap = '配对余弦差距（Pair Gap）：平均 1 − 余弦相似度，越小表示配对更接近。'
    help_gain = '完整视觉表征与文本的 Gap 减去 Caption 条件化 Said 表征与文本的 Gap。正值表示经过 Said 过滤后，视觉表征与当前文本更加匹配。'
    help_rmg = '相对模态差距（RMG）：配对距离除以配对距离与平均模态内距离之和；不能单独当作任务准确率。'
    cards([('初始 CLIP Pair Gap', metrics['base']['pair_gap'], help_gap),
           ('当前完整表征 Pair Gap', metrics['full']['pair_gap'], help_gap),
           ('Said 表征 Pair Gap', metrics['said']['pair_gap'], help_gap),
           ('语义平衡增益', metrics['balancing_gain'], help_gain),
           ('完整表征 RMG', metrics['full']['rmg'], help_rmg),
           ('Said 表征 RMG', metrics['said']['rmg'], help_rmg)])
    with st.expander('查看中心差距与完整高维指标'):
        st.caption('模态中心距离（L2M）：视觉中心与文本中心的欧氏距离；均在原始 embedding 维度计算。')
        st.dataframe(pd.DataFrame([{'表征': label, 'Pair Gap': metrics[key]['pair_gap'],
                                    'L2M': metrics[key]['l2m'], 'RMG': metrics[key]['rmg']}
                                   for key, label in REPRESENTATIONS.items()]), hide_index=True, width='stretch')

    st.subheader('跨模态表征空间')
    st.caption('联合主成分分析（PCA）：三图共享一个 basis 和相同坐标范围。圆点为视觉、三角为文本、十字为模态中心；虚线为 95% 协方差椭圆，灰线为固定的最多 30 对样本。二维图仅用于观察整体分布，正式 Gap 指标均在原始高维 embedding 中计算。')
    for column, (name, label) in zip(st.columns(3), REPRESENTATIONS.items()):
        with column:
            st.write(label)
            show_chart(pca_chart(pca, name), 270)
    st.caption('前两主成分解释方差：%.2f%% / %.2f%%。不同 checkpoint 各自联合拟合；不要跨 checkpoint 比较二维绝对坐标。' %
               tuple(100 * x for x in pca['explained_variance_ratio']))

    st.subheader('文本详细程度与表征差距')
    st.caption('该曲线使用 caption detail 作为信息覆盖代理，不将文本长度直接等同于真实语义覆盖率。重复 level 共用推理结果，各 level 保留同一组图像；不平滑、不筛选样本。')
    rows = [{'文本详细程度': DETAIL[r['level']], '表征': label, 'Pair Gap': r[key]['pair_gap']}
            for r in detail['levels'] for key, label in REPRESENTATIONS.items()]
    show_chart(line_chart(rows, '文本详细程度', 'Pair Gap', '表征', list(DETAIL.values())))
    rows = [{'文本详细程度': DETAIL[r['level']], '指标': '语义平衡增益', '语义平衡增益': r['balancing_gain']}
            for r in detail['levels']]
    show_chart(line_chart(rows, '文本详细程度', '语义平衡增益', '指标', list(DETAIL.values())), 170)
    with st.expander('查看 tokenizer 截断情况'):
        st.caption('使用训练相同的 LongCLIP tokenizer，最多 248 tokens，保留相同 truncate 行为。完整 dense caption 也可能被截断。')
        st.dataframe(pd.DataFrame([{'文本详细程度': DETAIL[r['level']], '平均使用 tokens': r['mean_used_tokens'],
                                   '截断样本数': r['truncated_count'], '样本数': r['n']}
                                  for r in detail['levels']]), hide_index=True, width='stretch')

    st.subheader('训练过程中的表征平衡变化')
    st.caption('以下均为固定 Probe 指标；不使用训练 minibatch 做 checkpoint 间正式比较。')
    for key, label in [('pair_gap', 'Pair Gap'), ('rmg', 'RMG')]:
        rows = [{'训练步数': e['step'], '表征': REPRESENTATIONS[k], label: e['metrics'][k][key]}
                for e in summary['checkpoints'].values() for k in ('full', 'said')]
        show_chart(line_chart(rows, '训练步数', label, '表征'), 190)
    rows = [{'训练步数': e['step'], '指标': '语义平衡增益', '语义平衡增益': e['metrics']['balancing_gain']}
            for e in summary['checkpoints'].values()]
    show_chart(line_chart(rows, '训练步数', '语义平衡增益', '指标'), 170)

    st.subheader('文本条件化有效性')
    cards([('正确 Caption Said Gap', metrics['said_own_gap'], '固定 Probe；使用本图 caption 条件化。'),
           ('打乱 Caption Said Gap', metrics['said_shuffle_gap'], '固定 Probe；使用下一张图 caption 条件化，仍与本图文本计算距离。'),
           ('Conditioning Margin', metrics['conditioning_gap_margin'], '打乱 Gap 减去正确 Gap；正值支持文本条件化有效性。')])
    st.caption('如果正确 caption 条件化后的 Said 表征明显比错误 caption 条件化后的 Said 表征更接近目标文本，说明 Said 分支确实响应文本条件，而不是固定视觉过滤器。这是条件化 sanity check，不是空间定位准确率。')
    st.write('训练 Batch 路由 Top-1：%s；训练 Batch 路由 Margin：%s' %
              (train.get('route_top1_acc', '未记录'), train.get('route_margin', '未记录')))
    st.subheader('标准 CLIP 能力保持')
    render_coco(entry.get('coco'))


def training_main():
    style()
    st.title('训练状态')
    st.caption('训练 Batch 诊断：以下为 rank 0 本地 batch，仅用于训练监控，不替代固定 Probe 的正式比较。')
    root = st.sidebar.text_input('表征诊断目录', os.environ.get('REPRESENTATION_BALANCE_ROOT', 'outputs/representation_balance'))
    try:
        data = read_json(Path(root) / 'batch_history.json')
        st.info(data['note'])
        records = data['records']
        if not records:
            st.info('尚无训练 Batch 日志。旧实验日志可以继续使用，缺失指标显示为未记录。')
            return
        st.write('日志实验：' + str(data.get('run', '未记录')))
        last = records[-1]
        st.write('已完成步数：%d；epoch：%d' % (last.get('completed_steps', last['step'] + 1), last['epoch']))
        labels = {'pair_gap_full': '完整表征 Pair Gap', 'pair_gap_said': 'Said 表征 Pair Gap',
                  'balancing_gain': '语义平衡增益', 'relative_balancing_gain': '相对平衡增益'}
        cards([(label, last.get(key), '训练 Batch 诊断') for key, label in labels.items()])
        st.dataframe(pd.DataFrame([{'训练指标': label, '数值': last.get(key)}
                                  for key, label in TRAIN_LABELS.items()]), hide_index=True, width='stretch')
        for keys, label in [(('pair_gap_full', 'pair_gap_said'), 'Pair Gap'),
                            (('balancing_gain',), '语义平衡增益'),
                            (('relative_balancing_gain',), '相对平衡增益')]:
            rows = [{'训练步数': r.get('completed_steps', r['step'] + 1), '指标': labels[k], label: r[k]}
                    for r in records for k in keys if k in r]
            if rows:
                show_chart(line_chart(rows, '训练步数', label, '指标'), 190)
    except (ValueError, KeyError, TypeError) as exc:
        st.warning('训练日志暂不可用。' + (str(exc) if isinstance(exc, ValueError) else '文件结构不完整。'))
