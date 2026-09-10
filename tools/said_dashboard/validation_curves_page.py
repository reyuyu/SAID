"""验证得分曲线：只读训练过程中写入的 validation_history.jsonl。

页面不加载 checkpoint、不运行模型、不重算任何指标：它只把每个 run 目录下
``validation_history.jsonl`` 里已经写好的验证点画成随训练步数变化的曲线。

Run (server, loopback only)::

    streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

from tools.said_dashboard.validation_curves_data import (
    BALANCING_GAIN_DEFINITION,
    BALANCING_GAIN_SIGN,
    CANONICAL_SIMILARITY_CHUNK,
    DEFAULT_METRICS,
    DIAGNOSTIC_KEYS,
    available_metrics,
    curve_rows,
    dataset_label,
    latest_points,
    load_runs,
    metric_label,
    non_canonical_chunks,
    total_wall_sec,
    variant_label,
    wide_rows,
)

DEFAULT_ROOT = 'runs_salu'
DEFAULT_EXPAND = ('image2text_R1', 'text2image_R1', 'balancing_gain')
FONT = ('Microsoft YaHei, PingFang SC, Noto Sans CJK SC, Source Han Sans SC, '
        'Arial Unicode MS, sans-serif')


def points_chart(rows, height=220):
    """One metric: value against training step, one line per series."""
    frame = pd.DataFrame(rows)
    return alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X('步数:Q', title='训练步数'),
        y=alt.Y('数值:Q', title=None, scale=alt.Scale(zero=False)),
        color=alt.Color('系列:N', title=None, legend=alt.Legend(orient='bottom')),
        tooltip=['实验:N', '数据集:N', '文本变体:N', '验证点:N', '步数:Q',
                 alt.Tooltip('数值:Q', format='.5f'),
                 alt.Tooltip('耗时(s):Q', format='.1f')],
    ).properties(height=height)


def show_chart(chart):
    st.altair_chart(chart.configure(font=FONT).configure_axis(
        labelFont=FONT, titleFont=FONT).configure_legend(
        labelFont=FONT, titleFont=FONT, orient='bottom'),
        width='stretch')


def render_matrix(rows, metrics):
    """Two charts per row so every metric keeps its own y scale."""
    for start in range(0, len(metrics), 2):
        for column, metric in zip(st.columns(2), metrics[start:start + 2]):
            subset = [row for row in rows if row['指标'] == metric_label(metric)]
            with column:
                st.markdown('**%s**' % metric_label(metric))
                if not subset:
                    st.caption('所选实验未记录该指标。')
                    continue
                show_chart(points_chart(subset))
                if metric in DIAGNOSTIC_KEYS:
                    st.caption('Said 特征只用于诊断，不进入检索排序。')


def main():
    st.markdown('<style>.stApp h1,.stApp h2,.stApp h3,.stApp p,'
                '.stApp [data-testid="stMetricValue"] {font-family:' + FONT + ' !important;}'
                '</style>', unsafe_allow_html=True)
    st.title('验证得分曲线')
    st.caption('只读训练过程中写入的验证记录（`<output_dir>/validation_history.jsonl`）：不加载 checkpoint，'
               '不重新推理。所有曲线的公平性来自固定 evaluator 与 canonical similarity_chunk = %d。'
               % CANONICAL_SIMILARITY_CHUNK)

    root = st.sidebar.text_input('运行目录', os.environ.get('SAID_RUNS_ROOT', DEFAULT_ROOT))
    runs = load_runs(root)
    if not runs:
        st.info('未找到验证记录。训练时用 `--val_every` / `--val_sharegpt4v` / `--eval_coco_initial` 打开验证，'
                '每个验证点会追加到 `<output_dir>/validation_history.jsonl`。')
        st.code('torchrun --nproc_per_node=4 train/train_salu.py \\\n'
                '  --val_every 100 --val_sharegpt4v --eval_coco_initial \\\n'
                '  --output_dir runs_salu/<run_name>')
        return

    names = [run['name'] for run in runs]
    chosen = st.sidebar.multiselect('实验', names, default=names)
    runs = [run for run in runs if run['name'] in chosen]
    if not runs:
        st.info('请在左侧至少选择一个实验。')
        return

    datasets = sorted({name for run in runs for name in run['datasets']})
    picked_datasets = st.sidebar.multiselect('数据集', datasets, default=datasets,
                                             format_func=dataset_label)
    variants = sorted({name for run in runs for name in run['variants']})
    picked_variants = st.sidebar.multiselect('文本变体', variants, default=variants,
                                             format_func=variant_label)
    if not picked_datasets or not picked_variants:
        st.info('请至少选择一个数据集与一个文本变体。')
        return

    metrics = available_metrics(runs, datasets=set(picked_datasets))
    default_metrics = [key for key in DEFAULT_METRICS if key in metrics] or metrics[:1]
    picked_metrics = st.sidebar.multiselect('指标', metrics, default=default_metrics,
                                            format_func=metric_label)
    compare = st.sidebar.radio('曲线分组', ('实验 · 数据集 · 变体', '只按文本变体'), index=0)
    show_table = st.sidebar.checkbox('显示数值表', value=True)
    if not picked_metrics:
        st.info('请至少选择一个指标。')
        return

    rows = curve_rows(runs, metrics=picked_metrics, datasets=set(picked_datasets),
                      variants=set(picked_variants),
                      series='variant' if compare == '只按文本变体' else 'auto')
    if not rows:
        st.info('当前筛选条件下没有验证记录。')
        return

    outliers = non_canonical_chunks(runs)
    if outliers:
        st.warning('以下验证点未使用 canonical similarity_chunk = %d，不能与其它曲线直接比较：'
                   % CANONICAL_SIMILARITY_CHUNK)
        st.dataframe(pd.DataFrame(outliers), hide_index=True, width='stretch')

    steps = sorted({row['步数'] for row in rows})
    first, second = st.columns(2)
    first.metric('验证点（步数）', '共 %d 个：%s' % (len(steps), ', '.join(str(s) for s in steps[:8])
                                                   + ('…' if len(steps) > 8 else '')))
    second.metric('验证耗时合计（秒）', '%.1f' % sum(total_wall_sec(runs).values()))

    st.subheader('各实验最新验证点')
    st.dataframe(pd.DataFrame(latest_points(runs, datasets=set(picked_datasets))),
                 hide_index=True, width='stretch')

    st.subheader('得分曲线')
    render_matrix(rows, picked_metrics)
    st.caption('Balancing Gain = Full Pair Gap − Said Pair Gap（%s）；正值表示 Caption 条件化 Said 表征'
               '比完整视觉表征更匹配当前文本。' % BALANCING_GAIN_SIGN)

    if show_table:
        st.subheader('验证记录')
        st.dataframe(pd.DataFrame(wide_rows(rows)), hide_index=True, width='stretch')

    with st.expander('记录字段与原始 JSONL'):
        st.markdown('- 每条记录：`step` / `epoch` / `dataset` / `caption_variant` / `reason` / '
                    '`metrics` / `wall_sec` / `protocol` / `similarity_chunk`\n'
                    '- `reason`：`initial`（step 0）/ `interval`（`--val_every`）/ `epoch_end` / `final`\n'
                    '- Balancing Gain 定义：`%s`（%s）' % (BALANCING_GAIN_DEFINITION, BALANCING_GAIN_SIGN))
        for run in runs:
            st.text('%s：%s（%d 条记录）' % (run['name'], run['path'], len(run['records'])))
        st.caption('验证只读训练已写入的 JSONL；指标可复现，且从不进入 loss 或检索排序。')


if __name__ == '__main__':
    main()
