"""验证得分曲线：只读训练过程中写入的 validation_history.jsonl。

页面不加载 checkpoint、不运行模型、不重算任何指标：它只把每个 run 目录下
``validation_history.jsonl`` 里已经写好的验证点画成随训练步数变化的曲线。

比较规则（Phase 2.7C merge 前加固）：

* 主曲线、latest points、主数值表**只使用** ``similarity_chunk == 512`` 的记录；
  其它 chunk 的记录仍然出现在警告、异常记录表与原始 JSONL 展开区，但绝不进入主曲线。
* 一条 series 永不合并两个 run：「只按文本变体」只在只选 1 个 run 时生效，
  多 run 时自动回退为「实验 · 数据集 · 变体」。

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
    curve_series_mode,
    dataset_label,
    dropped_raw_lines,
    is_canonical,
    latest_points,
    load_runs,
    metric_label,
    non_canonical_chunks,
    split_canonical,
    total_wall_sec,
    variant_label,
    wide_rows,
)

DEFAULT_ROOT = 'runs_salu'
FONT = ('Microsoft YaHei, PingFang SC, Noto Sans CJK SC, Source Han Sans SC, '
        'Arial Unicode MS, sans-serif')
GROUP_VARIANT = '只按文本变体'
GROUP_FULL = '实验 · 数据集 · 变体'


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


def render_dropped(runs):
    """非 canonical 记录：警告 + 异常表 + 原始行，但绝不进入主曲线。"""
    dropped = non_canonical_chunks(runs)
    if not dropped:
        return
    st.warning('以下验证点未使用 canonical similarity_chunk = %d，已被排除在主曲线、最新验证点与'
               '数值表之外（原始记录仍在下方展开区可见），不能与 canonical 曲线直接比较：'
               % CANONICAL_SIMILARITY_CHUNK)
    st.dataframe(pd.DataFrame(dropped), hide_index=True, width='stretch')


def render_raw_view(all_runs, dropped_count):
    """字段说明 + 每个 run 的文件与条数 + 被排除记录的原始 JSONL 行。"""
    with st.expander('记录字段、被排除的记录与原始 JSONL'):
        st.markdown('- 每条记录：`step` / `epoch` / `dataset` / `caption_variant` / `reason` / '
                    '`metrics` / `wall_sec` / `protocol` / `similarity_chunk`\n'
                    '- `reason`：`initial`（step 0）/ `interval`（`--val_every`）/ `epoch_end` / `final`\n'
                    '- 主曲线只用 `similarity_chunk = %d`（%d 条被排除，仍可在此查看原始行）\n'
                    '- Balancing Gain 定义：`%s`（%s）'
                    % (CANONICAL_SIMILARITY_CHUNK, dropped_count, BALANCING_GAIN_DEFINITION,
                       BALANCING_GAIN_SIGN))
        for run in all_runs:
            kept = sum(1 for record in run['records'] if is_canonical(record))
            st.text('%s：%s（%d 条记录，canonical %d 条）'
                    % (run['name'], run['path'], len(run['records']), kept))
        excluded = dropped_raw_lines(all_runs)
        if excluded:
            st.markdown('**被排除的原始 JSONL 行**')
            st.dataframe(pd.DataFrame([{'实验': item['实验'], '原因': item['原因']}
                                       for item in excluded]), hide_index=True, width='stretch')
            st.code('\n'.join(item['原始行'] for item in excluded))
        st.caption('验证只读训练已写入的 JSONL；指标可复现，且从不进入 loss 或检索排序。')


def main():
    st.markdown('<style>.stApp h1,.stApp h2,.stApp h3,.stApp p,'
                '.stApp [data-testid="stMetricValue"] {font-family:' + FONT + ' !important;}'
                '</style>', unsafe_allow_html=True)
    st.title('验证得分曲线')
    st.caption('只读训练过程中写入的验证记录（`<output_dir>/validation_history.jsonl`）：不加载 checkpoint，'
               '不重新推理。主曲线只使用 canonical similarity_chunk = %d 的记录，公平性来自'
               '固定 evaluator 与固定 chunk。' % CANONICAL_SIMILARITY_CHUNK)

    root = st.sidebar.text_input('运行目录', os.environ.get('SAID_RUNS_ROOT', DEFAULT_ROOT))
    all_runs = load_runs(root)
    if not all_runs:
        st.info('未找到验证记录。训练时用 `--val_every` / `--val_sharegpt4v` / `--eval_coco_initial` 打开验证，'
                '每个验证点会追加到 `<output_dir>/validation_history.jsonl`。')
        st.code('torchrun --nproc_per_node=4 train/train_salu.py \\\n'
                '  --val_every 100 --val_sharegpt4v --eval_coco_initial \\\n'
                '  --output_dir runs_salu/<run_name>')
        return

    names = [run['name'] for run in all_runs]
    chosen = st.sidebar.multiselect('实验', names, default=names)
    runs = [run for run in all_runs if run['name'] in chosen]
    if not runs:
        st.info('请在左侧至少选择一个实验。')
        return

    # 主曲线 / latest / 数值表只吃 canonical chunk 的记录
    canonical_runs, dropped = split_canonical(runs)

    datasets = sorted({name for run in canonical_runs for name in run['datasets']})
    picked_datasets = st.sidebar.multiselect('数据集', datasets, default=datasets,
                                             format_func=dataset_label)
    variants = sorted({name for run in canonical_runs for name in run['variants']})
    picked_variants = st.sidebar.multiselect('文本变体', variants, default=variants,
                                             format_func=variant_label)
    if not datasets or not variants:
        st.info('所选实验没有任何 canonical（similarity_chunk = %d）验证记录，无法绘制主曲线。'
                % CANONICAL_SIMILARITY_CHUNK)
        render_dropped(runs)
        render_raw_view(all_runs, len(dropped))
        return
    if not picked_datasets or not picked_variants:
        st.info('请至少选择一个数据集与一个文本变体。')
        return

    metrics = available_metrics(canonical_runs, datasets=set(picked_datasets))
    default_metrics = [key for key in DEFAULT_METRICS if key in metrics] or metrics[:1]
    picked_metrics = st.sidebar.multiselect('指标', metrics, default=default_metrics,
                                            format_func=metric_label)
    requested = st.sidebar.radio('曲线分组', (GROUP_FULL, GROUP_VARIANT), index=0)
    show_table = st.sidebar.checkbox('显示数值表', value=True)
    if not picked_metrics:
        st.info('请至少选择一个指标。')
        return

    series = curve_series_mode(len(runs), 'variant' if requested == GROUP_VARIANT else 'auto')
    if requested == GROUP_VARIANT and series != 'variant':
        st.warning('已选择 %d 个实验：「只按文本变体」会合并不同实验的同名变体，因此已自动改用'
                   '「%s」分组；只看单个实验时才允许按变体合并。' % (len(runs), GROUP_FULL))

    rows = curve_rows(canonical_runs, metrics=picked_metrics, datasets=set(picked_datasets),
                      variants=set(picked_variants), series=series)
    if not rows:
        st.info('当前筛选条件下没有 canonical 验证记录。')
        render_dropped(runs)
        render_raw_view(all_runs, len(dropped))
        return

    render_dropped(runs)

    steps = sorted({row['步数'] for row in rows})
    first, second = st.columns(2)
    first.metric('验证点（步数）', '共 %d 个：%s' % (len(steps), ', '.join(str(s) for s in steps[:8])
                                                   + ('…' if len(steps) > 8 else '')))
    second.metric('验证耗时合计（秒）', '%.1f' % sum(total_wall_sec(canonical_runs).values()))

    st.subheader('各实验最新验证点')
    st.dataframe(pd.DataFrame(latest_points(canonical_runs, datasets=set(picked_datasets))),
                 hide_index=True, width='stretch')

    st.subheader('得分曲线')
    st.caption('仅 canonical similarity_chunk = %d；每张图独立 y 轴。' % CANONICAL_SIMILARITY_CHUNK)
    render_matrix(rows, picked_metrics)
    st.caption('Balancing Gain = Full Pair Gap − Said Pair Gap（%s）；正值表示 Caption 条件化 Said 表征'
               '比完整视觉表征更匹配当前文本。' % BALANCING_GAIN_SIGN)

    if show_table:
        st.subheader('验证记录（canonical）')
        st.dataframe(pd.DataFrame(wide_rows(rows)), hide_index=True, width='stretch')

    render_raw_view(all_runs, len(dropped))


if __name__ == '__main__':
    main()
