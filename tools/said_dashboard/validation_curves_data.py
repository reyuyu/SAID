"""Pure helpers for the validation score curves page (no Streamlit import).

The page only reads the ``validation_history.jsonl`` records that training already
writes (``<output_dir>/validation_history.jsonl``, one JSON object per validation
point). It never loads a checkpoint, runs the model or recomputes a metric.

Every record carries ``dataset`` / ``caption_variant`` / ``reason`` / ``step`` /
``epoch`` / ``wall_sec`` / ``protocol`` / ``similarity_chunk`` and a ``metrics``
block whose shape depends on the dataset:

* ``sharegpt4v1k`` -> ``{'retrieval': {...}, 'diagnostics': {...}}``
* ``coco_val2017`` -> a flat retrieval dict

``flatten_metrics`` hides that difference from the charts.

Comparison rules (Phase 2.7C hardening):

* Only records with the canonical ``similarity_chunk`` may enter the main curves, the
  latest-point table and the numeric table. Records with any other chunk stay visible
  in the warning, in the anomaly table and in the raw JSONL view, but they are never
  plotted next to canonical numbers.
* A series never merges two runs: caption-variant-only grouping is allowed only when a
  single run is selected, otherwise the grouping falls back to run · dataset · variant.
"""
import json
from pathlib import Path

HISTORY_NAME = 'validation_history.jsonl'
CANONICAL_SIMILARITY_CHUNK = 512
# Same wording as ``eval.validation_protocol.BALANCING_GAIN_DEFINITION`` /
# ``BALANCING_GAIN_SIGN``; the dashboard cannot import that module (it pulls torch),
# so a test asserts the two definitions stay identical.
BALANCING_GAIN_DEFINITION = 'full_pair_gap - said_pair_gap'
BALANCING_GAIN_SIGN = 'positive = Said better; negative = Said worse'

RETRIEVAL_KEYS = ('image2text_R1', 'image2text_R5', 'image2text_R10',
                  'text2image_R1', 'text2image_R5', 'text2image_R10')
DIAGNOSTIC_KEYS = ('full_pair_gap', 'said_pair_gap', 'full_rmg', 'said_rmg',
                   'balancing_gain', 'relative_balancing_gain', 'conditioning_margin')
METRIC_LABELS = {
    'image2text_R1': 'I2T R@1 ↑', 'image2text_R5': 'I2T R@5 ↑', 'image2text_R10': 'I2T R@10 ↑',
    'text2image_R1': 'T2I R@1 ↑', 'text2image_R5': 'T2I R@5 ↑', 'text2image_R10': 'T2I R@10 ↑',
    'full_pair_gap': 'Full Pair Gap ↓', 'said_pair_gap': 'Said Pair Gap ↓',
    'full_rmg': 'Full RMG ↓', 'said_rmg': 'Said RMG ↓',
    'balancing_gain': 'Balancing Gain（Full − Said）↑',
    'relative_balancing_gain': 'Relative Balancing Gain ↑',
    'conditioning_margin': 'Conditioning Margin ↑',
}
DATASET_LABELS = {'sharegpt4v1k': 'ShareGPT4V-1K（1,000 路）',
                  'coco_val2017': 'COCO val2017（5,000 图 × 5 caption）'}
VARIANT_LABELS = {'first_sentence': '第一句', 'fixed_sparse': '训练同规则稀疏',
                  'full_dense': '完整 caption', 'coco_5captions': 'COCO 5 captions'}
REASON_LABELS = {'initial': '初始 step 0', 'interval': '间隔', 'epoch_end': 'epoch 结束',
                 'final': '训练结束'}
DEFAULT_METRICS = ('image2text_R1', 'image2text_R5', 'text2image_R1', 'text2image_R5',
                   'balancing_gain', 'conditioning_margin')


def metric_label(key):
    return METRIC_LABELS.get(key, key)


def dataset_label(name):
    return DATASET_LABELS.get(name, str(name))


def variant_label(name):
    return VARIANT_LABELS.get(name, str(name))


def reason_label(name):
    return REASON_LABELS.get(name, str(name) if name else '未记录')


def read_history(path):
    """Records of one run, oldest line first.

    A missing file raises ``ValueError``; blank and malformed lines (a run killed
    while appending) are skipped instead of breaking the page.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise ValueError('验证记录尚未生成：' + path.name) from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError('验证记录无法读取：' + path.name) from exc
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and 'step' in record:
            records.append(record)
    return records


def discover_runs(root):
    """Every run under ``root`` that has a validation history, sorted by name.

    ``root`` is either a parent directory of runs (``runs_salu``) or a single run
    directory (``runs_salu/<run_name>``, when the history sits directly inside it).
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    if (root / HISTORY_NAME).is_file():
        found.append({'name': root.name or str(root), 'path': str(root / HISTORY_NAME)})
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith('.') and (entry / HISTORY_NAME).is_file():
            found.append({'name': entry.name, 'path': str(entry / HISTORY_NAME)})
    return found


def load_runs(root):
    """Discovered runs with their records plus the datasets / variants they cover."""
    runs = []
    for run in discover_runs(root):
        try:
            records = read_history(run['path'])
        except ValueError:
            continue
        runs.append(summarize_run(run, records))
    return runs


def summarize_run(run, records):
    """``run`` plus the steps / datasets / variants its records cover."""
    records = list(records)
    return dict(run, records=records,
                steps=sorted({int(r['step']) for r in records}),
                datasets=sorted({str(r.get('dataset', '?')) for r in records}),
                variants=sorted({str(r.get('caption_variant', '?')) for r in records}))


def is_canonical(record, canonical=CANONICAL_SIMILARITY_CHUNK):
    """Canonical means the record was evaluated with the canonical similarity chunk."""
    return record.get('similarity_chunk') == canonical


def dropped_row(run_name, record):
    """One row of the anomaly table for a record that may not join the main curves."""
    return {'实验': run_name,
            '步数': int(record.get('step', -1)),
            '数据集': dataset_label(str(record.get('dataset', '?'))),
            '文本变体': variant_label(str(record.get('caption_variant', '?'))),
            '验证点': reason_label(record.get('reason')),
            'similarity_chunk': record.get('similarity_chunk')}


def split_canonical(runs, canonical=CANONICAL_SIMILARITY_CHUNK):
    """``(canonical_runs, dropped)``: curves / latest / tables may only use the first.

    ``dropped`` keeps every excluded record (wrong chunk, or a chunk that is missing)
    so the page can still warn about it, list it and show its raw line.
    """
    kept_runs, dropped = [], []
    for run in runs:
        kept_runs.append(summarize_run(run, [r for r in run['records']
                                             if is_canonical(r, canonical)]))
        dropped.extend(dropped_row(run['name'], r) for r in run['records']
                       if not is_canonical(r, canonical))
    return kept_runs, dropped


def raw_lines(path):
    """Non-empty raw lines of a history file (missing file yields no lines)."""
    try:
        text = Path(path).read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return []
    return [line for line in text.splitlines() if line.strip()]


def dropped_raw_lines(runs, canonical=CANONICAL_SIMILARITY_CHUNK):
    """Raw JSONL lines that may not join the curves: non-canonical or unparsable."""
    out = []
    for run in runs:
        for line in raw_lines(run['path']):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                out.append({'实验': run['name'], '原因': '无法解析的 JSONL 行', '原始行': line})
                continue
            if isinstance(record, dict) and not is_canonical(record, canonical):
                out.append({'实验': run['name'], '原因': 'similarity_chunk != %s' % canonical,
                            '原始行': line})
    return out


def curve_series_mode(n_runs, requested='auto'):
    """Series grouping that can never merge two runs into one line.

    ``'variant'`` (caption variant only) is only honoured when a single run is
    selected; with several runs the caller must fall back to ``'auto'``.
    """
    return 'variant' if (requested == 'variant' and int(n_runs) == 1) else 'auto'


def flatten_metrics(record):
    """Numeric leaves of one record, whatever its dataset shape is."""
    metrics = record.get('metrics') or {}
    blocks = [metrics] + [value for value in metrics.values() if isinstance(value, dict)]
    flat = {}
    for block in blocks:
        for key, value in block.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            flat.setdefault(str(key), float(value))
    return flat


def run_metrics(run):
    """Every metric key the run recorded, in canonical (retrieval first) order."""
    keys = set()
    for record in run['records']:
        keys.update(flatten_metrics(record))
    order = list(RETRIEVAL_KEYS) + list(DIAGNOSTIC_KEYS)
    return [key for key in order if key in keys] + sorted(keys - set(order))


def available_metrics(runs, datasets=None):
    """Metric keys present in the selected runs (optionally restricted to datasets)."""
    keys = set()
    for run in runs:
        for record in run['records']:
            if datasets and str(record.get('dataset')) not in datasets:
                continue
            keys.update(flatten_metrics(record))
    order = list(RETRIEVAL_KEYS) + list(DIAGNOSTIC_KEYS)
    return [key for key in order if key in keys] + sorted(keys - set(order))


def curve_rows(runs, metrics=None, datasets=None, variants=None, series='auto'):
    """Long-form chart rows: one row per (run, validation point, metric).

    ``series`` is ``'auto'`` (run · dataset · variant) or ``'variant'`` (caption
    variant only), which is what you want when comparing caption variants inside one
    run.
    """
    rows = []
    for run in runs:
        for record in run['records']:
            dataset = str(record.get('dataset', '?'))
            variant = str(record.get('caption_variant', '?'))
            if datasets and dataset not in datasets:
                continue
            if variants and variant not in variants:
                continue
            flat = flatten_metrics(record)
            for key in (metrics if metrics is not None else sorted(flat)):
                if key not in flat:
                    continue
                label = variant_label(variant)
                rows.append({
                    '实验': run['name'],
                    '步数': int(record['step']),
                    'epoch': int(record.get('epoch', 0)),
                    '数据集': dataset_label(dataset),
                    '文本变体': label,
                    '验证点': reason_label(record.get('reason')),
                    '指标': metric_label(key),
                    '数值': flat[key],
                    '系列': label if series == 'variant'
                            else '%s · %s · %s' % (run['name'], dataset_label(dataset), label),
                    '耗时(s)': round(float(record.get('wall_sec') or 0.0), 1),
                    'chunk': record.get('similarity_chunk'),
                    'protocol': record.get('protocol'),
                })
    rows.sort(key=lambda row: (row['实验'], row['数据集'], row['文本变体'], row['指标'], row['步数']))
    return rows


def wide_rows(rows):
    """Pivot long rows into one row per validation point (one column per metric)."""
    table, order = {}, []
    for row in rows:
        key = (row['实验'], row['步数'], row['数据集'], row['文本变体'], row['验证点'])
        if key not in table:
            table[key] = {'实验': key[0], '步数': key[1], '数据集': key[2],
                          '文本变体': key[3], '验证点': key[4], '耗时(s)': row['耗时(s)']}
            order.append(key)
        table[key][row['指标']] = row['数值']
    return [table[key] for key in order]


def latest_points(runs, datasets=None):
    """The newest validation point of every (run, dataset, variant) combination."""
    latest = {}
    for run in runs:
        for record in run['records']:
            dataset = str(record.get('dataset', '?'))
            if datasets and dataset not in datasets:
                continue
            key = (run['name'], dataset, str(record.get('caption_variant', '?')))
            current = latest.get(key)
            if current is None or int(record['step']) >= int(current['step']):
                latest[key] = record
    rows = []
    for (run_name, dataset, variant), record in sorted(latest.items()):
        flat = flatten_metrics(record)
        rows.append({
            '实验': run_name, '数据集': dataset_label(dataset), '文本变体': variant_label(variant),
            '步数': int(record['step']), 'epoch': int(record.get('epoch', 0)),
            '验证点': reason_label(record.get('reason')), '耗时(s)': round(float(record.get('wall_sec') or 0.0), 1),
            'chunk': record.get('similarity_chunk'),
            **{metric_label(key): flat[key] for key in available_metrics([{'records': [record]}])},
        })
    return rows


def non_canonical_chunks(runs, canonical=CANONICAL_SIMILARITY_CHUNK):
    """Records that did not use the canonical chunk, so mixed numbers are visible."""
    return split_canonical(runs, canonical)[1]


def total_wall_sec(runs):
    """Total validation wall time per run (monitoring only, never a training cost)."""
    return {run['name']: round(sum(float(r.get('wall_sec') or 0.0) for r in run['records']), 1)
            for run in runs}
