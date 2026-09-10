"""Phase 2.7C tests: validation score curves in the dashboard (read-only JSONL)."""
import json
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.validation_protocol import (  # noqa: E402
    BALANCING_GAIN_DEFINITION as CANONICAL_DEFINITION,
    BALANCING_GAIN_SIGN as CANONICAL_SIGN,
)
from tools.said_dashboard import validation_curves_data as curves  # noqa: E402


def sharegpt4v1k_record(step, variant, reason='interval', chunk=512, r1=0.63, gain=0.008,
                        full_gap=0.6755, said_gap=0.6675):
    return {
        'step': step, 'epoch': 0, 'dataset': 'sharegpt4v1k', 'caption_variant': variant,
        'reason': reason, 'protocol': 'sharegpt4v1k-fixed-captions-v1',
        'similarity_chunk': chunk, 'wall_sec': 17.7,
        'metrics': {'retrieval': {'image2text_R1': r1, 'image2text_R5': 0.864,
                                  'image2text_R10': 0.934, 'text2image_R1': 0.610,
                                  'text2image_R5': 0.836, 'text2image_R10': 0.912},
                    'diagnostics': {'full_pair_gap': full_gap, 'said_pair_gap': said_gap,
                                    'full_rmg': 0.5418, 'said_rmg': 0.5720,
                                    'balancing_gain': gain, 'relative_balancing_gain': 0.0118,
                                    'conditioning_margin': 0.0988,
                                    'balancing_gain_definition': 'full_pair_gap - said_pair_gap',
                                    'balancing_gain_sign': 'positive = Said better; negative = Said worse'}},
    }


def coco_record(step, reason):
    return {'step': step, 'epoch': 0, 'dataset': 'coco_val2017', 'caption_variant': 'coco_5captions',
            'reason': reason, 'protocol': 'coco-val2017-5captions-v1', 'similarity_chunk': 512,
            'wall_sec': 103.9,
            'metrics': {'image2text_R1': 0.517, 'image2text_R5': 0.7662, 'image2text_R10': 0.8428,
                        'text2image_R1': 0.3269, 'text2image_R5': 0.5776, 'text2image_R10': 0.6823}}


def write_run(root, name, records, trailing_garbage=False):
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(json.dumps(record, ensure_ascii=False) for record in records) + '\n'
    if trailing_garbage:
        text += '\n{"step": 99, "metrics": {"image2text_R1": 0.5}\n'   # killed mid-append
    (run_dir / curves.HISTORY_NAME).write_text(text, encoding='utf-8')
    return run_dir


def fixture_runs(tmp_path):
    write_run(tmp_path, 'run_a', [sharegpt4v1k_record(0, 'first_sentence', 'initial'),
                                  sharegpt4v1k_record(10, 'first_sentence'),
                                  sharegpt4v1k_record(20, 'first_sentence', 'final', r1=0.70),
                                  coco_record(0, 'initial'), coco_record(20, 'final')])
    write_run(tmp_path, 'run_b', [sharegpt4v1k_record(10, 'full_dense', gain=-0.0449)],
              trailing_garbage=True)
    (tmp_path / 'empty_run').mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def test_discover_and_load_runs_skips_dirs_without_history(tmp_path):
    fixture_runs(tmp_path)
    runs = curves.load_runs(tmp_path)
    assert [run['name'] for run in runs] == ['run_a', 'run_b']
    assert curves.discover_runs(tmp_path / 'missing') == []
    single = curves.load_runs(tmp_path / 'run_a')
    assert [run['name'] for run in single] == ['run_a']


def test_read_history_is_tolerant_but_missing_file_raises(tmp_path):
    run_dir = fixture_runs(tmp_path) / 'run_b'
    records = curves.read_history(run_dir / curves.HISTORY_NAME)
    assert [record['step'] for record in records] == [10]        # malformed tail skipped
    with pytest.raises(ValueError):
        curves.read_history(run_dir / 'absent.jsonl')


def test_flatten_metrics_handles_both_dataset_shapes():
    nested = curves.flatten_metrics(sharegpt4v1k_record(10, 'full_dense'))
    flat = curves.flatten_metrics(coco_record(0, 'initial'))
    assert nested['image2text_R1'] == 0.63 and nested['balancing_gain'] == 0.008
    assert flat['image2text_R1'] == 0.517 and 'balancing_gain' not in flat
    assert all(isinstance(value, float) for value in list(nested.values()) + list(flat.values()))


def test_curve_rows_filters_labels_and_series(tmp_path):
    runs = curves.load_runs(fixture_runs(tmp_path))
    rows = curves.curve_rows(runs, metrics=['image2text_R1', 'balancing_gain'],
                             datasets={'sharegpt4v1k'}, variants={'first_sentence'})
    assert {row['指标'] for row in rows} == {'I2T R@1 ↑', 'Balancing Gain（Full − Said）↑'}
    assert {row['步数'] for row in rows} == {0, 10, 20}
    assert {row['验证点'] for row in rows} == {'初始 step 0', '间隔', '训练结束'}
    assert all(row['系列'] == 'run_a · ShareGPT4V-1K（1,000 路） · 第一句' for row in rows)
    per_variant = curves.curve_rows(runs, metrics=['image2text_R1'],
                                    datasets={'sharegpt4v1k'}, series='variant')
    assert {row['系列'] for row in per_variant} == {'第一句', '完整 caption'}
    # COCO records only expose retrieval metrics
    coco_rows = curves.curve_rows(runs, metrics=['balancing_gain'], datasets={'coco_val2017'})
    assert coco_rows == []


def test_wide_rows_and_latest_points(tmp_path):
    runs = curves.load_runs(fixture_runs(tmp_path))
    rows = curves.curve_rows(runs, metrics=['image2text_R1', 'balancing_gain'],
                             datasets={'sharegpt4v1k'}, variants={'first_sentence'})
    wide = curves.wide_rows(rows)
    assert len(wide) == 3
    assert wide[2]['I2T R@1 ↑'] == 0.70 and wide[2]['Balancing Gain（Full − Said）↑'] == 0.008
    latest = curves.latest_points(runs, datasets={'sharegpt4v1k'})
    assert [row['步数'] for row in latest if row['实验'] == 'run_a'] == [20]
    assert any(row['实验'] == 'run_b' for row in latest)


def test_non_canonical_chunk_is_flagged(tmp_path):
    write_run(tmp_path, 'odd', [sharegpt4v1k_record(0, 'full_dense', chunk=1024)])
    runs = curves.load_runs(tmp_path)
    assert curves.non_canonical_chunks(runs) == [
        {'实验': 'odd', '步数': 0, '数据集': 'ShareGPT4V-1K（1,000 路）', 'similarity_chunk': 1024}]
    assert curves.non_canonical_chunks(curves.load_runs(fixture_runs(tmp_path / 'ok'))) == []


def test_balancing_gain_definition_matches_the_canonical_one():
    """The dashboard re-states the definition; it must not drift from the protocol."""
    assert curves.BALANCING_GAIN_DEFINITION == CANONICAL_DEFINITION == 'full_pair_gap - said_pair_gap'
    assert curves.BALANCING_GAIN_SIGN == CANONICAL_SIGN
    assert curves.CANONICAL_SIMILARITY_CHUNK == 512
    assert curves.metric_label('balancing_gain').startswith('Balancing Gain（Full − Said）')


# --------------------------------------------------------------------------- #
# page rendering (executed with streamlit's AppTest, not a source check)
# --------------------------------------------------------------------------- #
PAGE = 'from tools.said_dashboard.validation_curves_page import main\nmain()'


def test_page_renders_curves_for_a_fixture_run(tmp_path, monkeypatch):
    fixture_runs(tmp_path)
    monkeypatch.setenv('SAID_RUNS_ROOT', str(tmp_path))
    app = AppTest.from_string(PAGE).run()
    assert not app.exception
    assert app.title[0].value == '验证得分曲线'
    assert any('共 3 个' in metric.value for metric in app.metric)
    assert len(app.dataframe) >= 2                      # 最新验证点 + 数值表
    assert app.multiselect[0].value == ['run_a', 'run_b']


def test_page_warns_on_non_canonical_chunk(tmp_path, monkeypatch):
    write_run(tmp_path, 'odd', [sharegpt4v1k_record(7, 'full_dense', chunk=256)])
    monkeypatch.setenv('SAID_RUNS_ROOT', str(tmp_path))
    app = AppTest.from_string(PAGE).run()
    assert not app.exception
    assert any('canonical similarity_chunk' in warning.value for warning in app.warning)


def test_page_explains_how_to_produce_records_when_there_are_none(tmp_path, monkeypatch):
    monkeypatch.setenv('SAID_RUNS_ROOT', str(tmp_path / 'nothing_here'))
    app = AppTest.from_string(PAGE).run()
    assert not app.exception
    assert '未找到验证记录' in app.info[0].value
    assert '--val_sharegpt4v' in app.code[0].value


def test_app_navigation_exposes_the_curves_page():
    source = open(os.path.join(REPO_ROOT, 'tools', 'said_dashboard', 'app.py'), encoding='utf-8').read()
    assert "'验证得分曲线'" in source
    assert 'validation_curves_page' in source
