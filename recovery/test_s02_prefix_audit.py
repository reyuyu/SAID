import copy

import pytest

from s02_prefix_audit import compare_rows
from test_s02_full_gate import prefix


def test_numeric_difference_is_recorded_not_passed():
    baseline = prefix()
    actual = copy.deepcopy(baseline)
    actual[3]["loss"] += .005
    result = compare_rows(actual, baseline, baseline)
    assert result[3]["full_abs_difference"] > 0
    assert result[3]["exact_FSD_strings_and_tokens"]


def test_string_or_token_difference_is_not_numeric_noise():
    baseline = prefix()
    actual = copy.deepcopy(baseline)
    actual[0]["rank_health"][0]["stream_sha256"] = "different"
    with pytest.raises(RuntimeError, match="Input/LR drift"):
        compare_rows(actual, baseline, baseline)


def test_six_update_trace_cannot_be_prefix_only():
    baseline = prefix()
    with pytest.raises(RuntimeError, match="exactly five"):
        compare_rows(baseline + [dict(step=6)], baseline, baseline)
