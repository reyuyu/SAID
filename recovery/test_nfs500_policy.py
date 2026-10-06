import json
from pathlib import Path

import pytest

from recovery.nfs500_policy import cycle_state, wait_expired, source_changes
from recovery.s02_nfs500 import canonical_path, distribution, ROOT


def test_slow_batches_warn_and_continue():
    count = 0
    for seconds in [3, 7, 15, 4, 7, 15]:
        state = cycle_state(seconds, count)
        count = state['consecutive']
        assert not state['stop']
        assert state['warning'] == (seconds > 3)


def test_five_consecutive_over30_hard_stop_and_reset():
    count = 0
    for index in range(5):
        state = cycle_state(30.01, count)
        count = state['consecutive']
        assert state['stop'] == (index == 4)
    assert cycle_state(30, 4)['consecutive'] == 0


def test_wait_only_stops_while_unreturned():
    beat = dict(state='DATA_WAIT', started_monotonic=100)
    assert not wait_expired(beat, 160)
    assert wait_expired(beat, 160.001)
    assert not wait_expired(dict(beat, state='BATCH_RETURNED'), 200)


def test_no_path_fallback_or_escape(tmp_path):
    root = tmp_path / 'canonical'
    root.mkdir()
    assert canonical_path(root, 'a/b.jpg') == root / 'a/b.jpg'
    (root / 'mirror').symlink_to(tmp_path)
    for path in ['../b.jpg', '/root/b.jpg', 'mirror/b.jpg']:
        with pytest.raises(RuntimeError):
            canonical_path(root, path)


def test_only_exact_authorized_hash_change():
    old = dict(trainer='old', model='same')
    new = dict(trainer='new', model='same')
    assert source_changes(new, old, {'trainer':('old','new')})
    with pytest.raises(RuntimeError):
        source_changes(dict(new, model='drift'), old, {'trainer':('old','new')})
    with pytest.raises(RuntimeError):
        source_changes(dict(new, trainer='unknown'), old, {'trainer':('old','new')})


def test_frozen_config_math_identical():
    old = json.loads((ROOT / 'recovery/configs/summary02.json').read_text())
    new = json.loads((ROOT / 'recovery/configs/summary02_nfs500.json').read_text())
    assert new.pop('resource_policy') == 'nfs_reproduction500'
    assert new == old


def test_percentiles_all_steps():
    assert distribution([1, 2, 3, 100]) == dict(count=4, median=2.5, p95=85.44999999999996, p99=97.08999999999997, max=100)
