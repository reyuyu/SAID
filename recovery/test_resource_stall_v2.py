from unittest import mock

import pytest
import torch

from recovery import resource_stall_v2 as resource
from experiments.nest_clip_v1.armb_summary02_4epoch_v1.phase_train_v2 import TimedIterator


def test_summary_interpolates_p95():
    assert resource.summary([1, 2, 3]) == dict(count=3, median=2, p95=2.9, max=3)
    assert resource.summary([]) is None


def test_v1_reclaim_never_writes(tmp_path):
    before = dict(memory_current=1000, inactive_file=100 * resource.GIB)
    with mock.patch.object(resource, "memory_root", return_value=(tmp_path, 1)), \
            mock.patch.object(resource, "system_snapshot", return_value=before), \
            mock.patch.object(resource.os, "access") as access:
        result = resource.bounded_reclaim(before)
    assert result["attempted"] is False
    assert result["successful"] is False
    access.assert_not_called()


def test_v2_reclaim_requires_inactive_cache(tmp_path):
    target = tmp_path / "memory.reclaim"
    target.write_text("")
    before = dict(memory_current=1000, inactive_file=10 * resource.GIB)
    with mock.patch.object(resource, "memory_root", return_value=(tmp_path, 2)), \
            mock.patch.object(resource, "system_snapshot", return_value=before):
        result = resource.bounded_reclaim(before)
    assert result["attempted"] is False
    assert target.read_text() == ""


def test_bounded_v2_reclaim(tmp_path):
    target = tmp_path / "memory.reclaim"
    target.write_text("")
    before = dict(memory_current=200 * resource.GIB, inactive_file=100 * resource.GIB)
    after = dict(memory_current=136 * resource.GIB)
    with mock.patch.object(resource, "memory_root", return_value=(tmp_path, 2)), \
            mock.patch.object(resource, "system_snapshot", return_value=after), \
            mock.patch.object(resource.time, "sleep"):
        result = resource.bounded_reclaim(before)
    assert target.read_text() == str(64 * resource.GIB)
    assert result["successful"]
    assert not result["global_drop_caches"]


def test_timed_iterator_keeps_values_and_rng_order():
    class Recorder:
        def begin(self):
            self.current = {}

    recorder = Recorder()
    iterator = TimedIterator(iter(["one", "two"]), recorder)
    state = torch.get_rng_state().clone()
    assert list(iterator) == ["one", "two"]
    assert torch.equal(state, torch.get_rng_state())
    assert recorder.current is None


def test_pairs_and_pressure(tmp_path):
    path = tmp_path / "stat"
    path.write_text("file 100\nanon 200\n")
    assert resource.pairs(path) == dict(file=100, anon=200)
    path.write_text("some avg10=1.00 avg60=2.00 avg300=3.00 total=1234\n")
    assert resource.pressure(path)["some"]["total"] == 1234
    assert resource.pressure(path)["some"]["avg10"] == pytest.approx(1)
