import json

import pytest

import sa1b_recovery as recovery
import sa1b_throughput_benchmark as benchmark


def test_stage_shards_are_disjoint_and_never_completed_or_quarantined(tmp_path):
    rows = [dict(filename=f"sa_{position:06d}.tar", archive=str(tmp_path / f"sa_{position:06d}.tar"),
                 download_status="pending") for position in range(51)]
    rows[0]["download_status"] = "verified"
    rows[1]["extraction_status"] = "complete"
    rows[16]["download_status"] = recovery.QUARANTINED
    rows[17]["download_status"] = recovery.QUARANTINED
    (tmp_path / "sa_000002.tar").write_bytes(b"existing")
    stages = benchmark.select_stages(rows)
    names = [row["filename"] for concurrency, group in stages for row in group]
    assert len(names) == len(set(names)) == 20
    assert not {f"sa_{position:06d}.tar" for position in (0, 1, 2, 14, 16, 17)}.intersection(names)
    assert [len(group) for concurrency, group in stages] == [2, 4, 6, 8]


def test_select_lowest_stable_concurrency_near_highest_rate():
    stages = [dict(concurrency=2, stable=True, total_mib_s=5),
              dict(concurrency=4, stable=False, total_mib_s=100),
              dict(concurrency=6, stable=True, total_mib_s=19.5),
              dict(concurrency=8, stable=True, total_mib_s=20)]
    assert benchmark.choose_best(stages)[0] == 6
    assert benchmark.choose_best([])[0] == 2


def test_partial_byte_counter_does_not_sum_completed_and_partial(tmp_path):
    archive = tmp_path / "sa_000018.tar"
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    partial = cache / "sa_000018.tar.digest.incomplete"
    partial.write_bytes(b"12345")
    row = dict(archive=str(archive), filename=archive.name)
    assert benchmark.payload_bytes(row) == 5
    archive.write_bytes(b"123456789")
    assert benchmark.payload_bytes(row) == 9
    assert partial.read_bytes() == b"12345"


def test_incomplete_event_log_is_tolerated(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(dict(kind="metadata")) + '\n{"kind":')
    assert benchmark.events(path) == [dict(kind="metadata")]


def test_sdk_hash_encoded_partial_filename_is_counted(tmp_path):
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    partial = cache / "encoded-name=.digest.incomplete"
    partial.write_bytes(b"12345")
    row = dict(archive=str(tmp_path / "sa_000018.tar"), filename="sa_000018.tar", expected_lfs_sha256="digest")
    assert benchmark.payload_bytes(row) == 5


def test_recovery_rejects_unsafe_concurrency():
    with pytest.raises(ValueError, match="concurrency"):
        recovery.run(downloads_only=True, concurrency=0)


def test_tiers_cannot_be_shortened():
    with pytest.raises(ValueError, match="300"):
        benchmark.benchmark(12345, 299)


def test_retiring_paused_supervisor_releases_locks_without_touching_files(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *args, **kwargs:
        "100 Ts bash recovery/continue_sa1b_recovery.sh\n101 Tl python sa1b_recovery.py run --downloads-only\n")
    monkeypatch.setattr(benchmark.os, "killpg", lambda *args: calls.append(args))
    monkeypatch.setattr(benchmark.time, "sleep", lambda *args: None)
    result = benchmark.retire_paused_supervisor(100)
    assert result["partials_preserved"]
    assert calls == [(100, benchmark.signal.SIGKILL)]


def test_retirement_rejects_supervisor_not_fully_stopped(monkeypatch):
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *args, **kwargs:
        "100 Ts bash recovery/continue_sa1b_recovery.sh\n101 Rl python sa1b_recovery.py run --downloads-only\n")
    with pytest.raises(RuntimeError, match="fully stopped"):
        benchmark.retire_paused_supervisor(100)
