import json

import pytest

import sa1b_modelscope_repair as repair


def good_audit():
    rows = [dict(image_id=position, classification="BITWISE_EQUIVALENT", passed=True,
        id_basename_equal=True, dimensions_equal=True, rgb_pixels_exact_equal=True,
        preprocess_tensor_exact_equal=True, source_repository=repair.source.REPO,
        source_revision=repair.source.REVISION) for position in range(200)]
    return dict(passed=True, compared=200, samples=rows)


def test_repair_requires_complete_exact_source_equivalence():
    audit = good_audit()
    repair.require_equivalence(audit)
    audit["samples"][-1]["preprocess_tensor_exact_equal"] = False
    with pytest.raises(ValueError, match="failed or unrelated"):
        repair.require_equivalence(audit)


def test_duplicate_references_and_fewer_than200_never_open_repair_gate():
    audit = good_audit()
    audit["samples"][-1]["image_id"] = 0
    with pytest.raises(ValueError, match="200 distinct"):
        repair.require_equivalence(audit)
    audit["compared"] = 199
    audit["samples"] = audit["samples"][:199]
    with pytest.raises(ValueError, match="200 distinct"):
        repair.require_equivalence(audit)


def test_quarantined_source_404_does_not_trigger_another_download(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "CACHE", tmp_path)
    (tmp_path / "sa_195618.jpg.json").write_text(json.dumps(dict(image_id=195618,
        http_status=404, installed=False)))
    calls = []
    monkeypatch.setattr(repair, "urlopen", lambda *arguments, **kwargs: calls.append(arguments))
    result = repair.fetch_and_install(195618, None, "gate")
    assert not result["installed"] and result["http_status"] == 404
    assert not calls


def test_repair_plan_cannot_expand_beyond_frozen_missing_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "FOLDER", tmp_path)
    (tmp_path / "initial-missing-required-ids.json").write_text(json.dumps(dict(
        training_index_sha256=repair.recovery.RECORDS_SHA,
        shards={f"sa_{position:06d}.tar": [] for position in repair.SHARDS})))
    with pytest.raises(ValueError, match="exactly11174"):
        repair.required_plan()
