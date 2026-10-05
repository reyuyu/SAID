import json

import pytest

import audit
import validate


def test_missing_assets_never_mark_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(audit, "RECOVERY", tmp_path)
    audit.status()
    result = json.loads((tmp_path / "evidence/readiness.json").read_text())
    assert result["status"] == "NOT_READY_TO_RESUME_RESEARCH"
    assert not result["formal_training_started"]
    assert not result["formal_training_authorized"]


def test_failed_formal_prefix_does_not_claim_no_training_started(tmp_path, monkeypatch):
    recovery = tmp_path / "recovery"
    monkeypatch.setattr(audit, "RECOVERY", recovery)
    monkeypatch.setattr(audit, "EVIDENCE", recovery / "evidence")
    full = tmp_path / "experiments/nest_clip_v1/armb_summary02_4epoch_v1"
    full.mkdir(parents=True)
    (full / "FULL_PROGRESS.json").write_text(json.dumps(dict(status="BLOCKED_PREFIX_GATE", completed_updates=5,
                                                            formal_training_started=True)))
    (full / "FIRST_FIVE_GATE.json").write_text(json.dumps(dict(passed=False)))
    audit.status()
    result = json.loads((recovery / "evidence/readiness.json").read_text())
    assert result["formal_training_started"]
    assert not result["formal_training_authorized"]
    assert not result["requirements"]["s02_full_prefix"]
    assert result["s02_full_status"] == "BLOCKED_PREFIX_GATE"


def test_smoke_refuses_missing_prerequisites(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "EVIDENCE", tmp_path)
    calls = []
    monkeypatch.setattr(validate, "execute", lambda *arguments, **kwargs: calls.append(arguments))
    with pytest.raises(RuntimeError, match="Smoke not started"):
        validate.smoke()
    assert calls == []
    result = json.loads((tmp_path / "smoke-audit.json").read_text())
    assert not result["passed"] and not result["executed"]


def test_smoke_honors_current_user_prohibition(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "EVIDENCE", tmp_path)
    (tmp_path / "recovery-operation-policy.json").write_text(json.dumps({
        "smoke_authorized": False, "reason": "Meta official shard audit only"}))
    calls = []
    monkeypatch.setattr(validate, "execute", lambda *arguments, **kwargs: calls.append(arguments))
    with pytest.raises(RuntimeError, match="current user instruction forbids smoke"):
        validate.smoke()
    assert calls == []
    result = json.loads((tmp_path / "smoke-audit.json").read_text())
    assert not result["passed"] and not result["executed"]
    assert result["blocked_by"] == ["Meta official shard audit only"]


def test_all_frozen_hashes_have_sha256_length():
    for value in [audit.ANNOTATION_SHA, audit.RECORDS_SHA, audit.CLIP_SHA, audit.STEP0_SHA]:
        assert len(value) == 64
    for protocol in audit.PROTOCOLS.values():
        assert len(protocol[4]) == 64


def test_long_dci_protocol_is_reconstructed_only():
    assert "long_dci_reconstructed" in audit.PROTOCOLS
    assert "dci_full" not in audit.PROTOCOLS
    assert audit.PROTOCOLS["long_dci_reconstructed"][2:4] == (7602, 7602)
