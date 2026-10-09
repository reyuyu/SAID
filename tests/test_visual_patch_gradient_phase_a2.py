import ast
import inspect
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/nest_clip_v1/visual_patch_gradient_phase_a2_v1"


def test_a2_published_receipts_cover_both_nodes_and_batches():
    total = json.loads((OUT / "TOTAL_GRADIENT_A_B_COMPARISON.json").read_text())
    assert set(total["nodes"]) == {"500", "1217"}
    for node in total["nodes"].values():
        for stage in ("preflight16", "global1024_next1"):
            result = node[stage]
            assert result["A_B_forward_exact"]
            assert result["parameters_unchanged"]
            assert result["no_optimizer"] and result["no_parameter_updates"]
    assert total["nodes"]["500"]["global1024_next1"]["global_batch"] == 1024


def test_a2_route_and_resource_claims_are_explicit():
    patch = json.loads((OUT / "PATCH_BOUNDARY_GRADIENT_ATTRIBUTION.json").read_text())
    resources = json.loads((OUT / "GLOBAL1024_RESOURCE_VALIDATION.json").read_text())
    for node in ("500", "1217"):
        formal = patch["nodes"][node]["global1024_next1"]
        assert formal["A"]["patch_input_total_norm"] == 0
        assert formal["B"]["patch_input_total_norm"] > 0
        rr = resources["nodes"][node]["global1024_next1"]
        assert rr["global_batch"] == 1024 and not rr["oom"] and not rr["nan_or_inf"]
        assert rr["all_ranks_completed"] and rr["ddp_collectives_completed"]


def test_a2_nested_json_safety():
    from recovery.visual_patch_gradient_phase_a2 import json_safe

    value = json_safe({"nested": [torch.ones(2), {"x": torch.zeros(1)}]})
    assert value["nested"][0]["shape"] == [2]
    assert value["nested"][1]["x"]["finite"]
    json.dumps(value, allow_nan=False)


def test_a2_probe_has_no_optimizer_or_update_calls():
    from recovery import visual_patch_gradient_phase_a2 as probe

    tree = ast.parse(inspect.getsource(probe))
    forbidden = {"step", "AdamW", "SGD", "build_optimizer", "save_checkpoint"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
        for node in ast.walk(tree)
    )


def test_a2_report_classification_is_mechanism_only():
    report = (OUT / "VISUAL_PATCH_PHASE_A2_REPORT.md").read_text()
    assert "PHASE_B_FEASIBLE" in report
    assert "not claim an Urban or retrieval gain" in report
    assert "No training or evaluation was run" in report
