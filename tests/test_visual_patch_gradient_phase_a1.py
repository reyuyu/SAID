import json
import ast
import inspect
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_published_a1_receipts_are_complete_and_classified():
    folder=ROOT/'experiments/nest_clip_v1/visual_patch_gradient_phase_a1_v1'
    validation=json.loads((folder/'PHASE_A1_VALIDATION.json').read_text())
    diagnostic=json.loads((folder/'BF16_FP32_DIAGNOSTIC.json').read_text())
    assert validation['classification']=='PROBE_IMPLEMENTATION_ISSUE'
    assert validation['global1024_not_run'] and validation['node1217_not_run']
    assert diagnostic['precision_results']['bf16']['B']['native_backbone_separate_component_additivity']['pass'] is False
    assert diagnostic['precision_results']['fp32']['B']['native_backbone_separate_component_additivity']['pass'] is False
    assert diagnostic['precision_results']['fp32']['B']['native_backbone_one_pass_combined_control']['combined_vs_total_relative_L2_error']==0


def test_published_a1_keeps_a_b_route_evidence():
    folder=ROOT/'experiments/nest_clip_v1/visual_patch_gradient_phase_a1_v1'
    receipt=json.loads((folder/'VARIANT_A_B_GRADIENT_RECEIPTS.json').read_text())
    for precision in ('bf16','fp32'):
        for variant in ('A','B'):
            route=receipt['variants'][precision][variant]['patch_input_gradient_norms']
            if variant=='A':assert all(all(v==0 for v in values) for values in route.values())
            else:assert any(any(v>0 for v in values) for values in route.values())


def test_root_cause_report_has_no_training_claim():
    report=(ROOT/'experiments/nest_clip_v1/visual_patch_gradient_phase_a1_v1/ADDITIVITY_FAILURE_ROOT_CAUSE.md').read_text()
    assert 'PROBE_IMPLEMENTATION_ISSUE' in report
    assert 'No Phase B training is authorized' in report


def test_a1_probe_has_no_optimizer_or_update_calls():
    from recovery import visual_patch_gradient_phase_a1 as probe
    tree=ast.parse(inspect.getsource(probe))
    forbidden={'step','AdamW','SGD','build_optimizer','save_checkpoint'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden
                   for n in ast.walk(tree))
