import ast
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_change_is_only_visual_patch_detach_route():
    source = (ROOT / 'model/nested_fusion_mask.py').read_text()
    assert 'tokens = self.visual_adapter(hidden.float())' in source
    assert 'tokens = self.visual_adapter(hidden.detach().float())' not in source
    # The text condition path remains detached.
    assert 'hidden.detach().float().permute(1, 0, 2)' in source


def test_controller_freezes_single_arm_and_stop():
    from recovery import said_e2_visual_grad500 as controller

    assert controller.ARM == 'E2-VisualGrad'
    assert controller.STEP0_SHA == '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
    tree = ast.parse(inspect.getsource(controller.run_preflight))
    forbidden = {'step', 'AdamW', 'SGD', 'zero_grad', 'save_checkpoint'}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in forbidden for node in ast.walk(tree))


def test_phase_b_outputs_are_self_describing_when_present():
    folder = ROOT / 'experiments/nest_clip_v1/said_e2_visual_grad500_v1'
    if not (folder / 'SINGLE_VARIABLE_PROOF.json').exists():
        return
    proof = json.loads((folder / 'SINGLE_VARIABLE_PROOF.json').read_text())
    assert proof['only_detach_path_diff'] and proof['forward_loss_exact']
    assert proof['patch_gradient_A_zero'] and proof['patch_gradient_B_nonzero']
