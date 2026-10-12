import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from model.component_ablation import native_clip_logits
from model.nested_fusion_mask import pair_logits


ROOT = Path(__file__).resolve().parents[1]


def test_text_only_mask_does_not_depend_on_visual_values():
    gate = nn.Linear(8, 4, bias=False)
    module = SimpleNamespace(text_only_mask=True, fusion='balanced_stack',
                             fusion_branch=SimpleNamespace(gate=gate))
    text = (torch.zeros(3, 2), torch.randn(3, 4))
    visual_a = (torch.randn(5, 2), torch.randn(5, 4))
    visual_b = (torch.randn(5, 2) * 100, torch.randn(5, 4) * 100)
    assert torch.equal(pair_logits(module, visual_a, text),
                       pair_logits(module, visual_b, text))
    assert torch.equal(pair_logits(module, visual_a, text, paired=True),
                       pair_logits(module, visual_b, text, paired=True))
    unpaired = pair_logits(module, visual_a, text)
    assert unpaired.shape == (3, 5, 4)
    assert torch.equal(unpaired[:, 0], unpaired[:, 1])


def test_ablation_modes_are_explicit_and_loss_sources_are_isolated():
    source = (ROOT / 'model/component_ablation.py').read_text()
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {'JointMaskOnly', 'NestedAlignmentOnly', 'TextOnlySmartMask'} <= names
    assert '10 * af + sf' in source
    assert '1.35 * vf + 1.35 * vo + .30 * ve' in source


def test_common_step0_and_historical_reference_are_immutable_inputs():
    plan = (ROOT / 'experiments/nest_clip_v1/said_b16_component_ablation_stop3651_v1/CONFIG_DIFF.json').read_text()
    assert '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6' in plan
    assert 'max_updates' in plan and '4868' in plan


def test_native_logits_match_independent_normalized_clip_formula():
    torch.manual_seed(0)
    image = torch.randn(4, 8)
    text = torch.randn(6, 8)
    expected = 100 * torch.nn.functional.normalize(image.float(), dim=-1) @ \
        torch.nn.functional.normalize(text.float(), dim=-1).T
    assert torch.allclose(native_clip_logits(image, text), expected, atol=1e-6, rtol=1e-6)


def test_native_logits_are_invariant_to_positive_feature_scaling():
    torch.manual_seed(1)
    image = torch.randn(4, 8)
    text = torch.randn(6, 8)
    reference = native_clip_logits(image, text)
    scaled = native_clip_logits(image * torch.tensor([.2, 3., 17., 2.]).unsqueeze(1),
                                text * torch.tensor([2., .5, 4., 9., .1, 7.]).unsqueeze(1))
    assert torch.allclose(reference, scaled, atol=1e-5, rtol=1e-5)


def test_native_ablation_source_has_no_mask_or_sparsity_objective():
    source = (ROOT / 'model/component_ablation.py').read_text()
    native = source[source.index('class NestedAlignmentOnly'):source.index('class TextOnlySmartMask')]
    assert 'mask_net.requires_grad_(False)' in native
    assert 'F_sparse=0.' in native and 'O_sparse=0.' in native and 'E_sparse=0.' in native
    assert 'hns_' not in native
