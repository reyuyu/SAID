"""Production local evidence: numerical compatibility and gradient contracts."""
import copy
import pytest
import torch
from model import longclip
from model.model_longclip import VisionTransformer
from eval.salu.local_evidence import extract_stages, project_candidate


@pytest.fixture(scope='module')
def clip_model():
    torch.manual_seed(25)
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    return model.float().eval()


def test_production_matches_diagnostic_and_global(clip_model):
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        global_old, residual = clip_model.encode_image_with_patches(image)
        global_new, local = clip_model.encode_image_with_local_evidence(image)
        diagnostic = project_candidate(clip_model.visual,
            extract_stages(clip_model.visual, image, (11,))[11]['attention_delta'])
        assert local.shape == (1, 196, 512)
        assert torch.equal(global_old, global_new)
        assert torch.equal(global_new, clip_model.encode_image(image))
        assert (local-diagnostic).abs().max().item() <= 1e-6
        _, residual_again = clip_model.encode_image_with_patches(image)
        assert torch.equal(residual, residual_again)


def test_no_new_state_keys_and_checkpoint_rejection(clip_model):
    from pathlib import Path
    baseline = (Path(__file__).parent/'state_dict_keys_baseline.txt').read_text().splitlines()
    assert sorted(clip_model.state_dict()) == sorted(x for x in baseline if x)
    with pytest.raises(NotImplementedError, match='use_checkpoint'):
        clip_model.encode_image_with_local_evidence(torch.randn(1,3,224,224), use_checkpoint=True)


def test_local_extraction_is_one_visual_pass():
    v = VisionTransformer(32,16,64,3,1,32)
    calls=[]
    handles=[b.attn.register_forward_hook(lambda *args: calls.append(1))
             for b in v.transformer.resblocks]
    try:
        _, h=v.forward_with_local_evidence(torch.randn(2,3,32,32))
        assert h.requires_grad and len(calls)==3
    finally:
        for handle in handles:handle.remove()
