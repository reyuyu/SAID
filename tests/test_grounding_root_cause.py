import numpy as np
import torch

from eval.salu.grounding_root_cause import (captured_patches, choose_phrases,
                                          corr, logits_from_features, patch_order_probe)


def tiny_clip():
    from clip.model import CLIP
    return CLIP(embed_dim=32,image_resolution=224,vision_layers=2,
                vision_width=64,vision_patch_size=16,context_length=77,
                vocab_size=100,transformer_width=64,transformer_heads=1,
                transformer_layers=1).float().eval()


def test_all_patch_slots_and_detected_permutation():
    model=tiny_clip()
    with torch.inference_mode():
        assert patch_order_probe(model,'cpu')['pass']
        original=model.visual.transformer.forward
        model.visual.transformer.forward=lambda x: original(x).flip(0)
        result=patch_order_probe(model,'cpu')
        assert not result['pass'] and result['sequence_max_diff']>1e-3


def test_layer_hooks_removed_and_last_projection_matches_export():
    from eval.salu.semantic_grounding_eval import native_clip_patches
    model=tiny_clip();image=torch.randn(1,3,224,224)
    with torch.inference_mode():
        output=captured_patches(model,image,(0,1))
        torch.testing.assert_close(output[1],native_clip_patches(model,image))
    assert set(output)=={0,1}
    assert all(not block._forward_hooks for block in model.visual.transformer.resblocks)


def test_deterministic_selection_distinct_images_and_categories():
    phrases={str(i):{'image_id':str(i//2),'categories':[str(i%8)]} for i in range(250)}
    chosen=choose_phrases(phrases)
    assert chosen==choose_phrases(phrases)
    assert len(chosen)==len({phrases[k]['image_id'] for k in chosen})==100
    assert {c for k in chosen for c in phrases[k]['categories']}=={str(i) for i in range(8)}


def test_logit_rotation_correlation_equivalence():
    rng=np.random.default_rng(42)
    a,b=rng.normal(size=(2,14,14))
    assert np.isclose(corr(np.rot90(a,2),b),corr(a,np.rot90(b,2)))
    assert corr(a,np.ones_like(a)) is None


def test_independent_router_logits_match_original():
    from model.salu_modules import SaidRouter
    model=SaidRouter(dim=32).eval()
    t=torch.randn(4,32);h=torch.randn(196,32)
    with torch.inference_mode():
        actual=logits_from_features(h,t,model).div(.07).softmax(-1)
        expected=model(t,h[None].expand(4,-1,-1))[0]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
