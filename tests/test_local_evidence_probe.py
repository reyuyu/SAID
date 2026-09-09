import torch
import numpy as np
import pytest
from clip.model import CLIP

from eval.salu.local_evidence import (STAGES, decompose_block, extract_stages,
                                      project_candidate)


def model():
    return CLIP(embed_dim=32, image_resolution=224, vision_layers=12,
                vision_width=64, vision_patch_size=16, context_length=77,
                vocab_size=100, transformer_width=64, transformer_heads=1,
                transformer_layers=1).float().eval()


def test_block_decomposition_reconstructs_exact_forward():
    m=model(); block=m.visual.transformer.resblocks[0]; x=torch.randn(197,2,64)
    states=decompose_block(block,x)
    torch.testing.assert_close(states['after_attention'],states['x_in']+states['attention_delta'])
    torch.testing.assert_close(states['residual'],states['after_attention']+states['mlp_delta'])
    torch.testing.assert_close(states['residual'],block(x),atol=1e-6,rtol=1e-6)


def test_candidate_shapes_order_and_final_projection():
    m=model(); image=torch.randn(2,3,224,224)
    stages=extract_stages(m.visual,image)
    assert set(stages)=={3,6,9,11}
    for states in stages.values():
        assert set(states)==set(STAGES)
        for value in states.values(): assert value.shape==(2,196,64)
    projected=project_candidate(m.visual,stages[11]['residual'])
    assert projected.shape==(2,196,32)


def test_candidate_extractor_does_not_add_permutation():
    m=model(); image=torch.randn(1,3,224,224)
    stages=extract_stages(m.visual,image)
    final=m.visual.ln_post(stages[11]['residual'])@m.visual.proj
    from eval.salu.semantic_grounding_eval import native_clip_patches
    native=native_clip_patches(m,image)
    torch.testing.assert_close(final,native,atol=1e-5,rtol=1e-5)


def test_attention_delta_and_normalized_projection():
    m=model(); image=torch.randn(1,3,224,224)
    h=project_candidate(m.visual,extract_stages(m.visual,image,layers=(3,))[3]['attention_delta'])
    text=torch.randn(1,32); logits=torch.nn.functional.normalize(text,dim=-1)@torch.nn.functional.normalize(h[0],dim=-1).T
    attention=torch.softmax(logits/.07,-1)
    assert attention.shape==(1,196); torch.testing.assert_close(attention.sum(-1),torch.ones(1))


def test_longclip_patch_interface_exact_and_real_shapes():
    from model.model_longclip import VisionTransformer
    v=VisionTransformer(224,16,768,1,12,512).float().eval()
    image=torch.randn(1,3,224,224)
    with torch.inference_mode():
        states=extract_stages(v,image,(0,),include_input=True)[0]
        assert set(states)==set(STAGES)|{'x_in'}
        assert states['residual'].shape==(1,196,768)
        actual=project_candidate(v,states['residual'])
        assert actual.shape==(1,196,512)
        torch.testing.assert_close(actual,v(image,return_patches=True)[1],atol=1e-6,rtol=1e-6)


def test_orientation_and_pair_metrics_have_matching_denominators():
    from eval.salu.local_evidence_metrics import geometry,evaluate,orientation_margin
    phrases=[{'id':str(i),'entity_id':str(i),'phrase':str(i),'boxes':[box]}
             for i,box in enumerate([[0,0,16,16],[208,208,224,224]])]
    g=geometry([{'id':'i','entities':{str(i):p['boxes'] for i,p in enumerate(phrases)},'phrases':phrases}])
    logits=np.zeros((2,196));logits[0,0]=1;logits[1,-1]=1
    s,rows,att=evaluate(logits,g)
    assert s['pair_query_count']==2 and s['pairs']==1
    assert s['pointing']['mean']==1 and s['semantic_pointing_excess']['mean']==1
    assert s['semantic_mass_excess']['mean']==pytest.approx(s['switch_margin']['mean'])
    assert s['orientation_semantic_gap']>0
    np.testing.assert_array_equal(orientation_margin(logits,g['coverage']),[1,1])
    np.testing.assert_allclose(att.sum(1),1)


def test_rank_correlation_handles_ties():
    from eval.salu.local_evidence_metrics import correlation
    assert correlation([1,1,2],[2,2,1],True)==pytest.approx(-1)
    assert correlation([1,1],[2,3],True) is None
