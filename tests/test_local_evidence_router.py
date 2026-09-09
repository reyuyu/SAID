"""Production local evidence: numerical compatibility and gradient contracts."""
import copy
import pytest
import torch
from model import longclip
from model.model_longclip import VisionTransformer
from eval.salu.local_evidence import extract_stages, project_candidate
from model.salu_model import SALUModel, contrastive_loss, identifiable_said_loss
import torch.nn.functional as F


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


class TinyCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=VisionTransformer(32,16,64,3,1,32)
        self.text_projection=torch.nn.Parameter(torch.eye(32))
        self.logit_scale=torch.nn.Parameter(torch.tensor(2.0))
        self.mask_net=torch.nn.Linear(32,32)

    def encode_image(self,x):return self.visual(x)
    def encode_image_with_patches(self,x,use_checkpoint=False):
        return self.visual(x,return_patches=True,use_checkpoint=use_checkpoint)
    def encode_image_with_local_evidence(self,x,use_checkpoint=False):
        return self.visual.forward_with_local_evidence(x,use_checkpoint=use_checkpoint)
    def encode_text(self,t):return t@self.text_projection


@pytest.mark.parametrize('source',['residual','attention_delta'])
def test_state_and_old_checkpoint_compatibility(source):
    old=SALUModel(TinyCLIP())
    new=SALUModel(TinyCLIP(),said_feature_source=source)
    assert old.said_feature_source=='residual'
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
    result=new.load_state_dict(old.state_dict(),strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_residual_exact_phase22_objective_and_attention():
    torch.manual_seed(25)
    model=SALUModel(TinyCLIP(),said_feature_source='residual')
    images=torch.randn(3,3,32,32);texts=torch.randn(3,32)
    g,h=model.clip.encode_image_with_patches(images)
    t=F.normalize(model.clip.encode_text(texts),dim=-1)
    expected_a,_=model.said_router(t,h)
    z_pair,_=model.said_router.route_pairwise(t,h,chunk_size=64)
    expected=identifiable_said_loss(z_pair,t,model.logit_scale.exp().clamp(max=100))
    expected_global=contrastive_loss(F.normalize(g,dim=-1),t,model.logit_scale.exp().clamp(max=100))
    capture=[]
    handle=model.said_router.register_forward_hook(lambda m,a,o:capture.append(o[0]))
    actual=model.forward_train(images,texts)
    handle.remove()
    assert torch.equal(expected_a,capture[0])
    assert torch.equal(expected_global,actual['loss_global'])
    for key in expected:assert torch.equal(expected[key],actual[key])


def test_local_and_global_gradients():
    torch.manual_seed(25)
    model=SALUModel(TinyCLIP(),said_feature_source='attention_delta')
    images=torch.randn(3,3,32,32);texts=torch.randn(3,32)
    model.forward_train(images,texts)['loss_said'].backward()
    visual=model.clip.visual
    for parameter in [model.said_router.q_proj.weight,model.said_router.k_proj.weight,
                      visual.transformer.resblocks[-1].attn.in_proj_weight,
                      visual.transformer.resblocks[0].attn.in_proj_weight,visual.conv1.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    assert visual.transformer.resblocks[-1].mlp.c_fc.weight.grad is None
    model.zero_grad(set_to_none=True)
    model.forward_train(images,texts)['loss_total'].backward()
    assert visual.transformer.resblocks[-1].mlp.c_fc.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.clip.mask_net.parameters())


def test_cli_serialization_and_seed_stream():
    import json,random,numpy as np
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('phase25_training',Path(__file__).parents[1]/'train/train_salu.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    parse_args=training.parse_args
    from salu_reproducibility import seed_everything
    args=parse_args(['--said_feature_source','attention_delta','--seed','25'])
    assert json.loads(json.dumps(vars(args)))['said_feature_source']=='attention_delta'
    assert parse_args([]).said_feature_source=='residual'
    seed_everything(25);a=(random.random(),np.random.rand(),torch.rand(1))
    seed_everything(25);b=(random.random(),np.random.rand(),torch.rand(1))
    assert a==b
    with pytest.raises(ValueError):SALUModel(TinyCLIP(),said_feature_source='invalid')


def test_evaluator_direct_router_and_ranking():
    import numpy as np
    from eval.salu.grounding_root_cause import logits_from_features
    from eval.salu.local_router_eval import patch_geometry,full_gate
    model=SALUModel(TinyCLIP(),said_feature_source='attention_delta')
    _,h=model.encode_router_input(torch.randn(1,3,32,32));t=F.normalize(torch.randn(2,32),dim=-1)
    logits=logits_from_features(h[0],t,model.said_router)
    actual=model.said_router(t,h.expand(2,-1,-1))[0]
    torch.testing.assert_close(actual,torch.softmax(logits/.07,-1),atol=1e-6,rtol=1e-6)
    direct=logits_from_features(h[0],t)
    torch.testing.assert_close(direct,t@F.normalize(h[0],dim=-1).T)
    grid=np.arange(196)[None,:].astype(float)
    same=patch_geometry(grid,grid);opposite=patch_geometry(grid,-grid)
    assert same['spearman'][0]==pytest.approx(1) and opposite['spearman'][0]==pytest.approx(-1)
    assert all(same['top%d'%k][0]==1 and opposite['top%d'%k][0]==0 for k in [1,5,10,20])
    summary={key:{k:{'mean':value} for k in ['mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor']}
             for key,value in [('attention_delta_final_router',.1),('residual_final_router',-.1)]}
    assert full_gate(summary)
    summary['attention_delta_final_router']['mass_gain']['mean']=-.01
    assert not full_gate(summary)


def test_retrieval_recall_known_matching_features():
    from eval.salu.local_router_coco import recall_metrics
    images=torch.eye(12);texts=images.repeat_interleave(5,dim=0)
    result=recall_metrics(images,texts)
    assert all(v==1 for scores in result.values() for v in scores.values())
