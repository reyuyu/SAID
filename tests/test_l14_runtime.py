"""Strict backbone identity and resource-gated formal/evaluation scheduling."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.backbone import infer_base_model,validate_backbone
from model.model_longclip import CLIP,build_model
from experiments.nest_clip_v1.balanced_l14_4epoch_v1.run import BASE,L14Run
from model.balanced_hparam_search import hparams,trial_id


@pytest.mark.parametrize('patch,visual,text,native,tokens,expected',[
    (16,768,512,512,197,'ViT-B/16'),(14,1024,768,768,257,'ViT-L/14')])
def test_backbone_identity_comes_from_tensor_shapes(patch,visual,text,native,tokens,expected):
    state={'visual.conv1.weight':torch.empty(visual,3,patch,patch),
           'text_projection':torch.empty(text,native),
           'visual.positional_embedding':torch.empty(tokens,visual)}
    assert infer_base_model(state)==expected
    state['visual.positional_embedding']=torch.empty(tokens-1,visual)
    with pytest.raises(ValueError):
        infer_base_model(state)


def test_full_l14_validation_rejects_truncated_depth_and_wrong_heads():
    block=lambda heads:SimpleNamespace(attn=SimpleNamespace(num_heads=heads))
    model=SimpleNamespace(
        visual=SimpleNamespace(proj=torch.empty(1024,768),positional_embedding=torch.empty(257,1024),
                               conv1=SimpleNamespace(weight=torch.empty(1024,3,14,14)),input_resolution=224,
                               transformer=SimpleNamespace(resblocks=[block(16) for _ in range(24)])),
        text_projection=torch.empty(768,768),context_length=248,positional_embedding=torch.empty(248,768),
        positional_embedding_res=torch.empty(248,768),mask_net=SimpleNamespace(resblocks=[block(12)]),
        transformer=SimpleNamespace(resblocks=[block(12) for _ in range(12)]),
        token_embedding=SimpleNamespace(weight=torch.empty(49408,768)))
    assert validate_backbone(model,'ViT-L/14')['visual_tokens']==257
    model.visual.transformer.resblocks.pop()
    with pytest.raises(AssertionError):
        validate_backbone(model,'ViT-L/14')
    model.visual.transformer.resblocks.append(block(8))
    model.mask_net.resblocks[0].attn.num_heads=8
    with pytest.raises(AssertionError):
        validate_backbone(model,'ViT-L/14')


def test_openai_loading_allows_only_new_mask_initialization():
    torch.manual_seed(140)
    clip=CLIP(64,8,1,64,4,77,31,64,1,1,True)
    original={k:v for k,v in clip.state_dict().items() if not k.startswith('mask_net.')}
    loaded=build_model(dict(original),load_from_clip=True)
    for name,value in original.items():
        torch.testing.assert_close(loaded.state_dict()[name].float(),value.to(loaded.state_dict()[name].dtype).float(),atol=0,rtol=0)
    missing=dict(original);missing.pop('logit_scale')
    with pytest.raises(AssertionError):
        build_model(missing,load_from_clip=True)
    wrong=dict(original);wrong['visual.proj']=torch.empty(64,32)
    with pytest.raises(AssertionError):
        build_model(wrong,load_from_clip=True)


def scheduling_fixture(monkeypatch):
    runner=L14Run.__new__(L14Run)
    tid=trial_id(hparams(BASE))
    runner.state=dict(resources=[],l14_trial=tid,trials={tid:dict(hparams=hparams(BASE),budgets={})})
    monkeypatch.setattr(runner,'save',lambda:None)
    monkeypatch.setattr(runner,'publish',lambda:None)
    monkeypatch.setattr(runner,'sync',lambda message:None)
    return runner,tid


def test_failed_speed_gate_cannot_start_smoke_or_formal(monkeypatch):
    runner,_=scheduling_fixture(monkeypatch)
    resource=dict(passed=False,oom=False,config=BASE,acceptance=dict(resource_failure='measured full update exceeded3 seconds'))
    monkeypatch.setattr(runner,'probe',lambda label,cfg:resource)
    monkeypatch.setattr(runner,'train',lambda *args,**kwargs:pytest.fail('Unapproved training started'))
    runner.run()
    assert runner.state['status']=='resource_stopped'
    assert runner.state['formal_training_started'] is False


def test_passed_gate_runs_fresh_formal_and_keeps_final_separate_from_observed(monkeypatch):
    runner,tid=scheduling_fixture(monkeypatch)
    monkeypatch.setattr(runner,'probe',lambda label,cfg:dict(passed=True,config=BASE))
    trains=[];evaluations=[]
    def train(hp,stop,kind='formal'):
        trains.append((stop,kind))
        return Path('/formal-l14')
    def evaluate(tid,root,stop,training_root=None):
        evaluations.append((stop,training_root))
        score=.8 if stop==3651 else .7
        runner.state['trials'][tid]['budgets'][str(stop)]=dict(scores=dict(Score5_R1=score,J_long3=.7,J_long=.7))
    monkeypatch.setattr(runner,'train',train)
    monkeypatch.setattr(runner,'evaluate',evaluate)
    runner.run()
    assert trains==[(5,'smoke'),(4868,'formal')]
    assert evaluations==[(stop,Path('/formal-l14')) for stop in (0,500,3651,4868)]
    assert runner.state['status']=='completed'
    assert runner.state['final_updates']==4868 and runner.state['best_observed_updates']==3651
