import ast
import copy
import inspect
import pytest
import torch
from torch import nn
from model.nested_fusion_mask import FusionBranch
from recovery.visual_patch_gradient_phase_a import (visual_gradient_encode,compare_tensor,
    cosine,group,layer,slice_batch,summarize_vectors,COMPONENTS,write_new)


def test_only_patch_detach_removed_forward_and_downstream_gradients_exact():
    torch.manual_seed(17)
    branch=FusionBranch(nn.Sequential(nn.Linear(4,4)),6,4,'balanced_stack',3)
    hidden=torch.randn(2,3,6,requires_grad=True)
    params=list(branch.parameters())
    a=branch.encode_visual(hidden);b=visual_gradient_encode(branch,hidden)
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a.square().sum(),[hidden,*params],allow_unused=True,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),[hidden,*params],allow_unused=True)
    assert ga[0] is None and gb[0].norm()>0
    for x,y in zip(ga[1:],gb[1:]):
        assert x is None and y is None or torch.equal(x,y)


def test_live_global_path_and_extra_regularizer_path():
    torch.manual_seed(2);backbone=nn.Linear(3,3);adapter=nn.Linear(3,3)
    x=torch.randn(5,3);h=backbone(x)
    global_loss=h.square().sum()
    s=adapter(h.detach()).square().sum();live=adapter(h).square().sum()
    base=torch.autograd.grad(global_loss,backbone.weight,retain_graph=True)[0]
    zero=torch.autograd.grad(s,backbone.weight,allow_unused=True,retain_graph=True)[0]
    new=torch.autograd.grad(global_loss+live,backbone.weight)[0]
    assert base.norm()>0 and zero is None and (new-base).norm()>0


def test_forward_difference_detection():
    a=torch.tensor([1.,2.]);assert compare_tensor(a,a)['exact']
    assert compare_tensor(a,a+.01)['max_absolute_difference']>0
    with pytest.raises(AssertionError):compare_tensor(a,torch.tensor([float('nan'),2.]))


def test_real_vector_cosines_and_additivity():
    assert cosine(torch.tensor([1.,0.]),torch.tensor([-1.,0.]))==-1
    assert cosine(torch.zeros(2),torch.ones(2)) is None
    a=dict(zip(COMPONENTS,[torch.tensor([1.,0.]),torch.zeros(2),torch.zeros(2),torch.tensor([1.,0.])]))
    b=dict(zip(COMPONENTS,[torch.tensor([1.,1.]),torch.tensor([0.,2.]),torch.tensor([0.,3.]),torch.tensor([1.,6.])]))
    r=summarize_vectors(a,b);assert r['delta_g_norm']==6
    assert r['cosine_delta_base']==0 and r['gradient_additivity']['B']['relative_L2_error']==0


@pytest.mark.parametrize('name,expected',[
    ('clip.visual.proj','native_visual_backbone'),('clip.mask_net.x','text_mask_shared_pool'),
    ('clip.transformer.x','native_text_backbone'),('fusion_branch.visual_blocks.0.x','visual_mask'),
    ('fusion_branch.visual_adapter.weight','visual_adapter'),('fusion_branch.gate.weight','fusion_gate')])
def test_parameter_groups(name,expected):assert group(name)==expected


def test_layer_and_batch_selection():
    assert layer('clip.visual.transformer.resblocks.11.attn.weight')=='visual_transformer_layer_11'
    b=dict(sample_id=torch.arange(6),image=torch.ones(6,3),views=list(range(6)),partial_detail=True)
    sliced=slice_batch(b,2);assert len(sliced['sample_id'])==len(sliced['views'])==2 and sliced['partial_detail']
    assert len(b['sample_id'])==6


def test_exclusive_audit_outputs(tmp_path):
    p=tmp_path/'audit.json';write_new(p,dict(x=1))
    with pytest.raises(FileExistsError):write_new(p,dict(x=2))


def test_probe_never_calls_update_or_optimizer_constructor():
    from recovery import visual_patch_gradient_phase_a as probe
    tree=ast.parse(inspect.getsource(probe))
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ('step','save','AdamW','SGD','build_optimizer')
    # Production implementation still contains detach; the local override does not.
    assert 'hidden.detach().float()' in inspect.getsource(FusionBranch.encode_visual)
    override=ast.parse(inspect.getsource(visual_gradient_encode))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='detach'
                   for n in ast.walk(override))


def test_original_jsonl_sampling_schema_includes_index_hash(monkeypatch):
    from recovery import visual_patch_gradient_phase_a as probe
    live=dict(prefix_segments={1:3},lowest_selected_sentence_indices_sha256='frozen')
    monkeypatch.setattr(probe,'observe_selection',lambda batch:live)
    historical=dict(prefix_segments={'1':3},lowest_selected_sentence_indices_sha256='frozen')
    assert probe.serialized_sampling({})==historical


def test_additivity_gate_preserves_declared_failed_tolerance():
    from recovery.visual_patch_gradient_phase_a import additivity_failures
    value={'native_visual_backbone':dict(gradient_additivity={'A':dict(relative_L2_error=.01539450022053093)})}
    failure=additivity_failures(value)
    assert len(failure)==1 and not failure[0]['passed'] and failure[0]['tolerance']==.01


def test_stopped_report_does_not_invent_missing_gradient_evidence(tmp_path,monkeypatch):
    import json
    from recovery import collect_visual_patch_gradient_phase_a as collector
    from train import train_nested_semantic_mask as trainer
    run=tmp_path/'runtime';run.mkdir()
    (run/'node500-validated.log').write_text('\n'.join(
        f"[rank{r}]: AssertionError: {{'relative_L2_error': 0.01539450022053093}}" for r in range(4)))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('fixture')
    digest=collector.sha(checkpoint)
    monkeypatch.setattr(collector,'NODES',{500:(checkpoint,digest),1217:(checkpoint,digest)})
    monkeypatch.setattr(trainer,'code_manifest',lambda:{'frozen':'hash'})
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{'config':{'code_sha256':{'frozen':'hash'}}})
    out=tmp_path/'report';collector.stopped_preflight(run,out)
    result=json.loads((out/'VISUAL_GRAD_COMPONENT_AUDIT.json').read_text())
    assert result['conclusion']=='INCONCLUSIVE' and result['global1024_status']=='NOT_RUN'
    assert result['component_gradient_norms'].startswith('UNVERIFIED')
    assert not result['numerical_failure']['passed']
    with pytest.raises(FileExistsError):collector.stopped_preflight(run,out)
