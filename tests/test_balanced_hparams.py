"""Coefficient isolation, exact B0 recovery and checkpoint/optimizer migration contracts."""
import copy

import pytest
import torch
from torch.nn import functional as F

from model.balanced_hparam_search import BalancedSearch,hparams,migrate_legacy_optimizer,trial_id
from model.nested_fusion_mask import NestedFusionMask
from model.nested_semantic_mask import hard_st,inclusion
from tests.test_nested_fusion import TinyFusionCLIP,explicit_inputs,explicit_logits
from train.train_nested_semantic_mask import build_optimizer,optimizer_learning_rates,validate_resume_payload
from tests.test_nested_resume import payloads


def make(config=None):
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=config,fusion='balanced_stack',visual='patch',
                           text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


def inputs():
    return torch.randn(3,8),[torch.randint(0,31,(3,6)) for _ in range(3)],torch.tensor([1,0,1],dtype=torch.bool)


def weighted_reference(module,images,views,valid,completed):
    values,positives=[],[]
    for index,tokens in enumerate(views if int(valid.sum())>=2 else views[:1]):
        z,t,zv,zt=explicit_inputs(module,images,tokens)
        probability=explicit_logits(module,zv,zt).sigmoid();mask=hard_st(probability)
        scores=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*F.normalize(t,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if index==0 else valid
        labels=torch.arange(len(z),device=z.device)[enabled]
        ce=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)+F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        values.append((ce,mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean()))
        positives.append(probability.diagonal(dim1=0,dim2=1).T)
    hp=module.search_hparams
    if int(valid.sum())<2:return 10*values[0][0]+hp['sparsity_scale']*values[0][1]
    weights=hp['view_weights']
    return (10/sum(weights)*sum(w*pair[0] for w,pair in zip(weights,values))+
            hp['sparsity_scale']*(values[0][1]+2*values[1][1]+2*values[2][1])/3+
            hp['inclusion_max']*min(1,completed/200)*inclusion(*positives)[valid].mean())


def test_default_forward_gradients_architecture_identical():
    torch.manual_seed(907)
    clip=TinyFusionCLIP(32)
    old=NestedFusionMask(copy.deepcopy(clip),fusion='balanced_stack',visual='patch',text_tokens=6,
                         checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    new=BalancedSearch(copy.deepcopy(clip),fusion='balanced_stack',visual='patch',text_tokens=6,
                       checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    assert old.state_dict().keys()==new.state_dict().keys()
    images,views,valid=inputs()
    lo,_=old(images,*views,valid,61);ln,_=new(images,*views,valid,61)
    torch.testing.assert_close(lo,ln,atol=0,rtol=0)
    lo.backward();ln.backward()
    for name,p in old.named_parameters():
        q=dict(new.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    build_optimizer(old).step();build_optimizer(new).step()
    for name,p in old.named_parameters():torch.testing.assert_close(p,dict(new.named_parameters())[name],atol=0,rtol=0,msg=name)


def test_actual_four_groups_and_lr_effect_is_isolated():
    model=make(dict(fusion_lr=2e-4,visual_mask_lr_scale=.5))
    opt=build_optimizer(model)
    assert [g['name'] for g in opt.param_groups]==['backbone','text_mask_and_shared_pool','visual_mask','fusion_adapter']
    ids=[[id(p) for p in g['params']] for g in opt.param_groups]
    assert len([i for group in ids for i in group])==len(set(i for group in ids for i in group))
    assert id(model.fusion_branch.gate.weight) in ids[3]
    assert id(model.fusion_branch.visual_adapter.weight) in ids[3]
    assert id(model.clip.mask_net.attn_pool.attention.weight) in ids[1]
    rates=optimizer_learning_rates(model,500,3651)
    factor=.5*(1+__import__('math').cos(__import__('math').pi*500/3651))
    assert rates[1]==1e-3*factor and rates[2]==5e-4*factor and rates[3]==2e-4*factor


@pytest.mark.parametrize('field,value',[('sparsity_scale',.75),('inclusion_max',1.5),('view_weights',[2,1,1]),
                                      ('inclusion_max',2.),('view_weights',[1,1,2])])
def test_each_loss_coefficient_only_changes_its_term(field,value):
    torch.manual_seed(911)
    base=make();variant=copy.deepcopy(base);variant.search_hparams=hparams({field:value})
    images,views,valid=inputs()
    lb,logs=base(images,*views,valid,61);lv,_=variant(images,*views,valid,61)
    if field=='sparsity_scale':delta=(value-1)*(logs['F_sparse']+2*logs['O_sparse']+2*logs['E_sparse'])/3
    elif field=='inclusion_max':delta=(value-1)*logs['inc_weight']*logs['inc']
    else:
        terms=[logs[p+'_i2t']+logs[p+'_t2i'] for p in ['F','O','E']]
        delta=10/sum(value)*sum(w*x for w,x in zip(value,terms))-10/3*sum(terms)
    torch.testing.assert_close(lv-lb,delta,atol=3e-4,rtol=3e-5)
    torch.testing.assert_close(lv,weighted_reference(variant,images,views,valid,61),atol=3e-4,rtol=3e-5)
    if field=='sparsity_scale':
        valid.zero_();lb,logs=base(images,*views,valid,61);lv,_=variant(images,*views,valid,61)
        torch.testing.assert_close(lv-lb,(value-1)*logs['F_sparse'],atol=3e-4,rtol=3e-5)


@pytest.mark.parametrize('field,value',[('fusion_lr',2e-4),('visual_mask_lr_scale',.5),
                                      ('view_weights',[2,1,1]),('sparsity_scale',.75),('inclusion_max',1.5)])
def test_resume_checks_all_hparams(field,value):
    previous,current=payloads()
    previous['config'].update(hparam_search=True,trial_id=trial_id({}),**hparams({}))
    current.update(hparam_search=True,trial_id=trial_id({}),**hparams({}))
    current[field]=value
    with pytest.raises(AssertionError,match='hyperparameters'):validate_resume_payload(previous,current,'old-trainer')


def test_legacy_b0_adamw_state_split_is_exact_and_next_update_matches():
    torch.manual_seed(919)
    old=NestedFusionMask(TinyFusionCLIP(32),fusion='balanced_stack',visual='patch',text_tokens=6,
                         checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    old_opt=build_optimizer(old)
    images,views,valid=inputs()
    old(images,*views,valid,61)[0].backward();old_opt.step();old_opt.zero_grad(set_to_none=True)
    new=make();new.load_state_dict(old.state_dict(),strict=True)
    new_opt=build_optimizer(new)
    migrated=migrate_legacy_optimizer(old_opt.state_dict(),new,new_opt)
    assert migrated['steps']==[1]
    old(images,*views,valid,62)[0].backward();new(images,*views,valid,62)[0].backward()
    old_opt.step();new_opt.step()
    for name,p in old.named_parameters():torch.testing.assert_close(p,dict(new.named_parameters())[name],atol=0,rtol=0,msg=name)


def test_epoch_start_vs_boundary_loader_generator_state():
    generator=torch.Generator().manual_seed(0)
    epoch_start=generator.get_state()
    first_seed=int(torch.empty((),dtype=torch.int64).random_(generator=generator))
    boundary_state=generator.get_state()
    second_seed=int(torch.empty((),dtype=torch.int64).random_(generator=generator))
    middle_resume=torch.Generator();middle_resume.set_state(epoch_start)
    assert int(torch.empty((),dtype=torch.int64).random_(generator=middle_resume))==first_seed
    boundary_resume=torch.Generator();boundary_resume.set_state(boundary_state)
    assert int(torch.empty((),dtype=torch.int64).random_(generator=boundary_resume))==second_seed


def test_trial_identity_canonical_and_rank_uses_raw_scores():
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import scores,rank_key
    assert trial_id({})==trial_id(dict(view_weights=[1,1,1],fusion_lr=1e-4))
    assert trial_id({})!=trial_id(dict(fusion_lr=5e-5))
    lower={'scores':dict(Score5_R1=.72000001,J_long3=.99,J_long=.99)}
    higher={'scores':dict(Score5_R1=.72000002,J_long3=.1,J_long=.1)}
    assert rank_key(higher)>rank_key(lower)
