import copy
import math
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from model.finelip_prefix import (PrefixAggregation, scores, hinge_sums, eos_positions,
                                  legal_pairs, distributed_objective)
from train.finelip_prefix_state import group_size, lr_factor, validate_resume


def reference_score(v, t):
    rows = []
    for image in v:
        row = []
        for text in t:
            dot = image @ text.T
            activated = torch.where(dot >= 0, dot, dot * .1)
            row.append(sum(activated[p].max() for p in range(len(image))) / len(image)
                       + sum(activated[:, q].max() for q in range(len(text))) / len(text))
        rows.append(torch.stack(row))
    return torch.stack(rows)


def reference_loss(v, t, ids, tokens):
    matrix = reference_score(v, t)
    result = matrix.sum() * 0
    for i in range(len(ids)):
        for j in range(len(ids)):
            if ids[i] != ids[j] and not torch.equal(tokens[i], tokens[j]):
                result = result + torch.relu(.2 + matrix[i,j] - matrix[i,i]) + torch.relu(.2 + matrix[i,j] - matrix[j,j])
    return result


def test_aggregation_formula_scale_and_padding():
    torch.manual_seed(8)
    a = PrefixAggregation(dim=10, slots=39).double()
    x = torch.randn(2, 5, 10, dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[1,1,0,0,0], [1,1,1,1,1]], dtype=torch.bool)
    result, weights = a(x, valid)
    assert a.scale.shape == (1,1,1) and a.scale.item() == 1
    torch.testing.assert_close(weights.sum(-1), torch.ones(2,39, dtype=torch.double))
    normalized = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + a.norm.eps)
    hidden = F.gelu((normalized * a.norm.weight + a.norm.bias) @ a.network[0].weight.T + a.network[0].bias)
    logits = hidden @ a.network[2].weight.T + a.network[2].bias
    ref = []
    for b in range(2):
        ref.append(torch.stack([(logits[b,valid[b],s] * a.scale.squeeze()).softmax(0) @ x[b,valid[b]] for s in range(39)]))
    torch.testing.assert_close(result, torch.stack(ref))
    result.square().sum().backward()
    assert a.scale.grad.abs().item() > 1e-8
    assert weights[0,:,2:].count_nonzero() == 0


def test_scores_and_gradients_independent_reference():
    torch.manual_seed(4)
    v = F.normalize(torch.randn(3,4,10, dtype=torch.double), dim=-1).requires_grad_()
    t = F.normalize(torch.randn(3,5,10, dtype=torch.double), dim=-1).requires_grad_()
    actual, expected = scores(v,t), reference_score(v,t)
    torch.testing.assert_close(actual, expected)
    for a,b in zip(torch.autograd.grad(actual.sum(), (v,t), retain_graph=True), torch.autograd.grad(expected.sum(), (v,t))):
        torch.testing.assert_close(a,b)
    torch.testing.assert_close(scores(v,t[:,:4], matched=True), scores(v,t[:,:4]).diagonal())
    negative = scores(torch.tensor([[[1.,0.]]]), torch.tensor([[[-1.,0.]]]))
    torch.testing.assert_close(negative, torch.tensor([[-.2]]))


def test_active_ranking_sum_shift_and_zero():
    neg = torch.tensor([[.3]], requires_grad=True)
    row = torch.tensor([.4], requires_grad=True)
    col = torch.tensor([.45], requires_grad=True)
    valid = torch.ones(1,1,dtype=torch.bool)
    a,b,_,_ = hinge_sums(neg,row,col,valid)
    loss=a+b
    torch.testing.assert_close(loss, torch.tensor(.15))
    grads=torch.autograd.grad(loss,(neg,row,col))
    assert grads[0].item()==2 and grads[1].item()==-1 and grads[2].item()==-1
    a,b,_,_=hinge_sums(neg+3,row+3,col+3,valid)
    torch.testing.assert_close(a+b,loss)
    a,b,_,_=hinge_sums(neg,row,col,~valid)
    (a+b).backward()
    assert neg.grad.item()==0


def test_eos_content_zero_duplicates_and_tail_clock():
    ids=torch.tensor([[49406,0,2,49407,0], [49406,49407,0,0,0]])
    assert eos_positions(ids).tolist()==[3,1]
    before=torch.arange(5)[None]<eos_positions(ids)[:,None]
    assert before[0,1] and before[1].sum()==1
    image_ids=torch.tensor([4,4,5,6])
    captions=torch.tensor([[1,2],[1,3],[1,2],[1,4]])
    legal=legal_pairs(image_ids,image_ids,captions,captions)
    assert not legal[0,1] and not legal[0,2] and legal[0,3] and not legal.diagonal().any()
    assert [group_size(i,6) for i in range(6)]==[4,4,4,4,2,2]
    assert lr_factor(0,14604)==.005 and lr_factor(199,14604)==1 and lr_factor(200,14604)==1
    config=dict(batch_size=32,world_size=4,accumulation=4,seed=0,workers=8,schedule_epochs=6,
                lr_horizon_updates=12,manifest_sha256='a',init_sha256='b')
    payload=dict(objective='finelip_prefix',arm='FP0',config=config,accumulation_pending=0,
                 epoch=0,next_batch_index=6,micro_step=6,optimizer_step=2)
    assert validate_resume(payload,config,6)==(1,0)
    payload['next_batch_index']=5
    import pytest
    with pytest.raises(ValueError): validate_resume(payload,config,6)


class TinyFP0(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=PrefixAggregation(10,3)
        self.text=PrefixAggregation(10,3)
    def features(self,v,t):
        return F.normalize(self.visual(v)[0],dim=-1),F.normalize(self.text(t)[0],dim=-1)
    def forward(self,v,t,ids,tokens):
        return distributed_objective(*self.features(v,t),ids,tokens,chunk_text=2)[0]


def ddp_worker(rank, rendezvous, result):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,world_size=2,rank=rank)
    torch.manual_seed(18)
    net=TinyFP0().double()
    wrapped=torch.nn.parallel.DistributedDataParallel(net)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    torch.manual_seed(23)
    batches=[(torch.randn(4,5,10,dtype=torch.double),torch.randn(4,6,10,dtype=torch.double)) for _ in range(6)]
    import contextlib
    grads=[]
    for i,(v,t) in enumerate(batches):
        boundary=i in (3,5)
        sl=slice(rank*2,rank*2+2)
        with contextlib.nullcontext() if boundary else wrapped.no_sync():
            loss=wrapped(v[sl],t[sl],torch.arange(4)[sl],torch.arange(4)[:,None][sl])
            (loss * (i/6) / group_size(i,6)).backward()
        if boundary:
            grads.append(torch.cat([p.grad.flatten() for p in net.parameters()]))
            opt.step(); opt.zero_grad(set_to_none=True)
    if rank==0: torch.save({'model':net.state_dict(),'grads':grads},result)
    dist.destroy_process_group()


def test_two_rank_real_accumulation_updates_and_tail(tmp_path):
    mp.spawn(ddp_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'result.pt')),nprocs=2,join=True)
    actual=torch.load(tmp_path/'result.pt',weights_only=True)
    torch.manual_seed(18); net=TinyFP0().double()
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    torch.manual_seed(23)
    batches=[(torch.randn(4,5,10,dtype=torch.double),torch.randn(4,6,10,dtype=torch.double)) for _ in range(6)]
    n=0
    for i,(v,t) in enumerate(batches):
        loss=reference_loss(*net.features(v,t),torch.arange(4),torch.arange(4)[:,None])
        (loss * (i/6) / group_size(i,6)).backward()
        if i in (3,5):
            grad=torch.cat([p.grad.flatten() for p in net.parameters()])
            torch.testing.assert_close(grad,actual['grads'][n],rtol=3e-4,atol=5e-6)
            opt.step(); opt.zero_grad(set_to_none=True); n+=1
    for key,value in net.state_dict().items():
        torch.testing.assert_close(value,actual['model'][key],rtol=3e-4,atol=5e-6)


def test_strict_model_optimizer_boundary_restore(tmp_path):
    torch.manual_seed(8)
    net=TinyFP0().double()
    opt=torch.optim.AdamW(net.parameters(),lr=.002)
    v,t=torch.randn(4,5,10,dtype=torch.double),torch.randn(4,5,10,dtype=torch.double)
    def advance(model,optimizer):
        loss=reference_loss(*model.features(v,t),torch.arange(4),torch.arange(4)[:,None])
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    advance(net,opt)
    torch.save({'model':net.state_dict(),'optimizer':opt.state_dict()},tmp_path/'checkpoint.pt')
    advance(net,opt)
    restored=TinyFP0().double()
    other_opt=torch.optim.AdamW(restored.parameters(),lr=.002)
    payload=torch.load(tmp_path/'checkpoint.pt',weights_only=True)
    restored.load_state_dict(payload['model'],strict=True)
    other_opt.load_state_dict(payload['optimizer'])
    advance(restored,other_opt)
    for x,y in zip(net.parameters(),restored.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    import pytest
    payload['model'].pop('visual.scale')
    with pytest.raises(RuntimeError): restored.load_state_dict(payload['model'],strict=True)
