"""Four-rank explicit full-global reference for relation scaling and fallback."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.balanced_hparam_search import BalancedSearch
from model.nested_semantic_mask import hard_st
from tests.test_nested_fusion import TinyFusionCLIP, explicit_inputs, explicit_logits
from tests.test_balanced_hparams import weighted_reference
from tests.fusion_ddp_worker import fp32_encoders
from train.train_nested_semantic_mask import build_optimizer, optimizer_learning_rates


def reference_relation(module,images,tokens,valid,completed):
    base=weighted_reference(module,images,tokens,valid,completed)
    if int(valid.sum())<2:return base,base.new_tensor(0)
    z,tp,vp,qp=explicit_inputs(module,images,tokens[1])
    _,tr,vr,qr=explicit_inputs(module,images,tokens[2])
    masks=[]
    for v,q in ((vp,qp),(vr,qr)):
        all_masks=hard_st(explicit_logits(module,v,q).sigmoid())
        masks.append(all_masks.diagonal(dim1=0,dim2=1).T)
    mp,mr=masks
    pp=(F.normalize(z*mp,dim=-1,eps=1e-6)*F.normalize(tp,dim=-1,eps=1e-6)).sum(-1)
    rp=(F.normalize(z*mr.detach(),dim=-1,eps=1e-6)*F.normalize(tp,dim=-1,eps=1e-6)).sum(-1)
    rr=(F.normalize(z*mr,dim=-1,eps=1e-6)*F.normalize(tr,dim=-1,eps=1e-6)).sum(-1)
    pr=(F.normalize(z*mp.detach(),dim=-1,eps=1e-6)*F.normalize(tr,dim=-1,eps=1e-6)).sum(-1)
    sibling=.5*(F.relu(rp-pp)+F.relu(pr-rr))
    mean=sibling[valid].mean()
    return base+min(1,completed/200)*mean,mean


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    results=[]
    cases=[('all_valid',2,[1]*8),('rank0_only_valid',2,[1,1,0,0,0,0,0,0]),
           ('global1_fallback',2,[1,0,0,0,0,0,0,0]),('global0_fallback',2,[0]*8),('tail',1,[1,1,1,1])]
    for index,(name,batch,validity) in enumerate(cases):
        torch.manual_seed(821+index)
        module=BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(fusion_lr=2e-4),
            view_relation=True,sibling_coefficient=1,fusion='balanced_stack',visual='patch',
            text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2).cuda()
        reference=copy.deepcopy(module);fp32_encoders(module);fp32_encoders(reference)
        ddp=DDP(module,device_ids=[local],output_device=local,find_unused_parameters=True,static_graph=False)
        images=torch.randn(batch*4,8,device='cuda');views=[torch.randint(0,31,(batch*4,6),device='cuda') for _ in range(3)]
        valid=torch.tensor(validity,dtype=torch.bool,device='cuda');sl=slice(rank*batch,(rank+1)*batch)
        expected,sibling=reference_relation(reference,images,views,valid,61)
        loss,logs=ddp(images[sl],*[v[sl] for v in views],valid[sl],61)
        torch.testing.assert_close(logs['loss'],expected.detach(),atol=4e-4,rtol=3e-5)
        torch.testing.assert_close(torch.as_tensor(logs['sib_loss'],device='cuda'),sibling.detach(),atol=2e-6,rtol=1e-5)
        loss.backward();expected.backward()
        errors=[]
        for pname,p in module.named_parameters():
            q=dict(reference.named_parameters())[pname];assert (p.grad is None)==(q.grad is None),pname
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=1.5e-3,rtol=2e-4,msg=pname)
                errors.append(dict(name=pname,max_abs=float((p.grad-q.grad).abs().max())))
        result=dict(case=name,global_valid=int(valid.sum()),loss_error=float((logs['loss']-expected.detach()).abs()),
                    sibling_error=float((torch.as_tensor(logs['sib_loss'],device='cuda')-sibling.detach()).abs()),
                    F_candidates=logs['F_candidates'],gradients=errors)
        results.append(result);dist.barrier(device_ids=[local]);torch.cuda.synchronize(local)
        del ddp,module,reference
    gathered=[None]*4;dist.all_gather_object(gathered,dict(rank=rank,cases=results))
    if rank==0:Path(args.output).write_text(json.dumps(dict(passed=True,world_size=4,ranks=gathered),indent=2)+'\n')
    dist.barrier(device_ids=[local]);torch.cuda.synchronize(local);dist.destroy_process_group()


if __name__=='__main__':main()
