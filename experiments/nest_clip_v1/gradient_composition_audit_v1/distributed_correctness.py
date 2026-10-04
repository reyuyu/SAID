"""CPU/gloo four-rank oracle: DDP production gradients equal explicit global CE."""
import copy
from datetime import timedelta
import json
import os
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_balanced_hparams import make
from tests.test_nested_fusion import explicit_inputs,explicit_logits
from model.nested_semantic_mask import gather,hard_st,inclusion
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import AuditObjective,OBJECTIVES
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,dump


def explicit(model,images,views,valid):
    losses={};sparse=[];probabilities=[]
    for label,tokens in zip(['F','O','E'],views):
        z,text,zv,zt=explicit_inputs(model,images,tokens)
        logits=explicit_logits(model,zv,zt);probability=logits.sigmoid();mask=hard_st(probability)
        scores=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*F.normalize(text,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if label=='F' else valid
        labels=torch.arange(len(z))[enabled]
        losses[label+'_i2t']=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        losses[label+'_t2i']=F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        losses[label+'_combined']=losses[label+'_i2t']+losses[label+'_t2i']
        sparse.append(mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean())
        probabilities.append(probability.diagonal(dim1=0,dim2=1).T)
    z=model.clip.encode_image(images);t=model.clip.encode_text(views[0]);Q=100*(F.normalize(t,dim=-1)@F.normalize(z,dim=-1).T)
    labels=torch.arange(len(z));losses['native_i2t']=F.cross_entropy(Q.T,labels);losses['native_t2i']=F.cross_entropy(Q,labels);losses['native_combined']=losses['native_i2t']+losses['native_t2i']
    weights=model.search_hparams['view_weights'];losses['align_actual']=10/sum(weights)*sum(w*losses[k+'_combined'] for w,k in zip(weights,['F','O','E']))
    losses['sparse']=(sparse[0]+2*sparse[1]+2*sparse[2])/3
    losses['inc']=inclusion(*probabilities)[valid].mean()
    losses['total']=losses['align_actual']+losses['sparse']+losses['inc']
    return losses


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo',timeout=timedelta(minutes=3));rank=dist.get_rank();assert dist.get_world_size()==4
    results={}
    for kind,weights in [('RandomK',[1.,1.,1.]),('ArmB',[1.2,.6,1.2])]:
        torch.manual_seed(621);module=make({'view_weights':weights});reference=copy.deepcopy(module)
        torch.manual_seed(730+rank);images=torch.randn(2,8);views=[torch.randint(0,31,(2,6)) for _ in range(3)];valid=torch.tensor([1,0],dtype=torch.bool)
        global_images=gather(images,False);global_views=[gather(v,False) for v in views];global_valid=gather(valid,False)
        ddp=DDP(AuditObjective(module),find_unused_parameters=True)
        checks={}
        for objective in OBJECTIVES:
            ddp.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
            value=ddp(images,*views,valid,objective,False);value.backward()
            expected=explicit(reference,global_images,global_views,global_valid)[objective];expected.backward()
            avg=value.detach().clone();dist.all_reduce(avg);avg/=4
            torch.testing.assert_close(avg,expected.detach(),atol=1e-4,rtol=3e-6)
            maximum=0.;d2=n2=0.
            for name,p in module.named_parameters():
                q=dict(reference.named_parameters())[name]
                a=torch.zeros_like(p) if p.grad is None else p.grad;b=torch.zeros_like(q) if q.grad is None else q.grad
                torch.testing.assert_close(a,b,atol=2e-4,rtol=8e-5,msg=objective+':'+name)
                maximum=max(maximum,float((a-b).abs().max()));d2+=float((a-b).double().square().sum());n2+=float(b.double().square().sum())
            checks[objective]={'max_abs':maximum,'relative_L2':(d2/max(n2,1e-300))**.5,'passed':True}
        results[kind]=checks;del ddp,reference,module
    if rank==0:dump(EXP/'evidence/DDP_GLOBAL_CE_ORACLE.json',{'passed':True,'world_size':4,'backend':'gloo CPU correctness oracle','local_batch':2,'global_full_candidates':8,'global_partial_candidates':4,'objectives':results,'no_optimizer_created_or_used':True});print(json.dumps({'passed':True,'objectives_verified':32,'two_weightings':True}),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
