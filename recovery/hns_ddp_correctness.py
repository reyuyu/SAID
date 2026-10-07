"""Four-process CPU/gloo production-objective comparison to one global batch."""
import argparse
import copy
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.balanced_hparam_search import BalancedSearch
from recovery.s02_nfs500 import dump
from tests.test_nested_fusion import TinyFusionCLIP


def make(beta=(2.,2.)):
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3],inclusion_max=0),
        hns_enabled=True,hns_beta=beta,inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',
        text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--beta',type=float,nargs=2,default=[2.,2.]);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(44219)
    module=make(args.beta);reference=copy.deepcopy(module)
    images=torch.randn(8,8);views=[torch.randint(0,31,(8,6)) for _ in range(3)]
    valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
    # Compute the global single-process reference before joining any group.
    expected,reference_logs=reference(images,*views,valid,199);expected.backward()
    expected_grads={n:p.grad.clone() if p.grad is not None else None for n,p in reference.named_parameters()}
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==4
    ddp=DDP(module,find_unused_parameters=True)
    sl=slice(2*rank,2*rank+2)
    actual,logs=ddp(images[sl],*[t[sl] for t in views],valid[sl],199)
    loss_mean=actual.detach().clone();dist.all_reduce(loss_mean);loss_mean/=world
    torch.testing.assert_close(loss_mean,expected.detach(),atol=3e-4,rtol=3e-5)
    for key in ('HNS_align','HNS_original_sparse','V_DF_hard','V_3D_hard','HNS_surcharge','HNS_regularizer'):
        torch.testing.assert_close(logs[key],reference_logs[key],atol=3e-4,rtol=3e-5,msg=key)
    actual.backward();largest=0.
    for name,p in module.named_parameters():
        q=expected_grads[name];assert (p.grad is None)==(q is None)
        if q is not None:
            torch.testing.assert_close(p.grad,q,atol=8e-4,rtol=8e-5,msg=name)
            largest=max(largest,float((p.grad-q).abs().max()))
    optimizer=torch.optim.AdamW(module.parameters(),lr=1e-4,weight_decay=.01)
    optimizer.step()
    difference=0.
    for name,p in module.named_parameters():
        peer=p.detach().clone();dist.broadcast(peer,src=0)
        difference=max(difference,float((p-peer).abs().max()))
    assert difference==0
    if rank==0:dump(args.output,dict(passed=True,world=4,backend='gloo',production_objective=True,
        global_batch=8,valid_counts=[0,2,1,1],global_valid=4,completed_updates=199,lambda_h=.995,beta=args.beta,
        loss_matches_global_reference=True,all_trainable_gradients_match_global_reference=True,
        one_AdamW_update_all_ranks_exact=True,all_rank_parameters_exact=True,max_rank_difference=difference,
        global_reference_scope='Loss and every parameter gradient; cross-partition AdamW is not compared because near-zero reduction roundoff can be magnified by epsilon. Separate lambda0 equivalence tests compare the identical-partition AdamW update exactly.',
        max_gradient_difference=largest,gradient_tolerance=dict(atol=8e-4,rtol=8e-5),
        no_double_world_scaling=True,single_process_loss=float(expected.detach()),distributed_loss=float(loss_mean)))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
