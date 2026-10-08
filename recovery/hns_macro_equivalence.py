"""Actual fetched HNS forward versus opt-in macros: CPU/DDP and real BF16."""
import argparse
import copy
import json
import subprocess
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.balanced_hparam_search import BalancedSearch
from recovery.s02_nfs500 import ROOT,dump,now,sha

BRANCH='origin/experiment/nested-d3-hard-nested-sparsity500-v1'


def legacy_forward():
    source=subprocess.check_output(['git','show',BRANCH+':model/balanced_hparam_search.py'],cwd=ROOT,text=True)
    ns={'__name__':'model._macro_legacy','__package__':'model'}
    exec(compile(source,'fetched-HNS-v1.py','exec'),ns)
    return ns['BalancedSearch'].forward


def make():
    from tests.test_nested_fusion import TinyFusionCLIP
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3],inclusion_max=0),
        hns_enabled=True,inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',
        text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


def compare_gradients(a,b,*,exact=True):
    largest=0.;count=0
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=0 if exact else 8e-4,rtol=0 if exact else 8e-5,msg=name)
            largest=max(largest,float((p.grad-q.grad).abs().max()));count+=1
    return dict(trainable_parameters_checked=sum(p.requires_grad for p in a.parameters()),
        parameters_with_gradient=count,max_abs_error=largest,exact=exact)


def ddp(output):
    torch.set_num_threads(1);torch.manual_seed(44219)
    current=make();old=copy.deepcopy(current);old.forward=types.MethodType(legacy_forward(),old)
    global_model=copy.deepcopy(old)
    images=torch.randn(8,8);views=[torch.randint(0,31,(8,6)) for _ in range(3)]
    valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
    expected,_=global_model(images,*views,valid,199);expected.backward()
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==4
    a,b=DDP(current,find_unused_parameters=True),DDP(old,find_unused_parameters=True)
    sl=slice(rank*2,rank*2+2)
    x,logs=a(images[sl],*[t[sl] for t in views],valid[sl],199)
    y,oldlogs=b(images[sl],*[t[sl] for t in views],valid[sl],199)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    x.backward();y.backward();eq=compare_gradients(current,old)
    global_check=compare_gradients(current,global_model,exact=False)
    reduced=x.detach().clone();dist.all_reduce(reduced);reduced/=world
    torch.testing.assert_close(reduced,expected.detach(),atol=3e-4,rtol=3e-5)
    oa=torch.optim.AdamW(current.parameters(),lr=1e-4);ob=torch.optim.AdamW(old.parameters(),lr=1e-4)
    oa.step();ob.step()
    for p,q in zip(current.parameters(),old.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    if rank==0:dump(output,dict(passed=True,world=4,backend='gloo',production_forward=True,
        default_macros=[10,1,1],same_partition_old_new_gradient_exact=eq,
        single_global_batch_gradient_reference=global_check,global_valid_count=4,valid_per_rank=[0,2,1,1],
        AdamW_old_new_exact=True,no_double_world_scale=True,completed=199,ramp=.995))
    dist.barrier();dist.destroy_process_group()


def real(output,config):
    from model import longclip
    from model.balanced_hparam_search import hparams
    from recovery.s02_full_local_data import FullLocalDataset
    from recovery.s02_local500 import INDEX,IMAGES
    from recovery.s02_nfs500 import STEP0,STEP0_SHA
    from train.train_nested_semantic_mask import setup,seed_all,CappedSampler
    from train.nested_semantic_data import collate
    from torch.utils.data import DataLoader,DistributedSampler
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    cfg=json.loads(Path(config).read_text());assert sha(STEP0)==STEP0_SHA
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    payload=torch.load(STEP0,map_location='cpu',weights_only=False);clip.load_state_dict(payload['model'],strict=True);del payload
    a=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
        inclusion_hierarchy='detail_chain',fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
        image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
        checkpoint_pair_blocks=cfg['checkpoint_pair_blocks']).cuda(local).train()
    b=copy.deepcopy(a);b.forward=types.MethodType(legacy_forward(),b)
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0);sampler.set_epoch(0)
    batch=next(iter(DataLoader(dataset,batch_size=8,sampler=CappedSampler(sampler,8),collate_fn=collate,num_workers=0)))
    inputs=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    a.capture_hns_graph=b.capture_hns_graph=True
    x,logs=a(*inputs,199);y,oldlogs=b(*inputs,199)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    for k in ('HNS_align','HNS_original_sparse','V_DF_hard','V_3D_hard','HNS_surcharge','lambda_h'):
        assert float(logs[k])==float(oldlogs[k]),k
    x.backward();y.backward();localcheck=compare_gradients(a,b)
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:
            dist.all_reduce(pa.grad);pa.grad/=world
            dist.all_reduce(pb.grad);pb.grad/=world
    reducedcheck=compare_gradients(a,b)
    checks=[None]*world;dist.all_gather_object(checks,dict(rank=rank,local=localcheck,reduced=reducedcheck))
    if rank==0:dump(output,dict(passed=True,world=4,backend='nccl',fixed_real_batch=True,
        common0_sha256=STEP0_SHA,global_batch=32,batch_per_rank=8,
        precision='Unchanged native BF16 encoder autocast / FP32 masks and parameters',
        total_components_and_every_parameter_gradient_exact=True,DDP_mean_gradient_exact=True,
        checks=checks,formal_batch_unchanged=256,local_only=True,checked_utc=now()))
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--real',action='store_true')
    args=p.parse_args()
    if args.real:real(args.output,args.config)
    else:ddp(args.output)


if __name__=='__main__':main()
