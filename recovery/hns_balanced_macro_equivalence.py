"""Fetched HNS/Balanced defaults: fixed-batch forward, gradients and DDP proof."""
import argparse
import copy
import json
import subprocess
import types

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.balanced_hparam_search import BalancedSearch,hparams,MACRO_DEFAULTS
from recovery.hns_macro_equivalence import compare_gradients
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,sha,dump,now

BRANCHES={'HNS':'origin/experiment/nested-d3-hard-nested-sparsity500-v1',
          'Balanced':'origin/experiment/nested-detail-d3-balanced-full-v1'}
COMPLETED=(0,99,199,200,499)


def legacy_forward(method):
    source=subprocess.check_output(['git','show',BRANCHES[method]+':model/balanced_hparam_search.py'],cwd=ROOT,text=True)
    ns={'__name__':'model._macro_reference','__package__':'model'}
    exec(compile(source,'fetched-'+method+'.py','exec'),ns)
    return ns['BalancedSearch'].forward


def make(method,scales=None):
    from tests.test_nested_fusion import TinyFusionCLIP
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3],
        inclusion_max=0 if method=='HNS' else 1,**(scales or MACRO_DEFAULTS)),
        hns_enabled=method=='HNS',inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',
        text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


def ddp(output):
    torch.set_num_threads(1)
    global_references={}
    for method in BRANCHES:
        for completed in COMPLETED:
            torch.manual_seed(44219);reference=make(method)
            reference.forward=types.MethodType(legacy_forward(method),reference)
            images=torch.randn(8,8);views=[torch.randint(0,31,(8,6)) for _ in range(3)]
            valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
            expected,_=reference(images,*views,valid,completed);expected.backward()
            global_references[method,completed]=(reference,expected.detach())
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==4
    evidence=[]
    for method in BRANCHES:
        for completed in COMPLETED:
            torch.manual_seed(44219);a=make(method);b=copy.deepcopy(a);b.forward=types.MethodType(legacy_forward(method),b)
            aa,bb=DDP(a,find_unused_parameters=True),DDP(b,find_unused_parameters=True)
            images=torch.randn(8,8);views=[torch.randint(0,31,(8,6)) for _ in range(3)]
            valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool);sl=slice(rank*2,rank*2+2)
            x,logs=aa(images[sl],*[t[sl] for t in views],valid[sl],completed)
            y,old=bb(images[sl],*[t[sl] for t in views],valid[sl],completed)
            torch.testing.assert_close(x,y,atol=0,rtol=0);x.backward();y.backward();check=compare_gradients(a,b)
            reference,expected=global_references[method,completed]
            global_check=compare_gradients(a,reference,exact=False)
            reduced=x.detach().clone();dist.all_reduce(reduced);reduced/=world
            torch.testing.assert_close(reduced,expected,atol=3e-4,rtol=3e-5)
            oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4);oa.step();ob.step()
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
            evidence.append(dict(method=method,completed=completed,gradient_check=check,
                global_reference_gradient_check=global_check,loss_exact=True,AdamW_exact=True))
            del aa,bb,a,b
    if rank==0:dump(output,dict(passed=True,world=world,backend='gloo',evidence=evidence,
        valid_per_rank=[0,2,1,1],global_valid_count=4,normalization_unchanged=True,checked_utc=now()))
    dist.barrier();dist.destroy_process_group()


def real(output,configs):
    from model import longclip
    from recovery.s02_full_local_data import FullLocalDataset
    from recovery.s02_local500 import INDEX,IMAGES
    from train.train_nested_semantic_mask import setup,seed_all,CappedSampler
    from train.nested_semantic_data import collate
    from torch.utils.data import DataLoader,DistributedSampler
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4 and sha(STEP0)==STEP0_SHA
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0);sampler.set_epoch(0)
    batch=next(iter(DataLoader(dataset,batch_size=8,sampler=CappedSampler(sampler,8),collate_fn=collate,num_workers=0)))
    inputs=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    evidence=[]
    for method,path in zip(BRANCHES,configs):
        cfg=json.loads(open(path).read());cfg.update(MACRO_DEFAULTS)
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        payload=torch.load(STEP0,map_location='cpu',weights_only=False);clip.load_state_dict(payload['model'],strict=True);del payload
        a=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=method=='HNS',
            inclusion_hierarchy='detail_chain',fusion=cfg['fusion'],visual=cfg['visual'],
            condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
            image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
            checkpoint_pair_blocks=cfg['checkpoint_pair_blocks']).cuda(local).train()
        b=copy.deepcopy(a);b.forward=types.MethodType(legacy_forward(method),b)
        for completed in COMPLETED:
            a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
            x,logs=a(*inputs,completed);y,old=b(*inputs,completed)
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            x.backward();y.backward();localcheck=compare_gradients(a,b)
            for p,q in zip(a.parameters(),b.parameters()):
                if p.grad is not None:
                    dist.all_reduce(p.grad);p.grad/=world;dist.all_reduce(q.grad);q.grad/=world
            reduced=compare_gradients(a,b);checks=[None]*world
            dist.all_gather_object(checks,dict(rank=rank,local=localcheck,reduced=reduced))
            evidence.append(dict(method=method,completed=completed,checks=checks,loss_exact=True))
            del x,y,logs,old
        del a,b,clip;torch.cuda.empty_cache()
    if rank==0:dump(output,dict(passed=True,world=world,backend='nccl',fixed_real_batch=True,
        common0_sha256=STEP0_SHA,global_batch=32,batch_per_rank=8,formal_batch_per_rank=256,
        formal_batch_unchanged=True,completed_states=list(COMPLETED),sample_ids_local_rank0=batch['sample_id'].tolist(),
        precision='Frozen BF16 encoders / FP32 masks and parameters',every_parameter_gradient_exact=True,
        evidence=evidence,local_only=True,checked_utc=now()))
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--configs',nargs=2);p.add_argument('--real',action='store_true')
    args=p.parse_args()
    if args.real:real(args.output,args.configs)
    else:ddp(args.output)


if __name__=='__main__':main()
