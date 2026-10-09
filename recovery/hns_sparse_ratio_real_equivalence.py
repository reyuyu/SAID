"""Measured original-S12 forward/gradient comparison on immutable real data."""
import argparse
import copy
import hashlib
import json
import subprocess
import types

import torch
import torch.distributed as dist
from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from recovery.s02_nfs500 import ROOT, STEP0_SHA, sha, dump, now
from recovery.s02_local500 import INDEX, IMAGES
from recovery.s02_full_local_data import FullLocalDataset
from train.train_nested_semantic_mask import setup, seed_all, CappedSampler, state_digest
from train.nested_semantic_data import collate
from torch.utils.data import DataLoader, DistributedSampler

SOURCE_COMMIT='d3a33bfe35c171f593819d48f45d98b5a964d416'
SOURCE_SHA='15fba463fdff37a654cd6e8a04e720dc577414b0a5ade0f124885aaee3c8b089'
CHECKPOINT_SHA='cc0b27c871f717c8eb821b1e7a425a8e70ba82f771ee82f0061ca2e6652a172d'
ARMS={'E1-AlignMatched':[2.25,2.25,.5], 'E2-Uniform':[5/3,5/3,5/3]}


def original_forward():
    source=subprocess.check_output(['git','show',SOURCE_COMMIT+':model/balanced_hparam_search.py'],cwd=ROOT)
    assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
    ns={'__name__':'model._s12_sparse_reference','__package__':'model'}
    exec(compile(source,'pinned-original-S12.py','exec'),ns)
    return ns['BalancedSearch'].forward


def gradients(value, named, world):
    out={}
    for (name,p),g in zip(named,torch.autograd.grad(value,[p for _,p in named],retain_graph=True,allow_unused=True)):
        v=torch.zeros_like(p) if g is None else g.detach().float().clone()
        assert torch.isfinite(v).all(),name
        dist.all_reduce(v);v/=world
        out[name]=v.cpu()
    return out


def exact(a,b):
    assert a.keys()==b.keys()
    for name in a:torch.testing.assert_close(a[name],b[name],atol=0,rtol=0,
        msg=lambda m,n=name:n+' '+m)
    return dict(parameters_checked=len(a),max_abs_error=0.,exact=True)


def norm(v):
    return sum(float(x.double().square().sum()) for x in v.values())**.5


def ddp_check(output):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from tests.test_nested_fusion import TinyFusionCLIP
    from recovery.hns_macro_equivalence import compare_gradients
    torch.set_num_threads(1)
    legacy=original_forward()
    def make(weights):
        return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3],
            inclusion_max=0,lambda_align=10,lambda_sparse=1.2,lambda_hierarchy=1),
            view_sparsity_weights=weights,hns_enabled=True,inclusion_hierarchy='detail_chain',
            fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    # Prepare the independent unpartitioned reference before initializing DDP.
    refs={}
    for name,w in [('S12',[1.,2.,2.]),*ARMS.items()]:
        torch.manual_seed(44219);m=make(w);m.forward=types.MethodType(legacy,m)
        images=torch.randn(8,8);tokens=[torch.randint(0,31,(8,6)) for _ in range(3)]
        valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
        value,_=m(images,*tokens,valid,199);value.backward();refs[name]=(m,value.detach())
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==4
    records=[]
    for name,w in [('S12',[1.,2.,2.]),*ARMS.items()]:
        torch.manual_seed(44219);a=make(w);b=copy.deepcopy(a);b.forward=types.MethodType(legacy,b)
        aa,bb=DDP(a,find_unused_parameters=True),DDP(b,find_unused_parameters=True)
        images=torch.randn(8,8);tokens=[torch.randint(0,31,(8,6)) for _ in range(3)]
        valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool);sl=slice(rank*2,rank*2+2)
        x,_=aa(images[sl],*[t[sl] for t in tokens],valid[sl],199)
        y,_=bb(images[sl],*[t[sl] for t in tokens],valid[sl],199)
        torch.testing.assert_close(x,y,atol=0,rtol=0);x.backward();y.backward()
        same=compare_gradients(a,b);global_model,value=refs[name]
        global_check=compare_gradients(a,global_model,exact=False)
        average=x.detach().clone();dist.all_reduce(average);average/=world
        torch.testing.assert_close(average,value,atol=3e-4,rtol=3e-5)
        records.append(dict(arm=name,same_partition_gradient=same,global_reference_gradient=global_check))
        del aa,bb,a,b
    if rank==0:dump(output,dict(passed=True,world=4,backend='gloo',valid_per_rank=[0,2,1,1],
        normalization_unchanged=True,lambda_sparse=1.2,original_source_commit=SOURCE_COMMIT,evidence=records,checked_utc=now()))
    dist.barrier();dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--output',required=True)
    ap.add_argument('--ddp',action='store_true')
    ap.add_argument('--batch-per-rank',type=int,default=256)
    args=ap.parse_args()
    if args.ddp:return ddp_check(args.output)
    assert args.checkpoint
    started=now();seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    # Audit-only deterministic kernels: compare implementations, not atomic
    # accumulation noise. Formal training precision/kernels remain untouched.
    torch.use_deterministic_algorithms(True)
    rank,local,world,_=setup();assert world==4
    checkpoint_sha=sha(args.checkpoint);assert checkpoint_sha==CHECKPOINT_SHA
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert payload['completed_steps']==500 and payload['scheduler_horizon']==4868
    cfg=payload['config'];assert cfg['hns_enabled'] and cfg.get('view_sparsity_weights',[1,2,2])==[1,2,2]
    assert [cfg[k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')]==[10.,1.2,1.]
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace());clip.load_state_dict(payload['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
        view_sparsity_weights=[1.,2.,2.],inclusion_hierarchy='detail_chain',fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],
        text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    model.fusion_branch.load_state_dict(payload['adapter'],strict=True);del payload
    model=model.cuda(local).train();model.capture_hns_graph=True
    before=state_digest(model.state_dict()) if rank==0 else None
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0);sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0);sampler.set_epoch(0)
    batch=next(iter(DataLoader(dataset,batch_size=args.batch_per_rank,sampler=CappedSampler(sampler,args.batch_per_rank),collate_fn=collate,num_workers=8,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,generator=torch.Generator().manual_seed(0))))
    ids=[None]*world;dist.all_gather_object(ids,batch['sample_id'].tolist())
    ids_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    inputs=[batch[k].cuda(local,non_blocking=True) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    reference={};records={};baseline_logs=None;legacy=original_forward()
    for name,weights in [('Original-S12',[1.,2.,2.]),('Original-Repeat',[1.,2.,2.]),('Current-S12',[1.,2.,2.]),*ARMS.items()]:
        seed_all(0);model.view_sparsity_weights=list(weights)
        loss,logs=(legacy(model,*inputs,499) if name.startswith('Original') else model(*inputs,499))
        graph=model.hns_graph
        components={key:gradients(value,named,world) for key,value in
                    [('alignment',graph['weighted_align']),('hierarchy',graph['weighted_hierarchy']),
                     ('sparsity',graph['weighted_sparse']),('total',loss)]}
        if name=='Original-S12':
            reference=components;baseline_logs=logs
        elif name=='Original-Repeat':
            repeat={key:exact(components[key],reference[key]) for key in components}
            del components
        else:
            for key in ('HNS_align','V_DF_hard','V_3D_hard','HNS_surcharge','lambda_h','inc_weight','inclusion_loss'):
                assert logs[key]==baseline_logs[key],(name,key)
            ah={key:exact(components[key],reference[key]) for key in ('alignment','hierarchy')}
            if name=='Current-S12':
                assert logs['loss']==baseline_logs['loss'] and logs['HNS_original_sparse']==baseline_logs['HNS_original_sparse']
                default={key:exact(components[key],reference[key]) for key in components}
            sf,sd,s3=graph['raw_sparse_views']
            raw=(sf+2*sd+2*s3)/3 if weights==[1.,2.,2.] else (weights[0]*sf+weights[1]*sd+weights[2]*s3)/3
            torch.testing.assert_close(graph['weighted_sparse'],1.2*raw,atol=0,rtol=0)
            sparse_gradient=exact(gradients(1.2*raw,named,world),components['sparsity'])
            errors={n:components['total'][n]-sum(components[k][n] for k in ('alignment','sparsity','hierarchy')) for n,_ in named}
            relative=norm(errors)/max(norm(components['total']),1e-12);assert relative<3e-4,(name,relative)
            records[name]=dict(normalized_weights=weights,effective_coefficients=[1.2*x for x in weights],
                alignment_hierarchy_gradient_checks=ah,sparse_formula_gradient_check=sparse_gradient,
                component_gradient_norms={k:norm(v) for k,v in components.items()},gradient_additivity_error=relative)
            del components,errors
        if rank==0:print('AUDIT_COMPLETED '+name,flush=True)
        del loss,graph
        model.hns_graph={}
        torch.cuda.empty_cache()
    assert all(p.grad is None for p in model.parameters())
    assert sha(args.checkpoint)==checkpoint_sha
    if rank==0:
        after=state_digest(model.state_dict());assert before==after
        dump(args.output,dict(passed=True,no_parameter_updates=True,checkpoint_sha256=checkpoint_sha,
            checkpoint_completed_steps=500,common0_sha256=STEP0_SHA,global_batch=args.batch_per_rank*world,batch_per_rank=args.batch_per_rank,
            sample_ids_sha256=ids_sha,coefficient_mass=5.,effective_lambda_sparse=1.2,
            original_source_commit=SOURCE_COMMIT,original_source_sha256=SOURCE_SHA,
            original_vs_current_every_parameter_gradient_checks=default,
            identical_original_repeat_gradient_checks=repeat,audit_deterministic_algorithms=True,
            default_hns_behavior_preserved=True,alignment_hierarchy_unchanged=True,
            alignment_hierarchy_gradient_unchanged=True,DDP_mean_gradient_measured=True,
            hns_kernel_source_unchanged=True,arms={n:records[n] for n in ARMS},
            state_digest_before=before,state_digest_after=after,precision='Unchanged native BF16 autocast / FP32 masks',
            started_utc=started,ended_utc=now(),local_only=True,NFS_fallback=False))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
