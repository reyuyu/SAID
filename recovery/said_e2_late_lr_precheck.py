"""Real first epoch4 global1024 batch: loss/gradients before any LR update."""
import argparse
import gc
import hashlib
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from recovery import said_e2_late_lr_fourarm as control
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import setup, seed_all, restore_rng_state, rng_state, state_digest, code_manifest


def main():
    seed_all(0);torch.set_num_threads(4)
    rank,local,world,peers=setup();assert world==4
    def deny(*args,**kwargs):
        raise RuntimeError('Read-only precheck prohibits optimizer creation')
    torch.optim.Optimizer.__init__=deny
    checkpoint_sha=control.sha(control.PARENT)
    assert checkpoint_sha==control.PARENT_SHA
    p=torch.load(control.PARENT,map_location='cpu',weights_only=False);cfg=p['config']
    control.frozen(cfg);assert cfg['code_sha256']==code_manifest()
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(p['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
        view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
        fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
        checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
        shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    model.fusion_branch.load_state_dict(p['adapter'],strict=True)
    states=p['rng_per_rank'];del p
    model=model.cuda(local).train();before=state_digest(model.state_dict())
    dataset=FullLocalDataset(control.protocol.local.INDEX,control.protocol.local.IMAGES,'nested_detail_d3',0)
    dataset.set_epoch(3)
    sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(3)
    indices=list(iter(sampler))[:256]
    generator=torch.Generator();generator.set_state(states[rank]['loader_generator'])
    loader=DataLoader(dataset,batch_size=256,sampler=indices,collate_fn=collate,num_workers=8,
        pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,generator=generator,timeout=60)
    restore_rng_state(states[rank]);batch=next(iter(loader))
    sampling=json.loads(json.dumps(observe_selection(batch)))
    reference=control.rows(control.PARENT_RUN/'step4868/training/steps.jsonl')[0]
    assert sampling==next(h['sampling'] for h in reference['rank_health'] if h['rank']==rank)
    tensors=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    shared_rng=rng_state()
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    params=[v for _,v in named];baseline=None;baseline_loss=None;receipt={}
    torch.cuda.reset_peak_memory_stats()
    for name in ('E2-Uniform',*control.ARMS):
        restore_rng_state(shared_rng)
        loss,logs=model(*tensors,control.START)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        vectors=[]
        for (_,param),g in zip(named,grads):
            value=torch.zeros_like(param) if g is None else g.detach().float().clone()
            assert torch.isfinite(value).all()
            dist.all_reduce(value);value/=world;vectors.append(value.cpu())
        del grads
        if baseline is None:
            baseline=vectors;baseline_loss=loss.detach().cpu()
        checks={}
        for (key,_),g,ref in zip(named,vectors,baseline):
            error=float((g.double()-ref.double()).norm())/max(float(ref.double().norm()),1e-12)
            native=key.startswith(('clip.visual.','clip.transformer.','clip.token_embedding.'))
            assert error<=(.01 if native else .0003),(name,key,error)
            checks[key]=dict(relative_L2=error,exact=torch.equal(g,ref))
        assert torch.equal(loss.detach().cpu(),baseline_loss)
        assert state_digest(model.state_dict())==before and all(v.grad is None for v in params)
        receipt[name]=dict(loss=float(loss.detach()),loss_exact=True,
            maximum_parameter_gradient_relative_error=max(c['relative_L2'] for c in checks.values()),
            every_parameter_gradient_exact=all(c['exact'] for c in checks.values()),
            parameter_gradient_checks=checks,
            next_LR_multiplier=[1.]*4 if name=='E2-Uniform' else list(control.ARMS[name]),
            unchanged_forward_graph=True,no_optimizer_created=True,no_update=True)
        if name!='E2-Uniform':del vectors
        del loss;gc.collect();torch.cuda.empty_cache()
    receipts=[None]*world
    dist.all_gather_object(receipts,dict(rank=rank,sampling=sampling,state_digest=before,
        comparisons=receipt,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    assert len({r['state_digest'] for r in receipts})==1
    if rank==0:
        assert control.sha(control.PARENT)==checkpoint_sha
        control.dump(control.EXP/'PREUPDATE_EQUIVALENCE.json',dict(passed=True,global_batch=1024,
            batch_per_rank=256,next_update=3652,no_optimizer_created=True,no_parameter_updates=True,
            checkpoint_sha256=checkpoint_sha,checkpoint_immutable=True,production_manifest=code_manifest(),
            all_ranks_same_parameters=True,ranks=receipts,precision='native BF16 encoder; original checkpointing'))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
