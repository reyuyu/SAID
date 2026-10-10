"""Read-only common0 first real global1024 batch before any optimizer creation."""
import argparse
import gc
import json
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from recovery import said_e2_early_lr_threearm500 as c
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import setup, seed_all, rng_state, restore_rng_state, state_digest, code_manifest


def main():
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    receipt=dict(passed=False,rank=rank,started_utc=c.now(),no_optimizer_created=True,no_updates=True)
    def deny(*args,**kwargs):raise RuntimeError('Read-only precheck forbids optimizer creation')
    torch.optim.Optimizer.__init__=deny
    try:
        assert c.sha(c.STEP0)==c.STEP0_SHA
        cfg=c.config(next(iter(c.ARMS)));c.frozen(cfg,next(iter(c.ARMS)))
        initial=torch.load(c.STEP0,map_location='cpu',weights_only=False)
        assert initial['completed_steps']==0 and not initial['optimizer']['state']
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        clip.load_state_dict(initial['model'],strict=True)
        construction_rng=rng_state()
        model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
            view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
            fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
            checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
            shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
        restore_rng_state(construction_rng)
        baseline_cfg=c.read(c.BASE_RUN/'step500/config.json')
        assert state_digest(model.clip.state_dict())==baseline_cfg['component_initialization']['clip']
        assert state_digest(model.fusion_branch.state_dict())==baseline_cfg['adapter_initialization']['state_sha256']
        del initial
        model=model.cuda(local).train();before=state_digest(model.state_dict())
        digests=[None]*world;dist.all_gather_object(digests,before);assert len(set(digests))==1
        dataset=FullLocalDataset(c.runner.local.INDEX,c.runner.local.IMAGES,'nested_detail_d3',0)
        sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
        indices=list(iter(sampler))[:256]
        loader=DataLoader(dataset,batch_size=256,sampler=indices,collate_fn=collate,num_workers=8,
            pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
            generator=torch.Generator().manual_seed(0),timeout=60)
        batch=next(iter(loader));sampling=json.loads(json.dumps(observe_selection(batch)))
        ref=c.rows(c.BASE_RUN/'step500/steps.jsonl')[0]
        assert sampling==next(h['sampling'] for h in ref['rank_health'] if h['rank']==rank)
        tensors=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
        fixed_rng=rng_state();named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        params=[p for _,p in named];baseline=None;loss0=None;logs0=None;comparisons={}
        torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
        for arm in ('E2-Uniform',*c.ARMS):
            restore_rng_state(fixed_rng)
            loss,logs=model(*tensors,0)
            current_logs={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()}
            gradients=torch.autograd.grad(loss,params,allow_unused=True)
            vectors=[]
            for param,g in zip(params,gradients):
                value=torch.zeros_like(param) if g is None else g.detach().float().clone()
                assert torch.isfinite(value).all()
                dist.all_reduce(value);value/=world;vectors.append(value.cpu())
            del gradients
            if baseline is None:
                baseline=vectors;loss0=float(loss.detach());logs0=current_logs
            checks={n:dict(exact=torch.equal(v,b),maximum_absolute_error=float((v-b).abs().max()),
                relative_L2=float((v-b).norm())/max(float(b.norm()),1e-12))
                for (n,_),v,b in zip(named,vectors,baseline)}
            comparisons[arm]=dict(loss=float(loss.detach()),loss_exact=float(loss.detach())==loss0,
                every_forward_loss_telemetry_exact=current_logs==logs0,every_parameter_gradient_checks=checks,
                next_LR=list(c.expected_lrs(0,cfg)) if arm=='E2-Uniform' else list(c.scaled_rates(c.expected_lrs(0,cfg),arm)),
                unchanged_forward=True)
            # Persist this variant before the numerical assertion.
            receipt.update(comparisons=comparisons,state_digest_before=before)
            c.dump(c.RUN/f'PRECHECK-verified-rank{rank}.json',receipt)
            assert all(v['maximum_absolute_error'] <= 1e-5 and v['relative_L2'] <= 1e-6 for v in checks.values())
            assert abs(float(loss.detach())-loss0) <= 1e-5 and current_logs==logs0
            assert state_digest(model.state_dict())==before and all(p.grad is None for p in params)
            if arm!='E2-Uniform':del vectors
            del loss;gc.collect();torch.cuda.empty_cache()
        receipt.update(passed=True,state_digest_after=state_digest(model.state_dict()),
            production_manifest=code_manifest(),common0_sha256=c.sha(c.STEP0),all_rank_initial_parameters_exact=True,
            sampling_summary_sha256=c.digest(sampling),sample_count=256,global_batch=1024,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-tick,
            native_BF16_and_original_checkpointing=True,finished_utc=c.now())
        assert receipt['common0_sha256']==c.STEP0_SHA
        c.dump(c.RUN/f'PRECHECK-verified-rank{rank}.json',receipt)
        all_receipts=[None]*world;dist.all_gather_object(all_receipts,receipt)
        if rank==0:
            c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',dict(passed=True,global_batch=1024,
                no_optimizer_created=True,no_parameter_updates=True,ranks=all_receipts,
            every_parameter_gradient_within_tolerance=True,gradient_abs_tolerance=1e-5,
            gradient_relative_L2_tolerance=1e-6,common0_immutable=True))
    except BaseException as error:
        receipt.update(passed=False,error=repr(error),finished_utc=c.now())
        c.dump(c.RUN/f'PRECHECK-verified-rank{rank}.json',receipt)
        if rank==0:c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',receipt)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
