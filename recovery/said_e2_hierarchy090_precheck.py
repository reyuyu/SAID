"""Read-only native BF16 common0/global1024 coefficient and gradient controls."""
import argparse
import gc
import json
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from model import longclip
from model.balanced_hparam_search import BalancedSearch,hparams,macro_terms
from recovery import said_e2_hierarchy090_500 as c
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import setup,seed_all,rng_state,restore_rng_state,state_digest,code_manifest


def main():
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    receipt=dict(passed=False,rank=rank,started_utc=c.now(),no_optimizer_created=True,no_parameter_updates=True)
    def deny(*args,**kwargs):raise RuntimeError('Read-only precheck forbids optimizer creation')
    torch.optim.Optimizer.__init__=deny
    try:
        before_sha=c.sha(c.STEP0);assert before_sha==c.STEP0_SHA
        cfg=c.read(c.BASE_EXP/'config.json');candidate=c.config(c.ARM);c.frozen(candidate,c.ARM)
        initial=torch.load(c.STEP0,map_location='cpu',weights_only=False)
        assert initial['completed_steps']==0 and not initial['optimizer']['state']
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        clip.load_state_dict(initial['model'],strict=True);construction_rng=rng_state()
        model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
            view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
            fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
            checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
            shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
        restore_rng_state(construction_rng)
        reference_cfg=c.read(c.BASE_RUN/'step500/config.json')
        assert state_digest(model.clip.state_dict())==reference_cfg['component_initialization']['clip']
        assert state_digest(model.fusion_branch.state_dict())==reference_cfg['adapter_initialization']['state_sha256']
        del initial
        model=model.cuda(local).train();model.capture_hns_graph=True
        before=state_digest(model.state_dict());digests=[None]*world
        dist.all_gather_object(digests,before);assert len(set(digests))==1
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
        fixed_rng=rng_state();params=[p for p in model.parameters() if p.requires_grad]
        groups=model.optimizer_groups();mapping={id(p):g['name'] for g in groups for p in g['params']}
        def reduce_grad(value):
            grad=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
            vectors={k:[] for k in c.GROUPS}
            for p,g in zip(params,grad):
                v=torch.zeros_like(p) if g is None else g.detach().float().clone()
                assert torch.isfinite(v).all();dist.all_reduce(v);v/=world
                vectors[mapping[id(p)]].append(v.flatten().cpu())
            return {k:torch.cat(v) for k,v in vectors.items()}
        def global_value(value):
            v=value.detach().clone();dist.all_reduce(v);return float(v/world)
        tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
        # At update1 the unchanged native ramp is zero. Independent forwards
        # must reproduce the original loss and actual first-update gradients.
        controls={};first=None;logs_first=None
        for label,scales in [('E2-Uniform',cfg), (c.ARM,candidate)]:
            model.macro_hparams={k:scales[k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')}
            restore_rng_state(fixed_rng);loss,logs=model(*tensors,0)
            grads=reduce_grad(loss);norms={k:float(v.norm()) for k,v in grads.items()}
            vals={k:global_value(model.hns_graph[k]) for k in ('weighted_align','weighted_sparse','weighted_hierarchy')}
            actual_loss=global_value(loss)
            controls[label]=dict(global_loss=actual_loss,optimizer_group_gradient_norms=norms,weighted_losses=vals)
            receipt['first_update_controls']=controls;c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
            assert abs(actual_loss-ref['loss'])<=1e-5 and vals['weighted_hierarchy']==0
            if first is None:first=grads;logs_first=vals
            else:
                checks={k:dict(maximum_absolute_error=float((v-first[k]).abs().max()),
                    relative_L2=float((v-first[k]).norm())/max(float(first[k].norm()),1e-12)) for k,v in grads.items()}
                receipt['first_update_gradient_checks']=checks
                c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
                assert vals==logs_first and all(x['maximum_absolute_error']<=1e-5 and x['relative_L2']<=1e-6 for x in checks.values())
            del loss,grads;model.hns_graph=None;gc.collect();torch.cuda.empty_cache()
        del first
        # Positive-ramp control on common0, not a simulated training update.
        # A and B totals are derived from precisely the same captured graph.
        model.macro_hparams={k:cfg[k] for k in ('lambda_align','lambda_sparse','lambda_hierarchy')}
        restore_rng_state(fixed_rng);loss,logs=model(*tensors,200);graph=model.hns_graph
        a,s,h=macro_terms(graph['alignment'],graph['original_sparsity'],graph['weighted_hierarchy'],model.macro_hparams)
        aa,ss,hh=macro_terms(graph['alignment'],graph['original_sparsity'],graph['weighted_hierarchy'],
            dict(lambda_align=10.,lambda_sparse=1.2,lambda_hierarchy=.9))
        ga,gb,gh=reduce_grad((a+s)+h),reduce_grad((aa+ss)+hh),reduce_grad(h)
        checks={}
        for k in c.GROUPS:
            expected_delta=-.1*gh[k];error=gb[k]-ga[k]-expected_delta
            denom=max(float(ga[k].norm()),float(gb[k].norm()),1e-12)
            checks[k]=dict(baseline_norm=float(ga[k].norm()),candidate_norm=float(gb[k].norm()),
                hierarchy_norm=float(gh[k].norm()),delta_norm=float((gb[k]-ga[k]).norm()),
                expected_delta_norm=float(expected_delta.norm()),relative_total_scale_error=float(error.norm())/denom,
                absolute_error=float(error.norm()),maximum_absolute_error=float(error.abs().max()))
        scalar={k:global_value(v) for k,v in dict(baseline_align=a,candidate_align=aa,baseline_sparse=s,
            candidate_sparse=ss,baseline_hierarchy=h,candidate_hierarchy=hh,
            baseline_total=(a+s)+h,candidate_total=(aa+ss)+hh).items()}
        expected_forward={k:graph[k].detach().cpu().clone() for k in ('alignment','original_sparsity','raw_hierarchy','V_DF','V_3D')}
        expected_loss=((aa+ss)+hh).detach().clone()
        receipt.update(positive_ramp_control=dict(completed_argument=200,parameter_updates=0,scalars=scalar,
            gradient_groups=checks,same_forward_graph=True,gradient_tolerance=3e-4))
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        assert torch.equal(a,aa) and torch.equal(s,ss) and torch.equal(hh,.9*h)
        assert scalar['baseline_hierarchy']>0 and graph['lambda_h']==1.
        assert all(x['relative_total_scale_error']<=3e-4 for x in checks.values())
        assert checks['backbone']['hierarchy_norm']==0 and checks['backbone']['delta_norm']==0
        del ga,gb,gh,loss,graph,a,s,h,aa,ss,hh
        model.hns_graph=None;gc.collect();torch.cuda.empty_cache()
        model.macro_hparams=dict(lambda_align=10.,lambda_sparse=1.2,lambda_hierarchy=.9)
        restore_rng_state(fixed_rng);candidate_loss,_=model(*tensors,200)
        forward_checks={k:torch.equal(model.hns_graph[k].detach().cpu(),v) for k,v in expected_forward.items()}
        receipt.update(candidate_forward_raw_components_exact=forward_checks,
            candidate_forward_total_exact=torch.equal(candidate_loss.detach(),expected_loss))
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        assert all(forward_checks.values()) and receipt['candidate_forward_total_exact']
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in params)
        assert c.sha(c.STEP0)==before_sha
        receipt.update(passed=True,state_digest_before=before,state_digest_after=before,
            common0_sha256=before_sha,production_manifest=code_manifest(),sample_count=256,global_batch=1024,
            sampling_sha256=c.base.digest(sampling),all_rank_initial_parameters_exact=True,
            native_BF16_original_checkpointing=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            seconds=time.monotonic()-tick,finished_utc=c.now())
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        all_receipts=[None]*world;dist.all_gather_object(all_receipts,receipt)
        if rank==0:
            final=dict(passed=True,ranks=all_receipts,no_optimizer_created=True,no_parameter_updates=True,
                native_BF16_original_checkpointing=True,global_batch=1024,only_variable='lambda_hierarchy',
                baseline_lambda_hierarchy=1.,candidate_lambda_hierarchy=.9,positive_ramp_control=True)
            c.dump(c.EXP/'LOSS_AND_GRADIENT_EQUIVALENCE.json',final)
            c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',final)
    except BaseException as error:
        receipt.update(passed=False,error=repr(error),finished_utc=c.now())
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        if rank==0:c.dump(c.EXP/'LOSS_AND_GRADIENT_EQUIVALENCE.json',receipt)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
