"""Common0 real BF16/DDP equivalence and one-scalar loss isolation; no optimizer."""
import argparse
import ast
import gc
import json
import time
import subprocess

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from model import longclip
from model.balanced_hparam_search import BalancedSearch,hparams
from model import nested_fusion_mask as fusion
from recovery import said_e2_regular_hparam_fourarm500 as c
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import setup,seed_all,rng_state,restore_rng_state,state_digest,code_manifest


def legacy_score():
    source=subprocess.check_output(['git','show',c.PARENT_COMMIT+':model/nested_fusion_mask.py'],cwd=c.ROOT,text=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='score_block')
    namespace=dict(fusion.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'verified_legacy_score_block','exec'),namespace)
    return namespace['score_block']


def main():
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    receipt=dict(passed=False,rank=rank,no_optimizer_created=True,no_parameter_updates=True,started_utc=c.now())
    def deny(*a,**kw):raise RuntimeError('Read-only precheck forbids optimizer creation')
    torch.optim.Optimizer.__init__=deny
    original_score=fusion.score_block
    try:
        assert c.sha(c.STEP0)==c.STEP0_SHA
        cfg=c.config(next(iter(c.ARMS)))
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        initial=torch.load(c.STEP0,map_location='cpu',weights_only=False)
        assert initial['completed_steps']==0 and not initial['optimizer']['state']
        clip.load_state_dict(initial['model'],strict=True);construction=rng_state()
        model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(dict(cfg,lambda_sparse=1.2)),
            hns_enabled=True,view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
            fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
            checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
            shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
        restore_rng_state(construction);old=c.read(c.BASE_RUN/'step500/config.json')
        assert state_digest(model.clip.state_dict())==old['component_initialization']['clip']
        assert state_digest(model.fusion_branch.state_dict())==old['adapter_initialization']['state_sha256']
        del initial
        model=model.cuda(local).train();model.capture_hns_graph=True;before=state_digest(model.state_dict())
        rank_states=[None]*4;dist.all_gather_object(rank_states,before);assert len(set(rank_states))==1
        dataset=FullLocalDataset(c.runner.local.INDEX,c.runner.local.IMAGES,'nested_detail_d3',0)
        sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
        loader=DataLoader(dataset,batch_size=256,sampler=list(iter(sampler))[:256],collate_fn=collate,
            num_workers=8,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
            generator=torch.Generator().manual_seed(0),timeout=60)
        batch=next(iter(loader));sample=json.loads(json.dumps(observe_selection(batch)))
        reference=c.rows(c.BASE_RUN/'step500/steps.jsonl')[0]
        assert sample==next(h['sampling'] for h in reference['rank_health'] if h['rank']==rank)
        tensors=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
        fixed_rng=rng_state();named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
        group_by_id={id(p):g['name'] for g in model.optimizer_groups() for p in g['params']}
        groups={g:[i for i,(_,p) in enumerate(named) if group_by_id[id(p)]==g] for g in c.GROUPS}
        legacy=legacy_score();score_refs=None;grad_refs=None;baseline_logs=None;baseline_masks=None
        baseline_components={};arms={};torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        # Ramp completion is also covered by isolated kernel tests. At common0 the first update
        # uses the native ramp0; measured hierarchy gradients are intentionally zero here.
        for variant,(scale,sparse) in {'LEGACY_E2':(100.,1.2),'DEFAULT100':(100.,1.2),**c.ARMS}.items():
            model.contrastive_logit_scale=scale
            model.macro_hparams['lambda_sparse']=sparse;model.search_hparams['lambda_sparse']=sparse
            restore_rng_state(fixed_rng);blocks=[]
            def score(*args,**kwargs):
                result=(legacy if variant=='LEGACY_E2' else original_score)(*args,**kwargs)
                blocks.append(tuple(v.detach().cpu().clone() for v in result))
                return result
            fusion.score_block=score
            loss,logs=model(*tensors,0);graph=model.hns_graph
            gradients=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
            vectors=[]
            for p,g in zip(params,gradients):
                value=torch.zeros_like(p) if g is None else g.detach().float().clone()
                dist.all_reduce(value);value/=world;assert torch.isfinite(value).all();vectors.append(value.cpu())
            del gradients
            scalar_logs={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()}
            masks={k:v.detach().cpu().clone() for k,v in graph['masks'].items()}
            norms={g:float(torch.stack([vectors[i].square().sum() for i in ii]).sum().sqrt()) for g,ii in groups.items()}
            if variant=='LEGACY_E2':
                score_refs=blocks;grad_refs=vectors;baseline_logs=scalar_logs;baseline_masks=masks
            check=dict(global_loss=scalar_logs['loss'],optimizer_group_gradient_norms=norms,
                actual_scalars=dict(contrastive_logit_scale=scale,lambda_sparse=sparse),
                raw_sparse=scalar_logs['macro_raw_sparse'],weighted_sparse=scalar_logs['macro_weighted_sparse'])
            assert len(blocks)==len(score_refs)
            blockcheck=[]
            for actual,ref in zip(blocks,score_refs):
                assert torch.equal(actual[1],ref[1]) and torch.equal(actual[2],ref[2]),'Mask diagnostics changed at frozen state'
                if scale==100.:assert torch.equal(actual[0],ref[0])
                else:torch.testing.assert_close(actual[0],ref[0]*(scale/100.),rtol=1e-6,atol=1e-5)
                blockcheck.append(dict(score_default_exact=torch.equal(actual[0],ref[0]),mask_summary_exact=True,mask_diagnostics_exact=True))
            assert all(torch.equal(masks[k],baseline_masks[k]) for k in masks)
            assert scalar_logs['macro_raw_sparse']==baseline_logs['macro_raw_sparse']
            assert scalar_logs['macro_raw_hierarchy']==baseline_logs['macro_raw_hierarchy']
            assert math_close(scalar_logs['macro_weighted_sparse'],sparse*scalar_logs['macro_raw_sparse'])
            component_checks={}
            component_names=('weighted_sparse','weighted_hierarchy') if scale!=100 else ('weighted_align','weighted_hierarchy')
            if variant in ('LEGACY_E2','DEFAULT100'):component_names=('weighted_align','weighted_sparse','weighted_hierarchy')
            # Store actual gradients of each unchanged component, without making an additivity claim.
            for key in component_names:
                gs=torch.autograd.grad(graph[key],params,allow_unused=True,retain_graph=True)
                values=[]
                for p,g in zip(params,gs):
                    v=torch.zeros_like(p) if g is None else g.detach().float().clone()
                    dist.all_reduce(v);v/=world;assert torch.isfinite(v).all();values.append(v.cpu())
                if variant=='LEGACY_E2':baseline_components[key]=values
                ref=baseline_components[key]
                errs=[float((v-r).norm())/max(float(r.norm()),1e-12) for v,r in zip(values,ref)]
                assert max(errs)<=1e-6
                component_checks[key]=dict(maximum_relative_L2=max(errs),within_frozen_tolerance=True)
                del gs,values
            if variant in ('LEGACY_E2','DEFAULT100'):
                assert scalar_logs==baseline_logs
                checks={n:dict(exact=torch.equal(v,r),maximum_abs=float((v-r).abs().max()),relative_L2=float((v-r).norm())/max(float(r.norm()),1e-12))
                    for (n,_),v,r in zip(named,vectors,grad_refs)}
                check['default_every_parameter_gradient_checks']=checks
                assert all(v['maximum_abs']<=1e-5 and v['relative_L2']<=1e-6 for v in checks.values())
                check['default_forward_loss_exact']=True
            if scale==100:assert scalar_logs['macro_raw_align']==baseline_logs['macro_raw_align']
            check.update(blockchecks=blockcheck,masks_exact=True,unchanged_component_gradients=component_checks,
                alignment_symmetric=True,text_visual_detach_unchanged=True)
            arms[variant]=check
            receipt.update(arms=arms,global_batch=1024,state_digest_before=before,sampling_sha256=c.base.digest(sample))
            c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
            assert state_digest(model.state_dict())==before and all(p.grad is None for p in params)
            del loss,graph;model.hns_graph=None
            if variant!='LEGACY_E2':del vectors
            gc.collect();torch.cuda.empty_cache()
        receipt.update(passed=True,all_rank_initial_parameters_exact=True,state_digest_after=state_digest(model.state_dict()),
            common0_sha256=c.sha(c.STEP0),production_manifest=code_manifest(),seconds=time.monotonic()-started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,precision='Original BF16 encoder and checkpointing',
            default_gradient_tolerances=dict(relative_L2=1e-6,absolute=1e-5),finished_utc=c.now())
        assert receipt['common0_sha256']==c.STEP0_SHA
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        receipts=[None]*4;dist.all_gather_object(receipts,receipt)
        if rank==0:c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',dict(passed=True,ranks=receipts,
            global_batch=1024,default_source_vs_legacy_verified=True,no_optimizer_created=True,no_parameter_updates=True))
    except BaseException as error:
        receipt.update(passed=False,error=repr(error),finished_utc=c.now());c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        if rank==0:c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',receipt)
        raise
    finally:
        fusion.score_block=original_score
        if dist.is_initialized():dist.destroy_process_group()


def math_close(a,b):
    import math
    return math.isclose(a,b,rel_tol=4e-6,abs_tol=2e-6)


if __name__=='__main__':main()
