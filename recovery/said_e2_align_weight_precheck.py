"""Read-only common0/global1024 native BF16 regression and alignment isolation."""
import argparse
import gc
import math
import subprocess
import time
import types

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams, macro_terms
from recovery import said_e2_align_weight_fourarm500 as c
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import setup, seed_all, rng_state, restore_rng_state, state_digest, code_manifest


def legacy_class():
    source=subprocess.check_output(['git','show',c.PARENT_COMMIT+':model/balanced_hparam_search.py'],cwd=c.ROOT,text=True)
    namespace={'__name__':'model._frozen_e2_align_reference','__package__':'model'}
    exec(compile(source,'frozen_E2_production.py','exec'),namespace)
    return namespace['BalancedSearch']


def main():
    import json
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    receipt=dict(passed=False,rank=rank,no_optimizer_created=True,no_parameter_updates=True,started_utc=c.now(),arms={})
    def deny(*args,**kwargs):raise RuntimeError('Read-only precheck forbids optimizer creation')
    torch.optim.Optimizer.__init__=deny
    try:
        assert c.sha(c.STEP0)==c.STEP0_SHA
        cfg=c.read(c.BASE_EXP/'config.json')
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        initial=torch.load(c.STEP0,map_location='cpu',weights_only=False)
        assert initial['completed_steps']==0 and not initial['optimizer']['state']
        clip.load_state_dict(initial['model'],strict=True);construction=rng_state()
        model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
            view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
            fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
            checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
            shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
        restore_rng_state(construction);old=c.read(c.BASE_RUN/'step500/config.json')
        assert state_digest(model.clip.state_dict())==old['component_initialization']['clip']
        assert state_digest(model.fusion_branch.state_dict())==old['adapter_initialization']['state_sha256']
        del initial
        model=model.cuda(local).train();model.capture_hns_graph=True;before=state_digest(model.state_dict())
        states=[None]*world;dist.all_gather_object(states,before);assert len(set(states))==1
        dataset=FullLocalDataset(c.runner.local.INDEX,c.runner.local.IMAGES,'nested_detail_d3',0)
        sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
        loader=DataLoader(dataset,batch_size=256,sampler=list(iter(sampler))[:256],collate_fn=collate,
            num_workers=8,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
            generator=torch.Generator().manual_seed(0),timeout=60)
        batch=next(iter(loader));sample=json.loads(json.dumps(observe_selection(batch)))
        first=c.rows(c.BASE_RUN/'step500/steps.jsonl')[0]
        assert sample==next(h['sampling'] for h in first['rank_health'] if h['rank']==rank)
        tensors=[batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
        fixed_rng=rng_state();named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
        by_id={id(p):g['name'] for g in model.optimizer_groups() for p in g['params']}
        groups={g:[i for i,(_,p) in enumerate(named) if by_id[id(p)]==g] for g in c.GROUPS}
        def gradients(value):
            gs=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
            result=[]
            for param,g in zip(params,gs):
                v=torch.zeros_like(param) if g is None else g.detach().float().clone()
                dist.all_reduce(v);v/=world;assert torch.isfinite(v).all();result.append(v.cpu())
            return result
        def errors(values,reference):
            return {n:dict(exact=torch.equal(v,r),maximum_abs=float((v-r).abs().max()),
                relative_L2=float((v-r).norm())/max(float(r.norm()),1e-12)) for (n,_),v,r in zip(named,values,reference)}
        baseline_grad=None;baseline_logs=None;baseline_masks=None;baseline_components={};baseline_views={}
        native_forward=BalancedSearch.forward;legacy=legacy_class().forward
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        variants={'LEGACY_E2':([1.35,1.35,.3],10.),'DEFAULT_E2':([1.35,1.35,.3],10.),**c.ARMS}
        for variant,(weights,align) in variants.items():
            model.search_hparams['view_weights']=list(weights)
            model.search_hparams['lambda_align']=align;model.macro_hparams['lambda_align']=align
            model.forward=types.MethodType(legacy if variant=='LEGACY_E2' else native_forward,model)
            restore_rng_state(fixed_rng)
            loss,logs=model(*tensors,0);graph=model.hns_graph
            vectors=gradients(loss)
            scalar_logs={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()}
            masks={k:v.detach().cpu().clone() for k,v in graph['masks'].items()}
            norms={g:float(torch.stack([vectors[i].square().sum() for i in ii]).sum().sqrt()) for g,ii in groups.items()}
            check=dict(global_loss=scalar_logs['loss'],optimizer_group_gradient_norms=norms,
                view_weights=weights,lambda_align=align,unchanged_component_gradients={})
            receipt['arms'][variant]=check
            # Persist identifying evidence before any numerical assertion.
            c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
            if variant=='LEGACY_E2':
                baseline_grad=vectors;baseline_logs=scalar_logs;baseline_masks=masks
            assert all(torch.equal(masks[k],baseline_masks[k]) for k in masks)
            for name in ('F','O','E'):
                for key in ('i2t','t2i','sparse','keep_ratio'):
                    assert scalar_logs[name+'_'+key]==baseline_logs[name+'_'+key]
            for key in ('macro_raw_sparse','macro_raw_hierarchy','macro_weighted_sparse','macro_weighted_hierarchy'):
                assert scalar_logs[key]==baseline_logs[key]
            for key in ('weighted_sparse','weighted_hierarchy'):
                values=gradients(graph[key])
                if variant=='LEGACY_E2':baseline_components[key]=values
                checks=errors(values,baseline_components[key])
                check['unchanged_component_gradients'][key]=dict(maximum_relative_L2=max(x['relative_L2'] for x in checks.values()),
                    maximum_abs=max(x['maximum_abs'] for x in checks.values()))
                c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
                assert all(x['relative_L2']<=1e-6 and x['maximum_abs']<=1e-5 for x in checks.values())
                del values
            # Raw per-view CE tensors and gradients stay unchanged at frozen parameters.
            check['raw_view_gradient_checks']={}
            for view in ('F','Dall','D3'):
                values=gradients(graph['raw_view_'+view])
                if variant=='LEGACY_E2':baseline_views[view]=values
                checks=errors(values,baseline_views[view])
                check['raw_view_gradient_checks'][view]=dict(maximum_relative_L2=max(x['relative_L2'] for x in checks.values()),
                    maximum_abs=max(x['maximum_abs'] for x in checks.values()))
                c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
                assert all(x['relative_L2']<=1e-6 and x['maximum_abs']<=1e-5 for x in checks.values())
                del values
            af,ad,a3=[graph['raw_view_'+v] for v in ('F','Dall','D3')]
            wf,wd,w3=weights
            # Exactly the existing production arithmetic, not a new loss implementation.
            expected_legacy=10/(wf+wd+w3)*(wf*af+wd*ad+w3*a3)
            expected_align=macro_terms(expected_legacy,graph['raw_sparse'],graph['weighted_hierarchy'],model.macro_hparams)[0]
            assert torch.equal(graph['weighted_align'],expected_align)
            assert torch.equal(loss,(expected_align+graph['weighted_sparse'])+graph['weighted_hierarchy'])
            if weights==[1.35,1.35,.3]:
                assert scalar_logs['macro_raw_align']==baseline_logs['macro_raw_align']
            if variant in ('LEGACY_E2','DEFAULT_E2'):
                checks=errors(vectors,baseline_grad);check['default_every_parameter_gradient_checks']=checks
                check['default_forward_loss_exact']=scalar_logs==baseline_logs
                c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
                assert scalar_logs==baseline_logs
                assert all(x['relative_L2']<=1e-6 and x['maximum_abs']<=1e-5 for x in checks.values())
            check.update(masks_exact=True,raw_three_view_CE_exact=True,sparsity_and_HNS_exact=True,
                original_formula_and_macro_terms_exact=True,symmetric_CE_preserved=True,
                original_text_visual_detach_preserved=True,first_update_ramp_zero=True,
                nonzero_ramp_isolation_covered_by_CPU_tests=True)
            c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
            assert state_digest(model.state_dict())==before and all(p.grad is None for p in params)
            del loss,graph,af,ad,a3,expected_legacy,expected_align;model.hns_graph=None
            if variant!='LEGACY_E2':del vectors
            gc.collect();torch.cuda.empty_cache()
        receipt.update(passed=True,global_batch=1024,state_digest_before=before,state_digest_after=state_digest(model.state_dict()),
            sampling_sha256=c.base.digest(sample),common0_sha256=c.sha(c.STEP0),production_manifest=code_manifest(),
            all_rank_initial_parameters_exact=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            seconds=time.monotonic()-started,precision='Original BF16 encoder/checkpointing; native mean four-rank gradient reduction',
            default_gradient_tolerances=dict(relative_L2=1e-6,absolute=1e-5),finished_utc=c.now())
        assert receipt['common0_sha256']==c.STEP0_SHA
        c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        receipts=[None]*world;dist.all_gather_object(receipts,receipt)
        if rank==0:
            result=dict(passed=True,ranks=receipts,global_batch=1024,default_source_vs_frozen_E2_verified=True,
                no_optimizer_created=True,no_parameter_updates=True,source_scope=c.source_scope())
            c.dump(c.EXP/'CONFIG_AND_LOSS_EQUIVALENCE.json',result)
            c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',result)
    except BaseException as error:
        receipt.update(passed=False,error=repr(error),finished_utc=c.now());c.dump(c.RUN/f'PRECHECK-rank{rank}.json',receipt)
        if rank==0:
            c.dump(c.EXP/'CONFIG_AND_LOSS_EQUIVALENCE.json',receipt)
            c.dump(c.EXP/'PREUPDATE_EQUIVALENCE_VERIFIED.json',receipt)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
