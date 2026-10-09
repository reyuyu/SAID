"""Phase A.1: locate E2@500 global16 gradient-additivity failure.

This is a read-only four-rank probe. It creates no optimizer, performs no
parameter update and writes no checkpoint. It runs only 4 samples/rank.
"""
import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from recovery.e2_uniform_posthoc_audit import sha, write_new
from recovery.nested_d3_local_search import observe_selection
from recovery.s02_full_local_data import FullLocalDataset
from recovery.visual_patch_gradient_phase_a import (
    NODES, COMPONENTS, group, layer, reference_sampling, serialized_sampling,
    seed_all, setup, state_digest, rng_state, restore_rng_state,
    visual_gradient_encode, tensor_hash, compare_tensor,
)
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import code_manifest

BF16_NATIVE_TOL = .01
FP32_DIAGNOSTIC_TOL = .0003


def component_stats(records):
    """Finalize per-group/layer sums without retaining parameter-sized vectors."""
    result={}
    for name,v in records.items():
        total_norm=(v['total_norm2'])**.5
        err_norm=(v['error_norm2'])**.5
        result[name]=dict(
            parameter_count=v['parameter_count'],
            total_gradient_norm=total_norm,
            component_gradient_norms={k:v[k+'_norm2']**.5 for k in ('weighted_align','weighted_sparse','weighted_hierarchy')},
            component_sum_gradient_norm=v['sum_norm2']**.5,
            additivity_absolute_L2_error=err_norm,
            additivity_relative_L2_error=None if total_norm==0 else err_norm/total_norm,
            additivity_max_absolute_element=v['max_abs_error'],
            finite=v['finite'],
            parameter_names=v['parameter_names'],
        )
    return result


def new_record(name):
    return dict(parameter_count=0,total_norm2=0.,sum_norm2=0.,error_norm2=0.,max_abs_error=0.,finite=True,
                parameter_names=[],**{k+'_norm2':0. for k in COMPONENTS[:3]})


def accumulate(records, name, pname, grads):
    rec=records.setdefault(name,new_record(name));rec['parameter_count']+=1;rec['parameter_names'].append(pname)
    values=grads
    assert len(values)==4
    for value in values: assert torch.isfinite(value).all(), f'nonfinite gradient {pname}'
    summed=values[0]+values[1]+values[2];error=summed-values[3]
    rec['total_norm2']+=float(values[3].double().square().sum())
    rec['sum_norm2']+=float(summed.double().square().sum())
    rec['error_norm2']+=float(error.double().square().sum())
    rec['max_abs_error']=max(rec['max_abs_error'],float(error.abs().max()))
    for key,value in zip(COMPONENTS[:3],values):rec[key+'_norm2']+=float(value.double().square().sum())


def forward_summary(graph, captured):
    for kind in ('probabilities','masks'):
        for view,value in graph[kind].items():captured[f'{kind}_{view}']=value.detach().cpu().clone()
    for key in ('raw_view_F','raw_view_Dall','raw_view_D3','raw_sparse','raw_hierarchy',*COMPONENTS):
        captured[key]=graph[key].detach().cpu().clone()


def no_autocast(enabled):
    if enabled:
        return contextlib.nullcontext()
    # The model's production autocast calls resolve through this torch module.
    original=torch.autocast
    torch.autocast=lambda *args,**kwargs: contextlib.nullcontext()
    return contextlib.ExitStack()


def run_one(model,batch,node,variant,precision):
    rank=dist.get_rank();world=dist.get_world_size();device=next(model.parameters()).device
    tensors=[batch[k].to(device,non_blocking=True) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    before=state_digest(model.state_dict());input_hashes=[tensor_hash(x) for x in tensors]
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    original_branch=model.fusion_branch.encode_visual;original_visual=model.encode_visual;original_view=model.encode_view
    source={};captured={};view_counter=[0]
    def branch_encode(branch,hidden):
        source['patch']=hidden
        return original_branch(hidden) if variant=='A' else visual_gradient_encode(branch,hidden)
    def visual_encode(module,images):
        result=original_visual(images)
        captured['CLIP_global_image_embedding']=result[0].detach().cpu().clone()
        return result
    def view_encode(module,tokens):
        result=original_view(tokens)
        captured['CLIP_text_embedding_'+('F','Dall','D3')[view_counter[0]]]=result[0].detach().cpu().clone()
        view_counter[0]+=1
        return result
    model.fusion_branch.encode_visual=types.MethodType(branch_encode,model.fusion_branch)
    model.encode_visual=types.MethodType(visual_encode,model);model.encode_view=types.MethodType(view_encode,model)
    started=time.perf_counter();old_autocast=torch.autocast
    try:
        if precision=='fp32':torch.autocast=lambda *args,**kwargs: contextlib.nullcontext()
        loss,logs=model(*tensors,node);graph=model.hns_graph;forward_summary(graph,captured)
        groups={};layers={};routes_local={};component_local_norms={}
        for key in COMPONENTS:
            gradients=torch.autograd.grad(graph[key],params+[source['patch']],allow_unused=True,retain_graph=True)
            reduced=[]
            for (pname,param),gradient in zip(named,gradients[:-1]):
                value=torch.zeros_like(param) if gradient is None else gradient.detach().float().clone()
                dist.all_reduce(value);value/=world;reduced.append(value)
            patch_gradient=gradients[-1]
            patch_norm=0. if patch_gradient is None else float(patch_gradient.float().norm())
            patch_finite=patch_gradient is None or bool(torch.isfinite(patch_gradient).all())
            assert patch_finite
            norms=[None]*world;dist.all_gather_object(norms,patch_norm)
            routes_local[key]=norms
            component_local_norms[key]=norms
            for i,(pname,param) in enumerate(named):
                # Recompute all four component gradients are needed per parameter;
                # keep them in a small list keyed by parameter index.
                pass
            # Store reduced component vectors temporarily for this run; global16
            # has only 4 samples and these are released before the next variant.
            if key==COMPONENTS[0]:all_reduced=[]
            all_reduced.append(reduced)
            del gradients,patch_gradient
        # One backward through the algebraic sum is a control for the audit
        # method: production training also differentiates one total scalar.
        combined_target=sum(graph[k] for k in COMPONENTS[:3])
        combined_grad=torch.autograd.grad(combined_target,params+[source['patch']],allow_unused=True,retain_graph=True)
        combined_reduced=[]
        for param,gradient in zip(params,combined_grad[:-1]):
            value=torch.zeros_like(param) if gradient is None else gradient.detach().float().clone()
            dist.all_reduce(value);value/=world;combined_reduced.append(value)
        total_grad=all_reduced[3]
        control_group={};control_layer={}
        for i,(pname,param) in enumerate(named):
            values=[combined_reduced[i],total_grad[i],combined_reduced[i]-total_grad[i]]
            for records,name_fn in ((control_group,group),(control_layer,layer)):
                rec=records.setdefault(name_fn(pname),dict(parameter_count=0,total_norm2=0.,error_norm2=0.,max_abs_error=0.,finite=True,parameter_names=[]))
                rec['parameter_count']+=1;rec['parameter_names'].append(pname)
                rec['total_norm2']+=float(values[1].double().square().sum());rec['error_norm2']+=float(values[2].double().square().sum())
                rec['max_abs_error']=max(rec['max_abs_error'],float(values[2].abs().max()))
        def finalize_control(records):
            out={}
            for name,v in records.items():
                total=(v['total_norm2'])**.5;err=(v['error_norm2'])**.5
                out[name]=dict(parameter_count=v['parameter_count'],total_gradient_norm=total,
                    combined_vs_total_absolute_L2_error=err,
                    combined_vs_total_relative_L2_error=None if total==0 else err/total,
                    combined_vs_total_max_absolute_element=v['max_abs_error'],
                    parameter_names=v['parameter_names'])
            return out
        records_group={};records_layer={}
        for i,(pname,param) in enumerate(named):
            values=[all_reduced[j][i] for j in range(4)]
            accumulate(records_group,group(pname),pname,values);accumulate(records_layer,layer(pname),pname,values)
        group_result=component_stats(records_group);layer_result=component_stats(records_layer)
        # Exact forward and graph scalars are retained for A/B comparison.
        after=state_digest(model.state_dict())
        result=dict(node=node,variant=variant,precision=precision,global_batch=world*len(batch['sample_id']),
                    forward=captured,group_gradients=group_result,layerwise_gradients=layer_result,
                    patch_input_gradient_norms=component_local_norms,patch_route=routes_local,
                    state_digest_before=before,state_digest_after=after,parameters_unchanged=before==after,
                    input_hashes=input_hashes,no_optimizer=True,no_parameter_updates=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    seconds=time.perf_counter()-started,finite=True,
                    graph_total_minus_component_sum=float((graph['total_training']-combined_target).detach().abs().max()),
                    combined_component_gradients=finalize_control(control_group),
                    combined_component_layerwise=finalize_control(control_layer))
        assert all(p.grad is None for p in model.parameters())
        return result
    finally:
        torch.autocast=old_autocast
        model.fusion_branch.encode_visual=original_branch;model.encode_visual=original_visual;model.encode_view=original_view
        model.hns_graph={};gc.collect();torch.cuda.empty_cache()


def run_pair(model,batch,node,precision):
    shared=rng_state();results={}
    for variant in ('A','B'):
        restore_rng_state(shared);results[variant]=run_one(model,batch,node,variant,precision)
    a,b=results['A'],results['B']
    keys=sorted(set(a['forward']) & set(b['forward']))
    forward={k:compare_tensor(a['forward'][k],b['forward'][k]) for k in keys}
    assert all(v['exact'] for v in forward.values()),'A/B forward mismatch'
    for variant in ('A','B'):
        for group_name,info in results[variant]['group_gradients'].items():
            tolerance=BF16_NATIVE_TOL if precision=='bf16' and group_name=='native_visual_backbone' else FP32_DIAGNOSTIC_TOL
            info['tolerance']=tolerance;info['pass']=info['additivity_relative_L2_error']<=tolerance
    results['forward_equivalence']=forward
    results['A_pass']=all(v['pass'] for v in results['A']['group_gradients'].values())
    results['B_pass']=all(v['pass'] for v in results['B']['group_gradients'].values())
    results['native_A_pass']=results['A']['group_gradients']['native_visual_backbone']['pass']
    results['native_B_pass']=results['B']['group_gradients']['native_visual_backbone']['pass']
    results['state_unchanged']=all(results[v]['parameters_unchanged'] for v in ('A','B'))
    results['input_equal']=results['A']['input_hashes']==results['B']['input_hashes']
    return results


def serializable_result(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_hash(value),
                    finite=bool(torch.isfinite(value).all()),min=float(value.float().min()),
                    max=float(value.float().max()),mean=float(value.float().mean()))
    if isinstance(value,dict):return {str(k):serializable_result(v) for k,v in value.items()}
    if isinstance(value,list):return [serializable_result(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    # Deliberately do not monkeypatch torch.save here: no save/checkpoint API is
    # used. The output writer is JSON-only and exclusive-create.
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,peers=setup();assert world==4
    if rank==0:args.output_dir.mkdir(parents=True,exist_ok=False)
    dist.barrier();path,expected=NODES[500];assert sha(path)==expected
    payload=torch.load(path,map_location='cpu',weights_only=False);cfg=payload['config']
    assert payload['completed_steps']==500 and payload['scheduler_horizon']==4868
    assert cfg['code_sha256']==code_manifest() and cfg['hns_enabled'] and cfg['view_sparsity_weights']==[5/3]*3
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace());clip.load_state_dict(payload['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
        view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
        fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],checkpoint_encoders=True,
        image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=False)
    model.fusion_branch.load_state_dict(payload['adapter'],strict=True);states=payload['rng_per_rank'];del payload
    model=model.cuda(local).train();model.capture_hns_graph=True
    dataset=FullLocalDataset('/root/said_s02_stage500/data_index','/root/said_s02_stage500/ShareGPT4V','nested_detail_d3',0)
    sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    positions=list(itertools.islice(iter(sampler),500*256,501*256));assert len(positions)==256
    generator=torch.Generator();generator.set_state(states[rank]['loader_generator'])
    loader=DataLoader(dataset,batch_size=256,sampler=positions,collate_fn=collate,num_workers=8,drop_last=False,pin_memory=True,
                      multiprocessing_context='spawn',prefetch_factor=2,generator=generator,timeout=60)
    restore_rng_state(states[rank]);full_batch=next(iter(loader))
    assert serialized_sampling(full_batch)==reference_sampling(500,501,rank)
    batch={k:(v[:4] if isinstance(v,(torch.Tensor,list)) and len(v)==256 else v) for k,v in full_batch.items()}
    sampling=[None]*world;dist.all_gather_object(sampling,dict(rank=rank,sampling=serialized_sampling(batch),
        input_tensor_sha256={k:tensor_hash(batch[k]) for k in ('image','tokens_f','tokens_o','tokens_e','valid')}))
    all_results={}
    try:
        for precision in ('bf16','fp32'):
            all_results[precision]=run_pair(model,batch,500,precision)
        bf16_pass=all_results['bf16']['native_A_pass'] and all_results['bf16']['native_B_pass']
        fp32_pass=all_results['fp32']['native_A_pass'] and all_results['fp32']['native_B_pass']
        control_fp32=all(
            (all(x['combined_vs_total_relative_L2_error'] is None or
                 x['combined_vs_total_relative_L2_error']<=FP32_DIAGNOSTIC_TOL
                 for x in all_results['fp32'][v]['combined_component_gradients'].values()))
            for v in ('A','B'))
        graph_identity=all(all_results[p][v]['graph_total_minus_component_sum']==0.
                           for p in ('bf16','fp32') for v in ('A','B'))
        if graph_identity and control_fp32 and not fp32_pass:
            classification='PROBE_IMPLEMENTATION_ISSUE'
        elif fp32_pass and not bf16_pass:
            classification='BF16_NUMERICAL_EFFECT_SUPPORTED'
        elif not fp32_pass:
            classification='GRAPH_OR_GRADIENT_INCONSISTENCY'
        else:
            classification='INCONCLUSIVE'
    except BaseException as exc:
        classification='INCONCLUSIVE';all_results['exception']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        if rank==0:
            receipt=dict(status='COMPLETED',classification=classification if 'classification' in locals() else 'INCONCLUSIVE',
                checkpoint=dict(path=str(path),expected_sha256=expected,observed_sha256=sha(path),unchanged=True),
                source_manifest=code_manifest(),node=500,global_batch=16,per_rank_batch=4,sampling=sampling,
                precision_definitions=dict(bf16='Native production autocast/checkpointing path',fp32='Diagnostic only: torch.autocast disabled; same model/state/input and loss graph'),
                thresholds=dict(BF16_native_backbone=BF16_NATIVE_TOL,FP32_diagnostic_reference=FP32_DIAGNOSTIC_TOL),
                variants=serializable_result(all_results),no_training=True,no_optimizer_created=True,no_optimizer_step=True,no_checkpoint_saved=True,
                gpu_identity=peers)
            write_new(args.output_dir/'VARIANT_A_B_GRADIENT_RECEIPTS.json',receipt)
    dist.barrier();dist.destroy_process_group();return 0


if __name__=='__main__':sys.exit(main())
