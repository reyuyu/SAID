"""Phase A.2 read-only total-gradient probe for E2@500 and E2@1217.

Only the original visual Patch input detach is toggled. No optimizer is
constructed and no parameter/checkpoint update is possible in this module.
"""
import argparse
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
from recovery.s02_full_local_data import FullLocalDataset
from recovery.visual_patch_gradient_phase_a import (
    NODES, COMPONENTS, group, layer, reference_sampling, serialized_sampling,
    seed_all, setup, state_digest, rng_state, restore_rng_state,
    visual_gradient_encode, tensor_hash, compare_tensor,
)
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import code_manifest

TOL = 1e-12


def descriptor(value):
    if not isinstance(value,torch.Tensor):return value
    v=value.detach().float().cpu()
    return dict(shape=list(v.shape),dtype=str(value.dtype),sha256=tensor_hash(value),
                finite=bool(torch.isfinite(v).all()),min=float(v.min()),max=float(v.max()),mean=float(v.mean()))


def json_safe(value):
    """Convert nested audit results to JSON without retaining gradient tensors."""
    if isinstance(value, torch.Tensor):
        return descriptor(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def forward_capture(graph):
    result={}
    for kind in ('probabilities','masks'):
        for name,value in graph[kind].items():result[kind+'_'+name]=value.detach().cpu().clone()
    for key in ('raw_view_F','raw_view_Dall','raw_view_D3','raw_sparse','raw_hierarchy',*COMPONENTS):
        result[key]=graph[key].detach().cpu().clone()
    return result


def norm(v):return float(v.double().norm())


def cosine(a,b):
    a,b=a.reshape(-1),b.reshape(-1)
    d=a.double().norm()*b.double().norm()
    return None if not float(d) else float(torch.dot(a.double(),b.double())/d)


def run_total(model,batch,node,variant,label):
    rank=dist.get_rank();world=dist.get_world_size();device=next(model.parameters()).device
    tensors=[batch[k].to(device,non_blocking=True) for k in ('image','tokens_f','tokens_o','tokens_e','valid')]
    input_hashes=[tensor_hash(x) for x in tensors];before=state_digest(model.state_dict())
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    original_branch=model.fusion_branch.encode_visual;original_visual=model.encode_visual;original_view=model.encode_view
    source={};captured={};view_counter=[0]
    def branch_encode(branch,hidden):
        source['patch']=hidden
        return original_branch(hidden) if variant=='A' else visual_gradient_encode(branch,hidden)
    def visual_encode(module,images):
        result=original_visual(images);captured['global_image']=result[0].detach().cpu().clone();return result
    def view_encode(module,tokens):
        result=original_view(tokens);captured['text_'+('F','Dall','D3')[view_counter[0]]]=result[0].detach().cpu().clone();view_counter[0]+=1;return result
    model.fusion_branch.encode_visual=types.MethodType(branch_encode,model.fusion_branch)
    model.encode_visual=types.MethodType(visual_encode,model);model.encode_view=types.MethodType(view_encode,model)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        loss,logs=model(*tensors,node);graph=model.hns_graph;forward=forward_capture(graph)
        gradients=torch.autograd.grad(graph['total_training'],params+[source['patch']],allow_unused=True,retain_graph=True)
        group_parts={};layer_parts={}
        for (name,param),gradient in zip(named,gradients[:-1]):
            value=torch.zeros_like(param) if gradient is None else gradient.detach().float().clone()
            assert torch.isfinite(value).all(),('nonfinite',name)
            dist.all_reduce(value);value/=world
            group_parts.setdefault(group(name),[]).append(value.cpu().flatten())
            layer_parts.setdefault(layer(name),[]).append(value.cpu().flatten())
        patch_total=gradients[-1]
        assert patch_total is not None and torch.isfinite(patch_total).all() if variant=='B' else patch_total is None or torch.isfinite(patch_total).all()
        patch_total_cpu=torch.zeros_like(source['patch']).float().cpu() if patch_total is None else patch_total.detach().float().cpu()
        patch_norm=float(patch_total_cpu.norm())
        patch_norms=[None]*world;dist.all_gather_object(patch_norms,patch_norm)
        patch_components={};patch_grads={}
        for key in COMPONENTS[:3]:
            pg=torch.autograd.grad(graph[key],source['patch'],allow_unused=True,retain_graph=True)[0]
            if pg is None:pg=torch.zeros_like(source['patch'])
            assert torch.isfinite(pg).all();pg=pg.detach().float().cpu();patch_grads[key]=pg;patch_components[key]=dict(norm=norm(pg),finite=True)
        patch_sum=patch_grads['weighted_align']+patch_grads['weighted_sparse']+patch_grads['weighted_hierarchy']
        patch_components['sum_vs_total']=dict(sum_norm=norm(patch_sum),total_norm=patch_norm,
            absolute_L2_error=norm(patch_sum-patch_total_cpu),relative_L2_error=None if patch_norm==0 else norm(patch_sum-patch_total_cpu)/patch_norm,
            max_absolute_element=float((patch_sum-patch_total_cpu).abs().max()))
        vectors={k:torch.cat(v) for k,v in group_parts.items()};layer_vectors={k:torch.cat(v) for k,v in layer_parts.items()}
        after=state_digest(model.state_dict())
        assert after==before and all(p.grad is None for p in model.parameters())
        result=dict(node=node,label=label,variant=variant,global_batch=world*len(batch['sample_id']),forward=forward,
            group_vectors=vectors,layer_vectors=layer_vectors,patch_components=patch_components,patch_grads=patch_grads,patch_total=patch_total_cpu,
            patch_input_gradient_norms_by_rank=patch_norms,state_digest_before=before,state_digest_after=after,
            parameters_unchanged=True,input_hashes=input_hashes,loss_finite=True,no_optimizer=True,no_parameter_updates=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.perf_counter()-started)
        return result
    finally:
        model.fusion_branch.encode_visual=original_branch;model.encode_visual=original_visual;model.encode_view=original_view;model.hns_graph={}
        del tensors;gc.collect();torch.cuda.empty_cache()


def repeat_stats(first,second,vector_key='group_vectors'):
    result={}
    for name in first[vector_key]:
        a,b=first[vector_key][name],second[vector_key][name];d=b-a
        result[name]=dict(first_norm=norm(a),second_norm=norm(b),repeat_delta_norm=norm(d),
                          repeat_relative_to_first=None if norm(a)==0 else norm(d)/norm(a),repeat_cosine=cosine(a,b))
    return result


def compare_ab(a,b,vector_key='group_vectors'):
    result={}
    for name in a[vector_key]:
        ga,gb=a[vector_key][name],b[vector_key][name];delta=gb-ga
        result[name]=dict(g_A_norm=norm(ga),g_B_norm=norm(gb),delta_norm=norm(delta),
                          delta_over_A=None if norm(ga)==0 else norm(delta)/norm(ga),
                          cosine_delta_A=cosine(delta,ga),cosine_A_B=cosine(ga,gb))
    return result


def public_run(r):
    return json_safe({k:v for k,v in r.items() if k not in ('group_vectors','layer_vectors','patch_grads','patch_total')})


def public_vectors(r):
    return dict(groups={k:descriptor(v) for k,v in r['group_vectors'].items()},
                layers={k:descriptor(v) for k,v in r['layer_vectors'].items()},patch_total=descriptor(r['patch_total']))


def analyze_batch(model,batch,node,label,output):
    shared=rng_state();runs={}
    for variant in ('A','B'):
        restore_rng_state(shared);first=run_total(model,batch,node,variant,label+'_'+variant+'1')
        restore_rng_state(shared);second=run_total(model,batch,node,variant,label+'_'+variant+'2')
        # Repeated total gradients are the numerical noise reference.
        assert all(compare_tensor(first['forward'][k],second['forward'][k])['exact'] for k in first['forward'])
        runs[variant]=dict(first=first,second=second,repeat=repeat_stats(first,second),repeat_layerwise=repeat_stats(first,second,'layer_vectors'))
    assert all(compare_tensor(runs['A']['first']['forward'][k],runs['B']['first']['forward'][k])['exact'] for k in runs['A']['first']['forward'])
    ab=compare_ab(runs['A']['first'],runs['B']['first']);ab_layerwise=compare_ab(runs['A']['first'],runs['B']['first'],'layer_vectors')
    patch={}
    for variant in ('A','B'):
        p=runs[variant]['first']['patch_components'];total=p['sum_vs_total']['total_norm']
        patch[variant]=dict(components=p,patch_input_total_norm=total,
            weighted_component_relative_norms={k:(None if total==0 else p[k]['norm']/total) for k in COMPONENTS[:3]},
            cosine_component_total={k:cosine(runs[variant]['first']['patch_grads'][k],runs[variant]['first']['patch_total']) for k in COMPONENTS[:3]},
            repeat_total_noise_norm=norm(runs[variant]['second']['patch_total']-runs[variant]['first']['patch_total']))
    repeat={v:runs[v]['repeat'] for v in ('A','B')}
    return dict(node=node,label=label,global_batch=len(batch['sample_id'])*dist.get_world_size(),
        runs={v:{'first':public_run(runs[v]['first']),'second':public_run(runs[v]['second']),'vectors_first':public_vectors(runs[v]['first']),'vectors_second':public_vectors(runs[v]['second'])} for v in runs},
        A_B_total_gradient=ab,A_B_layerwise_gradient=ab_layerwise,repeat_noise=repeat,repeat_noise_layerwise={v:runs[v]['repeat_layerwise'] for v in ('A','B')},patch_boundary=patch,
        A_B_forward_exact=True,input_hashes_equal=runs['A']['first']['input_hashes']==runs['B']['first']['input_hashes'],
        parameters_unchanged=all(runs[v][q]['parameters_unchanged'] for v in runs for q in ('first','second')),
        no_optimizer=True,no_parameter_updates=True)


def load_model(node,local):
    path,expected=NODES[node];assert sha(path)==expected
    payload=torch.load(path,map_location='cpu',weights_only=False);cfg=payload['config']
    assert payload['completed_steps']==node and payload['scheduler_horizon']==4868 and cfg['code_sha256']==code_manifest()
    assert cfg['hns_enabled'] and cfg['view_sparsity_weights']==[5/3]*3 and cfg['view_weights']==[1.35,1.35,.3]
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace());clip.load_state_dict(payload['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,view_sparsity_weights=cfg['view_sparsity_weights'],
        inclusion_hierarchy=cfg['inclusion_hierarchy'],fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
        checkpoint_encoders=True,image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=False)
    model.fusion_branch.load_state_dict(payload['adapter'],strict=True);states=payload['rng_per_rank'];del payload
    model=model.cuda(local).train();model.capture_hns_graph=True
    return model,cfg,states


def get_batch(node,rank,world,states,cfg):
    epoch,cursor=divmod(node,1217);dataset=FullLocalDataset('/root/said_s02_stage500/data_index','/root/said_s02_stage500/ShareGPT4V','nested_detail_d3',0);dataset.set_epoch(epoch)
    sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(epoch)
    start=cursor*256;positions=list(itertools.islice(iter(sampler),start,start+256));assert len(positions)==256
    generator=torch.Generator();generator.set_state(states[rank]['loader_generator'])
    loader=DataLoader(dataset,batch_size=256,sampler=positions,collate_fn=collate,num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,generator=generator,timeout=60)
    restore_rng_state(states[rank]);batch=next(iter(loader));assert serialized_sampling(batch)==reference_sampling(node,node+1,rank)
    return batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rank,local,world,peers=setup();assert world==4
    if rank==0:a.output_dir.mkdir(parents=True,exist_ok=False)
    dist.barrier();all_nodes={}
    try:
        for node in (500,1217):
            model,cfg,states=load_model(node,local);full=get_batch(node,rank,world,states,cfg);small={k:(v[:4] if isinstance(v,(torch.Tensor,list)) and len(v)==256 else v) for k,v in full.items()}
            sampling=serialized_sampling(full);dist.barrier()
            small_result=analyze_batch(model,small,node,'preflight16',a.output_dir)
            formal_result=analyze_batch(model,full,node,'global1024_next1',a.output_dir)
            if rank==0:
                all_nodes[str(node)]=dict(checkpoint=dict(path=str(NODES[node][0]),expected_sha256=NODES[node][1],observed_sha256=sha(NODES[node][0])),
                    sampling=sampling,preflight16=small_result,global1024_next1=formal_result,resources={
                        'preflight16':small_result['runs'],'global1024_next1':formal_result['runs']},gpu_identity=peers)
            del model;gc.collect();torch.cuda.empty_cache();dist.barrier()
        if rank==0:
            write_new(a.output_dir/'A2_RAW_RECEIPT.json',dict(status='COMPLETED',nodes=all_nodes,source_manifest=code_manifest(),no_training=True,no_optimizer_created=True,no_optimizer_step=True,no_checkpoint_saved=True))
    except BaseException as exc:
        if rank==0:write_new(a.output_dir/'A2_FAILURE_RECEIPT.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),nodes=all_nodes,no_training=True,no_optimizer_step=True))
        raise
    finally:
        dist.barrier();dist.destroy_process_group()
    return 0


if __name__=='__main__':sys.exit(main())
