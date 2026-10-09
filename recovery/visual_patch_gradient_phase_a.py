"""Read-only four-rank E2 Patch-condition gradient probe; never trains.

torchrun --standalone --nproc_per_node=4 -m recovery.visual_patch_gradient_phase_a \
  --checkpoint-node 500 --output-dir /fresh/local/directory
Node1217 uses the same interface. Small4/rank preflight precedes two fixed
256/rank batches. All files are exclusive-create. No optimizer is constructed.
"""
import argparse
import gc
import hashlib
import itertools
import json
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from model.nested_fusion_mask import FusionBranch
from recovery.e2_uniform_posthoc_audit import sha, write_new
from recovery.s02_full_local_data import FullLocalDataset
from recovery.nested_d3_local_search import observe_selection
from train.nested_semantic_data import collate, sampling_diagnostics
from train.train_nested_semantic_mask import (code_manifest, rng_state,
    restore_rng_state, seed_all, setup, state_digest)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1')
NODES = {
    500: (RUNTIME/'hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform/step500/step000500.pt',
          '74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a'),
    1217: (RUNTIME/'hns-s12-uniform-e1-1217-v1/step1217/training/step001217.pt',
           'd276813e12b1b6da9a1f0247d9bbe13c6471276b45e852e06b149b9ad3cb5ead'),
}
COMPONENTS = ('weighted_align', 'weighted_sparse', 'weighted_hierarchy', 'total_training')
# BF16 encoder backward rounds shared-path accumulation; FP32 downstream keeps
# the existing audit tolerance. Declare tolerances before reading observations.
BF16_ADDITIVITY_TOL = .01
FP32_ADDITIVITY_TOL = 3e-4


def visual_gradient_encode(branch, hidden):
    """Exact production expression with only hidden.detach() removed."""
    tokens = branch.visual_adapter(hidden.float())
    return branch.visual_blocks(tokens.permute(1, 0, 2)).permute(1, 0, 2)


def group(name):
    if name.startswith('clip.visual.'): return 'native_visual_backbone'
    if name.startswith('clip.mask_net.'): return 'text_mask_shared_pool'
    if name.startswith('clip.'): return 'native_text_backbone'
    if name.startswith('fusion_branch.visual_blocks.'): return 'visual_mask'
    if name.startswith('fusion_branch.visual_adapter.'): return 'visual_adapter'
    return 'fusion_gate'


def layer(name):
    prefix = 'clip.visual.transformer.resblocks.'
    if name.startswith(prefix): return 'visual_transformer_layer_' + name[len(prefix):].split('.')[0]
    if name.startswith('clip.visual.'): return 'visual_' + name[len('clip.visual.'):].split('.')[0]
    return group(name)


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = a.norm()*b.norm()
    return None if not float(denominator) else float(torch.dot(a, b)/denominator)


def compare_tensor(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    error = (a.double()-b.double()).abs()
    denominator = torch.maximum(a.double().abs(), b.double().abs()).clamp_min(1e-12)
    return dict(shape=list(a.shape), dtype=str(a.dtype), exact=torch.equal(a,b),
                max_absolute_difference=float(error.max()),
                max_relative_difference=float((error/denominator).max()))


def summarize_vectors(a, b):
    da = b['weighted_align']-a['weighted_align']
    ds = b['weighted_sparse']-a['weighted_sparse']
    dh = b['weighted_hierarchy']-a['weighted_hierarchy']
    base, new = a['total_training'], b['total_training']
    delta = new-base
    result = dict(g_base_norm=float(base.norm()), g_new_norm=float(new.norm()),
        delta_g_norm=float(delta.norm()),
        delta_g_over_g_base=None if not float(base.norm()) else float(delta.norm()/base.norm()),
        cosine_delta_base=cosine(delta,base),
        cosine_added_alignment_base_alignment=cosine(da,a['weighted_align']),
        cosine_added_sparsity_base_alignment=cosine(ds,a['weighted_align']),
        cosine_added_hierarchy_base_alignment=cosine(dh,a['weighted_align']),
        cosine_added_sparsity_added_alignment=cosine(ds,da),
        cosine_added_hierarchy_added_alignment=cosine(dh,da),
        cosine_added_sparsity_added_hierarchy=cosine(ds,dh),
        added_component_norms=dict(alignment=float(da.norm()),sparsity=float(ds.norm()),hierarchy=float(dh.norm())),
        A_component_norms={k:float(v.norm()) for k,v in a.items()},
        B_component_norms={k:float(v.norm()) for k,v in b.items()},
        component_delta_max_abs={k:float((b[k]-a[k]).abs().max()) for k in COMPONENTS})
    result['gradient_additivity'] = {
        label:dict(relative_L2_error=float((v['total_training']-sum(v[k] for k in COMPONENTS[:3])).norm())/
                   max(float(v['total_training'].norm()),1e-12)) for label,v in [('A',a),('B',b)]}
    result['delta_additivity_relative_L2_error'] = float((delta-da-ds-dh).norm())/max(float(delta.norm()),1e-12)
    return result


def additivity_failures(results, layerwise=False):
    failures=[]
    for name,info in results.items():
        native=(name.startswith('visual_') and name not in ('visual_adapter','visual_mask')) if layerwise else name=='native_visual_backbone'
        tolerance=BF16_ADDITIVITY_TOL if native else FP32_ADDITIVITY_TOL
        for variant,entry in info['gradient_additivity'].items():
            entry.update(tolerance=tolerance,passed=entry['relative_L2_error']<=tolerance)
            if not entry['passed']:failures.append(dict(group=name,variant=variant,**entry))
    return failures


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def slice_batch(batch, count):
    length = len(batch['sample_id']); out = {}
    for key,value in batch.items():
        out[key] = value[:count] if isinstance(value,(torch.Tensor,list)) and len(value)==length else value
    return out


def serialized_sampling(batch):
    # Original training writes observe_selection to JSONL: integer histogram
    # keys become strings. Compare that exact schema, including index digest.
    return json.loads(json.dumps(observe_selection(batch)))


def deny_updates():
    def forbidden(*args, **kwargs):
        raise RuntimeError('Phase A forbids optimizer creation and checkpoint writes')
    torch.optim.Optimizer.__init__ = forbidden
    torch.save = forbidden


def reference_sampling(node, following_step, rank):
    parent = ('hns-s12-uniform-e1-1217-v1/step1217' if node==500 else
              'hns-s12-uniform-full4868-v1/step2434')
    with (RUNTIME/parent/'training/steps.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['step']==following_step:
                return next(h['sampling'] for h in row['rank_health'] if h['rank']==rank)
    raise RuntimeError('Missing original following-step sampling reference')


def load_probe(node, local):
    path, expected = NODES[node]
    assert sha(path)==expected, 'Checkpoint identity mismatch'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg=payload['config']
    assert payload['completed_steps']==node and payload['scheduler_horizon']==4868
    assert cfg['code_sha256']==code_manifest(), 'Production code manifest drift'
    assert (cfg['lambda_align'],cfg['lambda_sparse'],cfg['lambda_hierarchy'])==(10.,1.2,1.)
    assert cfg['view_sparsity_weights']==[5/3]*3 and cfg['view_weights']==[1.35,1.35,.3]
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['batch_size']==256
    assert cfg['checkpoint_encoders'] and cfg['workers']==8 and cfg['sampling_mode']=='nested_detail_d3'
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={node}
    assert payload['next_epoch']==node//1217 and payload['next_batch']==node%1217
    assert len(payload['rng_per_rank'])==4
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(payload['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),hns_enabled=True,
        view_sparsity_weights=cfg['view_sparsity_weights'],inclusion_hierarchy=cfg['inclusion_hierarchy'],
        fusion=cfg['fusion'],visual=cfg['visual'],condition_mode=cfg['condition_mode'],
        checkpoint_encoders=cfg['checkpoint_encoders'],image_chunk=cfg['image_chunk'],
        text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
        checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    model.fusion_branch.load_state_dict(payload['adapter'],strict=True)
    states=payload['rng_per_rank']; del payload
    model=model.cuda(local).train();model.capture_hns_graph=True
    return model,cfg,states


def matched_batch(model, batch, node, label, output, peers):
    rank=dist.get_rank();world=dist.get_world_size();device=next(model.parameters()).device
    input_keys=('image','tokens_f','tokens_o','tokens_e','valid')
    summaries=[None]*world
    dist.all_gather_object(summaries,dict(rank=rank,sampling=serialized_sampling(batch),
        image_sha256=tensor_hash(batch['image']),input_tensor_sha256={k:tensor_hash(batch[k]) for k in input_keys}))
    tensors=[batch[k].to(device,non_blocking=True) for k in input_keys]
    input_before=[tensor_hash(t) for t in tensors]
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for _,p in named]
    structure=[(n,list(p.shape),str(p.dtype),p.requires_grad,id(p)) for n,p in model.named_parameters()]
    before=state_digest(model.state_dict())
    hashes=[None]*world;dist.all_gather_object(hashes,before);assert len(set(hashes))==1
    original_branch=model.fusion_branch.encode_visual
    original_visual=model.encode_visual;original_view=model.encode_view
    shared_rng=rng_state();versions={};snapshots={};resources={};routes={}
    for variant in ('A','B'):
        restore_rng_state(shared_rng);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();started=time.perf_counter();captured={};source={};view_counter=[0]
        def branch_encode(branch, hidden):
            source['patch']=hidden
            return original_branch(hidden) if variant=='A' else visual_gradient_encode(branch,hidden)
        def visual_encode(module, images):
            z,condition=original_visual(images)
            captured['CLIP_global_image_embedding']=z.detach().cpu().clone()
            for i,value in enumerate(condition):captured['visual_condition_'+str(i)]=value.detach().cpu().clone()
            return z,condition
        def view_encode(module, tokens):
            text,condition=original_view(tokens);view=('F','Dall','D3')[view_counter[0]];view_counter[0]+=1
            captured['CLIP_text_embedding_'+view]=text.detach().cpu().clone()
            return text,condition
        def adapter_hook(module, arguments, result):captured['visual_adapter_tokens']=result.detach().cpu().clone()
        hook=model.fusion_branch.visual_adapter.register_forward_hook(adapter_hook)
        model.fusion_branch.encode_visual=types.MethodType(branch_encode,model.fusion_branch)
        model.encode_visual=types.MethodType(visual_encode,model);model.encode_view=types.MethodType(view_encode,model)
        try:
            loss,logs=model(*tensors,node)
            graph=model.hns_graph
            for kind in ('probabilities','masks'):
                for view,value in graph[kind].items():captured[kind+'_'+view]=value.detach().cpu().clone()
            for key in ('raw_view_F','raw_view_Dall','raw_view_D3','raw_sparse','raw_hierarchy',*COMPONENTS):
                captured[key]=graph[key].detach().cpu().clone()
            if variant=='B':
                assert all(compare_tensor(snapshots['A'][k],captured[k])['exact'] for k in captured), \
                    'Non-equivalent forwards; STOP before B gradient interpretation'
            snapshot_rng=rng_state()
            gradients_by_component={};route={}
            for key in COMPONENTS:
                print(json.dumps(dict(event='gradient_component',node=node,batch=label,variant=variant,component=key,rank=rank)),flush=True)
                gradients=torch.autograd.grad(graph[key],[*params,source['patch']],allow_unused=True,retain_graph=True)
                reduced=[]
                for parameter,gradient in zip(params,gradients[:-1]):
                    value=torch.zeros_like(parameter) if gradient is None else gradient.detach().float().clone()
                    assert torch.isfinite(value).all(), 'Nonfinite gradient'
                    dist.all_reduce(value);value/=world
                    assert torch.isfinite(value).all()
                    reduced.append(value.cpu())
                patch_grad=gradients[-1]
                patch_norm=0. if patch_grad is None else float(patch_grad.float().norm())
                assert patch_grad is None or torch.isfinite(patch_grad).all()
                norms=[None]*world;dist.all_gather_object(norms,patch_norm)
                route[key]=dict(local_patch_input_gradient_norms=norms,
                    patch_input_disconnected=all(n==0 for n in norms),
                    interpretation='Only derivative to the adapter-input Patch tensor, excludes existing CLS path')
                gradients_by_component[key]=reduced
                del gradients,reduced,patch_grad
            versions[variant]=gradients_by_component;snapshots[variant]=captured;routes[variant]=route
            after=state_digest(model.state_dict());assert after==before
            assert all(p.grad is None for p in model.parameters())
            assert [(n,list(p.shape),str(p.dtype),p.requires_grad,id(p)) for n,p in model.named_parameters()]==structure
            torch.cuda.synchronize()
            resource=dict(seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),state_digest_before=before,state_digest_after=after,
                local_rank=rank,global_batch=world*len(batch['sample_id']),parameters_unchanged=True)
            gathered=[None]*world;dist.all_gather_object(gathered,resource);resources[variant]=gathered
            if variant=='A':a_rng=snapshot_rng
            else:
                assert torch.equal(a_rng['cpu'],snapshot_rng['cpu']) and torch.equal(a_rng['cuda'],snapshot_rng['cuda'])
                assert a_rng['python']==snapshot_rng['python']
        finally:
            hook.remove();model.fusion_branch.encode_visual=original_branch
            model.encode_visual=original_visual;model.encode_view=original_view
        del graph,loss,source;model.hns_graph={};gc.collect()
    assert [tensor_hash(t) for t in tensors]==input_before
    equivalence={k:compare_tensor(snapshots['A'][k],snapshots['B'][k]) for k in snapshots['A']}
    assert all(x['exact'] for x in equivalence.values()), 'Non-equivalent forwards; STOP'
    equivalences=[None]*world;dist.all_gather_object(equivalences,dict(rank=rank,tensors=equivalence))
    results={};layer_results={}
    for grouping,target in ((group,results),(layer,layer_results)):
        names={grouping(n) for n,_ in named}
        for name in sorted(names):
            positions=[i for i,(n,_) in enumerate(named) if grouping(n)==name]
            pair={v:{c:torch.cat([versions[v][c][i].flatten() for i in positions]) for c in COMPONENTS} for v in ('A','B')}
            info=summarize_vectors(pair['A'],pair['B']);target[name]=info
            if grouping==group and name!='native_visual_backbone':
                assert all(v==0 for v in info['component_delta_max_abs'].values()), ('Unexpected downstream change',name)
            del pair
    old=results['native_visual_backbone']
    assert old['A_component_norms']['weighted_sparse']==old['A_component_norms']['weighted_hierarchy']==0
    assert all(v['patch_input_disconnected'] for v in routes['A'].values())
    assert not routes['B']['total_training']['patch_input_disconnected']
    assert old['delta_g_norm']>0
    failures=additivity_failures(results)+additivity_failures(layer_results,layerwise=True)
    if rank==0:
        write_new(output/(label+'.json'),dict(passed=not failures,failed_acceptance=failures,node=node,batch=label,global_batch=world*len(batch['sample_id']),
            same_parameter_objects=True,same_inputs=True,same_RNG=True,forward=equivalences,
            group_gradients=results,layerwise_gradients=layer_results,patch_route=routes,
            resources=resources,input_audit=summaries,no_optimizer=True,no_parameter_updates=True,
            no_checkpoint_saved=True,source_manifest=code_manifest(),gpu_identity=peers))
        print(json.dumps(dict(event='batch_passed',node=node,batch=label,visual=old)),flush=True)
    if failures:raise AssertionError(('Component additivity',failures))
    del versions,snapshots,tensors;gc.collect();torch.cuda.empty_cache();dist.barrier()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-node',dest='node',required=True,type=int,choices=NODES)
    parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    deny_updates();seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,peers=setup();assert world==4
    if rank==0:args.output_dir.mkdir(parents=True,exist_ok=False)
    dist.barrier();model,cfg,states=load_probe(args.node,local)
    epoch,cursor=divmod(args.node,1217)
    dataset=FullLocalDataset('/root/said_s02_stage500/data_index','/root/said_s02_stage500/ShareGPT4V','nested_detail_d3',0)
    assert dataset.metadata['records_sha256']==cfg['data']['records_sha256'];dataset.set_epoch(epoch)
    sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(epoch)
    positions=list(itertools.islice(iter(sampler),cursor*256,(cursor+2)*256));assert len(positions)==512
    generator=torch.Generator();generator.set_state(states[rank]['loader_generator'])
    loader=DataLoader(dataset,batch_size=256,sampler=positions,collate_fn=collate,num_workers=8,
        drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,generator=generator,timeout=60)
    restore_rng_state(states[rank]);batches=list(loader)
    for offset,batch in enumerate(batches):
        assert serialized_sampling(batch)==reference_sampling(args.node,args.node+offset+1,rank), 'Next-step data flow mismatch'
    if rank==0:
        write_new(args.output_dir/'PLAN.json',dict(node=args.node,checkpoint_path=str(NODES[args.node][0]),
            checkpoint_sha256=NODES[args.node][1],preflight_global_batch=16,formal_global_batch=1024,
            fixed_following_steps=[args.node+1,args.node+2],checkpoint_parameters_frozen=True,
            completed_updates_argument=args.node,production_sources_unchanged=True,no_optimizer=True,
            encoder_precision='native BF16 autocast; mask/fusion FP32; no external autocast',
            four_rank_gradients='native differentiable all_gather plus mean all_reduce, same as existing audit',
            original_sampling_summaries_match=True,
            image_augmentation_note='Native transform and restored worker generator on isolated suffix. Pixels regenerated; historical augmentation pixels not claimed identical. A/B share exact pixels.',
            additivity_tolerances=dict(BF16_native_backbone=BF16_ADDITIVITY_TOL,FP32_downstream=FP32_ADDITIVITY_TOL)))
    try:
        matched_batch(model,slice_batch(batches[0],4),args.node,'preflight16',args.output_dir,peers)
        for offset,batch in enumerate(batches):
            matched_batch(model,batch,args.node,'global1024_next'+str(offset+1),args.output_dir,peers)
    except BaseException:
        # Never retries. Preserve failure artifacts and release NCCL/GPU state.
        dist.destroy_process_group()
        raise
    del model;gc.collect();torch.cuda.empty_cache()
    path,expected=NODES[args.node];assert sha(path)==expected
    if rank==0:write_new(args.output_dir/'COMPLETED.json',dict(passed=True,checkpoint_unchanged=True,node=args.node,
        completed_batches=['preflight16','global1024_next1','global1024_next2'],no_updates=True))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
