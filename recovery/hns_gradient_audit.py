"""Immutable real-batch HNS decomposition and matched INC0 hierarchy telemetry."""
import argparse
import hashlib
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch,hparams
from model.nested_semantic_mask import global_sum
from model.hard_nested_sparsity import hard_telemetry
from recovery.s02_nfs500 import ROOT,dump,sha,now
from recovery.s02_local500 import INDEX,IMAGES
from recovery.s02_full_local_data import FullLocalDataset
from recovery.hns_preflight import BASE_RUN,EXP
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import CappedSampler,setup,seed_all,state_digest


def cosine(a,b):
    product=a.norm()*b.norm()
    return None if product.item()==0 else float(torch.dot(a,b)/product)


def output_direction(graph,name,parent_name,child_name,world):
    masks=graph['masks'];prob=graph['probabilities'];value=graph[name]
    result={}
    for label,tensors in [('HardST_output',masks),('soft_probability',prob)]:
        gp,gc=torch.autograd.grad(value,(tensors[parent_name],tensors[child_name]),retain_graph=True)
        stats=torch.stack((gp.square().sum(),gc.square().sum(),
            (gp>0).sum(),(gp<0).sum(),(gc>0).sum(),(gc<0).sum(),
            gp.min(),gp.max(),gc.min(),gc.max())).float()
        # Endpoint gradients of global mean = local surrogate gradients/world.
        norms=stats[:2].clone();dist.all_reduce(norms)
        norm_p,norm_c=(norms.sqrt()/world).tolist()
        counts=stats[2:6].clone();dist.all_reduce(counts)
        result[label]=dict(parent=parent_name,child=child_name,parent_norm=norm_p,child_norm=norm_c,
            parent_child_norm_ratio=None if norm_c==0 else norm_p/norm_c,
            parent_positive_count=int(counts[0]),parent_negative_count=int(counts[1]),
            child_positive_count=int(counts[2]),child_negative_count=int(counts[3]),
            gradient_descent_parent_expansion_verified=int(counts[0])==0 and int(counts[1])>0,
            gradient_descent_child_contraction_verified=int(counts[3])==0 and int(counts[2])>0)
        assert int(counts[0])==int(counts[3])==0
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-hierarchy',action='store_true')
    parser.add_argument('--run',type=str)
    parser.add_argument('--experiment',type=str)
    args=parser.parse_args();started=now();seed_all(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    run=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS' if args.run is None else __import__('pathlib').Path(args.run)
    experiment=EXP if args.experiment is None else __import__('pathlib').Path(args.experiment)
    checkpoint=(BASE_RUN if args.baseline_hierarchy else run)/'step500/step000500.pt'
    checkpoint_sha=sha(checkpoint);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['completed_steps']==500 and payload['scheduler_horizon']==4868
    cfg=payload['config'];assert cfg['inclusion_max']==0 and cfg['sampling_mode']=='nested_detail_d3'
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(payload['model'],strict=True)
    module=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),
        inclusion_hierarchy='detail_chain',hns_enabled=True,hns_beta=cfg.get('hns_beta',[2.,2.]),fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
        image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
        checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    module.fusion_branch.load_state_dict(payload['adapter'],strict=True);del payload
    module=module.cuda(local).train();module.capture_hns_graph=True
    before=state_digest(module.state_dict()) if rank==0 else None
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    loader=DataLoader(dataset,batch_size=256,sampler=CappedSampler(sampler,256),collate_fn=collate,
        num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(0),timeout=60)
    batch=list(loader)[0];batch_ids=[None]*world;dist.all_gather_object(batch_ids,batch['sample_id'].tolist())
    ids_digest=hashlib.sha256(json.dumps(batch_ids,separators=(',',':')).encode()).hexdigest()
    images=batch['image'].cuda(non_blocking=True);views=[batch[k].cuda(non_blocking=True) for k in ('tokens_f','tokens_o','tokens_e')]
    valid=batch['valid'].cuda();components={};endpoint={}
    if args.baseline_hierarchy:
        with torch.no_grad():_,logs=module(images,*views,valid,499)
    else:
        _,logs=module(images,*views,valid,499);graph=module.hns_graph
        names_params=[(n,p) for n,p in module.named_parameters() if p.requires_grad]
        parameters=[p for n,p in names_params]
        groups={
            'shared_visual_backbone':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.visual.')],
            'shared_text_backbone':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.') and not n.startswith(('clip.visual.','clip.mask_net.'))],
            'shared_text_mask_and_pool':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.mask_net.')],
            'shared_visual_mask':[i for i,(n,p) in enumerate(names_params) if n.startswith('fusion_branch.visual_blocks.')],
            'fusion_shared_module':[i for i,(n,p) in enumerate(names_params) if n.startswith('fusion_branch.') and not n.startswith('fusion_branch.visual_blocks.')],
        }
        assert sorted(i for group in groups.values() for i in group)==list(range(len(parameters)))
        vectors={}
        coefficients=dict(alignment=1.,original_sparsity=1.,V_DF=graph['beta'][0]/3*graph['lambda_h'],V_3D=graph['beta'][1]/3*graph['lambda_h'],total_HNS=1.,total_training=1.)
        for name in coefficients:
            gradients=torch.autograd.grad(graph[name],parameters,allow_unused=True,retain_graph=True)
            vector=torch.cat([torch.zeros_like(p).flatten() if g is None else g.float().flatten() for p,g in zip(parameters,gradients)])
            dist.all_reduce(vector);vector/=world;assert torch.isfinite(vector).all()
            offsets=[];cursor=0
            for p in parameters:offsets.append(cursor);cursor+=p.numel()
            vectors[name]={group:torch.cat([vector[offsets[i]:offsets[i]+parameters[i].numel()] for i in indices]) for group,indices in groups.items()}
            components[name]=dict(coefficient=coefficients[name],group_norms={group:dict(raw=float(v.norm()),weighted=float(v.norm())*coefficients[name]) for group,v in vectors[name].items()})
            outputs=list(graph['masks'].values())
            output_grads=torch.autograd.grad(graph[name],outputs,allow_unused=True,retain_graph=True)
            norm_sums=torch.stack([torch.zeros((),device='cuda') if g is None else g.square().sum() for g in output_grads])
            dist.all_reduce(norm_sums)
            components[name]['positive_HardST_output_gradient_norms']={view:dict(raw=float(norm)/world,
                weighted=float(norm)/world*coefficients[name],connected=gradient is not None)
                for view,norm,gradient in zip(graph['masks'],norm_sums.sqrt(),output_grads)}
            components[name]['output_scope_note']='These are actual paired positive output tensors. CE uses a separate all-pairs mask computation sharing parameters, so disconnected alignment at these tensors does not imply absence of per-view alignment gradient.'
            del gradients,vector
        hierarchy={group:coefficients['V_DF']*vectors['V_DF'][group]+coefficients['V_3D']*vectors['V_3D'][group] for group in groups}
        conflict={group:dict(hierarchy_norm=float(v.norm()),
            cosine_hierarchy_original_sparsity=cosine(v,vectors['original_sparsity'][group]),
            cosine_hierarchy_alignment=cosine(v,vectors['alignment'][group])) for group,v in hierarchy.items()}
        for name in components:
            components[name]['group_cosines']={group:dict(
                with_alignment=cosine(v,vectors['alignment'][group]),
                with_original_sparsity=cosine(v,vectors['original_sparsity'][group]))
                for group,v in vectors[name].items()}
        for group in groups:
            torch.testing.assert_close(vectors['total_HNS'][group],vectors['original_sparsity'][group]+hierarchy[group],atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(vectors['total_training'][group],vectors['alignment'][group]+vectors['total_HNS'][group],atol=3e-5,rtol=3e-4)
        endpoint={name:output_direction(graph,name,p,c,world) for name,p,c in [('V_DF','F','Dall'),('V_3D','Dall','D3')]}
        assert all(components[name]['group_norms'][group]['raw']==0 for name in ('original_sparsity','V_DF','V_3D','total_HNS') for group in ('shared_visual_backbone','shared_text_backbone'))
    assert sha(checkpoint)==checkpoint_sha and all(p.grad is None for p in module.parameters())
    if rank==0:
        assert state_digest(module.state_dict())==before
        telemetry={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items() if k.startswith('HNS_') or k in ('V_DF_hard','V_3D_hard','lambda_h')}
        value=dict(passed=True,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_unchanged=True,
            no_parameter_updates=True,state_digest_before=before,state_digest_after=before,completed_steps=500,
            protocol='First frozen seed0 epoch0 global1024 batch at immutable step500;256/rank; no optimizer step',
            sample_ids_sha256=ids_digest,records=1024,local_only=True,NFS_fallback=False,
            telemetry=telemetry,components=components,endpoint_gradient_direction=endpoint,
            shared_branch_note='F/Dall/D3 share mask parameters. Separate view branches below mean output tensors; parameter groups are shared, never fabricated private parameters.',
            native_backbone_regularizer_gradient_note='INC0 architecture detaches native token-hidden inputs to mask modules. Hierarchy and global sparsity therefore have zero native visual/text backbone gradient; inherited detach preserved. Alignment updates native backbones.',
            pairwise_equality_definition='Fraction of valid pairs with entire512-dimensional binary masks exactly equal; coordinate equality reported separately',
            started_utc=started,ended_utc=now())
        if not args.baseline_hierarchy:value.update(group_cosines=conflict,gradient_additivity_passed=True)
        destination=experiment/('INC0_HIERARCHY_REFERENCE.json' if args.baseline_hierarchy else 'GRADIENT_AUDIT.json')
        if not args.baseline_hierarchy:
            old=json.loads((experiment/'INC0_HIERARCHY_REFERENCE.json').read_text())
            assert old['sample_ids_sha256']==ids_digest
            value['matched_cohort_delta_vs_INC0']={k:telemetry[k]-old['telemetry'][k] for k in telemetry if isinstance(telemetry[k],(int,float)) and k in old['telemetry']}
            value['beta']=list(graph['beta'])
            reference=experiment/'HNS_V1_GRADIENT_REFERENCE.json'
            if reference.is_file():
                baseline=json.loads(reference.read_text());assert baseline['sample_ids_sha256']==ids_digest
                value['matched_cohort_delta_vs_HNS_v1']={k:telemetry[k]-baseline['telemetry'][k] for k in telemetry if isinstance(telemetry[k],(int,float)) and k in baseline['telemetry']}
                value['hierarchy_gradient_norm_ratio_vs_HNS_v1']={g:None if not b['hierarchy_norm'] else conflict[g]['hierarchy_norm']/b['hierarchy_norm'] for g,b in baseline['group_cosines'].items()}
                value['weighted_inner_outer_gradient_norm_ratio']={g:None if not components['V_DF']['group_norms'][g]['weighted'] else components['V_3D']['group_norms'][g]['weighted']/components['V_DF']['group_norms'][g]['weighted'] for g in conflict}
        dump(destination,value);print(json.dumps(dict(passed=True,baseline=args.baseline_hierarchy,output=str(destination))),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
