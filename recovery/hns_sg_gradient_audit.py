"""Read-only same-batch AND same-graph SG/no-SG audit at both500 checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from model.hard_nested_sparsity import hns_terms
from recovery.hns_gradient_audit import cosine
from recovery.hns_sg500 import BASE_RUN
from recovery.s02_nfs500 import ROOT, dump, sha, now
from recovery.s02_local500 import INDEX, IMAGES
from recovery.s02_full_local_data import FullLocalDataset
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import CappedSampler, setup, seed_all, state_digest


def grads(value,tensors):
    found=torch.autograd.grad(value,tensors,allow_unused=True,retain_graph=True)
    return [torch.zeros_like(p) if g is None else g for p,g in zip(tensors,found)]


def endpoint(value,graph,parent,child,world):
    result={}
    for label,tensors in [('HardST_output',graph['masks']),('soft_probability',graph['probabilities'])]:
        gp,gc=grads(value,[tensors[parent],tensors[child]])
        values=torch.stack([gp.square().sum(),gc.square().sum(),(gp>0).sum(),(gp<0).sum(),
            (gc>0).sum(),(gc<0).sum()]).float()
        dist.all_reduce(values)
        result[label]=dict(parent=parent,child=child,parent_norm=float(values[0].sqrt()/world),
            child_norm=float(values[1].sqrt()/world),child_gradient_exactly_zero=values[1].item()==0,
            parent_positive_count=int(values[2]),parent_negative_count=int(values[3]),
            child_positive_count=int(values[4]),child_negative_count=int(values[5]),
            parent_expansion_verified=int(values[2])==0 and int(values[3])>0,
            child_contraction_verified=int(values[5])==0 and int(values[4])>0)
    return result


def inspect_graph(module,graph,valid,world):
    names_params=[(n,p) for n,p in module.named_parameters() if p.requires_grad]
    parameters=[p for n,p in names_params]
    groups={
        'shared_visual_backbone':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.visual.')],
        'shared_text_backbone':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.') and not n.startswith(('clip.visual.','clip.mask_net.'))],
        'shared_text_mask_and_pool':[i for i,(n,p) in enumerate(names_params) if n.startswith('clip.mask_net.')],
        'shared_visual_mask':[i for i,(n,p) in enumerate(names_params) if n.startswith('fusion_branch.visual_blocks.')],
        'fusion_shared_module':[i for i,(n,p) in enumerate(names_params) if n.startswith('fusion_branch.') and not n.startswith('fusion_branch.visual_blocks.')],
    }
    assert sorted(i for g in groups.values() for i in g)==list(range(len(parameters)))
    def vectors(value):
        gradients=grads(value,parameters)
        out={}
        for group,indices in groups.items():
            vector=torch.cat([gradients[i].float().flatten() for i in indices])
            dist.all_reduce(vector);vector/=world
            assert torch.isfinite(vector).all();out[group]=vector
        return out
    base={n:vectors(graph[n]) for n in ('alignment','original_sparsity')}
    valid_count=valid.sum().float();dist.all_reduce(valid_count)
    masks=graph['masks'];modes={};scalar=[];reference_parent={}
    for name,sg in [('HNS-v1',False),('HNS-SG',True)]:
        hierarchy,terms=hns_terms(masks['F'],masks['Dall'],masks['D3'],valid,499,world,valid_count,detach_child=sg)
        values=dict(alignment=graph['alignment'],original_sparsity=graph['original_sparsity'],
            V_DF=terms['V_DF'],V_3D=terms['V_3D'],total_hierarchy=hierarchy,
            total_training=graph['alignment']+graph['original_sparsity']+hierarchy)
        scalar.append(torch.stack([v.detach() for v in values.values()]))
        endpoint_stats={}
        for edge,p,c in [('V_DF','F','Dall'),('V_3D','Dall','D3')]:
            gp,gc=grads(values[edge],[masks[p],masks[c]])
            if sg:
                assert torch.count_nonzero(gc)==0
                torch.testing.assert_close(gp,reference_parent[edge],atol=0,rtol=0)
            else:reference_parent[edge]=gp.clone()
            endpoint_stats[edge]=endpoint(values[edge],graph,p,c,world)
            d=endpoint_stats[edge]['HardST_output']
            assert d['parent_expansion_verified']
            assert d['child_gradient_exactly_zero'] if sg else d['child_contraction_verified']
        all_vectors=dict(base)
        all_vectors.update({k:vectors(values[k]) for k in ('V_DF','V_3D','total_hierarchy','total_training')})
        for group in groups:
            torch.testing.assert_close(all_vectors['total_hierarchy'][group],
                (2*all_vectors['V_DF'][group]+2*all_vectors['V_3D'][group])/3,atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(all_vectors['total_training'][group],
                base['alignment'][group]+base['original_sparsity'][group]+all_vectors['total_hierarchy'][group],atol=3e-5,rtol=3e-4)
        coefficient=dict(alignment=1.,original_sparsity=1.,V_DF=2/3,V_3D=2/3,total_hierarchy=1.,total_training=1.)
        components={}
        for component,v in all_vectors.items():
            out=grads(values[component],list(masks.values()))
            norms=torch.stack([g.square().sum() for g in out]);dist.all_reduce(norms)
            components[component]=dict(coefficient=coefficient[component],
                group_norms={g:dict(raw=float(a.norm()),weighted=float(a.norm())*coefficient[component]) for g,a in v.items()},
                positive_HardST_output_gradient_norms={label:float(n.sqrt()/world) for label,n in zip(masks,norms)})
        assert all(components['original_sparsity']['positive_HardST_output_gradient_norms'][v]>0 for v in masks)
        assert all(components[n]['group_norms'][g]['raw']==0 for n in ('original_sparsity','V_DF','V_3D','total_hierarchy') for g in ('shared_visual_backbone','shared_text_backbone'))
        conflict={g:dict(hierarchy_norm=float(a.norm()),
            cosine_hierarchy_original_sparsity=cosine(a,base['original_sparsity'][g]),
            cosine_hierarchy_alignment=cosine(a,base['alignment'][g])) for g,a in all_vectors['total_hierarchy'].items()}
        reduced=torch.stack([v.detach() for v in values.values()]);dist.all_reduce(reduced);reduced/=world
        modes[name]=dict(detach_child=sg,forward_values={k:float(v) for k,v in zip(values,reduced)},
            components=components,group_cosines=conflict,endpoint_gradient_direction=endpoint_stats,
            gradient_additivity_passed=True,parent_endpoint_gradient_exact_to_no_SG=True if sg else None,
            output_scope_note='Alignment uses separate all-pairs masks sharing parameters, so zero alignment gradient at these positive outputs does not mean no child alignment gradient.')
        del all_vectors
    torch.testing.assert_close(scalar[0],scalar[1],atol=0,rtol=0)
    return modes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True);args=parser.parse_args()
    started=now();seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    loader=DataLoader(dataset,batch_size=256,sampler=CappedSampler(sampler,256),collate_fn=collate,
        num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(0),timeout=60)
    batch=next(iter(loader));batch_ids=[None]*world;dist.all_gather_object(batch_ids,batch['sample_id'].tolist())
    ids_digest=hashlib.sha256(json.dumps(batch_ids,separators=(',',':')).encode()).hexdigest()
    images=batch['image'].cuda(non_blocking=True);views=[batch[k].cuda(non_blocking=True) for k in ('tokens_f','tokens_o','tokens_e')]
    valid=batch['valid'].cuda();results={}
    for name,run in [('HNS-v1',BASE_RUN),('HNS-SG',args.run)]:
        checkpoint=run/'step500/step000500.pt';checkpoint_sha=sha(checkpoint)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['completed_steps']==500 and payload['scheduler_horizon']==4868
        cfg=payload['config'];assert cfg['inclusion_max']==0 and cfg['sampling_mode']=='nested_detail_d3'
        assert cfg.get('hns_detach_child',False)==(name=='HNS-SG')
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        clip.load_state_dict(payload['model'],strict=True)
        module=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),
            inclusion_hierarchy='detail_chain',hns_enabled=True,hns_detach_child=name=='HNS-SG',fusion=cfg['fusion'],visual=cfg['visual'],
            condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
            image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
            checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
        module.fusion_branch.load_state_dict(payload['adapter'],strict=True);del payload
        module=module.cuda(local).train();module.capture_hns_graph=True
        before=state_digest(module.state_dict()) if rank==0 else None
        loss,logs=module(images,*views,valid,499);graph=module.hns_graph
        modes=inspect_graph(module,graph,valid,world)
        assert sha(checkpoint)==checkpoint_sha and all(p.grad is None for p in module.parameters())
        if rank==0:
            assert state_digest(module.state_dict())==before
            telemetry={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items() if k.startswith('HNS_') or k in ('V_DF_hard','V_3D_hard','lambda_h')}
            results[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_unchanged=True,
                state_digest_before=before,state_digest_after=before,telemetry=telemetry,modes=modes)
        del module,clip,graph,logs,loss;torch.cuda.empty_cache();dist.barrier()
    if rank==0:
        old=json.loads((ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1/GRADIENT_AUDIT.json').read_text())
        assert ids_digest==old['sample_ids_sha256']
        value=dict(passed=True,records=1024,sample_ids_sha256=ids_digest,checkpoints=results,
            protocol='First frozen seed0 epoch0 global1024 batch, immutable step500; SG/no-SG recomputed on IDENTICAL actual positive mask graph at EACH checkpoint. No optimizer update.',
            same_graph_forward_exact=True,child_hierarchy_gradient_exactly_zero=True,
            parent_hierarchy_gradient_exact_to_no_SG=True,original_sparsity_child_gradient_nonzero=True,
            alignment_sparsity_unmodified=True,global_valid_normalization=True,no_double_world_scaling=True,
            local_only=True,NFS_fallback=False,no_parameter_updates=True,
            shared_parameter_caveat='Dall is child on DF but parent on3D. Total hierarchy Dall gradient need not be zero. Shared parameter updates may indirectly change any view.',
            native_backbone_note='Original hidden-input detach preserved; hierarchy/sparse have zero native-backbone gradient, alignment remains active.',
            started_utc=started,ended_utc=now())
        dump(args.experiment/'GRADIENT_AUDIT.json',value);print(json.dumps(dict(passed=True,output=str(args.experiment/'GRADIENT_AUDIT.json'))),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
