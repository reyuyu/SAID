"""Immutable step500 matched global1024 macro-component parameter gradients."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from model import longclip
from model.balanced_hparam_search import BalancedSearch,hparams,macro_hparams
from recovery.s02_nfs500 import dump,sha,now
from recovery.s02_local500 import INDEX,IMAGES
from recovery.s02_full_local_data import FullLocalDataset
from train.train_nested_semantic_mask import CappedSampler,setup,seed_all,state_digest
from train.nested_semantic_data import collate


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return None if not float(denominator) else float(torch.dot(a,b)/denominator)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=now();seed_all(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    digest=sha(args.checkpoint);p=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert p['completed_steps']==500 and p['scheduler_horizon']==4868
    cfg=p['config'];hp=macro_hparams(cfg)
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(p['model'],strict=True)
    model=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(dict(cfg,**hp)),hns_enabled=cfg.get('hns_enabled',False),
        inclusion_hierarchy='detail_chain',fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
        image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],
        checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    model.fusion_branch.load_state_dict(p['adapter'],strict=True);del p
    model=model.cuda(local).train();model.capture_hns_graph=True
    before=state_digest(model.state_dict()) if rank==0 else None
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    batch=list(DataLoader(dataset,batch_size=256,sampler=CappedSampler(sampler,256),collate_fn=collate,
        num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(0),timeout=60))[0]
    ids=[None]*world;dist.all_gather_object(ids,batch['sample_id'].tolist())
    ids_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    _,logs=model(*[batch[k].cuda(local,non_blocking=True) for k in ('image','tokens_f','tokens_o','tokens_e','valid')],499)
    graph=model.hns_graph;named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    def group(name):
        if name.startswith('clip.visual.'):return 'native_visual_backbone'
        if name.startswith('clip.mask_net.'):return 'text_mask_shared_pool'
        if name.startswith('clip.'):return 'native_text_backbone'
        if name.startswith('fusion_branch.visual_blocks.'):return 'visual_mask'
        return 'fusion_adapter_gate'
    indices={}
    for i,(name,p) in enumerate(named):indices.setdefault(group(name),[]).append(i)
    vectors={};components={};coeff={'raw_align':hp['lambda_align'],'raw_sparse':hp['lambda_sparse'],
        'raw_hierarchy':hp['lambda_hierarchy']*graph['lambda_h'],
        'weighted_align':1.,'weighted_sparse':1.,'weighted_hierarchy':1.,'total_training':1.}
    for name,w in zip(('F','Dall','D3'),cfg['view_weights']):
        coeff['raw_view_'+name]=hp['lambda_align']*w/sum(cfg['view_weights'])
        coeff['weighted_view_'+name]=1.
    for key,weight in coeff.items():
        gradients=torch.autograd.grad(graph[key],params,allow_unused=True,retain_graph=True)
        reduced=[]
        for param,gradient in zip(params,gradients):
            v=torch.zeros_like(param) if gradient is None else gradient.detach().float().clone()
            dist.all_reduce(v);v/=world;assert torch.isfinite(v).all();reduced.append(v.flatten())
        vectors[key]={g:torch.cat([reduced[i] for i in ii]) for g,ii in indices.items()}
        components[key]=dict(coefficient=weight,group_norms={g:dict(raw=float(v.norm()),linear_scaling_estimate=float(v.norm())*weight)
            for g,v in vectors[key].items()})
        del gradients,reduced
    conflicts={};additivity={}
    for g in indices:
        a,s,h=[vectors[k][g] for k in ('raw_align','raw_sparse','raw_hierarchy')]
        weighted_a,weighted_s,weighted_h=[vectors[k][g] for k in ('weighted_align','weighted_sparse','weighted_hierarchy')]
        total=vectors['total_training'][g];summed=weighted_a+weighted_s+weighted_h
        error=float((total-summed).norm())/max(float(total.norm()),1e-12)
        tolerance=3e-4
        assert error<=tolerance,(g,error,tolerance)
        additivity[g]=dict(relative_L2_error=error,tolerance=tolerance,
            precision_note='Sum of independently differentiated actual weighted components, not raw-gradient scaling estimates')
        conflicts[g]=dict(cosine_hierarchy_sparsity=cosine(h,s),cosine_hierarchy_alignment=cosine(h,a),
            raw_alignment_norm=float(a.norm()),raw_sparsity_norm=float(s.norm()),raw_hierarchy_norm=float(h.norm()),
            weighted_alignment_norm=float(weighted_a.norm()),weighted_sparsity_norm=float(weighted_s.norm()),
            weighted_hierarchy_norm=float(weighted_h.norm()),total_norm=float(total.norm()))
        for raw,weighted in [('raw_align','weighted_align'),('raw_sparse','weighted_sparse'),('raw_hierarchy','weighted_hierarchy')]:
            components[raw]['group_norms'][g]['weighted']=float(vectors[weighted][g].norm())
            components[raw]['group_norms'][g]['weighted_method']='autograd of actual weighted loss'
    for g in ('native_visual_backbone','native_text_backbone'):
        assert conflicts[g]['raw_sparsity_norm']==conflicts[g]['raw_hierarchy_norm']==0
    views={g:{name:dict(raw_norm=float(vectors['raw_view_'+name][g].norm()),
        weighted_norm=float(vectors['weighted_view_'+name][g].norm())) for name in ('F','Dall','D3')} for g in indices}
    for g in indices:
        a,d,k=[vectors['weighted_view_'+v][g] for v in ('F','Dall','D3')]
        views[g].update(weighted_D3_Dall_ratio=None if not float(d.norm()) else float(k.norm()/d.norm()),
            cosine_F_Dall=cosine(a,d),cosine_Dall_D3=cosine(d,k),cosine_F_D3=cosine(a,k))
    assert all(p.grad is None for p in model.parameters()) and sha(args.checkpoint)==digest
    if rank==0:
        assert state_digest(model.state_dict())==before
        dump(args.output,dict(passed=True,protocol='Original immutable step500 first seed0 epoch0 global1024 batch;256/rank; mean all-reduced gradients; no optimizer step',
            sample_ids_sha256=ids_sha,records=1024,completed=499,ramp=graph['lambda_h'],macro_scales=hp,
            components=components,group_diagnostics=conflicts,gradient_additivity=additivity,view_gradients=views,
            checkpoint=dict(path=args.checkpoint,sha256=digest,unchanged=True,uploaded=False),
            no_parameter_updates=True,state_digest_before=before,state_digest_after=before,
            local_only=True,NFS_fallback=False,telemetry={k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()
                if k.startswith(('HNS_','macro_')) or k in ('V_DF_hard','V_3D_hard','lambda_h',
                    'inc','inc_weight','inclusion_loss','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation')},
            hierarchy_method='HNS' if cfg.get('hns_enabled',False) else 'Balanced',
            inherited_hidden_detach='Regularizers have zero native-backbone gradient; '+
                ('no mask-to-mask stop-gradient' if cfg.get('hns_enabled',False) else 'soft detached-child inclusion'),
            started_utc=started,ended_utc=now()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
