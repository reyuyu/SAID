"""Read-only spot8: views, actual alignment and native gradient only, no updates."""
import gc
import gzip
import hashlib
import json
import math
import statistics
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from train.nested_semantic_data import NestedDataset,collate,file_sha
from train.train_nested_semantic_mask import CappedSampler,state_digest,setup,parameter_agreement
from experiments.nest_clip_v1.gradient_composition_audit_v1 import run_audit as audit
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import AuditObjective
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import GROUPS,gram_matrices,linearity_errors
from experiments.nest_clip_v1.armb_summary04_500_v1.run import EXP,RUN,SOURCE,WEIGHTS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records

OBJECTIVES=['F_combined','O_combined','E_combined','align_actual','native_combined']


def metrics(grams):
    out={};indexes={key:i for i,key in enumerate(OBJECTIVES)}
    for group,G in grams.items():
        norm={key:math.sqrt(max(0,G[i][i])) for key,i in indexes.items()}
        def pair(key):
            i,j=indexes[key],indexes['native_combined'];dot=G[i][j];na,nb=norm[key],norm['native_combined']
            return dict(cosine=dot/(na*nb) if na>0 and nb>0 else None,projection=dot/(nb*nb) if nb>0 else None,dot=dot)
        weighted={key:w*norm[key] for key,w in zip(OBJECTIVES[:3],WEIGHTS)};denom=sum(weighted.values())
        out[group]=dict(norms=norm,native_alignment={key:pair(key) for key in OBJECTIVES[:4]},weighted_norms=weighted,
          effective_norm_share={key:v/denom if denom>0 else None for key,v in weighted.items()})
    return out


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,hardware=setup();calls=audit.guard_updates()
    result=load(EXP/'RESULTS.json');ckpt=result['checkpoint'];sha=result['checkpoint_sha256']
    stage=dict(checkpoint=ckpt,sha256=sha,mode='summary_random_detail',weights=WEIGHTS,config=result['config'],expected_updates=500)
    if rank==0:assert file_sha(ckpt)==sha
    model,named,identity=audit.build_model(stage);assert parameter_agreement(model)==0
    immutable=state_digest(model.state_dict());ddp=DDP(AuditObjective(model),device_ids=[local],output_device=local,find_unused_parameters=True)
    ds=NestedDataset(audit.INDEX,audit.IMAGE_ROOT,'summary_random_detail',0);ds.set_epoch(0)
    sampler=DistributedSampler(ds,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    loader=DataLoader(ds,batch_size=256,sampler=CappedSampler(sampler,8*256),collate_fn=collate,num_workers=8,pin_memory=True,
      multiprocessing_context='spawn',prefetch_factor=2,generator=torch.Generator().manual_seed(0))
    prior_inputs=load(audit.EXP/'evidence/batches_B.json')['batches'][:8]
    prior_raw_path=audit.RUN/'stage_B/batch_stats.jsonl'
    old_raw=records(prior_raw_path)[:8]
    rows=[];proofs=[]
    for index,batch in enumerate(loader):
        started=time.monotonic();before=state_digest(model.state_dict());assert before==immutable
        local_meta=audit.batch_provenance(batch,rank);peers=[None]*4;dist.all_gather_object(peers,local_meta)
        for a,b in zip(peers,prior_inputs[index]['ranks']):
            assert a['image_tensor_sha256']==b['image_tensor_sha256']
            for key in ['full_view_sha256','local_views_sha256','split_sha256','sample_id_sha256']:assert a['sampling'][key]==b['sampling'][key]
        images=batch['image'].cuda(non_blocking=True);views=[batch[k].cuda(non_blocking=True) for k in ['tokens_f','tokens_o','tokens_e']];valid=batch['valid'].cuda(non_blocking=True)
        gradients={};agreements={};losses={}
        for objective in OBJECTIVES:
            ddp.zero_grad(set_to_none=True);value=ddp(images,*views,valid,objective,index==0 and objective=='native_combined');value.backward()
            agreements[objective]=audit.synchronized_gradient_agreement(named)
            gradients[objective]={name:p.grad.detach().clone() if p.grad is not None else None for name,p in named.items()}
            mean=value.detach().clone();dist.all_reduce(mean);losses[objective]=float(mean/4)
        if rank==0:
            grams,_=gram_matrices(named,gradients,OBJECTIVES)
            values=metrics(grams);linear=linearity_errors(named,gradients,WEIGHTS)
            row=dict(batch=index+1,groups=values,losses=losses,linearity=linear,Gram_objectives=OBJECTIVES,Gram_FP64=grams,
              native_correctness=ddp.module.last_checks,input_matches_old_ArmB_audit=True)
            rows.append(row)
        dist.barrier();del gradients;ddp.zero_grad(set_to_none=True)
        after=state_digest(model.state_dict());assert after==before==immutable
        states=[None]*4;dist.all_gather_object(states,dict(rank=rank,before=before,after=after,update_calls=dict(calls)))
        if rank==0:
            proofs.append(dict(batch=index+1,states=states,DDP_gradient_agreements=agreements))
            print(json.dumps({'spot_batch':index+1,'total':8,'seconds':round(time.monotonic()-started,2),'alignment_native_cosine':values['native_backbone_total']['native_alignment']['align_actual']['cosine'],'unchanged':True}),flush=True)
        del images,views,valid,batch;gc.collect()
    if rank==0:
        assert file_sha(ckpt)==sha
        groups={}
        for g in GROUPS:
            groups[g]={}
            for name in OBJECTIVES:
                vals=[r['groups'][g]['norms'][name] for r in rows];groups[g].setdefault('norm_mean',{})[name]=statistics.fmean(vals)
            groups[g]['actual_alignment_native_cosine_mean']=statistics.fmean(r['groups'][g]['native_alignment']['align_actual']['cosine'] for r in rows) if rows[0]['groups'][g]['native_alignment']['align_actual']['cosine'] is not None else None
            groups[g]['old_S06_alignment_native_cosine_mean']=statistics.fmean(r['groups'][g]['pairs']['align_actual__native_combined']['cosine'] for r in old_raw) if old_raw[0]['groups'][g]['pairs']['align_actual__native_combined']['cosine'] is not None else None
            groups[g]['weighted_norm_share_mean']={name:statistics.fmean(r['groups'][g]['effective_norm_share'][name] for r in rows) if rows[0]['groups'][g]['effective_norm_share'][name] is not None else None for name in OBJECTIVES[:3]}
            groups[g]['old_S06_weighted_norm_share_mean']={name:statistics.fmean(r['groups'][g]['effective_norm_share'][name] for r in old_raw) if old_raw[0]['groups'][g]['effective_norm_share'][name] is not None else None for name in OBJECTIVES[:3]}
        summary=dict(passed=True,batches=8,world_size=4,local_batch=256,checkpoint_sha256=sha,optimizer_steps=0,
          checkpoint_before_after_SHA_equal=True,model_before_after_digest=immutable,groups=groups,
          note='Same first8 global batches as prior audit, with exact image/F/S/D hashes. Raw and weighted norms are diagnostic; norm shares are not literal optimizer update percentages. Different checkpoints prevent a causal attribution to weights alone.',
          precision='BF16 encoders,FP32 masks/scores/loss,FP64 tiled Gram',identity=identity,hardware=hardware)
        dump(EXP/'GRADIENT_SPOT_CHECK.json',summary);dump(EXP/'evidence/spot8-proof.json',dict(batches=proofs,optimizer_steps=0))
        with (EXP/'raw/GRADIENT_SPOT8.json.gz').open('wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as archive:archive.write(json.dumps(rows).encode())
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
