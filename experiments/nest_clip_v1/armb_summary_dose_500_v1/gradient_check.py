"""Read-only first8 native spot and final-checkpoint last50 input replay."""
import argparse
import gc
import gzip
import json
import math
import statistics
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Sampler
from train.nested_semantic_data import NestedDataset,collate,file_sha
from train.train_nested_semantic_mask import state_digest,setup,parameter_agreement
from experiments.nest_clip_v1.gradient_composition_audit_v1 import run_audit as audit
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import AuditObjective
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import GROUPS,gram_matrices
from experiments.nest_clip_v1.armb_summary_dose_500_v1.run import EXP,RUN,S04,ARMS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records


class FixedIndices(Sampler):
    def __init__(self,ids):self.indices=[sid-1000 for sid in ids]
    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)


def metrics(grams,objectives,weights):
    out={};index={key:i for i,key in enumerate(objectives)}
    for group,G in grams.items():
        norms={key:math.sqrt(max(0,G[i][i])) for key,i in index.items()}
        weighted={key:w*norms[key] for key,w in zip(objectives[:3],weights)};denom=sum(weighted.values())
        v=dict(norms=norms,weighted_norms=weighted,norm_shares={key:value/denom if denom>0 else None for key,value in weighted.items()})
        if 'native_combined' in index:
            n=index['native_combined'];native_norm=norms['native_combined'];alignment_norm=norms['align_actual'];a=index['align_actual']
            v['actual_alignment_native_cosine']=G[a][n]/(alignment_norm*native_norm) if alignment_norm>0 and native_norm>0 else None
            v['actual_alignment_native_projection']=G[a][n]/(native_norm*native_norm) if native_norm>0 else None
        out[group]=v
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--scope',choices=['spot8','last50'],required=True);args=p.parse_args()
    directory,weights=ARMS[args.arm];exp=EXP/directory;run=RUN/directory
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,hardware=setup();calls=audit.guard_updates()
    result=load(exp/'RESULTS.json');ckpt=result['checkpoint'];sha=result['checkpoint_sha256']
    if rank==0:assert file_sha(ckpt)==sha
    stage=dict(checkpoint=ckpt,weights=weights,config=result['config'],expected_updates=500)
    model,named,identity=audit.build_model(stage);assert parameter_agreement(model)==0
    immutable=state_digest(model.state_dict());ddp=DDP(AuditObjective(model),device_ids=[local],output_device=local,find_unused_parameters=True)
    source=load(S04/'RESULTS.json');reference=records(__import__('pathlib').Path(source['checkpoint']).parent/'steps.jsonl')
    positions=list(range(8)) if args.scope=='spot8' else list(range(450,500))
    ids=[sid for position in positions for sid in reference[position]['rank_health'][rank]['sampling']['sample_ids']]
    ds=NestedDataset(audit.INDEX,audit.IMAGE_ROOT,'summary_random_detail',0);ds.set_epoch(0)
    loader=DataLoader(ds,batch_size=256,sampler=FixedIndices(ids),collate_fn=collate,num_workers=8,pin_memory=True,
      multiprocessing_context='spawn',prefetch_factor=2,generator=torch.Generator().manual_seed(0))
    objectives=['F_combined','O_combined','E_combined']+(['align_actual','native_combined'] if args.scope=='spot8' else [])
    old_images=load(audit.EXP/'evidence/batches_B.json')['batches'][:8]
    rows=[];proofs=[]
    for i,batch in enumerate(loader):
        start=time.monotonic();before=state_digest(model.state_dict());assert before==immutable
        item=audit.batch_provenance(batch,rank);peers=[None]*4;dist.all_gather_object(peers,item)
        for a,b in zip(peers,reference[positions[i]]['rank_health']):
            for field in ['sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256']:
                assert a['sampling'][field]==b['sampling'][field]
            assert a['sampling']['random_detail_sampling']['selected_indices_sha256']==b['sampling']['random_detail_sampling']['selected_indices_sha256']
        if args.scope=='spot8':
            for a,b in zip(peers,old_images[i]['ranks']):assert a['image_tensor_sha256']==b['image_tensor_sha256']
        images=batch['image'].cuda(non_blocking=True);views=[batch[key].cuda(non_blocking=True) for key in ['tokens_f','tokens_o','tokens_e']];valid=batch['valid'].cuda(non_blocking=True)
        gradients={};agreements={};losses={}
        for objective in objectives:
            ddp.zero_grad(set_to_none=True);loss=ddp(images,*views,valid,objective,args.scope=='spot8' and i==0 and objective=='native_combined');loss.backward()
            agreements[objective]=audit.synchronized_gradient_agreement(named)
            gradients[objective]={name:p.grad.detach().clone() if p.grad is not None else None for name,p in named.items()}
            average=loss.detach().clone();dist.all_reduce(average);losses[objective]=float(average/4)
        if rank==0:
            grams,_=gram_matrices(named,gradients,objectives);values=metrics(grams,objectives,weights)
            rows.append(dict(batch_position=positions[i]+1,groups=values,losses=losses,objectives=objectives,Gram_FP64=grams))
        dist.barrier();del gradients;ddp.zero_grad(set_to_none=True)
        after=state_digest(model.state_dict());assert after==before==immutable and calls=={'optimizer_step':0,'scaler_step':0}
        states=[None]*4;dist.all_gather_object(states,dict(rank=rank,before=before,after=after,update_calls=dict(calls)))
        if rank==0:
            proofs.append(dict(batch_position=positions[i]+1,state_checks=states,gradient_agreements=agreements))
            if i==0 or (i+1)%5==0 or i+1==len(positions):print(json.dumps({'arm':args.arm,'scope':args.scope,'batches_completed':i+1,'total':len(positions),'seconds':round(time.monotonic()-start,2),'unchanged':True}),flush=True)
        del images,views,valid,batch;gc.collect()
    if rank==0:
        assert file_sha(ckpt)==sha
        summary=dict(passed=True,arm=args.arm,scope=args.scope,batches=len(positions),batch_positions=[p+1 for p in positions],weights=weights,
          checkpoint_sha256=sha,checkpoint_before_after_SHA_equal=True,model_digest_unchanged=True,optimizer_steps=0,
          groups={},scope_note='Final step500 frozen model gradients on selected input batches. last50 refers to training-input positions451..500, not saved per-update gradients. Norm shares are not optimizer displacement contributions.',hardware=hardware,identity=identity)
        for group in GROUPS:
            v=dict(raw_norm_means={key:statistics.fmean(r['groups'][group]['norms'][key] for r in rows) for key in objectives},
              weighted_norm_means={key:statistics.fmean(r['groups'][group]['weighted_norms'][key] for r in rows) for key in objectives[:3]},
              weighted_norm_share_means={key:statistics.fmean(r['groups'][group]['norm_shares'][key] for r in rows) if rows[0]['groups'][group]['norm_shares'][key] is not None else None for key in objectives[:3]})
            if args.scope=='spot8':
                v['actual_alignment_native_cosine_mean']=statistics.fmean(r['groups'][group]['actual_alignment_native_cosine'] for r in rows) if rows[0]['groups'][group]['actual_alignment_native_cosine'] is not None else None
                v['actual_alignment_native_projection_mean']=statistics.fmean(r['groups'][group]['actual_alignment_native_projection'] for r in rows) if rows[0]['groups'][group]['actual_alignment_native_projection'] is not None else None
            summary['groups'][group]=v
        name='GRADIENT_SPOT8' if args.scope=='spot8' else 'LAST50_GRADIENT_SHARES'
        dump(exp/(name+'.json'),summary);dump(exp/'evidence'/(name+'-proof.json'),dict(batches=proofs,optimizer_steps=0))
        with (exp/'raw'/(name+'.json.gz')).open('wb') as handle:
            with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=9) as archive:archive.write(json.dumps(rows).encode())
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
