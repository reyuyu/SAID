"""Real four-rank diagnostic backwards only. All update paths are forbidden."""
import argparse
from datetime import timedelta,datetime,timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch
from model.nested_semantic_mask import inclusion_weight
from train.nested_semantic_data import NestedDataset,collate,sampling_diagnostics,file_sha
from train.train_nested_semantic_mask import CappedSampler,seed_all,rng_state,restore_rng_state,state_digest,setup,parameter_agreement
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import AuditObjective,OBJECTIVES
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import GROUPS,parameter_group,gram_matrices,derive,linearity_errors

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/gradient_composition_audit_v1')
INDEX=Path('/root/lk_projects/SAID-nest-clip-v1/data_index')
IMAGE_ROOT=Path('/root/lk_projects/SAID-assets/training/ShareGPT4V')
SEARCH=ROOT/'experiments/nest_clip_v1/four_arm_text_search_500_v1'
SHARED=Path('/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt')
INIT_SHA='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'


def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(v,indent=2)+'\n');temporary.replace(p)

def stages():
    r=load(SEARCH/'BASELINE.json');b=load(SEARCH/'arm_B_summary_weight/RESULTS.json')
    return {
      '0':dict(name='Common step0',checkpoint=str(SHARED),sha256=INIT_SHA,mode='random_k',weights=[1.,1.,1.],config=r['config'],expected_updates=0),
      'R':dict(name='RandomK500',checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],mode='random_k',weights=[1.,1.,1.],config=r['config'],expected_updates=500),
      'B':dict(name='Summary downweight500',checkpoint=b['checkpoint'],sha256=b['checkpoint_sha256'],mode='summary_random_detail',weights=[1.2,.6,1.2],config=b['config'],expected_updates=500)}


def guard_updates():
    calls={'optimizer_step':0,'scaler_step':0}
    def forbidden(*args,**kwargs):
        calls['optimizer_step']+=1;raise AssertionError('Optimizer.step is forbidden in no-training gradient audit')
    def forbidden_scaler(*args,**kwargs):
        calls['scaler_step']+=1;raise AssertionError('GradScaler.step is forbidden')
    # No optimizer object is created. Hard traps cover standard exposed step paths.
    torch.optim.Optimizer.step=forbidden
    for name in dir(torch.optim):
        obj=getattr(torch.optim,name)
        if isinstance(obj,type) and issubclass(obj,torch.optim.Optimizer):obj.step=forbidden
    torch.amp.GradScaler.step=forbidden_scaler
    torch.cuda.amp.GradScaler.step=forbidden_scaler
    return calls


def checkpoint_steps(state):
    return sorted({float(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v})


def build_model(stage):
    cfg=stage['config'];seed_all(0)
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    common=torch.load(SHARED,map_location='cpu',weights_only=False)
    assert common['completed_steps']==0 and not common['optimizer']['state']
    clip.load_state_dict(common['model'],strict=True)
    saved=rng_state()
    model=BalancedSearch(clip.float(),search_hparams=dict(cfg,view_weights=stage['weights']),arm='A3',
      checkpoint_encoders=True,image_chunk=128,text_chunk=128,condition_mode='dual_branch',shuffle_seed=0,
      checkpoint_pair_blocks=False,fusion='balanced_stack',visual='patch')
    restore_rng_state(saved)
    assert state_digest(model.fusion_branch.state_dict())==cfg['adapter_initialization']['state_sha256']
    del common
    payload=torch.load(stage['checkpoint'],map_location='cpu',weights_only=False)
    assert payload['completed_steps']==stage['expected_updates']
    before_steps=checkpoint_steps(payload)
    if stage['expected_updates']:
        model.clip.load_state_dict(payload['model'],strict=True)
        model.fusion_branch.load_state_dict(payload['adapter'],strict=True)
        assert before_steps==[500.]
    else:assert before_steps==[]
    assert checkpoint_steps(payload)==before_steps
    optimizer_state_digest=hashlib.sha256(json.dumps(before_steps).encode()).hexdigest()
    del payload;gc.collect()
    model=model.cuda().train()
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    counts={g:sum(p.numel() for n,p in named.items() if parameter_group(n)==g) for g in GROUPS[:7]}
    assert sum(counts.values())==156931074
    return model,named,{'checkpoint_optimizer_steps':before_steps,'optimizer_step_metadata_digest':optimizer_state_digest,'parameter_groups':counts,'total_trainable':sum(counts.values())}


def synchronized_gradient_agreement(named,count=20):
    active=[n for n,p in named.items() if p.grad is not None]
    selected=random.Random(0).sample(active,min(count,len(active)));maximum=torch.zeros((),device='cuda')
    for name in selected:
        grad=named[name].grad;reference=grad.detach().clone();dist.broadcast(reference,src=0)
        maximum=torch.maximum(maximum,(grad-reference).abs().max())
    dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
    assert float(maximum)==0.,float(maximum)
    return {'tested_tensors':len(selected),'tensor_names':selected,'four_rank_max_abs_gradient_difference':float(maximum),'DDP_synchronized':True}


def batch_provenance(batch,rank):
    metadata=sampling_diagnostics(batch)
    image_sha=hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()
    return {'rank':rank,'sampling':metadata,'image_tensor_sha256':image_sha}


def compare_batch(stage,index,peers):
    baseline=load(SEARCH/'BASELINE.json')
    source=Path(baseline['root'])/'steps.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()[:32]]
    ref=rows[index]
    for p,r in zip(peers,ref['rank_health']):
        for key in ['sample_id_sha256','full_view_sha256','fixed_first_reference_stream_sha256']:
            assert p['sampling'][key]==r['sampling'][key],(stage,index,p['rank'],key)
        if stage in ['0','R']:
            for key in ['local_views_sha256','split_sha256']:assert p['sampling'][key]==r['sampling'][key]
    # Cross-stage actual image hashes and Full identity on every diagnostic batch.
    reference=EXP/'evidence/batches_0.json'
    if stage!='0' and reference.exists():
        common=load(reference)['batches'][index]['ranks']
        for p,r in zip(peers,common):
            assert p['image_tensor_sha256']==r['image_tensor_sha256']
            assert p['sampling']['full_view_sha256']==r['sampling']['full_view_sha256']
    if stage=='B':
        source=Path(stages()['B']['checkpoint']).parent/'steps.jsonl'
        old=[json.loads(line) for line in source.read_text().splitlines()[:32]][index]
        for p,r in zip(peers,old['rank_health']):
            for key in ['local_views_sha256','split_sha256']:assert p['sampling'][key]==r['sampling'][key]
            assert p['sampling']['random_detail_sampling']['selected_indices_sha256']==r['sampling']['random_detail_sampling']['selected_indices_sha256']


def repeat_error(named,old,new):
    maximum=0.;norm2=diff2=0.
    for name,p in named.items():
        a,b=old.get(name),new.get(name)
        assert (a is None)==(b is None)
        if a is None:continue
        d=(a-b).double();maximum=max(maximum,float(d.abs().max()));diff2+=float(d.square().sum());norm2+=float(a.double().square().sum())
    return {'max_abs':maximum,'relative_L2':math.sqrt(diff2/max(norm2,1e-300)),'bitwise_equal':maximum==0.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['0','R','B'],required=True);parser.add_argument('--batches',type=int,default=32);parser.add_argument('--pilot',action='store_true');parser.add_argument('--label',default='');parser.add_argument('--fp32-control',action='store_true');args=parser.parse_args()
    assert args.batches==32 or (args.pilot and args.batches==1)
    assert not args.fp32_control or (args.pilot and args.batches==1)
    if args.fp32_control:
        class FP32ControlAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self,device_type,*values,**options):
                options['enabled']=False
                super().__init__(device_type,*values,**options)
        torch.autocast=FP32ControlAutocast
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rank,local,world,hardware=setup();calls=guard_updates()
    stage=stages()[args.stage];destination=RUN/('pilot_'+args.stage if args.pilot else 'stage_'+args.stage)
    if args.label:destination=destination.with_name(destination.name+'_'+args.label)
    if rank==0:
        destination.mkdir(parents=True,exist_ok=False)
        assert file_sha(stage['checkpoint'])==stage['sha256']
        dump(RUN/'state.json',dict(status='RUNNING',stage=args.stage,batch=0,objective='loading',pilot=args.pilot,stage_pid=os.getpid()))
    dist.barrier()
    model,named,identity=build_model(stage)
    assert parameter_agreement(model)==0.
    immutable=state_digest(model.state_dict())
    wrapped=AuditObjective(model)
    ddp=DDP(wrapped,device_ids=[local],output_device=local,find_unused_parameters=True,static_graph=False)
    dataset=NestedDataset(INDEX,IMAGE_ROOT,stage['mode'],0);dataset.set_epoch(0)
    sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
    capped=CappedSampler(sampler,args.batches*256)
    loader=DataLoader(dataset,batch_size=256,sampler=capped,collate_fn=collate,num_workers=8,pin_memory=True,
      multiprocessing_context='spawn',prefetch_factor=2,generator=torch.Generator().manual_seed(0))
    metadata=[];raw=[];proofs=[]
    for batch_index,batch in enumerate(loader):
        start=time.monotonic();before=state_digest(model.state_dict());assert before==immutable
        local_meta=batch_provenance(batch,rank);peers=[None]*4;dist.all_gather_object(peers,local_meta)
        if rank==0:
            compare_batch(args.stage,batch_index,peers)
            global_ids=[sid for p in peers for sid in p['sampling']['sample_ids']]
            metadata.append(dict(batch=batch_index+1,ranks=peers,global_sample_ID_sha256=hashlib.sha256(json.dumps(global_ids,separators=(',',':')).encode()).hexdigest()))
            dump(RUN/'state.json',dict(status='RUNNING',stage=args.stage,batch=batch_index+1,objective='forward/backward',pilot=args.pilot,stage_pid=os.getpid()))
        images=batch['image'].cuda(non_blocking=True);views=[batch[k].cuda(non_blocking=True) for k in ['tokens_f','tokens_o','tokens_e']];valid=batch['valid'].cuda(non_blocking=True)
        gradients={};loss_values={};agreements={};native_checks=None;repeated=None
        for objective in OBJECTIVES:
            ddp.zero_grad(set_to_none=True)
            loss=ddp(images,*views,valid,objective,batch_index==0 and objective=='native_i2t')
            assert bool(torch.isfinite(loss))
            loss.backward()
            torch.cuda.synchronize()
            agreement=synchronized_gradient_agreement(named)
            gradients[objective]={name:p.grad.detach().clone() if p.grad is not None else None for name,p in named.items()}
            agreements[objective]=agreement
            mean=loss.detach().clone();dist.all_reduce(mean);loss_values[objective]=float(mean/world)
            if objective=='native_i2t':native_checks=wrapped.last_checks
            if objective.startswith('native'):
                assert all(g is None for name,g in gradients[objective].items() if parameter_group(name) in GROUPS[2:7])
            if rank==0 and batch_index==0:
                print(json.dumps({'stage':args.stage,'pilot':args.pilot,'batch':1,'objective':objective,'loss':loss_values[objective],'seconds':round(time.monotonic()-start,3)}),flush=True)
        # Same objective, same batch repeat at first/last: verify deterministic gradients.
        if batch_index in [0,args.batches-1]:
            ddp.zero_grad(set_to_none=True);repeat_loss=ddp(images,*views,valid,'F_i2t',False);repeat_loss.backward()
            synchronized_gradient_agreement(named)
            other={n:p.grad for n,p in named.items()};repeated=repeat_error(named,gradients['F_i2t'],other)
            assert repeated['bitwise_equal'],repeated
        torch.cuda.synchronize()
        if rank==0:
            grams,support=gram_matrices(named,gradients,OBJECTIVES)
            derived=derive(grams,support,OBJECTIVES,stage['weights'])
            linear=linearity_errors(named,gradients,stage['weights'])
            # Mask/scores paths are FP32; BF16 backbone summation is documented separately.
            assert all(v['relative_L2']<5e-5 for g,v in linear['groups'].items() if g in GROUPS[2:7]),linear
            if args.fp32_control:assert linear['relative_L2']<5e-5,linear
            row={'stage':args.stage,'batch':batch_index+1,'weights':stage['weights'],'loss_values':loss_values,
              'groups':derived,'Gram_objective_order':OBJECTIVES,'Gram_FP64':grams,'linearity':linear,'native_correctness':native_checks}
            raw.append(row)
            with (destination/'batch_stats.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        dist.barrier()
        del gradients;ddp.zero_grad(set_to_none=True)
        after=state_digest(model.state_dict());assert after==before==immutable
        checks=[None]*4;dist.all_gather_object(checks,dict(rank=rank,before_digest=before,after_digest=after,unchanged=True,update_calls=dict(calls)))
        assert all(c['before_digest']==c['after_digest']==immutable and c['update_calls']=={'optimizer_step':0,'scaler_step':0} for c in checks)
        if rank==0:
            proofs.append(dict(batch=batch_index+1,state_checks=checks,gradient_agreements=agreements,repeat_gradient=repeated))
            print(json.dumps({'stage':args.stage,'pilot':args.pilot,'batch_completed':batch_index+1,'total_batches':args.batches,'batch_seconds':round(time.monotonic()-start,3),'linearity_relative_L2':linear['relative_L2'],'model_unchanged':True,'fp32_control':args.fp32_control}),flush=True)
            dump(destination/'state-proof.json',dict(stage=args.stage,batches=proofs,optimizer_step_calls=0,scaler_step_calls=0))
        del images,views,valid,batch;gc.collect()
    if rank==0:
        assert file_sha(stage['checkpoint'])==stage['sha256']
        evidence=EXP/'evidence'/('pilot_'+args.stage if args.pilot else 'stage_'+args.stage)
        evidence=evidence.with_name(evidence.name+'_'+args.label) if args.label else evidence
        evidence.mkdir(parents=True,exist_ok=True)
        dump(evidence/'execution.json',dict(stage=stage,identity=identity,hardware=hardware,model_state_digest=immutable,
          checkpoint_file_SHA256_before_after_equal=True,batches=args.batches,optimizer_step_calls=0,scaler_step_calls=0,
          production_code_changed=False,precision=('Full FP32 control' if args.fp32_control else 'BF16 encoders,FP32 masks/scores/loss; FP64 tiled Gram accumulation'),
          inclusion_weight_main=inclusion_weight('A3',200),inclusion_weight_appendix=.5,
          git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        dump(evidence/'state-proof.json',dict(stage=args.stage,batches=proofs,optimizer_step_calls=0,scaler_step_calls=0))
        if not args.pilot:dump(EXP/'evidence'/('batches_'+args.stage+'.json'),dict(epoch=0,sampling_seed=0,sampler_seed=0,batch_selection='First32 rank-major global batches of official epoch0 DistributedSampler; no resampling',batches=metadata))
        dump(destination/'result.json',dict(status='COMPLETE',stage=args.stage,batches=args.batches,checkpoint_sha256=stage['sha256'],model_state_unchanged=True))
        dump(RUN/'state.json',dict(status='STAGE_COMPLETE',stage=args.stage,batch=args.batches,pilot=args.pilot))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
