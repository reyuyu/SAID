"""Named full-model/AdamW regression on a fixed real four-rank global1024 batch."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train.train_nested_semantic_mask import setup, seed_all, gradient_norm
from train.runtime_audit import finite_gradients, audit_step
from experiments.nest_clip_v1.runtime_optimization_v1.probe import (
    RUN, make_model, optimizer_for, set_lrs,
)


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


def snapshot(module, optimizer):
    return dict(parameters={n:cpu_copy(p) for n,p in module.named_parameters()},
                gradients={n:None if p.grad is None else cpu_copy(p.grad) for n,p in module.named_parameters()},
                optimizer={n:cpu_copy(optimizer.state[p]) for n,p in module.named_parameters() if p in optimizer.state})


def error(a,b,atol,rtol):
    assert a.shape==b.shape and a.dtype==b.dtype
    diff=(a.double()-b.double()).abs()
    return dict(max_abs=float(diff.max()) if diff.numel() else 0,
                max_relative=float((diff/a.double().abs().clamp_min(1e-8)).max()) if diff.numel() else 0,
                relative_l2=float(diff.norm()/a.double().norm().clamp_min(1e-10)),
                outside_tolerance=int((diff>atol+rtol*a.double().abs()).sum()),
                finite=bool(torch.isfinite(b).all()))


def compare_snapshots(a,b,bitwise=False):
    output={};passed=True
    for field in ('parameters','gradients','optimizer'):
        assert a[field].keys()==b[field].keys(),field
        results={}
        tolerance=(0.,0.) if bitwise else ((3e-7,3e-6) if field=='parameters' else (3e-4,2e-3))
        for name,x in a[field].items():
            y=b[field][name]
            if field=='optimizer':
                assert x.keys()==y.keys(),name
                records={k:error(x[k],y[k],*tolerance) for k in x if torch.is_tensor(x[k])}
                for k in x:
                    if not torch.is_tensor(x[k]):assert x[k]==y[k],(name,k)
                results[name]=records
                passed &= all(v['outside_tolerance']==0 and v['finite'] for v in records.values())
            elif x is None:
                results[name]={'both_absent':y is None};passed &= y is None
            elif y is None:
                results[name]={'missing_candidate':True};passed=False
            else:
                results[name]=error(x,y,*tolerance)
                passed &= results[name]['outside_tolerance']==0 and results[name]['finite']
        output[field]=results
    return dict(passed=bool(passed),bitwise_required=bitwise,by_parameter=output)


def capture_forward(module, fusion, balanced, batch, completed):
    captured={};view=['F'];seen=set()
    old_scores,old_logits= fusion.fusion_scores,fusion.pair_logits
    def scores(*args,**kwargs):
        name=('F','P','R')[len([k for k in captured if k.endswith('_scores')])]
        view[0]=name
        result=old_scores(*args,**kwargs)
        captured[name+'_scores']=cpu_copy(result[0])
        return result
    def logits(*args,**kwargs):
        result=old_logits(*args,**kwargs)
        if result.ndim==3 and view[0] not in seen:
            captured[view[0]+'_logits']=cpu_copy(result)
            captured[view[0]+'_probabilities']=cpu_copy(result.sigmoid())
            seen.add(view[0])
        return result
    fusion.fusion_scores=scores;fusion.pair_logits=logits
    try:
        with torch.no_grad():
            z,v=module.encode_visual(batch['image']);captured['image_embedding']=cpu_copy(z);captured['visual_condition']=cpu_copy(v[1])
            token_views=[batch[k] for k in ('tokens_f','tokens_o','tokens_e')]
            valid_count=int(balanced.gather(batch['valid'],False).sum())
            token_views=token_views if valid_count>=2 else token_views[:1]
            encoded=(module.encode_views(token_views) if valid_count>=2 and getattr(module,'runtime_fused_text',False) else
                     [module.encode_view(tokens) for tokens in token_views])
            for name,(text,condition) in zip(('F','P','R'),encoded):
                captured[name+'_embedding']=cpu_copy(text);captured[name+'_condition']=cpu_copy(condition[1])
            loss,_=module(batch['image'],batch['tokens_f'],batch['tokens_o'],batch['tokens_e'],batch['valid'],completed)
            captured['loss']=cpu_copy(loss)
    finally:fusion.fusion_scores=old_scores;fusion.pair_logits=old_logits
    return captured


def run_version(config, reference, initial, checkpoint, batch, rank, local, steps, scenarios):
    seed_all(0)
    device=torch.device('cuda',local)
    module,(balanced,fusion,semantic)=make_model(config,initial,device,reference)
    module.clip.load_state_dict(checkpoint['model'],strict=True)
    module.fusion_branch.load_state_dict(checkpoint['adapter'],strict=True)
    optimizer=optimizer_for(module,config)
    if not config.get('fused_adamw'):
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    else:
        state=copy.deepcopy(checkpoint['optimizer'])
        for group in state['param_groups']:group['fused']=True;group['foreach']=False
        optimizer.load_state_dict(state)
    ddp=DDP(module,device_ids=[local],output_device=local,
        find_unused_parameters=config.get('find_unused_parameters',True),static_graph=config.get('static_graph',False),
        gradient_as_bucket_view=config.get('gradient_as_bucket_view',False),bucket_cap_mb=config.get('bucket_cap_mb',25))
    results=[]
    for index in range(steps):
        selected=scenarios[index % len(scenarios)]
        valid=batch['valid'].clone()
        if selected!='normal':
            valid[:]=False
            if selected=='rare' and rank in (0,1):valid[0]=True
            if selected=='fallback' and rank==0:valid[0]=True
        fixed=dict(batch,valid=valid)
        completed=int(checkpoint['completed_steps'])+index
        full=audit_step(completed+1,config.get('audit_level','benchmark'))
        module.runtime_audit_active=full or not config.get('skip_observational_model_logs',False)
        forward=capture_forward(module,fusion,balanced,fixed,completed)
        set_lrs(optimizer,completed);optimizer.zero_grad(set_to_none=True)
        loss,logs=ddp(fixed['image'],fixed['tokens_f'],fixed['tokens_o'],fixed['tokens_e'],fixed['valid'],completed)
        loss.backward()
        if full:
            norms=[gradient_norm(g['params']) for g in optimizer.param_groups]
            norms.extend(gradient_norm([p]) for p in module.fusion_branch.parameters())
            finite=torch.isfinite(torch.stack(norms)).all().int()
        else:finite=finite_gradients(module)
        dist.all_reduce(finite,op=dist.ReduceOp.MIN);assert finite.item()
        gradients={name:cpu_copy(p.grad) if p.grad is not None else None for name,p in module.named_parameters()}
        optimizer.step()
        result=snapshot(module,optimizer);result['gradients']=gradients
        results.append(dict(scenario=selected,forward=forward,snapshot=result,loss=float(loss.detach())))
    del ddp,optimizer,module;gc.collect();torch.cuda.empty_cache();dist.barrier()
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--steps',type=int,default=1);parser.add_argument('--scenarios',default='normal')
    parser.add_argument('--bitwise',action='store_true');parser.add_argument('--self-repeat',action='store_true')
    args=parser.parse_args();assert 1<=args.steps<=3
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');rank,local,world,_=setup();assert world==4
    cfg=json.loads(Path(args.config).read_text());baseline=dict(cfg)
    for flag in ('fused_text_views','cache_gate_projection','cache_text_normalization','reduced_pair_score','fused_adamw','tf32','skip_observational_model_logs'):
        baseline[flag]=False
    baseline.update(encoder_checkpoint_strategy='full',checkpoint_encoders=True,audit_level='full',find_unused_parameters=True,static_graph=False,gradient_as_bucket_view=False)
    initial=torch.load(cfg.get('init_state','/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt'),map_location='cpu',weights_only=False)
    checkpoint=torch.load(cfg.get('regression_checkpoint','/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1/baseline/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step500/step000500.pt'),map_location='cpu',weights_only=False)
    cpu_batch=torch.load(RUN/f'profile-current/fixed-input-rank{rank}.pt',map_location='cpu',weights_only=False)
    batch={k:cpu_batch[k].cuda(local) for k in ('image','tokens_f','tokens_o','tokens_e','valid')}
    scenarios=args.scenarios.split(',');assert all(x in ('normal','rare','fallback') for x in scenarios)
    first=run_version(baseline,True,initial,checkpoint,batch,rank,local,args.steps,scenarios)
    torch.backends.cuda.matmul.allow_tf32=cfg.get('tf32',False);torch.set_float32_matmul_precision('high' if cfg.get('tf32',False) else 'highest')
    second=run_version(baseline if args.self_repeat else cfg,args.self_repeat,initial,checkpoint,batch,rank,local,args.steps,scenarios)
    comparisons=[]
    for a,b in zip(first,second):
        assert a['scenario']==b['scenario'] and a['forward'].keys()==b['forward'].keys()
        forward={k:error(a['forward'][k],b['forward'][k],0 if args.bitwise else 3e-4,0 if args.bitwise else 2e-4) for k in a['forward']}
        compare=compare_snapshots(a['snapshot'],b['snapshot'],args.bitwise)
        compare.update(forward=forward,scenario=a['scenario'],loss_error=abs(a['loss']-b['loss']))
        compare['passed'] &= all(v['outside_tolerance']==0 and v['finite'] for v in forward.values())
        comparisons.append(compare)
    result=dict(rank=rank,passed=all(x['passed'] for x in comparisons),steps=args.steps,comparisons=comparisons,
        deterministic_algorithms=True,cublas_workspace_config=os.getenv('CUBLAS_WORKSPACE_CONFIG'),
        real_input=True,logical_batch_per_rank=256,global_candidate_pool=1024,
        representative_logit_tile='first full128x128 tile per view; complete global1024 scores/loss/gradients/states checked',config=cfg)
    output=RUN/args.name
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    dist.barrier();(output/f'result-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    results=[None]*4;dist.all_gather_object(results,dict(rank=rank,passed=result['passed']))
    if rank==0:
        summary=dict(name=args.name,passed=all(x['passed'] for x in results),ranks=results,steps=args.steps,
            bitwise_required=args.bitwise,scenarios=scenarios,self_repeat=args.self_repeat)
        (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    dist.barrier(device_ids=[local]);torch.cuda.synchronize(local);dist.destroy_process_group()


if __name__=='__main__':main()
