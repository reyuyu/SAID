"""Four-rank real-data performance probe. Never saves a trained retrieval model."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from train.nested_semantic_data import NestedDataset, collate, sampling_diagnostics, file_sha
from train.train_nested_semantic_mask import seed_all, setup, CappedSampler, gradient_norm, state_digest
from train.runtime_audit import audit_step, finite_gradients, update_rolling_hash, ProfileRanges, profiler_summary


RUN = Path('/root/lk_projects/SAID-nest-clip-v1/runtime_optimization_v1')


def math_modules(reference):
    prefix = ('experiments.nest_clip_v1.runtime_optimization_v1.reference.model.' if reference else 'model.')
    return tuple(importlib.import_module(prefix+name) for name in (
        'balanced_hparam_search', 'nested_fusion_mask', 'nested_semantic_mask'))


def make_model(cfg, initial, device, reference=False):
    balanced, fusion, semantic = math_modules(reference)
    clip, _ = longclip.load_from_clip(cfg.get('base_model', 'ViT-B/16'), device='cpu', args=argparse.Namespace())
    assert initial['completed_steps'] == 0 and not initial['optimizer']['state']
    assert initial['provenance'].get('base_model', 'ViT-B/16') == cfg.get('base_model', 'ViT-B/16')
    clip.load_state_dict(initial['model'], strict=True)
    from model.backbone import validate_backbone
    validate_backbone(clip,cfg.get('base_model','ViT-B/16'))
    module = balanced.BalancedSearch(clip.float(), search_hparams=balanced.hparams(cfg),
        fusion='balanced_stack', visual='patch', arm='A3', condition_mode='dual_branch',
        checkpoint_encoders=cfg['checkpoint_encoders'], checkpoint_pair_blocks=False,
        image_chunk=cfg['image_chunk'], text_chunk=cfg['text_chunk'])
    if 'adapter' in initial and initial['adapter'] is not None:
        module.fusion_branch.load_state_dict(initial['adapter'], strict=True)
    if hasattr(module, 'configure_runtime'):
        module.configure_runtime(cfg)
    if cfg.get('encoder_checkpoint_strategy'):
        from model.runtime_execution import configure_checkpointing
        configure_checkpointing(module.clip, cfg['encoder_checkpoint_strategy'])
    module = module.to(device).train()
    assert all(p.dtype == torch.float32 for p in module.parameters())
    return module, (balanced, fusion, semantic)


def optimizer_for(module, cfg):
    options = {'fused': True} if cfg.get('fused_adamw') else {}
    return torch.optim.AdamW(module.optimizer_groups(), betas=(.9,.999), eps=1e-8, **options)


def set_lrs(optimizer, step):
    cosine = .5*(1+math.cos(math.pi*step/4868))
    backbone = 1e-6*(step+1)/200 if step < 200 else .5e-6*(1+math.cos(math.pi*(step-200)/(4868-200)))
    for index, group in enumerate(optimizer.param_groups):
        group['lr'] = backbone if index == 0 else group['peak_lr']*cosine


def make_loader(rank, steps):
    dataset = NestedDataset('/root/lk_projects/SAID-nest-clip-v1/data_index',
        '/root/lk_projects/SAID-assets/training/ShareGPT4V', 'random_k', 0, 'compact')
    assert len(dataset) == 1245901
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True, seed=0, drop_last=False)
    sampler.set_epoch(0); dataset.set_epoch(0)
    return DataLoader(dataset, batch_size=256, sampler=CappedSampler(sampler, steps*256),
        collate_fn=collate, num_workers=8, pin_memory=True, drop_last=False,
        multiprocessing_context='spawn', prefetch_factor=2, generator=torch.Generator().manual_seed(0))


def full_audit(module, optimizer, batch, logs, step, rank, elapsed, output, ranges, group_norms, auxiliary_norms):
    with ranges.range('integrity_audit_logging'):
        torch.cuda.synchronize()
        stream = json.dumps(dict(sample_ids=batch['sample_id'].tolist(), views=batch['views'],
            tokens=[batch[k].cpu().tolist() for k in ('tokens_f','tokens_o','tokens_e')]),
            ensure_ascii=False,separators=(',',':')).encode()
        health = dict(rank=rank, updates=step, batch=256, valid=int(batch['valid'].sum()),
            seconds=time.perf_counter()-elapsed, gradients_finite=True,
            reasons=dict(Counter(batch['reason'])),
            gradient_norms={k:float(v) for k,v in group_norms.items()},
            adapter_gradient_norms={k:float(v) for k,v in auxiliary_norms.items()},
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            stream_sha256=hashlib.sha256(stream).hexdigest(), sampling=sampling_diagnostics(batch))
        healths = [None]*4; dist.all_gather_object(healths, health)
        ids = math_modules(False)[2].gather(batch['image_id'].cuda(), False)
        unique,counts = ids.unique(return_counts=True); duplicates = unique[counts>1].cpu().tolist()
        if rank == 0:
            row = dict(step=step, rank_health=healths, duplicate_image_ids=duplicates,
                **{k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()})
            with (output/'steps.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            print(json.dumps(dict(step=step,loss=row['loss'],V=row['valid_global'])),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--save-fixed-input', action='store_true')
    args = parser.parse_args();cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(4);seed_all(0)
    relaxed = cfg.get('tf32', False)
    torch.backends.cuda.matmul.allow_tf32 = relaxed
    torch.set_float32_matmul_precision('high' if relaxed else 'highest')
    rank, local, world, peers = setup();assert world == 4
    device = torch.device('cuda',local)
    output = RUN/args.name
    if rank == 0:output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    initial_path = Path(cfg.get('init_state','/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt'))
    initial = torch.load(initial_path,map_location='cpu',weights_only=False)
    module, (balanced, fusion, semantic) = make_model(cfg,initial,device,args.reference)
    del initial
    ddp_options = dict(find_unused_parameters=cfg.get('find_unused_parameters',True),
        static_graph=cfg.get('static_graph',False),gradient_as_bucket_view=cfg.get('gradient_as_bucket_view',False),
        bucket_cap_mb=cfg.get('bucket_cap_mb',25))
    ddp = DDP(module,device_ids=[local],output_device=local,**ddp_options)
    optimizer = optimizer_for(module,cfg)
    ranges = ProfileRanges(args.profile);ranges.install(module,fusion,semantic,balanced)
    steps = 15 if args.profile else 35
    loader = make_loader(rank,steps);iterator = iter(loader)
    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0,warmup=5,active=10,repeat=1),record_shapes=False,
        profile_memory=False,with_stack=False) if args.profile else nullcontext()
    timings=[];losses=[];presence=None;rolling=hashlib.sha256();input_hashes=[];input_images=[];measured_start_unix=None
    if rank == 0:(output/'config.json').write_text(json.dumps(dict(cfg,args=vars(args),ddp=ddp_options,
        initial_sha256=file_sha(initial_path),torch=torch.__version__,cuda=torch.version.cuda,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
    torch.cuda.reset_peak_memory_stats()
    with profile as prof:
        for index in range(steps):
            if index==5:measured_start_unix=time.time()
            ranges.calls=0;start=time.perf_counter()
            with ranges.range('data_wait'):cpu_batch=next(iterator)
            input_images.append(cpu_batch['image'])
            if index == 0 and args.save_fixed_input:
                torch.save(cpu_batch,output/f'fixed-input-rank{rank}.pt')
            input_hashes.append(update_rolling_hash(rolling,cpu_batch))
            with ranges.range('host_to_device'):
                batch=dict(cpu_batch,**{k:cpu_batch[k].to(device,non_blocking=True) for k in (
                    'image','tokens_f','tokens_o','tokens_e','valid')})
            set_lrs(optimizer,index);optimizer.zero_grad(set_to_none=True)
            audited=audit_step(index+1,cfg.get('audit_level','benchmark'),final=index+1==steps)
            module.runtime_audit_active = audited or not cfg.get('skip_observational_model_logs',False)
            with ranges.range('model_forward'):
                loss,logs=ddp(batch['image'],batch['tokens_f'],batch['tokens_o'],batch['tokens_e'],batch['valid'],index)
            with ranges.range('nonfinite_detection'):
                finite=torch.isfinite(loss).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError('Nonfinite loss')
            with ranges.range('backward'):loss.backward()
            with ranges.range('gradient_diagnostics'):
                if audited:
                    group_norms={g['name']:gradient_norm(g['params']) for g in optimizer.param_groups}
                    auxiliary_norms={name:gradient_norm([p]) for name,p in module.fusion_branch.named_parameters()}
                    finite=torch.stack([torch.isfinite(x) for x in [*group_norms.values(),*auxiliary_norms.values()]]).all().int()
                else:finite=finite_gradients(module)
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError('Nonfinite gradients')
            if index == 0:presence={name:p.grad is not None for name,p in module.named_parameters() if p.requires_grad}
            with ranges.range('optimizer'):optimizer.step()
            if audited:full_audit(module,optimizer,batch,logs,index+1,rank,start,output,ranges,group_norms,auxiliary_norms)
            with ranges.range('step_synchronization'):
                if audited:dist.barrier(device_ids=[local])
                torch.cuda.synchronize(local)
            timings.append(time.perf_counter()-start)
            losses.append(float(loss.detach()))
            if torch.cuda.max_memory_allocated()/2**30>65:
                (output/f'failure-rank{rank}.json').write_text(json.dumps(dict(
                    reason='Allocated memory exceeded65GiB',step=index+1,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))+'\n')
                raise RuntimeError('Allocated memory exceeded65GiB')
            if args.profile:prof.step()
        if args.profile:
            prof.export_chrome_trace(str(output/f'profile-rank{rank}.json.gz'))
            (output/f'profile-summary-rank{rank}.json').write_text(json.dumps(profiler_summary(prof),indent=2)+'\n')
    measured_end_unix=time.time()
    shutdown=getattr(iterator,'_shutdown_workers',None)
    if shutdown is not None:shutdown()
    image_digest=hashlib.sha256()
    for tensor in input_images:image_digest.update(tensor.numpy().tobytes())
    local_result=dict(rank=rank,timings=timings,losses=losses,input_rolling_hashes=input_hashes,
        image_stream_sha256=image_digest.hexdigest(),
        measured_start_unix=measured_start_unix,measured_end_unix=measured_end_unix,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gradient_presence=presence)
    gathered=[None]*4;dist.all_gather_object(gathered,local_result)
    if rank == 0:
        measured=[max(x['timings'][step] for x in gathered) for step in range(5,steps)]
        result=dict(name=args.name,profile=args.profile,reference=args.reference,config=cfg,
            world_size=4,logical_local_batch=256,global_candidate_pool=1024,horizon=4868,
            warmup_updates=5,measured_updates=steps-5,ranks=gathered,
            mean_seconds=statistics.fmean(measured),median_seconds=statistics.median(measured),
            p95_seconds=statistics.quantiles(measured,n=100)[94],max_seconds=max(measured),
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in gathered),
            peak_reserved_gib=max(x['peak_reserved_gib'] for x in gathered),
            resource_feasible=max(measured)<=3 and max(x['peak_allocated_gib'] for x in gathered)<=65,
            classification='RELAXED' if relaxed else 'EXACT_CANDIDATE',
            text_rematerialization=cfg.get('fused_text_backward')=='per_view_recompute' and cfg.get('fused_text_views',False),
            ddp_logging=ddp._get_ddp_logging_data())
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','ddp_logging','config') }),flush=True)
    ranges.close();dist.barrier(device_ids=[local]);torch.cuda.synchronize(local);dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except torch.cuda.OutOfMemoryError as exc:
        name=sys.argv[sys.argv.index('--name')+1]
        path=RUN/name;path.mkdir(parents=True,exist_ok=True)
        (path/f'failure-rank{os.environ.get("RANK","unknown")}.json').write_text(json.dumps(dict(
            reason='CUDA OOM',error=str(exc),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))+'\n')
        raise
