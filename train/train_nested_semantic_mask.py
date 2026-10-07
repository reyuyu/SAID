"""torchrun-only NEST-CLIP v1 trainer; smoke and formal both require four ranks."""
import argparse
import gc
from collections import Counter
from datetime import timedelta
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.nested_semantic_mask import NestedSemanticMask, gather, _checkpoint_blocks
from model.nested_vcp_mask import NestedVCPMask
from model.nested_fusion_mask import NestedFusionMask
from model.balanced_hparam_search import BalancedSearch, hparams, migrate_legacy_optimizer
from train.nested_semantic_data import NestedDataset, collate, file_sha, sampling_diagnostics


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auxiliary_module(module):
    if hasattr(module, 'fusion_branch'):
        return module.fusion_branch
    joint = getattr(module, 'joint_adapter', None)
    return joint if joint is not None else getattr(module, 'vcp_query', None)


def build_optimizer(module):
    if isinstance(module, BalancedSearch):
        return torch.optim.AdamW(module.optimizer_groups(),betas=(.9,.999),eps=1e-8)
    mask_parameters = (module.mask_parameters() if hasattr(module, 'mask_parameters')
                       else module.clip.mask_net.parameters())
    mask_ids = {id(p) for p in mask_parameters if p.requires_grad}
    auxiliary = auxiliary_module(module)
    adapter_ids = ({id(p) for p in auxiliary.parameters() if p.requires_grad} - mask_ids
                   if auxiliary is not None else set())
    backbone, mask, adapter = [], [], []
    for p in module.parameters():
        if p.requires_grad:
            target = mask if id(p) in mask_ids else adapter if id(p) in adapter_ids else backbone
            target.append(p)
    params = backbone + mask + adapter
    assert len(params) == len({id(p) for p in params})
    assert len(mask) == len(mask_ids)
    assert len(adapter) == len(adapter_ids)
    groups = [
        dict(params=backbone, lr=1e-6, weight_decay=1e-2, name='backbone'),
        dict(params=mask, lr=1e-3, weight_decay=0., name='shared_mask')]
    if adapter:
        groups.append(dict(params=adapter, lr=1e-4, weight_decay=0., name='joint_adapter'))
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def learning_rates(s, horizon, include_adapter=False):
    backbone = 1e-6 * (s+1) / 200 if s < 200 else .5e-6 * (1 + math.cos(math.pi * (s-200)/(horizon-200)))
    values = (backbone, .5e-3 * (1 + math.cos(math.pi*s/horizon)))
    return values + ((.5e-4 * (1 + math.cos(math.pi*s/horizon))),) if include_adapter else values


def optimizer_learning_rates(module, s, horizon):
    if not isinstance(module, BalancedSearch):
        return learning_rates(s,horizon,auxiliary_module(module) is not None)
    backbone=learning_rates(s,horizon)[0]
    factor=.5*(1+math.cos(math.pi*s/horizon))
    hp=module.search_hparams
    return backbone,1e-3*factor,1e-3*hp['visual_mask_lr_scale']*factor,hp['fusion_lr']*factor


def training_horizon(config, batches_per_epoch):
    assert batches_per_epoch == 1217
    epochs = config['epochs']
    assert epochs in (3,4), 'Only three epochs or the authorized four-epoch Balanced followup'
    if epochs == 4:
        assert config.get('four_epoch_followup') and config.get('hparam_search')
        assert config.get('condition_mode') == 'dual_branch'
        assert config.get('fusion') == 'balanced_stack' and config.get('visual') == 'patch'
    else:
        assert not config.get('four_epoch_followup'), 'Four-epoch configuration must use four epochs'
    return epochs * batches_per_epoch


def gradient_norm(parameters):
    squares = [parameter.grad.float().square().sum()
               for parameter in parameters if parameter.grad is not None]
    if not squares:
        return torch.zeros((), device='cuda')
    return torch.stack(squares).sum().sqrt()


def validate_resume_payload(previous, current, expected_parent_trainer_sha256=None, allow_legacy_b0=False):
    """Permit a longer stop within the SAME horizon; pin a trainer-only migration.

    Model/objective/data code must match byte-for-byte. A changed trainer is
    accepted only when its recorded predecessor hash is explicitly supplied.
    """
    old = previous['config']
    if current.get('hparam_search',False):
        assert hparams(old)==hparams(current), 'Resume hyperparameters changed'
        assert old.get('inclusion_hierarchy','siblings') == current.get('inclusion_hierarchy','siblings'), 'Resume inclusion hierarchy changed'
        assert old.get('view_sparsity_weights',[1.,2.,2.]) == current.get('view_sparsity_weights',[1.,2.,2.]), 'Resume view sparsity allocation changed'
        assert old.get('regularizer_mode','independent') == current.get('regularizer_mode','independent'), 'Resume regularizer changed'
        assert old.get('support_bands') == current.get('support_bands'), 'Resume support bands changed'
        if not allow_legacy_b0:
            assert old.get('trial_id')==current.get('trial_id'), 'Resume trial identity changed'
    for key in ('arm', 'horizon', 'init_sha256', 'data', 'batch_size', 'world_size',
                'accumulation', 'epochs', 'seed', 'workers', 'checkpoint_encoders',
                'score_chunk'):
        assert old[key] == current[key], f'Resume mismatch: {key}'
    for key, default in (('image_chunk', 32), ('text_chunk', 64),
                         ('condition_mode', 'text_only'), ('shuffle_seed', 0),
                         ('checkpoint_pair_blocks', True), ('four_epoch_followup', False)):
        assert old.get(key, default) == current.get(key, default), f'Resume mismatch: {key}'
    assert old['run_type'] == current['run_type'] == 'formal', 'Only formal checkpoints may continue formally'
    for key, default in (('sampling_mode', 'fixed_first'), ('sampling_seed', 0)):
        assert old.get(key, default) == current[key], f'Resume mismatch: {key}'
    trainer = 'train/train_nested_semantic_mask.py'
    old_code, new_code = old['code_sha256'], current['code_sha256']
    if allow_legacy_b0:
        assert current.get('hparam_search') and hparams(old)==hparams({})
        assert previous['completed_steps']==500
        assert old['fusion']=='balanced_stack' and old['visual']=='patch'
        assert file_sha(current['resume'])=='c09e13d6d636c69491914643db81d8270b90d1c85ef0729513d240c4b108293d'
        assert set(new_code)==set(old_code)|{'model/balanced_hparam_search.py'}
    else:
        assert old_code.keys() == new_code.keys(), 'Resume code manifest keys changed'
    for path, digest in old_code.items():
        if digest != new_code[path]:
            if allow_legacy_b0 and path=='train/train_nested_semantic_mask.py':
                continue
            assert path == trainer and digest == expected_parent_trainer_sha256, f'Resume code mismatch: {path}'
    if expected_parent_trainer_sha256 is not None:
        assert old_code[trainer] == expected_parent_trainer_sha256
    completed = int(previous['completed_steps'])
    assert previous['scheduler_horizon'] == current['horizon']
    assert 0 <= completed < current['max_updates'] <= current['horizon'], 'Invalid continuation stop'
    assert previous['next_epoch'] == completed // current['batches_per_epoch']
    assert previous['next_batch'] == completed % current['batches_per_epoch']
    assert len(previous['rng_per_rank']) == current['world_size']
    return completed



class CappedSampler:
    """A rank-local prefix of DistributedSampler with identical order and padding."""

    def __init__(self, sampler, samples):
        self.sampler = sampler
        self.samples = min(int(samples), len(sampler))

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.samples)

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

def consumed_batch(epoch, batch_index, batches_per_epoch, completed):
    return epoch * batches_per_epoch + batch_index < completed


def atomic_save(payload, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.rename(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'])


def save_checkpoint(module, optimizer, config, completed, output):
    # This function MUST be entered by every rank.
    states = [None] * dist.get_world_size()
    local_state=rng_state()
    if hasattr(module,'_loader_generator'):
        boundary=completed % config['batches_per_epoch']==0
        local_state['loader_generator']= (module._loader_generator.get_state() if boundary else
                                         module._loader_epoch_generator_state)
    dist.all_gather_object(states, local_state)
    if dist.get_rank() == 0:
        adapter = auxiliary_module(module).state_dict() if auxiliary_module(module) is not None else None
        atomic_save(dict(model=module.clip.state_dict(), adapter=adapter,
                         optimizer=optimizer.state_dict(),
                         completed_steps=completed, scheduler_horizon=config['horizon'],
                         stop_updates=config['max_updates'], rng_per_rank=states,
                         next_epoch=completed // config['batches_per_epoch'],
                         next_batch=completed % config['batches_per_epoch'], config=config),
                    Path(output) / f'step{completed:06d}.pt')
    dist.barrier()


def save_emergency(module, optimizer, config, completed, output):
    destination = Path(output) / 'emergency'
    if dist.get_rank() == 0:
        destination.mkdir(exist_ok=False)
    dist.barrier()
    save_checkpoint(module, optimizer, config, completed, destination)


def code_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = ['model/nested_semantic_mask.py', 'model/nested_vcp_mask.py', 'model/nested_fusion_mask.py', 'train/nested_semantic_data.py',
             'train/train_nested_semantic_mask.py', 'train/random_detail_observer.py', 'model/model_longclip.py',
             'model/longclip.py', 'model/said_cls_cvssl.py']
    if (root/'model/balanced_hparam_search.py').exists():
        paths.append('model/balanced_hparam_search.py')
    if (root/'model/nested_support_band.py').exists():
        paths.append('model/nested_support_band.py')
    return {p: file_sha(root / p) for p in paths}


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def setup():
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    assert world == 4, 'Both real smoke and formal jobs require WORLD_SIZE=4'
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    check = torch.tensor(float(rank+1), device='cuda')
    started = time.perf_counter()
    dist.all_reduce(check)
    torch.cuda.synchronize()
    assert check.item() == 10.
    prop = torch.cuda.get_device_properties(local)
    identity = dict(rank=rank, local_rank=local, pid=os.getpid(), hostname=socket.gethostname(),
                    gpu_name=prop.name, uuid=str(prop.uuid), total_memory=prop.total_memory,
                    nccl_all_reduce=check.item(), nccl_seconds=time.perf_counter()-started)
    peers = [None] * world
    dist.all_gather_object(peers, identity)
    assert len({p['hostname'] for p in peers}) == 1
    assert len({p['uuid'] for p in peers}) == 4
    assert all('A100' in p['gpu_name'] and 'MIG' not in p['gpu_name'] for p in peers)
    # Gradient of each rank's differently weighted global sum must add at owners.
    from model.nested_semantic_mask import gather as diff_gather
    x = torch.tensor([[float(rank)]], device='cuda', requires_grad=True)
    (diff_gather(x).sum() * (rank+1)).backward()
    assert x.grad.item() == 10.
    return rank, local, world, peers


def parameter_agreement(module):
    difference = torch.zeros((), device='cuda')
    with torch.no_grad():
        for p in module.parameters():
            reference = p.detach().clone()
            work = dist.broadcast(reference, src=0, async_op=True)
            work.wait()
            difference = torch.maximum(difference, (p-reference).abs().max())
    torch.cuda.synchronize()
    dist.barrier()
    return float(difference)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--init-state', required=True)
    p.add_argument('--index-dir', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--run-type', choices=['probe', 'smoke', 'formal'], required=True)
    p.add_argument('--max-updates', type=int, required=True)
    p.add_argument('--resume')
    p.add_argument('--legacy-b0',action='store_true',help='Pinned default Balanced step500 optimizer migration only')
    p.add_argument('--expected-parent-trainer-sha256',
                   help='Explicitly pin a compatible predecessor trainer; model/data/objective hashes must still match')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get('hparam_search'):
        cfg.update(hparams(cfg))
    cfg.setdefault('sampling_mode', 'fixed_first')
    cfg.setdefault('sampling_seed', 0)
    cfg.setdefault('experiment_name', cfg['arm'])
    cfg.setdefault('condition_mode', 'text_only')
    cfg.setdefault('shuffle_seed', 0)
    cfg.setdefault('image_chunk', 32)
    cfg.setdefault('text_chunk', 64)
    cfg.setdefault('checkpoint_pair_blocks', True)
    cfg.setdefault('full_native_mix', 0.)
    cfg.setdefault('checkpoint_interval', 100)
    cfg.setdefault('save_initial_checkpoint', True)
    assert cfg['sampling_mode'] in ('fixed_first', 'random_k', 'summary_detail', 'summary_random_detail', 'summary_all_detail', 'nested_detail', 'nested_detail_d3', 'nested_detail_kr234', 'nested_detail_kr2m1', 'interior_random_k', 'summary_contiguous_detail')
    if cfg['sampling_mode'] in ('nested_detail', 'nested_detail_d3','nested_detail_kr234','nested_detail_kr2m1'):
        assert cfg.get('inclusion_hierarchy') == 'detail_chain'
        allowed = (([1.,1.,1.], [1.35,1.35,.30], [1.4,1.4,.20], [1.375,1.375,.25],
                    [1.325,1.325,.35], [1.3,1.3,.40]) if cfg['sampling_mode']=='nested_detail_d3'
                   else ([1.35,1.35,.30],) if cfg['sampling_mode'] in ('nested_detail_kr234','nested_detail_kr2m1')
                   else ([1.4,1.4,.2], [1.,1.,1.]))
        assert cfg['view_weights'] in allowed, 'Only reviewed Nested Detail weights are authorized'
    if cfg['sampling_mode'] in ('summary_random_detail', 'summary_contiguous_detail', 'summary_all_detail'):
        from train.random_detail_observer import install
        install()  # Read-only detached telemetry; model/loss source stays unchanged.
    assert cfg['condition_mode'] in ('text_only', 'joint_image', 'joint_shuffled_image', 'vcp_mask', 'dual_branch')
    assert cfg['full_native_mix'] == 0
    assert int(cfg['checkpoint_interval']) > 0
    assert (args.max_updates == 5 if args.run_type == 'smoke' else args.max_updates == 35 if args.run_type == 'probe' else args.max_updates > 0)
    assert cfg['batch_size'] == 256 and cfg['world_size'] == 4 and cfg['accumulation'] == 1
    assert cfg['seed'] == 0 and cfg['workers'] == 8
    training_horizon(cfg, 1217)
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rank, local, world, peers = setup()
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    dataset = NestedDataset(args.index_dir, args.image_root, cfg['sampling_mode'], cfg['sampling_seed'])
    base_sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                      seed=cfg['seed'], drop_last=False)
    full_batches_per_epoch = math.ceil(len(base_sampler) / cfg['batch_size'])
    # Stops below one epoch consume the exact same sampler prefix but let workers
    # reach StopIteration naturally instead of aborting on an early loop break.
    sampler = (CappedSampler(base_sampler, args.max_updates * cfg['batch_size'])
               if args.max_updates < full_batches_per_epoch else base_sampler)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], sampler=sampler, collate_fn=collate,
                        num_workers=cfg['workers'], drop_last=False, pin_memory=True,
                        multiprocessing_context='spawn', prefetch_factor=2,
                        generator=torch.Generator().manual_seed(cfg['seed']))
    horizon = training_horizon(cfg, full_batches_per_epoch)
    assert len(dataset) == 1245901, f'Unexpected dataset size {len(dataset)}; investigate before training'
    assert len(loader) == min(args.max_updates, full_batches_per_epoch)
    assert args.max_updates <= horizon, 'Stopping point must not exceed the configured horizon'
    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    initial = torch.load(args.init_state, map_location='cpu', weights_only=False)
    assert initial['completed_steps'] == 0 and initial['provenance']['source'] == 'OpenAI CLIP + original random MaskNetwork'
    assert not initial['optimizer']['state']
    clip.load_state_dict(initial['model'], strict=True)
    construction_rng = rng_state()
    module_class = (NestedFusionMask if cfg['condition_mode'] == 'dual_branch' else
                    NestedVCPMask if cfg['condition_mode'] == 'vcp_mask' else NestedSemanticMask)
    model_options = ({'fusion': cfg['fusion'], 'visual': cfg['visual']}
                     if cfg['condition_mode'] == 'dual_branch' else {})
    if cfg.get('hparam_search'):
        module_class=BalancedSearch
        model_options['search_hparams']=hparams(cfg)
        if 'inclusion_hierarchy' in cfg:
            model_options['inclusion_hierarchy']=cfg['inclusion_hierarchy']
        if 'view_sparsity_weights' in cfg:
            model_options['view_sparsity_weights']=cfg['view_sparsity_weights']
        if 'regularizer_mode' in cfg:
            model_options['regularizer_mode']=cfg['regularizer_mode']
        if 'support_bands' in cfg:
            model_options['support_bands']=cfg['support_bands']
        if 'summary_t2i_weight' in cfg:
            model_options['search_hparams']['summary_t2i_weight']=cfg['summary_t2i_weight']
    module = module_class(clip.float(), arm=cfg['arm'],
                          checkpoint_encoders=cfg['checkpoint_encoders'],
                          image_chunk=cfg['image_chunk'], text_chunk=cfg['text_chunk'],
                          condition_mode=cfg['condition_mode'],
                          shuffle_seed=cfg['shuffle_seed'],
                          checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'], **model_options)
    restore_rng_state(construction_rng)
    encoder_checkpoint_active = all(
        getattr(transformer.forward, '__func__', None) is _checkpoint_blocks
        for transformer in (module.clip.visual.transformer, module.clip.transformer))
    runtime_model = dict(image_chunk=module.image_chunk, text_chunk=module.text_chunk,
                         checkpoint_pair_blocks=module.checkpoint_pair_blocks,
                         checkpoint_encoders=module.checkpoint_encoders,
                         encoder_checkpoint_active=encoder_checkpoint_active,
                         condition_mode=module.condition_mode, arm=module.arm)
    assert runtime_model == dict(image_chunk=cfg['image_chunk'], text_chunk=cfg['text_chunk'],
                                 checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'],
                                 checkpoint_encoders=cfg['checkpoint_encoders'],
                                 encoder_checkpoint_active=cfg['checkpoint_encoders'],
                                 condition_mode=cfg['condition_mode'], arm=cfg['arm'])
    adapter_initialization = None
    auxiliary = auxiliary_module(module)
    if auxiliary is not None:
        adapter_initialization = dict(
            module=type(auxiliary).__name__, state_sha256=state_digest(auxiliary.state_dict()),
            parameters={name: dict(shape=list(parameter.shape),
                                   nonzero=int(parameter.count_nonzero()))
                        for name, parameter in auxiliary.named_parameters()})
    module = module.cuda().train()
    assert all(p.dtype == torch.float32 for p in module.parameters())
    ddp = DDP(module, device_ids=[local], output_device=local,
              find_unused_parameters=True, static_graph=False)
    optimizer = build_optimizer(module)
    if cfg.get('hparam_search'):
        module._loader_generator=loader.generator
        module._loader_epoch_generator_state=loader.generator.get_state()
    assert not optimizer.state
    del initial
    config = dict(**cfg, **vars(args), horizon=horizon, batches_per_epoch=full_batches_per_epoch,
                  loader_batches_this_epoch=len(loader), training_records=len(dataset),
                  sampler_num_samples=len(base_sampler),
                  sampler_padding=world*len(base_sampler)-len(dataset),
                  tail_batch=len(base_sampler) % cfg['batch_size'],
                  init_sha256=file_sha(args.init_state), data=dataset.metadata,
                  code_sha256=code_manifest(),
                  runtime_model=runtime_model,
                  adapter_initialization=adapter_initialization,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                  environment={k:v for k,v in os.environ.items() if k.startswith('NCCL') or k=='CUDA_VISIBLE_DEVICES'},
                  ranks=peers, gather_gradient_test={'expected':10., 'actual':10., 'passed':True})
    if model_options:
        config['runtime_model'].update(model_options)
        config['parameter_counts'] = dict(
            total=sum(p.numel() for p in module.parameters()),
            trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
            added=sum(p.numel() for p in module.fusion_branch.parameters()),
            optimizer_groups={g['name']: sum(p.numel() for p in g['params']) for g in optimizer.param_groups})
        branch = module.fusion_branch
        config['component_initialization'] = dict(
            clip=state_digest(module.clip.state_dict()),
            text_blocks=state_digest(module.clip.mask_net.resblocks.state_dict()),
            visual_blocks=state_digest(branch.visual_blocks.state_dict()),
            visual_adapter=state_digest(branch.visual_adapter.state_dict()),
            crossscore_query=state_digest(branch.query.state_dict()) if hasattr(branch, 'query') else None,
            crossscore_key=state_digest(branch.key.state_dict()) if hasattr(branch, 'key') else None,
            crossscore_readout=state_digest(branch.readout.state_dict()) if hasattr(branch, 'readout') else None,
            channel_gate=state_digest(branch.gate.state_dict()) if hasattr(branch, 'gate') else None)
    completed = 0
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=False)
        completed = validate_resume_payload(previous, config, args.expected_parent_trainer_sha256,args.legacy_b0)
        module.clip.load_state_dict(previous['model'], strict=True)
        auxiliary = auxiliary_module(module)
        if auxiliary is None:
            assert previous.get('adapter') is None
        else:
            auxiliary.load_state_dict(previous['adapter'], strict=True)
        if args.legacy_b0:
            config['optimizer_migration']=migrate_legacy_optimizer(previous['optimizer'],module,optimizer)
        else:
            optimizer.load_state_dict(previous['optimizer'])
        state = previous['rng_per_rank'][rank]
        restore_rng_state(state)
        if cfg.get('hparam_search') and 'loader_generator' in state:
            loader.generator.set_state(state['loader_generator'])
        config.update(parent_checkpoint_sha256=file_sha(args.resume), parent_completed_updates=completed,
                      parent_git_head=previous['config']['git_head'],
                      parent_code_sha256=previous['config']['code_sha256'])
        del previous
    config['start_updates'] = completed
    config['updates_planned_this_run'] = args.max_updates - completed
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        print(json.dumps({'event':'ready', 'horizon':horizon, 'start_updates':completed,
                          'stop_updates':args.max_updates, 'ranks':peers}), flush=True)
    if args.run_type != 'probe' and cfg['save_initial_checkpoint']:
        save_checkpoint(module, optimizer, config, completed, output)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    updates_here = 0
    probe_max_seconds = []
    resource_failure = None
    consecutive_slow = 0
    for epoch in range(cfg['epochs']):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)  # copied into fresh spawn workers before iter(loader)
        if completed >= (epoch+1)*full_batches_per_epoch:
            continue
        cycle_start = time.perf_counter()
        if cfg.get('hparam_search'):
            module._loader_epoch_generator_state=loader.generator.get_state()
        loader_iterator = iter(loader)
        for batch_index, batch in enumerate(loader_iterator):
            if consumed_batch(epoch, batch_index, full_batches_per_epoch, completed):
                cycle_start = time.perf_counter()
                continue
            tick = time.perf_counter()
            for k in ('image', 'tokens_f', 'tokens_o', 'tokens_e', 'valid'):
                batch[k] = batch[k].cuda(non_blocking=True)
            lrs = optimizer_learning_rates(module,completed,horizon)
            for group, lr in zip(optimizer.param_groups, lrs):
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, logs = ddp(batch['image'], batch['tokens_f'], batch['tokens_o'], batch['tokens_e'],
                              batch['valid'], completed)
            finite = torch.isfinite(loss).int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                if cfg.get('monitor_resources', False):
                    save_emergency(module, optimizer, config, completed, output)
                raise FloatingPointError(f'Nonfinite loss: {logs}')
            loss.backward()
            norms = {}
            for group in optimizer.param_groups:
                norms[group['name']] = gradient_norm(group['params'])
            adapter_norms = {}
            auxiliary = auxiliary_module(module)
            if auxiliary is not None:
                adapter_norms = {name: gradient_norm([parameter])
                                 for name, parameter in auxiliary.named_parameters()}
            finite = torch.stack([torch.isfinite(v) for v in norms.values()]).all().int()
            if adapter_norms:
                finite = finite * torch.stack(
                    [torch.isfinite(v) for v in adapter_norms.values()]).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                if cfg.get('monitor_resources', False):
                    save_emergency(module, optimizer, config, completed, output)
                raise FloatingPointError('Nonfinite parameter gradient on at least one rank')
            optimizer.step()
            if cfg.get('resource_policy') in ('nfs_reproduction500', 'local_reproduction500'):
                finite = torch.stack([torch.isfinite(p).all() for p in module.parameters()]).all().int()
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite parameter after AdamW on at least one rank')
            completed += 1
            updates_here += 1
            torch.cuda.synchronize()
            stream = json.dumps(dict(sample_ids=batch['sample_id'].tolist(), views=batch['views'],
                                     tokens=[batch[k].cpu().tolist() for k in ('tokens_f','tokens_o','tokens_e')]),
                                ensure_ascii=False, separators=(',', ':')).encode()
            health = dict(rank=rank, updates=updates_here, batch=len(batch['image']),
                          valid=int(batch['valid'].sum()), seconds=time.perf_counter()-tick,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          stream_sha256=hashlib.sha256(stream).hexdigest(),
                          reasons=dict(Counter(batch['reason'])),
                          gradient_norms={k:float(v) for k,v in norms.items()}, gradients_finite=True,
                          adapter_gradient_norms={k:float(v) for k,v in adapter_norms.items()},
                          sampling=sampling_diagnostics(batch))
            rank_health = [None]*world
            dist.all_gather_object(rank_health, health)
            ids = gather(batch['image_id'].cuda(), False)
            unique, counts = ids.unique(return_counts=True)
            duplicates = unique[counts>1].cpu().tolist()
            if rank == 0:
                row = dict(step=completed, s=completed-1, epoch=epoch,
                           lr_backbone=lrs[0], lr_mask=lrs[1],
                           lr_adapter=lrs[-1] if len(lrs) >= 3 else 0.,
                           actual_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                           rank_health=rank_health, duplicate_image_ids=duplicates,
                           **{k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()})
                with (output / 'steps.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps({'step':completed, 'loss':row['loss'], 'V':row['valid_global'],
                                  'seconds':max(x['seconds'] for x in rank_health)}), flush=True)
                if completed == 1:
                    (output / 'text_examples.json').write_text(json.dumps(dict(scope='rank0 first batch; raw full text only for explicit overlong fallback',
                        views=batch['views'][:8], reasons=batch['reason'][:8],
                        untruncated_lengths=batch['untruncated_lengths'][:8],
                        sampling_mode=cfg['sampling_mode'],
                        local_view_labels=(['Summary', 'Random Detail'] if cfg['sampling_mode']=='summary_random_detail' else
                                           ['Summary', 'Contiguous Detail'] if cfg['sampling_mode']=='summary_contiguous_detail' else
                                           ['Summary', 'All Detail'] if cfg['sampling_mode']=='summary_all_detail' else
                                           ['All Detail', 'Atomic Detail'] if cfg['sampling_mode']=='nested_detail' else
                                           ['All Detail', 'Partial Detail D3'] if cfg['sampling_mode']=='nested_detail_d3' else
                                           ['All Detail', 'Random Detail K234'] if cfg['sampling_mode']=='nested_detail_kr234' else
                                           ['All Detail', 'Random Detail K2..m-1'] if cfg['sampling_mode']=='nested_detail_kr2m1' else
                                           ['Summary', 'Detail'] if cfg['sampling_mode']=='summary_detail' else
                                           ['prefix', 'remainder'] if cfg['sampling_mode'] in ('random_k', 'interior_random_k') else ['overview','elaboration']),
                        sample_ids=batch['sample_id'][:8].tolist(),
                        n=batch['n'][:8].tolist(), K=batch['K'][:8].tolist()), indent=2))
            checkpoint_due = args.run_type != 'probe' and (
                completed % int(cfg['checkpoint_interval']) == 0 or completed == args.max_updates)
            if args.run_type == 'probe' or cfg.get('monitor_resources', False):
                # End-to-end cycle: real DataLoader wait, transfer, update, communication,
                # ordinary logging, then synchronization on the slowest rank.
                dist.barrier(device_ids=[local])
                torch.cuda.synchronize(local)
                full = torch.tensor(time.perf_counter() - cycle_start, device='cuda')
                dist.all_reduce(full, op=dist.ReduceOp.MAX)
                if args.run_type == 'probe' and updates_here > 5:
                    probe_max_seconds.append(float(full))
                if rank == 0:
                    timing_name = 'probe_timing.jsonl' if args.run_type == 'probe' else 'cycle_timing.jsonl'
                    with (output / timing_name).open('a') as f:
                        f.write(json.dumps(dict(step=completed, warmup=updates_here <= 5,
                                                four_rank_max_seconds=float(full)))+'\n')
                cycle_seconds = float(full)
                if args.run_type == 'probe':
                    if updates_here > 5 and cycle_seconds > 3:
                        resource_failure = 'measured full update exceeded 3 seconds'
                    elif updates_here >= 2 and cycle_seconds > cfg.get('feasibility_abort_seconds', float('inf')):
                        resource_failure = 'limited full-update feasibility probe exceeded its abort threshold'
                elif cfg.get('resource_policy') in ('nfs_reproduction500', 'local_reproduction500'):
                    if cfg['resource_policy'] == 'local_reproduction500':
                        from recovery.local500_policy import cycle_state
                    else:
                        from recovery.nfs500_policy import cycle_state
                    policy = cycle_state(cycle_seconds, consecutive_slow)
                    consecutive_slow = policy['consecutive']
                    if policy['warning'] and rank == 0:
                        print(json.dumps(dict(event='PERFORMANCE_WARNING', step=completed,
                                              full_cycle_s=cycle_seconds)), flush=True)
                    if policy['stop']:
                        resource_failure = ('single full update exceeded 60 seconds' if cfg['resource_policy'] == 'local_reproduction500'
                                            else 'five consecutive full updates exceeded 30 seconds')
                elif updates_here > 1:
                    consecutive_slow = consecutive_slow + 1 if cycle_seconds > 3 else 0
                    if consecutive_slow >= 3:
                        resource_failure = 'three consecutive full updates exceeded 3 seconds'
                if cfg.get('resource_policy') not in ('nfs_reproduction500', 'local_reproduction500') and torch.cuda.max_memory_allocated() / 2**30 > 65:
                    resource_failure = 'peak allocated memory exceeded 65 GiB'
                saturation_limit = cfg.get('saturation_abort_fraction')
                if cfg.get('resource_policy') not in ('nfs_reproduction500', 'local_reproduction500') and saturation_limit is not None and any(
                        float(value) >= saturation_limit for name, value in logs.items()
                        if name.endswith('_sigmoid_saturation')):
                    resource_failure = 'sigmoid saturation reached the recorded 99% stop criterion'
                failed = torch.tensor(int(resource_failure is not None), device='cuda')
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                if failed.item():
                    resource_failure = resource_failure or 'another rank exceeded the memory limit'
                    if args.run_type != 'probe' or resource_failure.startswith('sigmoid saturation'):
                        save_checkpoint(module, optimizer, config, completed, output)
                    break
            if checkpoint_due:
                checkpoint_started = time.perf_counter()
                save_checkpoint(module, optimizer, config, completed, output)
                checkpoint_seconds = torch.tensor(time.perf_counter() - checkpoint_started, device='cuda')
                dist.all_reduce(checkpoint_seconds, op=dist.ReduceOp.MAX)
                if rank == 0:
                    with (output / 'checkpoint_timing.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=completed, four_rank_max_seconds=float(checkpoint_seconds)))+'\n')
            cycle_start = time.perf_counter()
            # At a full epoch boundary let DataLoader exhaust naturally, so its
            # workers finish through StopIteration rather than only __del__.
            if completed >= args.max_updates and batch_index + 1 < len(loader):
                break
        # An early stop leaves most of the epoch unconsumed.  Join spawn workers
        # before any final NCCL collective so worker aborts cannot race teardown.
        shutdown = getattr(loader_iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
        del loader_iterator
        if completed >= args.max_updates or resource_failure is not None:
            break
    difference = parameter_agreement(module)
    final_nccl = torch.tensor(float(rank + 1), device='cuda')
    dist.all_reduce(final_nccl)
    torch.cuda.synchronize()
    assert final_nccl.item() == 10., 'Final four-rank NCCL all-reduce failed'
    summary = dict(rank=rank, completed_updates=completed, updates_this_run=updates_here,
                   max_parameter_difference_from_rank0=difference, seconds=time.perf_counter()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   final_nccl_all_reduce=final_nccl.item())
    results = [None]*world
    dist.all_gather_object(results, summary)
    assert all(x['completed_updates'] == (completed if resource_failure else args.max_updates) for x in results)
    assert all(x['max_parameter_difference_from_rank0'] == 0 for x in results)
    if rank == 0:
        speed_gate = None
        passed = resource_failure is None
        if args.run_type == 'probe':
            values = torch.tensor(probe_max_seconds, dtype=torch.float64)
            assert len(values) == 30 or resource_failure is not None
            speed_gate = dict(
                warmup_steps=5, measured_steps=len(values), threshold_seconds=3.,
                mean_seconds=float(values.mean()) if len(values) else None,
                median_seconds=float(values.median()) if len(values) else None,
                p95_seconds=float(torch.quantile(values, .95)) if len(values) else None,
                max_seconds=float(values.max()) if len(values) else None,
                all_steps_at_most_3s=bool((values <= 3.).all()) if len(values) == 30 else False,
                allocated_limit_gib=65.,
                every_rank_peak_allocated_at_most_65gib=all(
                    x['peak_allocated_gib'] <= 65. for x in results))
            passed = (passed and speed_gate['all_steps_at_most_3s'] and
                      speed_gate['every_rank_peak_allocated_at_most_65gib'])
        (output / 'acceptance.json').write_text(json.dumps(
            dict(passed=passed, ranks=results, speed_gate=speed_gate,
                 resource_failure=resource_failure), indent=2))
    dist.barrier(device_ids=[local])
    torch.cuda.synchronize(local)
    del ddp, optimizer, module
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(device_ids=[local])
    torch.cuda.synchronize(local)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
