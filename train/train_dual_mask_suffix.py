"""Minimal training entry for S0 Dual-Mask Suffix Clean v0.1.

This is a thin production path, not a compatibility runner.  It supports strict continuation
from a complete checkpoint, preserving optimizer, scheduler position, and the distributed data
stream position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from model import longclip  # noqa: E402
from model.dual_mask_suffix import (  # noqa: E402
    DualMaskSuffixTrainModule,
    SUFFIX_LAMBDA,
    split_caption_suffix,
)
from model.said_cls_cvssl import (  # noqa: E402
    said_mask_from_hidden,
)
from scheduler import cosine_lr  # noqa: E402
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate, stateless_seed  # noqa: E402
from train_said_cls_cvssl import load_init_state, state_digest  # noqa: E402


FORMAL_INIT = "/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt"
FORMAL_CONFIG = {
    "world_size": 4,
    "batch_size_per_gpu": 256,
    "max_steps": 500,
    "seed": 0,
    "accumulation": 1,
    "clip_lr": 1e-6,
    "mask_lr": 1e-3,
    "suffix_lr": 1e-4,
    "weight_decay": 1e-2,
    "mask_weight_decay": 0.0,
    "suffix_weight_decay": 0.0,
    "warmup": 200,
    "image_chunk": 16,
    "text_chunk": 32,
}


def split_suffix_record(record: Dict, prefix_k: int, caption_said: str) -> Tuple[str, str, str]:
    """Extract the original full caption and its valid suffix without drawing a new K."""
    full = record["conversations"][1]["value"].replace("\n", " ")
    prefix, suffix = split_caption_suffix(full, prefix_k, caption_said)
    return full, prefix, suffix


def _eot_token() -> int:
    return int(longclip._tokenizer.encoder["<|endoftext|>"])


def suffix_has_content(text: str) -> bool:
    """Use the project's SOT/EOT positions, rather than token-id nonzero counting."""
    tokens = longclip.tokenize([text], truncate=True)[0]
    positions = (tokens == _eot_token()).nonzero(as_tuple=False)
    eot_position = int(positions[0].item()) if positions.numel() else int(tokens.numel())
    return eot_position > 1


class DualMaskSuffixDataset(Share4VCvsslDataset):
    """The base dataset plus full caption, suffix text and a token-valid flag."""

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)
        record = self.json_data[index]
        full, prefix, suffix = split_suffix_record(record, sample["prefix_k"], sample["caption_said"])
        if prefix != sample["caption_said"]:
            raise AssertionError("base prefix stream changed while adding suffix")
        sample.update({
            "caption_full": full,
            "suffix_text": suffix,
            "suffix_valid": suffix_has_content(suffix),
        })
        return sample


def dual_mask_suffix_collate(samples: List[Dict]) -> Dict:
    """Collate only fields used by this experiment; preserve text as ragged Python lists."""
    batch = cvssl_collate(samples)
    batch.update({
        "caption_full": [s["caption_full"] for s in samples],
        "suffix_text": [s["suffix_text"] for s in samples],
        "suffix_valid": torch.tensor([s["suffix_valid"] for s in samples], dtype=torch.bool),
    })
    return batch


def build_optimizers(module: DualMaskSuffixTrainModule, args):
    """Separate CLIP, original S0 mask and suffix gate parameter groups."""
    mask_ids = {id(p) for p in module.clip.mask_net.parameters()}
    clip_params, mask_params = [], []
    for p in module.clip.parameters():
        if not p.requires_grad:
            continue
        (mask_params if id(p) in mask_ids else clip_params).append(p)
    clip_opt = torch.optim.AdamW(clip_params, lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    mask_opt = torch.optim.AdamW(mask_params, lr=args.mask_lr, weight_decay=0.0,
                                 betas=(0.9, 0.999), eps=1e-8)
    suffix_params = list(module.suffix_gate.parameters()) if module.suffix_gate is not None else []
    suffix_opt = torch.optim.AdamW(suffix_params, lr=args.suffix_lr, weight_decay=0.0,
                                   betas=(0.9, 0.999), eps=1e-8) if suffix_params else None
    return clip_opt, mask_opt, suffix_opt


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is present in the base env
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_ddp() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            dist.init_process_group(backend="gloo", rank=rank, world_size=world)
            return rank, local_rank, world
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    elif not dist.is_initialized():
        # The unchanged S0 helper uses autograd-aware all_gather even for a one-process run.
        # Give that path a real process group so single-GPU debug runs exercise the same graph.
        init_file = "/tmp/said_dual_mask_suffix_clean_v01_pg_%d" % os.getpid()
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
        dist.init_process_group(backend="gloo", init_method="file://" + init_file,
                                rank=0, world_size=1)
    return rank, local_rank, world


def _all_ranks_finite(value: torch.Tensor, world: int) -> bool:
    flag = torch.tensor(1 if bool(torch.isfinite(value).all()) else 0,
                        device=value.device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _check_gradients_finite(module: torch.nn.Module, world: int, device: torch.device) -> None:
    local_ok = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                   for parameter in module.parameters())
    flag = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise FloatingPointError("non-finite loss or gradient on at least one rank")


def _gradient_health(module):
    groups = {}
    for name, parameter in module.named_parameters():
        if name.startswith('suffix_gate.'):
            group = '.'.join(name.split('.')[:2])
        elif name.startswith('clip.visual.'):
            group = 'visual'
        elif name.startswith('clip.mask_net.'):
            group = 's0_mask'
        else:
            group = 'text'
        if parameter.grad is not None:
            value = parameter.grad.detach().float().square().sum()
            groups[group] = groups.get(group, 0.0) + value
    return {key: float(value.sqrt()) for key, value in groups.items()}


def _atomic_torch_save(payload: Dict, path: str) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str, module: DualMaskSuffixTrainModule, optimizers=None) -> Dict:
    """Strictly load a production checkpoint; missing mode-specific state is an error."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a dict")
    module.clip.load_state_dict(payload["clip_state"], strict=True)
    gate_state = payload.get("suffix_gate_state")
    if module.suffix_mode == "masked":
        if not isinstance(gate_state, dict) or not gate_state:
            raise RuntimeError("masked checkpoint is missing suffix_gate_state")
        module.suffix_gate.load_state_dict(gate_state, strict=True)
    elif gate_state is not None:
        raise RuntimeError("native checkpoint must store suffix_gate_state=None")
    if optimizers is not None:
        for key, optimizer in zip(("clip", "mask", "suffix"), optimizers):
            if optimizer is not None:
                optimizer.load_state_dict(payload["optimizer_states"][key])
    return payload


def export_bare_student(checkpoint_path: str, output_path: str) -> str:
    """Export only the CLIP student state for legacy S0 evaluation consumers."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("clip_state"), dict):
        raise TypeError("checkpoint has no clip_state mapping")
    _atomic_torch_save(payload["clip_state"], output_path)
    return output_path


def resume_position(payload: Dict, config: Dict) -> Tuple[int, int]:
    """Recover the next batch from completed updates, including legacy final checkpoints.

    The legacy loop fetched the next batch before its stop check, so its final saved
    step_in_epoch could be one ahead of the last update. completed_steps is authoritative.
    """
    previous = payload['config']
    fixed = ('suffix_mode', 'suffix_lambda', 'batch_size_per_gpu', 'world_size', 'epochs',
             'loader_batches', 'lr_horizon_steps', 'seed', 'total_len', 'image_chunk',
             'text_chunk', 'precision', 'run_type', 'init_state', 'accumulation', 'u_sparsity')
    for key in fixed:
        if previous.get(key) != config.get(key):
            raise RuntimeError(f'resume configuration mismatch: {key}')
    for key in ('base_model', 'lr', 'mask_lr', 'suffix_lr', 'weight_decay', 'warmup', 'num_workers'):
        if previous['arguments'].get(key) != config['arguments'].get(key):
            raise RuntimeError(f'resume argument mismatch: {key}')
    completed = int(payload['completed_steps'])
    if not 0 < completed < config['max_steps'] <= config['lr_horizon_steps']:
        raise RuntimeError('resume must continue positive updates within the original horizon')
    if previous.get('formal_optimizer_updates') != completed:
        raise RuntimeError('resume formal update count mismatch')
    return divmod(completed, config['loader_batches'])


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S0 Dual-Mask Suffix Clean v0.1")
    parser.add_argument("--suffix-mode", choices=("native", "masked"), required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="B16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--mask-lr", type=float, default=1e-3)
    parser.add_argument("--suffix-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", default=FORMAL_INIT)
    parser.add_argument("--image-chunk", type=int, default=16)
    parser.add_argument("--text-chunk", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-len", type=int, default=1000)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-type", choices=("debug", "formal"), default="debug")
    return parser.parse_args()


def main() -> int:
    args = _args()
    existing_log = os.path.join(args.output_dir, 'salu_log.jsonl')
    if not args.resume and os.path.exists(existing_log):
        raise SystemExit('output directory already contains a run; refusing to mix update counts')
    if args.resume and not os.path.isfile(args.resume):
        raise SystemExit(f'resume checkpoint does not exist: {args.resume}')
    _seed_everything(args.seed)
    rank, local_rank, world = _setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    base_name = {"B16": "ViT-B/16", "L14": "ViT-L/14"}.get(args.base_model, args.base_model)
    model, _ = longclip.load_from_clip(base_name, device="cpu", args=args)
    model.train()
    load_init_state(model, args.init_state, rank)
    module = DualMaskSuffixTrainModule(model, suffix_mode=args.suffix_mode, rank=rank,
                                       image_chunk=args.image_chunk, text_chunk=args.text_chunk).to(device)
    if world > 1:
        module = torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True, static_graph=False)
    inner = getattr(module, "module", module)
    clip_opt, mask_opt, suffix_opt = build_optimizers(inner, args)
    dataset = DualMaskSuffixDataset(seed=args.seed, total_len=args.total_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed) \
        if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.num_workers, pin_memory=True,
                        collate_fn=dual_mask_suffix_collate, drop_last=False)
    horizon = args.epochs * len(loader)
    clip_schedule = cosine_lr(clip_opt, args.lr, args.warmup, horizon)
    mask_schedule = cosine_lr(mask_opt, args.mask_lr, 0, horizon)
    suffix_schedule = cosine_lr(suffix_opt, args.suffix_lr, 0, horizon) if suffix_opt is not None else None
    os.makedirs(args.output_dir, exist_ok=True)
    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(args.resume, inner, (clip_opt, mask_opt, suffix_opt))
        if resume_payload.get('config', {}).get('suffix_mode') != args.suffix_mode:
            raise RuntimeError('resume suffix_mode does not match requested suffix_mode')
    config = {
        "objective": "s0_dual_mask_suffix_clean_v01", "suffix_mode": args.suffix_mode,
        "suffix_lambda": SUFFIX_LAMBDA, "suffix_gate": "Sequential(Linear(1024,512),GELU,Linear(512,512))",
        "suffix_gate_init": "seed=0;xavier_uniform,bias=0;last_weight=0,last_bias=log(8)",
        "batch_size_per_gpu": args.batch_size, "world_size": world, "epochs": args.epochs,
        "loader_batches": len(loader), "lr_horizon_steps": horizon, "max_steps": args.max_steps,
        "seed": args.seed, "total_len": args.total_len, "image_chunk": args.image_chunk,
        "text_chunk": args.text_chunk, "precision": "fp32 master + "+args.amp_dtype+" autocast",
        "run_type": args.run_type,
        "formal_optimizer_updates": 0,
        "debug_optimizer_updates": 0,
        "init_state": args.init_state,
        "training_sha": _git_head(), "arguments": vars(args),
        "accumulation": 1, "u_sparsity": 0.0,
        "communication_env": {k: os.environ.get(k) for k in
            ('NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 'GLOO_SOCKET_IFNAME', 'CUDA_VISIBLE_DEVICES')},
    }
    completed = 0
    resume_epoch = 0
    resume_step_in_epoch = -1
    if resume_payload is not None:
        completed = int(resume_payload.get('completed_steps', 0))
        resume_epoch, resume_next_batch = resume_position(resume_payload, config)
        resume_step_in_epoch = resume_next_batch - 1
        previous_config = resume_payload.get('config', {})
        previous_sha = previous_config.get('training_sha')
        # The only accepted predecessor is the immediately previous clean training SHA;
        # it contains the same model/objective and predates the continuation plumbing.
        compatible_previous = {
            'ff5ad1d4b918d56c6bfa48a2870dc5223e757237',
            '3b67bcec222691d06f0443f8e8129b197b800e82',
            config['training_sha'],
        }
        if previous_sha and previous_sha not in compatible_previous:
            raise RuntimeError('resume training SHA is not compatible with current code')
        config['resumed_from'] = os.path.abspath(args.resume)
        config['resume_completed_steps'] = completed
        config['resume_epoch'] = resume_epoch
        config['resume_step_in_epoch'] = resume_step_in_epoch
        config['resume_next_batch'] = resume_next_batch
        config['parent_training_sha'] = previous_sha
        config['parent_checkpoint_cursor'] = {k: resume_payload[k] for k in ('epoch', 'step_in_epoch')}
        config['formal_optimizer_updates'] = completed if args.run_type == 'formal' else 0
        config['debug_optimizer_updates'] = completed if args.run_type == 'debug' else 0
    log_path = os.path.join(args.output_dir, "salu_log.jsonl")
    if resume_payload is not None:
        with open(log_path, encoding='utf-8') as handle:
            previous_rows = [json.loads(line)['completed_steps'] for line in handle if line.strip()]
        if previous_rows != list(range(1, completed + 1)):
            raise RuntimeError('resume log is not exactly the inherited completed-update history')
    if rank == 0:
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as h:
            json.dump(config, h, indent=2, sort_keys=True)
        if resume_payload is None:
            _save_checkpoint(inner, (clip_opt, mask_opt, suffix_opt), config, args.output_dir, 0, 0, -1)
    if world > 1:
        dist.barrier()
    stream_digest = hashlib.sha256()
    consumed_samples = 0
    resume_completed = completed
    last_update_epoch, last_update_batch = divmod(completed - 1, len(loader)) if completed else (0, -1)
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step_in_epoch, batch in enumerate(loader):
            if completed >= args.max_steps:
                break
            step_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stream_record = {'sample_id': batch['sample_id'].tolist(),
                'image_id': batch['image_id'].tolist(), 'prefix_k': batch['prefix_k'].tolist(),
                'prefix': batch['caption_said'], 'suffix': batch['suffix_text']}
            stream_digest.update(json.dumps(stream_record, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            consumed_samples += len(batch['sample_id'])
            replay_index = epoch * len(loader) + step_in_epoch
            if replay_index < resume_completed:
                if replay_index + 1 == resume_completed:
                    expected_stream = resume_payload['config']['stream_summary'][rank]
                    replay_ok = (consumed_samples == expected_stream['consumed_samples'] and
                                 stream_digest.hexdigest() == expected_stream['stream_sha256'])
                    if not _all_ranks_finite(torch.tensor(0.0 if replay_ok else float('nan'), device=device), world):
                        raise RuntimeError('resume data replay differs from checkpoint sample/prefix/suffix stream')
                    print(f'RESUME_REPLAY_VERIFIED rank={rank} completed={completed} next_epoch={resume_epoch} '
                          f'next_batch={resume_next_batch} consumed_samples={consumed_samples}', flush=True)
                continue
            clip_schedule(completed); mask_schedule(completed)
            if suffix_schedule is not None:
                suffix_schedule(completed)
            image_a = batch["image_a"].to(device, non_blocking=True)
            prefix = longclip.tokenize(batch["caption_said"], truncate=True).to(device)
            suffix = longclip.tokenize(batch["suffix_text"], truncate=True).to(device)
            valid = batch["suffix_valid"].to(device)
            ids = batch["image_id"].to(device)
            clip_opt.zero_grad(set_to_none=True); mask_opt.zero_grad(set_to_none=True)
            if suffix_opt is not None:
                suffix_opt.zero_grad(set_to_none=True)
            amp_enabled = args.amp_dtype == "bf16" and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                out = module(image_a, prefix, suffix, valid, ids)
            if not _all_ranks_finite(out["loss_total"], world):
                raise FloatingPointError("non-finite loss on at least one rank")
            out["loss_total"].backward()
            _check_gradients_finite(inner, world, device)
            gradient_health = _gradient_health(inner) if completed < 3 or (completed + 1) % 100 == 0 else None
            clip_opt.step(); mask_opt.step()
            if suffix_opt is not None:
                suffix_opt.step()
            completed += 1
            last_update_epoch, last_update_batch = epoch, step_in_epoch
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            health = {'rank': rank, 'valid': int(valid.sum().item()),
                'gradient_norms_before_update': gradient_health,
                'step_seconds': time.perf_counter() - step_started,
                'peak_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else None,
                'peak_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == 'cuda' else None,
                'consumed_samples': consumed_samples, 'stream_sha256': stream_digest.hexdigest()}
            rank_health = [health]
            if world > 1:
                rank_health = [None for _ in range(world)]
                dist.all_gather_object(rank_health, health)
            config['formal_optimizer_updates'] = completed if args.run_type == 'formal' else 0
            config['debug_optimizer_updates'] = completed if args.run_type == 'debug' else 0
            config['stream_summary'] = [{k: h[k] for k in ('rank', 'consumed_samples', 'stream_sha256')} for h in rank_health]
            if rank == 0:
                record = {"completed_steps": completed, "epoch": epoch, "step_in_epoch": step_in_epoch,
                          "suffix_mode": args.suffix_mode, "lr": clip_opt.param_groups[0]["lr"],
                          "mask_lr": mask_opt.param_groups[0]["lr"],
                          "suffix_lr": suffix_opt.param_groups[0]["lr"] if suffix_opt else None,
                          "run_type": args.run_type,
                          "formal_optimizer_updates": completed if args.run_type == "formal" else 0,
                          "debug_optimizer_updates": completed if args.run_type == "debug" else 0}
                record["per_rank_valid"] = [h['valid'] for h in rank_health]
                record['rank_health'] = rank_health
                record['synchronized_step_seconds'] = max(h['step_seconds'] for h in rank_health)
                record["step_seconds"] = time.perf_counter() - step_started
                if device.type == "cuda":
                    record["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    record["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                for key, value in out.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        record[key] = float(value.detach().cpu())
                    elif value is None and key in ("m_u_keep_ratio", "m_u_probability_mean"):
                        record[key] = None
                with open(log_path, "a", encoding="utf-8") as h:
                    h.write(json.dumps(record, sort_keys=True) + "\n")
                if args.save_every and completed % args.save_every == 0:
                    _save_checkpoint(inner, (clip_opt, mask_opt, suffix_opt), config, args.output_dir,
                                     completed, epoch, step_in_epoch)
        if completed >= args.max_steps:
            break
    if rank == 0:
        config['training_seconds'] = time.perf_counter() - training_started
        _save_checkpoint(inner, (clip_opt, mask_opt, suffix_opt), config, args.output_dir,
                         completed, last_update_epoch, last_update_batch)
        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
    return 0


def _save_checkpoint(module: DualMaskSuffixTrainModule, optimizers, config: Dict,
                     output_dir: str, completed: int, epoch: int, step_in_epoch: int) -> str:
    path = os.path.join(output_dir, "s0_dual_mask_suffix_%s_step%06d.pt" %
                        (module.suffix_mode, completed))
    payload = {
        "clip_state": module.clip.state_dict(),
        "suffix_gate_state": None if module.suffix_gate is None else module.suffix_gate.state_dict(),
        "optimizer_states": {"clip": optimizers[0].state_dict(), "mask": optimizers[1].state_dict(),
                              "suffix": None if optimizers[2] is None else optimizers[2].state_dict()},
        "completed_steps": int(completed), "epoch": int(epoch), "step_in_epoch": int(step_in_epoch),
        "config": dict(config), "provenance": {"git_head": _git_head(), "state_digest": state_digest(module.clip.state_dict())},
    }
    _atomic_torch_save(payload, path)
    return path


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
