"""Phase 2 STEP 18: single-GPU forward/backward smoke with real CLIP + real data.

Uses the real OpenAI CLIP ViT-B/16 initialisation and a real ShareGPT4V batch
(default batch size 8), then checks that gradients reach the vision backbone, the
text encoder and both Said projections, while SmartCLIP's ``mask_net`` stays
gradient-free.

Precision: SALU training keeps fp32 master weights and uses bf16 autocast by
default. fp16 autocast combined with GradScaler's default 65536 scaling
overflows inside the fp16 autocast graph (verified on this server), so fp16 is
only available via ``--amp_dtype fp16`` with an explicit, smaller init scale.
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_DIR)
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_train_dataset  # noqa: E402


def grad_abs_sum(param):
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().abs().sum())


def main():
    parser = argparse.ArgumentParser(description='SALU single-GPU forward/backward smoke')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--tau_said', type=float, default=0.07)
    parser.add_argument('--lambda_global', type=float, default=1.0)
    parser.add_argument('--lambda_said', type=float, default=1.0)
    parser.add_argument('--backbone_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--scaler_init_scale', type=float, default=1024.0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    device = torch.device(args.device)
    torch.manual_seed(0)

    clip_model, _ = longclip.load_from_clip(args.base_model, device='cpu')
    salu = SALUModel(clip_model, tau_said=args.tau_said).to(device)
    salu.train()
    print('weight_dtype', salu.clip.visual.conv1.weight.dtype, 'amp_dtype', args.amp_dtype, flush=True)

    dataset = share4v_train_dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    images, texts = next(iter(loader))
    images = images.to(device)
    text_tokens = longclip.tokenize(texts, truncate=True).to(device)
    print('batch images', tuple(images.shape), 'text tokens', tuple(text_tokens.shape), flush=True)

    optimizer = torch.optim.AdamW(
        [
            {'params': salu.backbone_parameters(), 'lr': args.backbone_lr},
            {'params': salu.said_head_parameters(), 'lr': args.head_lr},
        ]
    )
    use_amp = args.amp_dtype != 'fp32'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'fp16'),
                                  init_scale=args.scaler_init_scale)

    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        out = salu(images, text_tokens, args.lambda_global, args.lambda_said)
        loss = out['loss_total']

    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)  # inspect true (unscaled) gradients

    losses = {k: float(out[k].detach()) for k in (
        'loss_global', 'loss_said', 'loss_total',
        'said_attention_entropy', 'said_effective_patch_count',
        'said_attention_max', 'said_attention_min', 'said_feature_norm')}

    checks = {
        'vision_backbone': grad_abs_sum(salu.clip.visual.conv1.weight),
        'vision_proj': grad_abs_sum(salu.clip.visual.proj),
        'text_encoder': grad_abs_sum(salu.clip.transformer.resblocks[0].attn.in_proj_weight),
        'text_projection': grad_abs_sum(salu.clip.text_projection),
        'logit_scale': grad_abs_sum(salu.clip.logit_scale),
        'said_q_proj': grad_abs_sum(salu.said_router.q_proj.weight),
        'said_k_proj': grad_abs_sum(salu.said_router.k_proj.weight),
        'mask_net_first': grad_abs_sum(next(salu.clip.mask_net.parameters())) if hasattr(salu.clip, 'mask_net') else 'n/a',
    }

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    ok = True
    for name in ('vision_backbone', 'text_encoder', 'said_q_proj', 'said_k_proj'):
        value = checks[name]
        if value is None or not (0.0 < value < float('inf')):
            ok = False
    if checks['mask_net_first'] not in (None, 'n/a'):
        ok = False  # mask_net must stay gradient-free

    result = {'losses': losses, 'grad_abs_sum': checks, 'pass': ok}
    print('SINGLE_GPU_SMOKE ' + json.dumps(result, sort_keys=True), flush=True)
    print('SINGLE_GPU_SMOKE_RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
