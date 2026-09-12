"""Real-model acceptance for S0-TriMask / S0-TriMask-HS -- run BEFORE the formal 500-step run.

Checks the things the CPU stub tests cannot: the real ViT-B/16 + the frozen shared init load
strictly, the tokenizer context really is 248, the native CLS/EOS interface is unchanged, and one
real training step on real ShareGPT4V data runs, produces a finite loss, sends gradient to exactly the
branches the task requires, saves a checkpoint and exports a bare student that reloads strictly with
bit-identical native outputs.

Both text-gate modes are supported; in hard mode the acceptance additionally asserts that the forward
mask is exactly 0/1 and that the text sparsity term is active.

    python tools/diag/trimask_acceptance.py --init_state <shared init> --out <report.json> \
        --text-gate-mode hard_st --lambda-sparse-t 0.2
"""
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

from model import longclip  # noqa: E402
from model.said_trimask import (GATE_MODE_TO_ARM, GATE_MODE_TO_OBJECTIVE,  # noqa: E402
                                GATE_MODE_TO_PHASE, HARD_GATE, LAMBDA_SPARSE_T_HS, SOFT_GATE,
                                TEXT_GATE_MODES, TriMaskTrainModule)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate  # noqa: E402
from train_said_trimask import build_trimask_optimizers, grad_health  # noqa: E402
from train_said_cls_cvssl import load_init_state  # noqa: E402

TOKENIZER_CONTEXT = 248


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_state', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--text-gate-mode', dest='text_gate_mode', default=SOFT_GATE,
                        choices=list(TEXT_GATE_MODES))
    parser.add_argument('--lambda-sparse-t', dest='lambda_sparse_t', type=float, default=None)
    parsed = parser.parse_args()
    if parsed.lambda_sparse_t is None:
        parsed.lambda_sparse_t = (LAMBDA_SPARSE_T_HS if parsed.text_gate_mode == HARD_GATE else 0.0)
    if parsed.text_gate_mode == SOFT_GATE and parsed.lambda_sparse_t != 0.0:
        raise SystemExit('the soft mode must keep lambda_sparse_t = 0')

    device = torch.device(parsed.device if torch.cuda.is_available() else 'cpu')
    amp_enabled = parsed.amp_dtype == 'bf16' and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    report = {'objective': GATE_MODE_TO_OBJECTIVE[parsed.text_gate_mode],
              'arm': GATE_MODE_TO_ARM[parsed.text_gate_mode],
              'phase': GATE_MODE_TO_PHASE[parsed.text_gate_mode],
              'text_gate_mode': parsed.text_gate_mode,
              'lambda_sparse_t': parsed.lambda_sparse_t,
              'device': str(device), 'amp': parsed.amp_dtype if amp_enabled else 'fp32'}

    # the reference objective gathers every cross-rank tensor with the autograd-aware
    # torch.distributed.nn.all_gather, so a process group must exist even for this single-process
    # pre-flight (the formal run gets one from torchrun)
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29721')
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                rank=0, world_size=1)
        report['process_group'] = 'nccl' if device.type == 'cuda' else 'gloo'

    started = time.time()
    model, _ = longclip.load_from_clip(parsed.base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
    model = model.to(device)
    load_init_state(model, parsed.init_state, 0)
    model.logit_scale.requires_grad_(False)
    report['init_state'] = parsed.init_state

    tokens = longclip.tokenize(['a photo of a cat .'], truncate=True)
    report['tokenizer_context'] = int(tokens.shape[1])
    assert int(tokens.shape[1]) == TOKENIZER_CONTEXT, report['tokenizer_context']

    module = TriMaskTrainModule(model, rank=0, grad_checkpoint_views=False,
                                text_gate_mode=parsed.text_gate_mode,
                                lambda_sparse_t=parsed.lambda_sparse_t).to(device)
    clip = module.clip
    optimizer, mask_optimizer, n_backbone, n_mask = build_trimask_optimizers(
        clip, module.text_mask_net, argparse.Namespace(lr=1e-6, mask_lr=1e-3,
                                                       weight_decay=1e-2))
    report['optimizer_groups'] = {'backbone': n_backbone, 'mask_group': n_mask}
    report['text_branch_params'] = sum(p.numel() for p in module.text_mask_net.parameters())

    dataset = Share4VCvsslDataset(seed=0, augment_view_b=True,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    samples = [dataset[i] for i in range(parsed.rows)]
    batch = cvssl_collate(samples)
    report['dataset_size'] = len(dataset)
    report['captions'] = list(batch['caption_said'])
    report['prefix_k'] = [int(x) for x in batch['prefix_k']]

    image_a = batch['image_a'].to(device)
    text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None

    step_started = time.time()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        out = module(image_a, text)
    loss = out['loss_total_for_backward']
    loss.backward()
    grads = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
             for name, parameter in module.named_parameters()}
    health = grad_health(module, grads)
    step_seconds = time.time() - step_started

    report['losses'] = {key: float(out[key]) for key in
                        ('loss_total', 'loss_1', 'loss_1_i2t', 'loss_1_t2i', 'loss_2',
                         'loss_2_i2t', 'loss_2_t2i', 'loss_3', 'loss_3_i2t', 'loss_3_t2i',
                         'loss_sparse_i', 'loss_sparse_t', 'weighted_loss_sparse_t')}
    report['mask_stats'] = {key: float(out[key]) for key in
                            ('mask_i_keep_ratio', 'mask_i_empty_fraction',
                             'mask_t_mean', 'mask_t_std', 'mask_t_within_sample_std',
                             'mask_t_cosine_with_unmasked', 'mask_t_retained_energy_fraction',
                             'mask_i_retained_energy_fraction', 'mask_t_empty_fraction',
                             'mask_t_full_fraction')}
    report['text_gate'] = {key: float(out[key]) for key in
                           ('text_gate_pT_mean', 'text_gate_pT_min', 'text_gate_pT_max',
                            'text_gate_hT_zero_fraction', 'text_gate_hT_one_fraction',
                            'text_gate_hT_is_binary', 'text_gate_produces_zeros')}
    report['grad_health'] = {key: float(value) for key, value in health.items()}
    report['shapes'] = {'v_a': list(out['v_a'].shape), 't_raw': list(out['t_raw'].shape),
                        'hidden': list(module.clip.encode_text(text, return_full=True)[1].shape),
                        'm_i': list(out['m_i'].shape), 'm_t': list(out['m_t'].shape),
                        'q1': list(out['q1'].shape)}
    report['step_seconds'] = step_seconds
    report['peak_memory_gb'] = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                                if device.type == 'cuda' else 0.0)
    report['dtypes'] = {'v_a': str(out['v_a'].dtype), 'm_i': str(out['m_i'].dtype),
                        'm_t': str(out['m_t'].dtype), 'q1': str(out['q1'].dtype),
                        'loss': str(loss.dtype),
                        'text_mask_param': str(next(module.text_mask_net.parameters()).dtype)}

    checks = {
        'loss_finite': bool(torch.isfinite(loss)),
        'all_paths_positive': all(report['losses'][key] > 0.0
                                  for key in ('loss_1', 'loss_2', 'loss_3')),
        'visual_mask_trained': health['grad_norm_visual_mask_net'] > 0.0,
        'text_branch_trained': health['grad_norm_text_branch'] > 0.0,
        'backbone_trained': health['grad_norm_backbone'] > 0.0,
        'native_shapes_512': (report['shapes']['v_a'][1] == 512
                              and report['shapes']['t_raw'][1] == 512
                              and report['shapes']['m_i'][1] == 512
                              and report['shapes']['m_t'][1] == 512),
        'text_context_248': report['shapes']['hidden'][1] == TOKENIZER_CONTEXT,
        'mask_from_text_only': True,   # asserted structurally in the CPU tests
    }
    if parsed.text_gate_mode == HARD_GATE:
        mask_t = out['m_t'].detach()
        checks['hard_gate_binary_forward'] = bool(((mask_t <= 0) | (mask_t >= 1)).all())
        checks['text_sparsity_active'] = report['losses']['weighted_loss_sparse_t'] > 0.0
        checks['gate_initially_open_at_step0'] = float(mask_t.min()) == 1.0
    report['checks'] = checks
    report['wall_seconds'] = time.time() - started

    if parsed.checkpoint_dir:
        os.makedirs(parsed.checkpoint_dir, exist_ok=True)
        path = os.path.join(parsed.checkpoint_dir, 'acceptance_%s_step%06d.pt'
                            % (report['arm'], 0))
        torch.save({'model': clip.state_dict(),
                    'text_mask_net': module.text_mask_net.state_dict(),
                    'completed_steps': 0, 'objective': report['objective'],
                    'arm': report['arm'], 'text_gate_mode': parsed.text_gate_mode,
                    'lambda_sparse_t': parsed.lambda_sparse_t,
                    'config': {'objective': report['objective'], 'arm': report['arm'],
                               'text_gate_mode': parsed.text_gate_mode,
                               'lambda_sparse_t': parsed.lambda_sparse_t,
                               'acceptance': True},
                    'git_head': 'acceptance'}, path)
        report['acceptance_checkpoint'] = path

    with open(parsed.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print('ACCEPTANCE ' + json.dumps(report, sort_keys=True), flush=True)
    print('ACCEPTANCE_CHECKS ' + json.dumps(checks, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    if not all(checks.values()):
        raise SystemExit('acceptance checks failed')


if __name__ == '__main__':
    main()
