#!/usr/bin/env python3
"""Read-only numerical audit for the Agreement-Mask at step 500.

The audit loads the frozen step-500 checkpoint, reconstructs the native
ShareGPT4V nested_detail_d3 stream, and evaluates positive pairs plus a
deterministic roll-1 negative pairing.  It never creates an optimizer or
writes model/checkpoint state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from model.nested_fusion_mask import pair_logits
from train.nested_semantic_data import NestedDataset, collate, file_sha


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def stats(x):
    x = x.detach().float().reshape(-1).cpu()
    if x.numel() == 0:
        return dict(count=0)
    q = torch.quantile(x, torch.tensor([.01, .05, .50, .95, .99]))
    return dict(count=int(x.numel()), mean=float(x.mean()), std=float(x.std(unbiased=False)),
                min=float(x.min()), max=float(x.max()), p01=float(q[0]), p05=float(q[1]),
                p50=float(q[2]), p95=float(q[3]), p99=float(q[4]))


def ratio_stats(num, den):
    return stats(num / den.clamp_min(1e-8))


def collect_view(name, unscaled, uv, ut, theta, out):
    # unscaled is l; the native model returns 5*l as mask logits.
    product = torch.tanh(uv / (uv.square().mean(-1, keepdim=True).add(1e-6).sqrt())) * \
              torch.tanh(ut / (ut.square().mean(-1, keepdim=True).add(1e-6).sqrt()))
    bias = .2 * torch.tanh(theta).view(1, -1)
    probability = torch.sigmoid(5. * unscaled)
    selected = unscaled >= 0
    out[name] = dict(
        raw_logits=stats(unscaled), scaled_logits=stats(5. * unscaled),
        probability=stats(probability),
        probability_saturation_fraction=float(((probability < .01) | (probability > .99)).float().mean()),
        probability_lt_001=float((probability < .01).float().mean()),
        probability_gt_099=float((probability > .99).float().mean()),
        hard_keep_ratio=float(selected.float().mean()),
        product=stats(product),
        bias=stats(bias.expand_as(unscaled)),
        product_sign_mismatch_fraction=float(((product >= 0) != (unscaled >= 0)).float().mean()),
        bias_rescue_negative_to_positive=float(((product < 0) & (unscaled >= 0)).float().mean()),
        bias_flip_positive_to_negative=float(((product >= 0) & (unscaled < 0)).float().mean()),
        visual_norm=stats(uv.norm(dim=-1)), text_norm=stats(ut.norm(dim=-1)),
        visual_text_norm_ratio=ratio_stats(uv.norm(dim=-1), ut.norm(dim=-1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--index-dir', required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--target-valid', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    root = Path(args.repo_root)
    cfg = json.loads(Path(args.config).read_text())
    ckpt_path = Path(args.checkpoint)
    assert torch.cuda.is_available(), 'GPU required for the native encoder audit'
    device = torch.device('cuda:0')

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert ckpt['completed_steps'] == 500
    assert ckpt['config']['joint_mask_mode'] == 'agreement'
    assert file_sha(ckpt_path) == '4908184d40c2db5886a9169dd57ac7698de8e617b58851cd76b5ad390ccc2b76'

    # Match the trainer's construction path exactly, then load the two frozen
    # state dictionaries from the full training checkpoint.
    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    search = hparams(cfg)
    module = BalancedSearch(
        clip.float(), arm=cfg['arm'], search_hparams=search,
        inclusion_hierarchy=cfg['inclusion_hierarchy'],
        view_sparsity_weights=cfg['view_sparsity_weights'],
        hns_enabled=cfg['hns_enabled'], hns_beta=cfg['hns_beta'],
        fusion=cfg['fusion'], visual=cfg['visual'],
        checkpoint_encoders=False, image_chunk=cfg['image_chunk'],
        text_chunk=cfg['text_chunk'], condition_mode=cfg['condition_mode'],
        shuffle_seed=cfg['shuffle_seed'], checkpoint_pair_blocks=False,
        joint_mask_mode=cfg['joint_mask_mode'])
    module.clip.load_state_dict(ckpt['model'], strict=True)
    module.fusion_branch.load_state_dict(ckpt['adapter'], strict=True)
    module.eval().to(device)
    theta = module.fusion_branch.agreement_theta.detach().to(device)

    dataset = NestedDataset(args.index_dir, args.image_root, 'nested_detail_d3', 0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate, pin_memory=True)
    records = {k: [] for k in ('F', 'Dall', 'D3')}
    negatives = {k: [] for k in ('F', 'Dall', 'D3')}
    ids, valid_ids, ks = [], [], []
    seen_valid = 0

    with torch.inference_mode():
        for batch in loader:
            valid = batch['valid'].bool()
            take = torch.ones_like(valid)
            if seen_valid + int(valid.sum()) > args.target_valid:
                need = args.target_valid - seen_valid
                valid_positions = valid.nonzero(as_tuple=False).flatten()
                take.zero_()
                if need > 0:
                    take[valid_positions[:need]] = True
            if not bool(take.any()):
                break
            images = batch['image'][take].to(device, non_blocking=True)
            tf = batch['tokens_f'][take].to(device, non_blocking=True)
            to = batch['tokens_o'][take].to(device, non_blocking=True)
            te = batch['tokens_e'][take].to(device, non_blocking=True)
            z, visual = module.encode_visual(images)
            for name, tokens in (('F', tf), ('Dall', to), ('D3', te)):
                _, condition = module.encode_view(tokens)
                uv, ut = visual[1], condition[1]
                positive = pair_logits(module, visual, condition, paired=True) / 5.
                rolled_visual = tuple(x.roll(1, 0) for x in visual)
                negative = pair_logits(module, rolled_visual, condition, paired=True) / 5.
                mask = torch.ones(positive.shape[0], dtype=torch.bool, device=device)
                if name != 'F':
                    # F is retained for every selected sample; Dall/D3 only for
                    # valid detail pairs, which is the trainer's enabled set.
                    mask = valid[take].to(device)
                records[name].append((positive[mask].cpu(), uv[mask].cpu(), ut[mask].cpu()))
                negatives[name].append((negative[mask].cpu(), uv.roll(1, 0)[mask].cpu(), ut[mask].cpu()))
            ids.extend(batch['sample_id'][take].tolist())
            valid_ids.extend(batch['sample_id'][take][valid[take]].tolist())
            ks.extend(batch['K'][take][valid[take]].tolist())
            seen_valid += int(valid[take].sum())
            if seen_valid >= args.target_valid:
                break

    result = dict(
        audit='agreement-mask-step500-readonly', checkpoint_sha256=file_sha(ckpt_path),
        checkpoint=str(ckpt_path), step=int(ckpt['completed_steps']), model='ViT-B/16',
        sampling=dict(mode='nested_detail_d3', seed=0, target_valid=args.target_valid,
                      selected_records=len(ids), valid_records=len(valid_ids),
                      sample_ids_sha256=digest(ids), valid_sample_ids_sha256=digest(valid_ids),
                      K_histogram={str(k): ks.count(k) for k in sorted(set(ks))},
                      index_metadata=json.loads((Path(args.index_dir)/'metadata.json').read_text()),
                      index_records_sha256=file_sha(Path(args.index_dir)/'records.jsonl')),
        theta=stats(theta), bounded_bias=stats(.2 * torch.tanh(theta)),
        theta_boundary_fraction=float((theta.abs() > 4.).float().mean()),
        bias_boundary_fraction=float((torch.tanh(theta).abs() > .999).float().mean()),
        positive={}, negative={}, sign_consistency={},
        implementation=dict(rule='l=tanh(RMSNorm(u_v))*tanh(RMSNorm(u_t))+0.2*tanh(theta)',
                            native_pair_logits='5*l', probability='sigmoid(5*l)',
                            hard_forward='l>=0', negative_pair_rule='visual.roll(1,0)',
                            positive_and_negative_same_function=True, rms_eps=1e-6,
                            bias_limit=0.2, theta_init='checkpoint_step500'),
    )
    for polarity, source in (('positive', records), ('negative', negatives)):
        view_tensors = {}
        for name in ('F', 'Dall', 'D3'):
            ls, vs, ts = [torch.cat(x, 0) for x in zip(*source[name])]
            collect_view(name, ls, vs, ts, theta.cpu(), result[polarity])
            view_tensors[name] = ls
        signs = torch.stack([(view_tensors[n] >= 0) for n in ('F', 'Dall', 'D3')])
        result['sign_consistency'][polarity] = dict(
            all_three_equal=float(signs.all(0).eq(signs.any(0)).float().mean()),
            pair_F_Dall=float((signs[0] == signs[1]).float().mean()),
            pair_F_D3=float((signs[0] == signs[2]).float().mean()),
            pair_Dall_D3=float((signs[1] == signs[2]).float().mean()),
            total_elements=int(signs.shape[1] * signs.shape[2]),
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({'output': str(out), 'valid_records': seen_valid,
                      'selected_records': len(ids)}, indent=2))


if __name__ == '__main__':
    main()
