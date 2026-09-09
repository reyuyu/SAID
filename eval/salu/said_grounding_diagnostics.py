"""Phase 2.1 Said grounding validation diagnostics (diagnostic only -- no training).

Answers whether the Said router learned *caption-conditioned* grounding rather
than generic image saliency. Uses a fixed diagnostic set (default 64 images from
the ShareGPT4V validation split) so every checkpoint is compared on exactly the
same images and the same caption permutation.

Diagnostics
-----------
1. Caption shuffle: A_s(I_i, C_i) vs A_s(I_i, C_j) with a fixed permutation j.
2. Positive alignment margin: cos(z_s(I_i,C_i), t_i) - cos(z_s(I_i,C_i), t_j).
3. Short vs long caption: A_s with the first sentence vs the full caption.

The full caption is used for *evaluation only*; it is never used as training
supervision. Heatmaps are written to ``outputs/`` and are never committed.

Usage::

    python eval/salu/said_grounding_diagnostics.py --tag initial
    python eval/salu/said_grounding_diagnostics.py \
        --checkpoint runs_salu/phase21/salu_said_only_last.pt --tag final
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.salu.visualize_said import (  # noqa: E402
    colorize,
    denormalize,
    js_divergence,
    upsample,
)
from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402
from sharegpt4v import share4v_val_dataset  # noqa: E402


def first_sentence(caption):
    parts = caption.split('. ')
    return parts[0] if parts else caption


def encode_caption(model, caption, device):
    tokens = longclip.tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        text_feature = model.clip.encode_text(tokens)
    return F.normalize(text_feature.float(), dim=-1)[0]


def said_for(model, patch_features, text_feature):
    with torch.no_grad():
        A_s, z_s = model.said_router(text_feature.unsqueeze(0), patch_features.unsqueeze(0))
    return A_s[0].detach().float().cpu().numpy(), z_s[0].detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Said grounding validation diagnostics')
    parser.add_argument('--checkpoint', default=None, help='SALU checkpoint (default: untrained init)')
    parser.add_argument('--tag', default='initial', help='label for this checkpoint, e.g. initial/step200/final')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--heatmap_count', type=int, default=8)
    parser.add_argument('--permutation_seed', type=int, default=0)
    parser.add_argument('--output_dir', default='outputs/salu_grounding')
    parser.add_argument('--tau_said', type=float, default=0.07)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.output_dir, 'heatmaps', args.tag)
    os.makedirs(heatmap_dir, exist_ok=True)

    clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip_model, tau_said=args.tau_said).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print('loaded %s (step %s)' % (args.checkpoint, ckpt.get('step')), flush=True)
    model.eval()

    dataset = share4v_val_dataset()
    indices = list(range(args.num_samples))
    image_ids = [dataset.json_data[i]['image'] for i in indices]

    # fixed permutation: image i is paired with the caption of perm[i] (never itself)
    perm = list(indices)
    random.Random(args.permutation_seed).shuffle(perm)
    for i in indices:
        if perm[i] == i:
            j = (i + 1) % len(indices)
            perm[i], perm[j] = perm[j], perm[i]

    diagnostic_set = {
        'num_samples': args.num_samples,
        'permutation_seed': args.permutation_seed,
        'indices': indices,
        'image_ids': image_ids,
        'permutation': perm,
        'source': 'share4v_val_dataset (first %d validation entries)' % args.num_samples,
    }
    with open(os.path.join(args.output_dir, 'diagnostic_set.json'), 'w') as fp:
        json.dump(diagnostic_set, fp, indent=2, sort_keys=True)

    shuffle_abs, shuffle_js, shuffle_cos = [], [], []
    margin_pos, margin_neg = [], []
    sl_abs, sl_js, sl_cos = [], [], []
    entropy_list, eff_list, max_list, min_list = [], [], [], []
    per_sample = []

    for i in indices:
        image_tensor, caption_i = dataset[i]
        _, caption_j = dataset[perm[i]]
        caption_short = first_sentence(caption_i)
        caption_long = caption_i

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
                _, patch_features = model.clip.encode_image_with_patches(image_tensor.unsqueeze(0).to(device))
            patch_features = patch_features[0].float()

        t_i = encode_caption(model, caption_i, device)
        t_j = encode_caption(model, caption_j, device)
        t_short = encode_caption(model, caption_short, device)
        t_long = encode_caption(model, caption_long, device)

        A_own, z_own = said_for(model, patch_features, t_i)
        A_shuf, z_shuf = said_for(model, patch_features, t_j)
        A_short, z_short = said_for(model, patch_features, t_short)
        A_long, z_long = said_for(model, patch_features, t_long)

        sim_pos = float(np.dot(z_own, t_i.detach().float().cpu().numpy()))
        sim_neg = float(np.dot(z_own, t_j.detach().float().cpu().numpy()))

        ent = float(SaidRouter.attention_entropy(torch.from_numpy(A_own)).item())
        shuffle_abs.append(float(np.abs(A_own - A_shuf).mean()))
        shuffle_js.append(js_divergence(A_own, A_shuf))
        shuffle_cos.append(float(np.dot(z_own, z_shuf)))
        margin_pos.append(sim_pos)
        margin_neg.append(sim_neg)
        sl_abs.append(float(np.abs(A_short - A_long).mean()))
        sl_js.append(js_divergence(A_short, A_long))
        sl_cos.append(float(np.dot(z_short, z_long)))
        entropy_list.append(ent)
        eff_list.append(float(np.exp(ent)))
        max_list.append(float(A_own.max()))
        min_list.append(float(A_own.min()))

        per_sample.append({
            'index': i,
            'image_id': image_ids[i],
            'shuffled_with': perm[i],
            'attention_entropy': ent,
            'shuffle_mean_abs_diff': float(np.abs(A_own - A_shuf).mean()),
            'shuffle_js': js_divergence(A_own, A_shuf),
            'shuffle_zs_cosine': float(np.dot(z_own, z_shuf)),
            'sim_pos': sim_pos,
            'sim_neg': sim_neg,
            'margin': sim_pos - sim_neg,
            'short_long_mean_abs_diff': float(np.abs(A_short - A_long).mean()),
            'short_long_js': js_divergence(A_short, A_long),
            'short_long_zs_cosine': float(np.dot(z_short, z_long)),
        })

        if i < args.heatmap_count:
            original = denormalize(image_tensor)
            Image.fromarray(original).save(os.path.join(heatmap_dir, 'sample%02d_image.png' % i))
            for tag, attn in (('own', A_own), ('shuffled', A_shuf), ('short', A_short), ('long', A_long)):
                grid = attn.reshape(14, 14)
                norm = (grid - grid.min()) / max(1e-8, grid.max() - grid.min())
                np.save(os.path.join(heatmap_dir, 'sample%02d_%s_attn14.npy' % (i, tag)), grid)
                heat = colorize(upsample(norm, 224))
                overlay = (0.5 * original.astype(np.float32) + 0.5 * heat.astype(np.float32)).astype(np.uint8)
                Image.fromarray(overlay).save(os.path.join(heatmap_dir, 'sample%02d_%s_overlay.png' % (i, tag)))

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    result = {
        'tag': args.tag,
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'indices': indices,
        'image_ids': image_ids,
        'caption_shuffle': {
            'mean_abs_diff': mean(shuffle_abs),
            'mean_js_divergence': mean(shuffle_js),
            'mean_zs_cosine': mean(shuffle_cos),
        },
        'positive_alignment_margin': {
            'mean_sim_pos': mean(margin_pos),
            'mean_sim_neg': mean(margin_neg),
            'mean_margin': mean([p - n for p, n in zip(margin_pos, margin_neg)]),
        },
        'short_vs_long_caption': {
            'mean_abs_diff': mean(sl_abs),
            'mean_js_divergence': mean(sl_js),
            'mean_zs_cosine': mean(sl_cos),
        },
        'said_attention': {
            'mean_entropy': mean(entropy_list),
            'mean_effective_patch_count': mean(eff_list),
            'mean_attention_max': mean(max_list),
            'mean_attention_min': mean(min_list),
        },
        'per_sample': per_sample,
        'note': 'full caption used for diagnostics only, never as training supervision',
    }
    out_path = os.path.join(args.output_dir, 'diagnostics_%s.json' % args.tag)
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
    print('DIAGNOSTICS ' + json.dumps({k: v for k, v in result.items() if k != 'per_sample'}, sort_keys=True), flush=True)
    print('DIAGNOSTICS_RESULT PASS tag=%s output=%s heatmaps=%s' % (args.tag, out_path, heatmap_dir), flush=True)


if __name__ == '__main__':
    main()
