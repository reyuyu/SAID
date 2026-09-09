"""Case sheets for Said-grounding review: original image + Said heatmaps + the exact captions.

For each fixed diagnostic image this writes one sheet containing

    [ original image ] [ own caption ] [ shuffled caption ] [ short caption ] [ long caption ]

where every column is the Said attention heatmap computed from the caption printed
directly underneath it, so image / heatmap / description can be checked by eye.

It also re-computes the attention from scratch (q_proj/k_proj -> normalize ->
softmax) and compares it with the stored ``*_attn14.npy`` from the diagnostics run,
and reports the top-attended patches and the attention mass concentration.

Figures go to ``outputs/`` and are never committed.
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.salu.visualize_said import colorize, denormalize, js_divergence as _js, upsample  # noqa: E402
from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_val_dataset  # noqa: E402

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
]
VARIANTS = ['own', 'shuffled', 'short', 'long']
VAR_LABEL = {
    'own': 'own caption (C_i)',
    'shuffled': 'shuffled caption (C_j)',
    'short': 'short caption (1st sentence of C_i)',
    'long': 'long caption (full C_i)',
}


def get_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def first_sentence(caption):
    parts = caption.split('. ')
    return parts[0] if parts else caption


def encode_caption(model, caption, device):
    tokens = longclip.tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        text_feature = model.clip.encode_text(tokens)
    return F.normalize(text_feature.float(), dim=-1)


def attention_from_scratch(model, patch_features, text_feature):
    """Independent recomputation of A_s = softmax(cos(q_proj(t), k_proj(H)) / tau)."""
    with torch.no_grad():
        q = F.normalize(model.said_router.q_proj(text_feature), dim=-1)
        k = F.normalize(model.said_router.k_proj(patch_features.unsqueeze(0)), dim=-1)
        logits = torch.einsum('bd,bnd->bn', q, k) / model.said_router.tau_said
        A_s = F.softmax(logits, dim=-1)[0]
        z_s = F.normalize(torch.einsum('bn,bnd->bd', A_s.unsqueeze(0), patch_features.unsqueeze(0)), dim=-1)[0]
    return A_s.detach().float().cpu().numpy(), z_s.detach().float().cpu().numpy()


def heat_tile(attn2d, scale_max, tile):
    norm = np.clip(attn2d / max(scale_max, 1e-12), 0.0, 1.0)
    return Image.fromarray(colorize(upsample(norm, tile)))


def main():
    parser = argparse.ArgumentParser(description='Said case sheets (image + heatmap + caption)')
    parser.add_argument('--checkpoint', default='runs_salu/phase21/salu_said_only_last.pt')
    parser.add_argument('--tag', default='final')
    parser.add_argument('--samples', default='0,1,2,3')
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--output_dir', default='outputs/salu_grounding/case_sheets')
    parser.add_argument('--tile', type=int, default=224)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip_model).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print('checkpoint %s (step %s)' % (args.checkpoint, ckpt.get('step')), flush=True)

    dataset = share4v_val_dataset()
    diag_set = json.load(open(os.path.join(args.diagnostics_dir, 'diagnostic_set.json')))
    perm = diag_set['permutation']

    samples = [int(s) for s in args.samples.split(',') if s.strip()]
    report = {'checkpoint': args.checkpoint, 'tag': args.tag, 'samples': []}

    for sample in samples:
        image_tensor, caption_own = dataset[sample]
        _, caption_shuf = dataset[perm[sample]]
        captions = {
            'own': caption_own,
            'shuffled': caption_shuf,
            'short': first_sentence(caption_own),
            'long': caption_own,
        }
        # same precision path as the diagnostics/training: bf16 image encode, fp32 router
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                _, patch_features_bf16 = model.clip.encode_image_with_patches(image_tensor.unsqueeze(0).to(device))
            _, patch_features_fp32 = model.clip.encode_image_with_patches(image_tensor.unsqueeze(0).to(device))
        patch_features = patch_features_bf16[0].float()
        patch_features_fp32 = patch_features_fp32[0].float()

        maps, zs = {}, {}
        for variant in VARIANTS:
            text_feature = encode_caption(model, captions[variant], device)
            maps[variant], zs[variant] = attention_from_scratch(model, patch_features, text_feature)

        # verification: sum-to-one, and equality with the stored diagnostics map
        checks = {}
        for variant in VARIANTS:
            stored_path = os.path.join(args.diagnostics_dir, 'heatmaps', args.tag,
                                       'sample%02d_%s_attn14.npy' % (sample, variant))
            entry = {
                'sum': float(maps[variant].sum()),
                'finite': bool(np.isfinite(maps[variant]).all()),
            }
            if os.path.exists(stored_path):
                stored = np.load(stored_path).reshape(-1)
                entry['max_abs_diff_vs_stored'] = float(np.abs(stored - maps[variant]).max())
            checks[variant] = entry

        # precision-noise floor: fp32 image encode vs bf16 image encode (same caption)
        text_own = encode_caption(model, captions['own'], device)
        map_own_fp32, _ = attention_from_scratch(model, patch_features_fp32, text_own)
        precision_noise = {
            'mean_abs_diff': float(np.abs(maps['own'] - map_own_fp32).mean()),
            'max_abs_diff': float(np.abs(maps['own'] - map_own_fp32).max()),
            'js_divergence': float(_js(maps['own'], map_own_fp32)),
        }

        own = maps['own']
        order = np.argsort(-own)
        top5 = [(int(idx // 14), int(idx % 14), float(own[idx])) for idx in order[:5]]
        topk_mass = {str(k): float(np.sort(own)[::-1][:k].sum()) for k in (1, 5, 10, 20)}
        entropy = float(-(np.clip(own, 1e-12, None) * np.log(np.clip(own, 1e-12, None))).sum())

        # ---- figure ----
        tile = args.tile
        col_w = 268
        header_h = 46
        label_h = 26
        caption_h = 210
        width = col_w * 5 + 16
        height = header_h + tile + label_h + caption_h
        canvas = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font_t = get_font(19)
        font_l = get_font(13)
        font_c = get_font(12)

        draw.text((8, 10), 'case sheet  sample %02d  image_id=%s  (%s, step %s)' % (
            sample, dataset.json_data[sample]['image'], args.tag, ckpt.get('step')),
            fill=(0, 0, 0), font=font_t)

        original = denormalize(image_tensor)
        scale_max = max(m.max() for m in maps.values())
        x = 8
        canvas.paste(Image.fromarray(original), (x, header_h))
        draw.text((x, header_h + tile + 4), 'original image', fill=(0, 0, 0), font=font_l)
        for k, line in enumerate(textwrap.wrap(dataset.json_data[sample]['image'], width=40)[:4]):
            draw.text((x, header_h + tile + label_h + k * 15), line, fill=(90, 90, 90), font=font_c)
        x += col_w

        for variant in VARIANTS:
            canvas.paste(heat_tile(maps[variant], scale_max, tile), (x, header_h))
            draw.rectangle([x, header_h, x + tile - 1, header_h + tile - 1], outline=(190, 190, 190))
            draw.text((x, header_h + tile + 4), VAR_LABEL[variant], fill=(0, 0, 0), font=font_l)
            wrapped = textwrap.wrap(captions[variant], width=42)[:12]
            for k, line in enumerate(wrapped):
                draw.text((x, header_h + tile + label_h + k * 15), line, fill=(60, 60, 60), font=font_c)
            x += col_w

        out_path = os.path.join(args.output_dir, 'case_sheet_sample%02d.png' % sample)
        canvas.save(out_path)

        entry = {
            'sample': sample,
            'image_id': dataset.json_data[sample]['image'],
            'shuffled_source_index': perm[sample],
            'shuffled_source_image_id': dataset.json_data[perm[sample]]['image'],
            'captions': captions,
            'verification': checks,
            'precision_noise_floor': precision_noise,
            'own_attention': {
                'entropy': entropy,
                'effective_patch_count': float(np.exp(entropy)),
                'max': float(own.max()),
                'top5_patches_row_col_weight': top5,
                'topk_mass': topk_mass,
            },
            'figure': out_path,
        }
        report['samples'].append(entry)

        print('SAMPLE %02d image=%s' % (sample, entry['image_id']), flush=True)
        print('   own      : %s' % captions['own'][:150], flush=True)
        print('   shuffled : %s' % captions['shuffled'][:150], flush=True)
        print('   short    : %s' % captions['short'][:150], flush=True)
        print('   attention: entropy=%.3f eff=%.1f max=%.4f top1=(%d,%d) top5_mass=%.3f top10_mass=%.3f' % (
            entropy, float(np.exp(entropy)), own.max(), top5[0][0], top5[0][1],
            topk_mass['5'], topk_mass['10']), flush=True)
        for variant in VARIANTS:
            c = checks[variant]
            extra = '' if 'max_abs_diff_vs_stored' not in c else ' stored_diff=%.2e' % c['max_abs_diff_vs_stored']
            print('   check %-8s sum=%.6f finite=%s%s' % (variant, c['sum'], c['finite'], extra), flush=True)
        print('   precision noise (fp32 vs bf16 image encode): mean_abs=%.6f max=%.6f jsd=%.6f' % (
            precision_noise['mean_abs_diff'], precision_noise['max_abs_diff'], precision_noise['js_divergence']), flush=True)
        print('   figure   : %s' % out_path, flush=True)

    with open(os.path.join(args.output_dir, 'case_sheet_report.json'), 'w') as fp:
        json.dump(report, fp, indent=2, sort_keys=True)
    print('CASE_SHEETS PASS')


if __name__ == '__main__':
    main()
