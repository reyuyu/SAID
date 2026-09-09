"""Phase 2 STEP 20/21: Said attention debug visualization + caption sensitivity.

Diagnostic only: these heatmaps are attention maps of the caption-conditioned
Said router. They are NOT explanation ground truth and carry no loss.

Outputs (never committed; ``outputs/`` is git-ignored):
    outputs/salu_debug/sample{i}_image.png
    outputs/salu_debug/sample{i}_attn14.png / _attn14.npy
    outputs/salu_debug/sample{i}_overlay.png
    outputs/salu_debug/sample{i}_captionA_overlay.png
    outputs/salu_debug/sample{i}_captionB_overlay.png
    outputs/salu_debug/caption_sensitivity.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_val_dataset  # noqa: E402

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def denormalize(tensor):
    """[3, 224, 224] normalized tensor -> uint8 HWC image."""
    image = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * CLIP_STD + CLIP_MEAN
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize(norm2d):
    """Simple blue -> green -> red heatmap for a [0, 1] array."""
    h = np.clip(norm2d, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def upsample(norm2d, size):
    image = Image.fromarray((np.clip(norm2d, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def js_divergence(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def attention_for(model, image_tensor, caption, device):
    tokens = longclip.tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            z_global, patch_features = model.clip.encode_image_with_patches(image_tensor.unsqueeze(0).to(device))
            text_feature = model.clip.encode_text(tokens)
            text_feature = torch.nn.functional.normalize(text_feature, dim=-1)
            A_s, z_s = model.said_router(text_feature, patch_features)
    return A_s[0].detach().float().cpu().numpy(), z_s[0].detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Said attention debug visualization')
    parser.add_argument('--checkpoint', default=None, help='SALU checkpoint (optional)')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output_dir', default='outputs/salu_debug')
    parser.add_argument('--tau_said', type=float, default=0.07)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip_model, tau_said=args.tau_said).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print('loaded SALU checkpoint %s (step %s)' % (args.checkpoint, ckpt.get('step')), flush=True)
    model.eval()

    dataset = share4v_val_dataset()
    grid = int(model.clip.visual.input_resolution // model.clip.visual.conv1.weight.shape[-1])
    summary = {'grid': grid, 'num_samples': args.num_samples, 'samples': [], 'caption_sensitivity': []}

    for i in range(args.num_samples):
        image_tensor, caption_a = dataset[i]
        _, caption_b = dataset[(i + 7) % len(dataset)]  # deliberately different caption

        attn_a, z_s_a = attention_for(model, image_tensor, caption_a, device)
        attn_b, z_s_b = attention_for(model, image_tensor, caption_b, device)

        original = denormalize(image_tensor)
        Image.fromarray(original).save(os.path.join(args.output_dir, 'sample%d_image.png' % i))

        for tag, attn in (('captionA', attn_a), ('captionB', attn_b)):
            grid2d = attn.reshape(grid, grid)
            np.save(os.path.join(args.output_dir, 'sample%d_%s_attn14.npy' % (i, tag)), grid2d)
            norm2d = (grid2d - grid2d.min()) / max(1e-8, (grid2d.max() - grid2d.min()))
            Image.fromarray((norm2d * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST).save(
                os.path.join(args.output_dir, 'sample%d_%s_attn14.png' % (i, tag)))
            heat = colorize(upsample(norm2d, 224))
            overlay = (0.5 * original.astype(np.float32) + 0.5 * heat.astype(np.float32)).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(args.output_dir, 'sample%d_%s_overlay.png' % (i, tag)))

        mean_abs_diff = float(np.abs(attn_a - attn_b).mean())
        jsd = js_divergence(attn_a, attn_b)
        cos_said = float(np.dot(z_s_a, z_s_b))
        entry = {
            'sample': i,
            'caption_A': caption_a[:160],
            'caption_B': caption_b[:160],
            'attention_mean_abs_diff': mean_abs_diff,
            'attention_js_divergence': jsd,
            'said_feature_cosine': cos_said,
            'attention_A_max': float(attn_a.max()),
            'attention_A_entropy': float(-(np.clip(attn_a, 1e-12, None) * np.log(np.clip(attn_a, 1e-12, None))).sum()),
        }
        summary['samples'].append(entry)
        summary['caption_sensitivity'].append({
            'sample': i, 'mean_abs_diff': mean_abs_diff, 'js_divergence': jsd, 'said_cosine': cos_said})
        print('SAMPLE %d mean_abs_diff=%.6f js_divergence=%.6f said_cosine=%.4f' % (
            i, mean_abs_diff, jsd, cos_said), flush=True)

    with open(os.path.join(args.output_dir, 'caption_sensitivity.json'), 'w') as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    print('VISUALIZE_RESULT PASS output_dir=' + args.output_dir, flush=True)


if __name__ == '__main__':
    main()
