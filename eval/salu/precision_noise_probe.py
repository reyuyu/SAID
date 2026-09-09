"""Quantify the attention precision-noise floor: bf16 vs fp32 image encoding.

The Said logits are scaled by 1/tau_said (=14.3 for tau=0.07), so tiny numerical
differences in the patch features can move the softmax noticeably. This probe
measures, on the fixed diagnostic set with the *same* captions, how much the Said
attention changes purely from encoding the image in bf16 vs fp32. That number is
the noise floor against which caption-induced changes must be compared.

Diagnostic only; writes a JSON into outputs/ (never committed).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.salu.visualize_said import js_divergence  # noqa: E402
from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_val_dataset  # noqa: E402


def encode_caption(model, caption, device):
    tokens = longclip.tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        return F.normalize(model.clip.encode_text(tokens).float(), dim=-1)


def attention(model, patch_features, text_feature):
    with torch.no_grad():
        q = F.normalize(model.said_router.q_proj(text_feature), dim=-1)
        k = F.normalize(model.said_router.k_proj(patch_features.unsqueeze(0)), dim=-1)
        logits = torch.einsum('bd,bnd->bn', q, k) / model.said_router.tau_said
        return F.softmax(logits, dim=-1)[0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Said attention precision-noise floor (bf16 vs fp32)')
    parser.add_argument('--checkpoint', default='runs_salu/phase21/salu_said_only_last.pt')
    parser.add_argument('--tag', default='final')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--output_dir', default='outputs/salu_grounding')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip_model).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dataset = share4v_val_dataset()
    abs_diffs, max_diffs, jsds = [], [], []
    for i in range(args.num_samples):
        image_tensor, caption = dataset[i]
        image = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                _, pf_bf16 = model.clip.encode_image_with_patches(image)
            _, pf_fp32 = model.clip.encode_image_with_patches(image)
        text_feature = encode_caption(model, caption, device)
        a_bf16 = attention(model, pf_bf16[0].float(), text_feature)
        a_fp32 = attention(model, pf_fp32[0].float(), text_feature)
        abs_diffs.append(float(np.abs(a_bf16 - a_fp32).mean()))
        max_diffs.append(float(np.abs(a_bf16 - a_fp32).max()))
        jsds.append(js_divergence(a_bf16, a_fp32))

    result = {
        'tag': args.tag,
        'num_samples': args.num_samples,
        'checkpoint': args.checkpoint,
        'precision_noise_floor': {
            'mean_abs_diff': float(np.mean(abs_diffs)),
            'max_abs_diff': float(np.mean(max_diffs)),
            'mean_js_divergence': float(np.mean(jsds)),
            'p95_js_divergence': float(np.percentile(jsds, 95)),
        },
    }
    diag_path = os.path.join(args.diagnostics_dir, 'diagnostics_final.json')
    if os.path.exists(diag_path):
        diag = json.load(open(diag_path))
        result['caption_shuffle_final'] = diag['caption_shuffle']
        result['comparison'] = {
            'caption_shuffle_jsd': diag['caption_shuffle']['mean_js_divergence'],
            'precision_noise_jsd': result['precision_noise_floor']['mean_js_divergence'],
            'ratio_caption_effect_over_noise': (
                diag['caption_shuffle']['mean_js_divergence']
                / max(1e-12, result['precision_noise_floor']['mean_js_divergence'])
            ),
        }
    out_path = os.path.join(args.output_dir, 'precision_noise_probe_%s.json' % args.tag)
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
    print('PRECISION_NOISE ' + json.dumps(result, sort_keys=True), flush=True)
    print('PRECISION_NOISE_PROBE PASS output=%s' % out_path, flush=True)


if __name__ == '__main__':
    main()
