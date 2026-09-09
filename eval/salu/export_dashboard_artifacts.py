"""Export precomputed Said-dashboard artifacts from checkpoints (no training, no UI).

Reads: SALU checkpoints, the fixed diagnostic set, ShareGPT4V validation captions.
Writes: outputs/salu_dashboard/{manifest.json, metrics.json, <tag>/...}

Per checkpoint / sample it stores
    sampleXX_image.png            (original image thumbnail, written once)
    sampleXX_{own,shuffled,short,long}.npy   (14x14 Said attention)
and per checkpoint it stores aggregate metrics (attention entropy / effective
patch count / own-vs-shuffled JSD / z_s cosine / route + evidence identification /
bf16-vs-fp32 precision noise floor / caption-conditioning ratio).

The Streamlit dashboard only reads these artifacts; it never loads a checkpoint.
All outputs stay under ``outputs/`` and are git-ignored.
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

from eval.salu.visualize_said import denormalize, js_divergence  # noqa: E402
from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_val_dataset  # noqa: E402

VARIANTS = ['own', 'shuffled', 'short', 'long']


def first_sentence(caption):
    parts = caption.split('. ')
    return parts[0] if parts else caption


def parse_checkpoints(spec):
    """'initial:,step100:/path/a.pt,final:/path/b.pt' -> [(tag, path_or_None)]."""
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        tag, _, path = item.partition(':')
        out.append((tag.strip(), path.strip() or None))
    return out


def encode_texts(model, captions, device):
    tokens = longclip.tokenize(captions, truncate=True).to(device)
    with torch.no_grad():
        features = model.clip.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


def attention(model, patch_features, text_feature):
    """A_s / z_s for one image and one caption (patch_features [N, D], text [D])."""
    with torch.no_grad():
        q = F.normalize(model.said_router.q_proj(text_feature.unsqueeze(0)), dim=-1)          # [1, D]
        k = F.normalize(model.said_router.k_proj(patch_features.unsqueeze(0)), dim=-1)        # [1, N, D]
        logits = torch.einsum('bd,bnd->bn', q, k) / model.said_router.tau_said
        A = F.softmax(logits, dim=-1)[0]                                                      # [N]
        z = F.normalize(torch.einsum('bn,bnd->bd', A.unsqueeze(0), patch_features.unsqueeze(0)), dim=-1)[0]
    return A.float().cpu().numpy(), z.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Export Said dashboard artifacts')
    parser.add_argument('--checkpoints', required=True,
                        help='comma list tag:path, empty path = untrained (e.g. initial:,final:/x.pt)')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--output_dir', default='outputs/salu_dashboard')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints = parse_checkpoints(args.checkpoints)
    if not checkpoints:
        raise SystemExit('no checkpoints parsed from --checkpoints')

    dataset = share4v_val_dataset()
    diag_set_path = os.path.join(args.diagnostics_dir, 'diagnostic_set.json')
    perm = json.load(open(diag_set_path))['permutation'] if os.path.exists(diag_set_path) \
        else list(range(args.num_samples))

    captions = {}
    for i in range(args.num_samples):
        _, own = dataset[i]
        _, shuf = dataset[perm[i]]
        captions[str(i)] = {
            'own': own,
            'shuffled': shuf,
            'short': first_sentence(own),
            'long': own,
            'image_id': dataset.json_data[i]['image'],
            'shuffled_with_index': perm[i],
            'shuffled_image_id': dataset.json_data[perm[i]]['image'],
        }

    manifest = {
        'num_samples': args.num_samples,
        'variants': VARIANTS,
        'checkpoints': [tag for tag, _ in checkpoints],
        'samples': [{'index': i, 'image_id': captions[str(i)]['image_id'],
                     'shuffled_with_index': perm[i]} for i in range(args.num_samples)],
        'captions': captions,
        'source': 'share4v_val_dataset, fixed diagnostic set (eval/salu/diagnostic_set.json)',
    }
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)

    metrics = {'checkpoints': {}, 'variants': VARIANTS}
    wrote_images = False

    for tag, ckpt_path in checkpoints:
        clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
        model = SALUModel(clip_model, said_loss_mode='identifiable', pair_chunk_size=None).to(device)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            step = ckpt.get('step')
        else:
            step = 0
        model.eval()
        tag_dir = os.path.join(args.output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        # text features for the 4 caption variants of every sample
        variant_texts = {v: [] for v in VARIANTS}
        for i in range(args.num_samples):
            for v in VARIANTS:
                variant_texts[v].append(captions[str(i)][v])
        text_features = {v: encode_texts(model, variant_texts[v], device) for v in VARIANTS}

        pf_bf16, pf_fp32 = [], []
        own_attn, noise_jsds, noise_abs = [], [], []
        per_sample = {}
        for i in range(args.num_samples):
            image_tensor, _ = dataset[i]
            image = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    _, p_bf16 = model.clip.encode_image_with_patches(image)
                _, p_fp32 = model.clip.encode_image_with_patches(image)
            pf_bf16.append(p_bf16[0].float())
            pf_fp32.append(p_fp32[0].float())
            if not wrote_images:
                from PIL import Image
                Image.fromarray(denormalize(image_tensor)).save(
                    os.path.join(tag_dir, 'sample%02d_image.png' % i))

        for i in range(args.num_samples):
            entry = {'image_id': captions[str(i)]['image_id'], 'attention': {}, 'zs': {}}
            attn_maps, zs_maps = {}, {}
            for v in VARIANTS:
                A, z = attention(model, pf_bf16[i], text_features[v][i])
                attn_maps[v], zs_maps[v] = A, z
                np.save(os.path.join(tag_dir, 'sample%02d_%s.npy' % (i, v)), A.reshape(14, 14))
                entry['attention'][v] = {
                    'entropy': float(-(np.clip(A, 1e-12, None) * np.log(np.clip(A, 1e-12, None))).sum()),
                    'max': float(A.max()),
                }
                entry['zs'][v] = z.tolist()
            A_own, z_own = attn_maps['own'], zs_maps['own']
            A_noise, _ = attention(model, pf_fp32[i], text_features['own'][i])
            noise_jsds.append(float(js_divergence(A_own, A_noise)))
            noise_abs.append(float(np.abs(A_own - A_noise).mean()))
            own_attn.append(A_own)
            entry['own_vs_shuffled'] = {
                'mean_abs_diff': float(np.abs(A_own - attn_maps['shuffled']).mean()),
                'js_divergence': float(js_divergence(A_own, attn_maps['shuffled'])),
                'zs_cosine': float(np.dot(z_own, zs_maps['shuffled'])),
            }
            entry['short_vs_long'] = {
                'mean_abs_diff': float(np.abs(attn_maps['short'] - attn_maps['long']).mean()),
                'js_divergence': float(js_divergence(attn_maps['short'], attn_maps['long'])),
                'zs_cosine': float(np.dot(zs_maps['short'], zs_maps['long'])),
            }
            per_sample[str(i)] = entry

        # pairwise route / evidence identification over the fixed set
        patch_stack = torch.stack(pf_bf16)                     # [S, N, D]
        text_stack = text_features['own']                      # [S, D]
        with torch.no_grad():
            z_pair, _ = model.said_router.route_pairwise(text_stack, patch_stack, chunk_size=None)
            scale = model.clip.logit_scale.exp().clamp(max=100)
            route_score = (scale * torch.einsum('ijd,id->ij', z_pair, text_stack)).float().cpu().numpy()
            evidence_score = (scale * torch.einsum('jid,id->ij', z_pair, text_stack)).float().cpu().numpy()
            z_pair_np = z_pair.float().cpu().numpy()

        labels = np.arange(args.num_samples)
        route_top1 = float((route_score.argmax(axis=1) == labels).mean())
        evidence_top1 = float((evidence_score.argmax(axis=1) == labels).mean())
        route_margin = float(np.mean(route_score[labels, labels] -
                                     (route_score.sum(axis=1) - route_score[labels, labels]) / (args.num_samples - 1)))
        evidence_margin = float(np.mean(evidence_score[labels, labels] -
                                         (evidence_score.sum(axis=1) - evidence_score[labels, labels]) / (args.num_samples - 1)))

        abs_diffs, jsds, cosines, entropies = [], [], [], []
        for i in range(args.num_samples):
            A_own, A_shuf = own_attn[i], np.load(os.path.join(tag_dir, 'sample%02d_shuffled.npy' % i)).reshape(-1)
            abs_diffs.append(float(np.abs(A_own - A_shuf).mean()))
            jsds.append(float(js_divergence(A_own, A_shuf)))
            cosines.append(float(np.dot(z_pair_np[i, i], z_pair_np[i, perm[i]])))
            entropies.append(float(-(np.clip(A_own, 1e-12, None) * np.log(np.clip(A_own, 1e-12, None))).sum()))

        metrics['checkpoints'][tag] = {
            'step': step,
            'checkpoint': ckpt_path,
            'said_attention': {
                'mean_entropy': float(np.mean(entropies)),
                'mean_effective_patch_count': float(np.mean(np.exp(entropies))),
                'mean_attention_max': float(np.mean([a.max() for a in own_attn])),
            },
            'caption_shuffle': {
                'mean_abs_diff': float(np.mean(abs_diffs)),
                'mean_js_divergence': float(np.mean(jsds)),
                'mean_zs_cosine': float(np.mean(cosines)),
            },
            'route_identification': {'top1_acc': route_top1, 'chance': 1.0 / args.num_samples,
                                     'top1_over_chance': route_top1 * args.num_samples, 'margin': route_margin},
            'evidence_identification': {'top1_acc': evidence_top1, 'chance': 1.0 / args.num_samples,
                                        'top1_over_chance': evidence_top1 * args.num_samples, 'margin': evidence_margin},
            'precision_noise': {
                'mean_js_divergence': float(np.mean(noise_jsds)),
                'mean_abs_diff': float(np.mean(noise_abs)),
            },
        }
        metrics['checkpoints'][tag]['caption_conditioning_ratio'] = (
            metrics['checkpoints'][tag]['caption_shuffle']['mean_js_divergence']
            / max(1e-12, metrics['checkpoints'][tag]['precision_noise']['mean_js_divergence'])
        )
        with open(os.path.join(tag_dir, 'per_sample_metrics.json'), 'w') as fp:
            json.dump(per_sample, fp, indent=2, sort_keys=True)
        print('EXPORTED %s (step %s) ratio=%.3f route_top1=%.4f evidence_top1=%.4f' % (
            tag, step, metrics['checkpoints'][tag]['caption_conditioning_ratio'], route_top1, evidence_top1), flush=True)
        wrote_images = True

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
    print('DASHBOARD_EXPORT PASS output_dir=%s checkpoints=%d samples=%d' % (
        args.output_dir, len(checkpoints), args.num_samples), flush=True)


if __name__ == '__main__':
    main()
