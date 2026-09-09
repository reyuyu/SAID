"""Phase 2.2 diagnostics: identifiable Said routing on the fixed 64-image set.

Computes, for one checkpoint:
  * route identification:   argmax_j route_score[i, j] == i  (caption varies, text fixed)
  * evidence identification: argmax_j evidence_score[i, j] == i (image varies, text fixed)
  * own-vs-shuffled caption attention (fixed permutation) JSD / mean abs diff / z_s cosine
  * own-caption attention entropy / effective patch count / max
  * caption-conditioning ratio = caption JSD / precision-noise JSD (if available)

Diagnostic only; writes JSON into outputs/ (never committed).
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


def main():
    parser = argparse.ArgumentParser(description='Identifiable Said routing diagnostics')
    parser.add_argument('--checkpoint', default=None, help='SALU checkpoint (default: untrained init)')
    parser.add_argument('--tag', default='p22_final')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    clip_model, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip_model, said_loss_mode='identifiable', pair_chunk_size=None).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print('loaded %s (step %s, mode %s)' % (args.checkpoint, ckpt.get('step'),
                                                ckpt.get('args', {}).get('said_loss_mode')), flush=True)
    model.eval()

    dataset = share4v_val_dataset()
    diag_set_path = os.path.join(args.diagnostics_dir, 'diagnostic_set.json')
    if os.path.exists(diag_set_path):
        perm = json.load(open(diag_set_path))['permutation']
    else:
        perm = list(range(args.num_samples))

    patches, texts = [], []
    for i in range(args.num_samples):
        image_tensor, caption = dataset[i]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                _, patch_features = model.clip.encode_image_with_patches(image_tensor.unsqueeze(0).to(device))
            tokens = longclip.tokenize([caption], truncate=True).to(device)
            text_feature = model.clip.encode_text(tokens)
        patches.append(patch_features[0].float())
        texts.append(F.normalize(text_feature.float(), dim=-1)[0])

    patch_features = torch.stack(patches)                      # [S, N, D]
    text_features = torch.stack(texts)                         # [S, D]

    with torch.no_grad():
        z_pair, A_pair = model.said_router.route_pairwise(
            text_features, patch_features, chunk_size=None, return_attention=True
        )
        scale = model.clip.logit_scale.exp().clamp(max=100)
        route_score = (scale * torch.einsum('ijd,id->ij', z_pair, text_features)).float().cpu().numpy()
        evidence_score = (scale * torch.einsum('jid,id->ij', z_pair, text_features)).float().cpu().numpy()
        A_pair = A_pair.float().cpu().numpy()

    labels = np.arange(args.num_samples)
    route_top1 = float((route_score.argmax(axis=1) == labels).mean())
    evidence_top1 = float((evidence_score.argmax(axis=1) == labels).mean())
    route_margin = float(np.mean(route_score[labels, labels] -
                                 (route_score.sum(axis=1) - route_score[labels, labels]) / (args.num_samples - 1)))
    evidence_margin = float(np.mean(evidence_score[labels, labels] -
                                     (evidence_score.sum(axis=1) - evidence_score[labels, labels]) / (args.num_samples - 1)))

    abs_diffs, jsds, cosines, entropies = [], [], [], []
    for i in range(args.num_samples):
        A_own = A_pair[i, i]
        A_shuf = A_pair[i, perm[i]]
        z_own, z_shuf = z_pair[i, i].float().cpu().numpy(), z_pair[i, perm[i]].float().cpu().numpy()
        abs_diffs.append(float(np.abs(A_own - A_shuf).mean()))
        jsds.append(float(js_divergence(A_own, A_shuf)))
        cosines.append(float(np.dot(z_own, z_shuf)))
        entropies.append(float(-(np.clip(A_own, 1e-12, None) * np.log(np.clip(A_own, 1e-12, None))).sum()))

    result = {
        'tag': args.tag,
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'route_identification': {
            'top1_acc': route_top1,
            'chance': 1.0 / args.num_samples,
            'top1_over_chance': route_top1 * args.num_samples,
            'margin': route_margin,
        },
        'evidence_identification': {
            'top1_acc': evidence_top1,
            'chance': 1.0 / args.num_samples,
            'top1_over_chance': evidence_top1 * args.num_samples,
            'margin': evidence_margin,
        },
        'caption_shuffle': {
            'mean_abs_diff': float(np.mean(abs_diffs)),
            'mean_js_divergence': float(np.mean(jsds)),
            'mean_zs_cosine': float(np.mean(cosines)),
        },
        'said_attention': {
            'mean_entropy': float(np.mean(entropies)),
            'mean_effective_patch_count': float(np.mean(np.exp(entropies))),
            'mean_attention_max': float(np.mean([A_pair[i, i].max() for i in range(args.num_samples)])),
        },
    }
    noise_path = os.path.join(args.diagnostics_dir, 'precision_noise_probe_%s.json' % args.tag)
    if os.path.exists(noise_path):
        noise = json.load(open(noise_path))['precision_noise_floor']['mean_js_divergence']
        result['caption_conditioning'] = {
            'caption_jsd': result['caption_shuffle']['mean_js_divergence'],
            'precision_noise_jsd': noise,
            'ratio': result['caption_shuffle']['mean_js_divergence'] / max(1e-12, noise),
        }

    out_path = os.path.join(args.diagnostics_dir, 'diagnostics_identifiable_%s.json' % args.tag)
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
    print('IDENTIFIABLE_DIAG ' + json.dumps(result, sort_keys=True), flush=True)
    print('IDENTIFIABLE_DIAG_RESULT PASS output=%s' % out_path, flush=True)


if __name__ == '__main__':
    main()
