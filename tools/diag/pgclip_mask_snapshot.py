"""Read-only post-hoc mask snapshot for PG-CLIP v0.1 (dashboard section I).

The trainer's scalar log deliberately carries only aggregate mask statistics, so the 768-d grid of the
page would otherwise have no measured values to show. This tool loads one PG-CLIP checkpoint, pushes a
fixed slice of the reference training stream through the model and the gate **once**, and writes the
per-coordinate hard masks plus the two energy metrics into the run directory.

Read-only by construction: no optimizer, no parameter update (`new_optimizer_updates = 0`), and the
CLIP parameter digest is verified unchanged before and after.

    python tools/diag/pgclip_mask_snapshot.py --checkpoint <pgclip_..._step000500.pt> \
        --out-dir <run dir> [--captions 8]
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (os.path.join(REPO, 'train'), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.pgclip import (NORM_EPS, OBJECTIVE, PreProjectionGate,             # noqa: E402
                          config_dict, energy_metrics, mask_statistics, state_digest)
from said_cvssl_data import Share4VCvsslDataset                                # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--captions', type=int, default=8,
                        help='how many consecutive samples of the reference stream to show')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--base_model', default='ViT-B/16')
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('objective') != OBJECTIVE:
        raise SystemExit('not a PG-CLIP checkpoint: objective=%r' % payload.get('objective'))
    device = torch.device(args.device)

    model, _ = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.load_state_dict(payload['clip'], strict=True)
    model = model.to(device).eval()
    gate = PreProjectionGate(device=device).to(device)
    gate.load_state_dict(payload['gate'])
    gate.eval()
    digest_before = state_digest(model.state_dict())

    dataset = Share4VCvsslDataset(seed=0, augment_view_b=False)
    count = max(1, int(args.captions))
    captions = []
    images = []
    for index in range(count):
        sample = dataset[index]
        captions.append(sample['caption_said'])
        images.append(sample['image_a'])
    images = torch.stack(images).to(device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            h_pre, native = model.encode_image_with_preprojection(images)
            hidden = model.encode_text_final_hidden(
                longclip.tokenize(captions, truncate=True).to(device))
        with torch.autocast(device_type='cuda', enabled=False):
            h = h_pre.float()
            H = hidden.float()
            masks, probabilities = gate(H)
            W = model.visual.proj.float()
            per_caption = masks.sum(dim=1)
            groups = []
            for index in range(count):
                groups.append({
                    'index': index,
                    'label': 'caption %d：%s' % (index, captions[index][:60]),
                    'kept': int(per_caption[index].item()),
                    'mask': [int(value) for value in (masks[index] >= 0.5).tolist()],
                    'probability_quantiles': [
                        float(value) for value in torch.quantile(
                            probabilities[index].float(),
                            torch.tensor([0.05, 0.5, 0.95], device=device))],
                })
            statistics = mask_statistics(masks, probabilities)
            energy = energy_metrics(h, W, masks)
    digest_after = state_digest(model.state_dict())
    if digest_before != digest_after:
        raise SystemExit('the parameter digest changed: this tool must stay read-only')

    snapshot = {
        'probe': 'pgclip_mask_snapshot', 'read_only': True, 'new_optimizer_updates': 0,
        'objective': OBJECTIVE, 'arm': payload.get('arm'), 'phase': payload.get('phase'),
        'completed_steps': payload.get('completed_steps'),
        'source': 'the first %d samples of the reference training stream (seed 0, image_a only), '
                  'one visual pass and one text pass, gate evaluated once' % count,
        'captions': [caption[:200] for caption in captions],
        'caption_sha256': __import__('hashlib').sha256(
            '\n'.join(captions).encode('utf-8')).hexdigest(),
        'coordinates': int(masks.shape[1]),
        'mask_groups': groups,
        'per_caption_keep': [int(value) for value in per_caption.tolist()],
        'statistics': statistics,
        'energy': energy,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': __import__('model.pgclip', fromlist=['file_sha256']).file_sha256(
            args.checkpoint),
        'clip_state_digest': digest_after,
        'config_identity': config_dict(),
        'not_run': ['any training or optimizer update', 'any causal reading of one coordinate',
                    'the grid is a coordinate index layout, not image space'],
    }
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pgclip_mask_snapshot.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('WROTE %s' % path)
    print('MASK_SNAPSHOT ' + json.dumps({
        'completed_steps': snapshot['completed_steps'],
        'kept_mean': statistics['mask_kept_mean'],
        'kept_min': statistics['mask_kept_min'],
        'kept_max': statistics['mask_kept_max'],
        'all_on_fraction': statistics['mask_all_on_fraction'],
        'all_off_fraction': statistics['mask_all_off_fraction'],
        'probability_mean': statistics['gate_probability_mean'],
        'probability_min': statistics['gate_probability_min'],
        'preproj_retained_energy_mean': energy['preproj_retained_energy_mean'],
        'projected_output_energy_ratio_mean': energy['projected_output_energy_ratio_mean'],
        'projected_output_energy_ratio_max': energy['projected_output_energy_ratio_max'],
        'projected_output_energy_ratio_above_one_fraction':
            energy['projected_output_energy_ratio_above_one_fraction'],
        'new_optimizer_updates': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
