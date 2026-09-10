"""Phase 3.0A.1c: forward-only complement sweep (no optimizer step, no C_U, no C_F).

One fixed ``(I, C_S)`` batch, one fixed initial checkpoint, and a grid of

    gap_anti_temperature in [2.0, 1.0, 0.5, 0.25, 0.10]  x  said_feature_source in
    [residual, attention_delta]

Nothing is trained: the script only forwards and measures. Every row records the batch
SHA so a reader can verify that all rows saw identical inputs and identical weights.

Writes ``outputs/phase30a_complement_diagnostic/temperature_sweep.json``.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, 'train')
for _p in (REPO, TRAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import longclip
from model import complement_diagnostics as cd
from model.gap_completion import (
    gap_completion_terms,
    gap_diagnostics,
    gap_discovery_loss,
    global_absorption_loss,
    soft_anti_said_attention,
    unsaid_feature_from_attention,
)
from model.salu_model import SALUModel
from model.unsaid_core import attention_jsd, attention_overlap
from model.salu_modules import SaidRouter
from sharegpt4v import share4v_train_dataset

DEFAULT_TEMPERATURES = (2.0, 1.0, 0.5, 0.25, 0.10)
FEATURE_SOURCES = ('residual', 'attention_delta')


def batch_sha(images, captions):
    digest = hashlib.sha256()
    digest.update(images.detach().float().cpu().contiguous().numpy().tobytes())
    digest.update(b'\x00')
    for caption in captions:
        digest.update(str(caption).encode('utf-8'))
        digest.update(b'\x01')
    return digest.hexdigest()


def model_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(model, images, tokens, tau, captions, feature_source, checkpoint_sha):
    """One forward-only measurement; no parameter is touched, no loss is differentiated."""
    z_global, patch_features = model.encode_router_input(images)
    text_feature = model.encode_text(tokens)
    t_s = F.normalize(text_feature, dim=-1)
    details = model.said_router.forward_with_details(t_s, patch_features)
    z_said = details['said']
    attention_said = details['attention']
    attention_unsaid = soft_anti_said_attention(details['scores'], temperature=tau)
    z_unsaid = unsaid_feature_from_attention(attention_unsaid, patch_features)

    terms = gap_completion_terms(z_global, z_said, z_unsaid)
    loss_gap = gap_discovery_loss(terms)
    loss_absorb = global_absorption_loss(z_global, z_said, z_unsaid)

    entropy_said = SaidRouter.attention_entropy(attention_said.detach().float())
    entropy_unsaid = SaidRouter.attention_entropy(attention_unsaid.detach().float())
    diagnostics = gap_diagnostics(terms)
    homogeneity = cd.patch_homogeneity_metrics(patch_features, z_global)
    raw_pools = cd.raw_pooling_metrics(attention_said, attention_unsaid, patch_features)
    relations = cd.global_relation_metrics(z_global, z_said, z_unsaid)

    row = {
        'feature_source': feature_source,
        'gap_anti_temperature': float(tau),
        'batch_sha256': batch_sha(images.cpu(), captions),
        'checkpoint_state_sha256': checkpoint_sha,
        'batch_size': int(images.shape[0]),
        'patch_count': int(patch_features.shape[1]),
        'said_attention_entropy': float(entropy_said.mean()),
        'unsaid_attention_entropy': float(entropy_unsaid.mean()),
        'said_unsaid_attention_overlap': float(
            attention_overlap(attention_said.detach().float(),
                              attention_unsaid.detach().float()).mean()),
        'said_unsaid_attention_jsd': float(
            attention_jsd(attention_said.detach().float(),
                          attention_unsaid.detach().float()).mean()),
        'loss_gap_discover': float(loss_gap),
        'loss_global_absorb': float(loss_absorb),
        'loss_total_if_weights_one': float(loss_gap + loss_absorb),
    }
    for source in (diagnostics, homogeneity, raw_pools, relations):
        for key, value in source.items():
            if key == 'patch_count':
                continue
            row[key] = float(value)
    return row


def main():
    parser = argparse.ArgumentParser(description='Phase 3.0A.1c forward-only sweep')
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default=os.path.join(
        'outputs', 'phase30a_complement_diagnostic', 'temperature_sweep.json'))
    parser.add_argument('--temperatures', default=','.join(str(t) for t in DEFAULT_TEMPERATURES))
    parser.add_argument('--feature_sources', default=','.join(FEATURE_SOURCES))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    temperatures = [float(item) for item in args.temperatures.split(',') if item.strip()]
    feature_sources = [item for item in args.feature_sources.split(',') if item.strip()]
    device = torch.device(args.device)

    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    torch.manual_seed(args.seed)
    clip_model, _ = longclip.load_from_clip(args.base_model, device='cpu')

    # one fixed batch: (I, C_S) from the real dataset, order fixed by seed + no shuffle
    dataset = share4v_train_dataset(caption_views=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, captions = next(iter(loader))
    captions = list(captions)
    images = images.to(device)
    tokens = longclip.tokenize(captions, truncate=True).to(device)
    fixed_batch_sha = batch_sha(images.cpu(), captions)
    print('fixed batch: %r  batch_sha256=%s' % (tuple(images.shape), fixed_batch_sha), flush=True)
    print('example C_S: %s' % captions[0][:100], flush=True)

    rows = []
    weight_shas = {}
    for feature_source in feature_sources:
        # every feature source starts from the exact same initial backbone + router
        torch.manual_seed(args.seed)
        model = SALUModel(clip_model, tau_said=0.07, said_loss_mode='identifiable',
                          pair_chunk_size=64, said_feature_source=feature_source)
        model = model.to(device).eval()
        model_sha_before = model_sha(model)
        weight_shas[feature_source] = model_sha_before
        with torch.no_grad():
            for tau in temperatures:
                row = measure(model, images, tokens, tau, captions, feature_source,
                              model_sha_before)
                rows.append(row)
                print('SWEEP %s tau=%.2f  jsd=%.6f overlap=%.6f cos(zS,zU)=%.6f '
                      'novel=%.6f closure=%.6f gap_after=%.6f patch_pair_cos=%.6f '
                      'centered_energy=%.6f raw_pool_cos=%.6f'
                      % (feature_source, tau, row['said_unsaid_attention_jsd'],
                         row['said_unsaid_attention_overlap'], row['cos_said_unsaid'],
                         row['unsaid_novel_component_norm'], row['gap_closure_ratio_mean'],
                         row['gap_after_mean'], row['patch_pair_cosine_mean'],
                         row['patch_centered_energy'], row['raw_pool_cosine']), flush=True)
        assert model_sha(model) == model_sha_before, 'forward-only sweep changed the weights'
        del model
        torch.cuda.empty_cache()

    payload = {
        'protocol': 'phase30a-1c-forward-only-temperature-sweep',
        'base_model': args.base_model,
        'seed': args.seed,
        'batch_size': int(images.shape[0]),
        'batch_sha256': fixed_batch_sha,
        'weights_sha256': weight_shas,
        'temperatures': temperatures,
        'feature_sources': feature_sources,
        'training': False,
        'optimizer_steps': 0,
        'c_f_used': False,
        'c_u_used': False,
        'notes': ('Forward only on one fixed (I, C_S); identical weights for every row. '
                  'loss_gap_discover / loss_global_absorb are reported for reference only '
                  'and are mathematically equal because target = normalize(s_ref + u_new) '
                  'and g_ref = normalize(g). Attention diversity is not useful closure: see '
                  'gap_after_mean and gap_closure_ratio_mean.'),
        'rows': rows,
    }
    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print('WROTE %s (%d rows)' % (output_path, len(rows)), flush=True)


if __name__ == '__main__':
    main()
