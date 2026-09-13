"""Read-only attention snapshot for CG-CLIP v0.1 (the caption gate and the two CLS attention rows).

CG-CLIP keeps the native CLIP encoder and computes the LAST visual block's CLS row twice. The
``native`` path is the untouched native computation. The ``conditional`` path re-uses the same native
CLS query, Keys and Values and the same ``out_proj`` / residuals / ``ln_2`` / MLP / ``ln_post`` /
``visual.proj``, but renormalises the native CLS attention weights with a 196-d per-(text, image)
patch gate produced by ``model/cgclip.py:CaptionGate``:

    m      = hard(p >= 0.5) + (p - p.detach())     # 196 patch gates, straight-through
    m_full = [1, m]                                # the CLS slot keeps gate 1
    a_cond = (a * m_full) / sum_p (a * m_full)     # no detached denominator, no clamp

The trainer's scalar log carries only aggregates, so the 14x14 grid of the page would otherwise have
no measured values to show and the two CLS attention rows would never be visible. This tool loads one
CG-CLIP checkpoint, pushes a small fixed slice of the reference training stream through the model and
the gate **once**, and writes the measured values as JSON.

Read-only by construction: no optimizer is imported, no parameter is updated, the CLIP and gate
parameter digests are verified unchanged before and after the forward pass, and the tool REFUSES any
output path that lies inside ``--run_dir`` so it can never write into the training run directory.
The sample is small and fixed on purpose; it is a display snapshot, never a training statistic.

    python tools/diag/cgclip_attention_snapshot.py \
        --checkpoint runs_salu/cgclip_v01/CG_CLIP_V01_step000500.pt \
        --output runs_salu/cgclip_v01_diag/cgclip_attention_snapshot.json
"""
import argparse
import datetime
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (os.path.join(REPO, 'train'), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.cgclip import (ARM, FIXED_SCALE, GATE_KIND, NORM_EPS, OBJECTIVE,   # noqa: E402
                          PATCH_TOKENS, CaptionGate, attention_map_grid,
                          attention_read_diagnostics, config_dict, conditional_cls_scores,
                          file_sha256, gate_config, gate_statistics, git_head, native_cls_row,
                          state_digest, visual_spec)
from said_cvssl_data import IMAGE_SIZE, Share4VCvsslDataset                       # noqa: E402

ARM_ALIASES = {'B16': 'ViT-B/16', 'B32': 'ViT-B/32', 'L14': 'ViT-L/14'}
PROBABILITY_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
CLS_SLOT_GATE_VALUE = 1.0
DEFAULT_BASE_MODEL = 'ViT-B/16'
DEFAULT_SAMPLES = 8
DEFAULT_GRID_SIDE = 14
SCOPE_NOTE = ('this is a small fixed READ-ONLY sample of the reference training stream, not a '
              'training statistic and not a training-set measurement')


def check_output_path(output, run_dir):
    """Refuse any output path inside the run directory: this tool must never write into a run."""
    if run_dir is None:
        return None
    run_dir = os.path.abspath(run_dir)
    output = os.path.abspath(output)
    if output == run_dir or output.startswith(run_dir + os.sep):
        raise SystemExit('refusing to write %s: the output must NOT live inside --run_dir %s '
                         '(this tool is read-only with respect to the run directory)'
                         % (output, run_dir))
    return run_dir


def verify_payload(payload, checkpoint):
    """Refuse anything that is not a complete CG-CLIP v0.1 checkpoint with both state dicts."""
    if not isinstance(payload, dict):
        raise SystemExit('checkpoint %s is not a dict payload (got %s)'
                         % (checkpoint, type(payload).__name__))
    missing = [key for key in ('clip', 'gate_state', 'completed_steps') if key not in payload]
    if missing:
        raise SystemExit('checkpoint %s is not a CG-CLIP payload: missing keys %r'
                         % (checkpoint, missing))
    if payload.get('objective') != OBJECTIVE:
        raise SystemExit('not a CG-CLIP checkpoint: objective=%r expected %r'
                         % (payload.get('objective'), OBJECTIVE))
    if payload.get('gate_kind') != GATE_KIND:
        raise SystemExit('not a CG-CLIP checkpoint: gate_kind=%r expected %r'
                         % (payload.get('gate_kind'), GATE_KIND))
    if not isinstance(payload['gate_state'], dict) or not payload['gate_state']:
        raise SystemExit('checkpoint %s carries no gate tensors: the caption gate cannot be rebuilt '
                         'and the conditional CLS row cannot be reproduced' % checkpoint)
    return payload


def load_sample(dataset, images, captions, sample_ids, count, device):
    """The first ``count`` samples of the reference stream: ``image_a`` only, plus their captions."""
    if len(dataset) < count:
        raise SystemExit('the dataset exposes %d samples, which is fewer than the %d requested'
                         % (len(dataset), count))
    for index in range(count):
        sample = dataset[index]
        captions.append(sample['caption_said'])
        images.append(sample['image_a'])
        sample_ids.append(int(sample['sample_id']))
    return torch.stack(images).to(device)


def conditional_attention(attention, mask):
    """Defensive reference of ``a_cond = (a * m_full) / sum_p (a * m_full)`` with the CLS slot at 1.

    The objective performs this renormalisation inside
    ``model/cgclip.py:_conditional_block``; the tool does not call this helper, it records the
    attention that the objective itself returned. The reference lives here so that the recorded
    values can be re-derived independently, and so that the class-slot convention (gate 1, never
    folded into the 196 patch gates) is stated in the code that writes the snapshot.
    """
    patch_mask = mask.float()
    mask_full = torch.cat([torch.ones_like(patch_mask[..., :1]), patch_mask], dim=-1)
    weighted = attention.unsqueeze(0) * mask_full.unsqueeze(2)
    denominator = weighted.sum(dim=-1, keepdim=True)
    if not torch.isfinite(denominator).all() or bool((denominator <= 0).any()):
        raise SystemExit('non-finite or non-positive conditional attention denominator: the gate '
                         'closed every slot of at least one (text, image) pair')
    return weighted / denominator, denominator


def mask_full_from_gate(mask):
    """``m_full = [1, m]``: the CLS slot keeps gate 1 and is never part of the 196 patch gates."""
    patch_mask = mask.float()
    return torch.cat([torch.ones_like(patch_mask[..., :1]), patch_mask], dim=-1)


def attention_rows(attention_row):
    """One CLS attention row ``[H, 197]`` as recorded lists, never folding the CLS slot into patches."""
    row = attention_row.detach().float()
    return {
        'per_head': [[float(v) for v in row[head].tolist()] for head in range(row.shape[0])],
        'head_mean': [float(v) for v in row.mean(dim=0).tolist()],
        'head_averaged_field': 'head_mean',
        'head_averaged_note': 'head_mean is a HEAD-AVERAGED DISPLAY VALUE; the per_head lists are '
                              'the measured per-head rows',
        'cls_self_mass_per_head': [float(v) for v in row[:, 0].tolist()],
        'patch_mass_per_head': [float(v) for v in row[:, 1:].sum(-1).tolist()],
        'cls_self_mass_head_mean': float(row[:, 0].mean()),
        'patch_mass_head_mean': float(row[:, 1:].sum(-1).mean()),
        'index_0_is': 'the CLS slot itself; indices 1..196 are the 14x14 patch tokens',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='CG-CLIP v0.1 training checkpoint (clip + gate_state)')
    parser.add_argument('--output', required=True,
                        help='JSON file to write; must NOT be inside --run_dir')
    parser.add_argument('--run_dir', default=None,
                        help='the training run directory (used only for the read-only assertion and '
                             'provenance; the tool never writes into it)')
    parser.add_argument('--init_state', default=None,
                        help='shared initial state of the run, recorded in the provenance record '
                             '(optional; never used to modify the loaded weights)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL,
                        help='native CLIP architecture to rebuild: %s (the trainer also accepts the '
                             'short alias B16)' % DEFAULT_BASE_MODEL)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help='how many consecutive samples of the reference stream to show '
                             '(the same convention as the PG-CLIP mask snapshot: 8 samples, '
                             'image_a only)')
    parser.add_argument('--grid_side', type=int, default=DEFAULT_GRID_SIDE,
                        help='side of the square token grid (14 for the 196 patch gates of ViT-B/16)')
    parser.add_argument('--split', default='train',
                        help='which slice of the reference stream to read; the CG-CLIP v0.1 run has '
                             'one stream, so only "train" is implemented and any other value is '
                             'refused rather than silently ignored')
    parser.add_argument('--pairs', type=int, default=0,
                        help='how many (image, caption) pairs to score; 0 = one closed cohort of '
                             '--samples images x --samples captions (the true positive pair of image '
                             'i is caption i)')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE,
                        help='the reference preprocessing size of image_a; it must equal the '
                             'training stream preprocessing (%d) and any other value is refused'
                             % IMAGE_SIZE)
    parsed = parser.parse_args()

    run_dir = check_output_path(parsed.output, parsed.run_dir)
    checkpoint = os.path.abspath(parsed.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit('checkpoint not found: %s' % checkpoint)
    output = os.path.abspath(parsed.output)
    base_model = ARM_ALIASES.get(parsed.base_model, parsed.base_model)
    side = int(parsed.grid_side)
    if side * side != PATCH_TOKENS:
        raise SystemExit('--grid_side %d does not describe the %d patch tokens of this model'
                         % (side, PATCH_TOKENS))
    if int(parsed.image_size) != int(IMAGE_SIZE):
        raise SystemExit('--image_size %d is not the preprocessing of the training stream (%d); '
                         'image_a must stay the reference view or the sample is not the same kind of '
                         'data the run used' % (int(parsed.image_size), int(IMAGE_SIZE)))
    if str(parsed.split) != 'train':
        raise SystemExit('--split %r is not implemented: the CG-CLIP v0.1 run reads one reference '
                         'stream, so only "train" (the first samples of that stream) exists'
                         % (parsed.split,))
    device = torch.device(parsed.device)
    use_autocast = device.type == 'cuda'

    payload = verify_payload(torch.load(checkpoint, map_location='cpu', weights_only=False),
                             checkpoint)
    count = max(1, int(parsed.samples))
    images_per_text = count
    if int(parsed.pairs) > 0:
        images_per_text = int(parsed.pairs)
    if images_per_text > count:
        raise SystemExit('--pairs %d needs %d images but only --samples %d are loaded '
                         '(raise --samples or lower --pairs)' % (images_per_text, images_per_text,
                                                                 count))

    model, _preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                args=argparse.Namespace())
    model.load_state_dict(payload['clip'], strict=True)
    model = model.to(device).eval()
    spec = visual_spec(model)
    gate = CaptionGate(device=device).to(device)
    gate.load_state_dict(payload['gate_state'], strict=True)
    gate.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in gate.parameters():
        parameter.requires_grad_(False)

    clip_digest_before = state_digest(model.state_dict())
    gate_digest_before = state_digest(gate.state_dict())

    dataset = Share4VCvsslDataset(seed=0, augment_view_b=False,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    dataset.set_epoch(0)
    captions, image_tensors, sample_ids = [], [], []
    images = load_sample(dataset, image_tensors, captions, sample_ids, count, device)
    texts = longclip.tokenize(captions, truncate=True).to(device)
    visual = model.visual
    last = visual.transformer.resblocks[-1]
    ln_post = visual.ln_post
    projection = visual.proj

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_autocast):
            tokens = model.encode_visual_prefinal(images)
            text_hidden = model.encode_text_final_hidden(texts)
        with torch.autocast(device_type=device.type, enabled=False):
            x11 = tokens['x11_raw'].float()
            hidden = text_hidden.float()
            eot, effective_length = model.eot_indices(texts)
            t_raw = hidden[torch.arange(hidden.shape[0], device=device), eot] \
                @ model.text_projection.float()
            t_unit = F.normalize(t_raw, dim=-1, eps=NORM_EPS)
            native = native_cls_row(last, x11, ln_post, projection, want_attention=True)
            qT = gate.project_query(t_unit)
            kG = gate.project_key(native['u'][:, 1:, :])
            mask, probability, logits = gate.gate_from_keys(qT, kG)
            native_scores = FIXED_SCALE * (F.normalize(native['projected'], dim=-1, eps=NORM_EPS)
                                           @ t_unit.t())
            capture = {}
            conditional_scores, _positive_mask = conditional_cls_scores(
                x11[:images_per_text], native, last, gate, t_unit[:images_per_text], qT, kG,
                projection, ln_post, use_checkpoint=False, capture=capture)
            attention_conditional = capture['attention_conditional']
            projected_conditional = capture['projected']
            cls_conditional = capture['cls']
            mask_conditional = capture['mask']
            probability_conditional = capture['probability']
        with torch.autocast(device_type=device.type, enabled=False):
            projected_native_matrix = native['projected'].unsqueeze(0)
            projected_cosine = F.cosine_similarity(projected_conditional, projected_native_matrix,
                                                   dim=-1)
            cls_cosine = F.cosine_similarity(cls_conditional, native['cls'].unsqueeze(0), dim=-1)
            diagnostics = attention_read_diagnostics(
                {'attention': native['attention'], 'cls': native['cls'],
                 'projected': native['projected'], 'out': native['out'], 'c': native['c']},
                {'attention_conditional': attention_conditional, 'cls': cls_conditional,
                 'projected': projected_conditional})
            statistics = gate_statistics(mask_conditional, probability_conditional)

    clip_digest_after = state_digest(model.state_dict())
    gate_digest_after = state_digest(gate.state_dict())
    if clip_digest_before != clip_digest_after:
        raise SystemExit('the CLIP parameter digest changed: this tool must stay read-only')
    if gate_digest_before != gate_digest_after:
        raise SystemExit('the caption-gate parameter digest changed: this tool must stay read-only')
    if native_scores.shape != conditional_scores.shape:
        raise SystemExit('the native and conditional score matrices disagree in shape: %s vs %s'
                         % (tuple(native_scores.shape), tuple(conditional_scores.shape)))

    quantile_axis = torch.tensor(PROBABILITY_QUANTILES, device=device)
    pairs = []
    for text_index in range(images_per_text):
        for image_index in range(images_per_text):
            patch_gate = mask_conditional[text_index, image_index]
            patch_probability = probability_conditional[text_index, image_index]
            grid = attention_map_grid(patch_gate, side=side)
            native_row = native['attention'][image_index].squeeze(1)
            conditional_row = attention_conditional[text_index, image_index]
            # independent re-derivation: re-project this (text, image) pair and re-apply the
            # straight-through gate, the CLS slot at 1 and the renormalisation. This is what makes
            # the recorded grid, the recorded probabilities and the recorded conditional attention
            # row one consistent set of numbers rather than three unrelated readouts.
            pair_logits = torch.einsum('d,id->i', qT[text_index].float(),
                                       kG[image_index].float()) / math.sqrt(gate.key_dim)
            pair_probability = torch.sigmoid(pair_logits + gate.bias.float())
            pair_mask = (pair_probability >= 0.5).to(pair_probability.dtype) \
                + (pair_probability - pair_probability.detach())
            pair_mask_full = mask_full_from_gate(pair_mask)
            pair_weighted = native_row.unsqueeze(0) * pair_mask_full.unsqueeze(1)
            pair_attention = pair_weighted / pair_weighted.sum(dim=-1, keepdim=True)
            gates_match = bool((pair_mask >= 0.5).equal(patch_gate.detach() >= 0.5))
            probability_max_abs_diff = float((pair_probability - patch_probability).abs().max())
            renormalised_max_abs_diff = float((pair_attention - conditional_row).abs().max())
            if not gates_match or probability_max_abs_diff > 1e-6:
                raise SystemExit('the recorded gate of (text %d, image %d) does not match an '
                                 'independent re-derivation: gates_match=%s '
                                 'probability_max_abs_diff=%r'
                                 % (text_index, image_index, gates_match,
                                    probability_max_abs_diff))
            if renormalised_max_abs_diff > 1e-4:
                raise SystemExit('the conditional CLS attention row of (text %d, image %d) does not '
                                 'reproduce a_cond = (a * m_full) / sum_p (a * m_full): '
                                 'max_abs_diff=%r' % (text_index, image_index,
                                                      renormalised_max_abs_diff))
            pairs.append({
                'pair_index': len(pairs),
                'text_index': text_index,
                'image_index': image_index,
                'is_true_positive_pair': bool(text_index == image_index),
                'caption': captions[text_index][:200],
                'caption_index': text_index,
                'image_sample_id': sample_ids[image_index],
                'gate_grid': {
                    'side': grid['side'],
                    'values': grid['grid'],
                    'note': grid['note'],
                    'cls_slot_gate_value': CLS_SLOT_GATE_VALUE,
                    'cls_slot_index': 0,
                    'cls_slot_note': 'the CLS slot keeps gate 1 by construction; it is recorded '
                                     'SEPARATELY here and is NEVER folded into the 14x14 grid',
                    'grid_scope': 'the 196 patch gates of this (text, image) pair, one gate shared '
                                  'by all 12 vision heads',
                },
                'gate_qc': {
                    'kept': int((patch_gate.detach() >= 0.5).sum()),
                    'kept_fraction': float((patch_gate.detach() >= 0.5).float().mean()),
                    'all_gates_open': bool((patch_gate.detach() >= 0.5).all()),
                    'any_gate_closed': bool((patch_gate.detach() < 0.5).any()),
                    'probability_mean': float(patch_probability.mean()),
                    'probability_min': float(patch_probability.min()),
                    'probability_max': float(patch_probability.max()),
                    'probability_quantiles_5_25_50_75_95': [
                        float(v) for v in torch.quantile(patch_probability.float(), quantile_axis)],
                    'logit_bias': float(gate.bias.detach()),
                },
                'gate_rederivation_check': {
                    'gates_match': gates_match,
                    'probability_max_abs_diff': probability_max_abs_diff,
                    'renormalised_attention_max_abs_diff': renormalised_max_abs_diff,
                    'mask_full_length': int(pair_mask_full.shape[-1]),
                    'note': 'the 196 patch gates, the CLS slot at 1 and the renormalised CLS '
                            'attention row are re-derived from qT and kG of this pair only',
                },
                'native_cls_attention_row': attention_rows(native_row),
                'conditional_cls_attention_row': attention_rows(conditional_row),
                'conditional_attention_row_sum_max_abs_deviation': float(
                    (conditional_row.sum(dim=-1) - 1.0).abs().max()),
                'conditional_patch_mass_head_mean': float(conditional_row[:, 1:].sum(-1).mean()),
                'native_patch_mass_head_mean': float(native_row[:, 1:].sum(-1).mean()),
                'closed_gate_attention_mass_fraction_of_patch_mass': float(
                    (conditional_row[:, 1:] * (patch_gate.detach().float() < 0.5)).sum()
                    / max(float(conditional_row[:, 1:].sum()), 1e-12)),
                'native_vs_conditional': {
                    'projected_cosine_similarity_512d': float(projected_cosine[text_index,
                                                                               image_index]),
                    'cls_cosine_similarity_768d': float(cls_cosine[text_index, image_index]),
                    'native_score_100x': float(native_scores[image_index, text_index]),
                    'conditional_score_100x': float(conditional_scores[image_index, text_index]),
                    'score_scope': 'both scores use the same fixed scale %.1f and the same '
                                   'normalised 512-d projected features' % FIXED_SCALE,
                },
            })

    per_pair_cosine = [float(v) for v in projected_cosine.reshape(-1).tolist()]
    per_pair_kept = [int(v) for v in
                     (mask_conditional.detach() >= 0.5).float().sum(dim=-1).reshape(-1).tolist()]
    snapshot = {
        'probe': 'cgclip_attention_snapshot',
        'arm': ARM,
        'objective': OBJECTIVE,
        'gate_kind': GATE_KIND,
        'read_only': True,
        'new_optimizer_updates': 0,
        'scope': SCOPE_NOTE,
        'scope_detail': {
            'kind': 'small fixed read-only sample',
            'samples': count,
            'pairs': len(pairs),
            'image_size': int(parsed.image_size),
            'grid_side': side,
            'device': str(device),
            'split': str(parsed.split),
            'not_run': ['any training or optimizer update',
                        'any import or construction of the optimizer',
                        'any write inside --run_dir',
                        'any claim about the training distribution: the sample is fixed and small',
                        'any causal reading of one patch coordinate',
                        'the grid is a token-order layout of the 14x14 patch sequence from block 11, '
                        'not a semantic map'],
        },
        'sample': {
            'source': 'the first %d samples of the reference training stream (seed 0, image_a only, '
                      'the openai-clip _transform(%d) preprocessing), one visual pass and one text '
                      'pass, the gate evaluated once' % (count, int(parsed.image_size)),
            'dataset': 'said_cvssl_data:Share4VCvsslDataset(seed=0, augment_view_b=False)',
            'sample_ids': list(sample_ids),
            'captions': [caption[:200] for caption in captions],
            'caption_sha256': __import__('hashlib').sha256(
                '\n'.join(captions).encode('utf-8')).hexdigest(),
            'effective_text_lengths': [int(v) for v in effective_length.tolist()],
            'pairs_scored': len(pairs),
        },
        'pairs': pairs,
        'per_pair_kept': per_pair_kept,
        'per_pair_projected_cosine_similarity_512d': per_pair_cosine,
        'gate_statistics': statistics,
        'gate_statistics_scope': 'model/cgclip.py:gate_statistics over the full %d x %d (text, image) '
                                 'tile of this fixed sample: %d pairs x %d patch gates; the CLS '
                                 'slot is excluded' % (count, count, count * count, PATCH_TOKENS),
        'attention_read_diagnostics': diagnostics,
        'attention_read_diagnostics_scope':
            'the native entries average over this sample\'s images; the conditional entries average '
            'over the (text, image) tiles of this small fixed sample, not over a training epoch',
        'cls_slot': {
            'gate_value': CLS_SLOT_GATE_VALUE,
            'note': 'the CLS slot keeps gate 1 by construction: it is never trainable, it is excluded '
                    'from the sparse term and it is recorded separately from the 196 patch gates',
        },
        'attention_definition': {
            'native': 'softmax_p(q_cls . K_p / sqrt(64)) over the native last-block CLS query and '
                      'the native last-block Keys',
            'conditional': 'a_cond = (a * m_full) / sum_p (a * m_full), m_full = [1, m], no detached '
                           'denominator and no clamp; the query, the Keys, the Values, out_proj, '
                           'both residuals, ln_2/MLP, ln_post and visual.proj are the SAME native '
                           'parameters for both paths',
            'training_only': 'the conditional path is training only and does not exist at inference',
            'cls_self_mass_unit': 'softmax probability mass on the CLS token itself',
        },
        'provenance': {
            'checkpoint': checkpoint,
            'checkpoint_sha256': file_sha256(checkpoint),
            'completed_steps': int(payload['completed_steps']),
            'clip_state_digest_recomputed': clip_digest_after,
            'clip_state_digest_stored': payload.get('clip_state_digest'),
            'clip_state_digest_matches_stored': payload.get('clip_state_digest') == clip_digest_after,
            'gate_state_digest_recomputed': gate_digest_after,
            'gate_state_digest_stored': payload.get('gate_state_digest'),
            'gate_state_digest_matches_stored': payload.get('gate_state_digest') == gate_digest_after,
            'parameter_digests_unchanged': bool(clip_digest_before == clip_digest_after
                                                and gate_digest_before == gate_digest_after),
            'git_head': git_head(REPO),
            'source_git_head': payload.get('git_head'),
            'device': str(device),
            'device_name': (torch.cuda.get_device_name(device) if device.type == 'cuda'
                            else 'cpu'),
            'dtype': {
                'master_weights': 'float32',
                'autocast_dtype': ('bfloat16 for the first 11 visual blocks and the text transformer'
                                   if use_autocast else 'disabled (cpu)'),
                'fp32_core': 'ln_1, Q/K/V, the native softmax, the caption gate, the '
                             'renormalisation, out_proj, both residuals, ln_2/MLP, ln_post, '
                             'visual.proj, the normalisation and the scores',
                'recorded_tensors': 'float32 (every recorded value is cast to float32)',
            },
            'torch_version': torch.__version__,
            'run_dir': run_dir,
            'run_dir_not_written': True,
            'init_state': (os.path.abspath(parsed.init_state) if parsed.init_state else None),
            'init_state_sha256': (file_sha256(os.path.abspath(parsed.init_state))
                                  if parsed.init_state and os.path.isfile(parsed.init_state)
                                  else None),
            'init_state_used': False,
            'init_state_note': 'the shared init is recorded for provenance only; the loaded weights '
                               'come from payload["clip"] and payload["gate_state"]',
            'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'config_identity': config_dict(),
            'gate_config': gate_config(gate),
            'visual_spec': spec,
        },
        'not_run': ['any training or optimizer update',
                    'any optimizer construction or state load',
                    'any write inside the training run directory'],
    }

    parent = os.path.dirname(output)
    if parent and not os.path.isdir(parent):
        if run_dir is not None and parent.startswith(run_dir + os.sep):
            raise SystemExit('refusing to create %s inside --run_dir %s' % (parent, run_dir))
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as error:
            raise SystemExit('cannot create the output directory %s: %s' % (parent, error))
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('WROTE %s' % output)
    print('CGCLIP_ATTENTION_SNAPSHOT ' + json.dumps({
        'probe': 'cgclip_attention_snapshot',
        'read_only': True,
        'new_optimizer_updates': 0,
        'arm': ARM,
        'completed_steps': int(payload['completed_steps']),
        'samples': count,
        'pairs': len(pairs),
        'grid_side': side,
        'gate_kept_mean': statistics['gate_kept_mean'],
        'gate_kept_min': statistics['gate_kept_min'],
        'gate_kept_max': statistics['gate_kept_max'],
        'gate_probability_mean': statistics['gate_probability_mean'],
        'gate_all_on_fraction': statistics['gate_all_on_fraction'],
        'gate_all_off_fraction': statistics['gate_all_off_fraction'],
        'native_cls_self_mass_mean': diagnostics['native_cls_self_mass_mean'],
        'native_patch_mass_mean': diagnostics['native_patch_mass_mean'],
        'conditional_cls_self_mass_mean': diagnostics['conditional_cls_self_mass_mean'],
        'conditional_patch_mass_mean': diagnostics['conditional_patch_mass_mean'],
        'conditional_vs_native_projected_cosine_mean':
            diagnostics['conditional_vs_native_projected_cosine_mean'],
        'cls_slot_gate_value': CLS_SLOT_GATE_VALUE,
        'clip_state_digest': clip_digest_after,
        'gate_state_digest': gate_digest_after,
        'device': str(device),
        'torch_version': torch.__version__,
        'scope': SCOPE_NOTE}, sort_keys=True))


if __name__ == '__main__':
    main()
