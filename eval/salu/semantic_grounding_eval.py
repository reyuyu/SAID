"""Phase 2.3: frozen-model phrase grounding audit; never trains or changes losses.

Run from repository root: python -m eval.salu.semantic_grounding_eval --help
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .flickr_entities import load_split
from .grounding_metrics import (patch_coverage, phrase_metrics, phrase_switch,
                                summarize, union_iou)

VARIANTS = ['initial_direct', 'phase21_router', 'phase22_router', 'phase22_direct']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def direct_attention(patches, texts, tau_eval=.07):
    import torch
    import torch.nn.functional as F
    if not np.isfinite(tau_eval) or tau_eval <= 0:
        raise ValueError('tau_eval must be finite and positive')
    if patches.ndim != 2 or texts.ndim != 2 or patches.shape[-1] != texts.shape[-1]:
        raise ValueError('expected patches [N,D] and phrase texts [P,D]')
    a = torch.softmax(F.normalize(texts.float(), dim=-1) @
                      F.normalize(patches.float(), dim=-1).T/tau_eval, dim=-1)
    if not torch.isfinite(a).all():
        raise ValueError('nonfinite direct attention')
    return a


def native_clip_patches(model, image):
    """Final projected patch tokens from unmodified OpenAI CLIP ViT.

    Same final ln_post/proj convention as the reviewed Phase 1 interface.
    Native CLIP text encoding remains the original 77-position model.
    """
    import torch
    v = model.visual
    x = v.conv1(image.type(model.dtype)).flatten(2).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype)[None, None].expand(len(x), 1, -1)
    x = v.ln_pre(torch.cat([cls, x], dim=1)+v.positional_embedding.to(x.dtype))
    x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    return v.ln_post(x[:, 1:]) @ v.proj


def switching_groups(records, min_distance=.2):
    """Deterministic 2-4 spatially separated entities; never select by scores."""
    groups = []
    for image in records:
        unique = {}
        for p in image['phrases']:
            key = p['entity_id']
            if key not in unique or (len(p['phrase']), p['id']) < (len(unique[key]['phrase']), unique[key]['id']):
                unique[key] = p
        chosen = []
        for p in sorted(unique.values(), key=lambda p: p['id']):
            box = np.asarray(p['boxes'])
            center = .5*(box[:, :2].min(0)+box[:, 2:].max(0))
            def separated(q):
                other = np.asarray(q['boxes'])
                other_center = .5*(other[:, :2].min(0)+other[:, 2:].max(0))
                return union_iou(p['boxes'], q['boxes']) < .1 and np.linalg.norm(center-other_center)/224 >= min_distance
            if all(separated(q) for q in chosen):
                chosen.append(p)
            if len(chosen) == 4:
                break
        if len(chosen) >= 2:
            groups.append({'image_id': image['id'], 'phrase_ids': [p['id'] for p in chosen]})
    return groups


def render_overlay(image, attention, boxes, peak_xy):
    from tools.said_dashboard.data import overlay_rgb
    out = Image.fromarray(overlay_rgb(image, attention))
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle(tuple(box), outline='lime', width=2)
    x, y = peak_xy
    draw.line((x-4, y, x+4, y), fill='red', width=2)
    draw.line((x, y-4, x, y+4), fill='red', width=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max_images', type=int, default=128, help='0 = full official split; run small first')
    parser.add_argument('--image_root', default=os.environ.get('FLICKR30K_ROOT'))
    parser.add_argument('--entities_root', default=os.environ.get('FLICKR30K_ENTITIES_ROOT'))
    parser.add_argument('--phase21_checkpoint', required=True)
    parser.add_argument('--phase22_checkpoint', required=True)
    parser.add_argument('--tau_eval', type=float, default=.07)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='outputs/semantic_grounding')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('max_images must be nonnegative')
    root = Path(args.output_dir)
    if (root/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory; refusing to mix audit runs')
    import torch
    import torch.nn.functional as F
    import clip
    from model import longclip
    from model.salu_model import SALUModel

    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records, exclusions = load_split(args.image_root, args.entities_root, args.split, args.max_images or None)
    groups = switching_groups(records)
    print('DATASET images=%d phrases=%d switching_images=%d exclusions=%s' %
          (len(records), sum(len(r['phrases']) for r in records), len(groups), exclusions), flush=True)
    for folder in ['images', 'attention', 'overlays']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    phrase_rows = {}
    image_rows = []
    for image in records:
        image_rows.append({k: image[k] for k in ['id', 'geometry', 'entities']})
        image_rows[-1]['phrase_ids'] = [p['id'] for p in image['phrases']]
        for p in image['phrases']:
            phrase_rows[p['id']] = {**p, 'image_id': image['id'], 'metrics': {}}
    manifest = {'schema_version': 1, 'status': 'running', 'dataset': 'Flickr30K Entities',
                'split': args.split, 'images': image_rows, 'num_images': len(records),
                'num_phrases': len(phrase_rows), 'variants': VARIANTS, 'grid': 14,
                'image_size': 224, 'tau_eval': args.tau_eval, 'precision': 'fp32, TF32 disabled',
                'phrase_queries_only': True, 'exclusions': exclusions,
                'switching_groups': groups, 'selection': 'official split file order',
                'split_sha256': sha256(Path(args.entities_root)/(args.split+'.txt')),
                'annotation_archive_sha256': sha256(Path(args.entities_root)/'annotations.zip')
                    if (Path(args.entities_root)/'annotations.zip').exists() else None,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'checkpoints': {}, 'geometry': 'Resize short side 224, CenterCrop 224; xyxy half-open',
                'background_definition': 'outside all visible annotated boxes, not a verified background label'}
    for stage, checkpoint in [('phase21', args.phase21_checkpoint), ('phase22', args.phase22_checkpoint)]:
        manifest['checkpoints'][stage] = {'name': Path(checkpoint).name, 'sha256': sha256(checkpoint)}
    write_json(root/'manifest.json', manifest)
    write_json(root/'per_phrase.json', phrase_rows)

    all_pairs, summary = {}, {}
    for stage in ['initial', 'phase21', 'phase22']:
        if stage == 'initial':
            native, preprocess = clip.load('ViT-B/16', device='cpu', jit=False)
            native = native.float().eval().to(args.device)
            tokenizer, backbone, salu = clip.tokenize, native, None
            active = ['initial_direct']
        else:
            checkpoint = getattr(args, stage+'_checkpoint')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = payload.get('args', {})
            mode = 'positive' if stage == 'phase21' else 'identifiable'
            if cfg.get('said_loss_mode', mode) != mode:
                raise ValueError('unexpected checkpoint loss mode for '+stage)
            backbone, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu')
            salu = SALUModel(backbone, tau_said=cfg.get('tau_said', .07), said_loss_mode=mode)
            salu.load_state_dict(payload['model'], strict=True)
            del payload
            salu = salu.float().eval().to(args.device)
            backbone, tokenizer = salu.clip, longclip.tokenize
            manifest['checkpoints'][stage].update({'tau_said': salu.tau_said, 'said_loss_mode': mode})
            active = [stage+'_router'] + (['phase22_direct'] if stage == 'phase22' else [])
        for variant in active:
            (root/'attention'/variant).mkdir(exist_ok=True)
        with torch.inference_mode():
            for idx, image in enumerate(records):
                with Image.open(image['path']) as original:
                    tensor = preprocess(original).unsqueeze(0).to(args.device)
                    g = image['geometry']
                    resized = original.resize(tuple(g['resized_size']), Image.BICUBIC)
                    x, y = g['crop_xy']
                    crop = resized.crop((x, y, x+224, y+224)).convert('RGB')
                if stage == 'initial':
                    crop.save(root/'images'/(image['id']+'.png'))
                    patches = native_clip_patches(backbone, tensor)[0]
                else:
                    patches = backbone.encode_image_with_patches(tensor)[1][0]
                if patches.shape != (196, 512):
                    raise ValueError('expected ViT-B/16 projected patches [196,512]')
                weights = {key: patch_coverage(boxes) for key, boxes in image['entities'].items() if boxes}
                phrases = image['phrases']
                for start in range(0, len(phrases), 64):
                    batch = phrases[start:start+64]
                    # Never pass the containing sentence and never silently truncate a phrase.
                    tokens = tokenizer([p['phrase'] for p in batch], truncate=False).to(args.device)
                    texts = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
                    maps = {}
                    if stage in ['initial', 'phase22']:
                        maps[stage+'_direct'] = direct_attention(patches, texts, args.tau_eval)
                    if salu is not None:
                        maps[stage+'_router'] = salu.said_router(texts, patches[None].expand(len(batch), -1, -1))[0]
                    for variant, attention in maps.items():
                        for p, a in zip(batch, attention.cpu().numpy().reshape(-1, 14, 14)):
                            entity = p['entity_id']
                            others = {k: b for k, b in image['entities'].items() if k != entity and b}
                            stats = phrase_metrics(a, p['boxes'], others, coverage=weights[entity], other_coverage=weights)
                            phrase_rows[p['id']]['metrics'][variant] = stats
                            np.save(root/'attention'/variant/(p['id']+'.npy'), a.astype(np.float32))
                if idx % 25 == 0:
                    print(stage, idx+1, '/', len(records), flush=True)
        for variant in active:
            pairs = []
            for group in groups:
                for aid, bid in itertools.combinations(group['phrase_ids'], 2):
                    pa, pb = phrase_rows[aid], phrase_rows[bid]
                    a = np.load(root/'attention'/variant/(aid+'.npy'))
                    b = np.load(root/'attention'/variant/(bid+'.npy'))
                    pair = phrase_switch(a, b, patch_coverage(pa['boxes']), patch_coverage(pb['boxes']))
                    pair.update({'image_id': group['image_id'], 'phrase_a': aid, 'phrase_b': bid})
                    pair['both_pointing_correct'] = pa['metrics'][variant]['pointing_correct'] and pb['metrics'][variant]['pointing_correct']
                    pairs.append(pair)
            all_pairs[variant] = pairs
            stats = [p['metrics'][variant] for p in phrase_rows.values()]
            summary[variant] = summarize(stats, pairs)
            categories = sorted({c for p in phrase_rows.values() for c in p['categories']})
            summary[variant]['categories'] = {
                cat: summarize([p['metrics'][variant] for p in phrase_rows.values() if cat in p['categories']], [])
                for cat in categories}
            ranked = sorted(phrase_rows, key=lambda key: (phrase_rows[key]['metrics'][variant]['mass_gain'], key))
            summary[variant]['worst50'] = ranked[:50]
            summary[variant]['best50'] = ranked[-50:][::-1]
            print('RESULT', variant, json.dumps({k: summary[variant][k] for k in ['pointing_correct', 'gt_mass', 'mass_gain', 'localization_margin', 'switch_margin']}), flush=True)
        write_json(root/'per_phrase.json', phrase_rows)
        write_json(root/'summary.json', summary)
        write_json(root/'switching.json', all_pairs)
        del backbone, salu
        if stage == 'initial':
            del native
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Persist overlays for all switching cases and per-variant best/worst cases.
    chosen = {p for group in groups[:32] for p in group['phrase_ids']}
    for variant in VARIANTS:
        chosen.update(summary[variant]['best50']+summary[variant]['worst50'])
    for variant in VARIANTS:
        (root/'overlays'/variant).mkdir(exist_ok=True)
        for key in sorted(chosen):
            p = phrase_rows[key]
            with Image.open(root/'images'/(p['image_id']+'.png')) as image:
                a = np.load(root/'attention'/variant/(key+'.npy'))
                render_overlay(image, a, p['boxes'], p['metrics'][variant]['peak_xy']).save(root/'overlays'/variant/(key+'.png'))
    manifest['status'] = 'complete'
    manifest['overlay_note'] = 'Individual exported overlays use their own max; dashboard comparisons use a shared max.'
    manifest['completed_variants'] = VARIANTS
    write_json(root/'manifest.json', manifest)
    print('AUDIT_COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
