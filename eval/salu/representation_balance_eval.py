"""Offline, fixed-cohort representation diagnostics; one GPU model at a time.

Run with ``python -m eval.salu.representation_balance_eval --help``.
Training and the artifact-only dashboard never import this exporter.
"""
import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from model import longclip
from model.salu_model import SALUModel
from .representation_probe import DETAIL_LEVELS, file_sha256, load_or_create_manifest, write_json
from .representation_balance_metrics import cyclic_indices, gap_comparison, joint_pca, modality_metrics

CHECKPOINTS = {'initial': 'salu_initial.pt', 'step100': 'salu_said_only_step000100.pt',
               'step200': 'salu_said_only_step000200.pt',
               'step400': 'salu_said_only_step000400.pt', 'final': 'salu_said_only_last.pt'}


def load_model(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    args = payload.get('args', {})
    if args.get('said_feature_source', 'residual') != 'residual':
        raise ValueError('representation balance production probe requires residual source')
    clip, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu')
    model = SALUModel(clip, tau_said=args.get('tau_said', .07),
                      said_loss_mode=args.get('said_loss_mode', 'positive'),
                      said_feature_source='residual')
    model.load_state_dict(payload['model'], strict=True)
    metadata = {'step': int(payload.get('step', 0)), 'epoch': payload.get('epoch', 0),
                'said_feature_source': 'residual', 'strict_load': True,
                'said_loss_mode': model.said_loss_mode}
    return model.float().eval().to(device), preprocess, metadata


def token_metadata(caption):
    count = len(longclip._tokenizer.encode(caption)) + 2
    return {'original_tokens_including_special': count, 'context_length': 248,
            'used_tokens_including_special': min(count, 248), 'truncated': count > 248}


@torch.inference_mode()
def infer(model, preprocess, samples, image_root, device, batch_size):
    images, patches = [], []
    equivalence = 0.
    for start in range(0, len(samples), batch_size):
        tensors = []
        for sample in samples[start:start + batch_size]:
            with Image.open(image_root / sample['image_id']) as image:
                tensors.append(preprocess(image.convert('RGB')))
        tensor = torch.stack(tensors).to(device)
        full, patch = model.encode_router_input(tensor)
        if start == 0:
            standard = model.encode_image(tensor)
            equivalence = float((standard - full).abs().max())
            if not torch.allclose(standard, full, atol=1e-6, rtol=1e-6):
                raise RuntimeError('z_full does not match standard encode_image')
        images.append(F.normalize(full.float(), dim=-1).cpu())
        patches.append(patch.float().cpu())
    full, patches = torch.cat(images), torch.cat(patches)

    def encode(captions):
        out = []
        for start in range(0, len(captions), batch_size):
            tokens = longclip.tokenize(captions[start:start + batch_size], truncate=True).to(device)
            out.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
        return torch.cat(out)

    def route(texts, image_indices):
        out = []
        for start in range(0, len(texts), batch_size):
            _, said = model.said_router(texts[start:start + batch_size].to(device),
                                        patches[image_indices[start:start + batch_size]].to(device))
            out.append(F.normalize(said.float(), dim=-1).cpu())
        return torch.cat(out)

    primary = [s['training_caption']['caption'] for s in samples]
    text = encode(primary)
    own = route(text, np.arange(len(samples)))
    shuffled = route(text[cyclic_indices(len(samples))], np.arange(len(samples)))
    captions, indices, mapping = [], [], []
    token_rows = []
    for index, sample in enumerate(samples):
        ladder = sample['caption_detail']
        offset = len(captions)
        for variant in ladder['variants']:
            captions.append(variant['caption'])
            indices.append(index)
        mapping.append([offset + ladder['levels'][level] for level in DETAIL_LEVELS])
        token_rows.append({'order': index, 'primary': token_metadata(primary[index]),
                           'variants': [token_metadata(v['caption']) for v in ladder['variants']]})
    detail_text = encode(captions)
    detail_said = route(detail_text, np.asarray(indices))
    mapping = np.asarray(mapping).T
    return {'z_full': full.numpy(), 'z_said': own.numpy(), 't': text.numpy(),
            'z_said_shuffle': shuffled.numpy(),
            'detail_t': detail_text[mapping].numpy(),
            'detail_said': detail_said[mapping].numpy()}, token_rows, equivalence


def detail_metrics(embeddings, samples, token_rows):
    rows = []
    for index, level in enumerate(DETAIL_LEVELS):
        target = embeddings['detail_t'][index]
        base, full, said = [modality_metrics(v, target) for v in
                           (embeddings['z_base'], embeddings['z_full'], embeddings['detail_said'][index])]
        tokens = [row['variants'][sample['caption_detail']['levels'][level]]
                  for row, sample in zip(token_rows, samples)]
        rows.append({'level': level, 'n': len(samples), 'base': base, 'full': full, 'said': said,
                     'balancing_gain': full['pair_gap'] - said['pair_gap'],
                     'mean_used_tokens': float(np.mean([t['used_tokens_including_special'] for t in tokens])),
                     'truncated_count': sum(t['truncated'] for t in tokens)})
    return {'levels': rows, 'tokens': token_rows, 'unique_variant_count':
            sum(len(s['caption_detail']['variants']) for s in samples),
            'note': '文本信息覆盖代理；长度不等于真实语义覆盖率。重复 level 共享一次推理，但各曲线保留同一组 N 张图像。'}


def read_log(path):
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_json', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--checkpoints_root', type=Path, default=Path('runs_salu/phase25/residual'))
    parser.add_argument('--output', type=Path, default=Path('outputs/representation_balance'))
    parser.add_argument('--batch_log', type=Path)
    parser.add_argument('--coco_artifact', type=Path)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(root / 'manifest.json', args.dataset_json, args.n, args.seed)
    manifest_sha = file_sha256(root / 'manifest.json')
    initial_sha = file_sha256(args.checkpoints_root / CHECKPOINTS['initial'])
    cache_path = root / 'base_cache.npz'
    base = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache['manifest_sha256']) != manifest_sha or str(cache['checkpoint_sha256']) != initial_sha:
                raise ValueError('base cache provenance mismatch')
            base = cache['z_base'].copy()
    training_log = read_log(args.checkpoints_root / 'salu_log.jsonl')
    batch_records = read_log(args.batch_log)
    write_json(root / 'batch_history.json', {'scope': '训练 Batch 诊断',
               'run': args.batch_log.parent.name if args.batch_log else None,
               'note': '独立 4 卡 × 10-step smoke；不属于下方 Phase 2.5 固定 Probe 训练轨迹。',
               'records': batch_records})
    summary = {'schema_version': 1, 'status': 'running', 'manifest_sha256': manifest_sha,
               'n': args.n, 'source_experiment': 'reviewed Phase 2.5 residual arm',
               'feature_source': 'residual', 'precision': 'fp32; TF32 disabled',
               'base_reference': 'fixed initial image embeddings paired with CURRENT checkpoint text',
               'export_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'checkpoints': {}}
    write_json(root / 'summary.json', summary)
    for tag, filename in CHECKPOINTS.items():
        print('EXPORT ' + tag, flush=True)
        path = args.checkpoints_root / filename
        checkpoint_sha = initial_sha if tag == 'initial' else file_sha256(path)
        model, preprocess, metadata = load_model(path, args.device)
        arrays, token_rows, equivalence = infer(model, preprocess, manifest['samples'],
                                               args.image_root, args.device, args.batch_size)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if base is None:
            if tag != 'initial':
                raise RuntimeError('initial embeddings must be exported first')
            base = arrays['z_full'].copy()
            np.savez_compressed(cache_path, z_base=base, manifest_sha256=manifest_sha,
                                checkpoint_sha256=initial_sha)
        arrays['z_base'] = base
        metrics = gap_comparison(base, arrays['z_full'], arrays['z_said'], arrays['t'], arrays['z_said_shuffle'])
        metrics.update({'checkpoint_sha256': checkpoint_sha, 'manifest_sha256': manifest_sha,
                        'standard_global_max_abs_error': equivalence, **metadata})
        detail = detail_metrics(arrays, manifest['samples'], token_rows)
        pca = joint_pca(base, arrays['z_full'], arrays['z_said'], arrays['t'])
        folder = root / tag
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / 'embeddings.npz', **arrays)
        write_json(folder / 'gap_metrics.json', metrics)
        write_json(folder / 'caption_detail.json', detail)
        write_json(folder / 'pca.json', pca)
        matches = [r for r in training_log if r.get('completed_steps', r['step'] + 1) == metadata['step']]
        entry = {'step': metadata['step'], 'epoch': metadata['epoch'], 'metrics': metrics,
                 'training': matches[-1] if matches else {}, 'coco': None}
        if tag == 'final' and args.coco_artifact and args.coco_artifact.exists():
            coco = json.loads(args.coco_artifact.read_text(encoding='utf-8'))['residual']
            if coco['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('COCO artifact checkpoint SHA does not match')
            entry['coco'] = {**coco, 'source': '复用 Phase 2.5 residual final 评估，checkpoint SHA256 已核对'}
        summary['checkpoints'][tag] = entry
        write_json(root / 'summary.json', summary)
        print(json.dumps({'tag': tag, 'step': metadata['step'], 'metrics': metrics}, ensure_ascii=False), flush=True)
    summary['status'] = 'complete'
    write_json(root / 'summary.json', summary)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
