"""Official CE CLS/EOS encoders on the unchanged frozen SAID protocols."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .native_adapter import BetaCLIPNativeAdapter, EXP, RUN, reconstruct, sha256

ROOT = EXP.parents[2]
ASSETS = Path('/root/lk_projects/SAID-assets')
BENCH = ASSETS / 'retrieval_benchmarks'
EXTENDED = ROOT / 'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'
LONG_SHA = '8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def extended_metric():
    # Execute only the unchanged frozen metric; do not import SAID's encoders.
    node = next(n for n in ast.parse(EXTENDED.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'metric')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(EXTENDED), 'exec'), namespace)
    return namespace['metric']


class Images(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


@torch.no_grad()
def encode(adapter, transform, paths, captions, device, batch, urban=False):
    # Raw FP32 features for the frozen COCO evaluator, which normalizes itself.
    ims = []
    loader = DataLoader(Images(paths, transform), batch_size=batch, shuffle=False,
                        num_workers=4, pin_memory=True)
    for index, images in enumerate(loader):
        ims.append(adapter.image_raw(images.to(device, non_blocking=True)).float().cpu())
        if index % 40 == 0:
            print(json.dumps({'phase': 'images', 'done': min((index+1)*batch, len(paths)),
                              'total': len(paths)}), flush=True)
    text = []
    # The frozen Urban evaluator encodes all 1000 captions in one GPU batch.
    text_batch = len(captions) if urban else batch
    for start in range(0, len(captions), text_batch):
        tokens = adapter.tokenize(captions[start:start+text_batch]).to(device)
        text.append(adapter.text_raw(tokens).float().cpu())
        if start % (batch*40) == 0:
            print(json.dumps({'phase': 'texts', 'done': min(start+text_batch, len(captions)),
                              'total': len(captions)}), flush=True)
    return torch.cat(ims), torch.cat(text)


def standardize_extended(metrics):
    return {direction: {key: value[direction] for key, value in metrics.items()}
            for direction in ('I2T', 'T2I')}


def coco_membership():
    ann = ASSETS / 'evaluation/coco/annotations/captions_val2017.json'
    payload = json.loads(ann.read_text())
    imgs = {r['id']: r for r in payload['images']}
    captions_by_id = {}
    for row in payload['annotations']:
        captions_by_id.setdefault(row['image_id'], []).append(row['caption'])
    ids = sorted(captions_by_id)  # torchvision.CocoCaptions: sorted(coco.imgs.keys()).
    assert ids == sorted(imgs) and len(ids) == 5000
    captions = [caption for i in ids for caption in captions_by_id[i][:5]]
    assert all(len(captions_by_id[i]) >= 5 for i in ids) and len(captions) == 25000
    paths = [ASSETS / 'evaluation/coco/val2017' / imgs[i]['file_name'] for i in ids]
    return paths, captions, {'annotation': str(ann), 'annotation_sha256': sha256(ann),
        'sorted_image_ids_sha256': digest_json(ids), 'caption_strings_sha256': digest_json(captions),
        'membership': 'sorted image IDs, first 5 captions in source annotation order',
        'similarity_chunk': 512, 'positive': '5 captions/image; unique owner image/caption'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', default=str(RUN / 'ce-native'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the untouched official evaluator's runtime default: Conv2d TF32
    # enabled, matmul TF32 disabled. This affects the vision patch embedding.
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    sanity = json.loads((RUN / 'ce-official-urban/result.json').read_text())
    assert sanity['consistency']['passed'] and sanity['reproduction_by_path']['cls'] == 'REPRODUCED'
    model, tokenizer, transform, metadata = reconstruct(args.checkpoint, args.device)
    assert metadata['identity'] == 'CE' and metadata['checkpoint_sha256'] == sanity['checkpoint']['checkpoint_sha256']
    adapter = BetaCLIPNativeAdapter(model, tokenizer)
    conditioner_calls = []
    def forbid_conditioner(module, inputs):
        conditioner_calls.append(True)
        raise AssertionError('Query-dependent conditioner must never execute during native evaluation')
    hook = model.text_conditioned_patches_block.register_forward_pre_hook(forbid_conditioner)
    from tools.urban1k_retrieval import image_caption_pairs, read_captions, _recall
    from eval.retrieval.coco_retrieval import retrieval_metrics
    metric = extended_metric()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    specs = [('Urban-1k', None, None, (1000, 1000)), ('COCO', None, None, (5000, 25000)),
        ('Flickr30k-test1k', 'flickr30k_test1k.jsonl', 'flickr30k', (1000, 5000)),
        ('DOCCI', 'docci_test.jsonl', 'docci', (5000, 5000)),
        ('Long-DCI', 'long_dci_reconstructed.jsonl', 'dci', (7602, 7602))]
    for name, manifest_name, image_folder, counts in specs:
        start = time.time()
        print(json.dumps({'dataset': name, 'phase': 'start'}), flush=True)
        if name == 'Urban-1k':
            pairs = image_caption_pairs(str(ASSETS / 'evaluation/Urban1k/Urban1k'))
            paths = [Path(p[0]) for p in pairs]
            captions = read_captions(pairs)
            protocol = {'positive': 'diagonal', 'caption_strings_sha256': digest_json(captions),
                        'metric_source': 'tools/urban1k_retrieval.py:_recall',
                        'metric_source_sha256': sha256(ROOT / 'tools/urban1k_retrieval.py'),
                        'similarity_device': args.device, 'text_encoding_batch': 1000}
        elif name == 'COCO':
            paths, captions, protocol = coco_membership()
            protocol.update(metric_source='eval/retrieval/coco_retrieval.py:retrieval_metrics',
                metric_source_sha256=sha256(ROOT / 'eval/retrieval/coco_retrieval.py'), similarity_device='cpu')
        else:
            manifest = BENCH / 'manifests' / manifest_name
            manifest_hash = sha256(manifest)
            if name == 'Long-DCI':
                assert manifest_hash == LONG_SHA
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            images, seen = [], set()
            for row in rows:
                if row['image_id'] not in seen:
                    seen.add(row['image_id'])
                    images.append((row['image_id'], row['image_path']))
            paths = [BENCH / image_folder / 'images' / p for _, p in images]
            captions = [r['caption'] for r in rows]
            ids = [r[0] for r in images]
            positives = [r['positive_image_id'] for r in rows]
            assert set(map(str, positives)) <= set(map(str, ids))
            protocol = {'manifest': str(manifest), 'manifest_sha256': manifest_hash,
                'caption_strings_sha256': digest_json(captions),
                'metric_source': str(EXTENDED.relative_to(ROOT)) + ':metric (unchanged AST extraction)',
                'metric_source_sha256': sha256(EXTENDED), 'positive': 'positive_image_id in frozen manifest',
                'similarity_device': 'cpu', 'image_order': 'first occurrence', 'caption_order': 'manifest row order'}
        assert (len(paths), len(captions)) == counts, (name, len(paths), len(captions))
        im_raw, tx_raw = encode(adapter, transform, paths, captions, args.device, args.batch_size,
                                urban=name == 'Urban-1k')
        if name == 'Urban-1k':
            im = im_raw.to(args.device)
            tx = tx_raw.to(args.device)
            im = im / im.norm(dim=-1, keepdim=True)
            tx = tx / tx.norm(dim=-1, keepdim=True)
            metrics = {direction: {f'R@{k}': result[f'R{k}'] for k in (1, 5, 10)}
                for direction, result in [('I2T', _recall(im @ tx.T)), ('T2I', _recall(tx @ im.T))]}
            print(json.dumps({'urban_gate_metrics': metrics, 'official': sanity['adapter_native_cls']}), flush=True)
            if any(abs(metrics[d]['R@1'] - sanity['adapter_native_cls'][d+'_R1']) >= 1e-6 for d in ('I2T', 'T2I')):
                torch.save({'image_raw': im_raw, 'text_raw': tx_raw}, output / 'urban-gate-debug.pt')
            for direction in ('I2T', 'T2I'):
                assert abs(metrics[direction]['R@1'] - sanity['adapter_native_cls'][direction+'_R1']) < 1e-6
            protocol['official_cls_R1_consistency_passed'] = True
            del im, tx
        elif name == 'COCO':
            raw = retrieval_metrics(im_raw, tx_raw, captions_per_image=5, similarity_chunk=512)
            metrics = {direction: {f'R@{k}': raw[f'{prefix}_R{k}'] for k in (1, 5, 10)}
                       for direction, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}
        else:
            im = F.normalize(im_raw.float(), dim=-1)
            tx = F.normalize(tx_raw.float(), dim=-1)
            metrics = standardize_extended(metric(im @ tx.T, ids, positives))
            del im, tx
        result = {'dataset': name, 'n_images': len(paths), 'n_captions': len(captions),
            'metrics': metrics, 'protocol': protocol, 'elapsed_seconds': time.time()-start,
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'native_cls_only': True}
        datasets[name] = result
        (output / (name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        del im_raw, tx_raw
    assert not conditioner_calls
    hook.remove()
    names = ['COCO', 'Urban-1k', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI']
    def score(selected):
        values = [datasets[n]['metrics'][d]['R@1'] for n in selected for d in ('I2T', 'T2I')]
        return sum(values)/len(values)
    result = {'status': 'COMPLETE', 'variant': 'CE', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': metadata, 'datasets': datasets, 'metrics_unit': 'fraction',
        'scores': {'Score5_R1': score(names), 'J_long3': score(['Urban-1k', 'DOCCI', 'Long-DCI']),
                   'J_long': score(['Urban-1k', 'DOCCI'])},
        'inference': 'query-independent official image-block CLS and text EOS; normalized plain inner product',
        'official_tokenizer': 'tokenizer.SimpleTokenizer(context_length=248)',
        'image_preprocess': 'Resize 224 bicubic, CenterCrop 224, RGB, ToTensor, OpenAI CLIP mean/std',
        'precision': 'FP32; matmul TF32 disabled; cuDNN convolution TF32 enabled (official runtime default); no autocast',
        'conditioner_forward_calls': 0,
        'no_training_or_finetuning': True, 'no_reranking_or_ensemble': True}
    (output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'COMPLETE', 'scores_percent': {k:v*100 for k,v in result['scores'].items()}}), flush=True)


if __name__ == '__main__':
    main()
