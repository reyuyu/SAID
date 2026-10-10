"""Validate the frozen production batch64 profile; preserve failed batch3 diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from model import longclip
from tools.eval_five_parallel import require_gpu_idle
from tools.eval_hyfl_native import encode, load_model
from tools.retrieval_bounded import ShardCache, atomic_json, retrieval, sha
from tools.urban1k_retrieval import image_caption_pairs, read_captions
from tools.validate_hyfl_inference import parameter_sha

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(checkpoint, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    require_gpu_idle((0, 1, 2, 3)); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    root = ROOT / 'local_assets/evaluation/Urban1k/Urban1k'
    pairs = image_caption_pairs(str(root))[:64]; captions = read_captions(pairs)
    rows = [{'image_id': str(i), 'caption_id': str(i), 'positive_image_id': str(i),
             'image_path': str(Path(ip).relative_to(root)), 'caption': captions[i]}
            for i, (ip, _) in enumerate(pairs)]
    images = [(r['image_id'], r['image_path']) for r in rows]
    model, pre = load_model(checkpoint, 'cpu'); before = parameter_sha(model)
    baseline = None; records = []; tolerance = 5e-6
    for gpu in range(4):
        device = f'cuda:{gpu}'; model.to(device)
        x = torch.stack([pre(Image.open(ip).convert('RGB')) for ip, _ in pairs]).to(device)
        old_im = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu()
        old_tx = torch.nn.functional.normalize(model.encode_text(longclip.tokenize(captions, truncate=True).to(device)).float(), dim=-1).cpu()
        cache = ShardCache(output / f'gpu{gpu}', {'model_sha': sha(checkpoint), 'batch': 64, 'gpu': gpu})
        im, tx = encode(model, pre, rows, images, root, cache, device, 64, 64, workers=0)
        if baseline is None:
            baseline = (old_im, old_tx)
        error = max(float((im - old_im).abs().max()), float((tx - old_tx).abs().max()),
                    float((im - baseline[0]).abs().max()), float((tx - baseline[1]).abs().max()))
        ids = [r['image_id'] for r in rows]
        _, details = retrieval(im, tx, ids, ids, ids, query_chunk=7, gallery_chunk=13)
        _, reference = retrieval(old_im, old_tx, ids, ids, ids, query_chunk=64, gallery_chunk=64)
        differences = {d: {str(k): [i for i, (a, b) in enumerate(zip(details[d]['hits'][str(k)], reference[d]['hits'][str(k)]))
                                   if a != b] for k in (1, 5, 10)} for d in ['I2T', 'T2I']}
        records.append({'gpu': gpu, 'batch': 64, 'max_abs': error, 'hit_differences': differences,
                        'pass': error <= tolerance and not any(v for ks in differences.values() for v in ks.values())})
        model.cpu(); torch.cuda.empty_cache()
    after = parameter_sha(model); assert after == before
    pass_all = all(r['pass'] for r in records)
    receipt = {'status': 'PASS' if pass_all else 'FAIL', 'tolerance': tolerance, 'records': records,
               'fixed_formal_batch': 64, 'checkpoint_sha256': sha(checkpoint), 'state_sha256_before': before,
               'state_sha256_after': after, 'batch_optimization_3': 'NOT_APPROVED; known cuDNN precision variation exceeds tolerance',
               'older_failed_batch3_receipts_preserved': True,
               'precision': 'Original native FP32 model/features; cuda matmul TF32 False; cuDNN TF32 True; autocast off'}
    atomic_json(output / 'GPU_NATIVE_BATCH64_TEST.json', receipt, exclusive=True)
    if not pass_all:
        raise AssertionError('Formal batch64 equivalence failed; receipt saved')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.checkpoint, a.output)
