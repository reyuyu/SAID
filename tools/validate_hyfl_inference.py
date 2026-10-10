"""Real four-GPU preflight against frozen native encoder calls, no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image
from tools.eval_hyfl_native import encode, load_model, MODEL_SHA
from tools.retrieval_bounded import ShardCache, atomic_json, retrieval, sha
from tools.urban1k_retrieval import image_caption_pairs, read_captions
from tools.eval_five_parallel import require_gpu_idle

ROOT = Path(__file__).resolve().parents[1]


def parameter_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def run(checkpoint, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    require_gpu_idle((0, 1, 2, 3))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    from model import longclip
    root = ROOT / 'local_assets/evaluation/Urban1k/Urban1k'
    pairs = image_caption_pairs(str(root))[:16]
    raw = [Path(cp).read_text().splitlines(keepends=True)[0] for _, cp in pairs]
    old_captions = read_captions(pairs)
    assert torch.equal(longclip.tokenize(raw, truncate=True), longclip.tokenize(old_captions, truncate=True))
    rows = [{'image_id': str(i), 'caption_id': str(i), 'positive_image_id': str(i), 'caption': t,
             'image_path': str(Path(ip).relative_to(root))} for i, ((ip, _), t) in enumerate(zip(pairs, raw))]
    images = [(r['image_id'], r['image_path']) for r in rows]
    model, pre = load_model(checkpoint, 'cpu'); before = parameter_sha(model)
    assert sha(checkpoint) == MODEL_SHA
    receipts = []; baseline = None; passed = True
    # Documented tolerances for FP32 batch/GPU variation, fixed before observation.
    tolerance = {'embedding_max_abs': 5e-6, 'correct_query_count_difference': 0}
    for gpu in range(4):
        device = f'cuda:{gpu}'; model.to(device)
        torch.cuda.reset_peak_memory_stats(device); t = time.monotonic()
        # Frozen old native extractor: entire text batch; images in batch64; L2 normalize.
        pixels = torch.stack([pre(Image.open(ip).convert('RGB')) for ip, _ in pairs]).to(device)
        old_im = model.encode_image(pixels).float()
        old_tx = model.encode_text(longclip.tokenize(old_captions, truncate=True).to(device)).float()
        old_im = (old_im / old_im.norm(dim=1, keepdim=True)).cpu()
        old_tx = (old_tx / old_tx.norm(dim=1, keepdim=True)).cpu()
        if baseline is None:
            baseline = (old_im, old_tx)
        for batch in (3, 16):
            cache = ShardCache(output / f'gpu{gpu}_batch{batch}', {'model_sha': MODEL_SHA, 'gpu': gpu, 'batch': batch})
            im, tx = encode(model, pre, rows, images, root, cache, device, batch, batch, workers=0)
            local_error = max(float((im - old_im).abs().max()), float((tx - old_tx).abs().max()))
            gpu_error = max(float((im - baseline[0]).abs().max()), float((tx - baseline[1]).abs().max()))
            ids = [r['image_id'] for r in rows]
            m, detail = retrieval(im, tx, ids, ids, ids, query_chunk=3, gallery_chunk=7)
            ref, ref_detail = retrieval(old_im, old_tx, ids, ids, ids, query_chunk=16, gallery_chunk=16)
            hit_diff = {d: {str(k): [i for i, (a, b) in enumerate(zip(detail[d]['hits'][str(k)], ref_detail[d]['hits'][str(k)]))
                                    if a != b] for k in (1, 5, 10)} for d in ('I2T', 'T2I')}
            # Compare direct whole-matrix torch.topk (frozen legacy rule) per query.
            sims = old_im @ old_tx.T; legacy_diff = {}
            for d, sim in [('I2T', sims), ('T2I', sims.T)]:
                idx = sim.topk(10, dim=1).indices
                legacy_diff[d] = {str(k): [i for i in range(16) if bool((idx[i, :k] == i).any())
                                             != detail[d]['hits'][str(k)][i]] for k in (1, 5, 10)}
            ok = max(local_error, gpu_error) <= tolerance['embedding_max_abs'] and not any(
                v for ds in [hit_diff, legacy_diff] for ks in ds.values() for v in ks.values())
            passed &= ok
            receipts.append({'gpu': gpu, 'batch': batch, 'old_new_max_abs': local_error,
                             'gpu0_baseline_max_abs': gpu_error, 'per_query_hit_differences': hit_diff,
                             'legacy_topk_hit_differences': legacy_diff, 'metrics': m, 'pass': ok})
        receipts[-1].update(elapsed_seconds=time.monotonic() - t,
                            peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
                            peak_cuda_reserved=torch.cuda.max_memory_reserved(device))
        model.cpu(); torch.cuda.empty_cache()
    after = parameter_sha(model); assert before == after and sha(checkpoint) == MODEL_SHA
    result = {'status': 'PASS' if passed else 'FAIL', 'tolerance': tolerance, 'receipts': receipts,
              'state_before_sha256': before, 'state_after_sha256': after, 'checkpoint_sha256': MODEL_SHA,
              'strict_load': '317 keys; zero missing/unexpected',
              'precision': 'FP32 model/input/features; autocast off; native cuDNN TF32 True; matmul TF32 False',
              'context_length': 248, 'preprocess': str(pre), 'urban_raw_strip_tokens_equal': True}
    atomic_json(output / 'GPU_INFERENCE_TEST.json', result, exclusive=True)
    if not passed:
        raise AssertionError('GPU inference equivalence failed; receipt saved')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.checkpoint, a.output)
