"""Read-only bare CLIP worker; verified batched feature shards and streaming R@K."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import resource
import re
import time
import traceback
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tools.retrieval_bounded import ShardCache, atomic_json, digest, retrieval, sha, validate_ids

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = '50634512e226e79d526e269ba1a6ca75d7248f47e75417537f1605ca71ac2a0e'
SOURCES = ('model/longclip.py', 'model/model_longclip.py', 'model/simple_tokenizer.py',
           'model/bpe_simple_vocab_16e6.txt.gz', 'tools/eval_hyfl_native.py', 'tools/retrieval_bounded.py')


def preprocess_identity(pre):
    # Function memory addresses in torchvision Compose.__repr__ are process-local.
    # Keep the actual transform classes/parameters/function name; remove only address.
    return re.sub(r' at 0x[0-9a-fA-F]+>', '>', str(pre))


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_model(checkpoint, device):
    from model import longclip
    if sha(checkpoint) != MODEL_SHA:
        raise ValueError('fixed checkpoint SHA mismatch')
    model, pre = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.float().eval().to(device)
    if model.context_length != 248 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('model context/precision mismatch')
    return model, pre


def rows_and_images(manifest):
    rows = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    images = {}
    for row in rows:
        iid = row['image_id']
        if iid in images and images[iid] != row['image_path']:
            raise ValueError('image ID/path mismatch')
        if not isinstance(row['caption'], str) or not row['caption'].strip():
            raise ValueError('empty/nonstring caption')
        images[iid] = row['image_path']
    validate_ids(list(images), [r['caption_id'] for r in rows], [r['positive_image_id'] for r in rows])
    if len(set(images.values())) != len(images):
        raise ValueError('different IDs alias the same image path')
    return rows, list(images.items())


class Images(Dataset):
    def __init__(self, images, root, pre):
        self.images, self.root, self.pre = images, Path(root), pre
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        path = self.root / self.images[i][1]
        with Image.open(path) as im:
            return self.pre(im.convert('RGB'))


def identity(job, checkpoint, rows, images, pre):
    return {'checkpoint_sha256': sha(checkpoint), 'manifest_sha256': sha(job['manifest']),
            'record_sha256': digest(rows), 'image_ids_paths_sha256': digest(images),
            'image_content_sha256': job['image_content_sha256'],
            'tokenizer': 'model.longclip.tokenize(truncate=True)', 'context_length': 248,
            'preprocess': preprocess_identity(pre),
            'precision': 'FP32 model/input/features; autocast off; native cuDNN TF32 True; matmul TF32 False',
            'backend_flags': {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                              'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
            'torch': torch.__version__, 'normalization': job.get('normalization', 'gpu'),
            'encoder_sources': {p: sha(ROOT / p) for p in SOURCES},
            'image_batch': job.get('image_batch', 64), 'text_batch': job.get('text_batch', 64)}


@torch.inference_mode()
def encode(model, pre, rows, images, root, cache, device, image_batch=64, text_batch=64,
           workers=4, normalization='gpu', timings=None):
    from model import longclip
    def norm(x):
        if normalization == 'cpu_raw_for_legacy':
            return x.float().cpu()
        if normalization == 'cpu':
            x = x.cpu()
        return (x.float() / x.float().norm(dim=-1, keepdim=True)).cpu()
    image_bank, text_bank = [], []
    for kind, records, batch in [('text', rows, text_batch), ('image', images, image_batch)]:
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        kind_started = time.monotonic()
        bank = text_bank if kind == 'text' else image_bank
        loader = None
        if kind == 'image':
            loader = iter(DataLoader(Images(images, root, pre), batch_size=batch,
                                    num_workers=workers, shuffle=False, pin_memory=True))
        for start in range(0, len(records), batch):
            subset = records[start:start + batch]
            ids = [r['caption_id'] for r in subset] if kind == 'text' else [r[0] for r in subset]
            x = cache.read(kind, start, ids)
            pixels = next(loader) if loader is not None else None
            if x is None:
                if kind == 'text':
                    tokens = longclip.tokenize([r['caption'] for r in subset], truncate=True).to(device)
                    x = norm(model.encode_text(tokens))
                else:
                    x = norm(model.encode_image(pixels.to(device)))
                cache.write(kind, start, ids, x)
            bank.append(x)
            if start % (batch * 20) == 0:
                print(f'{kind}: {start + len(subset)}/{len(records)}', flush=True)
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        if timings is not None:
            timings[kind + '_encoding_seconds'] = time.monotonic() - kind_started
    return torch.cat(image_bank), torch.cat(text_bank)


def legacy_metrics(images, texts, job, rows):
    """Frozen legacy ranking rules from the same newly inferred feature banks."""
    if job['name'] == 'COCO':
        from eval.retrieval.coco_retrieval import retrieval_metrics
        raw = retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=512)
        return {d: {str(k): round(raw[f'{prefix}_R{k}'] * (len(images) if d == 'I2T' else len(texts)))
                    for k in (1, 5, 10)} for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}
    # Historical extended evaluator normalizes on GPU, scores on CPU and uses torch.topk.
    # Urban used GPU scoring; preserve that backend for its independent regression.
    if job['name'] == 'Urban-1k':
        images, texts = images.to(job['device']), texts.to(job['device'])
    image_ids = list(dict.fromkeys(r['image_id'] for r in rows))
    iidx = {x: i for i, x in enumerate(image_ids)}
    pos = torch.tensor([iidx[r['positive_image_id']] for r in rows], device=images.device)
    counts = {'I2T': {str(k): 0 for k in (1, 5, 10)}, 'T2I': {str(k): 0 for k in (1, 5, 10)}}
    full_reference = None
    if job['name'] != 'Urban-1k':
        # Independent frozen legacy full-matrix reference ONLY for small historical datasets.
        # Full Flickr never reaches this path and is always streamed.
        if len(images) * len(texts) > 65_000_000:
            raise ValueError('legacy full-reference size guard')
        full_reference = images @ texts.T
    for direction, q, g in [('I2T', images, texts), ('T2I', texts, images)]:
        for start in range(0, len(q), 512):
            scores = (full_reference[start:start + 512] if direction == 'I2T'
                      else full_reference.T[start:start + 512]) if full_reference is not None else q[start:start + 512] @ g.T
            idx = scores.topk(min(10, len(g)), dim=1).indices
            correct = (pos[idx] == torch.arange(start, start + len(idx), device=idx.device)[:, None]
                       if direction == 'I2T' else idx == pos[start:start + len(idx), None])
            for k in (1, 5, 10):
                counts[direction][str(k)] += int(correct[:, :k].any(dim=1).sum())
    return counts


def run(job, checkpoint, output, cache_dir, device):
    started = time.monotonic()
    output = Path(output)
    if output.exists() or output.with_suffix('.failure.json').exists():
        raise FileExistsError(output)
    receipt = {'dataset': job['name'], 'protocol': job['protocol'], 'protocol_status': job['protocol_status'],
               'started_utc': utc(), 'device': device, 'physical_gpu': int(device.split(':')[-1]),
               'checkpoint_sha256': sha(checkpoint), 'status': 'RUNNING'}
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        # Preserve the frozen evaluator's native FP32 backend profile. cuDNN's
        # original default permits TF32 convolution with FP32 weights/outputs.
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        rows, images = rows_and_images(job['manifest'])
        if sha(job['manifest']) != job['manifest_sha256']:
            raise ValueError('manifest changed since preflight')
        model, pre = load_model(checkpoint, device)
        from tools.validate_hyfl_inference import parameter_sha
        receipt['state_sha256_before'] = parameter_sha(model)
        ident = identity(job, checkpoint, rows, images, pre)
        cache = ShardCache(cache_dir, ident)
        timings = {}
        imf, tf = encode(model, pre, rows, images, job['image_root'], cache, device,
                         job.get('image_batch', 64), job.get('text_batch', 64),
                         job.get('workers', 4), job.get('normalization', 'gpu'), timings=timings)
        receipt.update(timings)
        enc_seconds = time.monotonic() - started
        norm_im, norm_tx = imf, tf
        if job.get('normalization') == 'cpu_raw_for_legacy':
            norm_im = torch.nn.functional.normalize(imf, dim=-1)
            norm_tx = torch.nn.functional.normalize(tf, dim=-1)
        # Normalized feature banks fit comfortably; only scoring blocks are materialized.
        torch.cuda.synchronize(device)
        scoring_started = time.monotonic()
        metrics, detail = retrieval(norm_im.to(device), norm_tx.to(device), [i[0] for i in images],
                                    [r['caption_id'] for r in rows], [r['positive_image_id'] for r in rows],
                                    query_chunk=job.get('query_chunk', 256),
                                    gallery_chunk=job.get('gallery_chunk', 4096))
        torch.cuda.synchronize(device)
        receipt['scoring_seconds'] = time.monotonic() - scoring_started
        receipt['state_sha256_after'] = parameter_sha(model)
        if receipt['state_sha256_before'] != receipt['state_sha256_after']:
            raise ValueError('model state changed during inference')
        torch.save(detail, output.with_suffix('.queries.pt'))
        receipt.update(metrics=metrics, identity=ident, n_images=len(images), n_captions=len(rows),
                       image_batch=job.get('image_batch', 64), text_batch=job.get('text_batch', 64),
                       query_chunk=job.get('query_chunk', 256), gallery_chunk=job.get('gallery_chunk', 4096),
                       tie_rule='exact ties: ascending manifest index; near ties: FP32 score without rounding',
                       encoding_seconds=enc_seconds)
        if job.get('legacy_expected') or job['name'] == 'COCO':
            job = dict(job, device=device)
            counts = legacy_metrics(imf, tf, job, rows)
            if job['name'] == 'COCO':
                # Keep the user's original canonical COCO numeric rule as primary.
                # The generic streaming GPU output is a diagnostic, never selected
                # as primary according to benchmark performance.
                receipt['streaming_gpu_diagnostic_metrics'] = receipt['metrics']
                receipt['metrics'] = {d: {'query_count': len(imf) if d == 'I2T' else len(tf),
                                         'candidate_count': len(tf) if d == 'I2T' else len(imf),
                                         'correct': counts[d], 'recall_percent': {
                                             k: 100 * v / (len(imf) if d == 'I2T' else len(tf))
                                             for k, v in counts[d].items()}} for d in ('I2T', 'T2I')}
                receipt['ranking_protocol'] = 'original canonical COCO CPU FP32 chunk512/1D argsort'
            if not job.get('legacy_expected'):
                if sha(checkpoint) != MODEL_SHA:
                    raise ValueError('checkpoint changed during evaluation')
                receipt.update(status='COMPLETED', returncode=0)
                return
            r1_pass = all(counts[d]['1'] == job['legacy_expected'][d]['1'] for d in ('I2T', 'T2I'))
            receipt['legacy_regression'] = {'actual_counts': counts, 'expected_counts': job['legacy_expected'],
                                            'all_counts_match': counts == job['legacy_expected'], 'R1_pass': r1_pass,
                                            'R5_R10_query_deltas': {d: {str(k): counts[d][str(k)] - job['legacy_expected'][d][str(k)]
                                                                        for k in (5, 10)} for d in ('I2T', 'T2I')}}
            # Save mismatch evidence before declaring task failure.
            # User's §5.3 requires exact R1 counts and an explicit R5/R10 comparison.
            # Disclose ALL other differences; never label them exact regression PASS.
            if not r1_pass:
                raise ValueError('legacy R1 correct-query counts differ; see receipt')
        if sha(checkpoint) != MODEL_SHA:
            raise ValueError('checkpoint changed during evaluation')
        receipt.update(status='COMPLETED', returncode=0)
    except Exception as error:
        receipt.update(status='FAILED', returncode=1, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                       cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
                       peak_cuda_reserved=torch.cuda.max_memory_reserved(device))
        atomic_json(output if receipt['status'] == 'COMPLETED' else output.with_suffix('.failure.json'),
                    receipt, exclusive=True)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--cache-dir', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    run(json.loads(Path(a.job).read_text()), a.checkpoint, a.output, a.cache_dir, a.device)


if __name__ == '__main__':
    main()
