"""Independent frozen DOCCI encoder/metric reproduction after an observed R5 delta."""
import argparse
import json
from pathlib import Path
import time
import torch
from PIL import Image
from model import longclip
from tools.eval_hyfl_native import load_model, rows_and_images
from tools.retrieval_bounded import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(checkpoint, source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = sha(checkpoint); started = time.monotonic(); torch.set_num_threads(4)
    job = json.loads((source / 'DOCCI/attempt1/JOB.json').read_text())
    receipt = json.loads((source / 'DOCCI/attempt1/RESULT.failure.json').read_text())
    rows, images = rows_and_images(job['manifest']); model, pre = load_model(checkpoint, 'cuda:0')
    flags = {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
             'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
             'grad_mode': 'torch.no_grad (frozen old evaluator)', 'batch': 64}
    # These loops are the exact encoder order/math from frozen eval_extended_real.py.
    texts = []
    for start in range(0, len(rows), 64):
        tok = longclip.tokenize([r['caption'] for r in rows[start:start + 64]], truncate=True).to('cuda:0')
        texts.append(torch.nn.functional.normalize(model.encode_text(tok).float(), dim=-1).cpu())
    tf = torch.cat(texts); imgs = []
    for start in range(0, len(images), 64):
        pixels = torch.stack([pre(Image.open(Path(job['image_root']) / p).convert('RGB'))
                              for _, p in images[start:start + 64]]).to('cuda:0')
        imgs.append(torch.nn.functional.normalize(model.encode_image(pixels).float(), dim=-1).cpu())
    imf = torch.cat(imgs)
    old_im = torch.cat([torch.load(p, weights_only=True) for p in sorted((source / 'cache/DOCCI').glob('image_*.pt'))])
    old_tx = torch.cat([torch.load(p, weights_only=True) for p in sorted((source / 'cache/DOCCI').glob('text_*.pt'))])
    assert imf.shape == old_im.shape and tf.shape == old_tx.shape
    sim = imf @ tf.T; new_sim = old_im @ old_tx.T
    counts = {}; differences = {}
    ids = [i[0] for i in images]; cids = [r['caption_id'] for r in rows]
    for direction, scores, other in [('I2T', sim, new_sim), ('T2I', sim.T, new_sim.T)]:
        top = scores.topk(11, dim=1).indices; new_top = other.topk(11, dim=1).indices
        truth = torch.arange(len(scores))[:, None]; counts[direction] = {}; differences[direction] = {}
        for k in (1, 5, 10):
            hit = (top[:, :k] == truth).any(1); new_hit = (new_top[:, :k] == truth).any(1)
            counts[direction][str(k)] = int(hit.sum())
            diff = (hit != new_hit).nonzero().flatten().tolist()
            differences[direction][str(k)] = [{
                'query_index': i, 'query_id': (ids if direction == 'I2T' else cids)[i],
                'legacy_hit': bool(hit[i]), 'new_hit': bool(new_hit[i]),
                'legacy_top11': top[i].tolist(), 'new_top11': new_top[i].tolist(),
                'legacy_scores': scores[i, top[i]].tolist(), 'new_scores': other[i, new_top[i]].tolist(),
                'ground_truth_score_legacy': float(scores[i, i]), 'ground_truth_score_new': float(other[i, i]),
                'maximum_row_score_difference': float((scores[i] - other[i]).abs().max())} for i in diff]
    result = {'checkpoint_sha256': before, 'flags': flags, 'expected_counts': receipt['legacy_regression']['expected_counts'],
              'counts': counts, 'historical_exact_counts_match': counts == receipt['legacy_regression']['expected_counts'],
              'image_max_abs': float((imf - old_im).abs().max()), 'text_max_abs': float((tf - old_tx).abs().max()),
              'similarity_max_abs': float((sim - new_sim).abs().max()), 'query_differences': differences,
              'elapsed_seconds': time.monotonic() - started,
              'frozen_encoder_source_sha256': sha(ROOT / 'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py')}
    torch.save({'images': imf, 'texts': tf}, output / 'FROZEN_LEGACY_FEATURES.pt')
    assert sha(checkpoint) == before
    atomic_json(output / 'DOCCI_NUMERICAL_DIAGNOSIS.json', result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.checkpoint, a.source, a.output)
