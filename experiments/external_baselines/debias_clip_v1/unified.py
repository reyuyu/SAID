"""Frozen SAID metrics/data with official DeBias independent native encoders."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import (ASSETS, BENCH, CHECKPOINT, LONG_SHA, RUN, DeBiasCLIPNativeAdapter,
                     digest, dump, reconstruct, sha256)
from experiments.external_baselines.beta_clip_v1.evaluate_native import Images, coco_membership, extended_metric
from tools.urban1k_retrieval import image_caption_pairs, read_captions, _recall
from eval.retrieval.coco_retrieval import retrieval_metrics


@torch.no_grad()
def encode(adapter, transform, paths, captions, device):
    # Keep the official evaluator's encoder batch sizes. Candidate pools and
    # metric math are SAID's frozen definitions, independent of encoder batching.
    loader = DataLoader(Images(paths, transform), batch_size=1, num_workers=4,
                        pin_memory=True, shuffle=False)
    ims = []
    for index, images in enumerate(loader):
        ims.append(adapter.encode_image_native(images.to(device)).float().cpu())
        if index % 1000 == 0:
            print(json.dumps({'phase': 'images', 'done': index+1, 'total': len(paths)}), flush=True)
    texts = []
    for start in range(0, len(captions), 16):
        texts.append(adapter.encode_text_native(captions[start:start+16]).float().cpu())
        if start % 4000 == 0:
            print(json.dumps({'phase': 'texts', 'done': min(start+16, len(captions)), 'total': len(captions)}), flush=True)
    return torch.cat(ims), torch.cat(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    primary = json.loads((RUN/'base-full-amp/results.json').read_text())
    assert primary['status'] == 'COMPLETE' and len(primary['datasets']) == 5
    fp32_reference = json.loads((RUN/'urban-fp32/results.json').read_text())
    assert fp32_reference['status'] == 'COMPLETE'
    model, tokenizer, transform, meta = reconstruct(device=args.device, precision='fp32')
    adapter = DeBiasCLIPNativeAdapter(model, tokenizer)
    pca_calls = []
    def forbid_pca(*a, **kw):
        pca_calls.append(True)
        raise AssertionError('Training PCA must not execute in native retrieval')
    model.visual.PCA = forbid_pca
    metric = extended_metric()
    destination = RUN/'unified'
    destination.mkdir(exist_ok=True)
    datasets = {}
    specs = [('Urban-1k', None, None, (1000, 1000)), ('COCO', None, None, (5000, 25000)),
             ('Flickr30k-test1k', 'flickr30k_test1k.jsonl', 'flickr30k', (1000, 5000)),
             ('DOCCI', 'docci_test.jsonl', 'docci', (5000, 5000)),
             ('Long-DCI', 'long_dci_reconstructed.jsonl', 'dci', (7602, 7602))]
    with torch.no_grad():
        for name, manifest_name, folder, counts in specs:
            started = time.time()
            print(json.dumps({'dataset': name, 'phase': 'start'}), flush=True)
            if name == 'Urban-1k':
                pairs = image_caption_pairs(str(ASSETS/'evaluation/Urban1k/Urban1k'))
                paths = [Path(x[0]) for x in pairs]
                captions = read_captions(pairs)
                protocol = {'positive': 'diagonal', 'caption_strings_sha256': digest(captions),
                            'metric': 'tools.urban1k_retrieval._recall'}
            elif name == 'COCO':
                paths, captions, protocol = coco_membership()
                protocol['metric'] = 'eval.retrieval.coco_retrieval.retrieval_metrics'
            else:
                manifest = BENCH/'manifests'/manifest_name
                h = sha256(manifest)
                if name == 'Long-DCI':
                    assert h == LONG_SHA
                rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
                images, seen = [], set()
                for row in rows:
                    if row['image_id'] not in seen:
                        seen.add(row['image_id'])
                        images.append((row['image_id'], row['image_path']))
                paths = [BENCH/folder/'images'/p for _, p in images]
                captions = [r['caption'] for r in rows]
                ids = [i for i, _ in images]
                positives = [r['positive_image_id'] for r in rows]
                protocol = {'manifest': str(manifest), 'manifest_sha256': h,
                            'caption_strings_sha256': digest(captions),
                            'positive': 'positive_image_id from frozen manifest',
                            'metric': 'unchanged eval_extended_real.metric via AST extraction'}
            assert (len(paths), len(captions)) == counts
            if name == 'Urban-1k':
                # Use only indices from the official reference to match encoder
                # batch composition, including its final 8-caption batch. Raw
                # paths/captions still come from the frozen SAID lists; restore
                # the exact frozen candidate order before computing any metric.
                reference = torch.load(RUN/'urban-fp32/urban-embeddings.pt', weights_only=False)
                frozen_indices = {p.name: i for i, p in enumerate(paths)}
                grouping = [frozen_indices[Path(r['image']).name] for r in reference['raw_rows']]
                assert sorted(grouping) == list(range(1000))
                assert all(captions[i] == row['caption'] for i, row in zip(grouping, reference['raw_rows']))
                im, tx = encode(adapter, transform, [paths[i] for i in grouping],
                                [captions[i] for i in grouping], args.device)
                restore = torch.argsort(torch.tensor(grouping))
                im, tx = im[restore], tx[restore]
                protocol['encoding_batch_composition'] = 'Official reference batches, then restore unchanged SAID frozen candidate order; no raw text/membership change'
                del reference
            else:
                im, tx = encode(adapter, transform, paths, captions, args.device)
            if name == 'COCO':
                raw = retrieval_metrics(im, tx, captions_per_image=5, similarity_chunk=512)
                metrics = {d: {f'R@{k}': raw[f'{prefix}_R{k}'] for k in (1, 5, 10)}
                           for d, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]}
            else:
                im = F.normalize(im.float(), dim=-1)
                tx = F.normalize(tx.float(), dim=-1)
                if name == 'Urban-1k':
                    # The frozen metric uses full GPU matrices with FP32 scores.
                    torch.backends.cuda.matmul.allow_tf32 = False
                    sim_i = im.to(args.device) @ tx.to(args.device).T
                    sim_t = tx.to(args.device) @ im.to(args.device).T
                    metrics = {d: {f'R@{k}': values[f'R{k}'] for k in (1, 5, 10)}
                        for d, values in [('I2T', _recall(sim_i)), ('T2I', _recall(sim_t))]}
                    torch.backends.cuda.matmul.allow_tf32 = True
                    reference = torch.load(RUN/'urban-fp32/urban-embeddings.pt', weights_only=False)
                    off_rows = reference['raw_rows']
                    indices = {Path(r['image']).name: i for i, r in enumerate(off_rows)}
                    order = [indices[p.name] for p in paths]
                    reference_im = reference['image_features'][order].float()
                    reference_tx = reference['text_features'][order].float()
                    errors = {'image_max_abs': float((reference_im-im).abs().max()),
                              'text_max_abs': float((reference_tx-tx).abs().max())}
                    ref_metrics = fp32_reference['datasets']['Urban1k']['metrics']
                    gate = {'errors': errors, 'frozen_metrics': metrics,
                            'official_fp32_metrics': ref_metrics,
                            'official_default_amp_metrics': primary['datasets']['Urban1k']['metrics'],
                            'precision_policy': 'Phase A keeps official default AMP; Phase B native FP32 is fixed before scores. Official FP32 Urban is a separate adapter consistency diagnostic, not chosen as the primary reproduction result.'}
                    dump(destination/'urban-gate.json', gate)
                    assert max(errors.values()) < 1e-4, errors
                    for d in ('I2T', 'T2I'):
                        assert abs(metrics[d]['R@1']-ref_metrics[d]['R@1']) < 1e-6, gate
                    torch.save({'image_features': im, 'text_features': tx,
                                'image_basenames': [p.name for p in paths],
                                'similarity_i2t': sim_i.cpu(), 'similarity_t2i': sim_t.cpu()},
                               destination/'urban-embeddings.pt')
                    del sim_i, sim_t, reference
                    protocol['adapter_consistency_gate_passed'] = True
                else:
                    raw = metric(im @ tx.T, ids, positives)
                    metrics = {d: {key: value[d] for key, value in raw.items()} for d in ('I2T', 'T2I')}
            record = {'dataset': name, 'n_images': len(paths), 'n_captions': len(captions),
                      'metrics': metrics, 'protocol': protocol, 'elapsed_seconds': time.time()-started}
            datasets[name] = record
            dump(destination/(name+'.json'), record)
            print(json.dumps(record), flush=True)
            del im, tx
    assert not pca_calls
    def aggregate(names):
        values = [datasets[n]['metrics'][d]['R@1'] for n in names for d in ('I2T', 'T2I')]
        return sum(values)/len(values)
    scores = {'Score5_R1': aggregate([s[0] for s in specs]),
              'J_long3': aggregate(['Urban-1k', 'DOCCI', 'Long-DCI']),
              'J_long': aggregate(['Urban-1k', 'DOCCI'])}
    result = {'status': 'COMPLETE', 'checkpoint': meta, 'datasets': datasets, 'scores': scores,
              'units': 'raw fraction', 'precision': 'FP32 native encoders, official runtime flags; frozen FP32 metric scoring',
              'encoder_batches': {'image': 1, 'text': 16}, 'pca_calls': 0,
              'no_training_or_augmentation_or_reranking': True, 'official_default_amp_kept_for_phase_a': True}
    dump(destination/'results.json', result)
    print(json.dumps({'status': 'COMPLETE', 'scores_percent': {k: v*100 for k, v in scores.items()}}), flush=True)


if __name__ == '__main__':
    main()
