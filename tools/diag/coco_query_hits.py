"""Per-query COCO val2017 R@K hit vectors for one student, on the frozen evaluation protocol.

Why this exists
---------------
``eval/retrieval/coco_retrieval.py`` reports only *aggregate* R@K, so two arms can be compared
but not *paired*. COCO val2017 contributes 5,000 I2T queries and 25,000 T2I queries, and the
differences this project argues about (~0.5 pp) are inside the binomial noise of independent
samples (1 SE ~ 0.7 pp at p=0.6, n=5,000). Recording *which individual queries* each arm gets
right turns the comparison into a paired test on the same queries, which is what
``eval/paired_statistics.py`` (``paired_bootstrap``, ``mcnemar_exact``) consumes.

Protocol identity
-----------------
Not reimplemented: the image/text order, the ``captions[:5]`` text build, the 512-row similarity
chunk, the per-row ``argsort`` tie rule (``_top_indices``) and the aggregate R@K function
(``retrieval_metrics``) are imported from the evaluator library itself, and the library file's
sha256 is recorded in the output. The aggregate R@K recomputed here from the same feature tensors
must equal the number the standard evaluator already wrote for the same checkpoint; ``--expect``
turns that into an assertion instead of a hope. Run it only after the standard evaluation, and
never let this file's numbers replace the standard ones.

    python tools/diag/coco_query_hits.py \
        --label S0_smartclip@1000 --checkpoint <student_001000.pt> \
        --expect 0.6058,0.41236 --out <...>/query_hits_S0_smartclip_step001000.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COCO_LIB = os.path.join(REPO, 'eval', 'retrieval', 'coco_retrieval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'
K_VALUES = (1, 5, 10)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hit_vectors(coco_lib, images_n, texts_n, captions_per_image, chunk, k_values):
    """Per-query hits in both directions, mirroring the library's chunking and tie rule.

    ``images_n`` / ``texts_n`` are already L2-normalised exactly as ``retrieval_metrics``
    normalises them, so the two callers cannot disagree about the ranking.
    """
    n_images, n_texts = images_n.shape[0], texts_n.shape[0]
    top_k = max(k_values)
    image_chunk = min(chunk, n_images)
    text_chunk = min(chunk, n_texts)

    i2t = {k: [] for k in k_values}
    for start in range(0, n_images, image_chunk):
        block = images_n[start:start + image_chunk] @ texts_n.t()
        top = coco_lib._top_indices(block, top_k)
        for row in range(block.shape[0]):
            candidates = top[row].tolist()
            truth = range((start + row) * captions_per_image,
                          (start + row + 1) * captions_per_image)
            for k in k_values:
                i2t[k].append(int(any(candidate in candidates[:k] for candidate in truth)))

    t2i = {k: [] for k in k_values}
    for start in range(0, n_texts, text_chunk):
        block = texts_n[start:start + text_chunk] @ images_n.t()
        top = coco_lib._top_indices(block, top_k)
        for row in range(block.shape[0]):
            candidates = top[row].tolist()
            truth = ((start + row) // captions_per_image,)
            for k in k_values:
                t2i[k].append(int(any(candidate in candidates[:k] for candidate in truth)))
    return i2t, t2i


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='bare student or full checkpoint')
    parser.add_argument('--label', required=True, help='e.g. S0_smartclip@1000')
    parser.add_argument('--expect', default=None,
                        help='i2t_r1,t2i_r1 already reported by the standard evaluator; asserted')
    parser.add_argument('--expect-steps', type=int, default=None)
    parser.add_argument('--base-model', default='ViT-B/16')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--coco-root',
                        default=os.path.join(os.environ.get('COCO_DATA_ROOT', '/root/datasets/coco'),
                                             'val2017'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    global coco_lib
    coco_lib = load_module('coco_retrieval_frozen', COCO_LIB)
    # import the model package from THIS worktree before the loader module (which lives in another
    # checkout) prepends its own repository to sys.path; the module actually used is recorded below
    sys.path.insert(0, REPO)
    from model import longclip
    urban = load_module('eval_urban1k_cls', URBAN_EVAL)

    model, preprocess, meta = urban.load_student(args.checkpoint, args.base_model, args.device,
                                                 args.expect_steps)
    device = torch.device(args.device)

    from torchvision.datasets import CocoCaptions
    ann_file = os.path.join(os.path.dirname(os.path.abspath(args.coco_root)),
                            'annotations', 'captions_val2017.json')
    dataset = CocoCaptions(root=args.coco_root, annFile=ann_file, transform=preprocess)
    print('COCO val2017: %d images, annotations %s' % (len(dataset), ann_file), flush=True)

    image_features, text_features, caption_stream = [], [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            stop = min(start + args.batch_size, len(dataset))
            batch = [dataset[index] for index in range(start, stop)]
            images = torch.stack([item[0] for item in batch]).to(device)
            captions = [caption for _, caps in batch for caption in caps[:5]]
            if len(captions) != 5 * len(batch):
                raise SystemExit('image %d-%d has %d individual captions, expected %d'
                                 % (start, stop, len(captions), 5 * len(batch)))
            caption_stream.extend(captions)
            image_features.append(model.encode_image(images).detach().cpu().float())
            tokens = longclip.tokenize(captions, truncate=True).to(device)
            text_features.append(model.encode_text(tokens).detach().cpu().float())

    images = torch.cat(image_features)
    texts = torch.cat(text_features)
    images_n = F.normalize(images, dim=-1)
    texts_n = F.normalize(texts, dim=-1)

    recomputed = coco_lib.retrieval_metrics(images, texts, captions_per_image=5,
                                            similarity_chunk=coco_lib.DEFAULT_SIMILARITY_CHUNK,
                                            k_values=K_VALUES)
    i2t_hits, t2i_hits = hit_vectors(coco_lib, images_n, texts_n, 5,
                                     coco_lib.DEFAULT_SIMILARITY_CHUNK, K_VALUES)
    from_hits = {}
    for k in K_VALUES:
        from_hits['image2text_R%d' % k] = sum(i2t_hits[k]) / float(images_n.shape[0])
        from_hits['text2image_R%d' % k] = sum(t2i_hits[k]) / float(texts_n.shape[0])
    if from_hits != recomputed:
        raise SystemExit('the per-query pass does not reproduce the library aggregates:\n'
                         '  from hits  %s\n  library    %s' % (from_hits, recomputed))

    expected = None
    if args.expect:
        parts = [float(part) for part in args.expect.split(',')]
        if len(parts) != 2:
            raise SystemExit('--expect wants "i2t_r1,t2i_r1"')
        expected = {'image2text_R1': parts[0], 'text2image_R1': parts[1]}
        for key, value in expected.items():
            if abs(value - recomputed[key]) > 1e-9:
                raise SystemExit('recomputed %s=%r but the standard evaluator reported %r for %s'
                                 % (key, recomputed[key], value, args.checkpoint))

    payload = {
        'label': args.label,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': sha256_of(args.checkpoint),
        'completed_steps': meta.get('completed_steps'),
        'objective': meta.get('objective'),
        'protocol': 'coco-val2017-5caption-legacy-cls',
        'coco_root': os.path.abspath(args.coco_root),
        'annotations': ann_file,
        'caption_order_sha256': hashlib.sha256('\n'.join(caption_stream).encode('utf-8')).hexdigest(),
        'image_order_sha256': hashlib.sha256(
            json.dumps([int(i) for i in getattr(dataset, 'ids', [])]).encode('utf-8')).hexdigest(),
        'n_images': int(images_n.shape[0]),
        'n_texts': int(texts_n.shape[0]),
        'captions_per_image': 5,
        'similarity_chunk': int(coco_lib.DEFAULT_SIMILARITY_CHUNK),
        'library': {'coco_retrieval': COCO_LIB, 'coco_retrieval_sha256': sha256_of(COCO_LIB),
                    'longclip_module': getattr(longclip, '__file__', None),
                    'student_loader': URBAN_EVAL, 'student_loader_sha256': sha256_of(URBAN_EVAL)},
        'recomputed_metrics': recomputed,
        'expected_metrics': expected,
        'hits_i2t': {str(k): i2t_hits[k] for k in K_VALUES},
        'hits_t2i': {str(k): t2i_hits[k] for k in K_VALUES},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True)
    print('WROTE %s' % args.out)
    print('RECOMPUTED ' + json.dumps(recomputed, sort_keys=True))
    if expected:
        print('MATCHES_STANDARD_EVALUATOR ' + json.dumps(expected, sort_keys=True))


if __name__ == '__main__':
    main()
