"""Execute unchanged official dataset readers and compare raw membership/text."""
import ast
import copy
import json
import logging
import os
from pathlib import Path

from .common import ASSETS, BENCH, DATA, OFFICIAL, RUN, digest, dump, official_import, sha256


def main():
    module = official_import('training.data')
    factory = official_import('local_clip.factory')
    tokenizer = factory.get_tokenizer('ViT-B-16-longclip', is_siglip='none')
    urban = module.Urban1kTextDataset(str(DATA/'urban1k'), tokenizer=tokenizer)
    docci = module.DocciTextDataset(str(DATA/'docci'), tokenizer=tokenizer, split='test')
    coco = module.read_coco_pairs(str(DATA/'coco'), split='val')
    flickr = module.read_flickr_pairs(str(DATA/'flickr30k-images'), split='val')
    dci = module.read_dci_pairs(str(DATA/'dci'), split='test')
    raw = {'COCO': coco, 'Flickr-full': flickr, 'Urban1k': urban.old_data,
           'DOCCI': docci.old_data, 'DCI-full': dci}
    audit = {}
    for name, rows in raw.items():
        ids = list(dict.fromkeys(r['image_id'] for r in rows))
        paths = list(dict.fromkeys(r['image'] for r in rows))
        missing = [p for p in paths if not Path(p).is_file()]
        audit[name] = {'image_count': len(ids), 'unique_image_paths': len(paths),
            'caption_count': len(rows), 'full_image_candidates': len(ids), 'full_text_candidates': len(rows),
            'missing_images': len(missing), 'missing_examples': [Path(p).name for p in missing[:20]],
            'raw_caption_sha256': digest([r['caption'] for r in rows]),
            'pair_sha256': digest([(Path(r['image']).name, r['caption']) for r in rows]),
            'root': str(DATA), 'official_dataset_parser_executed': True}
    from tools.urban1k_retrieval import image_caption_pairs, read_captions
    pairs = image_caption_pairs(str(ASSETS/'evaluation/Urban1k/Urban1k'))
    said_urban = [(Path(p[0]).name, c) for p, c in zip(pairs, read_captions(pairs))]
    official_urban = [(Path(r['image']).name, r['caption']) for r in urban.old_data]
    off_by_image = dict(official_urban)
    diffs = [{'image': p, 'official': off_by_image.get(p), 'said': c}
             for p, c in said_urban if off_by_image.get(p) != c]
    urban_audit = {'official_count': len(official_urban), 'said_count': len(said_urban),
        'identical_count': len(said_urban)-len(diffs), 'different_count': len(diffs),
        'official_sorted_pair_sha256': digest(sorted(official_urban)),
        'said_sorted_pair_sha256': digest(sorted(said_urban)),
        'official_order_sha256': digest(official_urban), 'said_order_sha256': digest(said_urban),
        'order_identical': official_urban == said_urban, 'first20_diffs': diffs[:20]}
    docci_rows = [json.loads(l) for l in (BENCH/'manifests/docci_test.jsonl').read_text().splitlines() if l.strip()]
    off_docci = {Path(r['image']).name: r['caption'] for r in docci.old_data}
    docci_diffs = [{'image': r['image_path'], 'official': off_docci.get(r['image_path']), 'said': r['caption']}
        for r in docci_rows if off_docci.get(r['image_path']) != r['caption']]
    audit['DOCCI'].update(said_pair_count=len(docci_rows), raw_caption_different_count=len(docci_diffs),
                         sorted_pairs_identical=(sorted(off_docci.items()) == sorted((r['image_path'], r['caption']) for r in docci_rows)),
                         first20_diffs=docci_diffs[:20])
    anns = json.loads((ASSETS/'evaluation/coco/annotations/captions_val2017.json').read_text())
    caps = {}
    for row in anns['annotations']:
        caps.setdefault(row['image_id'], []).append(row['caption'])
    canonical = [c for i in sorted(caps) for c in caps[i][:5]]
    audit['COCO'].update(annotation_sha256=sha256(ASSETS/'evaluation/coco/annotations/captions_val2017.json'),
        said_image_count=5000, said_caption_count=25000,
        caption_count_histogram={str(k): sum(len(v)==k for v in caps.values()) for k in sorted(set(map(len, caps.values())))},
        official_extra_captions=len(coco)-len(canonical), candidate_order_identical=False,
        caption_protocol_equivalent=len(coco)==len(canonical))
    audit['DCI-full'].update(annotation_splits_sha256=sha256(DATA/'dci/densely_captioned_images/splits.json'),
        caption_construction="(short_caption + extra_caption).strip(); no inserted separator", all_splits=True,
        said_long_dci_manifest_sha256=sha256(BENCH/'manifests/long_dci_reconstructed.jsonl'), directly_comparable_to_said_long_dci=False)
    audit['Flickr-full'].update(said_test1k=(1000, 5000),
        composition='val + train + test from official reader; no split override', directly_comparable_to_said_test1k=False)
    dump(RUN/'dataset-protocol-audit.json', audit)
    dump(RUN/'urban-data-audit.json', urban_audit)
    # Runtime data lists are not Git artifacts or official input substitutes.
    dump(RUN/'official-raw-pairs.json', raw)
    print(json.dumps({'datasets': audit, 'urban': urban_audit}, indent=2), flush=True)


if __name__ == '__main__':
    main()
