"""Reproducible model-free source audit; never edits historical datasets/manifests."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tarfile
from PIL import Image
from tools.retrieval_bounded import atomic_json, digest, sha
from tools.eval_hyfl_native import rows_and_images

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'local_assets'
CHECKPOINT = '/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-s12-uniform-full4868-v1/step4868/evaluations/student_step4868.pt'


def write_rows(path, rows):
    # Audit output creation is exclusive; no historical files can be overwritten.
    if Path(path).exists():
        actual = [json.loads(s) for s in Path(path).read_text().splitlines()]
        if actual != rows:
            raise ValueError('preparation resume manifest mismatch')
        return
    with Path(path).open('x', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def image_inventory(images, root):
    def one(item):
        iid, rel = item
        p = (Path(root) / rel).resolve()
        if not p.is_relative_to(Path(root).resolve()) or not p.is_file():
            raise ValueError(f'missing/escaping image {p}')
        with Image.open(p) as im:
            im.verify()
        return {'id': iid, 'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size}
    with ThreadPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(one, images))
    return records


def row(iid, path, caption_id, caption, protocol):
    return {'image_id': str(iid), 'image_path': path, 'caption_id': str(caption_id),
            'caption': caption, 'positive_image_id': str(iid), 'protocol_id': protocol}


def run(out, evidence):
    out, evidence = Path(out), Path(evidence)
    out.mkdir(parents=True, exist_ok=True); evidence.mkdir(parents=True, exist_ok=True)
    if (out / 'DATA_PROVENANCE.json').exists():
        raise FileExistsError(out / 'DATA_PROVENANCE.json')
    bench = ASSETS / 'retrieval_benchmarks'
    historical = json.loads((ROOT / 'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1/step4868/RESULTS.json').read_text())['metrics']
    jobs, provenance = [], {}

    def add(name, manifest, image_root, protocol, status, source, legacy=None, norm='gpu'):
        rows, images = rows_and_images(manifest)
        print('Verify all image bytes/decoding:', name, len(images), flush=True)
        inventory = image_inventory(images, image_root)
        inventory_path = evidence / (name + '_image_inventory.json')
        if inventory_path.exists():
            if json.loads(inventory_path.read_text()) != inventory:
                raise ValueError('preparation resume image inventory mismatch')
        else:
            atomic_json(inventory_path, inventory, exclusive=True)
        m = {'name': name, 'manifest': str(Path(manifest).resolve()), 'manifest_sha256': sha(manifest),
             'image_root': str(Path(image_root).resolve()), 'image_content_sha256': digest(inventory),
             'n_images': len(images), 'n_captions': len(rows), 'protocol': protocol,
             'protocol_status': status, 'source': source, 'normalization': norm,
             'estimated_seconds': (len(images) + len(rows)) / 25,
             'work_units': len(images) + len(rows), 'image_batch': 64, 'text_batch': 64,
             'query_chunk': 256, 'gallery_chunk': 4096, 'workers': 4}
        if legacy:
            m['legacy_expected'] = {d: {str(k): round(historical[legacy][d][f'R@{k}'] *
                                                    (len(images) if d == 'I2T' else len(rows)))
                                        for k in (1, 5, 10)} for d in ('I2T', 'T2I')}
        jobs.append(m); provenance[name] = m

    # DOCCI official source strings, filename test selection and filename ordering.
    docci_source = ASSETS / 'downloads/docci_descriptions.jsonlines'
    raw = [json.loads(s) for s in docci_source.read_text().splitlines()]
    docci = [r for r in raw if r['split'] == 'test']
    old = [json.loads(s) for s in (bench / 'manifests/docci_test.jsonl').read_text().splitlines()]
    assert len(docci) == len(old) == 5000
    mapping = {r['example_id']: r for r in docci}
    assert all(r['caption'] == mapping[r['image_id']]['description'] and
               r['image_path'] == mapping[r['image_id']]['image_file'] for r in old)
    assert all('test' in Path(r['image_path']).name for r in old)
    assert [r['image_path'] for r in old] == sorted(r['image_path'] for r in old)
    add('DOCCI', bench / 'manifests/docci_test.jsonl', bench / 'docci/image_archive/images',
        'DOCCI-official-test-5000-cosine248', 'VERIFIED_SOURCE_PROTOCOL',
        {'url': 'https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines',
         'annotation_sha256': sha(docci_source), 'original_count': len(raw),
         'exact_description_filename_match': True, 'hyfl_annotation_bytes_available': False}, 'DOCCI')

    # Canonical torchvision/pycocotools order and first-five rule, confirmed TULIP source.
    cp = ASSETS / 'evaluation/coco/annotations/captions_val2017.json'
    coco = json.loads(cp.read_text()); imgs = {r['id']: r for r in coco['images']}
    caps = {iid: [] for iid in imgs}
    for a in coco['annotations']:
        caps[a['image_id']].append(a)
    assert len(imgs) == 5000 and len(coco['annotations']) == 25014 and min(map(len, caps.values())) >= 5
    rows = [row(iid, imgs[iid]['file_name'], a['id'], a['caption'], 'coco2017-val5k-first5')
            for iid in sorted(imgs) for a in caps[iid][:5]]
    cm = evidence / 'COCO.jsonl'; write_rows(cm, rows)
    add('COCO', cm, ASSETS / 'evaluation/coco/val2017', 'COCO2017-Val5K-first5',
        'VERIFIED_SOURCE_PROTOCOL', {'annotation_sha256': sha(cp), 'original_captions': 25014,
                                     'selected_captions': 25000, 'excluded_extra_annotations': 14}, 'COCO', 'cpu_raw_for_legacy')

    from tools.urban1k_retrieval import image_caption_pairs, dataset_fingerprint
    urban_root = ASSETS / 'evaluation/Urban1k/Urban1k'
    pairs = image_caption_pairs(str(urban_root)); assert len(pairs) == 1000
    rows = [row(Path(ip).stem, 'image/' + Path(ip).name, Path(cp).stem,
                Path(cp).read_text().splitlines(keepends=True)[0], 'urban-full1k') for ip, cp in pairs]
    um = evidence / 'Urban-1k.jsonl'; write_rows(um, rows)
    add('Urban-1k', um, urban_root, 'Urban-full1k-firstline-cosine248', 'VERIFIED_SOURCE_PROTOCOL',
        dataset_fingerprint(str(urban_root)), 'Urban-1k')

    add('Flickr30k-Test1K', bench / 'manifests/flickr30k_test1k.jsonl', bench / 'flickr30k/images',
        'SAID-existing-Flickr-test1k-5captions', 'LEGACY_ONLY',
        {'caption_csv_sha256': sha(ASSETS / 'downloads/test_1k_flickr.csv'),
         'note': 'Existing test1K; not independently relabeled Karpathy without split-ID proof.'}, 'Flickr30k-test1k')

    # Read the original frozen DCI archive directly, without NFS per-file latency.
    archive = ASSETS / 'downloads/dci.tar.gz'
    annotations = {}
    print('Read frozen DCI archive', flush=True)
    archive_sha = sha(archive)
    # Frozen recovery/fetch_assets.py and docs/extended_retrieval/status.json agree.
    # The older prepare_retrieval_benchmarks.py 9caff... constant is a documented typo.
    assert archive_sha == 'd865c244150168d3f25daaad0bf5b70b2123cf3e83ca7b0e207d5b26943c5fc7'
    with tarfile.open(archive, 'r:gz') as t:
        for member in t:
            if '/annotations/' in member.name and member.name.endswith('-data.json'):
                annotations[Path(member.name).name[:-10]] = json.load(t.extractfile(member))
    assert len(annotations) == 7805
    old_long = [json.loads(s) for s in (bench / 'manifests/long_dci_reconstructed.jsonl').read_text().splitlines()]
    valid = {k: v for k, v in annotations.items() if len(v['extra_caption']) != 0}
    assert len(valid) == len(old_long) == 7602 and set(valid) == {r['image_id'] for r in old_long}
    assert all(r['image_path'] == annotations[r['image_id']]['image'] and
               r['caption'] == annotations[r['image_id']]['extra_caption'].strip() for r in old_long)
    long_rows = [dict(r, caption=annotations[r['image_id']]['extra_caption'],
                      protocol_id='TULIP-constructor-original-extra-caption') for r in old_long]
    lm = evidence / 'Long-DCI.jsonl'; write_rows(lm, long_rows)
    add('Long-DCI', lm, bench / 'dci/image_archives', 'TULIP-constructor-reconstruction-7602',
        'PROTOCOL_UNVERIFIED', {'archive_sha256': archive_sha, 'raw_count': 7805, 'nonempty_count': 7602,
                                'excluded_empty_extra_caption': 203, 'official_csv_available': False,
                                'constructor_match': True, 'old_raw_string_differences': sum(
                                    a['caption'] != b['caption'] for a, b in zip(old_long, long_rows)),
                                'old_manifest_sha256': sha(bench / 'manifests/long_dci_reconstructed.jsonl'),
                                'limitation': 'Official dci_long.csv unavailable: no CSV row-by-row equivalence claim.'}, 'Long-DCI')

    # Explicit provisional DCI protocol; NEVER identify this as HyFL DCI_test.json.
    dci_rows = [row(k, v['image'], k, (v['short_caption'] + ' ' + v['extra_caption']).strip(),
                    'SAID-DCI-short-plus-extra-unverified') for k, v in sorted(annotations.items())]
    # Match official loader's filename sorting for this provisional annotation only.
    dci_rows.sort(key=lambda r: r['image_path'])
    dm = evidence / 'DCI.jsonl'; write_rows(dm, dci_rows)
    add('DCI', dm, bench / 'dci/image_archives', 'SAID-DCI-short-plus-extra-7805-NOT-HyFL-verified',
        'DCI_PROTOCOL_UNVERIFIED', {'archive_sha256': archive_sha, 'DCI_test_json_available': False,
                                    'caption_definition': 'short_caption + space + extra_caption; strip',
                                    'limitation': 'No evidence this matches HyFL caption construction/subset.'})
    blocked = {'Flickr30k-Full': {'status': 'BLOCKED_PROTOCOL_DATA', 'n_images': None, 'n_captions': None,
                                 'expected_only': {'n_images': 31783, 'n_captions': 158915},
                                 'reason': 'Only 1000-image archive present; official full images/token file not acquired.'}}
    atomic_json(out / 'DATA_PROVENANCE.json', {'datasets': provenance, 'blocked': blocked}, exclusive=True)
    atomic_json(out / 'EVALUATION_PLAN.json', {'checkpoint': CHECKPOINT, 'jobs': jobs, 'blocked': blocked}, exclusive=True)
    print('DATA AUDIT DONE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); run(a.output, a.evidence)


if __name__ == '__main__':
    main()
