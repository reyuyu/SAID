"""Official Flickr token parser with explicit #caption mapping; no implicit i//5."""
from collections import Counter
from pathlib import Path
import tarfile


def parse_token_lines(lines):
    rows = []; captions = set(); slots = {}; order = []
    for line in lines:
        if '\t' not in line:
            raise ValueError('malformed official Flickr token row')
        cid, text = line.split('\t', 1)
        if cid in captions or '#' not in cid or not text.strip():
            raise ValueError('duplicate/invalid/empty caption')
        image, index = cid.rsplit('#', 1)
        if index not in {'0', '1', '2', '3', '4'} or Path(image).name != image:
            raise ValueError('invalid image filename/caption index')
        if image not in slots:
            slots[image] = set(); order.append(image)
        slots[image].add(index); captions.add(cid)
        rows.append({'image_id': image, 'image_path': image, 'caption_id': cid,
                     'caption': text, 'positive_image_id': image, 'protocol_id': 'Flickr30k-Full-official-token'})
    if not rows or any(s != {'0', '1', '2', '3', '4'} for s in slots.values()):
        raise ValueError('missing caption slots; no filtering allowed')
    return rows


def read_official(path):
    path = Path(path)
    if path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as t:
            data = t.extractfile('results_20130124.token').read()
    else:
        data = path.read_bytes()
    return data, parse_token_lines(data.decode('utf8').splitlines(keepends=True))


def main():
    """Prepare a full job only when all authoritative annotation images exist."""
    import argparse
    import hashlib
    from tools.audit_hyfl_data import image_inventory, write_rows
    from tools.eval_hyfl_native import rows_and_images
    from tools.retrieval_bounded import atomic_json, digest, sha
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--token-file', required=True); p.add_argument('--image-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); raw, rows = read_official(a.token_file)
    ids = list(dict.fromkeys(r['image_id'] for r in rows))
    if len(ids) != 31783 or len(rows) != 158915:
        raise ValueError('Full Flickr annotation count differs; no subset substitution')
    # Complete image decoding/hash check BEFORE creating a runnable job.
    inventory = image_inventory([(iid, iid) for iid in ids], a.image_root)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    manifest = out / 'FULL_MANIFEST.jsonl'; write_rows(manifest, rows)
    job = {'name': 'Flickr30k-Full', 'manifest': str(manifest.resolve()), 'manifest_sha256': sha(manifest),
           'image_root': str(Path(a.image_root).resolve()), 'image_content_sha256': digest(inventory),
           'protocol': 'Flickr30k-Full-official-token-31783x158915', 'protocol_status': 'VERIFIED_SOURCE_PROTOCOL',
           'n_images': 31783, 'n_captions': 158915, 'annotation_sha256': hashlib.sha256(raw).hexdigest(),
           'estimated_seconds': (31783 + 158915) / 25, 'work_units': 31783 + 158915,
           'image_batch': 64, 'text_batch': 64, 'query_chunk': 256, 'gallery_chunk': 4096, 'workers': 4}
    atomic_json(out / 'IMAGE_INVENTORY.json', inventory, exclusive=True)
    atomic_json(out / 'JOB.json', job, exclusive=True)


if __name__ == '__main__':
    main()
