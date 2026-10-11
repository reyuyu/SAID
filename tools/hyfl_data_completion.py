"""Read-only source checks for Full Flickr and the author's Long-DCI TSV.

No model candidate selection, training, or changes to retrieval mathematics.
Large evidence stays in the caller's runtime directory.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
from PIL import Image
from tools.flickr_full_protocol import read_official
from tools.retrieval_bounded import atomic_json, digest, sha

ZIP_SHA = '2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391'
TOKEN_SHA = 'fad17aa6894489708a31f67d6055a6935fc5d0e3bc632ff45f5bbba6924bf9ee'


def zip_members(z):
    """Reject ambiguous IDs and escaping archive members before extraction."""
    files = [m for m in z.infolist() if not m.is_dir()]
    for m in files:
        p = Path(m.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in m.filename:
            raise ValueError('unsafe ZIP path')
        if ((m.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError('ZIP symlink rejected')
    metadata = []
    for m in files:
        if Path(m.filename).parts[0] == '__MACOSX' and Path(m.filename).name.startswith('._'):
            # The pinned publisher ZIP includes AppleDouble resource-fork files.
            # Confirm actual magic/version; never ignore a JPEG by name alone.
            with z.open(m) as f:
                header = f.read(26)
            if len(header) != 26 or header[:8] != bytes.fromhex('0005160700020000'):
                raise ValueError('invalid AppleDouble metadata; no image silently excluded')
            metadata.append(m)
    metadata_names = {m.filename for m in metadata}
    images = [m for m in files if m.filename not in metadata_names and
              Path(m.filename).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    counts = Counter(Path(m.filename).name for m in images)
    if any(v != 1 for v in counts.values()):
        raise ValueError('duplicate image basename')
    image_names = {m.filename for m in images}
    return {Path(m.filename).name: m for m in images}, [m.filename for m in files if m.filename not in image_names]


def restore_flickr(archive, token_file, output, receipt):
    proof = {'zip_path': str(archive), 'zip_sha256': sha(archive), 'expected_zip_sha256': ZIP_SHA,
             'status': 'RUNNING'}
    try:
        if proof['zip_sha256'] != ZIP_SHA:
            raise ValueError('ZIP SHA mismatch; no extraction')
        raw, rows = read_official(token_file)
        proof['token_sha256'] = hashlib.sha256(raw).hexdigest()
        if proof['token_sha256'] != TOKEN_SHA:
            raise ValueError('official caption SHA mismatch')
        official = {r['image_id'] for r in rows}
        if len(official) != 31783 or len(rows) != 158915:
            raise ValueError('official counts differ')
        with zipfile.ZipFile(archive) as z:
            members, other = zip_members(z)
            missing, extra = sorted(official - set(members)), sorted(set(members) - official)
            proof.update(zip_image_count=len(members), official_image_count=len(official),
                         official_caption_count=len(rows), missing_ids=missing, extra_ids=extra,
                         nonimage_members_count=len(other),
                         apple_double_metadata_count=sum(n.startswith('__MACOSX/') for n in other),
                         other_nonimage_members=[n for n in other if not n.startswith('__MACOSX/')],
                         zip_total_entries=len(z.infolist()))
            if missing or extra:
                raise ValueError('ZIP IDs differ; no subset evaluation')
            out = Path(output); out.mkdir(parents=True, exist_ok=False)
            inventory = []
            for iid in sorted(official):
                data = z.read(members[iid])  # verifies each member's CRC
                target = out / iid
                with target.open('xb') as f:
                    f.write(data)
                with Image.open(target) as im:
                    im.load()  # actually decode, beyond metadata verification
                inventory.append({'id': iid, 'member': members[iid].filename,
                                  'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            atomic_json(out.parent / 'FLICKR_PER_IMAGE_SOURCE.json', inventory, exclusive=True)
            proof.update(status='PASS', decoded_images=len(inventory), inventory_sha256=digest(inventory))
    except Exception as e:
        proof.update(status='FAILED', error=str(e)); raise
    finally:
        atomic_json(receipt, proof, exclusive=True)


def read_author_csv(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != ['filepath', 'title']:
            raise ValueError(f'unexpected TSV fields: {reader.fieldnames}')
        rows = list(reader)
    ids = [r['filepath'] for r in rows]
    if len(ids) != len(set(ids)) or any(not p or Path(p).name != p for p in ids):
        raise ValueError('duplicate/invalid author image filepath')
    if any(set(r) != {'filepath', 'title'} or not isinstance(r['title'], str) or not r['title'].strip()
           for r in rows):
        raise ValueError('missing/empty/malformed author record; no silent filtering')
    return rows


def compare_long(author, said, tokenize):
    """Match by filename, never SAID numeric annotation ID. Tokenize every row."""
    aa = {r['filepath']: r['title'] for r in author}
    ss = {r['image_path']: r['caption'] for r in said}
    if len(ss) != len(said):
        raise ValueError('duplicate SAID image filename')
    if any('positive_image_id' in r and r['positive_image_id'] != r['image_id'] for r in said):
        raise ValueError('SAID positive mapping differs from paired image')
    missing, extra = sorted(set(ss) - set(aa)), sorted(set(aa) - set(ss))
    ids = sorted(set(ss) & set(aa))
    raw_diffs = [p for p in ids if aa[p] != ss[p]]
    strip_diffs = [p for p in ids if aa[p].strip() != ss[p].strip()]
    token_diffs, hashes = [], {}
    for start in range(0, len(ids), 64):
        batch = ids[start:start + 64]
        a, s = tokenize([aa[p] for p in batch]), tokenize([ss[p] for p in batch])
        for p, at, st in zip(batch, a, s):
            if not at.equal(st): token_diffs.append(p)
            hashes[p] = {'author_token_sha256': hashlib.sha256(at.numpy().tobytes()).hexdigest(),
                         'said_token_sha256': hashlib.sha256(st.numpy().tobytes()).hexdigest()}
    order = [r['filepath'] for r in author] == [r['image_path'] for r in said]
    exact = not missing and not extra and not raw_diffs and order
    equivalent = not missing and not extra and not token_diffs and order
    return {'status': 'VERIFIED_EXACT' if exact else 'VERIFIED_TOKEN_EQUIVALENT' if equivalent
            else 'PROTOCOL_MISMATCH', 'author_records': len(author), 'said_records': len(said),
            'missing_author_ids': missing, 'extra_author_ids': extra,
            'raw_text_mismatch_count': len(raw_diffs), 'strip_text_mismatch_count': len(strip_diffs),
            'token_mismatch_count': len(token_diffs), 'token_mismatch_ids': token_diffs,
            'order_exact': order, 'candidate_and_positive_mapping_exact': not missing and not extra,
            'token_digest': digest(hashes), 'token_proof': hashes,
            'limitation': 'No equivalence claim for different ordering: batching and tie indices can differ.'}


def overlap_check(images, old_root, checkpoint, receipt):
    """Require all historical 1K images to be the same actual model inputs."""
    import torch
    from tools.eval_hyfl_native import load_model, MODEL_SHA
    proof = {'status': 'RUNNING', 'checkpoint_sha256_before': sha(checkpoint)}
    try:
        if proof['checkpoint_sha256_before'] != MODEL_SHA: raise ValueError('model SHA mismatch')
        model, pre = load_model(checkpoint, 'cpu')
        del model
        old = sorted(Path(old_root).glob('*.jpg'))
        if len(old) != 1000: raise ValueError('historical Flickr image count differs')
        records = []
        for op in old:
            np = Path(images) / op.name
            if not np.is_file(): raise ValueError('historical 1K image missing from Full')
            byte_equal = sha(np) == sha(op)
            record = {'filename': op.name, 'old_sha256': sha(op), 'new_sha256': sha(np),
                      'bytes_equal': byte_equal}
            if not byte_equal:
                import numpy as numpy
                with Image.open(op) as a, Image.open(np) as b:
                    a, b = a.convert('RGB'), b.convert('RGB')
                    record['pixels_equal'] = numpy.array_equal(numpy.asarray(a), numpy.asarray(b))
                    at, bt = pre(a), pre(b)
                    record['preprocess_equal'] = torch.equal(at, bt)
                    record['preprocess_max_abs'] = float((at - bt).abs().max())
                if not record['pixels_equal'] or not record['preprocess_equal']:
                    records.append(record); proof['first_mismatch'] = record
                    raise ValueError('mirror image version differs from historical model input')
            records.append(record)
        atomic_json(Path(receipt).with_suffix('.images.json'), records, exclusive=True)
        proof.update(status='PASS', overlap_images=len(records), byte_equal=sum(r['bytes_equal'] for r in records),
                     different_bytes_same_pixels_and_preprocess=sum(not r['bytes_equal'] for r in records),
                     per_image_digest=digest(records))
    except Exception as e:
        proof.update(status='FAILED', error=str(e)); raise
    finally:
        proof['checkpoint_sha256_after'] = sha(checkpoint)
        if proof['checkpoint_sha256_after'] != proof['checkpoint_sha256_before']:
            proof.update(status='FAILED', error='checkpoint changed during overlap check')
        atomic_json(receipt, proof, exclusive=True)
    if proof['status'] != 'PASS': raise ValueError(proof['error'])


def gpu_check(job, checkpoint, output):
    """Same batch64 direct vs worker check on Full Flickr, including exact tie rules."""
    import torch
    from model import longclip
    from tools.eval_hyfl_native import load_model, encode, rows_and_images, MODEL_SHA
    from tools.retrieval_bounded import ShardCache, retrieval
    from tools.validate_hyfl_inference import parameter_sha
    from tools.eval_five_parallel import require_gpu_idle
    proof = {'status': 'RUNNING', 'checkpoint_sha256_before': sha(checkpoint)}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    try:
        require_gpu_idle((0, 1, 2, 3))
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        device = 'cuda:0'; model, pre = load_model(checkpoint, device)
        before = parameter_sha(model)
        rows, images = rows_and_images(job['manifest']); rows, images = rows[:64], images[:64]
        # Five captions per image: use independent per-direction positives below.
        with torch.inference_mode():
            pixels = []
            for _, rel in images:
                with Image.open(Path(job['image_root']) / rel) as im:
                    pixels.append(pre(im.convert('RGB')))
            direct_im = torch.nn.functional.normalize(model.encode_image(torch.stack(pixels).to(device)).float(), dim=-1).cpu()
            direct_tx = torch.nn.functional.normalize(model.encode_text(longclip.tokenize([r['caption'] for r in rows], truncate=True).to(device)).float(), dim=-1).cpu()
            cache = ShardCache(out / 'cache', {'check': 'full-flickr-batch64', 'checkpoint': MODEL_SHA})
            timings = {}
            actual_im, actual_tx = encode(model, pre, rows, images, job['image_root'], cache, device,
                                         64, 64, workers=0, timings=timings)
            error = max(float((actual_im - direct_im).abs().max()), float((actual_tx - direct_tx).abs().max()))
            # Retrieval validation needs a caption for every image; use only the 13
            # positive images represented by these 64 captions, preserving explicit IDs.
            pos_ids = list(dict.fromkeys(r['positive_image_id'] for r in rows))
            indices = [next(i for i, x in enumerate(images) if x[0] == iid) for iid in pos_ids]
            args = (pos_ids, [r['caption_id'] for r in rows], [r['positive_image_id'] for r in rows])
            m, d = retrieval(actual_im[indices].to(device), actual_tx.to(device), *args, query_chunk=7, gallery_chunk=13)
            ref, rd = retrieval(direct_im[indices].to(device), direct_tx.to(device), *args, query_chunk=64, gallery_chunk=64)
            hits = all(d[x]['hits'] == rd[x]['hits'] for x in ('I2T', 'T2I'))
        after = parameter_sha(model)
        proof.update(status='PASS' if error <= 5e-6 and hits and before == after else 'FAILED',
                     max_abs=error, tolerance=5e-6, exact_query_hits=hits, batch=64,
                     direct_counts={k: v['correct'] for k, v in ref.items()},
                     worker_counts={k: v['correct'] for k, v in m.items()},
                     state_sha256_before=before, state_sha256_after=after, timings=timings,
                     peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
                     peak_cuda_reserved=torch.cuda.max_memory_reserved(device), device=device)
        if proof['status'] != 'PASS': raise ValueError('same-batch Full correctness failed')
    except Exception as e:
        proof.update(status='FAILED', error=str(e)); raise
    finally:
        proof['checkpoint_sha256_after'] = sha(checkpoint)
        if proof['checkpoint_sha256_after'] != proof['checkpoint_sha256_before']:
            proof.update(status='FAILED', error='checkpoint changed during GPU check')
        atomic_json(out / 'GPU_FLICKR_BATCH64_TEST.json', proof, exclusive=True)
    if proof['status'] != 'PASS': raise ValueError(proof['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('flickr'); f.add_argument('--zip', required=True)
    f.add_argument('--token-file', required=True); f.add_argument('--images', required=True)
    f.add_argument('--receipt', required=True)
    l = sub.add_parser('long'); l.add_argument('--csv', required=True)
    l.add_argument('--said', required=True); l.add_argument('--output', required=True)
    o = sub.add_parser('overlap'); o.add_argument('--images', required=True)
    o.add_argument('--old-root', required=True); o.add_argument('--checkpoint', required=True)
    o.add_argument('--receipt', required=True)
    g = sub.add_parser('gpu-check'); g.add_argument('--job', required=True)
    g.add_argument('--checkpoint', required=True); g.add_argument('--output', required=True)
    args = p.parse_args()
    if args.action == 'flickr':
        restore_flickr(args.zip, args.token_file, args.images, args.receipt)
    elif args.action == 'overlap':
        overlap_check(args.images, args.old_root, args.checkpoint, args.receipt)
    elif args.action == 'gpu-check':
        gpu_check(json.loads(Path(args.job).read_text()), args.checkpoint, args.output)
    else:
        from model import longclip
        said = [json.loads(s) for s in Path(args.said).read_text().splitlines()]
        proof = compare_long(read_author_csv(args.csv), said,
                             lambda texts: longclip.tokenize(texts, truncate=True))
        tokens = proof.pop('token_proof')
        atomic_json(Path(args.output).with_suffix('.tokens.json'), tokens, exclusive=True)
        proof.update(csv_sha256=sha(args.csv), csv_bytes=Path(args.csv).stat().st_size,
                     said_manifest_sha256=sha(args.said), tokenizer_context=248)
        atomic_json(args.output, proof, exclusive=True)
        print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
