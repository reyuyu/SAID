"""Verified single-shard ingestion; no network, models, or training.

The Windows transfer process supplies the pinned HF SHA256 before ingestion.
All image writes are atomic and restricted to IDs required by the unchanged JSON.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

from tools.data.audit_sharegpt4v import inspect_image, sha256_file, write_json

REPO = 'hanlincs/InternVL-SA1B-Caption-WebDataset'
REVISION = '4cdaea026f51899bb88d24d423121d2106a943ba'
SHARDS = tuple('%06d' % i for i in range(51))


def canonical_id(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe tar member: ' + name)
    base = path.name
    match = re.fullmatch(r'(sa_[0-9]+)(?:\.(?:jpg|jpeg|png|webp))?', base, re.IGNORECASE)
    if not match:
        raise ValueError('unrecognized image member: ' + name)
    return match.group(1).lower()


def required_ids(records):
    return {PurePosixPath(row['image']).stem for row in records
            if re.fullmatch(r'sam/images/sa_[0-9]+\.jpg', row.get('image', ''))}


def verify_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError('SHA256 mismatch: %s expected=%s actual=%s' % (path, expected, actual))
    return actual


def scan_tar(path, required, seen):
    images, auxiliary = [], 0
    with tarfile.open(path) as archive:
        for member in archive:
            # Never treat symlinks/devices/directories as readable image members.
            if not member.isfile():
                raise ValueError('non-regular tar member: ' + member.name)
            suffix = PurePosixPath(member.name).suffix.lower()
            if suffix in ('.json', '.txt'):
                auxiliary += 1
                continue
            key = canonical_id(member.name)
            images.append((key, member.name))
    counts = Counter(key for key, _ in images)
    ids = set(counts)
    return {'member_count': len(images) + auxiliary, 'image_count': len(images),
            'auxiliary_count': auxiliary, 'ids': sorted(ids),
            'duplicate_ids': sorted(k for k, n in counts.items() if n > 1),
            'required_intersection': len(ids & required), 'hf_only_ids': sorted(ids - required),
            'already_seen_duplicate_ids': sorted(ids & seen)}


def union_stats(shard_ids, required):
    counts = Counter(key for ids in shard_ids.values() for key in ids)
    union = set(counts)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    return {'required_total': len(required), 'union_total': len(union),
            'intersection_total': len(union & required),
            'missing_required': len(required - union), 'missing_samples': sorted(required - union)[:50],
            'extra_ids': len(union - required), 'extra_samples': sorted(union - required)[:50],
            'cross_shard_duplicates': len(duplicates), 'duplicate_samples': duplicates[:50],
            'duplicate_occurrences': sum(n - 1 for n in counts.values())}


def selective_extract(tar_path, destination, required):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = existing = 0
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile() or PurePosixPath(member.name).suffix.lower() in ('.json', '.txt'):
                continue
            key = canonical_id(member.name)
            if key not in required:
                continue
            target = destination / (key + '.jpg')
            if target.is_symlink():
                raise RuntimeError('refusing symlink target: ' + str(target))
            if inspect_image(target)[0] == 'resolved':
                existing += 1
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination, prefix=key + '.', suffix='.partial', delete=False) as out:
                    temporary = Path(out.name)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise OSError('unreadable tar member: ' + member.name)
                    with stream:
                        shutil.copyfileobj(stream, out)
                    out.flush()
                    os.fsync(out.fileno())
                status, error = inspect_image(temporary)
                if status != 'resolved':
                    raise RuntimeError('%s: %s: %s' % (member.name, status, error))
                temporary.replace(target)
                extracted += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return {'newly_extracted': extracted, 'existing_valid': existing,
            'validated_images': extracted + existing}


def initialize(json_path, metadata_path, root, output):
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    entries = {item['path']: item for item in metadata}
    records = json.loads(Path(json_path).read_text(encoding='utf-8'))
    ids = required_ids(records)
    write_json(output / 'required_sam_ids.json', {'json_sha256': sha256_file(json_path), 'ids': sorted(ids)})
    path = output / 'sam_shards.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous.get('revision') == REVISION:
            raise RuntimeError('already initialized; use ingest/status to resume')
    shards = []
    for shard in SHARDS:
        info = entries['data/sa_%s.tar' % shard]
        expected = info['lfs']['oid']
        if not re.fullmatch('[0-9a-f]{64}', expected) or info['size'] != info['lfs']['size']:
            raise ValueError('invalid pinned metadata')
        shards.append({'id': shard, 'tar_path': str(root / 'downloads/hf_sa1b' / ('sa_%s.tar' % shard)),
                       'expected_size': info['size'], 'expected_sha256': expected, 'status': 'missing',
                       'download_completed': False, 'upload_completed': False, 'hash_verified': False,
                       'extract_completed': False, 'image_verified': False, 'error': None})
    write_json(path, {'schema_version': 2, 'repo': REPO, 'revision': REVISION,
                     'source_note': '候选 RGB 来源；未与 Meta 原始图片比较 bytes；尺寸多样不排除等比缩放。',
                     'json_sha256': sha256_file(json_path), 'shards': shards,
                     'union': union_stats({}, ids), 'audit_time': datetime.now(timezone.utc).isoformat()})


def ingest(shard, root, output):
    if shard not in SHARDS:
        raise ValueError('shard outside 000000..000050')
    with (output / 'ingest.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'sam_shards.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        required = set(json.loads((output / 'required_sam_ids.json').read_text())['ids'])
        item = next(s for s in manifest['shards'] if s['id'] == shard)
        all_ids = {}
        for s in manifest['shards']:
            report = output / 'shard_ids' / (s['id'] + '.json')
            if report.exists() and s['id'] != shard:
                all_ids[s['id']] = json.loads(report.read_text())['ids']
        seen = set(k for ids in all_ids.values() for k in ids)
        try:
            tar_path = Path(item['tar_path'])
            if tar_path.stat().st_size != item['expected_size']:
                raise RuntimeError('tar size mismatch')
            item['sha256'] = verify_hash(tar_path, item['expected_sha256'])
            item.update(download_completed=True, upload_completed=True, hash_verified=True, status='downloaded')
            scan = scan_tar(tar_path, required, seen)
            item['id_audit'] = {k: v for k, v in scan.items() if k != 'ids'}
            if scan['duplicate_ids'] or scan['hf_only_ids'] or scan['already_seen_duplicate_ids']:
                raise RuntimeError('ID audit failed; see id_audit')
            write_json(output / 'shard_ids' / (shard + '.json'), scan)
            all_ids[shard] = scan['ids']
            manifest['union'] = union_stats(all_ids, required)
            write_json(path, manifest)
            item['extraction'] = selective_extract(tar_path, root / 'sam/images', required)
            if item['extraction']['validated_images'] != scan['required_intersection']:
                raise RuntimeError('extraction count mismatch')
            item.update(extract_completed=True, image_verified=True, status='verified', error=None)
        except Exception as exc:
            item['error'] = '%s: %s' % (type(exc).__name__, exc)
            write_json(path, manifest)
            raise
        manifest['audit_time'] = datetime.now(timezone.utc).isoformat()
        write_json(path, manifest)
        print(json.dumps({'shard': shard, 'status': item['status'], 'union': manifest['union']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'ingest'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/data_audit'))
    parser.add_argument('--json', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--shard')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'init':
        initialize(args.json, args.metadata, args.root, args.output)
    else:
        ingest(args.shard, args.root, args.output)


if __name__ == '__main__':
    main()
