"""Offline, exhaustive ShareGPT4V image audit; never filters training records."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import warnings

from PIL import Image


# Derived from the full JSON's observed paths, not caption text or guessed IDs.
SOURCE_PATTERNS = (
    ('COCO', r'coco/train2017/[0-9]{12}\.jpg'),
    ('LLaVA', r'llava/llava_pretrain/images/[0-9]{5}/[0-9]{9}\.jpg'),
    ('SAM', r'sam/images/sa_[0-9]+\.jpg'),
)
STATUSES = ('resolved', 'missing', 'unreadable', 'corrupt')


def source_for(path):
    for source, pattern in SOURCE_PATTERNS:
        if re.fullmatch(pattern, path):
            return source
    return 'unknown_source'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe_path(root, relative):
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or '..' in pure.parts or '\\' in relative:
        raise ValueError('unsafe relative image path: %r' % relative)
    resolved_root = Path(root).resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError('image path escapes data root: %r' % relative)
    return path


def inspect_image(path):
    """Permission/filesystem failures are unreadable; decodable format failures corrupt."""
    try:
        info = Path(path).stat()
        if not stat.S_ISREG(info.st_mode):
            return 'unreadable', 'not a regular file'
        # Count files inaccessible to normal users even when the audit runs as root.
        if not info.st_mode & 0o444:
            return 'unreadable', 'no read permission bits'
        with open(path, 'rb') as stream:
            stream.read(1)
    except FileNotFoundError as exc:
        return 'missing', str(exc)
    except OSError as exc:
        return 'unreadable', '%s: %s' % (type(exc).__name__, exc)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                rgb.load()
                rgb.close()
        return 'resolved', None
    except OSError as exc:
        kind = 'unreadable' if exc.errno in (errno.EACCES, errno.EPERM, errno.EIO) else 'corrupt'
        return kind, '%s: %s' % (type(exc).__name__, exc)
    except Exception as exc:
        # Unexpected decoder errors are retained, counted, and block the gate.
        return 'corrupt', '%s: %s' % (type(exc).__name__, exc)


def _inspect_job(job):
    root, relative = job
    try:
        path = safe_path(root, relative)
    except (ValueError, OSError) as exc:
        return relative, 'unreadable', str(exc)
    status, error = inspect_image(path)
    return relative, status, error


def split_manifest(records, root, json_sha256):
    entries = []
    val_ids, train_ids = set(), set()
    for index, record in enumerate(records):
        relative = record['image']
        try:
            canonical = safe_path(root, relative).relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            canonical = 'invalid:' + relative
        split = 'validation' if index < 1000 else 'training'
        (val_ids if split == 'validation' else train_ids).add(canonical)
        entries.append({'json_index': index, 'relative_image_path': relative,
                        'source': source_for(relative), 'split': split,
                        'canonical_identifier': canonical})
    overlap = sorted(val_ids & train_ids)
    return {'schema_version': 1, 'json_sha256': json_sha256,
            'val_records': min(1000, len(records)), 'train_records': max(0, len(records) - 1000),
            'overlap_count': len(overlap), 'overlap_samples': overlap[:50],
            'overlap_identifiers': overlap, 'entries': entries}


def audit_records(records, root, workers=1, progress=False, failures_path=None):
    counts = Counter(record['image'] for record in records)
    sources = {}
    for relative, count in counts.items():
        source = source_for(relative)
        if source not in sources:
            sources[source] = dict(records=0, unique_images=0, duplicate_path_records=0,
                                   **dict.fromkeys(STATUSES, 0))
        item = sources[source]
        item['records'] += count
        item['unique_images'] += 1
        item['duplicate_path_records'] += count - 1
    examples = defaultdict(list)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    jobs = ((str(root), relative) for relative in counts)
    results = executor.map(_inspect_job, jobs, chunksize=64) if executor else map(_inspect_job, jobs)
    failure_stream = open(failures_path, 'w', encoding='utf-8') if failures_path else None
    try:
        for completed, (relative, status, error) in enumerate(results, 1):
            sources[source_for(relative)][status] += counts[relative]
            if error:
                item = {'image': relative, 'status': status, 'records': counts[relative], 'error': error}
                if len(examples[status]) < 50:
                    examples[status].append(item)
                if failure_stream:
                    failure_stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            if progress and completed % 10000 == 0:
                print(json.dumps({'checked_unique_images': completed, 'total_unique_images': len(counts)}), flush=True)
    finally:
        if executor:
            executor.shutdown()
        if failure_stream:
            failure_stream.close()
    totals = {name + '_records': sum(item[name] for item in sources.values()) for name in STATUSES}
    patterns = {pattern: sum(n for p, n in counts.items() if re.fullmatch(pattern, p))
                for _, pattern in SOURCE_PATTERNS}
    unknown = {p: n for p, n in counts.items() if source_for(p) == 'unknown_source'}
    return {'records': len(records), 'unique_images': len(counts), **totals,
            'val_records': min(1000, len(records)), 'train_records': max(0, len(records) - 1000),
            'duplicate_path_records': len(records) - len(counts),
            'unknown_source_records': sum(unknown.values()),
            'unknown_source_samples': list(unknown.items())[:50],
            'directory_counts': dict(sorted(Counter(str(PurePosixPath(r['image']).parent) for r in records).items())),
            'path_pattern_records': patterns, 'sources': dict(sorted(sources.items())),
            'error_samples': dict(examples), 'image_validation': 'PIL.verify + reopen.convert(RGB).load',
            'scan_complete': True}


def full_data_gate(audit):
    try:
        return (audit['scan_complete'] is True and audit['records'] > 0
                and audit['resolved_records'] == audit['records']
                and all(audit[key] == 0 for key in ('missing_records', 'corrupt_records',
                         'unreadable_records', 'unknown_source_records')))
    except (KeyError, TypeError):
        return False


def audit_markdown(audit):
    lines = ['# ShareGPT4V 全量图片审计', '', '审计时间：' + audit['audit_time'],
             '', 'Full Data Gate：' + ('PASS' if full_data_gate(audit) else 'FAIL'), '',
             '| 来源 | Records | Unique | Resolved | Missing | Corrupt | Unreadable |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for source, item in audit['sources'].items():
        lines.append('| %s | %s |' % (source, ' | '.join(str(item[k]) for k in
                     ('records', 'unique_images', 'resolved', 'missing', 'corrupt', 'unreadable'))))
    lines += ['', '```json', json.dumps({k: v for k, v in audit.items()
               if k not in ('sources', 'directory_counts', 'error_samples')}, ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/data_audit'))
    parser.add_argument('--name', default='sharegpt4v_full_audit')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_file(args.json)
    records = json.loads(args.json.read_text(encoding='utf-8'))
    manifest = split_manifest(records, args.root, fingerprint)
    write_json(args.output_dir / 'sharegpt4v_split_manifest.json', manifest)
    overlap = manifest['overlap_count']
    del manifest
    result = audit_records(records, args.root, args.workers, progress=True,
                           failures_path=args.output_dir / (args.name + '_failures.jsonl'))
    if sha256_file(args.json) != fingerprint:
        raise RuntimeError('JSON changed during audit; result not published')
    result.update(schema_version=1, json_path=str(args.json.resolve()), json_sha256=fingerprint,
                  data_root=str(args.root.resolve()), overlap_count=overlap,
                  audit_time=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / (args.name + '.json'), result)
    (args.output_dir / (args.name + '.md')).write_text(audit_markdown(result), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('directory_counts', 'error_samples', 'path_pattern_records')}, ensure_ascii=False), flush=True)
    return 0 if full_data_gate(result) else 2


if __name__ == '__main__':
    raise SystemExit(main())
