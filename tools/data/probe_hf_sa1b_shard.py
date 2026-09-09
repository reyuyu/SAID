"""Probe one SA-1B WebDataset tar without extracting it."""
import argparse
from collections import Counter
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from statistics import median

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SA_ID = re.compile(r'^(sa_[0-9]+)$')


def canonical_key(name):
    stem = Path(name).name
    if '.' in stem:
        stem = stem.rsplit('.', 1)[0]
    return stem


def member_kind(name):
    suffix = Path(name).suffix.lower()
    return 'image' if suffix in IMAGE_EXTENSIONS else 'auxiliary'


def required_sam_ids(json_path):
    records = json.loads(Path(json_path).read_text(encoding='utf-8'))
    return {canonical_key(r['image']) for r in records
            if re.fullmatch(r'sam/images/sa_[0-9]+\.jpg', r.get('image', ''))}


def resolution_summary(values):
    widths = [v['width'] for v in values]
    heights = [v['height'] for v in values]
    resolutions = Counter((v['width'], v['height']) for v in values)
    return ({'width_min': min(widths), 'width_median': median(widths), 'width_max': max(widths),
             'height_min': min(heights), 'height_median': median(heights), 'height_max': max(heights),
             'unique_resolutions': len(resolutions),
             'possible_resized_source': bool(values) and len(resolutions) == 1
             and next(iter(resolutions)) == (512, 512)} if values else {})


def probe_tar(tar_path, required, sample_limit=1000):
    members = []
    image_members = []
    keys = Counter()
    with tarfile.open(tar_path, 'r:*') as archive:
        for member in archive:
            members.append(member.name)
            if member_kind(member.name) == 'image':
                image_members.append(member)
                keys[canonical_key(member.name)] += 1
        shard_keys = set(keys)
        matched = sorted(shard_keys & set(required))
        selected = matched[:sample_limit]
        by_key = {canonical_key(m.name): m for m in image_members}
        decoded, failed, resolutions, errors = [], [], [], []
        for key in selected:
            try:
                raw = archive.extractfile(by_key[key]).read()
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    resolutions.append({'id': key, 'width': image.width, 'height': image.height,
                                        'format': image.format, 'mode': image.mode})
                    image.convert('RGB').load()
                decoded.append(key)
            except PermissionError as exc:
                failed.append(key); errors.append({'id': key, 'kind': 'unreadable', 'error': str(exc)})
            except Exception as exc:
                failed.append(key); errors.append({'id': key, 'kind': 'corrupt', 'error': str(exc)})
    duplicate_keys = sorted(k for k, n in keys.items() if n > 1)
    return {'member_count': len(members), 'image_member_count': len(image_members),
            'auxiliary_member_count': len(members) - len(image_members),
            'image_extension_counts': dict(Counter(Path(m.name).suffix.lower() for m in image_members)),
            'image_key_count': len(shard_keys), 'duplicate_key_count': len(duplicate_keys),
            'duplicate_keys': duplicate_keys[:50], 'key_examples': sorted(shard_keys)[:50],
            'required_sam_count': len(required), 'intersection_count': len(matched),
            'intersection_ratio': len(matched) / len(shard_keys) if shard_keys else 0.0,
            'matched_examples': matched[:50], 'hf_only_examples': sorted(shard_keys - set(required))[:50],
            'decode_checked': len(selected), 'decode_success': len(decoded),
            'decode_failed': len(failed), 'decode_errors': errors[:50],
            'format_counts': dict(Counter(v['format'] for v in resolutions)),
            'mode_counts': dict(Counter(v['mode'] for v in resolutions)),
            'resolution_summary': resolution_summary(resolutions),
            'id_compatibility': 'PASS' if len(matched) > 0 else 'FAIL',
            'image_source_confidence': 'MEDIUM' if matched and len(resolutions) > 1 else 'UNKNOWN',
            'recommendation': 'NEEDS_REVIEW'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tar_path', type=Path); parser.add_argument('--json', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/data_audit'))
    args = parser.parse_args()
    required = required_sam_ids(args.json)
    result = probe_tar(args.tar_path, required)
    digest = hashlib.sha256()
    with args.tar_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    result.update(repo='hanlincs/InternVL-SA1B-Caption-WebDataset',
                  revision='4cdaea026f51899bb88d24d423121d2106a943ba',
                  shard='data/sa_000000.tar', file_size=args.tar_path.stat().st_size,
                  sha256=digest.hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'hf_sa1b_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'hf_sa1b_probe.md').write_text('# Hugging Face SA-1B 单 shard probe\n\n```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
