"""Bind a successful complete audit to the exact JSON, split, and data root."""
import json
import os
from pathlib import Path

from tools.data.audit_sharegpt4v import full_data_gate, sha256_file


def resolve_json_path(image_root, json_setting):
    """Resolve ``SHARE4V_JSON`` with the same semantics as the dataset loader.

    Absolute paths are used as-is; relative paths (the usual case, e.g.
    ``share-captioner_coco_lcs_sam_1246k_1107.json`` or ``debug/xxx.json``) are
    joined onto ``SHARE4V_DATA_ROOT``.
    """
    if not json_setting:
        raise ValueError('SHARE4V_JSON is empty')
    if os.path.isabs(json_setting):
        return json_setting
    return os.path.join(image_root, json_setting)


def require_full_data(audit_path, json_path, image_root):
    try:
        path = Path(audit_path)
        audit = json.loads(path.read_text(encoding='utf-8'))
        if not full_data_gate(audit):
            raise ValueError('完整数据尚未通过训练 Gate')
        if audit.get('overlap_count') != 0:
            raise ValueError('train/val overlap requires Review')
        if audit.get('json_sha256') != sha256_file(json_path):
            raise ValueError('JSON SHA256 differs from audited JSON')
        if Path(audit['data_root']).resolve() != Path(image_root).resolve():
            raise ValueError('data root differs from audited root')
        split = path.parent / 'sharegpt4v_split_manifest.json'
        if audit.get('manifest_sha256') != sha256_file(split):
            raise ValueError('split manifest SHA256 mismatch')
        return audit
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Full data fail-fast: %s' % exc) from exc
