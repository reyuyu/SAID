"""Read small audit artifacts only. Never enumerate images or run a model."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from tools.data.audit_sharegpt4v import full_data_gate


def read_artifact(path):
    try:
        with Path(path).open(encoding='utf-8') as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError('根节点必须是对象')
        return value, None
    except (OSError, ValueError) as exc:
        return None, '审计文件尚不可用：%s（%s）' % (Path(path).name, type(exc).__name__)


def gate_passes(audit, shards):
    if not audit or not shards or not full_data_gate(audit):
        return False
    entries = shards.get('shards', [])
    union = shards.get('union', {})
    expected = {'%06d' % i for i in range(51)}
    return (len(entries) == 51 and {s.get('id') for s in entries} == expected
            and all(s.get('image_verified') and s.get('hash_verified') and not s.get('error') for s in entries)
            and audit.get('json_sha256') == shards.get('json_sha256')
            and bool(audit.get('manifest_sha256')) and audit.get('overlap_count') == 0
            and union.get('required_total', 0) > 0
            and union.get('required_total') == union.get('union_total') == union.get('intersection_total')
            and all(union.get(key) == 0 for key in ('missing_required', 'extra_ids', 'cross_shard_duplicates')))


def main():
    from tools.said_dashboard.representation_balance_page import FONT
    st.markdown('<style>.stApp h1,.stApp h2,.stApp h3,.stApp p,'
                '.stApp [data-testid="stMetricValue"] {font-family:' + FONT + ' !important;}</style>',
                unsafe_allow_html=True)
    st.title('数据完整性')
    st.caption('只读离线审计结果；页面不扫描图片，不运行模型。')
    root = Path(st.sidebar.text_input('数据审计目录', os.environ.get('SAID_DATA_AUDIT_ROOT', 'outputs/data_audit')))
    audit, audit_error = read_artifact(root / 'sharegpt4v_full_audit.json')
    shards, shards_error = read_artifact(root / 'sam_shards.json')
    disk, _ = read_artifact(root / 'pre_download_audit.json')
    if gate_passes(audit, shards):
        st.success('完整数据已通过训练 Gate')
    else:
        st.error('完整数据尚未通过训练 Gate')
    for error in (audit_error, shards_error):
        if error:
            st.warning(error)
    if audit:
        columns = st.columns(4)
        for column, label, key in zip(columns, ['JSON 总记录数', '训练记录数', '验证记录数'],
                                       ['records', 'train_records', 'val_records']):
            column.metric(label, format(audit.get(key, 0), ','))
        total = audit.get('records', 0)
        columns[3].metric('已验证比例', '%.3f%%' % (100 * audit.get('resolved_records', 0) / total) if total else '—')
        for column, label, key in zip(st.columns(3), ['缺失图片', '损坏图片', '不可读图片'],
                                       ['missing_records', 'corrupt_records', 'unreadable_records']):
            column.metric(label, format(audit.get(key, 0), ','))
        st.subheader('按来源统计')
        names = {'records': '记录数', 'unique_images': '唯一图片数', 'resolved': '已验证',
                 'missing': '缺失', 'corrupt': '损坏', 'unreadable': '不可读'}
        rows = [{'来源': source, **{label: values.get(key, 0) for key, label in names.items()}}
                for source, values in audit.get('sources', {}).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.subheader('审计标识')
        st.text('JSON SHA256：' + str(audit.get('json_sha256', '未生成')))
        st.text('Split manifest SHA256：' + str(audit.get('manifest_sha256', '未生成')))
        st.text('最近审计时间：' + str(audit.get('audit_time', '未生成')))
    st.subheader('SAM 分片状态')
    if shards:
        rows = []
        for item in shards.get('shards', []):
            row = {'分片': item['id']}
            for key, label in [('download_completed', '下载'), ('upload_completed', '上传'),
                               ('hash_verified', 'Hash 验证'), ('extract_completed', '抽取'),
                               ('image_verified', '图片验证')]:
                row[label] = '完成' if item.get(key) else '待完成'
            row['错误'] = item.get('error') or ''
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption(shards.get('source_note', '候选 RGB 数据来源；未与 Meta 原始图片进行 bytes 比较。'))
    st.subheader('磁盘空间')
    if disk:
        st.text('数据目录占用：%.2f GiB' % (disk.get('data_directory_used_bytes', 0) / 1024**3))
        st.text('服务器剩余空间：%.2f GiB' % (disk.get('filesystem_free_bytes', 0) / 1024**3))
        st.caption('磁盘快照时间：' + str(disk.get('audit_time', '未知')))
    else:
        st.info('尚无磁盘快照。')
