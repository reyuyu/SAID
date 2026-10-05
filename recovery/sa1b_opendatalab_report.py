"""Render authorized mirror benchmark and independent split-recovery progress."""

import json
from pathlib import Path

import sa1b_recovery as recovery
from sa1b_throughput_benchmark import payload_bytes


def render():
    folder = recovery.EVIDENCE / "sa1b-opendatalab"
    result = recovery.load(folder / "authorized-probe/result.json", {})
    audit = recovery.load(folder / "authorized-probe/full-audit/audit.json", {})
    split = recovery.load(folder / "split-recovery/state.json", {})
    anomalies = recovery.load(folder / "authorized-anomaly-prefixes/results.json", {})
    authorization = recovery.load(folder / "authorized-download-check.json", {})
    state = recovery.load(recovery.STATE)
    rate = result.get("average_mib_s")
    hf_rate = result.get("hf_average_mib_s")
    pending = [row for row in state["shards"] if row.get("download_status") != recovery.QUARANTINED
               and row.get("extraction_status") != "complete"]
    remaining = {row["filename"]: max(0, row["expected_size_bytes"] - payload_bytes(row)) for row in pending}
    planned_odl = {f"sa_{position:06d}.tar" for position in range(39, 51)} if split else set()
    odl_remaining = sum(count for name, count in remaining.items() if name in planned_odl)
    hf_remaining = sum(count for name, count in remaining.items() if name not in planned_odl)
    eta = max(odl_remaining / (rate * 1024**2), hf_remaining / (hf_rate * 1024**2)) / 3600 if rate and hf_rate else None
    worker = recovery.load(folder / "authorized-probe/worker-result.json", {})
    counters = result.get("counters", {})
    active_duration = None
    if counters.get("payload_started_monotonic") and counters.get("last_payload_monotonic"):
        active_duration = counters["last_payload_monotonic"] - counters["payload_started_monotonic"]
    lines = ["# SA-1B OpenDataLab / OpenXLab benchmark", "",
        f"更新：{recovery.now()}。Repository：`OpenDataLab/SA-1B`；dataset ID：`6248`。",
        "HF 四路 supervisor process group 24215 / Python PID 24219 保持运行，本次没有停止、重启或修改该运行进程。没有启动 smoke 或正式训练。", "",
        "## 授权与原始清单", "",
        f"官方 AK/SK 登录成功：{authorization.get('login_success')}; SA-1B download check 通过：{authorization.get('download_check_passed')}。",
        "凭据只由官方 SDK 保存到 `/root/.openxlab`（目录 0700、config/token 0600），没有写入项目、下载 provenance 或报告。",
        "官方公共 metadata 分页确认 1000 个原命名 tar；所需前 51 个文件大小和公开 SHA256 全部与固定 HF revision 140d15308aff47dae3b00214083838d56a4319c6 一致。",
        "工具位于独立 `.opendatalab-venv`：opendatalab 0.0.10 / openxlab 0.1.3；训练和 HF 下载环境未增加依赖。", "",
        "## 十分钟单包 benchmark", "",
        "选择 `/raw/sa_000038.tar`：测试前 HF 尚未开始且零 partial；000030 已有此前 HF benchmark partial，故没有选它。",
        f"测量窗口：{result.get('duration_seconds')} 秒；完整文件：{result.get('archive_complete')}；SDK 任务时间：{worker.get('started_utc')}–{worker.get('finished_utc')}。",
        f"实际图片 payload 传输时间约 {active_duration} 秒。若在十分钟内完成，后续包含组装及空闲，保守窗口均速没有伪称为连续传输了十分钟，也没有重下已完成文件。",
        "只使用官方 `openxlab.dataset.download` 单文件调用。每个 shard SDK 默认 8 个 HTTP Range，不等于 8 个 shard 并发。", "",
        "| 指标 | 测量结果 |", "|---|---|",
        f"| ODL 保守全窗口平均 MiB/s | {rate} |",
        f"| ODL 5秒采样峰值 MiB/s | {result.get('peak_five_second_mib_s')} |",
        f"| 同窗口 HF 四路 MiB/s | {hf_rate} |",
        f"| 联合平均 MiB/s | {result.get('combined_mib_s')} |",
        f"| 联合 / 原五分钟 HF 5.300MiB/s | {result.get('combined_gain')} 倍 |",
        f"| HTTP / stream errors | {counters.get('errors')} |",
        f"| retry warnings | {counters.get('retries')} |",
        f"| HTTP status counts | `{json.dumps(counters.get('statuses', {}), sort_keys=True)}` |",
        f"| Range 206 实际观察 | {result.get('resume_actual_range_206')} |",
        "| 完整中断后重启恢复测试 | 尚未执行；Range 支持不是完整重启恢复验证 |",
        f"| 实际 CDN object 路径（无签名参数） | `{json.dumps(counters.get('hosts', []))}` |", "",
        "该路线实际走 OpenXLab CDN HTTP Range，不使用 HF Xet；没有进行无效的 Xet 开关对比。", "",
        "## 协议审计", "",
        f"分类：`{audit.get('classification', 'FULL_AUDIT_PENDING')}`；protocol_passed：{audit.get('protocol_passed', False)}。",
        f"大小：{audit.get('size_bytes')}；实际 MD5：`{audit.get('observed_md5')}`；与 checklist 一致：{audit.get('md5_matches_checklist')}。",
        f"实际 SHA256：`{audit.get('observed_sha256')}`；GNU tar integrity：{audit.get('tar_integrity_passed')}。",
        f"JPEG数：{audit.get('jpeg_count')}；完整解码通过数：{audit.get('decoded_jpegs')}；失败数：{len(audit.get('decode_failures', []))}。",
        f"原 image ID 范围：{audit.get('min_image_id')}–{audit.get('max_image_id')}。",
        f"冻结 training index 在该 ID 范围所需图片：{audit.get('required_images_found')}/{audit.get('required_images_in_id_range')}；缺失：{len(audit.get('required_missing_in_id_range', []))}。",
        "保留原始 basename 和编码字节；未 resize/recompress/rename/format conversion，mask JSON 未安装到训练目录。",
        "实际全量解码、逐 shard 必需 ID 集合和前16个 JPEG byte hash/尺寸记录在 full-audit/audit.json。", "",
        "## 分流与无覆盖交接", "",
        f"独立 ODL 分流状态：{split.get('enabled', False)}；ODL shard 并发：{split.get('odl_shard_concurrency', 0)}；HF shard 并发仍为4。",
        "初始速度 >=8MiB/s、联合提升 >=30%、完整 protocol audit 三个 gate 都通过才启用。后续每包保持速度/错误/联合增益检查。",
        "只考虑 000039–000050 中 HF 尚未开始、零 partial、距离 HF 当前活动 shard 至少8个位置的未来文件。000038 benchmark 文件仅在完整验证通过后交接。",
        "每包独立路径 → 固定 size/MD5/SHA → tar integrity → 全JPEG解码 → 冻结 training-index IDs → 无覆盖硬链接到 HF canonical cache。",
        "只为实际 bitwise 相同对象使用官方 huggingface_hub 本地 commit/etag metadata；旧 HF partial 不删除、不覆盖，HF 后续直接复用 archive，不再次请求该文件 payload。",
        "ODL provenance 保存为独立 sidecar，不并发覆盖正在运行 supervisor 的 SA1B_SHARDS.json。",
        "ODL audit/staging 不冒充 training images 已恢复；现有 HF supervisor 到达该队列位置时仍会 checksum/tar/extract/完整存在性审计并更新 completeness。",
        "若 HF 接近、已有 bytes、速度低于门槛或错误过多，停止 ODL 分流并保留断点。", "",
        "| Shard | ODL 状态 | MiB/s | Audit passed |", "|---|---|---:|---|" ]
    for entry in split.get("assignments", []):
        lines.append(f"| {entry['filename']} | {entry['status']} | {entry.get('average_mib_s', 'pending')} | {entry.get('audit_passed', 'pending')} |")
    lines.extend(["", "## 剩余下载估算", "",
        f"正常队列未下载 bytes：{sum(remaining.values())}；规划 ODL 分区：{odl_remaining}；其余 HF 分区：{hf_remaining}。",
        f"按以上各自均速和目前有限分区，纯下载估计约 {eta} 小时；使用 max(HF分区时间, ODL分区时间)，不能误用全部剩余量 / 联合速度。",
        "此估算不是全部数据 READY 的承诺：不包含 checksum/tar/解压/全量decode或三个异常 shard 的修复；后续 ODL protocol failure、账号过期、HF 接近和吞吐波动都会改变分区。", "",
        "## 隔离 shard 独立镜像探测", "",
        "对 000014/000016/000017 逐个执行了授权下载检查和独立 CDN 1MiB HTTP Range；副本保存到 authorized-anomaly-prefixes，不覆盖 HF evidence。",
        "三个文件的发布大小/SHA256 和实际首1MiB均与 HF 异常版本一致。没有实际第二完整 archive，通过前缀不能宣称完整 bitwise/tar通过。",
        "如果发布 SHA 准确，对相同对象换域名重下不能修复压缩包损坏，因此没有无意义地再下载三个11GB坏包。修复仍未证明，需有效不同原始 archive 或可恢复成员/原图补齐方案。", "",
        "| Shard | 发布对象匹配HF | 实际前缀匹配HF | 修复证明 |", "|---|---|---|---|" ])
    for entry in anomalies.get("probes", []):
        lines.append(f"| {entry['filename']} | {entry.get('published_object_matches_invalid_hf')} | {entry.get('prefix_matches_hf')} | 未证明 |")
    lines.extend(["", "## 证据", "",
        "`recovery/evidence/sa1b-opendatalab/authorized-download-check.json`：授权检查，不含密钥。",
        "`recovery/evidence/sa1b-opendatalab/authorized-probe/`：603秒样本、字节计数、status/CDN、完整 archive/JPEG audit。",
        "`recovery/evidence/sa1b-opendatalab/split-recovery/state.json`：逐 shard 进度，十秒级更新。",
        "`recovery/evidence/sa1b-opendatalab/authorized-anomaly-prefixes/results.json`：三个独立前缀探测。",
        "当前 HF job没有停止或重启；没有训练或smoke授权。建议本次恢复任务结束后轮换曾在聊天提供的 AK/SK。"])
    (recovery.RECOVERY / "SA1B_OPENDATALAB_BENCHMARK.md").write_text("\n".join(lines) + "\n")
    summary = dict(updated_utc=recovery.now(), authorization_passed=authorization.get("download_check_passed"),
                   repository="OpenDataLab/SA-1B", dataset_id=6248, test_shard="sa_000038.tar",
                   odl_average_mib_s=rate, hf_average_mib_s=hf_rate, protocol_audit=audit,
                   split=split, pure_download_estimate_hours=eta,
                   remaining_normal_download_bytes=sum(remaining.values()),
                   anomaly_repair_verified=False, hf_supervisor_untouched=True, training_started=False, smoke_started=False)
    recovery.atomic_json(folder / "authorized-assessment.json", summary)
    return summary


if __name__ == "__main__":
    summary = render()
    print(json.dumps({key:summary[key] for key in ("updated_utc", "authorization_passed", "odl_average_mib_s",
        "hf_average_mib_s", "pure_download_estimate_hours", "remaining_normal_download_bytes")}))
