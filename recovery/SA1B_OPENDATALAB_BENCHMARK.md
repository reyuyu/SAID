# SA-1B OpenDataLab / OpenXLab benchmark

更新：2026-10-05T03:52:26.290580+00:00。Repository：`OpenDataLab/SA-1B`；dataset ID：`6248`。
HF 四路 supervisor process group 24215 / Python PID 24219 保持运行，本次没有停止、重启或修改该运行进程。没有启动 smoke 或正式训练。

## 授权与原始清单

官方 AK/SK 登录成功：True; SA-1B download check 通过：True。
凭据只由官方 SDK 保存到 `/root/.openxlab`（目录 0700、config/token 0600），没有写入项目、下载 provenance 或报告。
官方公共 metadata 分页确认 1000 个原命名 tar；所需前 51 个文件大小和公开 SHA256 全部与固定 HF revision 140d15308aff47dae3b00214083838d56a4319c6 一致。
工具位于独立 `.opendatalab-venv`：opendatalab 0.0.10 / openxlab 0.1.3；训练和 HF 下载环境未增加依赖。

## 十分钟单包 benchmark

选择 `/raw/sa_000038.tar`：测试前 HF 尚未开始且零 partial；000030 已有此前 HF benchmark partial，故没有选它。
测量窗口：603.0843242034316 秒；完整文件：True；SDK 任务时间：2026-10-05T03:29:23.641715+00:00–2026-10-05T03:38:22.063113+00:00。
实际图片 payload 传输时间约 430.5404845997691 秒。若在十分钟内完成，后续包含组装及空闲，保守窗口均速没有伪称为连续传输了十分钟，也没有重下已完成文件。
只使用官方 `openxlab.dataset.download` 单文件调用。每个 shard SDK 默认 8 个 HTTP Range，不等于 8 个 shard 并发。

| 指标 | 测量结果 |
|---|---|
| ODL 保守全窗口平均 MiB/s | 17.826657926134846 |
| ODL 5秒采样峰值 MiB/s | 33.65325442966454 |
| 同窗口 HF 四路 MiB/s | 5.206568756611919 |
| 联合平均 MiB/s | 23.033226682746765 |
| 联合 / 原五分钟 HF 5.300MiB/s | 4.346060910415774 倍 |
| HTTP / stream errors | 0 |
| retry warnings | 0 |
| HTTP status counts | `{"200": 3, "206": 13, "302": 1}` |
| Range 206 实际观察 | True |
| 完整中断后重启恢复测试 | 尚未执行；Range 支持不是完整重启恢复验证 |
| 实际 CDN object 路径（无签名参数） | `["cdn-xlab-data.openxlab.org.cn/objects/0bdbf7ad845a66b0c295715a3911d31d32f224ab14c2afa16c3cb2947fb2a944"]` |

该路线实际走 OpenXLab CDN HTTP Range，不使用 HF Xet；没有进行无效的 Xet 开关对比。

## 协议审计

分类：`BITWISE_EQUIVALENT_MIRROR`；protocol_passed：True。
大小：11273217453；实际 MD5：`4723cf9f132e2f1305bff596c32ae0f1`；与 checklist 一致：True。
实际 SHA256：`0bdbf7ad845a66b0c295715a3911d31d32f224ab14c2afa16c3cb2947fb2a944`；GNU tar integrity：True。
JPEG数：11186；完整解码通过数：11186；失败数：0。
原 image ID 范围：425144–436332。
冻结 training index 在该 ID 范围所需图片：11167/11167；缺失：0。
保留原始 basename 和编码字节；未 resize/recompress/rename/format conversion，mask JSON 未安装到训练目录。
实际全量解码、逐 shard 必需 ID 集合和前16个 JPEG byte hash/尺寸记录在 full-audit/audit.json。

## 分流与无覆盖交接

独立 ODL 分流状态：True；ODL shard 并发：1；HF shard 并发仍为4。
初始速度 >=8MiB/s、联合提升 >=30%、完整 protocol audit 三个 gate 都通过才启用。后续每包保持速度/错误/联合增益检查。
只考虑 000039–000050 中 HF 尚未开始、零 partial、距离 HF 当前活动 shard 至少8个位置的未来文件。000038 benchmark 文件仅在完整验证通过后交接。
每包独立路径 → 固定 size/MD5/SHA → tar integrity → 全JPEG解码 → 冻结 training-index IDs → 无覆盖硬链接到 HF canonical cache。
只为实际 bitwise 相同对象使用官方 huggingface_hub 本地 commit/etag metadata；旧 HF partial 不删除、不覆盖，HF 后续直接复用 archive，不再次请求该文件 payload。
ODL provenance 保存为独立 sidecar，不并发覆盖正在运行 supervisor 的 SA1B_SHARDS.json。
ODL audit/staging 不冒充 training images 已恢复；现有 HF supervisor 到达该队列位置时仍会 checksum/tar/extract/完整存在性审计并更新 completeness。
若 HF 接近、已有 bytes、速度低于门槛或错误过多，停止 ODL 分流并保留断点。

| Shard | ODL 状态 | MiB/s | Audit passed |
|---|---|---:|---|
| sa_000039.tar | DOWNLOADING_ODL | 23.81497813516911 | pending |

## 剩余下载估算

正常队列未下载 bytes：329066875784；规划 ODL 分区：135399205540；其余 HF 分区：193667670244。
按以上各自均速和目前有限分区，纯下载估计约 9.85378554496338 小时；使用 max(HF分区时间, ODL分区时间)，不能误用全部剩余量 / 联合速度。
此估算不是全部数据 READY 的承诺：不包含 checksum/tar/解压/全量decode或三个异常 shard 的修复；后续 ODL protocol failure、账号过期、HF 接近和吞吐波动都会改变分区。

## 隔离 shard 独立镜像探测

对 000014/000016/000017 逐个执行了授权下载检查和独立 CDN 1MiB HTTP Range；副本保存到 authorized-anomaly-prefixes，不覆盖 HF evidence。
三个文件的发布大小/SHA256 和实际首1MiB均与 HF 异常版本一致。没有实际第二完整 archive，通过前缀不能宣称完整 bitwise/tar通过。
如果发布 SHA 准确，对相同对象换域名重下不能修复压缩包损坏，因此没有无意义地再下载三个11GB坏包。修复仍未证明，需有效不同原始 archive 或可恢复成员/原图补齐方案。

| Shard | 发布对象匹配HF | 实际前缀匹配HF | 修复证明 |
|---|---|---|---|
| sa_000014.tar | True | True | 未证明 |
| sa_000016.tar | True | True | 未证明 |
| sa_000017.tar | True | True | 未证明 |

## 证据

`recovery/evidence/sa1b-opendatalab/authorized-download-check.json`：授权检查，不含密钥。
`recovery/evidence/sa1b-opendatalab/authorized-probe/`：603秒样本、字节计数、status/CDN、完整 archive/JPEG audit。
`recovery/evidence/sa1b-opendatalab/split-recovery/state.json`：逐 shard 进度，十秒级更新。
`recovery/evidence/sa1b-opendatalab/authorized-anomaly-prefixes/results.json`：三个独立前缀探测。
当前 HF job没有停止或重启；没有训练或smoke授权。建议本次恢复任务结束后轮换曾在聊天提供的 AK/SK。
