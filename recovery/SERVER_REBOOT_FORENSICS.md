# SAID server reboot forensics

取证日期：2026-10-06（Asia/Shanghai；本文内核/文件时间若未另注均为 UTC）

## 最终结论

`REBOOT_CAUSE_UNDETERMINED`

没有直接 kernel/journal 证据证明 SAID 导致了 reboot，也没有直接证据证明是外部原因。更准确地说，本环境看到的是 **Kubernetes pod 被重建**，而不是宿主 Linux 在该时段重启：

- 宿主当前 boot 起点为 `2025-04-25 14:59:59 UTC`（`2025-04-25 22:59:59 Asia/Shanghai`），boot ID 为 `07af9760-3653-412b-bf96-d9b0698d102f`，取证时 `/proc/uptime` 为 `45668600s`（约 528.6 天）。
- 当前 pod PID 1 起始时间为 `2026-10-06 01:53:00 UTC`（`09:53:00 Asia/Shanghai`）。这不是 kernel boot 时间。
- journald 没有任何可读 journal，`journalctl --list-boots` 返回 `No journal files were found.`；因此无法确认上一 host boot 的结束时间，也无法读取 `-b -1` 的 kernel/system journal 或真正的 reboot 前 15 分钟原始 journal。
- `wtmp` 已失真/过旧：`who -b` 无输出，`last -x | head -30` 只有 2021 年会话，不能用于本次事件。

SAID 与 pod 重建时间接近，但仅能确认当时正在 NFS 到本地 SSD 的复制，不能由时间邻近推出因果。

## 1. Boot 与容器时间线

原始命令摘要：

```text
$ uptime -s
2025-04-25 14:59:59

$ who -b
<no output>

$ last -x | head -30
... only three 2021 login records ...
wtmp begins Wed Jun  9 09:36:39 2021

$ journalctl --list-boots --no-pager
No journal files were found.
```

可确认的边界：

| 对象 | 时间 (UTC) | 时间 (Asia/Shanghai) | 证据 |
|---|---:|---:|---|
| 当前 host boot 开始 | 2025-04-25 14:59:59 | 2025-04-25 22:59:59 | `uptime -s`, `/proc/uptime`, boot ID |
| 上一 host boot 结束 | 无法确认 | 无法确认 | 上一 boot journal/wtmp 均不可用 |
| SAID full copy 最后一条进度 | 2026-10-06 01:35:18 | 2026-10-06 09:35:18 | `local-copyfull.log`, `LOCAL_SSD_COPY_PROGRESS.json` mtime |
| 可见 Calico link-ready 事件 | 2026-10-06 01:50:27 | 2026-10-06 09:50:27 | 当前 host ring buffer；不能单独认定为旧 pod 结束或本 pod 开始 |
| 当前 pod PID 1 开始 | 2026-10-06 01:53:00 | 2026-10-06 09:53:00 | `ps -p 1`, `/proc/1`, `/etc/hostname` |

## 2. Kernel / journal 证据边界

上一 boot 的以下命令均无法取得数据，因为没有 journal 文件：

```text
journalctl -b -1 -k --no-pager
journalctl -b -1 --no-pager
```

作为受限替代，保存了当前 host boot 的 kernel ring buffer 最后 300 行未过滤上下文：

- `recovery/SERVER_REBOOT_KERNEL_CONTEXT_LAST300.txt`（恰好 300 行）

该文件不是上一 boot 日志，不能填补缺失的 reboot 前 journal。当前 ring buffer 在 `2026-10-05 05:47:33 UTC` 后直到 `2026-10-06 01:50:27 UTC` 没有记录；尤其在旧 pod 最后 SAID 进度附近没有 OOM、panic、lockup、hung task、NFS、filesystem 或 block-device 事件。

全 ring buffer 关键词扫描包含大量更早的、宿主其他 workload 的历史事件，不能归因于 SAID。事件窗口内未发现：

- `kernel panic`, `soft lockup`, `hard lockup`, `hung task`, `blocked for more than`
- `Out of memory`, `Killed process`, oom-killer invocation
- `NFS server not responding`, RPC timeout
- filesystem error、`I/O error`、NVMe error、PCIe/AER error

## 3. OOM 判断

**没有证据表明本次事件由 host OOM 或 SAID cgroup OOM 导致。**

- ring buffer 中最后一次可见 OOM 是 `2026-09-06 04:58:15 UTC`，被杀进程为 `lerobot-eval`（PID `3842860`，anon RSS `89510908 kB`）。它比本次 pod 重建早约 30 天，不是 SAID 本次任务。
- 事故前本项目已有资源报告在 `2026-10-06 00:13:18 UTC` 记录：`memory_current=497511981056` bytes，`memory.max=536870912000` bytes。
- 同一记录将内存拆分为 file/page cache `481803964416` bytes 和 RSS `2412883968` bytes；主体是文件缓存，不是进程真实 RSS。
- 同一记录明确为 `oom_kill=0`, `under_oom=0`，虽然累计 limit hits 为 `19288505`。limit hits 或接近 500 GiB 不等于 OOM kill。
- 旧 pod cgroup 随 pod 销毁，无法在当前 pod读取其最终 counters。当前新 pod 的 `memory.failcnt=0`, `oom_kill=0`, 最大使用约 `26.1 GB` 只描述新实例，不可倒推旧实例。

因此不得将“接近 500 GiB”表述为 OOM 导致 reboot。

## 4. GPU / driver

当前 `nvidia-smi -q` 成功，driver `550.127.05`，CUDA `12.4`，四张分配 GPU 均为 A100 80GB PCIe：

| GPU | PCI | 显存使用 | 温度 | GPU util | volatile ECC corrected/uncorrected |
|---:|---|---:|---:|---:|---:|
| 0 | `34:00.0` | 1 MiB | 30-31 C | 0% | 0 / 0 |
| 1 | `35:00.0` | 1 MiB | 31-32 C | 0% | 0 / 0 |
| 2 | `36:00.0` | 1 MiB | 30 C | 0% | 0 / 0 |
| 3 | `37:00.0` | 1 MiB | 30 C | 0% | 0 / 0 |

四卡均无进程、无 reset required、无 row-remap pending/failure。

宿主 ring buffer 有 Xid 31，但事故前最近一组在 `2026-10-04 13:28-13:29 UTC`，事故后另有一组在 `2026-10-06 02:51 UTC`；PCI 地址为 `9c:00`-`9e:00`，不是当前 pod 的 `34:00`-`37:00` 四卡。没有 `GPU has fallen off the bus` 或 SXid。它们不能证明当前四卡导致本次 pod 重建。

## 5. NFS / filesystem

项目位于 NFSv3：`192.168.210.100:/mnt/fs1/...`，取证时正常挂载，容量约 828T、使用 32%。

- ring buffer 确有严重 NFS timeout 历史：`2026-10-05 04:47:32`、`04:47:42`、`05:38:14`、`05:47:33 UTC` 出现 `server ... not responding` 和大量 suppressed RPC callbacks。
- 最后一条 NFS timeout 比当前 pod PID 1 启动早约 20 小时，比 full copy 最后进度早约 19 小时 48 分；不能直接解释本次 pod 重建。
- 事件窗口没有 NFS/RPC、hung/blocked task、filesystem、NVMe/local disk 或 I/O error 记录。
- 由于缺少 journald，仍不能排除 ring buffer 未记录或已丢失的存储/平台事件。

结论：历史 NFS stall 与已知 `data_wait` 问题一致，但没有证据显示本次事件前出现更严重的 kernel/NFS hang。

## 6. SAID 最后任务

事故前最后正式轨迹为 `armb_summary02_500gate_localssd_v3`：

1. 500-step training：supervisor PID `59920`；四个训练 rank PID `62292`-`62295`。step 500 在 `2026-10-06 00:20 UTC` 完成，峰值单卡 allocated GPU memory `27.76 GiB`。
2. 五组评估：最后 Long-DCI 评估在 `00:46:22 UTC` 完成，并通过 reproduction gate。
3. NFS -> local SSD full copy：active PID `65566`，16 workers；state 自报 `stage_started_at=00:59:32 UTC`，最后进度文件 mtime 为 `01:35:18 UTC`，已检查并 SHA 匹配 `496640 / 1245901`，约读取 `491.0 GB` payload、实际新写 `289.0 GB`。累计 elapsed 与文件 mtime 对精确开始时刻存在内部偏差，因此不把 `00:59:32` 当作独立确认的墙钟时间。
4. state 明确记录 `training_process_running=false`、`stage=local-copyfull`。没有证据表明事故时正在训练或大规模 decode；当时最可能正在 copy/staging。

复制日志无完成记录，且在 pod 重建前约 18 分钟停止更新。这个时间关系只证明两者同时段发生，不能证明 copy 导致 pod/host 事件。

## 7. 数据安全

以下目录均存在：

- `local_assets/training/ShareGPT4V/sam/images`
- `local_assets/training/ShareGPT4V/coco/train2017`
- `local_assets/training/ShareGPT4V/llava/llava_pretrain/images`
- common step0：`runtime/SAID-nest-clip-v1/shared/step000000.pt`、`student_step0.pt` 与 provenance JSON 均存在

只读取了现有 `recovery/FINAL_IMAGE_AUDIT.json`：状态 `DATA_AUDIT_PASS`，`1245901/1245901` 存在且此前全部 decode 通过，missing/corrupt/I/O error 均为 0；没有重跑全量 audit。

额外抽样使用已有 `local-image-path-proof.json` 的前 100 条代表性训练路径，映射回 NFS 后执行 stat、PIL verify 和完整 load：

```json
{"checked":100,"families":{"llava":46,"sam":44,"coco":10},"bytes_read_approx":55100247,"errors":[]}
```

NFS mount 在该 100 张样本检查中工作正常。

## 8. 判定依据与后续限制

支持 `REBOOT_CAUSE_UNDETERMINED` 的决定性依据：

- 没有上一 boot journal，也无法确认上一 host boot 结束时间。
- host uptime/boot ID 表明事件时 host kernel 并未重启；可确认的是 pod 生命周期重建。
- 事故窗口没有 kernel OOM/panic/lockup/hung/NFS/I/O/GPU 直接证据。
- 事故前 cgroup 接近 500 GiB，但约 482 GB 是 file cache，RSS 约 2.4 GB，`oom_kill=0`。
- SAID copy PID 在事故前活跃，但只有时间相关性，没有机制或 kernel/platform 证据建立因果。
- 平台侧 pod termination reason、Kubernetes events、node journal/BMC/IPMI 日志在当前环境不可见。

在获得平台侧证据前，不应启动后续 500-step/4868-step 训练、数据搬运或大规模 audit。要进一步定因，需由有宿主/集群权限的管理员提供旧 pod termination reason、Kubernetes events、node journal（含事件前后至少 15 分钟和最后 300 行原始上下文）以及 BMC/IPMI reset/power/watchdog 记录。
