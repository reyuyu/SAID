# RESOURCE_STALL_DIAGNOSIS_V2

Status: **RESOURCE_GATE_READY_FOR_500**

One fresh common-step0 replay stopped at40 optimizer updates. No500/full trajectory was started. The native3-second x3 consecutive protection is unchanged.

## Diagnosis and limits

The historical step31–33 stall was **not reproduced**. Its originating phase cannot be assigned retrospectively: that run lacked phase timings. Current measured steady-state data wait, H2D, backward/DDP and optimizer durations are normal; there is no evidence that NFS is the primary stall source.

Observation: `RESOURCE_STALL_NOT_REPRODUCED_WITHOUT_CACHE_RECLAIM`. Do not label this AFTER_CACHE_RECLAIM: no deliberate reclaim occurred. The readiness status certifies this40-update resource gate, not a root-cause resolution or guaranteed500-update throughput.

No data was moved: local SSD staging is not presently justified by measured data_wait. If a future authorized replay shows data_wait-dominated delays, stage the deterministic first500 image manifest using original bytes; never alter sampling/preprocessing.

A40-update CappedSampler naturally ends its prefetch queue earlier than a500-update loader; unchanged workers8/rank and identical first40 sample IDs do not eliminate that lifecycle difference. Cache pressure remains a residual risk.

## Audit workers and cgroup

Preflight and postflight process-tree plus /proc/PID/fd inspections found no residual final-decode audit or training workers, and no open training/audit image files in other inspected processes. lsof is unavailable; /proc evidence is retained. No unrelated process was killed and no image was deleted.

Cgroup v1 memory controller is mounted read-only. memory.current/max/events are unavailable under those v2 names; equivalent v1 usage_in_bytes/limit_in_bytes/stat/oom_control/failcnt values are normalized in JSON. There is no writable memory.reclaim. No global drop_caches or alternative system-level cache action was attempted.

| Metric | Before replay GiB | After replay GiB |
|---|---:|---:|
| memory_current | 463.010 | 458.479 |
| file | 450.678 | 445.931 |
| inactive_file | 42.686 | 22.231 |
| active_file | 407.989 | 423.697 |
| anon | 0.988 | 1.208 |

Limit: 500.0GiB. Replay peak current: 499.995GiB. v1 failcnt delta: 32095; OOM kills delta: 0; CPU quota throttled periods delta: 0.
Host memory PSI some delta: 28751us; IO PSI some delta: 1431648us. These are host-wide, **not current-cgroup PSI**. Reclaim-like allocation pressure existed; the counter alone does not identify which phase stalled.

## Replay metrics

| Window | Median s | p95 s | Max s |
|---|---:|---:|---:|
| All40, including startup/control gate | 2.072228 | 3.728177 | 27.180500 |
| Steady updates7–40 | 2.057229 | 2.234137 | 2.308884 |
| Focus updates31–40 | 2.061635 | 2.174645 | 2.230377 |

Step1 full_cycle=27.180500s includes DataLoader startup (rank0 data_wait=23.846660s). Step6 full_cycle=23.639568s includes the preserved first-five checkpoint/hard-invariant control gate (rank0 gate=21.325841s). Neither was removed from the native3x3 guard. They are isolated startup/control events, not the historical step31–33 failure.

Three consecutive full cycles>3s triggered: **False**. All40 updates/losses/gradients finite; first-five sample IDs/F/S/D strings/token hashes/LRs/config/source hashes and AdamW counter5 passed. All40960 sample IDs match the frozen DistributedSampler. All four rank parameter differences at5 and40 are0. Peak allocated GPU memory=27.759881GiB/card; final NCCL all-reduce=10.0.

### Phase timing (four-rank maximum per update, updates7–40)

| Phase | Median s | p95 s | Max s |
|---|---:|---:|---:|
| data_wait_s | 0.000345 | 0.000892 | 0.017520 |
| h2d_s | 0.023725 | 0.024024 | 0.024037 |
| forward_s | 0.516769 | 0.633538 | 0.781297 |
| backward_s | 1.360214 | 1.372356 | 1.374370 |
| optimizer_s | 0.011669 | 0.015354 | 0.104778 |
| gradient_checks_wall_s | 0.021935 | 0.028011 | 0.031068 |
| ddp_sync_s | 0.023931 | 0.132335 | 0.145766 |

CUDA events measure stream elapsed time, including host launch gaps, not exclusive GPU kernel time. Backward includes DDP reducer/NCCL and cannot safely be split further without intrusive hooks. ddp_sync_s measures explicit outside-model collectives only; these overlap wall phases and must not be summed as independent costs. All phases retain wall-time fields. Per-rank/per-step cgroup/PSI/GPU utilization samples and2-second system monitoring are preserved; no RNG calls or added CUDA synchronizations occur in telemetry.

## Local storage

Project/training images: NFS192.168.210.100:/mnt/fs1/... mounted at /opt/data/private. /tmp and root overlay have local NVMe backing (nvme2n1p2 ext4, also exposed via /tmp/nvidia-mps). Other visible NVMe devices are not available mounted storage; no mount/format operation was performed. /dev/shm is RAM, not staging SSD.

Local free=681.937GiB. Existing install inventories plus ZIP central directories report image payload=568.166GiB; expected count coverage=True. Full raw image payload fits with64GiB reserve=True. No recursive image enumeration, full data audit, image download or staging was performed.

- SAM: 569486/569486 required images represented in inventories; 524.330GiB payload. Inventory extras=920 are excluded from the estimate.
- COCO: 118287/118287 required images represented in inventories; 17.988GiB payload. Inventory extras=0 are excluded from the estimate.
- LLAVA: 558128/558128 required images represented in inventories; 25.848GiB payload. Inventory extras=0 are excluded from the estimate.

## Frozen initialization and isolation

Common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`. Start_updates=0; no resume argument. F/S/D=[1.4,0.2,1.4], horizon4868, batch256/rank, accumulation1, workers8/rank, original sampling/model/loss/optimizer unchanged. The original source hashes still match the passed smoke.

The native CLI uses run-type formal solely to retain the original protection; --max-updates40 and a second scheduler hook enforce the diagnostic cap. Dedicated diagnostic output is separate from formal trajectories. Diagnostic checkpoints5/40 were quarantined and marked DO_NOT_RESUME; checkpoint bytes are not committed to GitHub.

Runtime evidence: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/resource-stall-v2-20261005T184030Z`.
Local telemetry: `/tmp/said-resource-stall-v2/20261005T184030Z` (copied into runtime after completion).

CPU tests:71 passed (65 existing gate tests +6 resource-diagnostic tests). Training exited0; acceptance passed. All training/audit workers exited and GPUs are idle after the run.

Next action requires a new user instruction: start a fresh common-step0 500-step reproduction trajectory with horizon4868 and unchanged resource protection. Never resume diagnostic40 or stopped33. No500/full was automatically started.
