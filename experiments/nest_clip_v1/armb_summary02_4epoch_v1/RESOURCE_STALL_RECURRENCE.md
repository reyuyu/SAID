# S02 resource protection stop

**RESOURCE_STALL_RECURRED_BEFORE_500**

Stopped at optimizer update55; horizon4868. Native3-second x3 consecutive gate unchanged.

This is a resource/environment stop, not a retrieval/method failure. No automatic retry or continuation. Last20 full cycles and per-rank data/H2D/forward/backward-DDP/optimizer/cgroup/file-cache/PSI/GPU samples are in RESOURCE_STALL_RECURRENCE.json.

## Measured trigger and phase interpretation

| Step | Full cycle s | Maximum data wait s | Backward/DDP s range | Optimizer s range |
|---:|---:|---:|---:|---:|
| 53 | 3.410532 | 1.412658 | 1.353705–1.355840 | 0.010841–0.011158 |
| 54 | 6.930750 | 4.798252 | 1.361715–1.367387 | 0.011139–0.014049 |
| 55 | 14.281010 | 12.190608 | 1.358574–1.360280 | 0.010878–0.012850 |

At55, rank1 waited12.190608s for data, then its forward took0.513110s. Other ranks waited less for data but longer in forward: data_wait+forward was12.7018–12.7037s across all four ranks. This is strong evidence of **data supply starvation followed by rank synchronization wait**, not an increase in intrinsic GPU backward/optimizer compute. CUDA forward spans include collective waiting; do not misclassify the inflated forward timings as additional model compute.

The55-update run used a500-update loader; unlike the capped40-update diagnostic, its prefetch queue did not end around40. Passing the40-update diagnostic did not establish sustained500-update throughput.

At55, cgroup memory.current=499.988GiB out of500GiB; normalized v1 file/cache=462.237GiB, inactive_file=20.158GiB, active_file=425.743GiB, anon=26.260GiB. v1 failcnt increased8682 during55. Host IO PSI some total increased3.456746s during that cycle. These observations support memory/IO pressure; **they do not yet isolate NFS latency from CPU/cache-reclaim/DataLoader preparation**. PSI here is host-wide, not current-cgroup PSI. v1 cache is not equivalent to clean reclaimable inactive disk pages.

All55 losses and reported gradients remained finite. First-five IDs/F/S/D/token hashes/LRs/optimizer groups/horizon and AdamW counters passed; rank parameter differences at the first-five gate were zero. GPU peak remained27.759881GiB/card.

The native protection checkpoint step000055.pt was saved before a secondary DataLoader worker SIGABRT during early teardown. That teardown error is not the originating stall. No step56 was executed, no step500 retrieval evaluation occurred, and no4868 continuation or retry was launched.

The full20-update evidence window is steps36–55, with all four rank timing records present. Training/audit workers have exited and all four GPUs are idle. No image was deleted, recopied, reencoded or downloaded.

Next remediation requires a new instruction. A deterministic first500 image local-NVMe mirror and worker-side IO/decode/reclaim attribution are candidates; no dataset, worker count, model or objective change is authorized by this stop report.

## Read-only protection checkpoint audit

Saved `step000055.pt` SHA256: `d390a2c4a5e5b49f7c8b31f6eabf90dba2d629913a6ea7c1f799b15857626ed9`.

Audit passed: model/adapter/AdamW tensors finite; optimizer counters55; global_step55; scheduler horizon4868; all four CPU/CUDA/Python/NumPy and loader RNG states present; sampler/data cursor(next_epoch0,next_batch55). Common step0 origin and frozen native source hashes verified. The checkpoint was not modified or uploaded; resume is not authorized.

Current resource readiness is revoked: `RESOURCE_GATE_NOT_READY_FOR_500`. No500/4868 training is running. The historic40-step pass remains a bounded diagnostic observation, not proof that this longer loader can sustain throughput.
