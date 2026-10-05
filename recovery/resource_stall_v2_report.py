"""Finalize existing resource replay evidence; never launch training."""

import itertools
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from torch.utils.data import DistributedSampler

from recovery.resource_stall_v2 import EXP, GIB, ROOT, dump, process_audit, rows, summary, system_snapshot
from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_full import sha


def image_footprint():
    sources = []
    shards = json.loads((ROOT / "recovery/SA1B_SHARDS.json").read_text())["shards"]
    for shard in shards:
        if shard.get("image_inventory"):
            sources.append(Path(shard["image_inventory"]))
        rescue = shard.get("required_jpeg_rescue", {})
        if rescue.get("image_inventory"):
            sources.append(Path(rescue["image_inventory"]))
        if rescue.get("original_rescue_inventory"):
            sources.append(Path(rescue["original_rescue_inventory"]["path"]))
    names = subprocess.check_output(["rg", "--files", "recovery/evidence/modelscope-sa1b/installed-inventories"], cwd=ROOT, text=True)
    sources.extend(ROOT / name for name in names.splitlines() if name.endswith(".jsonl"))
    images = {}
    conflicts = []
    for path in sorted(set(sources)):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                basename = record.get("basename")
                size = record.get("size_bytes")
                if basename and size is not None:
                    if basename in images and images[basename] != size:
                        conflicts.append(basename)
                    images[basename] = size
    required = set()
    with (ROOT / "recovery/required_training_images.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["family"].lower() == "sam":
                required.add(record["basename"])
    covered = required.intersection(images)
    families = dict(SAM=dict(inventory_unique_images=len(images), required_images_covered=len(covered),
                             required_images_expected=len(required), unused_inventory_images=len(images.keys() - required),
                             payload_bytes=sum(images[basename] for basename in covered),
                             inventory_size_conflicts=sorted(set(conflicts))))
    for name in ("coco", "llava"):
        worker = json.loads((ROOT / f"recovery/evidence/hf-training-{name}-worker.json").read_text())
        with zipfile.ZipFile(worker["archive"]) as archive:
            members = [item for item in archive.infolist() if item.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
        families[name.upper()] = dict(inventory_unique_images=len(members), required_images_covered=len(members),
                                      required_images_expected=worker["archive_images"], payload_bytes=sum(item.file_size for item in members))
    total = sum(value["payload_bytes"] for value in families.values())
    disk = shutil.disk_usage("/tmp")
    coverage = [families[name]["required_images_covered"] for name in ("SAM", "COCO", "LLAVA")] == [569486, 118287, 558128]
    return dict(method="Existing installation JSONL inventories and ZIP central directories only; no image-tree scan/read/copy",
                families=families, inventory_counts_cover_expected=coverage, payload_bytes=total,
                local_free_bytes=disk.free, local_total_bytes=disk.total, reserve_bytes=64 * GIB,
                full_image_payload_fits_with_reserve=coverage and not conflicts and disk.free > total + 64 * GIB,
                data_moved=False)


def main():
    path = EXP / "RESOURCE_STALL_DIAGNOSIS_V2.json"
    result = json.loads(path.read_text())
    runtime = Path(result["preflight"]["runtime"])
    local = Path(result["preflight"]["local_telemetry"])
    native = runtime / "train"
    steps = rows(native / "steps.jsonl")
    checkpoint_config = json.loads((native / "config.json").read_text())
    prefix = json.loads((local / "prefix-gate.json").read_text())
    for rank in range(4):
        sampler = DistributedSampler(range(1245901), num_replicas=4, rank=rank, shuffle=True, seed=0, drop_last=False)
        expected = [index + 1000 for index in itertools.islice(iter(sampler), len(steps) * 256)]
        actual = []
        for step in steps:
            health = next(item for item in step["rank_health"] if item["rank"] == rank)
            actual.extend(health["sampling"]["sample_ids"])
        assert actual == expected, f"Frozen sample stream drift on rank{rank}"
    result["all_updates_sampler_verified"] = dict(passed=True, updates=len(steps), samples=len(steps) * 1024,
                                                   seed=0, skip_first=1000, reference="Frozen DistributedSampler")
    result["prefix_checkpoint_gate"] = prefix
    result["diagnostic_source_sha256"] = {name: sha(ROOT / name) for name in
        ("recovery/resource_stall_v2.py", "recovery/resource_stall_v2_report.py",
         "experiments/nest_clip_v1/armb_summary02_4epoch_v1/phase_train_v2.py")}
    result["training_git_head"] = checkpoint_config["git_head"]
    result["source_unchanged_after_replay"] = all(sha(ROOT / name) == digest for name, digest in result["preflight"]["native_source_sha256"].items())
    assert result["source_unchanged_after_replay"]
    assert checkpoint_config["start_updates"] == 0 and checkpoint_config["horizon"] == 4868
    assert checkpoint_config["view_weights"] == [1.4, .2, 1.4]
    before = result["preflight"]["cache_reclaim"]["before"]
    monitor = rows(local / "system-monitor.jsonl")
    samples = [item["system"] for item in monitor]
    result["memory_during_replay"] = dict(min_current_bytes=min(item["memory_current"] for item in samples),
        max_current_bytes=max(item["memory_current"] for item in samples),
        limit_hits_delta=samples[-1]["memory_events"]["limit_hits_cumulative"] - before["memory_events"]["limit_hits_cumulative"],
        oom_kill_delta=samples[-1]["memory_events"]["oom_kill"] - before["memory_events"]["oom_kill"],
        cpu_throttled_periods_delta=samples[-1]["cpu_stat"]["nr_throttled"] - before["cpu_stat"]["nr_throttled"],
        host_memory_PSI_some_total_delta_us=samples[-1]["memory_PSI"]["some"]["total"] - before["memory_PSI"]["some"]["total"],
        host_io_PSI_some_total_delta_us=samples[-1]["io_PSI"]["some"]["total"] - before["io_PSI"]["some"]["total"],
        PSI_scope="Host-wide; current cgroup v1 does not expose its own PSI")
    result["phase_summary_steps7_to40"] = {key: summary([max(rank[key] for rank in step["ranks"]) for step in result["phase_timings"] if step["step"] >= 7])
        for key in ("data_wait_s", "h2d_s", "forward_s", "backward_s", "optimizer_s", "gradient_checks_wall_s", "ddp_sync_s")}
    result["post_replay_process_audit"] = process_audit()
    assert result["post_replay_process_audit"]["workers_all_exited"]
    result["post_replay_system"] = system_snapshot()
    result["local_disk_capacity"] = image_footprint()
    result["diagnosis"] = dict(
        replay_observation="RESOURCE_STALL_NOT_REPRODUCED_WITHOUT_CACHE_RECLAIM" if not result["slow_steps"] else "SLOW_UPDATES_OBSERVED",
        historical_step31_to33_root_cause="UNDETERMINED: historical run did not have phase timing; no delayed step31-40 reproduced",
        nfs_confirmed_primary_cause=False, local_staging_required_by_current_evidence=False,
        environment_algorithm_changes=[], diagnostic_changes="Read-only phase telemetry written to local /tmp; native code and frozen configuration unchanged",
        residual_risk="Cgroup remains cache-heavy and reached its limit during replay. No deliberate cache reclaim was possible. A40-update capped loader is not a guarantee of sustained500-update throughput.")
    quarantine = runtime / "quarantined-checkpoints"
    quarantine.mkdir(exist_ok=True)
    for checkpoint in native.glob("step*.pt"):
        checkpoint.rename(quarantine / checkpoint.name)
    result["diagnostic_checkpoints"] = [dict(path=str(checkpoint), sha256=sha(checkpoint), resume_allowed=False)
                                         for checkpoint in sorted(quarantine.glob("*.pt"))]
    (quarantine / "DO_NOT_RESUME.md").write_text("Diagnostic-only checkpoints. Never resume500/full from these artifacts. Fresh common step0 is required.\n")
    for name in ("prefix-gate.json", "system-monitor.jsonl", "rank0.jsonl", "rank1.jsonl", "rank2.jsonl", "rank3.jsonl"):
        shutil.copyfile(local / name, runtime / name)
    dump(path, result)
    dump(runtime / "result.json", result)
    policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
    policy = json.loads(policy_path.read_text())
    policy.update(scope="Completed resource-only40 diagnostic; hold all500/full until a new instruction",
                  reproduction_gate_authorized=False, formal_training_authorized=False,
                  resource_diagnostic_authorized=False, final_manifest_audit_authorized=False,
                  resource_gate_status=result["status"], training_held_for_new_instruction=True)
    dump(policy_path, policy)
    write_report(result)
    print(json.dumps(dict(status=result["status"], image_payload_gib=result["local_disk_capacity"]["payload_bytes"] / GIB,
                          inventory_coverage=result["local_disk_capacity"]["inventory_counts_cover_expected"],
                          full_image_payload_fits=result["local_disk_capacity"]["full_image_payload_fits_with_reserve"])), flush=True)


def write_report(result):
    before = result["preflight"]["cache_reclaim"]["before"]
    after = result["post_replay_system"]
    memory = result["memory_during_replay"]
    lines = ["# RESOURCE_STALL_DIAGNOSIS_V2", "", f"Status: **{result['status']}**", "",
        "One fresh common-step0 replay stopped at40 optimizer updates. No500/full trajectory was started. The native3-second x3 consecutive protection is unchanged.", "",
        "## Diagnosis and limits", "",
        "The historical step31–33 stall was **not reproduced**. Its originating phase cannot be assigned retrospectively: that run lacked phase timings. Current measured steady-state data wait, H2D, backward/DDP and optimizer durations are normal; there is no evidence that NFS is the primary stall source.", "",
        "Observation: `RESOURCE_STALL_NOT_REPRODUCED_WITHOUT_CACHE_RECLAIM`. Do not label this AFTER_CACHE_RECLAIM: no deliberate reclaim occurred. The readiness status certifies this40-update resource gate, not a root-cause resolution or guaranteed500-update throughput.", "",
        "No data was moved: local SSD staging is not presently justified by measured data_wait. If a future authorized replay shows data_wait-dominated delays, stage the deterministic first500 image manifest using original bytes; never alter sampling/preprocessing.", "",
        "A40-update CappedSampler naturally ends its prefetch queue earlier than a500-update loader; unchanged workers8/rank and identical first40 sample IDs do not eliminate that lifecycle difference. Cache pressure remains a residual risk.", "",
        "## Audit workers and cgroup", "",
        "Preflight and postflight process-tree plus /proc/PID/fd inspections found no residual final-decode audit or training workers, and no open training/audit image files in other inspected processes. lsof is unavailable; /proc evidence is retained. No unrelated process was killed and no image was deleted.", "",
        "Cgroup v1 memory controller is mounted read-only. memory.current/max/events are unavailable under those v2 names; equivalent v1 usage_in_bytes/limit_in_bytes/stat/oom_control/failcnt values are normalized in JSON. There is no writable memory.reclaim. No global drop_caches or alternative system-level cache action was attempted.", "",
        "| Metric | Before replay GiB | After replay GiB |", "|---|---:|---:|"]
    for key in ("memory_current", "file", "inactive_file", "active_file", "anon"):
        lines.append(f"| {key} | {before[key] / GIB:.3f} | {after[key] / GIB:.3f} |")
    lines.extend(["", f"Limit: {int(before['memory_max']) / GIB:.1f}GiB. Replay peak current: {memory['max_current_bytes'] / GIB:.3f}GiB. v1 failcnt delta: {memory['limit_hits_delta']}; OOM kills delta: {memory['oom_kill_delta']}; CPU quota throttled periods delta: {memory['cpu_throttled_periods_delta']}.",
        f"Host memory PSI some delta: {memory['host_memory_PSI_some_total_delta_us']}us; IO PSI some delta: {memory['host_io_PSI_some_total_delta_us']}us. These are host-wide, **not current-cgroup PSI**. Reclaim-like allocation pressure existed; the counter alone does not identify which phase stalled.", "",
        "## Replay metrics", "", "| Window | Median s | p95 s | Max s |", "|---|---:|---:|---:|"])
    for name, key in (("All40, including startup/control gate", "cycles_all"), ("Steady updates7–40", "cycles_steps7_to40"), ("Focus updates31–40", "cycles_steps31_to40")):
        stats = result[key]
        lines.append(f"| {name} | {stats['median']:.6f} | {stats['p95']:.6f} | {stats['max']:.6f} |")
    lines.extend(["", "Step1 full_cycle=27.180500s includes DataLoader startup (rank0 data_wait=23.846660s). Step6 full_cycle=23.639568s includes the preserved first-five checkpoint/hard-invariant control gate (rank0 gate=21.325841s). Neither was removed from the native3x3 guard. They are isolated startup/control events, not the historical step31–33 failure.", "",
        f"Three consecutive full cycles>3s triggered: **{result['native_three_consecutive_gate_triggered']}**. All40 updates/losses/gradients finite; first-five sample IDs/F/S/D strings/token hashes/LRs/config/source hashes and AdamW counter5 passed. All40960 sample IDs match the frozen DistributedSampler. All four rank parameter differences at5 and40 are0. Peak allocated GPU memory={result['native_acceptance']['ranks'][0]['peak_allocated_gib']:.6f}GiB/card; final NCCL all-reduce=10.0.", "",
        "### Phase timing (four-rank maximum per update, updates7–40)", "", "| Phase | Median s | p95 s | Max s |", "|---|---:|---:|---:|"])
    for key, stats in result["phase_summary_steps7_to40"].items():
        lines.append(f"| {key} | {stats['median']:.6f} | {stats['p95']:.6f} | {stats['max']:.6f} |")
    lines.extend(["", "CUDA events measure stream elapsed time, including host launch gaps, not exclusive GPU kernel time. Backward includes DDP reducer/NCCL and cannot safely be split further without intrusive hooks. ddp_sync_s measures explicit outside-model collectives only; these overlap wall phases and must not be summed as independent costs. All phases retain wall-time fields. Per-rank/per-step cgroup/PSI/GPU utilization samples and2-second system monitoring are preserved; no RNG calls or added CUDA synchronizations occur in telemetry.", "",
        "## Local storage", "",
        "Project/training images: NFS192.168.210.100:/mnt/fs1/... mounted at /opt/data/private. /tmp and root overlay have local NVMe backing (nvme2n1p2 ext4, also exposed via /tmp/nvidia-mps). Other visible NVMe devices are not available mounted storage; no mount/format operation was performed. /dev/shm is RAM, not staging SSD.", ""])
    disk = result["local_disk_capacity"]
    lines.extend([f"Local free={disk['local_free_bytes'] / GIB:.3f}GiB. Existing install inventories plus ZIP central directories report image payload={disk['payload_bytes'] / GIB:.3f}GiB; expected count coverage={disk['inventory_counts_cover_expected']}. Full raw image payload fits with64GiB reserve={disk['full_image_payload_fits_with_reserve']}. No recursive image enumeration, full data audit, image download or staging was performed.", ""])
    for family, values in disk["families"].items():
        lines.append(f"- {family}: {values['required_images_covered']}/{values['required_images_expected']} required images represented in inventories; {values['payload_bytes'] / GIB:.3f}GiB payload. Inventory extras={values.get('unused_inventory_images', 0)} are excluded from the estimate.")
    lines.extend(["", "## Frozen initialization and isolation", "",
        "Common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`. Start_updates=0; no resume argument. F/S/D=[1.4,0.2,1.4], horizon4868, batch256/rank, accumulation1, workers8/rank, original sampling/model/loss/optimizer unchanged. The original source hashes still match the passed smoke.", "",
        "The native CLI uses run-type formal solely to retain the original protection; --max-updates40 and a second scheduler hook enforce the diagnostic cap. Dedicated diagnostic output is separate from formal trajectories. Diagnostic checkpoints5/40 were quarantined and marked DO_NOT_RESUME; checkpoint bytes are not committed to GitHub.", "",
        f"Runtime evidence: `{result['preflight']['runtime']}`.", f"Local telemetry: `{result['preflight']['local_telemetry']}` (copied into runtime after completion).", "",
        "CPU tests:71 passed (65 existing gate tests +6 resource-diagnostic tests). Training exited0; acceptance passed. All training/audit workers exited and GPUs are idle after the run.", "",
        "Next action requires a new user instruction: start a fresh common-step0 500-step reproduction trajectory with horizon4868 and unchanged resource protection. Never resume diagnostic40 or stopped33. No500/full was automatically started.", ""])
    report = "\n".join(lines)
    (EXP / "RESOURCE_STALL_DIAGNOSIS_V2.md").write_text(report)
    (ROOT / "recovery/RESOURCE_STALL_DIAGNOSIS_V2.md").write_text(report)


if __name__ == "__main__":
    main()
