"""Read-only CPU evidence for the trajectory stopped before its500-step gate."""

import statistics
from pathlib import Path

import torch

from . import reproduction_full as pipeline


def main():
    torch.set_num_threads(4)
    require, load, dump, sha = pipeline.require, pipeline.load, pipeline.dump, pipeline.sha
    output = pipeline.RUN / "step500"
    rows = [pipeline.json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    cycles = [pipeline.json.loads(line) for line in (output / "cycle_timing.jsonl").read_text().splitlines()]
    require([row["step"] for row in rows] == list(range(1, 34)), "Unexpected stopped trajectory")
    require(all(row["nonfinite"] == 0 and all(item["gradients_finite"] for item in row["rank_health"])
                for row in rows), "Nonfinite training state")
    require(all(row["four_rank_max_seconds"] > 3 for row in cycles[-3:]), "Missing resource-stop evidence")
    checkpoint_path = output / "step000033.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["global_step"] == checkpoint["completed_steps"] == 33 and
            checkpoint["scheduler_horizon"] == checkpoint["scheduler"]["horizon"] == 4868, "Step/horizon mismatch")
    require(checkpoint["data_cursor"] == dict(next_epoch=0, next_batch=33), "Checkpoint cursor mismatch")
    require(checkpoint["scaler"] == dict(enabled=False, dtype="bfloat16", state=None), "Historical BF16/scaler drift")
    require(len(checkpoint["rng_per_rank"]) == 4 and all(
        {"python", "numpy", "cpu", "cuda", "loader_generator"} <= set(state) for state in checkpoint["rng_per_rank"]),
        "Incomplete four-rank RNG state")
    require({int(state["step"]) for state in checkpoint["optimizer"]["state"].values()} == {33}, "AdamW counter mismatch")
    for state in [checkpoint["model"], checkpoint["adapter"], *checkpoint["optimizer"]["state"].values()]:
        require(all(bool(torch.isfinite(tensor).all()) for tensor in state.values()), "Nonfinite checkpoint tensor")
    frozen = load(pipeline.ROOT / "recovery/configs/summary02.json")
    require(all(checkpoint["config"][key] == value for key, value in frozen.items()), "Frozen hyperparameter drift")
    require(checkpoint["config"]["start_updates"] == 0 and checkpoint["config"]["resume"] is None and
            checkpoint["config"]["init_sha256"] == pipeline.native_pipeline.STEP0_SHA, "Incorrect initializer")
    require(all(sha(pipeline.ROOT / name) == digest for name, digest in checkpoint["config"]["code_sha256"].items()),
            "Native source drift")
    memory = Path("/sys/fs/cgroup/memory")
    memory_stat = dict(line.split() for line in (memory / "memory.stat").read_text().splitlines())
    cpu = Path("/sys/fs/cgroup/cpu")
    checkpoint_sha = sha(checkpoint_path)
    gate = load(pipeline.RUN / "first-five-gate.json")
    prior_state = load(pipeline.RUN / "state.json")
    report = dict(status="STOPPED_BEFORE_500_RESOURCE_GATE", completed_updates=33,
                  audited_at_utc=pipeline.now(), training_running=False, reproduction500_evaluated=False,
                  no_500_or_4868_result_claimed=True, first_five_hard_gate=gate,
                  first5_losses=[row["loss"] for row in rows[:5]], last_loss=rows[-1]["loss"],
                  every_recorded_loss_and_gradient_finite=True, checkpoint_tensors_finite=True,
                  optimizer_steps=[33], horizon=4868, all_frozen_config_fields_match=True,
                  frozen_config_sha256=sha(pipeline.ROOT / "recovery/configs/summary02.json"),
                  native_source_sha256=checkpoint["config"]["code_sha256"], executed_git_head=checkpoint["config"]["git_head"],
                  stop_trigger="Original native safety monitor: three consecutive full updates exceeded3 seconds",
                  last3_cycles=cycles[-3:], median_cycle_steps7_to30_seconds=statistics.median(
                      row["four_rank_max_seconds"] for row in cycles if 7 <= row["step"] <= 30),
                  per_step_cycles=cycles,
                  peak_allocated_gib=max(item["peak_allocated_gib"] for row in rows for item in row["rank_health"]),
                  secondary_failure="Early DataLoader shutdown raised SIGABRT worker error after step33 checkpoint was saved; terminal acceptance was not emitted",
                  root_cause="Unconfirmed: NFS/data wait, host-side synchronization and memcg page reclaim require investigation; not a reproduction-metric failure",
                  memory_snapshot=dict(limit_bytes=int((memory / "memory.limit_in_bytes").read_text()),
                                       usage_bytes=int((memory / "memory.usage_in_bytes").read_text()),
                                       max_usage_bytes=int((memory / "memory.max_usage_in_bytes").read_text()),
                                       file_cache_bytes=int(memory_stat["cache"]),
                                       failcnt_cumulative=int((memory / "memory.failcnt").read_text()),
                                       oom_control=(memory / "memory.oom_control").read_text(),
                                       caution="Counters are cumulative; no per-run baseline, so they do not establish causation"),
                  cpu_snapshot=dict(quota_us=int((cpu / "cpu.cfs_quota_us").read_text()),
                                    period_us=int((cpu / "cpu.cfs_period_us").read_text()),
                                    stat_cumulative=(cpu / "cpu.stat").read_text()),
                  cache_advice=load(pipeline.RUN / "private_archive_cache_advice.json"),
                  checkpoint=dict(path=str(checkpoint_path), sha256=checkpoint_sha,
                                  full_state_saved=True, resume_authorized=False),
                  safety_threshold_relaxed=False, production_code_changed=False,
                  no_data_download_or_recursive_scan_or_decode_reaudit=True,
                  next_action="Investigate/resolve storage or container-memory stalls; explicit fresh common-step0 retry only, never resume step33")
    dump(pipeline.EXP / "RESOURCE_STOP_AUDIT.json", report)
    (pipeline.EXP / "RESOURCE_STOP_AUDIT.md").write_text(
        "# S02 stopped before500: resource evidence\n\n"
        "**STOPPED_BEFORE_500_RESOURCE_GATE**. Exactly33 finite AdamW updates, from fresh common step0, horizon4868.\n\n"
        "Minimal first5 input/text/token/LR/seed/source invariants passed; four-rank parameter difference0. No cross-run numerical gate.\n\n"
        "| Update | Full-cycle seconds |\n|---:|---:|\n" +
        "\n".join(f"| {row['step']} | {row['four_rank_max_seconds']:.6f} |" for row in cycles[-3:]) +
        "\n\nOriginal native monitor stops after three consecutive cycles>3 seconds. It was not relaxed.\n"
        "Early worker shutdown subsequently raised SIGABRT, so there is no final native acceptance/DDP proof beyond the passed first5.\n\n"
        f"Median full cycle steps7–30: {report['median_cycle_steps7_to30_seconds']:.6f}s. Peak allocated: {report['peak_allocated_gib']:.6f}GiB/card.\n\n"
        "Container limit500GiB, substantial file cache, historical peak at the limit, no recorded OOM kill. Counters are cumulative, not proof of this run's cause.\n"
        "NFS/data waits and host-side synchronization/reclaim remain hypotheses, not confirmed causes.\n"
        "Advised only37 exact completed archive paths to release unused cache; no image/byte/deletion/global-cache operation. No material memory reduction and no new training trial.\n\n"
        "Full step33 CPU state audit passed: model/adapter/AdamW finite, counter33, manual scheduler4868, BF16/no scaler, four-rank RNG and epoch0/batch33 cursor.\n"
        "Step33 is evidence only and must not be resumed under the current500 continuation protocol. No step500 exists; retrieval gate is NOT RUN, not REPRODUCTION_FAIL_AT_500.\n")
    pending = dict(status="NOT_REACHED_500", completed_updates=33, evaluated=False, metrics=None, scores=None,
                   horizon=4868, blocker=report["stop_trigger"], resource_audit="RESOURCE_STOP_AUDIT.json")
    dump(pipeline.EXP / "STEP500_RESULTS.json", pending)
    (pipeline.EXP / "STEP500_REPRODUCTION.md").write_text(
        "# Step500 reproduction: NOT RUN\n\nFirst5 hard invariants passed; native resource monitor stopped at33. "
        "No step500 checkpoint or retrieval results exist. No score deltas or reproduction PASS/FAIL are claimed.\n\n"
        "See RESOURCE_STOP_AUDIT.md/json. Horizon stayed4868 and all frozen hyperparameters matched. "
        "Training is stopped; step33 is not an authorized resume point.\n")
    dump(pipeline.EXP / "CONTINUATION_GATE.json", dict(status="NOT_RUN_BEFORE_500", passed=False,
         evaluated=False, completed_updates=33, thresholds_percent=pipeline.THRESHOLDS, horizon=4868,
         continuation_authorized=False, blocker="RESOURCE_GATE_AT_33"))
    dump(pipeline.EXP / "CHECKPOINT_SHA256.json", {
        "step500/step000005.pt": sha(output / "step000005.pt"), "step500/step000033.pt": checkpoint_sha})
    dump(pipeline.EXP / "STRICT_EXPORT_PROVENANCE.json", dict(status="NOT_RUN", trajectory=str(pipeline.RUN),
                                                             reason="Neither500 nor4868 was reached"))
    prior_state.update(status=report["status"], stage="Stopped at33; resource gate before500",
                       active_pid=None, formal_training_authorized=False, resource_audit="RESOURCE_STOP_AUDIT.json")
    dump(pipeline.RUN / "state.json", prior_state)
    dump(pipeline.EXP / "FULL_PROGRESS.json", prior_state)
    dump(pipeline.EXP / "FULL_RESULTS.json", dict(report, full_evaluated=False))
    (pipeline.EXP / "FULL_RESULTS.md").write_text(
        "# S02 full trajectory stopped before500\n\n**STOPPED_BEFORE_500_RESOURCE_GATE**.\n\n"
        "First5 hard gate PASS, no loss/model/moment cross-run matching. From common step0, weights[1.4,0.2,1.4], horizon4868; all native source/config unchanged.\n\n"
        "Stopped at33 by original resource guard. See RESOURCE_STOP_AUDIT.md/json. No500 reproduction or full4868 retrieval result claimed.\n"
        "Checkpoints preserved locally; no implicit retry/resume/alternate experiment.\n")
    diagnostics = load(pipeline.EXP / "TRAINING_DIAGNOSTICS.json")
    diagnostics.update(status=report["status"], observed_updates=33, resource_audit="RESOURCE_STOP_AUDIT.json")
    dump(pipeline.EXP / "TRAINING_DIAGNOSTICS.json", diagnostics)
    policy_path = pipeline.ROOT / "recovery/evidence/recovery-operation-policy.json"
    policy = load(policy_path)
    policy.update(formal_training_authorized=False, resume_allowed=False, full_completed_updates=33,
                  full_gate_status=report["status"], reason="Native resource guard stopped before500; no metric gate evaluated or automatic restart authorized")
    dump(policy_path, policy)
    print(pipeline.json.dumps({key: report[key] for key in ("status", "completed_updates", "peak_allocated_gib", "checkpoint")}))


if __name__ == "__main__":
    main()
