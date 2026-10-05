"""Read-only checkpoint and phase evidence audit after a resource-protected stop."""

import json
import torch

from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as run
from recovery.resource_stall_v2 import process_audit


def main():
    proof_path = run.EXP / "RESOURCE_STALL_RECURRENCE.json"
    proof = run.load(proof_path)
    step = proof["completed_updates"]
    checkpoint = run.RUN / "step500" / f"step{step:06d}.pt"
    identity = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "adapter", "optimizer", "scheduler", "scaler", "rng_per_rank", "sampler", "data_cursor", "global_step"}
    assert required <= payload.keys()
    assert payload["global_step"] == payload["completed_steps"] == step == 55
    assert payload["scheduler_horizon"] == payload["scheduler"]["horizon"] == 4868
    assert payload["data_cursor"] == dict(next_epoch=0, next_batch=55)
    assert len(payload["rng_per_rank"]) == 4
    for state in payload["rng_per_rank"]:
        assert {"python", "numpy", "cpu", "cuda", "loader_generator"} <= state.keys()
    counters = sorted({int(state["step"]) for state in payload["optimizer"]["state"].values()})
    assert counters == [55]
    tensors = list(payload["model"].values()) + list(payload["adapter"].values())
    tensors.extend(tensor for state in payload["optimizer"]["state"].values() for tensor in state.values() if torch.is_tensor(tensor))
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    config = payload["config"]
    assert config["init_sha256"] == run.native_pipeline.STEP0_SHA
    assert config["start_updates"] == 0 and config["resume"] is None
    assert config["view_weights"] == [1.4, .2, 1.4] and config["workers"] == 8
    assert all(run.sha(run.ROOT / name) == digest for name, digest in config["code_sha256"].items())
    latest = checkpoint.stat()
    assert (identity.st_ino, identity.st_size, identity.st_mtime_ns) == (latest.st_ino, latest.st_size, latest.st_mtime_ns)
    proof["checkpoint_audit"] = dict(passed=True, path=str(checkpoint), sha256=run.sha(checkpoint),
        global_step=55, optimizer_steps=counters, scheduler_horizon=4868, four_rank_rng_complete=True,
        data_cursor=payload["data_cursor"], all_model_and_optimizer_tensors_finite=True,
        fresh_common_step0=True, native_source_unchanged=True, modified=False, resume_authorized=False)
    assert len(proof["recent20_steps"]) == 20
    assert all(len(row["ranks"]) == 4 and not any(rank.get("timing_unavailable") for rank in row["ranks"])
               for row in proof["recent20_steps"])
    latest_step = proof["recent20_steps"][-1]
    proof["phase_diagnosis"] = dict(primary_observed_phase="DATA_WAIT_WITH_DOWNSTREAM_FORWARD_SYNCHRONIZATION_WAIT",
        slowest_data_wait_s=max(rank["data_wait_s"] for rank in latest_step["ranks"]),
        data_plus_forward_s_by_rank={str(rank["rank"]): rank["data_wait_s"] + rank["forward_s"] for rank in latest_step["ranks"]},
        intrinsic_backward_optimizer_slowdown_observed=False,
        NFS_vs_CPU_reclaim_vs_worker_decode_cause="NOT_YET_ISOLATED",
        host_PSI_not_cgroup_PSI=True, method_failure=False)
    proof["post_stop_process_audit"] = process_audit()
    assert proof["post_stop_process_audit"]["workers_all_exited"]
    run.dump(proof_path, proof)
    run.dump(run.RUN / "resource-stall-recurrence.json", proof)
    policy_path = run.ROOT / "recovery/evidence/recovery-operation-policy.json"
    policy = run.load(policy_path)
    policy.update(resource_gate_status="RESOURCE_GATE_NOT_READY_FOR_500", training_held_for_new_instruction=True,
                  formal_training_authorized=False, reproduction_gate_authorized=False, resume_allowed=False)
    run.dump(policy_path, policy)
    state = run.load(run.EXP / "FULL_PROGRESS.json")
    state.update(training_process_running=False, last_training_pid=state.get("active_pid"), active_pid=None,
                 checkpoint_audit=proof["checkpoint_audit"], updated_at=run.now())
    run.dump(run.EXP / "FULL_PROGRESS.json", state)
    run.dump(run.RUN / "state.json", state)
    run.dump(run.EXP / "FULL_RESULTS.json", dict(state, full_evaluated=False))
    diagnostics = run.load(run.EXP / "TRAINING_DIAGNOSTICS.json")
    diagnostics.update(status=proof["status"], observed_updates=55, horizon=4868,
                       phase_diagnosis=proof["phase_diagnosis"], method_failure=False,
                       automatic_retry=False, recent20_evidence="RESOURCE_STALL_RECURRENCE.json")
    run.dump(run.EXP / "TRAINING_DIAGNOSTICS.json", diagnostics)
    pending_command = run.EXP / "commands/train500-recovery.json"
    if pending_command.exists():
        pending_command.replace(run.EXP / "commands/train500-500gate.json")
    print(json.dumps(dict(status=proof["status"], checkpoint_audit=proof["checkpoint_audit"],
                          primary_phase=proof["phase_diagnosis"]["primary_observed_phase"])), flush=True)


if __name__ == "__main__":
    main()
