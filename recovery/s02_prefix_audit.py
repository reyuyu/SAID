"""Reconcile a rejected S02 full prefix from saved evidence; never train."""

import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/nest_clip_v1/armb_summary02_4epoch_v1"
RUN = ROOT / "runtime/SAID-nest-clip-v1/armb_summary02_4epoch_recovery_v1"


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_rows(actual, reference, reproduced):
    if any([row["step"] for row in sequence] != [1, 2, 3, 4, 5]
           for sequence in (actual, reference, reproduced)):
        raise RuntimeError("Expected exactly five updates in each saved trajectory")
    comparison = []
    for full, old, new in zip(actual, reference, reproduced):
        ranks = {label: {rank["rank"]: rank for rank in row["rank_health"]}
                 for label, row in (("full", full), ("reference", old), ("replay", new))}
        exact_ids = all(ranks["full"][rank]["sampling"]["sample_ids"] ==
                        ranks["reference"][rank]["sampling"]["sample_ids"] ==
                        ranks["replay"][rank]["sampling"]["sample_ids"] for rank in range(4))
        exact_streams = all(ranks["full"][rank]["stream_sha256"] ==
                            ranks["reference"][rank]["stream_sha256"] ==
                            ranks["replay"][rank]["stream_sha256"] for rank in range(4))
        if not exact_ids or not exact_streams or full["actual_lrs"] != old["actual_lrs"] or old["actual_lrs"] != new["actual_lrs"]:
            raise RuntimeError("Input/LR drift, not merely a numeric trace difference")
        comparison.append(dict(step=full["step"], full_loss=full["loss"], reference_loss=old["loss"],
                               native_replay_loss=new["loss"], full_abs_difference=abs(full["loss"] - old["loss"]),
                               full_relative_difference_percent=100 * abs(full["loss"] - old["loss"]) / old["loss"],
                               exact_sample_ids=True, exact_FSD_strings_and_tokens=True, exact_LR=True,
                               rank_stream_sha256={str(rank): ranks["full"][rank]["stream_sha256"] for rank in range(4)}))
    return comparison


def main():
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_full import compact_step
    smoke = Path(load(ROOT / "recovery/evidence/s02-smoke-command.json")["quarantined_output_dir"])
    replay = Path((ROOT / "recovery/evidence/s02-native-repro-output.txt").read_text().strip())
    actual, reference, reproduced = (rows(folder / "steps.jsonl") for folder in (RUN / "train", smoke, replay))
    comparison = compare_rows(actual, reference, reproduced)
    configs = {name: load(folder / "config.json") for name, folder in (("full", RUN / "train"), ("reference", smoke), ("replay", replay))}
    fields = ("adapter_initialization", "component_initialization", "runtime_model", "data", "torch", "cuda", "nccl",
              "environment", "parameter_counts", "code_sha256")
    metadata = {key: configs["full"][key] == configs["reference"][key] == configs["replay"][key] for key in fields}
    if not all(metadata.values()):
        raise RuntimeError("Construction/environment/source metadata drift")
    accepted = load(replay / "acceptance.json")
    if not accepted["passed"] or not all(rank["completed_updates"] == 5 and rank["max_parameter_difference_from_rank0"] == 0
                                         for rank in accepted["ranks"]):
        raise RuntimeError("Original native diagnostic smoke acceptance failed")
    rejection = load(RUN / "first-five-gate.json")
    if rejection["passed"] or (RUN / "train/step000006.pt").exists():
        raise RuntimeError("Not a rejected prefix-only run")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proof = dict(status="BLOCKED_PREFIX_GATE", checked_at=timestamp, steps=comparison, metadata_exact_equal=metadata,
                 maximum_full_relative_difference_percent=max(row["full_relative_difference_percent"] for row in comparison),
                 common_step0_sha256=configs["full"]["init_sha256"],
                 same_native_replay_has_cross_run_loss_difference=any(new["loss"] != old["loss"] for new, old in zip(reproduced, reference)),
                 native_replay_acceptance=accepted, native_replay_output_dir=str(replay),
                 precise_backend_cause_confirmed=False, no_training_hyperparameters_changed=True,
                 numerical_gate_not_relaxed=True, formal_update6_executed=False, full4868_executed=False,
                 strict_final_export_executed=False, full_five_set_evaluation_executed=False)
    dump(EXP / "PREFIX_REPRODUCIBILITY.json", proof)
    (replay / "SMOKE_ONLY_DO_NOT_RESUME.md").write_text("# Numerical reproducibility diagnostic only\n\nExactly5 updates; do not resume full from this checkpoint.\n")
    (RUN / "DO_NOT_RESUME_FAILED_PREFIX.md").write_text("# Full prefix gate rejected\n\nFive updates only. Do not resume these forensic checkpoints. Any authorized retry must start from common step0.\n")
    state = load(RUN / "state.json")
    state.update(status="BLOCKED_PREFIX_GATE", completed_updates=5, formal_training_authorized=False,
                 active_pid=None, worker_processes_running=False, supervisor_process_running=False,
                 error=rejection["error"], reconciled_at=timestamp,
                 reconciliation_reason="Rejected synchronous gate, exact five-row native log, no live training processes; prior TRAINING0 snapshot stale",
                 evaluation_started=False, full_result_classification=None)
    dump(RUN / "state.json", state)
    dump(EXP / "FULL_PROGRESS.json", state)
    dump(EXP / "FULL_RESULTS.json", dict(state, scores=None, metrics=None, strict_final_export=None,
                                         prefix_reproducibility=proof, no_scientific_full_result_claimed=True))
    metrics = [compact_step(row) for row in actual]
    dump(EXP / "TRAINING_DIAGNOSTICS.json", dict(status="BLOCKED_PREFIX_GATE", observed_updates=5, updates=metrics,
         peak_allocated_gib=max(health["peak_allocated_gib"] for row in actual for health in row["rank_health"]),
         exact_first5_sample_text_token_stream=True, loss_tolerance_gate_passed=False,
         full_optimizer_state_forensics="PREFIX_STATE_FORENSICS.json", full_training_updates=5,
         diagnostic_native_replay_updates=5, isolated_operator_probe_updates=0))
    policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
    policy = load(policy_path)
    policy.update(formal_training_authorized=False, smoke_authorized=False, conditional_smoke_authorized=False,
                  full_gate_status="BLOCKED_PREFIX_GATE", full_completed_updates=5,
                  reason="S02 full prefix rejected; no automatic tolerance bypass, restart or checkpoint resume")
    dump(policy_path, policy)
    print(json.dumps(dict(status=proof["status"], maximum_loss_difference_percent=proof["maximum_full_relative_difference_percent"],
                         native_replay_difference_reproduced=proof["same_native_replay_has_cross_run_loss_difference"],
                         precise_cause_confirmed=False, full_completed_updates=5), indent=2))


if __name__ == "__main__":
    main()
