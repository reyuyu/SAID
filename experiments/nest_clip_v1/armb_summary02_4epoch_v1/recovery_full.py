"""One authorized S02 run from step0 to4868, followed by strict native five-set eval."""

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/SAID-nest-clip-v1"
RUN = RUNTIME / "armb_summary02_4epoch_recovery_v1"
TRAIN = RUN / "train"
ASSETS = ROOT / "local_assets"
BRANCH = "codex/nest-balanced-armb-summary02-4epoch-recovery-v1"
PYTHON = str(ROOT / ".venv/bin/python")
STEP0_SHA = "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6"
MODULE = "experiments.nest_clip_v1.armb_summary02_4epoch_v1"
DATASETS = ("COCO", "Urban-1k", "Flickr30k-test1k", "DOCCI", "Long-DCI")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight():
    policy = load(ROOT / "recovery/evidence/recovery-operation-policy.json")
    require(policy.get("formal_training_authorized") is True and policy.get("allowed_full_updates") == 4868
            and policy.get("resume_allowed") is False,
            "Current instruction/policy does not authorize a fresh full4868 attempt")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == BRANCH,
            "Wrong formal experiment branch")
    require(load(ROOT / "recovery/evidence/s02-smoke-audit.json")["status"] == "READY_TO_START_S02_FULL",
            "Final S02 smoke gate not passed")
    require(load(ROOT / "recovery/FINAL_IMAGE_AUDIT.json")["status"] == "DATA_AUDIT_PASS",
            "Existing data audit not passed; no data re-audit performed")
    require(load(ROOT / "recovery/evidence/evaluator-audit.json")["passed"], "Frozen evaluator audit not passed")
    config = load(ROOT / "recovery/configs/summary02.json")
    parent = load(ROOT / "experiments/nest_clip_v1/armb_summary_dose_500_v1/arm_E1_summary02/config.json")
    require({key for key in parent if parent[key] != config[key]} == {"experiment_name", "checkpoint_interval"},
            "Historical S02@500 hyperparameter drift")
    require(config["view_weights"] == [1.4, .2, 1.4] and config["sampling_mode"] == "summary_random_detail"
            and config["epochs"] == 4, "Incorrect S02 config")
    source = load(ROOT / "recovery/evidence/s02-smoke-command.json")
    for filename, digest in source["code_sha256"].items():
        require(sha(ROOT / filename) == digest, "Native source changed after smoke: " + filename)
    require(sha(RUNTIME / "shared/step000000.pt") == STEP0_SHA, "Common step0 SHA256 mismatch")
    require((Path.home() / ".cache/clip/ViT-B-16.pt").is_file(), "No model download permitted")
    require(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(),
            "GPUs occupied")
    metadata = load(RUNTIME / "data_index/metadata.json")
    require(metadata["training_records"] == 1245901, "Filtered training index count differs")
    proof = dict(passed=True, checked_at=now(), common_step0_sha256=STEP0_SHA,
                 resume=None, start_updates=0, stop_updates=4868, horizon=4868,
                 frozen_config=config, parent500_config=parent,
                 production_source_sha256=source["code_sha256"],
                 runtime_guard_sha256=sha(EXP / "recovery_train_gate.py"),
                 supervisor_sha256=sha(EXP / "recovery_full.py"),
                 data_audit_sha256=sha(ROOT / "recovery/FINAL_IMAGE_AUDIT.json"),
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 resource_monitor_unchanged=True, recursive_data_scan=False,
                 modifications="Read-only first-five comparison plus synchronous step5 checkpoint before update6; no model/objective/sampling/LR changes",
                 normal_tail="Historical DistributedSampler epoch tail180/rank/global720 retained")
    dump(EXP / "FULL_PREFLIGHT.json", proof)
    return proof


def compact_step(row):
    contribution = {label: 10 / 3 * weight * (row[prefix + "_i2t"] + row[prefix + "_t2i"])
                    for label, prefix, weight in zip(("F", "S", "D"), ("F", "O", "E"), (1.4, .2, 1.4))}
    return dict(step=row["step"], epoch=row["epoch"], loss=row["loss"],
                CE={label: dict(I2T=row[prefix + "_i2t"], T2I=row[prefix + "_t2i"])
                    for label, prefix in zip(("F", "S", "D"), ("F", "O", "E"))},
                weighted_alignment=contribution,
                sparsity=(row["F_sparse"] + 2 * row["O_sparse"] + 2 * row["E_sparse"]) / 3,
                inclusion_raw=row["inc"], inclusion_ramp=row["inc_weight"],
                inclusion_contribution=row["inc"] * row["inc_weight"],
                keep_ratio={label: row[prefix + "_keep_ratio"] for label, prefix in zip(("F", "S", "D"), ("F", "O", "E"))},
                gate_stats={key: value for key, value in row.items() if "_g_" in key or "saturation" in key
                            or key.endswith(("_all_open", "_all_closed", "_positive_keep_ratio", "_negative_keep_ratio"))},
                LR=row["actual_lrs"],
                rank_health=[{key: value for key, value in health.items() if key != "sampling"} for health in row["rank_health"]],
                nonfinite=row["nonfinite"], duplicate_image_ids=row["duplicate_image_ids"])


class Supervisor:
    def __init__(self):
        self.state = dict(status="PREFLIGHT", started_at=now(), supervisor_pid=os.getpid(),
                          branch=BRANCH, completed_updates=0, stop_updates=4868,
                          formal_training_authorized=True, formal_training_started=False,
                          resume=None, scheduled_evaluation="Frozen native five sets after strict step4868 export only")
        self.metrics = []
        self.offset = 0

    def save(self):
        self.state["updated_at"] = now()
        dump(RUN / "state.json", self.state)
        dump(EXP / "FULL_PROGRESS.json", self.state)

    def collect(self):
        gate_path = RUN / "first-five-gate.json"
        if gate_path.exists():
            require(load(gate_path)["passed"], "Synchronous first-five gate rejected; no continuation")
        path = TRAIN / "steps.jsonl"
        if not path.exists():
            return
        with path.open() as handle, (RUN / "training_metrics.jsonl").open("a") as output:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    self.offset = start
                    break
                row = json.loads(line)
                require(row["step"] == len(self.metrics) + 1 and row["step"] <= 4868,
                        "Optimizer update sequence changed or exceeded4868")
                require(row["nonfinite"] == 0 and all(health["gradients_finite"] for health in row["rank_health"]),
                        "Nonfinite full-training loss/gradients")
                compact = compact_step(row)
                self.metrics.append(compact)
                output.write(json.dumps(compact, allow_nan=False) + "\n")
                self.offset = handle.tell()
        if self.metrics:
            latest = self.metrics[-1]
            self.state.update(completed_updates=latest["step"], latest_loss=latest["loss"],
                              latest_update_seconds=max(rank["seconds"] for rank in latest["rank_health"]),
                              peak_allocated_gib=max(rank["peak_allocated_gib"] for rank in latest["rank_health"]))
            dump(EXP / "TRAINING_DIAGNOSTICS.json", dict(status=self.state["status"],
                 observed_updates=len(self.metrics), latest=latest,
                 last50_loss_mean=statistics.fmean(item["loss"] for item in self.metrics[-50:]),
                 raw_per_update_metrics=str(RUN / "training_metrics.jsonl"),
                 native_soft_gate_diagnostic_schedule="Unchanged; hard gate openness/keep logged each update"))
            if (RUN / "first-five-gate.json").exists():
                self.state["first_five_gate"] = load(RUN / "first-five-gate.json")
        self.save()

    def execute(self, name, command, training=False):
        self.state.update(stage=name, status="TRAINING" if training else "EVALUATING", stage_started_at=now())
        self.save()
        record = dict(command=command, cwd=str(ROOT), started_at=now())
        dump(EXP / "commands" / (name + "-recovery.json"), record)
        log = RUN / (name + ".log")
        with log.open("w") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                                                SAID_FULL_SUPERVISOR_PID=str(os.getpid())))
            self.state["active_pid"] = process.pid
            if training:
                self.state["formal_training_started"] = True
            self.save()
            try:
                while process.poll() is None:
                    if training:
                        self.collect()
                    time.sleep(30)
                if training:
                    self.collect()
                record.update(returncode=process.returncode, finished_at=now())
                dump(EXP / "commands" / (name + "-recovery.json"), record)
                require(process.returncode == 0, name + " failed; see " + str(log))
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise

    def run(self):
        self.save()
        preflight()
        policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
        policy = load(policy_path)
        policy.update(scope="One authorized S02 full4868 run from step0 and final frozen native five-set eval",
                      reason="User explicitly authorizes S02 full; no other trials",
                      formal_training_authorized=True, allowed_full_updates=4868, resume_allowed=False,
                      smoke_authorized=False, conditional_smoke_authorized=False,
                      authorized_experiments=["ArmB-Summary0.2-4epoch"])
        dump(policy_path, policy)
        command = [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                   "-m", MODULE + ".recovery_train_gate", "--config", str(ROOT / "recovery/configs/summary02.json"),
                   "--init-state", str(RUNTIME / "shared/step000000.pt"), "--index-dir", str(RUNTIME / "data_index"),
                   "--image-root", str(ASSETS / "training/ShareGPT4V"), "--output-dir", str(TRAIN),
                   "--run-type", "formal", "--max-updates", "4868"]
        self.execute("full-train4868", command, training=True)
        acceptance = load(TRAIN / "acceptance.json")
        require(acceptance["passed"] and len(self.metrics) == 4868 and
                all(rank["completed_updates"] == 4868 and rank["max_parameter_difference_from_rank0"] == 0
                    for rank in acceptance["ranks"]), "Full4868 native acceptance failed")
        require(load(RUN / "first-five-gate.json")["passed"], "First-five gate did not pass")
        checkpoint = TRAIN / "step004868.pt"
        bare = TRAIN / "student_step4868.pt"
        self.execute("strict-export4868", [PYTHON, "-m", "tools.nest_clip", "export", "--checkpoint", str(checkpoint),
                                         "--output", str(bare), "--expect-updates", "4868"])
        self.execute("strict-verify4868", [PYTHON, "-m", "tools.nest_clip", "verify-export", "--checkpoint", str(checkpoint),
                    "--bare", str(bare), "--output", str(TRAIN / "export-check.json"), "--index-dir", str(RUNTIME / "data_index"),
                    "--image-root", str(ASSETS / "training/ShareGPT4V")])
        for name, folder in (("coco", ASSETS / "evaluation/coco/val2017"), ("urban", ASSETS / "evaluation/Urban1k/Urban1k")):
            self.execute("eval4868-" + name, [PYTHON, "-m", "tools.eval_nest_native", "--checkpoint", str(bare),
                         "--dataset", name, "--root", str(folder), "--device", "cuda:0", "--batch-size", "64",
                         "--output", str(TRAIN / (name + "_native.json"))])
        bench = ASSETS / "retrieval_benchmarks"
        for name, manifest, folder in (("flickr_test1k", "flickr30k_test1k.jsonl", "flickr30k"),
                                      ("docci", "docci_test.jsonl", "docci"),
                                      ("long_dci", "long_dci_reconstructed.jsonl", "dci")):
            self.execute("eval4868-" + name, [PYTHON, "-m", "experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real",
                         "--checkpoint", str(bare), "--device", "cuda:0", "--batch-size", "64", "--output-dir", str(TRAIN / name),
                         f"{name}:{bench / 'manifests' / manifest}:{bench / folder / 'images'}"])
        self.finish(acceptance)
        policy.update(formal_training_authorized=False, full_completed_updates=4868,
                      reason="Authorized S02 full and five-set eval completed; no subsequent experiment authorized")
        dump(policy_path, policy)

    def finish(self, acceptance):
        from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores
        metrics, raw, sources = native_metrics(TRAIN)
        calculated = scores(metrics)
        calculated["Short4_R1"] = statistics.fmean(metrics[name][direction]["R@1"]
                                      for name in ("COCO", "Flickr30k-test1k") for direction in ("I2T", "T2I"))
        reference = load(EXP / "RANDOMK_4EPOCH_REFERENCE.json")
        reference_scores = dict(reference["scores"], Short4_R1=statistics.fmean(
            reference["metrics"][name][direction]["R@1"] for name in ("COCO", "Flickr30k-test1k") for direction in ("I2T", "T2I")))
        parent_scores = dict(Score5_R1=.70364367, J_long3=.73726611, J_long=.82765002, Short4_R1=.65321)
        percent = {key: 100 * value for key, value in calculated.items()}
        positive = percent["Score5_R1"] > 72.768147
        guard = percent["J_long3"] >= 76.870244
        decision = "FULL_POSITIVE" if positive and guard else "SHORT_LONG_TRADEOFF" if positive else "EARLY_SIGNAL_DID_NOT_SCALE"
        comparisons = dict(Score5_above_RandomK=positive, long3_guard_passed=guard,
            urban_both_directions_above_RandomK=all(metrics["Urban-1k"][direction]["R@1"] > reference["metrics"]["Urban-1k"][direction]["R@1"]
                                                  for direction in ("I2T", "T2I")),
            Short4_above_RandomK=calculated["Short4_R1"] > reference_scores["Short4_R1"],
            LongDCI_direction_deltas_pp={direction:100*(metrics["Long-DCI"][direction]["R@1"] - reference["metrics"]["Long-DCI"][direction]["R@1"])
                                        for direction in ("I2T", "T2I")})
        export = load(TRAIN / "export-check.json")
        dump(EXP / "STRICT_EXPORT_PROVENANCE.json", dict(export, inference="normalize(native image embedding) @ normalize(native full-caption text embedding).T",
                                                       training_updates=4868, no_mask_gate_rerank_ensemble=True))
        cycles = [json.loads(line) for line in (TRAIN / "cycle_timing.jsonl").read_text().splitlines()]
        normal = [item["four_rank_max_seconds"] for item in cycles if not item["warmup"]]
        resources = dict(mean_post_warmup_full_cycle_seconds=statistics.fmean(normal),
                         median_post_warmup_full_cycle_seconds=statistics.median(normal),
                         peak_allocated_gib=max(rank["peak_allocated_gib"] for rank in acceptance["ranks"]))
        checkpoints = {f"step{step:06d}.pt": sha(TRAIN / f"step{step:06d}.pt") for step in (5,1217,2434,3651,4868)}
        dump(EXP / "CHECKPOINT_SHA256.json", checkpoints)
        result = dict(status="COMPLETE", classification=decision, completed_updates=4868,
                      scores=calculated, scores_percent=percent, metrics=metrics, comparisons=comparisons,
                      delta_vs_RandomK_pp={key:100*(calculated[key]-reference_scores[key]) for key in calculated},
                      delta_vs_own500_pp={key:100*(calculated[key]-parent_scores[key]) for key in calculated},
                      RandomK_reference_scores_percent={key:100*value for key,value in reference_scores.items()},
                      S02_500_reference_scores_percent={key:100*value for key,value in parent_scores.items()},
                      resources=resources, strict_export=export, checkpoint_sha256=export["checkpoint_sha256"],
                      bare_student_sha256=export["bare_sha256"], common_step0_sha256=STEP0_SHA,
                      start_updates=0, resume=None, horizon=4868, first_five_gate=load(RUN / "first-five-gate.json"),
                      finished_at=now(), runtime=str(RUN), branch=BRANCH,
                      no_other_experiments=True, long_dci_protocol="Frozen reconstructed7602; not DCI Full")
        dump(EXP / "FULL_RESULTS.json", result)
        diagnostics = load(EXP / "TRAINING_DIAGNOSTICS.json")
        diagnostics.update(status="COMPLETE", observed_updates=4868, resources=resources,
                           epoch_boundaries=[self.metrics[step-1] for step in (1217,2434,3651,4868)],
                           first5=[item["loss"] for item in self.metrics[:5]], last50_loss_mean=statistics.fmean(item["loss"] for item in self.metrics[-50:]))
        dump(EXP / "TRAINING_DIAGNOSTICS.json", diagnostics)
        evidence = EXP / "evidence/recovery_full"
        evidence.mkdir(parents=True, exist_ok=True)
        with (RUN / "training_metrics.jsonl").open("rb") as source, gzip.open(evidence / "training_metrics.jsonl.gz", "wb") as destination:
            shutil.copyfileobj(source, destination)
        for name, source in sources.items():
            shutil.copy2(source, evidence / (name + ".json"))
        lines = ["# Arm B Summary S=0.2 full recovery result", "", "**" + decision + "**", "",
                 "Completed exactly4868 updates from common step0 (no resume), then strict native five-set evaluation.", "",
                 "| Model | Score5_R1 | J_long3 | J_long | Short4_R1 |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, values in (("RandomK formal4epoch", reference_scores), ("S02 historical500", parent_scores), ("S02 full4868", calculated)):
            lines.append("| " + name + " | " + " | ".join(f"{100*values[key]:.6f}" for key in calculated) + " |")
        lines += ["", "| Dataset | I2T R1 / R5 / R10 | T2I R1 / R5 / R10 |", "| --- | --- | --- |"]
        for name in DATASETS:
            lines.append("| " + name + " | " + " | ".join(" / ".join(f"{100*metrics[name][direction][recall]:.6f}"
                         for recall in ("R@1", "R@5", "R@10")) for direction in ("I2T", "T2I")) + " |")
        lines += ["", "## Requested conclusions", "", f"1. 500-step signal scales to full: {decision == 'FULL_POSITIVE'} (matched4868 comparison).",
                  f"2. Score5 >72.768147: {positive}.", f"3. J_long3 >=76.870244: {guard}.",
                  f"4. Urban both directions exceed RandomK: {comparisons['urban_both_directions_above_RandomK']}.",
                  f"5. Short4 exceeds RandomK: {comparisons['Short4_above_RandomK']}; deltas to RandomK/own500 are in FULL_RESULTS.json.",
                  "6. Long-DCI I2T/T2I deltas pp: " + json.dumps(comparisons["LongDCI_direction_deltas_pp"]), "",
                  "## Protocol/provenance", "", "F/S/D [1.4,0.2,1.4]; all model/objective/sampling/sparsity/inclusion200-ramp/optimizer/LR unchanged.",
                  "Only normalized bare native image/full-caption embeddings; no mask/gate/rerank/ensemble/Summary/Detail inference.",
                  "First-five exact sample/string/token/LR and model/optimizer comparisons passed before any sixth update.",
                  "Original distributed tail180/rank at each epoch preserved. Normal batch256/rank, global1024, accumulation1, seed0, four A10080GB.",
                  "Common step0 SHA256: `" + STEP0_SHA + "`.", "Final checkpoint SHA256: `" + export["checkpoint_sha256"] + "`.",
                  "Bare student SHA256: `" + export["bare_sha256"] + "`.", "Resources: " + json.dumps(resources), "",
                  "All checkpoints stay local. No extra experiment or fifth epoch. Training and evaluation have stopped."]
        (EXP / "FULL_RESULTS.md").write_text("\n".join(lines) + "\n")
        self.state.update(status="COMPLETE", stage="full4868 export and five-set evaluation completed; stopped", finished_at=now(),
                          classification=decision, scores_percent=percent, formal_training_authorized=False)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(load(RUN / "state.json"), indent=2))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (RUN / "state.json").exists() and not TRAIN.exists(), "Existing run preserved; no automatic restart/resume")
        supervisor = Supervisor()
        try:
            supervisor.run()
        except BaseException as error:
            supervisor.state.update(status="FAILED", error=type(error).__name__ + ": " + str(error), failed_at=now())
            if (RUN / "first-five-gate.json").exists():
                supervisor.state.update(status="BLOCKED_PREFIX_GATE", completed_updates=5,
                                        first_five_gate=load(RUN / "first-five-gate.json"))
            supervisor.state["formal_training_authorized"] = False
            supervisor.save()
            dump(EXP / "FULL_RESULTS.json", dict(supervisor.state, evaluated=False))
            (EXP / "FULL_RESULTS.md").write_text("# S02 full recovery blocked\n\n" + supervisor.state["error"] +
                "\n\nNo result is claimed; see FULL_PROGRESS.json and preserved local logs. No automatic resume or new trial.\n")
            raise


if __name__ == "__main__":
    main()
