"""Fresh0->500, read-only retrieval gate, conditional exact resume500->4868."""

import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from . import recovery_full as native_pipeline


ROOT = native_pipeline.ROOT
EXP = native_pipeline.EXP
RUNTIME = native_pipeline.RUNTIME
RUN = RUNTIME / "armb_summary02_500gate_recovery_v1"
PYTHON = native_pipeline.PYTHON
ASSETS = native_pipeline.ASSETS
MODULE = native_pipeline.MODULE
load = native_pipeline.load
dump = native_pipeline.dump
sha = native_pipeline.sha
require = native_pipeline.require
now = native_pipeline.now
REFERENCE = dict(Score5_R1=70.364367, J_long3=73.726611, J_long=82.765002, Short4_R1=65.321)
THRESHOLDS = dict(Score5_R1=70.164367, J_long3=73.403324, Urban_I2T=89.1, Urban_T2I=87.)


def reproduction_gate(percent, metrics, historical):
    values = dict(percent, Urban_I2T=100 * metrics["Urban-1k"]["I2T"]["R@1"],
                  Urban_T2I=100 * metrics["Urban-1k"]["T2I"]["R@1"])
    checks = {name: math.isfinite(values[name]) and values[name] >= threshold
              for name, threshold in THRESHOLDS.items()}
    deltas = {dataset: {direction: 100 * (metrics[dataset][direction]["R@1"] - historical[dataset][direction]["R@1"])
                        for direction in ("I2T", "T2I")} for dataset in native_pipeline.DATASETS}
    collapsed = [dataset + "/" + direction for dataset in native_pipeline.DATASETS for direction in ("I2T", "T2I")
                 if not math.isfinite(metrics[dataset][direction]["R@1"]) or
                 not 0 <= metrics[dataset][direction]["R@1"] <= 1 or
                 metrics[dataset][direction]["R@1"] < .5 * historical[dataset][direction]["R@1"]]
    checks["no_obvious_dataset_collapse"] = not collapsed
    return dict(status="REPRODUCTION_PASS" if all(checks.values()) else "REPRODUCTION_FAIL_AT_500",
                passed=all(checks.values()), checks=checks, thresholds_percent=THRESHOLDS,
                actual_percent=values, delta_vs_historical_S02_pp={key: percent[key] - value for key, value in REFERENCE.items()},
                dataset_R1_deltas_pp=deltas, collapsed_directions=collapsed,
                collapse_screen="Invalid/nonfinite R1 or loss of more than50% of historical R1 in any direction; no weight search",
                near_boundary={key: abs(values[key] - threshold) <= .05 for key, threshold in THRESHOLDS.items()},
                horizon=4868, continuation_checkpoint=str(RUN / "step500/step000500.pt"))


def publish(message):
    names = ("FULL_PROGRESS.json", "FULL_PREFLIGHT.json", "FIRST_FIVE_GATE.json", "TRAINING_DIAGNOSTICS.json",
             "STEP500_REPRODUCTION.md", "STEP500_RESULTS.json", "CONTINUATION_GATE.json", "STEP500_RESUME_AUDIT.json",
             "FULL_RESULTS.md", "FULL_RESULTS.json", "CHECKPOINT_SHA256.json", "STRICT_EXPORT_PROVENANCE.json")
    paths = [EXP / name for name in names if (EXP / name).is_file()]
    paths.extend((EXP / "commands").glob("*-500gate.json"))
    paths.extend((EXP / "evidence/reproduction").glob("*.json"))
    paths.append(ROOT / "recovery/evidence/recovery-operation-policy.json")
    relative = [str(path.relative_to(ROOT)) for path in paths]
    require(not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).strip(),
            "Refusing publication with unrelated staged changes")
    secrets = []
    for filename in ("config.json", "token.json"):
        path = Path.home() / ".openxlab" / filename
        if path.is_file():
            def collect(value, key=""):
                if isinstance(value, dict):
                    for child_key, child in value.items():
                        collect(child, child_key)
                elif isinstance(value, str) and re.search("secret|access.?key|token", key, re.I) and len(value) >= 12:
                    secrets.append(value.encode())
            collect(load(path))
    for path in paths:
        content = path.read_bytes()
        require(not any(secret in content for secret in secrets) and
                not re.search(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY|X-Amz-Signature=)", content),
                "Refusing secret-bearing publication: " + str(path.relative_to(ROOT)))
    subprocess.run(["git", "add", "-f", "--", *relative], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "-c", "user.name=Codex Recovery", "-c", "user.email=codex-recovery@users.noreply.github.com",
                        "commit", "-m", message], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", native_pipeline.BRANCH], cwd=ROOT, check=True, timeout=90)


class Supervisor(native_pipeline.Supervisor):
    def __init__(self):
        super().__init__()
        self.state.update(runtime=str(RUN), scheduled_evaluation="Pause at500; native five-set reproduction gate; then conditional4868",
                          first_stage_stop=500, horizon=4868, policy="Minimal first5 hard invariants; no cross-run numeric gate")
        self.train = RUN / "step500"
        native_pipeline.RUN = RUN
        native_pipeline.TRAIN = self.train

    def execute(self, name, command, training=False):
        os.environ["SAID_S02_STAGE"] = self.train.name
        super().execute(name, command, training)
        old = EXP / "commands" / (name + "-recovery.json")
        if old.exists():
            old.replace(EXP / "commands" / (name + "-500gate.json"))

    def train_stage(self, stop, resume=None):
        self.train = RUN / ("step500" if stop == 500 else "step4868")
        native_pipeline.TRAIN = self.train
        self.offset = 0
        command = [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                   "-m", MODULE + ".reproduction_train_gate", "--config", str(ROOT / "recovery/configs/summary02.json"),
                   "--init-state", str(RUNTIME / "shared/step000000.pt"), "--index-dir", str(RUNTIME / "data_index"),
                   "--image-root", str(ASSETS / "training/ShareGPT4V"), "--output-dir", str(self.train),
                   "--run-type", "formal", "--max-updates", str(stop)]
        if resume:
            require(stop == 4868 and resume == RUN / "step500/step000500.pt", "Invalid continuation checkpoint")
            command.extend(("--resume", str(resume)))
        self.execute("train" + str(stop), command, training=True)
        acceptance = load(self.train / "acceptance.json")
        require(acceptance["passed"] and len(self.metrics) == stop and
                all(rank["completed_updates"] == stop and rank["max_parameter_difference_from_rank0"] == 0
                    for rank in acceptance["ranks"]), "Native stage acceptance failed")
        require(load(RUN / "first-five-gate.json")["passed"], "Minimal hard gate missing")
        return acceptance

    def evaluate(self, step):
        checkpoint = self.train / f"step{step:06d}.pt"
        checkpoint_sha = sha(checkpoint)
        bare = self.train / f"student_step{step}.pt"
        self.execute(f"export{step}", [PYTHON, "-m", "tools.nest_clip", "export", "--checkpoint", str(checkpoint),
                                    "--output", str(bare), "--expect-updates", str(step)])
        self.execute(f"verify{step}", [PYTHON, "-m", "tools.nest_clip", "verify-export", "--checkpoint", str(checkpoint),
                                    "--bare", str(bare), "--output", str(self.train / "export-check.json"),
                                    "--index-dir", str(RUNTIME / "data_index"), "--image-root", str(ASSETS / "training/ShareGPT4V")])
        for name, folder in (("coco", ASSETS / "evaluation/coco/val2017"), ("urban", ASSETS / "evaluation/Urban1k/Urban1k")):
            self.execute(f"eval{step}-{name}", [PYTHON, "-m", "tools.eval_nest_native", "--checkpoint", str(bare),
                         "--dataset", name, "--root", str(folder), "--device", "cuda:0", "--batch-size", "64",
                         "--output", str(self.train / (name + "_native.json"))])
        bench = ASSETS / "retrieval_benchmarks"
        for name, manifest, folder in (("flickr_test1k", "flickr30k_test1k.jsonl", "flickr30k"),
                                      ("docci", "docci_test.jsonl", "docci"), ("long_dci", "long_dci_reconstructed.jsonl", "dci")):
            self.execute(f"eval{step}-{name}", [PYTHON, "-m", "experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real",
                         "--checkpoint", str(bare), "--device", "cuda:0", "--batch-size", "64", "--output-dir", str(self.train / name),
                         f"{name}:{bench / 'manifests' / manifest}:{bench / folder / 'images'}"])
        require(checkpoint_sha == sha(checkpoint), "Evaluation modified the resumable training checkpoint")
        from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores
        metrics, raw, sources = native_metrics(self.train)
        calculated = scores(metrics)
        calculated["Short4_R1"] = statistics.fmean(metrics[name][direction]["R@1"]
                                  for name in ("COCO", "Flickr30k-test1k") for direction in ("I2T", "T2I"))
        export = load(self.train / "export-check.json")
        require(export["passed"] and export["checkpoint_sha256"] == checkpoint_sha, "Strict export failed")
        evidence = EXP / "evidence/reproduction"
        evidence.mkdir(parents=True, exist_ok=True)
        for name, source in sources.items():
            shutil.copy2(source, evidence / (f"step{step}-" + name + ".json"))
        return dict(metrics=metrics, scores=calculated, scores_percent={key: value * 100 for key, value in calculated.items()},
                    strict_export=export, checkpoint_sha256=checkpoint_sha, evaluation_checkpoint_immutable=True,
                    step=step, horizon=4868, native_full_caption_only=True, finished_at=now())

    def sync(self, message):
        try:
            publish(message)
        except Exception as error:
            self.state["github_sync_error"] = type(error).__name__ + ": " + str(error)
            self.save()

    def run(self):
        policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
        policy = load(policy_path)
        require(policy.get("reproduction_gate_authorized") is True, "Latest500-gate authorization missing")
        self.save()
        preflight = native_pipeline.preflight()
        preflight.update(runtime_guard_sha256=sha(EXP / "reproduction_train_gate.py"),
                         supervisor_sha256=sha(EXP / "reproduction_full.py"), first_stage_stop=500,
                         modifications="Read-only minimal invariant gate; faithful native scheduler/scaler checkpoint metadata; conditional exact resume at500")
        dump(EXP / "FULL_PREFLIGHT.json", preflight)
        dump(EXP / "CONTINUATION_GATE.json", dict(status="PENDING_AT_500", horizon=4868, passed=False, thresholds_percent=THRESHOLDS))
        dump(EXP / "FULL_RESULTS.json", dict(status="RUNNING_0_TO_500", evaluated=False, horizon=4868, runtime=str(RUN)))
        (EXP / "FULL_RESULTS.md").write_text("# S02 full trajectory\n\nFresh common step0 ->500, horizon4868. Retrieval gate pending; no full result claimed.\n")
        acceptance500 = self.train_stage(500)
        result500 = self.evaluate(500)
        gate = reproduction_gate(result500["scores_percent"], result500["metrics"], load(EXP / "PARENT_500.json")["metrics"])
        gate.update(checkpoint_sha256=result500["checkpoint_sha256"], evaluated_at=now())
        result500.update(status=gate["status"], gate=gate, native_acceptance=acceptance500)
        dump(EXP / "STEP500_RESULTS.json", result500)
        dump(EXP / "CONTINUATION_GATE.json", gate)
        lines = ["# S02 step500 retrieval reproduction", "", "**" + gate["status"] + "**", "",
                 "Fresh common step0; scheduler horizon4868 throughout. Only500 updates so far, not a500-horizon trial.", "",
                 "| Metric | Actual pp | Historical pp | Delta pp |", "|---|---:|---:|---:|"]
        lines.extend(f"| {name} | {result500['scores_percent'][name]:.6f} | {reference:.6f} | {gate['delta_vs_historical_S02_pp'][name]:+.6f} |"
                     for name, reference in REFERENCE.items())
        lines.extend(["", "Gate checks: `" + json.dumps(gate["checks"]) + "`.",
                      "Collapse screen: " + gate["collapse_screen"] + ". All directional deltas and R@1/5/10 in STEP500_RESULTS.json.",
                      "Training checkpoint SHA256 unchanged before/after separate-process evaluation: `" + result500["checkpoint_sha256"] + "`.",
                      "Optimizer, manual scheduler, BF16/no-scaler metadata, four-rank RNG and exact sampler cursor saved.",
                      "No cross-run loss/model/moment equality gate. No hyperparameter or weight search."])
        (EXP / "STEP500_REPRODUCTION.md").write_text("\n".join(lines) + "\n")
        self.state.update(status=gate["status"], reproduction_gate=gate)
        if not gate["passed"]:
            self.state.update(formal_training_authorized=False, stage="Stopped at500; reproduction gate failed")
            self.save()
            dump(EXP / "FULL_RESULTS.json", dict(status=gate["status"], completed_updates=500, full_evaluated=False, step500=result500))
            (EXP / "FULL_RESULTS.md").write_text("# S02 full stopped\n\nREPRODUCTION_FAIL_AT_500. See STEP500_REPRODUCTION.md; no continuation or new experiment.\n")
            policy.update(formal_training_authorized=False, full_gate_status=gate["status"], full_completed_updates=500)
            dump(policy_path, policy)
            self.sync("Report S02 step500 reproduction failure; stop trajectory")
            return
        self.save()
        policy.update(resume_allowed=True, permitted_resume_checkpoint=gate["continuation_checkpoint"],
                      permitted_resume_sha256=gate["checkpoint_sha256"], full_gate_status="REPRODUCTION_PASS", full_completed_updates=500)
        dump(policy_path, policy)
        self.sync("Report S02 step500 retrieval gate; authorize same-trajectory continuation")
        checkpoint = RUN / "step500/step000500.pt"
        require(sha(checkpoint) == gate["checkpoint_sha256"], "Step500 changed before continuation")
        acceptance = self.train_stage(4868, checkpoint)
        require(load(EXP / "STEP500_RESUME_AUDIT.json")["passed"], "Missing exact resume proof")
        result = self.evaluate(4868)
        self.finish_result(result, acceptance, result500)
        policy.update(formal_training_authorized=False, resume_allowed=False, full_completed_updates=4868,
                      full_gate_status="COMPLETE", reason="This trajectory and both frozen native evaluations completed; no further experiment")
        dump(policy_path, policy)
        self.sync("Report S02 full4868 results and strict native five-set evaluation")

    def finish_result(self, result, acceptance, result500):
        reference = load(EXP / "RANDOMK_4EPOCH_REFERENCE.json")
        reference_scores = dict(reference["scores"], Short4_R1=statistics.fmean(
            reference["metrics"][name][direction]["R@1"] for name in ("COCO", "Flickr30k-test1k") for direction in ("I2T", "T2I")))
        positive = result["scores_percent"]["Score5_R1"] > 72.768147
        guard = result["scores_percent"]["J_long3"] >= 76.870244
        decision = "FULL_POSITIVE" if positive and guard else "SHORT_LONG_TRADEOFF" if positive else "EARLY_SIGNAL_DID_NOT_SCALE"
        cycles = [json.loads(line) for stage in ("step500", "step4868")
                  for line in (RUN / stage / "cycle_timing.jsonl").read_text().splitlines()]
        normal = [row["four_rank_max_seconds"] for row in cycles if not row["warmup"]]
        resources = dict(mean_post_warmup_cycle_seconds=statistics.fmean(normal),
                         median_post_warmup_cycle_seconds=statistics.median(normal),
                         peak_allocated_gib=max(rank["peak_allocated_gib"] for rank in acceptance["ranks"]))
        checkpoints = {str(path.relative_to(RUN)): sha(path) for path in
                       [RUN / "step500/step000005.pt", RUN / "step500/step000500.pt"] +
                       [self.train / f"step{step:06d}.pt" for step in (1217, 2434, 3651, 4868)]}
        dump(EXP / "CHECKPOINT_SHA256.json", checkpoints)
        dump(EXP / "STRICT_EXPORT_PROVENANCE.json", result["strict_export"])
        result.update(status="COMPLETE", classification=decision, completed_updates=4868,
                      common_step0_sha256=native_pipeline.STEP0_SHA, initial_start_updates=0,
                      resumed_same_trajectory_at500=True, continuation_gate=load(EXP / "CONTINUATION_GATE.json"),
                      exact_resume_proof=load(EXP / "STEP500_RESUME_AUDIT.json"), native_acceptance=acceptance,
                      resources=resources,
                      delta_vs_RandomK_pp={key: 100 * (value - reference_scores[key]) for key, value in result["scores"].items()},
                      delta_vs_historical_S02_500_pp={key: result["scores_percent"][key] - value for key, value in REFERENCE.items()},
                      delta_vs_same_trajectory_500_pp={key: result["scores_percent"][key] - result500["scores_percent"][key] for key in REFERENCE},
                      urban_both_above_RandomK=all(result["metrics"]["Urban-1k"][direction]["R@1"] > reference["metrics"]["Urban-1k"][direction]["R@1"]
                                                 for direction in ("I2T", "T2I")),
                      short4_above_RandomK=result["scores"]["Short4_R1"] > reference_scores["Short4_R1"],
                      LongDCI_direction_deltas_pp={direction: 100 * (result["metrics"]["Long-DCI"][direction]["R@1"] - reference["metrics"]["Long-DCI"][direction]["R@1"])
                                                  for direction in ("I2T", "T2I")},
                      runtime=str(RUN), no_other_experiments=True)
        dump(EXP / "FULL_RESULTS.json", result)
        diagnostics = load(EXP / "TRAINING_DIAGNOSTICS.json")
        diagnostics.update(status="COMPLETE", resources=resources, observed_updates=4868,
                           first5=[row["loss"] for row in self.metrics[:5]],
                           epoch_boundaries=[self.metrics[step - 1] for step in (1217, 2434, 3651, 4868)])
        dump(EXP / "TRAINING_DIAGNOSTICS.json", diagnostics)
        (EXP / "FULL_RESULTS.md").write_text("# S02 full4868 result\n\n**" + decision + "**\n\n" +
            "Fresh common step0; read-only500 retrieval gate passed; exact resume of the same trajectory to4868.\n\n" +
            "Scores (pp): `" + json.dumps(result["scores_percent"]) + "`.\n\n" +
            "Delta vs RandomK (pp): `" + json.dumps(result["delta_vs_RandomK_pp"]) + "`.\n\n" +
            "Urban both above RandomK: " + str(result["urban_both_above_RandomK"]) + "; Short4 above: " + str(result["short4_above_RandomK"]) + ".\n\n" +
            "LongDCI direction deltas pp: `" + json.dumps(result["LongDCI_direction_deltas_pp"]) + "`.\n\n" +
            "Strict bare native full-caption inference only; five frozen protocols; all bidirectional R@1/5/10 and SHA256 in FULL_RESULTS.json.\n" +
            "Training/evaluation stopped. No checkpoints uploaded; no additional experiment.\n")
        self.state.update(status="COMPLETE", stage="Full4868 and five-set evaluation complete; stopped",
                          formal_training_authorized=False, classification=decision, scores_percent=result["scores_percent"], finished_at=now())
        self.save()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (RUN / "state.json").exists() and not (RUN / "step500").exists(), "Preserve existing trajectory; no implicit restart")
        supervisor = Supervisor()
        try:
            supervisor.run()
        except BaseException as error:
            supervisor.state.update(status="STOPPED_ERROR", error=type(error).__name__ + ": " + str(error),
                                    formal_training_authorized=False, failed_at=now())
            supervisor.save()
            dump(EXP / "FULL_RESULTS.json", dict(supervisor.state, full_evaluated=False))
            (EXP / "FULL_RESULTS.md").write_text("# S02 trajectory stopped\n\n" + supervisor.state["error"] +
                                               "\n\nNo full result claimed. Preserved checkpoint/logs; no implicit restart or alternate experiment.\n")
            policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
            policy = load(policy_path)
            policy.update(formal_training_authorized=False, full_gate_status="STOPPED_ERROR")
            dump(policy_path, policy)
            supervisor.sync("Report S02 reproduction trajectory stopped safely")
            raise


if __name__ == "__main__":
    main()
