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
from recovery import local_ssd_stage as local_stage


ROOT = native_pipeline.ROOT
EXP = native_pipeline.EXP
RUNTIME = native_pipeline.RUNTIME
RUN = RUNTIME / "armb_summary02_500gate_localssd_v3"
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


def resource_recurrence(cycles):
    consecutive = 0
    for update, cycle in enumerate(cycles, start=1):
        if update > 1:
            consecutive = consecutive + 1 if cycle["four_rank_max_seconds"] > 3 else 0
            if consecutive >= 3:
                return dict(triggered=True, step=cycle["step"], threshold_seconds=3, consecutive_updates=3)
    return dict(triggered=False, threshold_seconds=3, consecutive_updates=3)


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
             "FULL_RESULTS.md", "FULL_RESULTS.json", "CHECKPOINT_SHA256.json", "STRICT_EXPORT_PROVENANCE.json",
             "RESOURCE_STALL_RECURRENCE.json", "RESOURCE_STALL_RECURRENCE.md", "LOCAL_SSD_STAGE500_REPORT.md",
             "LOCAL_SSD_STAGE500_READY.json", "LOCAL_SSD_MANIFEST_PROOF.json", "LOCAL_SSD_COPY_PROGRESS.json",
             "reproduction_full.py", "reproduction_train_gate.py", "local_image_dataset.py")
    paths = [EXP / name for name in names if (EXP / name).is_file()]
    paths.extend((EXP / "commands").glob("*-500gate.json"))
    paths.extend((EXP / "evidence/reproduction").glob("*.json"))
    paths.append(ROOT / "recovery/evidence/recovery-operation-policy.json")
    paths.extend(ROOT / "recovery" / name for name in ("local_ssd_stage.py", "test_local_ssd_stage.py"))
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
                elif isinstance(value, str) and re.search("secret|access.?key|token|^(ak|sk)$", key, re.I) and len(value) >= 12:
                    secrets.append(value.encode())
            collect(load(path))
    for path in paths:
        content = path.read_bytes()
        require(not any(secret in content for secret in secrets) and
                not re.search(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|^-----BEGIN [A-Z ]*PRIVATE KEY-----|X-Amz-Signature=[0-9a-fA-F]{32,})", content, re.M),
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
        if training:
            local = Path("/tmp/said-s02-full-phase-localssd-v3") / self.train.name
            local.mkdir(parents=True, exist_ok=False)
            os.environ["SAID_S02_PHASE_LOCAL"] = str(local)
            self.state["phase_telemetry"] = str(local)
        super().execute(name, command, training)
        self.state.update(active_pid=None, training_process_running=False)
        self.save()
        old = EXP / "commands" / (name + "-recovery.json")
        if old.exists():
            old.replace(EXP / "commands" / (name + "-500gate.json"))

    def train_stage(self, stop, resume=None):
        self.train = RUN / ("step500" if stop == 500 else "step4868")
        native_pipeline.TRAIN = self.train
        self.offset = 0
        command = [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                   "-m", MODULE + ".reproduction_train_gate", "--config", str(ROOT / "recovery/configs/summary02.json"),
                   "--init-state", str(RUNTIME / "shared/step000000.pt"), "--index-dir", str(local_stage.INDEX),
                   "--image-root", str(local_stage.IMAGES), "--output-dir", str(self.train),
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
        self.local_report()
        try:
            publish(message)
        except Exception as error:
            self.state["github_sync_error"] = type(error).__name__ + ": " + str(error)
            self.save()

    def local_report(self):
        manifest = load(EXP / "LOCAL_SSD_MANIFEST_PROOF.json") if (EXP / "LOCAL_SSD_MANIFEST_PROOF.json").exists() else {}
        copied = load(local_stage.META / "stage500-copy-ready.json") if (local_stage.META / "stage500-copy-ready.json").exists() else {}
        if not copied and (local_stage.META / "stage500-copy-progress.json").exists():
            copied = load(local_stage.META / "stage500-copy-progress.json")
        ready = load(local_stage.META / "stage500-ready.json") if (local_stage.META / "stage500-ready.json").exists() else {}
        result = load(EXP / "STEP500_RESULTS.json") if (EXP / "STEP500_RESULTS.json").exists() else {}
        if self.state["status"] == "LOCAL_SSD_COPY_IN_PROGRESS":
            result = {}
        lines = ["# Local SSD first500 staging and reproduction", "", "Updated UTC: " + now(), "",
                 "Trajectory: `" + str(RUN) + "`; status: `" + self.state["status"] + "`.", "",
                 "IO-only override; frozen sampling/preprocessing/model/loss/optimizer and horizon4868 unchanged. "
                 "Native3 consecutive full cycles>3s protection unchanged. Fresh common step0; no resume of diagnostic or stopped runs.", "",
                 "Local block-backed root-overlay/NVMe mirror: `" + str(local_stage.IMAGES) + "`.", "",
                 "## Manifest and copy", "",
                 "512000 records requested (500 x1024), de-duplicated images: `" + str(manifest.get("unique_images_count", "PENDING")) + "`.",
                 "Family counts: `" + json.dumps(manifest.get("family_counts", {})) + "`.",
                 "Payload bytes: `" + str(manifest.get("total_bytes", "PENDING")) + "`.",
                 "Payload GB / GiB: `" + str(round(manifest.get("total_bytes", 0) / 10**9, 3)) + " / " +
                     str(round(manifest.get("total_bytes", 0) / local_stage.GIB, 3)) + "`.",
                 "Full historical500 sample-ID stream / first5 FSD+tokens: `" + json.dumps({key:manifest.get(key) for key in
                     ("historical_sample_ids_matched", "historical_FSD_string_token_samples_matched", "offline_global_RNG_unchanged")}) + "`.",
                 "Copy: `" + json.dumps(copied) + "`.",
                 "Every required first500 image source SHA256 compared with SSD reread SHA256; atomic raw-byte copy. "
                 "O_DIRECT source reads; private destination-file cache advice only. No global drop_caches, no source-cache eviction.", "",
                 "1000-example byte/RGB/native preprocess proof and cgroup/process admission: `" + json.dumps(ready) + "`."]
        proof = []
        phase_root = Path("/tmp/said-s02-full-phase-localssd-v3")
        for stage in ("step500", "step4868"):
            for path in (phase_root / stage).glob("image-paths-*.jsonl"):
                proof.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        if proof:
            require(all(row["resolved_image_path"].startswith(str(local_stage.IMAGES) + "/") and
                        row["path_from_NFS"] is False for row in proof), "Actual worker path is not local")
            dump(EXP / "evidence/reproduction/local-image-path-proof.json", dict(passed=True, samples=len(proof), rows=proof))
        lines.extend(["", "## Training and retrieval", "",
                      f"Actual native worker resolved-path proofs: {len(proof)} samples, ranks: {sorted({row['rank'] for row in proof})}.",
                      "Phase telemetry: `" + str(phase_root) + "`; data_wait/H2D/forward/backward/DDP/optimizer/cgroup/PSI recorded."])
        cycle_path = RUN / "step500/cycle_timing.jsonl"
        if cycle_path.exists():
            from recovery.resource_stall_v2 import summary
            rows = [json.loads(line) for line in cycle_path.read_text().splitlines() if line]
            steady = [row["four_rank_max_seconds"] for row in rows if row["step"] >= 7]
            lines.append("Step7+ full cycles: `" + json.dumps(summary(steady)) + "`; protection: `" + json.dumps(resource_recurrence(rows)) + "`.")
        lines.extend(["Retrieval status: `" + str(result.get("status", "PENDING")) + "`; evaluated: `" + str(result.get("step") == 500 and "metrics" in result) + "`.",
                      "Scores percent: `" + json.dumps(result.get("scores_percent")) + "`.", "",
                      "Remaining images are NOT copied during500 training/evaluation. Only after REPRODUCTION_PASS: paused training, "
                      "manifest-driven full local coverage+SHA verification, unchanged same500 checkpoint, then exact-state continuation. "
                      "Old stall root cause is not claimed repaired; resource guard retained."])
        (EXP / "LOCAL_SSD_STAGE500_REPORT.md").write_text("\n".join(lines) + "\n")

    def resource_stop(self, error):
        cycle_path = self.train / "cycle_timing.jsonl"
        if not cycle_path.exists():
            return False
        cycles = [json.loads(line) for line in cycle_path.read_text().splitlines() if line]
        recurrence = resource_recurrence(cycles)
        if not recurrence["triggered"]:
            return False
        status = "RESOURCE_STALL_RECURRED_BEFORE_500" if self.train.name == "step500" else "RESOURCE_STALL_RECURRED_DURING_FULL"
        local = Path(self.state["phase_telemetry"])
        phases = {}
        for rank in range(4):
            path = local / f"rank{rank}.jsonl"
            phases[rank] = {row["step"]: row for row in
                            ([json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else [])}
        recent = [dict(step=row["step"], full_cycle_s=row["four_rank_max_seconds"],
                       ranks=[phases[rank].get(row["step"], dict(rank=rank, timing_unavailable=True)) for rank in range(4)])
                  for row in cycles[-20:]]
        proof = dict(status=status, method_failure=False, automatic_retry=False, trigger=recurrence,
                     completed_updates=self.state["completed_updates"], horizon=4868,
                     recent20_steps=recent, phase_telemetry=str(local), native_cycle_log=str(cycle_path),
                     error=type(error).__name__ + ": " + str(error), checkpoint_resume_authorized=False,
                     root_cause_previously_unresolved=True)
        dump(EXP / "RESOURCE_STALL_RECURRENCE.json", proof)
        dump(RUN / "resource-stall-recurrence.json", proof)
        destination = RUN / "phase-evidence" / self.train.name
        destination.mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            path = local / f"rank{rank}.jsonl"
            if path.exists():
                shutil.copyfile(path, destination / path.name)
        (EXP / "RESOURCE_STALL_RECURRENCE.md").write_text(
            "# S02 resource protection stop\n\n**" + status + "**\n\n" +
            f"Stopped at optimizer update{self.state['completed_updates']}; horizon4868. Native3-second x3 consecutive gate unchanged.\n\n" +
            "This is a resource/environment stop, not a retrieval/method failure. No automatic retry or continuation. " +
            "Last20 full cycles and per-rank data/H2D/forward/backward-DDP/optimizer/cgroup/file-cache/PSI/GPU samples are in RESOURCE_STALL_RECURRENCE.json.\n")
        self.state.update(status=status, formal_training_authorized=False, method_failure=False, active_pid=None,
                          training_process_running=False,
                          automatic_retry=False, failed_at=now(), resource_trigger=recurrence)
        self.save()
        dump(EXP / "FULL_RESULTS.json", dict(self.state, full_evaluated=False))
        (EXP / "FULL_RESULTS.md").write_text("# S02 trajectory stopped\n\n**" + status + "**\n\nResource gate, not method failure. No automatic restart; logs/checkpoint preserved.\n")
        if self.train.name == "step500":
            dump(EXP / "STEP500_RESULTS.json", dict(status=status, completed_updates=self.state["completed_updates"],
                                                    evaluated=False, scores_percent=None, method_failure=False, horizon=4868))
            dump(EXP / "CONTINUATION_GATE.json", dict(status=status, passed=False, evaluated=False,
                                                       automatic_retry=False, horizon=4868, thresholds_percent=THRESHOLDS))
            (EXP / "STEP500_REPRODUCTION.md").write_text("# S02 step500 reproduction\n\n**" + status + "**\n\nStep500 was not reached/evaluated. Retrieval reproduction cannot be judged. See RESOURCE_STALL_RECURRENCE.md.\n")
        policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
        policy = load(policy_path)
        policy.update(formal_training_authorized=False, reproduction_gate_authorized=False,
                      resume_allowed=False, full_gate_status=status, full_completed_updates=self.state["completed_updates"],
                      training_held_for_new_instruction=True)
        dump(policy_path, policy)
        self.sync("Report S02 resource stall recurrence; preserve20-step phase evidence and stop without retry")
        return True

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
        if not (local_stage.META / "manifest-proof.json").exists():
            self.execute("local-manifest", [PYTHON, "-m", "recovery.local_ssd_stage", "prepare"])
        if not (local_stage.META / "stage500-copy-ready.json").exists():
            self.execute("local-copy500", [PYTHON, "-m", "recovery.local_ssd_stage", "copy500"])
        self.execute("local-verify500", [PYTHON, "-m", "recovery.local_ssd_stage", "verify"])
        require(load(local_stage.META / "stage500-ready.json")["passed"], "Local mirror admission failed")
        require(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(),
                "GPUs became occupied during staging")
        self.sync("Record hash-verified local first500 mirror and unchanged native resource guard")
        dump(EXP / "CONTINUATION_GATE.json", dict(status="PENDING_AT_500", horizon=4868, passed=False, thresholds_percent=THRESHOLDS))
        dump(EXP / "STEP500_RESULTS.json", dict(status="PENDING_AT_500", evaluated=False, scores_percent=None,
                                                horizon=4868, runtime=str(RUN)))
        (EXP / "STEP500_REPRODUCTION.md").write_text("# S02 step500 reproduction\n\nPENDING_AT_500. Fresh common step0, horizon4868; no retrieval result claimed. Native resource protection unchanged.\n")
        dump(EXP / "FULL_RESULTS.json", dict(status="RUNNING_0_TO_500", evaluated=False, horizon=4868, runtime=str(RUN)))
        (EXP / "FULL_RESULTS.md").write_text("# S02 full trajectory\n\nFresh common step0 ->500, horizon4868. Retrieval gate pending; no full result claimed.\n")
        acceptance500 = self.train_stage(500)
        result500 = self.evaluate(500)
        gate = reproduction_gate(result500["scores_percent"], result500["metrics"], load(EXP / "PARENT_500.json")["metrics"])
        gate.update(checkpoint_sha256=result500["checkpoint_sha256"], evaluated_at=now())
        result500.update(status=gate["status"], gate=gate, native_acceptance=acceptance500, runtime=str(RUN))
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
        policy.update(resume_allowed=False, full_gate_status="REPRODUCTION_PASS_LOCAL_FULL_COPY_PENDING", full_completed_updates=500)
        dump(policy_path, policy)
        self.state.update(status="REPRODUCTION_PASS_LOCAL_FULL_COPY_PENDING", stage="Training paused; full local mirror required before501")
        self.save()
        self.sync("Report S02 reproduction pass; pause at500 for full local staging")
        self.execute("local-copyfull", [PYTHON, "-m", "recovery.local_ssd_stage", "copyfull"])
        full_ready = load(local_stage.META / "full-copy-ready.json")
        require(full_ready["checked"] == 1245901 and full_ready["families"] == local_stage.EXPECTED and
                full_ready["source_destination_SHA256_matches"] == 1245901 and full_ready["copy_workers_exited"],
                "Full local mirror incomplete")
        require(local_stage.process_audit()["workers_all_exited"], "Copy/audit workers remain")
        snapshot = local_stage.system_snapshot()
        require(snapshot["memory_current"] < .98 * int(snapshot["memory_max"]), "Cgroup still near limit after full local copy")
        dump(EXP / "evidence/reproduction/full-local-ready.json", dict(full_ready, system=snapshot))
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
            if isinstance(error, Exception) and supervisor.resource_stop(error):
                return
            supervisor.state.update(status="STOPPED_ERROR", error=type(error).__name__ + ": " + str(error), active_pid=None,
                                    training_process_running=False,
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
