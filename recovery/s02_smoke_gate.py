"""Verify and quarantine an already executed five-update S02 smoke, never train."""

import argparse
import datetime
import gc
import hashlib
import itertools
import json
import math
import mmap
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "recovery/evidence"
RUNTIME = ROOT / "runtime/SAID-nest-clip-v1"
STEP0_SHA = "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6"
WEIGHTS = [1.4, 0.2, 1.4]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_config(config):
    expected = dict(view_weights=WEIGHTS, sampling_mode="summary_random_detail",
                    batch_size=256, world_size=4, accumulation=1, seed=0, workers=8,
                    epochs=4, fusion="balanced_stack", visual="patch",
                    condition_mode="dual_branch", fusion_lr=2e-4,
                    visual_mask_lr_scale=1., sparsity_scale=1., inclusion_max=1.,
                    horizon=4868, batches_per_epoch=1217, max_updates=5,
                    start_updates=0, updates_planned_this_run=5, run_type="smoke",
                    training_records=1245901, init_sha256=STEP0_SHA,
                    sampling_seed=0, shuffle_seed=0, full_native_mix=0,
                    image_chunk=128, text_chunk=128, checkpoint_encoders=True,
                    checkpoint_pair_blocks=False, hparam_search=True,
                    four_epoch_followup=True, checkpoint_interval=1217)
    require(all(config.get(key) == value for key, value in expected.items()),
            "Runtime config differs from the frozen S02 five-update protocol")
    require(config.get("resume") is None and config.get("summary_t2i_weight", 1.) == 1.,
            "Old-checkpoint resume or nonhistorical summary-T2I weighting")
    require(len(config["ranks"]) == 4 and
            len({rank["uuid"] for rank in config["ranks"]}) == 4 and
            all("A100" in rank["gpu_name"] and rank["total_memory"] > 79 * 2**30
                for rank in config["ranks"]), "Four distinct A100 80GB GPUs not proven")
    require(config["gather_gradient_test"]["passed"], "DDP gather-gradient test failed")


def check_steps(rows, acceptance):
    require([row["step"] for row in rows] == [1, 2, 3, 4, 5],
            "Smoke did not complete exactly optimizer updates 1..5")
    require(acceptance["passed"], "Native acceptance failed: " +
            str(acceptance.get("resource_failure")))
    require(len(acceptance["ranks"]) == 4 and
            {rank["rank"] for rank in acceptance["ranks"]} == {0, 1, 2, 3},
            "Missing DDP rank acceptance")
    for rank in acceptance["ranks"]:
        require(rank["completed_updates"] == rank["updates_this_run"] == 5,
                "A rank did not update exactly five times")
        require(rank["max_parameter_difference_from_rank0"] == 0 and
                rank["final_nccl_all_reduce"] == 10., "Final DDP parameter/NCCL mismatch")
    metrics = []
    for row in rows:
        require(not row["duplicate_image_ids"], "Duplicate global image IDs")
        require(row["nonfinite"] == 0 and math.isfinite(row["loss"]), "Nonfinite loss")
        require(len(row["rank_health"]) == 4 and
                {health["rank"] for health in row["rank_health"]} == {0, 1, 2, 3},
                "Missing per-step rank health")
        for health in row["rank_health"]:
            require(health["batch"] == 256 and health["updates"] == row["step"] and
                    health["gradients_finite"], "Invalid batch/update/gradient health")
            require(all(math.isfinite(value) for value in health["gradient_norms"].values()),
                    "Nonfinite gradient norm")
        contribution = {label: 10 / sum(WEIGHTS) * weight *
                        (row[prefix + "_i2t"] + row[prefix + "_t2i"])
                        for label, prefix, weight in zip(("F", "S", "D"), ("F", "O", "E"), WEIGHTS)}
        sparse = (row["F_sparse"] + 2 * row["O_sparse"] + 2 * row["E_sparse"]) / 3
        inclusion = row["inc"] * row["inc_weight"]
        require(math.isclose(row["loss"], sum(contribution.values()) + sparse + inclusion,
                             rel_tol=2e-5, abs_tol=2e-4), "Native weighted loss decomposition mismatch")
        completed = row["step"] - 1
        factor = .5 * (1 + math.cos(math.pi * completed / 4868))
        expected_lr = dict(backbone=1e-6 * (completed + 1) / 200,
                           text_mask_and_shared_pool=1e-3 * factor,
                           visual_mask=1e-3 * factor, fusion_adapter=2e-4 * factor)
        require(row["actual_lrs"] == expected_lr, "LR not generated with historical H4868")
        metrics.append(dict(step=row["step"], loss=row["loss"],
                            CE={label: dict(I2T=row[prefix + "_i2t"], T2I=row[prefix + "_t2i"])
                                for label, prefix in zip(("F", "S", "D"), ("F", "O", "E"))},
                            weighted_alignment=contribution, sparsity=sparse,
                            inclusion=inclusion, inclusion_raw=row["inc"], inclusion_ramp=row["inc_weight"],
                            keep_ratio={label: row[prefix + "_keep_ratio"]
                                        for label, prefix in zip(("F", "S", "D"), ("F", "O", "E"))},
                            gate_stats={key: value for key, value in row.items()
                                        if "_g_" in key or "sigmoid_saturation" in key or
                                        key.endswith(("_all_open", "_all_closed", "_positive_keep_ratio", "_negative_keep_ratio"))},
                            LR=row["actual_lrs"], rank_health=[compact_health(health) for health in row["rank_health"]]))
    return metrics


def compact_health(health):
    result = {key: value for key, value in health.items() if key != "sampling"}
    if "sampling" in health:
        sampling = health["sampling"]
        result["sampling"] = {key: value for key, value in sampling.items()
                              if key.endswith("_sha256") or key in ("valid_count", "k1_count")}
        result["sampling"]["sample_ids_first8"] = sampling["sample_ids"][:8]
        result["sampling"]["random_detail_indices_sha256"] = sampling["random_detail_sampling"]["selected_indices_sha256"]
    return result


def replay_sample_stream(rows):
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views

    index = RUNTIME / "data_index"
    offsets = np.load(index / "offsets.npy", mmap_mode="r")
    with (index / "records.jsonl").open("rb") as handle:
        records = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for rank in range(4):
                sampler = DistributedSampler(range(1245901), num_replicas=4,
                                             rank=rank, shuffle=True, seed=0, drop_last=False)
                sampler.set_epoch(0)
                indices = list(itertools.islice(iter(sampler), 5 * 256))
                for row in rows:
                    health = next(item for item in row["rank_health"] if item["rank"] == rank)
                    selection = indices[(row["step"] - 1) * 256:row["step"] * 256]
                    ids = [index_id + 1000 for index_id in selection]
                    require(ids == health["sampling"]["sample_ids"], "DistributedSampler prefix mismatch")
                    samples = []
                    for index_id in selection:
                        record = json.loads(records[offsets[index_id]:offsets[index_id + 1]])
                        sample = sampled_text_views(record["caption"], "summary_random_detail", 0, 0, index_id + 1000)
                        if sample["valid"]:
                            parts = sample["views"][0].split(". ")
                            require(sample["views"][1] == parts[0] and
                                    sample["views"][2] == ". ".join(parts[position] for position in sample["detail_indices"])
                                    and 0 not in sample["detail_indices"], "Summary/random-detail construction mismatch")
                        samples.append(sample)
                    stream = json.dumps(dict(sample_ids=ids, views=[sample["views"] for sample in samples],
                                             tokens=[torch.stack([sample[key] for sample in samples]).tolist()
                                                     for key in ("tokens_f", "tokens_o", "tokens_e")]),
                                        ensure_ascii=False, separators=(",", ":")).encode()
                    require(hashlib.sha256(stream).hexdigest() == health["stream_sha256"],
                            "Actual F/S/D token/text stream differs from exact historical replay")
                    print(json.dumps(dict(event="sample_stream_verified", rank=rank, step=row["step"])), flush=True)
        finally:
            records.close()
    return dict(passed=True, ranks=4, updates=5, samples=5120,
                exact_distributed_sampler_prefix=True, exact_text_and_token_stream=True,
                no_images_read=True, full_dataset_reaudit=False)


def verify_checkpoint(checkpoint, rows):
    import torch
    torch.set_num_threads(4)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(payload["completed_steps"] == 5 and payload["scheduler_horizon"] == 4868
            and payload["stop_updates"] == 5, "Checkpoint step/horizon mismatch")
    steps = sorted({int(state["step"]) for state in payload["optimizer"]["state"].values()})
    require(steps == [5], "AdamW state does not prove five optimizer updates")
    require(len(payload["rng_per_rank"]) == 4 and payload["next_epoch"] == 0 and
            payload["next_batch"] == 5, "Checkpoint RNG/position mismatch")
    for group in payload["optimizer"]["param_groups"]:
        require(group["lr"] == rows[-1]["actual_lrs"][group["name"]] and
                group["betas"] == (.9, .999) and group["eps"] == 1e-8,
                "Serialized optimizer LR/betas/epsilon mismatch")
    for state in payload["optimizer"]["state"].values():
        require(all(bool(torch.isfinite(value).all()) for value in state.values() if torch.is_tensor(value)),
                "Nonfinite serialized optimizer state")
    initial = torch.load(RUNTIME / "shared/step000000.pt", map_location="cpu", weights_only=False)
    require(initial["completed_steps"] == 0 and not initial["optimizer"]["state"] and
            initial["provenance"]["source"] == "OpenAI CLIP + original random MaskNetwork", "Invalid initializer provenance")
    require(payload["model"].keys() == initial["model"].keys(), "Model state_dict keys changed")
    changed = 0
    for name, tensor in payload["model"].items():
        require(tensor.shape == initial["model"][name].shape and bool(torch.isfinite(tensor).all()),
                "Invalid checkpoint model tensor: " + name)
        changed += int(not torch.equal(tensor, initial["model"][name]))
    require(changed > 0, "Optimizer did not change any native model tensors")
    result = dict(passed=True, optimizer_steps=steps, optimizer_parameter_states=len(payload["optimizer"]["state"]),
                  changed_model_tensors=changed, state_dict_keys=len(payload["model"]),
                  rng_ranks=4, horizon=4868, next_batch=5,
                  scaler="Not used by the historical BF16-autocast/FP32-master trainer; no GradScaler state expected",
                  scheduler="Historical manual optimizer_learning_rates(completed,4868); serialized horizon and LR verified")
    del payload, initial
    gc.collect()
    return result


def native_export(checkpoint, output):
    bare = output / "bare-step000005.pt"
    proof = EVIDENCE / "s02-export-audit.json"
    commands = [[str(ROOT / ".venv/bin/python"), "-m", "tools.nest_clip", "export",
                 "--checkpoint", str(checkpoint), "--output", str(bare), "--expect-updates", "5"],
                [str(ROOT / ".venv/bin/python"), "-m", "tools.nest_clip", "verify-export",
                 "--checkpoint", str(checkpoint), "--bare", str(bare), "--output", str(proof),
                 "--index-dir", str(RUNTIME / "data_index"),
                 "--image-root", str(ROOT / "local_assets/training/ShareGPT4V")]]
    with (EVIDENCE / "s02-export.log").open("w") as handle:
        for command in commands:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = read_json(proof)
    require(result["passed"] and result["strict_load"] and
            result["image_max_abs"] == result["text_max_abs"] == 0., "Native bare encoder equality failed")
    import torch
    from model import longclip
    from train.nested_semantic_data import NestedDataset
    student, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=argparse.Namespace())
    student.load_state_dict(torch.load(bare, map_location="cpu", weights_only=True), strict=True)
    student.eval()
    dataset = NestedDataset(RUNTIME / "data_index", ROOT / "local_assets/training/ShareGPT4V")
    samples = [dataset[index] for index in (0, 1)]
    with torch.no_grad():
        images = student.encode_image(torch.stack([sample["image"] for sample in samples]))
        texts = student.encode_text(torch.stack([sample["tokens_f"] for sample in samples]))
        similarity = torch.nn.functional.normalize(images, dim=-1) @ torch.nn.functional.normalize(texts, dim=-1).T
    require(similarity.shape == (2, 2) and bool(torch.isfinite(similarity).all()), "Bare native similarity forward failed")
    result.update(native_similarity_forward_passed=True, similarity=similarity.tolist(),
                  context_length=student.positional_embedding.shape[0], inference="bare native CLIP image/text; no fusion/mask inference")
    write_json(proof, result)
    del student, dataset, samples
    gc.collect()
    return result


def publish_readiness(result):
    require(not result["passed"] or result["steps_completed"] == 5,
            "Cannot publish a passing gate without exactly five updates")
    write_json(EVIDENCE / "smoke-audit.json", dict(
        passed=result["passed"], executed=result["executed"],
        status=result["status"], blockers=result["blockers"],
        protocol="Summary+RandomDetail S02, not RandomK",
        configuration="recovery/configs/summary02.json", view_weights=WEIGHTS,
        scheduler_horizon=4868, max_updates=5,
        completed_updates=result["steps_completed"],
        source_evidence="s02-smoke-audit.json",
        quarantined_output_dir=result.get("quarantined_output_dir"),
        formal_training_started=False, formal_training_authorized=False))
    export = dict(result.get("export", {"passed": False}),
                  source_evidence="s02-export-audit.json", protocol="S02 step5 native bare student")
    write_json(EVIDENCE / "export-audit.json", export)
    policy_path = EVIDENCE / "recovery-operation-policy.json"
    policy = read_json(policy_path)
    policy.update(smoke_authorized=False, conditional_smoke_authorized=False,
                  smoke_gate_status=result["status"],
                  smoke_completed=result["steps_completed"] == 5,
                  formal_training_authorized=False,
                  reason="Authorized five-update S02 smoke finished; no further updates without a new instruction")
    write_json(policy_path, policy)


def main():
    command = read_json(EVIDENCE / "s02-smoke-command.json")
    require("returncode" in command, "Training process still running; not finalizing or quarantining")
    output = Path(command.get("quarantined_output_dir", command["output_dir"]))
    rows = read_rows(output / "steps.jsonl") if (output / "steps.jsonl").is_file() else []
    result = dict(passed=False, executed=True, status="NOT_READY_TO_START_S02_FULL", blockers=[],
                  checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  step0_sha256=command["step0_sha256"], formal_training_started=False,
                  formal_training_authorized=False, steps_completed=len(rows))
    try:
        require(command["returncode"] == 0, "torchrun failed; see s02-smoke-training.log")
        config = read_json(output / "config.json")
        check_config(config)
        frozen = read_json(ROOT / "recovery/configs/summary02.json")
        require(all(config.get(key) == value for key, value in frozen.items()),
                "Runtime differs from frozen configuration")
        require(all(config["code_sha256"].get(filename) == digest
                    for filename, digest in command["code_sha256"].items()
                    if filename != "tools/nest_clip.py"), "Runtime native code provenance mismatch")
        acceptance = read_json(output / "acceptance.json")
        result["native_acceptance"] = acceptance
        result["metrics"] = check_steps(rows, acceptance)
        result["sample_stream"] = replay_sample_stream(rows)
        checkpoint = output / "step000005.pt"
        result["checkpoint"] = verify_checkpoint(checkpoint, rows)
        result["export"] = native_export(checkpoint, output)
        cycles = read_rows(output / "cycle_timing.jsonl")
        require([item["step"] for item in cycles] == [1, 2, 3, 4, 5], "Missing full-cycle timings")
        update_seconds = [max(health["seconds"] for health in row["rank_health"]) for row in rows]
        cycle_seconds = [item["four_rank_max_seconds"] for item in cycles]
        peak = max(rank["peak_allocated_gib"] for rank in acceptance["ranks"])
        result["resources"] = dict(update_seconds=update_seconds, full_cycle_seconds=cycle_seconds,
                                   median_update_seconds=statistics.median(update_seconds),
                                   median_post_first_cycle_seconds=statistics.median(cycle_seconds[1:]),
                                   max_peak_allocated_gib=peak,
                                   first_cycle_startup_and_loading_seconds=cycle_seconds[0] - update_seconds[0],
                                   reference_seconds=2.1, reference_peak_gib=27.8,
                                   steady_state_not_established_by_five_warmup_updates=True)
        if max(update_seconds[1:] + cycle_seconds[1:]) > 4:
            result["blockers"].append("Post-first update/cycle exceeds 4 seconds; resource investigation required")
        if peak > 40:
            result["blockers"].append("Peak allocated GPU memory exceeds 40 GiB; resource investigation required")
        result["passed"] = not result["blockers"]
        if result["passed"]:
            result["status"] = "READY_TO_START_S02_FULL"
    except Exception as error:
        result["blockers"].append(type(error).__name__ + ": " + str(error))
    finally:
        quarantine = RUNTIME / "quarantine" / output.name
        if output != quarantine and output.exists():
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            require(not quarantine.exists(), "Quarantine destination already exists")
            output.rename(quarantine)
        if quarantine.exists():
            (quarantine / "SMOKE_ONLY_DO_NOT_RESUME.md").write_text(
                "# Smoke artifacts only\n\nNever resume formal training from this directory.\n"
                "Any authorized S02 full run must start from shared/step000000.pt.\n")
            command["quarantined_output_dir"] = str(quarantine)
            result["quarantined_output_dir"] = str(quarantine)
            write_json(EVIDENCE / "s02-smoke-command.json", command)
        write_json(EVIDENCE / "s02-smoke-audit.json", result)
        publish_readiness(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
