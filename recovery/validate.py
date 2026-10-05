"""Reuse pinned SAID constructors/tests; the only trainer allowed here stops at five updates."""

import argparse
import hashlib
import json
import math
import statistics
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from audit import ANNOTATION, ANNOTATION_SHA, ASSETS, CLIP_SHA, EVIDENCE, PINS, RECOVERY, ROOT, RUNTIME, STEP0_SHA, digest, dump, load


PYTHON = ROOT / ".venv/bin/python"
FORMAL = ROOT / "worktrees/randomk"


def execute(name, arguments, cwd=ROOT, cpu=False):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    with (EVIDENCE / (name + ".log")).open("w") as output:
        result = subprocess.run([str(argument) for argument in arguments], cwd=cwd, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    dump(EVIDENCE / (name + "-command.json"), dict(argv=[str(argument) for argument in arguments],
         cwd=str(cwd), cpu_only=cpu, exit_code=result.returncode, elapsed_seconds=time.monotonic() - started))
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {EVIDENCE / (name + '.log')}")


def prepare():
    assert digest(ANNOTATION) == ANNOTATION_SHA
    assert digest(ASSETS / "weights/openai/ViT-B-16.pt") == CLIP_SHA
    initial = RUNTIME / "shared/step000000.pt"
    index = RUNTIME / "data_index"
    if initial.is_file() and index.is_dir():
        print("Existing initializer/index retained; audit them instead of overwriting.")
        return
    if index.exists() or initial.exists():
        raise RuntimeError("Partial initializer/index exists; inspect it before recovery, no automatic deletion")
    execute("prepare-step0-index", [PYTHON, "-m", "tools.nest_clip", "prepare", "--annotation", ANNOTATION,
                                   "--index-dir", index, "--init-state", initial], cpu=True)


def tensor_identity(state):
    result = {}
    for key, tensor in state.items():
        content = tensor.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        result[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(content).hexdigest())
    return result


def step0():
    sys.path.insert(0, str(ROOT))
    import torch
    from model import longclip
    from model.nested_semantic_mask import NestedSemanticMask
    from train.train_nested_semantic_mask import build_optimizer, seed_all

    torch.set_num_threads(4)
    initial = RUNTIME / "shared/step000000.pt"
    payload = torch.load(initial, map_location="cpu", weights_only=False)
    assert payload["completed_steps"] == 0 and not payload["optimizer"]["state"]
    assert payload["provenance"]["seed"] == 0
    assert payload["provenance"]["source"] == "OpenAI CLIP + original random MaskNetwork"
    assert payload["provenance"]["original_clip_sha256"] == CLIP_SHA
    assert digest(ASSETS / "weights/openai/ViT-B-16.pt") == CLIP_SHA
    seed_all(0)
    reference, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=argparse.Namespace())
    module = NestedSemanticMask(reference, checkpoint_encoders=False)
    optimizer = build_optimizer(module)
    assert not optimizer.state
    torch.testing.assert_close(torch.get_rng_state(), payload["rng_cpu"], atol=0, rtol=0)
    expected = tensor_identity(reference.state_dict())
    actual = tensor_identity(payload["model"])
    assert actual == expected
    restored, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=argparse.Namespace())
    restored.load_state_dict(payload["model"], strict=True)
    reference.eval()
    restored.eval()
    images = torch.zeros(1, 3, 224, 224)
    tokens = longclip.tokenize(["Recovery deterministic constructor audit."], context_length=248, truncate=False)
    outputs = {}
    with torch.no_grad():
        for name, original, recovered in [
            ("image", reference.encode_image(images), restored.encode_image(images)),
            ("text", reference.encode_text(tokens), restored.encode_text(tokens)),
        ]:
            torch.testing.assert_close(original, recovered, atol=0, rtol=0)
            outputs[name] = dict(max_abs=float((original - recovered).abs().max()),
                                 sha256=hashlib.sha256(recovered.contiguous().numpy().tobytes()).hexdigest())
    observed = digest(initial)
    result = dict(passed=True, checkpoint=str(initial), expected_sha256=STEP0_SHA, observed_sha256=observed,
                  historical_file_sha_matches=observed == STEP0_SHA, optimizer_state_empty=True, seed=0,
                  provenance=payload["provenance"], context_length=248, state_dict=actual,
                  exact_keys_shapes_content_equal_original_seeded_constructor=True, rng_equal=True, outputs=outputs,
                  tensor_reference="Pinned original repository constructor + official CLIP + seed0, independently rebuilt; not a recovered historical tensor manifest.",
                  caveat="If file SHA differs, historical byte identity is NOT established. This audit establishes semantic deterministic reconstruction, without changing the model to force a SHA.")
    dump(EVIDENCE / "step0-audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "state_dict"}, indent=2))


def sampling():
    sys.path.insert(0, str(ROOT))
    import torch
    from train.nested_semantic_data import sample_detail_indices, sample_split_k, sampled_text_views, text_views

    torch.set_num_threads(4)
    random_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    for count in range(2, 20):
        for sample_id in range(1000, 2000):
            split = sample_split_k(count, 0, 0, sample_id)
            assert 1 <= split < count and split == sample_split_k(count, 0, 0, sample_id)
            detail = sample_detail_indices(count, 0, 0, sample_id)
            assert detail == sorted(set(detail)) and all(1 <= index < count for index in detail)
            assert detail == sample_detail_indices(count, 0, 0, sample_id)
            assert (2 <= len(detail) <= count - 2) if count >= 4 else len(detail) == 1
    assert random.getstate() == random_state and torch.equal(torch.get_rng_state(), torch_state)
    count = 0
    with (RUNTIME / "data_index/records.jsonl").open() as stream:
        for index, line in enumerate(stream):
            caption = json.loads(line)["caption"]
            original = text_views(caption)
            for mode in ("random_k", "summary_random_detail"):
                views = sampled_text_views(caption, mode, 0, 0, index + 1000)
                assert torch.equal(original["tokens_f"], views["tokens_f"])
                assert all(views[key].shape == (248,) for key in ("tokens_f", "tokens_o", "tokens_e"))
            count += 1
            if count == 1000:
                break
    assert count == 1000
    report = dict(passed=True, real_captions_checked=count, full_view_identical=True,
                  global_rng_untouched=True, randomk_bounds_checked=True, detail_excludes_summary=True,
                  detail_is_ordered_unique_subset=True, sampling_seed=0, first_sample_id=1000,
                  weights=load(RECOVERY / "configs/summary02.json")["view_weights"])
    assert report["weights"] == [1.4, 0.2, 1.4]
    dump(EVIDENCE / "sampling-audit.json", report)
    print(json.dumps(report, indent=2))


def cpu():
    selected = ["tests/test_nested_semantic_mask.py", "tests/test_nested_fusion.py", "tests/test_nested_randomk.py",
                "tests/test_nested_resume.py", "tests/test_balanced_hparams.py", "tests/test_summary_random_detail.py",
                "tests/test_armb_summary_dose.py", "tests/test_extended_retrieval.py"]
    results = {}
    for name, directory in [("research", ROOT), ("canonical", FORMAL)]:
        files = [path for path in selected if (directory / path).is_file()]
        try:
            execute("cpu-tests-" + name, [PYTHON, "-m", "pytest", "-q", *files], cwd=directory, cpu=True)
            execute("cpu-tests-" + name + "-legacy-retrieval", [PYTHON, "-m", "pytest", "-q", "tests/test_retrieval_eval.py"], cwd=directory, cpu=True)
            results[name] = dict(passed=True, selected_tests=files, legacy_retrieval="Separate subprocess: its original test changes sys.path and otherwise shadows train package in spawned workers.")
        except RuntimeError as error:
            results[name] = dict(passed=False, error=str(error), selected_tests=files)
    dump(EVIDENCE / "cpu-tests.json", dict(passed=all(result["passed"] for result in results.values()), checkouts=results))
    print(json.dumps(results, indent=2))


def construction():
    results = {}
    for name in ("randomk", "summary02"):
        execute("construction-" + name, [PYTHON, RECOVERY / "construct_configs.py", name], cpu=True)
        results[name] = load(EVIDENCE / ("construction-" + name + ".json"))
    dump(EVIDENCE / "construction-audit.json", dict(passed=all(result["passed"] for result in results.values()), configurations=results))


def smoke():
    policy = load(EVIDENCE / "recovery-operation-policy.json", {})
    if policy.get("smoke_authorized") is False:
        dump(EVIDENCE / "smoke-audit.json", dict(passed=False, executed=False,
            blocked_by=[policy.get("reason", "Smoke is not authorized by the current user instruction")],
            formal_training_started=False))
        raise RuntimeError("Smoke not started; current user instruction forbids smoke")
    prerequisites = ["git-audit.json", "environment-audit.json", "config-audit.json", "training-audit.json",
                     "evaluator-audit.json", "step0-audit.json", "cpu-tests.json", "sampling-audit.json",
                     "construction-audit.json", "training-image-completeness-audit.json", "sa1b-mirror-audit.json"]
    missing = [name for name in prerequisites if not load(EVIDENCE / name, {}).get("passed", False)]
    if missing:
        dump(EVIDENCE / "smoke-audit.json", dict(passed=False, executed=False, blocked_by=missing, formal_training_started=False))
        raise RuntimeError("Smoke not started; unverified prerequisites: " + ", ".join(missing))
    if digest(RUNTIME / "shared/step000000.pt") != STEP0_SHA:
        dump(EVIDENCE / "smoke-audit.json", dict(passed=False, executed=False,
            blocked_by=["Current common step0 SHA256 differs from the historical initializer"], formal_training_started=False))
        raise RuntimeError("Smoke not started; common step0 identity changed")
    head = subprocess.check_output(["git", "-C", str(FORMAL), "rev-parse", "HEAD"]).decode().strip()
    assert head == PINS["canonical"][1]
    subprocess.run(["git", "-C", str(FORMAL), "diff", "--exit-code"], check=True)
    gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                           "--format=csv,noheader"]).decode().strip()
    if gpu_processes:
        dump(EVIDENCE / "smoke-audit.json", dict(passed=False, executed=False,
            blocked_by=["GPU compute processes already active"], gpu_processes=gpu_processes,
            formal_training_started=False))
        raise RuntimeError("Smoke not started; GPU resources are already in use")
    output = RUNTIME / "recovery-smoke-randomk-5"
    if output.exists():
        raise RuntimeError("Smoke output exists; preserve it instead of overwriting")
    execute("four-gpu-smoke-5", [ROOT / ".venv/bin/torchrun", "--standalone", "--nnodes=1",
        "--nproc-per-node=4", "--max-restarts=0", "-m", "train.train_nested_semantic_mask",
        "--config", RECOVERY / "configs/randomk.json", "--init-state", RUNTIME / "shared/step000000.pt",
        "--index-dir", RUNTIME / "data_index", "--image-root", ASSETS / "training/ShareGPT4V",
        "--output-dir", output, "--run-type", "smoke", "--max-updates", "5"], cwd=FORMAL)
    acceptance = load(output / "acceptance.json")
    configuration = load(output / "config.json")
    assert acceptance["passed"] and configuration["horizon"] == 4868
    assert configuration["batches_per_epoch"] == 1217
    assert all(rank["completed_updates"] == 5 and rank["max_parameter_difference_from_rank0"] == 0 for rank in acceptance["ranks"])
    steps = [json.loads(line) for line in (output / "steps.jsonl").open()]
    assert [row["step"] for row in steps] == [1, 2, 3, 4, 5]
    assert all(math.isfinite(row["loss"]) and not row["duplicate_image_ids"] for row in steps)
    assert all(len(row["rank_health"]) == 4 and
        {health["rank"] for health in row["rank_health"]} == {0, 1, 2, 3} and
        all(health["batch"] == 256 and health["gradients_finite"] and len(health["stream_sha256"]) == 64
            for health in row["rank_health"]) for row in steps)
    peak = max(rank["peak_allocated_gib"] for rank in acceptance["ranks"])
    median_seconds = statistics.median(max(health["seconds"] for health in row["rank_health"]) for row in steps[1:])
    anomalies = []
    if not 27.8*.75 <= peak <= 27.8*1.25:
        anomalies.append("Peak allocation differs by more than25% from27.8GiB reference")
    if median_seconds > 2.1*1.5:
        anomalies.append("Median compute time of updates2..5 exceeds150% of2.1s reference; startup/NFS/system behavior needs investigation")
    smoke_record = dict(passed=False, executed=True, updates=5, horizon=4868,
        ddp_parameter_agreement=True, acceptance=acceptance, reference_seconds_per_update=2.1,
        reference_peak_allocated_gib=27.8, sample_stream_health_passed=True, loss_and_gradients_finite=True,
        resource_assessment=dict(peak_allocated_gib=peak, median_updates2to5_seconds=median_seconds,
            anomalies=anomalies, passed=not anomalies),
        timing_caveat="Five updates are startup/warmup, not a 495-cycle steady-state estimate.")
    dump(EVIDENCE / "smoke-audit.json", smoke_record)
    checkpoint = output / "step000005.pt"
    bare = output / "student_step5.pt"
    execute("native-export", [PYTHON, "-m", "tools.nest_clip", "export", "--checkpoint", checkpoint,
                              "--expect-updates", "5", "--output", bare], cwd=FORMAL, cpu=True)
    execute("native-verify-export", [PYTHON, "-m", "tools.nest_clip", "verify-export", "--checkpoint", checkpoint,
        "--bare", bare, "--output", EVIDENCE / "export-audit.json", "--index-dir", RUNTIME / "data_index",
        "--image-root", ASSETS / "training/ShareGPT4V"], cwd=FORMAL, cpu=True)
    assert load(EVIDENCE / "export-audit.json")["passed"]
    smoke_record.update(passed=not anomalies, native_bare_export_passed=True, common_step0_sha256=STEP0_SHA)
    dump(EVIDENCE / "smoke-audit.json", smoke_record)
    if anomalies:
        raise RuntimeError("Five-update smoke stopped; resource anomalies require investigation before full training")


def subset():
    if not load(EVIDENCE / "evaluator-audit.json", {}).get("passed", False):
        raise RuntimeError("Five frozen evaluation datasets must pass their audit before subset inference")
    sys.path.insert(0, str(ROOT))
    import torch
    from PIL import Image
    from torchvision.datasets import CocoCaptions
    from model import longclip
    from eval.retrieval.coco_retrieval import retrieval_metrics
    from tools.urban1k_retrieval import image_caption_pairs, read_captions, _recall
    from experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real import evaluate

    torch.set_num_threads(4)
    bare = RUNTIME / "shared/student_step0.pt"
    if not bare.exists():
        execute("native-step0-export", [PYTHON, "-m", "tools.nest_clip", "export", "--checkpoint",
            RUNTIME / "shared/step000000.pt", "--expect-updates", "0", "--output", bare], cpu=True)
    model, preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", args=argparse.Namespace())
    model.load_state_dict(torch.load(bare, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    results = {}
    coco = ASSETS / "evaluation/coco"
    dataset = CocoCaptions(str(coco / "val2017"), str(coco / "annotations/captions_val2017.json"), transform=preprocess)
    samples = [dataset[index] for index in range(2)]
    with torch.no_grad():
        images = model.encode_image(torch.stack([sample[0] for sample in samples])).float()
        texts = model.encode_text(longclip.tokenize([caption for sample in samples for caption in sample[1][:5]], context_length=248, truncate=True)).float()
        results["coco"] = dict(images=2, captions=10, metrics=retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=512))
        pairs = image_caption_pairs(str(ASSETS / "evaluation/Urban1k/Urban1k"))[:2]
        images = model.encode_image(torch.stack([preprocess(Image.open(path).convert("RGB")) for path, _ in pairs])).float()
        texts = model.encode_text(longclip.tokenize(read_captions(pairs), context_length=248, truncate=True)).float()
        matrix = torch.nn.functional.normalize(images, dim=-1) @ torch.nn.functional.normalize(texts, dim=-1).t()
        results["urban1k"] = dict(images=2, captions=2, image2text=_recall(matrix), text2image=_recall(matrix.t()))
    from audit import PROTOCOLS

    for name, (filename, folder, _, _, _) in PROTOCOLS.items():
        rows = [json.loads(line) for line in (ASSETS / "retrieval_benchmarks/manifests" / filename).open()]
        selected_ids = list(dict.fromkeys(row["image_id"] for row in rows))[:2]
        selected = [row for row in rows if row["image_id"] in selected_ids]
        manifest = EVIDENCE / ("subset-" + name + ".jsonl")
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        images, captions, metrics = evaluate(model, preprocess, manifest, ASSETS / "retrieval_benchmarks" / folder / "images", "cpu", 2)
        results[name] = dict(images=images, captions=captions, metrics=metrics)
    dump(EVIDENCE / "native-evaluator-subset.json", dict(passed=True, native_only=True, checkpoint=str(bare),
        checkpoint_step=0, protocols=results, scope="SUBSET_SANITY_ONLY; these are not official five-set retrieval scores"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "step0", "sampling", "cpu", "construction", "smoke", "subset"])
    args = parser.parse_args()
    globals()[args.command]()


if __name__ == "__main__":
    main()
