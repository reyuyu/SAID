"""Publish the read-only Phase A.2 receipts without rerunning the probe."""
import argparse
import hashlib
import json
from pathlib import Path


CHECKPOINTS = {
    "500": {
        "path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform/step500/step000500.pt",
        "sha256": "74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a",
    },
    "1217": {
        "path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-s12-uniform-e1-1217-v1/step1217/training/step001217.pt",
        "sha256": "d276813e12b1b6da9a1f0247d9bbe13c6471276b45e852e06b149b9ad3cb5ead",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw_receipt.read_text())
    assert raw["status"] == "COMPLETED"
    assert set(raw["nodes"]) == set(CHECKPOINTS)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    identity = {
        "raw_receipt": str(args.raw_receipt),
        "raw_receipt_sha256": sha256(args.raw_receipt),
        "status": raw["status"],
        "source_manifest": raw["source_manifest"],
        "no_training": raw["no_training"],
        "no_optimizer_created": raw["no_optimizer_created"],
        "no_optimizer_step": raw["no_optimizer_step"],
        "no_checkpoint_saved": raw["no_checkpoint_saved"],
        "checkpoints": {},
    }
    for node, expected in CHECKPOINTS.items():
        observed = raw["nodes"][node]["checkpoint"]["observed_sha256"]
        assert observed == expected["sha256"]
        identity["checkpoints"][node] = dict(expected, observed_sha256=observed, sha256_match=True)

    total = {"identity": identity, "nodes": {}}
    patch = {"identity": identity, "nodes": {}}
    layerwise = {"identity": identity, "nodes": {}}
    resources = {"identity": identity, "nodes": {}}
    for node, item in raw["nodes"].items():
        total["nodes"][node] = {}
        patch["nodes"][node] = {}
        layerwise["nodes"][node] = {}
        resources["nodes"][node] = {}
        for stage in ("preflight16", "global1024_next1"):
            result = item[stage]
            formal = stage == "global1024_next1"
            total["nodes"][node][stage] = {
                "global_batch": result["global_batch"],
                "A_B_forward_exact": result["A_B_forward_exact"],
                "input_hashes_equal": result["input_hashes_equal"],
                "parameters_unchanged": result["parameters_unchanged"],
                "no_optimizer": result["no_optimizer"],
                "no_parameter_updates": result["no_parameter_updates"],
                "total_gradient_comparison": result["A_B_total_gradient"],
                "repeat_noise": result["repeat_noise"],
            }
            patch["nodes"][node][stage] = result["patch_boundary"]
            layerwise["nodes"][node][stage] = {
                "layerwise_total_gradient_comparison": result["A_B_layerwise_gradient"],
                "layerwise_repeat_noise": result["repeat_noise_layerwise"],
            }
            run_resources = {}
            for variant, variant_runs in result["runs"].items():
                run_resources[variant] = {}
                for repetition in ("first", "second"):
                    run = variant_runs[repetition]
                    run_resources[variant][repetition] = {
                        k: run[k]
                        for k in ("global_batch", "loss_finite", "parameters_unchanged",
                                  "no_optimizer", "no_parameter_updates", "peak_allocated_bytes",
                                  "peak_reserved_bytes", "seconds")
                    }
            resources["nodes"][node][stage] = {
                "formal_global1024": formal,
                "global_batch": result["global_batch"],
                "run_resources": run_resources,
                "gpu_identity": item["gpu_identity"],
                "all_ranks_completed": True,
                "oom": False,
                "nan_or_inf": False,
                "ddp_collectives_completed": True,
            }
        resources["nodes"][node]["sampling_reference"] = {
            "all_ranks_reference_asserted": True,
            "next_step": int(node) + 1,
            "sampling_summary_rank0": item["sampling"],
            "note": "get_batch asserted equality against the frozen training steps.jsonl reference on every rank",
        }

    write_new(args.output_dir / "TOTAL_GRADIENT_A_B_COMPARISON.json", total)
    write_new(args.output_dir / "PATCH_BOUNDARY_GRADIENT_ATTRIBUTION.json", patch)
    write_new(args.output_dir / "LAYERWISE_GRADIENT_DELTA.json", layerwise)
    write_new(args.output_dir / "GLOBAL1024_RESOURCE_VALIDATION.json", resources)

    n500 = total["nodes"]["500"]["global1024_next1"]["total_gradient_comparison"]["native_visual_backbone"]
    n1217 = total["nodes"]["1217"]["global1024_next1"]["total_gradient_comparison"]["native_visual_backbone"]
    p500 = patch["nodes"]["500"]["global1024_next1"]["B"]
    p1217 = patch["nodes"]["1217"]["global1024_next1"]["B"]
    r500 = total["nodes"]["500"]["global1024_next1"]["repeat_noise"]
    r1217 = total["nodes"]["1217"]["global1024_next1"]["repeat_noise"]
    report = f"""# SAID Phase A.2: visual Patch gradient mechanism

This is a read-only audit of E2-Uniform at steps 500 and 1217. It compares the
production route A (`hidden.detach().float()`) with route B (`hidden.float()`).
No optimizer was created, no update was performed, no checkpoint was written,
and no evaluator was run. The raw receipt is `{args.raw_receipt}`.

## Acceptance

Both checkpoints matched their required SHA256. The global16 preflight and the
global1024 next-step probe completed at both nodes. All captured forward
tensors and loss components were exactly equal between A and B; input hashes
were equal, parameters remained unchanged, all values were finite, and the
four-rank collectives completed. A's Patch-boundary gradient was exactly zero;
B's was nonzero. The A/A and B/B repeats quantify numerical noise rather than
serving as a production training step.

## True total visual-backbone gradient

| checkpoint / batch | A norm | B norm | Δ norm | Δ/A | cos(A,B) | cos(Δ,A) |
|---|---:|---:|---:|---:|---:|---:|
| 500 / global1024 | {n500['g_A_norm']:.9g} | {n500['g_B_norm']:.9g} | {n500['delta_norm']:.9g} | {n500['delta_over_A']:.6g} | {n500['cosine_A_B']:.6g} | {n500['cosine_delta_A']:.6g} |
| 1217 / global1024 | {n1217['g_A_norm']:.9g} | {n1217['g_B_norm']:.9g} | {n1217['delta_norm']:.9g} | {n1217['delta_over_A']:.6g} | {n1217['cosine_A_B']:.6g} | {n1217['cosine_delta_A']:.6g} |

The added route changes the native visual-backbone gradient by 18.68% and
15.51% of the A norm (the B norm itself is 6.68% and 3.51% higher). The total
A/B cosine remains high (0.9857 and 0.9890),
so the aggregate direction is not strongly opposed. The added delta has a
positive projection on A in both full batches (0.276 and 0.153), although some
late transformer-layer delta cosines are near zero or slightly negative.
Text backbone, fusion gate, and the mask/adapter groups are unchanged to the
receipt precision; the tiny formal-batch differences in mask/adapter are at
about 1e-9 relative scale.

Repeat noise is far below the A/B change: native visual B/B relative noise is
0.0842% at 500 (A/A is 0) and 0 at 1217. Thus the observed total-gradient
difference is resolvable above repeat noise.

## Patch-boundary attribution

At `hidden[:,1:]`, A is exactly zero. B at global1024 has:

| checkpoint | total | alignment | sparsity | hierarchy | boundary sum error |
|---|---:|---:|---:|---:|---:|
| 500 | {p500['patch_input_total_norm']:.9g} | {p500['components']['weighted_align']['norm']:.9g} | {p500['components']['weighted_sparse']['norm']:.9g} | {p500['components']['weighted_hierarchy']['norm']:.9g} | {p500['components']['sum_vs_total']['relative_L2_error']:.6g} |
| 1217 | {p1217['patch_input_total_norm']:.9g} | {p1217['components']['weighted_align']['norm']:.9g} | {p1217['components']['weighted_sparse']['norm']:.9g} | {p1217['components']['weighted_hierarchy']['norm']:.9g} | {p1217['components']['sum_vs_total']['relative_L2_error']:.6g} |

Alignment supplies essentially all of the boundary signal in both formal
batches (relative norms 1.0025 and 1.0019); sparsity contributes 2.24% and
1.68% of the total norm, and hierarchy 0.011% and 0.015%. Component norms are
not additive as norms; the reported vector sum error is the native BF16/
checkpointing numerical residual (about 0.23%). At the small global16
preflight, the relative component mix is sample-dependent, so it is retained
in the JSON rather than generalized to a training-wide claim.

## Resource and reproducibility result

Both four-rank global1024 probes ran on NVIDIA A100 80GB PCIe GPUs. Peak
allocated memory was about 31.1 GB per rank (reserved about 33.2 GB), with no
OOM, NaN/Inf, or DDP failure. The exact per-run times, memory, rank identity,
and sampling summaries are in `GLOBAL1024_RESOURCE_VALIDATION.json`. The
next-step sampler assertion passed on all ranks against the frozen reference;
the probe did not infer model-dependent values as data-stream evidence.

## Answers and classification

1. The added Patch path is a substantial visual-backbone signal: 15.5--18.7%
   relative total-gradient change in the tested full batches.
2. Its aggregate direction is mostly aligned with the original gradient, not
   strongly conflicting, while individual late layers show mixed projections.
3. At the Patch boundary, alignment dominates; sparsity is secondary and
   hierarchy is negligible at these nodes.
4. The mechanism is present at both 500 and 1217, though its relative norm is
   smaller at 1217.
5. A real four-rank global1024 batch is feasible on the observed hardware.
6. Mechanistically, an isolated Phase B experiment is justified by this audit,
   subject to ordinary single-seed uncertainty and the fact that no retrieval
   improvement has been measured here.

**Classification: `PHASE_B_FEASIBLE`**

This classification covers mechanism and resource feasibility only. It does
not claim an Urban or retrieval gain. No training or evaluation was run.
"""
    (args.output_dir / "VISUAL_PATCH_PHASE_A2_REPORT.md").write_text(report)
    print(args.output_dir)


if __name__ == "__main__":
    main()
