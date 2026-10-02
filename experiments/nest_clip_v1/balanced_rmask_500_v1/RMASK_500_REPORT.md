# Balanced RMask: Single-Variable500 Experiment

Status: `running`. Stage: `five-native-evaluations`.

ViT-B/16,224,context248,seed0,full4x256. Only the R tensor construction differs: compact suffix versus F clone with exact BPE prefix-plus-separator positions replaced by PAD0. SOT, suffix IDs/positions, F EOT and trailing PAD remain unchanged. No new attention mask, view, architecture, loss or optimizer coefficient.

Fixed coefficients: fusion_lr2e-4, visual_mask_lr_scale1, weights[1,1,1], sparsity_scale1, inclusion_max1; four-epoch horizon4868 but stop at500. Each role has independent5-step smoke and a fresh formal start from the same common step0.

The exact prior four-epoch run has no step500 checkpoint; one matched old-R baseline is run as explicitly authorized. Historical69.990027% at horizon3651 is not the matched baseline.

## Main Comparison

| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|
| Balanced matched old-R@500 | 4868 | 69.900394 | 73.603324 | 82.340003 |
| Balanced RMask@500 | 4868 | 69.475768 | 73.126947 | 81.780003 |

## Matched old-R Complete Recall

Full checkpoint SHA256: `f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8`.
Bare-student SHA256: `90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e`.

| Dataset | Direction | R1 % | R5 % | R10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 59.960000 | 81.820000 | 88.680000 |
| COCO | T2I | 40.924000 | 66.660000 | 76.268000 |
| Urban-1k | I2T | 89.000005 | 98.200005 | 99.400002 |
| Urban-1k | T2I | 87.900007 | 98.200005 | 99.100006 |
| Flickr30k-test1k | I2T | 86.200000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.300000 | 90.520000 | 94.660000 |
| DOCCI | I2T | 76.320000 | 94.540000 | 97.560000 |
| DOCCI | T2I | 76.140000 | 94.480000 | 97.520000 |
| Long-DCI | I2T | 55.393318 | 74.875033 | 81.057616 |
| Long-DCI | T2I | 56.866614 | 76.229939 | 82.044199 |

### Resource Evidence

```json
{
  "mean_seconds": 2.098350592815515,
  "median_seconds": 2.077145576477051,
  "p95_seconds": 2.2119564533233644,
  "max_seconds": 2.3512561321258545,
  "normal_updates": 495,
  "approved_seconds": 3,
  "peak_allocated_gib": 27.759892463684082,
  "peak_reserved_gib": 28.44921875
}
```

## RMask Complete Recall

Full checkpoint SHA256: `f6f21be17a09aedf5697e891a8354259a30f700a32092b6781813c724add6783`.
Bare-student SHA256: `be4989b2fe5c7c85a802625a7ba30e7b6bee788bf71f250903d5cf6aecebc10f`.

| Dataset | Direction | R1 % | R5 % | R10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 58.740000 | 81.520000 | 88.000000 |
| COCO | T2I | 40.536000 | 66.364000 | 76.104000 |
| Urban-1k | I2T | 89.000005 | 98.000002 | 99.100006 |
| Urban-1k | T2I | 86.500007 | 98.000002 | 99.100006 |
| Flickr30k-test1k | I2T | 85.800000 | 97.200000 | 98.600000 |
| Flickr30k-test1k | T2I | 70.920000 | 90.500000 | 94.820000 |
| DOCCI | I2T | 75.800000 | 94.440000 | 97.660000 |
| DOCCI | T2I | 75.820000 | 94.200000 | 97.340000 |
| Long-DCI | I2T | 55.182847 | 74.533018 | 80.689292 |
| Long-DCI | T2I | 56.458827 | 75.690608 | 81.570639 |

### Resource Evidence

```json
{
  "mean_seconds": 2.1083403972664265,
  "median_seconds": 2.0913572311401367,
  "p95_seconds": 2.217528772354126,
  "max_seconds": 2.3211746215820312,
  "normal_updates": 495,
  "approved_seconds": 3,
  "peak_allocated_gib": 27.759892463684082,
  "peak_reserved_gib": 28.447265625
}
```

### Absolute-Position Statistics

```json
{
  "mean_F_eot_position": 170.452984375,
  "mean_prefix_boundary": 83.60286637392174,
  "mean_prefix_end_position": 82.60286637392174,
  "mean_suffix_first_position": 83.60286637392174,
  "mean_old_compact_suffix_first_position": 1.0,
  "suffix_first_bins_count": {
    "0-31": 86077,
    "32-63": 115125,
    "64-127": 211324,
    "128-191": 86906,
    "192-247": 12504
  },
  "suffix_first_bins_fraction": {
    "0-31": 0.1681401581447681,
    "32-63": 0.22488162582822852,
    "64-127": 0.41279378672334044,
    "128-191": 0.16975950118764846,
    "192-247": 0.024424928116014502
  },
  "valid_samples": 511936,
  "total_samples": 512000
}
```

## Ten R1 Deltas

| Dataset | Direction | RMask minus matched old-R (pp) |
|---|---|---:|
| COCO | I2T | -1.220000 |
| COCO | T2I | -0.388000 |
| Urban-1k | I2T | +0.000000 |
| Urban-1k | T2I | -1.400000 |
| Flickr30k-test1k | I2T | -0.400000 |
| Flickr30k-test1k | T2I | +0.620000 |
| DOCCI | I2T | -0.520000 |
| DOCCI | T2I | -0.320000 |
| Long-DCI | I2T | -0.210471 |
| Long-DCI | T2I | -0.407787 |

## Conclusion

RMask@500 is worse. Fixed seed0, measured raw Score5_R1; no rounding-based selection or statistical-significance claim.

## Evidence and Reproduction

evidence/real-samples.json preserves readable sentence/K/boundary/EOT and original/late suffix position evidence. Unit tests verify split extremes, BPE punctuation,248-token limits, F-only fallback and internal-PAD encoding. RMask500 trace hashes compare sample/F/P/K and original fixed-first reference streams to the matched baseline on every step/rank.

Each arm preserves config, command/exit records, acceptance, strict native export and raw five-dataset evaluator JSON. Only native normalized student embeddings are used; no mask or reranking, no DCI Full. Both roles stop at500, with no automatic promotion or new values.

```bash
cd /root/lk_projects/SAID-balanced-rmask-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.run --launch
```

Large weights, data and full token logs remain server-local. Code/configs/reports and compact evidence are committed to the independent RMask branch.
