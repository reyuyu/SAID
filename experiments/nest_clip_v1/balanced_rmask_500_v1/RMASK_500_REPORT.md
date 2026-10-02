# Balanced RMask: Single-Variable500 Experiment

Status: `running`. Stage: `five-native-evaluations`.

ViT-B/16,224,context248,seed0,full4x256. Only the R tensor construction differs: compact suffix versus F clone with exact BPE prefix-plus-separator positions replaced by PAD0. SOT, suffix IDs/positions, F EOT and trailing PAD remain unchanged. No new attention mask, view, architecture, loss or optimizer coefficient.

Fixed coefficients: fusion_lr2e-4, visual_mask_lr_scale1, weights[1,1,1], sparsity_scale1, inclusion_max1; four-epoch horizon4868 but stop at500. Each role has independent5-step smoke and a fresh formal start from the same common step0.

The exact prior four-epoch run has no step500 checkpoint; one matched old-R baseline is run as explicitly authorized. Historical69.990027% at horizon3651 is not the matched baseline.

## Main Comparison

| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|
| Balanced matched old-R@500 | 4868 | 69.900394 | 73.603324 | 82.340003 |
| Balanced RMask@500 | 4868 | not completed | not completed | not completed |

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

## Evidence and Reproduction

evidence/real-samples.json preserves readable sentence/K/boundary/EOT and original/late suffix position evidence. Unit tests verify split extremes, BPE punctuation,248-token limits, F-only fallback and internal-PAD encoding. RMask500 trace hashes compare sample/F/P/K and original fixed-first reference streams to the matched baseline on every step/rank.

Each arm preserves config, command/exit records, acceptance, strict native export and raw five-dataset evaluator JSON. Only native normalized student embeddings are used; no mask or reranking, no DCI Full. Both roles stop at500, with no automatic promotion or new values.

```bash
cd /root/lk_projects/SAID-balanced-rmask-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.run --launch
```

Large weights, data and full token logs remain server-local. Code/configs/reports and compact evidence are committed to the independent RMask branch.
