# Balanced RMask: Single-Variable500 Experiment

Status: `running`. Stage: `smoke5`.

ViT-B/16,224,context248,seed0,full4x256. Only the R tensor construction differs: compact suffix versus F clone with exact BPE prefix-plus-separator positions replaced by PAD0. SOT, suffix IDs/positions, F EOT and trailing PAD remain unchanged. No new attention mask, view, architecture, loss or optimizer coefficient.

Fixed coefficients: fusion_lr2e-4, visual_mask_lr_scale1, weights[1,1,1], sparsity_scale1, inclusion_max1; four-epoch horizon4868 but stop at500. Each role has independent5-step smoke and a fresh formal start from the same common step0.

The exact prior four-epoch run has no step500 checkpoint; one matched old-R baseline is run as explicitly authorized. Historical69.990027% at horizon3651 is not the matched baseline.

## Main Comparison

| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|
| Balanced matched old-R@500 | 4868 | not completed | not completed | not completed |
| Balanced RMask@500 | 4868 | not completed | not completed | not completed |

## Evidence and Reproduction

evidence/real-samples.json preserves readable sentence/K/boundary/EOT and original/late suffix position evidence. Unit tests verify split extremes, BPE punctuation,248-token limits, F-only fallback and internal-PAD encoding. RMask500 trace hashes compare sample/F/P/K and original fixed-first reference streams to the matched baseline on every step/rank.

Each arm preserves config, command/exit records, acceptance, strict native export and raw five-dataset evaluator JSON. Only native normalized student embeddings are used; no mask or reranking, no DCI Full. Both roles stop at500, with no automatic promotion or new values.

```bash
cd /root/lk_projects/SAID-balanced-rmask-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.run --launch
```

Large weights, data and full token logs remain server-local. Code/configs/reports and compact evidence are committed to the independent RMask branch.
