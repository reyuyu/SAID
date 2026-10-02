# R-SentenceDrop: Ordered Semantic Subsets at500

Status: `initializing`. Stage: `None`.

ViT-B/16,seed0,4x256,context248,encoder BF16/other paths FP32,encoder checkpoint ON,pair OFF. Stop500 with horizon4868. Same best coefficients and common step0. Only R construction differs: sample q uniformly from1..m inclusive, sample q suffix sentences without replacement, sort their original indices, join with the original separator and tokenize as compact text. No shuffle, in-place PAD, extra attention mask, view, model module or loss.

Sentence subset RNG is independent: SHA256(sampling_seed:epoch:sample_id:r_sentence_drop_v1). Original RandomK, F/P, image preprocessing and sampler are unchanged.

Reused matched baseline checkpoint SHA256: `f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8`.
Reused matched bare student SHA256: `90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e`.

Baseline already has all five frozen evaluations; it is not retrained. RMask is only a historical negative reference, not the initialization or sampling implementation.

## Main Scores

| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|
| Matched old-R@500 | 4868 | 69.900394 | 73.603324 | 82.340003 |
| R-SentenceDrop@500 | 4868 | not completed | not completed | not completed |

Historical RMask@500 Score5_R1: 69.475768% (negative reference).

## Reused Matched Baseline Complete Recall

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

## Evidence and Reproduction

Correctness and readable real-sample proofs are in evidence/. Statistics come from every valid sample over all500 updates/ranks. RDrop trace hashes match the reused baseline for every sample/F/P/K and original fixed-first reference stream. Model/loss sources stay byte-identical to fa19d12. Raw Recall fractions and strict export/acceptance evidence are retained.

Five native student protocols only, including reconstructed Long-DCI7602; no DCI Full, mask/gate inference, reranking or ensemble. No significance claim or rounded-score selection. No baseline retraining, promotion, extra q distributions, extra values or additional seeds.

```bash
cd /root/lk_projects/SAID-balanced-rdrop-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.run --launch
```

Exact runtime commits/commands/exits, native JSON and sampler evidence are preserved; large checkpoints/data/cache stay server-local. Stop after this single500 evaluation.
