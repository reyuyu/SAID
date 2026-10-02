# R-SentenceDrop: Ordered Semantic Subsets at500

Status: `completed`. Stage: `single500-final-comparison`.

ViT-B/16,seed0,4x256,context248,encoder BF16/other paths FP32,encoder checkpoint ON,pair OFF. Stop500 with horizon4868. Same best coefficients and common step0. Only R construction differs: sample q uniformly from1..m inclusive, sample q suffix sentences without replacement, sort their original indices, join with the original separator and tokenize as compact text. No shuffle, in-place PAD, extra attention mask, view, model module or loss.

Sentence subset RNG is independent: SHA256(sampling_seed:epoch:sample_id:r_sentence_drop_v1). Original RandomK, F/P, image preprocessing and sampler are unchanged.

Reused matched baseline checkpoint SHA256: `f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8`.
Reused matched bare student SHA256: `90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e`.

Baseline already has all five frozen evaluations; it is not retrained. RMask is only a historical negative reference, not the initialization or sampling implementation.

## Main Scores

| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|
| Matched old-R@500 | 4868 | 69.900394 | 73.603324 | 82.340003 |
| R-SentenceDrop@500 | 4868 | 69.893492 | 73.294487 | 82.005003 |
| Delta pp | - | -0.006902 | -0.308837 | -0.335000 |

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

## R-SentenceDrop Complete Recall

| Dataset | Direction | R1 % | R5 % | R10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 60.240000 | 82.260000 | 88.960000 |
| COCO | T2I | 41.208000 | 66.668000 | 76.468000 |
| Urban-1k | I2T | 88.400006 | 98.100007 | 99.200004 |
| Urban-1k | T2I | 87.100005 | 97.500002 | 98.800004 |
| Flickr30k-test1k | I2T | 86.900000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.820000 | 90.940000 | 95.000000 |
| DOCCI | I2T | 76.040000 | 94.960000 | 97.500000 |
| DOCCI | T2I | 76.480000 | 94.440000 | 97.300000 |
| Long-DCI | I2T | 55.196001 | 74.756643 | 81.254933 |
| Long-DCI | T2I | 56.550908 | 76.243094 | 81.781110 |

## Ten R1 Changes

| Dataset | Direction | RDrop minus old-R (pp) |
|---|---|---:|
| COCO | I2T | +0.280000 |
| COCO | T2I | +0.284000 |
| Urban-1k | I2T | -0.599998 |
| Urban-1k | T2I | -0.800002 |
| Flickr30k-test1k | I2T | +0.700000 |
| Flickr30k-test1k | T2I | +0.520000 |
| DOCCI | I2T | -0.280000 |
| DOCCI | T2I | +0.340000 |
| Long-DCI | I2T | -0.197316 |
| Long-DCI | T2I | -0.315706 |

## Sentence Subset Statistics

```json
{
  "valid_R_samples": 511936,
  "m_distribution": {
    "1": 79167,
    "4": 77536,
    "7": 34479,
    "6": 50008,
    "3": 78887,
    "5": 67551,
    "8": 23233,
    "2": 78902,
    "10": 6096,
    "9": 13486,
    "11": 1967,
    "12": 435,
    "14": 33,
    "13": 96,
    "16": 9,
    "17": 1,
    "15": 12,
    "31": 2,
    "33": 3,
    "18": 2,
    "21": 4,
    "27": 2,
    "26": 2,
    "68": 1,
    "24": 1,
    "69": 1,
    "20": 4,
    "25": 3,
    "29": 2,
    "90": 1,
    "43": 1,
    "53": 1,
    "41": 1,
    "23": 1,
    "42": 1,
    "39": 1,
    "36": 1,
    "30": 1,
    "57": 1,
    "28": 1
  },
  "q_distribution": {
    "1": 196590,
    "2": 116699,
    "5": 31778,
    "3": 77899,
    "4": 51589,
    "9": 2328,
    "7": 10236,
    "6": 18555,
    "8": 5167,
    "10": 795,
    "11": 225,
    "12": 44,
    "13": 13,
    "20": 3,
    "15": 5,
    "21": 3,
    "16": 1,
    "17": 2,
    "35": 1,
    "46": 1,
    "34": 1,
    "26": 1
  },
  "joint_m_q_distribution": {
    "1:1": 79167,
    "4:1": 19351,
    "7:1": 4942,
    "6:2": 8235,
    "6:5": 8305,
    "3:3": 26284,
    "5:4": 13544,
    "8:2": 2977,
    "2:1": 39638,
    "10:9": 582,
    "5:1": 13731,
    "2:2": 39264,
    "5:3": 13579,
    "5:5": 13418,
    "7:3": 4917,
    "8:5": 2838,
    "3:1": 26179,
    "7:7": 4976,
    "4:3": 19540,
    "4:4": 19234,
    "9:6": 1457,
    "3:2": 26424,
    "8:7": 2906,
    "9:1": 1489,
    "9:9": 1520,
    "6:3": 8392,
    "9:5": 1487,
    "6:1": 8361,
    "7:5": 4901,
    "8:4": 2962,
    "6:4": 8355,
    "4:2": 19411,
    "5:2": 13279,
    "7:6": 4937,
    "6:6": 8360,
    "9:3": 1444,
    "8:3": 2899,
    "7:2": 4808,
    "7:4": 4998,
    "9:2": 1447,
    "9:4": 1628,
    "10:1": 611,
    "10:3": 628,
    "8:8": 2820,
    "8:6": 2920,
    "10:2": 602,
    "10:10": 593,
    "8:1": 2911,
    "11:9": 166,
    "9:7": 1507,
    "12:3": 36,
    "10:5": 603,
    "11:6": 183,
    "12:7": 41,
    "10:8": 589,
    "9:8": 1507,
    "10:4": 623,
    "11:7": 185,
    "11:5": 185,
    "11:11": 178,
    "10:6": 649,
    "11:10": 156,
    "12:8": 40,
    "11:1": 168,
    "14:9": 4,
    "13:3": 10,
    "11:3": 166,
    "10:7": 616,
    "11:4": 191,
    "12:4": 37,
    "16:10": 1,
    "12:1": 27,
    "17:8": 1,
    "11:8": 196,
    "14:12": 4,
    "13:10": 10,
    "11:2": 193,
    "12:10": 34,
    "12:11": 36,
    "12:12": 31,
    "12:5": 28,
    "12:6": 33,
    "16:9": 2,
    "12:9": 49,
    "13:13": 5,
    "14:4": 5,
    "12:2": 43,
    "13:5": 7,
    "13:6": 8,
    "14:2": 4,
    "13:7": 3,
    "13:1": 12,
    "13:4": 6,
    "13:11": 8,
    "15:4": 3,
    "16:6": 1,
    "31:2": 1,
    "33:20": 1,
    "18:8": 1,
    "13:12": 9,
    "14:13": 3,
    "16:15": 1,
    "13:2": 8,
    "14:11": 2,
    "13:8": 5,
    "21:5": 1,
    "15:6": 2,
    "14:7": 1,
    "14:6": 4,
    "27:1": 1,
    "13:9": 5,
    "26:5": 1,
    "21:13": 1,
    "68:5": 1,
    "24:21": 1,
    "21:8": 1,
    "69:13": 1,
    "15:1": 1,
    "16:16": 1,
    "31:5": 1,
    "20:17": 1,
    "33:4": 1,
    "20:2": 1,
    "14:3": 2,
    "25:4": 1,
    "15:8": 1,
    "29:13": 1,
    "20:3": 1,
    "16:8": 1,
    "90:7": 1,
    "33:15": 1,
    "18:4": 1,
    "15:15": 2,
    "21:1": 1,
    "43:35": 1,
    "29:20": 1,
    "14:8": 3,
    "53:46": 1,
    "25:21": 1,
    "16:5": 1,
    "41:34": 1,
    "26:21": 1,
    "23:6": 1,
    "25:8": 1,
    "15:3": 1,
    "15:2": 1,
    "20:13": 1,
    "42:15": 1,
    "27:2": 1,
    "16:11": 1,
    "39:17": 1,
    "15:10": 1,
    "36:8": 1,
    "30:20": 1,
    "57:13": 1,
    "14:5": 1,
    "28:26": 1
  },
  "keep_fraction_mean": 0.6914001201539611,
  "keep_fraction_std_population": 0.2913653409690173,
  "keep_fraction_quantiles": {
    "q05": 0.2,
    "q25": 0.5,
    "q50": 0.7142857142857143,
    "q75": 1.0,
    "q95": 1.0
  },
  "keep_fraction_bins_count": {
    "(0,0.25]": 56945,
    "(0.25,0.5]": 136335,
    "(0.5,0.75]": 88489,
    "(0.75,1)": 34314,
    "1.0": 195853
  },
  "keep_fraction_bins_proportion": {
    "(0,0.25]": 0.11123460745093136,
    "(0.25,0.5]": 0.26631258594824353,
    "(0.5,0.75]": 0.1728516845855732,
    "(0.75,1)": 0.06702790973871733,
    "1.0": 0.38257321227653457
  },
  "R_drop_equal_R_old_fraction": 0.38257321227653457
}
```

## Resource Statistics

```json
{
  "mean_seconds": 2.1093895218589087,
  "median_seconds": 2.083280324935913,
  "p95_seconds": 2.237115812301636,
  "max_seconds": 2.4858462810516357,
  "regular_updates": 495,
  "limit_seconds": 3,
  "peak_allocated_gib": 27.759892463684082,
  "peak_reserved_gib": 28.447265625
}
```

Full checkpoint: `/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_500_v1/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step500/step000500.pt`; SHA256 `d55dbc0975718f1b69158d9c8ffa27356e5ba182578a51a6567bd5095c2af113`.
Bare student: `/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_500_v1/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step500/student_step500.pt`; SHA256 `62662d079cf59d3a88c1a76583735268b0054436b349c4152213866598c37eae`.

## Conclusion

R-SentenceDrop does not improve the matched baseline.

Reporting recovery: the callback ran before sampling/resource statistics were aggregated. The original error is retained in RESULTS.json; existing successful training and evaluation outputs were reused without rerunning either stage.

## Evidence and Reproduction

Correctness and readable real-sample proofs are in evidence/. Statistics come from every valid sample over all500 updates/ranks. RDrop trace hashes match the reused baseline for every sample/F/P/K and original fixed-first reference stream. Model/loss sources stay byte-identical to fa19d12. Raw Recall fractions and strict export/acceptance evidence are retained.

Five native student protocols only, including reconstructed Long-DCI7602; no DCI Full, mask/gate inference, reranking or ensemble. No significance claim or rounded-score selection. No baseline retraining, promotion, extra q distributions, extra values or additional seeds.

```bash
cd /root/lk_projects/SAID-balanced-rdrop-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.run --launch
```

Exact runtime commits/commands/exits, native JSON and sampler evidence are preserved; large checkpoints/data/cache stay server-local. Stop after this single500 evaluation.
