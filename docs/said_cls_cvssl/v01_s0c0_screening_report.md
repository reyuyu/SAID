# SAID-CLS-CVSSL v0.1 — S0 / C0 500-step screening

Scope: **performance screening only.** One startup blocker was fixed, S0 (`lambda_U = 0`) and C0
(`lambda_U = 1`) were each trained for 500 steps from the shared init, and canonical native-CLS
retrieval was run on Initial / step100 / step500. G0 and R0 were **not** started. No algorithm,
hyper-parameter, augmentation or tolerance was changed.

* branch `codex/said-cls-cvssl-v01`, screening run at the startup-fix commit (recorded per run as
  `git_head`; the fix commit itself is the one that carries this report)
* shared init: `runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`,
  `INIT_STATE_SHA256 = caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf`
  (matches the historical SmartCLIP reproduction's `initial_state_sha256`)
* raw results: `outputs/cvssl_screening/{S0,C0}_canonical.json`,
  `runs_salu/said_cls_cvssl/{ddpcheck_step20_*,ddpfix_step500_*}/`

## 1. Startup blockers fixed (A)

Two, both fatal at launch, neither algorithmic:

| # | symptom | fix |
| --- | --- | --- |
| 1 | `NameError: name 'image_b' is not defined` in the training loop | the view-b pixel digest read an outer `image_b` that only exists inside the step/probe helpers; changed to `tensor_digest(batch['image_b'][0])` |
| 2 | `NameError: name 'SaidClsCvsslTrainModule' is not defined` | the trainer used the class but never imported it; added to the existing `from model.said_cls_cvssl import (...)` |

A `symtable` scan of the whole trainer was run afterwards: no unresolved name remains except the
three local-import false positives (`random`/`np` in `seed_everything`, `subprocess` in `git_head`).

Nothing else was touched: no algorithm, no DDP redesign, no test tolerance, no new replication
matrix.

## 2. Configuration actually used

Everything below is read from the run's own `config.json` / `run_summary.json`, not from intent.

| item | value |
| --- | --- |
| model | ViT-B/16, LongCLIP 248 tokens, view-a = openai-clip `_transform(224)` |
| view-b | `shared_content_weak_v1`: same window, resample in [192, 224] bilinear, Gaussian blur k=3 σ~U[0.1, 1.0] |
| data | full ShareGPT4V, `share-captioner_coco_lcs_sam_1246k_1107.json`, 1,245,901 samples after the `[1000:]` slice |
| batch | 256 original pairs/rank, 4×A800 → 1024 global pairs, 2048 global views |
| seed / masks | seed 0, `rho = 0`, `tau_U = 0.1`, hard straight-through Said mask |
| losses | `lambda_align = 10`, `lambda_sparse = 2` |
| optimizer | AdamW backbone lr 1e-6 wd 1e-2; `mask_net` lr 1e-3 wd 0; warmup 200 / 0 |
| schedule | cosine over `3 × len(loader) = 3 × 1217 = 3651` steps; `max_steps=500` only truncates |
| DDP | real `DistributedDataParallel`, `find_unused_parameters=True`, `_set_static_graph()`, `ddp_gradient_averaging=1` |
| precision | fp32 master + bf16 autocast, non-reentrant activation checkpointing on both views |
| horizon check | `DATASET_SIZE 1245901 STEPS_PER_EPOCH 1217 LR_HORIZON 3651` |

**Measured dtype audit at launch (every rank, from the live parameters — not metadata):**

```
DTYPE_AUDIT rank=0 parameter_dtypes=[('torch.float32', 317)] backbone_group=['torch.float32']
            mask_group=['torch.float32'] bf16_tensors_in_model=0 amp_enabled=True amp_dtype=bf16
```

(The same line is printed by all four ranks.) The trainer now *raises* if the master weights or an
optimizer group are ever not fp32.

Correction to an earlier preflight number: `len(loader)` is **1217**, not
`ceil(1245901 / 256) = 4867`. On this torch version `DistributedSampler.__len__` already holds a
single replica's share, so `len(loader) = ceil(1245901 / (4 × 256)) = 1217`. The horizon actually
used is the correct one (3651), taken from the live loader inside the trainer.

## 3. Online running check, first 20 steps (C)

Each arm was launched once; the 20-step check is the first 20 steps of the 500-step run, plus a
separate 20-step launch used to validate checkpoint save/load early.

| check | S0 | C0 |
| --- | --- | --- |
| losses finite, no NaN/Inf over the 101 logged records | **PASS** (0 non-finite) | **PASS** (0 non-finite) |
| no OOM / no deadlock | **PASS** (peak 32.1 GB / 27.1 GB of 80 GB) | **PASS** |
| both optimizers update | **PASS** (101 distinct `rank_param_digest` values, `lr` 5e-09→9.82e-07, `mask_lr` 1e-3→9.55e-4) | **PASS** |
| DDP copy stays in sync | **PASS** (single shared parameter state; no rank diverged, no hang) | **PASS** |
| bf16 autocast path runs | **PASS** (`amp_enabled=True amp_dtype=bf16`) | **PASS** |
| checkpoint saves and reloads strictly | **PASS** (`STRICT_LOAD_OK missing=[] unexpected=[]`) | **PASS** |
| step-1 losses | `loss_sidm 3.2474143505096436`, `loss_dism 1.742993950843811`, `loss_sparsity 0.4442138671875`, `loss_smart 50.792510986328125` | **identical to S0** |

Nothing blocked the runs: no training error, no non-finite value, no synchronisation failure, no
data or positive-pairing error. Small fluctuations and the initial transient were ignored as
instructed. The gradient-angle / gradient-ratio probes stayed off.

## 4. Checkpoints and provenance (D)

| arm | run dir | steps | files | SHA256 |
| --- | --- | --- | --- | --- |
| S0 | `ddpfix_step500_S0_smartclip` | 20 / 100 / 500 | `cvssl_S0_smartclip_step{000020,000100,000500}.pt` | 500-step: `758dcdd2a9112d54ccb8784e211d329297209c5d7be26cd7a89e68185867c743` |
| C0 | `ddpfix_step500_C0_complement_vssl` | 20 / 100 / 500 | `cvssl_C0_complement_vssl_step{000020,000100,000500}.pt` | 500-step: `4f0bf8369dc6beeaf5302308bba1a5d38844441c68bdb4602094f37a5618f617` |

Save points were passed explicitly (`--save_completed_steps 0,20,100,500`), never left to a default.
For both arms:

* `completed_steps == 500` (also `RUN_SUMMARY` and `run_summary.json`);
* strict reload verified: `model.load_state_dict(..., strict=True)` → `missing=[] unexpected=[]`,
  317/317 tensors fp32; both optimizer state dicts load, `step=500`;
* the checkpoint really moved: 315/317 tensors differ from the init (S0 `max_abs 1.197e-01`,
  C0 `1.463e-01`);
* every checkpoint records `initial_state_sha256 = caf61198…`, `lr_horizon_steps = 3651`,
  `precision = fp32 master + bf16 autocast`, `objective = said_cls_cvssl`;
* S0@500 and C0@500 are genuinely different models (315/317 tensors differ, `max_abs 1.193e-01`).

New output directories; the pre-fix `smoke_*` artifacts were neither overwritten nor used.

## 5. Matched-run audit (E) — AUDIT_OK

Both at 20 steps and at 500 steps, S0 vs C0:

| quantity | S0 | C0 | verdict |
| --- | --- | --- | --- |
| `initial_state_sha256` | `caf61198…` | `caf61198…` | MATCH |
| `sample_stream_sha256` | `39c9885c41aaf11cd79a05b7828cf07b1cfa6af2d4ad507c978e870f3a422856` | same | MATCH |
| `caption_stream_sha256` | `799f8efae42541759ac92239643d2150dc8aa23d1da984676f029063c21c0fd4` | same | MATCH |
| `view_b_param_stream_sha256` | `cbfb7ffdfee3151a214beb57474a796adc9f5545702a73df4f8f1cb5fbbe7c19` | same | MATCH |
| `view_b_pixels_sha256_step0` | `81888072f9b73f6f` | same | MATCH |
| `steps_per_epoch` / `lr_horizon_steps` | 1217 / 3651 | 1217 / 3651 | MATCH |

* **Config diff:** the only differing keys are `arm`, `arm_mask`, `lambda_U`. Precision, optimizer,
  schedule, batch, seed, augmentation and data are byte-identical.
* **Step-1 identity:** `loss_sidm`, `loss_dism`, `loss_sparsity`, `loss_smart`,
  `loss_smart_global_mean`, `sidm_top1`, `dism_top1` are **bit-equal** between the arms — the same
  init and the same data stream.
* Different ranks see different data shards, as expected. S0 is the primary control as instructed;
  no exact historical SmartCLIP number was required before analysing C0.

## 6. Canonical retrieval protocol (F)

Frozen historical protocol, same evaluator and manifests for both arms:

```
protocol = phase30a-1d-fixed-cohort      usr_protocol = sharegpt4v1k-usr-v1
usr_manifest_sha256 = ba2298ad836be0d95e952633b8a2a747f08509dba843b41b4ab6e37d6a703b19
gap_anti_temperature = 1.0               c_u_is_evaluation_target_only = True
canonical: repr=legacy_cls pool=mean, loaded 317 tensors, skipped 0, router-absent 4
```

* Primary representation is exactly `normalize(clip.encode_image(image))` /
  `normalize(clip.encode_text(text))`. No masked Said/U features, no patch mean, no
  pair-conditioned reranking.
* COCO main protocol (val2017, I2T/T2I R@1/R@5/R@10); ShareGPT4V fixed 1K with `first_sentence`,
  `fixed_sparse`, `full_dense`. Candidate sizes and splits were **not** changed.
* The Existing evaluator already reads `payload['model']`, so the CVSSL checkpoints load as-is:
  **no format adapter and no evaluation change was needed.** Both JSONs reference the *same*
  `Initial` file (`sha256 c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`), so the
  Initial column is shared by construction.

## 7. Performance table (G)

Full 24-row table, every cell measured (also written to
`outputs/cvssl_screening/screening_table.md`):

| metric | Initial | S0@100 | C0@100 | S0@500 | C0@500 | C0−S0@500 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO I2T R@1 | 0.5170 | 0.5534 | 0.5598 | 0.6058 | 0.6042 | −0.0016 |
| COCO I2T R@5 | 0.7662 | 0.7964 | 0.7982 | 0.8220 | 0.8224 | +0.0004 |
| COCO I2T R@10 | 0.8428 | 0.8706 | 0.8774 | 0.8906 | 0.8896 | −0.0010 |
| COCO T2I R@1 | 0.3269 | 0.3630 | 0.3658 | 0.4124 | 0.4072 | −0.0051 |
| COCO T2I R@5 | 0.5776 | 0.6143 | 0.6167 | 0.6709 | 0.6655 | −0.0054 |
| COCO T2I R@10 | 0.6823 | 0.7164 | 0.7182 | 0.7662 | 0.7636 | −0.0026 |
| 1K first I2T R@1 | 0.5440 | 0.5970 | 0.5960 | 0.6620 | 0.6670 | +0.0050 |
| 1K first I2T R@5 | 0.7740 | 0.8130 | 0.8190 | 0.8940 | 0.8960 | +0.0020 |
| 1K first I2T R@10 | 0.8590 | 0.8990 | 0.8960 | 0.9520 | 0.9550 | +0.0030 |
| 1K first T2I R@1 | 0.5140 | 0.5680 | 0.5590 | 0.6350 | 0.6220 | −0.0130 |
| 1K first T2I R@5 | 0.7480 | 0.7910 | 0.7790 | 0.8580 | 0.8610 | +0.0030 |
| 1K first T2I R@10 | 0.8140 | 0.8510 | 0.8450 | 0.9290 | 0.9310 | +0.0020 |
| 1K sparse I2T R@1 | 0.7460 | 0.8460 | 0.8410 | 0.9010 | 0.9030 | +0.0020 |
| 1K sparse I2T R@5 | 0.9200 | 0.9630 | 0.9610 | 0.9820 | 0.9850 | +0.0030 |
| 1K sparse I2T R@10 | 0.9530 | 0.9780 | 0.9780 | 0.9970 | 0.9970 | +0.0000 |
| 1K sparse T2I R@1 | 0.7400 | 0.8370 | 0.8360 | 0.8790 | 0.8840 | +0.0050 |
| 1K sparse T2I R@5 | 0.9180 | 0.9530 | 0.9540 | 0.9710 | 0.9670 | −0.0040 |
| 1K sparse T2I R@10 | 0.9490 | 0.9670 | 0.9670 | 0.9820 | 0.9820 | +0.0000 |
| 1K full I2T R@1 | 0.7580 | 0.9110 | 0.9110 | 0.9650 | 0.9640 | −0.0010 |
| 1K full I2T R@5 | 0.9200 | 0.9880 | 0.9890 | 0.9980 | 0.9980 | +0.0000 |
| 1K full I2T R@10 | 0.9520 | 0.9980 | 0.9980 | 0.9990 | 0.9990 | +0.0000 |
| 1K full T2I R@1 | 0.7760 | 0.9010 | 0.8990 | 0.9620 | 0.9620 | +0.0000 |
| 1K full T2I R@5 | 0.9480 | 0.9860 | 0.9840 | 0.9970 | 0.9980 | +0.0010 |
| 1K full T2I R@10 | 0.9730 | 0.9940 | 0.9940 | 0.9990 | 0.9990 | +0.0000 |

The eight headline rows the brief asked for are the R@1 cells of this table.

### S0 − Initial and C0 − S0, step 500 (percentage points)

| metric | S0@500−Initial | C0@500−S0@500 |
| --- | ---: | ---: |
| COCO I2T R@1 | **+8.88** | −0.16 |
| COCO I2T R@5 | +5.58 | +0.04 |
| COCO I2T R@10 | +4.78 | −0.10 |
| COCO T2I R@1 | **+8.54** | −0.51 |
| COCO T2I R@5 | +9.33 | −0.54 |
| COCO T2I R@10 | +8.39 | −0.26 |
| 1K first I2T R@1 | **+11.80** | +0.50 |
| 1K first T2I R@1 | **+12.10** | −1.30 |
| 1K sparse I2T R@1 | **+15.50** | +0.20 |
| 1K sparse T2I R@1 | **+13.90** | +0.50 |
| 1K full I2T R@1 | **+20.70** | −0.10 |
| 1K full T2I R@1 | **+18.60** | +0.00 |

### 100 → 500 trend (percentage points)

| metric | S0 500−100 | C0 500−100 |
| --- | ---: | ---: |
| COCO I2T R@1 | +5.24 | +4.44 |
| COCO T2I R@1 | +4.93 | +4.14 |
| COCO T2I R@5 | +5.66 | +4.88 |
| 1K first I2T R@1 | +6.50 | +7.10 |
| 1K first T2I R@10 | +7.80 | +8.60 |
| 1K sparse I2T R@1 | +5.50 | +6.20 |
| 1K full T2I R@1 | +6.10 | +6.30 |

Both arms keep improving from 100 to 500 steps in every cell; the increments are nearly identical
for the two arms.

## 8. Training-side quantities

Averaged over steps 401–500 (mean over the 20 logged records in that window, both arms):

| quantity | S0 | C0 |
| --- | ---: | ---: |
| `loss_smart_global_mean` | 5.862 | 5.745 |
| `loss_vssl_global_mean` | 2.790 | 0.896 |
| U valid anchor fraction | 1.000 | 1.000 |
| Said mask keep ratio | 0.798 | 0.787 |
| U mask keep ratio | 0.202 | 0.213 |
| Said retained energy | 0.712 (unsaid 0.288) | 0.803 (unsaid 0.197) |
| `vssl_ab_top1` | 0.999 | 0.999 |
| `sidm_top1` / `dism_top1` | 0.940 / 0.931 | 0.944 / 0.932 |
| `lr` / `mask_lr` | 9.8e-07 / 9.63e-04 | 9.8e-07 / 9.63e-04 |
| wall time / step | 0.934 s | 1.209 s |
| peak GPU memory | 32.09 GB | 27.05 GB |
| wall clock for 500 steps | 574.5 s | 707.4 s |
| sync check | 101/101 distinct `rank_param_digest`, 0 non-finite | same |

Read this table carefully:

* the mask **keep ratios** are close (Said 0.798 vs 0.787, U 0.202 vs 0.213) but not identical;
* the **retained energy** does separate the arms — at step 500 the C0 `Said` mask keeps 0.803 of the
  image feature energy against S0's 0.712 (U energy 0.197 vs 0.288). At step 20 the two arms were
  within 0.002 of each other (0.806 vs 0.808), so this is a divergence that developed during
  training, consistent with C0 actually training on the U term;
* `loss_vssl_global_mean` differs only because C0 actively minimises U while S0 does not (its value
  is a passive by-product of the representation drift), so it is **not** comparable across arms;
* neither the energy split nor the U loss is a performance claim, and none of them is evidence of
  semantic preservation.

## 9. Answers to the three questions

**1. S0 vs Initial — what improved, what fell?**
Everything improved, in every one of the 24 cells. The largest gains are on ShareGPT4V-1K
full/sparse I2T R@1 (+20.7 / +15.5 pp) and the smallest on 1K full R@10 (+2.6 pp, already 0.952 at
init). Nothing fell. This is a 500-step screening on one seed, not a converged result.

**2. C0 vs S0 at step 500 — what improved, what fell?**
It is a wash. C0 is better on 5 of the 12 R@1 cells and worse on 7, with every difference inside
±1.3 pp; the largest single effects are COCO T2I R@1/R@5 (−0.51 / −0.54 pp, S0 better) and
1K first T2I R@1 (−1.30 pp, S0 better), while 1K first I2T R@1 (+0.50), 1K sparse T2I R@1 (+0.50)
and 1K first I2T R@5 (+0.20) marginally favour C0. On the COCO main protocol S0 is slightly ahead
on all six cells (−0.10 to −0.54 pp). No cell shows a difference that this design can call real:
there is a single seed, no R@5/R@10 pattern, and no confidence interval was computed for this
screening.

**3. What is the 100 → 500 trend?**
Monotone improvement for both arms in every cell, with almost identical increments (+4.1 to +8.6 pp
for C0, +4.4 to +8.1 pp for S0 on the R@1 cells). Extrapolating, the curves have not separated;
whatever separates them, if anything, is not visible in this window.

## 10. Verdict

* **500 steps is a screening, not a final judgement.** S0 and C0 are statistically
  indistinguishable on this run: the arm difference is at most 1.3 pp on R@1 and mostly ~0.1–0.5 pp,
  on a single seed with no interval estimate. Both are clearly far better than Initial.
* This is a **mixed / null** result for C0, reported as measured. The algorithm is **not** declared
  a failure and no hyper-parameter was tuned in response. G0/R0 were not run, so the comparison is
  not yet the pre-registered four-arm contrast.
* Even if C0 had won, that would only mean a performance gain was observed in this experiment.
  **Complementary semantic preservation remains NOT ESTABLISHED**: no withheld/attribute/concept
  benchmark was run, and none of the diagnostics above (U cross-view top-1, U loss, mask
  differences, low `cos(U, Global)`, retained energy) is evidence for it.

## 11. Not run

* G0_global_vssl, R0_random_vssl (deliberately not started)
* 3-epoch runs, hyper-parameter search, `lambda_U` / `rho` / `tau_U` scans, augmentation changes
* `weighted_vssl_to_smart_grad_ratio` and the gradient-angle probes (kept off on purpose)
* multi-seed repeats and confidence intervals for the screening numbers
* bf16 autocast numerics audit (CPU autocast cannot emulate it; must be done on the GPU run)
