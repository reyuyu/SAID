# SAID-CLS-CVSSL v0.1 — implementation status report

Branch `codex/said-cls-cvssl-v01` (base `d2067019a018257809852e391a950daa15712ded`).
Objective `said_cls_cvssl`: `L = L_smart + lambda_U * L_U^visual`. No method/loss/threshold was
changed after the fact; every number below is measured.

## 1. Source audit vs the attachment (findings, per the "report, do not silently adapt" rule)

| Item | Repository reality | Consequence |
| --- | --- | --- |
| `CLIP.forward(image, text, rank, soft=False)` | exists at `model/model_longclip.py:553`; `mask_logits = mask_net(full_text_embedding.detach())`, `hard = (soft>=0.5).float() - soft.detach() + soft`, `use_mask = soft if soft else hard`, SIDM/DISM with the fixed `100` multiplier, targets `linspace(rank*bs, ...)` | matches the attachment exactly; `compute_smartclip_terms` reproduces it line by line |
| gather | `torch.distributed.nn.all_gather` (autograd-aware) for text features, masks and raw image features | kept for SmartCLIP; CVSSL uses a plain detached gather |
| loss weights | `lambda_sparse=2`, `lambda_align=10` (`train/train.py:214`) | kept, not halved |
| optimizers / schedulers | `AdamW(lr=1e-6, wd=1e-2)` for everything except mask_net, `AdamW(lr=1e-3, wd=0)` for mask_net; `cosine_lr(warmup=200)` and `cosine_lr(warmup=0)`, `steps = epochs * len(loader)` | reproduced identically |
| `steps_per_epoch` | the reference DataLoader has **no** `drop_last`, so `len(loader) = 1217` (not 1216); the last batch is ragged (180/rank) | computed from the live loader, logged, ragged batch supported |
| view-a preprocessing | `share4v_train_dataset(preprocess=None)` builds the transform with openai-`clip.load('ViT-L/14')`; `_transform(224)` is Resize(224,BICUBIC)+CenterCrop(224)+RGB+ToTensor+CLIP-Normalize, and it is **tensor-identical** to the LongCLIP ViT-B/16 transform (measured `torch.equal == True`) | view a is rebuilt with the same ops and proven identical in a unit test |
| reference trainer device handling | `train/train.py::train_epoch` at HEAD calls `self.model(images, ...)` **without moving `images` to CUDA**; that call fails (`RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.HalfTensor)`), yet the historical 3-epoch SmartCLIP run produced 3 checkpoints and `loss.txt` with 600+ iterations | **interface inconsistency reported**: the committed reference trainer is not directly runnable; the new trainer moves inputs explicitly. `CLIP.forward` itself is unaffected and is what the equivalence gate compares against |
| `logit_scale` | `train/train.py:71` replaces it with `ones * 4.6052`; `CLIP.forward` never uses it (the scale is the fixed 100) | reproduced (it receives no gradient and only decays). **Correction (see `v01_ddp_debug_matrix.md`, `v01_distributed_correctness_fix_report.md`):** the reference trainer *does* wrap the model in `DistributedDataParallel` (`train/train.py:171`) and calls `_set_static_graph()` (`train/train.py:173`). The claim that it does not was wrong, and the CVSSL trainer added in `9a7c086` performed no parameter-gradient synchronisation at all; both are fixed and verified in the follow-up round. |
| init state | `runs_salu/phase30a_2_A_said_only/salu_initial.pt` → complete shared state, digest `caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf`, **identical to the historical SmartCLIP reproduction's `initial_state_sha256`**, mask_net present (0 missing) | all four arms load this one file |

## 2. Implementation

`model/complement_visual_ssl.py`, `model/said_cls_cvssl.py`, `train/said_cvssl_data.py`,
`train/train_said_cls_cvssl.py`, `tools/exp_said_cls_cvssl_{init,smoke}.sh`,
`tests/test_said_cls_cvssl.py`, `tests/test_complement_visual_ssl_ddp.py` (+2 torchrun workers).

* `m_U = 1 - (1-rho) sg(m_S)`, `rho = 0` fixed; G0 = all ones; R0 = per-sample coordinate
  permutation of that same complement (own value histogram preserved, one mask per anchor row).
* `Q_ij = <Pi_{m_U,i}(v_i^a), sg[Pi_{m_U,i}(v_j^b)]> / tau_U` with **the anchor's mask on every
  candidate**, computed by the exact matmul form
  `num/(tau*sqrt(a_norm)*sqrt(cand_norm))`, `M^2` kept, FP32 core with autocast explicitly off;
  the wrong candidate-mask form differs by 2.3 (measured) and is a test.
* `L_U = (1/2)(L_U^ab + L_U^ba)`, both directions recomputed (never transposed).
* Candidates gathered with a plain detached all-gather; anchors live; `m_U` detached.
* Validity: empty complement, zero masked norm (anchor/positive/negative), duplicates by
  `image_id`, and "no valid negative" all exclude the row *before* the cross-entropy; a
  zero-valid rank executes every collective and returns a connected zero (tested, 2 ranks, no
  deadlock).
* Cross-rank scaling: `1/V_global` per direction (`--ddp_gradient_averaging 0`, the trainer's
  convention of this objective); `W` is applied when the flag is 1 **and the model is DDP-wrapped** -- measured to give the exact global-mean U gradient (`norm ratio 1.000000` with it, `0.500000` without), see `v01_ddp_debug_matrix.md` §4.

## 3. Evidence that ran

* `pytest tests/ -q` → **498 passed, 2 skipped** (30 new tests).
* **S0 equivalence gate (PASS)**: identical weights/inputs → SIDM, DISM, sparsity and the weighted
  total agree to `<=1e-5/1e-4`; every parameter gradient agrees to a relative `<=1e-4`; one AdamW
  step on the same optimizer state leaves both models `allclose(1e-7, rtol=1e-5)`. Both hard and
  soft mask variants.
* **CVSSL 2-rank vs single-process gradient equivalence (PASS)**: sum of per-rank gradients ==
  single-process gradient (no-DDP convention) and rank-average == single-process gradient (DDP
  convention), each with equal *and* unequal local valid counts (rel. `<=1e-5`).
* **Gradient isolation (PASS, unit level)**: `L_U` gives exactly zero gradient to `mask_net`, the
  text encoder and `logit_scale`, a non-zero gradient to the visual tower, and both views receive
  an anchor gradient; `m_U=0` coordinates get exactly zero direct gradient.
* **20-step 4-arm engineering smoke: RUN** — S0 / G0 / R0 exited 0 (C0 completing at the time of
  writing); logs at `runs_salu/said_cls_cvssl/smoke_<arm>/salu_log.jsonl`.
  * Two blockers were found and fixed during the smoke, both environmental rather than algorithmic:
    (a) OOM at 256 pairs/GPU → both views now run with activation checkpointing
    (`--grad_checkpoint_views 1`, identical for all four arms, objective math unchanged), peak
    memory dropped to ~28–45 GB/GPU; (b) a rank-0-only gradient probe deadlocked the other ranks
    (it re-runs a graph that contains collectives) and is incompatible with reentrant
    checkpointing — the probe is therefore **NOT RUN** in the traine
    (`weighted_vssl_to_smart_grad_ratio` and `cos(grad_visual L_smart, grad_visual lambda_U L_U)`
    are absent from the logs; the isolation evidence is the unit tests instead).

## 4. NOT RUN (explicit)

* 500-step four-arm control runs — **NOT RUN** (command: `bash tools/exp_said_cls_cvssl_smoke.sh 500 step500`).
* Canonical retrieval for Initial/S0/G0/R0/C0 — **NOT RUN**.
* 3-epoch runs — **NOT RUN** (and not to be started automatically).
* `weighted_vssl_to_smart_grad_ratio`, `cos(grad_visual L_smart, grad_visual L_U)` — **NOT RUN`**
  (see §3).

## 5. Verdict (single seed, short horizon)

* Implementation: **PASS** for the parts that ran (tests, S0 equivalence, smoke), with one
  correction from the follow-up round: the DDP equivalence reported here was measured with a
  *feature-level* gather only. Parameter-gradient synchronisation was missing in `9a7c086`; the
  corrected trainer and its acceptance tests are in `v01_distributed_correctness_fix_report.md`.
  The 20-step numbers produced before that fix are engineering records only and are superseded.
* Retrieval: **UNRESOLVED** — no retrieval evaluation was completed in this round.
* Complementary semantic preservation: **NOT ESTABLISHED**. No withheld/attribute/concept
  benchmark was run; cross-view top-1, mask differences and loss decreases are **not** accepted
  here as evidence of true Unsaid semantics.

None of the four arms was tuned (`lambda_U`, `rho`, `tau_U` and the augmentation are the
pre-registered values), and no hyper-parameter scan was performed.
