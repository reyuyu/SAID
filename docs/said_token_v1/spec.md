# SAID-Token v1 (experiment T1) — frozen specification as implemented

Branch `codex/said-token-v1`, worktree `/root/SAID-token-v1`, base `9b47262f270bb099cda45feb63b3fd1a22853259`.

One sentence: aggregate the **native last-layer local tokens** of CLIP into 32 image slots and 32 text
slots per modality, pick (per image–text candidate pair) 16 visual slots as *Said* with a
text-conditioned router, run **one** fine-grained best-match ranking objective over
`{native CLS} ∪ {16 Said slots}` against `{32 text slots} ∪ {native EOS}`, and let the complementary 16
visual slots plus the observed text predict the complete feature of a **frozen copy of the initial
vision tower**:

```
L = L_Said + 0.1 * L_rec
```

There is deliberately **no** separate global image–text loss and **no** global pre-training stage: the
native CLS and the native EOS stay inside the same fine-grained matching set, and the native student CLS
is what the retrieval evaluation reads.

## 1. Last-layer interface

* image: `image_global_raw, image_patch_raw = clip.encode_image_with_patches(images, use_checkpoint=False)`
  — patches already went through `ln_post` and `visual.proj`.
* text: `text_global_raw, text_hidden = clip.encode_text(text, return_full=True)`, then
  `text_local = text_hidden @ clip.text_projection`, **once**.
* `d` is read from `text_projection.shape[-1]` (this CLIP has no `embed_dim` attribute).
* text validity: `eot_pos = argmax(token_ids, dim=-1)` (EOT id 49407); valid content lies **strictly
  between** SOT and EOT, so padding, SOT and EOT are all excluded. An empty caption is reported through
  an `empty` flag and never produces an all-`-inf` softmax (a dead row falls back to one legal
  position; the arithmetic stays finite and differentiable).

## 2. Self-aggregation (per modality, no shared parameters)

`A(X) = softmax_L(f(X)^T)`, `X' = A(X) X`, with
`f = LayerNorm(d) -> Linear(d, 102) -> GELU -> Linear(102, 32)` and a **fixed logit scale of 1**.
One aggregator per modality, CLS/EOS are never aggregated, text padding/SOT/EOT are never attended.

## 3. Router (text-conditioned, per candidate pair)

```
a[i, j, p] = <Norm(W_V V[i, p]), Norm(W_T g_T[j])> / tau_r
```

* `W_V`, `W_T` are `Linear(512, 128, bias=False)`; `tau_r = 0.07`.
* `V[i, p]` is the *p*-th **aggregated visual** slot; `g_T[j]` is the candidate text's **native global
  EOS feature**. The 32 aggregated text slots are used by the fine-grained match and are **not** part of
  the first-version router — there is no per-text-token reduction over `q`.
* `a` has shape `[B_image, B_text, 32]`; `hard = 1{p ∈ TopK(a, 16)}` over the trailing visual axis only.
  The native CLS is never a top-k candidate (it is not even in `a`).
* ties are broken by **ascending slot index** (stable sort of the negated logits, since `topk` does not
  guarantee the lowest index for equal values); no noise is injected, so the mask is a reproducible
  function of (pair, parameters) and never of the random stream.
* every candidate pair re-evaluates the router, so `h[i, j, p]` is a function of *that* candidate text.
* the positive pair's mask is `h[i, i, p]` and the U branch uses exactly it:
  `m_U[i, p] = 1 - h[i, i, p].detach()`.

## 4. Hard forward / score-layer backward proxy

```
s = s_hard + (s_tilde - sg(s_tilde))
```

`s_tilde` is built from `C_bar = C.detach()` with `w = [1; sigmoid(a)]` (the leading 1 belongs to the
native CLS), `eta = 0.1`:

```
s_v2t = Σ_p w_p max_q C_bar_pq / Σ_p w_p
s_t2v = (1/|T|) Σ_q eta [ logsumexp_p(C_bar_pq/eta + log w_p) - logsumexp_p(log w_p) ]
```

The forward value is exactly the hard score; the only gradient this term carries is through
`sigmoid(a)`. This is a **biased straight-through proxy**, not the exact derivative of top-k; the
top-k slice itself is not differentiable and the router is never left without a gradient.

## 5. Said fine-grained match

* visual side: `{native CLS} ∪ {16 selected slots}` = **17** tokens (masked slots are *removed* from the
  candidate set before the max — zero-filling them would let them win as soon as every legal cosine is
  negative; in the visual→text direction the same set is enforced through the mean's denominator).
* text side: `{32 text slots} ∪ {native EOS}` = **33** tokens.
* per-token normalisation, bidirectional **max**-similarity, `s_hard = mean over 17 + mean over 33`
  (no extra `/2`).
* ranking loss, margin `0.2`, mean over all valid (anchor, negative) incidences, no mining:
  `L_Said = 0.5 * (mean_i2t hinge + mean_t2i hinge)` with
  `hinge = [0.2 + s(anchor, negative) - s(anchor, anchor)]_+`.
* valid negatives exclude every pair whose two sides belong to the same original image (that one test
  drops the true positive as well as any duplicate caption of the same image).

## 6. U branch and reconstruction

```
m_U = 1 - sg(h_ii)
u   = (1/16) Σ_p m_U,ip V_ip                (raw aggregated slots; normalised only afterwards)
ĝ   = D([Norm(u); sg(g_T,i)])               D = Linear(1024,512) -> GELU -> Linear(512,512)
L_rec = mean_i (1 - <Norm(ĝ_i), sg(g0_i)>)
g0  = Norm(E_I^0(I_a))                       frozen deep copy of the *initial* vision tower
```

`EPS = 1e-6`; a sample whose `Norm(u)` is degenerate (norm ≤ EPS) is skipped, and if every sample is
invalid the loss stays a connected zero. A non-finite loss or prediction is a **hard error**
(`nan_to_num` is never used).

## 7. DDP conventions (measured, not assumed)

* The Said term couples every anchor with every candidate, so the gathered features are gathered with the
  **differentiable** functional collective (`torch.distributed.nn.functional.all_gather`). Its backward
  all-reduces the incoming gradients and hands each rank the part belonging to its own shard, which is
  what makes DDP's average over ranks reproduce the single-process gradient. The loss value therefore
  carries **no** `world_size` factor, and the counts are global on every rank (all-reducing them would
  multiply the denominator by `world_size`).
* The U branch is per sample and never mixes ranks, so its term **does** carry the explicit
  `world_size` factor (`W * Σ_local / V_global`), which is what turns DDP's average back into the global
  mean.
* Equal per-rank local batch sizes (including a ragged last batch) are required and checked with one
  tiny collective **before** any large gather; a mismatch raises identically on every rank.
* A single-process run keeps the identity gather, so the tests can compare a 2-rank step against a
  1-process reference.
* Two alternatives were implemented, measured and rejected in this round: a plain (non-differentiable)
  `all_gather` silently detached the local slice too — the text tower and the image projection received
  no gradient at all — and an "own anchors only" loss dropped every candidate-side partial gradient. Both
  were caught by the 2-rank equivalence test.

## 8. Chunking and precision

* `chunk_image = 8`, `chunk_text = 32`; the candidate pool is the whole global batch. Only small
  score/gate blocks live in the graph — the full `[B, B, 33, 33]` tensor is never materialised.
* fp32 master weights + bf16 autocast for the student; the router proxy, the cosine and the
  reconstruction cores run in fp32 with autocast off; the frozen reference is fp32, `eval()`, `no_grad`.
* standard `DistributedDataParallel(find_unused_parameters=True)`; `_set_static_graph()` is deliberately
  **not** called (the graph differs between the positive pass, the chunked grid and the U branch).

## 9. Optimizer groups (fixed, no search this round)

| group | contents | lr | weight decay | warmup |
|---|---|---|---|---|
| backbone | student visual + text tower incl. the original projections | 1e-6 | 1e-2 | 200 |
| aggregators + router | both aggregators and the router | 2e-4 | 1e-2 | 200 |
| decoder | `D` | 1e-4 | 0 | 200 |

The frozen reference tower, the historic `mask_net` and the unused `logit_scale` are in **no** optimizer.
The cosine horizon is `3 * len(loader)`; this round only runs `max_steps = 500`.

## 10. Frozen-out (must not appear in T1)

Separate global image–text loss / global warm-up stage / hidden alignment weight; silently keeping the old
SmartCLIP SIDM/DISM terms or their 10 and 2 weights in the total objective; the old 512-d `mask_net`, the
old `SaidRouter`, H11/final-block routing; the patch mean in place of the native CLS; the residual target
`g0 - W_s g_S`; Absorption; a Global-chasing decoder output; NCE reconstruction, dual-input
identification, CVSSL, orthogonality, energy balance, a new mask-sparsity loss; reading a hidden caption
suffix, USS, pseudo captions, an external strong teacher, EMA; automatically changing the loss weights,
the augmentation, the keep ratio or the lr in response to early results.
