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
  `L_Said = mean_i2t hinge + mean_t2i hinge` with
  `hinge = [0.2 + s(anchor, negative) - s(anchor, anchor)]_+`.
* valid negatives exclude every pair whose two sides belong to the same original image (that one test
  drops the true positive). Identical effective caption sequences through EOS are also excluded,
  ignoring padding. The legal ordered pair set is symmetric.

## 6. U branch and reconstruction

```
m_U = 1 - sg(h_ii)
u   = Σ_p m_U,ip V_ip / max(Σ_p m_U,ip, 1) (raw aggregated slots; normalised only afterwards)
ĝ   = D([Norm(u); sg(g_T,i)])               D = Linear(1024,512) -> GELU -> Linear(512,512)
L_rec = mean_i (1 - <Norm(ĝ_i), sg(g0_i)>)
g0  = Norm(E_I^0(I_a))                       frozen deep copy of the *initial* vision tower
```

`EPS = 1e-6`; a sample whose raw `u` norm ≤ EPS or complement count is zero is skipped, and if every sample is
invalid the loss stays a connected zero. A non-finite loss or prediction is a **hard error**
(`nan_to_num` is never used).

## 7. Corrected row ownership and DDP scaling (fix_v2)

Each rank computes local image rows against all global text candidates. Every ordered pair s_ij
contributes both [margin+s_ij-s_ii]+ and [margin+s_ij-s_jj]+. For the symmetric legal ordered set A
this equals the original two-direction objective. M=|A|, not 2M.

Text candidates, projected EOS queries and matched positive scores use autograd-aware gathers.
Images remain local and live. IDs, caption identities and validity use metadata gathers.
Said backward is W*(local_i2t_sum+local_t2i_sum)/max(M,1).
Rec backward is W*local_rec_sum/max(V_global,1); its coefficient 0.1 is applied once.
Logs reduce detached sums to true global means. A rank with no legal pairs or no valid U still
participates in the same collectives. Equal small last batches are supported; unequal local batch
sizes fail together before large gathers.

## 8. Chunking and precision

* Chunk sizes are engineering settings; the candidate pool remains the whole global batch (1024).
  Normalize tokens and project router inputs once per step. Each block creates one `[n,m,33,33]`
  cosine grid, shared by the hard directions and the detached soft view. Matched positives create
  `[b,33,33]`. Non-reentrant checkpointing recomputes large activations; collectives stay outside.
* fp32 master weights + bf16 autocast for the student; the router proxy, the cosine and the
  reconstruction cores run in fp32 with autocast off; the frozen reference is fp32, `eval()`, `no_grad`.
* standard `DistributedDataParallel(find_unused_parameters=True)`; `_set_static_graph()` is deliberately
  **not** called (the graph differs between the positive pass, the chunked grid and the U branch).

This revision separately restores the originally approved Said directional SUM (removes the old 0.5).
Positive-score detachment and the proxy formula/graph were two independent P0 errors. The corrected
proxy uses stable logsigmoid(a), CLS log-weight zero, a live numerator, and logsumexp(log_w) as the
normalizer. Only C is detached; gradients through router inputs are allowed. Empty captions match
EOS alone; numerical aggregator fallback slots are invalid for matching. Slot diagnostics distinguish
intra-image and all-image similarity without constructing an 8192×8192 matrix.
Historical buggy runs are INVALID_FOR_T1_METHOD_COMPARISON.

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
