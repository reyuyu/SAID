# T1 fix_v2: implementation and validation

The audited implementation detached its positive ranking baseline and used an incorrect router proxy.
Historical `t1_500step` artifacts remain untouched and are INVALID_FOR_T1_METHOD_COMPARISON.
Starting HEAD was 39612af4a7047933beae9153b4edaa2acdf1704b (clean); its only change since audited
8bdb632f1465bd37bce3f332e731febea807688a was the stopped-run ledger. No old T1/GPU job needed stopping.

## Corrections and mathematical mapping

- `ranking_sums` subtracts live matched scores in both hinges. Autograd-aware gather preserves global
  positives. Active 3x3 counterexample: correct SUM loss=.3, diagonal score gradients=-2/3,
  off-diagonal=1/3, common-shift derivative approximately zero. U gate remains detached.
- `soft_from_grid` uses C.detach(), logsigmoid(a), CLS log-weight=0, live logsumexp numerator and
  logsumexp(log_w) denominator. Constant C=c gives 2c and approximately zero gate gradient.
  Nonconstant independent weighted-exponential reference and FP64 finite differences pass.
- Separately restore the approved directional sum: remove the historical 0.5 in Said. Pair score
  also remains the sum of both token matching directions. L_total=L_Said+0.1*L_rec.
- FP32 numerical cores explicitly disable autocast; encoder outputs remain bf16, master weights FP32.
  U validity uses raw norm>1e-6 and nonzero actual complement count. Nonfinite inputs/features/target/
  prediction/loss fail rather than being repaired. Empty captions match EOS only.
- `PairwiseScorer.from_prepared` shares one cosine grid for hard and proxy directions. Matched positive
  path uses [b,33,33]. Local image rows x all global texts, live gathered candidate/query/positive
  gradients, and detached metadata retain all 1024 candidates. A symmetric legal set excludes identical
  image IDs and identical effective caption sequences, ignoring padding. M counts ordered pairs once.
- Backward Said=W*(local_I2T_sum+local_T2I_sum)/M; Rec=W*local_rec_sum/V, weighted 0.1 once.
  Detached sums produce true global logged means. First full batch M=1,047,550 (two extra duplicate
  exclusions beyond the 1024 diagonal pairs); local rank0 pairs=261,887.
- Token normalization/router projection happen once. Non-reentrant block checkpointing contains no
  collectives. 32x64 chunks give 128 blocks/rank at 256x1024; no repeated full global image grid.
  Intra-image slot cosine and all-image slot cosine use vector-sum identities, not 8192x8192 matrices.

## Validation

41 targeted/existing tests pass (37 unchanged tests in combined run, four DDP tests rerun after final
health-probe fix). Includes independent full small reference, scores/parameter gradients, matched
diagonal, one-grid count, checkpoint/chunks, exact duplicates, padding, empty text, invalid U, and
real 2-rank backward synchronization. Unequal local sizes fail before large gathers. No-loss rows keep
connected zero. Rec leaves text/router/frozen teacher unchanged; Said trains both backbones/modules.

FP64 continuous gradcheck: eps=1e-6, atol=1e-7, rtol=1e-5. FP32 independent parameter max error bound
2e-6+3e-5*max|reference|; DDP 2e-7+2e-5*max|reference|. No loss-factor or zero-positive failure is hidden.
The normal AdamW test verifies gradients and identical synchronized rank updates. Degenerate fixture
SGD updates verify single/DDP parameter equivalence because Adam's eps amplifies rounding in exactly
zero softmax-bias directions; production remains the original AdamW. No algorithm hyperparameters changed.

Actual CLI: two real-data GPU steps saved 0/1/2 with logs and unchanged teacher. This plumbing check used
small batch and is not a method result. Short full-configuration profile used four A800, 256/rank,
global1024, 32x64 chunks, checkpointing, three steps, original shared_init and horizon3651.
Synchronized step times 9.719/2.470/3.003 seconds; peak allocated 42.418 GiB on rank0.
Timing includes transfer/tokenization, compute, backward, optimizer and scalar reductions, excludes
loader wait/checkpoint save/cohort diagnostics. Profiler and first-step warmup affect these values.
Historical ~21.7sec and 55.6GiB are the erroneous run's observations, not a controlled same-math benchmark.
The only tested chunk candidate was 32x64; no further tuning was necessary. See fix_v2_profile.json.

Actual dtypes: student image/text=bf16; router projection/logits, hard/proxy, ranking, frozen reference,
decoder and reconstruction=float32. Common init digest caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf;
reference fingerprint 1428ea22801976c3358cf10905a64ba00bab7cadacda80e352016b3f8a2e1e9d.

## Formal experiment

Pending one new 500-step run after this code commit, from shared_init. Save0/20/100/500; first20 online
checks are part of that run. No old checkpoint resume. Exact fix_v2 step500 required for export/eval;
separate run evaluation directory prevents overwriting historical results. Frozen COCO/ShareGPT4V
protocol and existing S0/C0/C1 baselines will be reused; Urban1k remains a separate additional dataset.

UNSAID_SEMANTICS: NOT ESTABLISHED. U reconstruction incremental contribution: NOT ESTABLISHED (no T0).
T0, three epochs, hyperparameter search and new objectives: NOT RUN (outside authorized scope).
