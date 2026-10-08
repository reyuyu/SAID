# HNS macro-loss four-arm500

Mother: `experiment/nested-d3-hard-nested-sparsity500-v1`. Historical HNS-v1
raw results and checkpoint hashes are verified against fetched Git and persistent
server assets. Full/E3 results are context only;500-step ranking uses HNS@500.

| Arm | lambda_align | lambda_sparse | lambda_hierarchy |
|---|---:|---:|---:|
| E1-A12 | 12 | 1 | 1 |
| E2-S08 | 10 | 0.8 | 1 |
| E3-H075 | 10 | 1 | 0.75 |
| E4-H125 | 10 | 1 | 1.25 |

The four JSON configurations differ only in these three scalar fields. Each
arm runs fresh common0 smoke5, independent fresh common0 formal500, complete
512000-sample stream validation, immutable matched gradient audit, strict
bare export/verification, validated four-GPU native five-benchmark evaluation,
report/push/fetch. Only after this finishes does the next arm start. Never
resume smoke or another arm. Final stop500; no extra arms or continuation.

## Macro arithmetic

`LA = (1.35 AF + 1.35 AD + 0.30 A3)/3`.
`LS = (OmegaF + 2 OmegaD + 2 Omega3)/3`.
`LH = (2 VDF + 2 V3D)/3`, using actual positive Hard-ST masks and ReLU(child-parent).
Total: `lambdaA LA + lambdaS LS + lambdaH ramp LH`.
Ramp uses completed-before-update: steps1/100/200/500 are0/.495/.995/1.

The original alignment component already contains10; the opt-in wrapper
replaces its outer factor by multiplying once by `lambdaA/10`. The old HNS
kernel already contains beta2/2 and ramp; its whole output is multiplied once
by lambdaH. Sparsity gets only lambdaS. Default10/1/1 keeps the exact old
arithmetic. No double10, beta, sparsity scale or renormalization.

Only `model/balanced_hparam_search.py` changes among production source files;
the trainer, kernel, sampler, augmentation, optimizer/LR and inference remain
byte-identical to the mother. Internal ratios stay1.35/1.35/.30 and1/2/2;
beta2/2, no mask-to-mask SG, old soft inclusion0, fixedK3 and horizon4868 stay
unchanged. Regularizers retain the original hidden-input detach to native
encoders. Formal training is4xA100,256/rank,workers8,seed0,original BF16 strategy.

CPU tests compare the actual fetched HNS forward at steps1/100/200/500 and
valid counts0/1/2/many, every gradient and AdamW update. Four-process gloo
checks DDP mean normalization and uneven valid partitions. Real local-image
ViT-B/16 BF16/NCCL compares all330 trainable parameters at a fixed global32
batch (validation only; formal batch stays1024). Default loss/components and
local/reduced gradients are exact. Both equivalence gates must pass before
formal GPU training.

Gradient audit uses the original immutable step500 first frozen global1024
batch,256/rank,all-reduce/4,no optimizer updates. It reports raw LA/LS/LH norms,
actual weighted norms, hierarchy–sparsity/alignment cosines and total gradient
per native visual/text, text mask/shared pool, visual mask and fusion/gate
group. Actual weighted-loss gradients are differentiated separately; raw-gradient
linear scaling estimates are recorded separately because BF16 can round them.
Additive relative-L2 tolerance is0.03% for all groups using actual weighted terms.
Loss/equivalence gates themselves require exact default results.

## Execution and progress

```bash
python -m recovery.hns_macro_fourarm --prepare
python -m pytest tests/test_hns_macro.py tests/test_hns_macro_runner.py tests/test_nested_d3_hns500.py tests/test_eval_five_parallel.py
python -m recovery.hns_macro_fourarm --publish-setup
python -m recovery.hns_macro_fourarm --launch
```

Before publication/launch, write passing CPU/DDP/real BF16 correctness receipts.
Launch refuses duplicates. The detached session ignores SIGHUP, holds a lock,
supervises process groups and stops on technical failure without retries.
Progress: `QUEUE_STATE.json`, each arm's first-five gate/steps, and canonical
`runtime/SAID-nest-clip-v1/hns-macro-fourarm500-v1/runner.log`.

GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. Batch64 and native
full-caption normalized embedding retrieval unchanged. Training/audit GPU
processes must exit before evaluation. No mask/gate inference, rerank, ensemble,
TTA or mathematical evaluator changes.

All images/indexes are local-only under `/root/said_s02_stage500`. Missing,
symlink or escaped local paths fail; no NFS fallback/copy/full decode audit.
`/root` is disposable overlay cache; persistent NFS originals remain intact.
Checkpoints/bare/raw large logs remain in canonical NFS runtime. Only explicit
small source/config/test/report/JSON artifacts are published.

Final reports include complete recalls/deltas, component/semantic diagnostics,
mask equality/coverage, actual gradient evidence and champions. Score5 is
primary. Balanced champion uses predeclared guards Jlong3/Short4/UrbanT2I of
-0.2/-0.2/-0.3pp. Gains below0.05pp are small and unconfirmed; a500-step winner
is only a human E1/E2 validation candidate, never a full-training claim.
