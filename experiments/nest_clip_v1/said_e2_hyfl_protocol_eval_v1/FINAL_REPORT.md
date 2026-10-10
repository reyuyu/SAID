# E2@4868 HyFL protocol unification: final audit

Branch: `experiment/said-e2-hyfl-eval-unified-v1`, base `6dcae270f8e754323512648e1e09ed903ab698fc`. Only one fixed native bare model evaluated; no training/optimizer/checkpoint/model change. Production model/train/eval diff is empty (CODE_MANIFEST.json). Strict317-key FP32/context248 load passed; bare SHA before/after `50634512e226e79d526e269ba1a6ca75d7248f47e75417537f1605ca71ac2a0e`.

Outcome: real four-GPU run COMPLETED; protocol verification is PARTIAL. Three main source protocols verified (DOCCI, Urban, COCO), DCI and Long-DCI provisional, Full Flickr blocked. No claim of six fully aligned datasets, comprehensive superiority or unbiased SOTA.

| Model | DOCCI I2T/T2I | DCI I2T/T2I | Long-DCI I2T/T2I | Urban I2T/T2I |
| --- | --- | --- | --- | --- |
| E2@4868 (actual) | 80.200 / 81.220 | 69.058 / 69.289 (UNVERIFIED) | 60.274 / 60.668 (UNVERIFIED) | 93.900 / 92.900 |

| Model | COCO I2T/T2I | Flickr Full I2T/T2I |
| --- | --- | --- |
| E2@4868 (actual) | 61.860 / 43.000 | BLOCKED |

| Dataset | Direction | Queries | Candidates | R@1% (correct) | R@5% (correct) | R@10% (correct) |
| --- | --- | --- | --- | --- | --- | --- |
| DOCCI | I2T | 5000 | 5000 | 80.200000 (4010) | 96.420000 (4821) | 98.560000 (4928) |
| DOCCI | T2I | 5000 | 5000 | 81.220000 (4061) | 96.180000 (4809) | 98.420000 (4921) |
| DCI | I2T | 7805 | 7805 | 69.058296 (5390) | 86.367713 (6741) | 91.108264 (7111) |
| DCI | T2I | 7805 | 7805 | 69.288917 (5408) | 86.329276 (6738) | 90.787956 (7086) |
| Long-DCI | I2T | 7602 | 7602 | 60.273612 (4582) | 78.189950 (5944) | 83.925283 (6380) |
| Long-DCI | T2I | 7602 | 7602 | 60.668245 (4612) | 78.702973 (5983) | 83.701657 (6363) |
| Urban-1k | I2T | 1000 | 1000 | 93.900000 (939) | 99.100000 (991) | 99.600000 (996) |
| Urban-1k | T2I | 1000 | 1000 | 92.900000 (929) | 99.100000 (991) | 99.400000 (994) |
| COCO | I2T | 5000 | 25000 | 61.860000 (3093) | 83.660000 (4183) | 89.700000 (4485) |
| COCO | T2I | 25000 | 5000 | 43.000000 (10750) | 68.740000 (17185) | 78.160000 (19540) |
| Flickr30k-Full | I2T | NOT_RUN | NOT_RUN | N/A | N/A | N/A |
| Flickr30k-Full | T2I | NOT_RUN | NOT_RUN | N/A | N/A | N/A |
| Flickr30k-Test1K | I2T | 1000 | 5000 | 90.100000 (901) | 98.500000 (985) | 99.300000 (993) |
| Flickr30k-Test1K | T2I | 5000 | 1000 | 74.000000 (3700) | 92.080000 (4604) | 95.820000 (4791) |

Full Flickr official annotation:31783 images/158915 captions, exact #0–#4 mapping; available image filenames 1000, missing 30783, evaluated0. DCI actual7805/7805; Long-DCI actual7602/7602, officialCSV missing. The latter follows frozen TULIP constructor and is token-identical to the old7602 protocol, but CSV equivalence remains unproved.

Tests:14 CPU tests passed; real fourGPU old/new native FP32 comparison at the frozen formal batch64 passed with5e-6 tolerance and zero query-hit differences. Observed max difference:0.0. Full parameter-state digest unchanged. Long-DCI7602 token arrays/image/order matched exactly despite3025 whitespace-string differences. Historical five-dataset R@1 counts have a hard exact-match gate; R@5/10 are compared and remaining deviations, if any, are disclosed (LEGACY_COMPARISON.md).

Failed checks preserved: strict-TF32-off versus old/native features exceeded5e-6 (max0.000900492); native batch3 versus16 exceeded5e-6 (max0.000456773), despite unchanged small-sample hits. The first issue was the new controller's accidental precision override, now corrected to ORIGINAL native FP32 defaults; the second means batch3 optimization is not approved. Formal batch64 crossGPU equivalence passes; no threshold is increased. Both earlier aborted queues and all diagnosis receipts remain immutable. No claim that every attempted numerical check passed is made.

COCO primary retains its frozen CPU512/1D-argsort scoring rule and exactly reproduces the historical6 counts on new model features. GPU streaming's one I2T R1 query difference is kept as an explicit diagnostic; no old score was copied or chosen for being larger. CANONICAL_COCO_REPROOF.json independently proves the primary result. The exact worker version used for encoding/scoring is preserved as WORKER_EXECUTED_NATIVE_V3.py; the final reusable worker records both scoring paths and keeps canonical COCO primary by default.

| Job | Physical GPU | Seconds | Peak allocated GiB | Peak reserved GiB | Status |
| --- | --- | --- | --- | --- | --- |
| COCO | 0 | 86.775 | 1.197 | 1.822 | COMPLETED |
| DCI | 1 | 141.262 | 1.197 | 1.822 | COMPLETED |
| Long-DCI | 2 | 138.457 | 1.197 | 1.822 | COMPLETED |
| DOCCI | 3 | 87.030 | 1.197 | 1.822 | COMPLETED |
| Flickr30k-Test1K | 0 | 22.900 | 1.197 | 1.822 | COMPLETED |
| Urban-1k | 3 | 17.634 | 1.197 | 1.822 | COMPLETED |

FourGPU scheduled wall time:144.672s. See RUN_STATE.json for UTC, physical mapping, exit codes, CPU/disk preflight and worker logs. Completed jobs report no OOM/nonfinite features. Final compute process list: `EMPTY`. All controlled workers exited. Final GPU readings:

```
0, 0 %, 1 MiB
1, 0 %, 1 MiB
2, 0 %, 1 MiB
3, 0 %, 1 MiB
```

Remaining limitations: unavailable Full Flickr image library; unavailable HyFL DCI_test.json and caption-construction proof; unavailable official dci_long.csv and TULIP results.csv; no independently sealed validation protocol; public benchmarks repeatedly observed. Local mirrored image byte equality with HyFL-private datasets cannot be established. Full-Flickr end-to-end timing/memory and crossGPU sharded encoding remain unvalidated; only generic bounded-memory mathematics and dataset-level fourGPU scheduling are validated. No scores are fabricated or borrowed from legacy reports as new inference.

Preparation failures were model-free and preserved in PREPARATION_DIAGNOSTICS.json. Two incompatible evaluation queues stopped on legacy regression differences; original cancellation evidence and exact source diffs are in EVALUATION_RECOVERY_AUDIT.json and NATIVE_PRECISION_RESTORE_AUDIT.json. The final original-native FP32 profile was frozen and verified BEFORE the final run, with fresh cache/output namespace and no bypass of earlier identities. Complete query-level evidence, verified feature shards, official downloaded references and large logs remain under `/root/said_hyfl_protocol_eval_v1/` and the server runtime references directory. New small artifacts only are published. Historical E2 checkpoint and results remain untouched. Task stops here; no later experiment or training is started.
