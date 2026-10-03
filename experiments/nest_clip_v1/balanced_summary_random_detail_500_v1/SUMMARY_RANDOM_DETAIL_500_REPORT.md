# Summary + Random Detail Subset: strict500-update result

**MIXED.** Partial detail sampling addresses only part of the Summary–Detail failure.
Exactly500 optimizer updates from common step0 after separate smoke5, horizon4868.
No baseline retraining, full4868 confirmation or new sampling/weight/loss experiment follows.

## Three-way primary comparison

| Model | Score5_R1 % | J_long3 % | J_long % |
| --- | --- | --- | --- |
| RandomK | 69.900394 | 73.603324 | 82.340003 |
| Summary + All Detail | 69.405034 | 72.570390 | 81.570002 |
| Summary + Random Detail | 69.556218 | 72.553029 | 81.590002 |
| Delta versus RandomK pp | -0.344177 | -1.050295 | -0.750001 |
| Delta versus AllDetail pp | +0.151184 | -0.017361 | +0.019999 |

| Dataset | Dir | RandomK % | All-Detail % | Random-Detail % | Δ versus RandomK pp | Δ versus AllDetail pp |
| --- | --- | --- | --- | --- | --- | --- |
| COCO | I2T | 59.960000 | 59.920000 | 60.340000 | +0.380000 | +0.420000 |
| COCO | T2I | 40.924000 | 41.288000 | 41.704000 | +0.780000 | +0.416000 |
| Urban-1k | I2T | 89.000005 | 89.000005 | 88.500005 | -0.500000 | -0.500000 |
| Urban-1k | T2I | 87.900007 | 86.200005 | 86.900002 | -1.000005 | +0.699997 |
| Flickr30k-test1k | I2T | 86.200000 | 86.800000 | 86.700000 | +0.500000 | -0.100000 |
| Flickr30k-test1k | T2I | 70.300000 | 70.620000 | 71.500000 | +1.200000 | +0.880000 |
| DOCCI | I2T | 76.320000 | 75.880000 | 75.280000 | -1.040000 | -0.600000 |
| DOCCI | T2I | 76.140000 | 75.200000 | 75.680000 | -0.460000 | +0.480000 |
| Long-DCI | I2T | 55.393318 | 55.169692 | 53.485925 | -1.907393 | -1.683767 |
| Long-DCI | T2I | 56.866614 | 53.972639 | 55.472244 | -1.394370 | +1.499605 |

All scores/deltas use unrounded fractions. RandomK is the primary baseline; All-Detail is the
secondary failure-repair comparison. POSITIVE requires beating RandomK with no Long-DCI mean drop beyond0.2pp;
RESCUE-ONLY requires ≥0.30pp over AllDetail without beating RandomK. Remaining partial improvements are
MIXED; no improvement versus AllDetail is NEGATIVE. These are reporting rules, not training gates.

## Visible-only bounded sampling

Full raw strings/tokens preserve the formal baseline packing exactly. Summary is the first visible
sentence. D_pool is only the remaining complete visible sentences. K is uniform2..m-1 for m>=3;
m2 selects1 of2, m1 uses the sole detail, m0 Full-only. Independent SHA256(seed,epoch,sample_id,domain)
RNG chooses without replacement; selected indices are sorted. No shuffle, raw trailing sentence,
offset PAD, extra loss or changed candidate denominator runs. Old three modes remain reproducible.

Fixed100k simulation: Random Detail averages88.516732
effective tokens versus AllDetail155.920480;
mean K=4.030694, median4.0, K1 fallback0.100012%.
K=m count at m>=3 is0. Mean visible detail token coverage
57.455690%. All100k Full tensors/raw strings match;
all500×4 training sample-ID/Full/reference digests match the baseline, monitored live.

## Q1: Was All-Detail failure mainly excessive similarity to Full?

Before training, same common-step0 F/AllDetail mean cosine
0.903759;
F/RandomDetail0.781214.
On the same final500 model and same fixed1024 IDs, F/AllDetail
0.918021, F/RandomDetail0.795386.
Thus construction/embedding redundancy is measured separately from its retrieval utility.
The actual retrieval result and gate/mask/CE evidence above determine whether reduced redundancy
repairs the failure; cosine reduction alone is not causal proof that proximity caused the old degradation.
This run also removes old raw-tail budget violations; the all-visible-detail diagnostic control
distinguishes text-construction effects but is not a separate trained factorial control.
One seed and500 updates cannot prove a unique failure mechanism.
The actual last50 S-D mask IoU changes from
0.830646 (AllDetail model) to
0.852788 (RandomDetail model); masks do not become more separated
merely because text cosine decreases. The RandomDetail model's fixed step500 F-D IoU is
0.914523 (one read-only observation).
These are diagnostics, not additional constraints or an isolated causal training control.

| Same model stage | Text pair | Mean cosine | Median cosine |
| --- | --- | --- | --- |
| step0 | F_vs_All_Detail_raw | 0.903759 | 0.925702 |
| step0 | F_vs_All_Detail_visible | 0.904647 | 0.926611 |
| step0 | F_vs_Random_Detail | 0.781214 | 0.805299 |
| step0 | S_vs_Random_Detail | 0.660841 | 0.676482 |
| step0 | S_vs_All_Detail_raw | 0.641461 | 0.655122 |
| step0 | RandomK_F_vs_R | 0.759990 | 0.780549 |
| step0 | RandomK_P_vs_R | 0.668517 | 0.689805 |
| step500 | F_vs_All_Detail_raw | 0.918021 | 0.936489 |
| step500 | F_vs_All_Detail_visible | 0.918405 | 0.936763 |
| step500 | F_vs_Random_Detail | 0.795386 | 0.826841 |
| step500 | S_vs_Random_Detail | 0.573102 | 0.581048 |
| step500 | S_vs_All_Detail_raw | 0.576245 | 0.591465 |
| step500 | RandomK_F_vs_R | 0.729446 | 0.783338 |
| step500 | RandomK_P_vs_R | 0.587160 | 0.625969 |

## Q2: Does the subset recover T2I and per-view supervision?

Urban T2I deltas versus RandomK/AllDetail:-1.000005/+0.699997pp.
Long-DCI T2I deltas:-1.394370/+1.499605pp.
Actual last50 S_t2i/D_t2i:0.764769/
0.523805; actual AllDetail counterparts
0.749354/0.126268.

The task prompt's approximate prior S/D CE and IoU do not match saved AllDetail artifacts.
Actual prior S_t2i0.749354,D_t2i0.126268,S-D IoU0.830646. Original logs are authoritative;
baseline-diagnostic-correction.json records the discrepancy. Lower CE of an almost-Full view can
reflect an easier/less-partial task; it is not alone evidence of more useful supervision.

| View | CE direction | RandomK last50 | AllDetail actual last50 | RandomDetail last50 |
| --- | --- | --- | --- | --- |
| Full | I2T | 0.055952 | 0.052418 | 0.055656 |
| Full | T2I | 0.064539 | 0.062362 | 0.064523 |
| Summary/P | I2T | 0.196493 | 0.613796 | 0.628348 |
| Summary/P | T2I | 0.259438 | 0.749354 | 0.764769 |
| Detail/R | I2T | 0.887235 | 0.113485 | 0.467941 |
| Detail/R | T2I | 0.974187 | 0.126268 | 0.523805 |

## Ambiguity and mask diagnostics

Each view's nearest/top5/mean off-diagonal text cosine uses one fixed1024 set and the same model/tokenizer
within each stage. It measures semantic similarity, not confirmed false negatives or a modified loss.
No nearest neighbor changes training candidates. All-visible control is diagnostic-only.

| Stage | View | Nearest off-diagonal | Top5 negative mean | Mean off-diagonal |
| --- | --- | --- | --- | --- |
| step0 | Summary | 0.814489 | 0.774362 | 0.527796 |
| step0 | All_Detail_raw | 0.742623 | 0.705672 | 0.447234 |
| step0 | All_Detail_visible | 0.741752 | 0.704697 | 0.446092 |
| step0 | Random_Detail | 0.753255 | 0.722676 | 0.496093 |
| step0 | RandomK_P | 0.733179 | 0.691874 | 0.428412 |
| step0 | RandomK_R | 0.774106 | 0.745163 | 0.512996 |
| step500 | Summary | 0.739304 | 0.672184 | 0.322044 |
| step500 | All_Detail_raw | 0.618086 | 0.559785 | 0.234281 |
| step500 | All_Detail_visible | 0.618537 | 0.560056 | 0.234565 |
| step500 | Random_Detail | 0.619808 | 0.562350 | 0.246263 |
| step500 | RandomK_P | 0.642025 | 0.580150 | 0.259594 |
| step500 | RandomK_R | 0.640637 | 0.585505 | 0.256290 |

| Last50 read-only metric | RandomK | AllDetail | RandomDetail |
| --- | --- | --- | --- |
| F_keep_ratio | 0.891331 | 0.881888 | 0.910598 |
| O_keep_ratio | 0.869982 | 0.823891 | 0.838533 |
| E_keep_ratio | 0.871073 | 0.849838 | 0.869648 |
| oe_iou | 0.894495 | 0.830646 | 0.852788 |
| hard_inclusion_violation | 0.019033 | 0.024490 | 0.018356 |
| inc | 0.019186 | 0.019877 | 0.016118 |

Step1/100/200/500 and last50 record F/S/D lengths, CE directions, keep/gate stats,
S-D IoU, inclusion and hard violations. F-D IoU is read-only at the four existing diagnostic steps;
its last50 observation count is explicitly1 (step500), not50 measurements. Model source/math stays
byte-identical; detached observer tests show bitwise-equivalent loss, every gradient and actualAdamW update.

## Initialization, resources and strict export

Common initializer SHA54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6;
same architecture, four optimizer groups, best coefficients, seeds/batch/scheduler and F RNG streams.
Formal starts0, resume=None; smoke weights are not used. Loss/gradients finite and rank parameters agree.
4×A10080GB,256/rank,1024 full candidates, encoder checkpoint ON,pair OFF,128 image/text chunks.
Normal complete-cycle mean2.094615s,median2.072984s,
P952.207781s,max2.359459s.
Peak allocated/reserved27.759893/28.447266GiB.
≤3s mean/≤65GiB gate passed; checkpoint writes and first5 warmup recorded separately.
Strict export has optimizer steps[500], image/text native max_abs0.
Full checkpoint:/root/lk_projects/SAID-nest-clip-v1/balanced_summary_random_detail_500_v1/step500/step000500.pt; SHA256`259eb6e3753dcf6e341059a840c858d4ce548b13e5da7003edd5fce4bb203e9e`.
Bare student:/root/lk_projects/SAID-nest-clip-v1/balanced_summary_random_detail_500_v1/step500/student_step500.pt; SHA256`3b8162841dde46c4a9386243158bb6f0057e2659eea5aaa2e788fe1930de518c`.

## Complete frozen native recalls

Inference is normalized native image/full-caption text dot product only. No masks/gates/S/D,
PCA/reranking/ensemble/augmentation. COCO5000/25000,Urban1000,Flickr test1K1000/5000,
DOCCI5000,Long-DCI reconstructed7602 with immutable8890a2be... manifest. No DCI Full.

| Model | Dataset | Dir | R@1 % | R@5 % | R@10 % |
| --- | --- | --- | --- | --- | --- |
| RandomK | COCO | I2T | 59.960000 | 81.820000 | 88.680000 |
| RandomK | COCO | T2I | 40.924000 | 66.660000 | 76.268000 |
| RandomK | Urban-1k | I2T | 89.000005 | 98.200005 | 99.400002 |
| RandomK | Urban-1k | T2I | 87.900007 | 98.200005 | 99.100006 |
| RandomK | Flickr30k-test1k | I2T | 86.200000 | 97.200000 | 98.800000 |
| RandomK | Flickr30k-test1k | T2I | 70.300000 | 90.520000 | 94.660000 |
| RandomK | DOCCI | I2T | 76.320000 | 94.540000 | 97.560000 |
| RandomK | DOCCI | T2I | 76.140000 | 94.480000 | 97.520000 |
| RandomK | Long-DCI | I2T | 55.393318 | 74.875033 | 81.057616 |
| RandomK | Long-DCI | T2I | 56.866614 | 76.229939 | 82.044199 |
| AllDetail | COCO | I2T | 59.920000 | 82.240000 | 89.120000 |
| AllDetail | COCO | T2I | 41.288000 | 66.824000 | 76.876000 |
| AllDetail | Urban-1k | I2T | 89.000005 | 97.800004 | 99.400002 |
| AllDetail | Urban-1k | T2I | 86.200005 | 97.900003 | 98.900002 |
| AllDetail | Flickr30k-test1k | I2T | 86.800000 | 97.600000 | 99.200000 |
| AllDetail | Flickr30k-test1k | T2I | 70.620000 | 90.940000 | 94.960000 |
| AllDetail | DOCCI | I2T | 75.880000 | 94.860000 | 97.880000 |
| AllDetail | DOCCI | T2I | 75.200000 | 94.200000 | 97.140000 |
| AllDetail | Long-DCI | I2T | 55.169692 | 74.177848 | 79.992107 |
| AllDetail | Long-DCI | T2I | 53.972639 | 74.112076 | 80.110497 |
| RandomDetail | COCO | I2T | 60.340000 | 82.720000 | 89.400000 |
| RandomDetail | COCO | T2I | 41.704000 | 67.236000 | 77.044000 |
| RandomDetail | Urban-1k | I2T | 88.500005 | 97.700006 | 99.300003 |
| RandomDetail | Urban-1k | T2I | 86.900002 | 98.000002 | 99.000007 |
| RandomDetail | Flickr30k-test1k | I2T | 86.700000 | 97.200000 | 99.200000 |
| RandomDetail | Flickr30k-test1k | T2I | 71.500000 | 91.160000 | 95.380000 |
| RandomDetail | DOCCI | I2T | 75.280000 | 94.140000 | 97.360000 |
| RandomDetail | DOCCI | T2I | 75.680000 | 94.440000 | 97.260000 |
| RandomDetail | Long-DCI | I2T | 53.485925 | 73.467509 | 79.965798 |
| RandomDetail | Long-DCI | T2I | 55.472244 | 75.440674 | 81.333860 |

Config, structured commands, sampling/semantic audits, raw JSON, correctness and resource/export evidence
are in this directory. Large weights/data/embedding dumps stay outside Git. Completed at2026-10-04T03:50:17.809840+08:00.
