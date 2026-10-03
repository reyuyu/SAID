# Final comparison: manual checkpoint required

Status: **MANUAL_CHECKPOINT_REQUIRED**.390 model files were structurally inspected across /root and/tmp, including ignored files; no beta-CLIP candidate was found. Official CE/BCE downloads were then tried once each with20-second timeout and both failed. No training, model substitution or model/protocol adaptation was performed.

## CE primary comparison

CE is the closer loss-family comparison: SAID F/P/R alignments use softmax CE; beta-CLIP CE uses global CLS/EOS CE plus multi-positive fine-grained CE. The losses are not identical. BCE is secondary; its global alignment is still CE while the fine-grained objective differs.

All differences use **SAID minus beta-CLIP**. No difference is computable without beta-CLIP results.

| Dataset | beta-CLIP CE native CLS I2T % | SAID I2T % | Delta SAID-beta pp | beta-CLIP CE native CLS T2I % | SAID T2I % | Delta SAID-beta pp |
|---|---:|---:|---:|---:|---:|---:|
| COCO | N/A | 61.700000 | N/A | N/A | 42.220000 | N/A |
| Urban-1k | N/A | 92.100006 | N/A | N/A | 91.100007 | N/A |
| Flickr30k-test1k | N/A | 88.900000 | N/A | N/A | 72.440000 | N/A |
| DOCCI | N/A | 78.760000 | N/A | N/A | 79.980000 | N/A |
| Long-DCI | N/A | 59.668508 | N/A | N/A | 60.812944 | N/A |

SAID Score5_R1=72.768147%, J_long3=77.070244%, J_long=85.485003%.

## Required answers

| Question | CE primary | BCE secondary |
|---|---|---|
| Official checkpoint found? | No in scanned roots | No in scanned roots |
| SHA256? | Unknown: no file | Unknown: no file |
| Official Urban reproduced? | Not executed | Not executed |
| CLS/native adapter agreement? | Not tested | Not tested |
| Frozen five-dataset scores? | Not evaluated | Not evaluated |
| Score5/J_long3/J_long? | Unavailable | Unavailable |
| Ten directional differences? | Unavailable | Unavailable |

Published references, official checkpoint reproduction and unified SAID-protocol reevaluation are separate. Published Urban CE88.6 I2T/89.0 T2I and BCE92.3 I2T/91.8 T2I are not our results. No TCI value enters the fair comparison.

To resume, the required input is an official CE checkpoint transferred from the author Drive link, or its absolute server path if already uploaded. The identity script checks SHA, args/config, shapes and conditioner modules before any evaluation. CE proceeds independently of BCE availability. Preferred directory: /root/lk_projects/SAID-assets/external_baselines/beta_clip_v1/checkpoints/.
