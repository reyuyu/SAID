# Dataset and inference protocol audit

| Dataset | DeBias published protocol | Official code observed | SAID frozen protocol | Direct comparability |
| --- | --- | --- | --- | --- |
| COCO | Paper: val2017 5K; precise caption count unspecified | 5000/25014, all raw annotations | 5000/25000, first5/image, chunk512 | No exact match:14 extra official captions and evaluator/order differences |
| Urban1k | 1K diagonal pairs | 1000/1000; raw pairs identical to SAID | 1000/1000 frozen lexical order | Same raw protocol; AMP versus predeclared FP32 and metric implementation audited |
| Flickr | Full Flickr30k | 31014 images/155070 captions; val+train+test | test1K:1000/5000 | No for published/full result; unified test1K is comparable |
| DOCCI | test split 5K | 5000/5000; raw pair equality audited | 5000/5000 frozen test | Same raw pairs; precision/metric differences retained |
| DCI | short+long captions | 7805/7805, all splits, short+extra without inserted separator | DCI full not used in SAID fair table | No comparison to SAID Long-DCI |
| Long-DCI | Paper long-caption-only variant; no exact frozen-manifest identity supplied | No Long-DCI result in released base_full | reconstructed7602/7602 exact SHA | Only unified DeBias7602 versus SAID7602 is compared |

Phase A uses raw official data/reader inputs, not SAID manifests. Urban sorted raw pair fingerprints agree and all1000 captions match; enumeration order differs. DOCCI sorted image/caption pairs agree exactly. All official image references exist. Raw hashes, roots, counts and membership checks are in DATASET_PROTOCOL_AUDIT.json/URBAN_DATA_AUDIT.json.

Primary reproduction uses default AMP, while unified native FP32 was fixed before scores. The additional official FP32 Urban diagnostic changes only evaluation precision and confirms the adapter separately. URBAN_PRECISION_AUDIT.json preserves each changed query; primary AMP is not replaced by whichever precision scores higher.
