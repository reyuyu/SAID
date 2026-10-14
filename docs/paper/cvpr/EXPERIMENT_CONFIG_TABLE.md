# Experiment configuration table

All rows below identify the source commit and formal comparison node. “Updates” is the completed optimizer-update count, not a dataloader batch count.

## Core training configuration

| Family / node | Backbone | Fusion / views | GPUs × per-rank batch | Epochs / updates | Scheduler horizon | Seed / sampling | Checkpoint source |
|---|---|---|---:|---:|---:|---|---|
| B/16 HNS-v1 @3651 | ViT-B/16 | BalancedStack, F/Dall/D3, `nested_detail_d3` | 4 × 256 | 3 / 3651 | 4868 | seed 0, sampling seed 0, shuffle seed 0 | `experiment/nested-d3-hns-full-v1` @ `45468e1`, step003651 |
| L/14 HNS-v1 @6085 | ViT-L/14 | BalancedStack, F/Dall/D3, `nested_detail_d3` | 4 × 256 | 5 / 6085 | 6085 | seed 0, sampling seed 0, shuffle seed 0 | `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1` @ `8a0b6e8`, step006085 |
| ABCDE A.Full @3651 | ViT-B/16 | Full A / BalancedStack | 4 × 256 | 3 / 3651 | 4868 | seed 0; archive comparison | `archive/said-abcde-results-v1` @ `4ef949c` |
| ABCDE B @3651 | ViT-B/16 | Joint-Mask Only | 4 × 256 | 3 / 3651 | 4868 | seed 0; archive comparison | `archive/said-abcde-results-v1` @ `4ef949c` |
| ABCDE C @3651 | ViT-B/16 | Nested Alignment Only | 4 × 256 | 3 / 3651 | 4868 | seed 0; archive comparison | `archive/said-abcde-results-v1` @ `4ef949c` |
| ABCDE D @3651 | ViT-B/16 | Text-only SmartMask | 4 × 256 | 3 / 3651 | 4868 | seed 0; archive comparison | `archive/said-abcde-results-v1` @ `4ef949c` |
| ABCDE E @3651 | ViT-B/16 | Vanilla Long-Caption | 4 × 256 | 3 / 3651 | seed 0; archive comparison | `archive/said-abcde-results-v1` @ `4ef949c` |

## Frozen objective and optimizer settings

| Item | Formal value |
|---|---|
| Contrastive views | F, Dall, D3; symmetric I2T/T2I CE |
| Alignment view weights | `[1.35, 1.35, 0.30]` |
| Sparsity view weights | `[1, 2, 2]` |
| HNS | enabled; beta `[2, 2]`; detail chain; linear ramp over first 200 updates |
| Ordinary inclusion | disabled (`inclusion_max=0`) |
| Optimizer | AdamW; betas `(0.9, 0.999)`; epsilon `1e-8` |
| Weight decay | `1e-2` backbone; `0` masks/adapters |
| Peak LR | backbone `1e-6`; text/shared mask `1e-3`; visual mask `1e-3`; Fusion adapter `2e-4` |
| Precision / memory | BF16 autocast; encoder checkpointing on; pair-block checkpointing off |
| Data | ShareGPT4V; 1,245,901 training records after deterministic first-1,000 skip |

## Archived ABCDE headline comparison (Step3651)

Values are percentages from `ABCDE_COMPARISON.md`; they are included to preserve the historical component table and should not be substituted for the formal B/16 HNS-v1 `FULL_RESULTS.json` identity without checking provenance.

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A.Full | 73.774418 | 78.188029 | 87.040003 | 67.154 | 94.1 | 92.6 | 93.35 |
| B. Joint-Mask Only | 72.849872 | 76.789120 | 86.165001 | 66.941 | 93.3 | 91.5 | 92.40 |
| C. Nested Alignment Only | 73.232981 | 77.618968 | 86.535003 | 66.654 | 93.5 | 91.8 | 92.65 |
| D. Text-only SmartMask | 73.749518 | 78.126531 | 86.905003 | 67.184 | 94.0 | 92.2 | 93.10 |
| E. Vanilla Long-Caption | 71.956256 | 75.968426 | 85.470003 | 65.938 | 92.7 | 91.1 | 91.90 |

## Evaluation scope

The standard five are COCO-Val5K, Flickr30k-Test1K, DOCCI, reconstructed Long-DCI (7,602 pairs), and Urban-1k. Flickr30k-Full is a separate 31,783-image / 158,915-caption protocol and is not interchangeable with Flickr30k-Test1K or Long-DCI.
