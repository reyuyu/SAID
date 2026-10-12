# Provenance

| Artifact | SHA256 | Role |
|---|---|---|
| `inference/student_step4868.pt` | `4a1e5ed69b7aa728d7189518b9f285052e8293290963e079d886ddcab5fe819e` | strict-load native student |
| `training/hns_v1_l14_step4868_full.pt` | `60325144f586394ca75f7cc8fcd3432f777bc8a17324e1e76d84f7120f34d8bb` | complete optimizer/scheduler/RNG/cursor checkpoint |
| `training/common_step000000.pt` | `8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912` | verified untrained L14 initialization |

The full checkpoint is not a substitute for the bare student, and the bare student is not a continuation checkpoint. Large weights remain outside GitHub and are published only in the private ModelScope repository after identity, license and sensitive-metadata checks.
