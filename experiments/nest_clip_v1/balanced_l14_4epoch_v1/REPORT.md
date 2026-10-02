# Balanced L14: Fixed Four-Epoch Migration

Status: `running`. Stage: `6d44ae8d5c34-0-long_dci`.

OpenAI ViT-L/14 at224, text context248, seed0. Faithful Balanced coefficients: fusion_lr2e-4, visual_mask_lr_scale1, view_weights[1,1,1], sparsity_scale1, inclusion_max1. Horizon4868 from a new L14 step0, full4x256 direct logical batch, full1024 candidates (padded sampler tail720), original RandomK and loss definitions.

## Initialization and Correctness

L14 step0: `/root/lk_projects/SAID-nest-clip-v1/balanced_l14_4epoch_v1/shared/step000000.pt`.
L14 step0 SHA256: `93600bb47184e9d57383c5705a0fa2ec7779c0dfc2409c8bb05f72215f537e18`.
Official pretrained URL: https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
Official pretrained SHA256: `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.

Architecture/native/export/position checks use the real downloaded L14 weights. Ten two-rank edge cases and local resume tests use dimension-aware tiny native encoders with real768-channel,12-head MaskNet blocks and256 patches. A separate actual full-L14 two-rank reference on the fixed first two valid training samples checks every named gradient and AdamW update at the unchanged tolerances, using identical per-sample FP32 kernel shapes. B16 native inference and default loss/gradient/AdamW regressions are preserved.

Additional unnormalized Gaussian-image/short-caption full-L14 stress tests exceeded the original convolution-gradient tolerance and are preserved as failed evidence. They were not hidden or made to pass by increasing tolerances. Actual-training-input full-model tests and random-input dimension-aware fixtures passed; the numerical scope is explicit.

## Resource Gate

Each attempt measures5 warmup plus30 complete real-DataLoader updates, slowest rank. The approved limit is5s per regular update and65GiB allocated per GPU. Initialization, checkpoint writes and final agreement checks are separate.

The user explicitly approved updates within5s and instructed continuation. The original3s failure remains below as historical evidence; a fresh5s gate is required.

### direct-128x128

Passed: `False`. OOM: `False`.

```json
{
  "passed": false,
  "ranks": [
    {
      "rank": 0,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 165.5336399152875,
      "peak_allocated_gib": 47.36288833618164,
      "peak_reserved_gib": 48.59765625,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 1,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 165.46719890087843,
      "peak_allocated_gib": 47.35835361480713,
      "peak_reserved_gib": 48.548828125,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 2,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 165.2650107294321,
      "peak_allocated_gib": 47.363563537597656,
      "peak_reserved_gib": 48.59765625,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 3,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 165.47169978916645,
      "peak_allocated_gib": 47.36153793334961,
      "peak_reserved_gib": 48.62109375,
      "final_nccl_all_reduce": 10.0
    }
  ],
  "speed_gate": {
    "warmup_steps": 5,
    "measured_steps": 30,
    "threshold_seconds": 3.0,
    "mean_seconds": 4.308759291966756,
    "median_seconds": 4.308880805969238,
    "p95_seconds": 4.405271553993225,
    "max_seconds": 4.452944755554199,
    "all_steps_at_most_3s": false,
    "allocated_limit_gib": 65.0,
    "every_rank_peak_allocated_at_most_65gib": true
  },
  "resource_failure": "measured full update exceeded 3 seconds"
}
```

### direct-128x128-limit5s

Passed: `True`. OOM: `False`.

```json
{
  "passed": true,
  "ranks": [
    {
      "rank": 0,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 164.27521651238203,
      "peak_allocated_gib": 47.36120367050171,
      "peak_reserved_gib": 48.763671875,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 1,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 164.3818627372384,
      "peak_allocated_gib": 47.36153793334961,
      "peak_reserved_gib": 48.62109375,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 2,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 164.31319300830364,
      "peak_allocated_gib": 47.36288833618164,
      "peak_reserved_gib": 48.59765625,
      "final_nccl_all_reduce": 10.0
    },
    {
      "rank": 3,
      "completed_updates": 35,
      "updates_this_run": 35,
      "max_parameter_difference_from_rank0": 0.0,
      "seconds": 164.3443381935358,
      "peak_allocated_gib": 47.36044692993164,
      "peak_reserved_gib": 48.71484375,
      "final_nccl_all_reduce": 10.0
    }
  ],
  "speed_gate": {
    "warmup_steps": 5,
    "measured_steps": 30,
    "threshold_seconds": 5.0,
    "mean_seconds": 4.31757329305013,
    "median_seconds": 4.308165073394775,
    "p95_seconds": 4.440561532974243,
    "max_seconds": 4.527675151824951,
    "all_steps_at_most_limit": true,
    "all_steps_at_most_3s": false,
    "allocated_limit_gib": 65.0,
    "every_rank_peak_allocated_at_most_65gib": true
  },
  "resource_failure": null
}
```

## Native Scores

| Model | Updates | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|---:|
| B16 reference | 3651 | 4868 | 72.664715 | 76.972525 | 85.345001 |
| B16 reference | 4868 | 4868 | 72.768147 | 77.070244 | 85.485003 |
| L14 | 0 | 4868 | 54.511173 | 50.697956 | 59.890001 |
| L14 | 500 | 4868 | not run | not run | not run |
| L14 | 3651 | 4868 | not run | not run | not run |
| L14 | 4868 | 4868 | not run | not run | not run |

### L14 @0

Full checkpoint: `/root/lk_projects/SAID-nest-clip-v1/balanced_l14_4epoch_v1/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step4868/step000000.pt`; SHA256 `95055d73d824ffbb562035096b584fef1ba806f53636274555a7f031ea390bc9`.
Bare student: `/root/lk_projects/SAID-nest-clip-v1/balanced_l14_4epoch_v1/evaluations/step0/student_step0.pt`; SHA256 `94598e9732b4339a6985f74cc91d11ce691d23b396c3db81d3585ee8737e9e08`.

| Dataset | Direction | R1 % | R5 % | R10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 56.080000 | 79.540000 | 86.860000 |
| COCO | T2I | 35.324000 | 59.948000 | 70.144000 |
| Urban-1k | I2T | 68.300003 | 88.500005 | 93.600005 |
| Urban-1k | T2I | 53.100002 | 78.200006 | 86.000001 |
| Flickr30k-test1k | I2T | 84.600000 | 97.700000 | 99.200000 |
| Flickr30k-test1k | T2I | 64.920000 | 87.240000 | 92.040000 |
| DOCCI | I2T | 57.560000 | 83.680000 | 89.800000 |
| DOCCI | T2I | 60.600000 | 86.280000 | 91.900000 |
| Long-DCI | I2T | 33.307024 | 51.696922 | 58.563536 |
| Long-DCI | T2I | 31.320705 | 50.552486 | 58.537227 |

The prespecified final result is L14@4868. best_observed is selected only among500/3651/4868 by raw Score5_R1 and is explicitly benchmark-selected; no cross-checkpoint dataset mixing. The auxiliary mask/gate branches are not used in native inference.

## Reproduction

```bash
cd /root/lk_projects/SAID-balanced-l14-4epoch-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_l14_4epoch_v1.prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_l14_4epoch_v1.run --launch
```

Exact commands, hashes, code commits and exit statuses are retained in evidence and runtime state. Weights, optimizer/RNG checkpoints, data and caches remain local. Five frozen native protocols only, including reconstructed Long-DCI7602; no DCI Full, extra seeds, extra epochs or tuning trials.
