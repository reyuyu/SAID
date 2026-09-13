# Masked continuation from the completed step-500 run

The requested trajectory is the existing masked step 500 -> step 1000 -> native COCO
canonical and Urban-1k evaluation -> continuation from the same complete step-1000 checkpoint
through epoch 3 (3651 total updates). No new initialization and no native training arm.

The source is `dual_mask_suffix_masked_formal500_flatgather_v01/s0_dual_mask_suffix_masked_step000500.pt`,
SHA256 `24008b643672985978a89fc829bcfa42bb8f5d795b1a2e33ec0ed02423d7ead6`,
trained at code SHA `ff5ad1d4b918d56c6bfa48a2870dc5223e757237`.
CLIP, suffix gate, all three AdamW states, and the original cosine horizon are retained.

The legacy final checkpoint had `completed_steps=500`, but `step_in_epoch=500` because
the loop fetched the next batch before checking its stop condition. The next batch must
be index 500, not 501. Resume now derives the next position from completed updates and
loader length; final saves record the last actual update. Seeded loader replay verifies
each rank's entire saved sample/prefix/suffix digest before another optimizer update.
It also preserves cumulative sample counts and digest across segments, checks fixed
configuration and inherited log IDs, and refuses incompatible continuation.

Validation used the actual production main loop with a tiny model and CPU/Gloo: 18
uninterrupted updates over 3 epochs versus 9 + resume + 9, including an intentionally
one-ahead legacy checkpoint cursor. CLIP, suffix gate and AdamW-state max absolute errors
were 0.0, within fixed `atol=1e-7, rtol=1e-6`; LR indices, batch positions, and complete
data-stream digests agreed. This is a small-model resume check, not a second CLIP run.

Command:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/said-smartclip/bin/python tests/_dual_mask_suffix_resume_worker.py
```

## Retained invalid attempts

- `dual_mask_suffix_masked_continuation_v01`: mistakenly started from initialization;
  stopped after 344 updates, exit 1 after explicit termination. Excluded from the requested trajectory.
- First resume launch: SHA preflight mismatch, no optimizer updates (guard exit 91).
- Next resume launch: predecessor SHA rejection, no optimizer updates, exit 1. Stack retained
  in `resume_predecessor_failure.json`.
- `dual_mask_suffix_masked_from500_continuation_v01`: resumed with the incorrect legacy
  cursor; stopped at logged total 554 (54 additional updates), exit 1 after explicit termination.
  Excluded from the requested trajectory; no checkpoint from this segment is used.

Corrected trajectory directory: `runs_salu/dual_mask_suffix_masked_from500_continuation_v02`.
Its inherited 1..500 logs and step-500 checkpoint come directly from the original successful
run. The immutable original run and invalid-attempt artifacts remain on the server.

Checkpoint plan: preserve steps 0 and 500; save every 100 updates from 600 through 3600;
save final step 3651. Step 1000 also gets a strict bare-student export and frozen native
COCO canonical / Urban-1k results before further training. Failed stages do not auto-resume.
