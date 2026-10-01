# Balanced Fixed-Seed Hyperparameter Search

Base: ec7194a, unchanged Balanced-Stack-Patch architecture and shared step0.
Exactly five exposed coefficients: fusion_lr, visual_mask_lr_scale, view_weights,
sparsity_scale, inclusion_max. B0: 1e-4 / 1 / [1,1,1] / 1 / 1.

Coordinate order and candidate values are frozen in search.py. Both candidates
are always evaluated; no value, network, data, temperature or seed is appended.
Ten new500 trials at most. Ranking uses raw Score5_R1, then exact-tie J_long3,
then exact-tie J_long, without rounding or improvement thresholds. Five datasets
all participate in tuning; this is not a blind benchmark.

All trial IDs are SHA256 of canonical five-parameter JSON. Different trials start
at the common untrained checkpoint. Only same-trial promotions restore their
own full states: after all500-step evaluations, global Top2@500 continue directly
to3651. The user replaced the original Top3@500 to1217 / Top2@1217 to3651 plan;
no intermediate1217 screening or training is scheduled. Horizon stays3651.
B0@500/@3651 are verified and reused rather than retrained. Its legacy
three-group optimizer can be split by parameter name into four
groups without changing moments, steps or LR. Existing B0@3651 may be reused if
selected again; this avoids duplicate training of the identical configuration.

The change is applied by a scheduler handoff after the active child finishes.
Only the old supervisor is paused/replaced; the running torchrun/evaluator is
allowed to finish normally, with its exit status and artifacts preserved.

The network source model/nested_fusion_mask.py is byte-identical to the parent.
BalancedSearch adds only coefficient handling and extra detached diagnostics.
Backbone: peak1e-6/warmup200/WD.01; text_mask_and_shared_pool: peak1e-3/WD0;
visual_mask: peak1e-3*scale/WD0; fusion_adapter (A_V and W_g): fusion_lr/WD0.
The last three groups use the original3651 cosine without warmup.

Before search: exact default loss/gradient/AdamW regression, coefficient/optimizer
isolation, resume rejection and two-rank global score/loss/gradient/AdamW cases.
A common B0 real4x256 gate runs5+30 updates. Every new trial has5-step smoke,
then500 formal updates and all five frozen native evaluations. No per-trial gate
repetition is needed because hyperparameters leave computation unchanged.

DataLoader generation is saved per rank. Mid-epoch checkpoints preserve the
epoch-start generator state for exact batch replay; boundary checkpoints preserve
the generator state for the next epoch. Images use deterministic Resize/CenterCrop,
and RandomK uses SHA256(seed,epoch,sample_id), so no stochastic augmentation is
introduced. Full sampler1217 updates/epoch and padded720-candidate tails persist.

Commands:

```bash
python -m experiments.nest_clip_v1.balanced_hparam_search_v1.search --bootstrap
python -m experiments.nest_clip_v1.balanced_hparam_search_v1.search
```

Bootstrap runs only the common resource gate and verifies/reuses B0 references.
The second command executes the entire adaptive coordinate/promotion plan.
Outputs: /root/lk_projects/SAID-nest-clip-v1/balanced_hparam_search_v1/.
State, trial JSON, checkpoints, optimizer/RNG and full token logs stay there;
compact leaderboards/evidence/report are synced to this branch on completion.
Failures are preserved; no automatic hyperparameter repair or failed-stage retry.

Frozen five protocols: COCO canonical, Urban1k, Flickr test1K, DOCCI test5K,
Long-DCI reconstructed7602. Native normalized vectors only. No DCI Full, masked
rerank, multi-seed, extra structure, expanded search or further promotions.
