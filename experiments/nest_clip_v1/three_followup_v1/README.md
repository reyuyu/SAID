# Balanced Directed Followups

Parent: 835852a, completed fixed-seed Balanced hyperparameter search.
First-two branch: codex/nest-balanced-three-followup-v1.
Four-epoch and unified-report branch: codex/nest-balanced-four-epoch-v1.
First-two runtime: /root/lk_projects/SAID-nest-clip-v1/three_followup_v1/.
Four-epoch runtime: /root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch/.

The completed four-epoch scores and per-dataset comparison are in
[FOUR_EPOCH_RESULTS.md](FOUR_EPOCH_RESULTS.md). All three experiments have finished.

The user's opening instruction extends both new coefficient experiments to
three full epochs instead of the 500-step-only budget in the pasted details.
Each runs a five-update smoke from shared step0, an independent formal 500-update
run, five native evaluations, and an exact same-trial continuation to 3651 with
another five native evaluations. Both proceed to 3651 regardless of screening
scores. Experiments run sequentially on the full four GPUs.

| Experiment | fusion_lr | visual scale | F/P/R weights | sparsity | inclusion max | Horizon |
|---|---:|---:|---|---:|---:|---:|
| Inclusion++ | 2e-4 | 1 | [1,1,1] | 1 | 2 | 3651 |
| Remainder++ | 2e-4 | 1 | [1,1,2] | 1 | 1.5 | 3651 |
| Four-epoch fusion-only | 2e-4 | 1 | [1,1,1] | 1 | 1 | 4868 |

The architecture, shared step0, data, seed0, sampler, global candidates, loss
definitions and precision are unchanged. Inclusion uses the existing 200-step
ramp. Remainder alignment is 2.5*LF + 2.5*LP + 5*LR.

The user confirmed experiment3 as four epochs. Its selected parent is the
completed original A/B/C fusion-only winner, selected by full3651 results:
fusion_lr=2e-4, inclusion_max=1, other defaults. It starts from shared step0
with horizon4868, stops/evaluates at3651, then restores that same four-epoch
checkpoint for the last epoch and evaluates at4868. It does not resume the
old three-epoch checkpoint or restart its LR. At3651 the four-epoch run has
the same training length as the old best but a different cosine trajectory.

The four-epoch source is isolated in /root/lk_projects/SAID-balanced-four-epoch-v1
so live first-two source/checkpoint hashes remain unchanged. Its queued supervisor
waits for the first-two supervisor's exclusive lock to be released at normal exit.
It copies compact first-two evidence to produce the unified final report here.

The existing training and strict export/evaluation helpers are reused with
instance-specific runtime/evidence paths. The completed search outputs stay in
their own runtime directory. The common resource gate already passed on this
unchanged computation; each new candidate still receives its own real4x256 smoke.
Failures and interrupted logs are preserved, without automatic retries.

```bash
# First two, in /root/lk_projects/SAID-balanced-three-followup-v1:
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --launch
# Third, in /root/lk_projects/SAID-balanced-four-epoch-v1:
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.four_epoch --prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.four_epoch --launch
```

State, full optimizer/RNG checkpoints and detailed logs remain server-local.
THREE_FOLLOWUP_REPORT.md contains current references, every completed evaluation,
Recall tables, coefficient effects, diagnostics, hashes and reproduction commands.
Each supervisor commits/pushes compact evidence to its independent branch after
each evaluation; large weights, data and caches are excluded. The final unified
report stops after exactly these three experiments, with no additional trials.
