# Balanced Directed Followups

Parent: 835852a, completed fixed-seed Balanced hyperparameter search.
Branch: codex/nest-balanced-three-followup-v1.
Runtime: /root/lk_projects/SAID-nest-clip-v1/three_followup_v1/.

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

The architecture, shared step0, data, seed0, sampler, global candidates, loss
definitions and precision are unchanged. Inclusion uses the existing 200-step
ramp. Remainder alignment is 2.5*LF + 2.5*LP + 5*LR.

Experiment 3 is pending clarification: the opening instruction says all three
experiments use three epochs, but its detailed specification requires a fresh
four-epoch run with horizon4868 and evaluations at3651/4868. No experiment3
training is scheduled until its budget is resolved. Its selected parent is the
completed fusion-only winner (fusion_lr=2e-4, inclusion_max=1, other defaults).

The existing training and strict export/evaluation helpers are reused with
instance-specific runtime/evidence paths. The completed search outputs stay in
their own runtime directory. The common resource gate already passed on this
unchanged computation; each new candidate still receives its own real4x256 smoke.
Failures and interrupted logs are preserved, without automatic retries.

```bash
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --launch
```

State, full optimizer/RNG checkpoints and detailed logs remain server-local.
THREE_FOLLOWUP_REPORT.md contains current references, every completed evaluation,
Recall tables, coefficient effects, diagnostics, hashes and reproduction commands.
The supervisor commits/pushes compact evidence to this independent branch after
each evaluation; large weights, data and caches are excluded. It stops after
the two unambiguous experiments and records the unresolved experiment3 budget.
