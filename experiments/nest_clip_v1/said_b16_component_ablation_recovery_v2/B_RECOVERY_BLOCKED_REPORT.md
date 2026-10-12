# B recovery audit: blocked by parent model-source identity

The component-ablation recovery controller was isolated on
`experiment/said-b16-component-ablation-recovery-v2` and committed as
`3dd2e1ab8f4998b78a1ba2a55f99fad3e005ce92`. Its only production change is the
`component_ablation` resource policy: the historical 3-second boundary emits a
warning, a single complete update above 60 seconds stops, and existing memory,
nonfinite, DDP, and data-error stops remain active. Six CPU tests pass.

The independent four-rank smoke completed five updates with synchronized
parameters (`max_parameter_difference=0`), NCCL all-reduce result 10, and
13.252 GiB peak allocated memory per rank. The first full cycle was 31.47 s and
was recorded as a warning; no hard failure occurred.

The parent Step2261 checkpoint is readable and internally complete. Its SHA256
is `1b0e6e0566e2de0845ababb7876aa186a2e2c0cb19cac20cf9c94cbed2cdb3ba`; it
contains 2261 optimizer steps, horizon 4868, target 3651, epoch 1, batch 1044,
four rank RNG states, loader generator state, and finite model/adapter/optimizer
tensors. Step1217 is also readable and finite with the expected SHA256
`4edecc441f3ba593c305daca530268dd892ce42d73e4e6fb643b9deddc4b124f`.

Formal recovery was intentionally stopped before update 2262. The parent
manifest records `model/nested_fusion_mask.py` as
`87e7ac3a77f4ff20f89a9a551aaeba689d0d67e9d817fe05cb16544e61617bfb`, while the
only available B source is `5503cfca470b7ae35e59783ffe0c04fae35fc29f3cb9dee82fe0203ddbc70562`.
All other model, component, data, and environment manifest entries match; the
trainer mismatch is the explicitly recorded controller-only change. The exact
parent model source is absent from the current Git objects, remote branch, and
worktrees. Because the protocol requires all non-controller source hashes to
match, continuing would require disabling a required identity check and is
therefore disallowed.

C and D were not started. The original failed run, its checkpoints, logs, and
reports remain untouched.
