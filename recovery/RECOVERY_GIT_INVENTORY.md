# SAID recovery Git inventory

Recovery root: `/opt/data/private/lklk/SAID`.

## Pinned sources

| Role | Remote branch | Pinned commit |
| --- | --- | --- |
| Canonical RandomK Balanced-Stack-Patch B/16, H4868 | `codex/nest-balanced-four-epoch-v1` | `14653c92c6da9d552a2b624ab169eaaa275cdde8` |
| Summary+RandomDetail dose sampling and weighting | `codex/nest-balanced-armb-summary-dose-500-v1` | `00c088fe83a9529c25c83316013d82aea24aa607` |
| Gradient audit reports only; do not execute | `codex/nest-balanced-gradient-audit-v1` | `246566fbe618daf36e54e809ea79e25515c0e3b7` |
| S=0.2 full-preflight and current research checkout | `codex/nest-balanced-armb-summary02-4epoch-v1` | `52bb7ae2d77aae8c2b1f69877aed5f7cc5d85b18` |

All four branch tips were checked through the GitHub API and the local Git object
database. The complete, non-shallow all-branch clone and `git fetch --all` finished.
`git fsck --full` passed. There are 69 remote refs including `origin/HEAD`; no
experimental branches were merged. The current research checkout is detached at
`52bb7ae2d77aae8c2b1f69877aed5f7cc5d85b18`; `worktrees/randomk` is detached at
`14653c92c6da9d552a2b624ab169eaaa275cdde8`.

The initial GitHub network route stalled. A temporary loopback CONNECT tunnel to
a reachable GitHub edge was used, retaining end-to-end TLS certificate checking.
The remote URL remains the original GitHub repository, with no global proxy
configuration or alternate mirror replacing the source.

## Code ownership and differences

Use the **formal pinned commit in a detached worktree** for canonical RandomK.
Use the **S=0.2 pinned commit** for Summary+RandomDetail. Do not merge experimental
branches and do not transplant audited gradient instrumentation into training.

Git blob identities from the official repository's recursive commit trees:

| File | Formal blob | Dose/preflight blob |
| --- | --- | --- |
| `model/nested_fusion_mask.py` | `7e4c55cc7614d8c392463f5d1bbad47a6ba0eede` | `9abcaa67723d94b01cfd0c2636606188c894c9fe` |
| `model/balanced_hparam_search.py` | `bdf3ab8ccda1515789e3194fe06cab807473d9b7` | `7c56cd3e74956ede85c459033b704a7e9dc04b69` |
| `train/nested_semantic_data.py` | `a185444f0720ec5156e23999eee0c6e5f5443393` | `b6383d72a37b789a5c062f9c0d2203ecbb48b068` |
| `train/train_nested_semantic_mask.py` | `7ed75a005a711c5bbf8208bff44de4fb822f1d1d` | `93a1b4e8066e1c6bbb344c2864cd5bded62500dc` |

The preflight and dose commits have identical blobs for these four files. The
gradient commit differs in `model/balanced_hparam_search.py`
(`02473af16a824e48e1b71cc2299608607e5c1096`), so it is not the selected training source.

All four commits share the following initializer/export/evaluation implementations:

- `tools/nest_clip.py`: `ea06e0c6602554dc4c2c73c577703d21f39808f1`.
- `tools/eval_nest_native.py`: `914c8e021b3b3a4917419f5b9dc6fd8ac3ad2c05`.
- `experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py`:
  `47facbb9b90b64eae637dad9a24d5bd5ed6ea92e`.

The context-248 construction must remain `model/longclip.py` plus repository model
code, not a new implementation. Original image preprocessing must remain the
repository's `reference_view_a_transform`.

## Historical reports

- Formal four-epoch reference: `experiments/nest_clip_v1/three_followup_v1/THREE_FOLLOWUP_REPORT.md`
  and its `RESULTS.json` / `evidence/6d44ae8d5c34/step4868/`.
- Dose research: `experiments/nest_clip_v1/armb_summary_dose_500_v1/arm_E1_summary02/RESULTS.json`
  and sibling config, commands and evidence.
- Current S=0.2 preflight: `experiments/nest_clip_v1/armb_summary02_4epoch_v1/`.
- Gradient audit: `experiments/nest_clip_v1/gradient_composition_audit_v1/GRADIENT_AUDIT_REPORT.md`.
- Other historical experiments remain in fetched Git history; they are not recovery workloads.

The S=0.2 historical `run.py` calls `preflight(); train(); evaluate()` and depends on
the lost step500 parent. **Do not execute that script during recovery.** Restoring
the sampling/configuration from step0 is not resuming the lost step500 trajectory.

## Safety boundary

No 500/4868-step training is authorized by this recovery. Recovery validation is
limited to CPU/unit tests, configuration/sampling audits, and a gated four-GPU
five-update smoke with the original H4868 schedule after assets are verified.
