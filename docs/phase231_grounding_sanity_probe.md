# Phase 2.3.1 Grounding Sanity Probe: initial D4 gate record

> Superseded by [the completed root-cause audit](phase231_root_cause.md). Decision D: no implementation coordinate bug found. The stop below records an earlier phase of investigation, not the current conclusion.

**Status: STOP — possible spatial mapping bug. No model inference or training was started in this phase.**

The requested immediate-stop condition was triggered by non-identity transforms substantially outperforming identity on the complete 14,334-phrase, 1,000-image Flickr30K Entities validation artifacts. This is an alert, not a confirmed root-cause diagnosis. No attention orientation has been changed in the exporter, evaluator or dashboard.

## Repository history

Phase 2.3 contained exactly these commits beyond its starting main (`20c1770`):

- `cdc8125d3fe60eba173e9949cd7879bc2808b520` — feat: add phrase-level semantic grounding audit

- `e1544a6e4adf062997f2741ebfb1cd8ca647503c` — feat: add semantic grounding dashboard views

- `746294a21a6b45be7254bbfc89cefcea5a8c1a81` — docs: report Phase 2.3 semantic grounding audit results

Preceding dashboard fixes are also retained in history: `d7c50f6` (fix: include dashboard images for every checkpoint) and `20c1770` (fix: make Said dashboard exports self-contained). There were no omitted intermediate fix commits inside the Phase 2.3 range.

The clean Phase 2.3 branch was fetched/pulled, fast-forward merged into main and pushed. Work then moved to `codex/phase2.3.1-grounding-sanity-probe`. This probe branch has not been merged into main.

## Protocol

Read each saved NPY once; no model inference. Keep the original GT geometry unchanged and apply each of the eight D4 symmetries to the attention array. Recompute exact union-coverage GT mass and argmax patch-center pointing. Positive rotation is counterclockwise. `transpose_plus_flip` means transpose plus **both** axis flips (anti-diagonal reflection); a transpose plus a single flip would duplicate a 90-degree rotation. Identity reproduces the Phase 2.3 aggregate metrics.

```bash
python -m eval.salu.grounding_sanity_probe \
  --root outputs/semantic_grounding \
  --output outputs/grounding_sanity_probe
```

Outputs: `spatial_sweep.json` and `spatial_sweep.csv` under the output root. The optional `--include_shifts` implements the specified no-wrap, renormalized 25-offset sweep, but **was not run** because the D4 stop condition had already been reached.

## Full D4 results

| Model | Transform | Pointing | GT mass | Mass gain | Pointing delta (pp) |
|---|---|---:|---:|---:|---:|
| initial_direct | identity | 23.37% | 0.300584 | -0.019051 | +0.00 |
| initial_direct | horizontal_flip | 27.88% | 0.312437 | -0.007198 | +4.51 |
| initial_direct | vertical_flip | 29.91% | 0.316068 | -0.003567 | +6.54 |
| initial_direct | transpose | 30.26% | 0.316710 | -0.002925 | +6.89 |
| initial_direct | rotate90 | 31.67% | 0.318877 | -0.000757 | +8.30 |
| initial_direct | rotate180 | 32.15% | 0.321799 | 0.002164 | +8.78 |
| initial_direct | rotate270 | 31.72% | 0.319010 | -0.000625 | +8.35 |
| initial_direct | transpose_plus_flip | 31.00% | 0.317542 | -0.002092 | +7.63 |
| phase21_router | identity | 25.67% | 0.294295 | -0.025340 | +0.00 |
| phase21_router | horizontal_flip | 28.98% | 0.306398 | -0.013236 | +3.31 |
| phase21_router | vertical_flip | 30.11% | 0.312214 | -0.007421 | +4.44 |
| phase21_router | transpose | 31.09% | 0.313399 | -0.006236 | +5.42 |
| phase21_router | rotate90 | 31.37% | 0.315338 | -0.004297 | +5.71 |
| phase21_router | rotate180 | 31.65% | 0.317938 | -0.001697 | +5.98 |
| phase21_router | rotate270 | 31.89% | 0.316278 | -0.003357 | +6.22 |
| phase21_router | transpose_plus_flip | 30.40% | 0.313982 | -0.005652 | +4.74 |
| phase22_router | identity | 29.65% | 0.291263 | -0.028372 | +0.00 |
| phase22_router | horizontal_flip | 31.55% | 0.309335 | -0.010300 | +1.90 |
| phase22_router | vertical_flip | 29.90% | 0.305733 | -0.013902 | +0.25 |
| phase22_router | transpose | 29.64% | 0.310397 | -0.009238 | -0.01 |
| phase22_router | rotate90 | 30.89% | 0.314246 | -0.005389 | +1.24 |
| phase22_router | rotate180 | 30.24% | 0.314444 | -0.005190 | +0.59 |
| phase22_router | rotate270 | 30.44% | 0.313583 | -0.006051 | +0.79 |
| phase22_router | transpose_plus_flip | 30.03% | 0.311841 | -0.007794 | +0.38 |
| phase22_direct | identity | 23.24% | 0.283930 | -0.035705 | +0.00 |
| phase22_direct | horizontal_flip | 29.15% | 0.304131 | -0.015504 | +5.91 |
| phase22_direct | vertical_flip | 27.79% | 0.309446 | -0.010189 | +4.56 |
| phase22_direct | transpose | 30.90% | 0.312595 | -0.007040 | +7.66 |
| phase22_direct | rotate90 | 30.61% | 0.315777 | -0.003858 | +7.37 |
| phase22_direct | rotate180 | 30.75% | 0.320310 | 0.000675 | +7.51 |
| phase22_direct | rotate270 | 31.66% | 0.316032 | -0.003602 | +8.42 |
| phase22_direct | transpose_plus_flip | 30.17% | 0.313606 | -0.006029 | +6.93 |

## Gate evidence

- Initial direct: identity 23.37%; rotate180 32.15% (**+8.78 pp**), mean GT mass improves by 0.0212.

- Phase 2.1 Router: identity 25.67%; rotate270 31.89% (**+6.22 pp**), mean GT mass improves by 0.0220.

- Phase 2.2 Router: identity 29.65%; horizontal_flip 31.55% (**+1.90 pp**), mean GT mass improves by 0.0181.

- Phase 2.2 direct: identity 23.24%; rotate270 31.66% (**+8.42 pp**), mean GT mass improves by 0.0321.

Several transformations improve scores; different variants have different best transforms. Thus these results do not identify a unique inversion or establish a mapping bug. They are also compatible with spatial bias or anti-localization in contextual patch scores. Post-hoc transformed scores must not replace the original benchmark results. Under the explicit user gate, this is sufficient to stop and request Review before further probes.

## Completed automatic spatial tests

- Eight distinct D4 symmetries, expected one-hot locations, and inverse rotation.

- Shift discards out-of-bounds mass, never wraps, and normalizes surviving mass; all-mass-lost shifts raise an error.

- End-to-end one-hot at row 5 / column 7: NPY save, grounding loader, metrics, existing dashboard renderer, GT box and peak cross. Pointing True, GT mass 1, mass gain 195/196, red center (120,88) inside green box [112,80,128,96]. NPY reload equality is exact.

The one-hot test was implemented and executed while D4 was running, before its STOP result was inspected. It validates the stored-grid-to-renderer path; it does not validate model patch order or original image preprocessing.

Full regression suite: **66 passed, 1 skipped, 3 warnings in 16.10s**. All prior 63 passing tests still pass; the distributed-test skip and existing warnings remain. `git diff --check` passes.

## Deferred by immediate STOP

Not executed: shift sweep, 100-phrase checkpoint recomputation, independent conv patch-order probe, prompt forms, sentence-context diagnostic, checkpoint trajectory, pre-softmax ranking correlation, and identity-router exactness. No checkpoints were loaded. The 3-epoch decision is **NO** because the required sanity gates have not passed.

The new Grounding Sanity Probe dashboard page is deferred by the immediate STOP instruction. The existing Semantic Grounding Audit dashboard is unchanged. CSV/JSON and this report expose the completed D4 table for review.

## Conclusion

## Review update: clarified denominators and root cause

The complete paired table and current conclusion are in [phase231_root_cause.md](phase231_root_cause.md). All-phrase metrics use 14,334 phrases; true/shuffled/excess metrics use the same 3,674 directed observations from 1,837 pairs. Existing numerical results are unchanged. The earlier claim of a confirmed global mapping issue is withdrawn: saved/live, crop, EXIF, patch order and identity-router checks pass; the 180-degree signal already exists in live pre-softmax patch scores. Formal heatmaps remain unrotated.
