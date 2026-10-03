# Sampling audit before training

Complete indexed training corpus: 1245901 samples; skip1000 unchanged. Literal . split and cleanup reused; actual CLIP tokenizer,248 context.

| Metric | Matched RandomK | Summary–Detail |
|---|---:|---:|
| valid fraction (fixed100k) | 0.99988 | 0.99999 |
| P/S mean sentence segments | 4.020992519102292 | 1 |
| R/D mean sentence segments (fixed100k valid) | 4.036424370924511 | 7.158031580315803 |
| P/S mean tokens incl SOT/EOT | 83.49659959195104 | 18.602476024760247 |
| R/D mean tokens before truncation | 89.11432371884626 | 157.38638386383863 |
| Non-summary raw-detail coverage before truncation | 0.5704145933909043 | 1.0 |
| Effective detail coverage after truncation | 0.5704145933909043 | 0.9964532338625381 |
| Detail truncation fraction (whole corpus valid) | 0 | 0.029238794457567708 |

Coverage denominator is all raw non-summary CLIP BPE content tokens (excluding SOT/EOT). RandomK uses seed0,epoch0 and original sample IDs. Cases with no old local view contribute zero coverage, not an empty-caption loss. New coverage is1 before truncation; post-truncation coverage is measured independently.

The baseline packs Full into the longest complete-sentence prefix within248. Full remains unchanged. New Detail intentionally uses all raw remaining segments: 5.794058% of the fixed100k eligible cases extend beyond visible Full. Raw n>=2 validity changes 11 cases versus old visible-segment validity. Inclusion mathematics is unchanged, although semantic set inclusion does not imply post-budget token-set inclusion.

## Whole-corpus distributions

Sentence count:

```json
{
  "count": 1245901,
  "mean": 8.158626568242582,
  "p10": 6.0,
  "p25": 6.0,
  "p50": 8.0,
  "p75": 10.0,
  "p90": 11.0,
  "p95": 12.0,
  "median": 8.0
}
```

Summary tokens before truncation:

```json
{
  "count": 1245901,
  "mean": 18.62666375578798,
  "p50": 18.0,
  "p90": 26.0,
  "p95": 28.0,
  "p99": 34.0,
  "median": 18.0
}
```

Detail tokens before truncation:

```json
{
  "count": 1245879,
  "mean": 157.35960474492308,
  "p50": 152.0,
  "p90": 219.0,
  "p95": 236.0,
  "p99": 277.0,
  "median": 152.0
}
```

Raw Full tokens before packing:

```json
{
  "count": 1245901,
  "mean": 174.94568107738897,
  "p50": 169.0,
  "p90": 235.0,
  "p95": 252.0,
  "p99": 295.0,
  "median": 169.0
}
```

Single-sentence fraction: 1.7657903798134844e-05; two-sentence: 0.0001244079585777682; three-plus: 0.9998579341376241.

The fixed100k comparison also records actual packed Full lengths. F raw strings and tokens match1000/1000 randomly drawn real samples. SAMPLING_AUDIT.json and evidence/F_1000_EQUIVALENCE.json preserve complete statistics and fingerprints.
