# Fixed100k simulation before training

| View | Mean sentences | Mean effective tokens | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomK_P | 4.020993 | 83.496600 | 21.00 | 42.00 | 79.00 | 116.00 | 153.00 | 175.00 |
| RandomK_R | 4.036424 | 89.114324 | 25.00 | 47.00 | 83.00 | 123.00 | 162.00 | 184.00 |
| Summary | 1.000000 | 18.601862 | 12.00 | 14.00 | 18.00 | 22.00 | 26.00 | 28.00 |
| All_Detail_raw_previous | 7.158709 | 155.920480 | 101.00 | 119.00 | 152.00 | 190.00 | 219.00 | 236.00 |
| Random_Detail | 4.030694 | 88.516732 | 43.00 | 57.00 | 81.00 | 113.00 | 148.00 | 168.00 |
| All_Detail_visible_control | 7.057417 | 154.023123 | 100.00 | 119.00 | 152.00 | 190.00 | 213.00 | 221.00 |

Random Detail content-token coverage of visible detail: 57.455690%; raw-detail coverage: 56.877874%.
K mean=4.030694,median=4.0; K1 fallback fraction=0.0010001200144017282; K=m count for m>=3=0.

Full raw text/tokens match on all100k simulated records. Original Full packing/parser preserved; selected indices are strictly inside visibleF,exclude summary,sorted,no replacement,no reordering or PAD shifts.

Previous raw All-Detail and all-visible-detail control are both recorded to distinguish budget correction from subset selection. Exact histograms/quantiles/provenance are in SAMPLING_AUDIT.json.
