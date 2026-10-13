# Phase 2 report: DOCCI modality gap

## Protocol

The analysis uses 1,000 pairs sampled with `seed=0` from the frozen DOCCI test manifest. The same image-caption pairs are used for every model and caption setting. Caption prefixes are cut by the Long-CLIP BPE tokenization rule, retaining the required SOT/EOT tokens. All reported metrics use L2-normalized 512-dimensional features. The six main panels share one PCA fit; the Ours-native control is projected through that same transform.

## Observations

Long-CLIP's centroid gap is 1.048990 (Full), 1.035737 (50%), and 1.008667 (30%); it does not increase as the prefix is shortened in this sample. Ours masked has a lower gap at every setting—0.686359, 0.703691, and 0.734860—while its paired cosine falls from 0.332476 to 0.302094. Ours native has gaps 0.778371, 0.763776, and 0.770465, so the caption-conditioned masked feature is closer to the text centroid than the no-mask control by 0.092012, 0.060085, and 0.035605 respectively.

The mean mask keep ratio is 0.611574 (Full), 0.597969 (50%), and 0.601322 (30%). This is a mechanism observation, not an accuracy claim. The native control has higher paired cosine than the masked feature for all three settings in this probe, despite the masked feature's smaller centroid gap.

## Answers to the requested questions

1. On this fixed DOCCI sample, Long-CLIP's gap does not significantly increase from Full to 50% or 30%; it decreases slightly. This is evidence against assuming the expected trend without measuring it.
2. Ours masked has a smaller centroid gap than Long-CLIP in all three settings (about 34.6%, 32.1%, and 27.1% lower relative to Long-CLIP).
3. Ours native has a larger centroid gap than Ours masked by 0.0920 (Full), 0.0601 (50%), and 0.0356 (30%). Its paired cosine is higher, so centroid gap and paired alignment capture different aspects of the geometry.
4. The results are consistent with selective semantic masking reducing the image-text centroid separation for this checkpoint and sample, but they do not by themselves establish a general Information Imbalance or Modality Gap claim. The opposite trend in Long-CLIP, the native-versus-masked cosine tradeoff, one checkpoint, one seed, and 1,000 pairs are material limits.

No model was trained and no checkpoint was modified. See `provenance.json`, `ARTIFACT_SHA256SUMS.txt`, and `modality_gap_metrics.csv` for reproducibility details.
