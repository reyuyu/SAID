# Random Caption Completeness & First-Sentence Dependency

Inference-only analysis on fixed 1,000-pair DOCCI and Long-DCI samples. Historical Full, Prefix50 and First Sentence features were reused; only random/crop views were newly inferred. All reported vectors are original 512-dimensional L2-normalized features.

## Protocol and audit

- Random Distributed and Random Contiguous Crop were generated independently for seeds 0–4. The same generated text and sample order was used for all three models.
- NoFirst-Matched is a post-first-sentence contiguous crop matched to First Sentence BPE length. Invalid rows were excluded explicitly: DOCCI 980/1000 and Long-DCI 937/1000.
- Every point estimate has a paired-sample percentile bootstrap with 2,000 resamples; random-view seed variation is summarized by mean and sample standard deviation. Seeds are not independent examples.
- Mask keep ratios and KMeans diagnostics are reported only for Sema-CLIP Masked. Checkpoint, source and code hashes are in `RANDOM_CAPTION_PROVENANCE.json`.
- The historical DOCCI First Sentence feature extractor has three abbreviation/quoted-punctuation rows whose tokenization differs from the robust sampler (indices 96, 816, 876). Historical features remain the analysis reference; the discrepancy is recorded rather than silently replaced.

## Main quantitative results

Random rows are means over five seeds; historical rows are fixed 1,000-pair values. `Gap` is Euclidean centroid gap, `Ang` is 1 − centroid-direction cosine, and `PCos` is paired cosine.

### DOCCI

| Model | View | N | Gap | Gap SD | Ang | Text norm | PCos | PCos SD | Mask keep |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long-CLIP | full | 1000 | 1.0490 | 0.0000 | 0.6983 | 0.9428 | 0.3086 | 0.0000 | — |
| Long-CLIP | prefix50 | 1000 | 1.0357 | 0.0000 | 0.6974 | 0.9236 | 0.3098 | 0.0000 | — |
| Long-CLIP | first_sentence | 1000 | 0.9648 | 0.0000 | 0.6907 | 0.8166 | 0.3198 | 0.0000 | — |
| Long-CLIP | random_distributed_random50 | 1000 | 1.0395 | 0.0020 | 0.6976 | 0.9292 | 0.2994 | 0.0018 | — |
| Long-CLIP | random_contiguous_random50 | 1000 | 1.0433 | 0.0021 | 0.6984 | 0.9340 | 0.2960 | 0.0038 | — |
| Long-CLIP | random_distributed_random10 | 1000 | 1.0131 | 0.0009 | 0.6999 | 0.8853 | 0.2634 | 0.0014 | — |
| Long-CLIP | random_contiguous_random10 | 1000 | 0.9987 | 0.0177 | 0.6982 | 0.8640 | 0.2725 | 0.0172 | — |
| Long-CLIP | nofirst_matched | 980 | 1.0142 | 0.0009 | 0.6976 | 0.8899 | 0.2754 | 0.0043 | — |
| Sema-CLIP Native | full | 1000 | 0.7784 | 0.0000 | 0.8008 | 0.4812 | 0.3717 | 0.0000 | — |
| Sema-CLIP Native | prefix50 | 1000 | 0.7638 | 0.0000 | 0.7799 | 0.4683 | 0.3535 | 0.0000 | — |
| Sema-CLIP Native | first_sentence | 1000 | 0.7917 | 0.0000 | 0.7591 | 0.5532 | 0.3270 | 0.0000 | — |
| Sema-CLIP Native | random_distributed_random50 | 1000 | 0.7704 | 0.0016 | 0.7713 | 0.4934 | 0.3351 | 0.0031 | — |
| Sema-CLIP Native | random_contiguous_random50 | 1000 | 0.7722 | 0.0032 | 0.7743 | 0.4943 | 0.3281 | 0.0077 | — |
| Sema-CLIP Native | random_distributed_random10 | 1000 | 0.9094 | 0.0017 | 0.7525 | 0.7661 | 0.2320 | 0.0022 | — |
| Sema-CLIP Native | random_contiguous_random10 | 1000 | 0.8868 | 0.0277 | 0.7531 | 0.7289 | 0.2460 | 0.0260 | — |
| Sema-CLIP Native | nofirst_matched | 980 | 0.8617 | 0.0013 | 0.7545 | 0.6877 | 0.2575 | 0.0067 | — |
| Sema-CLIP Masked | full | 1000 | 0.6864 | 0.0000 | 0.7642 | 0.4812 | 0.3325 | 0.0000 | 0.6116 |
| Sema-CLIP Masked | prefix50 | 1000 | 0.7037 | 0.0000 | 0.7397 | 0.4683 | 0.3125 | 0.0000 | 0.5980 |
| Sema-CLIP Masked | first_sentence | 1000 | 0.7728 | 0.0000 | 0.7223 | 0.5532 | 0.2943 | 0.0000 | 0.6083 |
| Sema-CLIP Masked | random_distributed_random50 | 1000 | 0.7267 | 0.0035 | 0.7355 | 0.4934 | 0.2993 | 0.0027 | 0.6046 |
| Sema-CLIP Masked | random_contiguous_random50 | 1000 | 0.7230 | 0.0106 | 0.7395 | 0.4943 | 0.2939 | 0.0048 | 0.6049 |
| Sema-CLIP Masked | random_distributed_random10 | 1000 | 0.9039 | 0.0017 | 0.7276 | 0.7661 | 0.2293 | 0.0015 | 0.6098 |
| Sema-CLIP Masked | random_contiguous_random10 | 1000 | 0.8820 | 0.0311 | 0.7283 | 0.7289 | 0.2389 | 0.0176 | 0.6096 |
| Sema-CLIP Masked | nofirst_matched | 980 | 0.8588 | 0.0023 | 0.7298 | 0.6877 | 0.2458 | 0.0045 | 0.6101 |

First Sentence vs NoFirst-Matched on the exact common valid subset:

| Model | First gap | NoFirst gap | First PCos | NoFirst PCos |
|---|---:|---:|---:|---:|
| Long-CLIP | 0.9642 | 1.0142 | 0.3198 | 0.2754 |
| Sema-CLIP Native | 0.7924 | 0.8617 | 0.3267 | 0.2575 |
| Sema-CLIP Masked | 0.7741 | 0.8588 | 0.2942 | 0.2458 |

### LONG_DCI

| Model | View | N | Gap | Gap SD | Ang | Text norm | PCos | PCos SD | Mask keep |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long-CLIP | full | 1000 | 1.0281 | 0.0000 | 0.6943 | 0.9320 | 0.2942 | 0.0000 | — |
| Long-CLIP | prefix50 | 1000 | 1.0193 | 0.0000 | 0.6950 | 0.9181 | 0.2905 | 0.0000 | — |
| Long-CLIP | first_sentence | 1000 | 0.9750 | 0.0000 | 0.6976 | 0.8459 | 0.2859 | 0.0000 | — |
| Long-CLIP | random_distributed_random50 | 1000 | 1.0255 | 0.0011 | 0.6971 | 0.9250 | 0.2780 | 0.0018 | — |
| Long-CLIP | random_contiguous_random50 | 1000 | 1.0244 | 0.0007 | 0.6960 | 0.9246 | 0.2776 | 0.0033 | — |
| Long-CLIP | random_distributed_random10 | 1000 | 1.0087 | 0.0011 | 0.7021 | 0.8940 | 0.2454 | 0.0009 | — |
| Long-CLIP | random_contiguous_random10 | 1000 | 0.9992 | 0.0052 | 0.7007 | 0.8808 | 0.2530 | 0.0137 | — |
| Long-CLIP | nofirst_matched | 937 | 0.9999 | 0.0010 | 0.7005 | 0.8820 | 0.2526 | 0.0042 | — |
| Sema-CLIP Native | full | 1000 | 0.7895 | 0.0000 | 0.8610 | 0.4934 | 0.3104 | 0.0000 | — |
| Sema-CLIP Native | prefix50 | 1000 | 0.7869 | 0.0000 | 0.8289 | 0.5162 | 0.2914 | 0.0000 | — |
| Sema-CLIP Native | first_sentence | 1000 | 0.8629 | 0.0000 | 0.7805 | 0.6925 | 0.2632 | 0.0000 | — |
| Sema-CLIP Native | random_distributed_random50 | 1000 | 0.7940 | 0.0019 | 0.8201 | 0.5378 | 0.2736 | 0.0029 | — |
| Sema-CLIP Native | random_contiguous_random50 | 1000 | 0.7916 | 0.0005 | 0.8205 | 0.5329 | 0.2670 | 0.0078 | — |
| Sema-CLIP Native | random_distributed_random10 | 1000 | 0.9205 | 0.0014 | 0.7725 | 0.7872 | 0.1974 | 0.0013 | — |
| Sema-CLIP Native | random_contiguous_random10 | 1000 | 0.9063 | 0.0082 | 0.7702 | 0.7681 | 0.2104 | 0.0209 | — |
| Sema-CLIP Native | nofirst_matched | 937 | 0.8919 | 0.0025 | 0.7724 | 0.7443 | 0.2119 | 0.0075 | — |
| Sema-CLIP Masked | full | 1000 | 0.7142 | 0.0000 | 0.8387 | 0.4934 | 0.2879 | 0.0000 | 0.6482 |
| Sema-CLIP Masked | prefix50 | 1000 | 0.7372 | 0.0000 | 0.7922 | 0.5162 | 0.2709 | 0.0000 | 0.6284 |
| Sema-CLIP Masked | first_sentence | 1000 | 0.8446 | 0.0000 | 0.7463 | 0.6925 | 0.2536 | 0.0000 | 0.6122 |
| Sema-CLIP Masked | random_distributed_random50 | 1000 | 0.7594 | 0.0027 | 0.7875 | 0.5378 | 0.2551 | 0.0021 | 0.6315 |
| Sema-CLIP Masked | random_contiguous_random50 | 1000 | 0.7541 | 0.0034 | 0.7890 | 0.5329 | 0.2491 | 0.0067 | 0.6295 |
| Sema-CLIP Masked | random_distributed_random10 | 1000 | 0.9010 | 0.0015 | 0.7382 | 0.7872 | 0.2046 | 0.0011 | 0.6181 |
| Sema-CLIP Masked | random_contiguous_random10 | 1000 | 0.8922 | 0.0105 | 0.7395 | 0.7681 | 0.2141 | 0.0163 | 0.6145 |
| Sema-CLIP Masked | nofirst_matched | 937 | 0.8800 | 0.0042 | 0.7436 | 0.7443 | 0.2136 | 0.0054 | 0.6181 |

First Sentence vs NoFirst-Matched on the exact common valid subset:

| Model | First gap | NoFirst gap | First PCos | NoFirst PCos |
|---|---:|---:|---:|---:|
| Long-CLIP | 0.9753 | 0.9999 | 0.2855 | 0.2526 |
| Sema-CLIP Native | 0.8671 | 0.8919 | 0.2621 | 0.2119 |
| Sema-CLIP Masked | 0.8499 | 0.8800 | 0.2528 | 0.2136 |

## Interpretation

- **Prefix50 versus Random50:** replacing the prefix with distributed random content does not reproduce a large additional Long-CLIP gap reduction. Random50 remains close to Prefix50, while Masked Random50 is close to (and on DOCCI slightly above) Prefix50.
- **Random10:** roughly 10% content lowers paired cosine for all models and increases the Masked gap substantially. Long-CLIP gap changes are smaller than the paired-cosine loss, so Euclidean gap alone is not a semantic matching measure.
- **First Sentence versus NoFirst-Matched:** on the common subset, First Sentence has higher paired cosine and lower gap than NoFirst-Matched for all models and both datasets. This supports a first-sentence information advantage at matched length, without proving that the whole Prefix effect is caused by the first sentence.
- **Distributed versus contiguous:** the two randomization strategies are close but not identical. Contiguous crops generally show lower paired cosine and different norm behavior, making linguistic continuity a measurable nuisance factor.
- **Norm/angular decomposition:** each row includes `(r_image − r_text)^2` and `2 r_image r_text (1 − cosθ)` contributions and their fractions. Gap changes must therefore be interpreted together with text-centroid norm and angular gap.
- **Dataset dependence:** DOCCI and Long-DCI agree on the direction of First-vs-NoFirst paired-cosine and Random10 degradation, but differ in absolute gaps and mask ratios. This supports a recurring pattern with dataset-dependent magnitude, not a universal causal law.

## Cluster diagnostics and limitations

`RANDOM_CAPTION_CLUSTER_DIAGNOSTICS.csv` contains Masked-image KMeans k=2 silhouette, cluster sizes, minor fraction and center distance for Full, Prefix50, First Sentence, both random strategies and NoFirst-Matched. `RANDOM_CAPTION_CLUSTER_TRANSITIONS.json` contains ARI comparisons only where sample sets are identical. PCA figures use fixed Shared PCA per dataset/figure and seed 0 for random layouts; they are descriptive only.

No training, checkpoint modification, reranking or test-time selection was performed. Public benchmark reuse and the fixed single-seed sample protocol make these exploratory mechanism results rather than unbiased generalization evidence.
