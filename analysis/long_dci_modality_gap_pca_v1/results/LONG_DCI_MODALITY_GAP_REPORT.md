# Long-DCI Modality Gap PCA

This is a fixed seed=0, 1,000-pair, inference-only analysis using the reconstructed Long-DCI manifest. No training or checkpoint modification was performed.

## Caption audit

The manifest contains 7,602 unique image IDs and all image files are present. First-sentence parsing had 8 explicit no-boundary fallbacks (0.800%); these rows are listed in `samples.json`. Mean effective BPE lengths are Full=135.39, 50% Prefix=67.93, First Sentence=17.31.

## 512-D metrics

| Model | Caption | Gap | Paired Cosine | Paired L2 | Image centroid norm | Text centroid norm | Mask keep |
|---|---|---:|---:|---:|---:|---:|---:|
| longclip | full | 1.028106 [1.026199, 1.030310] | 0.294222 [0.292931, 0.295466] | 1.187966 [1.186916, 1.189053] | 0.804055 [0.802368, 0.806066] | 0.932028 [0.929799, 0.934183] | — |
| longclip | prefix50 | 1.019346 [1.017263, 1.021904] | 0.290479 [0.288948, 0.292001] | 1.191064 [1.189786, 1.192350] | 0.804055 [0.802368, 0.806066] | 0.918113 [0.915555, 0.920789] | — |
| longclip | first_sentence | 0.975039 [0.972647, 0.977894] | 0.285892 [0.283839, 0.287937] | 1.194751 [1.193044, 1.196455] | 0.804055 [0.802368, 0.806066] | 0.845892 [0.842808, 0.849190] | — |
| ours_masked | full | 0.714181 [0.707803, 0.720847] | 0.287937 [0.284598, 0.291130] | 1.192602 [1.189960, 1.195344] | 0.602069 [0.593970, 0.610612] | 0.493364 [0.487759, 0.499717] | 0.648221 [0.644705, 0.651582] |
| ours_masked | prefix50 | 0.737163 [0.730988, 0.743978] | 0.270936 [0.267738, 0.274114] | 1.206766 [1.204122, 1.209399] | 0.644354 [0.637394, 0.651758] | 0.516170 [0.510011, 0.523413] | 0.628367 [0.625365, 0.631092] |
| ours_masked | first_sentence | 0.844607 [0.837689, 0.852385] | 0.253573 [0.250753, 0.256259] | 1.221278 [1.219065, 1.223542] | 0.690083 [0.685289, 0.695475] | 0.692526 [0.684316, 0.701397] | 0.612191 [0.609568, 0.614643] |
| ours_native | full | 0.789548 [0.784735, 0.795267] | 0.310413 [0.306687, 0.314070] | 1.173342 [1.170233, 1.176471] | 0.688816 [0.684350, 0.693741] | 0.493364 [0.487759, 0.499717] | — |
| ours_native | prefix50 | 0.786929 [0.782184, 0.792689] | 0.291431 [0.287582, 0.295124] | 1.189325 [1.186225, 1.192576] | 0.688816 [0.684350, 0.693741] | 0.516170 [0.510011, 0.523413] | — |
| ours_native | first_sentence | 0.862906 [0.856786, 0.869765] | 0.263233 [0.259804, 0.266658] | 1.213007 [1.210157, 1.215838] | 0.688816 [0.684350, 0.693741] | 0.692526 [0.684316, 0.701397] | — |

## KMeans masked-image diagnosis

| Caption | Major/Minor | Silhouette | Center distance |
|---|---:|---:|---:|
| full | 883/117 | 0.155965 | 0.555190 |
| prefix50 | 513/487 | 0.047800 | 0.337889 |
| first_sentence | 541/459 | 0.064008 | 0.350457 |

## Long-DCI versus DOCCI

| Dataset | Model | Full Gap | 50% Gap | First Sentence Gap | Full Cosine | 50% Cosine | First Sentence Cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| DOCCI / Long-DCI | Sema-CLIP Masked | 0.6864 / 0.7142 | 0.7037 / 0.7372 | 0.7728 / 0.8446 | 0.3325 / 0.2879 | 0.3125 / 0.2709 | 0.2943 / 0.2536 |
| DOCCI / Long-DCI | Sema-CLIP Native | 0.7784 / 0.7895 | 0.7638 / 0.7869 | 0.7917 / 0.8629 | 0.3717 / 0.3104 | 0.3535 / 0.2914 | 0.3270 / 0.2632 |

## Answers to the study questions

- Masked has a smaller Long-DCI Centroid Gap than Native in all three settings: Full 0.7142 vs 0.7895, 50% 0.7372 vs 0.7869, First Sentence 0.8446 vs 0.8629.
- The Gap and Paired Cosine trends agree within Long-DCI Masked across caption shortening (Gap rises from 0.7142 to 0.8446, while cosine falls from 0.2879 to 0.2536), but Masked versus Native is not uniformly aligned: Masked has the smaller Gap yet also the lower cosine.
- The Masked-image double-cluster diagnosis partially reproduces the DOCCI behavior. Long-DCI Full has silhouette 0.1560 and a 883/117 split; the 50% and First Sentence silhouettes are weaker (0.0478, 0.0640). This is dataset- and caption-dependent, not a universal separation.
- Long-DCI is useful as a complementary cross-dataset figure because it exposes this dependence, but it does not uniformly supersede DOCCI as the single modality-gap main figure. Shared PCA is the cross-model reference; per-model PCA remains diagnostic only.

## Interpretation

The PCA figures use one Shared PCA for the Long-CLIP and Sema-CLIP Masked rows, plus a Native 1×3 control and per-model PCA controls. No points were removed or shifted, and PCA coordinates are not used as modality-gap values.
