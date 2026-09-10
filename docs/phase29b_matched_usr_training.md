# Phase 2.9B: matched USR training study (100-step feasibility)

Protocol `sharegpt4v1k-usr-v1`: Q = 868 queries, candidate pool = 868 (candidate j is the
withheld suffix target of query j). Every overall evaluation and every subgroup keeps the
full 868-candidate pool; subgroups only select query rows (`scores[subset, :]` with the
original labels). Duplicate suffix sentences are not handled (per instruction).

> **Old Phase 2.9B.2 subgroup numbers are INVALID — candidate-pool-subsetting bug
> (`scores[subset][:, subset]` shrank the candidate pool). They are not referenced here.**

## Matched initialization / streams

- `initial_state_sha256` = `4621b8d027f482ea...` for rank 0-3 of all three arms (12/12, measured).
- sampler / prefix / full / unsaid / has_unsaid stream digests equal across arms per rank;
  rank0 every-10-step batch digests (image/prefix/full/unsaid/has_unsaid): mismatch = 0.
- shared actual initial checkpoint = `runs_salu/phase29b3_GS/salu_initial.pt` (no `initial:` / `fresh_untracked`).

## Training trajectory (training-side; not a substitute for offline USR)

| Arm | Step | LG | LS | LU | LT | route top1 | evid top1 | U top1 i2t/t2i | U margin | cov raw/gated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GS | 0 | 0.5173 | 3.3339 | 0.0000 | 3.8512 | 0.0039 | 0.7461 | - | - | - |
| GS | 20 | 0.2488 | 3.1916 | 0.0000 | 3.4404 | 0.0078 | 0.8047 | - | - | - |
| GS | 50 | 0.2338 | 3.0775 | 0.0000 | 3.3112 | 0.0117 | 0.8359 | - | - | - |
| GS | 100 (final) | 0.1454 | 2.4206 | 0.0000 | 2.5660 | 0.1797 | 0.8789 | - | - | - |
| GSU_Raw | 0 | 0.5173 | 3.3339 | 3.9652 | 7.8164 | 0.0039 | 0.7461 | 0.1570/0.3004 | 3.958 | 0.5088/0.5088 |
| GSU_Raw | 20 | 0.2941 | 3.2150 | 3.7334 | 7.2425 | 0.0000 | 0.8125 | 0.1858/0.4204 | 4.261 | 0.5166/0.5166 |
| GSU_Raw | 50 | 0.2893 | 3.1396 | 3.5361 | 6.9651 | 0.0156 | 0.8164 | 0.2500/0.3571 | 4.269 | 0.5431/0.5431 |
| GSU_Raw | 100 (final) | 0.1745 | 2.8357 | 2.4961 | 5.5063 | 0.0547 | 0.8945 | 0.4670/0.4626 | 6.197 | 0.6199/0.6199 |
| GSU_Debiased | 0 | 0.5173 | 3.3339 | 3.9680 | 7.8192 | 0.0039 | 0.7461 | 0.1614/0.3004 | 3.905 | 0.5088/0.4396 |
| GSU_Debiased | 20 | 0.2951 | 3.2186 | 3.7713 | 7.2850 | 0.0039 | 0.8086 | 0.1770/0.4071 | 4.160 | 0.5178/0.4506 |
| GSU_Debiased | 50 | 0.2903 | 3.1454 | 3.5891 | 7.0248 | 0.0117 | 0.8086 | 0.2455/0.3393 | 4.094 | 0.5489/0.4862 |
| GSU_Debiased | 100 (final) | 0.1740 | 2.8540 | 2.5514 | 5.5794 | 0.0586 | 0.8984 | 0.4626/0.4581 | 6.004 | 0.6402/0.5702 |

## Complete offline USR table (10 model states x 4 scorers)

| Train Arm | Step | Scorer | R@1 | R@5 | R@10 | MRR | Mean Rank | Median Rank | candidate pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared initial | 0 | global | 0.2650 | 0.4482 | 0.5265 | 0.3514 | 84.1 | 8.0 | 868 |
| shared initial | 0 | anti_said | 0.0553 | 0.1279 | 0.1728 | 0.0997 | 190.5 | 128.0 | 868 |
| shared initial | 0 | raw | 0.0622 | 0.1406 | 0.1970 | 0.1108 | 181.6 | 111.0 | 868 |
| shared initial | 0 | debiased | 0.0588 | 0.1394 | 0.1912 | 0.1079 | 184.3 | 121.0 | 868 |
| GS | 20 | global | 0.2650 | 0.4493 | 0.5300 | 0.3522 | 82.8 | 8.0 | 868 |
| GS | 20 | anti_said | 0.0541 | 0.1233 | 0.1682 | 0.0965 | 192.9 | 129.0 | 868 |
| GS | 20 | raw | 0.0611 | 0.1429 | 0.1982 | 0.1097 | 180.2 | 111.0 | 868 |
| GS | 20 | debiased | 0.0611 | 0.1290 | 0.1878 | 0.1065 | 184.9 | 119.0 | 868 |
| GS | 50 | global | 0.2776 | 0.4620 | 0.5334 | 0.3639 | 77.5 | 8.0 | 868 |
| GS | 50 | anti_said | 0.0438 | 0.1014 | 0.1336 | 0.0810 | 203.5 | 144.0 | 868 |
| GS | 50 | raw | 0.0634 | 0.1371 | 0.1797 | 0.1067 | 178.9 | 122.0 | 868 |
| GS | 50 | debiased | 0.0553 | 0.1175 | 0.1636 | 0.0941 | 188.6 | 129.0 | 868 |
| GS | 100 | global | 0.2857 | 0.4758 | 0.5461 | 0.3753 | 71.8 | 7.0 | 868 |
| GS | 100 | anti_said | 0.0426 | 0.0806 | 0.1025 | 0.0689 | 216.9 | 164.0 | 868 |
| GS | 100 | raw | 0.0461 | 0.0899 | 0.1233 | 0.0758 | 186.0 | 141.0 | 868 |
| GS | 100 | debiased | 0.0449 | 0.0841 | 0.1118 | 0.0736 | 192.7 | 142.0 | 868 |
| GSU_Raw | 20 | global | 0.2684 | 0.4528 | 0.5334 | 0.3541 | 83.6 | 8.0 | 868 |
| GSU_Raw | 20 | anti_said | 0.0599 | 0.1382 | 0.1843 | 0.1062 | 186.8 | 120.0 | 868 |
| GSU_Raw | 20 | raw | 0.0783 | 0.1682 | 0.2327 | 0.1303 | 171.7 | 100.0 | 868 |
| GSU_Raw | 20 | debiased | 0.0668 | 0.1578 | 0.2189 | 0.1216 | 175.8 | 107.0 | 868 |
| GSU_Raw | 50 | global | 0.2730 | 0.4631 | 0.5461 | 0.3627 | 81.8 | 7.0 | 868 |
| GSU_Raw | 50 | anti_said | 0.0703 | 0.1728 | 0.2224 | 0.1253 | 171.1 | 91.0 | 868 |
| GSU_Raw | 50 | raw | 0.1290 | 0.2546 | 0.3203 | 0.1976 | 131.9 | 49.0 | 868 |
| GSU_Raw | 50 | debiased | 0.1083 | 0.2431 | 0.3053 | 0.1802 | 138.5 | 60.0 | 868 |
| GSU_Raw | 100 | global | 0.2926 | 0.4885 | 0.5703 | 0.3855 | 73.3 | 6.0 | 868 |
| GSU_Raw | 100 | anti_said | 0.0864 | 0.1947 | 0.2523 | 0.1452 | 154.5 | 76.0 | 868 |
| GSU_Raw | 100 | raw | 0.2085 | 0.3952 | 0.4620 | 0.2978 | 83.1 | 14.0 | 868 |
| GSU_Raw | 100 | debiased | 0.1912 | 0.3779 | 0.4539 | 0.2807 | 87.4 | 16.0 | 868 |
| GSU_Debiased | 20 | global | 0.2684 | 0.4528 | 0.5323 | 0.3542 | 83.6 | 8.0 | 868 |
| GSU_Debiased | 20 | anti_said | 0.0588 | 0.1359 | 0.1820 | 0.1051 | 187.3 | 120.0 | 868 |
| GSU_Debiased | 20 | raw | 0.0783 | 0.1705 | 0.2327 | 0.1310 | 171.6 | 99.0 | 868 |
| GSU_Debiased | 20 | debiased | 0.0703 | 0.1590 | 0.2200 | 0.1237 | 175.3 | 106.0 | 868 |
| GSU_Debiased | 50 | global | 0.2696 | 0.4597 | 0.5472 | 0.3606 | 82.3 | 7.0 | 868 |
| GSU_Debiased | 50 | anti_said | 0.0668 | 0.1636 | 0.2131 | 0.1186 | 174.6 | 94.0 | 868 |
| GSU_Debiased | 50 | raw | 0.1313 | 0.2558 | 0.3341 | 0.1999 | 131.5 | 46.0 | 868 |
| GSU_Debiased | 50 | debiased | 0.1106 | 0.2500 | 0.3076 | 0.1848 | 137.5 | 56.0 | 868 |
| GSU_Debiased | 100 | global | 0.2926 | 0.4850 | 0.5726 | 0.3853 | 74.3 | 6.0 | 868 |
| GSU_Debiased | 100 | anti_said | 0.0726 | 0.1763 | 0.2339 | 0.1307 | 163.0 | 83.0 | 868 |
| GSU_Debiased | 100 | raw | 0.2131 | 0.3975 | 0.4712 | 0.3019 | 85.8 | 14.0 | 868 |
| GSU_Debiased | 100 | debiased | 0.2051 | 0.3917 | 0.4516 | 0.2932 | 87.2 | 15.0 | 868 |

## Effect A: suffix supervision (GSU-Raw - GS)

| Step | Scorer | dR@1 | dR@5 | dR@10 | dMRR | d mean rank |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | raw | +0.0173 | +0.0253 | +0.0346 | +0.0206 | -8.6 |
| 20 | debiased | +0.0058 | +0.0288 | +0.0311 | +0.0151 | -9.0 |
| 50 | raw | +0.0657 | +0.1175 | +0.1406 | +0.0909 | -47.0 |
| 50 | debiased | +0.0530 | +0.1256 | +0.1417 | +0.0861 | -50.1 |
| 100 | raw | +0.1624 | +0.3053 | +0.3387 | +0.2220 | -102.9 |
| 100 | debiased | +0.1463 | +0.2938 | +0.3422 | +0.2071 | -105.2 |

## Effect B: suppression training (GSU-Debiased - GSU-Raw)

| Step | Scorer | dR@1 | dR@5 | dR@10 | dMRR | d mean rank |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | raw | +0.0000 | +0.0023 | +0.0000 | +0.0007 | -0.1 |
| 20 | debiased | +0.0035 | +0.0012 | +0.0012 | +0.0021 | -0.6 |
| 50 | raw | +0.0023 | +0.0012 | +0.0138 | +0.0024 | -0.4 |
| 50 | debiased | +0.0023 | +0.0069 | +0.0023 | +0.0046 | -1.0 |
| 100 | raw | +0.0046 | +0.0023 | +0.0092 | +0.0042 | +2.7 |
| 100 | debiased | +0.0138 | +0.0138 | -0.0023 | +0.0125 | -0.2 |

## Effect C: suppression inference (Debiased scorer - Raw scorer, same checkpoint)

| Checkpoint | dR@1 | dR@5 | dR@10 | dMRR | improved | unchanged | worsened | mean d rank | median d rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | -0.0035 | -0.0012 | -0.0058 | -0.0029 | 0.359 | 0.158 | 0.483 | -2.64 | +0.0 |
| GS_20 | +0.0000 | -0.0138 | -0.0104 | -0.0032 | 0.318 | 0.153 | 0.529 | -4.62 | -1.0 |
| GS_50 | -0.0081 | -0.0196 | -0.0161 | -0.0126 | 0.255 | 0.131 | 0.614 | -9.74 | -4.0 |
| GS_100 | -0.0012 | -0.0058 | -0.0115 | -0.0022 | 0.316 | 0.105 | 0.579 | -6.68 | -3.0 |
| GSU_Raw_20 | -0.0115 | -0.0104 | -0.0138 | -0.0087 | 0.318 | 0.158 | 0.524 | -4.19 | -1.0 |
| GSU_Raw_50 | -0.0207 | -0.0115 | -0.0150 | -0.0174 | 0.275 | 0.204 | 0.521 | -6.60 | -1.0 |
| GSU_Raw_100 | -0.0173 | -0.0173 | -0.0081 | -0.0170 | 0.251 | 0.311 | 0.438 | -4.36 | +0.0 |
| GSU_Debiased_20 | -0.0081 | -0.0115 | -0.0127 | -0.0073 | 0.312 | 0.158 | 0.530 | -3.74 | -1.0 |
| GSU_Debiased_50 | -0.0207 | -0.0058 | -0.0265 | -0.0151 | 0.285 | 0.205 | 0.510 | -6.02 | -1.0 |
| GSU_Debiased_100 | -0.0081 | -0.0058 | -0.0196 | -0.0087 | 0.289 | 0.342 | 0.369 | -1.42 | +0.0 |

## Offline trajectory (Raw / Debiased R@1 and MRR)

| Arm | 0 | 20 | 50 | 100 | slope 20-50 | slope 50-100 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GS raw | 0.0622 | 0.0611 | 0.0634 | 0.0461 | +0.00008 | -0.00035 | declining |
| GS debiased | 0.0588 | 0.0611 | 0.0553 | 0.0449 | -0.00019 | -0.00021 | declining |
| GSU-Raw raw | 0.0622 | 0.0783 | 0.1290 | 0.2085 | +0.00169 | +0.00159 | rising |
| GSU-Raw debiased | 0.0588 | 0.0668 | 0.1083 | 0.1912 | +0.00138 | +0.00166 | rising |
| GSU-Debiased raw | 0.0622 | 0.0783 | 0.1313 | 0.2131 | +0.00177 | +0.00164 | rising |
| GSU-Debiased debiased | 0.0588 | 0.0703 | 0.1106 | 0.2051 | +0.00134 | +0.00189 | rising |

## Corrected stratification (candidate pool stays 868)


### initial

| Group | n | pool | Raw R@1 | Raw MRR | Deb R@1 | Deb MRR | dMRR | mean d rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage Q1 (cov=0.493) | 217 | 868 | 0.0507 | 0.1029 | 0.0507 | 0.0995 | -0.0033 | -2.55 |
| coverage Q2 (cov=0.505) | 217 | 868 | 0.0415 | 0.0845 | 0.0507 | 0.0892 | +0.0048 | -4.58 |
| coverage Q3 (cov=0.512) | 217 | 868 | 0.0922 | 0.1379 | 0.0737 | 0.1250 | -0.0129 | -2.82 |
| coverage Q4 (cov=0.524) | 217 | 868 | 0.0645 | 0.1178 | 0.0599 | 0.1177 | -0.0001 | -0.60 |
| first_suffix | 334 | 868 | 0.0479 | 0.0946 | 0.0449 | 0.0927 | -0.0019 | -2.44 |
| later_suffix | 534 | 868 | 0.0712 | 0.1208 | 0.0674 | 0.1174 | -0.0035 | -2.76 |
| J/N tercile 0 (0.560) | 290 | 868 | 0.0862 | 0.1281 | 0.0828 | 0.1247 | -0.0034 | -0.47 |
| J/N tercile 1 (0.855) | 289 | 868 | 0.0588 | 0.1040 | 0.0554 | 0.1015 | -0.0025 | -3.63 |
| J/N tercile 2 (1.000) | 289 | 868 | 0.0415 | 0.1002 | 0.0381 | 0.0974 | -0.0028 | -3.82 |
| novelty low (0.254) | 290 | 868 | 0.1069 | 0.1709 | 0.1000 | 0.1640 | -0.0068 | -4.03 |
| novelty mid (0.389) | 289 | 868 | 0.0311 | 0.0866 | 0.0311 | 0.0862 | -0.0003 | -0.52 |
| novelty high (0.574) | 289 | 868 | 0.0484 | 0.0746 | 0.0450 | 0.0731 | -0.0015 | -3.36 |

### GS_100

| Group | n | pool | Raw R@1 | Raw MRR | Deb R@1 | Deb MRR | dMRR | mean d rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage Q1 (cov=0.489) | 217 | 868 | 0.0369 | 0.0662 | 0.0369 | 0.0628 | -0.0034 | -4.35 |
| coverage Q2 (cov=0.520) | 217 | 868 | 0.0507 | 0.0863 | 0.0553 | 0.0821 | -0.0042 | -5.60 |
| coverage Q3 (cov=0.539) | 217 | 868 | 0.0461 | 0.0726 | 0.0415 | 0.0713 | -0.0013 | -6.10 |
| coverage Q4 (cov=0.569) | 217 | 868 | 0.0507 | 0.0779 | 0.0461 | 0.0780 | +0.0001 | -10.66 |
| first_suffix | 334 | 868 | 0.0299 | 0.0606 | 0.0329 | 0.0597 | -0.0009 | -7.43 |
| later_suffix | 534 | 868 | 0.0562 | 0.0853 | 0.0524 | 0.0822 | -0.0030 | -6.20 |
| J/N tercile 0 (0.560) | 290 | 868 | 0.0586 | 0.0782 | 0.0586 | 0.0770 | -0.0011 | -7.21 |
| J/N tercile 1 (0.855) | 289 | 868 | 0.0346 | 0.0645 | 0.0277 | 0.0620 | -0.0025 | -7.32 |
| J/N tercile 2 (1.000) | 289 | 868 | 0.0450 | 0.0847 | 0.0484 | 0.0817 | -0.0030 | -5.49 |
| novelty low (0.418) | 290 | 868 | 0.0621 | 0.0850 | 0.0517 | 0.0780 | -0.0069 | -9.63 |
| novelty mid (0.652) | 289 | 868 | 0.0450 | 0.0746 | 0.0450 | 0.0725 | -0.0021 | -8.26 |
| novelty high (0.887) | 289 | 868 | 0.0311 | 0.0678 | 0.0381 | 0.0702 | +0.0024 | -2.14 |

### GSU_Raw_100

| Group | n | pool | Raw R@1 | Raw MRR | Deb R@1 | Deb MRR | dMRR | mean d rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage Q1 (cov=0.572) | 217 | 868 | 0.1705 | 0.2623 | 0.1475 | 0.2421 | -0.0202 | -0.06 |
| coverage Q2 (cov=0.602) | 217 | 868 | 0.2212 | 0.3124 | 0.2074 | 0.2950 | -0.0174 | -3.38 |
| coverage Q3 (cov=0.632) | 217 | 868 | 0.2120 | 0.3000 | 0.1797 | 0.2734 | -0.0265 | -5.73 |
| coverage Q4 (cov=0.683) | 217 | 868 | 0.2304 | 0.3164 | 0.2304 | 0.3122 | -0.0041 | -8.27 |
| first_suffix | 334 | 868 | 0.1886 | 0.2806 | 0.1677 | 0.2635 | -0.0171 | -5.52 |
| later_suffix | 534 | 868 | 0.2210 | 0.3085 | 0.2060 | 0.2915 | -0.0170 | -3.64 |
| J/N tercile 0 (0.560) | 290 | 868 | 0.2345 | 0.3323 | 0.2207 | 0.3224 | -0.0099 | -4.01 |
| J/N tercile 1 (0.855) | 289 | 868 | 0.1869 | 0.2661 | 0.1730 | 0.2512 | -0.0149 | -0.53 |
| J/N tercile 2 (1.000) | 289 | 868 | 0.2042 | 0.2947 | 0.1799 | 0.2684 | -0.0264 | -8.54 |
| novelty low (0.326) | 290 | 868 | 0.3586 | 0.4615 | 0.3379 | 0.4411 | -0.0204 | -5.29 |
| novelty mid (0.527) | 289 | 868 | 0.2111 | 0.3138 | 0.1799 | 0.2834 | -0.0304 | -6.04 |
| novelty high (0.750) | 289 | 868 | 0.0554 | 0.1174 | 0.0554 | 0.1171 | -0.0003 | -1.74 |

### GSU_Debiased_100

| Group | n | pool | Raw R@1 | Raw MRR | Deb R@1 | Deb MRR | dMRR | mean d rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage Q1 (cov=0.590) | 217 | 868 | 0.1843 | 0.2794 | 0.1613 | 0.2621 | -0.0173 | +0.31 |
| coverage Q2 (cov=0.620) | 217 | 868 | 0.2304 | 0.3213 | 0.2212 | 0.3125 | -0.0087 | -0.66 |
| coverage Q3 (cov=0.654) | 217 | 868 | 0.2442 | 0.3264 | 0.2350 | 0.3149 | -0.0115 | -3.64 |
| coverage Q4 (cov=0.707) | 217 | 868 | 0.1935 | 0.2807 | 0.2028 | 0.2834 | +0.0027 | -1.68 |
| first_suffix | 334 | 868 | 0.1886 | 0.2830 | 0.1826 | 0.2741 | -0.0089 | -2.69 |
| later_suffix | 534 | 868 | 0.2285 | 0.3138 | 0.2191 | 0.3052 | -0.0086 | -0.62 |
| J/N tercile 0 (0.560) | 290 | 868 | 0.2345 | 0.3309 | 0.2207 | 0.3257 | -0.0052 | -2.40 |
| J/N tercile 1 (0.855) | 289 | 868 | 0.1869 | 0.2660 | 0.1903 | 0.2648 | -0.0011 | +2.56 |
| J/N tercile 2 (1.000) | 289 | 868 | 0.2180 | 0.3088 | 0.2042 | 0.2890 | -0.0198 | -4.41 |
| novelty low (0.323) | 290 | 868 | 0.3621 | 0.4656 | 0.3483 | 0.4547 | -0.0109 | -2.68 |
| novelty mid (0.521) | 289 | 868 | 0.2145 | 0.3182 | 0.2042 | 0.3043 | -0.0140 | -3.46 |
| novelty high (0.742) | 289 | 868 | 0.0623 | 0.1214 | 0.0623 | 0.1202 | -0.0012 | +1.89 |

## Canonical standard validation

| Model | Variant | I2T R@1/5/10 | T2I R@1/5/10 | Full Gap | Said Gap | Cond. |
| --- | --- | --- | --- | --- | --- | --- |
| arm_S_step500 | first_sentence | 0.6840/0.9140/0.9590 | 0.6500/0.8820/0.9400 | 0.6788 | 0.6700 | 0.1184 |
| arm_S_step500 | fixed_sparse | 0.9140/0.9870/0.9960 | 0.8840/0.9740/0.9850 | 0.6521 | 0.6793 | 0.1956 |
| arm_S_step500 | full_dense | 0.9740/0.9980/0.9990 | 0.9600/0.9980/0.9990 | 0.6439 | 0.6835 | 0.2174 |
| arm_S_step500 | COCO | 0.5870/0.8132/0.8832 | 0.4030/0.6579/0.7586 | - | - | - |
| initial | first_sentence | 0.5440/0.7740/0.8590 | 0.5140/0.7480/0.8140 | 0.7044 | 0.7008 | 0.0001 |
| initial | fixed_sparse | 0.7460/0.9200/0.9530 | 0.7400/0.9180/0.9490 | 0.6776 | 0.7138 | 0.0000 |
| initial | full_dense | 0.7580/0.9200/0.9520 | 0.7760/0.9480/0.9730 | 0.6783 | 0.7410 | 0.0000 |
| initial | COCO | 0.5170/0.7662/0.8428 | 0.3269/0.5776/0.6823 | - | - | - |

## gap_* logging note

`gap_global_to_full_text` / `gap_global_to_said_text` / `gap_said_to_said_text` were computed
by `model/salu_model.py: forward_train()` but `train/train_salu.py` did not serialize them into
`salu_log.jsonl` for Phase 2.9B, so the historical per-batch values are unavailable. They were
not reconstructed or re-run. Legacy `pair_gap_full` / `pair_gap_said` / `balancing_gain` /
`relative_balancing_gain` remain available.

## Limitations

- 100 steps only, single seed, one candidate pool (868 suffix sentences, duplicates kept).
- Training-batch U top1 is a diagnostic and is not used as evidence of generalization.
- Global (plain CLIP) remains a strong baseline; conditional scorers are compared to it, not
  assumed to beat it at 100 steps.
- Old Phase 2.9B.2 subgroup tables are INVALID (candidate-pool-subsetting bug) and are not used.

## 500-step decision

See the phase report: suffix supervision (Effect A) is SUPPORTED with a large margin, the
GSU-Raw trajectory is still rising at step 50->100, suppression inference (Effect C) is not
supported, and suppression training (Effect B) is only marginally positive.

