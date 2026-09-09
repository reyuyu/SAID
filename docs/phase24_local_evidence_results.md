# Phase 2.4 results

Primary tau = 0.07; all maps use identity coordinates. All metric values below are fractions, not percentages.

## Small audit: 128 images / 1,908 phrases

| Candidate | pointing | gt_mass | mass_gain | semantic_mass_excess | target_gt_distractor | localization_margin | switch_margin | switch_positive_fraction | semantic_pointing_excess | orientation semantic gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_block03_residual | 0.309224 | 0.325036 | -0.008677 | -0.005676 | 0.502387 | -0.042949 | -0.005676 | 0.358650 | -0.023207 | -0.006905 |
| initial_block03_after_attention | 0.321803 | 0.326045 | -0.007668 | -0.004331 | 0.505370 | -0.041996 | -0.004331 | 0.371308 | -0.016878 | -0.005459 |
| initial_block03_attention_delta | 0.350629 | 0.332546 | -0.001167 | 0.001870 | 0.504773 | -0.031869 | 0.001870 | 0.582278 | 0.004219 | 0.001524 |
| initial_block03_mlp_delta | 0.344864 | 0.333873 | 0.000160 | -0.001929 | 0.500000 | -0.034595 | -0.001929 | 0.345992 | -0.018987 | -0.002415 |
| initial_block06_residual | 0.202306 | 0.312302 | -0.021411 | -0.012516 | 0.489260 | -0.052024 | -0.012516 | 0.198312 | -0.071730 | -0.015612 |
| initial_block06_after_attention | 0.210692 | 0.317714 | -0.016000 | -0.010754 | 0.489857 | -0.048898 | -0.010754 | 0.227848 | -0.035865 | -0.013871 |
| initial_block06_attention_delta | 0.299266 | 0.329641 | -0.004072 | -0.005507 | 0.499403 | -0.038883 | -0.005507 | 0.354430 | -0.006329 | -0.006980 |
| initial_block06_mlp_delta | 0.302411 | 0.328982 | -0.004732 | -0.002389 | 0.503580 | -0.037039 | -0.002389 | 0.405063 | -0.029536 | -0.002087 |
| initial_block09_residual | 0.203354 | 0.303047 | -0.030667 | -0.020023 | 0.486277 | -0.062871 | -0.020023 | 0.080169 | -0.004219 | -0.026147 |
| initial_block09_after_attention | 0.219078 | 0.308088 | -0.025625 | -0.017195 | 0.489857 | -0.058078 | -0.017195 | 0.126582 | -0.012658 | -0.022238 |
| initial_block09_attention_delta | 0.298218 | 0.331618 | -0.002095 | 0.003579 | 0.512530 | -0.032548 | 0.003579 | 0.578059 | 0.021097 | 0.004183 |
| initial_block09_mlp_delta | 0.235325 | 0.313431 | -0.020282 | -0.013387 | 0.490453 | -0.054046 | -0.013387 | 0.168776 | -0.004219 | -0.017366 |
| initial_block11_residual | 0.247904 | 0.315033 | -0.018680 | -0.014978 | 0.497017 | -0.052645 | -0.014978 | 0.164557 | -0.006329 | -0.019365 |
| initial_block11_after_attention | 0.281447 | 0.316433 | -0.017281 | -0.014249 | 0.495823 | -0.052500 | -0.014249 | 0.177215 | 0.002110 | -0.018686 |
| initial_block11_attention_delta | 0.641509 | 0.357991 | 0.024277 | 0.018517 | 0.525060 | -0.008076 | 0.018517 | 0.873418 | 0.310127 | 0.023186 |
| initial_block11_mlp_delta | 0.355346 | 0.328374 | -0.005339 | 0.000977 | 0.504177 | -0.033816 | 0.000977 | 0.540084 | 0.016878 | 0.001841 |
| phase22_block03_residual | 0.276205 | 0.319363 | -0.014350 | -0.006936 | 0.501790 | -0.046796 | -0.006936 | 0.350211 | -0.025316 | -0.007691 |
| phase22_block03_after_attention | 0.295073 | 0.321187 | -0.012526 | -0.005380 | 0.504773 | -0.044971 | -0.005380 | 0.392405 | -0.029536 | -0.006224 |
| phase22_block03_attention_delta | 0.360063 | 0.333298 | -0.000415 | 0.002477 | 0.509547 | -0.030745 | 0.002477 | 0.569620 | 0.008439 | 0.002247 |
| phase22_block03_mlp_delta | 0.331237 | 0.332826 | -0.000887 | -0.002237 | 0.502387 | -0.035673 | -0.002237 | 0.333333 | 0.012658 | -0.002617 |
| phase22_block06_residual | 0.205975 | 0.306692 | -0.027022 | -0.013732 | 0.490453 | -0.055455 | -0.013732 | 0.215190 | -0.035865 | -0.016813 |
| phase22_block06_after_attention | 0.226415 | 0.313710 | -0.020003 | -0.011862 | 0.489857 | -0.051402 | -0.011862 | 0.215190 | -0.054852 | -0.014996 |
| phase22_block06_attention_delta | 0.305556 | 0.326601 | -0.007112 | -0.005210 | 0.498807 | -0.040442 | -0.005210 | 0.350211 | -0.012658 | -0.007260 |
| phase22_block06_mlp_delta | 0.284067 | 0.326816 | -0.006898 | -0.002788 | 0.502387 | -0.038399 | -0.002788 | 0.392405 | -0.035865 | -0.002503 |
| phase22_block09_residual | 0.204927 | 0.294458 | -0.039255 | -0.019347 | 0.482100 | -0.066270 | -0.019347 | 0.088608 | -0.004219 | -0.023486 |
| phase22_block09_after_attention | 0.217505 | 0.300803 | -0.032910 | -0.017755 | 0.486874 | -0.061418 | -0.017755 | 0.122363 | -0.010549 | -0.021613 |
| phase22_block09_attention_delta | 0.412998 | 0.333203 | -0.000510 | 0.005951 | 0.513126 | -0.030422 | 0.005951 | 0.628692 | 0.109705 | 0.006789 |
| phase22_block09_mlp_delta | 0.207023 | 0.304295 | -0.029418 | -0.013783 | 0.485680 | -0.058400 | -0.013783 | 0.135021 | -0.008439 | -0.016463 |
| phase22_block11_residual | 0.241614 | 0.298650 | -0.035063 | -0.019651 | 0.488067 | -0.064092 | -0.019651 | 0.177215 | -0.018987 | -0.024097 |
| phase22_block11_after_attention | 0.225891 | 0.299452 | -0.034261 | -0.018264 | 0.490453 | -0.063617 | -0.018264 | 0.139241 | -0.004219 | -0.021740 |
| phase22_block11_attention_delta | 0.575472 | 0.353168 | 0.019454 | 0.018484 | 0.523270 | -0.009620 | 0.018484 | 0.856540 | 0.308017 | 0.022852 |
| phase22_block11_mlp_delta | 0.351677 | 0.330395 | -0.003319 | 0.002153 | 0.501193 | -0.031530 | 0.002153 | 0.590717 | 0.031646 | 0.002344 |

### Orientation and spatial prior controls

| Candidate | rotated mass gain | rotated semantic excess | mass gain gap | logit margin mean | median | positive fraction | prior Pearson | prior Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_block03_residual | -0.003977 | 0.001228 | -0.004700 | -0.003531 | -0.000490 | 0.439203 | -0.006464 | -0.000324 |
| initial_block03_after_attention | -0.003360 | 0.001128 | -0.004308 | -0.002982 | -0.000225 | 0.451782 | -0.004143 | 0.004310 |
| initial_block03_attention_delta | -0.001892 | 0.000346 | 0.000725 | -0.000688 | 0.000000 | 0.496855 | -0.461303 | -0.451686 |
| initial_block03_mlp_delta | 0.000214 | 0.000486 | -0.000054 | -0.000521 | -0.000181 | 0.453878 | 0.372796 | 0.354320 |
| initial_block06_residual | -0.003147 | 0.003097 | -0.018265 | -0.009615 | -0.005162 | 0.263627 | -0.394673 | -0.367752 |
| initial_block06_after_attention | -0.002471 | 0.003117 | -0.013528 | -0.007315 | -0.003577 | 0.307652 | -0.160931 | -0.148330 |
| initial_block06_attention_delta | 0.000321 | 0.001473 | -0.004393 | -0.001818 | -0.000409 | 0.449686 | -0.233642 | -0.231518 |
| initial_block06_mlp_delta | -0.000299 | -0.000302 | -0.004433 | -0.001714 | -0.000785 | 0.390985 | -0.507970 | -0.479921 |
| initial_block09_residual | 0.001839 | 0.006124 | -0.032506 | -0.016208 | -0.009731 | 0.163522 | -0.698763 | -0.685949 |
| initial_block09_after_attention | 0.001109 | 0.005043 | -0.026734 | -0.013658 | -0.007989 | 0.181342 | -0.611687 | -0.580929 |
| initial_block09_attention_delta | -0.001855 | -0.000604 | -0.000239 | 0.000418 | -0.000009 | 0.488470 | -0.526103 | -0.510671 |
| initial_block09_mlp_delta | 0.002253 | 0.003979 | -0.022535 | -0.010097 | -0.005161 | 0.237421 | -0.639485 | -0.611991 |
| initial_block11_residual | 0.001906 | 0.004387 | -0.020586 | -0.015438 | -0.007242 | 0.208595 | -0.381463 | -0.348152 |
| initial_block11_after_attention | 0.002548 | 0.004436 | -0.019828 | -0.015405 | -0.006728 | 0.221174 | -0.286576 | -0.235386 |
| initial_block11_attention_delta | -0.000091 | -0.004668 | 0.024368 | 0.008979 | 0.005464 | 0.825472 | 0.856347 | 0.867765 |
| initial_block11_mlp_delta | -0.002894 | -0.000864 | -0.002445 | 0.000607 | -0.000040 | 0.477987 | -0.650947 | -0.636865 |
| phase22_block03_residual | -0.004902 | 0.000755 | -0.009448 | -0.005674 | -0.001795 | 0.392034 | -0.132965 | -0.113881 |
| phase22_block03_after_attention | -0.003841 | 0.000844 | -0.008685 | -0.004779 | -0.000902 | 0.423480 | -0.148271 | -0.128242 |
| phase22_block03_attention_delta | -0.001940 | 0.000230 | 0.001525 | -0.000404 | 0.000011 | 0.504193 | -0.362744 | -0.358044 |
| phase22_block03_mlp_delta | -0.000386 | 0.000380 | -0.000501 | -0.000843 | -0.000223 | 0.448113 | 0.351623 | 0.328116 |
| phase22_block06_residual | -0.002787 | 0.003081 | -0.024235 | -0.012121 | -0.006561 | 0.224843 | -0.486283 | -0.462393 |
| phase22_block06_after_attention | -0.001395 | 0.003134 | -0.018608 | -0.009273 | -0.004612 | 0.266247 | -0.291761 | -0.275976 |
| phase22_block06_attention_delta | 0.000528 | 0.002049 | -0.007640 | -0.003072 | -0.001124 | 0.397275 | -0.676063 | -0.678796 |
| phase22_block06_mlp_delta | -0.001013 | -0.000285 | -0.005884 | -0.002396 | -0.001242 | 0.353774 | -0.620324 | -0.586152 |
| phase22_block09_residual | -0.000386 | 0.004139 | -0.038869 | -0.020163 | -0.012388 | 0.133648 | -0.746683 | -0.738122 |
| phase22_block09_after_attention | 0.000023 | 0.003857 | -0.032934 | -0.016836 | -0.010194 | 0.155136 | -0.676271 | -0.653621 |
| phase22_block09_attention_delta | -0.002438 | -0.000838 | 0.001928 | 0.001295 | 0.000055 | 0.512579 | -0.394454 | -0.404971 |
| phase22_block09_mlp_delta | -0.000389 | 0.002680 | -0.029029 | -0.013940 | -0.007792 | 0.181866 | -0.738984 | -0.741115 |
| phase22_block11_residual | 0.000195 | 0.004447 | -0.035258 | -0.025254 | -0.014004 | 0.168239 | -0.654205 | -0.636051 |
| phase22_block11_after_attention | 0.000151 | 0.003476 | -0.034412 | -0.022165 | -0.012005 | 0.170860 | -0.568437 | -0.539990 |
| phase22_block11_attention_delta | -0.001668 | -0.004367 | 0.021123 | 0.008118 | 0.004751 | 0.791929 | 0.352942 | 0.364744 |
| phase22_block11_mlp_delta | -0.001950 | -0.000190 | -0.001369 | 0.000610 | 0.000034 | 0.509434 | -0.726485 | -0.706062 |

Random patch-center pointing: 0.334850; mean GT area fraction: 0.333713.

## Full validation: 1,000 images / 14,334 phrases

| Candidate | pointing | gt_mass | mass_gain | semantic_mass_excess | target_gt_distractor | localization_margin | switch_margin | switch_positive_fraction | semantic_pointing_excess | orientation semantic gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_block11_attention_delta | 0.647691 | 0.344845 | 0.025210 | 0.018114 | 0.533093 | 0.008029 | 0.018114 | 0.867175 | 0.320904 | 0.023265 |
| phase22_block11_attention_delta | 0.592159 | 0.340245 | 0.020611 | 0.018543 | 0.535014 | 0.006839 | 0.018543 | 0.870441 | 0.310016 | 0.024100 |
| initial_block11_residual | 0.233710 | 0.300584 | -0.019051 | -0.014220 | 0.485074 | -0.039170 | -0.014220 | 0.160044 | -0.021502 | -0.018653 |
| phase22_block11_residual | 0.232385 | 0.283930 | -0.035705 | -0.020026 | 0.472749 | -0.051162 | -0.020026 | 0.134458 | -0.017148 | -0.025133 |

### Orientation and spatial prior controls

| Candidate | rotated mass gain | rotated semantic excess | mass gain gap | logit margin mean | median | positive fraction | prior Pearson | prior Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_block11_attention_delta | -0.002334 | -0.005151 | 0.027544 | 0.010366 | 0.006109 | 0.840100 | 0.929484 | 0.929707 |
| phase22_block11_attention_delta | -0.003806 | -0.005557 | 0.024416 | 0.009465 | 0.005346 | 0.809265 | 0.625877 | 0.618392 |
| initial_block11_residual | 0.002164 | 0.004433 | -0.021215 | -0.015924 | -0.007752 | 0.197781 | -0.395371 | -0.399398 |
| phase22_block11_residual | 0.000675 | 0.005107 | -0.036380 | -0.027022 | -0.014769 | 0.150272 | -0.705253 | -0.698034 |

Random patch-center pointing: 0.320884; mean GT area fraction: 0.319635.

## Block effects on the fixed small sample

These are changes in projected phrase-localization metrics, not a causal measurement of pixel dependence. Attention update is x_attn minus x_in; MLP update is x_out minus x_attn.

| Backbone | Block | Update | change mass gain | change semantic excess | change pointing |
|---|---:|---|---:|---:|---:|
| initial | 3 | attention | -0.000458 | +0.000924 | -0.001048 |
| initial | 3 | MLP | -0.001009 | -0.001346 | -0.012579 |
| initial | 6 | attention | -0.000173 | -0.001872 | -0.009434 |
| initial | 6 | MLP | -0.005412 | -0.001762 | -0.008386 |
| initial | 9 | attention | +0.003602 | +0.002619 | +0.021488 |
| initial | 9 | MLP | -0.005042 | -0.002827 | -0.015723 |
| initial | 11 | attention | +0.015218 | +0.008118 | +0.064990 |
| initial | 11 | MLP | -0.001400 | -0.000729 | -0.033543 |
| phase22 | 3 | attention | -0.000767 | +0.001271 | -0.008386 |
| phase22 | 3 | MLP | -0.001824 | -0.001556 | -0.018868 |
| phase22 | 6 | attention | -0.000548 | -0.001597 | -0.005241 |
| phase22 | 6 | MLP | -0.007018 | -0.001870 | -0.020440 |
| phase22 | 9 | attention | +0.004116 | +0.003247 | -0.003669 |
| phase22 | 9 | MLP | -0.006345 | -0.001591 | -0.012579 |
| phase22 | 11 | attention | +0.008541 | +0.002888 | +0.009958 |
| phase22 | 11 | MLP | -0.000802 | -0.001387 | +0.015723 |

## Small temperature robustness

Fixed selected candidates and matched final residual baselines; no temperature selection.

| Candidate | tau | mass gain | semantic excess / switch | target > distractor |
|---|---:|---:|---:|---:|
| initial_block11_attention_delta | 0.03 | 0.061007 | 0.047844 | 0.561456 |
| initial_block11_attention_delta | 0.05 | 0.034955 | 0.026810 | 0.536396 |
| initial_block11_attention_delta | 0.07 | 0.024277 | 0.018517 | 0.525060 |
| initial_block11_attention_delta | 0.1 | 0.016586 | 0.012624 | 0.516110 |
| initial_block11_attention_delta | 0.2 | 0.008036 | 0.006115 | 0.512530 |
| phase22_block11_attention_delta | 0.03 | 0.050311 | 0.048320 | 0.556086 |
| phase22_block11_attention_delta | 0.05 | 0.028322 | 0.026941 | 0.535800 |
| phase22_block11_attention_delta | 0.07 | 0.019454 | 0.018484 | 0.523270 |
| phase22_block11_attention_delta | 0.1 | 0.013179 | 0.012517 | 0.516706 |
| phase22_block11_attention_delta | 0.2 | 0.006326 | 0.006007 | 0.510143 |
| initial_block11_residual | 0.03 | -0.036536 | -0.024736 | 0.489260 |
| initial_block11_residual | 0.05 | -0.024518 | -0.018781 | 0.491647 |
| initial_block11_residual | 0.07 | -0.018680 | -0.014978 | 0.497017 |
| initial_block11_residual | 0.1 | -0.013833 | -0.011457 | 0.497613 |
| initial_block11_residual | 0.2 | -0.007453 | -0.006407 | 0.498210 |
| phase22_block11_residual | 0.03 | -0.060130 | -0.023691 | 0.462411 |
| phase22_block11_residual | 0.05 | -0.044325 | -0.022331 | 0.481504 |
| phase22_block11_residual | 0.07 | -0.035063 | -0.019651 | 0.488067 |
| phase22_block11_residual | 0.1 | -0.026721 | -0.016220 | 0.491050 |
| phase22_block11_residual | 0.2 | -0.014934 | -0.009961 | 0.494033 |

## Full temperature robustness

Fixed selected candidates and matched final residual baselines; no temperature selection.

| Candidate | tau | mass gain | semantic excess / switch | target > distractor |
|---|---:|---:|---:|---:|
| initial_block11_attention_delta | 0.03 | 0.064598 | 0.047902 | 0.570628 |
| initial_block11_attention_delta | 0.05 | 0.036572 | 0.026445 | 0.543657 |
| initial_block11_attention_delta | 0.07 | 0.025210 | 0.018114 | 0.533093 |
| initial_block11_attention_delta | 0.1 | 0.017122 | 0.012268 | 0.524610 |
| initial_block11_attention_delta | 0.2 | 0.008240 | 0.005895 | 0.515886 |
| phase22_block11_attention_delta | 0.03 | 0.054198 | 0.049505 | 0.569748 |
| phase22_block11_attention_delta | 0.05 | 0.030149 | 0.027169 | 0.546539 |
| phase22_block11_attention_delta | 0.07 | 0.020611 | 0.018543 | 0.535014 |
| phase22_block11_attention_delta | 0.1 | 0.013916 | 0.012525 | 0.526771 |
| phase22_block11_attention_delta | 0.2 | 0.006656 | 0.006004 | 0.515966 |
| initial_block11_residual | 0.03 | -0.037130 | -0.022577 | 0.468187 |
| initial_block11_residual | 0.05 | -0.024974 | -0.017670 | 0.478912 |
| initial_block11_residual | 0.07 | -0.019051 | -0.014220 | 0.485074 |
| initial_block11_residual | 0.1 | -0.014125 | -0.010930 | 0.490276 |
| initial_block11_residual | 0.2 | -0.007627 | -0.006133 | 0.496759 |
| phase22_block11_residual | 0.03 | -0.061826 | -0.025800 | 0.455062 |
| phase22_block11_residual | 0.05 | -0.045283 | -0.023183 | 0.467147 |
| phase22_block11_residual | 0.07 | -0.035705 | -0.020026 | 0.472749 |
| phase22_block11_residual | 0.1 | -0.027160 | -0.016327 | 0.480432 |
| phase22_block11_residual | 0.2 | -0.015172 | -0.009894 | 0.491397 |
