# HNS-S12 static weights:2434 ->3651 ->4868

Fixed macros10/1.2/1. No phase switching, additional arm, or training from bare weights. Initial checkpoint SHA: `4cde7d4b91af80755215f20687b80856266fa1faed4b2db864a88ae72b030975`.

| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |
|---:|---|---:|---:|---:|---:|---|
| 3651 | HNS_v1 | 73.774418 | 78.188029 | 87.040003 | 67.154000 | 94.100 / 92.600 |
| 3651 | D3_Balanced | 73.830182 | 78.205636 | 86.925002 | 67.267000 | 93.700 / 92.600 |
| 3651 | HNS_Half | 73.706809 | 78.066015 | 86.880002 | 67.168000 | 94.200 / 92.300 |
| 3651 | HNS-S12 | 73.843712 | 78.234853 | 87.005003 | 67.257000 | 93.900 / 92.200 |

## Step3651: matched-node deltas

HNS_v1: {"Score5": 0.06929436217033924, "J_long3": 0.04682393695058806, "J_long": -0.03499955892563378, "Short4": 0.10300000000000864, "Urban_I2T": -0.1999974250793457, "Urban_T2I": -0.40000081062316895} pp. Long-DCI directional R1 delta: {"I2T": 0.09208103130754486, "T2I": 0.3288608260983872} pp.
D3_Balanced: {"Score5": 0.013530381143723957, "J_long3": 0.029217301906228954, "J_long": 0.08000064373017324, "Short4": -0.009999999999990905, "Urban_I2T": 0.20000338554382324, "Urban_T2I": -0.40000081062316895} pp. Long-DCI directional R1 delta: {"I2T": -0.07892659826361781, "T2I": -0.06577216521967966} pp.
Positive Score5 difference below0.05pp: weak single-seed signal.
HNS_Half: {"Score5": 0.13690280385547737, "J_long3": 0.16883800642581548, "J_long": 0.12500128746033567, "Short4": 0.08899999999999864, "Urban_I2T": -0.29999613761901855, "Urban_T2I": -0.09999871253967285} pp. Long-DCI directional R1 delta: {"I2T": 0.2762430939226568, "T2I": 0.23677979479084232} pp.

| Dataset | I2T R1 / R5 / R10 (%) | T2I R1 / R5 / R10 (%) |
|---|---|---|
| COCO | 62.020000 / 83.980000 / 89.780000 | 43.048000 / 68.652000 / 78.172000 |
| Urban-1k | 93.900007 / 99.000007 / 99.600005 | 92.200005 / 99.200004 / 99.500006 |
| Flickr30k-test1k | 90.000000 / 98.500000 / 99.200000 | 73.960000 / 92.080000 / 95.900000 |
| DOCCI | 80.580000 / 96.360000 / 98.680000 | 81.340000 / 96.280000 / 98.420000 |
| Long-DCI | 60.457774 / 78.426730 / 84.122599 | 60.931334 / 78.808208 / 83.543804 |

Last50 mask telemetry: {"inc": 0.01272914968430996, "inc_weight": 0.0, "F_Dall_mask_iou": 0.8171857285499573, "Dall_D3_mask_iou": 0.7243854057788849, "Dall_F_hard_violation": 0.04707658909261227, "D3_Dall_hard_violation": 0.08129560858011246, "keep_ratios": {"F": 0.6285795557498932, "Dall": 0.6003553330898285, "D3": 0.5729792523384094}}.

E3 complete. E4 configuration remains fixed and queued; final scientific interpretation waits for4868.
All30 recalls and their matched-node deltas are in node RESULTS.json. Model/optimizer/sampler/CPU-CUDA-Python-NumPy RNG/loader restoration and next-five text/token/indices/LR evidence are in node RESUME_GATE.json.
Training/evaluator sources remain byte-identical to the evaluated2434 run; only control and read-only audit node support changed. Native four-GPU evaluator math/batch64/protocol unchanged.
Local-only disposable /root image cache; NFS originals retained. Full checkpoints, bare weights, raw logs stay persistent server-local and are never uploaded.
Stop4868 after final evaluation/report/sync; no new arms or continuation.
