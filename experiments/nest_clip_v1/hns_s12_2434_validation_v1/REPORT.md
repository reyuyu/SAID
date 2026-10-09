# Frozen HNS-S12 continuation:500 ->1217 ->2434

| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |
|---:|---|---:|---:|---:|---:|---|
| 1217 | HNS-v1 | 72.836442 | 76.971404 | 85.955001 | 66.634000 | 93.400 / 91.500 |
| 1217 | D3 Balanced | 72.812657 | 77.001762 | 86.040002 | 66.529000 | 93.600 / 91.900 |
| 1217 | HNS-S12 | 72.906355 | 76.999258 | 85.760003 | 66.767000 | 93.200 / 91.100 |

## Step1217 deltas and interpretation

HNS_v1: {"Score5": 0.06991269039252757, "J_long3": 0.027854483987567846, "J_long": -0.1949980688095252, "Short4": 0.13299999999999557, "Urban_I2T": -0.1999974250793457, "Urban_T2I": -0.3999948501586914}. Long-DCI directional R1 deltas: {"I2T": 0.5787950539331788, "T2I": 0.36832412523020164}.
Short4 benefit retained vs HNS_v1: True; Urban T2I decrease: True; Long-DCI R1 decreases in either direction: False.
D3_Balanced: {"Score5": 0.09369759042856174, "J_long3": -0.0025040159523967986, "J_long": -0.2799991178512613, "Short4": 0.23799999999999955, "Urban_I2T": -0.40000081062316895, "Urban_T2I": -0.7999956607818604}. Long-DCI directional R1 deltas: {"I2T": 0.6971849513285999, "T2I": 0.4077874243620161}.
Short4 benefit retained vs D3_Balanced: True; Urban T2I decrease: True; Long-DCI R1 decreases in either direction: False.

| Dataset | I2T R1 / R5 / R10 (%) | T2I R1 / R5 / R10 (%) |
|---|---|---|
| COCO | 61.880000 / 83.580000 / 89.620000 | 42.548000 / 68.292000 / 77.836000 |
| Urban-1k | 93.200004 / 99.000007 / 99.600005 | 91.100007 / 98.700005 / 99.600005 |
| Flickr30k-test1k | 89.500000 / 98.200000 / 99.200000 | 73.140000 / 92.060000 / 95.620000 |
| DOCCI | 79.060000 / 96.040000 / 98.340000 | 79.680000 / 95.800000 / 98.180000 |
| Long-DCI | 59.037096 / 77.400684 / 83.201789 | 59.918443 / 78.018942 / 83.175480 |

Last50 mask telemetry: {"inc": 0.01494597252458334, "inc_weight": 0.0, "F_Dall_mask_iou": 0.8644969487190246, "Dall_D3_mask_iou": 0.7891420674324036, "Dall_F_hard_violation": 0.03890041448175907, "D3_Dall_hard_violation": 0.05778378665447235, "keep_ratios": {"F": 0.7206451380252838, "Dall": 0.6968039608001709, "D3": 0.6521217107772828}}.

Full30 recalls and deltas, last50 keep/violations/losses, component/group gradients and cosines are in node JSONs.

All production source hashes match the evaluated S12@500 checkpoint. Macro10/1.2/1 is static; native horizon4868/ramp200 and original method/optimizer/data are frozen.
Both segment starts audit exact model/adapter/optimizer/CPU-CUDA-Python-NumPy RNG/loader generator and cursor, plus next five samples/text/tokens/indices/LR. Evaluation starts only after training and gradient processes exit.
Training images are local-only disposable /root overlay; persistent NFS originals are retained. Checkpoints/bare/raw logs are never uploaded.
Only this S12 trajectory is authorized. Stop2434; no3651/4868 or other arms.
