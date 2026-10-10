# B3-MASK085 epoch4 LR experiment

LR multipliers: {'backbone': 1.0, 'text_mask_and_shared_pool': 0.85, 'visual_mask': 0.85, 'fusion_adapter': 1.0}

All 1217 optimizer updates and 1245904 sample positions verified.
Checkpoint SHA256: f74d902bb537a0c6084887171e463b5962fc59352d2d8a2f49e6c3101b0bc4d4
Scores: {"Score5": 73.75880942529484, "J_long3": 78.16934904215807, "J_long": 87.03500225305557, "Short4": 67.143, "Urban_I2T": 93.90000700950623, "Urban_T2I": 92.90000200271606}
Delta pp: {"E2_Uniform": {"quality_delta_pp": {"Score5": -0.05337721652196592, "J_long3": -0.02429536086994233, "J_long": -0.020000000000010232, "Short4": -0.0969999999999942, "Urban_I2T": 0.0, "Urban_T2I": 0.0}, "recall_delta_pp": {"COCO": {"I2T": {"R@1": -0.12000000000000899, "R@5": -0.08000000000000229, "R@10": 0.05999999999999339}, "T2I": {"R@1": 0.012000000000000899, "R@5": -0.007999999999996898, "R@10": 0.0}}, "Urban-1k": {"I2T": {"R@1": 0.0, "R@5": 0.09999871253967285, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}, "Flickr30k-test1k": {"I2T": {"R@1": -0.30000000000000027, "R@5": 0.0, "R@10": 0.0}, "T2I": {"R@1": 0.019999999999997797, "R@5": 0.0, "R@10": -0.0200000000000089}}, "DOCCI": {"I2T": {"R@1": -0.0200000000000089, "R@5": 0.0400000000000067, "R@10": 0.0}, "T2I": {"R@1": -0.060000000000004494, "R@5": 0.019999999999997797, "R@10": -0.039999999999995595}}, "Long-DCI": {"I2T": {"R@1": -0.07892659826361781, "R@5": -0.0263088660878652, "R@10": 0.013154433043938152}, "T2I": {"R@1": 0.01315443304392705, "R@5": -0.026308866087876304, "R@10": 0.0263088660878652}}}}}

All recalls and diagnostics are saved in JSON. Public benchmark results are exploratory.
No changes to loss, model, detach, data, batch or evaluator mathematics.
