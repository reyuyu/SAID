# B2-BB115 epoch4 LR experiment

LR multipliers: {'backbone': 1.15, 'text_mask_and_shared_pool': 1.0, 'visual_mask': 1.0, 'fusion_adapter': 1.0}

All 1217 optimizer updates and 1245904 sample positions verified.
Checkpoint SHA256: 30122a42a0e5102ab6b105ab1c7091cd082350c65e30bb003f7f5920b4313041
Scores: {"Score5": 73.76966309538166, "J_long3": 78.1627718256361, "J_long": 87.03500225305557, "Short4": 67.18, "Urban_I2T": 93.90000700950623, "Urban_T2I": 92.90000200271606}
Delta pp: {"E2_Uniform": {"quality_delta_pp": {"Score5": -0.042523546435148774, "J_long3": -0.030872577391917844, "J_long": -0.020000000000010232, "Short4": -0.05999999999998806, "Urban_I2T": 0.0, "Urban_T2I": 0.0}, "recall_delta_pp": {"COCO": {"I2T": {"R@1": 0.0, "R@5": 0.019999999999997797, "R@10": 0.13999999999999568}, "T2I": {"R@1": 0.0, "R@5": -0.031999999999998696, "R@10": -0.007999999999996898}}, "Urban-1k": {"I2T": {"R@1": 0.0, "R@5": 0.09999871253967285, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}, "Flickr30k-test1k": {"I2T": {"R@1": -0.20000000000000018, "R@5": 0.0, "R@10": 0.0}, "T2I": {"R@1": -0.039999999999995595, "R@5": -0.039999999999995595, "R@10": 0.019999999999997797}}, "DOCCI": {"I2T": {"R@1": -0.0400000000000067, "R@5": -0.039999999999995595, "R@10": 0.0}, "T2I": {"R@1": -0.0400000000000067, "R@5": -0.039999999999995595, "R@10": 0.0}}, "Long-DCI": {"I2T": {"R@1": -0.052617732175741505, "R@5": 0.013154433043938152, "R@10": 0.0}, "T2I": {"R@1": -0.052617732175741505, "R@5": -0.026308866087876304, "R@10": -0.026308866087876304}}}}}

All recalls and diagnostics are saved in JSON. Public benchmark results are exploratory.
No changes to loss, model, detach, data, batch or evaluator mathematics.
