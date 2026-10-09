# B1-BB085 epoch4 LR experiment

LR multipliers: {'backbone': 0.85, 'text_mask_and_shared_pool': 1.0, 'visual_mask': 1.0, 'fusion_adapter': 1.0}

All 1217 optimizer updates and 1245904 sample positions verified.
Checkpoint SHA256: f925f3858e12f9f00f7db06f0ce66f06907f6988fb3a8d39feee699df1f94bbd
Scores: {"Score5": 73.79184841503438, "J_long3": 78.18574735839063, "J_long": 87.03000225305557, "Short4": 67.201, "Urban_I2T": 93.90000700950623, "Urban_T2I": 92.90000200271606}
Delta pp: {"E2_Uniform": {"quality_delta_pp": {"Score5": -0.020338226782428137, "J_long3": -0.007897044637388717, "J_long": -0.025000000000005684, "Short4": -0.03900000000000148, "Urban_I2T": 0.0, "Urban_T2I": 0.0}, "recall_delta_pp": {"COCO": {"I2T": {"R@1": 0.019999999999997797, "R@5": 0.08000000000000229, "R@10": 0.05999999999999339}, "T2I": {"R@1": -0.015999999999999348, "R@5": -0.0160000000000049, "R@10": 0.031999999999998696}}, "Urban-1k": {"I2T": {"R@1": 0.0, "R@5": 0.09999871253967285, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}, "Flickr30k-test1k": {"I2T": {"R@1": -0.20000000000000018, "R@5": 0.0, "R@10": 0.0}, "T2I": {"R@1": 0.039999999999995595, "R@5": 0.0200000000000089, "R@10": -0.0400000000000067}}, "DOCCI": {"I2T": {"R@1": -0.060000000000004494, "R@5": 0.0, "R@10": 0.019999999999997797}, "T2I": {"R@1": -0.0400000000000067, "R@5": -0.019999999999997797, "R@10": 0.0}}, "Long-DCI": {"I2T": {"R@1": 0.013154433043938152, "R@5": -0.03946329913180335, "R@10": 0.013154433043938152}, "T2I": {"R@1": 0.03946329913180335, "R@5": 0.026308866087876304, "R@10": 0.013154433043938152}}}}}

All recalls and diagnostics are saved in JSON. Public benchmark results are exploratory.
No changes to loss, model, detach, data, batch or evaluator mathematics.
