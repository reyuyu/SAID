# B4-MASK115 epoch4 LR experiment

LR multipliers: {'backbone': 1.0, 'text_mask_and_shared_pool': 1.15, 'visual_mask': 1.15, 'fusion_adapter': 1.0}

All 1217 optimizer updates and 1245904 sample positions verified.
Checkpoint SHA256: 79b18294b6c66d3f5b8842b64f086fdad902bc2c04b8e15bcfdb65855fab7f34
Scores: {"Score5": 73.8073866418168, "J_long3": 78.18697773636136, "J_long": 87.04500225305557, "Short4": 67.238, "Urban_I2T": 93.90000700950623, "Urban_T2I": 92.90000200271606}
Delta pp: {"E2_Uniform": {"quality_delta_pp": {"Score5": -0.004800000000003024, "J_long3": -0.006666666666660603, "J_long": -0.010000000000005116, "Short4": -0.001999999999995339, "Urban_I2T": 0.0, "Urban_T2I": 0.0}, "recall_delta_pp": {"COCO": {"I2T": {"R@1": -0.0400000000000067, "R@5": 0.019999999999997797, "R@10": 0.11999999999999789}, "T2I": {"R@1": 0.031999999999998696, "R@5": -0.028000000000005798, "R@10": 0.0200000000000089}}, "Urban-1k": {"I2T": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}, "Flickr30k-test1k": {"I2T": {"R@1": 0.0, "R@5": 0.10000000000000009, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}, "DOCCI": {"I2T": {"R@1": -0.0400000000000067, "R@5": 0.060000000000004494, "R@10": 0.0}, "T2I": {"R@1": 0.0, "R@5": 0.0400000000000067, "R@10": 0.0}}, "Long-DCI": {"I2T": {"R@1": -0.03946329913180335, "R@5": 0.013154433043938152, "R@10": 0.013154433043938152}, "T2I": {"R@1": 0.03946329913180335, "R@5": -0.03946329913180335, "R@10": -0.039463299131814455}}}}}

All recalls and diagnostics are saved in JSON. Public benchmark results are exploratory.
No changes to loss, model, detach, data, batch or evaluator mathematics.
