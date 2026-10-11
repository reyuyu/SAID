# HNS-v1 ViT-L/14 Full4 Epoch Final Report

The experiment trained the verified HNS-v1 ViT-L/14 initialization from step 0 to exactly 4868 optimizer updates with scheduler horizon 4868. Resource validation passed (5 warmups, 30 measurements; P95 4.435966 s, maximum 4.501598 s, peak allocated 47.364 GiB/GPU). Smoke and formal DDP training passed with four-rank parameter difference 0.0 and no resource failure.

Both forced checkpoints (3651 and 4868) were exported as strict bare students with zero image/text equivalence error. Standard native evaluation and the verified Flickr30k-Full protocol completed for both nodes. The full metric tables are in RESULTS_3651.json, RESULTS_4868.json, and FLICKR_FULL_RESULTS.json.

At step 4868, Score5 is 77.890862, J_long3 83.191078, J_long 90.670000, Short4 71.390, and Urban I2T/T2I 95.3/95.2 (mean 95.25). Relative to step 3651, the gains are small: Score5 +0.060338 and Urban mean +0.05 percentage points; T2I is unchanged at R@1. Flickr30k-Full changes from step 3651 to 4868 are descriptive and mixed (I2T R@1 -0.0378 pp; T2I R@1 +0.0277 pp).

No DCI evaluation was run. Checkpoints and large runtime artifacts remain on the server; this directory contains only small audit and result files.
