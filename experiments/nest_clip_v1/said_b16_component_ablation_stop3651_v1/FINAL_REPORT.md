# ViT-B/16 component ablation: final status

The authorized sequence stopped at B.  Historical A remains a read-only
reference with Score5 73.774418, J_long3 78.188029, J_long 87.040003,
Short4 67.154, Urban I2T 94.1 and Urban T2I 92.6.

B's explicit Joint-Mask-Only implementation passed CPU tests, a four-GPU
5-step smoke, and a 5-warmup/30-measurement resource probe.  The valid formal
run started from the immutable common step-0 SHA256
`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6` and used
the corrected fusion-adapter peak LR `2e-4` with the 4868 scheduler horizon.
It stopped at update 2261 when three consecutive full cycles exceeded the
existing 3-second resource stop gate (steps 2259–2261).  Teardown then
reported a DataLoader worker abort and torchrun exit 1.  The last checkpoint
and all logs remain server-local as failure evidence.

Because B did not reach step3651 and did not pass final audit, its bare export
and all five-set evaluations are explicitly **NOT RUN**.  C and D were never
started.  No incomplete arm is treated as a result, and no retry or resource
threshold change was made.
