# R-SentenceDrop500: One Variable

Reference:fa19d12, completed matched old-R B16@500/H4868.
Branch:codex/nest-balanced-rdrop-500-v1.
Runtime:/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_500_v1/.

Only compact R changes: take the existing suffix's m complete sentences,
draw q uniformly from1..m inclusive, sample q distinct indices with an independent
salted deterministic SHA256 RNG, sort indices to preserve original order, join
with the original separator and tokenize normally. No internal-PAD insertion,
absolute-position preservation, shuffle, fourth view or token-level mask is used.
When m1 or qm, R is exactly the old compact remainder. Never sample q0.

Original RandomK/F/P/images/data order, seed0, initialization, Balanced structure
and loss remain unchanged. Best coefficients stay2e-4/1/[1,1,1]/1/1. Use B16,
224,context248,4x256,global1024,BF16 encoder/other FP32,encoder checkpoint ON,
pair OFF,blocks128x128 and horizon4868, stopping at500 updates. Resource policy
is normal full updates<=3s and peak allocated<=65GiB/rank.

The matched baseline is strictly verified and reused (full SHA256 f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8).
Only one new5-step smoke and one new formal500 run are scheduled from the common
step0; smoke is not a warm start. All five frozen native protocols are evaluated.
Report raw recalls, ten R1 deltas, m/q/keep distributions and resource measurements
in RDROP_500_REPORT.md/RESULTS.json. Preserve failures without automatic retries.

Even a gain only warrants recommending full4epoch confirmation; do not actually
start4868, retune q, resume RMask, add views/losses or seeds. Compact evidence/code
are pushed to this independent branch; large checkpoints/data/cache stay local.
