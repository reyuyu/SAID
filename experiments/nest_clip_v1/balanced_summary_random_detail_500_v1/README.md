# Summary + Random Detail Subset: independent500-update test

Model/loss/optimizer/scheduler/native inference unchanged from14653c9. Old modes remain available.
New summary_random_detail uses only the unchanged packed Full-visible sentences. Summary=s1;
Detail is a uniformly drawn ordered subset froms2..sn, with independent SHA256(seed,epoch,sample_id,domain) RNG.
For m>=3, K is uniform2..m-1; m1/2 useK1; m0 Full-only. No raw trailing sentences enter local views.

Matched RandomK@500 and prior Summary+AllDetail@500 are reused, never retrained or used as initialization.
One smoke5, then fresh500 from common step0/horizon4868; complete frozen native five evaluation, then stop.
A detached read-only F/D IoU observer preserves original loss tensors; tests prove identical losses,
all gradients and actualAdamW updates. Model source itself is unchanged.

Fixed100k sampling and common-step0 same-model redundancy/ambiguity audits precede training.
After500 the same1024 diagnostic IDs are used for all text views on the same final model.
No weights/data/indexes/embedding dumps are submitted. Commands/evidence/raw results are retained.

The prompt's approximate prior per-view CE/IoU values differ from saved All-Detail results.
Actual prior S_t2i=0.74935388,D_t2i=0.12626782,S-D IoU=0.83064634;
comparisons use those original saved values (evidence/baseline-diagnostic-correction.json).
