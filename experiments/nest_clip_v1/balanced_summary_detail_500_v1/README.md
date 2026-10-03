# Summary–Detail single-variable 500-update experiment

Base:14653c92c6da9d552a2b624ab169eaaa275cdde8. Full remains the baseline visible-prefix
string/tokens bitwise. Summary is the first cleaned raw sentence segment; Detail is all remaining
raw segments in order, compactly retokenized at248. Invalid raw captions with fewer than2 segments
have no constructed local caption; collated PAD sentinels are excluded by the unchanged valid masks.

Only text construction changes. Frozen Balanced-Stack-Patch B/16 model/loss/optimizer/scheduler/evaluators
remain unchanged. Common step0 is reused. One smoke5, then an independent500 from step0 with horizon4868,
then all five frozen native datasets; no baseline retraining or automatic4868 run.

Baseline:69.900394% Score5_R1. Config, preflight, sampling correctness and whole-corpus statistics
are included. Data audit distinguishes raw captions, baseline visible Full and truncated effective Detail.
About5.79% of raw Detail strings extend beyond the packed visible Full; this is explicit,
not an assumption that their post-budget token sets are nested. Mathematical inclusion loss is unchanged.

Run stages with the said-repro Python environment:
python -m experiments.nest_clip_v1.balanced_summary_detail_500_v1.run preflight
python -m experiments.nest_clip_v1.balanced_summary_detail_500_v1.run smoke
python -m experiments.nest_clip_v1.balanced_summary_detail_500_v1.run formal
python -m experiments.nest_clip_v1.balanced_summary_detail_500_v1.run evaluate

Large checkpoints/indexes/data/embedding caches remain in /root/lk_projects/SAID-nest-clip-v1 and SAID-assets.
