# Summary–Detail@500: NEGATIVE

Fixed Summary–Detail decomposition does not improve the matched RandomK baseline at500.
All500 updates and five frozen native retrieval datasets completed. Matched baseline is reused;
common step0 and horizon4868 are preserved. No full4868 run is started.

Read [SUMMARY_DETAIL_500_REPORT.md](SUMMARY_DETAIL_500_REPORT.md) and [RESULTS.json](RESULTS.json)
for every directional R1/R5/R10, deltas, checkpoint hashes, matching and resource evidence.
[DATA_SAMPLING_AUDIT.md](DATA_SAMPLING_AUDIT.md) distinguishes raw captions from packed Full,
and records complete-corpus quantiles and actual Detail truncation/coverage.
Code/logs use tokens_o/e as unchanged internal slots; method/report names are Full/Summary/Detail.
Commands are structured arrays in commands/. Source base14653c9, branch
codex/nest-balanced-summary-detail-500-v1. Large weights/data/caches are outside Git.
