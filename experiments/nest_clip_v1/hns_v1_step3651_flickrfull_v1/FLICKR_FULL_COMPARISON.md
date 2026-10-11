# Flickr30k-Full: HNS-v1@3651 vs E2-Uniform@4868

| Direction | E2 (%) | HNS (%) | HNS − E2 (pp) | E2 correct | HNS correct | Net correct | Queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| I2T R@1 | 56.712708051 | 56.967561275 | +0.254853223 | 18025 | 18106 | +81 | 31783 |
| I2T R@5 | 78.969889564 | 79.020230941 | +0.050341377 | 25099 | 25115 | +16 | 31783 |
| I2T R@10 | 85.866658276 | 85.857219268 | -0.009439008 | 27291 | 27288 | -3 | 31783 |
| T2I R@1 | 36.725293396 | 36.776893308 | +0.051599912 | 58362 | 58444 | +82 | 158915 |
| T2I R@5 | 59.293332914 | 59.283893906 | -0.009439008 | 94226 | 94211 | -15 | 158915 |
| T2I R@10 | 68.391907624 | 68.394424692 | +0.002517069 | 108685 | 108689 | +4 | 158915 |

Bidirectional R@1 mean: E2 46.719000724%; HNS 46.872227291%; delta +0.153226568 pp.

Both R@1 directions improve (+81 image queries, +82 caption queries). Higher-K changes are mixed: I2T R@10 −3 queries; T2I R@5 −15 queries. This is not dominance across all six metrics or evidence sufficient to replace E2.

Net correct counts are aggregate differences, not paired correction counts. No significance claim.

This is one dataset, one seed, different training steps (3651 vs 4868), and a repeatedly observed public benchmark.
Historical HNS@3651 Score5 73.774418 is below E2@4868 73.812187. Full Flickr alone cannot justify replacing E2.
