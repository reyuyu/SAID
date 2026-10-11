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

Both directions improve at R@1: +81 net correct image queries and +82 net correct caption queries. This is not improvement across all six metrics: I2T R@10 loses 3 correct queries and T2I R@5 loses 15. I2T R@5 gains 16; T2I R@10 gains 4. The small mixed changes provide no statistical significance or model replacement claim.

Net correct counts are aggregate differences, not paired correction counts. No significance claim.

This is one dataset, one seed, different training steps (3651 vs 4868), and a repeatedly observed public benchmark.
Historical HNS@3651 Score5 73.774418 is below E2@4868 73.812187. Full Flickr alone cannot justify replacing E2.

## Identity and engineering validation

Base commit: `e305275c24bb6a80c2e87a674d50239d6a0f22f5`. Dedicated exact-SHA HNS entrypoint reuses unchanged native functions; frozen source diff is empty.
Bare SHA256: `a60150575f7e8f7b41effb730c0f9d866757954833a240813d4695d9a1cfcb44`. Training SHA256: `83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07`.
Manifest SHA256: `bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc`; image inventory digest: `e4fef70dd3f73034970b1ec0e20d81a4b421d1871e350e3266e41b744d3ab159`.
Strict bare load: ViT-B/16, context 248, FP32; all bare tensors equal original training model. Historical export image/text max abs 0.
All 31,783 images independently verified against frozen bytes and PIL decoding; all 158,915 captions exactly match official token rows. Five explicit positives per image.
Model/input/features FP32, autocast off, cuDNN TF32 True, CUDA matmul TF32 False. Batch64; image tail39, text tail3. Query256/gallery4096; exact ties by ascending manifest candidate index.
18 CPU tests passed; strict checkpoint/source identity, streaming TopK and cache rejection tests passed. Formal shard/query recount and state immutability passed.
GPU 0 A100 80GB: text 246.096s, image 75.926s, scoring 9.661s, total 345.451s.
CUDA allocated peak 1.477 GiB; reserved 2.432 GiB; CPU RSS peak 3.054 GiB.
No OOM, non-finite feature/score, or image read error. Weights unchanged before/after. Native worker saves boundary tie diagnostics in results.
Independent HNS feature banks and query-level top11/hit evidence retained at `/root/said_hns_v1_step3651_flickrfull_v1`; no E2 embedding reuse. Existing E2 result reused without running E2.
No training, optimizer creation/updates, other model evaluation, DCI evaluation, downloads, or inference adjustment.
See DELIVERY_VERIFICATION.json for the verified results commit and GPU/process cleanup evidence. Synchronization of the final delivery commit is recorded separately in the server-local `GITHUB_SYNC_VERIFIED.json` linked there, avoiding a self-referential commit hash.
