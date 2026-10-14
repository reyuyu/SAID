# Open questions and provenance limits

The following items are intentionally left unresolved. They should be settled before using the corresponding statement as a definitive paper claim.

1. **B/16 headline identity.** The formal HNS-v1 commit `45468e1`, `experiments/nest_clip_v1/nested_d3_hns_full_v1/FULL_RESULTS.json`, reports step3651 Score5 `73.638084`, J_long3 `78.014807`, J_long `86.915002`, and Short4 `67.073`. The later ABCDE archive commit `4ef949c` labels a component row `A.Full` at the same nominal step with Score5 `73.774418`, J_long3 `78.188029`, J_long `87.040003`, and Short4 `67.154`. The branch, checkpoint, and evaluation provenance are not identical in the files currently available, so this document preserves both identities instead of silently replacing one with the other.

2. **ABCDE B status conflict.** `said_b16_component_ablation_stop3651_v1/FINAL_REPORT.md` records that the B arm stopped at update 2261 and was not evaluated, whereas `archive/said-abcde-results-v1` reports B evaluated at step3651 and marks the result as passing. The archive result is presented as an archived comparison, but the underlying training-history discrepancy requires a provenance decision for publication.

3. **Exact wall-clock duration.** Runtime evidence records GPU model, rank count, memory, and health checks, but a single comparable end-to-end training duration is not uniformly available for both B/16 and L/14. No duration is asserted in the LaTeX paragraph.

4. **Text encoder depth/head wording.** The implementation is standard OpenAI CLIP, but the compact evidence set does not preserve a single audited table for every text-transformer layer/head count. The documents therefore identify tokenizer, context length, and encoder family without adding an independently reconstructed text-depth claim.

5. **Evaluation scheduler uniformity.** Later audit tooling documents four concurrent GPU jobs followed by Urban on the first released GPU. Historical reports do not prove that every earlier evaluation used exactly this scheduler. The extended document limits the four-GPU-parallel wording to runs with execution evidence.

6. **Caption protocol comparisons.** Long-DCI is a reconstructed fixed 7,602-pair protocol and is not DCI-full. Flickr30k-Test1K and Flickr30k-Full are separate candidate pools. A paper comparison must preserve these names and should not infer caption equivalence with external implementations without a per-sample audit.

7. **Data path versus paper description.** The audits use a local ShareGPT4V mirror and retain content hashes, but private server paths are deliberately omitted from the paper-facing files. A public release should provide a neutral manifest or instructions that reproduce the hashed training stream.

8. **Reported metrics are exploratory.** The public retrieval sets have been repeatedly observed across the project. The numbers support implementation comparison, but they should not be described as unbiased held-out model selection evidence without an independent validation protocol.
