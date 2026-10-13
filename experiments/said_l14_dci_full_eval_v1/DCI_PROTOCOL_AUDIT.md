# DCI-full protocol audit

The evaluation used the frozen 7,805-row manifest whose SHA256 is
`14530fb8bf3c7b4a75bb451412d4562c548f9ca1bdbe1fdbf39d6ee30fb4de24`.
The source archive is the local DCI release under `local_assets`, with 7,805
matching image files.  The evaluator used the native ViT-L/14 image and text
encoders, the repository tokenizer with context length 248, the repository
image preprocessing, L2-normalized embeddings, and the full 7,805 by 7,805
candidate pool. No mask, fusion, reranking, ensemble, or TTA was used.

There is a historical manifest inconsistency that is recorded rather than
silently corrected. The supplied frozen SHA corresponds to rows whose
`caption` field is the source `short_caption` value. Rebuilding rows with
`short_caption + " " + extra_caption` produces 7,805 rows but SHA256
`ebbb5a96d71a61ba8818b5e045e4a59ae249578d7e2056e8f7379a527ed68092`, so it is
not the supplied frozen protocol. The frozen captions and the concatenated
captions differ for 7,717/7,805 rows. With the 248-token tokenizer, the frozen
captions have mean effective length 28.04, median 25, maximum 217, and 0
truncated rows; concatenated captions have mean 154.66, median 149, maximum
248, and 1,394 rows reaching the context limit.

The official DeBias-CLIP repository was checked for a DCI loader or caption
construction implementation. Its checked source tree does not contain a DCI
loader or `splits.json` comparison implementation, so exact DeBias-CLIP
per-row equivalence is **UNVERIFIED**. The local DCI `splits.json` and source
annotations were independently counted; no resampling or split reconstruction
was used for the primary result. HyFL-CLIP caption identity is also
**UNVERIFIED**.

The primary score therefore represents the historically frozen SAID DCI-full
manifest. It must not be described as a new short-plus-extra result.
