# Common step0 audit

Expected initializer SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Observed initializer SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Official OpenAI CLIP B/16 SHA256: `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
Provenance: OpenAI CLIP + original random MaskNetwork; seed0; no training; empty optimizer state.
Context248 construction is the original `longclip.load_from_clip` and repository model code.
Step0 audit passed: True.
Historical file byte identity: True.

The semantic audit compares all keys/shapes/dtypes/tensor content against an independently
rebuilt pinned original seed0 constructor, RNG state, and exact CPU image/text output digests.
Per-tensor digests are in `evidence/step0-audit.json`. The historical tensor manifest itself was not
published; if file SHA differs, do not claim the original lost file was independently inspected.
No model parameters or constructor are modified to chase a serialization SHA.
