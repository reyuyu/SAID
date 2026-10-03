# Official Beta-CLIP evaluation: stopped at download gate

Official code is independently cloned at /root/lk_projects/B-CLIP-official and
pinned to v1 commit7be4476f84654b0febe7224d5788868d61ecae8b. SAID model source
is untouched. The beta-clip-official Python3.10 venv is isolated from said-repro.

Both official CE/BCE Google Drive downloads timed out, including a bounded
public no-cookies retry. A direct HTTPS connection check also timed out. No
checkpoint bytes were obtained. Per the user's explicit stop condition, no
Urban reproduction, native-adapter verification or five-dataset evaluation runs.
No alternative mirror, random initialization, original CLIP weights or fine-tuning
is substituted. All unknown checkpoint/inference fields are null, not guessed.

Evidence: CHECKPOINT_INVENTORY.json, environment.json, commands/ and raw/*/STATUS.json.
Source audit: OFFICIAL_SOURCE.md. Urban status: OFFICIAL_URBAN_REPRO.md.
The existing SAID four-epoch reference is reused in SAID_PROTOCOL_RESULTS.json
and BETA_CLIP_COMPARISON.md, with BetaCLIP-minus-SAID as the only delta direction.
No beta-CLIP score or comparison delta exists yet.

Reproduction commands are preserved verbatim as structured arrays in commands/.
Official-source/checkpoint availability must be restored before the unexecuted
model, tokenizer, strict-loader, CLS/TCI and frozen-protocol checks can proceed.
