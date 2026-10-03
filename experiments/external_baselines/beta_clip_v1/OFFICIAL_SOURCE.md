# Official source audit

Repository: [fzohra/B-CLIP](https://github.com/fzohra/B-CLIP).
Requested branch: v1. Verified detached HEAD: `7be4476f84654b0febe7224d5788868d61ecae8b`.
Local checkout: `/root/lk_projects/B-CLIP-official`, clean working tree.
Source SHA256 fingerprints are retained in CHECKPOINT_INVENTORY.json.

The [official README](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/README.md) and
[download script](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/download_ckpts.sh) advertise the two Drive IDs used in
commands/. Training examples specify B16,248 context, CE/BCE variants and beta0.5.
These are script expectations only; actual checkpoint args/architecture remain
unverified because neither file was downloaded.

The [official evaluator](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/validate_urban1k_distributed.py) exposes separate
retrieval_acc_cls_t2i/i2t and retrieval_acc_tci_t2i/i2t results. It builds its own
Urban dataset and224 bicubic Resize/CenterCrop/OpenAI normalization transform.
[Tokenizer](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/tokenizer.py) is official SimpleTokenizer; SAID tokenization
must not replace it. [Fine-tuned loading](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/main_clip_ft.py) strips module
prefixes, applies official state conversion and positional-extension setup,
then calls strict=True. No missing/unexpected-key or strict-load result is
claimed without weights.

[Model source](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/models_tome.py) has a native encode_image and encode_text
path, but conditioned evaluation uses encode_image_by_block. The last/intermediate/
mixed-block flags must be checked against real args and embeddings before
asserting CLS ranking equivalence. No structural equivalence is assumed here.

Official L14 architecture support: **yes**, CLIP_VITL14_OPENAI declares visual
width1024, text/embedding width768 and12 text heads at224.
Separate official released fine-tuned L14 checkpoint found: **no**, within the
pinned README/download-script scope. The two unavailable files' actual architecture
is unknown. No original OpenAI or third-party L14 weights are substituted.

Paper reference: [arXiv2512.12678](https://arxiv.org/abs/2512.12678).
The pinned README cites that paper; this source audit does not independently
establish conference acceptance or attribute README Urban scores to CLS versus TCI.
