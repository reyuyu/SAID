# Sema-CLIP ViT-L/14 DCI-full evaluation

This is a read-only native retrieval evaluation of HNS-v1 L/14 Full4 at step
4868. The bare student SHA256 is
`4a1e5ed69b7aa728d7189518b9f285052e8293290963e079d886ddcab5fe819e`; the
source branch is `experiment/hns-v1-l14-full4epoch4868-v1` at commit
`c2b70f0805124e79e9e4dcb528b176189124843f`.

| Model | DCI I2T R@1/5/10 | DCI T2I R@1/5/10 |
|---|---|---|
| Sema-CLIP L/14 @4868 | 53.1839 / 72.7354 / 79.6413 | 56.7841 / 75.7591 / 81.3709 |

The run used 7,805 images and 7,805 captions, GPU0 only, batch size 64, and
took 921.98 seconds. GPU1–3 remained unused. The Random Caption process was
observed in CPU-only statistics and was not stopped or modified. No training,
checkpoint mutation, masking inference, reranking, ensemble, or TTA occurred.

The result uses the frozen manifest SHA
`14530fb8bf3c7b4a75bb451412d4562c548f9ca1bdbe1fdbf39d6ee30fb4de24`. The
caption-field discrepancy with a freshly concatenated short-plus-extra
manifest is documented in `DCI_PROTOCOL_AUDIT.md`; the primary score follows
the supplied historical manifest exactly.

Existing Long-DCI results are a separate protocol and are not substituted for
this DCI-full result.
