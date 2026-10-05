# Evaluator audit

Only original native bare-student encoders and original retrieval metrics are used.
No condition gate, fusion, reranking, DCI Full substitution, or split replacement.

| Protocol | Images | Captions | Expected manifest SHA256 | Complete audit |
| --- | --- | --- | --- | --- |
| flickr30k_test1k | 1000 | 5000 | `113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc` | True |
| docci_test | 5000 | 5000 | `e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11` | True |
| long_dci_reconstructed | 7602 | 7602 | `8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b` | True |
| coco | 5000 | 25000 | See recovered historical identity evidence | True |
| urban1k | 1000 | 1000 | See recovered historical identity evidence | True |

COCO uses sorted `CocoCaptions.ids` and the first five original-order captions per image:
25014 source annotations are reduced by the original evaluator to 25000 candidates.
COCO similarity chunk is fixed at512 and uses the original row-wise argsort tie rule.
Urban1k uses original stem pairing, first caption line, and diagonal GT.
Urban1k archive SHA matches the historical pinned-revision report; all five image trees pass PIL verification.
COCO/Urban candidate-map hashes are fresh recovery evidence; no unpublished historical map hash is invented.
Flickr is the explicit test1K source. DOCCI is original Google test metadata, not translated captions.
Long-DCI is original `reconstruct_long_dci(extra_caption)`; 203 empty-extra-caption images are excluded
by the frozen original construction, not by a new split.

Historical extended manifest reconstruction:
```json
{
  "flickr30k_test1k.jsonl": {
    "sha256": "113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc",
    "rows": 5000,
    "historical_sha_matches": true
  },
  "docci_test.jsonl": {
    "sha256": "e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11",
    "rows": 5000,
    "historical_sha_matches": true
  },
  "long_dci_reconstructed.jsonl": {
    "sha256": "8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b",
    "rows": 7602,
    "historical_sha_matches": true,
    "source": "Original reconstruct_long_dci; extra_caption only; NOT DCI Full"
  }
}
```

A manifest SHA match does not imply that its image assets have finished downloading.
Full identity/candidate/GT checks and image-integrity checks are recorded in `evidence/evaluator-audit.json`.
Native step0 bare export and five-protocol small-sample evaluation passed: True.
This is a CPU subset sanity check, not five full-benchmark scores or a trained-step5 export.
No official full-benchmark scores were rerun during recovery.
