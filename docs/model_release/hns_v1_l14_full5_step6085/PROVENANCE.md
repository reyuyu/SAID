# Provenance

The model was trained from the audited ViT-L/14 common Step0 on the branch `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1`. The endpoint completed 6085 optimizer updates with scheduler horizon 6085. The source evaluation commit is `8a0b6e8894c166d40bced43625032fa7bbca61fe`.

The released bare student is the strict native encoder export from `evaluation_retry/bare_student_step006085.pt`. The full training checkpoint is `formal6085_retry/step006085.pt` and contains the model, adapter, optimizer, scheduler, RNG and sampler recovery state. The common Step0 is the independently audited initialization used by the run.

All three SHA256 values, model dimensions and native-load checks are recorded in `MODEL_IDENTITY.json`, `RELEASE_AUDIT.json` and `CHECKSUMS.sha256`.
