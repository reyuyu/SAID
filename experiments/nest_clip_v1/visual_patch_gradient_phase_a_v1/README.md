# Visual Patch gradient Phase A

This run stopped at the E2@500 global16 numerical preflight. Read the report;
there are no formal global1024 or@1217 gradient conclusions. No training ran.

CPU tests:

```bash
.venv/bin/python -m pytest -q tests/test_visual_patch_gradient_phase_a.py
```

Rebuild the stopped report from existing local evidence in a fresh directory:

```bash
.venv/bin/python -m recovery.collect_visual_patch_gradient_phase_a \
  --runtime /root/said_visual_patch_gradient_phaseA_v1 \
  --output-dir /path/to/fresh/report --stopped-preflight
```

The GPU probe command below is documented for reproducibility. This experiment
has stopped; no further GPU execution or Phase B training was started after the
failed numerical gate. A future run requires user direction.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m recovery.visual_patch_gradient_phase_a --checkpoint-node 500 \
  --output-dir /path/to/fresh/node500
```

Node1217 is supported by the same probe. It was not executed in this run.
The script has no optimizer, parameter updates or checkpoint writes, refuses
checkpoint/source/config drift and never retries after failed acceptance.
Every output is exclusive-create. Production code, tokenizer, data, BF16,
checkpointing, loss coefficients and four-rank reduction math are unchanged.

The first execution did not persist gradient norms before failing its numerical
gate. Published files explicitly mark these missing measurements UNVERIFIED.
The probe now saves failed numerical receipts before stopping; this reporting
revision has been CPU tested, not rerun on GPU. The original error and all
previous launch logs remain server-local.
