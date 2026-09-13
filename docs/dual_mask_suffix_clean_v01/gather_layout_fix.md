# Suffix gather layout correction and formal masked run

The Gloo failure recorded at `bb2d259` was reproduced with PyTorch 2.5.1+cu124:
the original two-dimensional suffix gather gave a 56.679512 maximum absolute
gradient difference and a 0.002000004 one-step parameter difference despite equal
scalar losses. Transposed scalar scores can send non-contiguous gradients into
the differentiable gather. Flattening only the new suffix communication to a
contiguous one-dimensional tensor, then restoring rank-ordered rows, reduced the
same probe's absolute gradient error to 3.814697e-6 and update difference to zero.
No S0 communication, candidate ordering, loss, detach boundary, or W/V factor changed.

The finite-only acceptance checks from the earlier evidence are superseded by
fixed elementwise assertions: gradient/value atol=2e-4, rtol=2e-5; updated
parameters atol=2e-6, rtol=2e-5. Relative report errors mean maximum absolute
difference divided by max(maximum absolute reference, 1e-12), separately from
absolute errors. Complete per-parameter errors and denominators are recorded in
`ddp_total_gloo.json` and `ddp_total_nccl.json`.

Both backends pass the production `ddp_model(...).loss_total.backward()` versus
an independent full-global-batch reference with non-open test gate parameters.
The test uses the original S0 mathematical formula plus explicit suffix pairs
conditioned on candidate j, with no collectives inside the reference. It compares
all visual/text/S0-mask/gate parameters, all three AdamW optimizer updates, and
rank parameter agreement over valid counts [1,3], [0,3], [0,0], [1,0], [2,2].
The production gate initialization remains all open. Standalone pair tests use
independent image/text gradient leaves and gate copies for 2x3 and 3x5 pools.

Formal authorization is masked only, 500 updates, followed by frozen native
COCO canonical and Urban-1k evaluation. No native comparison arm is trained.

```bash
NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/said-smartclip/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29610 \
  train/train_dual_mask_suffix.py --suffix-mode masked --run-type formal \
  --batch-size 256 --epochs 3 --max-steps 500 --seed 0 --num-workers 8 \
  --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state /root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt \
  --output-dir runs_salu/dual_mask_suffix_masked_formal500_flatgather_v01
```

Only observation and provenance fields were added to the trainer: initialization
checkpoint, actual counters, synchronized time, per-rank memory/gradient health,
and cumulative sample/prefix/suffix stream SHA-256. Sampling and schedules retain
their existing code paths. Initial and final file SHA-256, actual exit codes,
evaluation results, and elapsed time are recorded after the run in the report.
