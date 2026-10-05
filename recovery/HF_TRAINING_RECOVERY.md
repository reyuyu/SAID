# Original COCO/LLaVA HF recovery

Updated UTC: 2026-10-04T18:59:31.831298+00:00

No training/smoke commands in this supervisor. The SA1B supervisor is untouched.
All old curl partials are retained; SDK resumes use separate revision-pinned directories.
hf_xet is installed/enabled. Mirror HEAD currently supplies no Xet metadata; actual transfer uses SDK HTTP fallback.
Rate metric: SDK payload callback deltas / wall time, logged every 30 seconds.
Sources stop after 600 consecutive low-rate seconds or a 600-second payload average below 1MiB/s, including stalls.
Alternative ZIPs require identical SHA256/size. The 600-second average prevents brief buffered bursts evading the slow-source limit.

## COCO
Repo: `danasone/coco2017`
Pinned revision: `2e5331942f3d99a6f3f09591cdf4694abf2642e5`
Filename: `train2017.zip`
Expected SHA256: `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929`
Expected archive bytes: 19336861798
Status: complete; phase: complete
SDK payload bytes (including HTTP resume): 0/19336861798
Latest payload throughput MiB/s: 0.000
Verified/extracted required images: 118287/118287

## LLAVA
Repo: `332F/LLaVA-Pretrain`
Pinned revision: `698a737896f11f8cd9548d2989d2c2875405e955`
Filename: `images.zip`
Expected SHA256: `05459d8cb059bd32322b1c466c1cbd4568b09b1ce1db748425b7977236912660`
Expected archive bytes: 27356108382
Status: running; phase: downloading
SDK payload bytes (including HTTP resume): 21653094400/27356108382
Latest payload throughput MiB/s: 15.316
Verified/extracted required images: 0/558128

## Frozen training index
Snapshot UTC: 2026-10-04T18:59:31.809277+00:00
Records: 1245901
Missing paths: 1038262
Live rates: `evidence/hf-training-throughput.jsonl`
Worker logs: `evidence/hf-training-coco.log`, `evidence/hf-training-llava.log`
