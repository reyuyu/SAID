"""Extract training diagnostics for a C0 tuning run (windows + totals)."""
import json
import os
import statistics
import sys

ROOT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl/c0_tuning'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'C_L03'
PATH = os.path.join(ROOT, TAG)

KEYS = ['loss_smart_global_mean', 'loss_vssl_global_mean', 'lambda_U_effective',
        'vssl_valid_anchor_fraction', 'mask_s_keep_ratio', 'mask_u_keep_ratio',
        'said_retained_energy', 'unsaid_retained_energy', 'sec_per_step', 'peak_memory_gb']

records = [json.loads(line) for line in open(os.path.join(PATH, 'salu_log.jsonl')) if line.strip()]
summary = json.load(open(os.path.join(PATH, 'run_summary.json')))
config = json.load(open(os.path.join(PATH, 'config.json')))

print('== %s' % TAG)
print('   config: lambda_U=%s tau_U=%s u_weight_warmup_steps=%s lr_horizon=%s steps_per_epoch=%s'
      % (config['lambda_U'], config['tau_U'], config.get('u_weight_warmup_steps'),
         config['lr_horizon_steps'], config['loader_batches']))
print('   initial_state_sha256=%s' % config['initial_state_sha256'])
print('   completed_steps=%s wall=%.1fs mean_sec_per_step=%.4f peak_memory_gb=%.2f'
      % (summary['completed_steps'], summary['wall_sec'], summary['mean_sec_per_step'],
         summary['peak_memory_gb']))
for field in ('sample_stream_sha256', 'caption_stream_sha256', 'view_b_param_stream_sha256',
              'view_b_pixels_sha256_step0'):
    print('   %-30s %s' % (field, summary.get(field)))

for lo, hi, label in ((1, 25, 'steps 1-25'), (26, 100, 'steps 26-100'),
                      (101, 250, 'steps 101-250'), (251, 500, 'steps 251-500')):
    chunk = [row for row in records if lo <= row['completed_steps'] <= hi]
    if not chunk:
        continue
    print('   %s (%d records)' % (label, len(chunk)))
    for key in KEYS:
        values = [row[key] for row in chunk if isinstance(row.get(key), (int, float))]
        if values:
            print('      %-30s mean=%.6f min=%.6f max=%.6f'
                  % (key, statistics.mean(values), min(values), max(values)))

nonfinite = [(row['completed_steps'], key) for row in records for key, value in row.items()
             if isinstance(value, float) and value != value]
digests = [(row['completed_steps'], row['rank_param_digest']) for row in records]
print('   non-finite values: %d' % len(nonfinite))
print('   rank_param_digest: %d logged steps, %d distinct, changed=%s'
      % (len(digests), len({digest for _, digest in digests}), digests[0][1] != digests[-1][1]))
