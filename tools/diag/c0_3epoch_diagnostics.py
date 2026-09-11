"""Training diagnostics for the 3-epoch C_L03 run (windows + totals + cost)."""
import json
import os
import statistics

ROOT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl/c0_3epoch/L03'
KEYS = ['loss_smart_global_mean', 'loss_vssl_global_mean', 'lambda_U_effective',
        'vssl_valid_anchor_fraction', 'mask_s_keep_ratio', 'mask_u_keep_ratio',
        'said_retained_energy', 'unsaid_retained_energy', 'sidm_top1', 'dism_top1',
        'sec_per_step', 'peak_memory_gb']

records = [json.loads(line) for line in open(os.path.join(ROOT, 'salu_log.jsonl')) if line.strip()]
summary = json.load(open(os.path.join(ROOT, 'run_summary.json')))
config = json.load(open(os.path.join(ROOT, 'config.json')))

print('records=%d  completed=%s  epochs=%s  wall=%.1fs  mean_sec_per_step=%.4f  peak=%.2fGB'
      % (len(records), summary['completed_steps'], summary['epochs'], summary['wall_sec'],
         summary['mean_sec_per_step'], summary['peak_memory_gb']))
print('lambda_U=%s tau_U=%s u_weight_warmup_steps=%s lr_horizon=%s steps_per_epoch=%s'
      % (config['lambda_U'], config['tau_U'], config.get('u_weight_warmup_steps'),
         config['lr_horizon_steps'], config['loader_batches']))
print('initial_state_sha256=%s' % config['initial_state_sha256'])
print()
for lo, hi, label in ((1, 1217, 'epoch 1 (1-1217)'), (1218, 2434, 'epoch 2 (1218-2434)'),
                      (2435, 3651, 'epoch 3 (2435-3651)')):
    chunk = [row for row in records if lo <= row['completed_steps'] <= hi]
    if not chunk:
        continue
    print('%s  (%d logged records)' % (label, len(chunk)))
    for key in KEYS:
        values = [row[key] for row in chunk if isinstance(row.get(key), (int, float))]
        if values:
            print('   %-30s mean=%.6f  min=%.6f  max=%.6f'
                  % (key, statistics.mean(values), min(values), max(values)))
print()
nonfinite = [(row['completed_steps'], key) for row in records for key, value in row.items()
             if isinstance(value, float) and value != value]
digests = [(row['completed_steps'], row['rank_param_digest']) for row in records]
print('non-finite values = %d' % len(nonfinite))
print('rank_param_digest = %d logged steps, %d distinct, changed=%s'
      % (len(digests), len({d for _, d in digests}), digests[0][1] != digests[-1][1]))
print('stream digests: sample=%s caption=%s view_b_param=%s view_b_pixels_step0=%s'
      % (summary['sample_stream_sha256'][:16], summary['caption_stream_sha256'][:16],
         summary['view_b_param_stream_sha256'][:16], summary['view_b_pixels_sha256_step0']))
