"""Summarise a CVSSL run: summary json, step records, and the first-N-step online check."""
import json
import os
import sys

PATH = sys.argv[1]
CHECK_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print('== run_summary.json')
summary_path = os.path.join(PATH, 'run_summary.json')
if os.path.exists(summary_path):
    with open(summary_path) as handle:
        summary = json.load(handle)
    for key in sorted(summary):
        print('   %-32s %s' % (key, summary[key]))
else:
    print('   MISSING (run still going?)')

print('== salu_log.jsonl')
log_path = os.path.join(PATH, 'salu_log.jsonl')
if not os.path.exists(log_path):
    print('   MISSING')
    raise SystemExit(0)
records = [json.loads(line) for line in open(log_path) if line.strip()]
print('   records: %d  first_step=%s last_step=%s'
      % (len(records), records[0]['completed_steps'], records[-1]['completed_steps']))
keys = ['loss_smart_global_mean', 'loss_sidm', 'loss_dism', 'loss_sparsity',
        'loss_vssl_global_mean', 'vssl_valid_anchor_fraction', 'mask_s_keep_ratio',
        'mask_u_keep_ratio', 'said_retained_energy', 'unsaid_retained_energy']
header = 'step  ' + ' '.join('%-12s' % key[:12] for key in keys) + '  lr        mask_lr   sec    mem'
print(header)
for record in records:
    if record['completed_steps'] > CHECK_STEPS and record['completed_steps'] % 50 != 0:
        continue
    row = '%-5d ' % record['completed_steps']
    for key in keys:
        value = record.get(key)
        row += '%-12s ' % ('%.6f' % value if isinstance(value, (int, float)) else str(value))
    row += ' %.3e %.3e %5.2f %5.1f' % (record.get('lr', 0.0), record.get('mask_lr', 0.0),
                                       record.get('sec_per_step', 0.0),
                                       record.get('peak_memory_gb', 0.0))
    print(row)

print('== first %d steps online check' % CHECK_STEPS)
problems = []
for record in records:
    if record['completed_steps'] > CHECK_STEPS:
        continue
    for key, value in record.items():
        if isinstance(value, float) and value != value:
            problems.append((record['completed_steps'], key, value))
    if record.get('rank_param_digest') in (None, ''):
        problems.append((record['completed_steps'], 'rank_param_digest', None))
print('   non-finite values: %d' % len(problems))
for entry in problems[:10]:
    print('   %s' % (entry,))
steps_seen = sorted({record['completed_steps'] for record in records})
print('   steps logged: %s' % steps_seen[:24])
print('   max peak memory: %.1f GB' % max(record.get('peak_memory_gb', 0.0) for record in records))
print('   mean sec/step: %.2f' % (sum(record.get('sec_per_step', 0.0) for record in records)
                                  / max(len(records), 1)))
