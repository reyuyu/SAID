"""Summarise a C1-TCR run: losses, decoder behaviour, input-dependency diagnostics, cost."""
import json
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else '/root/SAID-c1-tcr/runs_salu/said_cls_c1'
LOG = os.path.join(PATH, 'salu_log.jsonl')
rows = []
for line in open(LOG):
    if line.startswith('LOG '):
        rows.append(json.loads(line[4:]))
print('records: %d  steps %s..%s' % (len(rows), rows[0]['completed_steps'],
                                     rows[-1]['completed_steps']))

MAIN = ['loss_smart', 'loss_rec', 'loss_rec_global_mean', 'cos_pred_reference',
        'prediction_norm', 'reference_norm', 'native_global_to_reference_cos',
        'rec_valid_fraction', 'r_u_norm', 'g_norm', 'coordinate_fraction_said',
        'coordinate_fraction_unsaid', 'retained_energy_said', 'retained_energy_unsaid',
        'lambda_rec', 'decoder_lr', 'sec_per_step', 'peak_memory_gb']
print()
print('%-8s %s' % ('step', ' '.join('%s' % key[:13] for key in MAIN)))
for row in rows:
    cells = []
    for key in MAIN:
        value = row.get(key)
        cells.append('%.5f' % value if isinstance(value, float) else str(value)[:13])
    print('%-8s %s' % (row['completed_steps'], ' '.join(cells)))

DIAG = [key for key in rows[-1] if key.startswith('diag_')]
if DIAG:
    print()
    print('== input-dependency diagnostics (least-squares error vs the frozen reference)')
    for row in rows:
        if not any(key in row for key in DIAG):
            continue
        print('   step %s' % row['completed_steps'])
        for key in sorted(DIAG):
            value = row.get(key)
            print('      %-32s %s' % (key, ('%.6f' % value) if isinstance(value, float) else value))

summary_path = os.path.join(PATH, 'run_summary.json')
if os.path.exists(summary_path):
    print()
    print('== run_summary')
    summary = json.load(open(summary_path))
    for key in sorted(summary):
        print('   %-40s %s' % (key, summary[key]))
