"""Extract the training-side quantities for the report (losses, masks, energy, cost, sync)."""
import json
import os
import statistics
import sys

ROOT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl'
S0 = os.path.join(ROOT, 'ddpfix_step500_S0_smartclip')
C0 = os.path.join(ROOT, 'ddpfix_step500_C0_complement_vssl')
CHECK = {'S0': os.path.join(ROOT, 'ddpcheck_step20_S0_smartclip'),
         'C0': os.path.join(ROOT, 'ddpcheck_step20_C0_complement_vssl')}

KEYS = ['loss_smart_global_mean', 'loss_vssl_global_mean', 'vssl_valid_anchor_fraction',
        'mask_s_keep_ratio', 'mask_u_keep_ratio', 'said_retained_energy',
        'unsaid_retained_energy', 'vssl_ab_top1', 'sidm_top1', 'dism_top1', 'lr', 'mask_lr']


def records(path):
    return [json.loads(line) for line in open(os.path.join(path, 'salu_log.jsonl')) if line.strip()]


def window(rows, lo, hi):
    return [row for row in rows if lo <= row['completed_steps'] <= hi]


def stat(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return (statistics.mean(values), min(values), max(values))


def main():
    payloads = {}
    for arm, path in (('S0', S0), ('C0', C0)):
        rows = records(path)
        summary = json.load(open(os.path.join(path, 'run_summary.json')))
        payloads[arm] = (rows, summary)
        print('== %s  records=%d  steps %d..%d' % (arm, len(rows), rows[0]['completed_steps'],
                                                   rows[-1]['completed_steps']))
        print('   completed_steps=%s wall_sec=%.1f mean_sec_per_step=%.4f peak_memory_gb=%.2f'
              % (summary['completed_steps'], summary['wall_sec'], summary['mean_sec_per_step'],
                 summary['peak_memory_gb']))
        for lo, hi, label in ((1, 20, 'steps 1-20'), (21, 100, 'steps 21-100'),
                              (101, 400, 'steps 101-400'), (401, 500, 'steps 401-500')):
            chunk = window(rows, lo, hi)
            if not chunk:
                continue
            print('   %s' % label)
            for key in KEYS:
                value = stat(chunk, key)
                if value:
                    print('      %-30s mean=%.6f min=%.6f max=%.6f' % (key, value[0], value[1],
                                                                       value[2]))
        # cross-rank synchronisation evidence: one rank_param_digest per logged step
        digests = [(row['completed_steps'], row['rank_param_digest']) for row in rows]
        unique = len({digest for _, digest in digests})
        print('   rank_param_digest: %d logged steps, %d distinct (a repeat would mean no update)'
              % (len(digests), unique))
        first, last = digests[0][1], digests[-1][1]
        print('   first digest=%s last digest=%s changed=%s' % (first, last, first != last))
        nonfinite = [(row['completed_steps'], key) for row in rows for key, value in row.items()
                     if isinstance(value, float) and value != value]
        print('   non-finite values: %d' % len(nonfinite))

    print()
    print('== 20-step online check runs (separate launches, arms stopped at 20)')
    for arm, path in CHECK.items():
        rows = records(path)
        summary = json.load(open(os.path.join(path, 'run_summary.json')))
        print('   %s completed_steps=%s max loss_smart=%.4f  min=%.4f  max loss_smart_global=%.4f'
              % (arm, summary['completed_steps'],
                 max(row['loss_smart'] for row in rows),
                 min(row['loss_smart'] for row in rows),
                 max(row['loss_smart_global_mean'] for row in rows)))
        nonfinite = [(row['completed_steps'], key) for row in rows for key, value in row.items()
                     if isinstance(value, float) and value != value]
        print('      non-finite=%d  checkpoints=%s'
              % (len(nonfinite), sorted(name for name in os.listdir(path) if name.endswith('.pt'))))

    # the two 500-step checkpoints must be different models
    import torch
    left = torch.load(os.path.join(S0, 'cvssl_S0_smartclip_step000500.pt'), map_location='cpu',
                      weights_only=False)['model']
    right = torch.load(os.path.join(C0, 'cvssl_C0_complement_vssl_step000500.pt'),
                       map_location='cpu', weights_only=False)['model']
    worst, changed = 0.0, 0
    for key in left:
        if key in right and torch.is_tensor(left[key]):
            difference = float((left[key].float() - right[key].float()).abs().max())
            worst = max(worst, difference)
            changed += 1 if difference > 0 else 0
    print()
    print('== S0@500 vs C0@500 weights differ in %d/%d tensors, max_abs=%.6e'
          % (changed, len(left), worst))


if __name__ == '__main__':
    main()
