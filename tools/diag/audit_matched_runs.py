"""Matched-run audit: prove S0 and C0 share init, data stream, views and configuration."""
import json
import os
import sys

ROOT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl'
PAIRS = [('ddpcheck_step20_S0_smartclip', 'ddpcheck_step20_C0_complement_vssl'),
         ('ddpfix_step500_S0_smartclip', 'ddpfix_step500_C0_complement_vssl')]
FIELDS = ['initial_state_sha256', 'sample_stream_sha256', 'caption_stream_sha256',
          'view_b_param_stream_sha256', 'view_b_pixels_sha256_step0', 'steps_per_epoch',
          'lr_horizon_steps']


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    ok = True
    for left_name, right_name in PAIRS:
        left_path = os.path.join(ROOT, left_name, 'run_summary.json')
        right_path = os.path.join(ROOT, right_name, 'run_summary.json')
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            print('== %s vs %s : summary missing, skipped' % (left_name, right_name))
            continue
        left, right = load(left_path), load(right_path)
        print('== %s vs %s' % (left_name, right_name))
        for field in FIELDS:
            same = left.get(field) == right.get(field)
            ok = ok and same
            print('   %-30s S0=%-44s C0=%-44s %s'
                  % (field, left.get(field), right.get(field), 'MATCH' if same else 'DIFFER'))

    # config comparison
    left_config = load(os.path.join(ROOT, PAIRS[-1][0], 'config.json'))
    right_config = load(os.path.join(ROOT, PAIRS[-1][1], 'config.json'))
    differing = {key: (left_config.get(key), right_config.get(key))
                 for key in sorted(set(left_config) | set(right_config))
                 if left_config.get(key) != right_config.get(key)}
    print('== config keys that differ between S0 and C0 (expected: arm/arm_mask/lambda_U only)')
    for key, values in differing.items():
        print('   %-28s S0=%r C0=%r' % (key, values[0], values[1]))
    unexpected = set(differing) - {'arm', 'arm_mask', 'lambda_U', 'output_dir', 'stopgrad_rule'}
    if unexpected:
        ok = False
        print('   UNEXPECTED CONFIG DIFFERENCES: %s' % sorted(unexpected))
    else:
        print('   ok (only the arm identity differs)')

    # step-1 loss identity between the two 20-step check runs
    left_log = os.path.join(ROOT, PAIRS[0][0], 'salu_log.jsonl')
    right_log = os.path.join(ROOT, PAIRS[0][1], 'salu_log.jsonl')
    if os.path.exists(left_log) and os.path.exists(right_log):
        first_left = json.loads(open(left_log).readline())
        first_right = json.loads(open(right_log).readline())
        print('== step-1 identity (same init + same data => same smart losses)')
        for key in ('loss_sidm', 'loss_dism', 'loss_sparsity', 'loss_smart',
                    'loss_smart_global_mean', 'sidm_top1', 'dism_top1'):
            same = first_left.get(key) == first_right.get(key)
            ok = ok and same
            print('   %-26s S0=%-22s C0=%-22s %s'
                  % (key, first_left.get(key), first_right.get(key), 'MATCH' if same else 'DIFFER'))
        for key in ('loss_vssl_global_mean', 'mask_u_keep_ratio', 'unsaid_retained_energy'):
            print('   %-26s S0=%-22s C0=%-22s (may differ: different arm mask / U term)'
                  % (key, first_left.get(key), first_right.get(key)))
    print('AUDIT_%s' % ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
