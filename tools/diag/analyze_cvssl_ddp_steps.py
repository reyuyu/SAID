"""Stage 4 analysis: DDP-synchronised per-term gradients + optimizer step equivalence.

Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.
"""

import json
import sys

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cvssl_diag/stage4'


def load(name):
    with open('%s/%s' % (OUT, name)) as handle:
        return json.load(handle)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def rows(reference, combined, key='grads'):
    out = []
    for name in sorted(reference):
        base, other = reference[name], combined.get(name)
        if base is None or other is None:
            continue
        out.append((name, float(tensor(base).abs().max()), float(tensor(other).abs().max()),
                    float((tensor(base) - other).abs().max())))
    return out


def report(tag, reference, combined, key='grads'):
    data = rows(reference, combined, key)
    worst_abs = max(row[3] for row in data)
    worst_rel = max(row[3] / max(row[1], 1e-30) for row in data)
    print('   %-34s worst_abs=%.3e  worst_rel(/(max|g|))=%.3e' % (tag, worst_abs, worst_rel))
    return worst_abs, worst_rel


def mean_of_ranks(runs, key='grads'):
    total = {}
    for run in runs:
        entry = run['steps'][0][key]
        for name, value in entry.items():
            if value is None:
                continue
            piece = tensor(value) / float(len(runs))
            total[name] = piece if name not in total else total[name] + piece
    return total


print('#' * 78)
print('# Per-term gradient, taken through the REAL DDP path (rank0 vs rank1 must be bit-equal)')
print('#' * 78)
for term in ('sidm', 'dism', 'sparse', 'smart', 'total'):
    runs = [load('s0_%s_rank%d.json' % (term, rank)) for rank in range(2)]
    single = load('s0_%s_single.json' % term)
    worst_rank = 0.0
    for name in single['steps'][0]['grads']:
        left, right = runs[0]['steps'][0]['grads'][name], runs[1]['steps'][0]['grads'][name]
        if left is None or right is None:
            continue
        worst_rank = max(worst_rank, float((tensor(left) - tensor(right)).abs().max()))
    absolute, relative = report('S0 %-6s DDP mean vs single' % term, single['steps'][0]['grads'],
                                mean_of_ranks(runs))
    print('      rank0-vs-rank1 maxabs = %.3e' % worst_rank)
for term in ('u', 'total'):
    runs = [load('a_%s_rank%d.json' % (term, rank)) for rank in range(2)]
    single = load('a_%s_single.json' % term)
    worst_rank = 0.0
    for name in single['steps'][0]['grads']:
        left, right = runs[0]['steps'][0]['grads'][name], runs[1]['steps'][0]['grads'][name]
        if left is None or right is None:
            continue
        worst_rank = max(worst_rank, float((tensor(left) - tensor(right)).abs().max()))
    report('C0/A %-6s DDP mean vs single' % term, single['steps'][0]['grads'],
           mean_of_ranks(runs))
    print('      rank0-vs-rank1 maxabs = %.3e' % worst_rank)

print()
print('#' * 78)
print('# Optimizer step equivalence (2 real steps: grads + params + AdamW state + digests)')
print('#' * 78)
for case in ('S0', 'A'):
    single = load('%s_opt_single.json' % case)
    runs = [load('%s_opt_rank%d.json' % (case, rank)) for rank in range(2)]
    for step in ('1', '2'):
        base = single['steps'][int(step) - 1]
        print('  case %s step %s' % (case, step))
        for rank, run in enumerate(runs):
            entry = run['steps'][int(step) - 1]
            param_abs = max(float((tensor(base['params'][name]) - tensor(entry['params'][name]))
                                  .abs().max()) for name in base['params'])
            param_scale = max(float(tensor(base['params'][name]).abs().max())
                              for name in base['params'])
            exp_abs = max(float((tensor(base['backbone_exp_avg'][key])
                                 - tensor(entry['backbone_exp_avg'][key])).abs().max())
                          for key in base['backbone_exp_avg']) if base['backbone_exp_avg'] else 0.0
            print('    rank%d param_digest_equal=%s backbone_opt_digest_equal=%s '
                  'mask_opt_digest_equal=%s maxabs(param)=%.3e (scale %.3e) maxabs(exp_avg)=%.3e'
                  % (rank, entry['param_digest'] == base['param_digest'],
                     entry['backbone_opt_digest'] == base['backbone_opt_digest'],
                     entry['mask_opt_digest'] == base['mask_opt_digest'],
                     param_abs, param_scale, exp_abs))
        print('    rank0-vs-rank1 param maxabs=%.3e; digest equal=%s'
              % (max(float((tensor(runs[0]['steps'][int(step) - 1]['params'][name])
                            - tensor(runs[1]['steps'][int(step) - 1]['params'][name])).abs().max())
                     for name in base['params']),
                 runs[0]['steps'][int(step) - 1]['param_digest']
                 == runs[1]['steps'][int(step) - 1]['param_digest']))

print()
print('#' * 78)
print('# Activation checkpointing ON/OFF equivalence (single process and 2-rank DDP)')
print('#' * 78)
off = load('ckoff.json')
on = load('ckon.json')
print('  single: param_digest off=%s on=%s equal=%s' % (off['steps'][0]['param_digest'],
                                                       on['steps'][0]['param_digest'],
                                                       off['steps'][0]['param_digest']
                                                       == on['steps'][0]['param_digest']))
print('  single: backbone_opt_digest equal=%s ; mask_opt_digest equal=%s'
      % (off['steps'][0]['backbone_opt_digest'] == on['steps'][0]['backbone_opt_digest'],
         off['steps'][0]['mask_opt_digest'] == on['steps'][0]['mask_opt_digest']))
print('  single: maxabs(param) between OFF and ON = %.3e'
      % max(float((tensor(off['steps'][0]['params'][name])
                   - tensor(on['steps'][0]['params'][name])).abs().max())
            for name in off['steps'][0]['params']))
ck_ranks = [load('ckon_rank%d.json' % rank) for rank in range(2)]
for rank, run in enumerate(ck_ranks):
    print('  2-rank checkpoint ON, rank%d: param_digest equal to single-ON: %s ; rank0==rank1: %s'
          % (rank, run['steps'][0]['param_digest'] == on['steps'][0]['param_digest'],
             ck_ranks[0]['steps'][0]['param_digest'] == ck_ranks[1]['steps'][0]['param_digest']))
