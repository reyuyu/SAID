"""Stage 2 analysis: residual character, term decomposition, U scaling, static-graph, checkpoint.

Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.
"""

import json
import sys

import torch

S2 = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cvssl_diag/stage2'
S3 = sys.argv[2] if len(sys.argv) > 2 else '/tmp/cvssl_diag/stage3'


def load(name):
    with open('%s/%s' % (S2, name)) as handle:
        return json.load(handle)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def rel(a, b):
    first, second = tensor(a), tensor(b)
    scale = max(float(first.abs().max()), float(second.abs().max()), 1e-30)
    return float((first - second).abs().max()) / scale


def grad_pair(first, second, key='grads', term=None):
    left = first['steps'][0][key] if term is None else first['steps'][0][key][term]
    right = second['steps'][0][key] if term is None else second['steps'][0][key][term]
    worst = 0.0
    for name in left:
        if left[name] is None or right[name] is None:
            continue
        worst = max(worst, rel(left[name], right[name]))
    return worst


def mean_of_ranks(runs, key='grads', term=None):
    values = [run['steps'][0][key] if term is None else run['steps'][0][key][term] for run in runs]
    total = {}
    for entry in values:
        for name, value in entry.items():
            if value is None:
                continue
            piece = tensor(value) / float(len(values))
            total[name] = piece if name not in total else total[name] + piece
    return total


def compare(tag, reference, combined):
    worst_rel, worst_abs = 0.0, 0.0
    ratios = []
    for name in sorted(reference):
        if reference[name] is None or name not in combined:
            continue
        base, other = tensor(reference[name]), combined[name]
        difference = float((base - other).abs().max())
        scale = max(float(base.abs().max()), float(other.abs().max()), 1e-30)
        worst_abs = max(worst_abs, difference)
        worst_rel = max(worst_rel, difference / scale)
        if float(base.norm()) > 1e-12 and float(other.norm()) > 1e-12:
            ratios.append(float(other.norm()) / float(base.norm()))
    ratios.sort()
    print('   %-34s worst_rel=%.3e worst_abs=%.3e  norm_ratio(min/p50/max)=%.6f/%.6f/%.6f'
          % (tag, worst_rel, worst_abs, ratios[0], ratios[len(ratios) // 2], ratios[-1]))
    return worst_rel


print('== (a) S0 residual character: DDP gradient vs single-process global gradient')
single_s0 = load('single_s0_terms.json')
s0_nostatic = [load('s0_nostatic_rank%d.json' % rank) for rank in range(2)]
s0_static = [load('s0_static_rank%d.json' % rank) for rank in range(2)]
print('   absolute max|grad| of the single reference (per parameter):')
for name in sorted(single_s0['steps'][0]['term_grads']['total']):
    value = single_s0['steps'][0]['term_grads']['total'][name]
    if value is None:
        continue
    print('      %-28s max|g|=%.6e' % (name, float(tensor(value).abs().max())))
compare('S0 DDP(no static) mean vs single', single_s0['steps'][0]['term_grads']['total'],
        mean_of_ranks(s0_nostatic, 'term_grads', 'total'))
compare('S0 DDP(static on) mean vs single', single_s0['steps'][0]['term_grads']['total'],
        mean_of_ranks(s0_static, 'term_grads', 'total'))
print('   rank0 vs rank1 (DDP, static off), term=total: %.3e'
      % grad_pair(s0_nostatic[0], s0_nostatic[1], 'term_grads', 'total'))
print('   rank0 vs rank1 (DDP, static on ), term=total: %.3e'
      % grad_pair(s0_static[0], s0_static[1], 'term_grads', 'total'))

print()
print('== (b) term decomposition, S0: DDP mean vs single (relative)')
for term in ('sidm', 'dism', 'sparse', 'smart', 'total'):
    reference = single_s0['steps'][0]['term_grads'][term]
    combined = mean_of_ranks(s0_nostatic, 'term_grads', term)
    print('   term=%-7s' % term, end='')
    compare('', reference, combined)

print()
print('== (c) term decomposition, A (full C0): DDP mean vs single (relative)')
single_a = load('single_a_terms.json')
a_ranks = [load('a_terms_rank%d.json' % rank) for rank in range(2)]
for term in ('sidm', 'dism', 'sparse', 'smart', 'u', 'total'):
    reference = single_a['steps'][0]['term_grads'][term]
    combined = mean_of_ranks(a_ranks, 'term_grads', term)
    print('   term=%-7s' % term, end='')
    compare('', reference, combined)

print()
print('== (d) U-only (C0 arm, terms=u): does the world-size factor give the global-mean gradient?')
single_u = load('u_single_won.json')
u_on = [load('u_ddp_won_rank%d.json' % rank) for rank in range(2)]
u_off = [load('u_ddp_woff_rank%d.json' % rank) for rank in range(2)]
print('   loss values: single loss=%.8f loss_global_mean=%.8f scale=%.3f'
      % (single_u['steps'][0]['loss_vssl_local_backward'],
         single_u['steps'][0]['loss_vssl_global_mean'], single_u['steps'][0]['loss_scale']))
for tag, runs in (('W ON ', u_on), ('W OFF', u_off)):
    print('   %s: rank losses=%s scales=%s global_mean=%s valid=%s/%s' % (
        tag,
        ['%.8f' % run['steps'][0]['loss_vssl_local_backward'] for run in runs],
        ['%.2f' % run['steps'][0]['loss_scale'] for run in runs],
        ['%.8f' % run['steps'][0]['loss_vssl_global_mean'] for run in runs],
        ['%.0f' % run['steps'][0]['valid_ab'] for run in runs],
        ['%.0f' % run['steps'][0]['valid_ba'] for run in runs]))
compare('U-only W ON : DDP mean vs single', single_u['steps'][0]['term_grads']['u'],
        mean_of_ranks(u_on, 'term_grads', 'u'))
compare('U-only W OFF: DDP mean vs single', single_u['steps'][0]['term_grads']['u'],
        mean_of_ranks(u_off, 'term_grads', 'u'))
print('   rank0 vs rank1, U-only W ON : %.3e' % grad_pair(u_on[0], u_on[1], 'term_grads', 'u'))
print('   rank0 vs rank1, U-only W OFF: %.3e' % grad_pair(u_off[0], u_off[1], 'term_grads', 'u'))
