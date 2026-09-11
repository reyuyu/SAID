"""Float64 control: if the residual is floating-point accumulation order it must collapse.

Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.
"""

import json
import sys

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cvssl_diag'
F32 = OUT + '/stage4'
F64 = OUT + '/stage5'


def load(path):
    with open(path) as handle:
        return json.load(handle)


def compare(tag, reference, runs):
    total = {}
    for run in runs:
        for name, value in run['steps'][0]['grads'].items():
            if value is None:
                continue
            piece = torch.tensor(value, dtype=torch.float64) / float(len(runs))
            total[name] = piece if name not in total else total[name] + piece
    worst_abs, worst_rel, scale = 0.0, 0.0, 0.0
    for name, value in reference['steps'][0]['grads'].items():
        if value is None:
            continue
        base = torch.tensor(value, dtype=torch.float64)
        magnitude = float(base.abs().max())
        difference = float((base - total[name]).abs().max())
        worst_abs = max(worst_abs, difference)
        scale = max(scale, magnitude)
        worst_rel = max(worst_rel, difference / max(magnitude, 1e-300))
    print('   %-40s max|g|=%.4e  worst_abs=%.3e  worst_rel=%.3e' % (tag, scale, worst_abs,
                                                                     worst_rel))
    return worst_abs, worst_rel


print('== float32 (production dtype) ==')
compare('S0 sidm  : DDP mean vs single',
        load(F32 + '/s0_sidm_single.json'),
        [load(F32 + '/s0_sidm_rank%d.json' % r) for r in range(2)])
compare('S0 smart : DDP mean vs single',
        load(F32 + '/s0_smart_single.json'),
        [load(F32 + '/s0_smart_rank%d.json' % r) for r in range(2)])
compare('C0 total : DDP mean vs single',
        load(F32 + '/a_total_single.json'),
        [load(F32 + '/a_total_rank%d.json' % r) for r in range(2)])
print()
print('== float64 (same math, no accumulation-order noise) ==')
compare('S0 sidm  : DDP mean vs single',
        load(F64 + '/f64_s0_sidm_single.json'),
        [load(F64 + '/f64_s0_sidm_rank%d.json' % r) for r in range(2)])
compare('S0 smart : DDP mean vs single',
        load(F64 + '/f64_s0_smart_single.json'),
        [load(F64 + '/f64_s0_smart_rank%d.json' % r) for r in range(2)])
compare('C0 total : DDP mean vs single',
        load(F64 + '/f64_a_total_single.json'),
        [load(F64 + '/f64_a_total_rank%d.json' % r) for r in range(2)])
