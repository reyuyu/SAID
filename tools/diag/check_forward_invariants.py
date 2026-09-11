"""Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.
"""

import json
import os
import subprocess
import sys

REPO = '/root/SAID-gap-completion'
WORKER = REPO + '/tests/_cvssl_ddp_trainer_worker.py'
PY = sys.executable
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/cvssl_diag/inv'
os.makedirs(OUT, exist_ok=True)


def run(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, cwd=REPO)
    if result.returncode != 0:
        print(result.stderr[-2500:])
        raise SystemExit('command failed')


for case in ('S0', 'C0', 'R0'):
    run([PY, WORKER, '--case', case, '--mode', 'single', '--out', OUT + '/%s_single.json' % case,
         '--steps', '1'])
    run([PY, '-m', 'torch.distributed.run', '--nproc_per_node=2', '--master_port', '29811',
         WORKER, '--case', case, '--mode', 'rank', '--out', OUT + '/%s_rank{rank}.json' % case,
         '--steps', '1'],
        {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29811', 'GLOO_SOCKET_IFNAME': 'lo'})
    single = json.load(open(OUT + '/%s_single.json' % case))['history']['1']
    ranks = [json.load(open(OUT + '/%s_rank%d.json' % (case, r)))['history']['1'] for r in range(2)]
    print('== case %s' % case)
    for key in ('loss_sidm', 'loss_dism', 'loss_sparsity', 'loss_smart', 'loss_vssl_global_mean'):
        values = [entry[key] for entry in ranks]
        mean = sum(values) / len(values)
        reference = single[key]
        print('   %-24s single=%.10f ranks=[%.10f, %.10f] mean=%.10f |mean-single|=%.3e rel=%.3e'
              % (key, reference, values[0], values[1], mean, abs(mean - reference),
                 abs(mean - reference) / max(abs(reference), 1e-30)))
        print('      rank0-vs-rank1 |diff|=%.3e (expected NONZERO: different anchors)'
              % abs(values[0] - values[1]))
