"""Orchestrate the C0 limited tuning: train -> eval -> score -> decide, stopping on a pass.

    python tools/diag/c0_tuning_search.py [--max-candidates N]

The candidate order is frozen in docs/said_cls_cvssl/c0_tuning/search_plan.md. Trials 3 and 4 depend
on earlier results, so their lambda_U / schedule are resolved from the ledger at launch time. The
loop stops immediately when the promotion gate is passed and never launches more than four new
training runs.
"""
import json
import os
import subprocess
import sys
import time

REPO = '/root/SAID-gap-completion'
TUNING = os.path.join(REPO, 'docs/said_cls_cvssl/c0_tuning')
PROGRESS = os.path.join(TUNING, 'progress.json')
RESULTS = os.path.join(TUNING, 'results.csv')
RUN_ROOT = os.path.join(REPO, 'runs_salu/said_cls_cvssl/c0_tuning')
EVAL_ROOT = os.path.join(REPO, 'outputs/cvssl_screening/c0_tuning')
LAUNCHER = os.path.join(REPO, 'tools/exp_said_cls_cvssl_c0_tuning.sh')
LEDGER = os.path.join(REPO, 'tools/diag/c0_tuning_ledger.py')
PY = '/root/miniconda3/envs/said-smartclip/bin/python'
STEPS = 500
ARM = 'C0_complement_vssl'

EXISTING = {'run': 'C_existing_L10', 'lambda_target': 1.0, 'u_warmup': 0, 'tau': 0.1}


def shell(command, check=True):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=REPO)
    if check and result.returncode != 0:
        raise SystemExit('command failed: %s\n%s\n%s' % (command, result.stdout[-2000:],
                                                         result.stderr[-2000:]))
    return result


def read_rows():
    import csv
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS, newline='') as handle:
        return list(csv.DictReader(handle))


def best_fixed_weight():
    """Best of the three constant-weight C0 candidates by J; ties prefer the smaller lambda_U."""
    rows = {row['run']: row for row in read_rows()}
    order = [('C_existing_L10', 1.0), ('C_L03', 0.3), ('C_L01', 0.1)]
    scored = [(name, lam, float(rows[name]['J'])) for name, lam in order if name in rows]
    if not scored:
        raise SystemExit('no fixed-weight C0 candidate has been scored yet')
    scored.sort(key=lambda item: (-item[2], item[1]))
    return scored[0]


def best_overall():
    """Best C0 candidate overall (never S0), used for the tau_U trial."""
    rows = [row for row in read_rows() if row['run'] != 'S0']
    if not rows:
        raise SystemExit('no C0 candidate has been scored yet')

    def key(row):
        return (-float(row['J']), float(row['lambda_target']),
                0 if float(row['u_warmup'] or 0) == 0 else 1)
    return sorted(rows, key=key)[0]


def train_and_eval(run_tag, lambda_u, tau_u, u_warmup):
    tail = shell('tail -1 %s/%s/train.log 2>/dev/null || true' % (RUN_ROOT, run_tag),
                 check=False).stdout.strip()
    print('[%s] launching train lambda_U=%s tau_U=%s u_warmup=%s' % (run_tag, lambda_u, tau_u,
                                                                     u_warmup), flush=True)
    launched = shell('bash %s train %s %s %s %s' % (LAUNCHER, run_tag, lambda_u, tau_u, u_warmup),
                     check=False)
    print(launched.stdout.strip(), flush=True)
    if launched.returncode != 0:
        raise SystemExit('launch failed for %s: %s' % (run_tag, launched.stderr[-1500:]))
    log = os.path.join(RUN_ROOT, run_tag, 'train.log')
    deadline = time.time() + 4 * 3600
    while time.time() < deadline:
        time.sleep(30)
        if not os.path.exists(log):
            continue
        content = open(log, errors='replace').read()
        if 'RUN_SUMMARY' in content:
            break
        if 'Traceback' in content or 'ChildFailedError' in content:
            raise SystemExit('training failed for %s; see %s' % (run_tag, log))
    else:
        raise SystemExit('training timed out for %s' % run_tag)
    summary = json.load(open(os.path.join(RUN_ROOT, run_tag, 'run_summary.json')))
    ckpt = os.path.join(RUN_ROOT, run_tag, 'cvssl_%s_step%06d.pt' % (ARM, STEPS))
    if summary['completed_steps'] != STEPS or not os.path.exists(ckpt):
        raise SystemExit('%s: completed_steps=%s checkpoint_exists=%s'
                         % (run_tag, summary['completed_steps'], os.path.exists(ckpt)))
    print('[%s] trained: completed=%s wall=%.1fs peak=%.2fGB' %
          (run_tag, summary['completed_steps'], summary['wall_sec'], summary['peak_memory_gb']),
          flush=True)
    print('[%s] strict-load check' % run_tag, flush=True)
    verify = shell('%s %s/tools/diag/verify_cvssl_ckpt.py %s %d' % (PY, REPO, ckpt, STEPS),
                   check=False)
    if verify.returncode != 0:
        raise SystemExit('%s: strict-load verification failed\n%s' % (run_tag,
                                                                      verify.stdout[-1500:]))
    print([line for line in verify.stdout.splitlines() if 'VERIFY_OK' in line
           or 'STRICT_LOAD_OK' in line], flush=True)

    print('[%s] evaluating' % run_tag, flush=True)
    evaluated = shell('bash %s eval %s' % (LAUNCHER, run_tag), check=False)
    if evaluated.returncode != 0:
        raise SystemExit('%s: evaluation failed\n%s\n%s' % (run_tag, evaluated.stdout[-1200:],
                                                            evaluated.stderr[-1200:]))
    canonical = os.path.join(EVAL_ROOT, '%s_canonical.json' % run_tag)
    if not os.path.exists(canonical):
        raise SystemExit('%s: no canonical json at %s' % (run_tag, canonical))
    scored = shell('%s %s score %s %s' % (PY, LEDGER, run_tag, canonical), check=False)
    print(scored.stdout, flush=True)
    if scored.returncode != 0:
        raise SystemExit('%s: scoring failed\n%s' % (run_tag, scored.stderr[-1500:]))
    row = [r for r in read_rows() if r['run'] == run_tag][0]
    return float(row['J']), row['passed'] in ('True', 'true', True)


def main():
    max_candidates = 4
    if '--max-candidates' in sys.argv:
        max_candidates = int(sys.argv[sys.argv.index('--max-candidates') + 1])
    os.makedirs(TUNING, exist_ok=True)
    done = {entry['run'] for entry in json.load(open(PROGRESS)).get('completed', [])} \
        if os.path.exists(PROGRESS) else set()
    # THIS ROUND: a single experiment only, lambda_U = 0.3. The remaining planned candidates
    # (C_L01, C_BEST_W200, C_BEST_T02) are intentionally NOT launched; they stay documented in
    # search_plan.md and are resolved through the ledger if a later round asks for them.
    all_planned = ['C_L03', 'C_L01', 'C_BEST_W200', 'C_BEST_T02']
    planned = all_planned[:max_candidates]
    verdict = 'SEARCH_BUDGET_EXHAUSTED'
    passed_run = None
    for run_tag in planned:
        if run_tag in done:
            print('[%s] already scored, skipping' % run_tag, flush=True)
            continue
        if run_tag == 'C_L03':
            config = (run_tag, 0.3, 0.1, 0)
        elif run_tag == 'C_L01':
            config = (run_tag, 0.1, 0.1, 0)
        elif run_tag == 'C_BEST_W200':
            name, lam, j_value = best_fixed_weight()
            print('   best fixed-weight C0 = %s (lambda_U=%s, J=%.10f)' % (name, lam, j_value),
                  flush=True)
            config = (run_tag, lam, 0.1, 200)
        else:
            row = best_overall()
            print('   best C0 overall = %s (lambda_U=%s, u_warmup=%s, tau_U=%s, J=%.10f)'
                  % (row['run'], row['lambda_target'], row['u_warmup'], row['tau_U'], float(row['J'])),
                  flush=True)
            config = (run_tag, float(row['lambda_target']), 0.2, int(float(row['u_warmup'] or 0)))
        j_value, passed = train_and_eval(*config)
        print('[%s] J=%.10f passed=%s' % (run_tag, j_value, passed), flush=True)
        if passed:
            verdict = 'CANDIDATE_PASSED_500STEP_SCREENING'
            passed_run = run_tag
            break
    payload = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}
    payload['verdict'] = verdict
    payload['passed_run'] = passed_run
    payload['search_finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(PROGRESS, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print('VERDICT %s passed_run=%s' % (verdict, passed_run), flush=True)
    if verdict != 'CANDIDATE_PASSED_500STEP_SCREENING':
        print('NO_CANDIDATE_PASSED', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
