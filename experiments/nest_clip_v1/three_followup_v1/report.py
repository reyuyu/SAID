"""Current references and complete native Recall for the directed followups."""
import json

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import DATASETS, rank_key


def write_report(state, output):
    lines = ['# Three Balanced Followups', '',
             f"Status: `{state['status']}`. Stage: `{state.get('stage')}`.", '',
             'Experiments1/2 use the latest instruction: full3651 updates, with evaluations at500 and3651. '
             'Both keep fusion_lr=2e-4. Experiment3 has conflicting3/4 epoch instructions and is pending clarification.', '',
             f"Experiment3 selected parent: `{state['experiment3']['trial_id']}`; "
             f"parameters `{json.dumps(state['experiment3']['hparams'], sort_keys=True)}`.", '',
             '## Scores', '',
             '| Configuration | Updates | Score5_R1 % | J_long3 % | J_long % |',
             '|---|---:|---:|---:|---:|']
    entries = []
    for label, reference in state['references'].items():
        hp = reference['hparams']
        name = ('Balanced B0' if hp['fusion_lr'] == 1e-4 else
                'Fusion-only C' if hp['inclusion_max'] == 1 else 'Previous inclusion1.5')
        entries.append((f"{name} ({label})", reference['updates'], reference))
    for experiment in state['experiments']:
        trial = state['trials'][experiment['trial_id']]
        for budget, result in trial['budgets'].items():
            entries.append((experiment['name'], int(budget), dict(result, hparams=trial['hparams'])))
    for label, budget, record in entries:
        score = record['scores']
        lines.append(f"| {label} | {budget} | {score['Score5_R1']*100:.6f} | "
                     f"{score['J_long3']*100:.6f} | {score['J_long']*100:.6f} |")
    lines += ['', '## Experiment Status', '']
    parent_id = next(r['trial_id'] for r in state['references'].values()
                     if r['hparams']['fusion_lr'] == 2e-4 and r['hparams']['inclusion_max'] == 1.5)
    for experiment in state['experiments']:
        trial = state['trials'][experiment['trial_id']]
        lines += [f"### {experiment['name']}", '',
                  f"Status: `{experiment['status']}`. Parameters: `{json.dumps(trial['hparams'],sort_keys=True)}`.", '']
        if experiment.get('error'):
            lines += [f"Failure: `{experiment['error']}`.", '']
        for budget, result in trial['budgets'].items():
            parent_key = f'{parent_id[:12]}@{budget}'
            if parent_key in state['references']:
                delta = 100*(result['scores']['Score5_R1']-state['references'][parent_key]['scores']['Score5_R1'])
                lines += [f"At{budget}, Score5_R1 change versus the inclusion1.5 parent: {delta:+.6f} pp.", '']
            lines += [f"At{budget}: full checkpoint SHA256 `{result['checkpoint_sha256']}`; "
                      f"bare student SHA256 `{result['bare_sha256']}`.", '',
                      f"Export/acceptance: `{result['export_check']['passed']}` / `{result['acceptance']['passed']}`.", '',
                      'Commands and exit codes are recorded in SEARCH_STATE.json and evidence/execution/. '
                      'Diagnostic summaries are in SEARCH_STATE.json; full step/token logs remain server-local.', '']
    lines += ['## Complete Native Recall', '']
    for label, budget, record in entries:
        lines += [f'### {label} @{budget}', '',
                  '| Dataset | Direction | R@1 % | R@5 % | R@10 % |', '|---|---|---:|---:|---:|']
        for dataset in DATASETS:
            for direction in ('I2T','T2I'):
                values = record['metrics'][dataset][direction]
                lines.append('| '+' | '.join([dataset,direction,*[f'{values[f"R@{k}"]*100:.6f}' for k in (1,5,10)]])+' |')
        lines.append('')
    finals = [(label,record) for label,budget,record in entries if budget == 3651]
    best_label, best_record = max(finals, key=lambda item: rank_key(item[1]))
    lines += ['## Current Conclusion', '',
              f"Highest completed3651 Score5_R1: {best_label}, {best_record['scores']['Score5_R1']*100:.6f}%.", '',
              'Unfinished experiments cannot be ranked at3651. Five datasets participate in selection; '
              'these are fixed-seed tuning results, without a significance or global optimality claim.', '',
              '## Reproduction', '', '```bash',
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --prepare',
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --launch',
              '```', '',
              'Each successful evaluation preserves exact train/export/verify/evaluator commands and runtime '
              'commit in evidence/execution/. Failed stages are not automatically retried. No fourth experiment '
              'or additional parameter values are scheduled. Large checkpoints, data and caches are not uploaded.']
    (output/'THREE_FOLLOWUP_REPORT.md').write_text('\n'.join(lines)+'\n')
