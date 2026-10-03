"""One exact R-SentenceDrop500 continuation to4868, five native evaluations, stop."""
import argparse
import copy
import fcntl
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search, REPO, PYTHON, hparams, load, native_metrics, now, rows,
)
from experiments.nest_clip_v1.balanced_rdrop_500_v1.run import summarize_sampling
from experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.verify import (
    BASE, BRANCH, EXP, RUN, PARENT_RUN, reference_records,
)

STREAM_KEYS = ('sample_id_sha256', 'full_view_sha256', 'split_sha256',
               'fixed_first_reference_stream_sha256', 'n', 'K',
               'prefix_segments', 'prefix_token_lengths')


def record_stream(paths):
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def compare_streams(actual_paths, reference_paths, expected_updates=4868):
    count = 0
    for actual, reference in itertools.zip_longest(record_stream(actual_paths), record_stream(reference_paths)):
        assert actual is not None and reference is not None, 'Stream length mismatch'
        count += 1
        assert actual['step'] == reference['step'] == count, 'Noncontiguous full trajectory'
        assert actual['nonfinite'] == 0
        assert len(actual['rank_health']) == len(reference['rank_health']) == 4
        for a, b in zip(actual['rank_health'], reference['rank_health']):
            assert a['rank'] == b['rank'] and a['gradients_finite']
            for key in STREAM_KEYS:
                assert a['sampling'][key] == b['sampling'][key], f'Matched stream drift step{count}: {key}'
    assert count == expected_updates
    return dict(passed=True, total_updates=count, ranks=4, compared_keys=list(STREAM_KEYS),
                original_baseline_has_prefix_hash=False,
                prefix_verification='Same full tokens, K, sentence counts, prefix segments/lengths; '
                                    'unchanged deterministic prefix construction. Explicit P-token hashes '
                                    'were additionally matched against the newer baseline over updates1..500.')


class Experiment(Search):
    def __init__(self):
        super().__init__(run_dir=RUN, experiment_dir=EXP, base_config=BASE)
        if not self.state['stages']:
            self.state['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
            self.save()
        if 'baseline' not in self.state:
            preflight = load(EXP/'evidence/preflight.json')
            assert preflight['passed']
            parent, baseline, logs = reference_records()
            tid = self.enroll(hparams(BASE))
            self.state['trials'][tid]['budgets']['500'] = dict(copy.deepcopy(parent), reused=True)
            self.state.update(status='prepared', stage='verified-continuation-500-to4868',
                              baseline=baseline, parent500=parent, rdrop_trial=tid,
                              baseline_logs=list(map(str, logs)), preflight=preflight,
                              baseline_retrained=False, horizon=4868, stop_updates=4868,
                              parent_experiment_commit=load(PARENT_RUN/'state.json')['result_commit'])
            dest = EXP/'evidence/baseline4868'
            dest.mkdir(parents=True, exist_ok=True)
            _, _, sources = native_metrics(Path(baseline['root']))
            for name, source in sources.items():
                shutil.copy2(source, dest/(name.lower().replace('-', '_')+'.json'))
            for name in ('config.json', 'acceptance.json', 'export-check.json'):
                shutil.copy2(Path(baseline['root'])/name, dest/name)
            source = REPO/'experiments/nest_clip_v1/balanced_rdrop_500_v1/evidence'/tid[:12]/'step500'
            shutil.copytree(source, EXP/'evidence/parent500', dirs_exist_ok=True)
            self.save()
            self.publish()

    def publish(self):
        super().publish()
        if 'baseline' in self.state:
            from experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.report import write_report
            write_report(self.state, EXP)

    def sync(self, message):
        self.publish()
        assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == BRANCH
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).strip()
        subprocess.run(['git', 'add', str(EXP.relative_to(REPO))], cwd=REPO, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPO, check=True)
        changes = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode
        assert changes in (0, 1)
        if changes:
            subprocess.run(['git', 'commit', '-m', message], cwd=REPO, check=True)
        push = subprocess.run(['git', 'push', 'origin', BRANCH], cwd=REPO, timeout=180)
        self.state.update(github_synced=push.returncode == 0,
                          result_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
        self.save()
        if push.returncode:
            raise RuntimeError('GitHub evidence push failed; all local results retained')

    def run(self):
        tid = self.state['rdrop_trial']
        parent = self.state['parent500']
        if '4868' not in self.state['trials'][tid]['budgets']:
            root = self.train(hparams(BASE), 4868, resume=Path(parent['checkpoint']))
            cfg = load(root/'config.json')
            assert cfg['start_updates'] == 500 and cfg['horizon'] == cfg['max_updates'] == 4868
            assert cfg['updates_planned_this_run'] == 4368 and cfg['remainder_mode'] == 'sentence_drop'
            assert cfg['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
            assert cfg['code_sha256'] == parent['config']['code_sha256']
            comparison = compare_streams([Path(parent['root'])/'steps.jsonl', root/'steps.jsonl'],
                                         self.state['baseline_logs'])
            self.state['full_stream_comparison'] = comparison
            self.save()
            self.evaluate(tid, root, 4868)
        result = self.state['trials'][tid]['budgets']['4868']
        assert result['step_range'] == [501, 4868]
        chain = [Path(parent['root']), Path(result['root'])]
        result['sentence_drop_statistics'] = summarize_sampling(
            list(record_stream([root/'steps.jsonl' for root in chain])))
        timing = [r['four_rank_max_seconds'] for root in chain
                  for r in rows(root/'cycle_timing.jsonl') if not r['warmup']]
        result['resource_summary'] = dict(mean_seconds=statistics.fmean(timing),
            median_seconds=statistics.median(timing), p95_seconds=statistics.quantiles(timing, n=100)[94],
            max_seconds=max(timing), regular_updates=len(timing),
            peak_allocated_gib=max(r['peak_allocated_gib'] for entry in (parent, result) for r in entry['acceptance']['ranks']),
            peak_reserved_gib=max(r['peak_reserved_gib'] for entry in (parent, result) for r in entry['acceptance']['ranks']),
            scope='Complete updates1..4868, excluding five warmup cycles of each process; checkpoints timed separately')
        result['stream_comparison'] = self.state['full_stream_comparison']
        result['complete_trajectory_range'] = [1, 4868]
        self.state.update(status='completed', stage='four-epoch-native-final-comparison', finished_utc=now(),
                          delta_scores_pp={k:100*(v-self.state['baseline']['scores'][k]) for k,v in result['scores'].items()},
                          delta_R1_pp={ds:{dr:100*(m['R@1']-self.state['baseline']['metrics'][ds][dr]['R@1'])
                                         for dr,m in metrics.items()} for ds,metrics in result['metrics'].items()})
        self.save()
        self.sync('Report R-SentenceDrop four-epoch native results against reused old-R baseline and stop')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            process = subprocess.Popen([PYTHON, '-u', '-m', 'experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.run'],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=process.pid, runtime=str(RUN))))
        return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN/'state.json').exists():
            state = load(RUN/'state.json')
            assert state['status'] not in ('failed', 'completed'), 'Preserve terminal result; no automatic retry'
        controller = Experiment()
        if args.prepare:
            controller.publish()
            print(json.dumps(dict(status='prepared', start=500, stop=4868, horizon=4868)))
            return
        try:
            controller.run()
        except Exception as exc:
            controller.state.update(status='failed', error=str(exc), failed_utc=now())
            controller.save()
            controller.publish()
            raise


if __name__ == '__main__':
    main()
