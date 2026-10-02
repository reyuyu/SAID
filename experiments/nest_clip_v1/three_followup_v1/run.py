"""Sequential directed coefficient experiments using the verified Balanced runner."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search, TrialFailure, REPO, RUN as PARENT_RUN, SHARED, PYTHON,
    hparams, load, now, rank_key, sha,
)


EXP = Path(__file__).resolve().parent
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/three_followup_v1')
BRANCH = 'codex/nest-balanced-three-followup-v1'
EXPERIMENTS = (
    ('Inclusion++', hparams(dict(fusion_lr=2e-4, inclusion_max=2))),
    ('Remainder++', hparams(dict(fusion_lr=2e-4, inclusion_max=1.5, view_weights=[1,1,2]))),
)
INIT_SHA256 = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'


def reference_records(parent):
    assert parent['status'] == 'completed', 'Wait for the previous experiment to finish'
    references = {}
    for tid, trial in parent['trials'].items():
        for budget, result in trial['budgets'].items():
            if '3651' in trial['budgets']:
                assert result['acceptance']['passed'] and result['export_check']['passed']
                references[f'{tid[:12]}@{budget}'] = dict(
                    trial_id=tid, hparams=trial['hparams'], updates=int(budget),
                    scores=result['scores'], metrics=result['metrics'],
                    checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
                )
    final = {tid: trial for tid, trial in parent['trials'].items() if '3651' in trial['budgets']}
    selected = max(final, key=lambda tid: rank_key(final[tid]['budgets']['3651']))
    assert selected == parent['best_3651']
    assert final[selected]['hparams']['fusion_lr'] == 2e-4
    return references, dict(trial_id=selected, hparams=final[selected]['hparams'],
                            status='awaiting_epoch_clarification', requested_epochs=[3,4])


class Followup(Search):
    def __init__(self):
        super().__init__(run_dir=RUN, experiment_dir=EXP)
        if 'experiments' not in self.state:
            parent = load(PARENT_RUN/'state.json')
            assert sha(SHARED) == INIT_SHA256
            assert parent['common_resource_gate']['passed']
            references, third = reference_records(parent)
            self.state.update(
                references=references, experiment3=third,
                parent_result_commit=parent['result_commit'],
                common_resource_gate=copy.deepcopy(parent['common_resource_gate']),
                policy='sequential_smoke5_then500_eval_then3651_eval',
                experiments=[],
            )
            for label, hp in EXPERIMENTS:
                tid = self.enroll(hp)
                self.state['experiments'].append(dict(name=label, trial_id=tid, status='pending'))
            self.state.update(status='ready', stage='prepared')
            self.save()
            self.publish()

    def publish(self):
        super().publish()
        if 'experiments' in self.state:
            from experiments.nest_clip_v1.three_followup_v1.report import write_report
            write_report(self.state, self.experiment_dir)

    def sync(self, message):
        branch = getattr(self, 'branch', BRANCH)
        assert subprocess.check_output(['git','branch','--show-current'], cwd=REPO, text=True).strip() == branch
        assert not subprocess.check_output(['git','diff','--cached','--name-only'], cwd=REPO, text=True).strip()
        self.publish()
        subprocess.run(['git','add',str(self.experiment_dir.relative_to(REPO))], cwd=REPO, check=True)
        subprocess.run(['git','diff','--cached','--check'], cwd=REPO, check=True)
        changed = subprocess.run(['git','diff','--cached','--quiet'], cwd=REPO).returncode
        assert changed in (0,1)
        if changed:
            subprocess.run(['git','commit','-m',message], cwd=REPO, check=True)
        pushed = subprocess.run(['git','push','origin',branch], cwd=REPO, timeout=180)
        self.state.update(github_synced=pushed.returncode == 0,
                          github_push_exit_code=pushed.returncode,
                          result_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip())
        self.save()
        if pushed.returncode:
            raise RuntimeError('Evidence push failed; local results are preserved')

    def run(self):
        for experiment in self.state['experiments']:
            tid = experiment['trial_id']
            trial = self.state['trials'][tid]
            if experiment['status'] in ('completed','failed'):
                continue
            experiment['status'] = 'running'
            self.save()
            self.publish()
            try:
                hp = trial['hparams']
                if '500' not in trial['budgets']:
                    self.train(hp, 5, 'smoke')
                    root = self.train(hp, 500)
                    self.evaluate(tid, root, 500)
                    self.sync(f"Record {experiment['name']} five-dataset results at500")
                if '3651' not in trial['budgets']:
                    root = self.train(hp, 3651, resume=Path(trial['budgets']['500']['checkpoint']))
                    self.evaluate(tid, root, 3651)
                experiment['status'] = 'completed'
                self.save()
                self.sync(f"Record {experiment['name']} complete three-epoch results")
            except (TrialFailure, subprocess.CalledProcessError) as exc:
                experiment.update(status='failed', error=str(exc))
                trial.update(status='failed', error=str(exc))
                self.save()
                self.sync(f"Preserve {experiment['name']} failure evidence")
        self.state.update(status='awaiting_epoch_clarification', stage='experiment3-budget',
                          first_two_finished_utc=now())
        self.save()
        self.sync('Report directed Balanced followups and pending experiment3 budget')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            child = subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.three_followup_v1.run'],
                                     cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=child.pid, runtime=str(RUN))))
        return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Followup()
        if args.prepare:
            print(json.dumps(dict(status=runner.state['status'], experiments=runner.state['experiments'])))
            return
        try:
            runner.run()
        except Exception as exc:
            runner.state.update(status='failed', error=str(exc), failed_utc=now())
            runner.save()
            runner.publish()
            raise


if __name__ == '__main__':
    main()
