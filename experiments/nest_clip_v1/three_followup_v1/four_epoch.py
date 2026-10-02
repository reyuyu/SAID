"""Wait for both live three-epoch experiments, then run the selected four-epoch model."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search, BASE_CONFIG, REPO, PYTHON, SHARED, RUN as SEARCH_RUN,
    hparams, load, now, sha, trial_id,
)
from experiments.nest_clip_v1.three_followup_v1.run import (
    Followup, EXP, RUN as PREVIOUS_RUN, INIT_SHA256, reference_records,
)


RUN = PREVIOUS_RUN/'four_epoch'
BRANCH = 'codex/nest-balanced-four-epoch-v1'
PREVIOUS_EXP = Path('/root/lk_projects/SAID-balanced-three-followup-v1/experiments/nest_clip_v1/three_followup_v1')


class FourEpoch(Followup):
    branch = BRANCH

    def __init__(self):
        config = dict(BASE_CONFIG, epochs=4, four_epoch_followup=True)
        Search.__init__(self, run_dir=RUN, experiment_dir=EXP, base_config=config)
        self.refresh_previous()

    def refresh_previous(self):
        parent = load(SEARCH_RUN/'state.json')
        references, selected = reference_records(parent)
        assert sha(SHARED) == INIT_SHA256
        previous = load(PREVIOUS_RUN/'state.json')
        hp = hparams(selected['hparams'])
        tid = trial_id(hp)
        self.state['trials'].update(copy.deepcopy(previous['trials']))
        self.enroll(hp)
        experiment = self.state.get('four_epoch_experiment', dict(
            name='Four-epoch fusion-only', trial_id=tid, status='pending'))
        self.state.update(
            references=references,
            experiments=copy.deepcopy(previous['experiments'])+[experiment],
            four_epoch_experiment=experiment,
            experiment3=dict(selected, status=experiment['status'], epochs=4, horizon=4868,
                             evaluation_updates=[3651,4868], selected_from='completed original A/B/C@3651'),
            parent_result_commit=parent['result_commit'],
            previous_runtime=str(PREVIOUS_RUN), previous_result_commit=previous.get('result_commit'),
            common_resource_gate=copy.deepcopy(previous['common_resource_gate']),
            policy='wait_for_first_two_then_shared_step0_horizon4868_eval3651_eval4868',
        )
        self.save()
        self.publish()

    def run(self):
        previous = load(PREVIOUS_RUN/'state.json')
        assert all(e['status'] in ('completed','failed') for e in previous['experiments'])
        assert previous['status'] == 'awaiting_epoch_clarification', previous['status']
        assert previous['experiment3']['trial_id'] == self.state['experiment3']['trial_id']
        for directory in ('configs','evidence'):
            shutil.copytree(PREVIOUS_EXP/directory, EXP/directory, dirs_exist_ok=True)
        experiment = self.state['four_epoch_experiment']
        tid = experiment['trial_id']
        trial = self.state['trials'][tid]
        experiment['status'] = 'running'
        self.state['experiment3']['status'] = 'running'
        self.save()
        for stop in (3651,4868):
            if str(stop) in trial['budgets']:
                continue
            resume = None if stop == 3651 else Path(trial['budgets']['3651']['checkpoint'])
            root = self.train(trial['hparams'], stop, resume=resume)
            assert load(root/'config.json')['horizon'] == 4868
            self.evaluate(tid, root, stop)
            self.sync(f'Record four-epoch Balanced horizon4868 native results at{stop}')
        experiment['status'] = 'completed'
        self.state['experiment3']['status'] = 'completed'
        self.state.update(status='completed', stage='all-three-followups-completed', finished_utc=now())
        self.save()
        self.sync('Report all three directed Balanced followups and full four-epoch comparison')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            child = subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.three_followup_v1.four_epoch'],
                                     cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(queue_supervisor_pid=child.pid, runtime=str(RUN))))
        return
    with (RUN/'supervisor.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            runner = FourEpoch()
            runner.state.update(status='queued', stage='waiting-for-first-two-followups')
            runner.save()
            runner.publish()
            print(json.dumps(runner.state['experiment3']))
            return
        # The live first-two supervisor owns this lock until its normal exit.
        runner = FourEpoch()
        runner.state.update(status='queued', stage='waiting-for-first-two-followups')
        runner.save()
        runner.publish()
        with (PREVIOUS_RUN/'supervisor.lock').open('a') as previous_lock:
            print(json.dumps(dict(status='queued', waiting_for=str(PREVIOUS_RUN))), flush=True)
            fcntl.flock(previous_lock, fcntl.LOCK_EX)
            runner.refresh_previous()
            try:
                runner.run()
            except Exception as exc:
                runner.state.update(status='failed', error=str(exc), failed_utc=now())
                runner.state['four_epoch_experiment'].update(status='failed', error=str(exc))
                runner.state['experiment3']['status'] = 'failed'
                runner.save()
                runner.publish()
                raise


if __name__ == '__main__':
    main()
