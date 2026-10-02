"""Correctness/resource gates before the sole authorized L14 four-epoch run."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search, REPO, PYTHON, TrialFailure, hparams, load, now, rank_key, trial_id,
)
from experiments.nest_clip_v1.balanced_l14_4epoch_v1.prepare import EXP,RUN,INIT,CONFIG


BRANCH='codex/nest-balanced-l14-4epoch-v1'
BASE=load(CONFIG)


class L14Run(Search):
    def __init__(self):
        super().__init__(run_dir=RUN,experiment_dir=EXP,base_config=BASE,init_state=INIT)
        if 'l14_trial' not in self.state:
            initialization=load(RUN/'initialization.json')
            tests=load(EXP/'evidence/correctness.json')
            ddp=load(EXP/'evidence/ddp-reference.json')
            real_ddp=load(EXP/'evidence/real-l14-ddp.json')
            assert initialization['passed'] and tests['passed'] and ddp['passed'] and real_ddp['passed']
            tid=trial_id(hparams(BASE))
            self.state['trials'].setdefault(tid,dict(hparams=hparams(BASE),seed=0,status='pending',budgets={}))
            self.state.update(initialization=initialization,correctness=tests,
                              resources=[],l14_trial=tid,
                              execution_path='direct',evaluation_updates=[0,500,3651,4868],
                              reference_commit=initialization['reference_commit'])
            self.save()
            self.publish()
        authorization_path=EXP/'evidence/resource-authorization-5s.json'
        if authorization_path.exists():
            authorization=load(authorization_path)
            assert authorization['approved_regular_update_limit_seconds']==BASE['max_update_seconds']
            if self.state.get('resource_authorization')!=authorization:
                assert not self.state.get('formal_training_started'), 'Cannot change policy during formal training'
                self.state.update(resource_authorization=authorization,resource_gate_passed=False,
                                  status='ready',stage='approved-resource-limit-recheck',
                                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
                for key in ('finished_utc','unmet','error'):
                    self.state.pop(key,None)
                self.save()
                self.publish()

    def publish(self):
        super().publish()
        if 'l14_trial' in self.state:
            from experiments.nest_clip_v1.balanced_l14_4epoch_v1.report import write_report
            write_report(self.state,self.experiment_dir)

    def sync(self,message):
        assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()==BRANCH
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        self.publish()
        subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
        assert changed in (0,1)
        if changed:
            subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
        pushed=subprocess.run(['git','push','origin',BRANCH],cwd=REPO,timeout=180)
        self.state.update(github_synced=pushed.returncode==0,github_push_exit_code=pushed.returncode,
                          result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        self.save()
        if pushed.returncode:
            raise RuntimeError('Evidence push failed; all local artifacts preserved')

    def probe(self,label,configuration):
        runner=Search(run_dir=RUN/'resources'/label,experiment_dir=EXP/'evidence/resources'/label,
                      base_config=configuration,init_state=INIT)
        runner.state['code_commit']=self.state['code_commit']
        runner.save()
        hp=hparams(BASE)
        tid=runner.enroll(hp)
        root=runner.run_dir/'trials'/tid/'probe'
        self.state.update(status='resource_gate',stage=label)
        self.save()
        try:
            runner.train(hp,35,'probe')
        except (RuntimeError,subprocess.CalledProcessError,TrialFailure) as exc:
            error=str(exc)
        else:
            error=None
        acceptance=load(root/'acceptance.json') if (root/'acceptance.json').exists() else None
        log=runner.run_dir/'execution'/f'{tid[:12]}-probe-35.console.txt'
        oom='out of memory' in log.read_text().lower() if log.exists() else False
        record=dict(label=label,config=configuration,root=str(root),acceptance=acceptance,
                    passed=bool(acceptance and acceptance['passed']),error=error,oom=oom)
        self.state['resources'].append(record)
        self.save()
        self.publish()
        return record

    def run(self):
        hp=hparams(BASE)
        if not self.state.get('resource_gate_passed'):
            limit=BASE['max_update_seconds']
            suffix=f'-limit{limit:g}s' if limit!=3 else ''
            resource=self.probe('direct-128x128'+suffix,copy.deepcopy(BASE))
            if not resource['passed']:
                memory_failed=resource['oom'] or ('memory' in str(resource['acceptance']).lower())
                # A speed failure cannot be repaired by redefining the batch or time budget.
                if memory_failed:
                    smaller=dict(BASE,image_chunk=64,text_chunk=64,checkpoint_pair_blocks=True)
                    resource=self.probe('direct-64x64-pair-checkpoint'+suffix,smaller)
                if not resource['passed']:
                    self.state.update(status='resource_stopped',stage='resource-gate-failed',finished_utc=now(),
                                      formal_training_started=False,
                                      unmet=f'Required complete logical updates<={limit:g}s and peak allocated<=65GiB per rank')
                    self.save()
                    self.sync('Report measured L14 resource gate; preserve stopped formal run')
                    return
            self.base_config=resource['config']
            self.state.update(resource_gate_passed=True,approved_training_config=resource['config'])
            self.save()
        else:
            self.base_config=self.state['approved_training_config']
        if not self.state.get('smoke_passed'):
            self.train(hp,5,'smoke')
            self.state.update(smoke_passed=True,status='ready',stage='validated-resource-gate-and-independent-smoke')
            self.save()
            self.sync('Record approved L14 resource gate and independent smoke before formal training')
        tid=self.state['l14_trial']
        if not self.state.get('formal_training_finished'):
            self.state.update(formal_training_started=True,status='running',stage='formal-training-starting')
            self.save()
            self.publish()
            root=self.train(hp,4868)
            self.state.update(formal_training_finished=True,formal_root=str(root))
            self.save()
        root=Path(self.state['formal_root'])
        for stop in (0,500,3651,4868):
            if str(stop) in self.state['trials'][tid]['budgets']:
                continue
            self.evaluate(tid,RUN/'evaluations'/f'step{stop}',stop,training_root=root)
            self.sync(f'Record Balanced L14 native five-dataset evaluation at{stop}')
        budgets=self.state['trials'][tid]['budgets']
        observed=max((500,3651,4868),key=lambda stop:rank_key(budgets[str(stop)]))
        self.state.update(status='completed',stage='final-and-best-observed',finished_utc=now(),
                          final_updates=4868,best_observed_updates=observed)
        self.save()
        self.sync('Report complete Balanced L14 four-epoch final and best observed results')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            child=subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.balanced_l14_4epoch_v1.run'],
                                   cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=child.pid,runtime=str(RUN))))
        return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=L14Run()
        try:
            runner.run()
        except Exception as exc:
            runner.state.update(status='failed',error=str(exc),failed_utc=now())
            runner.save()
            runner.publish()
            raise


if __name__=='__main__':
    main()
