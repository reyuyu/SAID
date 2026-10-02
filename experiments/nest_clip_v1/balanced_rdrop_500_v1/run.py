"""Reuse the verified baseline, run one SentenceDrop500 experiment, then stop."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search,REPO,PYTHON,load,now,hparams,rows,
)
from experiments.nest_clip_v1.balanced_rdrop_500_v1.verify import EXP,RUN,BASELINE_STATE,BASELINE_SHA


BASE=load(REPO/'configs/nest_balanced_rdrop_500_v1.json')
BRANCH='codex/nest-balanced-rdrop-500-v1'
MATCH_KEYS=('sample_id_sha256','full_view_sha256','prefix_view_sha256','split_sha256',
            'fixed_first_reference_stream_sha256')


def summarize_sampling(records):
    diagnostics=[h['sampling']['sentence_drop_diagnostics'] for r in records for h in r['rank_health']]
    ms=Counter();qs=Counter();joint=Counter()
    for d in diagnostics:
        ms.update(d['m_counts']);qs.update(d['q_counts']);joint.update(d['joint_m_q_counts'])
    count=sum(d['valid_samples'] for d in diagnostics)
    assert count==sum(joint.values())
    fractions=np.concatenate([np.full(n,int(key.split(':')[1])/int(key.split(':')[0]),dtype=np.float64)
                              for key,n in joint.items()])
    bins={'(0,0.25]':int(((fractions>0)&(fractions<=.25)).sum()),
          '(0.25,0.5]':int(((fractions>.25)&(fractions<=.5)).sum()),
          '(0.5,0.75]':int(((fractions>.5)&(fractions<=.75)).sum()),
          '(0.75,1)':int(((fractions>.75)&(fractions<1)).sum()),
          '1.0':int((fractions==1).sum())}
    assert sum(bins.values())==count
    return dict(valid_R_samples=count,m_distribution=dict(ms),q_distribution=dict(qs),
                joint_m_q_distribution=dict(joint),keep_fraction_mean=float(fractions.mean()),
                keep_fraction_std_population=float(fractions.std()),
                keep_fraction_quantiles={f'q{int(q*100):02}':float(np.quantile(fractions,q))
                                         for q in (.05,.25,.5,.75,.95)},
                keep_fraction_bins_count=bins,keep_fraction_bins_proportion={k:n/count for k,n in bins.items()},
                R_drop_equal_R_old_fraction=sum(d['same_old_r_count'] for d in diagnostics)/count)


class Experiment(Search):
    def __init__(self):
        super().__init__(run_dir=RUN,experiment_dir=EXP,base_config=BASE)
        if 'baseline' not in self.state:
            validation=load(EXP/'evidence/real-samples.json');tests=load(EXP/'evidence/correctness.json')
            assert validation['passed'] and tests['passed']
            previous=load(BASELINE_STATE)
            baseline=previous['roles']['baseline']['result']
            assert baseline['checkpoint_sha256']==BASELINE_SHA
            self.state.update(baseline=baseline,historical_rmask=previous['roles']['rmask']['result']['scores'],
                              validation=validation,tests=tests,reference_commit='fa19d12',
                              stop_updates=500,horizon=4868,baseline_retrained=False,rdrop_trial=self.enroll(hparams(BASE)))
            source=Path('/root/lk_projects/SAID-balanced-rmask-500-v1/experiments/nest_clip_v1/balanced_rmask_500_v1/baseline/evidence')
            shutil.copytree(source/self.state['rdrop_trial'][:12]/'step500',EXP/'evidence/baseline',dirs_exist_ok=True)
            self.save();self.publish()

    def publish(self):
        super().publish()
        if 'baseline' in self.state:
            from experiments.nest_clip_v1.balanced_rdrop_500_v1.report import write_report
            write_report(self.state,EXP)

    def sync(self,message):
        self.publish()
        assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()==BRANCH
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        change=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
        assert change in (0,1)
        if change:subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
        push=subprocess.run(['git','push','origin',BRANCH],cwd=REPO,timeout=180)
        self.state.update(github_synced=push.returncode==0,
                          result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        self.save()
        if push.returncode:raise RuntimeError('Evidence push failed; results retained locally')

    def run(self):
        hp=hparams(BASE);tid=self.state['rdrop_trial']
        if '500' not in self.state['trials'][tid]['budgets']:
            self.train(hp,5,'smoke')
            root=self.train(hp,500)
            config=load(root/'config.json')
            assert config['horizon']==4868 and config['max_updates']==500 and config['start_updates']==0
            assert config['resume'] is None and config['remainder_mode']=='sentence_drop'
            assert config['init_sha256']==self.state['baseline']['config']['init_sha256']
            assert config['component_initialization']==self.state['baseline']['config']['component_initialization']
            result=self.evaluate(tid,root,500,
                stream_reference=Path(self.state['baseline']['root'])/'steps.jsonl',stream_keys=MATCH_KEYS)
        result=self.state['trials'][tid]['budgets']['500']
        if 'sentence_drop_statistics' not in result:
            root=Path(result['root'])
            result['sentence_drop_statistics']=summarize_sampling(rows(root/'steps.jsonl'))
            timing=[r['four_rank_max_seconds'] for r in rows(root/'cycle_timing.jsonl') if not r['warmup']]
            result['resource_summary']=dict(mean_seconds=statistics.fmean(timing),
                median_seconds=statistics.median(timing),p95_seconds=statistics.quantiles(timing,n=100)[94],
                max_seconds=max(timing),regular_updates=len(timing),limit_seconds=3,
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['acceptance']['ranks']),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['acceptance']['ranks']))
        result=self.state['trials'][tid]['budgets']['500']
        self.state.update(status='completed',stage='single500-final-comparison',finished_utc=now(),
                          delta_scores_pp={k:100*(result['scores'][k]-self.state['baseline']['scores'][k])
                                           for k in result['scores']})
        self.save()
        self.sync('Report ordered R-SentenceDrop500 against reused matched old-R baseline and stop')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--launch-finalizer',action='store_true')
    parser.add_argument('--finish-after-current',action='store_true')
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.launch or args.launch_finalizer:
        command=[PYTHON,'-u','-m','experiments.nest_clip_v1.balanced_rdrop_500_v1.run']
        if args.launch_finalizer:command.append('--finish-after-current')
        with (RUN/('finalizer.console.txt' if args.launch_finalizer else 'supervisor.console.txt')).open('ab') as log:
            process=subprocess.Popen(command,
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=process.pid,runtime=str(RUN))));return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if args.finish_after_current else fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.finish_after_current:
            state=load(RUN/'state.json')
            if state['status']=='completed':
                print(json.dumps(dict(status='already completed')));return
            if ('500' not in state['trials'][state['rdrop_trial']]['budgets'] or
                    state.get('error') not in ("'sentence_drop_statistics'","'resource_summary'")):
                print(json.dumps(dict(status='no reporting-only recovery permitted',error=state.get('error'))));return
        controller=Experiment()
        if args.finish_after_current:
            controller.state['reporting_recovery']=dict(original_error=controller.state.pop('error'),
                original_failed_utc=controller.state.pop('failed_utc',None),recovered_utc=now(),
                cause='Report callback ran before the post-evaluation statistics were aggregated',
                training_or_evaluation_repeated=False)
            controller.save()
        try:controller.run()
        except Exception as exc:
            controller.state.update(status='failed',error=str(exc),failed_utc=now())
            controller.save();controller.publish();raise


if __name__=='__main__':main()
