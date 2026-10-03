"""One500-update view-relation experiment from shared step0; five native evals, stop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search,REPO,PYTHON,load,now,hparams,rows,native_metrics,
)
from experiments.nest_clip_v1.balanced_view_relation_500_v1.verify import EXP,RUN,BASELINE_STATE,BASELINE_SHA
from experiments.nest_clip_v1.balanced_view_relation_500_v1.report import METRICS

BASE=load(REPO/'configs/nest_balanced_view_relation_500_v1.json')
BRANCH='codex/nest-balanced-view-relation-500-v1'
MATCH_KEYS=('sample_id_sha256','full_view_sha256','prefix_view_sha256','local_views_sha256',
            'split_sha256','fixed_first_reference_stream_sha256')


def summarize_relation(records):
    def values(row):
        result={key:row[key] for key in METRICS if key in row}
        result['P_cosine_preference']=row['c_PP_mean']-row['c_RP_mean']
        result['R_cosine_preference']=row['c_RR_mean']-row['c_PR_mean']
        return result
    selected={str(row['step']):values(row) for row in records if row['step'] in (1,100,200,500)}
    assert set(selected)=={'1','100','200','500'}
    selected['last50']={key:statistics.fmean(values(row)[key] for row in records[-50:]) for key in METRICS}
    return selected


class Experiment(Search):
    def __init__(self):
        super().__init__(run_dir=RUN,experiment_dir=EXP,base_config=BASE)
        if not self.state['stages']:
            self.state['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        if 'baseline' not in self.state:
            preflight=load(EXP/'evidence/preflight.json');tests=load(EXP/'evidence/correctness.json')
            assert preflight['passed'] and tests['passed']
            baseline=load(BASELINE_STATE)['roles']['baseline']['result']
            assert baseline['checkpoint_sha256']==BASELINE_SHA
            tid=self.enroll(hparams(BASE))
            self.state.update(baseline=baseline,preflight=preflight,correctness=tests,relation_trial=tid,
                baseline_retrained=False,stop_updates=500,horizon=4868,
                original_math_reference='14653c92c6da9d552a2b624ab169eaaa275cdde8',
                runtime_reference='83e2a76',runtime_optimization_used=False)
            destination=EXP/'evidence/baseline';destination.mkdir(parents=True,exist_ok=True)
            _,_,sources=native_metrics(Path(baseline['root']))
            for name,path in sources.items():shutil.copy2(path,destination/(name.lower().replace('-','_')+'.json'))
            for name in ('config.json','acceptance.json','export-check.json'):
                shutil.copy2(Path(baseline['root'])/name,destination/name)
            self.save();self.publish()
        self.save()

    def publish(self):
        super().publish()
        if 'baseline' in self.state:
            from experiments.nest_clip_v1.balanced_view_relation_500_v1.report import write_report
            write_report(self.state,EXP)

    def sync(self,message):
        self.publish()
        assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()==BRANCH
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
        assert changed in (0,1)
        if changed:subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
        push=subprocess.run(['git','push','origin',BRANCH],cwd=REPO,timeout=180)
        self.state.update(github_synced=push.returncode==0,
            result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        self.save()
        if push.returncode:raise RuntimeError('GitHub push failed; all evidence remains local')

    def run(self):
        tid=self.state['relation_trial'];hp=hparams(BASE)
        if '500' not in self.state['trials'][tid]['budgets']:
            self.train(hp,5,'smoke')
            self.state['smoke_passed']=True;self.save()
            root=self.train(hp,500)
            config=load(root/'config.json')
            assert config['horizon']==4868 and config['max_updates']==500 and config['start_updates']==0
            assert config['resume'] is None and config['remainder_mode']=='compact'
            assert config['view_relation'] and config['sibling_coefficient']==1
            assert config['init_sha256']==self.state['baseline']['config']['init_sha256']
            assert config['component_initialization']==self.state['baseline']['config']['component_initialization']
            self.evaluate(tid,root,500,stream_reference=Path(self.state['baseline']['root'])/'steps.jsonl',stream_keys=MATCH_KEYS)
        result=self.state['trials'][tid]['budgets']['500']
        root=Path(result['root']);records=rows(root/'steps.jsonl')
        result['relation_diagnostics']=summarize_relation(records)
        times=[r['four_rank_max_seconds'] for r in rows(root/'cycle_timing.jsonl') if not r['warmup']]
        result['resource_summary']=dict(mean_seconds=statistics.fmean(times),median_seconds=statistics.median(times),
            p95_seconds=statistics.quantiles(times,n=100)[94],max_seconds=max(times),regular_updates=len(times),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['acceptance']['ranks']),
            peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['acceptance']['ranks']))
        self.state.update(status='completed',stage='view-relation-single500-complete',finished_utc=now(),
            delta_scores_pp={key:100*(value-self.state['baseline']['scores'][key]) for key,value in result['scores'].items()},
            delta_R1_pp={ds:{dr:100*(m['R@1']-self.state['baseline']['metrics'][ds][dr]['R@1']) for dr,m in metrics.items()}
                         for ds,metrics in result['metrics'].items()})
        self.save();self.sync('Report single-variable View-Relation500 native scores and diagnostics against reused baseline; stop')


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--launch',action='store_true');mode.add_argument('--prepare',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            child=subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.balanced_view_relation_500_v1.run'],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=child.pid,runtime=str(RUN))));return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'state.json').exists():
            previous=load(RUN/'state.json')
            assert previous['status'] not in ('failed','completed'),'Preserve terminal result; no automatic rerun'
        controller=Experiment()
        if args.prepare:
            controller.state.update(status='prepared',stage='verified-shared-step0-single500')
            controller.save();controller.publish();return
        try:controller.run()
        except Exception as exc:
            controller.state.update(status='failed',error=str(exc),failed_utc=now())
            controller.save();controller.sync('Preserve View-Relation500 stopped-stage evidence without retuning or retries');raise


if __name__=='__main__':main()
