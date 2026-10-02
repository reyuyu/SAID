"""Only matched old-R@500 and position-preserving RMask@500, then stop."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    Search,REPO,PYTHON,SHARED,load,now,hparams,trial_id,rows,
)


EXP=Path(__file__).resolve().parent
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1')
BRANCH='codex/nest-balanced-rmask-500-v1'
BASE=load(REPO/'configs/nest_balanced_rmask_500_v1.json')
ROLES=(('baseline','compact'),('rmask','prefix_pad'))
MATCH_KEYS=('sample_id_sha256','full_view_sha256','prefix_view_sha256','split_sha256',
            'fixed_first_reference_stream_sha256')


class Arm(Search):
    def __init__(self,role,mode):
        config=dict(BASE,remainder_mode=mode)
        super().__init__(run_dir=RUN/role,experiment_dir=EXP/role,base_config=config)
        self.role=role
        self.mode=mode
        self.tid=self.enroll(hparams(BASE))


class Experiment:
    def __init__(self):
        RUN.mkdir(parents=True,exist_ok=True)
        self.path=RUN/'state.json'
        if self.path.exists():
            self.state=load(self.path)
            assert self.state['status']!='completed'
        else:
            validation=load(EXP/'evidence/real-samples.json')
            tests=load(EXP/'evidence/correctness.json')
            assert validation['passed'] and tests['passed']
            self.state=dict(status='ready',started_utc=now(),reference_commit='14653c92c6da9d552a2b624ab169eaaa275cdde8',
                            roles={},validation=validation,tests=tests,
                            baseline_checkpoint_search=dict(found=False,
                              searched='/root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch',
                              decision='No step500 checkpoint in the exact B16 horizon4868 run; one matched baseline authorized'),
                            historical_score_only=dict(horizon=3651,Score5_R1=.6999002652589216),
                            stop_updates=500,horizon=4868,model='ViT-B/16',
                            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        self.state['supervisor_pid']=os.getpid()
        self.save()

    def save(self):
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state,indent=2)+'\n')
        temporary.replace(self.path)

    def publish(self):
        (EXP/'STATE.json').write_text(json.dumps(self.state,indent=2)+'\n')
        from experiments.nest_clip_v1.balanced_rmask_500_v1.report import write_report
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
        if push.returncode:raise RuntimeError('Push failed; local results preserved')

    def run(self):
        for role,mode in ROLES:
            if self.state['roles'].get(role,{}).get('status')=='completed':continue
            arm=Arm(role,mode)
            arm.state['code_commit']=self.state['code_commit'];arm.save()
            record=self.state['roles'].setdefault(role,dict(mode=mode,status='running',trial_id=arm.tid))
            self.state.update(status='running',active_role=role,stage='smoke5')
            self.save();self.publish()
            arm.train(hparams(BASE),5,'smoke')
            self.state['stage']='formal500';self.save()
            root=arm.train(hparams(BASE),500)
            config=load(root/'config.json')
            assert config['horizon']==4868 and config['start_updates']==0 and config['resume'] is None
            assert config['init_sha256']=='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
            self.state['stage']='five-native-evaluations';self.save()
            kwargs={}
            if role=='rmask':
                reference=Path(self.state['roles']['baseline']['result']['root'])/'steps.jsonl'
                kwargs=dict(stream_reference=reference,stream_keys=MATCH_KEYS)
            result=arm.evaluate(arm.tid,root,500,**kwargs)
            entries=rows(root/'steps.jsonl')
            timing=rows(root/'cycle_timing.jsonl')
            normal=[r['four_rank_max_seconds'] for r in timing if not r['warmup']]
            result['resource_summary']=dict(
                mean_seconds=statistics.fmean(normal),median_seconds=statistics.median(normal),
                p95_seconds=statistics.quantiles(normal,n=100)[94],max_seconds=max(normal),
                normal_updates=len(normal),approved_seconds=3,
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in result['acceptance']['ranks']),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in result['acceptance']['ranks']))
            if role=='rmask':
                diagnostics=[h['sampling']['position_diagnostics'] for r in entries for h in r['rank_health']]
                samples=sum(d['samples'] for d in diagnostics);valid=sum(d['valid_samples'] for d in diagnostics)
                bins={key:sum(d['suffix_first_position_counts'][key] for d in diagnostics)
                      for key in diagnostics[0]['suffix_first_position_counts']}
                result['position_summary']=dict(
                    mean_F_eot_position=sum(d['F_eot_sum'] for d in diagnostics)/samples,
                    mean_prefix_boundary=sum(d['prefix_boundary_sum'] for d in diagnostics)/valid,
                    mean_prefix_end_position=sum(d['prefix_end_position_sum'] for d in diagnostics)/valid,
                    mean_suffix_first_position=sum(d['suffix_first_position_sum'] for d in diagnostics)/valid,
                    mean_old_compact_suffix_first_position=sum(d['compact_suffix_first_position_sum'] for d in diagnostics)/valid,
                    suffix_first_bins_count=bins,suffix_first_bins_fraction={k:n/valid for k,n in bins.items()},
                    valid_samples=valid,total_samples=samples)
            record.update(status='completed',result=result)
            self.save()
            self.sync(f'Record {role} B16 horizon4868 matched500 native results')
        baseline=self.state['roles']['baseline']['result']
        rmask=self.state['roles']['rmask']['result']
        self.state['delta_scores_pp']={k:100*(rmask['scores'][k]-baseline['scores'][k]) for k in baseline['scores']}
        self.state.update(status='completed',stage='final500-comparison',finished_utc=now())
        self.save()
        self.sync('Report single-variable RMask versus matched compact remainder at500 and stop')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.launch:
        with (RUN/'supervisor.console.txt').open('ab') as log:
            child=subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.balanced_rmask_500_v1.run'],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        print(json.dumps(dict(supervisor_pid=child.pid,runtime=str(RUN))));return
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Experiment()
        try:controller.run()
        except Exception as exc:
            controller.state.update(status='failed',error=str(exc),failed_utc=now())
            controller.save();controller.publish();raise


if __name__=='__main__':main()
