"""Fixed seed coordinate descent and budget promotion, with immutable trial identities."""
import argparse
import copy
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from model.balanced_hparam_search import hparams,trial_id
from experiments.nest_clip_v1.stack_crossscore_v1.summarize import load,rows,metrics,sha,signatures


EXP=Path(__file__).resolve().parent
REPO=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_hparam_search_v1')
ASSETS=Path('/root/lk_projects/SAID-assets')
SHARED=Path('/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt')
PYTHON='/root/miniconda3/envs/said-repro/bin/python'
TORCHRUN='/root/miniconda3/envs/said-repro/bin/torchrun'
BRANCH='codex/nest-balanced-hparam-search-v1'
PARENT500=Path('/root/lk_projects/SAID-nest-clip-v1/mask_balance_cosine_v1/formal/Balanced-Stack-Patch')
PARENT3651=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_stack_3epoch_v1/formal/Balanced-Stack-Patch')
BASE_CONFIG=load(REPO/'configs/nest_balanced_stack_patch_3epoch.json')
DATASETS=('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')
ROUNDS=(('fusion_lr',(5e-5,2e-4)),('visual_mask_lr_scale',(.5,2.)),
        ('view_weights',([2,1,1],[1,1,1.5])),('sparsity_scale',(.75,1.25)),('inclusion_max',(.5,1.5)))


def scores(m):
    r1=[m[ds][dr]['R@1'] for ds in DATASETS for dr in ('I2T','T2I')]
    return dict(Score5_R1=statistics.fmean(r1),
                J_long3=statistics.fmean(m[ds][dr]['R@1'] for ds in ('Urban-1k','DOCCI','Long-DCI') for dr in ('I2T','T2I')),
                J_long=statistics.fmean(m[ds][dr]['R@1'] for ds in ('Urban-1k','DOCCI') for dr in ('I2T','T2I')))


def rank_key(record):
    # Raw floats, lexicographic tie-breakers only; no rounded scores or improvement threshold.
    return record['scores']['Score5_R1'],record['scores']['J_long3'],record['scores']['J_long']


def select_top2_500(trials):
    eligible=[tid for tid,trial in trials.items() if '500' in trial['budgets']]
    ordered=sorted(eligible,key=lambda tid:rank_key(trials[tid]['budgets']['500']),reverse=True)
    assert len(ordered)>=2
    return ordered[:2]


def native_metrics(root):
    m,raw,sources=metrics(root)
    long=load(root/'long_dci/long_dci.json')
    assert long['n_images']==long['n_captions']==7602
    assert long['manifest_sha256']=='8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b'
    m['Long-DCI']={dr:{rec:long['metrics'][rec][dr] for rec in ('R@1','R@5','R@10')} for dr in ('I2T','T2I')}
    raw['Long-DCI']=long;sources['Long-DCI']=root/'long_dci/long_dci.json'
    checks=load(root/'export-check.json')
    assert checks['passed'] and checks['strict_load']
    assert all(r['checkpoint_sha256']==checks['bare_sha256'] for r in raw.values())
    return m,raw,sources


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


class Search:
    def __init__(self,bootstrap=False,*,run_dir=RUN,experiment_dir=EXP,base_config=None,init_state=SHARED):
        self.run_dir=Path(run_dir)
        self.experiment_dir=Path(experiment_dir)
        self.base_config=copy.deepcopy(BASE_CONFIG if base_config is None else base_config)
        self.init_state=Path(init_state)
        self.state_path=self.run_dir/'state.json'
        if self.state_path.exists():
            self.state=load(self.state_path)
            assert self.state['status']!='completed','Search already complete'
        else:
            self.state=dict(status='initializing',started_utc=now(),trials={},rounds=[],promotions={},stages=[],
                            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        self.bootstrap=bootstrap
        self.state['supervisor_pid']=os.getpid()
        self.run_dir.mkdir(parents=True,exist_ok=True)
        (self.experiment_dir/'evidence').mkdir(parents=True,exist_ok=True)
        self.save()

    def save(self):
        temp=self.state_path.with_suffix('.tmp');temp.write_text(json.dumps(self.state,indent=2)+'\n');temp.replace(self.state_path)

    def command(self,key,command,gpu=True):
        log=self.run_dir/'execution'/f'{key}.console.txt'
        record=self.run_dir/'execution'/f'{key}.json'
        if record.exists():
            result=load(record)
            if result['exit_code']==0:return result
            raise RuntimeError(f'Preserved failed stage {key}: {log}; no automatic rerun')
        assert not log.exists(),f'Interrupted stage remains preserved: {log}'
        if gpu:
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert not occupied,f'GPU occupation; cannot launch {key}: {occupied}'
        log.parent.mkdir(exist_ok=True)
        self.state.update(status='running',stage=key,stage_started_utc=now());self.save()
        tick=time.monotonic()
        data=dict(command=command,cwd=str(REPO),started_utc=now(),
                  code_commit=self.state['code_commit'],stage=key)
        with log.open('x') as handle:
            child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            self.state['stage_pid']=child.pid;self.save()
            code=child.wait()
        data.update(exit_code=code,elapsed_seconds=time.monotonic()-tick,finished_utc=now())
        record.write_text(json.dumps(data,indent=2)+'\n');self.state['stages'].append(data);self.save()
        evidence=self.experiment_dir/'evidence/execution';evidence.mkdir(exist_ok=True)
        shutil.copy2(record,evidence/record.name);shutil.copy2(log,evidence/log.name)
        if code:raise subprocess.CalledProcessError(code,command)
        return data

    def config(self,hp):
        cfg=dict(self.base_config,**hparams(hp));tid=trial_id(hp)
        cfg.update(hparam_search=True,trial_id=tid,experiment_name=f'BalancedSearch-{tid[:12]}',
                   checkpoint_interval=100000,save_initial_checkpoint=False)
        if cfg.get('base_model')=='ViT-L/14':
            cfg['save_initial_checkpoint']=self.base_config.get('save_initial_checkpoint',True)
        path=self.run_dir/'configs'/f'{tid}.json';path.parent.mkdir(exist_ok=True)
        if path.exists():assert load(path)==cfg
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        compact_path=self.experiment_dir/'configs';compact_path.mkdir(exist_ok=True)
        shutil.copy2(path,compact_path/path.name)
        return tid,path

    def train(self,hp,stop,kind='formal',resume=None,legacy=False):
        tid,cfg=self.config(hp)
        root=self.run_dir/'trials'/tid/(kind if kind!='formal' else f'step{stop}')
        command=[TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
                 '-m','train.train_nested_semantic_mask','--config',str(cfg),'--init-state',str(self.init_state),
                 '--index-dir','/root/lk_projects/SAID-nest-clip-v1/data_index','--image-root',
                 str(ASSETS/'training/ShareGPT4V'),'--output-dir',str(root),'--run-type',kind,'--max-updates',str(stop)]
        if resume:
            command+=['--resume',str(resume)]
            if legacy:command+=['--legacy-b0']
        # The trainer creates only the final directory; ensure its unique parent exists.
        root.parent.mkdir(parents=True,exist_ok=True)
        self.command(f'{tid[:12]}-{kind}-{stop}',command)
        accepted=load(root/'acceptance.json')
        dest=self.experiment_dir/'evidence'/tid[:12];dest.mkdir(exist_ok=True)
        for filename in ('config.json','acceptance.json'):
            shutil.copy2(root/filename,dest/f'{kind}-{stop}-{filename}')
        if not accepted['passed']:
            if kind=='probe':raise RuntimeError(f'Common B0 resource gate failed: {accepted}')
            raise TrialFailure(f'Non-passing {kind} acceptance: {accepted}')
        return root

    def evaluate(self,tid,root,stop,reused=False,training_root=None):
        training_root=Path(training_root) if training_root is not None else root
        root.mkdir(parents=True,exist_ok=True)
        checkpoint=training_root/f'step{stop:06d}.pt';bare=root/f'student_step{stop}.pt'
        if not reused:
            prefix=f'{tid[:12]}-{stop}'
            self.command(prefix+'-export',[PYTHON,'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),
                                         '--expect-updates',str(stop),'--output',str(bare)],gpu=False)
            self.command(prefix+'-verify',[PYTHON,'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),
                                         '--bare',str(bare),'--output',str(root/'export-check.json'),
                                         '--index-dir','/root/lk_projects/SAID-nest-clip-v1/data_index','--image-root',str(ASSETS/'training/ShareGPT4V')],gpu=False)
            for name,path in [('coco',ASSETS/'evaluation/coco/val2017'),('urban',ASSETS/'evaluation/Urban1k/Urban1k')]:
                self.command(prefix+'-'+name,[PYTHON,'-m','tools.eval_nest_native','--checkpoint',str(bare),
                                             '--dataset',name,'--root',str(path),'--device','cuda:0','--batch-size','64',
                                             '--output',str(root/f'{name}_native.json')])
            bench=ASSETS/'retrieval_benchmarks'
            for name,manifest,images in [('flickr_test1k','flickr30k_test1k.jsonl','flickr30k/images'),
                                         ('docci','docci_test.jsonl','docci/images'),
                                         ('long_dci','long_dci_reconstructed.jsonl','dci/images')]:
                self.command(prefix+'-'+name,[PYTHON,'-m','experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                                             '--checkpoint',str(bare),'--device','cuda:0','--batch-size','64','--output-dir',str(root/name),
                                             f'{name}:{bench/"manifests"/manifest}:{bench/images}'])
        m,raw,sources=native_metrics(root)
        check=load(root/'export-check.json');assert check['optimizer_steps']==([] if stop==0 else [stop])
        score=scores(m)
        trial=self.state['trials'][tid]
        result=dict(root=str(root),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                    student=str(bare),bare_sha256=sha(bare),metrics=m,scores=score,reused=reused,
                    export_check=check,config=load(training_root/'config.json'),acceptance=load(training_root/'acceptance.json'))
        records=([r for r in rows(training_root/'steps.jsonl') if r['step']<=stop] if stop else [])
        result['step_range']=[records[0]['step'],records[-1]['step']] if records else [0,0]
        if stop:
            assert records and records[-1]['step']==stop
            assert [r['step'] for r in records]==list(range(records[0]['step'],stop+1))
        assert all(r['nonfinite']==0 and all(h['gradients_finite'] for h in r['rank_health']) for r in records)
        if stop==500:
            reference=rows(PARENT500/'steps.jsonl')
            assert signatures(records)==signatures(reference),'Sample/F/P/R/K stream drift'
            result['streams_equal_B0']=True
        result['diagnostics']={str(r['step']):{k:v for k,v in r.items() if k!='rank_health'} for r in records
                               if r['step'] in (1,100,200,500,1217,2000,2434,3000,3651,4868)}
        result['last50_mean']={k:statistics.fmean(r[k] for r in records[-50:] if k in r)
                               for k,v in (records[-1] if records else {}).items() if isinstance(v,(float,int))}
        if records and (training_root/'cycle_timing.jsonl').exists():
            cycles=[r for r in rows(training_root/'cycle_timing.jsonl') if r['step']<=stop]
            vals=[r['four_rank_max_seconds'] for r in cycles if r['step']>records[0]['step']]
            if vals:result['timing']=dict(mean_seconds=statistics.fmean(vals),median_seconds=statistics.median(vals),
                                          p95_seconds=statistics.quantiles(vals,n=100)[94],max_seconds=max(vals))
        trial.setdefault('budgets',{})[str(stop)]=result;trial['status']='completed';self.save()
        dest=self.experiment_dir/'evidence'/tid[:12]/f'step{stop}';dest.mkdir(parents=True,exist_ok=True)
        for name,source in sources.items():shutil.copy2(source,dest/f'{name.lower().replace("-","_")}.json')
        for filename in ('config.json','acceptance.json'):
            shutil.copy2(training_root/filename,dest/filename)
        shutil.copy2(root/'export-check.json',dest/'export-check.json')
        self.publish()
        return result

    def publish(self):
        (self.experiment_dir/'SEARCH_STATE.json').write_text(json.dumps(self.state,indent=2)+'\n')
        columns=['trial_id',*hparams({}),'Score5_R1','J_long3','J_long','checkpoint_sha256','reused']
        for budget in (500,1217,3651,4868):
            entries=[]
            for tid,trial in self.state['trials'].items():
                result=trial.get('budgets',{}).get(str(budget))
                if result:
                    entries.append((result,dict(trial_id=tid,**trial['hparams'],**result['scores'],
                                                 checkpoint_sha256=result['checkpoint_sha256'],reused=result['reused'])))
            entries.sort(key=lambda item:rank_key(item[0]),reverse=True)
            with (self.experiment_dir/f'leaderboard_{budget}.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
                for _,record in entries:
                    record['view_weights']=json.dumps(record['view_weights'],separators=(',',':'))
                    writer.writerow(record)

    def enroll(self,hp):
        tid,_=self.config(hp)
        self.state['trials'].setdefault(tid,dict(hparams=hparams(hp),seed=0,status='pending',budgets={}))
        self.save();return tid

    def candidate(self,hp):
        tid=self.enroll(hp);trial=self.state['trials'][tid]
        if '500' in trial['budgets']:return tid
        if trial['status']=='failed':return tid
        try:
            self.train(hp,5,'smoke')
            root=self.train(hp,500)
        except (subprocess.CalledProcessError,TrialFailure) as exc:
            # A numerical failure excludes this trial; no hyperparameter repair/retry.
            trial.update(status='failed',error=str(exc));self.save();self.publish();return tid
        self.evaluate(tid,root,500)
        return tid

    def run(self):
        b0=self.enroll({})
        if not self.state.get('common_resource_gate'):
            root=self.train({},35,'probe');self.state['common_resource_gate']=load(root/'acceptance.json');self.save()
        if '500' not in self.state['trials'][b0]['budgets']:
            self.evaluate(b0,PARENT500,500,reused=True)
        if '3651' not in self.state['trials'][b0]['budgets']:
            self.evaluate(b0,PARENT3651,3651,reused=True)
        if self.bootstrap:
            self.state.update(status='ready',stage='validated-common-path');self.save();self.publish();return
        best=b0
        for number,(field,values) in enumerate(ROUNDS,1):
            if len(self.state['rounds'])>=number:
                best=self.state['rounds'][number-1]['best_after'];continue
            old_best=best;participants=[best]
            for value in values:
                hp=copy.deepcopy(self.state['trials'][old_best]['hparams']);hp[field]=value
                participants.append(self.candidate(hp))
            eligible=[tid for tid in participants if '500' in self.state['trials'][tid]['budgets']]
            best=max(eligible,key=lambda tid:rank_key(self.state['trials'][tid]['budgets']['500']))
            self.state['rounds'].append(dict(round=number,coordinate=field,best_before=old_best,participants=participants,best_after=best))
            self.save();self.publish()
        top2=select_top2_500(self.state['trials'])
        self.state['promotions'].update(policy='global_top2_500_direct_to_3651',top2_500=top2)
        self.save();self.publish()
        for tid in top2:
            trial=self.state['trials'][tid]
            if '3651' not in trial['budgets']:
                root=self.train(trial['hparams'],3651,resume=Path(trial['budgets']['500']['checkpoint']),legacy=tid==b0)
                self.evaluate(tid,root,3651)
        final=max([b0,*top2],key=lambda tid:rank_key(self.state['trials'][tid]['budgets']['3651']))
        self.state.update(status='completed',stage='final-results',best_3651=final,finished_utc=now())
        self.save();self.publish()
        self.command('final-report',[PYTHON,'-m','experiments.nest_clip_v1.balanced_hparam_search_v1.report'],gpu=False)
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m','Report fixed-seed Balanced coordinate search and promotions'],cwd=REPO,check=True)
        pushed=subprocess.run(['git','push','origin',BRANCH],cwd=REPO)
        self.state.update(github_push_exit_code=pushed.returncode,result_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),github_synced=pushed.returncode==0)
        self.state['status']='completed' if pushed.returncode==0 else 'sync_failed';self.save()


class TrialFailure(RuntimeError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',action='store_true');args=parser.parse_args()
    runner=Search(args.bootstrap)
    try:runner.run()
    except Exception as exc:
        runner.state.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_utc=now())
        runner.save();runner.publish();raise


if __name__=='__main__':main()
