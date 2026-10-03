"""Four fixed independent arms; no adaptive or full-horizon training."""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import traceback

import torch
from torch.utils.data import DistributedSampler
from model.balanced_hparam_search import trial_id
from train.nested_semantic_data import file_sha
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1 import run as base

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/four_arm_text_search_500_v1')
BRANCH='codex/nest-balanced-four-arm-text-search-500-v1'
PREVIOUS=ROOT/'experiments/nest_clip_v1/balanced_summary_random_detail_500_v1'
ARMS={
 'A':('arm_A_interior_randomk','interior_random_k',[1.,1.,1.],1.),
 'B':('arm_B_summary_weight','summary_random_detail',[1.2,.6,1.2],1.),
 'C':('arm_C_summary_t2i_weight','summary_random_detail',[1.,1.,1.],.5),
 'D':('arm_D_contiguous_detail','summary_contiguous_detail',[1.,1.,1.],1.)}


def setup():
    RUN.mkdir(parents=True,exist_ok=True)
    baseline=base.load(PREVIOUS/'BASELINE.json')
    previous=base.load(PREVIOUS/'RESULTS.json')
    assert file_sha(base.SHARED)==base.INIT_SHA
    for record,full,bare in [(baseline,'checkpoint','student'),(previous,'checkpoint','bare_student')]:
        assert file_sha(record[full])==record['checkpoint_sha256']
        assert file_sha(record[bare])==record['bare_sha256']
    base.dump(EXP/'BASELINE.json',baseline)
    base.dump(EXP/'PREVIOUS_SUMMARY_RANDOM_DETAIL.json',previous)
    config=base.load(PREVIOUS/'config.json')
    for arm,(directory,mode,weights,t2i) in ARMS.items():
        c=dict(config,sampling_mode=mode,view_weights=weights,experiment_name=f'FourArm-{arm}-500')
        if t2i!=1.:c['summary_t2i_weight']=t2i
        c['trial_id']=trial_id(c)
        target=EXP/directory
        for folder in ['evidence','commands','raw']: (target/folder).mkdir(parents=True,exist_ok=True)
        base.dump(target/'config.json',c);base.dump(target/'BASELINE.json',baseline)
        allowed={'sampling_mode','experiment_name','trial_id','view_weights','summary_t2i_weight'}
        assert all(v==baseline['config'][k] for k,v in c.items() if k not in allowed)
    # Native evaluation, backbone, candidate construction and optimizer/scheduler unchanged.
    frozen=['model/nested_semantic_mask.py','model/nested_vcp_mask.py','model/model_longclip.py',
      'model/longclip.py','model/said_cls_cvssl.py','train/said_cvssl_data.py','eval/retrieval/coco_retrieval.py',
      'tools/urban1k_retrieval.py','tools/eval_nest_native.py','tools/nest_clip.py',
      'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py']
    hashes={}
    for path in frozen:
        old=subprocess.check_output(['git','show','3da12a3:'+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==old,path
        hashes[path]=hashlib.sha256(old).hexdigest()
    def functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    old=functions(subprocess.check_output(['git','show','14653c9:train/train_nested_semantic_mask.py'],cwd=ROOT,text=True))
    current=functions((ROOT/'train/train_nested_semantic_mask.py').read_text())
    names=['build_optimizer','learning_rates','optimizer_learning_rates','training_horizon','seed_all','rng_state','restore_rng_state']
    assert all(old[n]==current[n] for n in names)
    olddata=functions(subprocess.check_output(['git','show','3da12a3:train/nested_semantic_data.py'],cwd=ROOT,text=True))
    newdata=functions((ROOT/'train/nested_semantic_data.py').read_text())
    assert all(olddata[n]==newdata[n] for n in ['text_views','sample_split_k','sample_detail_indices'])
    assert base.load(base.INDEX/'metadata.json')==baseline['config']['data']
    assert file_sha(base.INDEX/'records.jsonl')==baseline['config']['data']['records_sha256']
    audit=base.load(EXP/'evidence/sampling-1000.json');assert audit['passed']
    rows=base.records(Path(baseline['root'])/'steps.jsonl');assert len(rows)==500
    for rank in range(4):
        sampler=DistributedSampler(range(1245901),num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        ids=[i+1000 for i in itertools.islice(iter(sampler),500*256)]
        for step,row in enumerate(rows):
            h=hashlib.sha256(json.dumps(ids[step*256:(step+1)*256],separators=(',',':')).encode()).hexdigest()
            assert h==row['rank_health'][rank]['sampling']['sample_id_sha256']
    initial=torch.load(base.SHARED,map_location='cpu',weights_only=False)
    assert initial['completed_steps']==0 and not initial['optimizer']['state']
    preflight=dict(passed=True,checked_at=base.now(),baseline_retrained=False,common_step0_sha256=base.INIT_SHA,
      frozen_sources=hashes,frozen_trainer_function_AST=names,historical_sampling_1000=audit,
      all_500x4_sampler_ID_hashes_match=True,mathematical_changes={'A':'interior split only','B':'alignment view weights only','C':'live Summary T2I weighting and global 12/11 normalization only','D':'contiguous detail only'},
      default_weight_loss_gradients_AdamW_bitwise_test=True,scheduler_horizon=4868,stop_updates=500)
    base.dump(EXP/'evidence/preflight.json',preflight)
    for arm,(directory,*_) in ARMS.items():
        base.dump(EXP/directory/'evidence/preflight.json',preflight)
        base.dump(RUN/directory/'state.json',dict(status='READY',stage='preflight passed',branch=BRANCH,started_at=base.now(),arm=arm))
    base.dump(RUN/'state.json',dict(status='RUNNING',stage='preregistered',branch=BRANCH,arms={},started_at=base.now(),supervisor_pid=os.getpid()))
    print(json.dumps({'preflight':'PASS','arms':list(ARMS)}),flush=True)


def semantic_audit(arm):
    root=base.RUN/'step500'
    argv=[base.PYTHON,'-m','experiments.nest_clip_v1.four_arm_text_search_500_v1.text_audit',
      '--arm',arm,'--checkpoint',str(root/'student_step500.pt'),'--output',str(base.EXP/'SUMMARY_AMBIGUITY.json')]
    base.execute('summary-ambiguity',argv,gpu=True)


def collect_training(arm):
    root=base.RUN/'step500';rows=base.records(root/'steps.jsonl');cycles=base.records(root/'cycle_timing.jsonl')
    last=rows[-50:]
    def summary(group):
        keys=sorted(set().union(*(r.keys() for r in group)))
        metrics={}
        for key in keys:
            values=[r[key] for r in group if isinstance(r.get(key),(int,float)) and not isinstance(r.get(key),bool)]
            if values:metrics[key]={'mean':statistics.fmean(values),'observations':len(values)}
        tokens={}
        for row in group:
            for rank in row['rank_health']:
                sampling=rank['sampling']
                for field in ['prefix_token_lengths','remainder_token_lengths']:
                    hist=sampling[field]
                    count=sum(hist.values());total=sum(int(k)*v for k,v in hist.items())
                    t=tokens.setdefault(field,{'count':0,'token_sum':0});t['count']+=count;t['token_sum']+=total
                for label,stat in sampling.get('Full_Summary_Detail_token_statistics',{}).items():
                    t=tokens.setdefault(label,{'count':0,'token_sum':0});t['count']+=stat['samples'];t['token_sum']+=stat['effective_token_sum']
        for t in tokens.values():t['mean']=t['token_sum']/max(1,t['count'])
        return {'metrics':metrics,'token_lengths':tokens}
    diag={'arm':arm,'steps':{str(i):summary([rows[i-1]]) for i in [1,100,200,500]},'last50':summary(last),
      'diagnostic_observation_note':'Gate and F-D IoU are collected only at scheduled diagnostic updates; last50 gate/FD values have one observation (step500), not a 50-update mean.'}
    base.dump(base.EXP/'TRAINING_DIAGNOSTICS.json',diag)
    normal=[r['four_rank_max_seconds'] for r in cycles if not r['warmup']]
    accept=base.load(root/'acceptance.json');config=base.load(root/'config.json')
    baseline=base.load(EXP/'BASELINE.json')
    assert config['adapter_initialization']['state_sha256']==baseline['config']['adapter_initialization']['state_sha256']
    if arm=='C':assert config['summary_t2i_weight']==.5
    resource={'normal_full_cycle_mean_seconds':statistics.fmean(normal),'normal_cycles':len(normal),
      'peak_allocated_gib':max(r['peak_allocated_gib'] for r in accept['ranks']),
      'passed':statistics.fmean(normal)<=3 and max(r['peak_allocated_gib'] for r in accept['ranks'])<=65}
    base.dump(base.EXP/'RESOURCE_SUMMARY.json',resource)
    for f in ['steps.jsonl','cycle_timing.jsonl']:
        shutil.copy2(root/f,base.EXP/'evidence'/f)
    return resource


def all_arms():
    setup()
    for arm,(directory,*_) in ARMS.items():
        base.EXP=EXP/directory;base.RUN=RUN/directory;base.ROOT=ROOT;base.BRANCH=BRANCH
        state=base.load(RUN/'state.json');state.update(stage=f'arm {arm}',active_arm=arm);base.dump(RUN/'state.json',state)
        try:
            base.train('smoke');base.train('formal')
            resource=collect_training(arm)
            base.evaluate()
            if arm in 'BCD':semantic_audit(arm)
            result=base.load(base.RUN/'result.json');result.update(status='COMPLETE',arm=arm,resources=resource)
            result['scores']['Short4_R1']=statistics.fmean(result['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I'])
            base.dump(base.EXP/'RESULTS.json',result)
            local=base.load(base.RUN/'state.json');local.update(status='COMPLETED',stage='five datasets and diagnostics complete');base.dump(base.RUN/'state.json',local)
            status='COMPLETE'
        except Exception as error:
            logs='\n'.join(p.read_text(errors='replace')[-16000:] for p in (base.RUN/'execution').glob('*.txt'))
            resource_fail=any(x in (str(error)+'\n'+logs).lower() for x in ['resource_abort','resource abort','feasibility','out of memory','65 gib','three consecutive'])
            # Failed resource gates can be bare assertions after a completed formal stage.
            if (base.RUN/'step500/cycle_timing.jsonl').exists():
                cycles=base.records(base.RUN/'step500/cycle_timing.jsonl');normal=[r['four_rank_max_seconds'] for r in cycles if not r['warmup']]
                resource_fail=resource_fail or bool(normal and statistics.fmean(normal)>3)
            status='RESOURCE_FAIL' if resource_fail else 'FAILED'
            failure={'status':status,'arm':arm,'error':repr(error),'traceback':traceback.format_exc()}
            base.dump(base.EXP/'RESULTS.json',failure)
            print(json.dumps(failure),flush=True)
        state=base.load(RUN/'state.json');state['arms'][arm]=status;base.dump(RUN/'state.json',state)
    from experiments.nest_clip_v1.four_arm_text_search_500_v1.report import generate
    generate()

if __name__=='__main__':all_arms()
