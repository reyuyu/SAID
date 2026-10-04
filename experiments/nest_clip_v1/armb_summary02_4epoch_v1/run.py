"""Authorized S0.2 continuation500→4868 with unchanged horizon and native5 eval."""
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

import torch
from torch.utils.data import DistributedSampler
from train.nested_semantic_data import file_sha
from train.train_nested_semantic_mask import validate_resume_payload,code_manifest,training_horizon,optimizer_learning_rates
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1 import run as base
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/armb_summary02_4epoch_v1')
BRANCH='codex/nest-balanced-armb-summary02-4epoch-v1'
SOURCE=ROOT/'experiments/nest_clip_v1/armb_summary_dose_500_v1/arm_E1_summary02'
STOP=4868


def preflight():
    parent=base.load(SOURCE/'RESULTS.json');simple=base.load(SOURCE/'config.json')
    assert parent['checkpoint_sha256']=='a15c5360fa34991fd0e9767ff158c232c97f1414762b7f58540ec40701e7769e'
    assert file_sha(parent['checkpoint'])==parent['checkpoint_sha256']
    assert file_sha(base.SHARED)==base.INIT_SHA
    config=dict(simple,experiment_name='ArmB-Summary0.2-4epoch',checkpoint_interval=1217)
    assert config['view_weights']==[1.4,.2,1.4] and config['sampling_mode']=='summary_random_detail'
    assert config['epochs']==4 and training_horizon(config,1217)==STOP
    assert {k for k in config if config[k]!=simple[k]}=={'experiment_name','checkpoint_interval'}
    frozen={}
    for path in parent['config']['code_sha256']:
        original=subprocess.check_output(['git','show','00c088f:'+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==original,path;frozen[path]=file_sha(ROOT/path)
    checkpoint=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
    current=dict(parent['config'],**config)
    current.update(max_updates=STOP,resume=parent['checkpoint'],code_sha256=code_manifest(),output_dir=str(RUN/'step4868'))
    assert validate_resume_payload(checkpoint,current)==500
    optimizer_steps=sorted({float(v['step']) for v in checkpoint['optimizer']['state'].values()})
    assert optimizer_steps==[500.]
    assert checkpoint['next_epoch']==0 and checkpoint['next_batch']==500
    assert all('loader_generator' in state for state in checkpoint['rng_per_rank'])
    optimizer_shapes={group['name']:len(group['params']) for group in checkpoint['optimizer']['param_groups']}
    before_parent_rng=[{'python_present':'python' in r,'torch_sha256':hashlib.sha256(r['cpu'].numpy().tobytes()).hexdigest(),
      'cuda_sha256':hashlib.sha256(r['cuda'].numpy().tobytes()).hexdigest(),'loader_generator_sha256':hashlib.sha256(r['loader_generator'].numpy().tobytes()).hexdigest()} for r in checkpoint['rng_per_rank']]
    del checkpoint
    assert base.load(base.INDEX/'metadata.json')==parent['config']['data']
    assert file_sha(base.INDEX/'records.jsonl')==parent['config']['data']['records_sha256']
    # Pin the established official RandomK H4868/4868 native reference, without retraining.
    candidates=base.load(ROOT/'experiments/nest_clip_v1/three_followup_v1/RESULTS.json')['evaluations']
    reference=next(r for r in candidates if r['updates']==4868 and r['config']['horizon']==4868)
    assert abs(reference['scores']['Score5_R1']*100-72.768147)<1e-5
    base.dump(EXP/'RANDOMK_4EPOCH_REFERENCE.json',reference)
    base.dump(EXP/'PARENT_500.json',parent);base.dump(EXP/'config.json',config)
    proof=dict(passed=True,parent_checkpoint=parent['checkpoint'],parent_sha256=parent['checkpoint_sha256'],
      common_step0_sha256=base.INIT_SHA,start_updates=500,stop_updates=STOP,updates_to_run=STOP-500,
      epochs=4,batches_per_epoch=1217,scheduler_horizon=STOP,optimizer_state_steps=optimizer_steps,
      optimizer_state_group_parameter_counts=optimizer_shapes,parent_rng_per_rank=before_parent_rng,
      resume_payload_validator_pass=True,production_source_hashes=frozen,
      checkpoint_IO_only_change='checkpoint_interval1217 saves complete full checkpoints at epoch boundaries',
      no_math_sampling_optimizer_scheduler_or_inference_changes=True,
      full_epoch_tail_local_batch=180,normal_local_batch=256,
      tail_note='Original full DistributedSampler tail remains180/rank,global720,once per epoch; ordinary global batches1024.')
    base.dump(EXP/'evidence/preflight.json',proof)
    base.dump(RUN/'state.json',dict(status='READY',stage='resume preflight passed',branch=BRANCH,parent_sha256=parent['checkpoint_sha256'],
      start_updates=500,stop_updates=STOP,started_at=base.now()))
    print(json.dumps({'preflight':'PASS','resume_step':500,'stop':STOP,'optimizer_step':500,'horizon':STOP}),flush=True)


def expected_stream():
    count=1245901;batch=256
    streams={}
    for rank in range(4):
        for epoch in range(4):
            sampler=DistributedSampler(range(count),num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(epoch)
            ids=[i+1000 for i in sampler]
            for b in range(1217):
                step=epoch*1217+b+1
                if step<=500:continue
                group=ids[b*batch:(b+1)*batch]
                h=hashlib.sha256(json.dumps(group,separators=(',',':')).encode()).hexdigest()
                streams.setdefault(step,{})[rank]=dict(epoch=epoch,batch_size=len(group),sample_id_sha256=h)
    assert len(streams)==4368
    return streams


def train():
    assert base.load(EXP/'evidence/preflight.json')['passed']
    parent=base.load(EXP/'PARENT_500.json');root=RUN/'step4868'
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not occupied,occupied
    command=[base.TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0','-m','train.train_nested_semantic_mask',
      '--config',str(EXP/'config.json'),'--init-state',str(base.SHARED),'--index-dir',str(base.INDEX),
      '--image-root',str(base.ASSETS/'training/ShareGPT4V'),'--output-dir',str(root),'--run-type','formal','--max-updates',str(STOP),'--resume',parent['checkpoint']]
    record=dict(argv=command,cwd=str(ROOT),env={'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},started_at=base.now())
    base.dump(EXP/'commands/formal-4868.json',record)
    execution=RUN/'execution';execution.mkdir(parents=True,exist_ok=True);log=execution/'formal-4868.console.txt';assert not log.exists()
    expected=expected_stream();compared=0;state=base.load(RUN/'state.json');state.update(status='RUNNING',stage='formal-4868',stage_started_at=base.now());base.dump(RUN/'state.json',state)
    started=time.monotonic()
    with log.open('w') as handle:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**record['env']),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        state['stage_pid']=child.pid;base.dump(RUN/'state.json',state)
        try:
            while child.poll() is None:
                if (root/'steps.jsonl').exists():
                    lines=(root/'steps.jsonl').read_text().splitlines()
                    for line in lines[compared:]:
                        try:row=json.loads(line)
                        except json.JSONDecodeError:break
                        step=row['step'];assert step==501+compared and row['epoch']==expected[step][0]['epoch']
                        for rank in row['rank_health']:
                            ref=expected[step][rank['rank']]
                            assert rank['sampling']['sample_id_sha256']==ref['sample_id_sha256'],(step,rank['rank'])
                            assert rank['batch']==ref['batch_size']
                        compared+=1
                        if step in [501,1000,1217,2000,2434,3000,3651,4000,4868]:print(json.dumps({'matched_step':step,'epoch':row['epoch'],'loss':row['loss']}),flush=True)
                time.sleep(2)
            code=child.wait()
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise
    record.update(exit_code=code,elapsed_seconds=time.monotonic()-started,finished_at=base.now());base.dump(EXP/'commands/formal-4868.json',record)
    assert code==0,f'4epoch continuation failed; preserve logs at {log}'
    rows=base.records(root/'steps.jsonl');assert len(rows)==4368 and rows[0]['step']==501 and rows[-1]['step']==STOP
    for row in rows:
        for rank in row['rank_health']:assert rank['sampling']['sample_id_sha256']==expected[row['step']][rank['rank']]['sample_id_sha256']
    acceptance=base.load(root/'acceptance.json');assert acceptance['passed']
    assert all(r['completed_updates']==STOP and r['updates_this_run']==4368 and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
    actual=base.load(root/'config.json');assert actual['horizon']==STOP and actual['start_updates']==500 and actual['parent_completed_updates']==500
    assert actual['parent_checkpoint_sha256']==parent['checkpoint_sha256'] and actual['code_sha256']==parent['config']['code_sha256']
    for checkpoint_step in [1217,2434,3651,4868]:assert (root/f'step{checkpoint_step:06d}.pt').exists()
    for name in ['config.json','acceptance.json','text_examples.json']:base.dump(EXP/'evidence'/name,base.load(root/name))
    base.dump(EXP/'evidence/stream-match.json',dict(passed=True,steps=4368,ranks=4,start_step=501,stop_step=4868,
      sampler_epoch_and_ID_and_tail_match=True,no_repeated_or_skipped_optimizer_updates=True))
    for offset in range(0,len(rows),500):
        chunk=rows[offset:offset+500];name=f'steps-{chunk[0]["step"]:06d}-{chunk[-1]["step"]:06d}.jsonl.gz'
        with (EXP/'evidence'/name).open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as archive:
                archive.write(('\n'.join(json.dumps(row) for row in chunk)+'\n').encode())
    for name in ['cycle_timing.jsonl','checkpoint_timing.jsonl']:
        with (root/name).open('rb') as src,(EXP/'evidence'/(name+'.gz')).open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as archive:shutil.copyfileobj(src,archive)
    state=base.load(RUN/'state.json');state.update(stage='4868 training complete',formal_updates=STOP);base.dump(RUN/'state.json',state)


def evaluate():
    base.EXP=EXP;base.RUN=RUN;base.ROOT=ROOT;base.BRANCH=BRANCH
    root=RUN/'step4868';ckpt=root/'step004868.pt';bare=root/'student_step4868.pt'
    base.execute('export',[base.PYTHON,'-m','tools.nest_clip','export','--checkpoint',str(ckpt),'--expect-updates',str(STOP),'--output',str(bare)])
    base.execute('verify-export',[base.PYTHON,'-m','tools.nest_clip','verify-export','--checkpoint',str(ckpt),'--bare',str(bare),
      '--output',str(root/'export-check.json'),'--index-dir',str(base.INDEX),'--image-root',str(base.ASSETS/'training/ShareGPT4V')])
    for name,path in [('coco',base.ASSETS/'evaluation/coco/val2017'),('urban',base.ASSETS/'evaluation/Urban1k/Urban1k')]:
        base.execute('eval-'+name,[base.PYTHON,'-m','tools.eval_nest_native','--checkpoint',str(bare),'--dataset',name,'--root',str(path),
          '--device','cuda:0','--batch-size','64','--output',str(root/(name+'_native.json'))],gpu=True)
    bench=base.ASSETS/'retrieval_benchmarks'
    for name,manifest,folder in [('flickr_test1k','flickr30k_test1k.jsonl','flickr30k'),('docci','docci_test.jsonl','docci'),('long_dci','long_dci_reconstructed.jsonl','dci')]:
        base.execute('eval-'+name,[base.PYTHON,'-m','experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
          '--checkpoint',str(bare),'--device','cuda:0','--batch-size','64','--output-dir',str(root/name),f'{name}:{bench/"manifests"/manifest}:{bench/folder/"images"}'],gpu=True)
    m,raw,_=native_metrics(root);s=scores(m);s['Short4_R1']=statistics.fmean(m[ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I'])
    result=dict(status='EVALUATED',checkpoint=str(ckpt),checkpoint_sha256=file_sha(ckpt),bare_student=str(bare),bare_sha256=file_sha(bare),
      metrics=m,scores=s,updates=STOP,epochs=4,horizon=STOP,parent500=base.load(EXP/'PARENT_500.json'),config=base.load(root/'config.json'),strict_export=base.load(root/'export-check.json'))
    base.dump(EXP/'RESULTS.json',result)
    for name,value in raw.items():base.dump(EXP/'raw'/(name+'.json'),value)
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import generate
    generate()


def main():preflight();train();evaluate()

if __name__=='__main__':main()
