"""Continue the pinned, evaluated local-only S02 checkpoint500 to exactly4868."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from recovery import s02_local500 as previous
from recovery.s02_local500 import ROOT, OUT, RUN, INDEX, CONFIG, STEP0, PYTHON, IMAGES, LOCAL, now, dump, rows, sha, distribution
from recovery.local500_policy import wait_expired

PARENT = RUN / 'step500/step000500.pt'
PARENT_SHA = '10d5c156dbef97a53cdac97c8eee7ad1879615e39a32208a1c04489e2f5f496d'
PHASE = LOCAL / 'formal-full-phase-20261006'
REVIEW = RUN / 'full-reviewed'
FINAL = RUN / 'step4868'


class LocalDataset(previous.LoggedLocalDataset):
    """Module-level class is spawn-picklable; worker path logs use full phase."""
    def __getitem__(self,index):
        previous.PHASE=PHASE
        return super().__getitem__(index)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def stream(batch):
    return digest(dict(sample_ids=batch['sample_id'].tolist(), views=batch['views'],
        tokens=[batch[k].cpu().tolist() for k in ('tokens_f','tokens_o','tokens_e')]))


def expected_lrs(completed, cfg):
    from train.train_nested_semantic_mask import learning_rates
    factor = .5 * (1 + math.cos(math.pi * completed / 4868))
    return [learning_rates(completed,4868)[0],1e-3*factor,
        1e-3*cfg['visual_mask_lr_scale']*factor,cfg['fusion_lr']*factor]


def reference():
    """Separate CPU process; frozen sampler/text only, no image decoding/RNG draw."""
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset = FullLocalDataset(INDEX,IMAGES,'summary_random_detail',0)
    before = torch.get_rng_state().clone()
    offsets = np.load(INDEX/'offsets.npy',mmap_mode='r')
    result = {}
    with (INDEX/'records.jsonl').open('rb') as handle:
        for rank in range(4):
            sampler = DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
            sampler.set_epoch(0)
            indices = list(itertools.islice(iter(sampler),500*256,505*256))
            for offset in range(5):
                chosen = indices[offset*256:(offset+1)*256]
                views = []
                for index in chosen:
                    handle.seek(int(offsets[index]))
                    record = json.loads(handle.read(int(offsets[index+1]-offsets[index])))
                    views.append(sampled_text_views(record['caption'],'summary_random_detail',0,0,index+1000))
                batch = dict(sample_id=torch.tensor([i+1000 for i in chosen]),views=[v['views'] for v in views])
                batch.update({k:torch.stack([v[k] for v in views]) for k in ('tokens_f','tokens_o','tokens_e')})
                result[f'{501+offset}:{rank}'] = dict(step=501+offset,rank=rank,sample_ids=batch['sample_id'].tolist(),stream_sha256=stream(batch))
    assert torch.equal(before,torch.get_rng_state()), 'Offline reference advanced RNG'
    return dict(passed=True,epoch=0,next_batch=500,first_update=501,horizon=4868,
        training_RNG_untouched=True,records=result,selection='Frozen DistributedSampler seed0, original ordered index, native F/S/D tokenizer')


def identity():
    import torch
    result = json.loads((OUT/'STEP500_RESULTS.json').read_text())
    actual = sha(PARENT)
    assert actual == PARENT_SHA == result['checkpoint_sha256'] == result['strict_export']['checkpoint_sha256']
    assert result['status']=='REPRODUCTION_PASS' and result['evaluation_checkpoint_immutable']
    checkpoint = torch.load(PARENT,map_location='cpu',weights_only=False)
    assert checkpoint['completed_steps']==checkpoint['global_step']==500
    assert checkpoint['scheduler_horizon']==checkpoint['scheduler']['horizon']==4868
    assert checkpoint['data_cursor']==dict(next_epoch=0,next_batch=500)
    assert len(checkpoint['rng_per_rank'])==4
    assert {int(s['step']) for s in checkpoint['optimizer']['state'].values()}=={500}
    frozen = json.loads(CONFIG.read_text())
    assert all(checkpoint['config'][key]==value for key,value in frozen.items())
    from train.train_nested_semantic_mask import code_manifest
    assert checkpoint['config']['code_sha256']==code_manifest(), 'Native source drift'
    proof = dict(status='IDENTITY_CONFIRMED',checkpoint=str(PARENT),sha256=actual,size_bytes=PARENT.stat().st_size,
        completed_steps=500,scheduler_horizon=4868,scheduler=checkpoint['scheduler'],sampler=checkpoint['sampler'],
        data_cursor=checkpoint['data_cursor'],optimizer_counters=[500],
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_count':len(g['params'])} for g in checkpoint['optimizer']['param_groups']],
        rng_per_rank=[dict(rank=r,keys=sorted(s),cpu_bytes=s['cpu'].numel(),cuda_devices=len(s['cuda']),
            loader_generator_sha256=hashlib.sha256(s['loader_generator'].numpy().tobytes()).hexdigest()) for r,s in enumerate(checkpoint['rng_per_rank'])],
        model_adapter_optimizer_present=True,native_source_sha256=checkpoint['config']['code_sha256'],
        evaluated_REPRODUCTION_PASS=True,evaluation_checkpoint_SHA_unchanged=True,
        user_authorization='Continue this exact step500 trajectory to4868; latest explicit user instruction',
        recorded_utc=now(),local_image_root=str(IMAGES),NFS_fallback=False)
    return proof


def worker():
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import local_image_dataset,reproduction_train_gate as gate,training_phase_timing as timing
    ref = json.loads((REVIEW/'resume-stream-reference.json').read_text())['records']
    cfg = json.loads(CONFIG.read_text())
    class HeartbeatIterator(timing.TimedIterator):
        def __init__(self,iterator,recorder):
            super().__init__(iterator,recorder)
            self.position=0
            self.epoch=iterator._dataset.epoch
        def __next__(self):
            self.recorder.begin()
            self.position+=1
            step=self.epoch*1217+self.position
            stamp=time.monotonic()
            path=PHASE/f'heartbeat-rank{self.recorder.rank}.json'
            event=dict(rank=self.recorder.rank,step=step,started_monotonic=stamp,utc=now(),replay_without_update=step<=500)
            dump(path,dict(event,state='DATA_WAIT'))
            try:
                value=next(self.iterator)
            except StopIteration:
                self.recorder.current=None
                dump(path,dict(event,state='EXHAUSTED'))
                raise
            except BaseException:
                dump(path,dict(event,state='DATA_WAIT_FAILED'))
                raise
            self.recorder.current['step']=step
            self.recorder.current['data_wait_s']=time.monotonic()-stamp
            if 501<=step<=505:
                expected=ref[f'{step}:{self.recorder.rank}']
                assert value['sample_id'].tolist()==expected['sample_ids'], 'Resume sample/cursor drift'
                assert stream(value)==expected['stream_sha256'], 'Resume F/S/D strings/token drift'
                dump(REVIEW/f'resume-batch-{step}-rank{self.recorder.rank}.json',dict(passed=True,**expected,epoch=self.epoch,batch_cursor=self.position-1))
            dump(path,dict(event,state='ACTIVE_STEP'))
            return value
    previous.PHASE=PHASE
    timing.TimedIterator=HeartbeatIterator
    local_image_dataset.LocalImageDataset=LocalDataset
    gate.RUN=RUN
    gate.EXP=REVIEW
    original_loader=trainer.DataLoader
    trainer.DataLoader=lambda *args,**kwargs: original_loader(*args,**kwargs,timeout=60)
    original_rates=trainer.optimizer_learning_rates
    def checked_rates(module,completed,horizon):
        values=original_rates(module,completed,horizon)
        assert horizon==4868
        if 500<=completed<505:
            assert list(values)==expected_lrs(completed,cfg), 'Resume scheduler LR drift'
            dump(REVIEW/f'resume-lr-{completed+1}-rank{os.environ["RANK"]}.json',dict(passed=True,step=completed+1,horizon=horizon,lrs=list(values)))
        return values
    trainer.optimizer_learning_rates=checked_rates
    gate.main()
    dump(PHASE/f'heartbeat-rank{os.environ["RANK"]}.json',dict(state='TRAINING_COMPLETE',utc=now()))


class Supervisor(previous.Supervisor):
    def __init__(self):
        super().__init__()
        self.train=FINAL
    def execute(self,name,command,training=False):
        from recovery.resource_stall_v2 import system_snapshot
        command=list(command)
        if name.startswith('verify'):
            command[command.index('--image-root')+1]=str(IMAGES)
            command[command.index('--index-dir')+1]=str(INDEX)
        log=RUN/(name+'.log')
        record=dict(name=name,command=command,started_utc=now(),raw_log=str(log))
        self.commands.append(record)
        dump(RUN/'full-commands.json',self.commands)
        env=dict(os.environ,SAID_FULL_SUPERVISOR_PID=str(os.getpid()),SAID_S02_STAGE='step4868',
            SAID_S02_PHASE_LOCAL=str(PHASE),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with log.open('w') as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            last_resource=0
            try:
                while process.poll() is None:
                    stamp=time.monotonic()
                    if stamp-last_resource>=5:
                        snapshot=dict(utc=now(),command=name,system=system_snapshot(),gpu=self.nvml.sample())
                        with (RUN/'full-resource-telemetry.jsonl').open('a') as output:
                            output.write(json.dumps(snapshot)+'\n')
                        last_resource=stamp
                        if snapshot['system']['memory_events'].get('oom_kill',0)>0:
                            raise RuntimeError('HARD_STOP: actual cgroup oom_kill')
                    if training:
                        for rank in range(4):
                            path=PHASE/f'heartbeat-rank{rank}.json'
                            if path.exists():
                                beat=json.loads(path.read_text())
                                if wait_expired(beat,stamp):
                                    raise RuntimeError(f'HARD_STOP: rank{rank} step{beat["step"]} {beat["state"]} >60s')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f'{name} failed returncode={process.returncode}; evidence={log}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                record.update(ended_utc=now(),returncode=process.returncode)
                dump(RUN/'full-commands.json',self.commands)
    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        assert not FINAL.exists() and not PHASE.exists(), 'Never overwrite/relaunch a trajectory'
        assert sha(PARENT)==PARENT_SHA
        ready=json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert not system_snapshot()['memory_events'].get('oom_kill',0)
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit() and int(entry.name)!=os.getpid():
                try:
                    args=(entry/'cmdline').read_bytes().split(b'\0')
                except OSError:
                    continue
                forbidden={'recovery.s02_full_stage','recovery.s02_local500','recovery.s02_nfs500','train.train_nested_semantic_mask'}
                assert not any(a.decode(errors='replace') in forbidden for a in args), 'Concurrent training/copy/audit'
        PHASE.mkdir()
        dump(RUN/'full-launch-provenance.json',dict(started_utc=self.started,supervisor_pid=os.getpid(),
            resume=str(PARENT),resume_sha256=PARENT_SHA,stop_updates=4868,horizon=4868,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            local_image_root=str(IMAGES),NFS_fallback=False,configuration_sha256=sha(CONFIG),
            launcher_sha256=sha(ROOT/'recovery/s02_local_full.py'),native_trainer_sha256=sha(ROOT/'train/train_nested_semantic_mask.py'),
            automatic_retry=False,replay='Native loader replays consumed500 batches without optimizer updates to retain loader worker sequencing'))
        command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
            '-m','recovery.s02_local_full','--worker','--config',str(CONFIG),'--init-state',str(STEP0),
            '--resume',str(PARENT),'--index-dir',str(INDEX),'--image-root',str(IMAGES),'--output-dir',str(FINAL),
            '--run-type','formal','--max-updates','4868']
        try:
            self.execute('train4868',command,training=True)
            self.acceptance=json.loads((FINAL/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates']==4868 and
                r['updates_this_run']==4368 and r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            import torch
            checkpoint=torch.load(FINAL/'step004868.pt',map_location='cpu',weights_only=False)
            assert checkpoint['completed_steps']==checkpoint['global_step']==4868
            assert checkpoint['scheduler_horizon']==4868 and checkpoint['data_cursor']==dict(next_epoch=4,next_batch=0)
            assert len(checkpoint['rng_per_rank'])==4 and {int(s['step']) for s in checkpoint['optimizer']['state'].values()}=={4868}
            self.final_checkpoint_proof=dict(passed=True,global_step=4868,data_cursor=checkpoint['data_cursor'],optimizer_steps=[4868],rng_ranks=4)
            del checkpoint
            pipeline.RUN=RUN
            pipeline.EXP=REVIEW
            self.result=pipeline.Supervisor.evaluate(self,4868)
            score=self.result['scores_percent']
            score.update(Score5=score['Score5_R1'],Short4=score['Short4_R1'])
            self.result['status']=classify(score['Score5'],score['J_long3'])
        except Exception as error:
            self.error=type(error).__name__+': '+str(error)
            print(self.error,flush=True)
        finally:
            dump(RUN/'full-supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
                final_checkpoint_proof=getattr(self,'final_checkpoint_proof',None),started_utc=self.started,ended_utc=now()))
        if self.error:
            raise RuntimeError(self.error)


def classify(score5,long3):
    if score5>72.768147:
        return 'FULL_POSITIVE' if long3>=76.870244 else 'SHORT_LONG_TRADEOFF'
    return 'EARLY_SIGNAL_DID_NOT_SCALE'


def prepare():
    assert not FINAL.exists(), 'Continuation already exists'
    REVIEW.mkdir(exist_ok=True)
    dump(OUT/'RESUME_PROVENANCE.json',identity())
    dump(REVIEW/'resume-stream-reference.json',reference())
    dump(REVIEW/'prelaunch-local-path-proof-5000.json',previous.path_proof())
    authorization=dict(status='REPRODUCTION_PASS',continuation_allowed=True,automatic_continuation=False,
        authorized_by='Explicit user instruction to resume exact evaluated checkpoint500 to4868',
        checkpoint=str(PARENT),sha256=PARENT_SHA,stop_updates=4868,authorized_utc=now())
    dump(REVIEW/'CONTINUATION_GATE.json',authorization)
    dump(OUT/'CONTINUATION_GATE.json',authorization)
    print(json.dumps(dict(event='CONTINUATION_PREPARED',parent_sha256=PARENT_SHA,stop=4868,local_only=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*remaining]
        worker()
    elif args.prepare:
        prepare()
    else:
        Supervisor().run()


if __name__=='__main__':
    main()
