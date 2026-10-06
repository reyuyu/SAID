"""Fresh local-only step0->500; strict native retrieval; never continue501."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

from recovery.s02_nfs500 import ROOT, OUT, STEP0, STEP0_SHA, PYTHON, now, dump, rows, sha, distribution
from recovery.s02_full_stage import LOCAL, IMAGES, MANIFEST_SHA, ensure_local_index, local_path
from recovery.s02_full_local_data import FullLocalDataset
from recovery.local500_policy import POLICY, wait_expired

RUN = ROOT / 'runtime/SAID-nest-clip-v1/s02-local500-20261006'
PHASE = LOCAL / 'formal-local500-phase-20261006'
INDEX = LOCAL / 'data_index'
CONFIG = OUT / 'configs/summary02_local500.json'


def path_proof(count=5000):
    import torch
    from torch.utils.data import DistributedSampler
    dataset=FullLocalDataset(INDEX,IMAGES,'summary_random_detail',0)
    before=torch.get_rng_state().clone()
    rng=random.Random(0)
    proof=[]
    for rank in range(4):
        positions=set(rng.sample(range(500*256),count//4))
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        for position,index in enumerate(itertools.islice(iter(sampler),max(positions)+1)):
            if position in positions:
                proof.append(dict(rank=rank,step=position//256+1,batch_position=position%256,
                    sample_id=index+1000,actual_path=str(dataset.resolved_path(index))))
    if len(proof)!=count or not torch.equal(before,torch.get_rng_state()):
        raise RuntimeError('Local frozen path proof count/RNG drift')
    return dict(passed=True,count=count,seed=0,epoch=0,all_local=True,NFS_fallback=False,
        selection='Random positions in frozen steps1..500,1250 per rank',training_RNG_untouched=True,rows=proof)


class LoggedLocalDataset(FullLocalDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.index_dir.resolve()!=INDEX.resolve():
            raise RuntimeError('Formal index must also be local')
        self._proofs=0

    def __getitem__(self,index):
        path=self.resolved_path(index)
        value=super().__getitem__(index)
        if self._proofs<4:
            record=dict(event='LOCAL_ONLY_IMAGE_READ',rank=int(os.environ['RANK']),worker_pid=os.getpid(),
                sample_id=value['sample_id'],actual_path=str(path),image_root=str(IMAGES),NFS_fallback=False)
            worker_phase = Path(os.environ['SAID_S02_PHASE_LOCAL'])
            with (worker_phase/f'image-paths-rank{os.environ["RANK"]}-pid{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            self._proofs+=1
        return value


def worker():
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import (
        local_image_dataset,reproduction_train_gate as gate,training_phase_timing as timing)
    class HeartbeatIterator(timing.TimedIterator):
        def __next__(self):
            self.recorder.begin()
            stamp=time.monotonic()
            path=PHASE/f'heartbeat-rank{self.recorder.rank}.json'
            event=dict(rank=self.recorder.rank,step=self.recorder.counter,started_monotonic=stamp,utc=now())
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
            self.recorder.current['data_wait_s']=time.monotonic()-stamp
            dump(path,dict(event,state='ACTIVE_STEP'))
            return value
    timing.TimedIterator=HeartbeatIterator
    local_image_dataset.LocalImageDataset=LoggedLocalDataset
    gate.RUN=RUN
    gate.EXP=RUN/'reviewed'
    provenance=json.loads((RUN/'launch-provenance.json').read_text())
    gate.AUTHORIZED_SOURCE_CHANGES={'train/train_nested_semantic_mask.py':(
        '56eaa83f7ee3ea9fa97a754a52288fc61adb28dd41129015ec39b129b82c57e3',
        provenance['source_sha256']['train/train_nested_semantic_mask.py'])}
    # Loader workers/prefetch/batch/sampler stay frozen. Only its failure timeout changes.
    original_loader=trainer.DataLoader
    def loader(*args,**kwargs):
        return original_loader(*args,**kwargs,timeout=60)
    trainer.DataLoader=loader
    gate.main()
    dump(PHASE/f'heartbeat-rank{os.environ["RANK"]}.json',dict(state='TRAINING_COMPLETE',utc=now()))


class Supervisor:
    def __init__(self):
        from recovery.resource_stall_v2 import Nvml
        self.train=RUN/'step500'
        self.nvml=Nvml()
        self.started=now()
        self.started_monotonic=time.monotonic()
        self.error=None
        self.result=None
        self.commands=[]
        self.acceptance=None

    def execute(self,name,command,training=False):
        from recovery.resource_stall_v2 import system_snapshot
        command=list(command)
        if name.startswith('verify'):
            command[command.index('--image-root')+1]=str(IMAGES)
            command[command.index('--index-dir')+1]=str(INDEX)
        log=RUN/(name+'.log')
        record=dict(name=name,command=command,started_utc=now(),raw_log=str(log))
        self.commands.append(record)
        dump(RUN/'commands.json',self.commands)
        env=dict(os.environ,SAID_FULL_SUPERVISOR_PID=str(os.getpid()),SAID_S02_STAGE='step500',
            SAID_S02_PHASE_LOCAL=str(PHASE),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with log.open('w') as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            last_resource=0
            try:
                while process.poll() is None:
                    stamp=time.monotonic()
                    if stamp-last_resource>=5:
                        snapshot=dict(utc=now(),command=name,system=system_snapshot(),gpu=self.nvml.sample())
                        with (RUN/'resource-telemetry.jsonl').open('a') as stream:
                            stream.write(json.dumps(snapshot)+'\n')
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
                dump(RUN/'commands.json',self.commands)

    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        ready=json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert ready['manifest']['sha256']==MANIFEST_SHA
        frozen=json.loads((OUT/'configs/summary02.json').read_text())
        cfg=json.loads(CONFIG.read_text())
        assert cfg.pop('resource_policy')==POLICY and cfg==frozen
        assert sha(STEP0)==STEP0_SHA
        hashes=ensure_local_index()
        assert not system_snapshot()['memory_events'].get('oom_kill',0)
        forbidden={'recovery.s02_full_stage','recovery.s02_nfs500','recovery.s02_stage500',
            'recovery.local_ssd_stage','train.train_nested_semantic_mask'}
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit() and int(entry.name)!=os.getpid():
                try:
                    arguments=(entry/'cmdline').read_bytes().split(b'\0')
                except OSError:
                    continue
                if any(a.decode(errors='replace') in forbidden for a in arguments):
                    raise RuntimeError('Concurrent SAID training/copy/audit; no launch')
        assert not RUN.exists(), 'Never overwrite or resume an earlier trajectory'
        RUN.mkdir(parents=True)
        (RUN/'reviewed').mkdir()
        PHASE.mkdir(exist_ok=False)
        dump(RUN/'prelaunch-local-path-proof-5000.json',path_proof())
        sources=['recovery/s02_local500.py','recovery/local500_policy.py','recovery/s02_full_local_data.py',
            'recovery/configs/summary02_local500.json','train/train_nested_semantic_mask.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py']
        dump(RUN/'launch-provenance.json',dict(started_utc=self.started,started_monotonic=self.started_monotonic,
            supervisor_pid=os.getpid(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha(ROOT/p) for p in sources},step0_sha256=STEP0_SHA,
            local_image_root=str(IMAGES),local_index=str(INDEX),local_index_sha256=hashes,
            NFS_fallback=False,stop_updates=500,horizon=4868,resume=None,initial_resources=system_snapshot(),
            disposable_cache='Ephemeral Docker overlay; NFS originals retained',automatic_retry=False))
        print(json.dumps(dict(event='LOCAL_ONLY_FORMAL_LAUNCH',image_root=str(IMAGES),stop=500,horizon=4868)),flush=True)
        command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
            '-m','recovery.s02_local500','--worker','--config',str(CONFIG),'--init-state',str(STEP0),
            '--index-dir',str(INDEX),'--image-root',str(IMAGES),'--output-dir',str(self.train),
            '--run-type','formal','--max-updates','500']
        try:
            self.execute('train500',command,training=True)
            self.acceptance=json.loads((self.train/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates']==500 and
                r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            assert json.loads((RUN/'first-five-gate.json').read_text())['passed']
            import torch
            checkpoint=torch.load(self.train/'step000500.pt',map_location='cpu',weights_only=False)
            assert checkpoint['global_step']==checkpoint['completed_steps']==500
            assert all(k in checkpoint for k in ('model','optimizer','scheduler','rng_per_rank','sampler','data_cursor'))
            assert checkpoint['scheduler']['horizon']==4868 and len(checkpoint['rng_per_rank'])==4
            assert {int(s['step']) for s in checkpoint['optimizer']['state'].values()}=={500}
            self.checkpoint_proof=dict(passed=True,global_step=500,horizon=4868,rng_ranks=4,
                data_cursor=checkpoint['data_cursor'],sampler=checkpoint['sampler'],optimizer_steps=[500],uploaded=False)
            del checkpoint
            pipeline.EXP=RUN/'reviewed'
            pipeline.RUN=RUN
            self.result=pipeline.Supervisor.evaluate(self,500)
            historical=json.loads((ROOT/'experiments/nest_clip_v1/armb_summary02_4epoch_v1/PARENT_500.json').read_text())
            gate=pipeline.reproduction_gate(self.result['scores_percent'],self.result['metrics'],historical['metrics'])
            self.result.update(status=gate['status'],reproduction_gate=gate)
            self.result['scores_percent'].update(Score5=self.result['scores_percent']['Score5_R1'],
                Short4=self.result['scores_percent']['Short4_R1'])
        except Exception as error:
            self.error=type(error).__name__+': '+str(error)
            print(self.error,flush=True)
        finally:
            self.report()
        if self.error:
            raise RuntimeError(self.error)

    def report(self):
        cycles=rows(self.train/'cycle_timing.jsonl')
        phases={r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
        lookup={r:{p['step']:p for p in records} for r,records in phases.items()}
        resources=rows(RUN/'resource-telemetry.jsonl')
        train_resources=[r for r in resources if r['command']=='train500']
        systems=[r['system'] for r in train_resources]+[p['system_after'] for records in phases.values() for p in records]
        waits=[max(lookup[r].get(c['step'],{}).get('data_wait_s',0) for r in range(4)) for c in cycles]
        compact=[dict(step=c['step'],full_cycle_s=c['four_rank_max_seconds'],ranks=[dict(rank=r,
            **{k:v for k,v in lookup[r].get(c['step'],{}).items() if k in ('data_wait_s','forward_s','backward_s','optimizer_s','wall_cycle_before_checkpoint_s')},
            system=lookup[r].get(c['step'],{}).get('system_after',{})) for r in range(4)]) for c in cycles]
        # Detailed per-step resources remain private; publish small scalars only.
        for c in compact:
            for p in c['ranks']:
                s=p.pop('system')
                p.update(memory_current=s.get('memory_current'),file_cache=s.get('file'),
                    IO_PSI_full_avg10=s.get('io_PSI',{}).get('full',{}).get('avg10'),
                    memory_PSI_full_avg10=s.get('memory_PSI',{}).get('full',{}).get('avg10'))
        proofs=[p for file in PHASE.glob('image-paths-*.jsonl') for p in rows(file)]
        valid=bool(proofs) and all(Path(p['actual_path']).is_relative_to(IMAGES) and not p['NFS_fallback'] for p in proofs)
        text=(RUN/'train500.log').read_text(errors='replace') if (RUN/'train500.log').exists() else ''
        errors=sorted(set(re.findall(r'Image failure sample=\d+[^\n]*|Input/output error[^\n]*|Missing local sample=\d+[^\n]*',text)))
        gpu_peaks={str(r):dict(allocated_GiB=max((p['peak_allocated_gib'] for c in rows(self.train/'steps.jsonl') for p in c['rank_health'] if p['rank']==r),default=None),
            reserved_GiB=max((p['peak_reserved_gib'] for c in rows(self.train/'steps.jsonl') for p in c['rank_health'] if p['rank']==r),default=None)) for r in range(4)}
        raw=list(RUN.glob('*.log'))+list(RUN.glob('*.jsonl'))+list(self.train.glob('*.jsonl'))+list(PHASE.glob('*.jsonl'))+[RUN/'prelaunch-local-path-proof-5000.json']
        inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),time_range_utc=[self.started,now()],uploaded=False) for p in sorted(raw)]
        stats=dict(started_utc=self.started,finished_utc=now(),completed_steps=len(cycles),
            full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
            data_wait_seconds_slowest_rank=distribution(waits),
            data_wait_seconds_all_rank_batches=distribution([p['data_wait_s'] for records in phases.values() for p in records]),
            rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in records]) for r,records in phases.items()},
            steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
            GPU_peak_memory=gpu_peaks,peak_cgroup_memory_bytes=max((s['memory_current'] for s in systems),default=0),
            peak_file_cache_bytes=max((s['file'] for s in systems),default=0),
            oom_kill=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0),
            PSI_scope='Host /proc/pressure on cgroup v1; not per-cgroup PSI',
            io_errors=errors,io_error_count=len(errors),io_error_scope='Actual native local image reads and training process log',
            stop_reason=self.error,slow_step_policy='>3s warning; single active step/unreturned batch >60s hard stop',
            actual_local_path_proof=dict(count=len(proofs),ranks=sorted({p['rank'] for p in proofs}),passed=valid,NFS_fallback=False),
            raw_local_artifacts=inventory)
        for kind in ('io_PSI','memory_PSI'):
            for pressure in ('some','full'):
                stats[f'{kind}_{pressure}_avg10']=distribution([s[kind][pressure]['avg10'] for s in systems if pressure in s[kind]])
        stats['GPU_utilization_percent']={str(r):distribution([g['gpu_percent'] for s in train_resources for g in s['gpu']
            if isinstance(g,dict) and g.get('index')==r and g.get('gpu_percent') is not None]) for r in range(4)}
        result=self.result or dict(status='INCOMPLETE_HARD_STOP',step=len(cycles),error=self.error,evaluated=False)
        result.update(local_image_root=str(IMAGES),runtime=str(RUN),step0_sha256=STEP0_SHA,
            first_five_gate=json.loads((RUN/'first-five-gate.json').read_text()) if (RUN/'first-five-gate.json').exists() else None,
            launch_provenance=json.loads((RUN/'launch-provenance.json').read_text()),
            stopped_at_500=len(cycles)==500,checkpoint_proof=getattr(self,'checkpoint_proof',None),
            acceptance=self.acceptance,checkpoint_uploaded=False,NFS_fallback=False,training_workers_exited=True)
        gate=dict(result.get('reproduction_gate',{}),status=result['status'],stop_updates=500,continuation_allowed=False,
            automatic_continuation=False,wait_for_next_user_instruction=True,github_sync='PENDING')
        dump(OUT/'STEP500_RESULTS.json',result)
        dump(OUT/'CONTINUATION_GATE.json',gate)
        dump(OUT/'LOCAL_RUNTIME_STATS.json',stats)
        dump(RUN/'local-per-step-summary.json',compact)
        # Only slow-step phase rows are published; aggregate covers all500.
        dump(OUT/'LOCAL_PER_STEP_SUMMARY.json',[c for c in compact if c['full_cycle_s']>3])
        lines=['# S=0.2 local-only step500 reproduction','',f'Status: `{result["status"]}`; completed {len(cycles)}/500, horizon4868.',
            '',f'Fresh common step0 SHA256: `{STEP0_SHA}`. No resume.',
            'Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch;4 A10080GB,256/rank,global1024,accum1,seed0; optimizer/LR/sparsity/inclusion/preprocessing unchanged.',
            f'Local-only image root: `{IMAGES}`. Missing/escaped/symlink image fails before native decode, no NFS fallback.',
            'Prelaunch random5000 frozen sampler positions all local; runtime path proofs: `'+json.dumps(stats['actual_local_path_proof'])+'`.',
            'Cache is ephemeral Docker overlay; NFS originals remain the persistent source of truth.',
            '', 'Full-cycle seconds: `'+json.dumps(stats['full_cycle_seconds'])+'`.',
            'Slowest-rank data_wait seconds: `'+json.dumps(stats['data_wait_seconds_slowest_rank'])+'`.',
            f'>3s:{stats["steps_gt3s"]}; >10s:{stats["steps_gt10s"]}. I/O errors:{len(errors)}; oom_kill:{stats["oom_kill"]}.',
            'Resources and all-rank distributions: LOCAL_RUNTIME_STATS.json. Slow-step phases: LOCAL_PER_STEP_SUMMARY.json.',
            'No speed-driven changes to batch/workers/mathematics; no sample skipping/substitution.',
            '', '| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | R1 delta I2T/T2I pp |', '|---|---|---|---|']
        for name,m in result.get('metrics',{}).items():
            values=[' / '.join(f'{m[d][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
            delta=gate['dataset_R1_deltas_pp'][name]
            lines.append(f'|{name}|{values[0]}|{values[1]}|{delta["I2T"]:+.6f} / {delta["T2I"]:+.6f}|')
        lines+=['','Scores percent: `'+json.dumps(result.get('scores_percent'))+'`.',
            'Historical score deltas pp: `'+json.dumps(gate.get('delta_vs_historical_S02_pp'))+'`.',
            'Gate checks: `'+json.dumps(gate.get('checks'))+'`.',
            'Collapse screen fixed before evaluation: invalid/nonfinite R1 or any direction below50% of historical R1.',
            f'Checkpoint stays local: `{self.train/"step000500.pt"}`. Strict export/evaluation must leave it immutable.',
            'PASS/FAIL both stop500; no automatic continuation. CPU/unit checks and GitHub receipt follow.',
            'Raw logs and checkpoints remain local; path/size/SHA/UTC inventory in LOCAL_RUNTIME_STATS.json.',
            'GitHub synchronization: PENDING.','Error: '+str(self.error)]
        (OUT/'STEP500_LOCAL_REPRODUCTION.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(event='FINAL',status=result['status'],completed_steps=len(cycles),error=self.error)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*remaining]
        worker()
    else:
        if remaining:
            parser.error('No arbitrary overrides allowed')
        def interrupted(sig,frame):
            raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        Supervisor().run()


if __name__=='__main__':
    main()
