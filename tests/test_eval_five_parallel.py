"""CPU-only scheduling, untouched historical arguments and failure receipts."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from tools import eval_five_parallel as ev


def test_mapping_and_frozen_original_commands(tmp_path):
    assert ev.gpu_mapping()==dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3)
    assert ev.gpu_mapping((3,2,1,0))['urban']==0
    serial=ev.build_jobs('/tmp/student.pt',tmp_path,serial=True)
    parallel=ev.build_jobs('/tmp/student.pt',tmp_path)
    for n in ev.ORDER:
        a=list(serial[n].command);b=list(parallel[n].command)
        assert a[a.index('--batch-size')+1]=='64'
        assert a[a.index('--device')+1]=='cuda:0'
        b[b.index('--device')+1]='cuda:0'
        assert a==b
        assert '-m' in a and 'torchrun' not in a and '--nproc-per-node' not in a
    old=json.loads((ev.ROOT/'experiments/nest_clip_v1/nested_d3_hns_sg500_v1/COMMANDS.json').read_text())
    for job in serial.values():
        module=job.command[2];protocol='flickr_test1k' if job.name=='flickr' else job.name
        candidates=[r['command'] for r in old if len(r['command'])>2 and r['command'][2]==module and
            (('--dataset' in r['command'] and r['command'][r['command'].index('--dataset')+1]==protocol) or
             any(a.startswith(protocol+':') for a in r['command']))]
        assert len(candidates)==1
        a=list(job.command);b=list(candidates[0])
        for flag in ('--checkpoint','--output','--output-dir'):
            if flag in a:b[b.index(flag)+1]=a[a.index(flag)+1]
        assert a==b


@pytest.mark.parametrize('gpus',[(0,1,2),(0,1,2,2),(-1,0,1,2)])
def test_invalid_gpu_mapping(gpus):
    with pytest.raises(ValueError):ev.gpu_mapping(gpus)


def test_duplicate_output_or_log_rejected(tmp_path):
    jobs=ev.build_jobs('/tmp/student.pt',tmp_path)
    jobs['urban']=replace(jobs['urban'],output=jobs['coco'].output)
    with pytest.raises(ValueError,match='Duplicate'):ev.validate_jobs(jobs)
    jobs=ev.build_jobs('/tmp/student.pt',tmp_path)
    jobs['urban']=replace(jobs['urban'],log=jobs['coco'].log)
    with pytest.raises(ValueError,match='Duplicate'):ev.validate_jobs(jobs)


class FakeProcesses:
    def __init__(self,jobs,fail=None,missing=None):
        self.jobs=jobs;self.fail=fail;self.missing=missing
        self.active=set();self.completed=[];self.lock=threading.Lock();self.spawned=[]

    def __call__(self,command,**kwargs):
        assert kwargs['start_new_session'] and kwargs['stderr']==ev.subprocess.STDOUT
        assert 'env' not in kwargs
        job=next(j for j in self.jobs.values() if list(j.command)==command)
        with self.lock:
            assert job.gpu not in self.active, 'Two simultaneous jobs on one GPU'
            self.active.add(job.gpu);self.spawned.append(job.name)
            if job.name=='urban':assert 'flickr' in self.completed
        owner=self
        class Process:
            pid=9999999;returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout=None):
                time.sleep(.02)
                if job.name!=owner.missing and job.name!=owner.fail:
                    job.output.parent.mkdir(parents=True,exist_ok=True)
                    job.output.write_text('{"mock_evaluator": true}')
                    if '--output-dir' in job.command:(job.output.parent/'extended_summary.json').write_text('{}')
                self.returncode=7 if job.name==owner.fail else 0
                with owner.lock:owner.active.remove(job.gpu);owner.completed.append(job.name)
                return self.returncode
        return Process()


@pytest.mark.parametrize('failure,missing',[(None,None),('docci',None),('flickr',None),(None,'long_dci')])
def test_subprocess_returncodes_missing_json_and_failure_propagation(tmp_path,failure,missing):
    out=tmp_path/'out';jobs=ev.build_jobs('/tmp/student.pt',out)
    fake=FakeProcesses(jobs,failure,missing)
    scheduler=ev.Scheduler(jobs,out,popen=fake)
    rc=scheduler.run({'mock':True},preflight=lambda g:None)
    receipt=json.loads((out/'EVAL_PARALLEL_RUN.json').read_text())
    assert rc==(1 if failure or missing else 0)
    assert receipt['status']==('FAILED' if rc else 'COMPLETED')
    assert receipt['jobs']['coco']['returncode']==0 and jobs['coco'].output.exists()
    assert 'duration_seconds' in receipt['jobs']['coco']
    if failure:assert receipt['jobs'][failure]['returncode']==7
    if missing:
        assert receipt['jobs'][missing]['returncode']==0
        assert 'Missing evaluator JSON' in receipt['jobs'][missing]['error']
    if failure=='flickr':
        assert 'urban' not in fake.spawned
        assert receipt['jobs']['urban']['status']=='BLOCKED_BY_PREDECESSOR'
    assert all(Path(r['log']).parent==out/'logs' for r in receipt['jobs'].values())


def test_serial_historical_order_and_stale_output_rejection(tmp_path):
    out=tmp_path/'out';jobs=ev.build_jobs('/tmp/student.pt',out,serial=True)
    # Use fake evaluator with a nonempty JSON and avoid invoking real models.
    calls=[]
    def serial_popen(command,**kwargs):
        job=next(j for j in jobs.values() if list(j.command)==command);calls.append(job.name)
        class P:
            pid=9999999;returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout=None):
                job.output.parent.mkdir(parents=True,exist_ok=True);job.output.write_text('{"mock":1}')
                if '--output-dir' in command:(job.output.parent/'extended_summary.json').write_text('{}')
                self.returncode=0;return 0
        return P()
    scheduler=ev.Scheduler(jobs,out,serial=True,popen=serial_popen)
    assert scheduler.run({},preflight=lambda g:None)==0 and calls==list(ev.ORDER)
    with pytest.raises(FileExistsError):ev.Scheduler(jobs,out).run({},preflight=lambda g:None)


def test_visibility_ambiguity_and_training_gpu_busy_rejected(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,0,1,3')
    with pytest.raises(RuntimeError,match='Ambiguous'):ev.require_gpu_idle({0,1,2,3})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    def output(command,**kwargs):
        return '0, GPU-zero\n1, GPU-one\n2, GPU-two\n3, GPU-three\n' if '--query-gpu=index,uuid' in command else 'GPU-one, 12345\n'
    monkeypatch.setattr(ev.subprocess,'check_output',output)
    with pytest.raises(RuntimeError,match='occupied'):ev.require_gpu_idle({0,1,2,3})
