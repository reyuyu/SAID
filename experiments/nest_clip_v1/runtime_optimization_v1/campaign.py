"""Serial real four-GPU35-update sweeps; atomic progress and no retrieval training."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from experiments.nest_clip_v1.runtime_optimization_v1.probe import RUN

REPO=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parent
PYTHON='/root/miniconda3/envs/said-repro/bin/python'
TORCHRUN='/root/miniconda3/envs/said-repro/bin/torchrun'


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run_probe(name,config,state):
    destination=RUN/name
    if (destination/'result.json').exists():
        return json.loads((destination/'result.json').read_text())
    assert not destination.exists(),f'Interrupted result retained; no implicit overwrite: {name}'
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not occupied,f'GPU occupied before{name}: {occupied}'
    path=RUN/'configs'/f'{name}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(config,indent=2)+'\n')
    state.update(status='running',stage=name,stage_started_utc=now());save(RUN/'campaign_state.json',state)
    samples=[];done=threading.Event()
    def monitor():
        while not done.is_set():
            started=time.time()
            try:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                readings=[]
                for line in output.splitlines():
                    index,util,memory=map(int,line.split(','));readings.append(dict(index=index,utilization_percent=util,memory_mib=memory))
                samples.append(dict(unix_time=started,gpus=readings))
            except (subprocess.SubprocessError,ValueError):pass
            done.wait(1)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    command=[TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
             '-m','experiments.nest_clip_v1.runtime_optimization_v1.probe','--config',str(path),'--name',name]
    started=time.monotonic()
    with (RUN/(name+'.console.txt')).open('x') as output:
        child=subprocess.Popen(command,cwd=REPO,stdout=output,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        state['stage_pid']=child.pid;save(RUN/'campaign_state.json',state)
        code=child.wait()
    done.set();watcher.join(timeout=6)
    execution=dict(name=name,command=command,exit_code=code,elapsed_seconds=time.monotonic()-started,
                   finished_utc=now(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    if not code:
        result=json.loads((destination/'result.json').read_text())
        lo=max(r['measured_start_unix'] for r in result['ranks']);hi=min(r['measured_end_unix'] for r in result['ranks'])
        measured=[sample for sample in samples if lo<=sample['unix_time']<=hi]
        result['gpu_utilization']={str(gpu):sum(s['gpus'][gpu]['utilization_percent'] for s in measured)/len(measured)
                                  for gpu in range(4)} if measured else {}
        result['execution']=execution
        save(destination/'result.json',result)
    else:
        result=dict(name=name,failed=True,config=config,execution=execution,
                    failure_evidence=[json.loads(p.read_text()) for p in destination.glob('failure-rank*.json')])
    state.setdefault('results',{})[name]=result;save(RUN/'campaign_state.json',state)
    save(RUN/(name+'.execution.json'),execution)
    (RUN/(name+'.gpu-utilization.json')).write_text(json.dumps(samples)+'\n')
    compact=EXP/'evidence/benchmarks';compact.mkdir(parents=True,exist_ok=True)
    save(compact/(name+'.json'),result)
    print(json.dumps({k:result.get(k) for k in ('name','mean_seconds','max_seconds','peak_allocated_gib','resource_feasible','failed')}),flush=True)
    return result


def b16(state):
    baseline=json.loads((RUN/'configs/benchmark-reference.json').read_text())
    baseline.update(audit_level='benchmark',skip_observational_model_logs=True)
    variants=[('b16-audit-sparse',dict(baseline,audit_level='sparse')),
              ('b16-audit-benchmark',baseline),
              ('b16-fused-batched',dict(baseline,fused_text_views=True)),
              ('b16-fused-strict',dict(baseline,fused_text_views=True,fused_text_backward='per_view_recompute')),
              ('b16-gate-cached',dict(baseline,cache_gate_projection=True)),
              ('b16-gate-strict',dict(baseline,cache_gate_projection=True,gate_backward='per_use_recompute')),
              ('b16-text-normalized',dict(baseline,cache_text_normalization=True)),
              ('b16-text-normalized-strict',dict(baseline,cache_text_normalization=True,normalization_backward='per_use_recompute')),
              ('b16-score-reduced',dict(baseline,reduced_pair_score=True))]
    for name,config in variants:run_probe(name,config,state)
    combined=dict(baseline,fused_text_views=True,fused_text_backward='per_view_recompute',
                  cache_gate_projection=True,gate_backward='per_use_recompute',cache_text_normalization=True,
                  normalization_backward='per_use_recompute')
    for image,text in ((128,128),(256,128),(128,256),(256,256),(512,256)):
        run_probe(f'b16-combined-{image}x{text}',dict(combined,image_chunk=image,text_chunk=text),state)
    # Checkpoint sweep excludes custom text rematerialization to genuinely test
    # full/partial/no encoder checkpointing rather than hiding recomputation.
    fastest=min((state['results'][f'b16-combined-{i}x{t}'] for i,t in ((128,128),(256,128),(128,256),(256,256),(512,256))
                 if not state['results'][f'b16-combined-{i}x{t}'].get('failed')),key=lambda r:r['mean_seconds'])
    shape={k:fastest['config'][k] for k in ('image_chunk','text_chunk')}
    for strategy in ('full','none','partial2','partial3'):
        config=dict(combined,**shape,fused_text_views=False,encoder_checkpoint_strategy=strategy,
                    checkpoint_encoders=strategy!='none')
        run_probe('b16-checkpoint-'+strategy,config,state)
    feasible=[r for name,r in state['results'].items() if name.startswith('b16-checkpoint-') and r.get('resource_feasible')]
    base=min(feasible,key=lambda r:r['mean_seconds'])['config'] if feasible else baseline
    for bucket in (25,50,100):
        run_probe(f'b16-ddp-{bucket}',dict(base,find_unused_parameters=False,static_graph=True,
                  gradient_as_bucket_view=True,bucket_cap_mb=bucket),state)
    ddps=[r for name,r in state['results'].items() if name.startswith('b16-ddp-') and r.get('resource_feasible')]
    fastest_config=min(ddps,key=lambda r:r['mean_seconds'])['config'] if ddps else base
    run_probe('b16-fused-adamw',dict(fastest_config,fused_adamw=True),state)
    run_probe('b16-tf32',dict(fastest_config,fused_adamw=True,tf32=True),state)
    state.update(status='b16_speed_candidates_complete',stage='awaiting_named_regression_and_l14_sweep',finished_b16_utc=now())
    save(RUN/'campaign_state.json',state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.launch:
        with (RUN/'campaign.console.txt').open('ab') as log:
            child=subprocess.Popen([PYTHON,'-u','-m','experiments.nest_clip_v1.runtime_optimization_v1.campaign'],cwd=REPO,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,runtime=str(RUN))));return
    path=RUN/'campaign_state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(status='initializing',results={},started_utc=now())
    try:b16(state)
    except Exception as exc:
        state.update(status='failed',error=str(exc),failed_utc=now());save(path,state);raise


if __name__=='__main__':main()
