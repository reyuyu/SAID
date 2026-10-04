"""Sequential audit stages and precision controls; never launches a trainer."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,ROOT,RUN,stages,dump
from train.nested_semantic_data import file_sha
PYTHON='/root/miniconda3/envs/said-repro/bin/python'
TORCHRUN='/root/miniconda3/envs/said-repro/bin/torchrun'


def command(name,args):
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not occupied,occupied
    path=RUN/'execution'/(name+'.console.txt');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    argv=[TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0','-m','experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit',*args]
    record={'argv':argv,'cwd':str(ROOT),'env':{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(ROOT)},'started_unix':time.time()}
    dump(EXP/'commands'/(name+'.json'),record)
    with path.open('w') as handle:
        process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,**record['env']),stdout=handle,stderr=subprocess.STDOUT)
        code=process.wait()
    record.update(exit_code=code,elapsed_seconds=time.time()-record['started_unix']);dump(EXP/'commands'/(name+'.json'),record)
    assert code==0,f'{name} failed; preserve evidence at {path}'
    shutil.copy2(path,EXP/'evidence'/(name+'.console.txt'))
    print(json.dumps({'stage':name,'exit_code':code,'seconds':record['elapsed_seconds']}),flush=True)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    provenance={}
    for key,stage in stages().items():
        actual=file_sha(stage['checkpoint']);assert actual==stage['sha256']
        provenance[key]={**stage,'sha256_before':actual,'identity_source':'common-step0 initializer' if key=='0' else 'b85f73a committed four-arm evidence','checkpoint_read_only':True}
    dump(EXP/'CHECKPOINT_PROVENANCE.json',provenance)
    for key in ['0','R','B']:command('audit-stage-'+key,['--stage',key,'--batches','32'])
    for key in ['0','R','B']:command('fp32-control-'+key,['--stage',key,'--batches','1','--pilot','--label','fp32','--fp32-control'])
    for key,stage in stages().items():
        after=file_sha(stage['checkpoint']);assert after==provenance[key]['sha256_before']
        provenance[key]['sha256_after']=after;provenance[key]['unchanged']=True
    dump(EXP/'CHECKPOINT_PROVENANCE.json',provenance)
    from experiments.nest_clip_v1.gradient_composition_audit_v1.report import generate
    generate()

if __name__=='__main__':main()
