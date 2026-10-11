"""Persist CPU checks before launching a single controlled Pure3 trajectory."""
import argparse
import os
import subprocess
from recovery.hns_v1_pure3 import ROOT, RUN, EXP, read, sha, dump, now

SUITES=['tests/test_hns_v1_pure3.py','tests/test_nested_d3_hns500.py','tests/test_hns_full.py',
        'tests/test_nested_detail_d3.py','tests/test_nested_resume.py','tests/test_retrieval_bounded.py',
        'tests/test_hns_pure3_full_identity.py']


def checks():
    command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*SUITES]
    result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=RUN/'CPU_TESTS.log'; log.write_text(result.stdout)
    dump(EXP/'CPU_TESTS.json',dict(passed=result.returncode==0,command=command,returncode=result.returncode,
        output=result.stdout,raw_log=str(log),log_sha256=sha(log),utc=now(),
        original_cpu4rank_DDP=read(EXP/'DDP_CORRECTNESS.json'),
        suite_sources={p:sha(ROOT/p) for p in SUITES}))
    assert result.returncode==0
    print(result.stdout,flush=True)


def launch():
    assert read(EXP/'STATE.json')['status']=='PREPARED'
    assert read(EXP/'CPU_TESTS.json')['passed'] and read(EXP/'DDP_CORRECTNESS.json')['passed']
    assert not (RUN/'launch.json').exists()
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle((0,1,2,3))
    command=[str(ROOT/'.venv/bin/python'),'-m','recovery.hns_v1_pure3','run']
    with (RUN/'controller.log').open('x') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'),start_new_session=True)
    receipt=dict(pid=p.pid,command=command,utc=now(),git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),one_launch_only=True)
    dump(RUN/'launch.json',receipt); print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=['checks','launch'])
    a=p.parse_args()
    if a.phase=='checks': checks()
    else: launch()
