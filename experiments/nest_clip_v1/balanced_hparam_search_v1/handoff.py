"""Finish the active child, then replace only the stopped old scheduler."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import EXP,REPO,RUN,PYTHON,TORCHRUN


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def proc(pid):
    # /proc fields after the closing comm parenthesis start at field3.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()


def main():
    state=json.loads((RUN/'state.json').read_text())
    parent,child=int(state['supervisor_pid']),int(state['stage_pid'])
    assert parent==1263400 and proc(parent)[0]=='T'
    assert int(proc(child)[1])==parent
    assert b'balanced_hparam_search_v1.search' in Path(f'/proc/{parent}/cmdline').read_bytes()
    key=state['stage']
    progress=RUN/'handoff-status.json'
    info=dict(status='waiting_for_active_child',old_supervisor=parent,active_child=child,
              stage=key,started_utc=now(),policy='global_top2_500_direct_to_3651')
    progress.write_text(json.dumps(info,indent=2)+'\n')
    while proc(child)[0]!='Z':
        time.sleep(1)
    wait_status=int(proc(child)[49])
    exit_code=os.waitstatus_to_exitcode(wait_status)
    record_path=RUN/'execution'/f'{key}.json'
    assert not record_path.exists()
    log=RUN/'execution'/f'{key}.console.txt'
    assert log.exists()
    assert key.endswith('-formal-500')
    prefix=key.split('-')[0]
    tid=next(t for t in state['trials'] if t.startswith(prefix))
    command=[TORCHRUN, '--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
             '-m','train.train_nested_semantic_mask','--config',str(RUN/'configs'/f'{tid}.json'),
             '--init-state','/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt',
             '--index-dir','/root/lk_projects/SAID-nest-clip-v1/data_index',
             '--image-root','/root/lk_projects/SAID-assets/training/ShareGPT4V',
             '--output-dir',str(RUN/'trials'/tid/'step500'),'--run-type','formal','--max-updates','500']
    accepted=RUN/'trials'/tid/'step500/acceptance.json'
    if exit_code==0:
        assert json.loads(accepted.read_text())['passed']
    record=dict(command=command,cwd=str(REPO),stage=key,code_commit=state['code_commit'],
                started_utc=state['stage_started_utc'],finished_utc=now(),exit_code=exit_code,
                elapsed_seconds=dt.datetime.now(dt.timezone.utc).timestamp()-dt.datetime.fromisoformat(state['stage_started_utc']).timestamp(),
                timing_source='UTC stage start to observed child exit, <=1s observation interval',
                exit_status_source='actual Linux child wait status from /proc while parent stopped')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    evidence=EXP/'evidence/execution'
    shutil.copy2(record_path,evidence/record_path.name);shutil.copy2(log,evidence/log.name)
    state['stages'].append(record)
    state['promotion_policy_revision']=dict(policy=info['policy'],requested_utc=info['started_utc'],
                                             handoff_utc=now(),active_child_exit_code=exit_code)
    state['status']='ready' if exit_code==0 else 'failed'
    state['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    temporary=RUN/'state.handoff.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(RUN/'state.json')
    # Signal only the verified old supervisor. Its child already finished normally.
    os.kill(parent,signal.SIGTERM)
    os.kill(parent,signal.SIGCONT)
    for _ in range(50):
        if not Path(f'/proc/{parent}').exists() or proc(parent)[0]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Old supervisor did not terminate; new scheduler not launched')
    info.update(active_child_exit_code=exit_code,child_finished_utc=record['finished_utc'])
    if exit_code==0:
        with (RUN/'search-top2-direct.console.txt').open('x') as output:
            process=subprocess.Popen([PYTHON,'-m','experiments.nest_clip_v1.balanced_hparam_search_v1.search'],
                                     cwd=REPO,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        info.update(status='completed',new_supervisor=process.pid)
    else:
        info.update(status='failed',error='Active child failed; preserved artifacts and did not restart')
    info['finished_utc']=now();progress.write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':main()
