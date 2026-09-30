"""Isolated, sequential resource gates and authorized 500-update candidate runs."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/stack_crossscore_v1')
EVIDENCE = EXP / 'evidence'
GROUPS = ('S-CLS', 'C-CLS', 'S-PATCH', 'C-PATCH')
PYTHON = '/root/miniconda3/envs/said-repro/bin/python'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def status(state):
    temp = RUN / 'status.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(RUN / 'status.json')


def assert_gpus_free():
    occupied = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not occupied, f'GPU occupation detected; will not contend: {occupied}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['gates', 'formal'], required=True)
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    state = dict(status='running', phase=args.phase, started_utc=now(),
                 supervisor_pid=os.getpid(), worktree=str(REPO), outcomes={},
                 code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    if args.phase == 'formal':
        gates = json.loads((RUN / 'resource-results.json').read_text())
        state['outcomes'] = gates
    start = time.monotonic()
    status(state)
    def stage(action, group):
        assert_gpus_free()
        name = f'{action}-{group}'
        record_path = EVIDENCE / f'{name}.execution.json'
        console = RUN / f'{name}.console.txt'
        assert not record_path.exists() and not console.exists(), f'Existing stage: {name}'
        command = ['bash', str(EXP / 'run.sh'), action, group]
        record = dict(command=command, cwd=str(REPO), code_commit=state['code_commit'],
                      started_utc=now(), stage=name)
        state.update(stage=name, stage_started_utc=record['started_utc'])
        status(state)
        tick = time.monotonic()
        with console.open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state['stage_pid'] = child.pid
            status(state)
            code = child.wait()
        record.update(exit_code=code, elapsed_seconds=time.monotonic() - tick, finished_utc=now())
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        # The rank0 update summaries are compact; full per-rank/token logs stay server-local.
        shutil.copy2(console, EVIDENCE / console.name)
        print(json.dumps(record), flush=True)
        if code:
            text = console.read_text()
            if action == 'probe' and 'CUDA out of memory' in text:
                return dict(passed=False, resource_failure='CUDA out of memory', exit_code=code)
            raise subprocess.CalledProcessError(code, command)
        if action in ('probe', 'smoke', 'formal'):
            out = RUN / action / group
            for filename in ('config.json', 'acceptance.json'):
                shutil.copy2(out / filename, EVIDENCE / f'{action}-{group}-{filename}')
            accepted = json.loads((out / 'acceptance.json').read_text())
            if action != 'probe':
                assert accepted['passed'], f'{name} failed: {accepted}'
            return accepted
        if action == 'verify-export':
            check = json.loads((RUN / 'formal' / group / 'export-check.json').read_text())
            assert check['passed'] and check['optimizer_steps'] == [500]
        return dict(exit_code=0)
    try:
        if args.phase == 'gates':
            assert not (RUN / 'resource-results.json').exists()
            assert json.loads((EVIDENCE / 'ddp-reference.json').read_text())['passed']
            for group in GROUPS:
                result = stage('probe', group)
                state['outcomes'][group] = dict(resource_status='passed' if result['passed'] else 'resource_infeasible',
                                               resource_result=result)
                status(state)
            (RUN / 'resource-results.json').write_text(json.dumps(state['outcomes'], indent=2) + '\n')
            shutil.copy2(RUN / 'resource-results.json', EVIDENCE / 'resource-results.json')
        else:
            for group in GROUPS:
                if state['outcomes'][group]['resource_status'] == 'passed':
                    stage('smoke', group)
            for group in GROUPS:
                if state['outcomes'][group]['resource_status'] != 'passed':
                    continue
                stage('formal', group)
                for action in ('export', 'verify-export', 'coco', 'urban', 'flickr_test1k', 'docci'):
                    stage(action, group)
                state['outcomes'][group]['formal_completed'] = True
                status(state)
            state['stage'] = 'report'
            status(state)
            subprocess.run([PYTHON, str(EXP / 'summarize.py')], cwd=REPO, check=True)
            state['stage'] = 'github-sync'
            status(state)
            assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
            subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
            subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
            subprocess.run(['git','commit','-m','Report Stack-Pool and CrossScore 500-step exploration'],cwd=REPO,check=True)
            push=subprocess.run(['git','push','origin','codex/nest-stack-crossscore-v1'],cwd=REPO)
            state['github_push_exit_code']=push.returncode
            state['result_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
            if push.returncode:
                raise subprocess.CalledProcessError(push.returncode,['git','push'])
            state['github_synced']=True
        state.update(status='completed',exit_code=0)
    except Exception as exc:
        state.update(status='failed',exit_code=getattr(exc,'returncode',1),error=f'{type(exc).__name__}: {exc}')
        print(state['error'],file=sys.stderr,flush=True)
    finally:
        state.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start)
        status(state)
    sys.exit(state['exit_code'])


if __name__ == '__main__':
    main()
