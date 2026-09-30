"""Continue the audited Balanced state only after the authorized Long-DCI job finishes."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_stack_3epoch_v1')
PREVIOUS = Path('/root/lk_projects/SAID-nest-clip-v1/mask_balance_cosine_v1/long-dci-status.json')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(state):
    temp = RUN / 'status.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(RUN / 'status.json')


def main():
    assert not (RUN / 'status.json').exists(), 'Continuation already scheduled'
    state = dict(status='waiting_for_long_dci', supervisor_pid=os.getpid(), started_utc=now(),
                 worktree=str(REPO), code_commit=subprocess.check_output(
                     ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                 command=['bash', str(EXP / 'run_training.sh')])
    start = time.monotonic()
    write_status(state)
    try:
        while True:
            previous = json.loads(PREVIOUS.read_text())
            if previous['status'] == 'completed':
                assert previous['exit_code'] == 0
                break
            if previous['status'] == 'failed':
                raise RuntimeError(f'Preceding evaluation failed: {previous.get("error")}')
            time.sleep(2)
        occupied = subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not occupied, f'GPUs occupied; will not contend: {occupied}'
        state.update(status='running', training_started_utc=now(),
                     preceding_evaluation_commit=previous.get('result_commit'))
        tick = time.monotonic()
        process = subprocess.Popen(state['command'],cwd=REPO,stdin=subprocess.DEVNULL)
        state['launcher_pid'] = process.pid
        write_status(state)
        code = process.wait()
        state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                     training_elapsed_seconds=time.monotonic()-tick)
    except Exception as exc:
        state.update(status='failed', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        print(state['error'],file=sys.stderr,flush=True)
    finally:
        state.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start)
        write_status(state)
    sys.exit(state['exit_code'])


if __name__ == '__main__':
    main()
