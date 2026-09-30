"""Run the authorized continuation detached, preserving its log and exit status."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve().with_name('run_training.sh')
REPO = SCRIPT.parents[3]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/vcp_mask_3epoch_v1')


def write_state(state):
    temporary = RUN / 'status.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(RUN / 'status.json')


def main():
    state = dict(status='running', supervisor_pid=os.getpid(),
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 command=['bash', str(SCRIPT)], worktree=str(REPO))
    started = time.monotonic()
    process = subprocess.Popen(state['command'], cwd=REPO)
    state['launcher_pid'] = process.pid
    write_state(state)
    code = process.wait()
    state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - started)
    write_state(state)
    sys.exit(code)


if __name__ == '__main__':
    main()
