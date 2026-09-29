"""Run one fast experiment stage and preserve command, source commit, console and exit code."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


here = Path(__file__).resolve().parent
repo = here.parents[3]
name, *command = sys.argv[1:]
assert name and command and '/' not in name
record = here / f'{name}.execution.json'
log = here / f'{name}.console.txt'
if record.exists() or log.exists():
    raise FileExistsError(name)
metadata = dict(name=name, command=command, cwd=str(repo),
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                 cwd=repo, text=True).strip())
record.write_text(json.dumps(metadata, indent=2) + '\n')
started = time.perf_counter()
with log.open('x') as handle:
    completed = subprocess.run(command, cwd=repo, stdout=handle,
                               stderr=subprocess.STDOUT)
metadata.update(exit_code=completed.returncode,
                wall_seconds=time.perf_counter() - started,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
record.write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
sys.exit(completed.returncode)
