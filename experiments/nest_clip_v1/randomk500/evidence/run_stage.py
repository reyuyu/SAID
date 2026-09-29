"""Record an explicitly requested command without modifying its environment or arguments."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
name, *command = sys.argv[1:]
assert name and command and '/' not in name
record = root / (name + '.execution.json')
log = root / (name + '.console.txt')
if record.exists() or log.exists():
    raise FileExistsError(name)
metadata = dict(name=name, command=command, cwd='/root/lk_projects/SAID',
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                cwd='/root/lk_projects/SAID', text=True).strip())
record.write_text(json.dumps(metadata, indent=2) + '\n')
start = time.perf_counter()
with log.open('x') as handle:
    completed = subprocess.run(command, cwd=metadata['cwd'], stdout=handle, stderr=subprocess.STDOUT)
metadata.update(exit_code=completed.returncode, wall_seconds=time.perf_counter() - start,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
record.write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
sys.exit(completed.returncode)
