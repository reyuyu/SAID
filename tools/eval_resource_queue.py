"""Dynamic dataset queue, one owned process per GPU, atomic state, explicit recovery."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from tools.retrieval_bounded import atomic_json, digest, sha
from tools.eval_five_parallel import require_gpu_idle, utc

ROOT = Path(__file__).resolve().parents[1]


def preflight(output, gpus):
    require_gpu_idle(gpus)
    mem = dict((line.split(':')[0], int(line.split()[1]) * 1024)
               for line in Path('/proc/meminfo').read_text().splitlines())
    disk = shutil.disk_usage(output)
    if mem['MemAvailable'] < 8 << 30 or disk.free < 10 << 30:
        raise RuntimeError('insufficient available CPU RAM/disk')
    return {'mem_available_bytes': mem['MemAvailable'], 'disk_free_bytes': disk.free,
            'cpu_count': os.cpu_count(), 'gpu_listing': subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
                '--format=csv,noheader'], text=True), 'utc': utc()}


def validate_jobs(jobs):
    names = [j['name'] for j in jobs]
    if not names or len(set(names)) != len(names):
        raise ValueError('empty/duplicate job names')
    for j in jobs:
        if not j['name'].replace('-', '').replace('_', '').isalnum():
            raise ValueError('unsafe job name')


class Queue:
    def __init__(self, plan, output, *, resume=False, gpus=(0, 1, 2, 3), resource_check=preflight):
        validate_jobs(plan['jobs'])
        if len(gpus) != len(set(gpus)) or not set(gpus) <= {0, 1, 2, 3}:
            raise ValueError('unauthorized/duplicate GPU')
        self.plan, self.output, self.gpus, self.resource_check = plan, Path(output), gpus, resource_check
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / 'RUN_STATE.json'
        self.active = {}; self.cancelled = False
        identity = digest(plan)
        if self.path.exists():
            if not resume:
                raise FileExistsError(self.path)
            self.state = json.loads(self.path.read_text())
            if self.state['plan_sha256'] != identity:
                raise ValueError('recovery plan mismatch')
            for name, record in self.state['jobs'].items():
                if record['status'] == 'COMPLETED':
                    if sha(record['output']) != record['output_sha256']:
                        raise ValueError('completed result changed')
                    payload = json.loads(Path(record['output']).read_text())
                    if payload.get('status') != 'COMPLETED':
                        raise ValueError('not a completed result')
                elif record['status'] == 'RUNNING':
                    # Refuse recovery while any recorded owned process still exists.
                    try:
                        os.kill(record['pid'], 0)
                    except ProcessLookupError:
                        record['status'] = 'INTERRUPTED'
                    else:
                        raise RuntimeError('recorded worker still exists')
        else:
            self.state = {'plan_sha256': identity, 'status': 'PREPARED', 'jobs': {},
                          'started_utc': utc(), 'attempts': [], 'blocked': plan.get('blocked', {})}
        self.persist()

    def persist(self):
        atomic_json(self.path, self.state)

    def cancel(self, *_):
        self.cancelled = True
        self.cancel_time = time.monotonic()
        for process, record, handle in self.active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        self.state['status'] = 'CANCELLING'; self.persist()

    def run(self):
        with ExitStack() as locks:
            lock = locks.enter_context((self.output / '.queue.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for gpu in self.gpus:
                f = locks.enter_context(Path(f'/tmp/said-eval-gpu-{gpu}.lock').open('a'))
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                evidence = self.resource_check(self.output, self.gpus)
            except Exception as error:
                self.state.update(status='FAILED_PREFLIGHT', error=repr(error), ended_utc=utc())
                self.persist()
                raise
            self.state['preflight'] = evidence; self.state['status'] = 'RUNNING'
            pending = [j for j in self.plan['jobs']
                       if self.state['jobs'].get(j['name'], {}).get('status') != 'COMPLETED']
            # Initial cost from actual full counts; update remaining predictions from measured rate.
            pending.sort(key=lambda j: j['estimated_seconds'], reverse=True)
            self.persist(); start = time.monotonic()
            old_handlers = {s: signal.signal(s, self.cancel) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                while pending or self.active:
                    if self.cancelled:
                        for j in pending:
                            self.state['jobs'][j['name']] = {'status': 'CANCELLED_BEFORE_START'}
                        pending.clear()
                    occupied = set(self.active)
                    for gpu in self.gpus:
                        if not pending or gpu in occupied or self.cancelled:
                            continue
                        # Recheck RAM and disk without mistaking our active workers for external processes.
                        avail = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                                         if s.startswith('MemAvailable:'))) * 1024
                        if avail < 8 << 30 or shutil.disk_usage(self.output).free < 10 << 30:
                            self.cancel(); break
                        j = pending.pop(0)
                        prior = self.state['jobs'].get(j['name'])
                        attempt = len([r for r in self.state['attempts'] if r['name'] == j['name']]) + 1
                        d = self.output / j['name'] / f'attempt{attempt}'
                        d.mkdir(parents=True, exist_ok=False)
                        job_path = d / 'JOB.json'; atomic_json(job_path, j, exclusive=True)
                        command = [sys.executable, '-m', 'tools.eval_hyfl_native', '--job', str(job_path),
                                   '--checkpoint', self.plan['checkpoint'], '--output', str(d / 'RESULT.json'),
                                   '--cache-dir', str(self.output / 'cache' / j['name']), '--device', f'cuda:{gpu}']
                        if 'command' in j:  # Explicit command jobs support scheduler tests/other read-only evaluators.
                            command = j['command']
                        log = (d / 'worker.log').open('xb')
                        try:
                            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                                       stdout=log, stderr=subprocess.STDOUT,
                                                       stdin=subprocess.DEVNULL, start_new_session=True)
                        except Exception:
                            log.close(); raise
                        r = {'name': j['name'], 'attempt': attempt, 'gpu': gpu, 'pid': process.pid,
                             'status': 'RUNNING', 'started_utc': utc(), 'start_monotonic': time.monotonic(),
                             'output': str(d / 'RESULT.json'), 'log': str(d / 'worker.log'), 'command': command,
                             'prior_attempt_status': prior.get('status') if prior else None}
                        self.active[gpu] = (process, r, log)
                        self.state['attempts'].append(r); self.state['jobs'][j['name']] = r; self.persist()
                    for gpu, (process, r, log) in list(self.active.items()):
                        rc = process.poll()
                        if rc is None:
                            continue
                        log.close(); del self.active[gpu]
                        r.update(returncode=rc, ended_utc=utc(), elapsed_seconds=time.monotonic() - r['start_monotonic'])
                        try:
                            payload = json.loads(Path(r['output']).read_text())
                            if rc != 0 or payload['status'] != 'COMPLETED':
                                raise ValueError('worker did not complete')
                            r.update(status='COMPLETED', output_sha256=sha(r['output']))
                            j = next(j for j in self.plan['jobs'] if j['name'] == r['name'])
                            observed_rate = r['elapsed_seconds'] / j.get('work_units', 1)
                            pending.sort(key=lambda p: p.get('work_units', 1) * observed_rate, reverse=True)
                        except Exception as error:
                            r.update(status='CANCELLED' if self.cancelled else 'FAILED', error=str(error))
                            # Retain successful peers; cancel only owned workers and queued jobs.
                            self.cancel()
                        self.persist()
                    if self.active:
                        if self.cancelled and time.monotonic() - self.cancel_time > 10:
                            for process, _, _ in self.active.values():
                                if process.poll() is None:
                                    os.killpg(process.pid, signal.SIGKILL)
                        time.sleep(0.2)
                self.state.update(status='FAILED' if self.cancelled else 'COMPLETED', ended_utc=utc(),
                                  wall_seconds=time.monotonic() - start)
                self.persist()
                return 1 if self.cancelled else 0
            except BaseException as error:
                self.cancel(); self.state.update(error=repr(error), status='FAILED'); self.persist()
                raise
            finally:
                for process, r, log in self.active.values():
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
                    log.close()
                    r.update(status='CANCELLED', returncode=process.returncode, ended_utc=utc())
                self.active.clear(); self.persist()
                for s, h in old_handlers.items():
                    signal.signal(s, h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    raise SystemExit(Queue(json.loads(Path(a.plan).read_text()), a.output, resume=a.resume).run())


if __name__ == '__main__':
    main()
