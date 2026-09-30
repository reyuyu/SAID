"""Run final VCP native evaluation and publish only compact result artifacts."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/vcp_mask_3epoch_v1')
OUT = RUN / 'formal/VCP-Mask'
EVIDENCE = EXP / 'evidence'
PYTHON = '/root/miniconda3/envs/said-repro/bin/python'
ASSETS = Path('/root/lk_projects/SAID-assets')
BRANCH = 'codex/nest-vcp-mask-3epoch-v1'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(state):
    path = RUN / 'evaluation-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def main():
    status_path = RUN / 'evaluation-status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    assert json.loads((RUN / 'status.json').read_text())['exit_code'] == 0
    acceptance = json.loads((OUT / 'acceptance.json').read_text())
    assert acceptance['passed'] and all(r['completed_updates'] == 3651 for r in acceptance['ranks'])
    activity = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not activity, f'GPUs are occupied: {activity}'
    bare = OUT / 'student_step3651.pt'
    destinations = [bare, OUT / 'export-check.json', OUT / 'coco_native.json',
                    OUT / 'urban_native.json', OUT / 'flickr_test1k', OUT / 'docci']
    assert not any(p.exists() for p in destinations), 'Refusing to overwrite evaluation artifacts'
    EVIDENCE.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    state = dict(status='running', stage='starting', started_utc=now(),
                 supervisor_pid=os.getpid(), worktree=str(REPO),
                 evaluation_git_head=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), stages=[])
    started = time.monotonic()
    write_status(state)

    def stage(name, command):
        evidence = EVIDENCE / f'{name}.execution.json'
        log = EVIDENCE / f'{name}.console.txt'
        assert not evidence.exists() and not log.exists()
        record = dict(stage=name, command=command, started_utc=now(),
                      cwd=str(REPO), git_head=state['evaluation_git_head'])
        state.update(stage=name, stage_started_utc=record['started_utc'])
        write_status(state)
        tick = time.monotonic()
        print(json.dumps(dict(event='stage_started', **record)), flush=True)
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT)
            state['stage_pid'] = process.pid
            write_status(state)
            code = process.wait()
        record.update(exit_code=code, finished_utc=now(), elapsed_seconds=time.monotonic() - tick)
        evidence.write_text(json.dumps(record, indent=2) + '\n')
        state['stages'].append(record)
        write_status(state)
        print(json.dumps(dict(event='stage_finished', stage=name, exit_code=code)), flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)

    try:
        stage('export', [PYTHON, '-m', 'tools.nest_clip', 'export', '--checkpoint',
                        str(OUT / 'step003651.pt'), '--expect-updates', '3651', '--output', str(bare)])
        stage('verify-export', [PYTHON, '-m', 'tools.nest_clip', 'verify-export', '--checkpoint',
                               str(OUT / 'step003651.pt'), '--bare', str(bare), '--output',
                               str(OUT / 'export-check.json'), '--index-dir',
                               '/root/lk_projects/SAID-nest-clip-v1/data_index', '--image-root',
                               str(ASSETS / 'training/ShareGPT4V')])
        check = json.loads((OUT / 'export-check.json').read_text())
        assert check['passed'] and check['strict_load'] and check['optimizer_steps'] == [3651]
        for name, root in [('coco', ASSETS / 'evaluation/coco/val2017'),
                           ('urban', ASSETS / 'evaluation/Urban1k/Urban1k')]:
            stage(name, [PYTHON, '-m', 'tools.eval_nest_native', '--checkpoint', str(bare),
                         '--dataset', name, '--root', str(root), '--device', 'cuda:0',
                         '--batch-size', '64', '--output', str(OUT / f'{name}_native.json')])
        bench = ASSETS / 'retrieval_benchmarks'
        for name, manifest, images in [
                ('flickr_test1k', 'flickr30k_test1k.jsonl', 'flickr30k/images'),
                ('docci', 'docci_test.jsonl', 'docci/images')]:
            stage(name, [PYTHON, '-m',
                         'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                         '--checkpoint', str(bare), '--device', 'cuda:0', '--batch-size', '64',
                         '--output-dir', str(OUT / name),
                         f'{name}:{bench / "manifests" / manifest}:{bench / images}'])
        stage('summarize', [PYTHON, str(EXP / 'summarize.py')])
        state['evaluation_completed'] = True
        write_status(state)
        state['stage'] = 'github-sync'
        write_status(state)
        # Add only this experiment's compact report/evidence, keeping server weights outside Git.
        paths = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).splitlines()
        assert not paths, 'Existing staged changes; results remain local'
        subprocess.run(['git', 'add', str(EXP.relative_to(REPO))], cwd=REPO, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPO, check=True)
        subprocess.run(['git', 'commit', '-m', 'Report VCP-Mask full three-epoch native results'],
                       cwd=REPO, check=True)
        pushed = subprocess.run(['git', 'push', 'origin', BRANCH], cwd=REPO)
        state['github_push_exit_code'] = pushed.returncode
        state['result_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        if pushed.returncode:
            raise subprocess.CalledProcessError(pushed.returncode, ['git', 'push', 'origin', BRANCH])
        state.update(status='completed', stage='finished', exit_code=0, github_synced=True)
    except Exception as exc:
        state.update(status='failed', exit_code=getattr(exc, 'returncode', 1),
                     error=f'{type(exc).__name__}: {exc}')
        print(state['error'], file=sys.stderr, flush=True)
    finally:
        state.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_status(state)
    sys.exit(state['exit_code'])


if __name__ == '__main__':
    main()
