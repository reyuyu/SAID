"""User-requested supplemental Long-DCI evaluation, separate from frozen four-set results."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.nest_clip_v1.stack_crossscore_v1.summarize import sha


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/mask_balance_cosine_v1')
FORMAL = RUN / 'formal'
EVIDENCE = EXP / 'evidence/long_dci'
GROUPS = ('Balanced-Stack-Patch', 'Cosine-CrossScore-CLS')
PYTHON = '/root/miniconda3/envs/said-repro/bin/python'
BENCH = Path('/root/lk_projects/SAID-assets/retrieval_benchmarks')
MANIFEST = BENCH / 'manifests/long_dci_reconstructed.jsonl'
IMAGES = BENCH / 'dci/images'
EXPECTED_MANIFEST_SHA = '8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def write_status(state):
    target = RUN / 'long-dci-status.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(target)


def main():
    assert not (RUN / 'long-dci-status.json').exists(), 'Supplement already started'
    primary = load(RUN / 'status.json')
    assert primary['status'] == 'completed' and primary['exit_code'] == 0
    assert sha(MANIFEST) == EXPECTED_MANIFEST_SHA
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert len(records) == 7602 and len({r['image_id'] for r in records}) == 7602
    assert all((IMAGES / r['image_path']).is_file() for r in records)
    reference_path = REPO / 'experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json'
    reference = load(reference_path)
    assert reference['evaluation_metadata']['TI-fast']['long-DCI']['manifest_sha256'] == EXPECTED_MANIFEST_SHA
    for group in GROUPS:
        check = load(FORMAL / group / 'export-check.json')
        assert check['passed'] and check['optimizer_steps'] == [500]
        assert sha(FORMAL / group / 'student_step500.pt') == check['bare_sha256']
        assert not (FORMAL / group / 'long_dci').exists(), 'Existing supplemental output'
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='starting', supervisor_pid=os.getpid(), started_utc=now(),
                 git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                 manifest_sha256=EXPECTED_MANIFEST_SHA, stages=[])
    started = time.monotonic()
    write_status(state)
    try:
        for group in GROUPS:
            active = subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert not active, f'GPUs occupied; refusing contention: {active}'
            out = FORMAL / group / 'long_dci'
            command = [PYTHON, '-m',
                       'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                       '--checkpoint', str(FORMAL / group / 'student_step500.pt'),
                       '--device', 'cuda:0', '--batch-size', '64', '--output-dir', str(out),
                       f'long_dci:{MANIFEST}:{IMAGES}']
            record = dict(group=group, command=command, cwd=str(REPO), git_head=state['git_head'], started_utc=now())
            state.update(stage=f'long_dci-{group}', stage_started_utc=record['started_utc'])
            write_status(state)
            tick = time.monotonic()
            with (EVIDENCE / f'{group}.console.txt').open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT,
                                         env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
                state['stage_pid'] = child.pid
                write_status(state)
                code = child.wait()
            record.update(exit_code=code, elapsed_seconds=time.monotonic()-tick, finished_utc=now())
            (EVIDENCE / f'{group}.execution.json').write_text(json.dumps(record, indent=2)+'\n')
            state['stages'].append(record)
            write_status(state)
            if code:
                raise subprocess.CalledProcessError(code, command)
            result = load(out / 'long_dci.json')
            assert result['n_images'] == result['n_captions'] == 7602
            assert result['manifest_sha256'] == EXPECTED_MANIFEST_SHA and result['native_student_only']
            assert result['checkpoint_sha256'] == load(FORMAL / group / 'export-check.json')['bare_sha256']
            shutil.copy2(out / 'long_dci.json', EVIDENCE / f'{group}.json')
        metrics = {'TI-fast@500': reference['metrics']['TI-fast']['long-DCI']}
        sources = {}
        for group in GROUPS:
            result = load(EVIDENCE / f'{group}.json')
            metrics[group] = {direction: {recall: result['metrics'][recall][direction]
                                         for recall in ('R@1','R@5','R@10')} for direction in ('I2T','T2I')}
            sources[group] = {key:value for key,value in result.items() if key != 'metrics'}
        deltas = {group: {direction: {recall:100*(metrics[group][direction][recall]-metrics['TI-fast@500'][direction][recall])
                                     for recall in ('R@1','R@5','R@10')} for direction in ('I2T','T2I')}
                  for group in GROUPS}
        final = dict(generated_utc=now(), protocol='long_dci', n_images=7602, n_captions=7602,
                     manifest_sha256=EXPECTED_MANIFEST_SHA, metrics=metrics, deltas_vs_TI500_pp=deltas,
                     evaluation_metadata=sources, execution=state['stages'],
                     evaluation_git_head=state['git_head'], reference_results_sha256=sha(reference_path),
                     note='Added by explicit user request after the frozen four-set 500-step experiment. '
                          'Native normalized image/text inner product, batch64, cuda:0. '
                          'J_long stays the Urban/DOCCI summary; Long-DCI is separate. '
                          'Original S-PATCH/C-CLS were not evaluated on Long-DCI; no unmeasured scores are inferred.')
        (EXP / 'LONG_DCI_RESULTS.json').write_text(json.dumps(final,indent=2)+'\n')
        lines=['# Supplemental Long-DCI: Balanced / Cosine @500', '',
               'This supplement was explicitly requested after the original four-dataset experiment. '
               'It keeps the same 7602 images/captions and full candidate pool as TI-fast@500. '
               'Native normalized student embeddings, batch64 on cuda:0; no mask, fusion or rerank.', '',
               'Values are percentages; changes from TI-fast@500 are percentage points. J_long remains unchanged.', '',
               '| Direction | Recall | TI-fast@500 | Balanced@500 | Cosine@500 | Balanced-TI pp | Cosine-TI pp |',
               '|---|---|---:|---:|---:|---:|---:|']
        for direction in ('I2T','T2I'):
            for recall in ('R@1','R@5','R@10'):
                values=[f'{metrics[g][direction][recall]*100:.2f}' for g in ('TI-fast@500',*GROUPS)]
                differences=[f'{deltas[g][direction][recall]:+.2f}' for g in GROUPS]
                lines.append('| '+' | '.join([direction,recall,*values,*differences])+' |')
        lines += ['', 'Original Stack-Patch/CrossScore-CLS have no Long-DCI measurements and are not assigned '
                  'estimated scores. TI-fast@3651 is a different training budget and is not included in this ranking. '
                  'These single-seed benchmark results do not establish statistical significance.', '',
                  f'Manifest SHA256: `{EXPECTED_MANIFEST_SHA}`.', '',
                  'Raw evaluator JSON, checkpoint hashes, commands, actual evaluator commit, timings and exit codes '
                  'are preserved under evidence/long_dci/. Weights and datasets remain server-local. No DCI Full '
                  'evaluation or additional training was started.', '']
        (EXP / 'LONG_DCI_REPORT.md').write_text('\n'.join(lines))
        state.update(evaluation_completed=True, stage='github-sync')
        write_status(state)
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add',str(EXP.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m','Add requested Long-DCI scores for balanced and cosine masks'],cwd=REPO,check=True)
        push = subprocess.run(['git','push','origin','codex/nest-mask-balance-cosine-v1'],cwd=REPO)
        state.update(result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                     github_push_exit_code=push.returncode)
        if push.returncode:
            raise subprocess.CalledProcessError(push.returncode,['git','push'])
        state.update(status='completed', exit_code=0, github_synced=True)
    except Exception as exc:
        state.update(status='failed', exit_code=getattr(exc,'returncode',1), error=f'{type(exc).__name__}: {exc}')
        print(state['error'],file=sys.stderr,flush=True)
    finally:
        state.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_status(state)
    sys.exit(state['exit_code'])


if __name__ == '__main__':
    main()
