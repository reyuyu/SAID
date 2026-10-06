"""Read-only step3651 strict native evaluation; never launches a trainer."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from recovery.s02_nfs500 import ROOT, PYTHON, dump, now, sha

RUN = ROOT / 'runtime/SAID-nest-clip-v1/s02-local500-20261006'
CHECKPOINT = RUN / 'step4868/step003651.pt'
EXPECTED_SHA = 'd3fbe825c70ba403fb867223d234d3fb6cab5276cdca5a4ed3bdec1f8c968a3a'
EVAL = RUN / 'epoch3-eval-step3651'
BARE = EVAL / 'student_step3651.pt'
OUT = ROOT / 'recovery'
RANDOMK = dict(Score5=72.768147, J_long3=77.070244, J_long=85.485003, Short4=66.315)
RANDOMK_R1 = {'COCO': (61.70, 42.22), 'Urban-1k': (92.10, 91.10),
              'Flickr30k-test1k': (88.90, 72.44), 'DOCCI': (78.76, 79.98),
              'Long-DCI': (59.668508, 60.812944)}


def execute(name, arguments):
    record = EVAL / (name + '.command.json')
    log = EVAL / (name + '.console.log')
    if record.exists():
        previous = json.loads(record.read_text())
        assert previous['exit_code'] == 0 and previous['command'] == arguments
        return
    assert not log.exists(), f'Interrupted stage preserved: {log}'
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    start = time.monotonic()
    state = dict(stage=name, command=arguments, started_at=now(), status='RUNNING')
    dump(EVAL / 'state.json', state)
    print(f'{now()} starting {name}', flush=True)
    with log.open('x') as handle:
        child = subprocess.Popen(arguments, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1'))
        dump(EVAL / 'state.json', dict(state, child_pid=child.pid))
        code = child.wait()
    state.update(exit_code=code, finished_at=now(), elapsed_seconds=time.monotonic()-start,
                 status='COMPLETED' if code == 0 else 'FAILED')
    dump(record, state)
    dump(EVAL / 'state.json', state)
    print(f'{now()} finished {name}: exit={code}', flush=True)
    if code:
        raise subprocess.CalledProcessError(code, arguments)


def main():
    import torch
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores

    EVAL.mkdir(parents=True, exist_ok=True)
    assert sha(CHECKPOINT) == EXPECTED_SHA, 'Epoch3 identity changed'
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert state['completed_steps'] == state['global_step'] == 3651
    assert state['scheduler_horizon'] == state['scheduler']['horizon'] == 4868
    assert state['data_cursor'] == dict(next_epoch=3, next_batch=0)
    counters = sorted({int(value['step']) for value in state['optimizer']['state'].values()})
    assert counters == [3651]
    assert state['trajectory_root'] == str(RUN)
    cfg = state['config']
    assert cfg['sampling_mode'] == 'summary_random_detail' and cfg['view_weights'] == [1.4, .2, 1.4]
    assert cfg['seed'] == 0 and cfg['batch_size'] == 256 and cfg['world_size'] == 4
    identity = dict(checkpoint=str(CHECKPOINT), sha256=EXPECTED_SHA, size_bytes=CHECKPOINT.stat().st_size,
                    completed_steps=3651, epoch=3, scheduler_horizon=4868,
                    data_cursor=state['data_cursor'], optimizer_counters=counters,
                    rng_rank_count=len(state['rng_per_rank']), trajectory_root=state['trajectory_root'],
                    sampling_mode=cfg['sampling_mode'], view_weights=cfg['view_weights'])
    dump(EVAL / 'identity.json', identity)
    del state
    execute('export', [PYTHON, '-m', 'tools.nest_clip', 'export', '--checkpoint', str(CHECKPOINT),
                       '--output', str(BARE), '--expect-updates', '3651'])
    execute('verify', [PYTHON, '-m', 'tools.nest_clip', 'verify-export', '--checkpoint', str(CHECKPOINT),
                       '--bare', str(BARE), '--output', str(EVAL / 'export-check.json'),
                       '--index-dir', '/root/said_s02_stage500/data_index',
                       '--image-root', '/root/said_s02_stage500/ShareGPT4V'])
    assets = ROOT / 'local_assets'
    for name, folder in [('coco', 'evaluation/coco/val2017'), ('urban', 'evaluation/Urban1k/Urban1k')]:
        execute(name, [PYTHON, '-m', 'tools.eval_nest_native', '--checkpoint', str(BARE),
                       '--dataset', name, '--root', str(assets / folder), '--device', 'cuda:0',
                       '--batch-size', '64', '--output', str(EVAL / (name + '_native.json'))])
    bench = assets / 'retrieval_benchmarks'
    for name, manifest, folder in [('flickr_test1k', 'flickr30k_test1k.jsonl', 'flickr30k'),
                                   ('docci', 'docci_test.jsonl', 'docci'),
                                   ('long_dci', 'long_dci_reconstructed.jsonl', 'dci')]:
        execute(name, [PYTHON, '-m',
                       'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                       '--checkpoint', str(BARE), '--device', 'cuda:0', '--batch-size', '64',
                       '--output-dir', str(EVAL / name),
                       f'{name}:{bench / "manifests" / manifest}:{bench / folder / "images"}'])
    assert sha(CHECKPOINT) == EXPECTED_SHA, 'Evaluation modified training checkpoint'
    metrics, raw, sources = native_metrics(EVAL)
    checks = json.loads((EVAL / 'export-check.json').read_text())
    assert checks['checkpoint_sha256'] == EXPECTED_SHA and checks['optimizer_steps'] == [3651]
    calculated = scores(metrics)
    percent = dict(Score5=100*calculated['Score5_R1'], J_long3=100*calculated['J_long3'],
                   J_long=100*calculated['J_long'],
                   Short4=100*statistics.fmean(metrics[d][r]['R@1']
                       for d in ('COCO', 'Flickr30k-test1k') for r in ('I2T', 'T2I')))
    epoch4 = json.loads((OUT / 'FULL_RESULTS.json').read_text())
    evidence = {name: dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))
                for name, path in sources.items()}
    logs = []
    for path in sorted(EVAL.glob('*.console.log')):
        cmd = json.loads(path.with_name(path.name.replace('.console.log', '.command.json')).read_text())
        logs.append(dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path),
                         started_at=cmd['started_at'], finished_at=cmd['finished_at'],
                         elapsed_seconds=cmd['elapsed_seconds']))
    result = dict(status='EPOCH3_EVALUATION_COMPLETE', identity=identity, strict_export=checks,
                  evaluation_checkpoint_immutable=True, native_full_caption_only=True,
                  protocol='normalize(native image embedding) @ normalize(native full-caption text embedding).T',
                  metrics=metrics, scores_percent=percent, finished_at=now(),
                  delta_vs_epoch4_pp={k:v-epoch4['scores_percent'][k] for k,v in percent.items()},
                  delta_vs_RandomK4epoch_pp={k:v-RANDOMK[k] for k,v in percent.items()},
                  dataset_R1_delta_vs_epoch4_pp={d:{r:100*(metrics[d][r]['R@1']-epoch4['metrics'][d][r]['R@1'])
                      for r in ('I2T','T2I')} for d in metrics},
                  dataset_R1_delta_vs_RandomK4epoch_pp={d:{r:100*metrics[d][r]['R@1']-RANDOMK_R1[d][i]
                      for i,r in enumerate(('I2T','T2I'))} for d in metrics},
                  native_evaluation_evidence=evidence, local_raw_logs_not_uploaded=logs,
                  commands=[json.loads(p.read_text()) for p in sorted(EVAL.glob('*.command.json'))],
                  git_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(OUT / 'EPOCH3_RESULTS.json', result)
    lines = ['# S=0.2 epoch3 strict native evaluation', '',
             'Same formal trajectory, step3651 (3 ×1217 updates), horizon4868. Evaluation only; no training.', '',
             '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |',
             '|---|---|---|']
    for d,m in metrics.items():
        fields = [' / '.join(f'{100*m[r][k]:.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
        lines.append(f'| {d} | {fields[0]} | {fields[1]} |')
    lines += ['', '| Score (%) | Epoch3 | Epoch4 | Delta vs epoch4 (pp) | Delta vs RandomK4epoch (pp) |',
              '|---|---:|---:|---:|---:|']
    for k,v in percent.items():
        lines.append(f'| {k} | {v:.6f} | {epoch4["scores_percent"][k]:.6f} | {result["delta_vs_epoch4_pp"][k]:+.6f} | {result["delta_vs_RandomK4epoch_pp"][k]:+.6f} |')
    lines += ['', 'RandomK comparison uses its 4epoch reference; it is not a matched epoch3 baseline.', '',
              f'Checkpoint: `{CHECKPOINT}`; SHA256 `{EXPECTED_SHA}`; bytes {identity["size_bytes"]}.',
              f'Bare student: `{BARE}`; SHA256 `{checks["bare_sha256"]}`.', '',
              'Strict model load; native image/text embedding equivalence exact. Checkpoint SHA unchanged before/after evaluation. '
              'All five results share the bare SHA. Long-DCI uses 7602 images/captions and the frozen manifest. '
              'No mask/gate/rerank/ensemble/Summary/Detail inference.', '',
              'Detailed dataset R1 deltas, command provenance and local raw-log paths/sizes/SHA/time ranges: `EPOCH3_RESULTS.json`. '
              'Checkpoints, bare student, dataset and raw logs remain local.']
    (OUT / 'EPOCH3_RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(EVAL / 'state.json', dict(status='EPOCH3_EVALUATION_COMPLETE', finished_at=now(), scores_percent=percent))
    print(json.dumps(percent, indent=2), flush=True)


if __name__ == '__main__':
    main()
