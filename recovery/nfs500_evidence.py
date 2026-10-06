"""Review completed-run evidence without training, copying, or dataset decoding."""
import datetime
import json
from pathlib import Path
import re
import subprocess
import time

from recovery.s02_nfs500 import ROOT, RUN, OUT, PHASE, dump, rows, sha


def kernel_since_launch():
    provenance = json.loads((RUN / 'launch-provenance.json').read_text())
    started = datetime.datetime.fromisoformat(provenance['started_utc'])
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
    boot_start = time.monotonic() - elapsed
    commands = json.loads((RUN / 'commands.json').read_text())
    ended_utc = commands[-1]['ended_utc']
    boot_end = boot_start + (datetime.datetime.fromisoformat(ended_utc) - started).total_seconds()
    result = subprocess.run(['dmesg', '--color=never'], capture_output=True, text=True)
    if result.returncode:
        return dict(available=False, reason='dmesg unavailable', exitcode=result.returncode)
    selected = []
    for line in result.stdout.splitlines():
        match = re.match(r'\[\s*([\d.]+)\]', line)
        if match and boot_start <= float(match.group(1)) <= boot_end:
            selected.append(line)
    raw = RUN / 'kernel-during-run.raw.log'
    raw.write_text('\n'.join(selected) + '\n')
    terms = {
        'NFS_RPC': r'NFS.*(?:not responding|error)|RPC.*(?:timeout|timed out)|rpc_check_timeout',
        'IO_FILESYSTEM': r'I/O error|Input/output error|EXT4-fs error|XFS.*(?:error|corrupt)|nvme.*(?:error|reset)',
        'GPU_DRIVER': r'NVRM.*(?:Xid|SXid)|fallen off the bus|AER.*(?:error|fatal)',
        'OOM_HANG': r'oom-killer|Out of memory|Killed process|hung task|blocked for more than|lockup|panic',
    }
    counts = {name:sum(bool(re.search(pattern, line, re.I)) for line in selected) for name,pattern in terms.items()}
    # Counts only: do not publish unreviewed host process names, paths or identifiers.
    return dict(available=True, scope='Host kernel ringbuffer entries since run launch; host-wide, no attribution by proximity',
                boot_seconds_start=boot_start, boot_seconds_end=boot_end,
                time_range_utc=[provenance['started_utc'], ended_utc], line_count=len(selected),
                keyword_counts=counts, raw_path=str(raw), raw_sha256=sha(raw), raw_bytes=raw.stat().st_size,
                uploaded=False)


def full_stream_check():
    actual = rows(RUN / 'step500/steps.jsonl')
    by_step = {r['step']:{p['rank']:p['sampling']['sample_ids'] for p in r['rank_health']} for r in actual}
    manifest = OUT / 'evidence/s02-stage500-local/ordered-records.jsonl'
    count = 0
    with manifest.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item['step'] not in by_step:
                continue
            if by_step[item['step']][item['rank']][item['batch_position']] != item['sample_id']:
                raise RuntimeError('Full500 sample order drift')
            count += 1
    expected = len(actual) * 1024
    if count != expected:
        raise RuntimeError('Full sample stream coverage failed')
    proof = json.loads((manifest.parent / 'manifest-proof.json').read_text())
    return dict(passed=True, records_checked=count, expected_records=expected, completed_steps=len(actual),
                manifest_sha256=sha(manifest), manifest_path=str(manifest),
                historical_manifest_proof=proof, separate_readonly_process=True,
                training_RNG_untouched=True, text_token_stream_first5='Exact historical reference gate')


def main():
    result = json.loads((OUT / 'STEP500_RESULTS.json').read_text())
    stats = json.loads((OUT / 'NFS_RUNTIME_STATS.json').read_text())
    kernel = kernel_since_launch()
    complete = result.get('stopped_at_500') and 'metrics' in result
    stream = full_stream_check() if complete else dict(passed=False, records_checked=0,
        reason='No optimizer step completed; first-five and full500 stream gate not reached')
    result['full500_sample_stream_proof'] = stream
    stats['kernel_during_run'] = kernel
    stats['nfs_completability'] = ('500/500 completed with canonical NFS; slow batches returned; no sample substitution' if complete else
        'This attempt could not proceed: all four ranks timed out on the first batch at60s; 500-step completion is unproven')
    stats['true_io_error_observed'] = bool(stats['io_error_count'] or
        kernel.get('keyword_counts', {}).get('NFS_RPC') or kernel.get('keyword_counts', {}).get('IO_FILESYSTEM'))
    phase_resources = [r['system_after'] for rank in range(4) for r in rows(PHASE / f'rank{rank}.jsonl')]
    phase_resources.extend(r['system'] for r in rows(RUN / 'resource-telemetry.jsonl'))
    stats['peak_cgroup_memory_bytes'] = max([stats['peak_cgroup_memory_bytes']] + [r['memory_current'] for r in phase_resources])
    stats['peak_file_cache_bytes'] = max([stats['peak_file_cache_bytes']] + [r['file'] for r in phase_resources])
    stats['memory_oom_kill_observed'] = any(r['memory_events'].get('oom_kill', 0) for r in phase_resources)
    from recovery.s02_nfs500 import distribution
    for name in ('memory_PSI', 'io_PSI'):
        stats[name + '_some_avg10'] = distribution([r[name]['some']['avg10'] for r in phase_resources if 'some' in r[name]])
        stats[name + '_full_avg10'] = distribution([r[name]['full']['avg10'] for r in phase_resources if 'full' in r[name]])
    if kernel.get('available'):
        stats['raw_local_artifacts'] = [r for r in stats['raw_local_artifacts'] if r['path'] != kernel['raw_path']]
        stats['raw_local_artifacts'].append(dict(path=kernel['raw_path'], bytes=kernel['raw_bytes'],
            sha256=kernel['raw_sha256'], time_range_utc=kernel['time_range_utc'], uploaded=False))
    staging = json.loads((OUT / 'S02_LOCAL_STAGE500.json').read_text())
    result['staging_stop'] = dict(status=staging['status'], copied_images=staging['copy']['copied'],
        copied_bytes=staging['copy']['bytes'], local_root=staging['local_storage']['local_root'],
        workers_exited=staging['copy']['copy_workers_exited'], stopped_utc=staging['stopped_at'],
        existing_images_partials_ledgers_preserved=True)
    stats['staging_raw_local_artifacts'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p),
        time_range_utc=[staging['generated_at'], staging['stopped_at']], uploaded=False)
        for p in sorted((OUT/'evidence/s02-stage500-local').glob('*')) if p.is_file() and p.name != 'task.lock']
    stats['excluded_large_data'] = [dict(path=staging['local_storage']['local_root'],
        description='Retained116817 local images, partial files and parent data_index; not used in formal NFS attempt'),
        dict(path=str(ROOT/'local_assets'), description='Canonical training/eval datasets and model caches; unchanged, not uploaded'),
        dict(path=str(ROOT/'runtime/SAID-nest-clip-v1/shared'), description='Common step0 checkpoints retained, not uploaded'),
        dict(path='/root/.cache/clip/ViT-B-16.pt', description='Downloaded335MiB bootstrap model cache, not uploaded')]
    if complete:
        result['scores_percent']['Score5'] = result['scores_percent']['Score5_R1']
        result['scores_percent']['Short4'] = result['scores_percent']['Short4_R1']
    else:
        log = (RUN / 'train500.log').read_text(errors='replace')
        failed_ranks = sorted({int(x) for x in re.findall(r'\[rank(\d)\]: RuntimeError: DataLoader timed out after 60 seconds', log)})
        if failed_ranks != [0, 1, 2, 3]:
            raise RuntimeError('Unexpected failure evidence; requires specific review')
        stats.update(unreturned_batch_timeouts=[dict(rank=r, step=1, data_wait_lower_bound_seconds=60,
                     right_censored=True, batch_returned=False) for r in failed_ranks],
                     batch_timeout_count=len(failed_ranks), completed_full_cycle_statistics_available=False,
                     cgroup_resource_anomaly=False,
                     limitation='First-batch wait includes worker startup, NFS reads, decoding and native text/preprocess. No exclusive NFS-causality claim.',
                     heartbeat_annotation='Launch3d1d041 finally-block incorrectly labelled failed waits BATCH_RETURNED. All four native timeout tracebacks are authoritative; raw heartbeat files retained. Fixed for future runs, no retry.')
        result.update(first_five_gate_status='NOT_REACHED', step500_checkpoint_exists=False,
                      scores_percent={k:None for k in ('Score5','J_long3','J_long','Short4')},
                      reproduction_verdict='NOT_EVALUATED: no step500 checkpoint; cannot label PASS or FAIL_AT_500',
                      automatic_retry=False, true_hard_stop='FIRST_BATCH_DATA_WAIT_TIMEOUT_60S',
                      training_ended_utc=json.loads((RUN/'commands.json').read_text())[-1]['ended_utc'],
                      source_read_failure_observed=False, physical_EIO_observed=False,
                      no_copy_workers_verified=True, all_training_workers_exited=True)
        historical = json.loads((ROOT/'experiments/nest_clip_v1/armb_summary02_4epoch_v1/PARENT_500.json').read_text())
        result['historical_reference'] = dict(metrics=historical['metrics'], scores_percent={
            'Score5':70.364367, 'J_long3':73.726611, 'J_long':82.765002, 'Short4':65.321})
        result['metrics'] = {name:None for name in historical['metrics']}
        result['historical_deltas_pp'] = None
        stats['hard_stop_evidence'] = dict(native_exception='DataLoader timed out after 60 seconds',
            ranks=failed_ranks, operator_steps_completed=0, source=str(RUN/'train500.log'), source_sha256=sha(RUN/'train500.log'))
        gate_path = OUT / 'CONTINUATION_GATE.json'
        gate = json.loads(gate_path.read_text())
        gate.update(true_hard_stop=result['true_hard_stop'], first_five_gate='NOT_REACHED',
                    step500_checkpoint_exists=False, reproduction_evaluated=False, automatic_retry=False)
        dump(gate_path, gate)
    dump(OUT / 'STEP500_RESULTS.json', result)
    dump(OUT / 'NFS_RUNTIME_STATS.json', stats)
    report = OUT / 'STEP500_NFS_REPRODUCTION.md'
    lines = ['', '## Reviewed completed-run evidence', '',
             (f"All {stream['records_checked']} sample IDs match the frozen historical500 stream; first5 strings, tokens and LR passed exact checks." if complete else
              'No optimizer step completed. First-five gate and full500 stream checks were not reached. Do not label this REPRODUCTION_PASS or REPRODUCTION_FAIL_AT_500.'),
             f"NFS completion: {stats['nfs_completability']}. Observed actual I/O failure: {stats['true_io_error_observed']}.",
             'Kernel evidence: `' + json.dumps(kernel) + '`; old host-wide kernel entries outside this run are excluded.',
             'Memory/IO PSI is host-scoped on this cgroup v1 system; no actual oom_kill observed: ' + str(not stats['memory_oom_kill_observed']) + '.', '',
             '| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | R1 deltas I2T/T2I (pp) |',
             '|---|---|---|---|']
    gate = result.get('reproduction_gate', {})
    for name, metrics in (result.get('metrics', {}) if complete else {}).items():
        values = [' / '.join(f'{metrics[d][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        delta = gate['dataset_R1_deltas_pp'][name]
        lines.append(f"| {name} | {values[0]} | {values[1]} | {delta['I2T']:+.6f} / {delta['T2I']:+.6f} |")
    if not complete:
        for name in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI'):
            lines.append(f'| {name} | NOT EVALUATED | NOT EVALUATED | N/A |')
        lines += ['', 'Four native DataLoader exceptions confirm first-batch wait reached60s without a batch; this is the requested hard-stop condition. No physical EIO, image read failure, NFS-server-not-responding, CUDA/OOM or kernel-hang evidence was observed during this attempt.',
                  'Completed-step median/p95/p99/max are unavailable (zero completed steps); >3s and >10s **completed step counts** are both0. Four unreturned batches are separately reported as right-censored waits>=60s, not zero-duration batches.',
                  'Cold worker startup, decode/text/preprocess and NFS reads all contribute to initial wait. Evidence establishes this attempt cannot pass the60s gate; it does not isolate NFS as the sole cause.',
                  'The original heartbeat finally-block marked exceptions BATCH_RETURNED; native tracebacks demonstrate these were failures. Raw evidence is preserved, future instrumentation is corrected, and no automatic retry occurred.']
    lines += ['', 'Scores (%): `' + json.dumps(result['scores_percent']) + '`.',
              'Historical score deltas (pp): `' + json.dumps(gate.get('delta_vs_historical_S02_pp')) + '`.',
              'Checks: `' + json.dumps(gate.get('checks')) + '`.',
              'Dataset collapse definition: invalid/nonfinite R1 or any direction below50% of historical R1. Thresholds are fixed before evaluation; no weight tuning.', '',
              ('Run is stopped500.' if complete else 'Run is stopped before step1; no step500 checkpoint exists.') + ' No continuation/retry is authorized by this result.']
    prefix = report.read_text().split('\n## Reviewed completed-run evidence')[0].split('\n## Reviewed run evidence')[0]
    if not complete:
        checkpoint = str(RUN/'step500/step000500.pt')
        prefix = prefix.replace('Complete resumable checkpoint remains local: `' + checkpoint + '`.',
            'No step500 checkpoint exists. Training stopped before its first optimizer update.')
        prefix = prefix.replace('Strict bare export and evaluation leave that checkpoint unchanged.',
            'Strict bare export and five-set evaluation were not run.')
        prefix = prefix.replace('Detailed phase telemetry includes backward/DDP, optimizer, memory, PSI and GPU utilization.',
            'Resource telemetry includes cgroup memory, file cache, host PSI and GPU utilization. No forward/backward/optimizer phase completed.')
        prefix = prefix.replace('Five-set strict native scores (%): `{}`.', 'Five-set results and all historical deltas: NOT EVALUATED.')
        prefix = prefix.replace('Both directions R@1/5/10 and historical deltas are in STEP500_RESULTS.json;',
            'STEP500_RESULTS.json records unavailable metrics and the historical reference;')
    lines[1] = '## Reviewed run evidence'
    report.write_text(prefix + '\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], stream_records=stream['records_checked'], kernel=kernel,
                         full_cycle=stats['full_cycle_seconds'], scores=result['scores_percent']), indent=2))


if __name__ == '__main__':
    main()
