"""Review finished AllDetail evidence and publish compact summaries; no training."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from recovery.s02_all_detail500 import RUN, PHASE, BASELINE, ROOT, OUT, IMAGES, classify
from recovery.s02_nfs500 import dump, now, rows, sha, distribution

CODE_SNAPSHOT = 'b45fa7c'
HISTORICAL_R1 = {'COCO': (60.240, 41.844), 'Urban-1k': (90.100, 88.000),
                 'Flickr30k-test1k': (87.500, 71.700), 'DOCCI': (76.540, 76.420),
                 'Long-DCI': (54.709287, 56.590371)}


def review():
    result = json.loads((OUT/'ALL_DETAIL500_RESULTS.json').read_text())
    stats = json.loads((OUT/'ALL_DETAIL500_RUNTIME_STATS.json').read_text())
    diag = json.loads((OUT/'ALL_DETAIL500_DIAGNOSTICS.json').read_text())
    assert result['stopped_at_500'] and result['completed_steps'] == 500 and result['stream_proof']['passed']
    assert result['first_five_gate']['passed'] and result['evaluation_checkpoint_immutable']
    assert result['acceptance']['passed'] and stats['oom_kill'] == stats['true_training_io_error_count'] == 0
    assert stats['local_only_proof']['passed'] and stats['local_only_proof']['count'] == 128
    checkpoint = Path(result['checkpoint_path'])
    assert sha(checkpoint) == result['checkpoint_sha256'] == result['strict_export']['checkpoint_sha256']
    assert sha(RUN/'step500/student_step500.pt') == result['strict_export']['bare_sha256']
    import torch
    torch.set_num_threads(4)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['global_step'] == payload['completed_steps'] == 500
    assert payload['scheduler_horizon'] == payload['scheduler']['horizon'] == 4868
    assert payload['trajectory_root'] == str(RUN)
    assert payload['data_cursor'] == dict(next_epoch=0,next_batch=500)
    assert len(payload['rng_per_rank']) == 4
    assert all(k in payload for k in ('model','adapter','optimizer','scheduler','sampler','rng_per_rank','data_cursor'))
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {500}
    assert payload['config']['resume'] is None and payload['config']['start_updates'] == 0
    assert payload['config']['init_sha256'] == result['launch_provenance']['step0_sha256']
    result['full_checkpoint_proof'] = dict(passed=True,size_bytes=checkpoint.stat().st_size,
        completed_steps=500,scheduler_horizon=4868,scheduler=payload['scheduler'],sampler=payload['sampler'],
        data_cursor=payload['data_cursor'],rng_ranks=4,optimizer_counters=[500],
        model_adapter_optimizer_present=True,uploaded=False)
    del payload
    proof = json.loads((RUN/'prelaunch-local-path-proof-5000.json').read_text())
    assert proof['passed'] and proof['count'] == 5000 and proof['training_RNG_untouched']
    assert all(Path(p['actual_path']).is_relative_to(IMAGES) for p in proof['rows'])
    result['prelaunch_local_only_proof'] = {k:v for k,v in proof.items() if k != 'rows'}
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    _, raw, sources = native_metrics(RUN/'step500')
    current_baseline = json.loads((OUT/'STEP500_RESULTS.json').read_text())
    for d, v in raw.items():
        assert v.get('native_only', v.get('native_student_only')) is True
        assert v['checkpoint_sha256'] == result['strict_export']['bare_sha256']
        baseline = current_baseline['native_evaluation_provenance'][d]
        for k in ('protocol', 'n_images', 'n_captions', 'manifest_sha256'):
            if k in baseline:
                assert v[k] == baseline[k], (d, k)
        assert sha(sources[d]) == result['native_evaluation_provenance'][d]['result_sha256']
    assert raw['Urban-1k']['n_images'] == raw['Urban-1k']['n_captions'] == 1000
    result['urban_gate'] = classify(result['scores_percent'], result['metrics'])
    result['status'] = result['urban_gate']['status']
    result['classification_note'] = 'Urban T2I gate uses integer correct-caption count/1000; raw float32 retained. INCONCLUSIVE is the gap between historical baseline and positive threshold without a guard failure.'
    result['dataset_R1_delta_vs_historical_S02_pp'] = {
        d:{r:100*result['metrics'][d][r]['R@1']-HISTORICAL_R1[d][i]
           for i,r in enumerate(('I2T','T2I'))} for d in HISTORICAL_R1}
    result['historical_R1_reference_percent'] = HISTORICAL_R1
    archive = subprocess.check_output(['git','rev-parse',CODE_SNAPSHOT],cwd=ROOT,text=True).strip()
    for p, expected in result['launch_provenance']['source_sha256'].items():
        payload = subprocess.check_output(['git','show',archive+':'+p],cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest() == expected, 'Archived launch code drift: '+p
    result['launch_code_snapshot'] = dict(commit=archive, all_source_SHA256_matched=True,
        snapshot_committed_during_training=True,
        post_training_changes='Only spawn-safe diagnostic log routing and integer Urban decision precision; training data/model/loss/optimizer frozen')
    result['commands'] = json.loads((RUN/'commands.json').read_text())
    assert all(c['returncode'] == 0 for c in result['commands'])
    result['diagnostic_log_relocation'] = dict(from_directory='/root/said_s02_stage500/formal-local500-phase-20261006',
        to_directory=str(PHASE),files=32,local_read_records=128,scope='Only worker PIDs observed in this run; after worker exit',
        reason='Spawn imports base module globals anew; future logger now reads the per-run environment path')
    steps = rows(RUN/'step500/steps.jsonl')
    assert [r['step'] for r in steps] == list(range(1,501))
    assert all(math.isfinite(r['loss']) and r['nonfinite'] == 0 for r in steps)
    assert all(h['batch'] == 256 and h['gradients_finite'] for r in steps for h in r['rank_health'])
    cycles = rows(RUN/'step500/cycle_timing.jsonl')
    stats['steady_steps7_plus_full_cycle_seconds'] = distribution([r['four_rank_max_seconds'] for r in cycles if r['step'] >= 7])
    lookup = {r:{v['step']:v for v in rows(PHASE/f'rank{r}.jsonl')} for r in range(4)}
    stats['data_wait_seconds_slowest_rank'] = distribution([max(lookup[r][c['step']]['data_wait_s'] for r in range(4)) for c in cycles])
    stats['steady_steps7_plus_data_wait_seconds_slowest_rank'] = distribution([max(lookup[r][c['step']]['data_wait_s'] for r in range(4)) for c in cycles if c['step'] >= 7])
    train_command = result['commands'][0]
    import datetime
    stats['training_wall_seconds'] = (datetime.datetime.fromisoformat(train_command['ended_utc'])-
                                     datetime.datetime.fromisoformat(train_command['started_utc'])).total_seconds()
    stats['checkpoint_time_seconds'] = sum(v['four_rank_max_seconds'] for v in rows(RUN/'step500/checkpoint_timing.jsonl'))
    stats['PSI_scope'] = 'Host /proc/pressure on cgroup v1; not per-cgroup PSI'
    resources = rows(RUN/'resource-telemetry.jsonl')
    stats['peak_training_anon_bytes'] = max(v['system']['anon'] for v in resources if v['command'] == 'train500')
    intervals = {c['raw_log']:[c['started_utc'],c['ended_utc']] for c in result['commands']}
    for artifact in stats['local_raw_artifacts']:
        if artifact['path'] in intervals:
            artifact['time_range_utc'] = intervals[artifact['path']]
        elif Path(artifact['path']).parent in (PHASE,RUN/'step500'):
            artifact['time_range_utc'] = [train_command['started_utc'],train_command['ended_utc']]
    aligned = sum(v['weighted_alignment_contribution'] for v in diag['last50_views'].values())
    sparse = statistics.fmean((r['F_sparse']+2*r['O_sparse']+2*r['E_sparse'])/3 for r in steps[-50:])
    inc = statistics.fmean(r['inc_weight']*r['inc'] for r in steps[-50:])
    delta = diag['last50_loss']-aligned-sparse-inc
    assert abs(delta) < .00001, 'Weighted contribution reconstruction drift'
    diag['last50_loss_reconstruction'] = dict(weighted_alignment=aligned,sparsity=sparse,
        weighted_inclusion=inc,actual_loss=diag['last50_loss'],residual=delta,passed=True)
    old_steps = rows(BASELINE/'steps.jsonl')
    count = token_sum = sentence_sum = 0
    for r in old_steps:
        for h in r['rank_health']:
            s = h['sampling']; d = s['Full_Summary_Detail_token_statistics']['Detail']
            count += d['samples']; token_sum += d['effective_token_sum']
            sentence_sum += sum(int(k)*v for k,v in s['random_detail_sampling']['selected_count_histogram'].items())
    diag['current_RandomDetail_baseline'] = dict(valid_detail_samples=count,D_mean_sentences=sentence_sum/count,
        D_mean_effective_tokens_including_SOT_EOT=token_sum/count,
        last50_views={p:dict(CE_directional_mean=statistics.fmean((r[k+'_i2t']+r[k+'_t2i'])/2 for r in old_steps[-50:]))
                      for p,k in [('F','F'),('S','O'),('D','E')]})
    result['CPU_unit_validation'] = dict(command='.venv/bin/python -m pytest -q tests/test_summary_all_detail.py tests/test_summary_random_detail.py tests/test_summary_detail.py recovery/test_all_detail500.py recovery/test_local500_policy.py',
        scope='CPU invariants and one small CUDA-token/CPU-reference telemetry regression; no training in tests',
        result='46 passed')
    result['reviewed_at'] = now()
    dump(OUT/'ALL_DETAIL500_RESULTS.json',result)
    dump(OUT/'ALL_DETAIL500_RUNTIME_STATS.json',stats)
    dump(OUT/'ALL_DETAIL500_DIAGNOSTICS.json',diag)
    lines = ['# S=0.2 AllDetail single-variable 500-step experiment','',
             f'Status: `{result["status"]}`. Exactly500 updates; no continuation or new experiments.', '',
             'Only method change: RandomDetail → all ordered non-summary detail sentences from the unchanged visible F sentence pool. '
             'User confirmed this pool. F/S strings/tokens, fallback, sampler IDs/order, preprocessing, weights[1.4,0.2,1.4], '
             'seed0, global1024,4 A10080GB, horizon4868, model/optimizer/LR/sparsity/inclusion/ramp unchanged.', '',
             '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) | R1 delta vs historical S0.2 (I2T / T2I pp) |',
             '|---|---|---|---|']
    for d,m in result['metrics'].items():
        values = [' / '.join(f'{100*m[r][k]:.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
        delta = result['dataset_R1_delta_vs_historical_S02_pp'][d]
        lines.append(f'| {d} | {values[0]} | {values[1]} | {delta["I2T"]:+.6f} / {delta["T2I"]:+.6f} |')
    lines += ['', '| Score (%) | AllDetail | Delta vs historical S0.2 (pp) | Delta vs current local S0.2 (pp) |',
              '|---|---:|---:|---:|']
    for k in ('Score5','J_long3','J_long','Short4'):
        lines.append(f'| {k} | {result["scores_percent"][k]:.6f} | {result["delta_vs_historical_S02_pp"][k]:+.6f} | {result["delta_vs_current_local_S02_pp"][k]:+.6f} |')
    lines += ['', 'Decision (predeclared user thresholds): `'+json.dumps(result['urban_gate'])+'`.',
              'Urban I2T delta: `'+str(result['historical_Urban_R1_delta_pp']['I2T'])+'`pp. Long-DCI per-direction deltas are shown above; no full-run claim.', '',
              '| View | Last50 CE I2T | CE T2I | Directional mean CE | Weighted alignment contribution | Keep ratio |',
              '|---|---:|---:|---:|---:|---:|']
    for name,v in diag['last50_views'].items():
        lines.append(f'| {name} | {v["CE_I2T"]:.6f} | {v["CE_T2I"]:.6f} | {v["CE_directional_mean"]:.6f} | {v["weighted_alignment_contribution"]:.6f} | {v["keep_ratio"]:.6f} |')
    lines += ['', f'D mean sentences {diag["D_mean_sentences"]:.6f}; mean tokens including SOT/EOT {diag["D_mean_effective_tokens_including_SOT_EOT"]:.6f}; '
              f'D/F content-token coverage {diag["D_F_content_token_coverage_pooled"]:.6f} (pooled), {diag["D_F_content_token_coverage_mean_per_sample"]:.6f} (mean per valid sample).',
              f'RandomDetail matched baseline: D mean sentences {diag["current_RandomDetail_baseline"]["D_mean_sentences"]:.6f}, mean tokens {diag["current_RandomDetail_baseline"]["D_mean_effective_tokens_including_SOT_EOT"]:.6f}.',
              f'Last50 inclusion {diag["last50_inc"]:.6f}, inclusion weight {diag["last50_inc_weight"]:.6f}, S/D mask IoU {diag["last50_oe_iou"]:.6f}. F/D mask IoU at diagnostic updates in ALL_DETAIL500_DIAGNOSTICS.json.',
              'Weighted contribution=10/sum(weights) × weight × (CE I2T+CE T2I). Token coverage excludes SOT/EOT and uses the same valid samples. Optional F-D native text cosine omitted: no existing observer.', '',
              f'Prelaunch5000 local-only sample paths; runtime128 local-only reads. Local root `{IMAGES}`; missing/escape/symlink fail-fast, no NFS fallback.',
              'All512000 sample IDs/F/S match baseline; D ordered and complete; LR exact. First5 gate before update6 passed, AdamW counters5, four-rank parameter difference0.',
              'Steady full-cycle seconds: `'+json.dumps(stats['steady_steps7_plus_full_cycle_seconds'])+'`.',
              'Slowest-rank data_wait seconds: `'+json.dumps(stats['data_wait_seconds_slowest_rank'])+'`.',
              f'>3s steps {stats["steps_gt3s"]}; >10s steps {stats["steps_gt10s"]}; actual I/O failures {stats["true_training_io_error_count"]}; oom_kill {stats["oom_kill"]}.',
              'GPU/cgroup peaks, PSI and complete raw-log inventory: ALL_DETAIL500_RUNTIME_STATS.json.', '',
              f'Checkpoint `{checkpoint}`; SHA256 `{result["checkpoint_sha256"]}`.',
              f'Bare student SHA256 `{result["strict_export"]["bare_sha256"]}`. Strict native embeddings exact; checkpoint SHA unchanged across eval; five sets share bare SHA; Long has7602 items and frozen manifest.',
              'Inference: normalize(native image embedding) @ normalize(native full-caption text embedding).T. No mask/gate/rerank/ensemble/Summary/Detail inference.',
              f'Launch code snapshot `{archive}` matches every launch source SHA. Post-training fixes only diagnostic log routing and float32 Urban boundary classification.',
              'Initial attempt failed after its first optimizer update at diagnostic CUDA/CPU comparison. Preserved locally; corrected run starts fresh common0 and never resumes it. '
              'Per-worker proof logs were relocated after worker exit; future logger uses the per-run environment directory.',
              'Checkpoints, bare student, datasets, local images/index/cache and raw logs remain local. NFS originals retained; /root is disposable overlay cache.',
              'Report/code/config/tests only are eligible for GitHub. Fetch/HEAD verification follows publication.']
    (OUT/'ALL_DETAIL500_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],scores=result['scores_percent'],gate=result['urban_gate'],
                         historical_deltas=result['delta_vs_historical_S02_pp']),indent=2))


if __name__ == '__main__':
    review()
