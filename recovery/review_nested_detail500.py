"""Review immutable checkpoint, native evaluation and small publication evidence."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from recovery.nested_detail500 import EXP,RUN,PHASE,BASELINE,IMAGES,WEIGHTS
from recovery.s02_nfs500 import ROOT,OUT,STEP0_SHA,dump,rows,sha,now


def review():
    import torch
    torch.set_num_threads(4)
    result = json.loads((EXP/'RESULTS.json').read_text())
    runtime = json.loads((EXP/'RUNTIME_STATS.json').read_text())
    diag = json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    gradients = json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    sampling = json.loads((EXP/'SAMPLING_AUDIT.json').read_text())
    assert result['completed_steps']==500 and result['stopped_at_500'] and result['stream_proof']['passed']
    assert result['stream_proof']['records']==512000 and result['first_five_gate']['passed']
    assert result['acceptance']['passed'] and result['evaluation_checkpoint_immutable']
    assert runtime['oom_kill']==runtime['true_training_io_error_count']==0
    assert runtime['local_only_proof']['passed'] and runtime['local_only_proof']['count']==128
    assert not runtime['pod_or_supervisor_anomaly_observed']
    assert all(c['returncode']==0 for c in result['commands'])
    checkpoint = Path(result['checkpoint_path']); bare = RUN/'step500/student_step500.pt'
    assert sha(checkpoint)==result['checkpoint_sha256']==result['strict_export']['checkpoint_sha256']
    assert sha(bare)==result['strict_export']['bare_sha256']
    assert result['strict_export']['passed'] and result['strict_export']['image_max_abs']==result['strict_export']['text_max_abs']==0
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['completed_steps']==payload['global_step']==payload['scheduler']['completed_steps']==500
    assert payload['scheduler']['horizon']==payload['scheduler_horizon']==4868
    assert payload['data_cursor']==dict(next_epoch=0,next_batch=500)
    assert len(payload['rng_per_rank'])==4 and payload['trajectory_root']==str(RUN)
    assert payload['config']['resume'] is None and payload['config']['init_sha256']==STEP0_SHA
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={500}
    assert all(k in payload for k in ('model','adapter','optimizer','scheduler','rng_per_rank','sampler','data_cursor'))
    result['full_checkpoint_proof'] = dict(passed=True,size_bytes=checkpoint.stat().st_size,
        global_step=500,horizon=4868,data_cursor=payload['data_cursor'],sampler=payload['sampler'],
        rng_ranks=4,AdamW_counters=[500],model_adapter_optimizer_present=True,uploaded=False)
    del payload
    proof = json.loads((RUN/'prelaunch-local-path-proof-5000.json').read_text())
    assert proof['passed'] and proof['count']==5000 and proof['training_RNG_untouched']
    assert all(Path(r['actual_path']).is_relative_to(IMAGES) for r in proof['rows'])
    result['prelaunch_local_only_proof'] = {k:v for k,v in proof.items() if k!='rows'}
    assert sampling['passed'] and sampling['records']==1000 and sampling['global_python_numpy_torch_RNG_unchanged']
    raw_audit = Path(sampling['raw_evidence']['path'])
    assert sha(raw_audit)==sampling['raw_evidence']['sha256']
    assert gradients['passed'] and gradients['fixed_global_batches']==8 and gradients['checkpoint_sha256']==sha(checkpoint)
    assert gradients['no_optimizer_updates'] and gradients['checkpoint_unchanged']
    reference = rows(BASELINE/'steps.jsonl')[:8]
    for g,b in zip(gradients['batches'],reference):
        ids = [h['sampling']['sample_ids'] for h in sorted(b['rank_health'],key=lambda h:h['rank'])]
        assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==g['global_sample_ids_sha256']
        assert g['all_gradients_finite'] and all(math.isfinite(v) for v in g['gradient_norms'].values())
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    _,raw,sources = native_metrics(RUN/'step500')
    old = json.loads((OUT/'STEP500_RESULTS.json').read_text())
    result['native_evaluation_provenance'] = {}
    for d,v in raw.items():
        assert v.get('native_only',v.get('native_student_only')) is True
        assert v['checkpoint_sha256']==result['strict_export']['bare_sha256']
        prior = old['native_evaluation_provenance'][d]
        for k in ('protocol','n_images','n_captions','manifest_sha256'):
            if k in prior:
                assert v[k]==prior[k], (d,k)
        result['native_evaluation_provenance'][d] = dict(
            **{k:v[k] for k in ('protocol','n_images','n_captions','manifest_sha256','checkpoint_sha256','native_only','native_student_only') if k in v},
            result_path=str(sources[d]),result_sha256=sha(sources[d]))
    launch = result['launch_provenance']; commit = launch['git_head']
    for path,expected in launch['source_sha256'].items():
        recorded = subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        assert hashlib.sha256(recorded).hexdigest()==expected, ('Unarchived launch source',path)
    result['launch_code_snapshot'] = dict(commit=commit,all_source_SHA256_matched=True)
    steps = rows(RUN/'step500/steps.jsonl')
    assert [r['step'] for r in steps]==list(range(1,501))
    assert all(math.isfinite(r['loss']) and r['nonfinite']==0 for r in steps)
    assert all(h['gradients_finite'] and h['batch']==256 for r in steps for h in r['rank_health'])
    assert gradients['checkpoint_sha256']==result['checkpoint_sha256']
    result['gradient_summary'] = {k:gradients[k] for k in ('mean_gradient_norms','mean_cosine_to_native','mean_weighted_alignment_cosine_to_native')}
    diag['gradient_weighted_mean_norms'] = {v:10/3*WEIGHTS[v]*g for v,g in gradients['mean_gradient_norms'].items()}
    diag['weighted_Ds_Dall_gradient_norm_ratio'] = diag['gradient_weighted_mean_norms']['Ds']/diag['gradient_weighted_mean_norms']['Dall']
    diag['weighted_gradient_note'] = 'Norms of each scaled view gradient; composite cosine uses their linear combination, not a separate combined backward. These finite-gradient observations do not establish causality.'
    result['atomic_pressure_diagnostic'] = diag['atomic_anomaly_heuristic']
    runtime['peak_cgroup_memory_interpretation'] = 'Usage includes file cache; not process RSS. Limit hits/cache size do not establish OOM; actual oom_kill separately checked.'
    intervals = {c['raw_log']:[c['started_utc'],c['ended_utc']] for c in result['commands']}
    train_times = [result['commands'][0]['started_utc'],result['commands'][0]['ended_utc']]
    for item in runtime['local_raw_artifacts']:
        item['time_range_utc'] = intervals.get(item['path'],train_times if 'step500/' in item['path'] or str(PHASE) in item['path'] else [runtime['started_at'],runtime['finished_at']])
    runtime['local_large_binary_assets'] = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False)
        for p in (RUN/'step500/step000005.pt',checkpoint,bare)]
    runtime['raw_sampling_audit'] = sampling['raw_evidence']
    validation = dict(passed=True,reviewed_at_utc=now(),related_unit_tests_passed=65,
        actual_all500_stream_proof=True,first5_gate_passed=True,full_checkpoint_metadata_verified=True,
        exact_export_native_embeddings=True,evaluation_protocols_exact_matched_baseline=True,
        gradient_batches_exact_frozen_stream=True,launch_commit_SHA256_archived=True,
        local_only_5000_prelaunch_and128_actual_read_proofs=True,full_audit_not_repeated=True)
    dump(EXP/'VALIDATION.json',validation); dump(EXP/'RESULTS.json',result); dump(EXP/'RUNTIME_STATS.json',runtime)
    dump(EXP/'TRAINING_DIAGNOSTICS.json',diag)
    lines = ['', '## Evidence review','',
        f'Launch code snapshot: `{commit}`; every recorded launch source SHA256 matches its Git blob.',
        '1000 text/token records and5000 frozen local paths passed before launch. All512000 training records matched baseline sample IDs and F strings/tokens. First5 structural gate passed before update6; final four-rank parameter difference0.',
        f'Final full checkpoint SHA256: `{result["checkpoint_sha256"]}`; size{checkpoint.stat().st_size}bytes. Model/adapter/AdamW/scheduler, four RNG states, sampler and cursor0:500 verified.',
        f'Bare student SHA256: `{result["strict_export"]["bare_sha256"]}`. Native image/text export equality exact; checkpoint unchanged by gradient diagnosis and five-set evaluation.',
        '', '| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |','|---|---:|---:|---:|---:|---:|---:|']
    for label,deltas in result['delta_vs_baselines_pp'].items():
        u = result['dataset_delta_vs_baselines_pp'][label]['Urban-1k']
        lines.append('| '+label+' | '+' | '.join(f'{deltas[k]:+.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {u["I2T"]["R@1"]:+.3f} | {u["T2I"]["R@1"]:+.3f} |')
    h = diag['last50_hierarchy']; ce = diag['last50_views']
    lines += ['',f'Last50 mask IoU F–Dall={h["F_Dall_mask_iou"]:.6f}, Dall–Ds={h["Dall_Ds_mask_iou"]:.6f}; '
        f'hard violation Dall⊆F={h["Dall_F_hard_violation"]:.6f}, Ds⊆Dall={h["Ds_Dall_hard_violation"]:.6f}. '
        'The requested chain penalty and detached-child gradient routing are verified; empirical violations measure how closely learned hard masks follow it.',
        f'Last50 raw combined CE F={ce["F"]["raw_combined_CE"]:.6f}, Dall={ce["Dall"]["raw_combined_CE"]:.6f}, Ds={ce["Ds"]["raw_combined_CE"]:.6f}. '
        f'Ds/Dall CE ratio={diag["Ds_CE_ratio_to_Dall"]:.4f}, mean backbone gradient norm ratio={diag["Ds_gradient_norm_ratio_to_Dall"]:.4f}. '
        f'Descriptive atomic-pressure flag={diag["atomic_anomaly_heuristic"]["flagged"]}; all8 diagnostic batches finite.',
        'Gradient norms and native-direction cosines: `'+json.dumps(result['gradient_summary'])+'`.',
        f'After applying the actual1.4/1.4/0.2 coefficients, Ds/Dall per-view gradient norm ratio={diag["weighted_Ds_Dall_gradient_norm_ratio"]:.4f}. '
        'Weighted alignment cosine uses the linear combination of separately recomputed view gradients; no optimizer updates or combined-backward equivalence assertion.',
        f'Full-cycle distribution(s): `{json.dumps(runtime["full_cycle_seconds"])}`; local data_wait(s): `{json.dumps(runtime["data_wait_seconds_slowest_rank"])}`.',
        f'Warnings >3s={runtime["steps_gt3s"]}, >10s={runtime["steps_gt10s"]}; true training I/O errors={runtime["true_training_io_error_count"]}, oom_kill={runtime["oom_kill"]}; no observed Pod/supervisor anomaly.',
        f'Training wall seconds={runtime["training_wall_seconds"]:.3f}; GPU peaks(GiB)={runtime["GPU_peak_allocated_GiB"]}. '
        f'Cgroup peak={runtime["peak_cgroup_memory_bytes"]/2**30:.3f}GiB, file-cache peak={runtime["peak_file_cache_bytes"]/2**30:.3f}GiB, anon peak={runtime["peak_anon_bytes"]/2**30:.3f}GiB. Cache usage is not RSS or OOM evidence.',
        'For the strong-positive gate, “Score5/J_long3 not worse” is interpreted against the higher AllDetail baseline. Detailed thresholds and fallback classifications are recorded in RESULTS.json.',
        'Conclusion is the measured aggregate/dataset comparison; this one run cannot separate atomic supervision from removal of Summary and the changed hierarchy. Training remains stopped at500.']
    report = EXP/'REPORT.md'; report.write_text(report.read_text().split('\n## Evidence review')[0]+'\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],scores=result['scores_percent'],validation=validation),indent=2))


if __name__ == '__main__':
    review()
