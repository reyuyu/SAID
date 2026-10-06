"""Check D3 checkpoint, frozen stream, native eval and archived launch evidence."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from recovery.nested_detail_d3_equal500 import (
    RUN,EXP,PHASE,BASELINE,ATOMIC_EXP,MODE,RELATED_TEST_COUNT,matched_rows,update_reports)
from recovery.s02_nfs500 import ROOT,OUT,STEP0_SHA,dump,rows,sha,now
from recovery.s02_full_stage import IMAGES

EXPECTED_WEIGHTS = [1,1,1]
ISOLATION_KEY = 'config_only_sampling_mode_changed'


def main():
    import torch
    torch.set_num_threads(4)
    result=json.loads((EXP/'RESULTS.json').read_text())
    runtime=json.loads((EXP/'RUNTIME_STATS.json').read_text())
    sampling=json.loads((EXP/'SAMPLING_AUDIT.json').read_text())
    gradients=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    assert result['completed_steps']==500 and result['stopped_at_500'] and result['stream_proof']['passed']
    assert result['first_five_gate']['passed'] and result['acceptance']['passed']
    assert all(r['completed_updates']==500 and r['max_parameter_difference_from_rank0']==0 for r in result['acceptance']['ranks'])
    assert result['evaluation_checkpoint_immutable'] and all(c['returncode']==0 for c in result['commands'])
    assert runtime['oom_kill']==runtime['true_training_io_error_count']==0
    assert runtime['local_only_proof']['passed'] and runtime['local_only_proof']['count']==128
    assert not runtime['pod_or_supervisor_anomaly_observed']
    checkpoint=Path(result['checkpoint_path']);bare=RUN/'step500/student_step500.pt'
    assert sha(checkpoint)==result['checkpoint_sha256']==result['strict_export']['checkpoint_sha256']
    assert sha(bare)==result['strict_export']['bare_sha256']
    assert result['strict_export']['passed'] and result['strict_export']['image_max_abs']==result['strict_export']['text_max_abs']==0
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['completed_steps']==payload['global_step']==payload['scheduler']['completed_steps']==500
    assert payload['scheduler_horizon']==payload['scheduler']['horizon']==4868
    assert payload['data_cursor']==dict(next_epoch=0,next_batch=500)
    assert len(payload['rng_per_rank'])==4 and payload['trajectory_root']==str(RUN)
    assert payload['config']['resume'] is None and payload['config']['init_sha256']==STEP0_SHA
    assert payload['config']['sampling_mode']==MODE and payload['config']['view_weights']==EXPECTED_WEIGHTS
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={500}
    assert all(k in payload for k in ('model','adapter','optimizer','scheduler','rng_per_rank','sampler','data_cursor'))
    result['full_checkpoint_proof']=dict(passed=True,size_bytes=checkpoint.stat().st_size,global_step=500,horizon=4868,
        data_cursor=payload['data_cursor'],sampler=payload['sampler'],rng_ranks=4,AdamW_counters=[500],
        model_adapter_optimizer_present=True,uploaded=False)
    del payload
    paths=json.loads((RUN/'prelaunch-local-path-proof-5000.json').read_text())
    assert paths['passed'] and paths['count']==5000 and paths['training_RNG_untouched']
    assert all(Path(r['actual_path']).is_relative_to(IMAGES) for r in paths['rows'])
    result['prelaunch_local_only_proof']={k:v for k,v in paths.items() if k!='rows'}
    assert sampling['passed'] and sampling['records']==1000 and sampling['global_python_numpy_torch_RNG_unchanged']
    assert sha(Path(sampling['raw_evidence']['path']))==sampling['raw_evidence']['sha256']
    steps=rows(RUN/'step500/steps.jsonl')
    assert [r['step'] for r in steps]==list(range(1,501))
    assert matched_rows(steps,rows(BASELINE/'steps.jsonl'))==512000
    assert gradients['passed'] and gradients['fixed_global_batches']==8 and gradients['no_optimizer_updates']
    assert gradients['checkpoint_unchanged'] and gradients['checkpoint_sha256']==sha(checkpoint)
    old_g=json.loads((ATOMIC_EXP/'GRADIENT_SPOTCHECK.json').read_text())
    assert all(gradients[k]==old_g[k] for k in ('fixed_global_batches','global_batch_size','selection','backbone_scope','gradients','native_reference'))
    for g,b,ref in zip(gradients['batches'],steps[:8],old_g['batches']):
        ids=[h['sampling']['sample_ids'] for h in sorted(b['rank_health'],key=lambda h:h['rank'])]
        digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        assert digest==g['global_sample_ids_sha256']==ref['global_sample_ids_sha256']
        assert g['all_gradients_finite'] and all(math.isfinite(v) for v in g['gradient_norms'].values())
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    _,raw,sources=native_metrics(RUN/'step500')
    old=json.loads((OUT/'STEP500_RESULTS.json').read_text())
    result['native_evaluation_provenance']={}
    for dataset,v in raw.items():
        assert v.get('native_only',v.get('native_student_only')) is True
        assert v['checkpoint_sha256']==result['strict_export']['bare_sha256']
        prior=old['native_evaluation_provenance'][dataset]
        for k in ('protocol','n_images','n_captions','manifest_sha256'):
            if k in prior:assert v[k]==prior[k],(dataset,k)
        result['native_evaluation_provenance'][dataset]=dict(
            **{k:v[k] for k in ('protocol','n_images','n_captions','manifest_sha256','checkpoint_sha256','native_only','native_student_only') if k in v},
            result_path=str(sources[dataset]),result_sha256=sha(sources[dataset]))
    launch=result['launch_provenance'];commit=launch['git_head']
    for path,expected in launch['source_sha256'].items():
        blob=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==expected,('Unarchived launch source',path)
        assert sha(ROOT/path)==expected,('Launch code modified',path)
    assert launch['isolation_proof'][ISOLATION_KEY] and launch['isolation_proof']['objective_model_sources_unchanged']
    result['launch_code_snapshot']=dict(commit=commit,all_source_SHA256_matched=True)
    runtime['peak_cgroup_memory_interpretation']='Includes file cache; not RSS. Actual oom_kill checked separately; cache/limit size does not establish OOM.'
    intervals={c['raw_log']:[c['started_utc'],c['ended_utc']] for c in result['commands']}
    train_range=[result['commands'][0]['started_utc'],result['commands'][0]['ended_utc']]
    for item in runtime['local_raw_artifacts']:
        item['time_range_utc']=intervals.get(item['path'],train_range if 'step500/' in item['path'] or str(PHASE) in item['path'] else [runtime['started_at'],runtime['finished_at']])
    runtime['local_large_binary_assets']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False)
        for p in (RUN/'step500/step000005.pt',checkpoint,bare)]
    runtime['raw_sampling_audit']=sampling['raw_evidence']
    dump(EXP/'RESULTS.json',result);dump(EXP/'RUNTIME_STATS.json',runtime)
    update_reports()
    validation=dict(passed=True,reviewed_at_utc=now(),related_unit_tests_passed=RELATED_TEST_COUNT,
        first5_gate_passed=True,all512000_sample_F_Dall_stream_exact=True,D3_strict_subsets_or_unchanged_fallback=True,
        full_checkpoint_metadata_verified=True,exact_export_native_embeddings=True,
        evaluation_protocols_exact_matched_baseline=True,gradient_batches_and_protocol_exact_atomic_equal=True,
        launch_commit_SHA256_archived=True,local_only_5000_prelaunch_and128_actual_read_proofs=True,
        method_change_only_lowest_view_sampling=True,no_full_audit_repeated=True)
    dump(EXP/'VALIDATION.json',validation)
    print(json.dumps(dict(status=result['status'],review_passed=True,scores=result['scores_percent'],validation=validation),indent=2))


if __name__=='__main__':main()
