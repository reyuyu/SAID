"""Read-only retry gates: real501 text metadata, code/data hashes, immutable failure evidence."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess

from recovery import said_e2_hierarchy090_full4868 as run
from recovery.s02_nfs500 import sha, rows

FAILED_COMMIT='12322b89b24a1c2bf01d030e43fdd39bf1ef6c34'
OLD_RUN=run.PROJECT/'runtime/SAID-nest-clip-v1/said-e2-hierarchy090-full4868-v1'
OLD_EXP=run.ROOT/'experiments/nest_clip_v1/said_e2_hierarchy090_full4868_v1'
CONTROLLER='recovery/said_e2_hierarchy090_full4868.py'


@functools.lru_cache(maxsize=1)
def history501():
    import numpy as np
    import torch
    from train.nested_semantic_data import collate, sampled_text_views
    from recovery.nested_d3_local_search import observe_selection
    torch.set_num_threads(4)
    records=run.protocol.common.read(OLD_RUN/'step4868/resume-reference.json')['records']
    offsets=np.load(run.protocol.local.INDEX/'offsets.npy',mmap_mode='r')
    expected=run.reference_rows()[0]
    output=[]
    with (run.protocol.local.INDEX/'records.jsonl').open('rb') as handle:
        for rank in range(4):
            ref=next(h for h in expected['rank_health'] if h['rank']==rank)
            samples=[]
            for sid in records[f'501:{rank}']['sample_ids']:
                idx=sid-1000;handle.seek(int(offsets[idx]))
                row=json.loads(handle.read(int(offsets[idx+1]-offsets[idx])))
                text=sampled_text_views(row['caption'],'nested_detail_d3',0,0,sid)
                samples.append(dict(text,sample_id=sid,image_id=0,image=torch.empty(0)))
            batch=collate(samples);actual=observe_selection(batch)
            old=run.protocol.common.read(OLD_RUN/f'step4868/batch-501-rank{rank}.json')
            assert old['passed'] and run.stream(batch)==old['stream_sha256']==ref['stream_sha256']
            assert actual!=ref['sampling'] and run.normalize_json(actual)==run.normalize_json(ref['sampling'])
            output.append(dict(rank=rank,step=501,passed=True,actual_sampling=actual,reference_sampling=ref['sampling'],
                stream_sha256=run.stream(batch),raw_equality=False,normalized_full_equality=True,
                collision_check_passed=True,CPU_text_only=True,optimizer_created=False,images_decoded=False))
    return output


def failure_snapshot():
    files=[p for directory in (OLD_EXP,OLD_RUN) for p in directory.rglob('*') if p.is_file()]
    assert files and run.protocol.common.read(OLD_EXP/'STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
    return [dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]


def immutable_snapshot(files):
    assert all(Path(v['path']).stat().st_size==v['size_bytes'] and sha(v['path'])==v['sha256'] for v in files)
    return dict(passed=True,files_checked=len(files),all_old_failure_evidence_unchanged=True)


def provenance():
    from train.train_nested_semantic_mask import code_manifest
    from recovery.s02_full_stage import MANIFEST_SHA
    history=history501()
    native=code_manifest()
    original=run.protocol.common.read(OLD_EXP/'RESUME_PROVENANCE.json')['sources']
    assert native==original
    for name,digest in native.items():
        for commit in (FAILED_COMMIT,run.PARENT_COMMIT):
            assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=run.ROOT)).hexdigest()==digest
    blob=subprocess.check_output(['git','show',FAILED_COMMIT+':'+CONTROLLER],cwd=run.ROOT)
    old_sha=hashlib.sha256(blob).hexdigest()
    assert old_sha==run.protocol.common.read(OLD_EXP/'FAILURE_AUDIT.json')['original_failed_controller_sha256']
    diff=subprocess.check_output(['git','diff',FAILED_COMMIT,'--',CONTROLLER],cwd=run.ROOT,text=True)
    assert diff and run.PARENT_SHA==sha(run.PARENT)
    meta=run.protocol.common.read(run.protocol.local.INDEX/'metadata.json')
    records=run.protocol.local.INDEX/'records.jsonl'
    assert sha(records)==meta['records_sha256']=='0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8'
    assert sha(meta['annotation'])==meta['annotation_sha256']=='5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c'
    ready=run.protocol.common.read(run.protocol.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    manifest=Path(ready['manifest']['path'])
    assert sha(manifest)==ready['manifest']['sha256']==MANIFEST_SHA
    known=run.protocol.common.read(run.ROOT/'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1/posthoc_audit/FULL_STREAM_1218_4868_PROOF.json')
    known_hash={v['actual_log']:v['actual_log_sha256'] for v in known['segments']}
    # The500->1217 reference is anchored in its original source report.
    suffix=run.protocol.common.read(run.ROOT/'experiments/nest_clip_v1/hns_s12_uniform_e1_1217_v1/step1217/SAMPLING_AND_LR_PROOF.json')
    logs=[]
    for index,path in enumerate(run.REFERENCE_LOGS):
        digest=sha(path)
        if index:assert digest==known_hash[str(path)]
        else:
            actual=rows(path);assert [r['step'] for r in actual]==list(range(501,1218))
            assert suffix['passed'] and suffix['updates']==717
        logs.append(dict(path=str(path),sha256=digest,independent_posthoc_hash_match=index>0,
            prefix_reference_identity='Original completed717-update report' if not index else None))
    return dict(passed=True,failed_commit=FAILED_COMMIT,failed_controller_sha256=old_sha,
        repaired_controller_sha256=sha(run.ROOT/CONTROLLER),exact_controller_diff=diff,
        production_source_sha256=native,production_source_changes=[],
        other_comparison_review={'observe':'Normalize complete dictionaries with collision rejection',
            'check_row_sampling':'Both operands already deserialize from JSON; strict full equality unchanged',
            'LR_and_macro_config':'String-keyed mappings; no representation mismatch',
            'model_optimizer_RNG_restore':'Native tensor/state equality preserved'},
        historical501_fourrank=history,reference_logs=logs,
        dataset=dict(annotation_sha256=meta['annotation_sha256'],records_sha256=meta['records_sha256'],
            offsets_sha256=sha(run.protocol.local.INDEX/'offsets.npy'),manifest_sha256=MANIFEST_SHA,
            local_image_root=str(run.protocol.local.IMAGES),prior_complete_local_verification=ready['verification'],
            image_collection_not_rehashed_in_full=True,NFS_fallback=False),
        failure_evidence_snapshot=failure_snapshot(),
        restart_from_completed500=True,reexecute501_authorized=True,restart_from0=False,
        retry_attempts_allowed=1,automatic_further_retry=False)
