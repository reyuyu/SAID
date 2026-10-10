"""CPU-only diagnosis of the stopped501 sampling receipt, never a restart.

Reconstruct deterministic text metadata from the frozen local index. Images,
models, optimizers and evaluators are not loaded. Preserve the failing runner.
"""
import json
import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

from recovery import said_e2_hierarchy090_full4868 as run
from recovery import nested_d3_local_search as search
from recovery.s02_nfs500 import dump, sha, now
from train.nested_semantic_data import sampled_text_views, collate


def json_normalize(value):
    return json.loads(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False))


def key_type_differences(actual, reference, prefix=''):
    out=[]
    if isinstance(actual,dict) and isinstance(reference,dict):
        for key,value in actual.items():
            name=prefix+'/'+str(key)
            if key not in reference and str(key) in reference:
                out.append(dict(path=name,in_memory_key_type=type(key).__name__,logged_key_type='str',key=str(key)))
                old=reference[str(key)]
            elif key in reference:old=reference[key]
            else:continue
            out.extend(key_type_differences(value,old,name))
    return out


def main():
    torch.set_num_threads(4)
    assert run.protocol.common.read(run.EXP/'STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
    dest=run.EXP/'FAILURE_AUDIT.json'
    assert not dest.exists(), 'Preserve all failure receipts'
    runtime=run.segment(4868)
    reference=run.reference_rows()[0]
    replay=run.protocol.common.read(runtime/'resume-reference.json')['records']
    offsets=np.load(run.protocol.local.INDEX/'offsets.npy',mmap_mode='r')
    proofs=[]
    with (run.protocol.local.INDEX/'records.jsonl').open('rb') as handle:
        for rank in range(4):
            expected=next(h for h in reference['rank_health'] if h['rank']==rank)
            samples=[]
            for sid in replay[f'501:{rank}']['sample_ids']:
                idx=sid-1000;handle.seek(int(offsets[idx]))
                record=json.loads(handle.read(int(offsets[idx+1]-offsets[idx])))
                text=sampled_text_views(record['caption'],'nested_detail_d3',0,0,sid)
                # Empty placeholder is never decoded, forwarded, or audited as an image.
                samples.append(dict(text,sample_id=sid,image_id=0,image=torch.empty(0)))
            batch=collate(samples)
            value=search.observe_selection(batch)
            changes=key_type_differences(value,expected['sampling'])
            pre=run.protocol.common.read(runtime/f'batch-501-rank{rank}.json')
            proof=dict(rank=rank,step=501,CPU_text_only_reconstruction=True,
                live_preupdate_stream_receipt=pre,
                reconstructed_stream_sha256=run.stream(batch),
                raw_python_dictionary_equal=value==expected['sampling'],
                JSON_normalized_dictionary_equal=json_normalize(value)==expected['sampling'],
                changed_key_types=changes)
            assert pre['passed'] and pre['stream_sha256']==run.stream(batch)==expected['stream_sha256']
            assert not proof['raw_python_dictionary_equal'] and proof['JSON_normalized_dictionary_equal'] and changes
            proofs.append(proof)
    dump(run.EXP/'READONLY_SAMPLING_DIAGNOSIS.json',dict(passed=True,ranks=proofs,
        evidence_type='Deterministic CPU reconstruction checked against actual preupdate stream receipts and real E2 logs',
        exact_failed_live_sampling_object_was_not_saved=True,models_loaded=False,optimizer_created=False,
        optimizer_step=False,images_decoded=False,evaluation=False,utc=now()))
    protected=run.protocol.common.read(run.EXP/'PROTECTED_ARTIFACTS.json')
    assert all(sha(v['path'])==v['sha256'] for v in protected['files'])
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    assert not gpu, 'GPU compute process remains'
    restore=run.protocol.common.read(runtime/'RESTORE_AUDIT.json')
    assert restore['passed'] and all(r['parameter_difference']==0 for r in restore['ranks'])
    assert not list((runtime/'training').glob('step*.pt'))
    log=(runtime/'train.log').read_text(errors='replace')
    assert log.count("AssertionError: Full-stage K/detail/token sampling mismatch")==4
    dump(dest,dict(status='ENGINEERING_FAILURE_STOPPED',classification='AUDIT_DICTIONARY_SERIALIZATION_MISMATCH',
        first_failed_update=501,optimizer_update501_executed_in_memory=True,
        optimizer_counters501_inferred_from_passed_preceding_assert=True,
        durable_new_checkpoint=False,last_verified_recoverable_update=500,
        target4868_reached=False,strict_export_run=False,five_set_evaluation_run=False,
        root_cause='In-memory histogram dictionaries have integer keys; historical JSON logs have string keys. Raw dict equality is false while JSON-normalized equality is true.',
        root_cause_scope='New isolated audit controller; production model/loss/data/LR unchanged',
        live_restore_audit=restore,CPU_tests=run.protocol.common.read(run.EXP/'CPU_TESTS.json'),
        readonly_diagnosis='READONLY_SAMPLING_DIAGNOSIS.json',
        protected_artifacts_unchanged=protected,training_log=dict(path=str(runtime/'train.log'),sha256=sha(runtime/'train.log')),
        original_failed_controller_sha256=sha(Path(run.__file__)),automatic_retry=False,
        GPU_compute_processes=[],utc=now()))
    lines=['# E2-Hierarchy090 full4868: engineering stop','',
        'The authorized4868 endpoint was NOT reached. No new native Recall or aggregate result exists; final comparison with E2 is UNAVAILABLE.',
        '', 'Resume from the exact H0.9@500 checkpoint succeeded. All four ranks restored model, fusion adapter, AdamW, CPU/CUDA/Python/NumPy RNG and DataLoader generator exactly, with zero parameter difference. Horizon4868, cursorepoch0/batch500 and next-update501 LR passed. Original checkpoint SHA256 remains `'+run.PARENT_SHA+'`.',
        '', 'The native trainer executed one new optimizer update501. During its post-update sampling telemetry, the new controller asserted equality between the in-memory Python sampling dictionary and a JSON-deserialized historical dictionary. All four ranks raised this assertion. No new recoverable training checkpoint was saved; update501 tensors were discarded on process exit. There is no checkpoint from which update502 can resume.',
        '', 'Root cause: sampling histograms use integer keys in Python, whereas JSON object keys are strings. CPU-only reconstruction for all four step501 batches reproduces the actual pre-update stream hashes and the E2 reference stream. Raw dictionary equality fails; JSON-normalized equality passes exactly for everyrank, including sample IDs, text/token digests, K and selected indices. This supports a representation bug in the audit controller rather than a detected data-stream drift.',
        '', 'Limitation: the failed live sampling dictionaries were not persisted before the assertion. The schema diagnosis therefore combines live pre-update receipts, frozen production code and deterministic text-only reconstruction; it is not a saved post-update sampling receipt. The original failure evidence and controller are preserved unchanged.',
        '', 'CPU regression43 passed before launch, but these tests compared deserialized log dictionaries and missed the in-memory versus JSON key-type boundary. The new read-only diagnostic regression covers this boundary. No OOM, nonfinite-loss/gradient, source drift or checkpoint corruption was observed before the telemetry failure. This does not constitute successful501–505 acceptance or a full4368-update audit.',
        '', 'No restart, coefficient change, second arm, export or public evaluation followed the failure. The original H0.9 and E2 checkpoint/bare artifacts remain hash-identical. All training workers and the supervisor exited; four GPUs have no compute processes. Large raw logs remain server-local.',
        '', 'Original E2@4868 reference: Score5=73.812186642, J_long3=78.193644403, J_long=87.055002253, Short4=67.240, Urban93.9/92.9, Mean93.4. H0.9@4868 values and deltas are UNAVAILABLE; no conclusion about long-text or Urban improvement can be drawn.',
        '', 'FAILURE_AUDIT.json and READONLY_SAMPLING_DIAGNOSIS.json contain the receipts. The failing controller has not been changed or re-launched.']
    (run.EXP/'HIERARCHY090_FULL4868_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='ENGINEERING_FAILURE_STOPPED',raw_equal=False,normalized_equal_all_four=True,GPU_idle=True)))


def finalize():
    dest=run.EXP/'FAILURE_VALIDATION.json'
    assert not dest.exists()
    proof=run.protocol.common.read(run.EXP/'FAILURE_AUDIT.json')
    assert proof['status']=='ENGINEERING_FAILURE_STOPPED'
    runtime=run.segment(4868)
    receipts=[dict(rank=r,
        LR=run.protocol.common.read(runtime/f'lr-501-rank{r}.json'),
        batch=run.protocol.common.read(runtime/f'batch-501-rank{r}.json')) for r in range(4)]
    assert all(v['LR']['passed'] and v['batch']['passed'] for v in receipts)
    command=[str(run.PROJECT/'.venv/bin/python'),'-m','pytest','-q','tests/test_hierarchy090_full_failure_audit.py']
    check=subprocess.run(command,cwd=run.ROOT,capture_output=True,text=True,env=dict(__import__('os').environ,OMP_NUM_THREADS='4'))
    assert check.returncode==0
    dump(dest,dict(engineering_fullrun_passed=False,readonly_diagnosis_passed=True,
        failure_regression_test=dict(passed=True,command=command,returncode=0,summary=check.stdout+check.stderr),
        before_launch_CPU_tests_passed=43,actual_step501_preupdate_receipts=receipts,
        formal_retry=False,final4868_metrics_available=False,utc=now()))
    print(json.dumps(dict(readonly_diagnosis_passed=True,failure_regression_passed=True,training_restarted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.finalize:finalize()
    else:main()
