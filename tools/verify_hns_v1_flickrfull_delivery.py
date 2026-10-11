"""Validate completed results, Git equality and idle GPUs; no model execution."""
import argparse
import json
from pathlib import Path
import subprocess
from tools.audit_hns_v1_flickrfull import BASE, E2_RESULT, EXP, ROOT, RUN, TRAIN, TRAIN_SHA
from tools import eval_hns_v1_flickrfull as hns
from tools.eval_five_parallel import require_gpu_idle
from tools.retrieval_bounded import atomic_json, sha

BRANCH = 'experiment/hns-v1-step3651-flickrfull-eval-v1'
PERSIST = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-v1-step3651-flickrfull-eval-v1')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def verify(final):
    hns.bind()
    assert git('branch', '--show-current') == BRANCH
    local = git('rev-parse', 'HEAD')
    remote = git('rev-parse', 'refs/remotes/origin/' + BRANCH)
    assert local == remote, 'fetch and verify origin HEAD before delivery'
    assert not git('diff', BASE, '--', *hns.FROZEN)
    sources = json.loads((EXP / 'SOURCE_EQUIVALENCE.json').read_text())
    for name, expected in sources['new_scripts'].items():
        assert sha(ROOT / name) == expected
    tests = json.loads((EXP / 'EVAL_CORRECTNESS_TESTS.json').read_text())
    assert tests['status'] == 'PASS' and tests['tests_run'] == 18
    assert tests['formal_inference'] == tests['same_frozen_E2_protocol'] == 'PASS'
    result = json.loads((EXP / 'FLICKR_FULL_RESULTS.json').read_text())
    assert result['status'] == 'COMPLETED'
    assert result['state_sha256_before'] == result['state_sha256_after']
    assert sha(hns.CHECKPOINT) == hns.MODEL_SHA and sha(TRAIN) == TRAIN_SHA
    comparison = json.loads((EXP / 'COMPARISON.json').read_text())
    assert sha(E2_RESULT) == comparison['E2_raw_result_sha256']
    assert not git('diff', BASE, '--', str(E2_RESULT.relative_to(ROOT)))
    require_gpu_idle((0, 1, 2, 3))
    queue = json.loads((RUN / 'eval/RUN_STATE.json').read_text())
    assert queue['status'] == 'COMPLETED' and len(queue['attempts']) == 1
    assert queue['attempts'][0]['returncode'] == 0
    assert not Path('/proc', str(queue['attempts'][0]['pid'])).exists()
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True).strip()
    assert not compute, 'GPU compute process remains'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                   '--format=csv'], text=True)
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    # The current verifier and shell are evidence writers only, never model jobs.
    forbidden = [s for s in processes.splitlines() if (' -m tools.eval_hns_v1_flickrfull ' in s or
                  ' -m tools.audit_hns_v1_flickrfull execute' in s) and 'ps -eo' not in s]
    assert not forbidden, forbidden
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(EXP.iterdir()) if p.is_file()}
    value = {
        'status': 'PASS', 'branch': BRANCH, 'verified_commit': local, 'remote_tracking_head': remote,
        'local_remote_head_equal_after_fetch': True, 'base_commit': BASE,
        'bare_sha256': hns.MODEL_SHA, 'training_sha256': TRAIN_SHA,
        'original_E2_result_unchanged': True, 'original_encoder_worker_queue_unchanged': True,
        'tests_run': 18, 'all_tests_and_formal_checks_passed': True,
        'all_four_gpus_idle': True, 'gpu_listing': gpu,
        'GPU_compute_processes': compute, 'owned_worker_pid': queue['attempts'][0]['pid'],
        'owned_worker_exited': True, 'remaining_model_jobs': forbidden,
        'small_artifact_sha256': manifest,
        'large_cache_and_queries': str(RUN),
        'final_commit_sync_proof': str(PERSIST / 'GITHUB_SYNC_VERIFIED.json'),
    }
    PERSIST.mkdir(parents=True, exist_ok=True)
    target = PERSIST / 'GITHUB_SYNC_VERIFIED.json' if final else EXP / 'DELIVERY_VERIFICATION.json'
    if final:
        assert not git('diff', '--exit-code') and not git('diff', '--cached', '--exit-code')
    atomic_json(target, value, exclusive=True)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final', action='store_true')
    a = p.parse_args(); verify(a.final)
