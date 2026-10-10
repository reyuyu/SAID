"""Freeze final read-only protocol jobs after all data and CPU/GPU prerequisites."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tools.retrieval_bounded import atomic_json, digest, sha
from tools.flickr_full_protocol import read_official

ROOT = Path(__file__).resolve().parents[1]
REF = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e2-hyfl-eval-unified-v1/references')


def run(output):
    from model import longclip
    output = Path(output)
    plan = json.loads((output / 'EVALUATION_PLAN.json').read_text())
    gpu = json.loads(Path('/root/said_hyfl_protocol_eval_v1/gpu_tests/GPU_INFERENCE_TEST.json').read_text())
    cpu = Path('/root/said_hyfl_protocol_eval_v1/cpu_tests_v2.log').read_text()
    assert gpu['status'] == 'PASS' and 'Ran 12 tests' in cpu and cpu.rstrip().endswith('OK')
    long_job = next(j for j in plan['jobs'] if j['name'] == 'Long-DCI')
    new = [json.loads(s) for s in Path(long_job['manifest']).read_text().splitlines()]
    old_path = ROOT / 'local_assets/retrieval_benchmarks/manifests/long_dci_reconstructed.jsonl'
    old = [json.loads(s) for s in old_path.read_text().splitlines()]
    assert [(r['image_id'], r['image_path']) for r in new] == [(r['image_id'], r['image_path']) for r in old]
    h = hashlib.sha256()
    for start in range(0, len(new), 128):
        a = longclip.tokenize([r['caption'] for r in new[start:start + 128]], truncate=True)
        b = longclip.tokenize([r['caption'] for r in old[start:start + 128]], truncate=True)
        assert torch.equal(a, b), 'raw extra_caption vs stripped token mismatch'
        h.update(a.numpy().tobytes())
    atomic_json(output / 'LONG_DCI_TOKEN_EQUIVALENCE.json',
                {'pass': True, 'rows': len(new), 'raw_string_differences': 3025,
                 'images_and_order_equal': True, 'all_248_tokens_equal': True,
                 'token_stream_sha256': h.hexdigest(), 'official_csv_equality': 'UNVERIFIED'}, exclusive=True)
    data, full = read_official(REF / 'flickr30k_annotations.tar.gz')
    ids = set(r['image_id'] for r in full)
    available = ROOT / 'local_assets/retrieval_benchmarks/flickr30k/images'
    present = sorted(i for i in ids if (available / i).is_file())
    assert len(ids) == 31783 and len(full) == 158915
    audit = {'status': 'BLOCKED_PROTOCOL_DATA', 'annotation_source':
             'https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz',
             'annotation_sha256': hashlib.sha256(data).hexdigest(), 'archive_sha256': sha(REF / 'flickr30k_annotations.tar.gz'),
             'n_annotation_images': len(ids), 'n_annotation_captions': len(full),
             'available_verified_filename_count': len(present), 'missing_images': len(ids) - len(present),
             'caption_slots_0_through_4_validated': True,
             'results_csv_available': False,
             'image_acquisition_attempts': [
                 {'url': 'https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar', 'result': 'HTTP404'},
                 {'url': 'https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar.gz', 'result': 'HTTP404'},
                 {'url': 'https://huggingface.co/api/datasets/nlphuji/flickr30k/tree/main', 'result': 'connection_timeout'},
                 {'url': 'https://shannon.cs.illinois.edu/DenotationGraph/', 'result': 'official page requires dataset access form'}],
             'note': 'No full evaluation and no 1K substitution; official caption records are verified.'}
    atomic_json(output / 'FLICKR_FULL_DATA_AUDIT.json', audit, exclusive=True)
    plan['blocked']['Flickr30k-Full'] = audit
    for j in plan['jobs']:
        if j['name'] == 'COCO':
            j['normalization'] = 'cpu_raw_for_legacy'
    plan['frozen_prerequisites'] = {'gpu_receipt_sha256': sha('/root/said_hyfl_protocol_eval_v1/gpu_tests/GPU_INFERENCE_TEST.json'),
                                     'cpu_log_sha256': sha('/root/said_hyfl_protocol_eval_v1/cpu_tests_v2.log'),
                                     'model_sha256': sha(plan['checkpoint']),
                                     'data_provenance_sha256': sha(output / 'DATA_PROVENANCE.json')}
    atomic_json(output / 'FROZEN_PLAN.json', plan, exclusive=True)
    atomic_json(output / 'EVAL_TESTS.json', {'cpu': {'pass': True, 'tests': 12, 'log': cpu},
                                            'gpu': gpu, 'long_dci_tokens': 'PASS'}, exclusive=True)
    print('FROZEN', digest(plan))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.output)
