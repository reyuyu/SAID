"""Reproducible read-only preparation, single queue job, and receipt validation."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
import unittest
import torch
from tools import eval_hns_v1_flickrfull as hns
from tools.audit_hyfl_data import image_inventory
from tools.eval_resource_queue import Queue, preflight
from tools.flickr_full_protocol import read_official
from tools.retrieval_bounded import ShardCache, atomic_json, digest, sha
from tools.validate_hyfl_inference import parameter_sha

ROOT = Path(__file__).resolve().parents[1]
BASE = 'e305275c24bb6a80c2e87a674d50239d6a0f22f5'
EXP = ROOT / 'experiments/nest_clip_v1/hns_v1_step3651_flickrfull_v1'
RUN = Path('/root/said_hns_v1_step3651_flickrfull_v1')
HISTORY = Path('/opt/data/private/lklk/SAID/experiments/nest_clip_v1/epoch_curve_eval_v1')
TRAIN = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1/step4868/step003651.pt')
TRAIN_SHA = '83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07'
TOKEN = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e2-hyfl-eval-unified-v1/references/flickr30k_annotations.tar.gz')
E2_RESULT = ROOT / 'experiments/nest_clip_v1/said_e2_hyfl_data_completion_v1/FLICKR_FULL_RESULTS.json'
FIXED_JOB = Path('/root/said_hyfl_data_completion_v1/full_protocol/JOB.json')


def save(name, value):
    atomic_json(EXP / name, value, exclusive=True)


def prepare():
    EXP.mkdir(parents=True, exist_ok=False)
    RUN.mkdir(parents=True, exist_ok=False)
    hns.bind()
    job = json.loads(FIXED_JOB.read_text())
    hns.validate_job(job, hns.CHECKPOINT)
    print('Verify bare and full training checkpoint bytes', flush=True)
    assert sha(hns.CHECKPOINT) == hns.MODEL_SHA
    assert sha(TRAIN) == TRAIN_SHA
    train = torch.load(TRAIN, map_location='cpu', weights_only=False)
    bare = torch.load(hns.CHECKPOINT, map_location='cpu', weights_only=True)
    assert train['completed_steps'] == 3651 and train['scheduler_horizon'] == 4868
    assert train['next_epoch'] == 3 and train['next_batch'] == 0
    assert train['scheduler']['horizon'] == 4868
    counters = sorted({int(v['step']) for v in train['optimizer']['state'].values()})
    assert counters == [3651]
    assert set(bare) == set(train['model'])
    assert all(torch.equal(bare[k], train['model'][k]) for k in bare)
    export = json.loads((HISTORY / 'evidence/HNS_v1/step3651/export-check.json').read_text())
    audit = export['strict_export']
    assert audit['passed'] and audit['strict_load']
    assert audit['image_max_abs'] == audit['text_max_abs'] == 0
    assert audit['bare_sha256'] == hns.MODEL_SHA and audit['checkpoint_sha256'] == TRAIN_SHA
    inventory = json.loads((HISTORY / 'CHECKPOINT_INVENTORY.json').read_text())['models']['HNS_v1']['3651']
    assert inventory['sha256'] == TRAIN_SHA and inventory['update'] == 3651
    model, pre = hns.native.load_model(hns.CHECKPOINT, 'cpu')
    e2 = json.loads(E2_RESULT.read_text())
    assert hns.native.preprocess_identity(pre) == e2['identity']['preprocess']
    model_identity = {
        'model': 'HNS-v1', 'step': 3651, 'epoch': 3, 'scheduler_horizon': 4868,
        'bare_checkpoint': hns.CHECKPOINT, 'bare_sha256': hns.MODEL_SHA,
        'training_checkpoint': str(TRAIN), 'training_sha256': TRAIN_SHA,
        'strict_bare_training_tensor_equality': True, 'state_dict_keys': len(bare),
        'strict_load': True, 'context_length': model.context_length, 'dtype': 'torch.float32',
        'state_sha256': parameter_sha(model), 'optimizer_steps': counters,
        'scheduler': train['scheduler'], 'data_cursor': train['data_cursor'],
        'configuration': train['config'], 'historical_export': export,
        'historical_inventory_sha256': sha(HISTORY / 'CHECKPOINT_INVENTORY.json'),
        'historical_export_sha256': sha(HISTORY / 'evidence/HNS_v1/step3651/export-check.json'),
        'preprocess': hns.native.preprocess_identity(pre),
        'inference': 'Bare image/text encoders only; no mask/fusion/adapter/HNS inference',
    }
    del train, bare, model
    assert sha(hns.CHECKPOINT) == hns.MODEL_SHA and sha(TRAIN) == TRAIN_SHA
    save('MODEL_IDENTITY.json', model_identity)
    print('Verify all official captions and all 31,783 image bytes/decoding', flush=True)
    assert sha(job['manifest']) == job['manifest_sha256']
    rows, images = hns.native.rows_and_images(job['manifest'])
    raw, official = read_official(TOKEN)
    assert hashlib.sha256(raw).hexdigest() == job['annotation_sha256']
    assert rows == official and len(rows) == 158915 and len(images) == 31783
    assert set(Counter(r['positive_image_id'] for r in rows).values()) == {5}
    inv = image_inventory(images, job['image_root'])
    assert digest(inv) == job['image_content_sha256']
    original_inv = json.loads(Path('/root/said_hyfl_data_completion_v1/full_protocol/IMAGE_INVENTORY.json').read_text())
    assert inv == original_inv
    disk_images = {p.name for p in Path(job['image_root']).iterdir() if p.is_file()}
    assert disk_images == {p for _, p in images}
    atomic_json(RUN / 'IMAGE_INVENTORY.json', inv, exclusive=True)
    save('DATA_IDENTITY.json', {
        'status': 'PASS', 'manifest': job['manifest'], 'manifest_sha256': sha(job['manifest']),
        'image_root': job['image_root'], 'image_content_sha256': digest(inv),
        'n_images': len(images), 'n_captions': len(rows), 'captions_per_image': 5,
        'official_token_sha256': job['annotation_sha256'], 'official_rows_exact_match': True,
        'all_image_bytes_and_PIL_verification': True, 'missing_images': 0, 'extra_images': 0,
        'record_sha256': digest(rows), 'image_ids_paths_sha256': digest(images),
        'independent_inventory': str(RUN / 'IMAGE_INVENTORY.json'),
    })
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_retrieval_bounded', 'tests.test_hns_v1_flickrfull'])
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (RUN / 'CPU_TESTS.log').write_text(stream.getvalue())
    save('EVAL_CORRECTNESS_TESTS.json', {
        'status': 'PASS' if tests.wasSuccessful() else 'FAIL', 'tests_run': tests.testsRun,
        'failures': len(tests.failures), 'errors': len(tests.errors), 'skipped': len(tests.skipped),
        'test_log': stream.getvalue(), 'strict_load': True,
        'training_bare_parameter_exact_match': True, 'preprocess_matches_E2': True,
        'context_length': 248, 'image_batch': 64, 'text_batch': 64,
        'source_fingerprint_checks': 'PASS',
    })
    assert tests.wasSuccessful(), 'CPU regression failed; receipts retained'
    resources = preflight(RUN, (0,))
    atomic_json(RUN / 'PREFLIGHT.json', resources, exclusive=True)
    sources = {p: sha(ROOT / p) for p in hns.FROZEN}
    assert sources == hns.FROZEN
    diff = subprocess.check_output(['git', 'diff', BASE, '--', *hns.FROZEN], cwd=ROOT, text=True)
    assert not diff, 'frozen evaluator/encoder/queue changed'
    save('SOURCE_EQUIVALENCE.json', {
        'base_commit': BASE, 'frozen_source_sha256': sources,
        'frozen_files_git_diff': diff, 'worker_byte_identical': True,
        'minimal_entry_sha256': sha(ROOT / hns.ENTRY),
        'change': 'Only in the dedicated process: MODEL_SHA bound to exact HNS SHA; SOURCES appends wrapper',
        'functions_reused': ['load_model', 'identity', 'encode', 'run', 'retrieval'],
        'new_scripts': {p: sha(ROOT / p) for p in [hns.ENTRY, 'tools/audit_hns_v1_flickrfull.py',
                                                      'tests/test_hns_v1_flickrfull.py']},
    })
    output = RUN / 'eval/Flickr30k-Full/attempt1/RESULT.json'
    cache = RUN / 'eval/cache/Flickr30k-Full'
    job['command'] = [sys.executable, '-m', 'tools.eval_hns_v1_flickrfull',
                      '--job', str(RUN / 'JOB.json'), '--checkpoint', hns.CHECKPOINT,
                      '--output', str(output), '--cache-dir', str(cache), '--device', 'cuda:0']
    atomic_json(RUN / 'JOB.json', job, exclusive=True)
    save('EVALUATION_PLAN.json', {'checkpoint': hns.CHECKPOINT, 'jobs': [job], 'blocked': {},
                                'cache_policy': 'new independent HNS-only cache; no E2 feature reuse'})
    print('PREFLIGHT PASS: ready for one Full Flickr evaluation', flush=True)


def execute():
    assert json.loads((EXP / 'EVAL_CORRECTNESS_TESTS.json').read_text())['status'] == 'PASS'
    assert sha(hns.CHECKPOINT) == hns.MODEL_SHA
    hns.bind()
    plan = json.loads((EXP / 'EVALUATION_PLAN.json').read_text())
    assert not (RUN / 'eval/RUN_STATE.json').exists(), 'existing evaluation: do not duplicate'
    return Queue(plan, RUN / 'eval', gpus=(0,)).run()


def report():
    hns.bind()
    queue = json.loads((RUN / 'eval/RUN_STATE.json').read_text())
    assert queue['status'] == 'COMPLETED' and len(queue['attempts']) == 1
    path = Path(queue['jobs']['Flickr30k-Full']['output'])
    result = json.loads(path.read_text())
    assert sha(path) == queue['jobs']['Flickr30k-Full']['output_sha256']
    assert result['status'] == 'COMPLETED' and result['returncode'] == 0
    assert result['checkpoint_sha256'] == hns.MODEL_SHA
    assert result['state_sha256_before'] == result['state_sha256_after']
    assert result['state_sha256_before'] == json.loads((EXP / 'MODEL_IDENTITY.json').read_text())['state_sha256']
    assert sha(hns.CHECKPOINT) == hns.MODEL_SHA and sha(TRAIN) == TRAIN_SHA
    e2 = json.loads(E2_RESULT.read_text())
    assert e2['status'] == 'COMPLETED' and e2['checkpoint_sha256'] == hns.E2_SHA
    ident = result['identity']; eident = e2['identity']
    for key in eident:
        if key not in ('checkpoint_sha256', 'encoder_sources'):
            assert ident[key] == eident[key], 'E2 protocol identity difference: ' + key
    assert {k: ident['encoder_sources'][k] for k in eident['encoder_sources']} == eident['encoder_sources']
    rows, images = hns.native.rows_and_images(ident['manifest_sha256'] and json.loads(FIXED_JOB.read_text())['manifest'])
    cache = ShardCache(RUN / 'eval/cache/Flickr30k-Full', ident)
    features = {}
    for kind, records in [('image', images), ('text', rows)]:
        shape = 0; error = 0.; shards = 0
        for start in range(0, len(records), 64):
            subset = records[start:start + 64]
            ids = [r[0] for r in subset] if kind == 'image' else [r['caption_id'] for r in subset]
            x = cache.read(kind, start, ids)
            assert x is not None and x.dtype == torch.float32 and list(x.shape) == [len(ids), 512]
            assert torch.isfinite(x).all()
            error = max(error, float((x.norm(dim=1) - 1).abs().max()))
            shape += len(x); shards += 1
        assert shape == len(records) and error < 1e-6
        features[kind] = {'shape': [shape, 512], 'dtype': 'torch.float32', 'shards': shards,
                          'tail_batch': len(records) % 64, 'max_normalization_error': error,
                          'all_shard_hashes_ids_and_identity_verified': True}
    detail = torch.load(path.with_suffix('.queries.pt'), map_location='cpu', weights_only=True)
    for d, n in [('I2T', 31783), ('T2I', 158915)]:
        assert list(detail[d]['top_indices'].shape) == [n, 11]
        assert torch.isfinite(detail[d]['top_scores']).all()
        for k in ('1', '5', '10'):
            m = result['metrics'][d]
            assert m['query_count'] == n and len(detail[d]['hits'][k]) == n
            assert sum(detail[d]['hits'][k]) == m['correct'][k]
            assert m['recall_percent'][k] == 100 * m['correct'][k] / n
    save('FLICKR_FULL_RESULTS.json', result)
    stats = {k: result[k] for k in ['device', 'physical_gpu', 'started_utc', 'ended_utc',
                                   'text_encoding_seconds', 'image_encoding_seconds', 'scoring_seconds',
                                   'encoding_seconds', 'elapsed_seconds', 'peak_cuda_allocated',
                                   'peak_cuda_reserved', 'cpu_peak_rss_bytes']}
    stats.update(features=features, queue_wall_seconds=queue['wall_seconds'],
                 OOM=False, NaN_Inf=False, image_read_error=False,
                 unchanged_checkpoint_sha256=sha(hns.CHECKPOINT), unchanged_training_sha256=sha(TRAIN),
                 state_sha256_before=result['state_sha256_before'], state_sha256_after=result['state_sha256_after'],
                 query_evidence=str(path.with_suffix('.queries.pt')),
                 query_evidence_sha256=sha(path.with_suffix('.queries.pt')),
                 result_sha256=sha(path), cache_identity_sha256=digest(ident))
    save('RUNTIME_STATS.json', stats)
    lines = ['# Flickr30k-Full: HNS-v1@3651 vs E2-Uniform@4868', '',
             '| Direction | E2 (%) | HNS (%) | HNS − E2 (pp) | E2 correct | HNS correct | Net correct | Queries |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    comparison = {}
    for d in ('I2T', 'T2I'):
        comparison[d] = {}
        for k in ('1', '5', '10'):
            a = e2['metrics'][d]['recall_percent'][k]; b = result['metrics'][d]['recall_percent'][k]
            ac = e2['metrics'][d]['correct'][k]; bc = result['metrics'][d]['correct'][k]
            n = result['metrics'][d]['query_count']
            comparison[d][k] = {'E2_percent': a, 'HNS_percent': b, 'delta_pp': b-a,
                                'E2_correct': ac, 'HNS_correct': bc, 'net_correct': bc-ac, 'queries': n}
            lines.append(f'| {d} R@{k} | {a:.9f} | {b:.9f} | {b-a:+.9f} | {ac} | {bc} | {bc-ac:+d} | {n} |')
    means = {name: sum(r['metrics'][d]['recall_percent']['1'] for d in ('I2T', 'T2I')) / 2
             for name, r in [('E2', e2), ('HNS', result)]}
    save('COMPARISON.json', {'directional': comparison, 'bidirectional_R1_mean': means,
                             'mean_delta_pp': means['HNS']-means['E2'], 'E2_raw_result_sha256': sha(E2_RESULT)})
    lines += ['', f"Bidirectional R@1 mean: E2 {means['E2']:.9f}%; HNS {means['HNS']:.9f}%; delta {means['HNS']-means['E2']:+.9f} pp.", '',
              'Net correct counts are aggregate differences, not paired correction counts. No significance claim.', '',
              'This is one dataset, one seed, different training steps (3651 vs 4868), and a repeatedly observed public benchmark.',
              'Historical HNS@3651 Score5 73.774418 is below E2@4868 73.812187. Full Flickr alone cannot justify replacing E2.']
    (EXP / 'FLICKR_FULL_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    tests = json.loads((EXP / 'EVAL_CORRECTNESS_TESTS.json').read_text())
    tests.update(formal_inference='PASS', same_frozen_E2_protocol='PASS',
                 all_feature_shards='PASS', query_hit_recount='PASS',
                 checkpoint_and_state_immutable='PASS', NaN_Inf_OOM_image_errors=False)
    atomic_json(EXP / 'EVAL_CORRECTNESS_TESTS.json', tests)
    report_lines = lines + ['', '## Identity and engineering validation', '',
        f'Base commit: `{BASE}`. Dedicated exact-SHA HNS entrypoint reuses unchanged native functions; frozen source diff is empty.',
        f'Bare SHA256: `{hns.MODEL_SHA}`. Training SHA256: `{TRAIN_SHA}`.',
        f"Manifest SHA256: `{ident['manifest_sha256']}`; image inventory digest: `{ident['image_content_sha256']}`.",
        'Strict bare load: ViT-B/16, context 248, FP32; all bare tensors equal original training model. Historical export image/text max abs 0.',
        'All 31,783 images independently verified against frozen bytes and PIL decoding; all 158,915 captions exactly match official token rows. Five explicit positives per image.',
        'Model/input/features FP32, autocast off, cuDNN TF32 True, CUDA matmul TF32 False. Batch64; image tail39, text tail3. Query256/gallery4096; exact ties by ascending manifest candidate index.',
        f"{tests['tests_run']} CPU tests passed; strict checkpoint/source identity, streaming TopK and cache rejection tests passed. Formal shard/query recount and state immutability passed.",
        f"GPU {stats['physical_gpu']} A100 80GB: text {stats['text_encoding_seconds']:.3f}s, image {stats['image_encoding_seconds']:.3f}s, scoring {stats['scoring_seconds']:.3f}s, total {stats['elapsed_seconds']:.3f}s.",
        f"CUDA allocated peak {stats['peak_cuda_allocated']/2**30:.3f} GiB; reserved {stats['peak_cuda_reserved']/2**30:.3f} GiB; CPU RSS peak {stats['cpu_peak_rss_bytes']/2**30:.3f} GiB.",
        'No OOM, non-finite feature/score, or image read error. Weights unchanged before/after. Native worker saves boundary tie diagnostics in results.',
        f"Independent HNS feature banks and query-level top11/hit evidence retained at `{RUN}`; no E2 embedding reuse. Existing E2 result reused without running E2.",
        'No training, optimizer creation/updates, other model evaluation, DCI evaluation, downloads, or inference adjustment.',
        'See DELIVERY_VERIFICATION.json for final Git commit/remote equality and GPU/process cleanup evidence.']
    (EXP / 'FINAL_REPORT.md').write_text('\n'.join(report_lines) + '\n')
    print(json.dumps({'metrics': result['metrics'], 'means': means, 'runtime': stats}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'execute', 'report'])
    a = p.parse_args()
    try:
        if a.phase == 'prepare': prepare()
        elif a.phase == 'execute': return execute()
        else: report()
    except BaseException as error:
        if RUN.exists():
            atomic_json(RUN / (a.phase + '_FAILURE.json'), {
                'phase': a.phase, 'error': repr(error), 'traceback': traceback.format_exc()}, exclusive=True)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
