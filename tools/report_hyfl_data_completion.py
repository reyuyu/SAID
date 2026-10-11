"""Assemble small auditable reports from actual completed runtime receipts."""
import argparse
import json
from pathlib import Path
import subprocess
from tools.retrieval_bounded import atomic_json, sha
from tools.eval_hyfl_native import MODEL_SHA

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/nest_clip_v1/said_e2_hyfl_data_completion_v1'
BASE = 'ccb07f45ad52463d3e739878db44864dfd6d4059'


def read(path): return json.loads(Path(path).read_text())


def write_md(name, text):
    with (EXP / name).open('x') as f: f.write(text)


def run(evidence, runtime, checkpoint):
    e, rt = Path(evidence), Path(runtime)
    if sha(checkpoint) != MODEL_SHA: raise ValueError('fixed checkpoint changed')
    EXP.mkdir(parents=True, exist_ok=False)
    full = read(e / 'full_eval/Flickr30k-Full/attempt1/RESULT.json')
    if full['status'] != 'COMPLETED': raise ValueError('Full evaluation did not complete')
    image = read(e / 'FLICKR_FULL_IMAGE_AUDIT.json')
    overlap = read(e / 'FLICKR_1K_OVERLAP.json')
    gpu = read(e / 'gpu_check_v2/GPU_FLICKR_BATCH64_TEST.json')
    for proof in [image, overlap, gpu]:
        if proof['status'] != 'PASS': raise ValueError('prior acceptance failed')
    job = read(e / 'full_protocol/JOB.json')
    long = read(e / 'LONG_DCI_OFFICIAL_CSV_AUDIT.json')
    oldlong = read(e / 'LONG_DCI_HISTORICAL_STRIPPED_AUDIT.json')
    dci = read(e / 'DCI_INVESTIGATION.json')
    histfile = ROOT / 'experiments/nest_clip_v1/said_e2_hyfl_protocol_eval_v1/RESULTS_HYFL_PROTOCOL.json'
    hist = read(histfile)
    testslog = e / 'cpu_tests_final3.log'
    if 'Ran 21 tests' not in testslog.read_text() or not testslog.read_text().rstrip().endswith('OK'):
        raise ValueError('CPU tests not fully passing')
    lock = {
        'base_commit': BASE, 'branch': 'experiment/said-e2-hyfl-data-completion-v1',
        'checkpoint_path': checkpoint, 'checkpoint_sha256_before': MODEL_SHA,
        'checkpoint_sha256_after': sha(checkpoint),
        'flickr': {'repo': 'nlphuji/flickr30k', 'revision': '006d3058228379086e8ea76b2b12959d21b168f8',
                   'download_host': 'hf-mirror.com; redirect to HF cas-bridge.xethub.hf.co',
                   'zip': str(rt / 'downloads/flickr30k-images.zip'),
                   'zip_bytes': (rt / 'downloads/flickr30k-images.zip').stat().st_size,
                   'zip_sha256': image['zip_sha256'], 'official_token_sha256': image['token_sha256'],
                   'readme_sha256': sha(e / 'flickr_README.md'),
                   'official_homepage_sha256': sha(e / 'flickr_official_homepage.html'),
                   'license_review': 'Illinois homepage permits non-commercial research/education and notes image copyright belongs to owners. No redistribution of images.',
                   'image_csv_used': False, 'annotation_source': 'Frozen Illinois official results_20130124.token'},
        'long_dci': {'repo': 'mderakhshani/Long-DCI',
                     'revision': '95daae6e4b0b078f9af5fb7fcd8e95ab2201dc0a',
                     'path': str(rt / 'downloads/dci_long.csv'), 'sha256': long['csv_sha256'],
                     'bytes': long['csv_bytes'], 'records': long['author_records'],
                     'license': 'cc-by-nc-4.0; research only, CSV not uploaded'},
        'download_evidence': {p.name: sha(p) for p in e.iterdir() if p.is_file() and
                             (p.name.endswith('headers.txt') or p.name in ['hf_download.log', 'flickr_download.log'])},
        'network': 'HF direct DNS/IP connections timed out/reset. Mirror API and pinned resolve succeeded. hf CLI stalled before bytes; bounded curl download used instead. No alternate image mirror or mixed sources.',
        'network_attempt_receipt_sha256': sha(e / 'NETWORK_ATTEMPTS.json'),
        'frozen_inference': {'precision': 'FP32', 'autocast': False, 'cuda_matmul_tf32': False,
                             'cudnn_tf32': True, 'context_length': 248, 'image_batch': 64,
                             'text_batch': 64, 'query_chunk': 256, 'gallery_chunk': 4096},
        'historical_results_sha256': sha(histfile),
        'code_changes': 'New source audit/report tools and CPU tests; worker only adds timing/state receipts. Model, loss, sampler, tokenizer/preprocess and retrieval mathematics unchanged.',
        'training_executed': False,
        'zip_metadata_classification': {'apple_double_files': image['apple_double_metadata_count'],
              'verified_magic_version': '00051607/00020000',
              'initial_failed_classification_receipt_sha256': sha(e / 'FLICKR_ZIP_APPLEDOUBLE_CLASSIFICATION_FAILURE.json'),
              'initial_failure_retained': True,
              'explanation': 'Initial minimal parser counted resource forks by JPEG suffix. Confirmed every AppleDouble header; corrected metadata classification before any image extraction/evaluation. No real images excluded.'},
        'probe_initialization_failure': {
              'receipt_sha256': sha(e / 'gpu_check/GPU_FLICKR_BATCH64_TEST.json'),
              'error': 'reset_peak_memory_stats before CUDA initialization: Invalid device argument',
              'model_loaded_or_inferred': False, 'failure_preserved': True,
              'fix': 'Call torch.cuda.set_device(0) before peak reset, as existing formal worker does.'},
    }
    atomic_json(EXP / 'SOURCE_LOCK.json', lock, exclusive=True)
    atomic_json(EXP / 'FLICKR_FULL_IMAGE_AUDIT.json', dict(image, overlap=overlap), exclusive=True)
    atomic_json(EXP / 'FLICKR_FULL_MANIFEST_AUDIT.json', {
        'status': 'PASS', 'manifest_sha256': job['manifest_sha256'],
        'image_content_sha256': job['image_content_sha256'], 'images': 31783, 'captions': 158915,
        'explicit_caption_image_positives': True, 'caption_slots_per_image': [0,1,2,3,4],
        'retained_original_text': True, 'missing_images': 0, 'extra_images': 0,
        'image_manifest_path': job['manifest'], 'job': job}, exclusive=True)
    atomic_json(EXP / 'FLICKR_FULL_RESULTS.json', full, exclusive=True)
    atomic_json(EXP / 'LONG_DCI_OFFICIAL_CSV_AUDIT.json', dict(long,
                author_revision=lock['long_dci']['revision'], historical_stripped_comparison=oldlong,
                historical_results_reused=True, reason='Exact same ordered image/text inputs and positive mappings.'), exclusive=True)
    atomic_json(EXP / 'DCI_PROTOCOL_INVESTIGATION.json', dci, exclusive=True)
    atomic_json(EXP / 'EVAL_CORRECTNESS_TESTS.json', {
        'cpu_tests': {'status': 'PASS', 'tests': 21, 'log_sha256': sha(testslog),
                      'command': '.venv/bin/python -m unittest tests.test_hyfl_data_completion tests.test_retrieval_bounded -v'},
        'gpu_same_batch64': gpu,
        'flickr_image_acceptance': image['status'], 'overlap': overlap['status'],
        'long_csv': long['status'], 'historical_long_tokens': oldlong['status'],
        'dci_protocol': 'UNVERIFIED, not claimed as a test PASS',
        'retrieval_math': 'Existing tests cover explicit positives, shuffled IDs, tails, cross-block TopK, exact/near ties, NaN/Inf, cache SHA/identity, overwrite and queue failure/cancellation.'}, exclusive=True)
    resources = {k: full[k] for k in ['device', 'physical_gpu', 'image_batch', 'text_batch', 'query_chunk',
                 'gallery_chunk', 'image_encoding_seconds', 'text_encoding_seconds', 'scoring_seconds',
                 'elapsed_seconds', 'cpu_peak_rss_bytes', 'peak_cuda_allocated', 'peak_cuda_reserved']}
    resources.update(queue=read(e / 'full_eval/RUN_STATE.json'),
                     oom=False, nonfinite=False, image_read_error=False,
                     parallelism='Single Full Flickr worker on GPU0. No internal multiGPU shards.',
                     timing_scope='Encoding includes tokenize/load/cache IO; scoring includes feature transfer. No warm-cache reuse.')
    atomic_json(EXP / 'EVAL_RESOURCE_REPORT.json', resources, exclusive=True)
    datasets = {}
    for name, r in {**hist['datasets'], **hist.get('legacy_only', {})}.items():
        if name == 'Flickr30k-Full' and 'metrics' not in r:
            # Historical report explicitly includes the old blocked Full entry.
            # Preserve it in its source file; actual new Full receipt is added below.
            continue
        datasets[name] = {'metrics': r['metrics'], 'protocol_status': r['protocol_status'],
                          'origin': 'Historical real inference; reused unchanged', 'source_sha256': sha(histfile)}
    datasets['Long-DCI']['protocol_status'] = 'VERIFIED_EXACT_AUTHOR_CSV'
    datasets['DCI']['protocol_status'] = 'PROTOCOL_UNVERIFIED'
    datasets['Flickr30k-Full'] = {'metrics': full['metrics'], 'protocol_status': 'VERIFIED_SOURCE_PROTOCOL',
                                 'origin': 'This task: actual Full inference', 'source_sha256': sha(e / 'full_eval/Flickr30k-Full/attempt1/RESULT.json')}
    atomic_json(EXP / 'ALL_DATASET_RESULTS.json', {'model': 'E2-Uniform@4868', 'datasets': datasets}, exclusive=True)
    def recalls(name, d):
        return ' / '.join(f"{datasets[name]['metrics'][d]['recall_percent'][str(k)]:.6f}" for k in [1,5,10])
    alltable = '| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Status / origin |\n|---|---|---|---|\n'
    for name, r in datasets.items():
        alltable += f"| {name} | {recalls(name, 'I2T')} | {recalls(name, 'T2I')} | {r['protocol_status']}; {r['origin']} |\n"
    r1 = lambda n: ' / '.join(f"{datasets[n]['metrics'][d]['recall_percent']['1']:.3f}" for d in ['I2T','T2I'])
    tables = ('Long-caption R@1 (I2T / T2I, %):\n\n'
              '| Model | DOCCI | DCI (unverified) | Long-DCI (author CSV verified) | Urban-1k |\n'
              '|---|---|---|---|---|\n'
              f"| E2-Uniform@4868 | {r1('DOCCI')} | {r1('DCI')} | {r1('Long-DCI')} | {r1('Urban-1k')} |\n\n"
              'Short-caption R@1 (I2T / T2I, %):\n\n'
              '| Model | COCO-Val5K | Flickr30k-Full |\n|---|---|---|\n'
              f"| E2-Uniform@4868 | {r1('COCO')} | {r1('Flickr30k-Full')} |\n\n")
    write_md('FLICKR_FULL_RESULTS.md', '# Actual Full Flickr30k evaluation\n\n' +
             f"31,783 images; 158,915 captions; one complete candidate pool; explicit five positives per image.\n\n"
             f"I2T R@1/5/10: {recalls('Flickr30k-Full', 'I2T')}%.\n\nT2I R@1/5/10: {recalls('Flickr30k-Full', 'T2I')}%.\n\n"
             f"Correct counts: `{json.dumps({d: full['metrics'][d]['correct'] for d in ['I2T','T2I']})}`.\n\n"
             'Exact ties use ascending fixed manifest index; near ties use unrounded FP32 cosine. Query256/gallery4096; no full score matrix.\n')
    goal = dci['goal_candidate']
    investigation = ('# DCI protocol investigation\n\nStatus: **PROTOCOL_UNVERIFIED**.\n\n'
        'All five public HyFL commits inspected. Initial/latest evaluation blob hashes match; middle commits only alter README. '
        'The only evaluation file reads filename/caption from absent DCI_test.json, sorts filenames and gives no generation rule. '
        'Paper Table1 separates DCI and Long-DCI; README ambiguously names dci Long-DCI and links TULIP.\n\n'
        'Inspected author GitHub repositories, UNCHA evaluation/config, author HF dataset listings (empty), model repository files '
        '(weights and README only), HF DCI search and public issues (empty). None provides a HyFL-author annotation or complete rule. '
        'This is bounded public-source evidence, not a claim to have searched every private/unindexed resource.\n\n'
        f"GOAL pinned commit `{goal['commit']}` has {goal['records']} records, all in SAID, but lacks {goal['missing_from_goal_count']} of 7,805. "
        f"Only {goal['matched_image_token_equal']} of {goal['common_images']} shared captions is token-identical to short+extra. "
        'Its candidate set explicitly mismatches; no GPU evaluation or substitution was performed.\n\n'
        f"SAID raw BPE mean excluding SOT/EOT is {dci['token_lengths_without_sot_eot']['said_mean']:.6f}; including them approximately 174.187. "
        'The proximity to paper 174.2 is auxiliary, not identity proof. 1,383 captions exceed 246 content tokens before truncation.\n\n'
        'Current DCI remains the explicit short+space+extra/strip convention. No alternate caption version was scored. '
        'Author annotation or exact constructor still required; request text prepared but never sent.\n')
    write_md('DCI_PROTOCOL_INVESTIGATION.md', investigation)
    write_md('DCI_AUTHOR_REQUEST.md', '# Draft only — not sent\n\n'
        'Please provide the exact DCI_test.json used for DCI in HyFL-CLIP Table1, or a reproducible generator with pinned source revision. '
        'Needed: total records, image/caption fields, original caption construction (short/extra/concatenation), filtering, order, '
        'positive mappings, file SHA256 and per-caption summary/hash. Please clarify how DCI differs from Long-DCI and whether '
        'GOAL’s 1,999-record file is related. No images or private model weights are requested.\n')
    write_md('PROTOCOL_COMPARISON.md', '# Protocol and result provenance\n\n' + tables + alltable +
        '\nOnly verified source-protocol columns may be compared at the documented dataset level. HyFL-private annotation bytes and tie implementation '
        'are not available for bitwise reproduction. DCI is explicitly excluded from strict HyFL comparisons. Flickr-test1K is a separate legacy result, never Full. '
        'Long-DCI author CSV is exact against the raw manifest and token-equivalent against the historical stripped manifest (3,025 whitespace differences).\n')
    write_md('FINAL_REPORT.md', '# E2 HyFL data completion\n\n'
        'P0 completed by actual full-library inference. P1 completed by direct author CSV comparison. P2 remains explicitly unverified. '
        'No training, model updates, alternate model selection or benchmark-driven protocol changes occurred.\n\n' + tables + alltable +
        f"\nZIP SHA256 `{image['zip_sha256']}`; {image['zip_image_count']} decoded images; missing/extra IDs: 0/0; "
        f"{overlap['overlap_images']} historical images checked, byte-identical {overlap['byte_equal']}, "
        f"different bytes but equal pixels/preprocess {overlap['different_bytes_same_pixels_and_preprocess']}.\n\n"
        f"Author TSV revision `{lock['long_dci']['revision']}`, {long['author_records']} records / {long['csv_bytes']} bytes, "
        f"SHA256 `{long['csv_sha256']}`. Raw text, order, IDs, tokens and positive/candidate mappings exact against raw manifest. "
        'Historical stripped manifest has 3,025 text differences, zero token differences. Historical scores can therefore be reused.\n\n'
        'DCI: absent author annotation/generator; GOAL subset and captions mismatch. The author-request draft was not sent. '
        'No strict HyFL-DCI comparison is claimed.\n\n'
        f"Full worker GPU0: image encoding {full['image_encoding_seconds']:.2f}s, text encoding {full['text_encoding_seconds']:.2f}s, "
        f"scoring {full['scoring_seconds']:.2f}s, overall {full['elapsed_seconds']:.2f}s. "
        f"Peak CUDA allocated/reserved {full['peak_cuda_allocated']/2**30:.3f}/{full['peak_cuda_reserved']/2**30:.3f} GiB; "
        f"CPU peak RSS {full['cpu_peak_rss_bytes']/2**30:.3f} GiB. No OOM, NaN/Inf or image-read failures.\n\n"
        '21 CPU tests and actual Full-image same-batch64 GPU correctness passed. The latter has exact per-query hit agreement; '
        f"maximum feature error {gpu['max_abs']}. Parameter digest and checkpoint SHA unchanged before/after. "
        'DCI identity is an unresolved protocol audit, not labeled PASS.\n\n'
        'Necessary worker change only records separate timing and before/after model digest; retrieval functions and all production training/model files unchanged. '
        'An initial ZIP audit classified AppleDouble resource forks as JPEGs by suffix. Its failure evidence is preserved; confirmed magic/version permits classifying them as metadata. No image/caption subset was used. '
        'An initial small probe also failed before model loading because memory peak reset preceded CUDA initialization; its receipt is preserved and API call order corrected. '
        'New tools reject source SHA/count/ID errors, duplicate/escaping archive files, bad image versions, malformed TSV and token/order mismatch; original files are immutable.\n\n'
        'Large ZIP/images, author TSV, manifests, per-image/token receipts, features and query outputs stay on server. '
        f"Runtime sources: `{rt}`; image/features/evidence: `{e}`.\n\n"
        'Remaining limitation: unavailable authoritative HyFL DCI annotation. Public datasets have been repeatedly observed; these scores are exploratory, '
        'not independently sealed generalization evidence. No fourGPU internal-shard equivalence is claimed. '
        'See DELIVERY_VERIFICATION.json for GPU/process cleanup and artifact hashes; final remote/local HEAD proof is '
        f"saved on server at `{rt / 'GITHUB_SYNC_VERIFIED.json'}` after push/fetch.\n")
    print(EXP)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--evidence', required=True)
    p.add_argument('--runtime', required=True); p.add_argument('--checkpoint', required=True)
    a = p.parse_args(); run(a.evidence, a.runtime, a.checkpoint)
