"""Assemble evidence from real worker receipts; missing datasets remain missing."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
from tools.retrieval_bounded import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
REF = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e2-hyfl-eval-unified-v1/references')
ORDER = ['DOCCI', 'DCI', 'Long-DCI', 'Urban-1k', 'COCO', 'Flickr30k-Full']
BASE = '6dcae270f8e754323512648e1e09ed903ab698fc'


def md(path, text):
    with Path(path).open('x', encoding='utf8') as f:
        f.write(text.rstrip() + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def run(output, runtime):
    out, runtime = Path(output), Path(runtime)
    plan_name = 'FROZEN_NATIVE_PLAN.json' if (out / 'FROZEN_NATIVE_PLAN.json').exists() else 'FROZEN_PLAN_V2.json'
    plan = json.loads((out / plan_name).read_text())
    state = json.loads((runtime / 'RUN_STATE.json').read_text())
    data = json.loads((out / 'DATA_PROVENANCE.json').read_text())
    tests = json.loads((out / 'EVAL_TESTS.json').read_text())
    native_error = max(r['max_abs'] for r in tests['gpu']['records'])
    actual = {}
    for name, r in state['jobs'].items():
        p = Path(r['output'])
        if r['status'] == 'COMPLETED':
            assert sha(p) == r['output_sha256']
            result = json.loads(p.read_text()); assert result['status'] == 'COMPLETED'
            assert result['checkpoint_sha256'] == plan['frozen_prerequisites']['model_sha256']
            if 'legacy_regression' in result:
                assert result['legacy_regression']['R1_pass']
            actual[name] = result
        else:
            fail = p.with_suffix('.failure.json')
            actual[name] = json.loads(fail.read_text()) if fail.exists() else r
    # Preserve actual worker output unchanged; independently reprove the original
    # COCO scoring rule on exactly these newly inferred, SHA-verified features.
    coco_proof = json.loads((out / 'CANONICAL_COCO_REPROOF.json').read_text())
    assert coco_proof['status'] == 'PASS'
    actual['COCO']['streaming_gpu_diagnostic_metrics'] = actual['COCO']['metrics']
    actual['COCO']['metrics'] = coco_proof['metrics']
    actual['COCO']['ranking_protocol'] = coco_proof['protocol']
    actual['COCO']['canonical_reproof_sha256'] = sha(out / 'CANONICAL_COCO_REPROOF.json')
    assert sha(plan['checkpoint']) == plan['frozen_prerequisites']['model_sha256']
    statuses = {n: actual[n] if n in actual else plan['blocked'].get(n, {'status': 'NOT_RUN'}) for n in ORDER}
    raw = {'model': 'SAID E2-Uniform@4868', 'base_commit': BASE, 'checkpoint': plan['checkpoint'],
           'checkpoint_sha256': sha(plan['checkpoint']), 'evaluation_status': state['status'],
           'protocol_completion': 'PARTIAL_PROTOCOL_VERIFICATION', 'datasets': statuses,
           'legacy_only': {'Flickr30k-Test1K': actual.get('Flickr30k-Test1K')},
           'missing_metrics_are_not_zero': True, 'scientific_status': 'EXPLORATORY_REPEATED_PUBLIC_BENCHMARKS'}
    atomic_json(out / 'RESULTS_HYFL_PROTOCOL.json', raw, exclusive=True)
    atomic_json(out / 'RUN_STATE.json', state, exclusive=True)
    references = {
        'paper': {'url': 'https://arxiv.org/html/2607.00428v1', 'sha256': sha(REF / 'paper.html'),
                  'read_sections': ['Table1', 'Table3', 'SupplementaryS1.1']},
        'hyfl': {'repo': 'janeyeon/hyfl-clip', 'commit': 'e0fcacceea751d95b36893d0758c808dd34e0c4d',
                 'files': ['hyfl_unified_pinned.py']},
        'longclip': {'repo': 'beichenzbc/Long-CLIP', 'commit': '3966af9ae9331666309a22128468b734db4672a7',
                     'files': ['longclip_flickr_pinned.py']},
        'tulip': {'repo': 'ivonajdenkoska/tulip', 'commit': '1387d2721c59a4fb23b3f7c597118a574367c625',
                 'files': ['tulip_dci.py', 'tulip_prep.py', 'tulip_coco.py', 'tulip_flickr.py']},
        'dci': {'repo': 'facebookresearch/DCI', 'commit': '8326c972847f88b45a01d3f553930558bf60a136',
                'files': ['dci_official_download.py']}}
    for n, r in references.items():
        if 'files' in r:
            r['files'] = {p: sha(REF / p) for p in r['files']}
    atomic_json(out / 'REFERENCE_VERSIONS.json', references, exclusive=True)
    paths = [p for folder in ['model', 'train', 'eval'] for p in (ROOT / folder).rglob('*.py')]
    assert not subprocess.check_output(['git', 'diff', BASE, '--', 'model', 'train', 'eval'], cwd=ROOT)
    manifest = {'base_commit': BASE, 'production_diff': 'EMPTY',
                'production_python_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
                'new_tools_sha256': {str(p.relative_to(ROOT)): sha(p) for pattern in ['*hyfl*.py', '*native*batch*.py', 'restore_native_eval_profile.py', 'flickr_full_protocol.py'] for p in (ROOT / 'tools').glob(pattern)},
                'retrieval_bounded_sha256': sha(ROOT / 'tools/retrieval_bounded.py'),
                'queue_sha256': sha(ROOT / 'tools/eval_resource_queue.py')}
    atomic_json(out / 'CODE_MANIFEST.json', manifest, exclusive=True)
    prior = json.loads((out / 'NATIVE_PRECISION_RESTORE_AUDIT.json').read_text())
    first = json.loads((out / 'EVALUATION_RECOVERY_AUDIT.json').read_text())
    timing = {'first_incompatible_queue_wall_seconds': first['original_state']['wall_seconds'],
              'second_incompatible_queue_wall_seconds': prior['second_state']['wall_seconds'],
              'final_native_queue_wall_seconds': state['wall_seconds'],
              'legacy_independent_reproduction_seconds': first['independent_diagnosis']['elapsed_seconds'],
              'first_queue_start_to_final_queue_end_seconds': (dt.datetime.fromisoformat(state['ended_utc']) -
                  dt.datetime.fromisoformat(first['original_state']['started_utc'])).total_seconds()}
    timing['all_queue_wall_seconds_sum'] = sum(timing[k] for k in ['first_incompatible_queue_wall_seconds',
                                                                 'second_incompatible_queue_wall_seconds', 'final_native_queue_wall_seconds'])
    atomic_json(out / 'EVAL_TIMING.json', timing, exclusive=True)
    f = json.loads((out / 'FLICKR_FULL_DATA_AUDIT.json').read_text())
    protocol_rows = []
    for n in ORDER:
        if n in actual:
            r = actual[n]; protocol_rows.append([n, r['protocol_status'], r.get('n_images'), r.get('n_captions'), r['protocol']])
        else:
            protocol_rows.append([n, 'BLOCKED_PROTOCOL_DATA', '0 evaluated / 31783 annotated', '0 evaluated / 158915 annotated', 'Full images missing'])
    md(out / 'PROTOCOL_AUDIT.md', f'''# HyFL/SAID protocol audit

Fixed model: E2@4868, SHA256 `{sha(plan['checkpoint'])}`. No training, optimizer restoration, model changes, mask/fusion/HNS inference, reranking, TTA or ensemble. Final actual inference uses native ViT-B/16, original preprocess, LongCLIP tokenizer248 with truncate=True, FP32 weights/inputs/outputs/normalized features, L2 normalization and cosine. Autocast is disabled. Original native backend settings are preserved: CUDA matmul TF32=False, cuDNN TF32=True (permitted TF32 image convolution with FP32 tensors).

Read arXiv v1 Table1/Table3/S1.1. S1.1 specifies full Flickr30k, COCO2017-Val5K, 248-token truncation, and normalized cosine for CLIP baselines. HyFL's Lorentzian similarity is NOT introduced into SAID. Published R@1 and our added R@5/10 are distinct.

{table(['Dataset', 'Status', 'Images', 'Captions', 'Protocol'], protocol_rows)}

`VERIFIED_SOURCE_PROTOCOL` means the available official source, selection and metric agree with the published/official-loader definition. It does not imply a bytewise comparison against unavailable HyFL-private derived JSON or pixels.

DOCCI: official docci_descriptions.jsonlines test descriptions and image filenames are exact; 5000 test records, filename ordering, basename-test filtering match HyFL loader. Local images originate from the previously fingerprinted mirror; every image has been decoded and hashed, but HyFL-private image byte identities are unavailable.

DCI: original archive contains 7805 records/images. HyFL reads DCI_test.json and sorts filename, but neither its JSON nor its caption construction was available. The provisional evaluation explicitly uses the existing SAID short_caption + space + extra_caption convention and filename sort. It is NOT a verified HyFL DCI result, and is not ranked against the paper.

Long-DCI: official TULIP constructor sorts annotation filenames, uses raw extra_caption and skips exactly empty strings. 7805 raw records, 203 empty exclusions, 7602 nonempty pairs. We reproduced this constructor using the verified official DCI archive, without additional filtering. 3025 historical captions differ in leading/trailing whitespace. All 7602 image/order identities and all 248-token arrays match the historical stripped manifest (LONG_DCI_TOKEN_EQUIVALENCE.json). The official dci_long.csv is still unavailable; CSV filepath/title and missing-row equivalence remain UNVERIFIED. The reconstruction result is provisional, not a full official CSV replication.

COCO: official 5000 images and 25014 annotations. Explicit positive mappings retain the first five captions per image (25000); 14 extras are excluded by the original documented rule. Sorted image IDs and original annotation order reproduce torchvision CocoCaptions. TULIP eval/coco.py explicitly uses captions[0:5].

Urban: 1000 pairs, fixed stems, first-line captions. Raw newline versus legacy strip yields identical tokenizer output; full candidate pools preserved.

Flickr Full: official Illinois annotation archive retrieved; results_20130124.token has 31783 distinct images and 158915 unique caption IDs with slots #0–#4 per image. Caption-to-image positives are parsed explicitly, without i//5. Source token SHA256 `{f['annotation_sha256']}`. Only {f['available_verified_filename_count']} named image files are available, {f['missing_images']} missing. The official site requests image access via a form; direct image URLs return404 and HF endpoints timed out. No form/message was submitted. No full Flickr inference ran. TULIP results.csv was not available for row comparison.

Flickr test1K remains a separate legacy-only 1000/5000 protocol. Its exact Karpathy origin has not been independently proved, so it is not relabeled as Karpathy or Full.

Ranking rule: descending FP32 similarity; exactly equal candidates ordered by ascending fixed manifest index. Near ties are not rounded. Query/gallery blocks retain global top11, allowing exact top10 and K/K+1 margin auditing. GPU GEMM block shape can alter last-bit near ties; per-query receipts and boundary counts are retained locally. Frozen legacy ranking is separately regressed, including canonical COCO CPU chunk512/1D argsort. Any new deterministic-tie differences are disclosed in LEGACY_COMPARISON.md.

COCO exception required by the existing canonical protocol: PRIMARY results retain CPU FP32/query512/full gallery/per-row1D argsort from the frozen original evaluator. Generic bounded GPU scores are retained only as diagnostics. On the same freshly inferred SHA-verified feature bank, GPU streaming yields I2T R1=61.84 versus canonical61.86, one query (image456303/query3934). Four highest candidates have exactly equal similarity, and the ascending-index rule chooses a different candidate than the frozen per-row argsort rule. CANONICAL_COCO_REPROOF.json records every query difference and independently reproduces all historical counts. No better-score selection is used; the previously established original COCO numeric rule controls primary results regardless of direction of the difference. Full Flickr remains bounded GPU streaming.

Precision compatibility: initial implementation incorrectly disabled the frozen evaluator's default cuDNN TF32 permission. Those incompatible attempts were stopped and retained. Independent old-path DOCCI inference reproduced all historical counts exactly; a single-variable TF32 flag test matched old/new cached image embeddings bit-for-bit. Query test_02516 moved rank6→5 in the incompatible strict-TF32-off variant, changing I2T R5 by one. Across all DOCCI images the two profiles differ by max0.000900492, exceeding5e-6. The final system restores the ORIGINAL FP32 backend profile, verified on all four GPUs at the actual batch64 before launch. We do not conflate FP32 tensor dtype with pure IEEE32 contraction everywhere.

Native batch3 versus batch16 also exceeds5e-6 (max0.000456773), with no small-sample query-hit differences. This numerical batch variation remains a FAILED optimization check; no tolerance is increased. Only the verified original batch64 is authorized by the final plan. Cache identity includes batch and backend flags; changing them refuses cache reuse. CrossGPU/same-batch64 tests pass. See NATIVE_PRECISION_RESTORE_AUDIT.json and EVAL_TESTS.json for failed and passing receipts.

Source hashes/commits: REFERENCE_VERSIONS.json. Full image/caption inventories and feature shards stay local. No silent missing-image deletion, invalid-caption filtering or dataset substitution.
''')
    metrics_rows = []
    for n in ORDER + ['Flickr30k-Test1K']:
        r = actual.get(n)
        for d in ['I2T', 'T2I']:
            if r and 'metrics' in r:
                m = r['metrics'][d]
                metrics_rows.append([n, d, m['query_count'], m['candidate_count']] +
                                    [f"{m['recall_percent'][str(k)]:.6f} ({m['correct'][str(k)]})" for k in [1, 5, 10]])
            else:
                metrics_rows.append([n, d, 'NOT_RUN', 'NOT_RUN', 'N/A', 'N/A', 'N/A'])
    def pair(n):
        r = actual.get(n, {})
        if 'metrics' not in r:
            return 'BLOCKED'
        value = ' / '.join(f"{r['metrics'][d]['recall_percent']['1']:.3f}" for d in ['I2T', 'T2I'])
        return value + (' (UNVERIFIED)' if 'UNVERIFIED' in r['protocol_status'] else '')
    main = table(['Model', 'DOCCI I2T/T2I', 'DCI I2T/T2I', 'Long-DCI I2T/T2I', 'Urban I2T/T2I'],
                 [['E2@4868 (actual)', *[pair(n) for n in ORDER[:4]]]])
    short = table(['Model', 'COCO I2T/T2I', 'Flickr Full I2T/T2I'], [['E2@4868 (actual)', pair('COCO'), pair('Flickr30k-Full')]])
    all_metrics = table(['Dataset', 'Direction', 'Queries', 'Candidates', 'R@1% (correct)', 'R@5% (correct)', 'R@10% (correct)'], metrics_rows)
    md(out / 'RESULTS_HYFL_PROTOCOL.md', '# Actual native retrieval results\n\n' + main + '\n\n' + short + '\n\n' +
       all_metrics + '\n\nDCI/Long-DCI are provisional protocols. Full Flickr has no result. Flickr1K is legacy-only. Missing scores are not zero; no six-set aggregate is reported. All public benchmark observations remain exploratory.\n')
    legacy_rows = []
    for n, r in actual.items():
        if 'legacy_regression' not in r:
            continue
        for d in ['I2T', 'T2I']:
            for k in [1, 5, 10]:
                old = r['legacy_regression']['expected_counts'][d][str(k)]
                legacy = r['legacy_regression']['actual_counts'][d][str(k)]
                new = r['metrics'][d]['correct'][str(k)]
                legacy_rows.append([n, d, k, old, legacy, new, new - old])
    md(out / 'LEGACY_COMPARISON.md', '# Historical protocol regression\n\n' +
       table(['Dataset', 'Direction', 'K', 'Historical count', 'Frozen legacy count (new inference)',
              'New deterministic count', 'Delta queries'], legacy_rows) +
       '\n\nAll frozen legacy counts come from newly inferred embeddings. Historical scores are used ONLY as expected regression targets. COCO retains original raw-CPU normalization and chunk512/1D argsort for independent compatibility verification. Extended legacy datasets use their frozen CPU full-matrix/topk reference (size-guarded; never used for Full Flickr), while the new main evaluator uses bounded GPU blocks. Urban retains its native GPU scoring.\n\n'
       'New exact ties use manifest-index ordering, with no score quantization. Any count delta above is disclosed as an evaluator numerical difference, not concealed as training improvement. Earlier incompatible TF32-off attempts are retained separately; the final FP32 native backend is restored to its original defaults after an exact single-variable diagnosis. Independent old-default DOCCI inference reproduced ALL historical counts. Failed batch3 variation is retained and batch optimization is not approved; formal batch64 is frozen and validated across four GPUs. Evidence is in EVALUATION_RECOVERY_AUDIT.json and NATIVE_PRECISION_RESTORE_AUDIT.json. Any remaining count deviations are NOT marked exact regression PASS. Query-level top11/score/hit artifacts remain local.\n\n'
       'COCO primary results retain the original canonical CPU512/1D-argsort numerical protocol on new feature banks. Generic GPU streaming gives I2T R1=61.84 instead of61.86; query3934/image456303 has four exactly tied highest candidates, where ascending-index versus original per-row argsort selects different items. That replacement is not promoted to primary. CANONICAL_COCO_REPROOF.json supplies full query evidence and all exact canonical counts.\n\n'
       'Flickr-test1K (1000/5000) is not Full (31783/158915). Long-DCI reconstruction changed only raw whitespace; token/image/order equivalence is proven, CSV equivalence is not. DCI short-plus-extra is independent and not used as a replacement for verified Long-DCI. Old Score5/J_long3/J_long/Short4 definitions remain historical; no renamed six-set aggregate is used.\n')
    resource_rows = [[n, r.get('physical_gpu'), f"{r.get('elapsed_seconds', 0):.3f}",
                      f"{r.get('peak_cuda_allocated', 0)/(1<<30):.3f}",
                      f"{r.get('peak_cuda_reserved', 0)/(1<<30):.3f}", r.get('status')]
                     for n, r in actual.items()]
    resources = table(['Job', 'Physical GPU', 'Seconds', 'Peak allocated GiB', 'Peak reserved GiB', 'Status'], resource_rows)
    md(out / 'EVAL_SCHEDULER_REPORT.md', f'''# Resource-aware evaluator

Dynamic arbitrary-length dataset queue on explicit cuda:0–3 (no CUDA_VISIBLE_DEVICES remapping), one owned worker per GPU. Frozen estimated cost sorts largest jobs first; observed seconds/work-unit informs remaining priorities. Available RAM/disk checked before launch; GPU/process/CPU/disk evidence recorded. One GPU lock per lane, one output lock per queue. No external process is killed. Each job has an exclusive attempt directory and log.

On failure, successful peers are retained; only owned worker process groups are cancelled, with a bounded TERM/KILL cleanup. Pending jobs are cancelled, evidence saved. Explicit --resume verifies frozen plan identity and completed-result SHA, refuses live recorded workers, and creates new attempt directories. Verified feature shards resume without repeated feature rows/counts. Any orphan/corrupt shard is rejected.

Full Flickr uses the same general worker and verified explicit-positive parser once a complete manifest/root is supplied; missing full images currently block that job. No cross-GPU feature sharding was enabled. Data-level concurrency was sufficient; multiGPU sharding is not claimed validated.

Features are encoded in frozen batches64, loaders4 workers, original native FP32 backend settings. Scoring query chunk256/gallery chunk4096 bounds each similarity block to4MiB (plus merge buffers), rather than a20.2GB Full Flickr similarity matrix. Only feature banks and global top11 are retained. Exact ties use ascending candidate index; near ties retain unrounded scores. Feature cache binds checkpoint/manifest/record/image-content/tokenizer/context/preprocess/precision/backend/batch and encoder-version hashes, each tensor shard has SHA/IDs/shape checks.

Scheduler wall: {state.get('wall_seconds', 0):.3f} seconds, including loading/encoding/scoring/legacy regression. GPU small-batch tests are reported separately in EVAL_TESTS.json.

All queue attempts summed:{timing['all_queue_wall_seconds_sum']:.3f}s; includes both retained incompatible runs. Independent frozen DOCCI reproduction:{timing['legacy_independent_reproduction_seconds']:.3f}s. First queue start to final completion:{timing['first_queue_start_to_final_queue_end_seconds']:.3f}s (includes diagnosis, code correction, preflight and gaps). EVAL_TIMING.json keeps these denominators separate.

{resources}

RUN_STATE.json includes start/end UTC, PIDs, commands, return codes, logs, physical GPU binding, CPU RAM/disk preflight and full worker durations. Per-worker CPU peak RSS, CUDA allocated/reserved peaks, counts/candidates and precision are in RESULTS_HYFL_PROTOCOL.json. No OOM was reported by completed jobs. Full-Flickr resource viability at its actual scale remains UNVERIFIED because images are unavailable.
''')
    diagnostics = {'preparation_attempts': [
        {'log': '/root/said_hyfl_protocol_eval_v1/data_audit.log', 'failure': 'DOCCI individual symlink path rejected by root guard',
         'resolution': 'Use the same images in their canonical archive directory; no images changed.'},
        {'log': '/root/said_hyfl_protocol_eval_v1/data_audit_v2.log', 'failure': 'DCI expected SHA copied from historical script typo',
         'resolution': 'Verified official DCI download.py checksum and existing frozen recovery/status; corrected new audit constant only.'},
        {'log': '/root/said_hyfl_protocol_eval_v1/data_audit_v3.log', 'status': 'PASS'}],
        'formal_worker_attempts': {n: r['attempt'] for n, r in state['jobs'].items()},
        'evaluation_recovery_audit': 'EVALUATION_RECOVERY_AUDIT.json' if (out / 'EVALUATION_RECOVERY_AUDIT.json').exists() else None,
        'historical_results_preserved': True, 'production_code_preserved': True}
    atomic_json(out / 'PREPARATION_DIAGNOSTICS.json', diagnostics, exclusive=True)
    gpu_listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                           '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    atomic_json(out / 'FINAL_RESOURCE_CHECK.json', {'gpus': gpu_listing, 'compute_apps': apps,
                                                   'idle': not apps.strip(), 'checkpoint_sha256_after': sha(plan['checkpoint'])}, exclusive=True)
    md(out / 'FINAL_REPORT.md', f'''# E2@4868 HyFL protocol unification: final audit

Branch: `experiment/said-e2-hyfl-eval-unified-v1`, base `{BASE}`. Only one fixed native bare model evaluated; no training/optimizer/checkpoint/model change. Production model/train/eval diff is empty (CODE_MANIFEST.json). Strict317-key FP32/context248 load passed; bare SHA before/after `{sha(plan['checkpoint'])}`.

Outcome: real four-GPU run {state['status']}; protocol verification is PARTIAL. Three main source protocols verified (DOCCI, Urban, COCO), DCI and Long-DCI provisional, Full Flickr blocked. No claim of six fully aligned datasets, comprehensive superiority or unbiased SOTA.

{main}

{short}

{all_metrics}

Full Flickr official annotation:31783 images/158915 captions, exact #0–#4 mapping; available image filenames {f['available_verified_filename_count']}, missing {f['missing_images']}, evaluated0. DCI actual7805/7805; Long-DCI actual7602/7602, officialCSV missing. The latter follows frozen TULIP constructor and is token-identical to the old7602 protocol, but CSV equivalence remains unproved.

Tests:14 CPU tests passed; real fourGPU old/new native FP32 comparison at the frozen formal batch64 passed with5e-6 tolerance and zero query-hit differences. Observed max difference:{native_error}. Full parameter-state digest unchanged. Long-DCI7602 token arrays/image/order matched exactly despite3025 whitespace-string differences. Historical five-dataset R@1 counts have a hard exact-match gate; R@5/10 are compared and remaining deviations, if any, are disclosed (LEGACY_COMPARISON.md).

Failed checks preserved: strict-TF32-off versus old/native features exceeded5e-6 (max0.000900492); native batch3 versus16 exceeded5e-6 (max0.000456773), despite unchanged small-sample hits. The first issue was the new controller's accidental precision override, now corrected to ORIGINAL native FP32 defaults; the second means batch3 optimization is not approved. Formal batch64 crossGPU equivalence passes; no threshold is increased. Both earlier aborted queues and all diagnosis receipts remain immutable. No claim that every attempted numerical check passed is made.

COCO primary retains its frozen CPU512/1D-argsort scoring rule and exactly reproduces the historical6 counts on new model features. GPU streaming's one I2T R1 query difference is kept as an explicit diagnostic; no old score was copied or chosen for being larger. CANONICAL_COCO_REPROOF.json independently proves the primary result. The exact worker version used for encoding/scoring is preserved as WORKER_EXECUTED_NATIVE_V3.py; the final reusable worker records both scoring paths and keeps canonical COCO primary by default.

{resources}

FourGPU scheduled wall time:{state.get('wall_seconds', 0):.3f}s. See RUN_STATE.json for UTC, physical mapping, exit codes, CPU/disk preflight and worker logs. Completed jobs report no OOM/nonfinite features. Final compute process list: `{apps.strip() or 'EMPTY'}`. All controlled workers exited. Final GPU readings:\n\n```\n{gpu_listing.rstrip()}\n```

Remaining limitations: unavailable Full Flickr image library; unavailable HyFL DCI_test.json and caption-construction proof; unavailable official dci_long.csv and TULIP results.csv; no independently sealed validation protocol; public benchmarks repeatedly observed. Local mirrored image byte equality with HyFL-private datasets cannot be established. Full-Flickr end-to-end timing/memory and crossGPU sharded encoding remain unvalidated; only generic bounded-memory mathematics and dataset-level fourGPU scheduling are validated. No scores are fabricated or borrowed from legacy reports as new inference.

Preparation failures were model-free and preserved in PREPARATION_DIAGNOSTICS.json. Two incompatible evaluation queues stopped on legacy regression differences; original cancellation evidence and exact source diffs are in EVALUATION_RECOVERY_AUDIT.json and NATIVE_PRECISION_RESTORE_AUDIT.json. The final original-native FP32 profile was frozen and verified BEFORE the final run, with fresh cache/output namespace and no bypass of earlier identities. Complete query-level evidence, verified feature shards, official downloaded references and large logs remain under `/root/said_hyfl_protocol_eval_v1/` and the server runtime references directory. New small artifacts only are published. Historical E2 checkpoint and results remain untouched. Task stops here; no later experiment or training is started.
''')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--runtime', required=True)
    a = p.parse_args(); run(a.output, a.runtime)
