"""Write the three separate result layers and small, credential-free evidence."""
import json
from pathlib import Path
import shutil

from .common import EXP, OFFICIAL, ROOT, RUN, digest, dump, sha256

NAMES = ['COCO', 'Urban-1k', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI']
PUBLISHED = {'COCO': {'I2T': 61.3, 'T2I': 43.0},
    'Flickr-full': {'I2T': 56.6, 'T2I': 36.6},
    'Urban1k': {'I2T': 93.1, 'T2I': 93.0}, 'DOCCI': {'I2T': 79.7, 'T2I': 80.0},
    'DCI-full': {'I2T': 68.5, 'T2I': 67.6}, 'Long-DCI (paper)': {'I2T': 57.8, 'T2I': 57.4}}


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def pct(v):
    return f'{v*100:.6f}'


def small_console(source, target):
    # Preserve every non-progress log line; full raw console remains in runtime.
    lines = Path(source).read_text(errors='replace').replace('\r', '\n').splitlines()
    lines = [l.rstrip() for l in lines if not ('it/s]' in l or 's/it]' in l)]
    Path(target).write_text('\n'.join(lines)+'\n')


def compact_rankings(path, value):
    # Keep each query on one line so the complete required top10 evidence is
    # small and reviewable. This changes formatting only, not JSON contents.
    pieces = []
    for direction, data in value['directions'].items():
        metadata = {k: v for k, v in data.items() if k != 'per_query_top10'}
        body = json.dumps(metadata, separators=(',', ':'))[:-1]
        rows = ',\n'.join(json.dumps(r, separators=(',', ':')) for r in data['per_query_top10'])
        pieces.append(json.dumps(direction)+':'+body+',"per_query_top10":[\n'+rows+'\n]}')
    top = {k: v for k, v in value.items() if k != 'directions'}
    text = json.dumps(top, separators=(',', ':'))[:-1]+',"directions":{\n'+',\n'.join(pieces)+'\n}}\n'
    assert json.loads(text) == value
    Path(path).write_text(text)


def main():
    official = read(RUN/'base-full-amp/results.json')
    native = read(RUN/'unified/results.json')
    fp32 = read(RUN/'urban-fp32/results.json')
    sanity = read(RUN/'urban-amp/results.json')
    assert official['status'] == native['status'] == 'COMPLETE'
    assert len(official['datasets']) == len(native['datasets']) == 5
    said = read(ROOT/'experiments/external_baselines/beta_clip_v1/SAID_PROTOCOL_RESULTS.json')['said_reference']
    source = read(RUN/'official-source.json')
    dataset_audit = read(RUN/'dataset-protocol-audit.json')
    urban_audit = read(RUN/'urban-data-audit.json')
    gate = read(RUN/'unified/urban-gate.json')
    precision = read(RUN/'urban-precision-audit.json')
    rank_consistency = read(RUN/'urban-native-ranking-consistency.json')
    initial_gate = read(RUN/'urban-gate-initial-order-failure.json')
    errors = read(RUN/'urban-error-analysis.json')
    ranking = read(RUN/'urban-ranking-agreement.json')
    deltas = {n: {d: (said['metrics'][n][d]['R@1']-native['datasets'][n]['metrics'][d]['R@1'])*100
                  for d in ('I2T', 'T2I')} for n in NAMES}
    score_deltas = {k: (said['scores'][k]-v)*100 for k, v in native['scores'].items()}
    native.update(said_reference=said, delta_direction='SAID minus DeBias',
                  delta_R1_pp=deltas, delta_scores_pp=score_deltas,
                  urban_adapter_consistency=gate, source=source)
    dump(EXP/'SAID_UNIFIED_RESULTS.json', native)
    official.update(source=source, checkpoint=read(EXP/'CHECKPOINT_INVENTORY.json'),
        phase_a_uses_said_manifest=False, published_reference=PUBLISHED,
        first_urban_sanity=sanity, separate_fp32_urban_diagnostic=fp32,
        dataset_protocol_audit=dataset_audit)
    dump(EXP/'OFFICIAL_REPRO_RESULTS.json', official)
    dump(EXP/'URBAN_DATA_AUDIT.json', urban_audit)
    dump(EXP/'URBAN_ERROR_ANALYSIS.json', errors)
    compact_rankings(EXP/'URBAN_RANKING_AGREEMENT.json', ranking)
    dump(EXP/'URBAN_PRECISION_AUDIT.json', precision)
    dump(EXP/'DATASET_PROTOCOL_AUDIT.json', dataset_audit)
    inventory = read(EXP/'CHECKPOINT_INVENTORY.json')
    inventory.update(status='COMPLETE_VERIFIED_OFFICIAL_PIPELINE', strict_loading=official['strict_loading'],
                     architecture='ViT-B/16', actual_model_construction=native['checkpoint']['construction'],
                     raw_checkpoint_tensor_mismatches_after_official_loading=native['checkpoint']['raw_checkpoint_tensor_mismatches_after_official_loading'])
    dump(EXP/'CHECKPOINT_INVENTORY.json', inventory)
    for n, record in official['datasets'].items():
        dump(EXP/'raw/official'/f'{n}.json', record)
    for n, record in native['datasets'].items():
        dump(EXP/'raw/unified'/f'{n}.json', record)
    dump(EXP/'raw_logs/official-source-fingerprints.json', source)
    dump(EXP/'raw_logs/data-preparation.json', read(RUN/'data-preparation.json'))
    dump(EXP/'raw_logs/flickr-full-transfer.json', read(RUN/'flickr-full-transfer.json'))
    dump(EXP/'raw_logs/said-reference-urban-check.json', read(RUN/'said-urban-predictions.json'))
    dump(EXP/'raw_logs/native-urban-consistency.json', gate)
    dump(EXP/'raw_logs/native-urban-ranking-consistency.json', rank_consistency)
    dump(EXP/'raw_logs/native-urban-initial-batch-diagnostic.json', {
        'initial': initial_gate, 'final': gate,
        'recalls_unchanged': initial_gate['frozen_metrics'] == gate['frozen_metrics'],
        'resolution': 'Match official encoder batch composition, including final8 captions, then restore the exact frozen SAID candidate order. Raw text/images/checkpoint/evaluator unchanged; tolerance not loosened.'})
    metric_files = ['eval/retrieval/coco_retrieval.py', 'tools/urban1k_retrieval.py',
                    'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py']
    dump(EXP/'raw_logs/frozen-metric-source-fingerprints.json', {p: sha256(ROOT/p) for p in metric_files})
    for name in ['base-full-amp', 'urban-amp', 'urban-fp32', 'unified', 'said-urban-predictions']:
        small_console(RUN/(name+'.console.txt'), EXP/'raw_logs'/(name+'.summary.console.txt'))
    python = '/root/miniconda3/envs/debias-clip-official/bin/python'
    module = 'experiments.external_baselines.debias_clip_v1.'
    for name, suffix, argv in [('official-base-full', 'official_repro', []),
                              ('official-urban-sanity', 'official_repro', ['--urban-only']),
                              ('official-urban-fp32-diagnostic', 'official_repro', ['--urban-only', '--precision', 'fp32']),
                              ('unified-five', 'unified', []), ('urban-errors', 'urban_errors', [])]:
        env = {'PYTHONPATH': str(ROOT)}
        if name == 'official-urban-fp32-diagnostic':
            env['CUDA_VISIBLE_DEVICES'] = '1'
        dump(EXP/'commands'/(name+'.json'), {'cwd': str(ROOT), 'env': env,
            'argv': [python, '-m', module+suffix, *argv], 'exit_code': 0})
    dump(EXP/'commands/official-expanded-test-arguments.json', {
        'python_main_equivalent': [python, str(OFFICIAL/'main.py'), *official['argv']],
        'executed_via': 'official_repro observer calls unchanged main.main(argv)',
        'normalization': 'official shell syntax normalized; evaluation arguments unchanged; machine paths/log destinations adapted',
        'source_modified': False})
    pub_table = table(['Dataset/protocol', 'Published I2T R1 %', 'Published T2I R1 %'],
                      [[n, f"{p['I2T']:.1f}", f"{p['T2I']:.1f}"] for n, p in PUBLISHED.items()])
    official_rows = []
    for n, r in official['datasets'].items():
        for d in ('I2T', 'T2I'):
            value = r['metrics'][d]['R@1']*100
            delta = value-PUBLISHED[n][d]
            classification = 'REPRODUCED' if abs(delta)<.05 else 'NEAR-REPRODUCED' if abs(delta)<=.200001 else 'NOT REPRODUCED'
            official_rows.append([n, d, f'{PUBLISHED[n][d]:.1f}', f'{value:.6f}', f'{delta:+.6f}', classification])
    off_table = table(['Official dataset', 'Direction', 'Published %', 'Released checkpoint + official evaluator %',
                       'Δ reproduced−published pp', 'Classification'], official_rows)
    unified_rows = [[n, d, pct(native['datasets'][n]['metrics'][d]['R@1']), pct(said['metrics'][n][d]['R@1']),
                     f'{deltas[n][d]:+.6f}'] for n in NAMES for d in ('I2T', 'T2I')]
    unified_table = table(['Frozen dataset', 'Direction', 'DeBias B/16 R1 %', 'SAID B/16 R1 %', 'Δ SAID−DeBias pp'], unified_rows)
    scores_table = table(['Metric', 'DeBias B/16 %', 'SAID B/16 %', 'Δ SAID−DeBias pp'],
        [[k, pct(v), pct(said['scores'][k]), f'{score_deltas[k]:+.6f}'] for k, v in native['scores'].items()])
    urban = official['datasets']['Urban1k']
    answer = 'YES' if urban['classification']=='REPRODUCED' else 'NEAR' if urban['classification']=='NEAR-REPRODUCED' else 'NO'
    urban_native_table = table(['Direction', 'R@1 %', 'R@5 %', 'R@10 %'],
        [[d, *[pct(native['datasets']['Urban-1k']['metrics'][d][f'R@{k}']) for k in (1, 5, 10)]] for d in ('I2T', 'T2I')])
    protocol_rows = [
        ['COCO', 'Paper: val2017 5K; precise caption count unspecified', '5000/25014, all raw annotations', '5000/25000, first5/image, chunk512', 'No exact match:14 extra official captions and evaluator/order differences'],
        ['Urban1k', '1K diagonal pairs', '1000/1000; raw pairs identical to SAID', '1000/1000 frozen lexical order', 'Same raw protocol; AMP versus predeclared FP32 and metric implementation audited'],
        ['Flickr', 'Full Flickr30k', '31014 images/155070 captions; val+train+test', 'test1K:1000/5000', 'No for published/full result; unified test1K is comparable'],
        ['DOCCI', 'test split 5K', '5000/5000; raw pair equality audited', '5000/5000 frozen test', 'Same raw pairs; precision/metric differences retained'],
        ['DCI', 'short+long captions', '7805/7805, all splits, short+extra without inserted separator', 'DCI full not used in SAID fair table', 'No comparison to SAID Long-DCI'],
        ['Long-DCI', 'Paper long-caption-only variant; no exact frozen-manifest identity supplied', 'No Long-DCI result in released base_full', 'reconstructed7602/7602 exact SHA', 'Only unified DeBias7602 versus SAID7602 is compared']]
    protocol_table = table(['Dataset', 'DeBias published protocol', 'Official code observed', 'SAID frozen protocol', 'Direct comparability'], protocol_rows)
    (EXP/'PROTOCOL_AUDIT.md').write_text('# Dataset and inference protocol audit\n\n'+protocol_table+'\n\nPhase A uses raw official data/reader inputs, not SAID manifests. Urban sorted raw pair fingerprints agree and all1000 captions match; enumeration order differs. DOCCI sorted image/caption pairs agree exactly. All official image references exist. Raw hashes, roots, counts and membership checks are in DATASET_PROTOCOL_AUDIT.json/URBAN_DATA_AUDIT.json.\n\nPrimary reproduction uses default AMP, while unified native FP32 was fixed before scores. The additional official FP32 Urban diagnostic changes only evaluation precision and confirms the adapter separately. URBAN_PRECISION_AUDIT.json preserves each changed query; primary AMP is not replaced by whichever precision scores higher.\n')
    raw_recalls = table(['Protocol', 'Dataset', 'Direction', 'R@1 %', 'R@5 %', 'R@10 %'],
        [[label, n, d, *[pct(r['metrics'][d][f'R@{k}']) for k in (1, 5, 10)]]
         for label, ds in [('Official default AMP', official['datasets']), ('SAID frozen / native FP32', native['datasets'])]
         for n, r in ds.items() for d in ('I2T', 'T2I')])
    report = f'''# DeBias-CLIP official B/16 reproduction and SAID frozen re-evaluation

Completed using the released `debias_vitb_3e.pt`, pinned official source, untouched model/evaluator,
and two separately audited dataset protocols. No training, checkpoint selection or inference augmentation ran.

## Q1: Official Urban T2I=93.0 reproduction

**{answer}.** Published93.0%; official default-AMP reproduction {urban['metrics']['T2I']['R@1']*100:.6f}%,
{urban['correct_T2I_queries']}/1000 correct; difference {urban['delta_to_published_T2I_pp']:+.6f}pp,
{urban['delta_correct_T2I_queries']:+d} queries. Official I2T {urban['metrics']['I2T']['R@1']*100:.6f}%
versus paper93.1%, retained alongside T2I.

## Table A: Published reference only

{pub_table}

Sources: [pinned official README](https://github.com/TRAILab/DeBias-CLIP/blob/18a06c98bcb50018e22c3febca2aefc8c4a12e0d/README.md)
and [paper2602.22419v2 Tables3/4](https://arxiv.org/html/2602.22419v2).
Table5's ablation DOCCI T2I79.7 differs from the primary80.0; both source entries are recorded.
Published full Flickr and DCI/Long-DCI protocols are not mixed into a SAID Score5.

## Table B: Official released-checkpoint reproduction

{off_table}

Actual full candidates: COCO5000/25014; Flickr31014/155070; Urban1000/1000;
DOCCI5000/5000; DCI7805/7805. The original base_full main entry point and its dataset readers
and ranking functions were called. Shell continuation syntax and machine paths were normalized;
evaluation flags were preserved. All official R@1/5/10 are retained in raw/official and the JSON report.
Long-DCI is a published reference but is not a separate dataset returned by released base_full.

## Q2 and Table C: Unified frozen SAID protocol

{unified_table}

Unified Urban complete Recall:

{urban_native_table}

{scores_table}

Score5 averages all ten raw fractional directional R1 values; J_long3 averages Urban/DOCCI/Long-DCI's
six directions, J_long averages Urban/DOCCI's four. Differences are **SAID minus DeBias**.
All primary fair results use one checkpoint, native independent official encoders and plain normalized dot products.
No PCA, drop-first, sentence shuffle, shifted-padding augmentation, ensemble, reranking or conditional image feature runs.

## Q3: Located differences rather than a generic protocol label

Urban:1000 sorted raw image/caption pairs are identical (different_count={urban_audit['different_count']});
sorted pair SHA256 `{urban_audit['said_sorted_pair_sha256']}`. Official JSON uses filesystem enumeration,
SAID uses frozen lexical order; positive IDs are remapped by basename. Tokenizer remains the official
open_clip SimpleTokenizer at248 and preprocessing remains the official checkpoint factory transform.
Official default AMP and separate official FP32 differ in {precision['directions']['T2I']['top1_changed_count']}
T2I top1 queries, with correct counts {precision['directions']['T2I']['amp_correct']} versus
{precision['directions']['T2I']['fp32_correct']}. Exact changed IDs are in URBAN_PRECISION_AUDIT.json.
This numerical precision difference is isolated using identical official raw rows/order/checkpoint.
Native FP32 embedding errors, frozen metric outputs and official FP32 consistency outcomes are in
raw_logs/native-urban-consistency.json; no default-AMP score is replaced by the diagnostic.
The first consistency gate found text max_abs0.0001925528 when text batch composition differed.
Matching the official batches (including final8 captions), then restoring the unchanged frozen candidate
order, reduces image/text max_abs to5.96046448e-8. Every R@1/5/10 value remains unchanged; no tolerance
is relaxed. Top1/ranking agreement is measured in raw_logs/native-urban-ranking-consistency.json.

COCO: released reader keeps25014 captions, frozen SAID keeps25000; it also changes image/caption ordering
and canonical chunk512/row-wise sorting. Flickr:31014 full-image candidates versus1000 test candidates,
not interchangeable scores. DOCCI: raw test membership/text are equal; precision and official argsort
versus frozen topk implementation are distinct. DCI: all7805 split entries with short+extra are different
from reconstructed7602 long captions. PROTOCOL_AUDIT.md gives the dataset-by-dataset evidence.
Dataset differences are established from raw data; numerical changes not isolated to a single factor
are not claimed as causally proved.

## Checkpoint and official loader audit

File size {inventory['size_bytes']} bytes; SHA256 `{inventory['sha256']}`; saved epoch3.
303 state tensors /149835265 tensor elements. Actual B/16 shapes: vision conv[768,3,16,16],
vision positions[197,768], joint projection[768,512], text positions/base and residual[248,512].
The file saves epoch/name/state_dict/optimizer/scaler, but no args/config. Script flags are provenance,
not independently stored checkpoint hyperparameters. Manual transfer is from the author-provided official
Drive checkpoint through user cloud disk; access passwords/signed URLs are not published.

Official strict loading has no missing/unexpected keys and matches every effective post-conversion tensor.
There is one raw-state mismatch: the pinned factory replaces saved `text.positional_embedding_res`
with the base `text.positional_embedding` for this already-stretched custom checkpoint.
Maximum difference0.000978708267211914. Both phases preserve this same official behavior; no fix is
applied to chase published scores. Thus strict key compatibility does **not** mean every raw tensor
survives the official loader unchanged. This anomaly is explicit in CHECKPOINT_INVENTORY.json.

The official Urban same-input adapter audit has zero image/text differences and zero direct scaled-score
differences; own Recall matches the evaluator. Primary AMP normalized outputs are FP32,[1000,512],
mean norms1.0, full[1000,1000] similarity. Scaling/unscaled top1 equivalence is measured in the raw JSON.
Phase B's PCA-call guard confirms zero training PCA calls.

## Q4: Fair comparison scope and error analysis

The reference is fixed SAID Balanced-Stack-Patch B/16, four epochs/4868 updates, fusion_lr2e-4,
inclusion_max1, view_weights[1,1,1], sparsity_scale1, native inference. Reference scores are reused,
and its Urban predictions were independently recomputed to match92.10/91.10 exactly within float32 mean.
Bare checkpoint SHA256 `{said['bare_sha256']}`; full `{said['checkpoint_sha256']}`.
This is a same-test-protocol inference comparison, not matched training compute: DeBias has three
released training epochs/global batch256; SAID has four epochs/global full candidates1024.
Neither model was trained or selected again in this task. No L/14 model is evaluated.

Urban correctness categories, every query ID and top10 candidates are saved in
URBAN_ERROR_ANALYSIS.json and URBAN_RANKING_AGREEMENT.json. No manual hard-query selection is used.

## Complete measured Recall@1/5/10

{raw_recalls}

Evidence: CHECKPOINT_INVENTORY.json, PUBLISHED_REFERENCE.json, OFFICIAL_REPRO_RESULTS.json,
SAID_UNIFIED_RESULTS.json, PROTOCOL_AUDIT.md, NATIVE_INFERENCE_AUDIT.md, URBAN_DATA_AUDIT.json,
URBAN_PRECISION_AUDIT.json, environment.json, structured commands/, raw/ and small raw_logs/.
Source remains clean at `{source['commit']}`. Large weights, datasets, embeddings and environments remain on server.
'''
    (EXP/'DEBIAS_VS_SAID.md').write_text(report)
    (EXP/'OFFICIAL_REPRO_REPORT.md').write_text('# Official default-AMP reproduction\n\n'+
        f'Urban T2I target93.0%: **{answer}**, measured{urban["metrics"]["T2I"]["R@1"]*100:.6f}%, {urban["correct_T2I_queries"]}/1000.\n\n'+
        off_table+'\n\nSource commit, strict loader anomaly, observed pools and precision diagnosis are retained in OFFICIAL_REPRO_RESULTS.json and DEBIAS_VS_SAID.md. All measured directions and recalls are retained, including unfavorable results.\n')
    (EXP/'README.md').write_text('''# DeBias-CLIP official B/16 reproduction: complete

Read [DEBIAS_VS_SAID.md](DEBIAS_VS_SAID.md) for the separate published/reference,
official default-AMP reproduction, and frozen SAID/native-FP32 result tables.
The released three-epoch B/16 checkpoint is used throughout; no training or L/14 evaluation runs.
Both phases preserve the pinned official loader, including the recorded positional-residual overwrite.
Only machine paths/shell syntax and documented annotation schema are adapted; official Python source remains clean.

Code: common.py native adapter/strict audit; prepare_data.py/audit_data.py raw protocol checks;
official_repro.py observes the untouched main/evaluator; unified.py uses frozen SAID metrics;
precision_audit.py/said_urban.py/urban_errors.py support requested query diagnosis;
write_reports.py writes small sanitized evidence. Structured argv arrays are in commands/.
Official checkout, released weight, source datasets and embedding caches remain outside Git.
''')
    inference = (EXP/'NATIVE_INFERENCE_AUDIT.md').read_text()
    marker = 'Default AMP official retrieval can create half-precision'
    if marker in inference:
        inference = inference.split(marker)[0]
    inference += '\nCompleted measured audit: primary AMP encoders return normalized FP32 image/text features; both norm means1.0. Same-input adapter differences and direct-score differences are0. Official loader raw positional overwrite is confirmed. Unified PCA calls0. AMP versus native FP32 changes one Urban T2I top1 query,930→931 correct; no image/caption/checkpoint change. Encoder batch composition was aligned for the consistency check and frozen candidate order restored, reducing feature max_abs to5.96e-8 while all six recalls remained identical. Same-input and frozen-ranking evidence is retained in raw_logs/.\n'
    (EXP/'NATIVE_INFERENCE_AUDIT.md').write_text(inference)
    state = read(RUN/'state.json')
    state.update(status='EVALUATION_COMPLETE_PENDING_GIT', phase='final report and Git synchronization',
                 official_urban_T2I_answer=answer, scores=native['scores'], delta_scores_pp=score_deltas)
    dump(RUN/'state.json', state)
    print(json.dumps({'answer_Q1': answer, 'scores_percent': {k: v*100 for k, v in native['scores'].items()},
                      'delta_scores_pp': score_deltas}), flush=True)


if __name__ == '__main__':
    main()
