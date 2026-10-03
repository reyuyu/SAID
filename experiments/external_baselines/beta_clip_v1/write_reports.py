"""Publish only small, credential-free evidence from the completed CE evaluation."""
import json
from pathlib import Path
import shutil

from .native_adapter import EXP, RUN

NAMES = ['COCO', 'Urban-1k', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI']


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (EXP / name).write_text(json.dumps(value, indent=2)+'\n')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def main():
    native = read(RUN / 'ce-native/results.json')
    official = read(RUN / 'ce-official-urban/result.json')
    assert native['status'] == 'COMPLETE' and official['consistency']['passed']
    old = read(EXP / 'SAID_PROTOCOL_RESULTS.json')
    said = old['said_reference']
    source = old['source']
    source['l14_release_search_scope'] = 'Pinned v1 README/download script: no separate fine-tuned L14 link; transferred CE checkpoint is confirmed B/16. BCE absent.'
    transfer = read(RUN / 'ce-transfer.json')
    transfer['identity_status'] = 'verified: actual checkpoint args/state, strict load and official Urban CLS reproduction'
    candidate = read(RUN / 'ce-internal-inventory.json')
    candidate['provenance'] = transfer
    candidate['strict_load'] = native['checkpoint']
    bce = {'status': 'CHECKPOINT_UNAVAILABLE', 'identity': 'BCE',
           'official_google_drive_id': '1sFQXDQZOzM5XoFkRD_9OYORfQdJrHmN4',
           'reason': 'No BCE checkpoint found in manual discovery; supplied cloud share contained CE only.',
           'evaluation': None, 'blocks_ce': False}
    inventory = {'status': 'CE_VERIFIED_BCE_UNAVAILABLE', 'source': source,
        'CE': candidate, 'BCE': bce, 'historical_discovery': 'raw_logs/checkpoint-discovery.json',
        'historical_download_failures': 'commands/ and raw_logs/*once.console.txt; superseded by manual CE transfer',
        'no_training_or_finetuning': True, 'no_model_substitution': True}
    write('CHECKPOINT_INVENTORY.json', inventory)
    official['runtime_precision'] = native['precision']
    official['published_reference']['matched_path'] = 'CLS (established by measured output)'
    official['published_reference']['TCI_reference_available'] = False
    official['reproduction_note'] = 'CLS reproduces the published CE pair exactly. TCI does not equal that CLS reference; its comparison label is not evidence of a failed TCI-specific published claim.'
    write('OFFICIAL_URBAN_REPRO_CE.json', official)
    metrics = {n: native['datasets'][n]['metrics'] for n in NAMES}
    delta = {n: {d: (said['metrics'][n][d]['R@1']-metrics[n][d]['R@1'])*100
                 for d in ('I2T', 'T2I')} for n in NAMES}
    delta_scores = {k: (said['scores'][k]-v)*100 for k, v in native['scores'].items()}
    native.update(said_reference=said, delta_direction='SAID minus BetaCLIP',
                  delta_R1_pp=delta, delta_scores_pp=delta_scores)
    write('CE_SAID_PROTOCOL_RESULTS.json', native)
    unified = {'status': 'COMPLETE_CE_BCE_UNAVAILABLE', 'source': source, 'beta_clip_ce': native,
        'beta_clip_bce': None, 'bce_availability': bce,
        'official_urban_cls': {k: official['official_evaluator']['results'][k] for k in
                              ('retrieval_acc_cls_i2t', 'retrieval_acc_cls_t2i')},
        'official_urban_tci': {k: official['official_evaluator']['results'][k] for k in
                              ('retrieval_acc_tci_i2t', 'retrieval_acc_tci_t2i')},
        'native_adapter_validation': official['consistency'], 'said_reference': said,
        'delta_direction': 'SAID minus BetaCLIP', 'delta_R1_pp': delta, 'delta_scores_pp': delta_scores,
        'no_training_or_finetuning': True, 'no_model_substitution': True, 'said_model_code_modified': False}
    write('SAID_PROTOCOL_RESULTS.json', unified)
    write('raw/ce/STATUS.json', {'status': 'COMPLETE', 'checkpoint_sha256': candidate['sha256'],
        'official_urban_reproduced': True, 'native_consistency_passed': True, 'frozen_five_datasets_complete': True})
    write('raw/bce/STATUS.json', bce)
    for n in NAMES:
        write('raw/ce/'+n+'.json', native['datasets'][n])
    write('raw/ce/strict_load.json', native['checkpoint'])
    env = read(EXP / 'environment.json')
    env['evaluation_precision'] = native['precision']
    write('environment.json', env)
    gate = read(RUN / 'urban-gate-diagnosis.json')
    gate.update(cause='Unified harness initially disabled cuDNN TF32 convolution, unlike the official evaluator default. Match the official runtime setting; no model or data change.',
               resolution='PASS: final frozen Urban I2T 88.600004%, T2I 89.000005%; R1 equals official CLS within float32 metric precision.',
               final_cudnn_allow_tf32=True, final_matmul_allow_tf32=False)
    write('raw_logs/urban-gate-diagnosis.json', gate)
    for source_name, dest in [('ce-official-urban.console.txt', 'official-urban-ce.console.txt'),
                             ('ce-native.console.txt', 'native-five-ce.console.txt'),
                             ('ce-native.cudnn-false-gate-failure.console.txt', 'native-urban-cudnn-false-diagnostic.console.txt')]:
        shutil.copyfile(RUN / source_name, EXP / 'raw_logs' / dest)
    python = '/root/miniconda3/envs/beta-clip-official/bin/python'
    checkpoint = candidate['path']
    common = {'cwd': str(EXP.parents[2]), 'env': {'PYTHONPATH': str(EXP.parents[2]), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}}
    write('commands/official-urban-ce.json', {**common, 'argv': [python.replace('/python', '/torchrun'),
        '--standalone', '--nnodes=1', '--nproc-per-node=4', '--max-restarts=0', '-m',
        'experiments.external_baselines.beta_clip_v1.official_urban', '--checkpoint', checkpoint, '--variant', 'ce'],
        'exit_code': 0, 'log': 'raw_logs/official-urban-ce.console.txt'})
    write('commands/native-five-ce.json', {**common, 'argv': [python, '-m',
        'experiments.external_baselines.beta_clip_v1.evaluate_native', '--checkpoint', checkpoint],
        'exit_code': 0, 'log': 'raw_logs/native-five-ce.console.txt'})
    sha = candidate['sha256']
    consistency = f'''# Native CLS consistency: PASS

Official Urban CLS and adapter on identical inputs: image/text raw max absolute difference 0,
scaled scores max absolute difference 0, top-1 mismatches 0/1000 in both directions,
and scaled full-ranking query mismatches 0/1000 in both directions across four ranks.
R1: I2T 88.60%, T2I 89.00%. Raw evidence: [OFFICIAL_URBAN_REPRO_CE.json](OFFICIAL_URBAN_REPRO_CE.json).

The adapter extracts CLS from official `encode_image_by_block` with the saved last/intermediate-block flags,
then uses official `encode_text` for EOS. Both towers are independent. The standard `encode_image`
implementation differs by at most 3.576279e-6 on tested raw CLS values; the adapter follows the exact
official CLS path. No conditioned representation or conditioner forward runs in native evaluation
(a raising forward hook verifies zero calls).

Frozen SAID Urban raw captions: 1000; different_count=0; both caption-list SHA256 values:
`473427e3f0fe30f38e0dbec4fd7f6701d0bbe27ad29ada57bd47883dc1772abb`.
Final SAID frozen metric R1: I2T 88.600004%, T2I 89.000005% (float32 mean).

During the unified-harness gate, disabling cuDNN TF32 convolution shifted one T2I query (89.0→88.9%).
The official runtime defaults to convolution TF32 enabled. Matching that default resolves the difference.
Matmul TF32 remains disabled, weights/activations FP32, no autocast. No data, tokenizer, loss or model
math was changed. Details: [raw_logs/urban-gate-diagnosis.json](raw_logs/urban-gate-diagnosis.json).
This diagnostic failure occurred before the remaining four datasets ran.

Checkpoint: `{sha}`. Strict load: 362/362 tensors, empty missing/unexpected keys,
all loaded tensors exactly equal to the converted fine-tuned checkpoint. Avoiding timm's redundant
pretrained network bootstrap retains no temporary initializer tensors. Official sources are unmodified.
'''
    (EXP / 'NATIVE_CLS_CONSISTENCY.md').write_text(consistency)
    urban_rows = [['CE (measured)', '88.60', '89.00', '85.40', '95.50'],
                  ['BCE (checkpoint absent)', 'N/A', 'N/A', 'N/A', 'N/A']]
    urban_table = table(['Checkpoint', 'CLS I2T %', 'CLS T2I %', 'TCI I2T %', 'TCI T2I %'], urban_rows)
    urban_text = '# Official Urban reproduction\n\nCE published reference: I2T 88.6%, T2I 89.0%. The untouched official evaluator reproduces this pair in CLS: **REPRODUCED**, delta 0 pp.\n\n'+urban_table+'\n\nThe measured TCI path uses the official evaluator settings (`use_model_text_settings=False`, `use_text_eos=True`, `return_first=True`, `return_avg_tci=False`). TCI is query-dependent and excluded from the fair main table and all aggregates. The supplied README pair matches CLS; it is not a separate TCI reference. BCE published I2T/T2I 92.3/91.8 remains untested.\n'
    (EXP / 'OFFICIAL_URBAN_REPRO.md').write_text(urban_text)
    main_rows = [[n, f"{metrics[n]['I2T']['R@1']*100:.6f}", f"{said['metrics'][n]['I2T']['R@1']*100:.6f}",
                    f"{delta[n]['I2T']:+.6f}", f"{metrics[n]['T2I']['R@1']*100:.6f}",
                    f"{said['metrics'][n]['T2I']['R@1']*100:.6f}", f"{delta[n]['T2I']:+.6f}"] for n in NAMES]
    main_table = table(['Dataset', 'β-CLIP CE CLS I2T %', 'SAID I2T %', 'Δ SAID−β pp',
                        'β-CLIP CE CLS T2I %', 'SAID T2I %', 'Δ SAID−β pp'], main_rows)
    score_table = table(['Metric', 'β-CLIP CE CLS %', 'SAID %', 'Δ SAID−β pp'],
        [[k, f'{native["scores"][k]*100:.6f}', f'{said["scores"][k]*100:.6f}', f'{delta_scores[k]:+.6f}']
         for k in ('Score5_R1', 'J_long3', 'J_long')])
    recalls = table(['Model', 'Dataset', 'Direction', 'R@1 %', 'R@5 %', 'R@10 %'],
        [[label, n, d, *[f'{values[n][d]["R@"+str(k)]*100:.6f}' for k in (1,5,10)]]
         for label, values in [('β-CLIP CE CLS', metrics), ('SAID Balanced B/16', said['metrics'])]
         for n in NAMES for d in ('I2T', 'T2I')])
    ce_wins = sum(delta[n][d]<0 for n in NAMES for d in ('I2T','T2I'))
    report = f'''# β-CLIP CE official checkpoint versus SAID Balanced B/16

**CE complete; BCE checkpoint unavailable.** The CE file transferred from the user cloud share
strictly loads the fine-tuned B/16 model and reproduces published Urban CLS 88.60/89.00% (I2T/T2I).
Native adapter embeddings and rankings match the official CLS path exactly on identical inputs.
All five SAID frozen protocols have now been evaluated with query-independent official CLS/EOS.
SAID wins {10-ce_wins}/10 directional R1 comparisons; CE wins {ce_wins}/10.
Score5 difference (SAID−β) is {delta_scores['Score5_R1']:+.6f} percentage points.

## Unified SAID-protocol re-evaluation (fair main result)

{main_table}

{score_table}

Aggregates use unrounded fractions: Score5_R1 averages ten directional R1 values;
J_long3 averages Urban/DOCCI/Long-DCI's six directions; J_long averages Urban/DOCCI's four.
All differences use **SAID minus β-CLIP**. TCI is excluded.

## Published reference and official checkpoint reproduction

Published CE I2T/T2I 88.6/89.0%; published BCE 92.3/91.8%, from the
[pinned official README](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/README.md).
These are reference values, not five-dataset SAID-protocol results. Measured CE:

{urban_table}

The published CE pair matches measured CLS, **REPRODUCED** (0 pp delta).
Measured TCI 85.40/95.50% uses the untouched official evaluator's EOS conditioning settings;
no separate TCI published reference is asserted. BCE is unavailable, not assigned its published
numbers as measured results. CE was completed without waiting for BCE.

## Provenance, loading and actual saved configuration

Source: official author-provided Google Drive checkpoint, ID `1nNbYLlU_1VcbEioLL7yX_rzARAzxD6q_`;
user manually downloaded it, transferred through USTC cloud disk to the evaluation server.
Filename `ce_checkpoint_10.pt`; size {candidate['size_bytes']} bytes; epoch 10;
SHA256 `{sha}`. Official source pinned to `7be4476f84654b0febe7224d5788868d61ecae8b`, clean checkout.

Actual args: `CLIP_VITB16_OPENAI`, context 248, β=0.5, max_concepts=30,
`fg_loss_fn=cls+tcil`, `text_conditioning_mode=attn_pooling_mlp`, EOS/concepts/conditioned patches enabled.
Saved last-block and intermediate-block flags are both false; the official mixed-block CLS path is used.
Checkpoint loss mode is legacy `1_k_positives` with `use_softmax_for_multi_positives=True`,
where the current CE training script uses `k_positives_ce`. This difference is recorded without
rewriting saved args or running a training loss. CE identity combines saved softmax configuration,
model/conditioner structure, declared official transfer provenance and exact Urban CLS reproduction.

Loading uses official state conversion and positional setup plus only removal of leading `module.`.
362 state tensors strictly load, missing_keys=[], unexpected_keys=[], every loaded tensor equals
the converted checkpoint. Fine-tuned vision/text towers and all 11 conditioner tensors are loaded.
Model construction skips a redundant timm pretrained download; the strict full-state equality check
proves no temporary initializer weight survives. No substitute model or partial load is used.

## Protocol and limitations

COCO: 5000 images/25000 captions, sorted image IDs, first five source-order captions,
five positives/image, canonical similarity_chunk=512 and frozen row-wise argsort semantics.
Urban: 1000/1000 diagonal, raw captions equal to official captions (0 differences).
Flickr: frozen test1K manifest, 1000/5000. DOCCI: frozen 5000/5000.
Long-DCI: exact reconstructed 7602/7602 manifest, SHA256
`8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b`;
no DCI Full or paper DCI number is used. Frozen metric functions and positive mappings are reused;
dataset records, caption fingerprints and metric-source SHA256 values are in the raw JSON.

Official `SimpleTokenizer(context_length=248)` is used on unchanged SAID caption strings.
Its original truncation behavior (including no forced EOT replacement for overlong captions)
is retained. Image transform: RGB, 224 bicubic Resize/CenterCrop, OpenAI CLIP normalization.
Precision: FP32/no autocast; matmul TF32 off, cuDNN convolution TF32 on to match official defaults.
The initial Urban gate detected a one-query difference when convolution TF32 was off; this was
diagnosed and resolved before the remaining datasets, without changing model/data/evaluator math.
Native evaluation has zero conditioner forward calls; no concepts, reranking, ensemble or augmentation.

SAID reference is reused: Balanced-Stack-Patch B/16, fusion_lr=2e-4, inclusion_max=1,
view_weights=[1,1,1], sparsity_scale=1, four epochs/4868 updates, native inference.
Full checkpoint SHA256 `{said['checkpoint_sha256']}`;
bare SHA256 `{said['bare_sha256']}`. SAID was not retrained or modified.
CE is the primary closer loss-family comparison: SAID F/P/R use softmax CE alignment, while
β-CLIP CE uses global CE and multi-positive fine-grained CE; their losses are not identical.
This compares released checkpoints with the same raw test protocols and native inference,
not equal training compute: β-CLIP saved training budget is 10 epochs versus SAID's four.

Official L/14 architecture support: yes. Separate released fine-tuned L/14 checkpoint found:
no within pinned README/download-script scope. This transferred checkpoint is B/16; BCE and L/14
fine-tuned weights are not claimed verified.

## Complete Recall@1/5/10

{recalls}

## Evidence and commands

[CHECKPOINT_INVENTORY.json](CHECKPOINT_INVENTORY.json),
[OFFICIAL_URBAN_REPRO_CE.json](OFFICIAL_URBAN_REPRO_CE.json),
[NATIVE_CLS_CONSISTENCY.md](NATIVE_CLS_CONSISTENCY.md),
[CE_SAID_PROTOCOL_RESULTS.json](CE_SAID_PROTOCOL_RESULTS.json),
[environment.json](environment.json), `raw/ce/`, `raw_logs/`, and `commands/`.
Cloud passwords, signed URLs, weights, datasets, cached embeddings and environments are excluded.
Earlier download failures remain historical evidence and do not describe current CE availability.
'''
    (EXP / 'FINAL_COMPARISON.md').write_text(report)
    (EXP / 'BETA_CLIP_COMPARISON.md').write_text(report)
    (EXP / 'README.md').write_text('''# Official β-CLIP CE evaluation: complete

The manually transferred official CE checkpoint reproduces published Urban CLS 88.60/89.00%
(I2T/T2I), passes native consistency, and completes all five frozen SAID protocols.
BCE checkpoint is unavailable and did not block CE. No training or finetuning was performed.

Read [FINAL_COMPARISON.md](FINAL_COMPARISON.md) for every directional R1 difference,
aggregate scores, CLS versus TCI distinction and complete R@1/5/10.
[CE_SAID_PROTOCOL_RESULTS.json](CE_SAID_PROTOCOL_RESULTS.json) stores raw fractions/protocol hashes;
[CHECKPOINT_INVENTORY.json](CHECKPOINT_INVENTORY.json) stores actual args, SHA256 and strict loading.
Commands are structured argv arrays in `commands/official-urban-ce.json` and `commands/native-five-ce.json`.
Run from the SAID worktree using the independent beta-clip-official Python3.10 environment.
The pinned official checkout remains outside this repository and unmodified.

Implementation: `native_adapter.py`, `official_urban.py`, `evaluate_native.py`,
read-only `find_checkpoints.py`; `write_reports.py` publishes small runtime results only.
Old Google Drive timeout logs remain historical evidence, superseded by successful manual CE transfer.
No checkpoint, dataset, embedding cache, cloud access secret or Python environment is committed.
''')
    print(json.dumps({'status': unified['status'], 'scores_percent': {k:v*100 for k,v in native['scores'].items()},
                      'delta_scores_pp': delta_scores}))


if __name__ == '__main__':
    main()
