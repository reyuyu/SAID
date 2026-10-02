"""Explicit incomplete/resource-limited states and frozen five-dataset comparisons."""
import json

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import DATASETS,REPO


def write_report(state,output):
    references=json.loads((REPO/'experiments/nest_clip_v1/three_followup_v1/RESULTS.json').read_text())
    b16=[r for r in references['evaluations'] if r['name']=='Four-epoch fusion-only']
    assert {r['updates'] for r in b16}=={3651,4868}
    budgets=state['trials'][state['l14_trial']]['budgets']
    limit=state.get('resource_authorization',{}).get('approved_regular_update_limit_seconds',3)
    lines=['# Balanced L14: Fixed Four-Epoch Migration','',
           f"Status: `{state['status']}`. Stage: `{state.get('stage')}`.",'',
           'OpenAI ViT-L/14 at224, text context248, seed0. Faithful Balanced coefficients: '
           'fusion_lr2e-4, visual_mask_lr_scale1, view_weights[1,1,1], sparsity_scale1, inclusion_max1. '
           'Horizon4868 from a new L14 step0, full4x256 direct logical batch, full1024 candidates '
           '(padded sampler tail720), original RandomK and loss definitions.','',
           '## Initialization and Correctness','',
           f"L14 step0: `{state['initialization']['initial_checkpoint']}`.",
           f"L14 step0 SHA256: `{state['initialization']['initial_checkpoint_sha256']}`.",
           f"Official pretrained URL: {state['initialization']['provenance']['pretrained_url']}",
           f"Official pretrained SHA256: `{state['initialization']['provenance']['pretrained_sha256']}`.",'',
           'Architecture/native/export/position checks use the real downloaded L14 weights. '
           'Ten two-rank edge cases and local resume tests use dimension-aware tiny native encoders with '
           'real768-channel,12-head MaskNet blocks and256 patches. A separate actual full-L14 two-rank '
           'reference on the fixed first two valid training samples checks every named gradient and AdamW '
           'update at the unchanged tolerances, using identical per-sample FP32 kernel shapes. '
           'B16 native inference and default loss/gradient/AdamW regressions are preserved.','',
           'Additional unnormalized Gaussian-image/short-caption full-L14 stress tests exceeded the original '
           'convolution-gradient tolerance and are preserved as failed evidence. They were not hidden or '
           'made to pass by increasing tolerances. Actual-training-input full-model tests and random-input '
           'dimension-aware fixtures passed; the numerical scope is explicit.','',
           '## Resource Gate','',
           'Each attempt measures5 warmup plus30 complete real-DataLoader updates, slowest rank. '
           f'The approved limit is{limit:g}s per regular update and65GiB allocated per GPU. '
           'Initialization, checkpoint writes and final agreement checks are separate.']
    if state.get('resource_authorization'):
        lines += ['','The user explicitly approved updates within5s and instructed continuation. '
                  'The original3s failure remains below as historical evidence; a fresh5s gate is required.']
    for attempt in state['resources']:
        lines += ['',f"### {attempt['label']}",'',f"Passed: `{attempt['passed']}`. OOM: `{attempt['oom']}`.",'',
                  '```json',json.dumps(attempt['acceptance'],indent=2),'```']
    if state['status']=='resource_stopped':
        lines += ['','Formal4868 training and all four evaluation nodes were not run: resource acceptance failed. '
                  'No B16 score is presented as an L14 result. No new speed budget, smaller candidate pool, '
                  'different backbone or Gradient Cache path was silently substituted.']
    if state.get('error'):
        lines += ['',f"Failure: `{state['error']}`."]
    lines += ['','## Native Scores','','| Model | Updates | Horizon | Score5_R1 % | J_long3 % | J_long % |',
              '|---|---:|---:|---:|---:|---:|']
    for record in sorted(b16,key=lambda r:r['updates']):
        scores=record['scores']
        lines.append(f"| B16 reference | {record['updates']} | 4868 | {scores['Score5_R1']*100:.6f} | "
                     f"{scores['J_long3']*100:.6f} | {scores['J_long']*100:.6f} |")
    for stop in (0,500,3651,4868):
        if str(stop) not in budgets:
            lines.append(f'| L14 | {stop} | 4868 | not run | not run | not run |')
            continue
        record=budgets[str(stop)];scores=record['scores']
        lines.append(f"| L14 | {stop} | 4868 | {scores['Score5_R1']*100:.6f} | "
                     f"{scores['J_long3']*100:.6f} | {scores['J_long']*100:.6f} |")
    for stop in (0,500,3651,4868):
        if str(stop) not in budgets:
            continue
        record=budgets[str(stop)]
        lines += ['',f'### L14 @{stop}','',
                  f"Full checkpoint: `{record['checkpoint']}`; SHA256 `{record['checkpoint_sha256']}`.",
                  f"Bare student: `{record['student']}`; SHA256 `{record['bare_sha256']}`.",'',
                  '| Dataset | Direction | R1 % | R5 % | R10 % |','|---|---|---:|---:|---:|']
        for dataset in DATASETS:
            for direction in ('I2T','T2I'):
                m=record['metrics'][dataset][direction]
                lines.append('| '+' | '.join([dataset,direction,*[f'{m[f"R@{k}"]*100:.6f}' for k in (1,5,10)]])+' |')
    if '4868' in budgets:
        parent=next(r for r in b16 if r['updates']==4868)
        gain=100*(budgets['4868']['scores']['Score5_R1']-parent['scores']['Score5_R1'])
        lines += ['',f'Final L14@4868 minus B16@4868 Score5_R1: {gain:+.6f} pp.']
    lines += ['','The prespecified final result is L14@4868. best_observed is selected only among500/3651/4868 '
              'by raw Score5_R1 and is explicitly benchmark-selected; no cross-checkpoint dataset mixing. '
              'The auxiliary mask/gate branches are not used in native inference.','',
              '## Reproduction','','```bash','cd /root/lk_projects/SAID-balanced-l14-4epoch-v1',
              "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_l14_4epoch_v1.prepare",
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_l14_4epoch_v1.run --launch',
              '```','',
              'Exact commands, hashes, code commits and exit statuses are retained in evidence and runtime state. '
              'Weights, optimizer/RNG checkpoints, data and caches remain local. Five frozen native protocols '
              'only, including reconstructed Long-DCI7602; no DCI Full, extra seeds, extra epochs or tuning trials.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'RESULTS.json').write_text(json.dumps(dict(state=state,b16_reference=b16,
        final=budgets.get('4868'),best_observed_updates=state.get('best_observed_updates')),indent=2)+'\n')
