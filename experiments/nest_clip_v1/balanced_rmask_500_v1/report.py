"""Fair horizon4868 baseline, raw recalls, R1 deltas and position evidence."""
import json


DATASETS=('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')


def write_report(state,output):
    lines=['# Balanced RMask: Single-Variable500 Experiment','',
           f"Status: `{state['status']}`. Stage: `{state.get('stage')}`.",'',
           'ViT-B/16,224,context248,seed0,full4x256. Only the R tensor construction differs: '
           'compact suffix versus F clone with exact BPE prefix-plus-separator positions replaced by PAD0. '
           'SOT, suffix IDs/positions, F EOT and trailing PAD remain unchanged. '
           'No new attention mask, view, architecture, loss or optimizer coefficient.','',
           'Fixed coefficients: fusion_lr2e-4, visual_mask_lr_scale1, weights[1,1,1], '
           'sparsity_scale1, inclusion_max1; four-epoch horizon4868 but stop at500. '
           'Each role has independent5-step smoke and a fresh formal start from the same common step0.','',
           'The exact prior four-epoch run has no step500 checkpoint; one matched old-R baseline is run '
           'as explicitly authorized. Historical69.990027% at horizon3651 is not the matched baseline.','',
           '## Main Comparison','','| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |',
           '|---|---:|---:|---:|---:|']
    for role,label in [('baseline','Balanced matched old-R@500'),('rmask','Balanced RMask@500')]:
        result=state['roles'].get(role,{}).get('result')
        values=[f'{result["scores"][key]*100:.6f}' for key in ('Score5_R1','J_long3','J_long')] if result else ['not completed']*3
        lines.append('| '+' | '.join([label,'4868',*values])+' |')
    if state.get('delta_scores_pp'):
        lines.append('| Delta (pp) | - | '+' | '.join(f'{state["delta_scores_pp"][key]:+.6f}' for key in ('Score5_R1','J_long3','J_long'))+' |')
    for role,label in [('baseline','Matched old-R'),('rmask','RMask')]:
        record=state['roles'].get(role,{})
        result=record.get('result')
        if not result:continue
        lines += ['',f'## {label} Complete Recall','',
                  f"Full checkpoint SHA256: `{result['checkpoint_sha256']}`.",
                  f"Bare-student SHA256: `{result['bare_sha256']}`.",'',
                  '| Dataset | Direction | R1 % | R5 % | R10 % |','|---|---|---:|---:|---:|']
        for ds in DATASETS:
            for direction in ('I2T','T2I'):
                m=result['metrics'][ds][direction]
                lines.append('| '+' | '.join([ds,direction,*[f'{m[f"R@{k}"]*100:.6f}' for k in (1,5,10)]])+' |')
        lines += ['','### Resource Evidence','','```json',json.dumps(result['resource_summary'],indent=2),'```']
        if role=='rmask':
            lines += ['','### Absolute-Position Statistics','','```json',json.dumps(result['position_summary'],indent=2),'```']
    if all(state['roles'].get(role,{}).get('result') for role in ('baseline','rmask')):
        base=state['roles']['baseline']['result'];mask=state['roles']['rmask']['result']
        lines += ['','## Ten R1 Deltas','','| Dataset | Direction | RMask minus matched old-R (pp) |','|---|---|---:|']
        for ds in DATASETS:
            for direction in ('I2T','T2I'):
                d=100*(mask['metrics'][ds][direction]['R@1']-base['metrics'][ds][direction]['R@1'])
                lines.append(f'| {ds} | {direction} | {d:+.6f} |')
        difference=mask['scores']['Score5_R1']-base['scores']['Score5_R1']
        outcome='RMask@500 is better' if difference>0 else 'RMask@500 is worse' if difference<0 else 'Essentially tied'
        lines += ['','## Conclusion','',outcome+'. Fixed seed0, measured raw Score5_R1; no rounding-based selection or statistical-significance claim.']
        if difference>0:lines += ['','Worth confirming with a full4868 run; this task does not start it.']
    if state.get('error'):lines += ['','Failure preserved: `'+state['error']+'`.']
    lines += ['','## Evidence and Reproduction','',
              'evidence/real-samples.json preserves readable sentence/K/boundary/EOT and original/late suffix '
              'position evidence. Unit tests verify split extremes, BPE punctuation,248-token limits, '
              'F-only fallback and internal-PAD encoding. RMask500 trace hashes compare sample/F/P/K '
              'and original fixed-first reference streams to the matched baseline on every step/rank.','',
              'Each arm preserves config, command/exit records, acceptance, strict native export and raw '
              'five-dataset evaluator JSON. Only native normalized student embeddings are used; no mask '
              'or reranking, no DCI Full. Both roles stop at500, with no automatic promotion or new values.','',
              '```bash','cd /root/lk_projects/SAID-balanced-rmask-500-v1',
              "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.verify",
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rmask_500_v1.run --launch','```','',
              'Large weights, data and full token logs remain server-local. Code/configs/reports and '
              'compact evidence are committed to the independent RMask branch.']
    (output/'RMASK_500_REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'RESULTS.json').write_text(json.dumps(state,indent=2)+'\n')
