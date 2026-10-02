"""Native five-dataset comparison, whole-sentence sampling stats and bounded conclusions."""
import json


DATASETS=('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')


def write_report(state,output):
    base=state['baseline'];result=state['trials'][state['rdrop_trial']]['budgets'].get('500')
    lines=['# R-SentenceDrop: Ordered Semantic Subsets at500','',
           f"Status: `{state['status']}`. Stage: `{state.get('stage')}`.",'',
           'ViT-B/16,seed0,4x256,context248,encoder BF16/other paths FP32,encoder checkpoint ON,pair OFF. '
           'Stop500 with horizon4868. Same best coefficients and common step0. Only R construction differs: '
           'sample q uniformly from1..m inclusive, sample q suffix sentences without replacement, sort their '
           'original indices, join with the original separator and tokenize as compact text. No shuffle, '
           'in-place PAD, extra attention mask, view, model module or loss.','',
           'Sentence subset RNG is independent: SHA256(sampling_seed:epoch:sample_id:r_sentence_drop_v1). '
           'Original RandomK, F/P, image preprocessing and sampler are unchanged.','',
           f"Reused matched baseline checkpoint SHA256: `{base['checkpoint_sha256']}`.",
           f"Reused matched bare student SHA256: `{base['bare_sha256']}`.",'',
           'Baseline already has all five frozen evaluations; it is not retrained. RMask is only a historical '
           'negative reference, not the initialization or sampling implementation.','',
           '## Main Scores','','| Model | Horizon | Score5_R1 % | J_long3 % | J_long % |',
           '|---|---:|---:|---:|---:|']
    for label,record in [('Matched old-R@500',base),('R-SentenceDrop@500',result)]:
        values=[f'{record["scores"][key]*100:.6f}' for key in ('Score5_R1','J_long3','J_long')] if record else ['not completed']*3
        lines.append('| '+' | '.join([label,'4868',*values])+' |')
    if state.get('delta_scores_pp'):
        lines.append('| Delta pp | - | '+' | '.join(f'{state["delta_scores_pp"][k]:+.6f}'
                     for k in ('Score5_R1','J_long3','J_long'))+' |')
    lines += ['',f"Historical RMask@500 Score5_R1: {state['historical_rmask']['Score5_R1']*100:.6f}% (negative reference)."]
    for label,record in [('Reused Matched Baseline',base),('R-SentenceDrop',result)]:
        if not record:continue
        lines += ['',f'## {label} Complete Recall','',
                  '| Dataset | Direction | R1 % | R5 % | R10 % |','|---|---|---:|---:|---:|']
        for ds in DATASETS:
            for direction in ('I2T','T2I'):
                m=record['metrics'][ds][direction]
                lines.append('| '+' | '.join([ds,direction,*[f'{m[f"R@{k}"]*100:.6f}' for k in (1,5,10)]])+' |')
    if result:
        lines += ['','## Ten R1 Changes','','| Dataset | Direction | RDrop minus old-R (pp) |','|---|---|---:|']
        for ds in DATASETS:
            for direction in ('I2T','T2I'):
                delta=100*(result['metrics'][ds][direction]['R@1']-base['metrics'][ds][direction]['R@1'])
                lines.append(f'| {ds} | {direction} | {delta:+.6f} |')
        lines += ['','## Sentence Subset Statistics','','```json',
                  json.dumps(result.get('sentence_drop_statistics',{'status':'pending aggregation'}),indent=2),'```',
                  '','## Resource Statistics','','```json',json.dumps(result.get('resource_summary',{'status':'pending aggregation'}),indent=2),'```',
                  '',f"Full checkpoint: `{result['checkpoint']}`; SHA256 `{result['checkpoint_sha256']}`.",
                  f"Bare student: `{result['student']}`; SHA256 `{result['bare_sha256']}`.",'',
                  '## Conclusion','']
        if result['scores']['Score5_R1']>base['scores']['Score5_R1']:
            lines.append('R-SentenceDrop@500 improves the matched Balanced baseline and is worth full4-epoch confirmation. No full run is started by this task.')
        else:
            lines.append('R-SentenceDrop does not improve the matched baseline.')
    if state.get('error'):lines += ['','Preserved failure: `'+state['error']+'`.']
    lines += ['','## Evidence and Reproduction','',
              'Correctness and readable real-sample proofs are in evidence/. Statistics come from every '
              'valid sample over all500 updates/ranks. RDrop trace hashes match the reused baseline for '
              'every sample/F/P/K and original fixed-first reference stream. Model/loss sources stay '
              'byte-identical to fa19d12. Raw Recall fractions and strict export/acceptance evidence are retained.','',
              'Five native student protocols only, including reconstructed Long-DCI7602; no DCI Full, '
              'mask/gate inference, reranking or ensemble. No significance claim or rounded-score selection. '
              'No baseline retraining, promotion, extra q distributions, extra values or additional seeds.','',
              '```bash','cd /root/lk_projects/SAID-balanced-rdrop-500-v1',
              "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.verify",
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_500_v1.run --launch','```','',
              'Exact runtime commits/commands/exits, native JSON and sampler evidence are preserved; '
              'large checkpoints/data/cache stay server-local. Stop after this single500 evaluation.']
    (output/'RDROP_500_REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'RESULTS.json').write_text(json.dumps(state,indent=2)+'\n')
