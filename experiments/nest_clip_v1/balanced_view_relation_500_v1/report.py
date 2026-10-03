"""Matched native500 scores plus relation diagnostics, with safe intermediate publication."""
import json

DATASETS=('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')
SCORES=('Score5_R1','J_long3','J_long')
METRICS=('inc','sib_loss','relation_loss','P_violation_rate','R_violation_rate',
         'P_violation_magnitude','R_violation_magnitude','c_PP_mean','c_RP_mean','c_RR_mean','c_PR_mean',
         'P_cosine_preference','R_cosine_preference','oe_iou','F_keep_ratio','O_keep_ratio','E_keep_ratio')


def write_report(state,output):
    base=state['baseline'];result=state['trials'][state['relation_trial']]['budgets'].get('500')
    lines=['# Balanced View-Relation500','','Status: `'+state['status']+'`; stage: `'+str(state.get('stage'))+'`.','',
      'Single variable: add0.5*[ReLU(c_RP-c_PP)+ReLU(c_PR-c_RR)] to the original inclusion term, '
      'using unscaled FP32 cosine with normalize eps1e-6. Only cross masks detach; correct masks '
      'retain Hard-ST gradients and z/t keep gradients. Reuse positive masks and cached existing '
      'P/R encoder outputs. No extra encoder, mask generation, model module, fourth view, '
      'transpose constraints, margin or CE denominator change.','',
      'Frozen compact old-R Balanced recipe: B16,224,context248,seed0,sampling_seed0,4x256/global1024, '
      'accum1,workers8,encoder checkpoint ON,pair OFF,chunks128x128. Original audited runtime; '
      'no cancelled runtime optimizations. fusion2e-4,visual scale1,view1:1:1,sparse1,inclusion_max1. '
      'The original min(1,completed/200) scales inclusion+sibling together. H4868, stop500.','',
      'Run a5-step smoke, then independently start formal500 from the shared step0, never smoke '
      'or an old500 checkpoint. Reuse the previously completed matched baseline.','',
      '## Main Scores','','| Model | Score5_R1 % | J_long3 % | J_long % |','|---|---:|---:|---:|']
    for name,record in [('Matched Balanced old-R@500',base),('View-Relation@500',result)]:
        values=[f'{100*record["scores"][key]:.6f}' for key in SCORES] if record else ['pending']*3
        lines.append('| '+' | '.join([name,*values])+' |')
    if result:lines.append('| Delta pp | '+' | '.join(f'{100*(result["scores"][k]-base["scores"][k]):+.6f}' for k in SCORES)+' |')
    for name,record in [('Matched Baseline',base),('View-Relation',result)]:
        if not record:continue
        lines+=['','## '+name+' Complete Recall','','| Dataset | Direction | R@1 % | R@5 % | R@10 % |','|---|---|---:|---:|---:|']
        for ds in DATASETS:
            for dr in ('I2T','T2I'):
                lines.append('| '+' | '.join([ds,dr,*[f'{100*record["metrics"][ds][dr][f"R@{k}"]:.6f}' for k in (1,5,10)]])+' |')
    if result:
        lines+=['','## R@1 Changes','','| Dataset | Direction | Delta pp |','|---|---|---:|']
        for ds in DATASETS:
            for dr in ('I2T','T2I'):
                lines.append(f'| {ds} | {dr} | {100*(result["metrics"][ds][dr]["R@1"]-base["metrics"][ds][dr]["R@1"]):+.6f} |')
        diagnostics=result.get('relation_diagnostics')
        lines+=['','## Relation Diagnostics','']
        if diagnostics:
            lines+=['| Metric | step1 | step100 | step200 | step500 | last50 |','|---|---:|---:|---:|---:|---:|']
            for key in METRICS:
                lines.append('| '+' | '.join([key,*[f'{diagnostics[col][key]:.8f}' for col in ('1','100','200','500','last50')]])+' |')
        else:lines.append('Pending final diagnostic aggregation.')
        lines+=['','## Resource Summary','','```json',json.dumps(result.get('resource_summary',{'status':'pending aggregation'}),indent=2),'```','',
                '## Conclusion','',('View-Relation@500 improves the matched baseline and is worth full 4-epoch confirmation.'
                  if result['scores']['Score5_R1']>base['scores']['Score5_R1'] else 'View-Relation@500 does not improve the matched baseline.'),
                'No statistical significance claim. Violation reduction alone does not qualify a better model. '
                'Stop after the five500-step native evaluations; no full4868 run, coefficient/margin/constraint variants or seeds.']
    lines+=['','## Correctness, Guard and Provenance','',
      'Unit tests cover no violation, isolated P/R mask gradient directions, equality at zero margin, '
      'coefficient0 bitwise loss/all gradients/AdamW recovery, no extra encoder calls and F-only fallback. '
      'A four-rank explicit full-global reference checks normal, sparse, zero-valid, V1 and tail cases. '
      'Global means use sum_valid then the original world/valid_count DDP scaling; no repeated averaging.','',
      'Frozen collapse guard:25-update running means; after8 consecutive bad windows stop if '
      'P/R IoU < max(0.15,0.25*matched-baseline-window-IoU), or any positive keep ratio outside(0.02,0.995). '
      'Guard only stops and records, never retunes loss, coefficient, masks or valid samples.','',
      f"Reused baseline full SHA256: `{base['checkpoint_sha256']}`.",
      f"Reused baseline bare SHA256: `{base['bare_sha256']}`."]
    if result:lines += [f"New full SHA256: `{result['checkpoint_sha256']}`.",f"New bare SHA256: `{result['bare_sha256']}`."]
    if state.get('error'):lines+=['','Preserved stop/failure: `'+state['error']+'`.']
    lines+=['','Strict bare export and all five frozen native protocols use normalized image/full-text '
      'embeddings and plain inner product only. Reconstructed Long-DCI7602, no DCI Full, training mask, '
      'gate, sibling score, reranking or ensemble in inference. Raw JSON and small evidence are here; '
      'large checkpoints/data/cache remain on the server.','', '## Reproduction','','```bash',
      'cd /root/lk_projects/SAID-balanced-view-relation-500-v1',
      "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.verify",
      '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.run --launch','```']
    (output/'VIEW_RELATION_500_REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
    (output/'RESULTS.json').write_text(json.dumps(state,indent=2)+'\n')
    if result:
        for key in ('relation_diagnostics','resource_summary'):
            if key in result:(output/(key+'.json')).write_text(json.dumps(result[key],indent=2)+'\n')
