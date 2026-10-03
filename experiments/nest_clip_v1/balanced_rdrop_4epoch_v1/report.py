"""Four-epoch native comparison with a safe intermediate evaluation callback."""
import json

DATASETS = ('COCO', 'Urban-1k', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI')
SCORES = ('Score5_R1', 'J_long3', 'J_long')


def write_report(state, output):
    baseline = state['baseline']
    result = state['trials'][state['rdrop_trial']]['budgets'].get('4868')
    lines = ['# R-SentenceDrop: Four-Epoch Confirmation', '',
             f"Status: `{state['status']}`; stage: `{state.get('stage')}`.", '',
             'User-authorized continuation of the completed R-SentenceDrop500 experiment. '
             'Restore its full checkpoint, AdamW state, all four rank RNG states and loader '
             'cursor at update500; continue to4868 with the original horizon4868. The complete '
             'trajectory starts at the same shared step0 as the compact old-R baseline.', '',
             'ViT-B/16,224,context248,seed0,sampling_seed0,4x256,global1024,accumulation1,workers8/rank, '
             'BF16 encoders and FP32 parameters/mask/gate/scoring/loss. Frozen coefficients '
             'fusion_lr=2e-4,visual_mask_lr_scale=1,view_weights=[1,1,1],sparsity_scale=1,inclusion_max=1. '
             'Encoder checkpoint ON,pair OFF,image/text chunks128. Training/data/model source files '
             'are identical to the verified500 run. Only the authorized stopping point changes.', '',
             'R remains an ordered subset of complete suffix sentences, with uniform q in1..m '
             'and independent salted SHA256 sampling. No additional tuning, seed, view or loss.', '',
             '## Main Scores', '', '| Model | Score5_R1 % | J_long3 % | J_long % |',
             '|---|---:|---:|---:|']
    for name, record in [('Matched old-R@4868', baseline), ('R-SentenceDrop@4868', result)]:
        values = [f'{record["scores"][k]*100:.6f}' for k in SCORES] if record else ['pending']*3
        lines.append('| '+' | '.join([name, *values])+' |')
    if result:
        lines.append('| Delta pp | '+' | '.join(
            f'{100*(result["scores"][k]-baseline["scores"][k]):+.6f}' for k in SCORES)+' |')
    for name, record in [('Reused Old-R Baseline', baseline), ('R-SentenceDrop', result)]:
        if record is None:
            continue
        lines += ['', f'## {name}: All Recall', '',
                  '| Dataset | Direction | R@1 % | R@5 % | R@10 % |', '|---|---|---:|---:|---:|']
        for ds in DATASETS:
            for dr in ('I2T', 'T2I'):
                values = [f'{record["metrics"][ds][dr][f"R@{k}"]*100:.6f}' for k in (1, 5, 10)]
                lines.append('| '+' | '.join([ds, dr, *values])+' |')
    if result:
        lines += ['', '## Ten R@1 Changes', '', '| Dataset | Direction | Delta pp |', '|---|---|---:|']
        for ds in DATASETS:
            for dr in ('I2T', 'T2I'):
                delta = 100*(result['metrics'][ds][dr]['R@1']-baseline['metrics'][ds][dr]['R@1'])
                lines.append(f'| {ds} | {dr} | {delta:+.6f} |')
        for key in ('sentence_drop_statistics', 'resource_summary', 'stream_comparison'):
            lines += ['', '## '+key, '', '```json',
                      json.dumps(result.get(key, {'status': 'pending aggregation'}), indent=2), '```']
        lines += ['', '## Conclusion', '',
                  ('R-SentenceDrop@4868 improves the matched old-R four-epoch baseline.'
                   if result['scores']['Score5_R1'] > baseline['scores']['Score5_R1'] else
                   'R-SentenceDrop@4868 does not improve the matched old-R four-epoch baseline.'),
                  'Single seed only; no statistical significance claim. Stop after these five native evaluations.']
    lines += ['', '## Provenance', '',
              f"Parent RDrop500 full SHA256: `{state['parent500']['checkpoint_sha256']}`.",
              f"Baseline4868 full SHA256: `{baseline['checkpoint_sha256']}`.",
              f"Baseline4868 bare SHA256: `{baseline['bare_sha256']}`."]
    if result:
        lines += [f"RDrop4868 full SHA256: `{result['checkpoint_sha256']}`.",
                  f"RDrop4868 bare SHA256: `{result['bare_sha256']}`."]
    if state.get('error'):
        lines += ['', 'Preserved failure: `'+state['error']+'`.']
    lines += ['', 'Preflight, raw five-dataset JSON, strict export verification, exact commands/commits '
              'and acceptance evidence are in evidence/. Baseline is reused without retraining. '
              'Native normalized image/text inner product only; reconstructed Long-DCI7602, no DCI Full. '
              'Full weights, data and per-update logs remain local.', '', '## Reproduction', '', '```bash',
              'cd /root/lk_projects/SAID-balanced-rdrop-4epoch-v1',
              "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.verify",
              '/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.run --launch', '```', '']
    (output/'RDROP_4EPOCH_REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
    (output/'RESULTS.json').write_text(json.dumps(state, indent=2)+'\n')
    if result:
        for key in ('sentence_drop_statistics', 'resource_summary', 'stream_comparison'):
            if key in result:
                (output/(key+'.json')).write_text(json.dumps(result[key], indent=2)+'\n')
