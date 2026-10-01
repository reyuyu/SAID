"""Final search-space winner, complete recalls, promotion lineage and limitations."""
import json
from pathlib import Path

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import EXP,RUN,DATASETS,load,scores,rank_key


def main():
    state=load(RUN/'state.json')
    assert state.get('best_3651'), 'Final promotions have not completed'
    trials=state['trials'];best=state['best_3651']
    references={}
    ti_path=EXP.parents[2]/'experiments/nest_clip_v1/ti_fast_3epoch_v1/FULL3EPOCH_RESULTS.json'
    ti=load(ti_path)
    ti_metrics=dict(ti['metrics_step3651'],**{'Long-DCI':ti['long_dci_evaluation']['step3651']})
    references['TI-fast@3651']=dict(metrics=ti_metrics,scores=scores(ti_metrics),
                                   checkpoint_sha256=ti['checkpoint_sha256'],bare_sha256=ti['bare_sha256'])
    background_path=Path('/root/lk_projects/SAID-publish-suffix3-3epoch/experiments/results.json')
    background=load(background_path)
    original=next(e for e in background['experiments'] if e['experiment']=='S0-DualMask-Suffix3-U0' and e['steps']==3651)
    mapping={'coco':'COCO','urban':'Urban-1k','flickr_test1k':'Flickr30k-test1k','docci':'DOCCI','long_dci':'Long-DCI'}
    m={mapping[r['protocol']]:{dr:{rec:r['metrics'][rec][dr] for rec in ('R@1','R@5','R@10')}
                              for dr in ('I2T','T2I')} for r in original['results'] if r['protocol'] in mapping}
    assert set(m)==set(DATASETS)
    references['3/0@3651 background']=dict(metrics=m,scores=scores(m),bare_sha256=original['bare_student_sha256'],
                                           source=str(background_path),note='Different objective/initialization lineage, background only; no DCI Full rerun')
    final=trials[best]['budgets']['3651']
    results=dict(title='Balanced hparam search best',search_state=state,best_trial_id=best,best_hparams=trials[best]['hparams'],
                 best_final_checkpoint=final['checkpoint'],best_final_checkpoint_sha256=final['checkpoint_sha256'],
                 best_metrics=final['metrics'],best_scores=final['scores'],references=references,
                 scope='Highest measured score among the requested coordinate-search configurations and promoted budgets, seed0; not global hyperparameter optimality',
                 selection='Raw Score5_R1, then exact-tie J_long3, then exact-tie J_long; no display rounding or improvement threshold',
                 limitations=['All five benchmarks were used for tuning; results are not blind/generalization estimates',
                              '500-step screening can miss late-emerging configurations not promoted',
                              'No multi-seed significance claims',
                              'Legacy B0 final checkpoint/results reused rather than duplicated; legacy final gate diagnostics were not collected'])
    (EXP/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Balanced Hyperparameter Search: Fixed Seed0','',
           '## Winner','',f"**Balanced hparam search best**: `{best}`.",'',
           f"Final checkpoint SHA256: `{final['checkpoint_sha256']}`.",'',
           f"Chosen parameters: `{json.dumps(trials[best]['hparams'],sort_keys=True)}`.",'',
           'This is the highest measured score in the requested coordinate search/promotion procedure, seed0. '
           'It is not a claim of global hyperparameter optimality or statistical significance.','',
           '## Search Rules','',
           'The Balanced network, data, RandomK, backbone/text/pool LR, precision, hard-ST and candidate pool '
           'are frozen. Coordinates: fusion LR5e-5/2e-4; visual mask LR scale0.5/2; normalized view weights '
           '[2,1,1]/[1,1,1.5]; sparsity scale0.75/1.25; inclusion maximum0.5/1.5. Each round evaluates both '
           'new candidates from the shared untrained state, retaining the current best as the third option. '
           'All histories remain eligible for global Top3.','',
           'Default B0@500/@3651 are reused after strict checkpoint/metric verification. Same-trial promotions '
           'restore optimizer moments, all RNG and loader cursor, without resetting the3651 cosine horizon. '
           'The B0 legacy optimizer is split by parameter name with exact moments/step/LR preservation. '
           'Default loss/gradient/next AdamW update regression is exact.','',
           'Ranking uses original Recall floats: Score5_R1 (ten equally weighted R1 directions), then J_long3 '
           '(Urban/DOCCI/Long-DCI), then J_long (Urban/DOCCI). No minimum gain is imposed.','',
           '## Coordinates','', '| Round | Coordinate | Previous best | Candidate IDs | New best |',
           '|---|---|---|---|---|']
    for r in state['rounds']:
        lines.append(f"| {r['round']} | {r['coordinate']} | {r['best_before'][:12]} | "
                     f"{','.join(t[:12] for t in r['participants'][1:])} | {r['best_after'][:12]} |")
    for budget in (500,1217,3651):
        entries=[(tid,t,t['budgets'][str(budget)]) for tid,t in trials.items() if str(budget) in t['budgets']]
        entries.sort(key=lambda e:rank_key(e[2]),reverse=True)
        lines += ['',f'## Leaderboard {budget}','',
                  '| Trial | fusion_lr | visual scale | F/P/R weights | sparsity | inclusion | Score5_R1 % | J_long3 % | J_long % | Reused |',
                  '|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
        for tid,t,result in entries:
            hp=t['hparams'];s=result['scores']
            lines.append(f"| {tid[:12]} | {hp['fusion_lr']} | {hp['visual_mask_lr_scale']} | {hp['view_weights']} | "
                         f"{hp['sparsity_scale']} | {hp['inclusion_max']} | {s['Score5_R1']*100:.6f} | "
                         f"{s['J_long3']*100:.6f} | {s['J_long']*100:.6f} | {result['reused']} |")
    lines += ['', '## Final References','', '| Model | Score5_R1 % | J_long3 % | J_long % |', '|---|---:|---:|---:|']
    for label,result in [('Search best@3651',final),*references.items()]:
        s=result['scores'];lines.append(f"| {label} | {s['Score5_R1']*100:.6f} | {s['J_long3']*100:.6f} | {s['J_long']*100:.6f} |")
    lines += ['', '## Complete Native Recall','',
              'Each table includes all successful500 trials, Top3@1217 and final3651 results. Values are percentages. '
              'Native bare-student normalized image/text inner products only; no training mask or rerank.','']
    for budget in (500,1217,3651):
        for tid,trial in trials.items():
            record=trial['budgets'].get(str(budget))
            if record is None:continue
            lines += [f'### {tid[:12]} @{budget}','', '| Dataset | Direction | R1 | R5 | R10 |','|---|---|---:|---:|---:|']
            for ds in DATASETS:
                for direction in ('I2T','T2I'):
                    m=record['metrics'][ds][direction]
                    lines.append('| '+' | '.join([ds,direction,*[f'{m["R@"+str(k)]*100:.3f}' for k in (1,5,10)]])+' |')
    lines += ['', '## Diagnostics and Failures','',
              'Actual LR and gradients are logged for backbone, text_mask_and_shared_pool, visual_mask, fusion_adapter. '
              'Requested gate moments/quantiles/saturation, F/P/R differences, modality norms and replaced-image '
              'mask switching at1/100/200/500/1217/2000/3000/3651 are preserved per successful stage. '
              'Diagnostics do not enter selection. All full-step measurements include real DataLoader and ordinary logs; '
              'checkpoint writes remain separate. Full candidate/tail handling and finite gradients are validated.','']
    failures={tid:t.get('error') for tid,t in trials.items() if t.get('status')=='failed'}
    lines.append('Failures: '+(json.dumps(failures) if failures else 'none.'))
    lines += ['', '## Artifacts','',
              'Parameter JSON hashes are trial IDs. SEARCH_STATE.json records runtime commits, commands/exit codes, '
              'checkpoint/student hashes, metric JSON, stage costs, promotion lineage and all failure statuses. '
              'leaderboard_500.csv / leaderboard_1217.csv / leaderboard_3651.csv preserve original score floats. '
              'Raw evaluator JSON and compact evidence are committed here; large training checkpoints, RNG/optimizer '
              'payloads, full token logs, data and caches remain server-local.','',
              'Only the five authorized native protocols are used: COCO5000/25000, Urban1000/1000, Flickr1000/5000, '
              'DOCCI5000/5000, reconstructed Long-DCI7602/7602. Image batch64; COCO chunk512. No DCI Full run. '
              'All five sets were used in selection, so these are tuning results on previously examined benchmarks.', '']
    (EXP/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(best=best,parameters=trials[best]['hparams'],scores=final['scores']),indent=2))


if __name__=='__main__':main()
