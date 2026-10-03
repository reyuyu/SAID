"""Fixed long guard, Score5 and deterministic near-tie decision after four arms."""
import json
from pathlib import Path
import statistics
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP,RUN,ARMS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import dump,load,now

GUARD=.73403324
BASE_SCORE=.69900394
DATASETS=['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']

def add_short(record):
    scores=dict(record['scores']);scores['Short4_R1']=statistics.fmean(record['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I']);return scores


def selection(arms,baseline_score=BASE_SCORE):
    complete={a:r for a,r in arms.items() if r['status']=='COMPLETE'}
    eligible={a:r for a,r in complete.items() if r['scores']['J_long3']>=GUARD}
    ordered=sorted(complete,key=lambda a:(complete[a]['scores']['Score5_R1'],a),reverse=True)
    best=ordered[0] if ordered else None
    champion=None;near=[]
    if eligible:
        top=max(r['scores']['Score5_R1'] for r in eligible.values())
        near=[a for a,r in eligible.items() if top-r['scores']['Score5_R1']<.0005]
        champion=max(near,key=lambda a:(eligible[a]['scores']['J_long3'],eligible[a]['metrics']['Urban-1k']['T2I']['R@1'],eligible[a]['metrics']['Long-DCI']['T2I']['R@1'],eligible[a]['scores']['Short4_R1'],a))
    if not eligible:status='NO_SAFE_WINNER'
    elif any(r['scores']['Score5_R1']>baseline_score for r in eligible.values()):status='WINNER'
    elif any(r['scores']['Short4_R1']>add_short(load(EXP/'BASELINE.json'))['Short4_R1'] or any(r['metrics'][ds][dr]['R@1']>load(EXP/'BASELINE.json')['metrics'][ds][dr]['R@1'] for ds in DATASETS for dr in ['I2T','T2I']) for r in complete.values()):status='TRADEOFF'
    else:status='NEGATIVE'
    return {'status':status,'champion':champion,'eligible_arms':list(eligible),'best_observed_Score5_arm':best,
      'Score5_ranking':ordered,'long_guard_raw_fraction':GUARD,'long_guard_percent':100*GUARD,
      'near_tie_Score5_distance_pp':.05,'near_tie_candidates':near,
      'tie_definition':'Within strictly less than0.05pp of highest eligible Score5, compare J_long3,Urban T2I,Long T2I,Short4 lexicographically; arm ID final deterministic fallback.',
      'overall_improvement':bool(eligible and any(r['scores']['Score5_R1']>baseline_score for r in eligible.values())),
      'full_training_recommendation':champion or best,'full_training_launched':False}


def generate():
    baseline=load(EXP/'BASELINE.json');previous=load(EXP/'PREVIOUS_SUMMARY_RANDOM_DETAIL.json')
    arms={a:load(EXP/directory/'RESULTS.json') for a,(directory,*_) in ARMS.items()}
    decision=selection(arms,baseline['scores']['Score5_R1'])
    result={'status':decision['status'],'completed_at':now(),'units':'raw fractions in JSON; percentages and percentage-point deltas in Markdown','baseline':baseline,'previous':previous,'arms':arms,'selection':decision}
    dump(EXP/'SEARCH_RESULTS.json',result);dump(EXP/'SELECTION.json',decision)
    table=['| Model | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    for name,r in [('RandomK baseline',baseline),('Summary+RandomDetail previous',previous)]+[(a,arms[a]) for a in ARMS]:
        if r.get('status') in ['RESOURCE_FAIL','FAILED']:table.append(f'| {name}: {r["status"]} | — | — | — | — |');continue
        s=add_short(r);table.append('| '+name+' | '+' | '.join(f'{s[k]*100:.6f}' for k in ['Score5_R1','J_long3','J_long','Short4_R1'])+' |')
    five=['| Arm | Dataset | I2T R1 | T2I R1 | Δ RandomK I/T pp | Δ previous I/T pp |','|---|---|---:|---:|---|---|']
    recalls=['| Arm | Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|---|']
    for a,r in arms.items():
        if r['status']!='COMPLETE':continue
        for ds in DATASETS:
            m=r['metrics'][ds]
            deltas=lambda ref:' / '.join(f'{100*(m[dr]["R@1"]-ref["metrics"][ds][dr]["R@1"]):+.6f}' for dr in ['I2T','T2I'])
            five.append(f'| {a} | {ds} | {m["I2T"]["R@1"]*100:.6f} | {m["T2I"]["R@1"]*100:.6f} | {deltas(baseline)} | {deltas(previous)} |')
            recalls.append('| '+a+' | '+ds+' | '+' | '.join(' / '.join(f'{m[dr][k]*100:.6f}' for k in ['R@1','R@5','R@10']) for dr in ['I2T','T2I'])+' |')
    analysis=['# Short / long analysis','',f'Status: **{decision["status"]}**. Eligible: {decision["eligible_arms"]}. Guard J_long3 ≥73.403324%.','',*table,'']
    for a,r in arms.items():
        if r['status']!='COMPLETE':analysis +=[f'{a}: {r["status"]}.', ''];continue
        s=r['scores'];ref=baseline if a=='A' else previous
        analysis +=[f'{a}: vs {"RandomK" if a=="A" else "previous Summary+RandomDetail"}, ΔScore5 {100*(s["Score5_R1"]-ref["scores"]["Score5_R1"]):+.6f}pp, ΔJ_long3 {100*(s["J_long3"]-ref["scores"]["J_long3"]):+.6f}pp, ΔShort4 {100*(s["Short4_R1"]-add_short(ref)["Short4_R1"]):+.6f}pp.','']
        if a=='A':
            analysis +=['Removing extreme splits '+('supports a sampling-only improvement in this matched500-step run.' if s['Score5_R1']>baseline['scores']['Score5_R1'] else 'did not improve overall retrieval in this matched500-step run.'),'']
        if a=='C':
            recovered=[ds for ds in ['Urban-1k','DOCCI','Long-DCI'] if r['metrics'][ds]['T2I']['R@1']>previous['metrics'][ds]['T2I']['R@1']]
            analysis +=[f'Summary T2I downweight recovered T2I on {recovered}. This {"supports" if len(recovered)==3 else "provides partial or limited evidence for"} the ambiguity hypothesis; a single500-step seed does not establish causality. The12/11 normalization also strengthens the other five directions, as preregistered.','']
        if a=='D':
            deltas=[100*(r['metrics']['Long-DCI'][dr]['R@1']-previous['metrics']['Long-DCI'][dr]['R@1']) for dr in ['I2T','T2I']]
            analysis +=[f'Contiguous Detail changes Long-DCI by {deltas[0]:+.6f}/{deltas[1]:+.6f}pp versus discrete RandomDetail. '+('Both directions recover, consistent with the continuity hypothesis.' if min(deltas)>0 else 'The directional results do not support a uniform continuity benefit.')+' No statistical significance claim is made without replicated seeds.','']
        if a in 'BCD':
            audit=load(EXP/ARMS[a][0]/'SUMMARY_AMBIGUITY.json');m=audit['metrics']['Summary'];diag=load(EXP/ARMS[a][0]/'TRAINING_DIAGNOSTICS.json')['last50']['metrics']
            analysis +=[f'Summary cosine nearest negative {m["nearest_off_diagonal_cosine"]["mean"]:.6f}, mean off-diagonal {m["mean_off_diagonal_cosine"]["mean"]:.6f}; last50 CE Summary I/T {diag["O_i2t"]["mean"]:.6f}/{diag["O_t2i"]["mean"]:.6f}, Detail I/T {diag["E_i2t"]["mean"]:.6f}/{diag["E_t2i"]["mean"]:.6f}.','']
    (EXP/'SHORT_LONG_ANALYSIS.md').write_text('\n'.join(analysis))
    resources=['| Arm | Mean normal full cycle seconds | Peak allocated GiB | Status |','|---|---:|---:|---|']
    for a,r in arms.items():
        resource=r.get('resources',{});resources.append(f'| {a} | {resource.get("normal_full_cycle_mean_seconds",0):.4f} | {resource.get("peak_allocated_gib",0):.4f} | {r["status"]} |')
    (EXP/'RESOURCE_SUMMARY.md').write_text('# Resource summary\n\n'+'\n'.join(resources)+'\n')
    report=['# Four-arm text-supervision search','',f'**{decision["status"]}**. Selected eligible champion: **{decision["champion"]}**; best observed Score5: **{decision["best_observed_Score5_arm"]}**.','',
      'Four preregistered arms ran sequentially:5-step smoke followed by fresh500 from identical commonstep0, seed0,4×256 candidates,4868 scheduler horizon. Native inference is unchanged. No baseline retraining, adaptive tuning or4868-step continuation.','',
      'Long guard: J_long3≥73.403324%; Score5 near-ties strictly<0.05pp use J_long3,Urban T2I,Long T2I,Short4. Raw fractions determine selection.','',*table,'',*five,'',*recalls,'',*resources,'',
      'Correctness:93 initial targeted/regression tests passed, including1000 real-sample exact3da12a3 sampling replay and default-weight bitwise loss/gradients/AdamW. Final additional selection tests are recorded in CORRECTNESS_TESTS.md. All live500×4 ID/Full/token/reference hashes are checked against RandomK. Image augmentation equality is supported by unchanged augmentation/sampler/worker seeds, private sampling RNG and dataset image replay tests; the historical baseline did not record actual image tensor digests, so retrospective500×4 image-byte equality cannot be asserted.','',
      'Strict exports and frozen five-dataset metadata/checkpoint SHA checks are stored per arm. Long-DCI is7602/7602, manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b.','',
      'Gate/F-D diagnostics are scheduled at1/100/200/500; last50 gate and F-D entries have one observation, while CE/keep/inclusion/S-D IoU use all50 updates. See TRAINING_DIAGNOSTICS.json per arm.','',
      f'Most useful candidate for a future full-training confirmation: {decision["full_training_recommendation"]}. This recommendation does not authorize or launch full training.','',
      'See SHORT_LONG_ANALYSIS.md for hypothesis-specific evidence and the limits of a single-seed500-step search. Branch: '+ 'codex/nest-balanced-four-arm-text-search-500-v1'+'.']
    (EXP/'FOUR_ARM_SEARCH_REPORT.md').write_text('\n'.join(report)+'\n')
    state=load(RUN/'state.json');state.update(status='EXPERIMENTS_COMPLETED',stage='report generated; pending Git synchronization',selection=decision,completed_at=now());dump(RUN/'state.json',state)
    print(json.dumps({'selection':decision,'scores':{a:r.get('scores') for a,r in arms.items()}}),flush=True)
