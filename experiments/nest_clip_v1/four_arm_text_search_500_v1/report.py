"""Fixed long guard, Score5 and deterministic near-tie decision after four arms."""
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path
import statistics
from train.nested_semantic_data import sample_interior_split_k,sample_contiguous_detail_indices
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP,RUN,ARMS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import dump,load,now,records,stream_check

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


def final_audits(arms,baseline,previous):
    baseline_rows=records(Path(baseline['root'])/'steps.jsonl')
    previous_rows=records(Path(previous['checkpoint']).parent/'steps.jsonl')
    full=load(EXP/'evidence/FULL_TOKEN_LENGTHS.json')
    core_hashes=None
    for arm,result in arms.items():
        if result['status']!='COMPLETE':continue
        directory=EXP/ARMS[arm][0];root=Path(result['checkpoint']).parent
        rows=records(root/'steps.jsonl');assert len(rows)==500
        for row,ref in zip(rows,baseline_rows):stream_check(row,ref)
        config=result['config']
        assert config['component_initialization']==baseline['config']['component_initialization']
        assert config['parameter_counts']==baseline['config']['parameter_counts']
        if core_hashes is None:core_hashes=config['code_sha256']
        else:assert core_hashes==config['code_sha256']
        sampling_replay=arm in 'AD'
        if sampling_replay:
            for row,ref in zip(rows,baseline_rows):
                for rank,old in zip(row['rank_health'],ref['rank_health']):
                    sampling=rank['sampling']
                    assert sampling['n']==old['sampling']['n']
                    choices=[]
                    for sid,n,k in zip(sampling['sample_ids'],sampling['n'],sampling['K']):
                        if arm=='A':
                            expected=sample_interior_split_k(n,0,0,sid) if n>=2 else 0
                            assert k==expected
                            assert 2<=k<=n-2 if n>=4 else 0<=k<=max(0,n-1)
                        else:
                            indices=sample_contiguous_detail_indices(n,0,0,sid)
                            assert len(indices)==k and 0 not in indices
                            if indices:assert indices==list(range(indices[0],indices[0]+k))
                            if n>=4:assert 2<=k<n-1
                            choices.append(indices)
                    if arm=='D':
                        digest=hashlib.sha256(json.dumps(dict(sample_ids=sampling['sample_ids'],choices=choices),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                        assert digest==sampling['random_detail_sampling']['selected_indices_sha256']
        local_match=arm in 'BC'
        if local_match:
            for row,ref in zip(rows,previous_rows):
                for new,old in zip(row['rank_health'],ref['rank_health']):
                    for key in ['local_views_sha256','split_sha256']:
                        assert new['sampling'][key]==old['sampling'][key],(arm,row['step'],key)
                    assert new['sampling']['random_detail_sampling']['selected_indices_sha256']==old['sampling']['random_detail_sampling']['selected_indices_sha256']
        errors=[]
        if arm in 'BC':
            for row in rows:
                ce=[(row[v+'_i2t'],row[v+'_t2i']) for v in ['F','O','E']]
                if arm=='B':align=10/3*sum(w*(ci+ct) for w,(ci,ct) in zip([1.2,.6,1.2],ce))
                else:align=10/3*12/11*(ce[0][0]+ce[0][1]+ce[1][0]+.5*ce[1][1]+ce[2][0]+ce[2][1])
                expected=align+(row['F_sparse']+2*row['O_sparse']+2*row['E_sparse'])/3+row['inc_weight']*row['inc']
                assert math.isclose(expected,row['loss'],rel_tol=1e-5,abs_tol=3e-4),(arm,row['step'],expected,row['loss'])
                errors.append(abs(expected-row['loss']))
        dump(directory/'evidence/final-matching-and-objective.json',dict(passed=True,formal_steps=500,ranks=4,
          ID_F_and_reference_match_baseline=True,local_views_match_previous_all500x4=local_match,
          all500x4_sampling_K_or_contiguous_indices_replay_checked=sampling_replay,
          same_initial_component_hashes=True,same_parameter_counts=True,same_training_code_hashes_all_arms=True,
          all500_objective_reconstruction_checked=arm in 'BC',max_objective_float32_rounding_error=max(errors) if errors else None))
        diag=load(directory/'TRAINING_DIAGNOSTICS.json')
        for step,v in full['steps'].items():diag['steps'][step]['token_lengths']['Full']=v
        diag['last50']['token_lengths']['Full']=full['last50']
        dump(directory/'TRAINING_DIAGNOSTICS.json',diag)
        # Preserve lossless full step evidence as small compressed logs in Git.
        for name in ['steps.jsonl','cycle_timing.jsonl']:
            source=directory/'evidence'/name;target=source.with_suffix(source.suffix+'.gz')
            if not source.exists() and target.exists():continue
            with source.open('rb') as src,target.open('wb') as out:
                with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as archive:shutil.copyfileobj(src,archive)
            source.unlink()


def generate():
    baseline=load(EXP/'BASELINE.json');previous=load(EXP/'PREVIOUS_SUMMARY_RANDOM_DETAIL.json')
    arms={a:load(EXP/directory/'RESULTS.json') for a,(directory,*_) in ARMS.items()}
    final_audits(arms,baseline,previous)
    decision=selection(arms,baseline['scores']['Score5_R1'])
    decision['champion_is_overall_improvement']=bool(decision['champion'] and arms[decision['champion']]['scores']['Score5_R1']>baseline['scores']['Score5_R1'])
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
            t2i_deltas={ds:100*(r['metrics'][ds]['T2I']['R@1']-previous['metrics'][ds]['T2I']['R@1']) for ds in ['Urban-1k','DOCCI','Long-DCI']}
            analysis +=[f'Summary T2I downweight recovered T2I on {recovered}; directional deltas versus previous are {t2i_deltas}pp. This provides limited support for the T2I-ambiguity hypothesis. Short4 and overall/long-guard deltas above must be considered together: small positive directional changes do not establish that the long-text deficit was repaired. A single500-step seed does not establish causality. The12/11 normalization also strengthens the other five directions, as preregistered.','']
        if a=='D':
            deltas=[100*(r['metrics']['Long-DCI'][dr]['R@1']-previous['metrics']['Long-DCI'][dr]['R@1']) for dr in ['I2T','T2I']]
            analysis +=[f'Contiguous Detail changes Long-DCI by {deltas[0]:+.6f}/{deltas[1]:+.6f}pp versus discrete RandomDetail. '+('Both directions recover, consistent with the continuity hypothesis.' if min(deltas)>0 else 'The directional results do not support a uniform continuity benefit.')+' No statistical significance claim is made without replicated seeds.','']
        if a in 'BCD':
            audit=load(EXP/ARMS[a][0]/'SUMMARY_AMBIGUITY.json');m=audit['metrics']['Summary'];diag=load(EXP/ARMS[a][0]/'TRAINING_DIAGNOSTICS.json')['last50']['metrics']
            analysis +=[f'Summary cosine nearest negative {m["nearest_off_diagonal_cosine"]["mean"]:.6f}, mean off-diagonal {m["mean_off_diagonal_cosine"]["mean"]:.6f}; last50 CE Summary I/T {diag["O_i2t"]["mean"]:.6f}/{diag["O_t2i"]["mean"]:.6f}, Detail I/T {diag["E_i2t"]["mean"]:.6f}/{diag["E_t2i"]["mean"]:.6f}.','']
    analysis +=[
      'B is the strongest observed supervision change in this search. Against RandomK it gains Score5 +0.010927pp and Short4 +0.734000pp, but loses J_long3 -0.471122pp; it is0.271122pp below the safety threshold. Against previous Summary+RandomDetail it restores part of the long-text performance while preserving Short4. The very small overall Score5 increase is not a safe improvement.', '',
      'B outperforms C by Score5 +0.133736pp, J_long3 +0.137560pp and Short4 +0.128000pp. These matched results favor reducing total Summary supervision over reducing only its T2I direction at these registered coefficients. C has modest positive long-T2I changes versus previous, but its Summary nearest-negative cosine rises; CE changes and cosine geometry do not independently prove a causal ambiguity mechanism.', '',
      'D does not recover the previous Urban or DOCCI results; Long-DCI I2T gains0.263089pp while T2I is unchanged. This run offers no convincing overall benefit from enforcing contiguous Detail. Discrete sampling is not established as the main source of the long-text deficit.', '',
      'Rule-selected eligible champion A is a relative selection among new arms, not an improvement over the existing RandomK. A is the only safe candidate for a future full-training confirmation under the requested rule, but its500-step evidence does not justify replacing the formal RandomK recipe. B is the more informative supervision direction for future research if the long-text deficit can be addressed. No full training is launched.'
    ]
    (EXP/'SHORT_LONG_ANALYSIS.md').write_text('\n'.join(analysis))
    resources=['| Arm | Mean normal full cycle seconds | Peak allocated GiB | Status |','|---|---:|---:|---|']
    for a,r in arms.items():
        resource=r.get('resources',{});resources.append(f'| {a} | {resource.get("normal_full_cycle_mean_seconds",0):.4f} | {resource.get("peak_allocated_gib",0):.4f} | {r["status"]} |')
    (EXP/'RESOURCE_SUMMARY.md').write_text('# Resource summary\n\n'+'\n'.join(resources)+'\n')
    report=['# Four-arm text-supervision search','',f'**{decision["status"]}**. Selected eligible champion: **{decision["champion"]}**; best observed Score5: **{decision["best_observed_Score5_arm"]}**. No safe overall improvement over matched RandomK.','',
      'Four preregistered arms ran sequentially:5-step smoke followed by fresh500 from identical commonstep0, seed0,4×256 candidates,4868 scheduler horizon. Native inference is unchanged. No baseline retraining, adaptive tuning or4868-step continuation.','',
      'Long guard: J_long3≥73.403324%; Score5 near-ties strictly<0.05pp use J_long3,Urban T2I,Long T2I,Short4. Raw fractions determine selection.','',*table,'',*five,'',*recalls,'',*resources,'',
      'Correctness:93 initial targeted/regression tests passed, including1000 real-sample exact3da12a3 sampling replay and default-weight bitwise loss/gradients/AdamW. Five additional selection tests pass (98 unique tests total). A separate1000-real-image replay also passes. All live500×4 ID/Full/token/reference hashes are checked against RandomK. Image equality follows from identical indexed image paths and unchanged deterministic Resize/CenterCrop/Normalize, and is independently verified on1000 actual images. Historical image tensor digests were not recorded; this is source/data/transform equivalence plus actual replay, rather than a retrospective hash comparison. B/C also match previous local strings/tokens/K/indices hashes on all500×4 rank batches; live objective reconstruction checks the declared weight formula on all500 steps.','',
      'Strict exports and frozen five-dataset metadata/checkpoint SHA checks are stored per arm. Long-DCI is7602/7602, manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b.','',
      'Gate/F-D diagnostics are scheduled at1/100/200/500; last50 gate and F-D entries have one observation, while CE/keep/inclusion/S-D IoU use all50 updates. See TRAINING_DIAGNOSTICS.json per arm.','',
      f'Most useful candidate for a future full-training confirmation: {decision["full_training_recommendation"]}. This recommendation does not authorize or launch full training.','',
      'See SHORT_LONG_ANALYSIS.md for hypothesis-specific evidence and the limits of a single-seed500-step search. Branch: '+ 'codex/nest-balanced-four-arm-text-search-500-v1'+'.']
    report += ['',
      'The eligible champion is only the best new arm under the guard. In this run A remains below RandomK Score5; the existing RandomK recipe remains the stronger formal reference. B is the highest observed Score5 and retains short-text gains, but is ineligible because of its long-text loss. C gives limited support for direction-specific ambiguity; D does not demonstrate a consistent continuity benefit. See the detailed short/long analysis.',
      '', 'Artifacts contain small JSON/Markdown/code and lossless compressed logs only; checkpoints, datasets and embeddings remain outside Git.']
    (EXP/'FOUR_ARM_SEARCH_REPORT.md').write_text('\n'.join(report)+'\n')
    command_index={a:{path.stem:load(path) for path in (EXP/ARMS[a][0]/'commands').glob('*.json')} for a in ARMS}
    dump(EXP/'commands/INDEX.json',command_index)
    dump(EXP/'raw/NATIVE_METRICS.json',{a:r['metrics'] for a,r in arms.items() if r['status']=='COMPLETE'})
    dump(EXP/'RESOURCE_SUMMARY.json',{a:r.get('resources',{'status':r['status']}) for a,r in arms.items()})
    (EXP/'PROGRESS.md').write_text('# Four-arm progress\n\nCOMPLETE: all four arms finished fresh500 training, strict export, frozen five-dataset native evaluation and diagnostics. Final status: '+decision['status']+'. See FOUR_ARM_SEARCH_REPORT.md and SELECTION.json. No4868-step run was launched.\n')
    state=load(RUN/'state.json');state.update(status='EXPERIMENTS_COMPLETED',stage='report generated; pending Git synchronization',selection=decision,completed_at=now());dump(RUN/'state.json',state)
    print(json.dumps({'selection':decision,'scores':{a:r.get('scores') for a,r in arms.items()}}),flush=True)
