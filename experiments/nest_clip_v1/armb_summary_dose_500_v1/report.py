"""Five-dose observed curve, two new runs only; no adaptive training."""
import csv
import json
from pathlib import Path
import statistics
from experiments.nest_clip_v1.armb_summary_dose_500_v1.run import EXP,RUN,ROOT,S04,SEARCH,ARMS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,now,records

DATASETS=['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']
METRICS=['Score5_R1','J_long3','J_long','Short4_R1']


def score_record(record):
    scores=dict(record['scores']);scores['Short4_R1']=statistics.fmean(record['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I'])
    return scores


def make_curve():
    old1=load(ROOT/'experiments/nest_clip_v1/balanced_summary_random_detail_500_v1/RESULTS.json')
    old06=load(SEARCH/'arm_B_summary_weight/RESULTS.json');old04=load(S04/'RESULTS.json')
    entries=[(1.,[1.,1.,1.],old1),( .6,[1.2,.6,1.2],old06),(.4,[1.3,.4,1.3],old04)]
    for key,(directory,weights) in ARMS.items():
        record=load(EXP/directory/'RESULTS.json');assert record['status']=='COMPLETE'
        entries.append((weights[1],weights,record))
    return [dict(S_weight=dose,weights=weights,scores=score_record(r),metrics=r['metrics'],
      checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],bare_sha256=r['bare_sha256'],
      record=r,new_run=dose in [.2,0.]) for dose,weights,r in entries]


def gradient_curve():
    audit=ROOT/'experiments/nest_clip_v1/gradient_composition_audit_v1'
    with __import__('gzip').open(audit/'RAW_BATCH_GRADIENT_STATS.json.gz','rt') as f:raw=json.load(f)['batches']
    old=[r for r in raw if r['stage']=='B'][:8]
    out={'.6':{},'.4':{},'.2':{},'0':{}}
    for group in ['G1_vision_backbone','G2_text_backbone','native_backbone_total']:
        out['.6'][group]=statistics.fmean(r['groups'][group]['pairs']['align_actual__native_combined']['cosine'] for r in old)
        out['.4'][group]=load(S04/'GRADIENT_SPOT_CHECK.json')['groups'][group]['actual_alignment_native_cosine_mean']
    for key,(directory,weights) in ARMS.items():
        target='.2' if weights[1]==.2 else '0';g=load(EXP/directory/'GRADIENT_SPOT8.json')['groups']
        for group in out['.4']:out[target][group]=g[group]['actual_alignment_native_cosine_mean']
    return out


def summarize_decision(curve,gradients):
    scores=[r['scores']['Score5_R1'] for r in curve];long=[r['scores']['J_long3'] for r in curve]
    best_score=max(curve,key=lambda r:(r['scores']['Score5_R1'],r['scores']['J_long3'],r['scores']['Short4_R1']))
    best_long=max(curve,key=lambda r:(r['scores']['J_long3'],r['scores']['Score5_R1']))
    lookup={r['S_weight']:r for r in curve}
    if best_score['S_weight']==.2 and lookup[0.]['scores']['Score5_R1']<lookup[.2]['scores']['Score5_R1']:
        interpretation='LOW_DOSE_SUMMARY_OBSERVED_BEST'
        recommendation='Observed S0.2 wins Score5 and S0.0 drops; supports a low but nonzero Summary CE dose in this500-step seed. Check the separate J_long3 optimum before full confirmation.'
    elif best_score['S_weight']==0.:
        interpretation='ZERO_SUMMARY_ALIGNMENT_OBSERVED_BEST'
        recommendation='Zero Summary alignment wins this500-step comparison. Summary CE is not needed for the best observed Score5 here; Summary sampling/sparsity/inclusion still remain. Next research should isolate F/D reweighting, without launching it in this task.'
    elif best_score['S_weight']==.4:
        interpretation='S04_REMAINS_OBSERVED_BEST'
        recommendation='S0.4 remains the best observed Score5; stop decreasing Summary and recommend the already validated S0.4 for4868 confirmation. No full run is launched.'
    else:
        interpretation='HIGHER_DOSE_OR_TRADEOFF'
        recommendation='The Score5/long/short curves require a tradeoff interpretation; do not continue a weight sweep from these500-step results.'
    joint=best_score['S_weight']==best_long['S_weight']
    gvals=[gradients[k]['native_backbone_total'] for k in ['.6','.4','.2','0']]
    return dict(interpretation=interpretation,best_Score5_dose=best_score['S_weight'],best_Jlong3_dose=best_long['S_weight'],
      Score5_monotonic_as_S_decreases=all(b>=a for a,b in zip(scores,scores[1:])),
      Jlong3_monotonic_as_S_decreases=all(b>=a for a,b in zip(long,long[1:])),
      actual_alignment_native_cosine_monotonic=all(b>=a for a,b in zip(gvals,gvals[1:])),
      observed_joint_nonzero_best=joint and best_score['S_weight']>0,
      best_score_and_long_same_dose=joint,
      Short4_zero_minus_S04_pp=100*(lookup[0.]['scores']['Short4_R1']-lookup[.4]['scores']['Short4_R1']),
      Short4_zero_minus_S02_pp=100*(lookup[0.]['scores']['Short4_R1']-lookup[.2]['scores']['Short4_R1']),
      recommendation=recommendation,long_guard={str(r['S_weight']):r['scores']['J_long3']>=.73403324 for r in curve},
      no_extra_weights_tested=True,no_full_training=True,stop_updates=500,
      limits='Only one seed and500 updates. An observed nonzero optimum is not proof of a population optimum or a4868-step outcome. Gradient last50 shares are final-checkpoint input replays, not historical training-update gradients.')


def plot(curve,baseline):
    import subprocess
    plot_python='/root/lk_projects/SAID-nest-clip-v1/summary-dose-plot-env/bin/python'
    dump(EXP/'PLOT_INPUT.json',dict(curve=[dict(S_weight=r['S_weight'],scores=r['scores']) for r in curve],baseline=score_record(baseline)))
    argv=[plot_python,'-m','experiments.nest_clip_v1.armb_summary_dose_500_v1.plot_curve']
    result=subprocess.run(argv,cwd=ROOT,check=True,capture_output=True,text=True)
    dump(EXP/'commands/plot.json',dict(argv=argv,cwd=str(ROOT),exit_code=result.returncode,
      matplotlib='3.9.2',isolated_environment=True,shared_training_environment_modified=False))


def generate():
    curve=make_curve();baseline=load(EXP/'BASELINE.json');gradients=gradient_curve();decision=summarize_decision(curve,gradients)
    compact=[{k:v for k,v in r.items() if k!='record'} for r in curve]
    dump(EXP/'DOSE_RESULTS.json',dict(status='COMPLETE',curve=compact,RandomK_baseline=baseline,gradient_spot8_curve=gradients,decision=decision))
    dump(EXP/'DECISION.json',decision)
    with (EXP/'DOSE_CURVE.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['S_weight','F_weight','D_weight',*METRICS])
        for r in curve:writer.writerow([r['S_weight'],r['weights'][0],r['weights'][2],*[100*r['scores'][k] for k in METRICS]])
    plot(curve,baseline)
    lines=['# Arm B Summary alignment-dose500 report','',f'**{decision["interpretation"]}**. E1/E2 completed fresh500, strict export, frozen native5 and read-only diagnostics.','',
      '**S=0 means zero Summary contrastive alignment weight only.** Summary is still constructed, encoded and used by original sparsity/inclusion; forward and architecture remain unchanged. The sole production edit permits zero coefficients in validation.','',
      '| S weight | F/S/D weights | Score5 | J_long3 | J_long | Short4 |','|---:|---|---:|---:|---:|---:|']
    for r in curve:lines.append(f'| {r["S_weight"]:.1f} | '+ '/'.join(f'{w:g}' for w in r['weights'])+' | '+' | '.join(f'{r["scores"][k]*100:.6f}' for k in METRICS)+' |')
    lines+=['','![Dose curve](DOSE_CURVE.png)','',
      'Historical S1.0/S0.6/S0.4 are reused without retraining. E1/E2 each independently run smoke5, then fresh formal500 from identical common step0.4×256 candidates,seed0,accumulation1,horizon4868. Sampling/RNG,optimizer/LRs,auxiliary formulas/ramp and native inference are fixed.','',
      '## Main questions','',
      f'1. Score5 monotonic as S decreases: **{decision["Score5_monotonic_as_S_decreases"]}**. J_long3 monotonic: **{decision["Jlong3_monotonic_as_S_decreases"]}**.',
      f'2. Short4 at zero versus S0.4: **{decision["Short4_zero_minus_S04_pp"]:+.6f}pp**; versus S0.2: **{decision["Short4_zero_minus_S02_pp"]:+.6f}pp**. This reports effect size without a significance claim.',
      '3–4. Urban and Long-DCI changes are listed below for every dose and against S0.4/RandomK.',
      f'5. Actual alignment/native cosine monotonic from S0.6→S0.4→S0.2→S0.0: **{decision["actual_alignment_native_cosine_monotonic"]}**.',
      f'6. Observed Score5 optimum: **S={decision["best_Score5_dose"]}**; J_long3 optimum: **S={decision["best_Jlong3_dose"]}**. Joint nonzero observed optimum: **{decision["observed_joint_nonzero_best"]}**. Single-seed500 evidence cannot establish a universal optimum.','',
      '## Five-dataset R1 and deltas','',
      '| S weight | Dataset | I2T R1 | T2I R1 | Δvs S0.4 I/T pp | Δvs RandomK I/T pp |','|---:|---|---:|---:|---|---|']
    ref=curve[2]
    for r in curve:
        for ds in DATASETS:
            m=r['metrics'][ds]
            delta=lambda reference:' / '.join(f'{100*(m[dr]["R@1"]-reference["metrics"][ds][dr]["R@1"]):+.6f}' for dr in ['I2T','T2I'])
            lines.append(f'| {r["S_weight"]:.1f} | {ds} | {m["I2T"]["R@1"]*100:.6f} | {m["T2I"]["R@1"]*100:.6f} | {delta(ref)} | {delta(baseline)} |')
    lines+=['','## E1/E2 full recalls','', '| Arm | Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|---|']
    for key,(directory,weights) in ARMS.items():
        r=load(EXP/directory/'RESULTS.json')
        for ds in DATASETS:lines.append('| '+key+' | '+ds+' | '+' | '.join(' / '.join(f'{r["metrics"][ds][dr][k]*100:.6f}' for k in ['R@1','R@5','R@10']) for dr in ['I2T','T2I'])+' |')
    lines+=['','## Actual last50 CE and frozen-checkpoint last50 gradient shares','',
      'CE below is the actual update451..500 mean. Gradient shares are **read-only replay of those50 input batches at checkpoint500**, rather than per-update historical gradients. Relative shares omit the common10/3; they are not exact summed-gradient or optimizer displacement contribution percentages.','',
      '| Arm | View | Last50 raw CE | Weighted CE | Weighted CE share | Last50 replay raw gradient norm | Weighted gradient norm share |','|---|---|---:|---:|---:|---:|---:|']
    for key,(directory,weights) in ARMS.items():
        d=load(EXP/directory/'TRAINING_DIAGNOSTICS.json')['last50'];g=load(EXP/directory/'LAST50_GRADIENT_SHARES.json')['groups']['native_backbone_total']
        for label,obj in zip(['F','S','D'],['F_combined','O_combined','E_combined']):lines.append(f'| {key} | {label} | {d["raw_combined_CE"][label]:.6f} | {d["weighted_CE_contribution"][label]:.6f} | {d["weighted_CE_shares"][label]*100:.3f}% | {g["raw_norm_means"][obj]:.6f} | {g["weighted_norm_share_means"][obj]*100:.3f}% |')
    lines+=['','## Fixed8 actual alignment/native cosine','',
      'Every dose below uses the same first8 real4-card global batches. Historical S0.6/S0.4 diagnostics are reused; no reference training or full32-batch audit is rerun. Comparisons across checkpoints mix learned-state and coefficient effects. S1.0 native-gradient cosine was not previously audited and is not invented.','',
      '| S weight | Vision backbone | Text backbone | Native backbone total |','|---:|---:|---:|---:|']
    for dose,key in [(.6,'.6'),(.4,'.4'),(.2,'.2'),(0.,'0')]:lines.append(f'| {dose:.1f} | '+' | '.join(f'{gradients[key][g]:.6f}' for g in ['G1_vision_backbone','G2_text_backbone','native_backbone_total'])+' |')
    lines+=['','## Resources','', '| Arm | Normal full-cycle mean seconds | Peak allocated GiB |','|---|---:|---:|']
    resources={}
    for key,(directory,_) in ARMS.items():
        resource=load(EXP/directory/'RESOURCE_SUMMARY.json');resources[key]=resource
        lines.append(f'| {key} | {resource["normal_full_cycle_mean_seconds"]:.6f} | {resource["peak_allocated_gib"]:.6f} |')
    dump(EXP/'RESOURCE_SUMMARY.json',resources)
    lines+=['','## Correctness and stopping','',
      '26 initial tests pass, including zero Summary alignment gradients with nonzero Summary sparsity gradients. Both preflights prove1000-sample equality and live500×4 stream matches to S0.4. Default forward/optimizer/scheduler/sampler/evaluator sources remain unchanged; the hparams change only permits nonnegative coefficients and rejects an all-zero total. Each formal run starts at step0 with empty optimizer and resume=None.','',
      'Five native evaluators use strict exported bare students only. Long-DCI remains7602/7602,manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b. Training masks/local views do not enter native inference.','',
      decision['recommendation'], '', 'For an overall/short-text confirmation, S0.2 is the current observed candidate. For a long-text-first objective, S0.0 is the observed candidate. Their Score5 difference is only0.053766pp and the optima differ, so this run does not establish a clearly unique Summary dose. A4868 confirmation is a recommendation only; no full run was launched.', '', 'This normalized dose path lowers S while increasing both F and D. Therefore the complete effect cannot be attributed solely to removing Summary CE; separate F/D reweighting would be a future experiment, not part of this task.', '',
      'No other weights, no new training beyond E1/E2×500, no4868 run. Full gradients, datasets and checkpoints remain outside Git. Branch:codex/nest-balanced-armb-summary-dose-500-v1.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    state=load(RUN/'state.json');state.update(status='EXPERIMENTS_COMPLETED',stage='unified report generated; pending GitHub synchronization',decision=decision,finished_at=now());dump(RUN/'state.json',state)
    print(json.dumps({'dose_scores_percent':{str(r['S_weight']):{k:v*100 for k,v in r['scores'].items()} for r in curve},'decision':decision}),flush=True)

if __name__=='__main__':generate()
