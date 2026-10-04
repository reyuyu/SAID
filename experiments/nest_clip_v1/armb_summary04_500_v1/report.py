"""Frozen S0.4 comparison against S0.6 and matched RandomK; no continuation."""
import json
from pathlib import Path
import statistics
from experiments.nest_clip_v1.armb_summary04_500_v1.run import EXP,RUN,SOURCE,WEIGHTS
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records,now

DATASETS=['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']


def add_short(r):
    s=dict(r['scores']);s['Short4_R1']=statistics.fmean(r['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I']);return s


def classify(s,old,randomk):
    # Literal percent thresholds supplied by the user; decisions use unrounded fractions.
    strong=s['Score5_R1']>=.70000394 and s['J_long3']>=.73403324
    positive=s['Score5_R1']>.69911321 and s['J_long3']>.73132202
    negative=s['Score5_R1']<old['Score5_R1'] or s['J_long3']<old['J_long3']
    tradeoff=s['Short4_R1']>=old['Short4_R1'] and s['J_long3']<=old['J_long3']
    status='STRONG POSITIVE' if strong else 'POSITIVE' if positive else 'NEGATIVE' if negative else 'SHORT-LONG TRADEOFF' if tradeoff else 'NO CLEAR GAIN'
    return dict(status=status,positive=positive,strong_positive=strong,negative_vs_S06=negative,short_long_tradeoff=tradeoff,
      Short4_advantage_over_RandomK_pp=100*(s['Short4_R1']-randomk['Short4_R1']),
      long_guard_pass=s['J_long3']>=.73403324,full4868_recommended=strong,
      full_training_launched=False,stop_updates=500,
      status_overlap_note='If Short4 is maintained but J_long3 drops below S0.6, NEGATIVE is primary and short_long_tradeoff=true records the overlapping user rule.',
      thresholds_percent=dict(positive_Score5_gt=69.911321,positive_Jlong3_gt=73.132202,strong_Score5_ge=70.000394,strong_Jlong3_ge=73.403324))


def generate():
    r=load(EXP/'RESULTS.json');old=load(EXP/'ARM_B_S06.json');baseline=load(EXP/'BASELINE.json')
    s=add_short(r);os=add_short(old);bs=add_short(baseline);decision=classify(s,os,bs)
    delta={name:{k:100*(s[k]-ref[k]) for k in s} for name,ref in [('ArmB_S06',os),('RandomK',bs)]}
    per_dataset={ds:{name:{dr:100*(r['metrics'][ds][dr]['R@1']-ref['metrics'][ds][dr]['R@1']) for dr in ['I2T','T2I']} for name,ref in [('ArmB_S06',old),('RandomK',baseline)]} for ds in DATASETS}
    r.update(status='COMPLETE',classification=decision,delta_pp=delta,dataset_R1_delta_pp=per_dataset,gradient_spot_check=load(EXP/'GRADIENT_SPOT_CHECK.json'))
    dump(EXP/'RESULTS.json',r);dump(EXP/'DECISION.json',decision)
    diag=load(EXP/'TRAINING_DIAGNOSTICS.json');last=diag['last50'];spot=r['gradient_spot_check'];g='native_backbone_total';new_grad=spot['groups'][g]
    old_diag=load(SOURCE/'TRAINING_DIAGNOSTICS.json')['last50']['metrics']
    old_raw={label:old_diag[p+'_i2t']['mean']+old_diag[p+'_t2i']['mean'] for label,p in [('F','F'),('S','O'),('D','E')]}
    old_weighted={label:w*old_raw[label] for label,w in zip(['F','S','D'],[1.2,.6,1.2])}
    old_loss_shares={label:v/sum(old_weighted.values()) for label,v in old_weighted.items()}
    summary_loss_share=last['weighted_CE_share']['S'];largest_loss=max(last['weighted_CE_contribution'],key=last['weighted_CE_contribution'].get)
    shares=new_grad['weighted_norm_share_mean'];largest_gradient=max(shares,key=shares.get)
    findings=dict(last50_weighted_CE_share=last['weighted_CE_share'],old_S06_last50_weighted_CE_share=old_loss_shares,largest_weighted_CE_view=largest_loss,
      spot8_raw_norms=new_grad['norm_mean'],spot8_weighted_norm_shares=shares,
      spot8_largest_weighted_norm_view=largest_gradient,
      Summary_dominates_weighted_CE=largest_loss=='S',Summary_dominates_weighted_gradient_norm=largest_gradient=='O_combined',
      actual_alignment_native_cosine=new_grad['actual_alignment_native_cosine_mean'],
      old_S06_same8_actual_alignment_native_cosine=new_grad['old_S06_alignment_native_cosine_mean'],
      important_scope='Loss shares and gradient norm shares are separate diagnostics; neither is an AdamW displacement contribution percentage. The spot check is limited to8 training global batches.')
    dump(EXP/'SUMMARY_DOMINANCE.json',findings)
    lines=['# Arm B Summary weight0.4 matched500 experiment','',f'**{decision["status"]}**. All500 updates, strict bare export, frozen native5 evaluations and read-only spot8 are complete.','',
      'Only normalized alignment weights changed: F/S/D[1.2,0.6,1.2]→[1.3,0.4,1.3]. Sampling/RNG, model, candidates, directional weights, sparsity, inclusion/ramp, optimizer/LR,horizon4868 and native inference are unchanged. Smoke5 and formal500 independently begin at common step0; no resume or full continuation.','',
      '| Model | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    for label,vals in [('RandomK baseline',bs),('Arm B S=0.6',os),('Arm B S=0.4',s)]:lines.append('| '+label+' | '+' | '.join(f'{vals[k]*100:.6f}' for k in ['Score5_R1','J_long3','J_long','Short4_R1'])+' |')
    lines+=['','| Reference | ΔScore5 pp | ΔJ_long3 pp | ΔJ_long pp | ΔShort4 pp |','|---|---:|---:|---:|---:|']
    for name,d in delta.items():lines.append('| '+name+' | '+' | '.join(f'{d[k]:+.6f}' for k in ['Score5_R1','J_long3','J_long','Short4_R1'])+' |')
    lines+=['','| Dataset | I2T R1 | T2I R1 | Δvs S0.6 I/T pp | Δvs RandomK I/T pp |','|---|---:|---:|---|---|']
    for ds in DATASETS:
        v=r['metrics'][ds];d=per_dataset[ds]
        lines.append(f'| {ds} | {v["I2T"]["R@1"]*100:.6f} | {v["T2I"]["R@1"]*100:.6f} | '+ ' / '.join(f'{d["ArmB_S06"][dr]:+.6f}' for dr in ['I2T','T2I'])+' | '+' / '.join(f'{d["RandomK"][dr]:+.6f}' for dr in ['I2T','T2I'])+' |')
    lines+=['','| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|']
    for ds in DATASETS:lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{r["metrics"][ds][dr][k]*100:.6f}' for k in ['R@1','R@5','R@10']) for dr in ['I2T','T2I'])+' |')
    lines+=['','## Summary loss and gradient dominance','',
      'Raw CE and weighted CE are separately recorded at1/100/200/500 and last50. Last50 combined CE and relative weighted contributions (common10/3 omitted here):','',
      '| View | Raw combined CE | Weighted contribution | Weighted CE share | Old S0.6 weighted CE share |','|---|---:|---:|---:|---:|']
    for label in ['F','S','D']:lines.append(f'| {label} | {last["raw_combined_CE"][label]:.6f} | {last["weighted_CE_contribution"][label]:.6f} | {last["weighted_CE_share"][label]*100:.3f}% | {old_loss_shares[label]*100:.3f}% |')
    lines+=['',f'Summary is {"still" if findings["Summary_dominates_weighted_CE"] else "not"} the largest weighted CE route. Largest: {largest_loss}. Raw loss magnitude alone is not a gradient conclusion.','',
      '| Spot8 native backbone view | Raw gradient norm mean | Weighted norm share | Old S0.6 weighted norm share |','|---|---:|---:|---:|']
    for label,key in zip(['F','S','D'],['F_combined','O_combined','E_combined']):
        lines.append(f'| {label} | {new_grad["norm_mean"][key]:.6f} | {shares[key]*100:.3f}% | {new_grad["old_S06_weighted_norm_share_mean"][key]*100:.3f}% |')
    lines+=['',f'Summary is {"still" if findings["Summary_dominates_weighted_gradient_norm"] else "not"} the largest weighted gradient-norm route. Largest: {largest_gradient}. Norm shares are diagnostics, not percentages of the summed gradient or AdamW parameter displacement.','',
      '| Spot8 group | Actual alignment/native cosine S0.6 | Actual alignment/native cosine S0.4 | Delta |','|---|---:|---:|---:|']
    for group in ['G1_vision_backbone','G2_text_backbone','native_backbone_total']:
        v=spot['groups'][group];a=v['old_S06_alignment_native_cosine_mean'];b=v['actual_alignment_native_cosine_mean'];lines.append(f'| {group} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines+=['','Spot8 uses exactly the same first8 seed0/epoch0 global batches as the prior audit. Actual image/F/S/D digests match. Five objectives per batch are synchronized DDP backwards with20 parameter-tensor agreement checks; model/checkpoint remain unchanged. Different learned checkpoints limit attribution of gradient changes to immediate weighting alone. No full32-batch audit was rerun.','',
      '## Resources and correctness','',
      f'Normal full-cycle mean {r["resources"]["normal_full_cycle_mean_seconds"]:.6f}s; peak allocated {r["resources"]["peak_allocated_gib"]:.6f}GiB. All500×4 F/S/D/token/K/indices stream hashes match Arm B. Common initialization, trainable parameter counts and production source hashes match. Fixed1000 sampling replay and explicit CE/gradient tests passed. All500 actual loss values reconstruct from the declared weighting and unchanged auxiliary terms.','',
      'Gate/F-D fields are scheduled diagnostics; last50 gate/F-D values have one observation. CE/keep/inclusion/S-D IoU use all50. Full details are in TRAINING_DIAGNOSTICS.json and compressed complete logs.','',
      'Long-DCI remains7602/7602 with manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b. Native inference uses normalized image and full-caption text embeddings only.','',
      '## Decision','',
      f'Full4868 verification recommended: **{decision["full4868_recommended"]}** under the registered STRONG POSITIVE rule. Full training launched: **False**. This experiment stops at500.', '',
      'POSITIVE requires both Score5>69.911321% and J_long3>73.132202%; STRONG POSITIVE additionally requires Score5≥70.000394% and J_long3≥73.403324%. Raw fractions, not rounded display values, determine decisions. Overlapping negative/tradeoff conditions are retained in DECISION.json.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    state=load(RUN/'state.json');state.update(status='EXPERIMENT_COMPLETED',stage='report generated; pending GitHub synchronization',decision=decision,full_training_launched=False,finished_at=now());dump(RUN/'state.json',state)
    print(json.dumps({'status':decision['status'],'scores_percent':{k:100*v for k,v in s.items()},'deltas_pp':delta,'Summary_dominance':findings}),flush=True)

if __name__=='__main__':generate()
