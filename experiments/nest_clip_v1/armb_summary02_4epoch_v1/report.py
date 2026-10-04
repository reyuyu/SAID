"""Matched four-epoch native comparison, with separate500-step parent evidence."""
import json
from pathlib import Path
import statistics
from experiments.nest_clip_v1.armb_summary02_4epoch_v1.run import EXP,RUN,ROOT,BRANCH,STOP
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records,now

DATASETS=['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']
METRICS=['Score5_R1','J_long3','J_long','Short4_R1']


def with_short(record):
    s=dict(record['scores']);s['Short4_R1']=statistics.fmean(record['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I']);return s


def generate():
    r=load(EXP/'RESULTS.json');ref=load(EXP/'RANDOMK_4EPOCH_REFERENCE.json');parent=load(EXP/'PARENT_500.json')
    rs=with_short(r);bs=with_short(ref);ps=with_short(parent)
    delta={k:100*(rs[k]-bs[k]) for k in METRICS};parent_delta={k:100*(rs[k]-ps[k]) for k in METRICS}
    cycle=records(RUN/'step4868/cycle_timing.jsonl');normal=[v['four_rank_max_seconds'] for v in cycle if not v['warmup']]
    acceptance=load(RUN/'step4868/acceptance.json')
    resources=dict(normal_full_cycle_mean_seconds=statistics.fmean(normal),normal_cycles=len(normal),
      continuation_updates=4368,completed_updates=4868,
      peak_allocated_gib=max(v['peak_allocated_gib'] for v in acceptance['ranks']))
    assert len(cycle)==4368 and resources['normal_full_cycle_mean_seconds']<=3 and resources['peak_allocated_gib']<=65
    diag_rows=records(RUN/'step4868/steps.jsonl');last=diag_rows[-50:]
    diagnostic={}
    for key in sorted(set().union(*(v.keys() for v in last))):
        values=[v[key] for v in last if isinstance(v.get(key),(int,float)) and not isinstance(v.get(key),bool)]
        if values:diagnostic[key]=dict(mean=statistics.fmean(values),observations=len(values))
    raw={label:diagnostic[p+'_i2t']['mean']+diagnostic[p+'_t2i']['mean'] for label,p in [('F','F'),('S','O'),('D','E')]}
    weighted={label:w*raw[label] for label,w in zip(['F','S','D'],[1.4,.2,1.4])}
    dump(EXP/'TRAINING_DIAGNOSTICS.json',dict(last50=diagnostic,raw_combined_CE=raw,weighted_CE_contributions=weighted,
      epoch_boundaries={str(step):next(v for v in diag_rows if v['step']==step) for step in [1217,2434,3651,4868]},
      note='Last50 means actual updates4819..4868. Scheduled gate/F-D fields have only one observation in this window. Loss share is not gradient contribution.'))
    improved=delta['Score5_R1']>0
    decision=dict(status='IMPROVED_SCORE5' if improved else 'NO_SCORE5_IMPROVEMENT',delta_vs_RandomK4epoch_pp=delta,
      delta_vs_own500_pp=parent_delta,
      note='Primary comparison is matched4868 updates/H4868/native5. Parent500 is a budget trajectory reference, not an equally trained baseline.',
      long3_improved=delta['J_long3']>0,short4_improved=delta['Short4_R1']>0,training_complete=True,
      epochs=4,updates=4868,no_extra_doses=True,no_fifth_epoch=True)
    r.update(status='COMPLETE',scores=rs,resources=resources,decision=decision,delta_vs_RandomK4epoch_pp=delta,
      per_dataset_R1_delta_pp={ds:{dr:100*(r['metrics'][ds][dr]['R@1']-ref['metrics'][ds][dr]['R@1']) for dr in ['I2T','T2I']} for ds in DATASETS})
    dump(EXP/'RESULTS.json',r);dump(EXP/'DECISION.json',decision);dump(EXP/'RESOURCE_SUMMARY.json',resources)
    lines=['# Arm B S0.2 full four-epoch result','',f'**{decision["status"]}**. Total4868 optimizer updates are complete; resumed exact same trial from500 with unchangedH4868.','',
      '| Model | Updates | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|---:|']
    for name,updates,s in [('RandomK formal4epoch',4868,bs),('Arm B S0.2 parent',500,ps),('Arm B S0.2 full4epoch',4868,rs)]:lines.append(f'| {name} | {updates} | '+' | '.join(f'{s[k]*100:.6f}' for k in METRICS)+' |')
    lines+=['','| Comparison | ΔScore5 pp | ΔJ_long3 pp | ΔJ_long pp | ΔShort4 pp |','|---|---:|---:|---:|---:|']
    for name,d in [('vs matched RandomK4epoch',delta),('vs own500 checkpoint (longer budget)',parent_delta)]:lines.append('| '+name+' | '+' | '.join(f'{d[k]:+.6f}' for k in METRICS)+' |')
    lines+=['','| Dataset | S0.2 I2T R1 | S0.2 T2I R1 | RandomK I2T/T2I R1 | ΔI2T/T2I pp |','|---|---:|---:|---|---|']
    for ds in DATASETS:
        m=r['metrics'][ds];d=r['per_dataset_R1_delta_pp'][ds]
        lines.append(f'| {ds} | {m["I2T"]["R@1"]*100:.6f} | {m["T2I"]["R@1"]*100:.6f} | '+' / '.join(f'{ref["metrics"][ds][dr]["R@1"]*100:.6f}' for dr in ['I2T','T2I'])+' | '+ ' / '.join(f'{d[dr]:+.6f}' for dr in ['I2T','T2I'])+' |')
    lines+=['','| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|']
    for ds in DATASETS:lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{r["metrics"][ds][dr][k]*100:.6f}' for k in ['R@1','R@5','R@10']) for dr in ['I2T','T2I'])+' |')
    lines+=['','## Recipe and continuation correctness','',
      'F/S/D weights[1.4,0.2,1.4], Summary=first visible sentence, same deterministic random-detail subset sampler, Balanced-Stack-Patch ViT-B/16.4 GPUs×256/rank,accumulation1,seed0,encoder BF16/mask and loss FP32. Optimizer/LR groups, sparsity/inclusion/ramp and inference are unchanged.','',
      'Parent full checkpoint SHA256:a15c5360fa34991fd0e9767ff158c232c97f1414762b7f58540ec40701e7769e. The parent had500 AdamW steps and horizon4868. Model/fusion weights, full optimizer state and all four rank RNG states were restored; no LR/warmup reset. Resume validator requires byte-identical production code and exact trial/data/sampling/hparams. First resumed update501, final4868;4368 new updates.','',
      'Checkpoint interval1217 is an I/O-only change. Complete epoch-boundary checkpoints are retained at1217,2434,3651,4868. Original tail180/rank/global720 is preserved once per epoch, with256/rank/global1024 otherwise. All4368×4 sample-ID/epoch/batch-size streams match the official DistributedSampler.','',
      'The first preflight attempt used the wrong CPU RNG metadata field name and stopped before launching training; the report-only field was corrected tocpu. Failed preflight log is preserved.28 resume/scheduler/sampling regression tests passed.','',
      'Strict bare export is verified; frozen native5 uses normalized image/full-caption embeddings only. Long-DCI remains7602/7602,manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b. Neither training masks nor Summary/Detail inference views are used.','',
      f'Normal continuation full-cycle mean {resources["normal_full_cycle_mean_seconds"]:.6f}s,peak allocated {resources["peak_allocated_gib"]:.6f}GiB. Checkpoint writes are recorded separately from normal cycle time.','',
      'RandomK reference is the published formal4epoch4868 result under the same horizon/evaluator; it is reused without retraining. Its sampling/weights differ, so this comparison assesses the full S0.2 method against the formal RandomK method.','',
      'No S0.0 full run, no other dose, no fifth epoch. Training has stopped after completing the requested4 epochs. Checkpoints/datasets stay outside Git; complete small compressed log chunks and raw metrics are committed.', '',
      'Branch:'+BRANCH+'.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    state=load(RUN/'state.json');state.update(status='EXPERIMENT_COMPLETED',stage='final report generated; pending GitHub synchronization',decision=decision,finished_at=now());dump(RUN/'state.json',state)
    print(json.dumps({'scores_percent':{k:v*100 for k,v in rs.items()},'delta_vs_RandomK4epoch_pp':delta,'decision':decision}),flush=True)

if __name__=='__main__':generate()
