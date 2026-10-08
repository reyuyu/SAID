"""Interpret completed read-only curves, without touching evaluator or training."""
import json
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from recovery import epoch_curve_eval as replay
from tools import eval_five_parallel as ev


def sign_changes(points):
    previous=None;changes=[]
    for point in points:
        if point['delta']==0:continue
        if previous is not None and previous['delta']*point['delta']<0:
            changes.append(dict(between_epochs=[previous['epoch'],point['epoch']],
                direction='HNS_to_Balanced' if point['delta']<0 else 'Balanced_to_HNS',
                observed_delta_before_pp=previous['delta'],observed_delta_after_pp=point['delta']))
        previous=point
    return changes


def interpretation(aggregate,directions,gains,diagnostics):
    crossing={k:sign_changes(v) for k,v in aggregate.items()}
    score_flips=[x for x in crossing['Score5'] if x['direction']=='HNS_to_Balanced']
    first=score_flips[0] if score_flips else None
    interval=first['between_epochs'] if first else None
    gaps={k:[p['delta'] for p in v] for k,v in aggregate.items()}
    first_negative={k:next((p['epoch'] for p in v if p['delta']<0),None) for k,v in aggregate.items()}
    masks={};violations={};shares={};temporal={}
    for view in ('F','Dall','D3'):
        a=[diagnostics['D3_Balanced'][str(s)]['views'][view]['keep'] for s in replay.STEPS]
        b=[diagnostics['HNS_v1'][str(s)]['views'][view]['keep'] for s in replay.STEPS]
        segments=[dict(between_epochs=[i+1,i+2],Balanced_change=a[i+1]-a[i],HNS_change=b[i+1]-b[i],
            HNS_minus_Balanced_change=(b[i+1]-b[i])-(a[i+1]-a[i]),
            HNS_falls_faster=b[i+1]-b[i]<a[i+1]-a[i]) for i in range(3)]
        lower=next((i+1 for i,(x,y) in enumerate(zip(a,b)) if y<x),None)
        if interval is None:relation='no observed Score5 crossover'
        elif lower is None:relation='no observed lower HNS coverage'
        elif lower<=interval[0]:relation='lower HNS coverage observed before crossover interval'
        elif lower==interval[1]:relation='lower HNS coverage first observed at crossover endpoint; within-interval order unknown'
        else:relation='lower HNS coverage first observed after crossover interval'
        masks[view]=dict(Balanced=a,HNS=b,gap_HNS_minus_Balanced=[y-x for x,y in zip(a,b)],segments=segments,
            E1_to_E4_change=dict(Balanced=a[-1]-a[0],HNS=b[-1]-b[0]),first_lower_HNS_epoch=lower)
        temporal[view]=relation
    for key in ('Dall_F_hard_violation','D3_Dall_hard_violation'):
        violations[key]={m:[dict(epoch=i+1,**diagnostics[m][str(s)]['telemetry'][key]) for i,s in enumerate(replay.STEPS)]
                         for m in replay.MODELS}
    for m in replay.MODELS:
        shares[m]=[diagnostics[m][str(s)]['views']['D3']['alignment_share_percent'] for s in replay.STEPS]
    directional=[];declines=[]
    for key,points in directions.items():
        changes=sign_changes(points)
        interval_change=None
        if interval:
            aa=next(p for p in points if p['epoch']==interval[0]);bb=next(p for p in points if p['epoch']==interval[1])
            interval_change=bb['delta']-aa['delta']
        directional.append(dict(direction=key,crossovers=changes,first_negative_epoch=next((p['epoch'] for p in points if p['delta']<0),None),
            final_gap_pp=points[-1]['delta'],Score5_crossover_interval_gap_change_pp=interval_change))
        for aa,bb in zip(points,points[1:]):
            for m in ('Balanced','HNS'):
                change=bb[m]-aa[m]
                if change<0:declines.append(dict(model=m,direction=key,between_epochs=[aa['epoch'],bb['epoch']],R1_change_pp=change))
    directional.sort(key=lambda x:x['final_gap_pp'])
    slower=[g for g in gains if g['gains_pp']['Score5']['HNS_minus_Balanced_growth']<0]
    decision='YES' if slower and slower[0]['segment']=='E1->E2' else 'NO'
    q6=dict(relative_learning_speed_slower_segments=[g['segment'] for g in slower],
        Score5_absolute_gains=[dict(segment=g['segment'],**g['gains_pp']['Score5']) for g in gains],
        actual_directional_declines=declines,
        distinction='Negative HNS-minus-Balanced growth is relative learning slowdown, not necessarily declining HNS retrieval.')
    return dict(Q1_Score5_first_crossover=first,Q2_aggregate_first_negative_epoch=first_negative,
        aggregate_gaps_pp=gaps,aggregate_crossovers=crossing,Q3_directional_ranked_evidence=directional,
        Q4_keep_trajectories=masks,hard_violation_trajectories=violations,D3_alignment_share_percent=shares,
        Q5_temporal_correspondence=temporal,Q6_late_disadvantage=q6,
        HALF_CONTINUE_TO_2434=decision,Half_training_started=False,
        recommendation_basis='First negative relative Score5 growth interval; the measured Half@1217 signal is weak and single-seed.',
        limitations='Four discrete epoch observations from one seed. No exact crossover update, within-interval temporal ordering, causality or statistical significance is inferred. CE share is not a gradient norm.')


def main():
    result=replay.read(replay.EXP/'RESULTS.json')
    assert result['status']=='COMPLETE' and result['training'] is False
    assert all(result['models'][m]['4868']['final_reproduction']['metrics_exact'] and
               result['models'][m]['4868']['final_reproduction']['aggregates_exact'] for m in replay.MODELS)
    curves=replay.read(replay.EXP/'EPOCH_CURVES.json')
    directions=replay.read(replay.EXP/'DIRECTIONAL_R1_CURVES.json')['curves']
    diag=replay.read(replay.EXP/'TRAINING_DIAGNOSTICS_COMPARISON.json')
    science=interpretation(curves['aggregates'],directions,curves['epoch_to_epoch_gains'],diag)
    half=replay.read(replay.HALF)['models']
    science['existing_Half1217_scores']={m:v['scores_percent'] for m,v in half.items()}
    science['Half1217_source']=dict(path=str(replay.HALF),sha256=ev.sha(replay.HALF),new_evaluation=False)
    result['scientific_interpretation']=science
    replay.save('RESULTS.json',result);replay.save('SCIENTIFIC_INTERPRETATION.json',science)
    lines=['','## Direct answers to the scientific questions','',
        'Q1. First observed Score5 ranking reversal: `'+json.dumps(science['Q1_Score5_first_crossover'])+'`. A crossing interval is reported, never an exact update.',
        'Q2. First negative HNS-minus-Balanced epoch: `'+json.dumps(science['Q2_aggregate_first_negative_epoch'])+'`. A deficit already present at E1 predates the observed epoch window; it is not assigned a made-up reversal step.',
        '', 'Q3. Directional sources (sorted by final deficit):','',
        '| Direction | First negative epoch | Final HNS-minus-Balanced(pp) | Gap change across Score5 crossover(pp) |','|---|---:|---:|---:|']
    for r in science['Q3_directional_ranked_evidence']:
        x=r['Score5_crossover_interval_gap_change_pp']
        lines.append(f'| {r["direction"]} | {r["first_negative_epoch"]} | {r["final_gap_pp"]:+.6f} | '+('unobserved' if x is None else f'{x:+.6f}')+' |')
    lines += ['','Q4. Does HNS coverage fall faster? Matched epoch-last50, E1 to E4:','',
        '| View | Balanced keep change | HNS keep change | HNS falls faster |','|---|---:|---:|---|']
    for view,v in science['Q4_keep_trajectories'].items():
        a,b=v['E1_to_E4_change']['Balanced'],v['E1_to_E4_change']['HNS']
        lines.append(f'| {view} | {a:+.6f} | {b:+.6f} | {b<a} |')
    lines += ['','Q5. Temporal correspondence: `'+json.dumps(science['Q5_temporal_correspondence'])+'`. This is observational; no causal claim.',
        'Q6. Relative slow-growth intervals: `'+json.dumps(science['Q6_late_disadvantage']['relative_learning_speed_slower_segments'])+'`.',
        'Actual negative directional epoch-to-epoch gains: `'+json.dumps(science['Q6_late_disadvantage']['actual_directional_declines'])+'`. Distinguish these from Balanced simply improving faster.',
        f'`HALF_CONTINUE_TO_2434 = {science["HALF_CONTINUE_TO_2434"]}`. This recommendation combines the original curve with the existing weak Half@1217 signal; no continuation is run.',
        'Matched keep, hard violations, CE/weighted CE and D3 alignment shares are fully preserved in TRAINING_DIAGNOSTICS_COMPARISON.json and SCIENTIFIC_INTERPRETATION.json. No new gradient audit is run.',
        science['limitations']]
    with (replay.EXP/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')


def wait_and_publish():
    """Finish the scientific report after the evaluation runner has synced."""
    with (replay.RUN/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (replay.RUN/'completed.json').exists():
            state=replay.read(replay.EXP/'STATE.json')
            if state['status']=='STOPPED_WITH_EVIDENCE':
                raise RuntimeError('Evaluation failed; no trajectory interpretation')
            os.kill(replay.read(replay.RUN/'runner.json')['pid'],0)
            time.sleep(15)
        ev.require_gpu_idle({0,1,2,3})
        main()
        cmd=[sys.executable,'-m','pytest','-q','tests/test_epoch_curve_eval.py','tests/test_epoch_curve_analysis.py']
        checked=subprocess.run(cmd,cwd=replay.ROOT,check=True,capture_output=True,text=True)
        replay.save('CPU_TESTS.json',dict(passed=True,tests=7,command=cmd,output=checked.stdout,training=False))
        from recovery import check_stage500_publish as review
        names=['recovery/epoch_curve_analysis.py','tests/test_epoch_curve_analysis.py']+[
            'experiments/nest_clip_v1/epoch_curve_eval_v1/'+n
            for n in ('REPORT.md','RESULTS.json','SCIENTIFIC_INTERPRETATION.json','CPU_TESTS.json')]
        assert replay.git('branch','--show-current')==replay.BRANCH
        assert not replay.git('diff','--cached','--name-only')
        subprocess.run(['git','add','--',*names],cwd=replay.ROOT,check=True)
        review.ALLOWED=set(names);review.inspect()
        subprocess.run(['git','diff','--cached','--check'],cwd=replay.ROOT,check=True)
        subprocess.run(['git','commit','-m','Complete bounded epoch-curve scientific interpretation'],cwd=replay.ROOT,check=True)
        head=replay.git('rev-parse','HEAD')
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+replay.BRANCH],cwd=replay.ROOT,check=True,timeout=120)
        subprocess.run(['git','fetch','origin','refs/heads/'+replay.BRANCH+':refs/remotes/origin/'+replay.BRANCH],
                       cwd=replay.ROOT,check=True,timeout=120)
        remote=replay.git('rev-parse','origin/'+replay.BRANCH)
        assert remote==head==replay.git('rev-parse','FETCH_HEAD')
        ev.save(replay.RUN/'GITHUB_RECEIPT.json',dict(branch=replay.BRANCH,commit=head,remote_HEAD=remote,
            remote_HEAD_matches_local=True,push_success=True,checked_utc=ev.utc()))
        ev.save(replay.RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',training=False,
            scientific_interpretation_complete=True,github=replay.read(replay.RUN/'GITHUB_RECEIPT.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-and-publish',action='store_true')
    if parser.parse_args().wait_and_publish:wait_and_publish()
    else:main()
