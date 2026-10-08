from recovery import epoch_curve_analysis as analysis


def test_sign_changes_preserve_unresolved_zero_intervals():
    points=[dict(epoch=1,delta=.1),dict(epoch=2,delta=0),dict(epoch=3,delta=-.1)]
    found=analysis.sign_changes(points)
    assert found[0]['between_epochs']==[1,3]
    assert found[0]['direction']=='HNS_to_Balanced'
    assert analysis.sign_changes([dict(epoch=1,delta=0),dict(epoch=2,delta=-.1)])==[]


def test_relative_slowdown_is_not_absolute_decline_and_temporal_claim_is_bounded():
    from recovery import epoch_curve_eval as replay
    points=[dict(epoch=i+1,delta=d,Balanced=70+i,HNS=70+i+d)
            for i,d in enumerate((.1,.2,-.1,-.2))]
    aggregates={k:points for k in replay.KEYS}
    gains=[dict(segment=f'E{i+1}->E{i+2}',gains_pp={
        'Score5':dict(Balanced=1,HNS=1+points[i+1]['delta']-points[i]['delta'],
                     HNS_minus_Balanced_growth=points[i+1]['delta']-points[i]['delta'])}) for i in range(3)]
    diagnostics={m:{} for m in replay.MODELS}
    for m in replay.MODELS:
        for i,step in enumerate(replay.STEPS):
            keep=.9-.1*i if m=='D3_Balanced' else .8-.08*i
            diagnostics[m][str(step)]=dict(views={v:dict(keep=keep,alignment_share_percent=50)
                for v in ('F','Dall','D3')},telemetry={k:dict(mean=.01,observations=50,steps=[],scope='epoch-last50')
                for k in ('Dall_F_hard_violation','D3_Dall_hard_violation')})
    r=analysis.interpretation(aggregates,{'Long-DCI/T2I':points},gains,diagnostics)
    assert r['Q1_Score5_first_crossover']['between_epochs']==[2,3]
    assert r['HALF_CONTINUE_TO_2434']=='NO'
    assert r['Q6_late_disadvantage']['actual_directional_declines']==[]
    assert r['Q6_late_disadvantage']['relative_learning_speed_slower_segments']==['E2->E3','E3->E4']
    assert 'before crossover interval' in r['Q5_temporal_correspondence']['D3']
    assert r['Q4_keep_trajectories']['D3']['E1_to_E4_change']['HNS']>r['Q4_keep_trajectories']['D3']['E1_to_E4_change']['Balanced']
