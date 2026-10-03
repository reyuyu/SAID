"""Selection guard, strict near-tie boundary and no unsafe champion."""
import copy
from experiments.nest_clip_v1.four_arm_text_search_500_v1.report import selection,GUARD
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load

def arm(score,long,urban=.88,dcit=.56):
    r=copy.deepcopy(load(EXP/'BASELINE.json'));r['status']='COMPLETE';r['scores'].update(Score5_R1=score,J_long3=long,Short4_R1=.6435)
    r['metrics']['Urban-1k']['T2I']['R@1']=urban;r['metrics']['Long-DCI']['T2I']['R@1']=dcit
    return r

def test_no_unsafe_champion_even_if_best_score_exceeds_baseline():
    s=selection({'A':arm(.71,GUARD-.001),'B':arm(.72,GUARD-.01)})
    assert s['status']=='NO_SAFE_WINNER' and s['champion'] is None and s['best_observed_Score5_arm']=='B'

def test_guard_equality_eligible_and_near_tie_prefers_long():
    s=selection({'A':arm(.705,GUARD),'B':arm(.7046,.74)})
    assert s['status']=='WINNER' and s['champion']=='B' and set(s['eligible_arms'])=={'A','B'}

def test_score_distance_outside_near_tie_prefers_score():
    s=selection({'A':arm(.705,GUARD),'B':arm(.7044,.75)})
    assert s['champion']=='A'

def test_equal_long_uses_urban_then_dci_then_short():
    s=selection({'A':arm(.705,.74,.89,.57),'B':arm(.705,.74,.90,.56)})
    assert s['champion']=='B'
    s=selection({'A':arm(.705,.74,.90,.57),'B':arm(.705,.74,.90,.56)})
    assert s['champion']=='A'

def test_safe_but_no_overall_gain_is_not_WINNER():
    s=selection({'A':arm(.698,GUARD)})
    assert s['status']!='WINNER' and not s['overall_improvement']
