"""The user-selected policy uses only global500 scores and exact-tie auxiliaries."""
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import select_top2_500


def trial(score,long3=.7,long=.8,other=None):
    budgets={'500':{'scores':dict(Score5_R1=score,J_long3=long3,J_long=long)}}
    if other is not None:
        budgets['1217']={'scores':dict(Score5_R1=other,J_long3=long3,J_long=long)}
    return dict(budgets=budgets)


def test_top2_uses_all_historical_500_trials_without_1217_reranking():
    trials={'b0':trial(.698),'early':trial(.705,other=.1),'latest':trial(.703,other=.9),
            'failed':{'budgets':{}},'low':trial(.69)}
    assert select_top2_500(trials)==['early','latest']


def test_top2_preserves_raw_score_priority_and_tie_breakers():
    trials={'rounded_lower':trial(.70000001,.99,.99),'higher':trial(.70000002,.1,.1),
            'tie':trial(.70000001,.98,.99)}
    assert select_top2_500(trials)==['higher','rounded_lower']
