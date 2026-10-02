"""Exact weighted keep-fraction summaries and interval boundaries."""
import math

import pytest

from experiments.nest_clip_v1.balanced_rdrop_500_v1.run import summarize_sampling


def test_population_summary_keeps_all_pairs_and_exact_bin_edges():
    records=[dict(rank_health=[dict(sampling=dict(sentence_drop_diagnostics=dict(
        valid_samples=5,m_counts={'4':4,'1':1},q_counts={'1':2,'2':1,'3':1,'4':1},
        joint_m_q_counts={'4:1':1,'4:2':1,'4:3':1,'4:4':1,'1:1':1},same_old_r_count=2)))])]
    result=summarize_sampling(records)
    assert result['valid_R_samples']==5
    assert result['keep_fraction_mean']==pytest.approx(.7)
    assert result['keep_fraction_std_population']==pytest.approx(math.sqrt(.085))
    assert result['keep_fraction_quantiles']['q25']==.5
    assert result['keep_fraction_quantiles']['q50']==.75
    assert result['keep_fraction_bins_count']=={'(0,0.25]':1,'(0.25,0.5]':1,'(0.5,0.75]':1,'(0.75,1)':0,'1.0':2}
    assert result['R_drop_equal_R_old_fraction']==.4
