"""Exact weighted keep-fraction summaries and interval boundaries."""
import math

import pytest

from experiments.nest_clip_v1.balanced_rdrop_500_v1.run import summarize_sampling
from experiments.nest_clip_v1.balanced_rdrop_500_v1.report import write_report,DATASETS


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


def test_report_handles_evaluation_callback_before_statistics(tmp_path):
    record=dict(scores=dict(Score5_R1=.7,J_long3=.7,J_long=.7),checkpoint_sha256='full',bare_sha256='bare',
                checkpoint='/full.pt',student='/bare.pt',metrics={
                    ds:{dr:{f'R@{k}':.7 for k in (1,5,10)}for dr in ('I2T','T2I')}for ds in DATASETS})
    state=dict(status='running',baseline=record,historical_rmask=record['scores'],rdrop_trial='trial',
               trials={'trial':dict(budgets={'500':record})})
    write_report(state,tmp_path)
    assert 'pending aggregation' in (tmp_path/'RDROP_500_REPORT.md').read_text()
