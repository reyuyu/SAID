"""Decision boundaries and diagnostic contribution arithmetic."""
import pytest

from recovery.s02_all_detail500 import classify, diagnostics


@pytest.mark.parametrize('urban,score,long_score,expected', [
    (89,70.264367,73.403324,'URBAN_STRONG_POSITIVE'),
    (88.5,70.164367,73.403324,'URBAN_POSITIVE'),
    (89,70.264367,73.403323,'NEGATIVE'),
    (88,71,74,'NEGATIVE'),
    (88.4,71,74,'INCONCLUSIVE'),
])
def test_decision_boundary(urban,score,long_score,expected):
    result=classify(dict(Score5=score,J_long3=long_score),{'Urban-1k':{'T2I':{'R@1':urban/100}}})
    assert result['status']==expected
    assert not result['automatic_continuation'] and not result['automatic_new_experiments']


def test_diagnostics_weights_and_token_coverage():
    row=dict(step=500,inc=2,inc_weight=1,oe_iou=.4,hard_inclusion_violation=.1,loss=42)
    for p in ('F','O','E'):
        row.update({p+'_i2t':2,p+'_t2i':1,p+'_keep_ratio':.5,p+'_positive_keep_ratio':.6})
    s=dict(Full_Summary_Detail_token_statistics={'Detail':dict(samples=2,effective_token_sum=24)},
           random_detail_sampling={'selected_count_histogram':{'3':2}},
           detail_full_token_coverage=dict(full_content_token_sum=40,per_sample_ratio_sum=1,samples=2))
    row['rank_health']=[dict(sampling=s)]
    result=diagnostics([row])
    assert result['last50_views']['F']['weighted_alignment_contribution']==pytest.approx(14)
    assert result['last50_views']['S']['weighted_alignment_contribution']==pytest.approx(2)
    assert result['last50_views']['D']['weighted_alignment_contribution']==pytest.approx(14)
    assert result['D_mean_sentences']==3 and result['D_mean_content_tokens']==10
    assert result['D_F_content_token_coverage_pooled']==.5


@pytest.mark.parametrize('raw,expected', [
    (.8899999856948853, 'URBAN_STRONG_POSITIVE'),
    (.8849999904632568, 'URBAN_POSITIVE'),
])
def test_native_float32_urban_threshold_representation(raw,expected):
    result=classify(dict(Score5=71,J_long3=74),{'Urban-1k':{'T2I':{'R@1':raw}}})
    assert result['status']==expected
