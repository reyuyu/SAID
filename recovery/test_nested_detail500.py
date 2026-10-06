"""Latest user decision gates including canonical Urban threshold rounding."""
import pytest
from recovery.nested_detail500 import classify


@pytest.mark.parametrize('urban,score,long_score,anomaly,status',[
    (88.7,70.684209,74.267014,False,'NESTED_DETAIL_STRONG_POSITIVE'),
    (88.3,70.48,74.,False,'NESTED_DETAIL_POSITIVE'),
    (88.2,70.8,74.4,False,'GLOBAL_POSITIVE_URBAN_INCONCLUSIVE'),
    (88.3,70.4,73.9,False,'URBAN_LONG_OR_GLOBAL_TRADEOFF'),
    (88.1,70.,73.,True,'ATOMIC_DETAIL_OVERCONSTRAINED'),
    (88.1,70.,73.,False,'NESTED_DETAIL_NEGATIVE'),
])
def test_decision(urban,score,long_score,anomaly,status):
    result = classify(dict(Score5=score,J_long3=long_score),
        {'Urban-1k':{'I2T':{'R@1':.9},'T2I':{'R@1':urban/100}}},anomaly)
    assert result['status'] == status and not result['automatic_continuation']


def test_float32_threshold_rounding():
    result = classify(dict(Score5=71,J_long3=75),
        {'Urban-1k':{'I2T':{'R@1':.9},'T2I':{'R@1':.8869999647140503}}})
    assert result['status'] == 'NESTED_DETAIL_STRONG_POSITIVE'
