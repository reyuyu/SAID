"""Meaningful controls for fresh full-run boundaries and predeclared decisions."""
import copy
import json
import pytest

from recovery.nested_detail_d3_balanced_full import CONFIG,REFERENCE_EXP,frozen_config
from recovery.nested_detail_d3_balanced_full_evidence import decision,view_means


def baseline():
    return dict(scores_percent=dict(Score5=73.209163,J_long3=77.137938,J_long=85.945002,Short4=67.316),
        metrics={'Urban-1k':dict(I2T={'R@1':.928},T2I={'R@1':.918})})


def test_full_config_no_method_changes():
    actual=json.loads(CONFIG.read_text());expected=json.loads((REFERENCE_EXP/'config.json').read_text())
    assert actual==expected
    frozen_config(actual)


@pytest.mark.parametrize('key,value',[
    ('sampling_seed',1),('view_weights',[1.,1.,1.]),('inclusion_max',2),('sparsity_scale',.9),('workers',4),('epochs',3)])
def test_frozen_drift_rejected(key,value):
    c=json.loads(CONFIG.read_text());c[key]=value
    with pytest.raises(AssertionError):frozen_config(c)


@pytest.mark.parametrize('delta,urban,status',[
    ({'Score5':.2,'J_long3':.3,'Short4':-.1}, {'I2T':92.8,'T2I':92.0},'FULL_STRONG_POSITIVE'),
    ({'Score5':.2,'J_long3':-.1}, {'I2T':92.8,'T2I':91.8},'FULL_POSITIVE'),
    ({'Score5':-.4,'J_long3':.3,'Short4':-.3}, {'I2T':92.8,'T2I':92.0},'FULL_TRADEOFF'),
    ({'Score5':-.4,'J_long3':-.3}, {'I2T':92.5,'T2I':91.5},'FULL_NEGATIVE'),
    ({}, {'I2T':92.8,'T2I':91.8},'FULL_INCONCLUSIVE')])
def test_classification_boundaries(delta,urban,status):
    b=baseline();s=dict(b['scores_percent'])
    s.update({k:s[k]+v for k,v in delta.items()})
    assert decision(s,urban,b)['status']==status


def test_weighted_diagnostic_reference():
    row={}
    for p,x in [('F',.1),('O',.2),('E',1.2)]:
        row.update({p+'_'+k:v for k,v in dict(i2t=x/2,t2i=x/2,keep_ratio=.8,
            g_mean=.3,g_variance=.01,g_saturation=.02).items()})
    actual=view_means([row]);expected={'F':.45,'Dall':.9,'D3':1.2}
    for view,value in expected.items():
        assert actual[view]['weighted_CE']==pytest.approx(value)
        assert actual[view]['alignment_share_percent']==pytest.approx(100*value/sum(expected.values()))
