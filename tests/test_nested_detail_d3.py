"""Strict subset boundaries, uniform sampling, RNG isolation and frozen F/Dall."""
from collections import Counter
import itertools
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from train import nested_semantic_data as data
from recovery.nested_detail_d3_equal500 import changed_config,classify,gradient_pressure,SCORES


@pytest.mark.parametrize('m,k',[(0,0),(1,1),(2,1),(3,2),(4,3),(5,3),(8,3)])
def test_strict_subset_boundaries_and_order(m,k):
    for sid in range(50):
        selected=data.sample_partial_detail_indices(m+1,0,0,sid)
        assert len(selected)==k and selected==sorted(set(selected))
        assert all(1<=i<=m for i in selected)
        if m>=2:assert len(selected)<m


def test_uniform_without_replacement():
    counts=Counter(tuple(data.sample_partial_detail_indices(6,0,0,s)) for s in range(6000))
    possible=set(itertools.combinations(range(1,6),3))
    assert set(counts)==possible
    expected=6000/len(possible)
    assert all(abs(n-expected)<expected*.2 for n in counts.values())


def test_rng_untouched_and_epoch_sample_determinism():
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    caption='. '.join(['A bright city plaza']+[f'Detail {i} shows people walking' for i in range(8)])
    draws=[]
    for epoch in (0,1):
        for sid in range(20):
            a=data.sampled_text_views(caption,'nested_detail_d3',0,epoch,sid)
            b=data.sampled_text_views(caption,'nested_detail_d3',0,epoch,sid)
            assert a['views']==b['views'] and torch.equal(a['tokens_e'],b['tokens_e'])
            draws.append(tuple(a['detail_indices']))
    assert len(set(draws))>1
    assert random.getstate()==before[0] and torch.equal(before[2],torch.get_rng_state())
    after=np.random.get_state()
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]


@pytest.mark.parametrize('caption',[
    'One caption', 'Summary. Only one detail', 'Summary. First detail. Second detail',
    'Summary. First detail. Second detail. Third detail',
    'Summary. One detail. Two details. Three details. Four details. Five details',
    '. '.join(['A scene']+['Some long detail about a landscape '+('clouds '*70)]*8),
    'overlong '*300,
])
def test_full_dall_and_fallback_frozen(caption):
    old=data.sampled_text_views(caption,'nested_detail',0,0,42)
    new=data.sampled_text_views(caption,'nested_detail_d3',0,0,42)
    assert new['views'][:2]==old['views'][:2] and new['valid']==old['valid']
    assert torch.equal(new['tokens_f'],old['tokens_f']) and torch.equal(new['tokens_o'],old['tokens_o'])
    if not new['valid'] or new['detail_pool_size']==1:
        assert new['views']==old['views'] and torch.equal(new['tokens_e'],old['tokens_e'])
    else:
        indices=new['detail_indices'];parts=new['views'][0].split('. ')
        assert new['views'][2]=='. '.join(parts[i] for i in indices)
        assert set(indices)<set(new['dall_indices']) and 0 not in indices


@pytest.mark.parametrize('invalid_only',[False,True])
def test_sampling_diagnostics_include_all_records_and_degenerates(invalid_only):
    captions=['One caption','Summary. One detail'] if invalid_only else [
        'Summary. First detail. Second detail. Third detail. Fourth detail','One caption','Summary. One detail']
    if invalid_only:captions=['One caption','overlong '*300]
    samples=[dict(image=torch.zeros(3,2,2),image_id=i,sample_id=i,
        **data.sampled_text_views(c,'nested_detail_d3',0,0,i)) for i,c in enumerate(captions)]
    diag=data.sampling_diagnostics(data.collate(samples))
    assert diag['nested_d3_exact']
    stats=diag['nested_d3_statistics']
    assert stats['records']==len(samples) and sum(stats['K_eff_histogram'].values())==len(samples)
    assert stats['degenerate_samples']==2
    assert stats['strict_subset_samples']==(0 if invalid_only else 1)


def test_only_lowest_sampling_config_changes():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/config.json').read_text())
    new=json.loads((root/'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/config.json').read_text())
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'sampling_mode'}
    changed_config(new)
    with pytest.raises(AssertionError,match='Config drift'):changed_config(dict(new,view_weights=[1.4,1.4,.2]))


def test_runtime_provenance_fields_do_not_reject_frozen_config():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/config.json').read_text())
    changed_config(dict(c,completed_steps=5,code_sha256={'trainer':'example'},resume=None))
    with pytest.raises(AssertionError,match='Config drift'):
        changed_config(dict(c,workers=4,completed_steps=5))


@pytest.mark.parametrize('delta,urban,dominant,status',[
    (0,88.5,False,'D3_EQUAL_WEIGHT_STRONG_POSITIVE'),
    (-.1,88.3,False,'D3_OPTIMIZATION_POSITIVE'),
    (-.3,88.6,False,'D3_TRADEOFF'),
    (-.3,88.0,True,'D3_EQUAL_WEIGHT_NEGATIVE'),
])
def test_predeclared_classification(delta,urban,dominant,status):
    r=classify(dict(Score5=SCORES['Score5']+delta,J_long3=SCORES['J_long3']+delta),
        {'Urban-1k':{'I2T':{'R@1':.91},'T2I':{'R@1':urban/100}}},
        dict(CE_ratio_vs_atomic_equal=.5,alignment_share_drop_pp=30,gradient_dominant=dominant,
            both_gradient_ratio_improvement=True))
    assert r['status']==status and not r['automatic_continuation']


def test_dominance_same_protocol_as_atomic():
    g=dict(mean_gradient_norms=dict(F=1.,Dall=1.,D3=3.),
        batches=[dict(gradient_norms=dict(F=1.,Dall=1.,D3=3.)) for _ in range(8)])
    p=gradient_pressure(g)
    assert p['dominant'] and p['D3_larger_than_F_plus_Dall_batches']==8
