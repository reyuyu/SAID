"""Bounded subset, legacy replay, visibility/RNG and diagnostic math isolation."""
import ast
import copy
import random
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from model import longclip
from train import nested_semantic_data as data
from tests.test_balanced_hparams import make, inputs
from train.random_detail_observer import install
from train.train_nested_semantic_mask import build_optimizer


@pytest.mark.parametrize('n', [2,3,4,5,8,12])
def test_K_rules_and_preserved_order(n):
    parts = [f'Part {i}' for i in range(n-1)]+[f'Part {n-1}.']
    caption = '. '.join(parts)
    for sample_id in range(128):
        v = data.sampled_text_views(caption, 'summary_random_detail', 0, 0, sample_id)
        old = data.sampled_text_views(caption, 'random_k', 0, 0, sample_id)
        assert torch.equal(v['tokens_f'], old['tokens_f']) and v['views'][0]==old['views'][0]
        assert v['views'][1]==parts[0] and v['valid']
        indexes=v['detail_indices']; assert indexes==sorted(set(indexes)) and 0 not in indexes
        assert v['views'][2]=='. '.join(parts[i] for i in indexes)
        assert v['K']==len(indexes)
        m=n-1
        assert v['K']==1 if m<=2 else 2<=v['K']<=m-1
        assert torch.equal(v['tokens_e'],longclip.tokenize([v['views'][2]],truncate=False)[0])


@pytest.mark.parametrize('caption', ['Only one.', 'word '*300, 'word '*300+'. A cat.'])
def test_no_complete_visible_detail_fallback(caption):
    v=data.sampled_text_views(caption,'summary_random_detail')
    old=data.sampled_text_views(caption,'random_k')
    assert not v['valid'] and v['detail_pool_size']==0 and not v['detail_indices']
    assert v['views'][1:]==[None,None]
    assert not v['tokens_o'].any() and not v['tokens_e'].any()
    assert torch.equal(v['tokens_f'],old['tokens_f'])


def test_raw_trailing_sentences_are_never_used():
    caption='A street. A bus. A dog. A shop. '+'forbidden '*300
    for sid in range(32):
        v=data.sampled_text_views(caption,'summary_random_detail',sample_id=sid)
        assert 'forbidden' not in v['views'][0] and 'forbidden' not in v['views'][2]
        assert v['n']==4 and v['detail_pool_size']==3 and v['K']==2


def test_independent_replay_epoch_change_and_global_rng():
    py=random.getstate(); np_state=np.random.get_state(); t=torch.get_rng_state().clone()
    n=9
    a=[data.sample_detail_indices(n,0,0,i) for i in range(256)]
    b=[data.sample_detail_indices(n,0,1,i) for i in range(256)]
    assert a==[data.sample_detail_indices(n,0,0,i) for i in range(256)]
    assert any(x!=y for x,y in zip(a,b))
    assert random.getstate()==py and torch.equal(t,torch.get_rng_state())
    now=np.random.get_state(); assert now[0]==np_state[0] and np.array_equal(now[1],np_state[1]) and now[2:]==np_state[2:]


def test_old_modes_are_unchanged_in_outputs():
    root=Path(__file__).resolve().parents[1]
    old=subprocess.check_output(['git','show','2cb9081:train/nested_semantic_data.py'],cwd=root,text=True)
    names=['text_views','sample_split_k','sampled_text_views']
    nodes=[n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'longclip':longclip,'torch':torch,'hashlib':__import__('hashlib'),'random':random}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'old_sampling','exec'),scope)
    for mode in ['fixed_first','random_k','summary_detail']:
        for caption in ['One. Two. Three. Four.','One.','A scene. '+ 'word '*300]:
            a=scope['sampled_text_views'](caption,mode,0,0,99);b=data.sampled_text_views(caption,mode,0,0,99)
            assert a.keys()==b.keys()
            for k in a:
                assert torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k]


@pytest.mark.parametrize('valid_case',['all','some'])
def test_readonly_F_D_iou_preserves_loss_gradients_and_AdamW(valid_case):
    torch.manual_seed(713)
    old=make();new=copy.deepcopy(old)
    images,views,valid=inputs()
    if valid_case=='all':valid.fill_(True)
    loss_old,logs_old=old(images,*views,valid,0);loss_old.backward()
    restore=install()
    try:
        loss_new,logs_new=new(images,*views,valid,0);loss_new.backward()
    finally:restore()
    torch.testing.assert_close(loss_old,loss_new,atol=0,rtol=0)
    assert 'E_F_D_positive_mask_iou' in logs_new
    assert 0<=float(logs_new['E_F_D_positive_mask_iou'])<=1
    for n,p in old.named_parameters():
        q=dict(new.named_parameters())[n]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=n)
    build_optimizer(old).step();build_optimizer(new).step()
    for n,p in old.named_parameters():torch.testing.assert_close(p,dict(new.named_parameters())[n],atol=0,rtol=0,msg=n)


def test_readonly_observer_resets_after_full_only_fallback():
    torch.manual_seed(714)
    model=make();images,views,valid=inputs()
    restore=install()
    try:
        _,fallback=model(images,*views,torch.zeros_like(valid),0)
        assert 'E_F_D_positive_mask_iou' not in fallback
        _,regular=model(images,*views,valid,99)
        assert 'E_F_D_positive_mask_iou' in regular
        assert 0<=float(regular['E_F_D_positive_mask_iou'])<=1
    finally:restore()
