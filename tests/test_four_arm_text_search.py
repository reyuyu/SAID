"""Preregistered bounds, exact historical replay and live directional gradients."""
import ast
import copy
import hashlib
import json
import mmap
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from model import longclip
from model.nested_fusion_mask import directional_ce
from train import nested_semantic_data as data
from tests.test_balanced_hparams import make, inputs
from train.train_nested_semantic_mask import build_optimizer

@pytest.mark.parametrize('n', [2,3,4,5,8])
def test_interior_complement_and_fallback(n):
    parts=[f'Sentence {i}' for i in range(n)]
    for sid in range(128):
        a=data.sampled_text_views('. '.join(parts),'interior_random_k',sample_id=sid)
        k=a['K']
        assert 2<=k<=n-2 if n>=4 else k==data.sample_split_k(n,0,0,sid)
        assert a['views'][1:]==['. '.join(parts[:k]),'. '.join(parts[k:])]
        assert '. '.join(a['views'][1:])==a['views'][0]

@pytest.mark.parametrize('n',[1,2,3,4,5,8])
def test_contiguous_visible_detail(n):
    parts=[f'Sentence {i}' for i in range(n)]
    for sid in range(64):
        v=data.sampled_text_views('. '.join(parts),'summary_contiguous_detail',sample_id=sid)
        if n==1:
            assert not v['valid'];continue
        ids=v['detail_indices'];assert ids==list(range(ids[0],ids[0]+len(ids)))
        assert 0 not in ids and v['views'][2]=='. '.join(parts[i] for i in ids)
        assert len(ids)==1 if n<4 else 2<=len(ids)<=n-2
    v=data.sampled_text_views('A. B. C. D. '+'forbidden '*300,'summary_contiguous_detail')
    assert 'forbidden' not in v['views'][2]

@pytest.mark.parametrize('mode',['interior_random_k','summary_contiguous_detail'])
def test_global_rng_and_image_stream(mode,tmp_path):
    from PIL import Image
    Image.fromarray(np.random.default_rng(4).integers(0,256,(320,400,3),dtype=np.uint8)).save(tmp_path/'image.png')
    index=tmp_path/'index';index.mkdir()
    raw=json.dumps({'image':'image.png','caption':'First. Second. Third. Fourth.'}).encode()+b'\n'
    (index/'records.jsonl').write_bytes(raw);np.save(index/'offsets.npy',np.array([0,len(raw)]))
    (index/'metadata.json').write_text(json.dumps({'training_records':1}))
    samples=[]
    for name in ['random_k',mode]:
        random.seed(10);np.random.seed(10);torch.manual_seed(10)
        samples.append(data.NestedDataset(index,tmp_path,name)[0])
    assert samples[0]['sample_id']==samples[1]['sample_id']
    assert torch.equal(samples[0]['image'],samples[1]['image'])
    assert torch.equal(samples[0]['tokens_f'],samples[1]['tokens_f'])
    py=random.getstate();ts=torch.get_rng_state().clone();ns=np.random.get_state()
    for sid in range(64):data.sampled_text_views('A. B. C. D. E.',mode,sample_id=sid)
    assert py==random.getstate() and torch.equal(ts,torch.get_rng_state())
    now=np.random.get_state();assert ns[0]==now[0] and np.array_equal(ns[1],now[1]) and ns[2:]==now[2:]


def test_summary_sampling_exact_1000_real_samples():
    root=Path(__file__).resolve().parents[1]
    old=subprocess.check_output(['git','show','3da12a3:train/nested_semantic_data.py'],cwd=root,text=True)
    nodes=[n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name in ['text_views','sample_split_k','sample_detail_indices','sampled_text_views']]
    scope=dict(longclip=longclip,torch=torch,hashlib=hashlib,random=random)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical','exec'),scope)
    index=Path('/root/lk_projects/SAID-nest-clip-v1/data_index');offset=np.load(index/'offsets.npy',mmap_mode='r')
    digest=hashlib.sha256()
    with (index/'records.jsonl').open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as records:
        for i in random.Random(0).sample(range(1245901),1000):
            caption=json.loads(records[offset[i]:offset[i+1]])['caption']
            a=scope['sampled_text_views'](caption,'summary_random_detail',0,0,i+1000)
            b=data.sampled_text_views(caption,'summary_random_detail',0,0,i+1000)
            assert a.keys()==b.keys()
            for key in a:assert torch.equal(a[key],b[key]) if torch.is_tensor(a[key]) else a[key]==b[key]
            for mode in ['interior_random_k','summary_contiguous_detail']:
                v=data.sampled_text_views(caption,mode,0,0,i+1000)
                assert v['views'][0]==a['views'][0] and torch.equal(v['tokens_f'],a['tokens_f'])
            digest.update(str(i+1000).encode());digest.update(b['tokens_e'].numpy().tobytes())
    out=root/'experiments/nest_clip_v1/four_arm_text_search_500_v1/evidence/sampling-1000.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'passed':True,'samples':1000,'seed':0,'epoch':0,'historical_commit':'3da12a3','B_C_all_fields_bitwise_equal':True,'A_D_F_raw_and_tokens_equal':True,'digest':digest.hexdigest()},indent=2)+'\n')


def test_directional_synthetic_formula_and_gradients():
    torch.manual_seed(91)
    scores=[torch.randn(5,5,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    ce=[(F.cross_entropy(s.T,torch.arange(5)),F.cross_entropy(s,torch.arange(5))) for s in scores]
    actual=(10/3)*(12/11)*sum(directional_ce(ci,ct,ci*0,.5 if i==1 else 1.) for i,(ci,ct) in enumerate(ce))
    expected=(10/3)*(12/11)*(ce[0][0]+ce[0][1]+ce[1][0]+.5*ce[1][1]+ce[2][0]+ce[2][1])
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-14)
    ga=torch.autograd.grad(actual,scores,retain_graph=True);ge=torch.autograd.grad(expected,scores)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=0,atol=1e-15)


def test_default_directional_loss_gradients_optimizer_exact():
    torch.manual_seed(94)
    root=Path(__file__).resolve().parents[1]
    oldsource=subprocess.check_output(['git','show','3da12a3:model/balanced_hparam_search.py'],cwd=root,text=True)
    scope={'__name__':'historical_balanced'};exec(compile(oldsource,'historical_balanced','exec'),scope)
    old=make();old.forward=scope['BalancedSearch'].forward.__get__(old)
    new=make({'summary_t2i_weight':1.});new.load_state_dict(old.state_dict())
    images,views,valid=inputs()
    lo,_=old(images,*views,valid,61);ln,_=new(images,*views,valid,61)
    torch.testing.assert_close(lo,ln,rtol=0,atol=0);lo.backward();ln.backward()
    for name,p in old.named_parameters():
        q=dict(new.named_parameters())[name]
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    build_optimizer(old).step();build_optimizer(new).step()
    for name,p in old.named_parameters():torch.testing.assert_close(p,dict(new.named_parameters())[name],rtol=0,atol=0)


def test_C_live_forward_formula_and_B_weights():
    torch.manual_seed(95)
    base=make();images,views,valid=inputs();valid.fill_(True)
    for config in [{'summary_t2i_weight':.5},{'view_weights':[1.2,.6,1.2]}]:
        model=copy.deepcopy(base)
        if 'summary_t2i_weight' in config:model.summary_t2i_weight=.5
        else:model.search_hparams['view_weights']=config['view_weights']
        loss,logs=model(images,*views,valid,61)
        if 'summary_t2i_weight' in config:
            expected=10/3*12/11*(logs['F_i2t']+logs['F_t2i']+logs['O_i2t']+.5*logs['O_t2i']+logs['E_i2t']+logs['E_t2i'])
        else:
            assert sum(config['view_weights'])==3
            expected=10/3*sum(w*(logs[p+'_i2t']+logs[p+'_t2i']) for w,p in zip(config['view_weights'],['F','O','E']))
        expected+=(logs['F_sparse']+2*logs['O_sparse']+2*logs['E_sparse'])/3+logs['inc_weight']*logs['inc']
        torch.testing.assert_close(loss,expected,rtol=2e-6,atol=1e-4)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
