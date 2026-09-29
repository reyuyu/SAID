"""Tests limited to the new local-view split and its data/logging integration."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import longclip
from model.nested_semantic_mask import NestedSemanticMask
import train.nested_semantic_data as data
from tests.test_nested_semantic_mask import TinyCLIP


def legacy_views(caption):
    """Execute the exact reviewed function, not a reimplementation of the rule."""
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git','show','190b477:train/nested_semantic_data.py'], cwd=root, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='text_views')
    namespace = {'longclip':longclip}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'reviewed_text_views','exec'),namespace)
    return namespace['text_views'](caption)


def assert_old_fields(actual, expected):
    for key,value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(actual[key],value),key
        else:
            assert actual[key]==value,key


@pytest.mark.parametrize('caption', ['First. Second. Last.', 'Only one.', 'word '*247,
                                    'Short. '+'word '*230+'. '+'tail '*100, ' First. \nSecond. Last. '])
def test_fixed_first_matches_reviewed_function(caption):
    assert_old_fields(data.sampled_text_views(caption), legacy_views(caption))


def test_force_k1_matches_loss_and_gradients(monkeypatch):
    monkeypatch.setattr(data,'sample_split_k',lambda *args:1)
    captions=['A cat. A table. The end.', 'A dog. A bench. The last sentence.']
    old=[legacy_views(x) for x in captions]
    new=[data.sampled_text_views(x,'random_k',sample_id=i) for i,x in enumerate(captions)]
    for a,b in zip(new,old):assert_old_fields(a,b)
    torch.manual_seed(11)
    clip=TinyCLIP(); clip.token_embedding=nn.Embedding(len(longclip._tokenizer.encoder),8)
    modules=[NestedSemanticMask(m,arm='A3',checkpoint_encoders=False) for m in (clip,copy.deepcopy(clip))]
    images=torch.randn(2,8); outputs=[]
    for module,views in zip(modules,(old,new)):
        tokens=[torch.stack([v[k] for v in views]) for k in ('tokens_f','tokens_o','tokens_e')]
        loss,logs=module(images,*tokens,torch.ones(2,dtype=torch.bool),200)
        assert logs['inc_weight']==1
        loss.backward(); outputs.append((loss, [p.grad for p in module.parameters()]))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=3e-5,rtol=3e-6)
    for a,b in zip(outputs[0][1],outputs[1][1]):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-5)


def test_visible_full_unchanged_and_contiguous_prefix_remainder():
    caption='A room. A table. A window. A cat. The final visible sentence. '+'extra '*300
    original=legacy_views(caption)
    assert original['valid'] and 'extra' not in original['views'][0]
    parts=original['views'][0].split('. ')
    observed=set()
    for sample_id in range(128):
        v=data.sampled_text_views(caption,'random_k',sample_id=sample_id)
        n,k=v['n'],v['K']; observed.add(k)
        assert n==len(parts) and 1<=k<n
        assert v['views']==[original['views'][0],'. '.join(parts[:k]),'. '.join(parts[k:])]
        assert torch.equal(v['tokens_f'],original['tokens_f'])
        assert v['views'][2].endswith(parts[-1])
        assert max(v['untruncated_lengths'])<=248
        assert torch.equal(torch.stack([v[x] for x in ('tokens_f','tokens_o','tokens_e')]),
                           longclip.tokenize(v['views'],truncate=False))
    assert observed==set(range(1,len(parts)))


def test_two_segments_fallbacks_and_token_boundaries():
    for caption in ('One. Two.', 'Single.', 'word '*247, 'word '*246):
        original=legacy_views(caption)
        for sample_id in range(8):
            v=data.sampled_text_views(caption,'random_k',sample_id=sample_id)
            assert_old_fields(v,original)
            assert v['K']==(1 if original['valid'] else 0)
    with pytest.raises(ValueError,match='empty'):
        data.sampled_text_views(' \n ','random_k')


def test_token_overflow_is_reported_without_redraw(monkeypatch):
    real_encode=longclip._tokenizer.encode
    def encode(text):
        # A tokenizer boundary regression: prefix can exceed the whole F length.
        return [100]*247 if text=='a. b' else real_encode(text)
    monkeypatch.setattr(longclip._tokenizer,'encode',encode)
    calls=[]
    def choose(*args):
        calls.append(args); return 2
    monkeypatch.setattr(data,'sample_split_k',choose)
    assert data.text_views('a. b. c')['views'][0]=='a. b. c'
    with pytest.raises(ValueError,match=r'sample_id=123.*epoch=7.*n=3.*K=2.*249'):
        data.sampled_text_views('a. b. c','random_k',epoch=7,sample_id=123)
    assert len(calls)==1


def test_replay_and_global_rng_isolation():
    python_state=random.getstate(); numpy_state=np.random.get_state(); torch_state=torch.get_rng_state().clone()
    calls=[data.sampled_text_views('One. Two. Three. Four. Five.','random_k',epoch=2,sample_id=i) for i in range(32)]
    replay=[data.sampled_text_views('One. Two. Three. Four. Five.','random_k',epoch=2,sample_id=i) for i in range(32)]
    assert [v['K'] for v in calls]==[v['K'] for v in replay]
    assert python_state==random.getstate()
    current=np.random.get_state()
    assert numpy_state[0]==current[0] and np.array_equal(numpy_state[1],current[1]) and numpy_state[2:]==current[2:]
    assert torch.equal(torch_state,torch.get_rng_state())
    assert any(data.sample_split_k(5,0,2,i)!=data.sample_split_k(5,0,3,i) for i in range(32))


def test_grouped_k_distribution(capsys):
    histogram={n:dict(Counter(data.sample_split_k(n,0,0,i) for i in range(4096))) for n in (2,3,5,8)}
    for n,counts in histogram.items():
        assert set(counts)==set(range(1,n)) and sum(counts.values())==4096
    # Report the actual finite-sample histogram; do not force exact balance.
    with capsys.disabled():print('K distribution:',json.dumps(histogram,sort_keys=True))


def make_dataset(tmp_path):
    index=tmp_path/'index'; index.mkdir()
    Image.new('RGB',(32,32),(20,40,60)).save(tmp_path/'image.png')
    offsets=[0]
    with (index/'records.jsonl').open('wb') as f:
        for i in range(24):
            row=json.dumps(dict(image='image.png',caption=f'Item {i}. Two. Three. Four. Five. Last.')).encode()+b'\n'
            f.write(row); offsets.append(offsets[-1]+len(row))
    np.save(index/'offsets.npy',np.array(offsets,dtype=np.int64))
    (index/'metadata.json').write_text(json.dumps({'training_records':24}))
    return index


def collect_workers(index,root,workers,epoch):
    dataset=data.NestedDataset(index,root,sampling_mode='random_k',sampling_seed=0)
    dataset.set_epoch(epoch)
    kwargs=dict(batch_size=6,num_workers=workers,collate_fn=data.collate,
                generator=torch.Generator().manual_seed(0),persistent_workers=False)
    if workers:kwargs['multiprocessing_context']='spawn'
    loader=DataLoader(dataset,**kwargs)
    result=[]
    for batch in loader:
        result.extend(zip(batch['sample_id'].tolist(),batch['K'].tolist()))
        health=data.sampling_diagnostics(batch)
        legacy=dict(sample_ids=batch['sample_id'].tolist(),views=batch['reference_views'],
                    tokens=[batch['tokens_f'].tolist(),batch['reference_tokens_o'].tolist(),batch['reference_tokens_e'].tolist()])
        assert health['fixed_first_reference_stream_sha256']==hashlib.sha256(json.dumps(legacy,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert health['valid_count']==len(batch['valid'])
    return result


def test_spawn_workers_and_epoch_propagation(tmp_path):
    index=make_dataset(tmp_path)
    zero=collect_workers(index,tmp_path,0,0)
    two=collect_workers(index,tmp_path,2,0)
    epoch1=collect_workers(index,tmp_path,2,1)
    assert zero==two
    assert epoch1==collect_workers(index,tmp_path,0,1)
    assert any(a[1]!=b[1] for a,b in zip(two,epoch1))
