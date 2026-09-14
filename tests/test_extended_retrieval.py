import csv, json
import numpy as np
import pytest
from tools.eval_extended_retrieval import evaluate_features, cache_identity, recall_at_k
from tools.prepare_retrieval_benchmarks import parse_docci, parse_long_dci, parse_flickr, parse_dci, integrity

def test_one_to_one_and_five_caption_and_order_invariance():
    im=np.eye(2,dtype=np.float32); tx=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1]],np.float32)
    ids=['a','b']; pos=[['a']]*5+[['b']]
    assert evaluate_features(im,tx,ids,pos)['R@1'][0] == 1.0
    shuffled=[5,3,0,4,2,1]; assert recall_at_k(im,tx[shuffled],ids,[pos[i] for i in shuffled],1)[0] == 1.0
    assert recall_at_k(im,tx,ids,pos,5,query_chunk=2) == recall_at_k(im,tx,ids,pos,5)

def test_ties_are_stable_and_cache_mismatch():
    im=np.array([[1,0]],np.float32); tx=np.array([[1,0],[1,0]],np.float32)
    assert recall_at_k(im,tx,['x'],[['x'],['y']],1)[0] == 1.0
    assert cache_identity('a','m','tok',248,'p','fp32') != cache_identity('b','m','tok',248,'p','fp32')

def test_manifest_parsers(tmp_path):
    d=tmp_path/'d.jsonl'; d.write_text(json.dumps({'split':'test','example_id':'t1','description':'hello','image_file':'t1.jpg'})+'\n')
    m=tmp_path/'d.manifest'; assert parse_docci(d,m)[1]==1
    t=tmp_path/'l.tsv'; t.write_text('filepath\ttitle\nq.jpg\tlong text\n'); assert parse_long_dci(t,tmp_path/'l.manifest')[1]==1
    f=tmp_path/'f.csv'; f.write_text('image_id,captions,split\na,"[""c1"",""c2"",""c3"",""c4"",""c5""]",test\n'); assert parse_flickr(f,tmp_path/'f.manifest',True)[1]==5

def test_invalid_empty_and_duplicate(tmp_path):
    d=tmp_path/'d.jsonl'; d.write_text(json.dumps({'split':'test','example_id':'t1','description':'','image_file':'t1.jpg'})+'\n')
    with pytest.raises(ValueError): parse_docci(d,tmp_path/'m')
    d.write_text('\n'.join(json.dumps({'split':'test','example_id':'t1','description':'ok','image_file':'t1.jpg'}) for _ in range(2)))
    with pytest.raises(ValueError): parse_docci(d,tmp_path/'m2')

def test_integrity_reports_missing_image(tmp_path):
    m=tmp_path/'m'; m.write_text(json.dumps({'protocol_id':'x','image_id':'i','image_path':'missing.jpg','caption_id':'c','caption':'ok','positive_image_id':'i'})+'\n')
    out=integrity(m,tmp_path/'images'); assert out['missing_images']==1 and out['decode_failures']==0
