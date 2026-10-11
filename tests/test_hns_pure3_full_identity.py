"""Negative checks for the dedicated Full evaluation identity binding."""
import copy
import pytest
from recovery.hns_v1_pure3 import TRAIN
from recovery.hns_v1_pure3_evidence import aggregate, DATASETS
from tools import eval_hns_pure3_flickrfull as full


def test_authorized_full_job_and_only_own_export():
    full.validate_job(full.FIXED, TRAIN/'student_step3651.pt')
    with pytest.raises(ValueError):
        full.validate_job(full.FIXED, TRAIN/'step003651.pt')
    with pytest.raises(ValueError):
        full.validate_job(full.FIXED, '/tmp/other-student.pt')


@pytest.mark.parametrize('key,value', [('manifest_sha256','invalid'),('n_images',1000),
    ('n_captions',5000),('image_batch',32),('text_batch',256),('query_chunk',512),
    ('gallery_chunk',8192),('normalization','cpu'),('legacy_expected',{}),
    ('protocol','test1k')])
def test_protocol_drift_rejected(key,value):
    job=copy.deepcopy(full.FIXED); job[key]=value
    with pytest.raises(ValueError): full.validate_job(job,TRAIN/'student_step3651.pt')


def test_aggregates_use_only_historical_five_and_original_math():
    metrics={d:{r:{'R@1':.5,'R@5':.8,'R@10':.9} for r in ('I2T','T2I')} for d in DATASETS}
    assert set(aggregate(metrics).values())=={50.}
    metrics['Flickr30k-Full']={r:{'R@1':1.} for r in ('I2T','T2I')}
    assert set(aggregate(metrics).values())=={50.}
