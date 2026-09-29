"""Continuation contract: preserve horizon, reject semantic changes, no batch drift."""
import copy

import pytest
from torch.utils.data import DataLoader, DistributedSampler

from train.nested_semantic_data import NestedDataset, collate
from train.train_nested_semantic_mask import consumed_batch, learning_rates, validate_resume_payload
from tests.test_nested_randomk import make_dataset


def payloads():
    old = dict(arm='A3', horizon=3651, init_sha256='init', data={'training_records':1245901},
               batch_size=256, world_size=4, accumulation=1, epochs=3, seed=0, workers=8,
               checkpoint_encoders=True, score_chunk=64, run_type='formal',
               sampling_mode='random_k', sampling_seed=0, max_updates=500,
               batches_per_epoch=1217,
               code_sha256={'model/nested_semantic_mask.py':'objective',
                            'train/nested_semantic_data.py':'data',
                            'train/train_nested_semantic_mask.py':'old-trainer'})
    previous=dict(config=old, completed_steps=500, scheduler_horizon=3651,
                  next_epoch=0, next_batch=500, rng_per_rank=[None]*4)
    current=copy.deepcopy(old)
    current['max_updates']=3651
    current['code_sha256']['train/train_nested_semantic_mask.py']='new-trainer'
    return previous,current


def test_stop_extension_preserves_horizon_and_requires_trainer_pin():
    previous,current=payloads()
    with pytest.raises(AssertionError,match='code mismatch'):
        validate_resume_payload(previous,current)
    assert validate_resume_payload(previous,current,'old-trainer')==500
    with pytest.raises(AssertionError):
        validate_resume_payload(previous,current,'wrong-trainer')
    assert learning_rates(500,3651)!=learning_rates(0,3651)
    assert learning_rates(3650,3651)[0]>0 and learning_rates(3650,3651)[1]>0


@pytest.mark.parametrize('field,value', [('horizon',500),('sampling_mode','fixed_first'),
    ('sampling_seed',1),('batch_size',128),('seed',1),('score_chunk',32),('run_type','smoke')])
def test_resume_rejects_changed_method_or_schedule(field,value):
    previous,current=payloads();current[field]=value
    with pytest.raises(AssertionError):validate_resume_payload(previous,current,'old-trainer')


def test_trainer_pin_does_not_allow_objective_or_data_change():
    for name in ('model/nested_semantic_mask.py','train/nested_semantic_data.py'):
        previous,current=payloads();current['code_sha256'][name]='changed'
        with pytest.raises(AssertionError,match='code mismatch'):
            validate_resume_payload(previous,current,'old-trainer')


def test_resume_rejects_bad_cursor_or_finished_stop():
    previous,current=payloads();previous['next_batch']=501
    with pytest.raises(AssertionError):validate_resume_payload(previous,current,'old-trainer')
    previous,current=payloads();current['max_updates']=500
    with pytest.raises(AssertionError,match='continuation stop'):
        validate_resume_payload(previous,current,'old-trainer')


def test_resume_stream_across_epochs_and_tail_batches(tmp_path):
    index=make_dataset(tmp_path)
    for rank in range(4):
        def collect(completed):
            dataset=NestedDataset(index,tmp_path,'random_k',0)
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
            loader=DataLoader(dataset,batch_size=4,sampler=sampler,drop_last=False,collate_fn=collate,num_workers=0)
            result=[]
            for epoch in range(3):
                sampler.set_epoch(epoch);dataset.set_epoch(epoch)
                for batch_index,batch in enumerate(loader):
                    if consumed_batch(epoch,batch_index,len(loader),completed):continue
                    result.append((epoch,batch_index,batch['sample_id'].tolist(),batch['K'].tolist(),
                                   batch['tokens_f'].tolist(),batch['tokens_o'].tolist(),batch['tokens_e'].tolist()))
                    completed+=1
            return result
        full=collect(0)
        assert len(full)==6 and [len(x[2]) for x in full]==[4,2]*3
        for start in (1,2,3,5):assert collect(start)==full[start:]
