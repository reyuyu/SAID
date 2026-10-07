"""Fail closed on method drift while permitting pinned default-beta migration."""
import copy
import json

import pytest

from recovery import hns_full as full
from train.train_nested_semantic_mask import validate_resume_payload


def payloads():
    cfg=json.loads((full.BASE_EXP/'config.json').read_text())
    cfg.update(horizon=4868,init_sha256='common0',data={'records':1245901},run_type='formal',
        batches_per_epoch=1217,max_updates=500,
        code_sha256={p:'old:'+p for p in full.MIGRATED}|{'train/nested_semantic_data.py':'frozen'})
    previous=dict(config=cfg,completed_steps=500,global_step=500,scheduler_horizon=4868,
        trajectory_root=str(full.BASE_RUN),data_cursor=dict(next_epoch=0,next_batch=500),
        next_epoch=0,next_batch=500,rng_per_rank=[None]*4)
    current=copy.deepcopy(cfg)
    current.update(resume=str(full.PARENT),max_updates=4868)
    current['code_sha256'].update({p:'new:'+p for p in full.MIGRATED})
    proof=dict(predecessor_sources=copy.deepcopy(cfg['code_sha256']),current_sources=copy.deepcopy(current['code_sha256']),
        audited_equivalent_changes={p:[cfg['code_sha256'][p],current['code_sha256'][p]] for p in full.MIGRATED})
    return previous,current,proof


def test_pinned_equivalent_migration_keeps_actual_predecessor_provenance():
    previous,current,proof=payloads();before=copy.deepcopy(previous)
    assert full.validate_resume(previous,current,proof,validate_resume_payload)==500
    assert previous==before
    assert current['code_sha256']!=previous['config']['code_sha256']


@pytest.mark.parametrize('key,value',[
    ('view_weights',[1.,1.,1.]),('sampling_mode','nested_detail_kr234'),('hns_beta',[1.,2.]),
    ('inclusion_max',1.),('workers',4),('view_sparsity_weights',[1,2,3]),
    ('horizon',500),('resume','other-step500.pt'),('max_updates',500),('batch_size',128)])
def test_method_or_checkpoint_drift_rejected(key,value):
    previous,current,proof=payloads();current[key]=value
    with pytest.raises(AssertionError):full.validate_resume(previous,current,proof,validate_resume_payload)


def test_unknown_source_change_or_changed_migration_pin_rejected():
    previous,current,proof=payloads();current['code_sha256']['train/nested_semantic_data.py']='modified'
    with pytest.raises(AssertionError):full.validate_resume(previous,current,proof,validate_resume_payload)
    previous,current,proof=payloads();proof['audited_equivalent_changes'][next(iter(full.MIGRATED))][1]='wrong'
    with pytest.raises(AssertionError):full.validate_resume(previous,current,proof,validate_resume_payload)


@pytest.mark.parametrize('field,value',[
    ('next_batch',0),('next_epoch',1),('completed_steps',0),('global_step',33),
    ('data_cursor',{'next_epoch':0,'next_batch':0}),('trajectory_root','other-run')])
def test_cursor_or_trajectory_drift_rejected(field,value):
    previous,current,proof=payloads();previous[field]=value
    with pytest.raises(AssertionError):full.validate_resume(previous,current,proof,validate_resume_payload)


def test_frozen500_config_exact_and_final_no_additional_arm():
    cfg=json.loads((full.BASE_EXP/'config.json').read_text());full.frozen_config(cfg)
    assert cfg['hns_enabled'] and cfg.get('hns_beta',[2.,2.])==[2.,2.]
    assert cfg['checkpoint_interval']==1217 and cfg['epochs']==4
    assert cfg['workers']==8 and cfg['batch_size']*cfg['world_size']==1024
    assert full.TRAIN.is_relative_to(full.ROOT/'runtime')


def test_export_and_evaluation_receipt_gates_are_required(monkeypatch):
    monkeypatch.setattr(full,'read',lambda path:dict(passed=False))
    with pytest.raises(AssertionError):full.resume_gate()
