import copy
import pytest
from recovery import said_e2_hierarchy090_full4868 as r


def test_frozen_continuation_does_not_accept_parameter_changes():
    cfg=r.protocol.common.read(r.BASE_EXP/'config.json');r.frozen(cfg)
    assert r.TARGETS==(4868,) and r.predecessor(4868)==(r.PARENT,500)
    for key,value in [('lambda_hierarchy',1.),('batch_size',128),('epochs',5),('fusion_lr',1e-4)]:
        bad=dict(cfg,**{key:value})
        with pytest.raises(AssertionError):r.frozen(bad)
    with pytest.raises(AssertionError):r.predecessor(1217)


@pytest.mark.parametrize('step',[501,1217,1218,2434,2435,3651,3652,4868])
def test_native_LR_counters_epoch_and_stream_drift_rejected(step):
    ref=next(x for x in r.reference_rows() if x['step']==step)
    a=copy.deepcopy(ref)
    a['macro_lambda_hierarchy']=.9
    a['macro_weighted_hierarchy']=.9*a['macro_raw_hierarchy']
    a['loss']=sum(a['macro_weighted_'+k] for k in ('align','sparse','hierarchy'))
    for h in a['rank_health']:h['updates']=step-500
    assert r.check_row(a,ref)==(720 if step%1217==0 else 1024)
    for mutation in ('counter','text','sampling','batch','lr','epoch','nonfinite'):
        bad=copy.deepcopy(a)
        if mutation=='counter':bad['rank_health'][0]['updates']-=1
        if mutation=='text':bad['rank_health'][0]['stream_sha256']='wrong'
        if mutation=='sampling':bad['rank_health'][0]['sampling']={}
        if mutation=='batch':bad['rank_health'][0]['batch']-=1
        if mutation=='lr':bad['actual_lrs']['backbone']*=1.1
        if mutation=='epoch':bad['epoch']+=1
        if mutation=='nonfinite':bad['nonfinite']=1
        with pytest.raises(AssertionError):r.check_row(bad,ref)


def test_no_missing_or_repeated_updates():
    ref=r.reference_rows()
    assert len(ref)==4368
    with pytest.raises(AssertionError):r.full_stream_proof(ref[:-1],ref)
    with pytest.raises(AssertionError):r.full_stream_proof([ref[0],*ref],ref)
