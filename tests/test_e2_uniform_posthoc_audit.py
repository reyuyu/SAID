import copy
import pytest
import torch
from recovery.e2_uniform_posthoc_audit import exact_mcnemar, paired_summary, stream_row_invariant, query_rows, write_new, write_urban


def test_exact_mcnemar():
    assert exact_mcnemar(0,0)==1
    assert exact_mcnemar(3,0)==.25
    assert exact_mcnemar(10,10)==1
    assert exact_mcnemar(5,1)==exact_mcnemar(1,5)


def test_pairing_counts_gains_and_regressions():
    s=paired_summary([1,1,0,0],[1,0,1,0])
    assert s['counts']==dict(both_correct=1,E2_only_correct=1,S12_only_correct=1,both_wrong=1)
    assert s['net_correct']==0 and s['mcnemar_exact_two_sided_p']==1


def fixture_rows():
    a=dict(step=1218,s=1217,epoch=1,actual_lrs={'x':1},rank_health=[
        dict(rank=r,updates=1,batch=256,stream_sha256='a',sampling=dict(sample_ids=list(range(256)))) for r in range(4)])
    return a,copy.deepcopy(a)


@pytest.mark.parametrize('failure',['step','rank','sample','stream','LR','optimizer','batch'])
def test_stream_rejects_drift(failure):
    a,b=fixture_rows()
    if failure=='step': b['step']=1219
    elif failure=='rank':b['rank_health'].pop()
    elif failure=='sample':b['rank_health'][0]['sampling']['sample_ids'][0]=999
    elif failure=='stream':b['rank_health'][0]['stream_sha256']='b'
    elif failure=='LR':b['actual_lrs']['x']=2
    elif failure=='optimizer':b['rank_health'][0]['updates']=2
    elif failure=='batch':b['rank_health'][0]['batch']=180
    with pytest.raises(AssertionError):stream_row_invariant(a,b,1218,1217,{'x':1})


def test_stream_excludes_model_values():
    a,b=fixture_rows();a['loss']=1;b['loss']=999
    stream_row_invariant(a,b,1218,1217,{'x':1})


def test_topk_query_rank_and_margin():
    sim=torch.eye(10)*2
    sim[0,1]=3
    rows=query_rows(sim,[str(i) for i in range(10)])
    assert not rows[0]['correct'] and rows[0]['top1_ID']=='1'
    assert rows[0]['ground_truth_rank']==2 and rows[0]['ground_truth_minus_best_wrong_similarity']==-1
    assert rows[1]['correct'] and rows[1]['ground_truth_rank']==1


def test_outputs_cannot_overwrite_existing_artifacts(tmp_path):
    target=tmp_path/'original.json'
    write_new(target, {'immutable': True})
    with pytest.raises(FileExistsError): write_new(target, {'immutable': False})
    assert 'true' in target.read_text()


def test_tie_bounds_preserve_native_topk():
    sim=torch.ones(10,10)
    rows=query_rows(sim,[str(i) for i in range(10)])
    native=sim.topk(1,dim=1).indices[:,0].tolist()
    for i,row in enumerate(rows):
        assert row['top1_ID']==str(native[i])
        assert row['correct']==(native[i]==i)
        assert row['rank_min_with_ties']==1 and row['rank_max_with_ties']==10


def test_small_query_artifacts_preserve_every_row(tmp_path):
    import json
    value=dict(directions={d:dict(queries=[dict(query_ID=str(i)) for i in range(1000)]) for d in ('I2T','T2I')})
    write_urban(tmp_path,value)
    summary=json.loads((tmp_path/'URBAN_4868_PAIRED_AUDIT.json').read_text())
    for d,item in summary['directions'].items():
        assert item['query_count']==1000
        assert len(json.loads((tmp_path/item['queries_file']).read_text())['queries'])==1000
    assert len(value['directions']['I2T']['queries'])==1000
