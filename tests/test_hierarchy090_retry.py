import copy
import json
import pytest
from recovery.said_e2_hierarchy090_full4868 import normalize_json
from recovery import hierarchy090_retry_preflight as check


def native():return {'histogram':{0:7,1:12,3:237},'nested':[{'tokens':{248:256},'K':3}],'digest':'abc'}


def test_integer_keys_and_JSON_string_keys_match_exactly():
    a=native();b=json.loads(json.dumps(a))
    assert a!=b and normalize_json(a)==normalize_json(b)


@pytest.mark.parametrize('change',['count','K','digest','missing','extra'])
def test_real_changes_are_not_hidden(change):
    a=native();b=normalize_json(a)
    if change=='count':b['histogram']['3']+=1
    if change=='K':b['nested'][0]['K']=2
    if change=='digest':b['digest']='changed'
    if change=='missing':del b['histogram']['0']
    if change=='extra':b['unexpected_field']=0
    with pytest.raises(AssertionError):assert normalize_json(a)==normalize_json(b)


@pytest.mark.parametrize('bad',[{1:2,'1':2},{'nested':[{1:2,'1':3}]},{None:2,'null':2},{True:1,'true':1}])
def test_JSON_key_collision_is_rejected_even_when_values_equal(bad):
    with pytest.raises(ValueError,match='JSON key collision'):normalize_json(bad)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_value_is_rejected(value):
    with pytest.raises(ValueError):normalize_json({'nested':[{'value':value}]})


@pytest.mark.parametrize('rank',range(4))
def test_real_historical_step501_all_fields_and_mutations(rank):
    proof=check.history501()[rank]
    assert proof['passed'] and proof['collision_check_passed']
    a=proof['actual_sampling'];b=proof['reference_sampling']
    assert normalize_json(a)==normalize_json(b)
    for key in a:
        missing=copy.deepcopy(b);missing.pop(key)
        with pytest.raises(AssertionError):assert normalize_json(a)==normalize_json(missing)
    changed=copy.deepcopy(b);changed['valid_count']+=1
    assert normalize_json(a)!=normalize_json(changed)


def test_production_code_and_sampling_remain_frozen():
    from train.train_nested_semantic_mask import code_manifest
    original=check.run.protocol.common.read(check.OLD_EXP/'RESUME_PROVENANCE.json')['sources']
    assert code_manifest()==original
