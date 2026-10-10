import copy
import json
import pytest

from recovery import said_e2_early_lr_threearm500 as c


def test_frozen_arm_axes_and_order():
    assert c.ARMS == {'C1-Fusion085':(1,1,1,.85),'C2-Fusion115':(1,1,1,1.15),
                      'C3-MaskBalance':(1,1.15,.85,1)}
    assert list(c.ARMS)==['C1-Fusion085','C2-Fusion115','C3-MaskBalance']


@pytest.mark.parametrize('arm',c.ARMS)
def test_native_schedule_scale_and_zero(arm):
    for s in (0,1,199,200,499,4868):
        native=c.expected_lrs(s,c.config(arm))
        scaled=c.scaled_rates(native,arm)
        assert scaled==tuple(v*m for v,m in zip(native,c.ARMS[arm]))
        assert scaled[0]==native[0]
    assert c.scaled_rates((0,0,0,0),arm)==(0,0,0,0)


@pytest.mark.parametrize('arm',c.ARMS)
def test_independent_fresh_smoke_and_formal_commands(arm):
    for smoke in (True,False):
        cmd=c.training_command(arm,smoke)
        assert '--resume' not in cmd and '--max-restarts=0' in cmd
        assert cmd[cmd.index('--init-state')+1]==str(c.STEP0)
        assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
        assert cmd[cmd.index('--run-type')+1]==('smoke' if smoke else 'formal')
        assert cmd[cmd.index('--output-dir')+1]==str(c.RUN/(arm+'.smoke5' if smoke else arm)/'step500')


def test_full_reference_stream_accepts_only_declared_LR_changes():
    reference=c.rows(c.BASE_RUN/'step500/steps.jsonl')[:2]
    for arm in c.ARMS:
        actual=copy.deepcopy(reference)
        for row in actual:
            row['actual_lrs']=dict(zip(c.GROUPS,c.scaled_rates(list(row['actual_lrs'].values()),arm)))
        assert c.matched_stream(actual,reference,arm)['records']==2048
        broken=copy.deepcopy(actual)
        broken[1]['rank_health'][0]['stream_sha256']='wrong'
        with pytest.raises(AssertionError):c.matched_stream(broken,reference,arm)
        broken=copy.deepcopy(actual);broken[1]['actual_lrs']['fusion_adapter']+=1e-8
        with pytest.raises(AssertionError):c.matched_stream(broken,reference,arm)
        broken=copy.deepcopy(actual);broken[1]['step']=1
        with pytest.raises(AssertionError):c.matched_stream(broken,reference,arm)
        broken=copy.deepcopy(actual);broken[1]['rank_health'][0]['sampling']['K'][0]=999
        with pytest.raises(AssertionError):c.matched_stream(broken,reference,arm)


def test_configuration_unchanged_for_all_arms():
    original=c.read(c.BASE_EXP/'config.json')
    for arm in c.ARMS:
        assert c.config(arm)==original
        c.frozen(c.config(arm),arm)
        changed=c.config(arm);changed['lambda_sparse']=1.
        with pytest.raises(AssertionError):c.frozen(changed,arm)


def test_sampling_receipt_uses_JSON_representation():
    from recovery.nested_d3_local_search import observe_selection
    from train.nested_semantic_data import collate
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(c.runner.local.INDEX,c.runner.local.IMAGES,'nested_detail_d3',0)
    # Real text/token metadata; image pixels do not enter observe_selection.
    batch=collate([dataset[0]])
    raw=observe_selection(batch)
    stored=json.loads(json.dumps(raw))
    assert c.digest(stored)==c.digest(json.loads(json.dumps(raw)))
    assert stored['sample_ids']==raw['sample_ids']
    assert stored['full_view_sha256']==raw['full_view_sha256']
    assert stored['lowest_selected_sentence_indices_sha256']==raw['lowest_selected_sentence_indices_sha256']
