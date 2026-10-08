import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from recovery import hns_macro_fourarm as r


def test_exact_four_configs_only_macro_fields_differ():
    base=r.base_config()
    assert list(r.ARMS)==['E1-A12','E2-S08','E3-H075','E4-H125']
    for arm,scales in r.ARMS.items():
        cfg=r.config(arm);r.frozen(cfg,arm)
        assert {k:v for k,v in cfg.items() if k not in r.MACRO_KEYS}==base
        assert tuple(cfg[k] for k in r.MACRO_KEYS)==scales
        bad=copy.deepcopy(cfg);bad['sparsity_scale']=.8
        with pytest.raises(AssertionError):r.frozen(bad,arm)


def test_every_smoke_and_formal_fresh0_local_only_stop500():
    for arm in r.ARMS:
        for smoke in (False,True):
            cmd=r.training_command(arm,smoke)
            assert '--resume' not in cmd
            assert cmd[cmd.index('--init-state')+1]==str(r.STEP0)
            assert cmd[cmd.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
            assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
            assert cmd[cmd.index('--run-type')+1]==('smoke' if smoke else 'formal')
        assert r.training_command(arm,False)!=r.training_command(arm,True)


def test_champion_baseline_guard_and_unconfirmed_gain():
    q=dict(Score5=71.2,J_long3=75.15,J_long=84.16,Short4=65.27,Urban_T2I=89.7)
    assert r.champion({'HNS-v1':q,'E1-A12':dict(q,Score5=71.1)})['decision']=='KEEP_HNS_V1'
    choice=r.champion({'HNS-v1':q,'E1-A12':dict(q,Score5=71.22)})
    assert choice['BEST_SCORE5_ARM']=='E1-A12' and choice['gain_is_small_unconfirmed']
    bad=dict(q,Score5=71.4,J_long3=74)
    assert r.champion({'HNS-v1':q,'E1-A12':bad})['BEST_BALANCED_ARM']=='HNS-v1'


def test_smoke_failure_blocks_formal_eval_and_next_arm():
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp)/'run';run.mkdir();exp=Path(tmp)/'exp';exp.mkdir()
        r.dump(exp/'QUEUE_STATE.json',dict(status='PREPARED'))
        for name in ('CPU_TESTS.json','DDP_EQUIVALENCE.json','REAL_BF16_EQUIVALENCE.json'):r.dump(exp/name,dict(passed=True))
        r.dump(exp/'BASELINE_PROVENANCE.json',dict(production_sources={}))
        def activate(arm,smoke=False):
            r.local.RUN=run/(arm+'.smoke5' if smoke else arm)
            r.local.PHASE=Path(tmp)/(arm+('-smoke' if smoke else '-formal'))
        with patch.object(r,'RUN',run),patch.object(r,'EXP',exp),patch.object(r,'activate',side_effect=activate), \
                patch.object(r,'sha',return_value=r.STEP0_SHA),patch.object(r,'git',return_value='HEAD'), \
                patch.object(r,'evaluate') as evaluate,patch.object(r,'publish') as publish, \
                patch.object(r,'evaluator_proof'),patch('train.train_nested_semantic_mask.code_manifest',return_value={}), \
                patch('tools.eval_five_parallel.require_gpu_idle'),patch.object(r.signal,'signal'), \
                patch.object(r.local,'Supervisor') as supervisor,patch.object(r.local,'RUN'),patch.object(r.local,'PHASE'):
            supervisor.return_value.execute.side_effect=RuntimeError('smoke failed')
            with pytest.raises(RuntimeError,match='smoke failed'):r.run()
            assert supervisor.return_value.execute.call_count==1
            evaluate.assert_not_called();publish.assert_not_called()
            assert r.read(exp/'QUEUE_STATE.json')['status']=='STOPPED_WITH_EVIDENCE'


def test_default_interface_keeps_legacy_config_schema():
    from model.balanced_hparam_search import hparams
    base=r.base_config();assert not any(k in hparams(base) for k in r.MACRO_KEYS)
    for arm in r.ARMS:assert all(k in hparams(r.config(arm)) for k in r.MACRO_KEYS)
