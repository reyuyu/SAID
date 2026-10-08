"""CPU checks for read-only provenance, sequential replay and discrete curves."""
import copy
import json

import pytest

from recovery import epoch_curve_eval as replay


def test_epoch_interleaved_queue():
    assert replay.queue()==[(m,s) for s in (1217,2434,3651,4868) for m in ('D3_Balanced','HNS_v1')]


def test_strict_export_identity_gate():
    good=dict(passed=True,strict_load=True,optimizer_steps=[2434],image_max_abs=0.,text_max_abs=0.,
        checkpoint_sha256='checkpoint',bare_sha256='bare')
    assert replay.valid_export(good,'checkpoint','bare',2434)
    for key,value in [('passed',False),('strict_load',False),('optimizer_steps',[1217]),
                      ('image_max_abs',1e-9),('text_max_abs',1e-9),('bare_sha256','other')]:
        assert not replay.valid_export(dict(good,**{key:value}),'checkpoint','bare',2434)


def test_crossing_intervals_are_discrete_and_ties_are_not_invented():
    assert replay.crossings([dict(epoch=1,delta=.1),dict(epoch=2,delta=-.2)])==[
        dict(between_epochs=[1,2],delta_before=.1,delta_after=-.2)]
    assert replay.crossings([dict(epoch=1,delta=0),dict(epoch=2,delta=-.2)])==[]


def test_gains_and_directional_deltas_preserve_units():
    results={m:{} for m in replay.MODELS}
    for i,s in enumerate(replay.STEPS):
        for m in replay.MODELS:
            x=70+i if m=='D3_Balanced' else 70.2+.8*i
            results[m][str(s)]=dict(scores_percent={k:x for k in replay.KEYS},
                metrics={'COCO':{d:{'R@1':x/100} for d in ('I2T','T2I')}})
    aggregate,directions,gains=replay.curves(results)
    assert aggregate['Score5'][0]['delta']==pytest.approx(.2)
    assert aggregate['Score5'][-1]['delta']==pytest.approx(-.4)
    assert directions['COCO/I2T'][-1]['delta']==pytest.approx(-.4)
    assert gains[0]['gains_pp']['Score5']['Balanced']==1
    assert gains[0]['gains_pp']['Score5']['HNS']==pytest.approx(.8)
    assert gains[0]['gains_pp']['Score5']['HNS_minus_Balanced_growth']==pytest.approx(-.2)


def test_runner_stops_on_failed_checkpoint_without_next_eval_or_analysis(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,'EXP',tmp_path/'reports');monkeypatch.setattr(replay,'RUN',tmp_path/'run')
    replay.EXP.mkdir();replay.RUN.mkdir()
    replay.save('CHECKPOINT_INVENTORY.json',dict(models={'D3_Balanced':{'1217':{}}}))
    replay.save('CPU_TESTS.json',dict(passed=True))
    monkeypatch.setattr(replay,'frozen_sources',lambda x:None)
    monkeypatch.setattr(replay.signal,'signal',lambda *x:None)
    calls=[]
    def fail(*args):calls.append(args[:2]);raise RuntimeError('FINAL_REPRODUCTION_MISMATCH')
    monkeypatch.setattr(replay,'evaluate',fail)
    monkeypatch.setattr(replay,'report',lambda *args:pytest.fail('Analysis after mismatch'))
    monkeypatch.setattr(replay,'publish',lambda:pytest.fail('Publishing unfinished trajectory'))
    with pytest.raises(RuntimeError,match='MISMATCH'):replay.run()
    assert calls==[('D3_Balanced',1217)]
    assert json.loads((replay.EXP/'STATE.json').read_text())['status']=='STOPPED_WITH_EVIDENCE'
