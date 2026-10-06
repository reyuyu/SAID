import json
from pathlib import Path
from recovery.local500_policy import cycle_state,wait_expired,POLICY


def test_slow_steps_only_warn_even_five_over30():
    for seconds in (3,7,15,31,31,31,31,31,60):
        result=cycle_state(seconds,100)
        assert not result['stop']
        assert result['warning']==(seconds>3)
    assert cycle_state(60.01)['stop']


def test_unreturned_and_compute_watchdogs_and_failure_states():
    for state in ('DATA_WAIT','ACTIVE_STEP'):
        beat=dict(state=state,started_monotonic=100)
        assert not wait_expired(beat,160)
        assert wait_expired(beat,160.01)
    for state in ('DATA_WAIT_FAILED','EXHAUSTED','TRAINING_COMPLETE'):
        assert not wait_expired(dict(state=state),200)


def test_frozen_local_configuration_changes_only_resource_control():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'recovery/configs/summary02.json').read_text())
    new=json.loads((root/'recovery/configs/summary02_local500.json').read_text())
    assert new.pop('resource_policy')==POLICY
    assert new==old
