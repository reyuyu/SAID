import ast
import inspect
import pytest

from recovery import said_e2_late_lr_fourarm as c


def test_declared_arm_order_and_only_axes():
    assert list(c.ARMS)==['B1-BB085','B2-BB115','B3-MASK085','B4-MASK115']
    assert c.ARMS['B1-BB085']==(.85,1,1,1)
    assert c.ARMS['B2-BB115']==(1.15,1,1,1)
    assert c.ARMS['B3-MASK085']==(1,.85,.85,1)
    assert c.ARMS['B4-MASK115']==(1,1.15,1.15,1)


@pytest.mark.parametrize('arm',c.ARMS)
def test_only_scales_native_schedule_after_3651(arm):
    values=(1e-6,1e-3,1e-3,2e-4)
    assert c.scaled_rates(values,arm,3650)==values
    assert c.scaled_rates(values,arm,3651)==tuple(v*m for v,m in zip(values,c.ARMS[arm]))
    assert c.scaled_rates((0,0,0,0),arm,4868)==(0,0,0,0)


@pytest.mark.parametrize('arm',c.ARMS)
def test_every_arm_has_same_parent_stop_and_full_batch(arm):
    cmd=c.training_command(arm)
    assert cmd[cmd.index('--resume')+1]==str(c.PARENT)
    assert cmd[cmd.index('--max-updates')+1]=='4868'
    assert c.STOP-c.START==1217
    assert 'step003651.pt' in str(c.PARENT)
    assert '--max-restarts=0' in cmd


def test_worker_does_not_change_optimizer_or_loss_definition():
    source=inspect.getsource(c.worker)
    assert 'original_optimizer(module)' in source
    assert 'original_rates(module, completed, horizon)' in source
    assert 'scaled_rates(values, arm, completed)' in source
    assert 'lambda_align' not in source
    assert 'resume_lineage' not in source  # Reviewed resume protocol supplies lineage.


def test_full_stream_covers_entire_epoch_and_reference():
    source=inspect.getsource(c.full_stream)
    assert 'list(range(START+1, STOP+1))' in source
    assert "h['sampling'] == ref['sampling']" in source
    assert "h['stream_sha256'] == ref['stream_sha256']" in source
    assert 'records == 1245904' in source


def test_readonly_precheck_forbids_optimizer():
    from recovery import said_e2_late_lr_precheck as p
    tree=ast.parse(inspect.getsource(p.main))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
        n.func.attr in ('AdamW','SGD','step') for n in ast.walk(tree))
