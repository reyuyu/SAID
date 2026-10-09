import copy
import subprocess
import sys
import pytest
import torch
from model.balanced_hparam_search import MACRO_DEFAULTS
from recovery.hns_balanced_macro_equivalence import make,COMPLETED,compare_gradients
from recovery import hns_static_strength_twoarm as r


@pytest.mark.parametrize('scales',[(10.,1.2,1.),(10.,1.,4.)])
@pytest.mark.parametrize('completed',COMPLETED)
def test_S12_H4_only_scale_one_component_actual_forward_gradient(scales,completed):
    torch.manual_seed(173);base=make('HNS');new=copy.deepcopy(base)
    new.macro_hparams=dict(zip(r.runner.MACRO_KEYS,scales));base.capture_hns_graph=new.capture_hns_graph=True
    images=torch.randn(6,8);views=[torch.randint(0,31,(6,6)) for _ in range(3)];valid=torch.ones(6,dtype=torch.bool)
    old,oldlog=base(images,*views,valid,completed);loss,logs=new(images,*views,valid,completed)
    for name in ('raw_align','raw_sparse','raw_hierarchy'):
        torch.testing.assert_close(base.hns_graph[name],new.hns_graph[name],atol=0,rtol=0)
    for name,scale in zip(('align','sparse','hierarchy'),(1,scales[1],scales[2])):
        torch.testing.assert_close(new.hns_graph['weighted_'+name],base.hns_graph['weighted_'+name]*scale,atol=0,rtol=0)
    assert logs['lambda_h']==min(1,completed/200)
    assert logs['inc_weight']==logs['inclusion_loss']==0 and logs['HNS_enabled']
    assert logs['V_DF_hard']==oldlog['V_DF_hard'] and logs['V_3D_hard']==oldlog['V_3D_hard']
    rawA=torch.autograd.grad(base.hns_graph['raw_hierarchy'],list(base.parameters()),retain_graph=True,allow_unused=True)
    rawB=torch.autograd.grad(new.hns_graph['raw_hierarchy'],list(new.parameters()),retain_graph=True,allow_unused=True)
    for a,b in zip(rawA,rawB):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_exact_two_configs_fresh_common0_local_commands():
    code='''from recovery import hns_static_strength_twoarm as r
r.configure()
assert list(r.runner.ARMS)==['E1-HNS-S12','E2-HNS-H4']
base=r.runner.read(r.REF['exp']/'config.json')
for arm,scales in r.ARMS.items():
 cfg=r.config(arm);r.shared.frozen(cfg,arm)
 assert {k:v for k,v in cfg.items() if k not in r.runner.MACRO_KEYS}==base
 assert sum(a!=b for a,b in zip(scales,(10,1,1)))==1
 for smoke in (True,False):
  cmd=r.runner.training_command(arm,smoke)
  assert '--resume' not in cmd
  assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
  assert cmd[cmd.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
  assert cmd[cmd.index('-m')+1]==r.ENTRY
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_mask_guard_only_complete_endpoint_streak():
    def logs(v):return {'HNS_'+k+'_keep':v for k in ('F','Dall','D3')}
    previous=None;count=0
    for _ in range(5):previous,count=r.endpoint_streak(logs(0),previous,count)
    assert count==5 and previous=='empty'
    assert r.endpoint_streak(logs(.05),previous,count)==(None,0)
    assert r.endpoint_streak(logs(1),previous,count)==('full',1)
    assert r.endpoint_streak({},previous,count)==(None,0)
    with pytest.raises(AssertionError):r.endpoint_streak(logs(float('nan')),None,0)


def test_selection_preserves_baseline_and_labels_small_gain():
    q={n:dict(Score5=71.1) for n in ('HNS-v1',*r.ARMS)};q['HNS-v1']['Score5']=71.202056
    assert r.selection(q)['RECOMMENDED_NEXT_VALIDATION']=='KEEP_HNS_V1'
    q['E1-HNS-S12']['Score5']=71.22
    assert r.selection(q)['weak_single_seed_signal']


@pytest.mark.parametrize('fail_smoke',[False,True])
def test_explicit_two_arm_queue_success_stop_and_smoke_failure(fail_smoke):
    code='''import tempfile
from pathlib import Path
from unittest.mock import patch
from recovery import hns_static_strength_twoarm as r
r.configure();q=r.runner;events=[];fail_smoke=FAIL_SMOKE
with tempfile.TemporaryDirectory() as tmp:
 run=Path(tmp)/'run';run.mkdir();exp=Path(tmp)/'exp';exp.mkdir();r.RUN=run;r.EXP=exp;q.RUN=run;q.EXP=exp
 q.dump(exp/'QUEUE_STATE.json',dict(status='PREPARED'))
 for name in r.GATES:q.dump(exp/name,dict(passed=True))
 q.dump(exp/'BASELINE_PROVENANCE.json',dict(production_sources={}))
 def activate(arm,smoke=False):
  q.search.ARM=arm;q.local.RUN=run/(arm+'.smoke5' if smoke else arm)
  q.local.PHASE=Path(tmp)/(arm+('-smoke' if smoke else '-formal'))
 def execute(name,cmd,training=False):
  arm=q.search.ARM;events.append((arm,name))
  if fail_smoke and name=='smoke5':raise RuntimeError('smoke failed')
  if name in ('smoke5','train500'):
   assert '--resume' not in cmd
   step=5 if name=='smoke5' else 500
   q.dump(q.local.RUN/'step500/acceptance.json',dict(passed=True,ranks=[dict(completed_updates=step,updates_this_run=step,max_parameter_difference_from_rank0=0) for _ in range(4)]))
   if name=='train500':q.dump(q.local.RUN/'first-five-gate.json',dict(passed=True))
 def evaluate(supervisor,arm):events.append((arm,'eval'));return {}
 def report(arm,result,supervisor):events.append((arm,'report'))
 def publish():events.append((q.search.ARM,'publish'))
 with patch.object(r,'activate',side_effect=activate),patch.object(r,'sha',return_value=q.STEP0_SHA),patch.object(q,'git',return_value='HEAD'),patch.object(q,'evaluate',side_effect=evaluate),patch.object(r,'report_arm',side_effect=report),patch.object(r,'publish',side_effect=publish),patch.object(r,'combined') as combined,patch.object(r,'validate_controls'),patch.object(q,'evaluator_proof'),patch.object(q,'identity',return_value={'passed':True}),patch.object(r,'rows',return_value=[]),patch.object(r.shared,'matched_stream',return_value={'passed':True,'records':512000}),patch.object(q.search,'sampling_audit'),patch('train.train_nested_semantic_mask.code_manifest',return_value={}),patch('tools.eval_five_parallel.require_gpu_idle'),patch.object(r.signal,'signal'),patch.object(q.local,'Supervisor') as supervisor:
  supervisor.return_value.execute.side_effect=execute
  if fail_smoke:
   try:r.run()
   except RuntimeError as e:assert str(e)=='smoke failed'
   else:raise AssertionError('failure ignored')
   assert events==[('E1-HNS-S12','smoke5')]
   combined.assert_not_called()
   assert q.read(exp/'QUEUE_STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
  else:
   r.run();combined.assert_called_once()
   expected=[(arm,stage) for arm in r.ARMS for stage in ('smoke5','train500','gradient-audit500','eval','report','publish')]
   assert events==expected,events
   assert q.read(exp/'QUEUE_STATE.json')['status']=='TWO_COMPLETED_GPU_IDLE'
   assert q.read(run/'completed.json')['arms']==list(r.ARMS)
'''.replace('FAIL_SMOKE',repr(fail_smoke))
    subprocess.run([sys.executable,'-c',code],check=True)
