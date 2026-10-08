import copy
import pytest
from recovery import hns_balanced_macro_fourarm as r


def test_four_unique_arms_only_one_macro_variable_and_no_hns_repeats():
    assert list(r.ARMS)==['E1-HNS-A8','E2-Balanced-A8','E3-Balanced-S12','E4-Balanced-H125']
    for arm,scales in r.ARMS.items():
        base=r.runner.read(r.REFERENCES[r.METHODS[arm]]['exp']/'config.json');cfg=r.config(arm);r.frozen(cfg,arm)
        assert {k:v for k,v in cfg.items() if k not in r.MACRO_KEYS}==base
        assert sum(a!=b for a,b in zip(scales,(10,1,1)))==1
        assert tuple(cfg[k] for k in r.MACRO_KEYS)==scales
        bad=copy.deepcopy(cfg);bad['inclusion_max']=.5
        with pytest.raises(AssertionError):r.frozen(bad,arm)


def test_selection_checks_hns_gap_and_weak_signal():
    q={n:dict(Score5=71.1,J_long3=75,Short4=65.2,Urban_T2I=89.7) for n in ('HNS-v1','D3 Balanced',*r.ARMS)}
    q['HNS-v1']['Score5']=71.2;q['E2-Balanced-A8']['Score5']=71.18
    assert r.selection(q)['RECOMMENDED_NEXT_VALIDATION']=='KEEP_HNS_V1'
    q['E2-Balanced-A8']['Score5']=71.22
    result=r.selection(q);assert result['BEST_OVERALL_500']=='E2-Balanced-A8' and result['weak_single_seed_signal']


def test_parent_runner_fresh_local_500_commands_and_gates():
    # Configure in a child to avoid mutating the archived runner in this pytest process.
    import subprocess,sys
    code='''from recovery import hns_balanced_macro_fourarm as r
r.configure()
assert len(r.runner.ARMS)==4
for arm in r.ARMS:
 for smoke in (True,False):
  cmd=r.runner.training_command(arm,smoke)
  assert '--resume' not in cmd
  assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
  assert cmd[cmd.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
  assert cmd[cmd.index('-m')+1]==r.ENTRY
assert 'REAL_BF16_EQUIVALENCE.json' in r.runner.GATES
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_complete_queue_orders_smoke_fresh_train_audit_eval_report_publish_then_stops():
    import subprocess,sys
    code='''import tempfile
from pathlib import Path
from unittest.mock import patch
from recovery import hns_balanced_macro_fourarm as r
r.configure();q=r.runner;events=[]
with tempfile.TemporaryDirectory() as tmp:
 run=Path(tmp)/'run';run.mkdir();exp=Path(tmp)/'exp';exp.mkdir()
 q.RUN=run;q.EXP=exp
 q.dump(exp/'QUEUE_STATE.json',dict(status='PREPARED'))
 for name in q.GATES:q.dump(exp/name,dict(passed=True))
 q.dump(exp/'BASELINE_PROVENANCE.json',dict(production_sources={}))
 def activate(arm,smoke=False):
  q.search.ARM=arm;q.local.RUN=run/(arm+'.smoke5' if smoke else arm)
  q.local.PHASE=Path(tmp)/(arm+('-smoke' if smoke else '-formal'))
 def execute(name,cmd,training=False):
  arm=q.search.ARM;events.append((arm,name))
  if name in ('smoke5','train500'):
   assert '--resume' not in cmd
   assert cmd[cmd.index('--init-state')+1]==str(q.STEP0)
   step=5 if name=='smoke5' else 500
   q.dump(q.local.RUN/'step500/acceptance.json',dict(passed=True,ranks=[dict(completed_updates=step,updates_this_run=step,max_parameter_difference_from_rank0=0) for _ in range(4)]))
   if name=='train500':q.dump(q.local.RUN/'first-five-gate.json',dict(passed=True))
 def evaluate(supervisor,arm):events.append((arm,'eval'));return {}
 def report(arm,result,supervisor):events.append((arm,'report'))
 def publish():events.append((q.search.ARM,'publish'))
 with patch.object(q,'activate',side_effect=activate),patch.object(q,'sha',return_value=q.STEP0_SHA),patch.object(q,'git',return_value='HEAD'),patch.object(q,'evaluate',side_effect=evaluate),patch.object(q,'report_arm',side_effect=report),patch.object(q,'publish',side_effect=publish),patch.object(q,'combined') as combined,patch.object(q,'evaluator_proof'),patch.object(q,'identity',return_value={'passed':True}),patch.object(q,'rows',return_value=[]),patch.object(q,'matched_stream',return_value={'passed':True,'records':512000}),patch.object(q.search,'sampling_audit'),patch('train.train_nested_semantic_mask.code_manifest',return_value={}),patch('tools.eval_five_parallel.require_gpu_idle'),patch.object(q.signal,'signal'),patch.object(q.local,'Supervisor') as supervisor:
  supervisor.return_value.execute.side_effect=execute
  q.run();combined.assert_called_once()
 expected=[(arm,stage) for arm in r.ARMS for stage in ('smoke5','train500','gradient-audit500','eval','report','publish')]
 assert events==expected,events
 assert q.read(exp/'QUEUE_STATE.json')['status']=='FOUR_COMPLETED_GPU_IDLE'
 assert q.read(run/'completed.json')['arms']==list(r.ARMS)
'''
    subprocess.run([sys.executable,'-c',code],check=True)
