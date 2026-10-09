import subprocess
import sys
import pytest


def invoke(source):
    subprocess.run([sys.executable,'-c',source],check=True)


def test_only_e3_e4_full_checkpoint_parents_frozen_configuration():
    invoke('''from recovery import hns_s12_full4868 as task
task.configure();r=task.protocol
assert r.TARGETS==(3651,4868)
assert r.predecessor(3651)==(task.PARENT,2434)
assert r.predecessor(4868)==(r.segment(3651)/'training/step003651.pt',3651)
assert task.PARENT_SHA=='4cde7d4b91af80755215f20687b80856266fa1faed4b2db864a88ae72b030975'
r.frozen(r.common.read(task.BASE/'config.json'))
for n in r.TARGETS:
 c=r.training_command(n)
 assert c[c.index('--resume')+1]==str(r.predecessor(n)[0])
 assert c[c.index('--max-updates')+1]==str(n)
 assert c[c.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
 assert c[c.index('-m')+1]==task.ENTRY
 assert 'student' not in c[c.index('--resume')+1]
for n in (500,1217,2434,6000):
 try:r.predecessor(n)
 except AssertionError:pass
 else:raise AssertionError('Unauthorized target accepted')
''')


@pytest.mark.parametrize('failure',[False,True])
def test_sequence_requires_e3_evaluation_before_e4_and_stops4868(failure):
    invoke('''import tempfile
from pathlib import Path
from unittest.mock import patch
from tools import eval_five_parallel
from recovery import hns_s12_full4868 as task
task.configure();r=task.protocol;events=[];fail=FAILURE
with tempfile.TemporaryDirectory() as tmp:
 r.RUN=Path(tmp)/'run';r.EXP=Path(tmp)/'exp';r.RUN.mkdir()
 r.dump(r.EXP/'STATE.json',dict(status='PREPARED'));r.dump(r.EXP/'CPU_TESTS.json',dict(passed=True))
 for n in r.TARGETS:
  r.segment(n).mkdir();r.dump(r.segment(n)/'training/acceptance.json',dict(passed=True,ranks=[dict(completed_updates=n,updates_this_run=1217,max_parameter_difference_from_rank0=0)]*4))
 class Supervisor:
  def execute(self,name,command,training=False):events.append((name,list(command)))
 def evaluate(s,n):
  events.append(('eval',n))
  if fail:raise RuntimeError('E3 evaluator failed')
  return {}
 with patch.object(eval_five_parallel,'require_gpu_idle',lambda _:None),patch.object(r.local,'Supervisor',Supervisor),patch.object(r,'phase',lambda n:Path(tmp)/('phase'+str(n))),patch.object(r,'identity',lambda p,n:dict(sha256=task.PARENT_SHA)),patch.object(r,'resume_gate',lambda n:events.append(('gate',n))),patch.object(r,'evaluate',evaluate),patch.object(r,'report',lambda n,s,v:events.append(('report',n))),patch.object(r,'combined',lambda n:None),patch.object(r,'publish',lambda:events.append(('publish',None))):
  if fail:
   try:r.run()
   except RuntimeError as error:assert 'E3 evaluator failed' in str(error)
   else:raise AssertionError('Failure silently ignored')
   assert [e[0] for e in events]==['train','gate','gradient','eval']
   assert r.common.read(r.EXP/'STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
  else:
   r.run()
   assert [e[0] for e in events]==['train','gate','gradient','eval','report','publish']*2
   assert r.common.read(r.RUN/'completed.json')['stop']==4868
   assert r.common.read(r.RUN/'completed.json')['nodes']==[3651,4868]
   assert r.common.read(r.EXP/'STATE.json')['status']=='COMPLETED_GPU_IDLE'
   trains=[v for k,v in events if k=='train']
   assert trains[0][trains[0].index('--resume')+1]==str(task.PARENT)
   assert trains[1][trains[1].index('--resume')+1]==str(r.segment(3651)/'training/step003651.pt')
'''.replace('FAILURE',repr(failure)))


@pytest.mark.parametrize('key,value',[
    ('lambda_sparse',1),('lambda_align',8),('lambda_hierarchy',4),
    ('hns_half_after500',True),('hns_detach_child',True),('hns_beta',[2,4]),
    ('view_weights',[1,1,1]),('sampling_mode','nested_detail_kr234')])
def test_rejects_loss_sampling_and_phase_switch_drift(key,value):
    invoke('''from recovery import hns_s12_full4868 as task
task.configure();r=task.protocol;c=r.common.read(task.BASE/'config.json');r.frozen(c)
c[KEY]=VALUE
try:r.frozen(c)
except AssertionError:pass
else:raise AssertionError('Frozen method drift accepted')
'''.replace('KEY',repr(key)).replace('VALUE',repr(value)))


def test_matched_node_report_defers_interpretation_until_e4():
    invoke('''import copy,tempfile
from pathlib import Path
from recovery import hns_s12_full4868 as task
task.configure();r=task.protocol
history,_=task.archived(r.REFERENCE_BRANCH,r.REFERENCE_FILE)
half,_=task.archived(task.HALF_BRANCH,task.HALF_FILE)
diagnostics,_=task.archived(r.REFERENCE_BRANCH,task.DIAG_FILE)
refs={m:{str(n):history['models'][m][str(n)] for n in task.TARGETS} for m in ('HNS_v1','D3_Balanced')}
refs['HNS_Half']={str(n):half['models'][str(n)] for n in task.TARGETS}
e2=r.common.read(task.BASE/'step2434/RESULTS.json');mask=r.common.read(task.BASE/'step2434/MASK_HIERARCHY_AUDIT.json')
with tempfile.TemporaryDirectory() as tmp:
 task.EXP=r.EXP=Path(tmp)
 r.dump(task.EXP/'REFERENCES.json',dict(models=refs))
 r.dump(task.EXP/'REFERENCE_DIAGNOSTICS.json',dict(models=diagnostics,S12_epoch2_results=e2,S12_epoch2_masks=mask))
 for n in task.TARGETS:
  value=copy.deepcopy(refs['HNS_v1'][str(n)])
  value['scores_percent']['Score5']+=.01
  value['comparisons']={m:r.common.compare(value,refs[m][str(n)]) for m in refs}
  assert abs(value['comparisons']['HNS_v1']['quality_delta_pp']['Score5']-.01)<1e-9
  r.dump(task.EXP/f'step{n}/RESULTS.json',value)
  r.dump(task.EXP/f'step{n}/MASK_HIERARCHY_AUDIT.json',mask)
  r.dump(task.EXP/f'step{n}/GRADIENT_AUDIT.json',dict(group_diagnostics={}))
 task.combined([3651])
 assert r.common.read(task.EXP/'RESULTS.json')['status']=='PARTIAL'
 assert not (task.EXP/'SCIENTIFIC_DIAGNOSTICS.json').exists()
 task.combined([3651,4868])
 full=r.common.read(task.EXP/'RESULTS.json')
 assert full['status']=='COMPLETED' and full['stop_updates']==4868
 assert full['scientific_diagnostics']['S12_exceeds_HNS_at_both_late_nodes']
 assert (task.EXP/'SCIENTIFIC_DIAGNOSTICS.json').exists()
 assert 'weak single-seed signal' in (task.EXP/'REPORT.md').read_text()
''')
