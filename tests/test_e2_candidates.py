"""Control tests: frozen mothers, exact ramp convention, stateful queue bounds."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch

from recovery import e2_candidates as e


class SchedulingTests(unittest.TestCase):
    def test_completed_before_update_convention(self):
        from model.nested_semantic_mask import inclusion_weight
        for update in (1,100,200,500,501,1217,2434):
            completed=update-1
            self.assertEqual(inclusion_weight('A3',completed),min(1.,completed/200.))
            self.assertEqual(e.ramp500('A3',completed),min(1.,completed/500.))
        self.assertEqual(e.ramp500('A2',1000),0.)

    def test_same_detached_child_gradient_only_scale_changes(self):
        from model.balanced_hparam_search import detail_chain_inclusion
        f=torch.tensor([[.1,.5,.7]],requires_grad=True)
        d=torch.tensor([[.4,.6,.3]],requires_grad=True)
        lowest=torch.tensor([[.5,.2,.4]],requires_grad=True)
        base=detail_chain_inclusion(f,d,lowest).mean()
        for completed in (0,1,99,199,499,500,1217):
            old=min(1.,completed/200.)
            new=e.ramp500('A3',completed)
            before=torch.autograd.grad(base*old,(f,d),retain_graph=True)
            after=torch.autograd.grad(base*new,(f,d),retain_graph=True)
            for a,b in zip(before,after):
                self.assertTrue(torch.allclose(a*new,b*old,rtol=1e-6,atol=1e-8))
        self.assertIsNone(torch.autograd.grad(base,lowest,allow_unused=True)[0])

    def test_only_B_config_extension_and_A_exact_mother(self):
        for candidate in ('A','B'):
            cfg=e.mother_config(candidate)
            if candidate=='B':cfg['inclusion_ramp_steps']=500
            e.assert_frozen(candidate,cfg)
            for key,value in [('workers',4),('sampling_seed',1),('view_weights',[1.,1.,1.]),('inclusion_max',.25)]:
                bad=copy.deepcopy(cfg);bad[key]=value
                with self.assertRaises(AssertionError):e.assert_frozen(candidate,bad)

    def test_queue_has_only_two_E2_candidates(self):
        self.assertEqual(list(e.SPEC),['A','B'])
        self.assertEqual(e.SPEC['A']['segments'],[(1217,2434)])
        self.assertEqual(e.SPEC['B']['segments'],[(0,500),(500,1217),(1217,2434)])
        for candidate in ('A','B'):
            for start,stop in e.SPEC[candidate]['segments']:
                cmd=e.training_command(candidate,start,stop)
                self.assertEqual(cmd[cmd.index('--max-updates')+1],str(stop))
                self.assertLessEqual(stop,2434)
                self.assertEqual(cmd[cmd.index('--image-root')+1],str(e.IMAGES))
                self.assertEqual('--resume' in cmd,start>0)
                if candidate=='A':self.assertEqual(cmd[cmd.index('--resume')+1],str(e.PARENT))
                elif start==0:self.assertEqual(cmd[cmd.index('--init-state')+1],str(e.STEP0))

    def test_classification_respects_long_guard(self):
        m={'test':{'I2T':{'R@1':.5},'T2I':{'R@1':.5}}}
        base=dict(scores_percent=dict(zip(e.KEYS,[73.6,78,87,67])),metrics=m)
        good=copy.deepcopy(base);good['scores_percent']['Score5']+=.1
        self.assertEqual(e.classify(good,base),'STRONG_CANDIDATE')
        bad=copy.deepcopy(good);bad['scores_percent']['J_long3']-=.2
        self.assertNotEqual(e.classify(bad,base),'STRONG_CANDIDATE')

    def test_resume_rejects_wrong_optimizer_counter_and_cursor(self):
        # Real serialization schema, mock binary I/O; no CUDA or model required.
        p=dict(completed_steps=1217,global_step=1217,scheduler=dict(completed_steps=1217,horizon=4868),
            scheduler_horizon=4868,data_cursor=dict(next_epoch=1,next_batch=0),next_epoch=1,next_batch=0,
            model={'a':1},adapter={},optimizer={'state':{'a':{'step':torch.tensor(500)}}},
            scaler=dict(enabled=False,dtype='bfloat16',state=None))
        with patch.object(e,'sha',return_value=e.PARENT_SHA),patch('torch.load',return_value=p):
            with self.assertRaises(AssertionError):e.checkpoint_identity(e.PARENT,1217,e.PARENT_SHA)
        p['optimizer']['state']['a']['step']=torch.tensor(1217)
        p['data_cursor']['next_batch']=1
        with patch.object(e,'sha',return_value=e.PARENT_SHA),patch('torch.load',return_value=p):
            with self.assertRaises(AssertionError):e.checkpoint_identity(e.PARENT,1217,e.PARENT_SHA)

    def test_failure_blocks_second_candidate(self):
        class FailedProcess:
            pid=123456
            def wait(self,**kw): return 2
            def poll(self): return 2
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'QUEUE_RUN',Path(tmp)), \
                patch.object(e.signal,'signal'),patch.object(e.subprocess,'Popen',return_value=FailedProcess()) as start:
            with self.assertRaises(RuntimeError):e.queue()
            self.assertEqual(start.call_count,1)
            self.assertEqual(e.read(Path(tmp)/'QUEUE_STATE.json')['status'],'FAILED')


if __name__=='__main__':unittest.main()
