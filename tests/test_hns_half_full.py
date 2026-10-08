"""Continuation bounds, exact ancestry and fail-closed epoch sequencing."""
import copy
import unittest
from unittest.mock import patch
from recovery import e2_candidates as e,hns_half_full as h


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.old={k:copy.deepcopy(getattr(e,k)) for k in ('SPEC','PARENT','PARENT_SHA','LIMIT','INITIAL_A_START','ENTRY','CODE','PUBLICATION_TITLE')}
        h.configure()

    def tearDown(self):
        for k,v in self.old.items():setattr(e,k,v)

    def test_two_segments_resume_correct_parent_never_fresh(self):
        self.assertEqual(e.SPEC['A']['segments'],[(2434,3651),(3651,4868)])
        for start,stop in h.SEGMENTS:
            cmd=e.training_command('A',start,stop)
            self.assertEqual(cmd[cmd.index('--max-updates')+1],str(stop))
            self.assertIn('recovery.hns_half_full',cmd)
            self.assertEqual(cmd[cmd.index('--resume')+1],str(e.parent_for_segment('A',start)))
            self.assertEqual(cmd[cmd.index('--image-root')+1],str(e.IMAGES))
        self.assertEqual(e.parent_for_segment('A',2434),h.PARENT)
        self.assertEqual(e.parent_for_segment('A',3651),e.paths('A')[1]/'step3651/step003651.pt')
        self.assertEqual(e.LIMIT,4868)
        self.assertLess(h.ORDER.index('strict export/evaluate/report/sync3651'),h.ORDER.index('train3652..4868'))

    def test_config_and_half_schedule_frozen(self):
        from recovery.hns_half1217 import half_weight
        cfg=e.mother_config('A');e.assert_frozen('A',cfg)
        for key,value in [('workers',4),('view_weights',[1,1,1]),('hns_half_after500',False),('inclusion_max',.5)]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(AssertionError):e.assert_frozen('A',bad)
        for completed in (2434,3650,3651,4867):self.assertEqual(half_weight(completed),.5)

    def test_original_e2_defaults_are_opt_in(self):
        for k,v in self.old.items():setattr(e,k,v)
        self.assertEqual(e.LIMIT,2434)
        self.assertEqual(e.INITIAL_A_START,1217)
        self.assertEqual(e.ENTRY,'recovery.e2_candidates')

    def test_failed_training_blocks_evaluation_and_second_segment(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp)/'exp';runtime=Path(tmp)/'runtime';exp.mkdir();runtime.mkdir()
            e.dump(exp/'STATE.json',dict(status='PREPARED'))
            e.dump(exp/'CPU_TESTS.json',dict(passed=True))
            e.dump(exp/'RESUME_PROVENANCE.json',dict(production_sources={}))
            with patch.object(e,'paths',return_value=(exp,runtime)),patch.object(e,'sha',return_value=h.PARENT_SHA), \
                    patch.object(e,'evaluator_sources'),patch.object(e,'git',return_value='commit'), \
                    patch.object(e,'checkpoint_identity',return_value={}),patch.object(e,'publish') as publish, \
                    patch.object(e,'Supervisor') as supervisor,patch('train.train_nested_semantic_mask.code_manifest',return_value={}), \
                    patch('tools.eval_five_parallel.require_gpu_idle'),patch('recovery.resource_stall_v2.system_snapshot',return_value={'memory_events':{}}), \
                    patch.object(h.signal,'signal'),patch.object(e,'IMAGES',Path(tmp)/'ShareGPT4V'):
                supervisor.return_value.execute.side_effect=RuntimeError('training failed')
                with self.assertRaisesRegex(RuntimeError,'training failed'):h.run()
                self.assertEqual(supervisor.return_value.execute.call_count,1)
                supervisor.return_value.evaluate.assert_not_called();publish.assert_not_called()
                self.assertEqual(e.read(exp/'STATE.json')['status'],'FAILED')


if __name__=='__main__':unittest.main()
