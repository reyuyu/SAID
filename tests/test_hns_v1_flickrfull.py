import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import eval_hns_v1_flickrfull as hns


class HNSBindingTests(unittest.TestCase):
    def test_only_identity_changes_functions_are_reused(self):
        worker = hns.native
        functions = {n: getattr(worker, n) for n in
                     ('load_model', 'identity', 'encode', 'run', 'retrieval', 'rows_and_images')}
        with patch.object(worker, 'MODEL_SHA', hns.E2_SHA), patch.object(worker, 'SOURCES', worker.SOURCES):
            hns.bind()
            self.assertEqual(worker.MODEL_SHA, hns.MODEL_SHA)
            self.assertIn(hns.ENTRY, worker.SOURCES)
            for n, f in functions.items():
                self.assertIs(getattr(worker, n), f)
            hns.bind()
            self.assertEqual(worker.SOURCES.count(hns.ENTRY), 1)

    def test_wrong_checkpoint_rejected_before_model_creation(self):
        with patch.object(hns.native, 'MODEL_SHA', hns.E2_SHA), patch.object(hns.native, 'SOURCES', hns.native.SOURCES):
            hns.bind()
            with patch.object(hns.native, 'sha', return_value=hns.E2_SHA):
                with self.assertRaisesRegex(ValueError, 'checkpoint SHA mismatch'):
                    hns.native.load_model('wrong.pt', 'cpu')

    def test_source_drift_rejected(self):
        with patch.object(hns, 'sha', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'frozen source changed'):
                hns.bind()

    def test_fixed_protocol_no_substitution(self):
        job = json.loads(Path('/root/said_hyfl_data_completion_v1/full_protocol/JOB.json').read_text())
        hns.validate_job(job, hns.CHECKPOINT)
        for key, value in [('image_batch', 32), ('normalization', 'cpu'),
                           ('gallery_chunk', 8192), ('name', 'Urban-1k')]:
            wrong = copy.deepcopy(job); wrong[key] = value
            with self.assertRaises(ValueError):
                hns.validate_job(wrong, hns.CHECKPOINT)
        with self.assertRaises(ValueError):
            hns.validate_job(job, '/tmp/other.pt')


if __name__ == '__main__':
    unittest.main()
