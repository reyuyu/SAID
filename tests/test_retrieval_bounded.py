import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from tools.retrieval_bounded import (ShardCache, atomic_json, retrieval, select,
                                     stream_topk, validate_ids)
from tools.eval_resource_queue import Queue
from tools.flickr_full_protocol import parse_token_lines
from tools.eval_hyfl_native import preprocess_identity


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(7)

    def direct(self, im, tx, ids, cids, positive):
        scores = im @ tx.T
        ii = torch.arange(len(tx)).expand(len(im), -1)
        ti = torch.arange(len(im)).expand(len(tx), -1)
        # Independent scalar sorting reference, not the merge implementation.
        ix = torch.tensor([sorted(range(len(tx)), key=lambda j: (-float(row[j]), j))[:11]
                           for row in scores])
        it = torch.tensor([sorted(range(len(im)), key=lambda j: (-float(row[j]), j))[:11]
                           for row in scores.T])
        mapping = {s: i for i, s in enumerate(ids)}
        return {
            'I2T': {str(k): sum(any(positive[j] == ids[i] for j in row[:k])
                               for i, row in enumerate(ix.tolist())) for k in (1, 5, 10)},
            'T2I': {str(k): sum(mapping[positive[j]] in row[:k]
                               for j, row in enumerate(it.tolist())) for k in (1, 5, 10)}}

    def check_case(self, n, five=False, reorder=False):
        im = torch.nn.functional.normalize(torch.randn(n, 17), dim=1)
        tx = im.repeat_interleave(5 if five else 1, 0) + torch.randn(n * (5 if five else 1), 17) * .1
        tx = torch.nn.functional.normalize(tx, dim=1)
        ids = [f'i{i}' for i in range(n)]
        cids = [f'c{i}' for i in range(len(tx))]
        pos = [ids[i // (5 if five else 1)] for i in range(len(tx))]
        if reorder:
            p = torch.randperm(n); im = im[p]; ids = [ids[i] for i in p]
            p = torch.randperm(len(tx)); tx = tx[p]
            cids = [cids[i] for i in p]; pos = [pos[i] for i in p]
        direct = self.direct(im, tx, ids, cids, pos)
        for q, g in [(1, 1), (3, 7), (13, 9), (50, 500)]:
            m, _ = retrieval(im, tx, ids, cids, pos, query_chunk=q, gallery_chunk=g)
            self.assertEqual({d: m[d]['correct'] for d in m}, direct)

    def test_one_to_one(self):
        self.check_case(23)

    def test_one_to_five(self):
        self.check_case(23, True)

    def test_noncontiguous_positives_and_both_reorders(self):
        self.check_case(23, True, True)

    def test_duplicate_and_missing_ids(self):
        for ids, cids, pos in [(['a', 'a'], ['x', 'y'], ['a', 'a']),
                               (['a', 'b'], ['x', 'x'], ['a', 'b']),
                               (['a'], ['x'], ['missing']), (['a', 'b'], ['x'], ['a']),
                               (['a'], ['x', 'y'], ['a'])]:
            with self.assertRaises(ValueError):
                validate_ids(ids, cids, pos)

    def test_global_topk_across_blocks(self):
        q = torch.tensor([[1., 0.]])
        g = torch.stack([torch.tensor([i / 100, 1.]) for i in range(43)])
        idx, _ = stream_topk(q, g, 1, 3)
        self.assertEqual(idx[0].tolist(), list(range(42, 31, -1)))

    def test_exact_ties_and_near_ties(self):
        q = torch.tensor([[1., 0.]])
        g = torch.tensor([[1., 1.], [1., 1.], [1.0000001192092896, 1.], [.99999988, 1.]])
        for block in [1, 2, 4]:
            idx, _ = stream_topk(q, g, 1, block)
            self.assertEqual(idx[0].tolist(), [2, 0, 1, 3])

    def test_nan_inf_zero_dtype_rejected(self):
        for value in [float('nan'), float('inf'), 0.]:
            with self.assertRaises(ValueError):
                stream_topk(torch.tensor([[value]]), torch.ones(1, 1))
        with self.assertRaises(ValueError):
            stream_topk(torch.ones(1, 1, dtype=torch.float16), torch.ones(1, 1))

    def test_cache_tail_shards_merge_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            c = ShardCache(d, {'checkpoint_sha': 'a', 'context': 248})
            x = torch.randn(11, 4)
            chunks = []
            for start in range(0, 11, 4):
                ids = [str(i) for i in range(start, min(start + 4, 11))]
                c.write('image', start, ids, x[start:start + 4])
                chunks.append(c.read('image', start, ids))
            self.assertTrue(torch.equal(torch.cat(chunks), x))
            with self.assertRaises(ValueError):
                ShardCache(d, {'checkpoint_sha': 'wrong'})
            with self.assertRaises(ValueError):
                c.read('image', 0, ['wrong'])
            with self.assertRaises(FileExistsError):
                c.write('image', 0, ['0', '1', '2', '3'], x[:4])
            path, receipt = c.paths('image', 0)
            m = json.loads(receipt.read_text()); m['sha256'] = 'wrong'
            receipt.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                c.read('image', 0, ['0', '1', '2', '3'])

    def test_incomplete_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c = ShardCache(d, {'x': 1})
            p, r = c.paths('image', 0); torch.save(torch.ones(1, 4), p)
            with self.assertRaises(ValueError):
                c.read('image', 0, ['a'])
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'orphan').touch()
            with self.assertRaises(ValueError):
                ShardCache(d, {'x': 1})

    def test_no_overwrite_result(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'RESULT.json'); atomic_json(p, {'status': 'ok'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                atomic_json(p, {}, exclusive=True)

    def test_official_flickr_explicit_mapping_and_invalid_rows(self):
        rows = parse_token_lines([f'{im}.jpg#{i}\tcaption {i}\n' for i in range(5) for im in ['a', 'b']])
        self.assertEqual([r['positive_image_id'] for r in rows], ['a.jpg', 'b.jpg'] * 5)
        for lines in [['a.jpg#0\ttext\n'], ['../a.jpg#0\ttext'], ['bad'],
                      ['a.jpg#0\ttext'] * 5, ['a.jpg#0\t  ']]:
            with self.assertRaises(ValueError):
                parse_token_lines(lines)

    def test_process_independent_preprocess_cache_identity(self):
        a = 'Compose(Resize(224), <function convert_rgb at 0x123abc>, Normalize(mean=0.5))'
        b = 'Compose(Resize(224), <function convert_rgb at 0x456def>, Normalize(mean=0.5))'
        self.assertEqual(preprocess_identity(a), preprocess_identity(b))
        self.assertNotEqual(preprocess_identity(a), preprocess_identity(b.replace('224', '256')))

    def test_scheduler_failure_cancel_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            fail = Path(d, 'failflag'); fail.touch()
            # The exact same command succeeds after the external failure is resolved.
            command = [sys.executable, '-c',
                       f"import pathlib,json,sys; root=pathlib.Path({d!r}); "
                       "sys.exit(3) if (root/'failflag').exists() else None; "
                       "p=next(root.glob('run/a/attempt*/JOB.json')); "
                       "p=max(root.glob('run/a/attempt*/JOB.json')); "
                       "(p.parent/'RESULT.json').write_text(json.dumps({'status':'COMPLETED'}))"]
            plan = {'checkpoint': 'unused', 'jobs': [
                {'name': 'a', 'estimated_seconds': 10, 'command': command},
                {'name': 'b', 'estimated_seconds': 1, 'command': [sys.executable, '-c', 'pass']}]}
            q = Queue(plan, Path(d, 'run'), gpus=(0,), resource_check=lambda *_: {})
            self.assertEqual(q.run(), 1)
            self.assertEqual(q.state['jobs']['a']['status'], 'FAILED')
            self.assertEqual(q.state['jobs']['b']['status'], 'CANCELLED_BEFORE_START')
            with self.assertRaises(FileExistsError):
                Queue(plan, Path(d, 'run'), gpus=(0,), resource_check=lambda *_: {})
            fail.unlink()
            # Verify explicit recovery preserves failure and produces a new attempt.
            q = Queue(plan, Path(d, 'run'), resume=True, gpus=(0,), resource_check=lambda *_: {})
            q.run()
            self.assertEqual(q.state['jobs']['a']['status'], 'COMPLETED')
            self.assertEqual(len([r for r in q.state['attempts'] if r['name'] == 'a']), 2)
            self.assertTrue(Path(d, 'run/a/attempt1/worker.log').exists())

    def test_scheduler_preflight_failure_is_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            def fail(*_):
                raise RuntimeError('occupied GPU')
            q = Queue({'jobs': [{'name': 'a', 'estimated_seconds': 1}], 'checkpoint': 'unused'},
                      d, gpus=(0,), resource_check=fail)
            with self.assertRaisesRegex(RuntimeError, 'occupied GPU'):
                q.run()
            self.assertEqual(json.loads(Path(d, 'RUN_STATE.json').read_text())['status'], 'FAILED_PREFLIGHT')


if __name__ == '__main__':
    unittest.main()
