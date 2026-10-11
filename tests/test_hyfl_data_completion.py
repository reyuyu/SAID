import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import torch
from tools.hyfl_data_completion import compare_long, read_author_csv, zip_members
from tools.eval_hyfl_native import encode
from tools.retrieval_bounded import ShardCache
from unittest.mock import patch


def tokenize(texts):
    return torch.tensor([[len(s.strip()), sum(s.strip().encode())] for s in texts])


class BackfillTests(unittest.TestCase):
    def test_archive_duplicate_escape_symlink(self):
        for names in [('a/x.jpg', 'b/x.jpg'), ('../x.jpg',), ('/x.jpg',), ('a\\x.jpg',)]:
            with zipfile.ZipFile(io.BytesIO(), 'w') as z:
                for name in names: z.writestr(name, b'x')
                with self.assertRaises(ValueError): zip_members(z)
        with zipfile.ZipFile(io.BytesIO(), 'w') as z:
            m = zipfile.ZipInfo('link.jpg'); m.external_attr = 0o120777 << 16
            z.writestr(m, 'target')
            with self.assertRaises(ValueError): zip_members(z)

    def test_csv_quoted_newlines_and_reject_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'a.tsv')
            with p.open('w', newline='') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['filepath', 'title'])
                w.writerow(['sa_1.jpg', 'a\nb\tquoted "caption"'])
            self.assertEqual(read_author_csv(p)[0]['title'], 'a\nb\tquoted "caption"')
            with p.open('a') as f: f.write('sa_1.jpg\tduplicate\n')
            with self.assertRaises(ValueError): read_author_csv(p)

    def test_real_appledouble_is_metadata_but_misnamed_jpeg_is_rejected(self):
        for data, valid in [(bytes.fromhex('0005160700020000') + bytes(18), True),
                            (b'JPEG masquerading as metadata', False)]:
            with zipfile.ZipFile(io.BytesIO(), 'w') as z:
                z.writestr('images/a.jpg', b'actual image')
                z.writestr('__MACOSX/images/._a.jpg', data)
                if valid:
                    images, other = zip_members(z)
                    self.assertEqual(set(images), {'a.jpg'})
                    self.assertEqual(other, ['__MACOSX/images/._a.jpg'])
                else:
                    with self.assertRaises(ValueError): zip_members(z)

    def test_exact_token_equivalence_and_real_difference(self):
        a = [{'filepath': 'sa_1.jpg', 'title': 'cat'}]
        s = [{'image_id': 'numeric_annotation_id', 'image_path': 'sa_1.jpg', 'caption': 'cat'}]
        self.assertEqual(compare_long(a, s, tokenize)['status'], 'VERIFIED_EXACT')
        a[0]['title'] = ' cat\n'
        self.assertEqual(compare_long(a, s, tokenize)['status'], 'VERIFIED_TOKEN_EQUIVALENT')
        a[0]['title'] = 'dog'
        self.assertEqual(compare_long(a, s, tokenize)['status'], 'PROTOCOL_MISMATCH')
        a[0]['filepath'] = 'sa_2.jpg'
        self.assertEqual(compare_long(a, s, tokenize)['status'], 'PROTOCOL_MISMATCH')

    def test_reordering_is_not_assumed_equivalent(self):
        a = [{'filepath': 'a.jpg', 'title': 'cat'}, {'filepath': 'b.jpg', 'title': 'dog'}]
        s = [{'image_path': 'b.jpg', 'caption': 'dog'}, {'image_path': 'a.jpg', 'caption': 'cat'}]
        p = compare_long(a, s, tokenize)
        self.assertFalse(p['order_exact']); self.assertEqual(p['status'], 'PROTOCOL_MISMATCH')

    def test_timing_receipt_does_not_change_features_or_tail(self):
        class Model:
            def encode_text(self, t): return t.float() + 1
            def encode_image(self, p): return p.float() + 1
        rows = [{'caption_id': str(i), 'caption': str(i)} for i in range(5)]
        images = [(str(i), str(i)) for i in range(5)]
        with tempfile.TemporaryDirectory() as d, \
             patch('model.longclip.tokenize', side_effect=lambda texts, **kw: torch.tensor([[int(x), 2] for x in texts])), \
             patch('tools.eval_hyfl_native.DataLoader', return_value=[torch.tensor([[0, 2], [1, 2], [2, 2]]), torch.tensor([[3, 2], [4, 2]])]):
            timing = {}
            cache = ShardCache(Path(d, 'a'), {'test': 'timing'})
            im, tx = encode(Model(), None, rows, images, '.', cache, 'cpu', 3, 3, workers=0, timings=timing)
            raw = torch.tensor([[i, 2] for i in range(5)]).float() + 1
            expected = raw / raw.norm(dim=-1, keepdim=True)
            self.assertTrue(torch.equal(im, expected)); self.assertTrue(torch.equal(tx, expected))
            self.assertEqual(set(timing), {'image_encoding_seconds', 'text_encoding_seconds'})
            self.assertTrue(all(x >= 0 for x in timing.values()))

    def test_incorrect_positive_mapping_rejected(self):
        a = [{'filepath': 'a.jpg', 'title': 'cat'}]
        s = [{'image_id': 'a', 'positive_image_id': 'b', 'image_path': 'a.jpg', 'caption': 'cat'}]
        with self.assertRaises(ValueError): compare_long(a, s, tokenize)


if __name__ == '__main__': unittest.main()
