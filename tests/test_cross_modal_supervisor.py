import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from recovery.cross_modal_supervisor import eligible, scores
from model.nested_fusion_mask import FusionBranch, _interaction_regions, interaction_delta, pair_logits


class InteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.visual = (torch.randn(4), torch.randn(4, 16), torch.randn(4, 16, 16))
        self.text = (torch.randn(4), torch.randn(4, 16))

    def module(self, arm):
        branch = FusionBranch(nn.Identity(), 16, 16, f'cross_modal_{arm}', 196, rank=32)
        return SimpleNamespace(fusion=f'cross_modal_{arm}', fusion_branch=branch)

    def test_initial_zero_residual_matches_original_gate(self):
        for arm in ('x1', 'x2'):
            m = self.module(arm)
            for paired in (False, True):
                gate = m.fusion_branch.balanced_gate(self.visual[1], self.text[1], paired)
                expected = (gate * self.text[1] + (1 - gate) * self.visual[1] if paired else
                            gate * self.text[1][:, None] + (1 - gate) * self.visual[1][None])
                torch.testing.assert_close(pair_logits(m, self.visual, self.text, paired), expected, atol=0, rtol=0)

    def test_pairwise_diagonal_chunk_and_explicit_formulas(self):
        for arm in ('x1', 'x2'):
            m = self.module(arm)
            nn.init.normal_(m.fusion_branch.wo.weight)
            b = m.fusion_branch
            full = interaction_delta(m, self.visual, self.text)
            paired = interaction_delta(m, self.visual, self.text, True)
            torch.testing.assert_close(paired, full[range(4), range(4)], atol=2e-5, rtol=2e-5)
            chunks = torch.cat([interaction_delta(m, self.visual,
                                 tuple(x[i:i+2] for x in self.text)) for i in (0, 2)])
            torch.testing.assert_close(full, chunks, atol=2e-5, rtol=2e-5)
            if arm == 'x1':
                expected = b.wo(b.pt(self.text[1])[:, None] * b.pv(self.visual[1])[None])
            else:
                q, k, v = b.wq(self.text[1]), b.wk(self.visual[2]), b.wv(self.visual[2])
                values = []
                for text in q:
                    images = []
                    for keys, value in zip(k, v):
                        attn = (text @ keys.T / math.sqrt(32)).softmax(-1)
                        images.append(b.wo(text * (attn @ value)))
                    values.append(torch.stack(images))
                expected = torch.stack(values)
            torch.testing.assert_close(full, expected, atol=2e-5, rtol=2e-5)

    def test_both_modalities_and_new_parameters_have_gradients(self):
        for arm in ('x1', 'x2'):
            m = self.module(arm)
            # At zero Wo, upstream projections are correctly zero on update1.
            loss = interaction_delta(m, self.visual, self.text).square().sum()
            signal = interaction_delta(m, self.visual, self.text).sum()
            signal.backward()
            self.assertGreater(m.fusion_branch.wo.weight.grad.norm().item(), 0)
            # Diagnostic nonzero readout probes connectivity without training.
            m.fusion_branch.zero_grad(set_to_none=True)
            nn.init.normal_(m.fusion_branch.wo.weight)
            tv = tuple(x.clone().requires_grad_() for x in self.visual)
            tt = tuple(x.clone().requires_grad_() for x in self.text)
            y = interaction_delta(m, tv, tt)
            y.square().sum().backward()
            self.assertGreater(tt[1].grad.norm().item(), 0)
            self.assertGreater(tv[2 if arm == 'x2' else 1].grad.norm().item(), 0)
            for name, param in m.fusion_branch.named_parameters():
                if name.startswith(('pt.', 'pv.', 'wq.', 'wk.', 'wv.', 'wo.')):
                    self.assertIsNotNone(param.grad, name)
                    self.assertGreater(param.grad.norm().item(), 0, name)
            v_alt = tuple(x.flip(0) for x in tv)
            t_alt = tuple(x.flip(0) for x in tt)
            self.assertGreater((y - interaction_delta(m, v_alt, tt)).norm().item(), 0)
            self.assertGreater((y - interaction_delta(m, tv, t_alt)).norm().item(), 0)

    def test_regions_are_row_major_spatial_pool(self):
        tokens = torch.arange(196, dtype=torch.float32).reshape(1, 196, 1)
        actual = _interaction_regions(tokens)
        expected = torch.nn.functional.adaptive_avg_pool2d(
            tokens.reshape(1, 14, 14, 1).permute(0, 3, 1, 2), (4, 4)).flatten(2).transpose(1, 2)
        self.assertEqual(actual.shape, (1, 16, 1))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_x3_existing_defect_is_detected_and_not_relabelled(self):
        m = self.module('x3')
        nn.init.normal_(m.fusion_branch.wo.weight)
        matrix = interaction_delta(m, self.visual, self.text)
        paired = interaction_delta(m, self.visual, self.text, True)
        self.assertGreater((matrix[range(4), range(4)] - paired).norm().item(), 0.1)


class DecisionTests(unittest.TestCase):
    def test_requires_both_metrics(self):
        base = dict(Score5=72.8, J_long3=77.0)
        self.assertFalse(eligible(dict(Score5=72.9, J_long3=76.9), base))
        self.assertFalse(eligible(dict(Score5=72.7, J_long3=77.1), base))
        self.assertTrue(eligible(dict(Score5=72.8, J_long3=77.0), base))

    def test_score_uses_all_five_datasets_and_both_directions(self):
        names = ['COCO', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI', 'Urban-1k']
        metrics = {name: {d: {'R@1': 0.5} for d in ('I2T', 'T2I')} for name in names}
        result = scores(metrics)
        self.assertTrue(all(value == 50 for value in result.values()))
        metrics['COCO']['T2I']['R@1'] = 0.6
        self.assertEqual(scores(metrics)['Score5'], 51)
        self.assertEqual(scores(metrics)['J_long3'], 50)


if __name__ == '__main__':
    unittest.main()
