"""Exercise the export-to-loader contract with tiny CPU model/data doubles."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from eval.salu import export_dashboard_artifacts as export
from tools.said_dashboard import data as dash


def test_export_includes_images_for_every_checkpoint(tmp_path, monkeypatch):
    class Dataset:
        json_data = [{'image': 'image%d.jpg' % i} for i in range(2)]

        def __getitem__(self, i):
            return torch.full((3, 224, 224), float(i)), 'caption %d. More text.' % i

    class Router:
        q_proj = staticmethod(lambda x: x)
        k_proj = staticmethod(lambda x: x)
        tau_said = 0.07

        def route_pairwise(self, texts, patches, chunk_size=None):
            logits = torch.einsum('jd,ind->ijn', texts, F.normalize(patches, dim=-1))
            a = torch.softmax(logits / self.tau_said, dim=-1)
            return F.normalize(torch.einsum('ijn,ind->ijd', a, patches), dim=-1), a

    class Model:
        def __init__(self, *args, **kwargs):
            self.said_router = Router()
            self.clip = SimpleNamespace(logit_scale=torch.tensor(0.),
                                        encode_image_with_patches=self.encode)

        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, image):
            patches = torch.arange(196 * 4, dtype=torch.float32).reshape(1, 196, 4) + 1
            return patches.mean(1), patches

    monkeypatch.setattr(export, 'share4v_val_dataset', Dataset)
    monkeypatch.setattr(export.longclip, 'load_from_clip', lambda *a, **kw: (None, None))
    monkeypatch.setattr(export, 'SALUModel', Model)
    monkeypatch.setattr(export, 'encode_texts', lambda model, captions, device:
                        F.normalize(torch.ones(len(captions), 4), dim=-1))
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoints', 'initial:,final:',
                                    '--num_samples', '2', '--device', 'cpu',
                                    '--diagnostics_dir', str(tmp_path / 'diagnostics'),
                                    '--output_dir', str(tmp_path / 'artifacts')])
    export.main()
    root = tmp_path / 'artifacts'
    for tag in dash.checkpoints(dash.load_manifest(root)):
        for i in range(2):
            image = dash.load_image(root, tag, i)
            assert image.size == (224, 224)
            np.testing.assert_array_equal(image, dash.load_image(root, 'initial', i))
            for variant in dash.VARIANTS:
                assert dash.load_attention(root, tag, i, variant).shape == (14, 14)
