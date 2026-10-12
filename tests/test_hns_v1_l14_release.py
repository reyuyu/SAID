import argparse
import torch
from model import longclip
from model.backbone import load_native_state

def test_l14_student_shape_and_strict_load():
    base, _ = longclip.load_from_clip('ViT-L/14', device='cpu', args=argparse.Namespace())
    state = base.state_dict()
    model, _, spec = load_native_state(state, base_model='ViT-L/14', device='cpu')
    assert spec['embedding_dim'] == 768
    assert spec['patch_count'] == 256
    assert model.context_length == 248
