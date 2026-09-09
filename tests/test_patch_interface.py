"""Phase 1 tests: patch-token interface for the SmartCLIP / LongCLIP visual encoder.

These tests only exercise the new *interface*:
    CLIP.encode_image_with_patches(image) -> (global [B, 512], patches [B, 196, 512])

They assert that the original global-image path is bit-for-bit unchanged and that no
new parameters were introduced.
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import longclip  # noqa: E402
from model.model_longclip import CLIP, ModifiedResNet, VisionTransformer  # noqa: E402

BASELINE_KEYS_PATH = os.path.join(REPO_ROOT, 'tests', 'state_dict_keys_baseline.txt')
BATCH = 2
RESOLUTION = 224


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    net, _ = longclip.load_from_clip('ViT-B/16', device='cpu')
    net.eval()
    return net


@pytest.fixture(scope='module')
def images():
    torch.manual_seed(0)
    return torch.randn(BATCH, 3, RESOLUTION, RESOLUTION)


def test_a_global_compatibility(model, images):
    """TEST A: encode_image() and the global half of the new API must agree."""
    with torch.no_grad():
        g_old = model.encode_image(images)
        g_new, patches = model.encode_image_with_patches(images)
    max_abs_diff = (g_old - g_new).abs().max().item()
    print('max_abs_diff', max_abs_diff)
    assert max_abs_diff < 1e-6, 'max_abs_diff too large: %r' % max_abs_diff
    assert torch.allclose(g_old, g_new, atol=1e-6, rtol=1e-5)
    assert patches is not None


def test_b_shapes(model, images):
    """TEST B: ViT-B/16@224 shapes, plus the untouched checkpoint path."""
    with torch.no_grad():
        g, patches = model.encode_image_with_patches(images)
        g_ckpt = model.encode_image_with_checkpoint(images)
    print('global', tuple(g.shape), 'patches', tuple(patches.shape), 'ckpt', tuple(g_ckpt.shape))
    assert tuple(g.shape) == (BATCH, 512)
    assert tuple(patches.shape) == (BATCH, 196, 512)
    assert tuple(g_ckpt.shape) == (BATCH, 512)


def test_c_dtype_device(model, images):
    """TEST C: patch features live on the same device / dtype as the global feature."""
    with torch.no_grad():
        g, patches = model.encode_image_with_patches(images)
    assert patches.device == g.device
    assert patches.dtype == g.dtype


def test_d_finite(model, images):
    """TEST D: no NaN / Inf in either output."""
    with torch.no_grad():
        g, patches = model.encode_image_with_patches(images)
    assert bool(torch.isfinite(g).all())
    assert bool(torch.isfinite(patches).all())


def test_e_patch_count_from_model(model, images):
    """TEST E: patch count must follow from the model geometry, not a hard-coded 196."""
    visual = model.visual
    patch_size = int(visual.conv1.weight.shape[-1])
    expected_grid = int(visual.input_resolution) // patch_size
    expected_patches = expected_grid ** 2
    print('patch_size', patch_size, 'expected_grid', expected_grid, 'expected_patches', expected_patches)
    assert expected_patches == 196
    with torch.no_grad():
        _, patches = model.encode_image_with_patches(images)
    assert patches.shape[1] == expected_patches


def test_f_no_new_parameters(model):
    """TEST F: the interface must not add any parameter / buffer."""
    keys_now = sorted(model.state_dict().keys())
    with open(BASELINE_KEYS_PATH, 'r', encoding='utf8') as fp:
        keys_baseline = sorted(line.strip() for line in fp if line.strip())
    print('key_count_now', len(keys_now), 'key_count_baseline', len(keys_baseline))
    assert len(keys_now) == len(keys_baseline)
    assert keys_now == keys_baseline
    assert not any('patch' in key for key in keys_now)


def test_g_checkpoint_reload(model, images, tmp_path):
    """TEST G: save -> reload -> both APIs still work, no missing/unexpected keys."""
    state_dict = model.state_dict()
    ckpt_path = str(tmp_path / 'patch_interface_ckpt.pt')
    torch.save(state_dict, ckpt_path)

    reloaded, _ = longclip.load(ckpt_path, device='cpu')
    missing, unexpected = reloaded.load_state_dict(state_dict, strict=False)
    print('missing', len(missing), 'unexpected', len(unexpected))
    assert list(missing) == []
    assert list(unexpected) == []

    reloaded.eval()
    with torch.no_grad():
        g = reloaded.encode_image(images)
        g2, patches = reloaded.encode_image_with_patches(images)
    assert tuple(g.shape) == (BATCH, 512)
    assert tuple(g2.shape) == (BATCH, 512)
    assert tuple(patches.shape) == (BATCH, 196, 512)
    assert torch.allclose(g, g2, atol=1e-6, rtol=1e-5)


def test_h_resnet_backbone_is_rejected():
    """STEP 4: patch features are only defined for VisionTransformer backbones."""
    resnet = ModifiedResNet(layers=(3, 4, 6, 3), output_dim=512, heads=32,
                            input_resolution=224, width=64)
    dummy = type('DummyCLIP', (), {})()
    dummy.visual = resnet
    with pytest.raises(NotImplementedError):
        CLIP.encode_image_with_patches(dummy, torch.randn(1, 3, RESOLUTION, RESOLUTION))
