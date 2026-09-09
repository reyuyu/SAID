"""Spatial checks runnable without model checkpoints or evaluation data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.salu.grounding_sanity_probe import D4_NAMES, d4_transform, shift_attention
from eval.salu.grounding_metrics import phrase_metrics


def test_d4_eight_distinct_symmetries_and_locations():
    a = np.zeros((14,14)); a[5,7] = 1
    expected = [(5,7), (5,6), (8,7), (7,5), (6,5), (8,6), (7,8), (6,8)]
    for name, peak in zip(D4_NAMES, expected):
        out = d4_transform(a, name)
        assert np.unravel_index(out.argmax(), out.shape) == peak
        assert out.sum() == 1
    grid = np.arange(196).reshape(14,14)
    assert len({d4_transform(grid, name).tobytes() for name in D4_NAMES}) == 8
    np.testing.assert_array_equal(d4_transform(d4_transform(grid,'rotate90'),'rotate270'), grid)


def test_shift_retains_only_in_bounds_mass():
    a = np.zeros((14,14)); a[0,0] = .25; a[13,13] = .75
    out = shift_attention(a, 1, 1)
    assert out[1,1] == 1 and out[0,0] == 0 and out.sum() == 1
    np.testing.assert_array_equal(shift_attention(a,0,0), a)
    with pytest.raises(ValueError, match='all attention'):
        shift_attention(a,1,-1)


def test_one_hot_npy_loader_metric_renderer_end_to_end(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/said_dashboard'))
    import grounding_data as gd
    a = np.zeros((14,14), dtype=np.float32); a[5,7] = 1
    folder = tmp_path/'attention/phase22_router'; folder.mkdir(parents=True)
    np.save(folder/'onehot.npy', a)
    loaded = gd.load_map(tmp_path, 'phase22_router', 'onehot')
    np.testing.assert_array_equal(a, loaded)
    boxes = [[112,80,128,96]]
    metrics = phrase_metrics(loaded, boxes, {})
    assert metrics['pointing_correct'] is True
    assert metrics['gt_mass'] == 1
    assert metrics['mass_gain'] == pytest.approx(1-1/196)
    assert metrics['peak_xy'] == [120,88]
    image = gd.annotated_image(Image.new('RGB',(224,224),'white'), boxes,
                               attention=loaded, peak_xy=metrics['peak_xy'])
    assert image.getpixel((120,88)) == (255,0,0)
    assert image.getpixel((112,80)) == (0,255,0)
    assert 112 < 120 < 128 and 80 < 88 < 96
