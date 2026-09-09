import json

import numpy as np
import pytest
import torch
from PIL import Image

from eval.salu.flickr_entities import load_split, read_annotation, read_sentences
from eval.salu.grounding_metrics import (clip_geometry, contains, patch_coverage,
                                       phrase_metrics, phrase_switch, transform_boxes,
                                       union_area, union_iou)
from eval.salu.semantic_grounding_eval import direct_attention, switching_groups


def test_fractional_patch_overlap():
    w = patch_coverage([[8, 0, 24, 16]])
    assert w[0, 0] == .5 and w[0, 1] == .5
    assert w.sum() == 1


def test_multi_box_union_neither_double_counts_nor_fills_gaps():
    boxes = [[0, 0, 16, 16], [8, 0, 24, 16], [48, 0, 64, 16]]
    w = patch_coverage(boxes)
    assert w[0, :4].tolist() == [1, .5, 0, 1]
    assert union_area(boxes) == 640
    assert union_iou(boxes, boxes) == 1
    assert union_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]]) == pytest.approx(1/3)


def test_pointing_uses_patch_center_and_mass_uses_overlap():
    a = np.zeros((14, 14)); a[0, 0] = 1
    stats = phrase_metrics(a, [[0, 0, 8, 16]], {})
    assert stats['pointing_correct'] is False  # x=8 lies outside half-open box
    assert stats['gt_mass'] == .5
    assert stats['gt_area_fraction'] == pytest.approx(.5/196)
    assert stats['mass_gain'] == pytest.approx(.5-.5/196)
    assert stats['localization_margin'] is None
    assert contains([[0, 0, 16, 16]], 8, 8)


def test_uniform_attention_has_zero_mass_gain():
    a = np.ones((14, 14))/196
    s = phrase_metrics(a, [[1, 2, 74, 99], [60, 70, 200, 210]], {})
    assert s['mass_gain'] == pytest.approx(0, abs=1e-12)
    assert s['mass_lift'] == pytest.approx(1)


def test_distractors_exclude_overlapping_entities_and_classify_peak():
    a = np.zeros((14, 14)); a[0, 0] = .2; a[0, 3] = .8
    target = [[0, 0, 16, 16]]
    other = {'nested': [[0, 0, 16, 16]], 'separate': [[48, 0, 64, 16]]}
    s = phrase_metrics(a, target, other)
    assert s['distractor_count'] == 1
    assert s['distractor_mass'] == .8
    assert s['localization_margin'] == pytest.approx(-.6)
    assert s['target_gt_distractor'] is False
    assert s['peak_location'] == 'other_object'
    assert phrase_metrics(a, target, {})['peak_location'] == 'background'


def test_switch_margin_correct_wrong_and_identical_maps():
    a = np.zeros((14, 14)); a[0, 0] = 1
    b = np.zeros((14, 14)); b[-1, -1] = 1
    wa = patch_coverage([[0, 0, 16, 16]])
    wb = patch_coverage([[208, 208, 224, 224]])
    assert phrase_switch(a, b, wa, wb)['switch_margin'] == 1
    assert phrase_switch(b, a, wa, wb)['switch_margin'] == -1
    assert phrase_switch(a, a, wa, wb)['switch_margin'] == 0


def test_direct_clip_attention_uses_cosine_without_router():
    h = torch.tensor([[3., 0.], [0., 7.]])
    t = torch.eye(2)
    a = direct_attention(h, t)
    assert a.shape == (2, 2)
    torch.testing.assert_close(a.sum(-1), torch.ones(2))
    assert a.argmax(-1).tolist() == [0, 1]
    torch.testing.assert_close(a, direct_attention(h*10, t*4))
    with pytest.raises(ValueError):
        direct_attention(h, t, tau_eval=0)


@pytest.mark.parametrize('boxes', [[[1, 1, 1, 2]], [[4, 2, 1, 3]], [[0, 0, float('nan'), 1]], [[1, 2, 3]]])
def test_malformed_boxes_rejected(boxes):
    with pytest.raises(ValueError):
        patch_coverage(boxes)


def test_empty_boxes_are_not_scored_as_failed_grounding():
    assert patch_coverage([]).sum() == 0
    with pytest.raises(ValueError, match='no visible boxes'):
        phrase_metrics(np.ones((14, 14))/196, [], {})


@pytest.mark.parametrize('width,height', [(501, 333), (333, 501), (225, 224), (600, 400)])
def test_geometry_matches_torchvision_resize_crop(width, height):
    from torchvision.transforms import Resize, CenterCrop, InterpolationMode
    g = clip_geometry(width, height)
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8))
    expected = CenterCrop(224)(Resize(224, interpolation=InterpolationMode.BICUBIC)(image))
    x, y = g['crop_xy']
    actual = image.resize(tuple(g['resized_size']), Image.BICUBIC).crop((x, y, x+224, y+224))
    np.testing.assert_array_equal(actual, expected)
    boxes = transform_boxes([[0, 0, width, height]], g)
    np.testing.assert_allclose(boxes, [[0, 0, 224, 224]])


def test_crop_exclusion_and_partial_box():
    g = clip_geometry(448, 224)
    assert not len(transform_boxes([[0, 0, 100, 100]], g))
    np.testing.assert_allclose(transform_boxes([[100, 0, 128, 16]], g), [[0, 0, 16, 16]])


def test_official_sentence_markup_and_box_convention(tmp_path):
    sent = tmp_path/'1.txt'
    sent.write_text('[/EN#1/people A man] holds [/EN#2/animals a dog].\n')
    phrases = read_sentences(sent)
    assert [p['phrase'] for p in phrases] == ['A man', 'a dog']
    assert phrases[0]['sentence'] == 'A man holds a dog.'
    assert phrases[1]['categories'] == ['animals']
    xml = tmp_path/'1.xml'
    xml.write_text('<annotation><size><width>224</width><height>224</height></size>'
                   '<object><name>1</name><name>2</name><bndbox>'
                   '<xmin>1</xmin><ymin>1</ymin><xmax>16</xmax><ymax>16</ymax>'
                   '</bndbox></object><object><name>3</name><scene>1</scene></object></annotation>')
    w, h, boxes, flags = read_annotation(xml)
    assert boxes['1'] == boxes['2'] == [[0, 0, 16, 16]]
    assert flags['3'] == 'scene'


def test_missing_roots_and_files_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv('FLICKR30K_ROOT', raising=False)
    monkeypatch.delenv('FLICKR30K_ENTITIES_ROOT', raising=False)
    with pytest.raises(ValueError, match='FLICKR30K_ROOT'):
        load_split()
    (tmp_path/'val.txt').write_text('1\n')
    with pytest.raises(FileNotFoundError, match='no silent image skipping'):
        load_split(str(tmp_path), str(tmp_path))


def test_switch_groups_select_distinct_spatial_entities():
    def phrase(key, entity, text, box):
        return {'id': key, 'entity_id': entity, 'phrase': text, 'boxes': [box]}
    records = [{'id': '1', 'phrases': [phrase('a', '1', 'a man', [0, 0, 30, 30]),
                 phrase('b', '1', 'man', [0, 0, 30, 30]),
                 phrase('c', '2', 'a dog', [150, 150, 200, 200])]}]
    assert switching_groups(records)[0]['phrase_ids'] == ['b', 'c']


def test_native_clip_patch_extraction_preserves_spatial_order():
    from clip.model import CLIP
    from eval.salu.semantic_grounding_eval import native_clip_patches
    model = CLIP(embed_dim=32, image_resolution=224, vision_layers=1,
                 vision_width=64, vision_patch_size=16, context_length=77,
                 vocab_size=100, transformer_width=64, transformer_heads=1,
                 transformer_layers=1).float().eval()
    captured = []
    handle = model.visual.transformer.register_forward_hook(lambda module, inputs, out: captured.append(out.detach()))
    image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        global_feature = model.encode_image(image)
        sequence = captured[-1].permute(1, 0, 2)
        expected = model.visual.ln_post(sequence[:, 1:]) @ model.visual.proj
        actual = native_clip_patches(model, image)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2, 196, 32)
        torch.testing.assert_close(model.visual.ln_post(sequence[:, 0]) @ model.visual.proj, global_feature)
    handle.remove()
