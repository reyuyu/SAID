"""Required sampling/Full-equivalence/RNG gates for a real training change."""
import ast
import json
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from model import longclip
from train import nested_semantic_data as data


@pytest.mark.parametrize('caption,parts', [
    ('A street. A bus.', ['A street', 'A bus.']),
    ('One. Two. Three. Four. Five.', ['One', 'Two', 'Three', 'Four', 'Five.'])])
def test_summary_detail_construction(caption, parts):
    v = data.sampled_text_views(caption, 'summary_detail', sample_id=111)
    old = data.sampled_text_views(caption, 'random_k', sample_id=111)
    assert v['valid'] and v['K'] == 1 and v['n'] == len(parts)
    assert v['views'][1:] == [parts[0], '. '.join(parts[1:])]
    assert v['views'][0] == old['views'][0]
    assert torch.equal(v['tokens_f'], old['tokens_f'])
    expected = longclip.tokenize(v['views'][1:], context_length=248, truncate=True)
    assert torch.equal(v['tokens_o'], expected[0])
    assert torch.equal(v['tokens_e'], expected[1])


def test_single_sentence_has_no_constructed_local_caption():
    v = data.sampled_text_views('Only a cat.', 'summary_detail')
    assert not v['valid'] and v['views'][1:] == [None, None]
    assert not v['tokens_o'].any() and not v['tokens_e'].any()
    assert v['K'] == 0


def test_all_raw_detail_even_when_full_visible_prefix_is_shorter():
    caption = 'A scene. A red bus. Several people. ' + 'extra '*300
    v = data.sampled_text_views(caption, 'summary_detail')
    old = data.sampled_text_views(caption, 'random_k')
    assert 'extra' not in v['views'][0] and 'extra' in v['views'][2]
    assert v['untruncated_lengths'][2] > 248
    assert torch.equal(v['tokens_f'], old['tokens_f'])
    assert int(v['tokens_e'][-1]) == longclip._tokenizer.encoder['<|endoftext|>']


def test_raw_two_sentence_validity_and_summary_truncation():
    caption = 'word '*300 + '. A small cat.'
    v = data.sampled_text_views(caption, 'summary_detail')
    old = data.sampled_text_views(caption, 'random_k')
    assert v['valid'] and v['n'] == 2 and v['untruncated_lengths'][1] > 248
    assert torch.equal(v['tokens_f'], old['tokens_f'])


def test_no_K_draw_no_subset_no_shuffle_and_compact_detail(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Unexpected random local-view draw')
    monkeypatch.setattr(data, 'sample_split_k', forbidden)
    monkeypatch.setattr(random, 'sample', forbidden)
    monkeypatch.setattr(random, 'shuffle', forbidden)
    v = data.sampled_text_views('First. Second. Third. Fourth.', 'summary_detail')
    assert v['views'][2] == 'Second. Third. Fourth.'
    encoded = longclip._tokenizer.encode(v['views'][2])
    assert v['tokens_e'][0].item() == longclip._tokenizer.encoder['<|startoftext|>']
    assert v['tokens_e'][1].item() == encoded[0]
    assert torch.equal(v['tokens_e'], longclip.tokenize([v['views'][2]], truncate=True)[0])


def test_global_python_numpy_torch_rng_untouched():
    py = random.getstate(); np_state = np.random.get_state(); ts = torch.get_rng_state().clone()
    for epoch in (0, 1, 3):
        for sample in range(32):
            data.sampled_text_views('First. Second. Third. Fourth.', 'summary_detail', epoch=epoch, sample_id=sample)
    assert random.getstate() == py
    now = np.random.get_state()
    assert now[0] == np_state[0] and np.array_equal(now[1], np_state[1]) and now[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), ts)


def test_baseline_functions_unchanged_at_ast_level():
    root = Path(__file__).resolve().parents[1]
    old = subprocess.check_output(['git', 'show', '14653c9:train/nested_semantic_data.py'], cwd=root, text=True)
    new = (root/'train/nested_semantic_data.py').read_text()
    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    assert a['text_views'] == b['text_views'] and a['sample_split_k'] == b['sample_split_k']


def test_dataset_image_rng_and_sample_id_match(tmp_path):
    image = np.random.default_rng(4).integers(0, 256, (320, 400, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path/'image.png')
    index = tmp_path/'index'; index.mkdir()
    raw = json.dumps({'image': 'image.png', 'caption': 'First. Second. Third. Fourth.'}).encode()+b'\n'
    (index/'records.jsonl').write_bytes(raw)
    np.save(index/'offsets.npy', np.array([0, len(raw)], dtype=np.int64))
    (index/'metadata.json').write_text(json.dumps({'training_records': 1}))
    datasets = [data.NestedDataset(index, tmp_path, mode, 0) for mode in ('random_k', 'summary_detail')]
    samples = []
    for ds in datasets:
        random.seed(10); np.random.seed(10); torch.manual_seed(10)
        samples.append(ds[0])
    assert samples[0]['sample_id'] == samples[1]['sample_id']
    assert torch.equal(samples[0]['image'], samples[1]['image'])
    assert torch.equal(samples[0]['tokens_f'], samples[1]['tokens_f'])
