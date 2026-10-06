"""One-variable AllDetail sampling, fallback and process-RNG invariants."""
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from model import longclip
from train import nested_semantic_data as data


@pytest.mark.parametrize('caption', [
    'First. Second.', 'First. Second. Third. Fourth. Fifth.',
    'A street. A bus. A shop. ' + 'invisible '*300,
    'Only one.', 'word '*300, 'word '*300 + '. A cat.',
])
def test_only_detail_changes(caption):
    for epoch in (0, 1, 3):
        for sid in (0, 7, 1024):
            baseline = data.sampled_text_views(caption, 'summary_random_detail', 0, epoch, sid)
            actual = data.sampled_text_views(caption, 'summary_all_detail', 0, epoch, sid)
            for k in ('tokens_f', 'tokens_o', 'reference_tokens_o', 'reference_tokens_e'):
                assert torch.equal(actual[k], baseline[k])
            for k in ('valid', 'n', 'detail_pool_size', 'reference_views'):
                assert actual[k] == baseline[k]
            assert actual['views'][:2] == baseline['views'][:2]
            if actual['valid']:
                sentences = actual['views'][0].split('. ')
                assert actual['detail_indices'] == list(range(1, len(sentences)))
                assert actual['views'][2] == '. '.join(sentences[1:])
                assert torch.equal(actual['tokens_e'], longclip.tokenize([actual['views'][2]], truncate=False)[0])
                assert actual['K'] == actual['detail_pool_size']
            else:
                assert actual.keys() == baseline.keys()
                for k in actual:
                    assert torch.equal(actual[k], baseline[k]) if torch.is_tensor(actual[k]) else actual[k] == baseline[k]


def test_no_random_draw_and_no_unpacked_trailing_sentences(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('AllDetail must not draw random subsets')
    monkeypatch.setattr(data, 'sample_detail_indices', forbidden)
    monkeypatch.setattr(data, 'sample_contiguous_detail_indices', forbidden)
    value = data.sampled_text_views('First. Second. Third. ' + 'unpacked '*300, 'summary_all_detail')
    assert 'unpacked' not in value['views'][0] and 'unpacked' not in value['views'][2]


def test_process_rng_unchanged():
    py, np_state, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for sid in range(32):
        data.sampled_text_views('First. Second. Third. Fourth.', 'summary_all_detail', sample_id=sid)
    assert py == random.getstate() and torch.equal(ts, torch.get_rng_state())
    current = np.random.get_state()
    assert current[0] == np_state[0] and np.array_equal(current[1], np_state[1]) and current[2:] == np_state[2:]


def test_config_single_variable():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / 'recovery/configs/summary02_local500.json').read_text())
    new = json.loads((root / 'recovery/configs/summary02_all_detail500.json').read_text())
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {'sampling_mode'}
    assert new['sampling_mode'] == 'summary_all_detail'


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_diagnostics_with_CPU_references_and_device_tokens(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    samples = []
    for sid, caption in enumerate(('First. Second. Third. Fourth.', 'Only one.')):
        v = data.sampled_text_views(caption, 'summary_all_detail', sample_id=sid)
        samples.append(dict(image=torch.zeros(3, 2, 2), image_id=sid, sample_id=sid, **v))
    batch = data.collate(samples)
    for k in ('tokens_f', 'tokens_o', 'tokens_e', 'valid'):
        batch[k] = batch[k].to(device)
    result = data.sampling_diagnostics(batch)
    assert result['summary_baseline_exact'] and result['all_detail_selection_complete']
    assert result['detail_full_token_coverage']['samples'] == 1
