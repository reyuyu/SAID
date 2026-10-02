"""CPU contracts for stateless, compact sentence subsets of the old remainder."""
from collections import Counter
import copy
import hashlib
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import longclip
from model.nested_semantic_mask import NestedSemanticMask
from tests.test_nested_semantic_mask import TinyCLIP, global_reference
from train import nested_semantic_data as data


CONTEXT = 248
CAPTION = 'Prefix zero. Prefix one. Suffix zero. Suffix one. Suffix two. Suffix three. Suffix four.'
UNICODE_CAPTION = 'Café naïve 東京 👩🏽‍🚀. Straße coöperate. Ελληνικά 中文. Fin.'
NEAR_LIMIT_VISIBLE = 'Start. ' + 'word ' * 243 + 'end'
NEAR_LIMIT_CAPTION = NEAR_LIMIT_VISIBLE + '. ' + 'tail ' * 40
DROP_FIELDS = (
    'drop_m', 'drop_q', 'drop_selected_indices_before_sort',
    'drop_selected_indices_after_sort', 'drop_same_old_r', 'old_remainder_text',
)
PRESERVED_FIELDS = (
    'tokens_f', 'tokens_o', 'n', 'K', 'valid', 'reason',
    'reference_views', 'reference_tokens_o', 'reference_tokens_e',
)


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def assert_fields_equal(actual, expected, fields):
    for field in fields:
        if torch.is_tensor(expected[field]):
            assert torch.equal(actual[field], expected[field]), field
        else:
            assert actual[field] == expected[field], field


def assert_subset(m, subset):
    q, before, after = subset
    before, after = list(before), list(after)
    assert 1 <= q <= m
    assert len(before) == len(after) == q
    assert len(set(before)) == q
    assert all(0 <= index < m for index in before)
    assert after == sorted(before)
    return q, before, after


def assert_drop_view(dropped, compact):
    assert_fields_equal(dropped, compact, PRESERVED_FIELDS)
    assert dropped['valid']
    assert dropped['views'][:2] == compact['views'][:2]
    assert dropped['untruncated_lengths'][:2] == compact['untruncated_lengths'][:2]
    parts = compact['views'][0].split('. ')
    suffix = parts[compact['K']:]
    assert compact['views'][2] == '. '.join(suffix)
    assert dropped['old_remainder_text'] == compact['views'][2]
    assert dropped['drop_m'] == len(suffix) == compact['n'] - compact['K']
    q, _, selected = assert_subset(len(suffix), (
        dropped['drop_q'], dropped['drop_selected_indices_before_sort'],
        dropped['drop_selected_indices_after_sort'],
    ))
    selected_text = '. '.join(suffix[index] for index in selected)
    assert selected_text and dropped['views'][2] == selected_text
    assert dropped['drop_same_old_r'] is (q == len(suffix))
    assert dropped['untruncated_lengths'] == [
        len(longclip._tokenizer.encode(text)) + 2 for text in dropped['views']
    ]
    assert max(dropped['untruncated_lengths']) <= CONTEXT
    expected = longclip.tokenize(dropped['views'], context_length=CONTEXT, truncate=False)
    actual = torch.stack([dropped[key] for key in ('tokens_f', 'tokens_o', 'tokens_e')])
    assert torch.equal(actual, expected)
    assert actual.device.type == 'cpu'
    remainder = dropped['tokens_e']
    eot = int(remainder.argmax())
    assert eot == dropped['untruncated_lengths'][2] - 1
    assert int(remainder[0]) == longclip._tokenizer.encoder['<|startoftext|>']
    assert int(remainder[eot]) == longclip._tokenizer.encoder['<|endoftext|>']
    assert torch.all(remainder[1:eot] != 0), 'Sentence drop must compact, without internal PAD'
    assert torch.count_nonzero(remainder[eot + 1:]) == 0
    if q == len(suffix):
        assert torch.equal(remainder, compact['tokens_e'])
        assert dropped['views'] == compact['views']
        assert dropped['untruncated_lengths'] == compact['untruncated_lengths']


def test_helper_uses_independent_versioned_hash_seed(monkeypatch):
    real_random = random.Random
    captured = []

    def local_random(seed):
        captured.append(seed)
        return real_random(seed)

    monkeypatch.setattr(data.random, 'Random', local_random)
    for seed, epoch, sample_id in ((0, 0, 1000), (73, 9, 123456), (-7, 2, 0)):
        captured.clear()
        assert_subset(7, data.sample_sentence_subset(7, seed, epoch, sample_id))
        material = f'{seed}:{epoch}:{sample_id}:r_sentence_drop_v1'.encode('utf-8')
        expected = int.from_bytes(hashlib.sha256(material).digest(), 'big')
        assert captured == [expected]
        captured.clear()
        data.sample_split_k(8, seed, epoch, sample_id)
        assert captured == [int.from_bytes(hashlib.sha256(
            f'{seed}:{epoch}:{sample_id}'.encode('utf-8')).digest(), 'big')]
        assert captured[0] != expected


@pytest.mark.parametrize('m', [1, 2, 5, 12])
def test_helper_resume_is_independent_of_call_order_and_seed_epoch_sample(m):
    keys = [(seed, epoch, sample_id)
            for seed in (0, 73) for epoch in (0, 3) for sample_id in range(32)]
    first = {key: assert_subset(m, data.sample_sentence_subset(m, *key)) for key in keys}
    # Other draws and reversed access order cannot advance this local stream.
    for sample_id in range(32):
        data.sample_sentence_subset(m + 1, 99, 8, sample_id)
        data.sample_split_k(9, 99, 8, sample_id)
    resumed = {key: assert_subset(m, data.sample_sentence_subset(m, *key))
               for key in reversed(keys)}
    assert first == resumed
    if m == 1:
        assert set((q, tuple(before), tuple(after)) for q, before, after in first.values()) == {
            (1, (0,), (0,))
        }
    else:
        baseline = [first[0, 0, i] for i in range(32)]
        for seed, epoch in ((73, 0), (0, 3)):
            assert baseline != [first[seed, epoch, i] for i in range(32)]
        assert len({(q, tuple(before)) for q, before, _ in baseline}) > 1


def test_q_inclusive_range_full_reach_and_reported_histograms(capsys):
    histograms = {}
    saw_unsorted = False
    for m in (1, 2, 5, 8):
        counts = Counter()
        for sample_id in range(4096):
            q, before, after = assert_subset(m, data.sample_sentence_subset(m, 29, 3, sample_id))
            counts[q] += 1
            saw_unsorted |= before != after
            if q == m:
                assert after == list(range(m))
        assert set(counts) == set(range(1, m + 1))
        assert sum(counts.values()) == 4096
        histograms[m] = dict(sorted(counts.items()))
    assert saw_unsorted
    # Finite-sample counts are evidence, not a requirement for perfect balance.
    with capsys.disabled():
        print('Sentence-drop q histogram:', json.dumps(histograms, sort_keys=True))


@pytest.mark.parametrize('sampling_mode', ['fixed_first', 'random_k'])
@pytest.mark.parametrize('sampling_seed', [0, 73])
def test_preserves_full_prefix_split_and_all_global_rng_states(sampling_mode, sampling_seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    observed = set()
    for epoch in (0, 1, 7):
        for sample_id in (0, 1, 7, 1000, 1009, 123456):
            kwargs = dict(sampling_mode=sampling_mode, sampling_seed=sampling_seed,
                          epoch=epoch, sample_id=sample_id)
            compact = data.sampled_text_views(CAPTION, remainder_mode='compact', **kwargs)
            default = data.sampled_text_views(CAPTION, **kwargs)
            assert_fields_equal(default, compact, compact.keys())
            dropped = data.sampled_text_views(CAPTION, remainder_mode='sentence_drop', **kwargs)
            assert_drop_view(dropped, compact)
            q, before, after = data.sample_sentence_subset(
                compact['n'] - compact['K'], sampling_seed, epoch, sample_id)
            assert dropped['drop_q'] == q
            assert list(dropped['drop_selected_indices_before_sort']) == list(before)
            assert list(dropped['drop_selected_indices_after_sort']) == list(after)
            observed.add(compact['K'])
    assert python_state == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_state[0] == current_numpy[0]
    assert np.array_equal(numpy_state[1], current_numpy[1])
    assert numpy_state[2:] == current_numpy[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert len(observed) > 1 if sampling_mode == 'random_k' else observed == {1}


@pytest.mark.parametrize('before', [(0,), (4,), (3, 0, 2), (4, 0, 3, 1, 2)])
def test_forced_local_suffix_indices_are_sorted_whole_sentences(monkeypatch, before):
    monkeypatch.setattr(data, 'sample_split_k', lambda *args: 2)
    calls = []

    def forced_subset(m, sampling_seed, epoch, sample_id):
        calls.append((m, sampling_seed, epoch, sample_id))
        return len(before), list(before), sorted(before)

    monkeypatch.setattr(data, 'sample_sentence_subset', forced_subset)
    kwargs = dict(sampling_mode='random_k', sampling_seed=73, epoch=7, sample_id=1009)
    compact = data.sampled_text_views(CAPTION, remainder_mode='compact', **kwargs)
    dropped = data.sampled_text_views(CAPTION, remainder_mode='sentence_drop', **kwargs)
    assert calls == [(5, 73, 7, 1009)]
    assert_drop_view(dropped, compact)
    assert dropped['drop_q'] == len(before)
    assert list(dropped['drop_selected_indices_before_sort']) == list(before)
    assert list(dropped['drop_selected_indices_after_sort']) == sorted(before)
    assert 'Prefix' not in dropped['views'][2]
    if list(before) != sorted(before):
        wrong_order = '. '.join(compact['views'][2].split('. ')[i] for i in before)
        assert dropped['views'][2] != wrong_order


@pytest.mark.parametrize('sampling_mode', ['fixed_first', 'random_k'])
def test_single_suffix_is_exact_old_remainder(sampling_mode):
    for epoch, sample_id in ((0, 1000), (9, 12345)):
        kwargs = dict(sampling_mode=sampling_mode, sampling_seed=29,
                      epoch=epoch, sample_id=sample_id)
        compact = data.sampled_text_views('First. Last.', **kwargs)
        dropped = data.sampled_text_views('First. Last.', remainder_mode='sentence_drop', **kwargs)
        assert_drop_view(dropped, compact)
        assert dropped['drop_m'] == dropped['drop_q'] == 1
        assert list(dropped['drop_selected_indices_before_sort']) == [0]
        assert list(dropped['drop_selected_indices_after_sort']) == [0]
        assert dropped['drop_same_old_r'] is True


@pytest.mark.parametrize('caption', [
    'First. Second. Third. Last.',
    ' First. \nSecond. Last. ',
    'Wait?. A (red) object?. (Signed), “yes”: façade. Done.',
    UNICODE_CAPTION,
])
def test_actual_selected_text_uses_normal_tokenization(monkeypatch, caption):
    monkeypatch.setattr(data, 'sample_split_k', lambda *args: 1)

    def endpoints(m, *args):
        before = [m - 1, 0] if m > 1 else [0]
        return len(before), before, sorted(before)

    monkeypatch.setattr(data, 'sample_sentence_subset', endpoints)
    compact = data.sampled_text_views(caption, 'random_k')
    dropped = data.sampled_text_views(caption, 'random_k', remainder_mode='sentence_drop')
    assert_drop_view(dropped, compact)


@pytest.mark.parametrize('visible', [
    NEAR_LIMIT_VISIBLE, 'Start. Second. ' + 'word ' * 241 + 'end',
])
@pytest.mark.parametrize('full_subset', [False, True])
def test_visible_full_at_248_tokens_excludes_overflow_tail(monkeypatch, visible, full_subset):
    monkeypatch.setattr(data, 'sample_split_k', lambda *args: 1)

    def forced_subset(m, *args):
        before = list(reversed(range(m))) if full_subset else [m - 1]
        return len(before), before, sorted(before)

    monkeypatch.setattr(data, 'sample_sentence_subset', forced_subset)
    caption = visible + '. ' + 'tail ' * 40
    assert len(longclip._tokenizer.encode(visible)) + 2 == CONTEXT
    assert len(longclip._tokenizer.encode(caption)) + 2 > CONTEXT
    original = data.text_views(caption)
    assert original['views'][0] == visible
    assert 'tail' not in original['views'][0]
    for epoch, sample_id in ((0, 1000), (5, 1001)):
        kwargs = dict(sampling_mode='random_k', epoch=epoch, sample_id=sample_id)
        compact = data.sampled_text_views(caption, **kwargs)
        dropped = data.sampled_text_views(caption, remainder_mode='sentence_drop', **kwargs)
        assert_drop_view(dropped, compact)
        m = len(visible.split('. ')) - 1
        assert (dropped['n'], dropped['K'], dropped['drop_m'], dropped['drop_q']) == (
            m + 1, 1, m, m if full_subset else 1,
        )
        assert int(dropped['tokens_f'].argmax()) == CONTEXT - 1
        assert torch.equal(dropped['tokens_f'], original['tokens_f'])


@pytest.mark.parametrize('sampling_mode', ['fixed_first', 'random_k'])
@pytest.mark.parametrize('caption,reason,n', [
    ('Only one sentence.', 'single_visible_segment', 1),
    ('word ' * 246, 'single_visible_segment', 1),
    ('Short. ' + 'word ' * 247, 'single_visible_segment', 1),
    ('word ' * 247 + '. Short tail.', 'first_segment_overlong', 0),
])
def test_invalid_fallback_keeps_all_old_fields_and_never_draws(monkeypatch, sampling_mode, caption, reason, n):
    def unexpected_draw(*args):
        pytest.fail('Invalid captions must bypass sentence-drop sampling')

    monkeypatch.setattr(data, 'sample_sentence_subset', unexpected_draw)
    original = data.text_views(caption)
    for epoch, sample_id in ((0, 1000), (9, 12345)):
        kwargs = dict(sampling_mode=sampling_mode, epoch=epoch, sample_id=sample_id)
        compact = data.sampled_text_views(caption, **kwargs)
        dropped = data.sampled_text_views(caption, remainder_mode='sentence_drop', **kwargs)
        assert_fields_equal(dropped, compact, compact.keys())
        assert_fields_equal(dropped, original, original.keys())
        assert not dropped['valid'] and dropped['reason'] == reason
        assert (dropped['n'], dropped['K']) == (n, 0)
        assert all(field not in dropped for field in DROP_FIELDS)
        assert torch.equal(dropped['tokens_e'], longclip.tokenize('', context_length=CONTEXT)[0])


def test_empty_caption_keeps_old_error():
    with pytest.raises(ValueError, match='empty'):
        data.sampled_text_views(' \n ', 'random_k', remainder_mode='sentence_drop')


@pytest.mark.parametrize('valid_count', [0, 1])
def test_zero_or_one_valid_keeps_full_only_loss_and_gradients(monkeypatch, valid_count):
    monkeypatch.setattr(data, 'sample_split_k', lambda *args: 1)
    monkeypatch.setattr(data, 'sample_sentence_subset', lambda m, *args: (1, [m - 1], [m - 1]))
    captions = ['Single.', 'word ' * 247, 'Short. ' + 'word ' * 247]
    if valid_count:
        captions[0] = 'A room. A table. A window. A cat.'
    compact = [data.sampled_text_views(caption, 'random_k', sample_id=i)
               for i, caption in enumerate(captions)]
    dropped = [data.sampled_text_views(caption, 'random_k', sample_id=i,
                                      remainder_mode='sentence_drop')
               for i, caption in enumerate(captions)]
    valid = torch.tensor([view['valid'] for view in compact])
    assert int(valid.sum()) == valid_count
    if valid_count:
        assert_drop_view(dropped[0], compact[0])
        assert not torch.equal(compact[0]['tokens_e'], dropped[0]['tokens_e'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        clip = TinyCLIP()
        clip.token_embedding = nn.Embedding(len(longclip._tokenizer.encoder), 8)
        images = torch.randn(len(captions), 8)
    tokens = [torch.stack([view[key] for view in compact])
              for key in ('tokens_f', 'tokens_o', 'tokens_e')]
    reference = copy.deepcopy(clip)
    expected_loss = global_reference(reference, images, tokens, valid, 'A3', 200)
    expected_loss.backward()
    expected_gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                          for name, parameter in reference.named_parameters()}
    results = []
    for local_views in (compact, dropped):
        module = NestedSemanticMask(copy.deepcopy(clip), arm='A3', checkpoint_encoders=False)
        local_tokens = [torch.stack([view[key] for view in local_views])
                        for key in ('tokens_f', 'tokens_o', 'tokens_e')]
        loss, logs = module(images, *local_tokens, valid, completed=200)
        assert loss.device.type == 'cpu' and torch.isfinite(loss)
        assert logs['nonfinite'] == 0 and logs['valid_global'] == valid_count
        assert logs['inc_weight'] == logs['O_candidates'] == logs['E_candidates'] == 0
        torch.testing.assert_close(loss, expected_loss, atol=3e-5, rtol=3e-6)
        loss.backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                     for name, parameter in module.clip.named_parameters()}
        for name, gradient in gradients.items():
            expected = expected_gradients[name]
            assert (gradient is None) == (expected is None), name
            if gradient is not None:
                assert torch.isfinite(gradient).all()
                torch.testing.assert_close(gradient, expected, atol=3e-4, rtol=3e-5, msg=name)
        results.append((loss.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for name, gradient in results[0][1].items():
        other = results[1][1][name]
        assert (gradient is None) == (other is None), name
        if gradient is not None:
            torch.testing.assert_close(gradient, other, atol=0, rtol=0, msg=name)


def make_index(root):
    index = root / 'index'
    index.mkdir()
    captions = [f'Item {i}. Two. Three. Four. Five. Last.' for i in range(8)]
    captions += [NEAR_LIMIT_CAPTION, 'Single.', 'word ' * 247 + '. Tail.',
                 'Short. ' + 'word ' * 247]
    offsets = [0]
    with (index / 'records.jsonl').open('wb') as records:
        for i, caption in enumerate(captions):
            filename = f'image_{i}.png'
            Image.new('RGB', (35 + i, 40), (20 + i, 40, 60)).save(root / filename)
            row = json.dumps({'image': filename, 'caption': caption}).encode() + b'\n'
            records.write(row)
            offsets.append(offsets[-1] + len(row))
    np.save(index / 'offsets.npy', np.asarray(offsets, dtype=np.int64))
    (index / 'metadata.json').write_text(json.dumps({'training_records': len(captions)}))
    return index, captions


def collate_with_replay_samples(samples):
    # Keep per-item metadata across spawn even if collate only packs model inputs.
    return data.collate(samples), samples


def collect_workers(dataset, workers):
    kwargs = dict(batch_size=4, num_workers=workers, collate_fn=collate_with_replay_samples,
                  persistent_workers=False, generator=torch.Generator().manual_seed(11))
    if workers:
        kwargs.update(multiprocessing_context='spawn', timeout=60)
    return list(DataLoader(dataset, **kwargs))


def test_spawn_old_new_streams_match_images_full_prefix_k_and_subset_resume(tmp_path):
    index, captions = make_index(tmp_path)
    kwargs = dict(sampling_mode='random_k', sampling_seed=29)
    compact_dataset = data.NestedDataset(index, tmp_path, remainder_mode='compact', **kwargs)
    drop_dataset = data.NestedDataset(index, tmp_path, remainder_mode='sentence_drop', **kwargs)
    snapshots = {}
    for epoch in (0, 3):
        compact_dataset.set_epoch(epoch)
        drop_dataset.set_epoch(epoch)
        old = collect_workers(compact_dataset, workers=2)
        new = collect_workers(drop_dataset, workers=2)
        # A separate serial reader keeps parent mmaps out of spawn pickling.
        serial_dataset = data.NestedDataset(index, tmp_path, remainder_mode='sentence_drop', **kwargs)
        serial_dataset.set_epoch(epoch)
        serial = collect_workers(serial_dataset, workers=0)
        assert len(old) == len(new) == len(serial) == 3
        sample_ids, subsets = [], []
        for (old_batch, old_samples), (new_batch, new_samples), (serial_batch, serial_samples) in zip(old, new, serial):
            assert_fields_equal(new_batch, old_batch, (
                'image', 'image_id', 'sample_id', *PRESERVED_FIELDS,
            ))
            assert_fields_equal(new_batch, serial_batch, new_batch.keys())
            assert all(new_batch[key].device.type == 'cpu'
                       for key in ('image', 'tokens_f', 'tokens_o', 'tokens_e'))
            for baseline, dropped, direct in zip(old_samples, new_samples, serial_samples):
                assert_fields_equal(dropped, direct, dropped.keys())
                assert_fields_equal(dropped, baseline, ('image', 'image_id', 'sample_id'))
                sample_id = dropped['sample_id']
                sample_ids.append(sample_id)
                replay = data.sampled_text_views(captions[sample_id - 1000], epoch=epoch,
                                                  sample_id=sample_id, remainder_mode='sentence_drop', **kwargs)
                assert_fields_equal(dropped, replay, replay.keys())
                if dropped['valid']:
                    assert_drop_view(dropped, baseline)
                    subsets.append((sample_id, dropped['drop_q'],
                                    tuple(dropped['drop_selected_indices_before_sort']),
                                    tuple(dropped['drop_selected_indices_after_sort'])))
                else:
                    assert_fields_equal(dropped, baseline, baseline.keys())
                    if any(field in dropped for field in DROP_FIELDS):
                        assert dropped['drop_m'] == dropped['drop_q'] == 0
                        assert list(dropped['drop_selected_indices_before_sort']) == []
                        assert list(dropped['drop_selected_indices_after_sort']) == []
                        assert dropped['drop_same_old_r'] is True
                        assert dropped['old_remainder_text'] == baseline['views'][2]
        assert sample_ids == list(range(1000, 1000 + len(captions)))
        snapshots[epoch] = new, subsets
    assert snapshots[0][1] != snapshots[3][1]
    # Returning to a previous epoch reproduces the complete stream after resume.
    drop_dataset.set_epoch(0)
    resumed = collect_workers(drop_dataset, workers=0)
    for (expected_batch, expected_samples), (actual_batch, actual_samples) in zip(snapshots[0][0], resumed):
        assert_fields_equal(actual_batch, expected_batch, expected_batch.keys())
        for actual, expected in zip(actual_samples, expected_samples):
            assert_fields_equal(actual, expected, expected.keys())
