"""CPU regressions for remainder masking in the original full-caption positions."""
import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import longclip
from model.model_longclip import CLIP
from model.nested_semantic_mask import NestedSemanticMask
from tests.test_nested_semantic_mask import TinyCLIP, global_reference
from train import nested_semantic_data as data


CONTEXT = 248
NEAR_LIMIT_VISIBLE = 'Start. ' + 'word ' * 243 + 'end'
NEAR_LIMIT_CAPTION = NEAR_LIMIT_VISIBLE + '. ' + 'tail ' * 40
UNICODE_CAPTION = 'Café naïve 東京 👩🏽‍🚀. Straße coöperate. Ελληνικά 中文. Fin.'


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


def assert_positioned_suffix(masked, compact):
    """Locate the suffix backwards from EOT, independently of the masking helper."""
    full, remainder = masked['tokens_f'], masked['tokens_e']
    tokenizer = longclip._tokenizer
    eot_id = tokenizer.encoder['<|endoftext|>']
    eot_positions = torch.nonzero(full == eot_id).flatten()
    assert eot_positions.numel() == 1
    eot = int(eot_positions[0])
    suffix_ids = full.new_tensor(tokenizer.encode(compact['views'][2]))
    first = eot - len(suffix_ids)
    assert 1 < first < eot
    assert torch.equal(full[first:eot], suffix_ids)

    assert remainder.shape == full.shape == (CONTEXT,)
    assert remainder.dtype == full.dtype
    assert int(full[0]) == tokenizer.encoder['<|startoftext|>']
    assert remainder[0] == full[0]
    assert torch.count_nonzero(remainder[1:first]) == 0
    assert torch.equal(remainder[first:], full[first:])
    assert remainder[eot] == eot_id
    assert int(remainder.argmax()) == int(full.argmax()) == eot

    # Compare actual non-PAD positions and their IDs, not just decoded strings.
    expected_positions = torch.nonzero(full[:eot] != 0).flatten()
    expected_positions = expected_positions[expected_positions >= first]
    actual_positions = torch.nonzero(remainder[1:eot] != 0).flatten() + 1
    assert torch.equal(actual_positions, expected_positions)
    assert torch.equal(remainder[actual_positions], full[expected_positions])
    assert int(actual_positions[0]) == first
    assert torch.count_nonzero(remainder[eot + 1:]) == 0
    compact_positions = torch.nonzero(compact['tokens_e'][1:] != 0).flatten() + 1
    assert int(compact_positions[0]) == 1
    assert first > int(compact_positions[0])

    assert masked['prefix_token_boundary'] == first
    assert masked['full_eot_index'] == eot
    assert masked['suffix_first_index'] == first
    assert masked['compact_suffix_first_index'] == 1
    assert first == 1 + len(tokenizer.encode(masked['views'][1] + '. '))
    return first, eot


@pytest.mark.parametrize('sampling_mode', ['fixed_first', 'random_k'])
@pytest.mark.parametrize('sampling_seed', [0, 73])
def test_modes_preserve_full_prefix_and_split_across_epochs_and_ids(sampling_mode, sampling_seed):
    caption = 'A room. A table. A window. A cat. A lamp. The end.'
    original = data.text_views(caption)
    observed = set()
    for epoch in (0, 1, 7):
        for sample_id in (0, 1, 7, 1000, 1009, 123456):
            kwargs = dict(sampling_mode=sampling_mode, sampling_seed=sampling_seed,
                          epoch=epoch, sample_id=sample_id)
            default = data.sampled_text_views(caption, **kwargs)
            compact = data.sampled_text_views(caption, remainder_mode='compact', **kwargs)
            masked = data.sampled_text_views(caption, remainder_mode='prefix_pad', **kwargs)
            assert_fields_equal(default, compact, compact.keys())
            assert_fields_equal(masked, compact, (
                'tokens_f', 'tokens_o', 'n', 'K', 'valid', 'reason', 'views',
                'reference_views', 'reference_tokens_o', 'reference_tokens_e'))
            assert torch.equal(masked['tokens_f'], original['tokens_f'])
            assert torch.equal(masked['tokens_o'], longclip.tokenize(
                masked['views'][1], context_length=CONTEXT)[0])
            assert masked['untruncated_lengths'][:2] == compact['untruncated_lengths'][:2]
            assert_positioned_suffix(masked, compact)
            observed.add(masked['K'])
    if sampling_mode == 'random_k':
        assert len(observed) > 1
    else:
        assert observed == {1}


@pytest.mark.parametrize('split', ['first', 'last'])
@pytest.mark.parametrize('caption', [
    'First. Second. Third. Last.',
    'A (red) object?. (Signed), “yes”: façade. Done.',
    UNICODE_CAPTION,
])
def test_k_extremes_preserve_bpe_suffix_positions(monkeypatch, split, caption):
    n = len(data.text_views(caption)['views'][0].split('. '))
    k = 1 if split == 'first' else n - 1
    monkeypatch.setattr(data, 'sample_split_k', lambda *args: k)
    compact = data.sampled_text_views(caption, 'random_k')
    masked = data.sampled_text_views(caption, 'random_k', remainder_mode='prefix_pad')
    assert masked['valid'] and masked['K'] == k and masked['n'] == n
    assert len(masked['views'][1].split('. ')) == k
    assert len(masked['views'][2].split('. ')) == n - k
    assert_positioned_suffix(masked, compact)
    # Also exercise the public helper and make mutation of its input observable.
    original = compact['tokens_f'].clone()
    remainder, boundary = data.mask_remainder_prefix(compact['tokens_f'], compact['views'][1])
    assert torch.equal(compact['tokens_f'], original)
    assert remainder.data_ptr() != compact['tokens_f'].data_ptr()
    assert torch.equal(remainder, masked['tokens_e'])
    assert boundary == masked['prefix_token_boundary']


def test_separator_belongs_to_masked_prefix_even_when_punctuation_merges():
    compact = data.sampled_text_views('Wait?. Tail.')
    masked = data.sampled_text_views('Wait?. Tail.', remainder_mode='prefix_pad')
    first, _ = assert_positioned_suffix(masked, compact)
    tokenizer = longclip._tokenizer
    prefix_ids = tokenizer.encode('Wait?')
    delimited_ids = tokenizer.encode('Wait?. ')
    # The question mark and separator form a different BPE token together.
    assert prefix_ids != delimited_ids
    assert torch.equal(compact['tokens_f'][1:first], compact['tokens_f'].new_tensor(delimited_ids))
    assert compact['tokens_f'][first - 1] == delimited_ids[-1]
    assert masked['tokens_e'][first - 1] == 0
    assert masked['tokens_e'][first] == tokenizer.encode('Tail.')[0]


def test_longest_visible_caption_at_full_context_limit():
    tokenizer = longclip._tokenizer
    assert len(tokenizer.encode(NEAR_LIMIT_VISIBLE)) + 2 == CONTEXT
    assert len(tokenizer.encode(NEAR_LIMIT_CAPTION)) + 2 > CONTEXT
    baseline = data.text_views(NEAR_LIMIT_CAPTION)
    assert baseline['views'][0] == NEAR_LIMIT_VISIBLE
    assert 'tail' not in baseline['views'][0]
    for epoch in (0, 5):
        for sample_id in (1000, 1001):
            kwargs = dict(sampling_mode='random_k', epoch=epoch, sample_id=sample_id)
            compact = data.sampled_text_views(NEAR_LIMIT_CAPTION, **kwargs)
            masked = data.sampled_text_views(NEAR_LIMIT_CAPTION, remainder_mode='prefix_pad', **kwargs)
            assert masked['valid'] and (masked['n'], masked['K']) == (2, 1)
            assert torch.equal(masked['tokens_f'], baseline['tokens_f'])
            _, eot = assert_positioned_suffix(masked, compact)
            assert eot == CONTEXT - 1


@pytest.mark.parametrize('sampling_mode', ['fixed_first', 'random_k'])
@pytest.mark.parametrize('caption,reason,n', [
    ('Only one sentence.', 'single_visible_segment', 1),
    ('word ' * 246, 'single_visible_segment', 1),
    ('Short. ' + 'word ' * 247, 'single_visible_segment', 1),
    ('word ' * 247 + '. Short tail.', 'first_segment_overlong', 0),
])
def test_invalid_samples_keep_baseline_empty_remainder(sampling_mode, caption, reason, n):
    baseline = data.text_views(caption)
    empty = longclip.tokenize('', context_length=CONTEXT)[0]
    for epoch, sample_id in ((0, 1000), (9, 12345)):
        masked = data.sampled_text_views(caption, sampling_mode, epoch=epoch,
                                         sample_id=sample_id, remainder_mode='prefix_pad')
        assert_fields_equal(masked, baseline, baseline.keys())
        assert not masked['valid'] and masked['reason'] == reason
        assert (masked['n'], masked['K']) == (n, 0)
        assert torch.equal(masked['tokens_e'], empty)
        assert int(masked['tokens_e'].argmax()) == 1


@pytest.mark.parametrize('full,prefix', [
    ('A cat. A table.', 'A dog'),
    ('A cat? A table.', 'A cat'),
    ('Café. 東京.', 'Cafe'),
    ('First. Last.', 'First. Last.'),
])
def test_helper_rejects_real_prefix_mismatch_without_mutation(full, prefix):
    tokens = longclip.tokenize(full, context_length=CONTEXT)[0]
    before = tokens.clone()
    with pytest.raises(ValueError):
        data.mask_remainder_prefix(tokens, prefix)
    assert torch.equal(tokens, before)


def test_helper_rejects_corrupted_prefix_token():
    tokens = longclip.tokenize('A cat. A table.', context_length=CONTEXT)[0]
    tokens[2] = longclip._tokenizer.encode('dog')[0]
    before = tokens.clone()
    with pytest.raises(ValueError):
        data.mask_remainder_prefix(tokens, 'A cat')
    assert torch.equal(tokens, before)


def test_real_clip_cpu_encoding_uses_unchanged_eot_and_is_finite():
    torch.manual_seed(512)
    clip = CLIP(embed_dim=512, image_resolution=16, vision_layers=1,
                vision_width=64, vision_patch_size=16, context_length=CONTEXT,
                vocab_size=49408, transformer_width=64, transformer_heads=1,
                transformer_layers=1, load_from_clip=False).cpu().eval()
    # Production normally fills the residual positional table from a checkpoint.
    nn.init.normal_(clip.positional_embedding_res, std=.01)
    views = [data.sampled_text_views(caption, remainder_mode='prefix_pad')
             for caption in ('First. Second. Last.', UNICODE_CAPTION, NEAR_LIMIT_CAPTION)]
    full = torch.stack([view['tokens_f'] for view in views])
    remainder = torch.stack([view['tokens_e'] for view in views])
    eot_id = longclip._tokenizer.encoder['<|endoftext|>']
    assert torch.equal(full.argmax(-1), remainder.argmax(-1))
    assert torch.all(remainder[torch.arange(len(views)), full.argmax(-1)] == eot_id)
    with torch.no_grad():
        pooled, hidden = clip.encode_text(remainder, return_full=True)
        expected = hidden[torch.arange(len(views)), full.argmax(-1)] @ clip.text_projection
        full_features = clip.encode_text(full)
    assert pooled.shape == full_features.shape == (len(views), 512)
    assert hidden.shape == (len(views), CONTEXT, 64)
    assert pooled.device.type == hidden.device.type == 'cpu'
    assert all(torch.isfinite(value).all() for value in (pooled, hidden, full_features))
    torch.testing.assert_close(pooled, expected)


@pytest.mark.parametrize('valid_count', [0, 1])
def test_global_zero_or_one_valid_keeps_full_only_loss_and_gradients(valid_count):
    captions = ['Single.', 'word ' * 247, 'Short. ' + 'word ' * 247]
    if valid_count:
        captions[0] = 'A room. A table. A window. A cat.'
    compact = [data.sampled_text_views(caption, 'random_k', sample_id=i)
               for i, caption in enumerate(captions)]
    masked = [data.sampled_text_views(caption, 'random_k', sample_id=i,
                                    remainder_mode='prefix_pad')
              for i, caption in enumerate(captions)]
    valid = torch.tensor([view['valid'] for view in compact])
    assert int(valid.sum()) == valid_count
    if valid_count:
        assert not torch.equal(compact[0]['tokens_e'], masked[0]['tokens_e'])
    torch.manual_seed(43)
    clip = TinyCLIP()
    clip.token_embedding = nn.Embedding(49408, 8)
    images = torch.randn(len(captions), 8)
    tokens = [torch.stack([view[key] for view in compact])
              for key in ('tokens_f', 'tokens_o', 'tokens_e')]
    with torch.no_grad():
        expected_loss = global_reference(clip, images, tokens, valid, 'A3', 200)
    results = []
    for local_views in (compact, masked):
        module = NestedSemanticMask(copy.deepcopy(clip), arm='A3', checkpoint_encoders=False)
        local_tokens = [torch.stack([view[key] for view in local_views])
                        for key in ('tokens_f', 'tokens_o', 'tokens_e')]
        loss, logs = module(images, *local_tokens, valid, completed=200)
        assert torch.isfinite(loss) and logs['nonfinite'] == 0
        assert logs['valid_global'] == valid_count and logs['inc_weight'] == 0
        assert logs['O_candidates'] == logs['E_candidates'] == 0
        torch.testing.assert_close(loss, expected_loss, atol=3e-5, rtol=3e-6)
        loss.backward()
        gradients = {name: None if param.grad is None else param.grad.clone()
                     for name, param in module.named_parameters()}
        assert all(torch.isfinite(grad).all() for grad in gradients.values() if grad is not None)
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
    captions += ['Single.', 'word ' * 247 + '. Tail.']
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
    return index


def collect_batches(index, root, epoch, workers, remainder_mode=None):
    kwargs = dict(sampling_mode='random_k', sampling_seed=29)
    if remainder_mode is not None:
        kwargs['remainder_mode'] = remainder_mode
    dataset = data.NestedDataset(index, root, **kwargs)
    dataset.set_epoch(epoch)
    loader_kwargs = dict(batch_size=5, num_workers=workers, collate_fn=data.collate,
                         persistent_workers=False, generator=torch.Generator().manual_seed(11))
    if workers:
        loader_kwargs.update(multiprocessing_context='spawn', timeout=60)
    return list(DataLoader(dataset, **loader_kwargs))


def test_spawn_propagates_remainder_mode_and_epoch_preserving_images_full_prefix_and_k(tmp_path):
    index = make_index(tmp_path)
    epoch_splits = []
    for epoch in (0, 3):
        old = collect_batches(index, tmp_path, epoch, workers=2)
        new = collect_batches(index, tmp_path, epoch, workers=2, remainder_mode='prefix_pad')
        serial = collect_batches(index, tmp_path, epoch, workers=0, remainder_mode='prefix_pad')
        assert len(old) == len(new) == len(serial) == 2
        splits, sample_ids = [], []
        for baseline, masked, direct in zip(old, new, serial):
            assert_fields_equal(masked, baseline, (
                'image', 'image_id', 'sample_id', 'tokens_f', 'tokens_o', 'n', 'K',
                'valid', 'reason', 'views', 'reference_views',
                'reference_tokens_o', 'reference_tokens_e'))
            assert_fields_equal(masked, direct, masked.keys())
            sample_ids.extend(masked['sample_id'].tolist())
            splits.extend(masked['K'].tolist())
            assert all(tensor.device.type == 'cpu' for tensor in (
                masked['image'], masked['tokens_f'], masked['tokens_o'], masked['tokens_e']))
            for i, enabled in enumerate(masked['valid'].tolist()):
                if enabled:
                    # Recheck positions after collating and crossing the process boundary.
                    fields = ('tokens_f', 'tokens_e', 'views', 'prefix_token_boundary',
                              'full_eot_index', 'suffix_first_index', 'compact_suffix_first_index')
                    sample = {field: masked[field][i] for field in fields}
                    compact = {field: baseline[field][i] for field in ('tokens_e', 'views')}
                    assert_positioned_suffix(sample, compact)
                else:
                    assert torch.equal(masked['tokens_e'][i], baseline['tokens_e'][i])
        assert sample_ids == list(range(1000, 1010))
        epoch_splits.append(splits)
    assert any(a != b for a, b in zip(*epoch_splits))
