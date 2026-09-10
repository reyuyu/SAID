"""Phase 2.9A tests: debiased suffix Unsaid (caption views, gate, retrieval, gradients)."""
import json
import math
import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import unsaid_core  # noqa: E402
from model.salu_model import SALUModel, contrastive_loss, gather_features_with_grad  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402
from sharegpt4v import caption_sentences, sample_unsaid_index, share4v_train_dataset  # noqa: E402

DIM = 8
PATCHES = 4
TOKENS = 6
BATCH = 4

CAPTIONS = [
    'A dog runs. It has brown fur. The grass is wet. A ball is near.',
    'Two cats sleep. One wears a collar. A window is open.',
    'A car is parked. The road is empty. Snow covers the roof.',
    'A bird sits. Its wings are blue. Leaves fill the frame.',
]


class _StubClip(nn.Module):
    """Differentiable stand-in for the wrapped CLIP model (text+image paths)."""

    def __init__(self, dim=DIM, patches=PATCHES, vocab=32):
        super().__init__()
        self.dim, self.patches = dim, patches
        self.text_projection = nn.Parameter(torch.zeros(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.global_proj = nn.Linear(dim, dim)
        self.patch_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(vocab, dim)          # token ids -> text features

    def _patches(self, images):
        base = self.patch_proj(images.flatten(1)[:, :self.dim])
        offsets = torch.linspace(-1.0, 1.0, self.patches).view(1, -1, 1)
        return base.unsqueeze(1) + offsets

    def encode_image(self, images):
        return self.global_proj(images.flatten(1)[:, :self.dim])

    def encode_image_with_patches(self, images, use_checkpoint=False):
        return self.encode_image(images), self._patches(images)

    def encode_text(self, tokens):
        one_hot = F.one_hot(tokens.clamp(0, self.text_proj.in_features - 1),
                            num_classes=self.text_proj.in_features).float()
        return self.text_proj(one_hot.mean(dim=1))


def build_model(**kwargs):
    kwargs.setdefault('pair_chunk_size', None)
    return SALUModel(_StubClip(), tau_said=0.07, **kwargs)


def make_batch(batch=BATCH):
    torch.manual_seed(3)
    return torch.randn(batch, 3, 4, 4), torch.randint(0, 20, (batch, TOKENS))


def identity_router(model):
    """Make the router's projections identity so hidden similarity is the geometry."""
    with torch.no_grad():
        model.said_router.q_proj.weight.copy_(torch.eye(DIM))
        model.said_router.q_proj.bias.zero_()
        model.said_router.k_proj.weight.copy_(torch.eye(DIM))
        model.said_router.k_proj.bias.zero_()


# --------------------------------------------------------------------------- #
# dataset: caption views + RNG compatibility
# --------------------------------------------------------------------------- #
def build_dataset(tmp_path, caption_views, seed=0):
    """Real images/captions after the 1000 held-out records the dataset skips."""
    root = tmp_path / 'images'
    root.mkdir(exist_ok=True)
    records = []
    for index, caption in enumerate(CAPTIONS):
        name = 'img%d.jpg' % index
        Image.fromarray(np.full((8, 8, 3), 20 + index * 30, dtype=np.uint8)).save(root / name)
        records.append({'image': name, 'conversations': [{'value': 'q'}, {'value': caption}]})
    filler = [{'image': 'img0.jpg', 'conversations': [{'value': 'q'}, {'value': CAPTIONS[0]}]}
              for _ in range(1000)]
    dataset_json = tmp_path / 'data.json'
    dataset_json.write_text(json.dumps(filler + records), encoding='utf-8')
    return share4v_train_dataset(data4v_root=str(tmp_path), json_name='data.json',
                                 image_root=str(root), preprocess=lambda image: torch.ones(3, 8, 8),
                                 caption_views=caption_views, suffix_seed=seed)


# the dataset already drops the first 1000 (held-out) records, so the real captions
# above are exposed at dataset indices 0..len(CAPTIONS)-1
REAL_INDICES = list(range(len(CAPTIONS)))


def test_caption_views_do_not_change_the_legacy_prefix_stream(tmp_path):
    legacy = build_dataset(tmp_path, caption_views=False)
    tri = build_dataset(tmp_path, caption_views=True)
    assert len(legacy) == len(tri) == len(CAPTIONS)

    random.seed(1234)
    legacy_pairs = [(legacy[index][1], len(legacy[index][1].split('. '))) for index in REAL_INDICES]
    random.seed(1234)
    tri_pairs = [(tri[index]['caption_said'], tri[index]['prefix_k']) for index in REAL_INDICES]

    assert legacy_pairs == tri_pairs                      # identical K and C_said, same seed
    assert all(1 <= prefix_k for _, prefix_k in tri_pairs)
    random.seed(1234)
    again = [(legacy[index][1], len(legacy[index][1].split('. '))) for index in REAL_INDICES]
    assert again == legacy_pairs                          # the dataset itself is deterministic


def test_unsaid_sentence_is_a_real_withheld_suffix_sentence(tmp_path):
    tri = build_dataset(tmp_path, caption_views=True)
    random.seed(7)
    saw_unsaid = 0
    for index in REAL_INDICES:
        sample = tri[index]
        sentences = caption_sentences(sample['caption_full'])
        assert sample['num_sentences'] == len(sentences)
        assert sample['caption_said'] == '. '.join(sentences[:sample['prefix_k']])
        if sample['has_unsaid']:
            saw_unsaid += 1
            assert sample['prefix_k'] < sample['num_sentences']
            assert sample['prefix_k'] < sample['unsaid_sentence_index'] <= sample['num_sentences']
            assert sample['caption_unsaid'] == sentences[sample['unsaid_sentence_index'] - 1]
            assert sample['caption_unsaid'] not in sample['caption_said']
        else:
            assert sample['prefix_k'] == sample['num_sentences']
            assert sample['caption_unsaid'] == '' and sample['unsaid_sentence_index'] == 0
    assert saw_unsaid >= 1                               # at least one prefix leaves a suffix
    assert saw_unsaid <= len(CAPTIONS)


def test_sample_unsaid_index_is_stateless_and_bounded():
    assert sample_unsaid_index(0, 3, 3, 0) is None
    for index in range(5):
        for prefix_k in range(1, 4):
            picked = sample_unsaid_index(index, 4, prefix_k, seed=11)
            assert prefix_k < picked <= 4
            assert picked == sample_unsaid_index(index, 4, prefix_k, seed=11)   # deterministic
    assert sample_unsaid_index(0, 4, 2, 11) != sample_unsaid_index(0, 4, 2, 12) or True


# --------------------------------------------------------------------------- #
# gate: synthetic attention tests (A-E of the spec)
# --------------------------------------------------------------------------- #
def test_gate_range_and_uniform_scores_are_finite():
    uniform = torch.zeros(2, PATCHES)
    gate = unsaid_core.said_suppression_gate(uniform, 0.1, 1.0)
    assert torch.isfinite(gate).all()
    assert (gate >= 0.1 - 1e-6).all() and (gate <= 1.0 + 1e-6).all()
    assert gate.min().item() == pytest.approx(0.55, abs=1e-6)      # sigmoid(0) = 0.5
    assert (gate > 0).all()                                        # never all zeros

    scores = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    gate = unsaid_core.said_suppression_gate(scores, 0.1, 1.0)
    assert gate.argmax().item() == 0                               # lowest Said score -> highest gate
    assert gate.argmin().item() == 4
    assert (gate >= 0.1).all() and (gate <= 1.0).all()
    for floor in (0.05, 0.1, 0.5):
        bounded = unsaid_core.said_suppression_gate(scores, floor, 1.0)
        assert float(bounded.min()) >= floor - 1e-6
    with pytest.raises(ValueError):
        unsaid_core.said_suppression_gate(scores, 1.0, 1.0)


def test_gate_detaches_the_said_score_path():
    scores = torch.randn(2, PATCHES, requires_grad=True)
    gate = unsaid_core.said_suppression_gate(scores, 0.1, 1.0)
    assert gate.requires_grad is False
    assert gate.grad_fn is None                    # no graph back to the Said scores
    # the scores themselves still receive gradients from their own (non-gate) use
    scores.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_lower_said_coverage_patch_wins_at_equal_hidden_similarity():
    hidden = torch.zeros(1, 1, PATCHES)                   # identical hidden similarity
    gate = torch.tensor([[0.9, 0.7, 0.5, 0.3]])           # patch 0 least used by the prefix
    attention = unsaid_core.debiased_unsaid_attention(hidden, gate, 0.07, 1.0)
    assert attention[0, 0].argmax().item() == 0
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 1), atol=1e-6)
    ordered = attention[0, 0].argsort(descending=True).tolist()
    assert ordered == [0, 1, 2, 3]                        # monotone in the gate


def test_soft_gate_never_hard_removes_a_strongly_hidden_patch():
    gate = torch.tensor([[0.1, 0.9]])                     # patch 0 heavily used by Said
    hidden = torch.tensor([[[5.0, 0.0]]])                 # ... but it matches the hidden text
    attention = unsaid_core.debiased_unsaid_attention(hidden, gate, 0.07, 1.0)
    assert attention[0, 0, 0].item() > attention[0, 0, 1].item()
    assert attention[0, 0].argmax().item() == 0           # still allowed to be top-1
    assert attention[0, 0, 0].item() > 0.0                # never hard-masked


def test_beta_zero_is_pure_hidden_semantic_attention():
    hidden = torch.randn(2, 3, PATCHES)
    gate = torch.rand(2, PATCHES) * 0.9 + 0.1
    plain = unsaid_core.debiased_unsaid_attention(hidden, gate, 0.07, beta=0.0)
    assert torch.allclose(plain, torch.softmax(hidden / 0.07, dim=-1), atol=1e-6)


def test_coverage_diagnostic_prefers_the_gated_attention():
    hidden = torch.zeros(1, 1, PATCHES)
    gate = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
    raw = unsaid_core.debiased_unsaid_attention(hidden, gate, 0.07, beta=0.0)
    gated = unsaid_core.debiased_unsaid_attention(hidden, gate, 0.07, beta=1.0)
    raw_coverage = unsaid_core.said_coverage(raw[:, 0], gate).mean()
    gated_coverage = unsaid_core.said_coverage(gated[:, 0], gate).mean()
    assert gated_coverage < raw_coverage


# --------------------------------------------------------------------------- #
# pairwise missing-text retrieval
# --------------------------------------------------------------------------- #
def test_pairwise_retrieval_loss_top1_and_margins():
    scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    result = unsaid_core.pairwise_retrieval_loss(scores)
    assert float(result['top1_i2t']) == 1.0 and float(result['top1_t2i']) == 1.0
    assert float(result['margin_i2t']) > 0 and float(result['margin_t2i']) > 0
    assert torch.isfinite(result['loss']) and result['batch_size'] == 2


def test_pairwise_retrieval_loss_is_safe_below_two_samples():
    for size in (0, 1):
        scores = torch.randn(size, size, requires_grad=True)
        result = unsaid_core.pairwise_retrieval_loss(scores)
        assert float(result['loss']) == 0.0
        result['loss'].backward()                       # differentiable zero
        assert scores.grad is not None and torch.isfinite(scores.grad).all()
        assert scores.grad.abs().sum().item() == 0.0
    with pytest.raises(ValueError):
        unsaid_core.pairwise_retrieval_loss(torch.randn(2, 3))


def test_score_unsaid_candidates_retrieves_the_matching_hidden_text():
    model = build_model()
    identity_router(model)
    with torch.no_grad():
        patches = torch.zeros(2, PATCHES, DIM)
        semantic_a = torch.zeros(DIM); semantic_a[0] = 1.0
        semantic_b = torch.zeros(DIM); semantic_b[1] = 1.0
        patches[0, :, 0] = 1.0
        patches[1, :, 1] = 1.0
        prefix = torch.zeros(2, DIM); prefix[:, 2] = 1.0      # orthogonal -> uniform gate
        candidates = torch.stack([semantic_a, semantic_b])

    scores = model.score_unsaid_candidates(patches, prefix, candidates)
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]
    assert float(unsaid_core.pairwise_retrieval_loss(scores)['top1_i2t']) == 1.0
    details = model.score_unsaid_candidates(patches, prefix, candidates, return_details=True)
    assert set(details) == {'scores', 'attention', 'gate', 'hidden_logits'}
    assert torch.allclose(details['attention'].sum(-1), torch.ones(2, 2), atol=1e-5)
    assert (details['gate'] >= 0.1).all()


def test_score_said_conditioned_returns_a_pair_matrix():
    model = build_model()
    _, tokens = make_batch()
    with torch.no_grad():
        _, patches = model.encode_router_input(torch.randn(BATCH, 3, 4, 4))
        texts = model.clip.encode_text(tokens)
    scores = model.score_said_conditioned(patches, texts)
    assert scores.shape == (BATCH, BATCH)
    assert torch.isfinite(scores).all()


# --------------------------------------------------------------------------- #
# debiased branch: gradients, valid handling, backward compatibility
# --------------------------------------------------------------------------- #
def tri_forward(model, images, tokens, **kwargs):
    kwargs.setdefault('lambda_unsaid', 1.0)
    kwargs.setdefault('unsaid_mode', 'debiased_suffix')
    kwargs.setdefault('global_caption_view', 'full')
    kwargs.setdefault('has_unsaid', torch.tensor([True, True, False, True]))
    return model.forward_train(images, tokens, 1.0, 1.0,
                               texts_full=tokens.flip(0), texts_unsaid=tokens.flip(1), **kwargs)


def test_debiased_unsaid_gradients_reach_q_k_visual_and_text_paths():
    model = build_model()
    images, tokens = make_batch()
    out = tri_forward(model, images, tokens)
    assert out['unsaid_enabled'] is True
    assert torch.isfinite(out['loss_unsaid'])
    out['loss_unsaid'].backward()

    parameters = dict(model.named_parameters())
    for name in ('said_router.q_proj.weight', 'said_router.k_proj.weight',
                 'clip.patch_proj.weight', 'clip.text_proj.weight'):
        grad = parameters[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name


def test_debiased_unsaid_diagnostics_are_finite_and_typed():
    model = build_model()
    images, tokens = make_batch()
    out = tri_forward(model, images, tokens)
    debiased_keys = ('unsaid_valid_ratio', 'unsaid_valid_batch_size', 'unsaid_retrieval_top1_i2t',
                     'unsaid_retrieval_top1_t2i', 'unsaid_retrieval_margin_i2t',
                     'unsaid_retrieval_margin_t2i', 'unsaid_gate_mean', 'unsaid_gate_min',
                     'unsaid_gate_max', 'said_coverage_under_raw', 'said_coverage_under_gated',
                     'unsaid_attention_entropy', 'unsaid_effective_patch_count',
                     'said_attention_entropy', 'said_effective_patch_count',
                     'said_unsaid_attention_overlap')
    for key in debiased_keys:
        value = out[key]
        assert value is not None, key
        assert torch.isfinite(value).all(), key
    # residual-only diagnostics stay unset in the debiased mode (never fabricated)
    assert out['unsaid_residual_norm_mean'] is None
    assert out['unsaid_target_cosine'] is None
    assert float(out['unsaid_valid_batch_size']) == 3.0
    assert 0.1 - 1e-6 <= float(out['unsaid_gate_min']) <= float(out['unsaid_gate_max']) <= 1.0
    assert 0.0 <= float(out['said_coverage_under_raw']) <= 1.0
    assert 0.0 <= float(out['said_coverage_under_gated']) <= 1.0


def test_debiased_unsaid_with_one_valid_sample_is_a_finite_zero():
    model = build_model()
    images, tokens = make_batch()
    out = tri_forward(model, images, tokens,
                      has_unsaid=torch.tensor([False, True, False, False]))
    assert float(out['unsaid_valid_batch_size']) == 1.0
    assert float(out['loss_unsaid']) == 0.0
    out['loss_unsaid'].backward()                     # still a legal graph
    assert torch.isfinite(out['loss_total'])


def test_prefix_view_and_disabled_unsaid_keep_the_legacy_objective():
    model = build_model()
    images, tokens = make_batch()
    out = model.forward_train(images, tokens, 0.7, 1.3, lambda_unsaid=0.0,
                              global_caption_view='prefix', unsaid_mode='debiased_suffix',
                              texts_full=tokens.flip(0), texts_unsaid=tokens.flip(1),
                              has_unsaid=torch.tensor([True, True, False, True]))

    z_global, patches = model.encode_router_input(images)
    t = F.normalize(model.clip.encode_text(tokens), dim=-1)
    z_g = F.normalize(z_global, dim=-1)
    scale = model.clip.logit_scale.exp().clamp(max=100)
    reference = contrastive_loss(gather_features_with_grad(z_g), gather_features_with_grad(t), scale)

    assert torch.equal(out['loss_global'], reference)      # prefix view, unchanged objective
    assert float(out['loss_unsaid']) == 0.0
    assert torch.equal(out['loss_total'], 0.7 * reference + 1.3 * out['loss_said'])
    assert all(out[key] is None for key in unsaid_core.DIAGNOSTIC_KEYS)


def test_full_caption_view_changes_only_the_global_alignment():
    model = build_model()
    images, tokens = make_batch()
    prefix = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=0.0,
                                 global_caption_view='prefix')
    full = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=0.0,
                               global_caption_view='full', texts_full=tokens.flip(0))
    assert not torch.equal(prefix['loss_global'], full['loss_global'])
    assert torch.equal(prefix['loss_said'], full['loss_said'])       # Said still uses the prefix
    assert torch.equal(prefix['said_attention_entropy'], full['said_attention_entropy'])
    with pytest.raises(ValueError):
        model.forward_train(images, tokens, 1.0, 1.0, 0.0, global_caption_view='full')


def test_debiased_mode_adds_no_parameters_or_state_dict_keys():
    model = build_model()
    keys = sorted(model.state_dict())
    assert not any('unsaid' in key for key in keys)
    assert sorted(model.said_router.state_dict()) == [
        'k_proj.bias', 'k_proj.weight', 'q_proj.bias', 'q_proj.weight']
    reference = build_model()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    assert keys == sorted(reference.state_dict())


def test_legacy_modules_are_untouched_by_the_new_mode():
    model = build_model()
    images, tokens = make_batch()
    with torch.no_grad():
        assert torch.equal(model.encode_image(images), model.clip.encode_image(images))
        assert torch.equal(model.encode_text(tokens), model.clip.encode_text(tokens))
    out = tri_forward(model, images, tokens)
    assert torch.isfinite(out['loss_total'])
    with torch.no_grad():
        assert torch.equal(model.encode_image(images), model.clip.encode_image(images))
