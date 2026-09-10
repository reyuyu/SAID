"""Phase 2.9B tests: candidate-dimension chunking equivalence (score / loss / gradient)."""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_debiased_unsaid import build_model, make_batch  # noqa: E402

HAS_UNSAID = torch.tensor([True, True, False, True, True, True])
EQUIVALENCE_ATOL = 1e-6          # fp32 accumulation-order difference of the chunked einsum
# Pair scores are multiplied by the CLIP logit scale (~20 here), so the same fp32
# accumulation-order noise appears as a slightly larger *absolute* difference while the
# relative difference stays at fp32 level. Both bounds are asserted, never relaxed silently.
SCORE_ATOL = 5e-6                # absolute, for scores at logit-scale magnitude
SCORE_RTOL = 1e-6                # relative, the meaningful bound


def run_branch(chunk_size, seed=0, batch=6):
    torch.manual_seed(seed)
    model = build_model()
    images, tokens = make_batch(batch)
    out = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=1.0,
                              unsaid_mode='debiased_suffix', global_caption_view='full',
                              texts_full=tokens.flip(0), texts_unsaid=tokens.flip(1),
                              has_unsaid=HAS_UNSAID[:batch],
                              unsaid_candidate_chunk_size=chunk_size)
    out['loss_unsaid'].backward()
    grads = {name: parameter.grad.clone()
             for name, parameter in model.named_parameters() if parameter.grad is not None}
    return out, grads


@pytest.mark.parametrize('chunk_size', [1, 2, 5, 6, 64])
def test_chunked_unsaid_matches_the_unchunked_result(chunk_size):
    base, base_grads = run_branch(0)
    chunked, chunked_grads = run_branch(chunk_size)

    # the loss itself is bit-identical: chunking only splits the candidate dimension
    assert torch.equal(base['loss_unsaid'], chunked['loss_unsaid'])
    assert float(base['unsaid_valid_batch_size']) == float(chunked['unsaid_valid_batch_size']) == 5.0
    for key in ('unsaid_retrieval_top1_i2t', 'unsaid_retrieval_top1_t2i',
                'unsaid_retrieval_margin_i2t', 'unsaid_retrieval_margin_t2i',
                'said_attention_entropy', 'unsaid_attention_entropy',
                'said_unsaid_attention_overlap', 'said_unsaid_attention_jsd',
                'unsaid_gate_mean', 'unsaid_gate_min', 'unsaid_gate_max',
                'said_coverage_under_raw', 'said_coverage_under_gated'):
        assert torch.allclose(base[key], chunked[key], atol=EQUIVALENCE_ATOL), key

    assert set(base_grads) == set(chunked_grads)
    for name in base_grads:
        assert torch.allclose(base_grads[name], chunked_grads[name],
                              atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_ATOL), name


def test_chunking_keeps_only_the_own_positive_attention_rows():
    out, _ = run_branch(2)
    diagnostics = [out[key] for key in ('unsaid_attention_entropy', 'said_coverage_under_raw',
                                       'said_coverage_under_gated')]
    for value in diagnostics:
        assert value is not None and torch.isfinite(value).all()
    assert 0.1 - 1e-6 <= float(out['unsaid_gate_min']) <= float(out['unsaid_gate_max']) <= 1.0


def test_invalid_candidate_chunk_size_is_rejected():
    torch.manual_seed(0)
    model = build_model()
    images, tokens = make_batch(4)
    with pytest.raises(ValueError):
        model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=1.0,
                            unsaid_mode='debiased_suffix', global_caption_view='full',
                            texts_full=tokens.flip(0), texts_unsaid=tokens.flip(1),
                            has_unsaid=torch.tensor([True, True, True, True]),
                            unsaid_candidate_chunk_size=-1)


# --------------------------------------------------------------------------- #
# inference API chunking (Phase 2.9B.1)
# --------------------------------------------------------------------------- #
def api_inputs(seed=0, images=5):
    torch.manual_seed(seed)
    model = build_model()
    _, tokens = make_batch(images)
    with torch.no_grad():
        _, patches = model.encode_router_input(torch.randn(images, 3, 4, 4))
        candidates = model.clip.encode_text(tokens)
        prefix = model.clip.encode_text(tokens.flip(0))
    return model, patches, prefix, candidates


@pytest.mark.parametrize('chunk_size', [1, 2, 4, 5, 32])
def test_inference_api_score_matrix_equivalence(chunk_size):
    model, patches, prefix, candidates = api_inputs()
    reference = model.score_unsaid_candidates(patches, prefix, candidates,
                                              candidate_chunk_size=0)
    chunked = model.score_unsaid_candidates(patches, prefix, candidates,
                                            candidate_chunk_size=chunk_size)
    assert reference.shape == chunked.shape == (5, 5)
    max_diff = (reference - chunked).abs().max().item()
    scale = max(1.0, float(reference.abs().max()))
    assert max_diff <= SCORE_ATOL, max_diff                       # documented fp32 bound
    assert max_diff / scale <= SCORE_RTOL, (max_diff, scale)      # tight relative bound
    assert torch.allclose(reference, chunked, atol=SCORE_ATOL, rtol=SCORE_RTOL)


def test_chunked_details_never_carry_the_full_pairwise_tensors():
    model, patches, prefix, candidates = api_inputs()
    legacy = model.score_unsaid_candidates(patches, prefix, candidates,
                                           candidate_chunk_size=0, return_details=True)
    assert {'scores', 'attention', 'hidden_logits', 'gate'} <= set(legacy)
    assert legacy['attention'].shape == (5, 5, patch_features_count(model))

    chunked = model.score_unsaid_candidates(patches, prefix, candidates,
                                            candidate_chunk_size=2, return_details=True)
    assert chunked['mode'] == 'chunked' and chunked['n_chunks'] == 3
    assert 'attention' not in chunked and 'hidden_logits' not in chunked   # memory safe
    assert chunked['own_attention'].shape == (5, patch_features_count(model))
    assert torch.allclose(chunked['scores'], legacy['scores'], atol=EQUIVALENCE_ATOL)
    assert torch.allclose(chunked['gate'], legacy['gate'], atol=EQUIVALENCE_ATOL)
    # B != C: no well-defined diagonal, and nothing fabricated
    rectangular = model.score_unsaid_candidates(patches[:4], prefix[:4], candidates,
                                                candidate_chunk_size=2, return_details=True)
    assert rectangular['scores'].shape == (4, 5)
    assert rectangular['own_attention'] is None


def patch_features_count(model):
    _, tokens = make_batch(1)
    with torch.no_grad():
        _, patches = model.encode_router_input(torch.randn(1, 3, 4, 4))
    return int(patches.shape[1])


def test_inference_api_rejects_invalid_chunk_sizes():
    model, patches, prefix, candidates = api_inputs()
    with pytest.raises(ValueError):
        model.score_unsaid_candidates(patches, prefix, candidates, candidate_chunk_size=-1)


def test_inference_api_matches_the_training_branch_valid_subset():
    """Valid filtering + compressed diagonal labels: branch loss == API reference loss."""
    from model import unsaid_core

    torch.manual_seed(4)
    model = build_model()
    images, tokens = make_batch(6)
    has_unsaid = torch.tensor([True, False, True, True, False, True])
    out = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=1.0,
                              unsaid_mode='debiased_suffix', global_caption_view='full',
                              texts_full=tokens.flip(0), texts_unsaid=tokens.flip(1),
                              has_unsaid=has_unsaid, unsaid_candidate_chunk_size=2)
    assert float(out['unsaid_valid_batch_size']) == 4.0

    index = torch.nonzero(has_unsaid, as_tuple=False).flatten()
    with torch.no_grad():
        _, patches = model.encode_router_input(images)
        unsaid_texts = model.clip.encode_text(tokens.flip(1))
        prefix_texts = model.clip.encode_text(tokens)
    scores = model.score_unsaid_candidates(patches[index], prefix_texts[index], unsaid_texts[index],
                                           candidate_chunk_size=0)
    reference = unsaid_core.pairwise_retrieval_loss(scores)
    assert torch.allclose(out['loss_unsaid'], reference['loss'], atol=EQUIVALENCE_ATOL)
    assert torch.allclose(out['unsaid_retrieval_top1_i2t'], reference['top1_i2t'],
                          atol=EQUIVALENCE_ATOL)
    assert torch.allclose(out['unsaid_retrieval_margin_i2t'], reference['margin_i2t'],
                          atol=EQUIVALENCE_ATOL)
    # a wrong (uncompressed) mapping is a different problem, so labels 0..3 must be right
    scrambled = scores[torch.randperm(scores.shape[0], generator=torch.Generator().manual_seed(0))]
    assert not torch.allclose(unsaid_core.pairwise_retrieval_loss(scrambled)['loss'],
                              reference['loss'], atol=1e-6)


def test_own_positive_attention_order_across_remainder_chunks():
    """B_u = 5 with chunk 2 -> chunks [0,1] [2,3] [4]; the diagonal rows must stay aligned."""
    model, patches, prefix, candidates = api_inputs(images=5)
    chunked = model.score_unsaid_candidates(patches, prefix, candidates, candidate_chunk_size=2,
                                            return_details=True)
    legacy = model.score_unsaid_candidates(patches, prefix, candidates, candidate_chunk_size=0,
                                           return_details=True)
    diagonal = torch.stack([legacy['attention'][k, k] for k in range(5)])
    assert torch.allclose(chunked['own_attention'], diagonal, atol=EQUIVALENCE_ATOL)
    assert torch.allclose(chunked['own_raw_attention'],
                          torch.stack([torch.softmax(legacy['hidden_logits'][k, k] / 0.07, dim=-1)
                                       for k in range(5)]), atol=EQUIVALENCE_ATOL)
    assert chunked['own_attention'].shape == diagonal.shape == (5, patch_features_count(model))


def test_explicit_gap_diagnostics_are_finite_and_never_in_the_loss():
    model = build_model()
    images, tokens = make_batch()
    full = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=0.0,
                               global_caption_view='full', texts_full=tokens.flip(0))
    for key in ('gap_global_to_full_text', 'gap_global_to_said_text', 'gap_said_to_said_text'):
        assert full[key] is not None, key
        assert torch.isfinite(full[key]), key
        assert 0.0 <= float(full[key]) <= 2.0 + 1e-6, key
    prefix = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=0.0,
                                 global_caption_view='prefix')
    assert prefix['gap_global_to_full_text'] is None          # no full text in prefix view
    assert prefix['gap_global_to_said_text'] is not None
    assert torch.isfinite(prefix['gap_said_to_said_text'])
    assert 'pair_gap_full' in full and 'pair_gap_said' in full   # legacy fields untouched
    assert float(full['gap_global_to_said_text']) == pytest.approx(
        float(1.0 - (torch.nn.functional.normalize(
            model.encode_image(images).detach().float(), dim=-1)
            * torch.nn.functional.normalize(
                model.clip.encode_text(tokens).detach().float(), dim=-1)).sum(-1).mean()), abs=1e-6)
