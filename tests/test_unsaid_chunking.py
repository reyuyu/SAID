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
