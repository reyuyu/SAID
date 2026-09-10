"""Phase 2.7B tests: exact chunked retrieval evaluation + Full Data Gate path semantics."""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.retrieval.coco_retrieval import retrieval_metrics  # noqa: E402
from tools.data.full_data_gate import resolve_json_path  # noqa: E402

ALL_KEYS = {'%s_R%d' % (d, k) for d in ('image2text', 'text2image') for k in (1, 5, 10)}


def _legacy_reference(image_features, text_features, captions_per_image=5):
    """Reference copy of ``train_utils.eval_coco``'s metric loop (argsort based)."""
    images = image_features / image_features.norm(dim=-1, keepdim=True)
    texts = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = images @ texts.t()
    n = images.shape[0]
    out = {}
    for k in (1, 5, 10):
        hits = 0
        for i in range(n):
            top = similarity[i].argsort()[-k:]
            if any(captions_per_image * i + j in top for j in range(captions_per_image)):
                hits += 1
        out['image2text_R%d' % k] = hits / float(n)
    similarity_t = similarity.t()
    for k in (1, 5, 10):
        hits = 0
        for i in range(n * captions_per_image):
            top = similarity_t[i].argsort()[-k:]
            if (i // captions_per_image) in top:
                hits += 1
        out['text2image_R%d' % k] = hits / float(n * captions_per_image)
    return out


def _perfect(dim=8, n=8, captions_per_image=5):
    """One-hot features: every caption of image i is exactly image i's feature."""
    images = torch.zeros(n, dim)
    texts = torch.zeros(n * captions_per_image, dim)
    for i in range(n):
        images[i, i] = 1.0
        for j in range(captions_per_image):
            texts[i * captions_per_image + j, i] = 1.0
    return images, texts


def test_i2t_and_t2i_perfect_alignment():
    images, texts = _perfect()
    metrics = retrieval_metrics(images, texts, captions_per_image=5)
    assert set(metrics) == ALL_KEYS
    for key, value in metrics.items():
        assert value == pytest.approx(1.0), key


def test_five_captions_per_image_mapping():
    """A hit is any of the image's own 5 captions, not only the first one."""
    torch.manual_seed(11)
    dim, n, c = 4, 2, 5
    images = torch.eye(dim)[:n].clone()
    # captions that "belong" to image i lean towards image i's feature but are
    # unique, so no row is a tie
    texts = torch.zeros(n * c, dim)
    for i in range(n):
        for j in range(c):
            noise = torch.randn(dim)
            texts[i * c + j] = torch.nn.functional.normalize(images[i] * 0.5 + noise * 0.1, dim=-1)
    # move the strongest caption of each image away from its first position
    texts[0], texts[2] = texts[2].clone(), texts[0].clone()
    texts[6], texts[5] = texts[5].clone(), texts[6].clone()

    metrics = retrieval_metrics(images, texts, captions_per_image=c)
    assert metrics['image2text_R1'] == pytest.approx(1.0)
    assert metrics['text2image_R1'] == pytest.approx(1.0)


def test_cross_image_captions_are_not_hits():
    dim, n, c = 2, 2, 5
    images = torch.eye(dim)[:n].clone()
    texts = torch.zeros(n * c, dim)
    texts[0:c] = images[1]        # all captions of image 0 point at image 1
    texts[c:2 * c] = images[0]    # and vice versa
    metrics = retrieval_metrics(images, texts, captions_per_image=c)
    assert metrics['image2text_R1'] == pytest.approx(0.0)
    assert metrics['text2image_R1'] == pytest.approx(0.0)


def test_known_partial_recall_value():
    dim, n = 4, 4
    images = torch.eye(dim).clone()
    texts = torch.eye(dim).clone()
    texts[1] = images[3]          # image 1 and 3 swap their captions -> both miss
    texts[3] = images[1]
    metrics = retrieval_metrics(images, texts, captions_per_image=1)
    assert metrics['image2text_R1'] == pytest.approx(0.5)
    assert metrics['text2image_R1'] == pytest.approx(0.5)
    assert metrics['image2text_R5'] == pytest.approx(1.0)


def test_chunked_equals_full_exactly():
    torch.manual_seed(0)
    images = torch.randn(53, 16)
    texts = torch.randn(53 * 5, 16)
    full = retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=None)
    for chunk in (1, 2, 7, 13, 53, 4096):
        chunked = retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=chunk)
        assert chunked == full, 'chunk=%r differs from full' % (chunk,)


def test_chunked_equals_full_for_single_caption_retrieval():
    torch.manual_seed(1)
    images = torch.randn(29, 8)
    texts = torch.randn(29, 8)
    full = retrieval_metrics(images, texts, captions_per_image=1, similarity_chunk=None)
    for chunk in (1, 3, 7):
        assert retrieval_metrics(images, texts, captions_per_image=1, similarity_chunk=chunk) == full


def test_normalisation_and_scale_invariance():
    torch.manual_seed(2)
    images = torch.randn(20, 12)
    texts = torch.randn(20 * 5, 12)
    reference = retrieval_metrics(images, texts, captions_per_image=5)
    scaled = retrieval_metrics(images * 13.5, texts * 0.02, captions_per_image=5)
    normalized = retrieval_metrics(
        torch.nn.functional.normalize(images, dim=-1),
        torch.nn.functional.normalize(texts, dim=-1),
        captions_per_image=5,
    )
    assert scaled == reference
    assert normalized == reference


def test_metrics_are_deterministic():
    torch.manual_seed(3)
    images = torch.randn(31, 10)
    texts = torch.randn(31 * 5, 10)
    first = retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=4)
    second = retrieval_metrics(images, texts, captions_per_image=5, similarity_chunk=4)
    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first.values())


def test_invalid_inputs_are_rejected():
    images = torch.randn(4, 6)
    with pytest.raises(ValueError):
        retrieval_metrics(images, torch.randn(4, 6), captions_per_image=0)
    with pytest.raises(ValueError):
        retrieval_metrics(images, torch.randn(9, 6), captions_per_image=5)  # 9 != 4 * 5
    with pytest.raises(ValueError):
        retrieval_metrics(torch.randn(4), torch.randn(4), captions_per_image=1)
    with pytest.raises(ValueError):
        retrieval_metrics(torch.randn(4, 6), torch.randn(4, 7), captions_per_image=1)


def test_matches_legacy_reference_including_ties():
    """Exact parity with train_utils.eval_coco's argsort-based selection, ties included."""
    torch.manual_seed(5)
    n, c, d = 12, 5, 6
    images = torch.randn(n, d)
    texts = torch.randn(n * c, d)
    texts[7] = texts[3].clone()      # duplicate caption -> exact tie
    texts[20] = texts[21].clone()
    reference = _legacy_reference(images, texts, captions_per_image=c)
    assert retrieval_metrics(images, texts, captions_per_image=c, similarity_chunk=4) == reference
    assert retrieval_metrics(images, texts, captions_per_image=c, similarity_chunk=None) == reference
    assert retrieval_metrics(images, texts, captions_per_image=c, similarity_chunk=1) == reference


def test_gate_json_path_relative_and_absolute():
    root = '/root/datasets/ShareGPT4V'
    assert resolve_json_path(root, 'share-captioner_coco_lcs_sam_1246k_1107.json') == os.path.join(
        root, 'share-captioner_coco_lcs_sam_1246k_1107.json')
    assert resolve_json_path(root, 'debug/share4v_smoke_nosam.json') == os.path.join(
        root, 'debug/share4v_smoke_nosam.json')
    absolute = '/mnt/data/some.json'
    assert resolve_json_path(root, absolute) == absolute
    with pytest.raises(ValueError):
        resolve_json_path(root, '')
    # relative paths follow the dataset loader's own join convention
    import sharegpt4v
    assert resolve_json_path(root, sharegpt4v.json_name) == os.path.join(root, sharegpt4v.json_name)
