import hashlib
import io

from PIL import Image
import pytest
import torch

import sa1b_modelscope_equivalence as equivalent
import sa1b_modelscope_source as source


def jpeg(color="red", size=(32, 24)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG")
    return stream.getvalue()


def transform(image):
    return torch.tensor(list(image.tobytes()), dtype=torch.float32)


def inputs(tmp_path, candidate):
    original = jpeg()
    local_path, source_path = tmp_path / "original.jpg", tmp_path / "source.jpg"
    local_path.write_bytes(original)
    source_path.write_bytes(candidate)
    row = dict(image_id=123, basename="sa_123.jpg", shard="sa_000000.tar", local_path=str(local_path),
        source_url=source.FULL_IMAGE_PREFIX+"sa_123.jpg", source_field="opensource_url", parquet_path="data/test.parquet",
        parquet_sha256="a"*64, repo=source.REPO, revision=source.REVISION, sha256=hashlib.sha256(original).hexdigest())
    proof = dict(downloaded=True, source_path=str(source_path), sha256=hashlib.sha256(candidate).hexdigest())
    return row, proof


def test_same_encoded_jpeg_is_bitwise_equivalent(tmp_path):
    row, proof = inputs(tmp_path, jpeg())
    result = equivalent.compare(row, proof, transform)
    assert result["classification"] == "BITWISE_EQUIVALENT"
    assert result["rgb_pixels_exact_equal"] and result["preprocess_tensor_exact_equal"]


def test_metadata_only_difference_can_be_training_input_equivalent(tmp_path):
    row, proof = inputs(tmp_path, jpeg()+b"metadata-only-tail")
    result = equivalent.compare(row, proof, transform)
    assert result["classification"] == "TRAINING_INPUT_EQUIVALENT"
    assert not result["encoded_bytes_exact_equal"]


def test_changed_pixels_are_not_equivalent_even_if_a_transform_hides_them(tmp_path):
    row, proof = inputs(tmp_path, jpeg("blue"))
    result = equivalent.compare(row, proof, lambda image: torch.ones(3))
    assert result["classification"] == "NOT_EQUIVALENT"
    assert not result["rgb_pixels_exact_equal"] and result["preprocess_tensor_exact_equal"]


def test_resized_image_closes_gate(tmp_path):
    row, proof = inputs(tmp_path, jpeg(size=(16, 12)))
    result = equivalent.compare(row, proof, transform)
    assert result["classification"] == "NOT_EQUIVALENT"
    assert not result["dimensions_equal"]


def test_local_reference_must_match_original_archive_inventory(tmp_path):
    row, proof = inputs(tmp_path, jpeg())
    row["sha256"] = "wrong"
    with pytest.raises(ValueError, match="original tar inventory"):
        equivalent.compare(row, proof, transform)


def test_single_mismatch_or_missing_sample_never_authorizes_replacement():
    good = [dict(image_id=position, original_shard="normal", classification="BITWISE_EQUIVALENT", passed=True)
            for position in range(200)]
    assert equivalent.summary(good, 200, {})["passed"]
    assert not equivalent.summary(good[:-1], 200, {})["passed"]
    good[-1]["classification"] = "NOT_EQUIVALENT"
    assert not equivalent.summary(good, 200, {})["replacement_authorized"]


def test_duplicate_ids_do_not_count_as200_distinct_references():
    rows = [dict(image_id=1, original_shard="normal", classification="BITWISE_EQUIVALENT", passed=True)]*200
    assert not equivalent.summary(rows, 200, {})["replacement_authorized"]


def test_local_crop_and_other_repository_objects_are_never_full_image_candidates():
    assert source.full_image_id(source.FULL_IMAGE_PREFIX+"sa_123.jpg") == 123
    assert source.full_image_id("https://modelscope.cn-beijing.oss.aliyuncs.com/open_data/sa-1b-mask-caption/prod/crop.jpg") is None
    assert source.full_image_id(source.FULL_IMAGE_PREFIX+"sa_123.jpg?resize=1") is None
