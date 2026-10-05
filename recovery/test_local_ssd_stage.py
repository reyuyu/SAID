import hashlib
import os

import pytest
from torch.utils.data.distributed import DistributedSampler

from recovery import local_ssd_stage as stage


@pytest.mark.parametrize("name", ["/absolute.jpg", "../escape.jpg", "sam/../../escape.jpg", "bad\0.jpg", ""])
def test_reject_unsafe_paths(name):
    with pytest.raises(ValueError):
        stage.safe_relative(name)


def test_sampler_preserves_global_rng():
    before = stage.rng_digest()
    assert before == stage.rng_digest()
    sampler = DistributedSampler(range(1245901), num_replicas=4, rank=0, seed=0, shuffle=True)
    assert list(sampler)[:4] == [1137253, 1120898, 579725, 622156]
    assert stage.rng_digest() == before


def test_copy_and_existing_destination_verification(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "cache/images/source.jpg"
    payload = bytes(range(256)) * 509
    source.write_bytes(payload)
    monkeypatch.setattr(stage, "CACHE", tmp_path / "cache")
    native_open = os.open
    monkeypatch.setattr(stage.os, "open", lambda path, flags, *args, **kwargs:
                        native_open(path, flags & ~os.O_DIRECT, *args, **kwargs))
    first = stage.atomic_copy(source, destination)
    expected = hashlib.sha256(payload).hexdigest()
    assert first["source_sha256"] == first["destination_sha256"] == expected
    assert first["copied"] and destination.read_bytes() == payload
    second = stage.atomic_copy(source, destination, expected)
    assert not second["copied"]
    destination.write_bytes(b"corruption")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        stage.atomic_copy(source, destination, expected)


def test_reject_symlink_destination(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"original")
    destination = cache / "source.jpg"
    destination.symlink_to(source)
    monkeypatch.setattr(stage, "CACHE", cache)
    with pytest.raises(RuntimeError, match="escapes"):
        stage.atomic_copy(source, destination)


def test_local_io_subclass_preserves_native_dataset_return(monkeypatch):
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.local_image_dataset import LocalImageDataset
    from train.nested_semantic_data import NestedDataset
    expected = {"sample_id": 1234, "views": ["full", "summary", "detail"], "image": object()}
    monkeypatch.setattr(NestedDataset, "__getitem__", lambda self, index: expected)
    dataset = object.__new__(LocalImageDataset)
    dataset._path_proof_count = 8
    before = stage.rng_digest()
    assert dataset[234] is expected
    assert stage.rng_digest() == before
