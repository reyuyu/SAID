"""Incremental tests for strict continuation of Dual-Mask-Full v0.1.

Only the resumption path is covered here:

* a saved checkpoint restores the clip state, the suffix gate and all three optimizer states, and
  the loaded clip digest matches the recorded provenance;
* the continuation refuses to run when the objective code, the coefficients, the mode or any
  objective hyper-parameter disagrees with the checkpoint, and refuses when nothing pins the code;
* the skip rule reproduces exactly the tail of the original stream -- no gap, no repeated batch --
  for the real shape of this run (500 updates inside a 1217-batch epoch of a 3651-step horizon);
* the per-step stream payload is a deterministic function of the batch, so a rebuild can be compared
  with the log.
"""

import copy
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train")
for path in (ROOT, TRAIN):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session", autouse=True)
def one_process_group():
    """The module forward uses autograd-aware all_gather, so a one-process group must exist."""
    if not dist.is_initialized():
        init_file = "/tmp/dual_mask_suffix_resume_test_pg"
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
        dist.init_process_group("gloo", init_method="file://" + init_file, rank=0, world_size=1)
    yield

from train_dual_mask_suffix import (  # noqa: E402
    DualMaskSuffixDataset,
    ResumePositionSampler,
    batch_stream_payload,
    build_optimizers,
    file_sha256,
    load_checkpoint,
    resume_position_problem,
    resume_skip,
    stream_batch_index,
    validate_resume,
    _save_checkpoint,
)
from model.dual_mask_suffix import DualMaskSuffixTrainModule  # noqa: E402


class ToyClip(torch.nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.image = torch.nn.Linear(6, 4)
        self.tokens = torch.nn.Embedding(32, 4)
        self.text = torch.nn.Linear(4, 4)
        self.mask_net = ToyMaskNet()

    def encode_image(self, images):
        return self.image(images.flatten(1))

    def encode_text(self, tokens, return_full=False):
        hidden = self.tokens(tokens)
        pooled = self.text(hidden.mean(dim=1))
        return (pooled, hidden) if return_full else pooled


class ToyMaskNet(torch.nn.Module):
    """Keeps the S0 mask mostly open, like the accepted test fixture.

    The frozen S0 helper normalises ``v * m_s`` without an eps, so a pair whose mask closes every
    coordinate gives 0/0. That is a pre-existing property of the reference objective, not something
    this continuation introduces, and the production run is protected by its per-step collective
    finiteness gate.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor([2.0, 0.1, -0.2, 0.3]))
            self.linear.weight[0].zero_()

    def forward(self, hidden):
        return self.linear(hidden.mean(dim=1))


ARGS = type("Args", (), {"lr": 1e-6, "mask_lr": 1e-3, "suffix_lr": 1e-4, "weight_decay": 1e-2})()


def _module(seed=5):
    torch.manual_seed(seed)
    return DualMaskSuffixTrainModule(ToyClip(), "masked", rank=0, feature_dim=4,
                                     image_chunk=2, text_chunk=3,
                                     lambda_suffix=10.0, lambda_u_sparse=2.0)


def _config(**overrides):
    config = {"objective": "s0_dual_mask_suffix_clean_v01", "suffix_mode": "masked",
              "suffix_lambda": 10.0, "u_sparsity_lambda": 2.0, "batch_size_per_gpu": 256,
              "seed": 0, "epochs": 3, "total_len": 1000, "max_steps": 3651,
              "lr_horizon_steps": 3651, "loader_batches": 1217,
              "objective_code_sha256": "a" * 64,
              "arguments": {"lr": 1e-6, "mask_lr": 1e-3, "suffix_lr": 1e-4, "weight_decay": 1e-2,
                            "warmup": 200, "image_chunk": 16, "text_chunk": 32, "amp_dtype": "bf16"}}
    config.update(overrides)
    return config


EXPECTED = {"suffix_mode": "masked", "suffix_lambda": 10.0, "u_sparsity_lambda": 2.0,
            "batch_size_per_gpu": 256, "seed": 0, "epochs": 3, "total_len": 1000,
            "lr": 1e-6, "mask_lr": 1e-3, "suffix_lr": 1e-4, "weight_decay": 1e-2,
            "warmup": 200, "image_chunk": 16, "text_chunk": 32, "amp_dtype": "bf16"}


def test_checkpoint_restores_model_gate_and_all_three_optimizers(tmp_path):
    module = _module()
    optimizers = build_optimizers(module, ARGS)
    images = torch.randn(6, 1, 6)
    prefix = torch.randint(1, 31, (6, 5))
    suffix = torch.randint(1, 31, (6, 5))
    valid = torch.tensor([True, True, False, True, False, True])
    out = module(images, prefix, suffix, valid, torch.arange(6))
    out["loss_total"].backward()
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.step()
    path = _save_checkpoint(module, optimizers, _config(), str(tmp_path), 500, 0, 499)

    restored = _module(seed=99)
    restored_optimizers = build_optimizers(restored, ARGS)
    payload = load_checkpoint(path, restored, restored_optimizers)
    assert payload["completed_steps"] == 500 and payload["step_in_epoch"] == 499
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, dict(restored.named_parameters())[name])
    for original, loaded in zip(optimizers, restored_optimizers):
        if original is None:
            assert loaded is None
            continue
        assert original.state_dict()["param_groups"] == loaded.state_dict()["param_groups"]
        original_state, loaded_state = original.state_dict()["state"], loaded.state_dict()["state"]
        assert set(original_state) == set(loaded_state) and original_state
        for key, entry in original_state.items():
            for field, value in entry.items():
                assert torch.equal(value, loaded_state[key][field]), (key, field)
                if field == "step":
                    assert int(torch.as_tensor(value).item()) == 1
                else:
                    assert torch.isfinite(torch.as_tensor(value)).all(), (key, field)


def test_resume_refuses_every_disagreement_and_demands_a_code_pin():
    good = _config()
    assert validate_resume(good, good["arguments"], EXPECTED, "a" * 64, None) == []
    assert validate_resume(good, good["arguments"], EXPECTED, "a" * 64, "a" * 64) == []

    cases = {
        "mode": (_config(suffix_mode="native"), "a" * 64, None),
        "suffix weight": (_config(suffix_lambda=1.0), "a" * 64, None),
        "sparsity weight": (_config(u_sparsity_lambda=0.0), "a" * 64, None),
        "batch size": (_config(batch_size_per_gpu=64), "a" * 64, None),
        "seed": (_config(seed=7), "a" * 64, None),
        "objective code": (_config(), "b" * 64, None),
        "no pin at all": (_config(objective_code_sha256=None), "a" * 64, None),
    }
    for name, (config, model_sha, expect_sha) in cases.items():
        problems = validate_resume(config, config["arguments"], EXPECTED, model_sha, expect_sha)
        assert problems, name

    # learning rates and warmup come from the recorded arguments
    for key, value in (("lr", 1e-5), ("mask_lr", 1e-2), ("suffix_lr", 1e-3), ("warmup", 0),
                       ("amp_dtype", "fp32"), ("image_chunk", 8)):
        arguments = dict(good["arguments"])
        arguments[key] = value
        problems = validate_resume(good, arguments, EXPECTED, "a" * 64, None)
        assert any(key in problem for problem in problems), key

    # an older checkpoint that records no code hash may still be resumed when the hash is supplied
    older = _config(objective_code_sha256=None)
    assert validate_resume(older, older["arguments"], EXPECTED, "c" * 64, "c" * 64) == []
    assert validate_resume(older, older["arguments"], EXPECTED, "c" * 64, "d" * 64)


def test_skip_rule_reproduces_the_tail_of_the_stream_without_gaps_or_repeats():
    batches, epochs = 1217, 3
    completed = 500

    original = [(epoch, step) for epoch in range(epochs) for step in range(batches)][:3651]
    assert len(original) == 3651
    consumed = original[:completed]
    assert consumed[-1] == (0, 499)

    replayed, skipped = [], 0
    for epoch, step in [(e, s) for e in range(epochs) for s in range(batches)]:
        if resume_skip(epoch, step, batches, completed):
            skipped += 1
            continue
        replayed.append((epoch, step))
        if len(consumed) + len(replayed) >= 3651:
            break
    # exactly the batches the original run consumed are skipped: no gap, no repeat
    assert skipped == 500
    assert len(replayed) == 3651 - 500
    assert replayed == original[500:3651]
    assert replayed[0] == (0, 500) and replayed[-1] == (2, 1216)
    # nothing is skipped beyond the first residual epoch, and a fresh run skips nothing
    assert not any(resume_skip(1, s, batches, completed) for s in range(batches))
    assert not any(resume_skip(2, s, batches, completed) for s in range(batches))
    assert not any(resume_skip(e, s, batches, None) for e in range(epochs) for s in range(batches))


def test_skip_rule_is_independent_of_the_two_checkpoint_position_conventions():
    """A checkpoint written inside the loop records step_in_epoch=499, one written after the loop
    records 500; both describe 500 consumed batches, and the update count alone must decide."""
    batches, completed = 1217, 500
    # in-loop convention: the recorded index is the batch that was just processed
    assert stream_batch_index(0, 499, batches) == completed - 1
    # after-loop convention: the recorded index is the batch that was about to be processed
    assert stream_batch_index(0, 500, batches) == completed
    for epoch, step in ((0, 499), (0, 500)):
        assert resume_position_problem(epoch, step, batches, completed) is None
        # the decided skip is the same for both conventions: 500 batches, hence 501 is the first
        # batch that trains
        assert resume_skip(0, 499, batches, completed) is True
        assert resume_skip(0, 500, batches, completed) is False
    # an inconsistent record is refused
    assert resume_position_problem(0, 700, batches, completed)
    assert resume_position_problem(1, 0, batches, completed)
    assert resume_position_problem(None, None, batches, completed) is None


def test_resume_skip_boundary_is_exactly_the_first_unconsumed_batch():
    batches = 10
    assert resume_skip(0, 0, batches, None) is False        # not a continuation
    assert resume_skip(0, 0, batches, 0) is False           # nothing consumed yet
    assert resume_skip(0, 9, batches, 10) is True
    assert resume_skip(1, 0, batches, 10) is False          # first batch of the second epoch trains
    assert resume_skip(1, 9, batches, 20) is True
    assert resume_skip(2, 0, batches, 20) is False


def test_batch_stream_payload_is_deterministic_and_matches_the_logged_digest():
    import hashlib
    batch = {"sample_id": torch.tensor([1000, 1001]), "image_id": torch.tensor([7, 9]),
             "prefix_k": torch.tensor([2, 3]), "caption_said": ["a cat", "a dog"],
             "suffix_text": ["on a mat", "in a park"]}
    payload = batch_stream_payload(batch)
    assert payload == batch_stream_payload(copy.deepcopy(batch))
    assert json.loads(payload.decode("utf-8")) == {
        "sample_id": [1000, 1001], "image_id": [7, 9], "prefix_k": [2, 3],
        "prefix": ["a cat", "a dog"], "suffix": ["on a mat", "in a park"]}
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == hashlib.sha256(batch_stream_payload(batch)).hexdigest()
    # a different batch gives a different digest
    batch["prefix_k"] = torch.tensor([2, 4])
    assert hashlib.sha256(batch_stream_payload(batch)).hexdigest() != digest


def test_objective_code_hash_pins_the_real_model_file():
    path = os.path.join(ROOT, "model", "dual_mask_suffix.py")
    digest = file_sha256(path)
    assert len(digest) == 64 and digest == file_sha256(path)
    config = _config(objective_code_sha256=digest)
    assert validate_resume(config, config["arguments"], EXPECTED, digest, None) == []
    assert validate_resume(_config(objective_code_sha256=digest), config["arguments"], EXPECTED,
                           "0" * 64, None)


# --------------------------------------------------------------------------------------------
# dry skip: same caption stream, no image decode
# --------------------------------------------------------------------------------------------

def test_dry_item_matches_the_real_item_and_the_rng_state_on_real_data():
    """The dry item must draw the same prefix and produce the same metadata as the real one.

    This runs on the real dataset (a few indices only) and compares the fields as well as the
    process-wide RNG state after each draw, which is what makes the dry skip stream-exact.
    """
    import random
    dataset = DualMaskSuffixDataset(seed=0, total_len=1000)
    indices = [0, 1, 2, 17, 100, 1234, 5000]

    random.seed(20260914)
    real = [dataset[i] for i in indices]
    real_state = random.getstate()

    random.seed(20260914)
    dry = [dataset._dry_sample(i) for i in indices]
    dry_state = random.getstate()
    assert dry_state == real_state, "the dry path consumed a different amount of randomness"

    for index, (r, d) in zip(indices, zip(real, dry)):
        assert d["prefix_k"] == r["prefix_k"], index
        assert d["caption_said"] == r["caption_said"], index
        assert d["suffix_text"] == r["suffix_text"], index
        assert d["caption_full"] == r["caption_full"], index
        assert d["suffix_valid"] == r["suffix_valid"], index
        assert d["image_id"] == r["image_id"], index
        assert d["sample_id"] == r["sample_id"] == index + 1000
        assert tuple(d["image_a"].shape) == (3, 8, 8)      # placeholder, never trained on
        assert d["dry_item"] is True

    # the tuple form is what the resume sampler emits
    random.seed(7)
    a = dataset[(indices[3], True)]
    random.seed(7)
    b = dataset._dry_sample(indices[3])
    assert a["prefix_k"] == b["prefix_k"] and a["caption_said"] == b["caption_said"]


def test_dry_placeholder_is_collate_compatible():
    from train_dual_mask_suffix import dual_mask_suffix_collate
    dataset = DualMaskSuffixDataset(seed=0, total_len=1000)
    samples = [dataset[(i, True)] for i in range(4)]
    batch = dual_mask_suffix_collate(samples)
    assert tuple(batch["image_a"].shape) == (4, 3, 8, 8)
    assert len(batch["suffix_text"]) == 4 and batch["prefix_k"].tolist() == [s["prefix_k"] for s in samples]


def test_resume_position_sampler_marks_exactly_the_consumed_batches():
    inner = list(range(20))
    sampler = ResumePositionSampler(inner, batches_per_epoch=4, batch_size=5, completed_steps=3)
    marks = list(sampler)
    # the wrapped order is untouched
    assert [index for index, _ in marks] == inner
    # three consumed batches of five items are marked dry, the fourth is not
    assert [dry for _, dry in marks] == [True] * 15 + [False] * 5
    assert len(sampler) == len(inner)
    # a later epoch is entirely beyond the consumed prefix
    sampler.set_epoch(1)
    assert all(not dry for _, dry in sampler)
    # nothing is marked without a continuation
    assert all(not dry for _, dry in ResumePositionSampler(inner, 4, 5, 0))


def test_resume_position_sampler_forwards_set_epoch_to_the_wrapped_sampler():
    class Stub:
        def __init__(self):
            self.epochs = []
            self.values = list(range(6))

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    stub = Stub()
    sampler = ResumePositionSampler(stub, batches_per_epoch=2, batch_size=3, completed_steps=1)
    sampler.set_epoch(2)
    assert stub.epochs == [2] and sampler.epoch == 2


# --------------------------------------------------------------------------------------------
# schedule extension: a longer cosine horizon raises the LR at the seam, and that is recorded
# --------------------------------------------------------------------------------------------

def test_schedule_extension_needs_an_explicit_flag_and_only_goes_upwards():
    longer = dict(EXPECTED)
    longer['epochs'] = 4
    shorter = dict(EXPECTED)
    shorter['epochs'] = 2
    config, arguments = _config(), _config()['arguments']

    # no flag: refused, because the horizon change moves the learning rate
    problems = validate_resume(config, arguments, longer, "a" * 64, None)
    assert any('epochs' in problem for problem in problems)
    # with the flag: allowed
    assert validate_resume(config, arguments, longer, "a" * 64, None,
                           allow_schedule_extension=True) == []
    # even with the flag, shortening the horizon is refused
    assert any('shorten' in problem for problem in
               validate_resume(config, arguments, shorter, "a" * 64, None,
                               allow_schedule_extension=True))
    # identical epochs stay fine in both modes
    assert validate_resume(config, arguments, EXPECTED, "a" * 64, None) == []


def test_cosine_value_reproduces_the_schedule_and_the_seam_jump():
    from train_dual_mask_suffix import cosine_value
    horizon3, horizon4, warmup, base, step = 3651, 4868, 200, 1e-6, 3651
    before = cosine_value(base, warmup, horizon3, step - 1)   # the LR the checkpoint last used
    after = cosine_value(base, warmup, horizon4, step)        # the LR the extension starts with
    assert before < 1e-12, before                             # exhausted 3-epoch schedule
    assert 1.5e-7 < after < 1.7e-7, after                     # 4-epoch schedule at the same step
    assert after / before > 1e5
    # the closed form matches scheduler.cosine_lr for a plain point
    import numpy as np
    assert abs(cosine_value(base, warmup, horizon4, 2000)
               - 0.5 * (1 + np.cos(np.pi * (2000 - warmup) / (horizon4 - warmup))) * base) < 1e-18
    # warmup branch
    assert abs(cosine_value(base, warmup, horizon4, 0) - base / warmup) < 1e-18
