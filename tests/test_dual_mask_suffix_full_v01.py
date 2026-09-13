"""Incremental tests for Dual-Mask-Full v0.1 (`10 L_S + 2 S_S + 10 L_U + 2 S_U`).

Only the newly added behaviour is covered here:

* the two new weights really enter the total loss, and the S0 part is untouched;
* the valid-positive gate sparsity is extracted from the same tiled scoring, once per valid
  positive, with a non-square tile grid, a ragged tail and non-contiguous validity;
* its gradient follows ``abs(hard-ST(sigmoid(logits)))`` exactly, reaches only the new gate, and
  gives no extra gradient to closed coordinates;
* the two new coefficients survive save/load, the full gate loads strictly, and the bare student
  still contains the CLIP state only;
* the two-process DDP worker reproduces the new objective from an independent global reference.

The old configuration (lambda_suffix=1.0, lambda_u_sparse=0.0) is covered by
``tests/test_dual_mask_suffix.py``, which must keep passing unchanged.
"""

import copy
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train")
for path in (ROOT, TRAIN):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.dual_mask_suffix import (  # noqa: E402
    DualMaskSuffixTrainModule,
    build_suffix_gate,
    hard_st,
    pairwise_masked_scores,
    u_sparse_terms,
)


@pytest.fixture(scope="session", autouse=True)
def one_process_group():
    if not dist.is_initialized():
        init_file = "/tmp/dual_mask_suffix_full_v01_test_pg"
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
        dist.init_process_group("gloo", init_method="file://" + init_file, rank=0, world_size=1)
    yield


class ToyClip(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.image = nn.Linear(6, 4)
        self.tokens = nn.Embedding(32, 4)
        self.text = nn.Linear(4, 4)
        self.mask_net = ToyMaskNet()

    def encode_image(self, images):
        return self.image(images.flatten(1))

    def encode_text(self, tokens, return_full=False):
        hidden = self.tokens(tokens)
        pooled = self.text(hidden.mean(dim=1))
        return (pooled, hidden) if return_full else pooled


class ToyMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor([2.0, 0.1, -0.2, 0.3]))
            self.linear.weight[0].zero_()

    def forward(self, hidden):
        return self.linear(hidden.mean(dim=1))


def _module(lambda_suffix, lambda_u_sparse, seed=17):
    torch.manual_seed(seed)
    return DualMaskSuffixTrainModule(ToyClip(), "masked", rank=0, feature_dim=4,
                                     image_chunk=2, text_chunk=3,
                                     lambda_suffix=lambda_suffix,
                                     lambda_u_sparse=lambda_u_sparse)


def _batch(rows=6, tokens=7, dim=4, seed=5):
    torch.manual_seed(seed)
    images = torch.randn(rows, 1, 6)
    prefix = torch.randint(1, 31, (rows, tokens))
    suffix = torch.randint(1, 31, (rows, tokens))
    valid = torch.tensor([bool(i % 3) for i in range(rows)], dtype=torch.bool)
    if int(valid.sum()) < 2:
        valid[0] = valid[1] = True
    return images, prefix, suffix, valid


# --------------------------------------------------------------------------------------------
# weight check: the total really is 10 L_S + 2 S_S + 10 L_U + 2 S_U
# --------------------------------------------------------------------------------------------

def test_new_weights_enter_the_total_loss_and_the_s0_part_is_unchanged():
    images, prefix, suffix, valid = _batch()
    old = _module(1.0, 0.0)
    new = _module(10.0, 2.0)
    new.suffix_gate.load_state_dict(old.suffix_gate.state_dict())
    with torch.no_grad():
        out_old = old(images, prefix, suffix, valid, torch.arange(images.shape[0]))
        out_new = new(images, prefix, suffix, valid, torch.arange(images.shape[0]))

    # identity in the new objective
    weighted = (out_new["weighted_s0_align"] + out_new["weighted_s0_sparse"]
                + out_new["weighted_u_align"] + out_new["weighted_u_sparse"])
    assert torch.allclose(out_new["loss_total"], weighted, atol=1e-5, rtol=1e-6)
    # the S0 part is exactly what it was: 10 L_S + 2 S_S
    assert torch.allclose(out_old["loss_s0"], out_new["loss_s0"], atol=0.0, rtol=0.0)
    assert torch.allclose(out_old["loss_s0"], out_old["weighted_s0_align"] + out_old["weighted_s0_sparse"],
                          atol=1e-5, rtol=1e-6)
    # the alignment log does not depend on the sparsity weight
    assert torch.allclose(out_old["loss_u_align_global"], out_new["loss_u_align_global"],
                          atol=0.0, rtol=0.0)
    # the new weighted sparsity is exactly 2 * S_U
    assert torch.allclose(out_new["weighted_u_sparse"], 2.0 * out_new["loss_u_sparse_global"],
                          atol=1e-6, rtol=1e-6)
    # and the whole difference is exactly 9 L_U + 2 S_U
    assert torch.allclose(out_new["loss_total"] - out_old["loss_total"],
                          9.0 * out_new["loss_u_align_global"] + 2.0 * out_new["loss_u_sparse_global"],
                          atol=1e-4, rtol=1e-5)


def test_initialization_is_fully_open_so_the_difference_is_nine_l_u_plus_two():
    """At the production initialization every coordinate is open, so S_U = 1 and 2 S_U = 2."""
    images, prefix, suffix, valid = _batch()
    model = _module(10.0, 2.0, seed=3)  # untouched initialization: last layer weight 0, bias log(8)
    with torch.no_grad():
        out = model(images, prefix, suffix, valid, torch.arange(images.shape[0]))
        old = _module(1.0, 0.0, seed=3)
        out_old = old(images, prefix, suffix, valid, torch.arange(images.shape[0]))
    assert float(out["m_u_keep_ratio"]) == 1.0
    assert float(out["m_u_positive_keep_ratio"]) == 1.0
    assert float(out["u_sparse_local_sum"]) == float(out["valid_local_count"])  # 512 * V summed = V
    assert abs(float(out["loss_u_sparse_global"]) - 1.0) < 1e-6
    delta = float(out["loss_total"] - out_old["loss_total"])
    expected = 9.0 * float(out["loss_u_align_global"]) + 2.0
    assert abs(delta - expected) < 2e-4, (delta, expected)


# --------------------------------------------------------------------------------------------
# valid-positive index extraction
# --------------------------------------------------------------------------------------------

def _reference_positive(g, m_s, gate, valid, rank):
    """Untiled reference: the diagonal positive pair of each valid global row, taken once.

    Returns the sum of ``mean_d |hard-ST(sigmoid(logits))|`` over valid positives, the matching
    count, their mean keep ratio, and the full probability/mask tensors for extra inspection.
    """
    rows, candidates, dim = g.shape[0], m_s.shape[0], g.shape[1]
    g_norm = F.normalize(g.float(), dim=-1, eps=1e-6)
    m_det = m_s.detach().float()
    pair_g = g_norm.detach()[:, None, :].expand(-1, candidates, -1)
    pair_r = g_norm.detach()[:, None, :] * m_det[None, :, :]
    x = torch.cat((pair_g, pair_r), dim=-1).reshape(-1, 2 * dim)
    probability = torch.sigmoid(gate(x.float())).reshape(rows, candidates, dim)
    mask = hard_st(probability)
    per_pair = mask.abs().mean(dim=-1)
    index = torch.arange(min(rows, candidates))
    columns = int(rank) * rows + index
    inside = columns < candidates
    index, columns = index[inside], columns[inside]
    chosen = valid[index] if index.numel() else torch.zeros(0, dtype=torch.bool)
    diagonal = per_pair[index, columns]
    count = int(chosen.sum())
    return {"sum": (diagonal * chosen.float()).sum(), "count": count,
            "keep_ratio": float(diagonal[chosen].mean()) if count else float("nan"),
            "probability": probability, "mask": mask}


@pytest.mark.parametrize("rows,candidates,image_chunk,text_chunk", [(5, 5, 2, 3), (6, 9, 4, 2), (7, 7, 3, 3)])
def test_positive_sparsity_uses_the_candidate_column_and_counts_each_valid_pair_once(
        rows, candidates, image_chunk, text_chunk):
    torch.manual_seed(11)
    g = torch.randn(rows, 4, requires_grad=True)
    m_s = torch.rand(candidates, 4)
    t = torch.randn(candidates, 4)
    gate = build_suffix_gate(seed=2, input_dim=8, hidden_dim=6, output_dim=4)
    with torch.no_grad():
        gate[2].weight.normal_(0, 0.3)
        gate[2].bias.copy_(torch.tensor([0.4, -0.6, 0.2, -0.1]))
    valid = torch.tensor([bool(i % 2) for i in range(rows)], dtype=torch.bool)
    rank = 0
    _, logs = pairwise_masked_scores(g, m_s, t, gate, image_chunk=image_chunk,
                                     text_chunk=text_chunk, rank=rank, valid_local=valid)
    reference = _reference_positive(g, m_s, gate, valid, rank)
    assert torch.allclose(logs["u_sparse_positive_sum"], reference["sum"], atol=1e-6, rtol=1e-6)
    # validity decided by the same rule on both sides (the rank offset can push a column out)
    assert float(logs["u_sparse_positive_count"]) == float(reference["count"])
    if reference["count"]:
        assert abs(float(logs["m_u_positive_keep_ratio"]) - reference["keep_ratio"]) < 1e-6
    # the score matrix still covers every candidate
    scores, _ = pairwise_masked_scores(g, m_s, t, gate, image_chunk=image_chunk,
                                       text_chunk=text_chunk, rank=rank, valid_local=valid)
    assert scores.shape == (rows, candidates)


def test_sparsity_ignores_non_positive_columns_and_invalid_rows():
    torch.manual_seed(23)
    g = torch.randn(6, 4)
    m_s = torch.rand(6, 4)
    t = torch.randn(6, 4)
    gate = build_suffix_gate(seed=5, input_dim=8, hidden_dim=6, output_dim=4)
    with torch.no_grad():
        gate[2].weight.normal_(0, 0.25)
        gate[2].bias.zero_()
    valid = torch.tensor([True, True, False, False, True, False])
    _, base = pairwise_masked_scores(g, m_s, t, gate, image_chunk=2, text_chunk=2,
                                     rank=0, valid_local=valid)
    scores_base, _ = pairwise_masked_scores(g, m_s, t, gate, image_chunk=2, text_chunk=2,
                                            rank=0, valid_local=valid)
    # perturb only the prefix of an invalid row: its own positive changes, valid ones do not
    m_changed = m_s.clone()
    m_changed[2] = 1.0 - m_changed[2]
    scores_changed, changed = pairwise_masked_scores(g, m_changed, t, gate, image_chunk=2,
                                                     text_chunk=2, rank=0, valid_local=valid)
    assert not torch.allclose(scores_base, scores_changed)  # the perturbation is real
    assert torch.allclose(base["u_sparse_positive_sum"], changed["u_sparse_positive_sum"],
                          atol=1e-6, rtol=1e-6)
    assert float(base["u_sparse_positive_count"]) == float(changed["u_sparse_positive_count"])


# --------------------------------------------------------------------------------------------
# sparse gradient: abs convention, only the new gate, no gradient for closed coordinates
# --------------------------------------------------------------------------------------------

def test_sparsity_gradient_matches_independent_abs_reference_and_reaches_only_the_gate():
    torch.manual_seed(7)
    g = torch.randn(5, 4, requires_grad=True)
    m_s = torch.rand(5, 4, requires_grad=True)
    t = torch.randn(5, 4, requires_grad=True)
    gate = build_suffix_gate(seed=9, input_dim=8, hidden_dim=6, output_dim=4)
    with torch.no_grad():
        gate[2].weight.normal_(0, 0.4)
        # deliberately closes some coordinates
        gate[2].bias.copy_(torch.tensor([0.9, -1.4, 0.05, -0.8]))
    valid = torch.tensor([True, False, True, True, False])
    _, logs = pairwise_masked_scores(g, m_s, t, gate, image_chunk=2, text_chunk=3,
                                     rank=0, valid_local=valid)
    actual = logs["u_sparse_positive_sum"]
    gate_reference = copy.deepcopy(gate)
    reference = _reference_positive(g.detach(), m_s.detach(), gate_reference, valid, 0)
    assert torch.allclose(actual, reference["sum"], atol=1e-6, rtol=1e-6)

    gate_grads = torch.autograd.grad(actual, list(gate.parameters()), retain_graph=True)
    reference_grads = torch.autograd.grad(reference["sum"], list(gate_reference.parameters()))
    for got, want in zip(gate_grads, reference_grads):
        assert torch.allclose(got, want, atol=2e-6, rtol=2e-5)

    # the term is a function of the gate only: the image/text features receive nothing
    assert torch.autograd.grad(actual, g, retain_graph=True, allow_unused=True)[0] is None
    assert torch.autograd.grad(actual, t, retain_graph=True, allow_unused=True)[0] is None
    assert m_s.grad is None or torch.allclose(m_s.grad, torch.zeros_like(m_s))

    # closed coordinates must not receive the extra hidden gradient of the ST term: a gate that
    # closes every coordinate has S_U = 0 and no gradient anywhere in the gate
    all_closed_gate = build_suffix_gate(seed=11, input_dim=8, hidden_dim=6, output_dim=4)
    with torch.no_grad():
        all_closed_gate[2].weight.zero_()
        all_closed_gate[2].bias.fill_(-9.0)  # sigmoid(-9) < 0.5 everywhere
    _, closed_logs = pairwise_masked_scores(g.detach(), m_s.detach(), t, all_closed_gate,
                                            image_chunk=2, text_chunk=3, rank=0,
                                            valid_local=valid)
    assert float(closed_logs["u_sparse_positive_sum"]) == 0.0
    closed_grads = torch.autograd.grad(closed_logs["u_sparse_positive_sum"],
                                       list(all_closed_gate.parameters()), allow_unused=True)
    for got in closed_grads:
        assert got is None or torch.allclose(got, torch.zeros_like(got))
    assert float(closed_logs["m_u_positive_all_closed_fraction"]) == 1.0


def test_sparsity_backward_value_is_w_over_v_and_the_log_is_the_global_mean():
    local = torch.tensor(3.0, requires_grad=True)
    gate_logs = {"u_sparse_positive_sum": local}
    valid_count = torch.tensor(6.0)
    backward, global_value, fields = u_sparse_terms(gate_logs, valid_count, 2, local, "masked")
    assert torch.allclose(backward, (2.0 / 6.0) * local)
    # single process: the logged global mean is local_sum / V, and with two equal ranks the
    # reduction of the two (W/V) * local_sum backward values is exactly this global mean
    assert torch.allclose(global_value, local.detach() / 6.0)
    assert torch.allclose(backward.detach(), 2.0 * global_value)
    assert float(fields["weight_over_valid"]) == pytest.approx(2.0 / 6.0)
    zeros = u_sparse_terms(gate_logs, torch.tensor(1.0), 2, local, "masked")
    assert float(zeros[0]) == 0.0 and float(zeros[1]) == 0.0
    native = u_sparse_terms({"u_sparse_positive_sum": None}, valid_count, 2, local, "native")
    assert float(native[0]) == 0.0 and float(native[1]) == 0.0


# --------------------------------------------------------------------------------------------
# checkpoint / export
# --------------------------------------------------------------------------------------------

def test_new_coefficients_survive_save_load_and_the_bare_student_is_clip_only(tmp_path):
    from train_dual_mask_suffix import (build_optimizers, export_bare_student, load_checkpoint,
                                        _save_checkpoint)

    module = _module(10.0, 2.0)
    args = type("Args", (), {"lr": 1e-3, "mask_lr": 1e-3, "suffix_lr": 1e-3,
                             "weight_decay": 0.0})()
    optimizers = build_optimizers(module, args)
    config = {"objective": "s0_dual_mask_suffix_clean_v01", "suffix_lambda": 10.0,
              "u_sparsity_lambda": 2.0, "u_sparsity": 2.0}
    path = _save_checkpoint(module, optimizers, config, str(tmp_path), 5, 0, 4)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["suffix_lambda"] == 10.0
    assert payload["config"]["u_sparsity_lambda"] == 2.0
    assert payload["suffix_gate_state"] is not None
    assert payload["completed_steps"] == 5

    restored = _module(10.0, 2.0)
    restore_optimizers = build_optimizers(restored, args)
    load_checkpoint(path, restored, restore_optimizers)
    for name, parameter in module.named_parameters():
        assert torch.allclose(parameter, dict(restored.named_parameters())[name])

    bare = str(tmp_path / "bare_student.pt")
    export_bare_student(path, bare)
    state = torch.load(bare, map_location="cpu", weights_only=False)
    assert set(state) == set(module.clip.state_dict())
    assert not any(key.startswith("suffix_gate") for key in state)

    # a module built with the old coefficients must not silently adopt the new ones
    check = _module(1.0, 0.0)
    assert check.lambda_suffix == 1.0 and check.lambda_u_sparse == 0.0
    with pytest.raises(ValueError):
        DualMaskSuffixTrainModule(ToyClip(), "native", feature_dim=4, lambda_u_sparse=2.0)


# --------------------------------------------------------------------------------------------
# two-process DDP: the new objective against an independent global reference
# --------------------------------------------------------------------------------------------

def test_two_process_ddp_reproduces_the_full_objective(tmp_path):
    worker = os.path.join(ROOT, "tests", "_dual_mask_suffix_ddp_worker.py")
    output = str(tmp_path / "ddp_full.json")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
               "--master_addr=127.0.0.1", "--master_port=29593", worker, "--out", output,
               "--lambda-suffix", "10", "--lambda-u-sparse", "2"]
    env = dict(os.environ)
    env["GLOO_SOCKET_IFNAME"] = "lo"
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, env=env)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    report = json.loads(open(output, encoding="utf-8").read())
    assert report["lambda_suffix"] == 10.0 and report["lambda_u_sparse"] == 2.0
    results = report["cases"]
    # 1 valid vs 3 valid, a zero-valid rank, and two V < 2 cases
    assert [row["global_valid"] for row in results] == [4, 3, 0, 1, 4]
    assert results[0]["valid_per_rank"] == [1, 3]
    assert results[1]["valid_per_rank"] == [0, 3]
    assert report["status"] == "passed"
    for row in results:
        assert row["status"] == "within_fixed_tolerance"
        assert row["u_sparse_weighted_error"]["relative_error"] < 1e-4
        if row["global_valid"] < 2:
            assert row["u_sparse_global"] == 0.0
            assert row["u_sparse_locally_backpropagated"] == 0.0
            assert row["u_sparse_positive_count_local"] is None
        else:
            assert row["u_sparse_positive_count_local"] == row["valid_per_rank"][0]
            # a valid positive pair always exists here, and the test gate is deliberately not
            # fully open, so the global sparsity and its gradient path are non-trivial
            assert row["u_sparse_global"] > 0.0
            assert row["u_sparse_reference"] == pytest.approx(row["u_sparse_global"], rel=1e-4)
