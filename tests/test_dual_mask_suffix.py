"""Focused clean-v0.1 tests: independent formulas, checkpoint strictness and DDP smoke."""

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
    native_scores,
    pairwise_masked_scores,
    split_caption_suffix,
    suffix_loss_from_scores,
)


@pytest.fixture(scope="session", autouse=True)
def one_process_group():
    if not dist.is_initialized():
        init_file = "/tmp/dual_mask_suffix_clean_v01_test_pg"
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
        dist.init_process_group("gloo", init_method="file://" + init_file, rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class ToyClip(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.image = nn.Linear(6, 4)
        self.text = nn.Linear(4, 4)
        self.mask_net = ToyMaskNet()

    def encode_image(self, images):
        return self.image(images.flatten(1))

    def encode_text(self, tokens, return_full=False):
        pooled = tokens.float().mean(dim=1, keepdim=True).repeat(1, 4)
        hidden = pooled[:, None, :].repeat(1, 5, 1)
        value = self.text(hidden[:, 0, :])
        return (value, hidden) if return_full else value


class ToyMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden):
        return self.linear(hidden.mean(dim=1))


def test_caption_split_keeps_original_prefix_and_excludes_last_nonempty_sentence():
    full = "A. B. C. D"
    assert split_caption_suffix(full, 1, "A") == ("A", "B. C")
    assert split_caption_suffix(full, 2, "A. B") == ("A. B", "C")
    assert split_caption_suffix(full, 3, "A. B. C") == ("A. B. C", "")
    assert split_caption_suffix("A. B. C. D. ", 2, "A. B")[1] == "C"
    assert split_caption_suffix("A\nB. C", 1, "A B")[1] == ""


def test_gate_initialization_is_all_open_and_restores_rng():
    torch.manual_seed(99)
    expected = torch.rand(5)
    torch.manual_seed(99)
    gate = build_suffix_gate(seed=0, input_dim=8, hidden_dim=6, output_dim=4)
    actual = torch.rand(5)
    assert torch.allclose(expected, actual)
    x = torch.randn(3, 8)
    assert torch.all(torch.sigmoid(gate(x)) > 0.5)
    assert torch.allclose(gate[2].weight, torch.zeros_like(gate[2].weight))
    assert torch.allclose(gate[2].bias, torch.full_like(gate[2].bias, torch.log(torch.tensor(8.0))))


@pytest.mark.parametrize('rows,candidates', [(2, 3), (3, 5)])
def test_pairwise_masked_scores_matches_independent_pair_formula_and_gradients(rows, candidates):
    torch.manual_seed(3)
    g = torch.randn(rows, 4, requires_grad=True)
    m = torch.rand(candidates, 4, requires_grad=True)
    t = torch.randn(candidates, 4, requires_grad=True)
    gate = build_suffix_gate(seed=4, input_dim=8, hidden_dim=6, output_dim=4)
    with torch.no_grad():
        gate[2].weight.normal_(0, 0.2)
        gate[2].bias.zero_()
    q, _ = pairwise_masked_scores(g, m, t, gate, image_chunk=1, text_chunk=2)

    g_ref = g.detach().clone().requires_grad_(True)
    t_ref = t.detach().clone().requires_grad_(True)
    gate_ref = copy.deepcopy(gate)
    g_norm = F.normalize(g_ref.float(), dim=-1, eps=1e-6)
    t_norm = F.normalize(t_ref.float(), dim=-1, eps=1e-6)
    expected = []
    for i in range(rows):
        row = []
        for j in range(candidates):
            x = torch.cat((g_norm[i].detach(), (g_norm[i].detach() * m[j].detach())))
            p = torch.sigmoid(gate_ref(x))
            hard = (p >= 0.5).float() + (p - p.detach())
            u = F.normalize(g_norm[i] * hard, dim=-1, eps=1e-6)
            row.append(100.0 * (u * t_norm[j]).sum())
        expected.append(torch.stack(row))
    expected = torch.stack(expected)
    assert torch.allclose(q, expected, atol=1e-6, rtol=1e-6)
    q.sum().backward(); expected.sum().backward()
    assert torch.allclose(g.grad, g_ref.grad, atol=2e-6, rtol=2e-5)
    assert torch.allclose(t.grad, t_ref.grad, atol=2e-6, rtol=2e-5)
    for actual, reference in zip(gate.parameters(), gate_ref.parameters()):
        assert actual.grad is not None and reference.grad is not None
        assert torch.allclose(actual.grad, reference.grad, atol=2e-6, rtol=2e-5)
    assert m.grad is None


def test_all_open_masked_scores_equal_native_scores():
    torch.manual_seed(5)
    g = torch.randn(2, 4)
    m = torch.randint(0, 2, (3, 4)).float()
    t = torch.randn(3, 4)
    gate = build_suffix_gate(seed=0, input_dim=8, hidden_dim=6, output_dim=4)
    masked, _ = pairwise_masked_scores(g, m, t, gate, image_chunk=2, text_chunk=3)
    assert torch.allclose(masked, native_scores(g, t), atol=1e-5, rtol=1e-5)


def test_candidate_column_condition_and_same_image_counterexample():
    torch.manual_seed(31)
    g = torch.randn(2, 4)
    t = torch.randn(3, 4)
    masks = torch.zeros(3, 4)
    gate = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 4))
    with torch.no_grad():
        gate[0].weight.zero_(); gate[0].bias.zero_()
        gate[0].weight[0, 4] = 1.0
        gate[2].weight.zero_(); gate[2].bias.zero_()
        gate[2].weight[0, 0] = 10.0
    captured = []
    hook = gate[0].register_forward_hook(lambda _m, inp, _out: captured.append(inp[0].detach().clone()))
    q0, _ = pairwise_masked_scores(g, masks, t, gate, image_chunk=2, text_chunk=2)
    masks_changed = masks.clone(); masks_changed[1] = 1.0
    q1, _ = pairwise_masked_scores(g, masks_changed, t, gate, image_chunk=2, text_chunk=2)
    hook.remove()
    assert torch.allclose(q0[:, 0], q1[:, 0]) and torch.allclose(q0[:, 2], q1[:, 2])
    assert not torch.allclose(q0[:, 1], q1[:, 1])
    assert captured and torch.allclose(captured[0][0, :4], captured[0][1, :4])

    g_same = g[:1].expand(2, -1).clone()
    q_same, _ = pairwise_masked_scores(g_same, masks_changed, t, gate, image_chunk=1, text_chunk=2)
    assert torch.allclose(q_same[0], q_same[1])


def test_suffix_loss_preserves_global_labels_and_matches_compact_reference():
    torch.manual_seed(11)
    q_local = torch.randn(4, 8, requires_grad=True)
    q_remote = torch.randn(4, 8, requires_grad=True)
    q_global = torch.cat((q_local, q_remote), dim=0)
    q_local_ref = q_local.detach().clone().requires_grad_(True)
    q_remote_ref = q_remote.detach().clone().requires_grad_(True)
    q_global_ref = torch.cat((q_local_ref, q_remote_ref), dim=0)
    valid_local = torch.tensor([False, True, False, True])
    valid_global = torch.tensor([False, True, False, True, True, False, False, True])
    got = suffix_loss_from_scores(q_local, valid_local, 0, 2, valid_global, q_global=q_global)
    j = torch.tensor([1, 3, 4, 7])
    targets = torch.tensor([0, 1])
    ref_i2t = F.cross_entropy(q_local_ref[[1, 3]][:, j], targets, reduction="sum")
    ref_t2i = F.cross_entropy(q_global_ref.T[[1, 3]][:, j], targets, reduction="sum")
    expected = (ref_i2t + ref_t2i) / 2.0
    assert torch.allclose(got["loss"], expected, atol=1e-6)
    got["loss"].backward(); expected.backward()
    assert torch.allclose(q_local.grad, q_local_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(q_remote.grad, q_remote_ref.grad, atol=1e-6, rtol=1e-6)


def test_v_less_than_two_is_a_connected_zero():
    q = torch.randn(3, 5, requires_grad=True)
    result = suffix_loss_from_scores(q, torch.tensor([True, False, False]), 0, 1,
                                     torch.tensor([True, False, False, False, False]))
    assert result["loss"].item() == 0.0
    result["loss"].backward()
    assert q.grad is not None


def test_model_does_not_read_suffix_tokens_and_strict_checkpoint_load(tmp_path):
    from train_dual_mask_suffix import (build_optimizers, export_bare_student, load_checkpoint,
                                        _save_checkpoint)

    torch.manual_seed(21)
    model = ToyClip()
    module = DualMaskSuffixTrainModule(model, "masked", feature_dim=4, image_chunk=2, text_chunk=3)
    module.eval()
    images = torch.randn(3, 1, 6)
    prefix = torch.randint(1, 9, (3, 5))
    suffix_a = torch.randint(1, 9, (3, 5))
    suffix_b = torch.randint(1, 9, (3, 5))
    valid = torch.ones(3, dtype=torch.bool)
    ids = torch.arange(3)
    out_a = module(images, prefix, suffix_a, valid, ids)
    out_b = module(images, prefix, suffix_b, valid, ids)
    # Suffix tokens may change the suffix score, but they must not alter the gate's
    # image/S0-derived readout or its mask statistics.
    assert torch.allclose(out_a["m_u_probability_mean"], out_b["m_u_probability_mean"], atol=1e-6)
    assert torch.allclose(out_a["m_u_keep_ratio"], out_b["m_u_keep_ratio"], atol=1e-6)
    args = type("Args", (), {"lr": 1e-3, "mask_lr": 1e-3, "suffix_lr": 1e-3,
                              "weight_decay": 0.0})()
    opts = build_optimizers(module, args)
    path = _save_checkpoint(module, opts, {"suffix_mode": "masked"}, str(tmp_path), 2, 0, 1)
    bare = str(tmp_path / "bare_student.pt")
    export_bare_student(path, bare)
    assert set(torch.load(bare, map_location="cpu", weights_only=False)) == set(module.clip.state_dict())
    restored = DualMaskSuffixTrainModule(ToyClip(), "masked", feature_dim=4, image_chunk=2, text_chunk=3)
    load_checkpoint(path, restored, build_optimizers(restored, args))
    for a, b in zip(module.suffix_gate.parameters(), restored.suffix_gate.parameters()):
        assert torch.allclose(a, b)


def test_real_two_process_ddp_dynamic_valid_sets(tmp_path):
    worker = os.path.join(ROOT, "tests", "_dual_mask_suffix_ddp_worker.py")
    output = str(tmp_path / "ddp.json")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
               "--master_addr=127.0.0.1", "--master_port=29591", worker, "--out", output]
    env = dict(os.environ)
    env["GLOO_SOCKET_IFNAME"] = "lo"
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    report = json.loads(open(output, encoding="utf-8").read())
    results = report["cases"]
    assert [row["global_valid"] for row in results] == [4, 3, 0, 1, 4]
    assert results[0]['valid_per_rank'] == [1, 3]
    assert results[1]['valid_per_rank'] == [0, 3]
    assert report['status'] == 'passed'
    # The worker enforces elementwise fixed atol/rtol on every gradient and update.
    assert all(row['status'] == 'within_fixed_tolerance' for row in results)
