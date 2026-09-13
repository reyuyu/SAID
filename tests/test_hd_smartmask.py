import math
import os
import tempfile

import torch
import torch.nn as nn

from model.hd_smartmask import (
    HighDimGate, atomic_torch_save, blocked_masked_units, build_optimizer,
    conditional_scores, config_dict, four_losses, checkpoint_payload,
    validate_checkpoint,
)


def test_gate_initialises_open_and_backpropagates_through_st():
    gate = HighDimGate()
    hidden = torch.randn(3, 248, 512)
    mask, probability = gate(hidden)
    assert torch.all(mask == 1)
    assert torch.allclose(probability, torch.full_like(probability, 8 / 9), atol=1e-6)
    loss = mask.mean()
    loss.backward()
    assert gate.projection.weight.grad is not None
    assert torch.isfinite(gate.projection.weight.grad).all()


def test_full_open_decode_equals_full_decode_not_native():
    torch.manual_seed(0)
    z = torch.randn(4, 2048)
    decoder = nn.Linear(2048, 512, bias=False)
    nn.init.normal_(decoder.weight)
    masks = torch.ones(4, 2048)
    t = torch.nn.functional.normalize(torch.randn(4, 512), dim=-1)
    q = conditional_scores(z, decoder, masks, t, image_chunk=2, text_chunk=2)
    full = torch.nn.functional.normalize(decoder(z), dim=-1)
    expected = 100 * (full @ t.t())
    assert torch.allclose(q, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(full, torch.nn.functional.normalize(z[:, :512], dim=-1))


def test_candidate_mask_changes_only_its_score_column():
    torch.manual_seed(1)
    z = torch.randn(3, 2048)
    decoder = nn.Linear(2048, 512, bias=False)
    masks = torch.ones(3, 2048)
    t = torch.nn.functional.normalize(torch.randn(3, 512), dim=-1)
    baseline = conditional_scores(z, decoder, masks, t, image_chunk=3, text_chunk=2)
    changed = masks.clone(); changed[1, 0] = 0
    altered = conditional_scores(z, decoder, changed, t, image_chunk=3, text_chunk=2)
    assert not torch.allclose(baseline[:, 1], altered[:, 1])
    assert torch.allclose(baseline[:, 0], altered[:, 0])
    assert torch.allclose(baseline[:, 2], altered[:, 2])


def test_losses_use_sum_over_512_and_only_positive_consistency():
    q = torch.zeros(2, 2)
    g = torch.zeros(2, 512); g[:, 0] = 1
    f = torch.zeros_like(g)
    masked_pos = torch.zeros_like(g)
    masks = torch.zeros(2, 2048)
    terms = four_losses(q, masked_pos, f, g, masks)
    assert terms["rec"].item() == 1.0
    assert terms["cons"].item() == 1.0
    assert terms["sparse"].item() == 0.0
    assert terms["weighted_rec"].item() == 1.0


def test_three_parameter_groups_and_single_decoder_registration():
    clip = nn.Module()
    clip.visual = nn.Module(); clip.visual.proj = nn.Parameter(torch.randn(2, 2))
    clip.logit_scale = nn.Parameter(torch.ones(()))
    clip.mask_net = nn.Linear(2, 2)
    clip.other = nn.Parameter(torch.randn(2))
    latent = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 3), nn.ReLU())
    decoder = nn.Linear(3, 2, bias=False)
    gate = nn.Linear(2, 2)
    opt, groups = build_optimizer(clip, latent, decoder, gate)
    assert len(opt.param_groups) == 3
    assert id(clip.visual.proj) in {id(p) for p in groups["clip"]}
    assert id(clip.logit_scale) not in {id(p) for p in sum(groups.values(), [])}
    assert all(p.requires_grad is False for p in clip.mask_net.parameters())


def test_production_checkpoint_round_trip_and_atomic_write():
    clip = nn.Module(); clip.visual = nn.Module(); clip.visual.proj = nn.Parameter(torch.randn(2, 2))
    clip.logit_scale = nn.Parameter(torch.ones(())); clip.mask_net = nn.Linear(2, 2)
    latent = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 3), nn.ReLU())
    decoder = nn.Linear(3, 2, bias=False); gate = nn.Linear(2, 2); gate.config = lambda: {"test_gate": True}
    opt, _ = build_optimizer(clip, latent, decoder, gate)
    payload = checkpoint_payload(clip, latent, decoder, gate, opt, 20, {"step": 20}, config_dict(), {"epoch": 0}, {"torch": torch.get_rng_state()}, {"git_head": "test"})
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "step20.pt")
        atomic_torch_save(payload, path)
        loaded = torch.load(path, weights_only=False)
        validate_checkpoint(loaded, clip, latent, decoder, gate)
        assert loaded["completed_steps"] == 20
        assert os.path.exists(path + ".tmp") is False
