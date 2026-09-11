"""SAID-CLS-CVSSL v0.1 tests: SmartCLIP equivalence gate + CVSSL semantics + isolation."""
import copy
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import complement_visual_ssl as cvssl  # noqa: E402
from model import longclip  # noqa: E402
from model.said_cls_cvssl import (ARMS, ARM_LAMBDA_U, ARM_MASK,  # noqa: E402
                                  SaidClsCvsslObjective, compute_smartclip_terms,
                                  said_mask_from_hidden)
from said_cvssl_data import (CLIP_MEAN, CLIP_STD, Share4VCvsslDataset,  # noqa: E402
                             cvssl_collate, image_id_from_path, reference_view_a_transform,
                             shared_content_weak_v1, stateless_seed)

DIM = 16
PATCHES = 4
TOKENS = 5


@pytest.fixture(scope='session', autouse=True)
def _single_process_group():
    """``torch.distributed.nn.all_gather`` (the reference SmartCLIP gather) needs a live group."""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='file:///tmp/cvssl_test_pg',
                                rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class _ToyClip(nn.Module):
    """Minimal CLIP-shaped module whose ``forward`` is the *reference* SmartCLIP forward."""

    def __init__(self, dim=DIM, text_width=24, mask_width=12):
        super().__init__()
        self.embed_dim = dim
        self.text_width = text_width
        self.context_length = 248
        torch.manual_seed(0)
        self.text_projection = nn.Parameter(torch.zeros(text_width, dim))
        self.text_proj = nn.Linear(text_width, dim)
        self.visual_proj = nn.Linear(text_width, dim)
        self.mask_net = _ToyMaskNet(text_width, mask_width, dim)

    def encode_image(self, images):
        pooled = images.flatten(1)[:, :self.text_width]
        return self.visual_proj(pooled)

    def encode_text(self, tokens, return_full=False, return_pool=False):
        hidden = tokens.float().mean(dim=1, keepdim=True).repeat(1, self.text_width)
        hidden = hidden.unsqueeze(1).repeat(1, TOKENS, 1)          # [b, T, W]
        pooled = self.text_proj(hidden[:, 0, :])
        if return_full:
            return pooled, hidden
        return pooled

    def forward(self, image, text, rank, soft=False):
        """Reference SmartCLIP forward (the object of the S0 equivalence gate)."""
        import torch.distributed.nn as nn_dist
        unnorm_image_features_long = self.encode_image(image)
        unnorm_text_features, full_text_embedding = self.encode_text(text, return_full=True)
        text_features = unnorm_text_features / unnorm_text_features.norm(dim=1, keepdim=True)

        mask_logits = self.mask_net(full_text_embedding.detach())
        soft_mask = torch.sigmoid(mask_logits)
        hard_mask = (soft_mask >= 0.5).float() - soft_mask.detach() + soft_mask
        use_mask = soft_mask if soft else hard_mask

        text_features_all = torch.cat(nn_dist.all_gather(text_features), dim=0)
        loss_sparsity = torch.mean(torch.abs(use_mask))
        use_mask_all = torch.cat(nn_dist.all_gather(use_mask), dim=0)
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

        rep_use_mask_all_sidm = use_mask_all.repeat(bs, 1)
        rep_image = unnorm_image_features_long.repeat_interleave(len(use_mask_all), dim=0)
        rep_image = rep_image * rep_use_mask_all_sidm
        rep_image = rep_image / rep_image.norm(dim=1, keepdim=True)
        sidm_sim = torch.sum(rep_image * text_features_all.repeat(bs, 1), dim=1)
        sidm_sim = 100 * sidm_sim.view(bs, len(use_mask_all))
        loss_sidm = F.cross_entropy(sidm_sim, targets)

        image_all = torch.cat(nn_dist.all_gather(unnorm_image_features_long), dim=0)
        ns = len(use_mask_all)
        rep_mask = use_mask.repeat_interleave(ns, dim=0)
        rep_image_dism = image_all.repeat(bs, 1) * rep_mask
        rep_image_dism = rep_image_dism / rep_image_dism.norm(dim=1, keepdim=True)
        dism_sim = torch.sum(rep_image_dism * text_features.repeat_interleave(ns, 0), dim=1)
        dism_sim = 100 * dism_sim.view(bs, len(use_mask_all))
        loss_dism = F.cross_entropy(dism_sim, targets)
        return {'loss_sidm': loss_sidm, 'loss_dism': loss_dism, 'loss_sparsity': loss_sparsity,
                'num_mean': torch.sum(hard_mask, dim=1).float().mean().item()}


class _ToyMaskNet(nn.Module):
    def __init__(self, width, mask_width, dim):
        super().__init__()
        self.proj = nn.Linear(width, mask_width)
        self.out = nn.Linear(mask_width, dim)

    def forward(self, hidden):
        return self.out(torch.tanh(self.proj(hidden.mean(dim=1))))


def make_batch(batch=6, seed=3, dim=DIM):
    torch.manual_seed(seed)
    return {
        'image_a': torch.randn(batch, 3, 4, 4),
        'image_b': torch.randn(batch, 3, 4, 4),
        'caption_said': ['a cat . a dog .'] * batch,
        'image_id': torch.arange(batch, dtype=torch.long),
        'sample_id': torch.arange(batch, dtype=torch.long),
    }


def make_feature_model(seed=0, batch=6):
    torch.manual_seed(seed)
    model = _ToyClip()
    model.train()
    data = make_batch(batch)
    text = torch.randint(0, 10, (batch, TOKENS))
    return model, data, text


# --------------------------------------------------------------------------- #
# 1. SmartCLIP equivalence gate (S0 == reference CLIP.forward)
# --------------------------------------------------------------------------- #
def _reference_and_new(model, data, text, soft_mask=False):
    """Run the reference CLIP.forward and the new feature-level terms on the same weights."""
    reference_model = copy.deepcopy(model)
    new_model = copy.deepcopy(model)

    reference = reference_model(data['image_a'], text, 0, soft=soft_mask)
    v_a = new_model.encode_image(data['image_a'])
    t_raw, hidden = new_model.encode_text(text, return_full=True)
    m_s, _, _ = said_mask_from_hidden(new_model.mask_net, hidden, soft_mask=soft_mask)
    terms = compute_smartclip_terms(v_a, t_raw, m_s, 0)
    return reference_model, new_model, reference, terms


def test_reference_mask_ste_is_binary_in_forward_and_differentiable():
    model, _, text = make_feature_model()
    _, hidden = model.encode_text(text, return_full=True)
    m_s, soft, logits = said_mask_from_hidden(model.mask_net, hidden)
    assert torch.all((m_s.detach() == 0) | (m_s.detach() == 1))
    (m_s.sum()).backward()
    assert model.mask_net.proj.weight.grad is not None
    assert float(model.mask_net.proj.weight.grad.abs().sum()) > 0.0
    assert torch.allclose(soft, torch.sigmoid(logits))
    # the text hidden state feeds the mask net DETACHED (reference behaviour)
    live_hidden = hidden.detach().clone().requires_grad_(True)
    live_m_s, _, _ = said_mask_from_hidden(model.mask_net, live_hidden)
    gradient = torch.autograd.grad(live_m_s.sum(), live_hidden, allow_unused=True)[0]
    assert gradient is None


@pytest.mark.parametrize('soft_mask', [False, True])
def test_s0_equals_reference_smartclip_losses_and_gradients(soft_mask):
    model, data, text = make_feature_model()
    reference_model, new_model, reference, terms = _reference_and_new(model, data, text, soft_mask)

    for key, tol in (('loss_sidm', 1e-5), ('loss_dism', 1e-5), ('loss_sparsity', 1e-5)):
        assert abs(reference[key].item() - terms[key].item()) <= tol, key
    reference_total = 10.0 * (reference['loss_sidm'] + reference['loss_dism']) \
        + 2.0 * reference['loss_sparsity']
    assert abs(reference_total.item() - terms['loss_smart'].item()) <= 1e-4

    # gradients: same weights, same loss -> same gradients for every parameter group
    reference_total.backward()
    terms['loss_smart'].backward()
    other_parameters = dict(reference_model.named_parameters())
    checked = 0
    for name, parameter in new_model.named_parameters():
        other = other_parameters[name]
        if parameter.grad is None and other.grad is None:
            continue                                   # unused in this toy configuration
        assert (parameter.grad is None) == (other.grad is None), name
        difference = float((parameter.grad - other.grad).abs().max())
        scale = max(float(other.grad.abs().max()), 1e-8)
        assert difference / scale <= 1e-4, (name, difference, scale)
        checked += 1
    assert checked >= 4                                # visual, text, mask_net, at least one bias


def test_s0_one_optimizer_step_matches_the_reference():
    model, data, text = make_feature_model()
    reference_model, new_model, reference, terms = _reference_and_new(model, data, text)
    reference_total = 10.0 * (reference['loss_sidm'] + reference['loss_dism']) \
        + 2.0 * reference['loss_sparsity']

    optimizers = []
    for target, loss in ((reference_model, reference_total), (new_model, terms['loss_smart'])):
        optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3, weight_decay=1e-2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizers.append(optimizer)

    for name, parameter in new_model.named_parameters():
        other = dict(reference_model.named_parameters())[name]
        assert torch.allclose(parameter, other, atol=1e-7, rtol=1e-5), name


def test_s0_arm_is_exactly_the_smartclip_objective():
    model, data, text = make_feature_model()
    objective = SaidClsCvsslObjective(model, rank=0, arm='S0_smartclip',
                                      lambda_u=ARM_LAMBDA_U['S0_smartclip'])
    out = objective(data['image_a'], data['image_b'], text, data['image_id'],
                    random_generator=torch.Generator().manual_seed(0))
    assert abs(float(out['loss_total']) - float(out['loss_smart'])) <= 1e-8
    assert float(out['loss_vssl_raw']) > 0.0                     # diagnostics still computed
    assert not out['loss_vssl_raw'].requires_grad
    out['loss_total'].backward()
    assert model.mask_net.proj.weight.grad is not None


# --------------------------------------------------------------------------- #
# 2. anchor-mask semantics
# --------------------------------------------------------------------------- #
def test_one_row_uses_the_anchor_mask_for_every_candidate():
    torch.manual_seed(1)
    anchors = torch.randn(2, 8)
    candidates = torch.randn(5, 8)
    mask = torch.tensor([[1.0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    logits, _ = cvssl.anchor_masked_cosine_logits(anchors, candidates, mask, tau_u=0.1)

    def masked_cosine(a, k, m):
        return F.cosine_similarity((a * m), (k * m), dim=-1) / 0.1

    for i in range(2):
        for j in range(5):
            expected = masked_cosine(anchors[i], candidates[j], mask[i])
            assert abs(float(logits[i, j]) - float(expected)) < 1e-5

    # the forbidden candidate-mask form must disagree (this is the shortcut we forbid)
    wrong = torch.stack([masked_cosine(anchors[i], candidates[j], mask[j % 2])
                         for i in range(2) for j in range(5)]).reshape(2, 5)
    assert float((logits - wrong).abs().max()) > 1e-3


def test_masked_cosine_keeps_mask_squared():
    torch.manual_seed(2)
    anchors = torch.randn(3, 6)
    candidates = torch.randn(4, 6)
    mask = torch.rand(3, 6) * 0.7 + 0.3                        # soft values, rho > 0 style
    exact, _ = cvssl.anchor_masked_cosine_logits(anchors, candidates, mask, tau_u=0.1)
    reference = cvssl.explicit_broadcast_reference(anchors, candidates, mask, tau_u=0.1)
    assert float((exact - reference).abs().max()) < 1e-5

    without_square = (anchors[:, None, :] * candidates[None, :, :] * mask[:, None, :]).sum(-1)
    a_norm = (anchors[:, None, :].square() * mask[:, None, :]).sum(-1).sqrt()
    k_norm = (candidates[None, :, :].square() * mask[:, None, :]).sum(-1).sqrt()
    wrong = without_square / (a_norm * k_norm) / 0.1
    assert float((exact - wrong).abs().max()) > 1e-3


def test_directions_are_recomputed_not_transposed():
    torch.manual_seed(3)
    v_a = torch.randn(4, 8)
    v_b = torch.randn(4, 8)
    mask = (torch.rand(4, 8) > 0.3).float()
    ab = cvssl.cvssl_direction(v_a, v_b, mask, torch.arange(4), torch.arange(4),
                               torch.arange(4), tau_u=0.1)
    ba = cvssl.cvssl_direction(v_b, v_a, mask, torch.arange(4), torch.arange(4),
                               torch.arange(4), tau_u=0.1)
    assert float((ab['logits'] - ba['logits'].t()).abs().max()) > 1e-3


def test_positive_minus_negative_and_log_valid_candidate_count():
    """All features identical and valid -> loss == log(#valid candidates)."""
    torch.manual_seed(4)
    batch, candidates, dim = 3, 5, 8
    anchor = torch.randn(1, dim).repeat(batch, 1)
    bank = anchor[0:1].repeat(candidates, 1)
    mask = torch.ones(batch, dim)
    result = cvssl.cvssl_direction(anchor, bank, mask, torch.arange(batch),
                                   torch.arange(batch), torch.arange(candidates), tau_u=0.1)
    expected = torch.log(torch.tensor(float(candidates)))
    assert abs(float(result['per_anchor_loss']) / batch - float(expected)) < 1e-4
    # with one duplicate candidate removed, that anchor sees one fewer candidate
    ids = torch.arange(candidates)
    ids[4] = 0                                            # candidate 4 duplicates anchor 0's image
    result_dup = cvssl.cvssl_direction(anchor, bank, mask, torch.arange(batch),
                                       torch.arange(batch), ids, tau_u=0.1)
    expected_dup = (float(torch.log(torch.tensor(4.0))) + 2 * float(expected)) / batch
    assert abs(float(result_dup['per_anchor_loss']) / batch - expected_dup) < 1e-4


# --------------------------------------------------------------------------- #
# 3. gradient isolation
# --------------------------------------------------------------------------- #
def test_vssl_cannot_reach_mask_net_text_or_logit_scale():
    model, data, text = make_feature_model()
    objective = SaidClsCvsslObjective(model, rank=0, arm='C0_complement_vssl', lambda_u=1.0)
    out = objective(data['image_a'], data['image_b'], text, data['image_id'],
                    random_generator=torch.Generator().manual_seed(0))
    out['loss'].backward()
    for name, parameter in model.mask_net.named_parameters():
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name
    for name, parameter in model.text_proj.named_parameters():
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name
    visual_grad = model.visual_proj.weight.grad
    assert visual_grad is not None and float(visual_grad.abs().sum()) > 0.0


def test_both_views_receive_anchor_gradient():
    torch.manual_seed(5)
    v_a = torch.randn(4, 8, requires_grad=True)
    v_b = torch.randn(4, 8, requires_grad=True)
    mask = torch.ones(4, 8)
    loss = cvssl.complement_visual_contrastive_loss(v_a, v_b, mask, torch.arange(4),
                                                    tau_u=0.1)['loss']
    loss.backward()
    assert float(v_a.grad.abs().sum()) > 0.0
    assert float(v_b.grad.abs().sum()) > 0.0


def test_zero_coordinates_get_no_direct_gradient():
    torch.manual_seed(6)
    v_a = torch.randn(2, 6, requires_grad=True)
    v_b = torch.randn(3, 6)
    mask = torch.zeros(2, 6)
    mask[:, :3] = 1.0
    logits, _ = cvssl.anchor_masked_cosine_logits(v_a, v_b, mask, tau_u=0.1)
    logits.sum().backward()
    assert float(v_a.grad[:, 3:].abs().sum()) == 0.0
    assert float(v_a.grad[:, :3].abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# 4. degenerate cases
# --------------------------------------------------------------------------- #
def test_empty_complement_is_finite_and_reported_not_faked():
    v_a = torch.randn(2, 6, requires_grad=True)
    v_b = torch.randn(2, 6)
    mask = torch.zeros(2, 6)                                    # empty complement, no fallback
    out = cvssl.complement_visual_contrastive_loss(v_a, v_b, mask, torch.arange(2), tau_u=0.1)
    assert torch.isfinite(out['loss']).all()
    assert float(out['loss']) == 0.0
    assert float(out['valid_ab']) == 0.0 and float(out['valid_ba']) == 0.0
    assert float(out['invalid_norm_count']) >= 0.0


def test_duplicate_images_are_not_negatives():
    torch.manual_seed(7)
    v_a = torch.randn(4, 8)
    v_b = torch.randn(4, 8)
    mask = torch.ones(4, 8)
    ids = torch.tensor([1, 1, 2, 2])
    with_duplicates = cvssl.complement_visual_contrastive_loss(v_a, v_b, mask, ids, tau_u=0.1,
                                                              duplicate_policy='exclude')
    without = cvssl.complement_visual_contrastive_loss(v_a, v_b, mask, ids, tau_u=0.1,
                                                      duplicate_policy='none')
    assert float(with_duplicates['duplicate_candidates']) > 0.0
    assert float(with_duplicates['loss']) != float(without['loss'])


def test_degenerate_negative_candidates_are_removed_from_the_denominator():
    anchor = torch.randn(1, 6)
    bank = torch.randn(3, 6)
    bank[2] = 0.0                                               # zero feature -> zero masked norm
    mask = torch.ones(1, 6)
    result = cvssl.cvssl_direction(anchor, bank, mask, torch.tensor([0]),
                                   torch.tensor([0]), torch.tensor([0, 1, 2]), tau_u=0.1)
    assert float(result['per_anchor_loss']) < float(torch.log(torch.tensor(3.0))) + 1.0
    assert float(result['mean_valid_negatives']) == 1.0


# --------------------------------------------------------------------------- #
# 5. arms
# --------------------------------------------------------------------------- #
def test_arm_masks_have_the_expected_geometry():
    m_s = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.5, 0.5]])
    ones, _ = cvssl.build_arm_mask(m_s, 'ones')
    assert float(ones.sum()) == 6.0
    comp, _ = cvssl.build_arm_mask(m_s, 'complement', rho=0.0)
    assert torch.allclose(comp, 1.0 - m_s)
    comp_rho, _ = cvssl.build_arm_mask(m_s, 'complement', rho=0.1)
    assert torch.allclose(comp_rho, 1.0 - 0.9 * m_s)
    random_mask, diagnostics = cvssl.build_arm_mask(m_s, 'random', generator=torch.Generator()
                                                   .manual_seed(0))
    assert diagnostics['mask_value_histogram_identical_to_complement']
    assert float(random_mask.sum()) == float((1.0 - m_s).sum())


def test_rho_one_is_global_vssl_under_the_same_definition():
    m_s = torch.rand(2, 5)
    mask, _ = cvssl.build_arm_mask(m_s, 'complement', rho=1.0)
    assert torch.allclose(mask, torch.ones_like(mask))
    ones, _ = cvssl.build_arm_mask(m_s, 'ones')
    assert torch.allclose(mask, ones)


def test_arm_registry_matches_the_pre_registered_table():
    assert ARM_MASK['S0_smartclip'] == 'none' and ARM_LAMBDA_U['S0_smartclip'] == 0.0
    assert ARM_MASK['G0_global_vssl'] == 'ones' and ARM_LAMBDA_U['G0_global_vssl'] == 1.0
    assert ARM_MASK['R0_random_vssl'] == 'random' and ARM_LAMBDA_U['R0_random_vssl'] == 1.0
    assert ARM_MASK['C0_complement_vssl'] == 'complement'
    assert ARM_LAMBDA_U['C0_complement_vssl'] == 1.0
    assert len(ARMS) == 4


def test_random_mask_rng_is_independent_and_reproducible():
    generator = torch.Generator().manual_seed(0)
    before = torch.rand(3, generator=generator)
    generator = torch.Generator().manual_seed(0)
    torch.rand(3, generator=torch.Generator().manual_seed(12345))    # unrelated stream
    after = torch.rand(3, generator=generator)
    assert torch.allclose(before, after)
    assert stateless_seed(0, 0, 5, 'view_b') == stateless_seed(0, 0, 5, 'view_b')
    assert stateless_seed(0, 0, 5, 'view_b') != stateless_seed(0, 1, 5, 'view_b')


# --------------------------------------------------------------------------- #
# 6. data layer
# --------------------------------------------------------------------------- #
def test_view_a_is_the_reference_preprocessing(tmp_path):
    from clip.clip import _transform as openai_transform_factory
    from PIL import Image
    transform = reference_view_a_transform()
    openai_transform = openai_transform_factory(224)
    generator = torch.Generator().manual_seed(0)
    raw = (torch.rand(260, 320, 3, generator=generator) * 255).byte().numpy()
    image = Image.fromarray(raw)
    a = transform(image)
    b = openai_transform(image)
    assert torch.equal(a, b)
    assert a.shape == (3, 224, 224)
    assert CLIP_MEAN[0] == pytest.approx(0.48145466) and CLIP_STD[0] == pytest.approx(0.26862954)


def test_view_b_changes_pixels_but_keeps_the_window():
    from PIL import Image
    window = Image.fromarray((torch.rand(224, 224, 3,
                                         generator=torch.Generator().manual_seed(1)) * 255
                              ).byte().numpy())
    view_b, parameters = shared_content_weak_v1(window, torch.Generator().manual_seed(0))
    to_tensor = reference_view_a_transform()
    a = to_tensor(window)
    b = to_tensor(view_b)
    assert b.shape == a.shape
    assert float((a - b).abs().mean()) > 0.0
    assert 192 <= parameters['resample_size'] <= 224
    assert 0.1 <= parameters['blur_sigma'] <= 1.0
    assert parameters['blur_kernel'] == 3
    assert parameters['resample_interpolation'] == 'bilinear'


def test_collate_and_image_ids():
    batch = [{'image_a': torch.zeros(3, 4, 4), 'image_b': torch.ones(3, 4, 4),
              'caption_said': 'x . y .', 'image_id': 7, 'sample_id': 1000, 'prefix_k': 1,
              'num_sentences': 2, 'view_b_resample_size': 200, 'view_b_blur_sigma': 0.5}] * 2
    collated = cvssl_collate(batch)
    assert collated['image_a'].shape == (2, 3, 4, 4)
    assert collated['caption_said'] == ['x . y .'] * 2
    assert collated['image_id'].tolist() == [7, 7]
    assert image_id_from_path('a/b.jpg') == image_id_from_path('a/b.jpg')
    assert image_id_from_path('a/b.jpg') != image_id_from_path('a/c.jpg')


def test_tokenizer_is_still_longclip_248():
    tokens = longclip.tokenize(['a cat on a mat'] * 2, truncate=True)
    assert tokens.shape[1] == 248


# --------------------------------------------------------------------------- #
# 7. objective surface
# --------------------------------------------------------------------------- #
def _code_without_prose(path):
    import tokenize
    pieces = []
    with open(path, encoding='utf-8') as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return ' '.join(pieces).lower()


def test_objective_rejects_unknown_arm_and_forbidden_ingredients():
    model, data, text = make_feature_model()
    with pytest.raises(ValueError):
        SaidClsCvsslObjective(model, arm='not_an_arm')
    source = _code_without_prose(os.path.join(REPO_ROOT, 'model', 'said_cls_cvssl.py'))
    for banned in ('identifiable_said_loss', 'finalblockclsreadout', 'said_router',
                   'exgap', 'forward_prefinal', 'encode_visual_prefinal'):
        assert banned not in source, banned
    data_source = _code_without_prose(os.path.join(REPO_ROOT, 'model',
                                                  'complement_visual_ssl.py'))
    for banned in ('queue', 'momentum', 'ema_', 'projection_head', 'predictor'):
        assert banned not in data_source, banned


def test_all_four_arms_run_and_log_the_required_fields():
    model, data, text = make_feature_model()
    required = ('loss_total', 'loss_smart', 'loss_sidm', 'loss_dism', 'loss_sparsity',
                'loss_vssl_raw', 'loss_vssl_weighted', 'sidm_top1', 'dism_top1',
                'actual_global_candidate_count', 'vssl_ab_top1', 'vssl_ba_top1',
                'positive_minus_negative_margin', 'vssl_valid_anchor_fraction',
                'invalid_norm_count', 'valid_negative_count', 'mask_s_keep_ratio',
                'mask_u_keep_ratio', 'cos_u_g', 'view_pixel_distance',
                'unmasked_crossview_cos')
    for arm in ARMS:
        objective = SaidClsCvsslObjective(model, arm=arm, lambda_u=ARM_LAMBDA_U[arm])
        out = objective(data['image_a'], data['image_b'], text, data['image_id'],
                        random_generator=torch.Generator().manual_seed(0))
        for key in required:
            assert key in out, (arm, key)
            assert torch.isfinite(torch.as_tensor(float(out[key]))), (arm, key)
        if ARM_LAMBDA_U[arm] > 0:
            assert float(out['loss_vssl_raw']) > 0.0
        else:
            assert abs(float(out['loss_total']) - float(out['loss_smart'])) < 1e-8
