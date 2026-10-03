"""Exact fusion contractions, branch ownership, detach and explicit global references."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from model.model_longclip import VisionTransformer, MaskNetwork
from model.nested_fusion_mask import (FusionBranch, NestedFusionMask, pair_logits,
                                     pool_summary, stack_logits, fusion_scores)
from model.nested_semantic_mask import hard_st, inclusion, inclusion_weight
from tests.test_nested_jointmask import TinyJointCLIP, TinyVisual
from train.train_nested_semantic_mask import build_optimizer


class TinyFusionCLIP(TinyJointCLIP):
    def __init__(self, width=8):
        super().__init__()
        if width != 8:
            self.visual = TinyVisual(hidden=width, output=width)
            self.token_embedding = nn.Embedding(31, width)
            self.text_projection = nn.Parameter(torch.randn(width, width) * .1)
            self.mask_net = MaskNetwork(width, layers=1, heads=2)
        self.visual.register_buffer('positional_embedding', torch.zeros(197, width))

    def encode_image(self, images, return_hidden=False, return_token_hidden=False):
        hidden = self.visual.hidden(images.float())
        z = hidden @ self.visual.proj
        if return_token_hidden:
            offsets = torch.arange(197, device=images.device).view(1, 197, 1) / 197
            return z, hidden[:, None] + offsets
        return (z, hidden) if return_hidden else z


def explicit_inputs(module, images, tokens):
    z, hidden = module.clip.encode_image(images, return_token_hidden=True)
    zv = module.fusion_branch.encode_visual(hidden[:, :1] if module.visual == 'cls' else hidden[:, 1:])
    text, ht = module.clip.encode_text(tokens, return_full=True)
    zt = module.clip.mask_net.resblocks(ht.detach().permute(1, 0, 2)).permute(1, 0, 2)
    return z, text, zv, zt


def explicit_logits(module, zv, zt):
    if module.fusion == 'balanced_stack':
        ut = module.clip.mask_net.attn_pool(zt)
        uv = module.clip.mask_net.attn_pool(zv)
        joined = torch.cat((ut[:, None].expand(-1, len(uv), -1),
                            uv[None].expand(len(ut), -1, -1)), dim=-1)
        gate = module.fusion_branch.gate(joined).sigmoid()
        return gate * ut[:, None] + (1 - gate) * uv[None]
    if module.fusion == 'stack_pool':
        paired = torch.cat((zv[None].expand(len(zt), -1, -1, -1),
                            zt[:, None].expand(-1, len(zv), -1, -1)), dim=-2)
        return module.clip.mask_net.attn_pool(paired.flatten(0, 1)).reshape(len(zt), len(zv), -1)
    branch = module.fusion_branch
    q, k = branch.query(zv), branch.key(zt)
    if module.fusion == 'cosine_crossscore':
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        relation = torch.einsum('iak,btk->biat', q, k)
    else:
        relation = torch.einsum('iak,btk->biat', q, k) / math.sqrt(branch.rank)
    return branch.readout(relation.flatten(-2))


def reference_loss(module, images, views, valid, completed):
    enabled_views = views if int(valid.sum()) >= 2 else views[:1]
    values, probabilities = [], []
    for index, tokens in enumerate(enabled_views):
        z, text, zv, zt = explicit_inputs(module, images, tokens)
        logits = explicit_logits(module, zv, zt)
        mask = hard_st(logits.sigmoid())
        scores = 100 * (F.normalize(z[None] * mask, dim=-1, eps=1e-6) *
                        F.normalize(text, dim=-1, eps=1e-6)[:, None]).sum(-1)
        enabled = torch.ones_like(valid) if index == 0 else valid
        labels = torch.arange(len(z), device=z.device)[enabled]
        align = (F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None], -torch.inf), labels) +
                 F.cross_entropy(scores[enabled].masked_fill(~enabled[None], -torch.inf), labels))
        positive = mask.diagonal(dim1=0, dim2=1).T
        soft = logits.sigmoid().diagonal(dim1=0, dim2=1).T
        values.append((align, positive[enabled].abs().mean()))
        probabilities.append(soft)
    if int(valid.sum()) < 2:
        return 10 * values[0][0] + values[0][1]
    return (10 / 3 * sum(x[0] for x in values) +
            (values[0][1] + 2 * values[1][1] + 2 * values[2][1]) / 3 +
            inclusion_weight('A3', completed) * inclusion(*probabilities)[valid].mean())


def make_model(fusion, visual='cls', width=8):
    return NestedFusionMask(TinyFusionCLIP(width), fusion=fusion, visual=visual,
                            checkpoint_encoders=False, image_chunk=2, text_chunk=2, text_tokens=6)


def test_visual_post_ln_unprojected_tokens_are_exact_and_native_unchanged():
    torch.manual_seed(702)
    model = VisionTransformer(8, 4, 8, 2, 2, 4).eval()
    images = torch.randn(2, 3, 8, 8)
    calls = []
    hook = model.transformer.resblocks[-1].register_forward_hook(lambda _, __, value: calls.append(value.detach()))
    native, tokens = model(images, return_token_hidden=True)
    hook.remove()
    assert len(calls) == 1 and tokens.shape == (2, 5, 8)
    expected = model.ln_post(calls[0].permute(1, 0, 2))
    torch.testing.assert_close(tokens, expected, atol=0, rtol=0)
    torch.testing.assert_close(native, tokens[:, 0] @ model.proj, atol=0, rtol=0)
    torch.testing.assert_close(native, model(images), atol=0, rtol=0)


@pytest.mark.parametrize('visual_tokens', [1, 3, 196])
def test_stack_logsumexp_decomposition_logit_and_gradient(visual_tokens):
    torch.manual_seed(706)
    v = torch.randn(2, visual_tokens, 8, dtype=torch.float64, requires_grad=True)
    t = torch.randn(3, 6, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    # This independent float64 reference checks the formula without downcasting production FP32 helpers.
    ev, et = F.linear(v, w, b).squeeze(-1), F.linear(t, w, b).squeeze(-1)
    sv = (ev.logsumexp(-1), (ev.softmax(-1)[..., None] * v).sum(-2))
    st = (et.logsumexp(-1), (et.softmax(-1)[..., None] * t).sum(-2))
    actual = stack_logits(sv, st)
    joined = torch.cat([v[None].expand(3, -1, -1, -1), t[:, None].expand(-1, 2, -1, -1)], -2)
    energy = F.linear(joined, w, b)
    expected = (energy.softmax(-2) * joined).sum(-2)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    gradients_a = torch.autograd.grad(actual.square().sum(), (v, t, w, b), retain_graph=True)
    gradients_b = torch.autograd.grad(expected.square().sum(), (v, t, w, b))
    for left, right in zip(gradients_a, gradients_b):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize('visual_tokens', [1, 3])
def test_crossscore_flatten_order_raw_qk_and_exact_contraction_adamw(visual_tokens):
    torch.manual_seed(709)
    branch = FusionBranch(nn.Identity(), 8, 8, 'crossscore_flat', visual_tokens, text_tokens=6, rank=4).double()
    reference = copy.deepcopy(branch)
    q = torch.randn(2, visual_tokens, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(3, 6, 4, dtype=torch.float64, requires_grad=True)
    raw = torch.einsum('iak,btk->biat', q, k) / 2
    explicit = reference.readout(raw.flatten(-2))
    if visual_tokens == 1:
        c = torch.einsum('dt,btk->bdk', branch.readout.weight, k) / 2
        actual = branch.cross_logits(q, c)
    else:
        weight = branch.readout.weight.reshape(8,visual_tokens,6)
        c = torch.einsum('dat,btk->bdak',weight,k) / 2
        actual = branch.cross_logits(q, c)
    torch.testing.assert_close(actual, explicit, atol=2e-14, rtol=2e-13)
    actual.square().sum().backward(retain_graph=True)
    explicit.square().sum().backward()
    left, right = dict(branch.named_parameters()), dict(reference.named_parameters())
    for name in left:
        assert (left[name].grad is None) == (right[name].grad is None), name
        if left[name].grad is not None:
            torch.testing.assert_close(left[name].grad, right[name].grad, atol=2e-12, rtol=2e-11, msg=name)
    torch.optim.AdamW(branch.parameters(), lr=1e-4, weight_decay=0).step()
    torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0).step()
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=2e-14, rtol=2e-13, msg=name)
    assert branch.readout.weight.count_nonzero() == branch.readout.weight.numel()
    assert branch.readout.bias.count_nonzero() == 0 or branch.readout.bias.grad is not None


@pytest.mark.parametrize('fusion,visual', [('stack_pool','cls'),('stack_pool','patch'),
                                         ('crossscore_flat','cls'),('crossscore_flat','patch')])
def test_branches_once_rng_isolated_detach_optimizer_groups(fusion, visual):
    torch.manual_seed(711)
    clip = TinyFusionCLIP()
    state = torch.get_rng_state().clone()
    model = NestedFusionMask(clip, fusion=fusion, visual=visual, text_tokens=6,
                             checkpoint_encoders=False, image_chunk=2, text_chunk=2)
    assert torch.equal(state, torch.get_rng_state())
    source_ids = {id(x) for x in clip.mask_net.resblocks.parameters()}
    target_ids = {id(x) for x in model.fusion_branch.visual_blocks.parameters()}
    assert source_ids.isdisjoint(target_ids)
    calls = dict(text=0, visual=0, image=0)
    hooks = [clip.mask_net.resblocks.register_forward_hook(lambda *_: calls.__setitem__('text',calls['text']+1)),
             model.fusion_branch.visual_blocks.register_forward_hook(lambda *_:calls.__setitem__('visual',calls['visual']+1))]
    images = torch.randn(3, 8, requires_grad=True)
    tokens = [torch.randint(0,31,(3,6)) for _ in range(3)]
    z, vv = model.encode_visual(images)
    text, tt = model.encode_view(tokens[0])
    pair_logits(model, vv, tt).sum().backward()
    assert images.grad is None and clip.token_embedding.weight.grad is None
    model.zero_grad(set_to_none=True)
    calls['text'] = calls['visual'] = 0
    loss, logs = model(images, *tokens, torch.ones(3,dtype=torch.bool), 41)
    assert calls['text'] == 3 and calls['visual'] == 1
    loss.backward()
    for hook in hooks:hook.remove()
    assert model.fusion_branch.visual_adapter.weight.grad is not None
    assert any(p.grad is not None for p in model.fusion_branch.visual_blocks.parameters())
    if fusion == 'crossscore_flat':
        assert not clip.mask_net.attn_pool.attention.weight.requires_grad
        assert clip.mask_net.attn_pool.attention.weight.grad is None
    optimizer = build_optimizer(model)
    groups = {g['name']:{id(p) for p in g['params']} for g in optimizer.param_groups}
    assert target_ids <= groups['shared_mask']
    assert id(model.fusion_branch.visual_adapter.weight) in groups['joint_adapter']
    flat = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(flat) == len(set(flat))
    assert torch.isfinite(loss) and logs['inc_weight'] == 41/200


@pytest.mark.parametrize('fusion,visual', [('stack_pool','cls'),('stack_pool','patch'),
                                         ('crossscore_flat','cls'),('crossscore_flat','patch')])
def test_score_diagonals_regularizers_explicit_reference_labels_and_conditions(fusion, visual):
    torch.manual_seed(719)
    model = make_model(fusion, visual)
    images = torch.randn(3, 8)
    views = [torch.randint(0,31,(3,6)) for _ in range(3)]
    z, vv = model.encode_visual(images)
    t, tt = model.encode_view(views[0])
    logits = pair_logits(model, vv, tt)
    direct = explicit_logits(model, *explicit_inputs(model, images, views[0])[2:])
    torch.testing.assert_close(logits, direct, atol=3e-6, rtol=3e-5)
    positive = pair_logits(model, vv, tt, paired=True)
    torch.testing.assert_close(positive, logits.diagonal(dim1=0,dim2=1).T, atol=3e-6, rtol=3e-5)
    alternate = pair_logits(model, tuple(x.roll(1,0) for x in vv), tt)
    assert not torch.equal(logits, alternate)
    changed = pair_logits(model, vv, tuple(x.roll(1,0) for x in tt))
    assert not torch.equal(logits, changed)
    valid = torch.tensor([1,0,1],dtype=torch.bool)
    loss, _ = model(images,*views,valid,61)
    expected = reference_loss(model,images,views,valid,61)
    torch.testing.assert_close(loss,expected,atol=3e-4,rtol=3e-5)
    scores,_,_=fusion_scores(model,z,t,vv,tt,torch.ones_like(valid),torch.ones_like(valid))
    first = F.cross_entropy(scores,torch.arange(3))
    second = F.cross_entropy(scores,torch.tensor([1,0,2]))
    assert torch.isfinite(first + second)
    after,_,_=fusion_scores(model,z,t,vv,tt,torch.ones_like(valid),torch.ones_like(valid))
    torch.testing.assert_close(scores,after,atol=0,rtol=0)


def test_shared_initialization_across_four_arms():
    torch.manual_seed(727)
    clip = TinyFusionCLIP()
    models = [NestedFusionMask(copy.deepcopy(clip),fusion=f,visual=v,text_tokens=6,checkpoint_encoders=False)
              for f,v in [('stack_pool','cls'),('crossscore_flat','cls'),('stack_pool','patch'),('crossscore_flat','patch')]]
    for model in models[1:]:
        torch.testing.assert_close(model.fusion_branch.visual_adapter.weight,models[0].fusion_branch.visual_adapter.weight,atol=0,rtol=0)
        for key,value in model.fusion_branch.visual_blocks.state_dict().items():
            torch.testing.assert_close(value,models[0].fusion_branch.visual_blocks.state_dict()[key],atol=0,rtol=0)
    torch.testing.assert_close(models[1].fusion_branch.query.weight,models[3].fusion_branch.query.weight,atol=0,rtol=0)
    torch.testing.assert_close(models[1].fusion_branch.key.weight,models[3].fusion_branch.key.weight,atol=0,rtol=0)


@pytest.mark.parametrize('fusion,visual', [('stack_pool','cls'),('crossscore_flat','cls'),
                                         ('stack_pool','patch'),('crossscore_flat','patch')])
def test_nonreentrant_checkpoint_matches_named_gradients_and_bare_state(fusion, visual):
    torch.manual_seed(733)
    plain=make_model(fusion,visual)
    remat=copy.deepcopy(plain)
    remat.checkpoint_pair_blocks=True
    images=torch.randn(3,8)
    views=[torch.randint(0,31,(3,6)) for _ in range(3)]
    valid=torch.tensor([1,0,1],dtype=torch.bool)
    loss_plain,_=plain(images,*views,valid,61)
    loss_remat,_=remat(images,*views,valid,61)
    loss_plain.backward()
    loss_remat.backward()
    torch.testing.assert_close(loss_plain,loss_remat,atol=0,rtol=0)
    a,b=dict(plain.named_parameters()),dict(remat.named_parameters())
    for name in a:
        assert (a[name].grad is None)==(b[name].grad is None),name
        if a[name].grad is not None:
            torch.testing.assert_close(a[name].grad,b[name].grad,atol=0,rtol=0,msg=name)
    bare=TinyFusionCLIP()
    bare.load_state_dict(plain.clip.state_dict(),strict=True)
    torch.testing.assert_close(bare.encode_image(images),plain.clip.encode_image(images),atol=0,rtol=0)
    torch.testing.assert_close(bare.encode_text(views[0]),plain.clip.encode_text(views[0]),atol=0,rtol=0)


def test_full_patch_parameter_count_and_initial_bias_are_frozen_definition():
    torch.manual_seed(739)
    branch=FusionBranch(nn.Identity(),768,512,'crossscore_flat',196,248,64)
    assert branch.readout.weight.shape==(512,48608)
    assert branch.readout.weight.numel()==24887296
    assert branch.readout.bias.shape==(512,) and branch.readout.bias.count_nonzero()==0
    assert float(branch.readout.weight.abs().max())<=1/math.sqrt(48608)
