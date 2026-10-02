"""L14 dimensions with tiny native encoders and production mask transformer blocks."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from model.balanced_hparam_search import BalancedSearch
from model.model_longclip import MaskNetwork
from model.nested_fusion_mask import NestedFusionMask, fusion_scores, pair_logits
from model.nested_semantic_mask import hard_st
from tests.test_nested_fusion import TinyFusionCLIP, reference_loss
from train.train_nested_semantic_mask import build_optimizer, optimizer_learning_rates


class L14Visual(nn.Module):
    def __init__(self, hidden=1024, native=768, patches=256):
        super().__init__()
        self.hidden = nn.Linear(8, hidden)
        self.proj = nn.Parameter(torch.randn(hidden, native) * hidden ** -.5)
        self.positional_embedding = nn.Parameter(torch.randn(patches + 1, hidden) * .05)

    def forward(self, images, return_hidden=False, return_token_hidden=False):
        tokens = self.hidden(images.float())[:, None] + self.positional_embedding[None]
        native = tokens[:, 0] @ self.proj
        if return_token_hidden:
            return native, tokens
        return (native, tokens[:, 0]) if return_hidden else native


class L14FixtureCLIP(nn.Module):
    """Dimension-aware fixture; native scores must use both encode_* methods."""
    def __init__(self, visual_hidden=1024, text_hidden=768, native=768,
                 patches=256, mask_heads=12):
        super().__init__()
        self.visual = L14Visual(visual_hidden, native, patches)
        self.token_embedding = nn.Embedding(31, text_hidden)
        self.text_projection = nn.Parameter(torch.randn(text_hidden, native) * text_hidden ** -.5)
        self.mask_net = MaskNetwork(text_hidden, layers=1, heads=mask_heads)

    def encode_image(self, images, return_hidden=False, return_token_hidden=False):
        return self.visual(images, return_hidden=return_hidden,
                           return_token_hidden=return_token_hidden)

    def encode_text(self, tokens, return_full=False):
        hidden = self.token_embedding(tokens)
        pooled = hidden[torch.arange(len(tokens), device=tokens.device), tokens.argmax(-1)]
        native = pooled @ self.text_projection
        return (native, hidden) if return_full else native


def make_l14_model(*, search=True, search_hparams=None, fusion='balanced_stack',
                   image_chunk=2, text_chunk=2, checkpoint_pair_blocks=False):
    options = dict(fusion=fusion, visual='patch', text_tokens=6,
                   checkpoint_encoders=False, image_chunk=image_chunk, text_chunk=text_chunk,
                   checkpoint_pair_blocks=checkpoint_pair_blocks)
    if search:
        return BalancedSearch(L14FixtureCLIP(), search_hparams=search_hparams, **options)
    return NestedFusionMask(L14FixtureCLIP(), **options)


def l14_inputs(batch=3, device='cpu'):
    return (torch.randn(batch, 8, device=device),
            [torch.randint(0, 31, (batch, 6), device=device) for _ in range(3)])


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('fusion', ['balanced_stack', 'crossscore_flat'])
def test_l14_dimensions_native_outputs_and_actual_patch_count(fusion):
    torch.manual_seed(1401)
    module = make_l14_model(search=False, fusion=fusion)
    images, views = l14_inputs(2)
    with torch.no_grad():
        z, hidden = module.clip.encode_image(images, return_token_hidden=True)
        text, ht = module.clip.encode_text(views[0], return_full=True)
        assert hidden.shape == (2, 257, 1024) and ht.shape == (2, 6, 768)
        assert z.shape == text.shape == (2, 768)
        torch.testing.assert_close(z, module.clip.encode_image(images), atol=0, rtol=0)
        torch.testing.assert_close(text, module.clip.encode_text(views[0]), atol=0, rtol=0)
        assert module.fusion_branch.visual_tokens == 256
        assert module.fusion_branch.visual_adapter.weight.shape == (768, 1024)
        for blocks in (module.clip.mask_net.resblocks, module.fusion_branch.visual_blocks):
            assert len(blocks) == 1
            assert blocks[0].attn.embed_dim == 768 and blocks[0].attn.num_heads == 12
        lengths = []
        hook = module.fusion_branch.visual_blocks.register_forward_pre_hook(
            lambda _, args: lengths.append(args[0].shape[0]))
        try:
            _, vv = module.encode_visual(images)
            _, tt = module.encode_view(views[0])
            assert pair_logits(module, vv, tt).shape == (2, 2, 768)
        finally:
            hook.remove()
        assert lengths == [256]
        if fusion == 'crossscore_flat':
            assert module.fusion_branch.readout.weight.shape == (768, 256 * 6)


def test_fixture_without_positional_embedding_retains_196_patch_fallback():
    module = NestedFusionMask(TinyFusionCLIP(), fusion='stack_pool', visual='patch',
                              text_tokens=6, checkpoint_encoders=False)
    assert not hasattr(module.clip.visual, 'positional_embedding')
    assert module.fusion_branch.visual_tokens == 196


def test_l14_independent_branches_shared_pool_once_and_half_gate():
    torch.manual_seed(1403)
    clip = L14FixtureCLIP()
    rng = torch.get_rng_state().clone()
    module = BalancedSearch(clip, fusion='balanced_stack', visual='patch', text_tokens=6,
                            checkpoint_encoders=False, image_chunk=2, text_chunk=2)
    assert torch.equal(rng, torch.get_rng_state())
    text = dict(clip.mask_net.resblocks.named_parameters())
    visual = dict(module.fusion_branch.visual_blocks.named_parameters())
    assert text.keys() == visual.keys()
    for name in text:
        assert text[name].data_ptr() != visual[name].data_ptr(), name
        torch.testing.assert_close(text[name], visual[name], atol=0, rtol=0, msg=name)
    name = next(iter(text))
    before = text[name].detach().clone()
    with torch.no_grad():
        visual[name].add_(.125)
    torch.testing.assert_close(text[name], before, atol=0, rtol=0)
    pool = clip.mask_net.attn_pool
    assert [name for name, child in module.named_modules(remove_duplicate=False)
            if child is pool] == ['clip.mask_net.attn_pool']
    aliases = list(module.named_parameters(remove_duplicate=False))
    for name, parameter in pool.named_parameters():
        assert [n for n, p in aliases if p is parameter] == ['clip.mask_net.attn_pool.' + name]
        assert sum(p is parameter for p in module.mask_parameters()) == 1
    optimizer = build_optimizer(module)
    assert [g['name'] for g in optimizer.param_groups] == [
        'backbone', 'text_mask_and_shared_pool', 'visual_mask', 'fusion_adapter']
    flat = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(flat) == len(set(flat)) == sum(p.requires_grad for p in module.parameters())
    groups = {g['name']: {id(p) for p in g['params']} for g in optimizer.param_groups}
    assert {id(p) for p in pool.parameters()} <= groups['text_mask_and_shared_pool']
    assert {id(p) for p in visual.values()} <= groups['visual_mask']
    assert id(module.fusion_branch.gate.weight) in groups['fusion_adapter']
    assert id(module.fusion_branch.visual_adapter.weight) in groups['fusion_adapter']
    images, views = l14_inputs(2)
    with torch.no_grad():
        _, vv = module.encode_visual(images)
        _, tt = module.encode_view(views[0])
        assert module.fusion_branch.gate.weight.shape == (768, 1536)
        for paired in (False, True):
            gate = module.fusion_branch.balanced_gate(vv[1], tt[1], paired=paired)
            torch.testing.assert_close(gate, torch.full_like(gate, .5), atol=0, rtol=0)
        logits = pair_logits(module, vv, tt)
        torch.testing.assert_close(logits, .5 * (tt[1][:, None] + vv[1][None]), atol=0, rtol=0)
        assert not torch.equal(logits, pair_logits(module, tuple(x.roll(1, 0) for x in vv), tt))
        assert not torch.equal(logits, pair_logits(module, vv, tuple(x.roll(1, 0) for x in tt)))


def test_l14_conditions_detach_but_native_scoring_trains_backbone():
    torch.manual_seed(1409)
    module = make_l14_model()
    images, views = l14_inputs(2)
    images.requires_grad_(True)
    _, vv = module.encode_visual(images)
    _, tt = module.encode_view(views[0])
    pair_logits(module, vv, tt).square().mean().backward()
    assert images.grad is None
    backbone = [p for name, p in module.clip.named_parameters() if not name.startswith('mask_net.')]
    assert all(p.grad is None for p in backbone)
    for part in (module.fusion_branch.visual_adapter, module.fusion_branch.visual_blocks,
                 module.clip.mask_net.resblocks, module.clip.mask_net.attn_pool, module.fusion_branch.gate):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    module.zero_grad(set_to_none=True)
    calls = dict(visual=0, text=0)
    hooks = [module.fusion_branch.visual_blocks.register_forward_hook(
        lambda *_: calls.__setitem__('visual', calls['visual'] + 1)),
        module.clip.mask_net.resblocks.register_forward_hook(
        lambda *_: calls.__setitem__('text', calls['text'] + 1))]
    try:
        loss, _ = module(images, *views, torch.ones(2, dtype=torch.bool), 61)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == dict(visual=1, text=3)
    assert images.grad is not None and images.grad.abs().sum() > 0
    for p in backbone:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize('validity', [(1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)])
def test_l14_default_objective_unchanged_and_matches_explicit_reference(validity):
    torch.manual_seed(1423)
    old = make_l14_model(search=False)
    new = BalancedSearch(copy.deepcopy(old.clip), fusion='balanced_stack', visual='patch',
                         text_tokens=6, checkpoint_encoders=False, image_chunk=2, text_chunk=2)
    assert old.state_dict().keys() == new.state_dict().keys()
    images, views = l14_inputs()
    valid = torch.tensor(validity, dtype=torch.bool)
    lo, logs_old = old(images, *views, valid, 61)
    ln, logs_new = new(images, *views, valid, 61)
    with torch.no_grad():
        expected = reference_loss(new, images, views, valid, 61)
    torch.testing.assert_close(ln, expected, atol=3e-4, rtol=3e-5)
    torch.testing.assert_close(lo, ln, atol=0, rtol=0)
    assert logs_old['valid_global'] == logs_new['valid_global'] == sum(validity)
    assert logs_new['inc_weight'] == (61 / 200 if sum(validity) >= 2 else 0.)
    lo.backward()
    ln.backward()
    named = dict(new.named_parameters())
    for name, p in old.named_parameters():
        q = named[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('images_count,texts_count', [(5, 3), (1, 1)])
def test_l14_uneven_chunks_and_singleton_tail_scores_and_gradients(images_count, texts_count):
    torch.manual_seed(1427)
    module = make_l14_model(checkpoint_pair_blocks=True)
    with torch.no_grad():
        module.fusion_branch.gate.weight.normal_(std=.003)
        images, _ = l14_inputs(images_count)
        tokens = torch.randint(0, 31, (texts_count, 6))
        z, vv = module.encode_visual(images)
        text, tt = module.encode_view(tokens)
    z, text = z.detach().requires_grad_(), text.detach().requires_grad_()
    vv = tuple(x.detach().requires_grad_() for x in vv)
    tt = tuple(x.detach().requires_grad_() for x in tt)
    vi = torch.arange(images_count) % 2 == 0
    vt = torch.arange(texts_count) % 2 == 0
    actual, summary, diagnostics = fusion_scores(module, z, text, vv, tt, vi, vt)
    ut = tt[1][:, None].expand(-1, images_count, -1)
    uv = vv[1][None].expand(texts_count, -1, -1)
    gate = module.fusion_branch.gate(torch.cat((ut, uv), -1)).sigmoid()
    mask = hard_st((gate * ut + (1 - gate) * uv).sigmoid())
    expected = 100 * (F.normalize(z[None] * mask, dim=-1, eps=1e-6) *
                      F.normalize(text, dim=-1, eps=1e-6)[:, None]).sum(-1)
    assert actual.shape == (texts_count, images_count)
    torch.testing.assert_close(actual, expected, atol=8e-5, rtol=3e-6)
    selected = mask.detach().bool()
    active = vt[:, None] & vi[None]
    expected_summary = torch.stack((selected.masked_fill(~active[..., None], False).float().sum(),
                                    active.sum() * 768,
                                    (selected.all(-1) & active).float().sum(),
                                    ((~selected).all(-1) & active).float().sum()))
    torch.testing.assert_close(summary, expected_summary, atol=0, rtol=0)
    assert diagnostics.count_nonzero() == 0
    variables = (z, text, *vv, *tt, module.fusion_branch.gate.weight)
    weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) + 1
    ga = torch.autograd.grad((actual * weights).mean(), variables, allow_unused=True)
    gb = torch.autograd.grad((expected * weights).mean(), variables, allow_unused=True)
    for index, (left, right) in enumerate(zip(ga, gb)):
        assert (left is None) == (right is None), index
        if left is not None:
            torch.testing.assert_close(left, right, atol=1.5e-3, rtol=1e-4, msg=str(index))


def test_l14_save_restore_next_adamw_and_rng_equivalent(tmp_path):
    random.seed(1433)
    np.random.seed(1433)
    torch.manual_seed(1433)
    module = make_l14_model()
    optimizer = build_optimizer(module)
    loader = torch.Generator().manual_seed(19)

    def rng_state():
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    cpu=torch.get_rng_state(), loader=loader.get_state())

    def step(model, opt, completed):
        images, views = l14_inputs(2)
        images = images + random.random() + float(np.random.random())
        order = torch.randperm(2, generator=loader)
        images, views = images[order], [v[order] for v in views]
        opt.zero_grad(set_to_none=True)
        for group, rate in zip(opt.param_groups, optimizer_learning_rates(model, completed, 4868)):
            group['lr'] = rate
        loss, _ = model(images, *views, torch.ones(2, dtype=torch.bool), completed)
        loss.backward()
        opt.step()
        return loss.detach(), images, views

    step(module, optimizer, 0)
    path = tmp_path / 'l14-resume.pt'
    torch.save(dict(model=module.clip.state_dict(), adapter=module.fusion_branch.state_dict(),
                    optimizer=optimizer.state_dict(), rng=rng_state(), completed_steps=1,
                    scheduler_horizon=4868), path)
    uninterrupted = step(module, optimizer, 1)
    after = rng_state()
    resumed = make_l14_model()
    resumed_optimizer = build_optimizer(resumed)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    resumed.clip.load_state_dict(payload['model'], strict=True)
    resumed.fusion_branch.load_state_dict(payload['adapter'], strict=True)
    resumed_optimizer.load_state_dict(payload['optimizer'])
    state = payload['rng']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    loader.set_state(state['loader'])
    assert payload['scheduler_horizon'] == 4868
    completed = payload['completed_steps']
    del payload
    replay = step(resumed, resumed_optimizer, completed)
    for left, right in zip((uninterrupted[0], uninterrupted[1], *uninterrupted[2]),
                           (replay[0], replay[1], *replay[2])):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    replay_rng = rng_state()
    assert after['python'] == replay_rng['python']
    assert after['numpy'][0] == replay_rng['numpy'][0]
    np.testing.assert_array_equal(after['numpy'][1], replay_rng['numpy'][1])
    assert after['numpy'][2:] == replay_rng['numpy'][2:]
    for name in ('cpu', 'loader'):
        assert torch.equal(after[name], replay_rng[name])
    target = dict(resumed.named_parameters())
    for name, p in module.named_parameters():
        q = target[name]
        torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
        left, right = optimizer.state[p], resumed_optimizer.state[q]
        assert left.keys() == right.keys(), name
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=name + '.' + key)
    for left, right in zip(optimizer.param_groups, resumed_optimizer.param_groups):
        assert {k: v for k, v in left.items() if k != 'params'} == {
            k: v for k, v in right.items() if k != 'params'}
