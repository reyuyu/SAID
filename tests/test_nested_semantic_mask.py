"""Focused method tests; the real smoke is deliberately a separate four-GPU job."""
import copy
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from model.model_longclip import MaskNetwork
from model.nested_semantic_mask import (NestedSemanticMask, inclusion, inclusion_weight,
                                        masked_scores, view_terms)
from model.said_cls_cvssl import said_mask_from_hidden, compute_smartclip_terms
from train.nested_semantic_data import text_views, collate
from train.train_nested_semantic_mask import build_optimizer, learning_rates


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(8, 8)
        self.token_embedding = nn.Embedding(30, 8)
        self.text_projection = nn.Parameter(torch.randn(8, 8) * .1)
        self.mask_net = MaskNetwork(8, layers=1, heads=2)

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens, return_full=False):
        h = self.token_embedding(tokens)
        t = h.mean(1) @ self.text_projection
        return (t, h) if return_full else t


def literal_scores(z, t, m):
    return 100 * (F.normalize(z[:, None] * m[None], dim=-1, eps=1e-6) *
                  F.normalize(t, dim=-1, eps=1e-6)[None]).sum(-1)


def global_reference(clip, images, tokens, valid, arm, s):
    z = clip.encode_image(images)
    values = []
    probabilities = []
    for k, tok in enumerate(tokens if int(valid.sum()) >= 2 else tokens[:1]):
        t, h = clip.encode_text(tok, return_full=True)
        m, p, _ = said_mask_from_hidden(clip.mask_net, h)
        probabilities.append(p)
        selected = torch.ones_like(valid) if k == 0 else valid
        scores = literal_scores(z[selected], t[selected], m[selected])
        labels = torch.arange(len(scores), device=z.device)
        align = F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)
        values.append((align, m[selected].abs().mean()))
    if int(valid.sum()) < 2:
        return 10 * values[0][0] + values[0][1]
    inc = inclusion(*probabilities)[valid].mean()
    return 10/3*sum(x[0] for x in values) + (values[0][1]+2*values[1][1]+2*values[2][1])/3 + inclusion_weight(arm,s)*inc


def test_text_views():
    v = text_views(' First. \nSecond. Last sentence. ')
    assert v['views'] == ['First. Second. Last sentence.', 'First', 'Second. Last sentence.']
    assert v['valid'] and v['views'][0] == '. '.join(v['views'][1:])
    assert not text_views('Only one sentence.')['valid']
    assert text_views('word '*300 + '. tail')['reason'] == 'first_segment_overlong'
    with pytest.raises(ValueError, match='empty'):
        text_views(' \n ')
    v = text_views('Short. ' + 'word '*230 + '. ' + 'tail '*100)
    assert v['valid'] and max(v['untruncated_lengths']) <= 248
    assert 'tail' not in v['views'][0]
    assert v['views'][2] in v['views'][0]
    assert v['tokens_f'].shape == (248,)
    # Raw (not truncated) lengths are checked independently.
    from model.longclip import _tokenizer
    assert v['untruncated_lengths'] == [len(_tokenizer.encode(s))+2 for s in v['views']]
    assert text_views('word '*246)['reason'] == 'single_visible_segment'
    assert text_views('word '*247)['reason'] == 'first_segment_overlong'


def test_collate_preserves_sample_views():
    import json
    samples=[dict(image=torch.zeros(3,2,2),image_id=i,sample_id=i,**text_views(s))
             for i,s in enumerate(('One. Two. End.', 'Single.'))]
    batch=collate(samples)
    assert batch['views'] == [s['views'] for s in samples]
    assert batch['valid'].tolist() == [True,False]
    json.dumps(dict(views=batch['views'],lengths=batch['untruncated_lengths']))


def test_inclusion_gradient_and_schedule():
    pf, po, pe = [torch.tensor([[v, v]], requires_grad=True) for v in (.1, .8, .6)]
    inc = inclusion(pf, po, pe).mean()
    inc.backward()
    assert (pf.grad < 0).all() and po.grad is None and pe.grad is None
    assert inclusion(torch.ones_like(pf), po, pe).item() == 0
    assert [inclusion_weight('A3', s) for s in (0,200,499)] == [0,1,1]
    assert [inclusion_weight('A2', s) for s in (0,200,499)] == [0,0,0]
    assert learning_rates(0,3651) == (5e-9, .001)
    assert learning_rates(200,3651)[0] == 1e-6
    assert learning_rates(499,3651)[1] > 0


def test_closed_and_open_masks_finite():
    z, t = torch.randn(3,8,requires_grad=True), torch.randn(3,8,requires_grad=True)
    m = torch.zeros(3,8,requires_grad=True)
    masked_scores(z,t,m).sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (z,t,m))
    torch.testing.assert_close(masked_scores(z,t,torch.ones_like(m)), literal_scores(z,t,torch.ones_like(m)), atol=3e-5,rtol=2e-6)


def test_shared_parameters():
    m = NestedSemanticMask(TinyCLIP(), checkpoint_encoders=False)
    all_named = list(m.named_parameters(remove_duplicate=False))
    assert len(all_named) == len({id(v) for _,v in all_named})
    optimizer = build_optimizer(m)
    assert len(optimizer.param_groups) == 2
    params = [p for g in optimizer.param_groups for p in g['params']]
    assert len(params) == len({id(x) for x in params})


def test_original_smartclip_regression():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group('gloo', init_method='file://'+tmp+'/pg', rank=0, world_size=1)
        try:
            old = TinyCLIP()
            new = copy.deepcopy(old)
            images, tokens = torch.randn(4,8), torch.randint(0,30,(4,6))
            # SGD exposes gradient scaling directly. Adam additionally has a
            # documented FP32 sensitivity for attention key-bias null directions.
            opts = [torch.optim.SGD(m.parameters(), lr=1e-4) for m in (old,new)]
            results=[]
            for m, opt, original in zip((old,new),opts,(True,False)):
                z=m.encode_image(images); t,h=m.encode_text(tokens,return_full=True)
                mask,_,_=said_mask_from_hidden(m.mask_net,h)
                if original:
                    terms=compute_smartclip_terms(z,t,mask,0)
                    loss=terms['loss_smart']; align=terms['loss_sidm']+terms['loss_dism']
                else:
                    valid=torch.ones(4,dtype=torch.bool)
                    align,sparse,_=view_terms(z,t,mask,valid,valid)
                    loss=10*align+2*sparse
                loss.backward()
                results.append((loss.detach(),align.detach(),mask.detach(),[p.grad.clone() for p in m.parameters()]))
                opt.step()
            for a,b in zip(results[0][:3], results[1][:3]):
                torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-6)
            for a,b in zip(results[0][3],results[1][3]):
                torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-5)
            for a,b in zip(old.parameters(),new.parameters()):
                torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
            sgd_error=max(float((a-b).abs().max()) for a,b in zip(old.parameters(),new.parameters()))
            for m in (old,new):
                torch.optim.AdamW(m.parameters(),lr=1e-4,eps=1e-8).step()
            adam_error=0.
            for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
                adam_error=max(adam_error,float((a-b).abs().max()))
                if name.endswith('attn.in_proj_bias'):
                    width=a.numel()//3
                    assert max(float(a.grad[width:2*width].abs().max()),float(b.grad[width:2*width].abs().max())) < 1e-4
                    torch.testing.assert_close(a[width:2*width],b[width:2*width],atol=2.01e-4,rtol=0)
                    torch.testing.assert_close(a[:width],b[:width],atol=5e-6,rtol=3e-5)
                    torch.testing.assert_close(a[2*width:],b[2*width:],atol=5e-6,rtol=3e-5)
                elif name == 'mask_net.attn_pool.attention.bias':
                    assert max(float(a.grad.abs().max()),float(b.grad.abs().max())) < 1e-4
                    torch.testing.assert_close(a,b,atol=2.01e-4,rtol=0)
                else:
                    torch.testing.assert_close(a,b,atol=5e-6,rtol=3e-5,msg=name)
            import json
            print(json.dumps(dict(test='original_smartclip',
                                  loss_error=float((results[0][0]-results[1][0]).abs()),
                                  alignment_error=float((results[0][1]-results[1][1]).abs()),
                                  mask_max_error=float((results[0][2]-results[1][2]).abs().max()),
                                  gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(results[0][3],results[1][3])),
                                  sgd_update_max_error=sgd_error,adamw_update_max_error=adam_error)))
        finally:
            dist.destroy_process_group()
