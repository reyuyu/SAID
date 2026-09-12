import torch
from model.said_cls_reconstruction import reconstruction_alpha

def test_vwarm_absolute_schedule():
    assert reconstruction_alpha(0) == 0
    assert reconstruction_alpha(99) == 0
    assert reconstruction_alpha(100) == 0.01
    assert reconstruction_alpha(198) == 0.99
    assert reconstruction_alpha(200) == 1
    assert reconstruction_alpha(500) == 1

def test_un_normalization_equivalence_when_nonzero():
    v = torch.tensor([[3., 4., 0.]])
    m = torch.tensor([[1., 0., 1.]])
    old = torch.nn.functional.normalize(torch.nn.functional.normalize(v, dim=-1) * m, dim=-1)
    new = torch.nn.functional.normalize(v * m, dim=-1)
    assert torch.allclose(old, new)

def test_un_masked_coordinate_has_zero_direct_gradient():
    v = torch.tensor([[3., 4., 5.]], requires_grad=True)
    m = torch.tensor([[1., 0., 1.]])
    u = torch.nn.functional.normalize(v * m, dim=-1)
    u.sum().backward()
    assert v.grad[0, 1].item() == 0
