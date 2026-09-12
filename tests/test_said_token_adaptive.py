import torch
from model.said_token_v1 import hard_gate, hard_from_grid

def test_adaptive_counts():
    x=torch.tensor([[-10.,-1.,0.,1.,10.],[0.,0.,0.,0.,0.]])
    g=hard_gate(x,mode='adaptive_sigmoid',threshold=.5)
    assert g.sum(-1).tolist()==[3.,5.]

def test_threshold_boundary():
    x=torch.tensor([[0.]])
    assert hard_gate(x,mode='adaptive_sigmoid',threshold=.5).item()==1

def test_k0_negative_masked_max():
    c=torch.tensor([[[[-.8,-.7],[-.9,-.6],[-.95,-.85]]]])
    gate=torch.zeros(1,1,2)
    valid=torch.ones(1,1,2,dtype=torch.bool)
    s,_,_=hard_from_grid(c,gate,valid)
    assert torch.isfinite(s).all() and s.item() < -1.0

