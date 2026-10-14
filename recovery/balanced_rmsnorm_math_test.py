#!/usr/bin/env python3
"""CPU-only formula test for BalancedStack-RMSNorm."""
import torch


def rmsnorm(x):
    x = x.float()
    return x / x.square().mean(-1, keepdim=True).add(1e-6).sqrt()


def main():
    torch.manual_seed(0)
    v = torch.randn(7, 512)
    t = torch.randn(5, 512)
    wv = torch.randn(512, 512)
    wt = torch.randn(512, 512)
    nv, nt = rmsnorm(v), rmsnorm(t)
    assert torch.isfinite(nv).all() and torch.isfinite(nt).all()
    assert torch.allclose(nv.square().mean(-1), torch.ones(7), atol=2e-5)
    assert torch.allclose(nt.square().mean(-1), torch.ones(5), atol=2e-5)
    g = torch.sigmoid(nt[:, None] @ wt.T + nv[None] @ wv.T)
    logits = g * nt[:, None] + (1 - g) * nv[None]
    assert logits.shape == (5, 7, 512)
    assert torch.isfinite(logits).all()
    print('{"passed":true,"shape":[5,7,512],"eps":1e-6,"affine":false}')


if __name__ == '__main__':
    main()
