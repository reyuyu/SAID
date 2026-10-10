"""Reuse native read-only gradient audit, restoring the recorded CE scale."""
import sys
import torch
from model.balanced_hparam_search import BalancedSearch
from recovery import hns_macro_gradient as audit


def main():
    checkpoint=sys.argv[sys.argv.index('--checkpoint')+1]
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    scale=payload['config']['contrastive_logit_scale'];del payload
    original=BalancedSearch.__init__
    def init(module,*args,**kwargs):
        return original(module,*args,contrastive_logit_scale=scale,**kwargs)
    BalancedSearch.__init__=init
    audit.main()


if __name__=='__main__':main()
