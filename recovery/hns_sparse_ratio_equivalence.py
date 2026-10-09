"""Fixed-state coefficient audit for the HNS-S12 sparse-ratio ablation."""
import hashlib
import json
import math
from pathlib import Path

ARMS={
    'E1-AlignMatched':[2.25,2.25,.50],
    'E2-Uniform':[5./3.,5./3.,5./3.],
}


def audit(output):
    for name,weights in ARMS.items():
        assert len(weights)==3 and all(math.isfinite(x) and x>0 for x in weights)
        assert math.isclose(sum(weights),5.,rel_tol=0.,abs_tol=1e-12)
        effective=[1.2*x for x in weights]
        assert math.isclose(sum(effective),6.,rel_tol=0.,abs_tol=1e-12)
        # The total sparse component is exactly the weighted sum divided by3;
        # this is the same expression used by the production forward graph.
        raw=[.17,.23,.31]
        sparse=sum(c*x for c,x in zip(weights,raw))/3
        assert math.isclose(1.2*sparse,sum(1.2*c*x/3 for c,x in zip(weights,raw)),rel_tol=0.,abs_tol=1e-15)
    value={
        'passed':True,
        'scope':'Scalar coefficient arithmetic only; no production forward/backward or DDP assertions',
        'coefficient_mass':5.0,
        'effective_lambda_sparse':1.2,
        'arms':{n:{'normalized_weights':w,'effective_coefficients':[1.2*x for x in w],
                   'raw_ratio_sum':sum(w),'sparse_formula':'1.2*sum(w_i*Omega_i)/3'} for n,w in ARMS.items()},
        'fixed_state_formula_check':True,
        'note':'This arithmetic check is not a training-equivalence gate. Run hns_sparse_ratio_real_equivalence before training.'
    }
    Path(output).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();audit(args.output)
