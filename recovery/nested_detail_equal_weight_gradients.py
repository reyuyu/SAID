"""Same read-only8-batch protocol, directed at the equal-weight checkpoint."""
from recovery import nested_detail_gradients as protocol
from recovery.nested_detail_equal_weight500 import RUN,EXP,WEIGHTS
from recovery.s02_nfs500 import dump
import json


def main():
    protocol.RUN,protocol.EXP,protocol.WEIGHTS = RUN,EXP,WEIGHTS
    protocol.main()
    # Only the rank0 writer enriches its result; no extra forward/backward.
    import os
    if os.environ['RANK']=='0':
        path = EXP/'GRADIENT_SPOTCHECK.json'
        data = json.loads(path.read_text()); m = data['mean_gradient_norms']
        data['mean_norm_ratios'] = dict(Ds_Dall=m['Ds']/m['Dall'],Ds_F=m['Ds']/m['F'])
        data['per_batch_norm_ratios'] = [dict(batch=b['batch'],
            Ds_Dall=b['gradient_norms']['Ds']/b['gradient_norms']['Dall'],
            Ds_F=b['gradient_norms']['Ds']/b['gradient_norms']['F']) for b in data['batches']]
        dump(path,data)


if __name__=='__main__':
    main()
