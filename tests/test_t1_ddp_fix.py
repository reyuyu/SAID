import json
import os
import socket
import subprocess
import sys
import pytest
import torch
from test_said_token_v1 import WORKER, REPO_ROOT


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


@pytest.mark.parametrize('scenario',['invalid_rank','all_invalid','duplicates'])
def test_actual_ddp_backward_and_update_with_uneven_valid_counts(tmp_path,scenario):
    outputs=[]
    for mode in ['single','rank']:
        path=str(tmp_path/(mode+'_{rank}.json'))
        port=str(free_port())
        command=[sys.executable]
        if mode=='rank':
            command+=['-m','torch.distributed.run','--nproc_per_node=2','--master_port',port]
        command+=[WORKER,'--mode',mode,'--rows','2','--scenario',scenario,'--optimizer','sgd','--out',path]
        result=subprocess.run(command,capture_output=True,text=True,cwd=REPO_ROOT,
            env={**os.environ,'MASTER_ADDR':'127.0.0.1','MASTER_PORT':port,'GLOO_SOCKET_IFNAME':'lo','OMP_NUM_THREADS':'1'},timeout=120)
        assert result.returncode==0,result.stderr[-5000:]
        outputs.append([json.load(open(path.format(rank=r))) for r in range(1 if mode=='single' else 2)])
    single=outputs[0][0]
    ranks=outputs[1]
    for rank in ranks:
        assert rank['loss_total_global_mean']==pytest.approx(single['loss_total_global_mean'],abs=2e-6,rel=2e-5)
        assert rank['said_scaling_abs_diff']<1e-6 and rank['rec_scaling_abs_diff']<1e-6
        for category in ['grads','updated']:
            for name,value in single[category].items():
                other=rank[category][name]
                if value is None:
                    assert other is None
                    continue
                a,b=torch.tensor(value),torch.tensor(other)
                difference=(a-b).abs().max()
                assert difference<=2e-7+2e-5*a.abs().max(),(category,name,difference)
        if scenario=='invalid_rank':
            assert rank['loss_rec_valid_global']==2
        if scenario=='all_invalid':
            assert rank['loss_rec_valid_global']==0 and rank['loss_rec_global_mean']==0
    assert ranks[0]['grads']==ranks[1]['grads']
    assert ranks[0]['updated']==ranks[1]['updated']
