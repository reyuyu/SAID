"""Real production graph: only post500 hierarchy scale may change."""
import copy

import pytest
import torch

from recovery import hns_half1217 as half
from recovery.hns_ddp_correctness import make
from tests.test_balanced_hparams import inputs
from tests.test_hns_full import payloads
from train.train_nested_semantic_mask import validate_resume_payload


@pytest.mark.parametrize('completed,weight',[(0,0),(199,.995),(499,1),(500,.5),(1216,.5)])
def test_optimizer_update_boundary(completed,weight):
    assert half.half_weight(completed)==weight


@pytest.mark.parametrize('completed',[499,500,1216])
def test_actual_graph_component_values_and_gradients(monkeypatch,completed):
    from model import hard_nested_sparsity as hns, balanced_hparam_search as model
    torch.manual_seed(9793);a=make();b=copy.deepcopy(a)
    a.capture_hns_graph=b.capture_hns_graph=True
    images,views,valid=inputs();la,al=a(images,*views,valid,completed)
    monkeypatch.setattr(hns,'hierarchy_weight',half.half_weight)
    monkeypatch.setattr(model,'hierarchy_weight',half.half_weight)
    lb,bl=b(images,*views,valid,completed)
    ratio=1 if completed<500 else .5
    for key in ('HNS_align','HNS_original_sparse','V_DF_hard','V_3D_hard'):
        torch.testing.assert_close(al[key],bl[key],atol=0,rtol=0)
    torch.testing.assert_close(bl['HNS_surcharge'],ratio*al['HNS_surcharge'],atol=0,rtol=0)
    pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
    for key in ('alignment','original_sparsity','hierarchy'):
        ga=a.hns_graph[key] if key!='hierarchy' else (a.hns_graph['V_DF']+a.hns_graph['V_3D'])*2/3
        gb=b.hns_graph[key] if key!='hierarchy' else (b.hns_graph['V_DF']+b.hns_graph['V_3D'])*2/3*ratio
        aa=torch.autograd.grad(ga,pa,allow_unused=True,retain_graph=True)
        bb=torch.autograd.grad(gb,pb,allow_unused=True,retain_graph=True)
        for x,y in zip(aa,bb):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(y,x*(ratio if key=='hierarchy' else 1),atol=0,rtol=0)
    assert bl['lambda_h']==half.half_weight(completed) and not bl['HNS_detach_child']


def test_half_resume_pinned_states_and_stop(monkeypatch):
    previous,current,proof=payloads();current.update(max_updates=1217,hns_half_after500=True)
    current['resume']=str(half.PARENT)
    before=copy.deepcopy(previous)
    assert half.validate_resume(previous,current,proof,validate_resume_payload)==500
    assert previous==before
    for key,value in [('max_updates',4868),('hns_detach_child',True),('hns_half_after500',False),
        ('view_weights',[1,1,1]),('workers',4),('hns_beta',[1,2]),('resume','other.pt')]:
        changed=dict(current);changed[key]=value
        with pytest.raises(AssertionError):half.validate_resume(previous,changed,proof,validate_resume_payload)


def test_torchrun_delimiter_stop_local_root():
    from torch.distributed.run import get_args_parser
    command=half.train_command();args=get_args_parser().parse_args(command[1:])
    assert args.training_script==half.ENTRY and '--max-updates' in args.training_script_args
    assert command[command.index('--max-updates')+1]=='1217'
    assert command[command.index('--resume')+1]==str(half.PARENT)
    assert command[command.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'


def test_declared_decision_primary_retrieval():
    assert half.classify(dict(Score5=.1,J_long3=0,J_long=-.01,Short4=0),False)=='PROMISING_FOR_FULL'
    assert half.classify(dict(Score5=-.1,J_long3=-.1,J_long=-.1,Short4=.1),True)=='NEGATIVE'
