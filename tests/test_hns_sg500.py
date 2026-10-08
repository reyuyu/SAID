"""Edge-local child stop-gradient; pinned HNS and INC0 production comparisons."""
import ast
import copy
import json
import subprocess
import types

import pytest
import torch

from model.hard_nested_sparsity import hard_violation, hns_terms, hard_telemetry
from model.nested_semantic_mask import hard_st
from recovery.hns_ddp_correctness import make
from recovery import hns_sg500 as sg
from tests.test_balanced_hparams import inputs
from tests.test_nested_d3_hns500 import base_forward
from tests.test_nested_fusion import explicit_inputs, explicit_logits


@pytest.mark.parametrize('child,parent,value',[(0,0,0),(0,1,0),(1,1,0),(1,0,1)])
def test_actual_hard_st_truth_table_and_exact_child_zero(child,parent,value):
    c=torch.tensor([[2. if child else -2.]],requires_grad=True)
    p=torch.tensor([[2. if parent else -2.]],requires_grad=True)
    cm,pm=hard_st(c.sigmoid()),hard_st(p.sigmoid())
    a=hard_violation(cm,pm).sum();b=hard_violation(cm,pm,True).sum()
    assert a.item()==b.item()==value
    old_c,old_p=torch.autograd.grad(a,(c,p),retain_graph=True)
    gc,gp=torch.autograd.grad(b,(c,p),allow_unused=True,retain_graph=True)
    assert gc is None or torch.count_nonzero(gc)==0
    torch.testing.assert_close(gp,old_p,atol=0,rtol=0)
    if value:assert gp.item()<0 and old_c.item()>0
    else:assert gp.item()==old_c.item()==old_p.item()==0
    sparse_grad=torch.autograd.grad(cm.abs().mean(),c)[0]
    if child:assert sparse_grad.item()>0


def test_all_binary_counts_and_forward_exact():
    torch.manual_seed(383)
    masks=[hard_st(torch.rand(11,512,requires_grad=True)) for _ in range(3)]
    valid=torch.tensor([True]*9+[False]*2)
    for completed in (0,99,199,499):
        a,aa=hns_terms(*masks,valid,completed,4,36)
        b,bb=hns_terms(*masks,valid,completed,4,36,detach_child=True)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for k in ('V_DF','V_3D'):torch.testing.assert_close(aa[k],bb[k],atol=0,rtol=0)
    telemetry,_=hard_telemetry(*masks,valid)
    for edge,c,p in [('DF',masks[1],masks[0]),('3D',masks[2],masks[1])]:
        assert telemetry[edge+'_violation_count']==((c[valid]==1)&(p[valid]==0)).sum()


@pytest.mark.parametrize('completed',[0,99,199,499])
def test_production_single_variable_forward_and_independent_gradients(completed):
    torch.manual_seed(9793);module=make(detach_child=True);ref=copy.deepcopy(module)
    ref.forward=types.MethodType(base_forward(),ref)
    images,views,valid=inputs();module.capture_hns_graph=True
    actual,logs=module(images,*views,valid,completed);baseline,_=ref(images,*views,valid,completed)
    masks=[]
    for tokens in views:
        z,t,zv,zt=explicit_inputs(ref,images,tokens)
        m=hard_st(explicit_logits(ref,zv,zt).sigmoid());masks.append(m.diagonal(dim1=0,dim2=1).T)
    from model.hard_nested_sparsity import hierarchy_weight
    expected=baseline+hierarchy_weight(completed)*2/3*(
        torch.relu(masks[1].detach()-masks[0])[valid].mean()+
        torch.relu(masks[2].detach()-masks[1])[valid].mean())
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(ref.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=8e-5,msg=name)
    assert logs['HNS_detach_child'] and logs['inc_weight']==logs['inclusion_loss']==0


def test_same_model_forward_alignment_sparse_gradients_and_groups_exact():
    torch.manual_seed(9793);a=make();b=copy.deepcopy(a);b.hns_detach_child=True
    a.capture_hns_graph=b.capture_hns_graph=True
    images,views,valid=inputs();la,al=a(images,*views,valid,499);lb,bl=b(images,*views,valid,499)
    torch.testing.assert_close(la,lb,atol=0,rtol=0)
    for k in ('HNS_align','HNS_original_sparse','V_DF_hard','V_3D_hard','HNS_surcharge'):
        torch.testing.assert_close(al[k],bl[k],atol=0,rtol=0)
    for key in ('alignment','original_sparsity'):
        pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
        ga=torch.autograd.grad(a.hns_graph[key],pa,allow_unused=True,retain_graph=True)
        gb=torch.autograd.grad(b.hns_graph[key],pb,allow_unused=True,retain_graph=True)
        for x,y in zip(ga,gb):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    for graph in (a.hns_graph,b.hns_graph):
        grads=torch.autograd.grad(graph['original_sparsity'],list(graph['masks'].values()),retain_graph=True)
        assert all(g.abs().sum()>0 for g in grads)
    from train.train_nested_semantic_mask import build_optimizer
    assert build_optimizer(a).state_dict()['param_groups']==build_optimizer(b).state_dict()['param_groups']


def test_current_default_matches_fetched_HNS_loss_all_gradients_and_AdamW():
    source=sg.git_blob(sg.BASE_SHA,'model/balanced_hparam_search.py').decode()
    terms=sg.git_blob(sg.BASE_SHA,'model/hard_nested_sparsity.py').decode()
    ns={'__name__':'model._hns_sg_reference','__package__':'model'};exec(compile(source,'fetched-HNS-v1','exec'),ns)
    oldns={};exec(compile(terms,'fetched-HNS-terms','exec'),oldns);ns['hns_terms']=oldns['hns_terms']
    torch.manual_seed(9793);a=make();b=copy.deepcopy(a);b.forward=types.MethodType(ns['BalancedSearch'].forward,b)
    images,views,valid=inputs();la,_=a(images,*views,valid,499);lb,_=b(images,*views,valid,499)
    torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward()
    for n,p in a.named_parameters():
        q=dict(b.named_parameters())[n];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    from train.train_nested_semantic_mask import build_optimizer
    build_optimizer(a).step();build_optimizer(b).step()
    for n,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[n],atol=0,rtol=0)


def test_pinned_optimizer_scheduler_rng_and_checkpoint_unchanged():
    path='train/train_nested_semantic_mask.py'
    before=ast.parse(sg.git_blob(sg.BASE_SHA,path));after=ast.parse((sg.ROOT/path).read_text())
    functions=('build_optimizer','learning_rates','optimizer_learning_rates','seed_all','rng_state','restore_rng_state','training_horizon','save_checkpoint','consumed_batch')
    old={n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,ast.FunctionDef)}
    new={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,ast.FunctionDef)}
    for name in functions:assert old[name]==new[name]


def test_only_config_flag_and_native_admission_rejects_full_resume_NFS():
    baseline=json.loads(sg.git_blob(sg.BASE_SHA,str((sg.BASE_EXP/'config.json').relative_to(sg.ROOT))))
    cfg=json.loads((sg.EXP/'config.json').read_text());assert cfg==dict(baseline,hns_detach_child=True)
    assert cfg==json.loads((sg.ROOT/'configs/nested_d3_hns_sg500.json').read_text())
    from train.train_nested_semantic_mask import validate_nested_view_weights
    validate_nested_view_weights(cfg,'formal',500,None,'/root/said_s02_stage500/ShareGPT4V')
    validate_nested_view_weights(cfg,'smoke',5,None,'/root/said_s02_stage500/ShareGPT4V')
    for run_type,updates,resume,root in [('formal',4868,None,'/root/said_s02_stage500/ShareGPT4V'),
        ('formal',500,'old.pt','/root/said_s02_stage500/ShareGPT4V'),('formal',500,None,'/opt/data/private/lklk/SAID/local_assets')]:
        with pytest.raises(AssertionError):validate_nested_view_weights(cfg,run_type,updates,resume,root)


def test_retrieval_primary_classification_no_violation_veto():
    base=dict(Score5=.7120205634766327,J_long3=.7515209391277212,Short4=.65277)
    q=dict(base,Score5=base['Score5']+.00001)
    assert sg.decide(q,base,False)['classification']=='STRONG_POSITIVE'
    q=dict(base,Score5=base['Score5']-.0002,Short4=base['Short4']+.0011)
    assert sg.decide(q,base,False)['classification']=='POSITIVE'
    q=dict(base,Score5=base['Score5']-.0021)
    assert sg.decide(q,base,True)['classification']=='MIXED'
    assert sg.decide(q,base,False)['classification']=='NEGATIVE'


def test_checkpoint_metadata_adapter_does_not_duplicate_states_or_mutate(monkeypatch):
    tensor=torch.ones(5)
    current={'config':{'hns_detach_child':True,'runtime_model':{'hns_detach_child':True}},'model':{'weight':tensor}}
    reference={'config':{}}
    def check(a,b):
        assert a['model']['weight'] is tensor and 'hns_detach_child' not in a['config']['runtime_model']
        return {'passed':True}
    monkeypatch.setattr(sg,'OriginalInvariants',check)
    assert sg.checkpoint_invariants(current,reference)['passed']
    assert current['config']['runtime_model']['hns_detach_child'] is True


def test_actual_audit_same_graph_endpoint_and_group_decomposition(monkeypatch):
    from recovery import hns_sg_gradient_audit as audit
    monkeypatch.setattr(audit.dist,'all_reduce',lambda tensor:None)
    torch.manual_seed(9793);module=make(detach_child=True);module.capture_hns_graph=True
    images,views,valid=inputs();module(images,*views,valid,499)
    modes=audit.inspect_graph(module,module.hns_graph,valid,1)
    assert modes['HNS-v1']['forward_values']==modes['HNS-SG']['forward_values']
    for edge in ('V_DF','V_3D'):
        assert modes['HNS-SG']['endpoint_gradient_direction'][edge]['HardST_output']['child_gradient_exactly_zero']
        assert modes['HNS-v1']['endpoint_gradient_direction'][edge]['HardST_output']['child_contraction_verified']


def test_final_report_with_real_schemas_only_in_temporary_fixture(tmp_path,monkeypatch):
    from recovery import nested_d3_local_search as search
    from recovery.s02_nfs500 import dump
    from experiments.nest_clip_v1.balanced_hparam_search_v1 import search as native
    exp=tmp_path/'exp';run=tmp_path/'run';exp.mkdir();run.mkdir()
    for name in ('RESULTS.json','TRAINING_DIAGNOSTICS.json','MASK_HIERARCHY_AUDIT.json','RUNTIME_STATS.json'):
        dump(exp/name,sg.read(sg.BASE_EXP/name))
    modes={n:dict(group_cosines={'shared_visual_mask':{'cosine_hierarchy_original_sparsity':-.8}},
        endpoint_gradient_direction={'synthetic_fixture':True}) for n in ('HNS-SG','HNS-v1')}
    dump(exp/'GRADIENT_AUDIT.json',dict(passed=True,checkpoints={'HNS-SG':{'modes':modes}}))
    dump(exp/'VALIDATION.json',{'passed':True})
    dump(run/'supervisor-result.json',dict(acceptance={'passed':True},commands=[],started_utc='synthetic',ended_utc='synthetic'))
    for name in ('launch-provenance.json','first-five-gate.json','full-stream-proof.json','commands.json'):
        dump(run/name,{'synthetic_fixture':True})
    (run/'HNS-SG-matched-gradient-audit500.log').write_text('Synthetic test fixture\n')
    result=sg.read(exp/'RESULTS.json')
    values=dict(sg.read(sg.BASE_EXP/'TRAINING_DIAGNOSTICS.json')['HNS']['last50'])
    values['HNS_detach_child']=True
    monkeypatch.setattr(sg,'EXP',exp);monkeypatch.setattr(search,'RUN',run)
    monkeypatch.setattr(sg,'SMOKE',tmp_path/'smoke');monkeypatch.setattr(sg,'SMOKE_PHASE',tmp_path/'phase')
    monkeypatch.setattr(sg,'rows',lambda p:[dict(step=i,**values) for i in range(1,501)])
    monkeypatch.setattr(native,'native_metrics',lambda p:(result['metrics'],result['native_evaluation_provenance'],{}))
    sg.summarize()
    actual=sg.read(exp/'RESULTS.json')
    assert set(actual['all_baseline_comparisons'])=={'INC0','Anchor','HNS-v1'}
    assert len(list((exp/'evaluations').glob('*.json')))==5
    assert 'ReLU(child.detach()-parent)' in (exp/'REPORT.md').read_text()
    assert actual['decision']['automatic_full'] is False


def test_torchrun_module_argument_delimiter_and_optional_recovery_artifacts(tmp_path,monkeypatch):
    from torch.distributed.run import get_args_parser
    command=sg.audit_command('/tmp/sg-run','/tmp/sg-exp')
    args=get_args_parser().parse_args(command[1:])
    assert args.module and args.training_script=='recovery.hns_sg_gradient_audit'
    assert args.training_script_args==['--run','/tmp/sg-run','--experiment','/tmp/sg-exp']
    monkeypatch.setattr(sg,'EXP',tmp_path)
    assert not any(p.name in sg.OPTIONAL_REPORTS for p in sg.publication_paths())
    (tmp_path/'POST500_RECOVERY.json').write_text('{}')
    assert tmp_path/'POST500_RECOVERY.json' in sg.publication_paths()
