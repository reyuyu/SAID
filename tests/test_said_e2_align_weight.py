import copy
import ast
import subprocess
import sys
import types
import pytest
import torch
from model.balanced_hparam_search import BalancedSearch, hparams, macro_terms, trial_id
from tests.test_nested_fusion import TinyFusionCLIP
from recovery import said_e2_align_weight_fourarm500 as c
from recovery.said_e2_align_weight_precheck import legacy_class


def tiny(weights=(1.35,1.35,.3),align=10.):
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=list(weights),
        inclusion_max=0,lambda_align=align,lambda_sparse=1.2,lambda_hierarchy=1),
        view_sparsity_weights=[5/3]*3,hns_enabled=True,inclusion_hierarchy='detail_chain',
        fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


def inputs():
    return torch.randn(6,8),[torch.randint(0,31,(6,6)) for _ in range(3)]


def gradients(value,params):return torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)


def exact_gradients(a,b):
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_plan_exact_unique_trials_and_single_axes():
    assert c.ARMS=={'A1-View-DallPlus':([1.3,1.4,.3],10.),'A2-View-FPlus':([1.4,1.3,.3],10.),
                    'A3-Align95':([1.35,1.35,.3],9.5),'A4-Align105':([1.35,1.35,.3],10.5)}
    original=c.read(c.BASE_EXP/'config.json');ids=[]
    for arm in c.ARMS:
        cfg=c.config(arm);c.frozen(cfg,arm);ids.append(trial_id(cfg))
        expected={'view_weights'} if arm.startswith(('A1','A2')) else {'lambda_align'}
        assert {k for k,v in original.items() if cfg[k]!=v and k!='trial_id'}==expected
        assert cfg['trial_id']==trial_id(cfg) and cfg['trial_id']!=original['trial_id']
        assert sum(cfg['view_weights'])==3 and cfg['view_weights'][2]==.3
    assert len(set(ids))==4 and trial_id(original) not in ids


def test_exact_production_scope_and_formula_preserved():
    assert set(c.source_scope())==set(c.PRODUCTION_CHANGES)


@pytest.mark.parametrize('completed',[0,99,200,499])
@pytest.mark.parametrize('valid_count',[0,1,2,6])
def test_default_vs_frozen_E2_forward_loss_and_all_gradients_exact(completed,valid_count):
    torch.manual_seed(8921);model=tiny();old=copy.deepcopy(model)
    old.forward=types.MethodType(legacy_class().forward,old)
    image,views=inputs();valid=torch.arange(6)<valid_count
    x,logs=model(image,*views,valid,completed);y,ref=old(image,*views,valid,completed)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    for k,v in ref.items():
        if torch.is_tensor(v):torch.testing.assert_close(logs[k],v,rtol=0,atol=0,msg=k)
        else:assert logs[k]==v,k
    x.backward();y.backward()
    exact_gradients([p.grad for p in model.parameters()],[p.grad for p in old.parameters()])


@pytest.mark.parametrize('completed',[0,99,200,499])
@pytest.mark.parametrize('arm',list(c.ARMS))
def test_each_arm_changes_only_alignment_formula_with_other_gradients_exact(arm,completed):
    torch.manual_seed(7311);model=tiny();model.capture_hns_graph=True
    image,views=inputs();valid=torch.tensor([1,1,0,1,1,0],dtype=torch.bool)
    _,_=model(image,*views,valid,completed);old=model.hns_graph
    params=[p for p in model.parameters() if p.requires_grad]
    refs={k:gradients(old[k],params) for k in ('weighted_sparse','weighted_hierarchy','raw_view_F','raw_view_Dall','raw_view_D3')}
    weights,align=c.ARMS[arm]
    model.search_hparams['view_weights']=list(weights)
    model.search_hparams['lambda_align']=align;model.macro_hparams['lambda_align']=align
    loss,logs=model(image,*views,valid,completed);g=model.hns_graph
    for key in refs:
        torch.testing.assert_close(g[key],old[key],rtol=0,atol=0)
        exact_gradients(gradients(g[key],params),refs[key])
    for key in g['masks']:torch.testing.assert_close(g['masks'][key],old['masks'][key],rtol=0,atol=0)
    af,ad,a3=[g['raw_view_'+v] for v in ('F','Dall','D3')];wf,wd,w3=weights
    legacy=10/(wf+wd+w3)*(wf*af+wd*ad+w3*a3)
    wa,ws,wh=macro_terms(legacy,g['raw_sparse'],g['weighted_hierarchy'],model.macro_hparams)
    torch.testing.assert_close(g['weighted_align'],wa,rtol=0,atol=0)
    torch.testing.assert_close(loss,(wa+g['weighted_sparse'])+g['weighted_hierarchy'],rtol=0,atol=0)
    exact_gradients(gradients(loss,params),gradients((wa+g['weighted_sparse'])+g['weighted_hierarchy'],params))
    assert not logs['inclusion_enabled'] and logs['inc_weight']==0
    assert g['lambda_h']==min(1,completed/200)
    for p,gg in zip(params,gradients(g['weighted_sparse']+g['weighted_hierarchy'],params)):
        if any(p is q for q in model.clip.visual.parameters()):assert gg is None or not gg.any()


@pytest.mark.parametrize('weights',[[1.31,1.39,.3],[1.4,1.4,.2],[1.,1.,1.],[1.3,1.4,.31]])
def test_unreviewed_HNS_view_allocation_rejected(weights):
    with pytest.raises(AssertionError,match='Unreviewed HNS alignment'):tiny(weights)


def trainer_weights():
    tree=ast.parse((c.ROOT/'train/train_nested_semantic_mask.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='allowed' for t in n.targets))
    return lambda mode:eval(compile(ast.Expression(node.value),'trainer_whitelist','eval'),{'cfg':{'sampling_mode':mode}})


def test_trainer_whitelist_preserves_old_and_only_adds_two_D3_entries():
    allowed=trainer_weights()
    assert allowed('nested_detail_d3')==([1.,1.,1.],[1.35,1.35,.3],[1.4,1.4,.2],
        [1.375,1.375,.25],[1.325,1.325,.35],[1.3,1.3,.4],[1.3,1.4,.3],[1.4,1.3,.3])
    assert [1.31,1.39,.3] not in allowed('nested_detail_d3')
    assert allowed('nested_detail_kr234')==([1.35,1.35,.3],)
    assert allowed('nested_detail')==([1.4,1.4,.2],[1.,1.,1.])


def test_full_stream_checks_alignment_native_LR_and_all_sampling_fields():
    reference=c.rows(c.BASE_RUN/'step500/steps.jsonl')[:2]
    for arm,(weights,align) in c.ARMS.items():
        actual=copy.deepcopy(reference)
        for row in actual:
            row['alignment_view_weights']=weights;row['macro_lambda_align']=align
            row['macro_weighted_align']=align*row['macro_raw_align']
            row['loss']=sum(row['macro_weighted_'+k] for k in ('align','sparse','hierarchy'))
        assert c.matched_stream(actual,reference,arm)['records']==2048
        bad=copy.deepcopy(actual);bad[-1]['actual_lrs']['backbone']*=1.01
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,arm)
        bad=copy.deepcopy(actual);bad[-1]['rank_health'][0]['sampling']['K'][0]=100
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,arm)
        bad=copy.deepcopy(actual);bad[-1]['alignment_view_weights']=[1.31,1.39,.3]
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,arm)


def test_commands_independent_common0_not_resume_and_stop500():
    # Controllers intentionally rebind shared helpers. Isolate that process
    # state so historical command tests keep their original runner functions.
    subprocess.run([sys.executable,'-c',"""
from recovery import said_e2_align_weight_fourarm500 as c
c.configure()
for arm in c.ARMS:
    for smoke in (True,False):
        cmd=c.base.training_command(arm,smoke)
        assert '--resume' not in cmd
        assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
        assert cmd[cmd.index('--init-state')+1]==str(c.STEP0)
        assert cmd[cmd.index('-m')+1]==c.ENTRY
"""],check=True)
