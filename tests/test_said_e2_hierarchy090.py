import copy
import json
import subprocess
import sys
import pytest
import torch
from model.balanced_hparam_search import BalancedSearch,trial_id
from tests.test_nested_fusion import TinyFusionCLIP
from recovery import said_e2_hierarchy090_500 as c


def test_only_one_macro_variable_and_native_trial_id():
    original=c.read(c.BASE_EXP/'config.json');candidate=c.config(c.ARM)
    assert {k for k in candidate if candidate[k]!=original.get(k)}=={'lambda_hierarchy','trial_id'}
    assert candidate['lambda_hierarchy']==.9 and candidate['trial_id']==trial_id(candidate)
    c.frozen(candidate,c.ARM)
    for key,value in [('lambda_sparse',1.1),('lambda_hierarchy',.8),('view_weights',[1.3,1.4,.3]),('fusion_lr',1e-4)]:
        changed=copy.deepcopy(candidate);changed[key]=value
        with pytest.raises(AssertionError):c.frozen(changed,c.ARM)
    with pytest.raises(AssertionError):c.config('second-arm')


@pytest.mark.parametrize('completed',[0,99,200,499])
@pytest.mark.parametrize('valid_count',[0,1,4,6])
def test_real_forward_only_hierarchy_changes(completed,valid_count):
    torch.manual_seed(943)
    hp=dict(view_weights=[1.35,1.35,.3],inclusion_max=0,lambda_align=10,lambda_sparse=1.2,lambda_hierarchy=1)
    m=BalancedSearch(TinyFusionCLIP(32),search_hparams=hp,view_sparsity_weights=[5/3]*3,
        hns_enabled=True,inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',
        text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    m.capture_hns_graph=True
    images=torch.randn(6,8);tokens=[torch.randint(0,31,(6,6)) for _ in range(3)]
    valid=torch.arange(6)<valid_count
    before={k:v.clone() for k,v in m.state_dict().items()}
    params=[p for p in m.parameters() if p.requires_grad]
    def grad(v):return torch.autograd.grad(v,params,allow_unused=True,retain_graph=True)
    loss,oldlogs=m(images,*tokens,valid,completed)
    if valid_count<2:
        # Native fallback does not expose the optional captured three-view graph.
        old=grad(loss);m.macro_hparams['lambda_hierarchy']=.9
        newloss,logs=m(images,*tokens,valid,completed)
        torch.testing.assert_close(loss,newloss,atol=0,rtol=0)
        for x,y in zip(grad(newloss),old):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
        assert oldlogs['macro_weighted_hierarchy']==logs['macro_weighted_hierarchy']==0
        assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
        return
    g=m.hns_graph
    old={k:grad(g[k]) for k in ('weighted_align','weighted_sparse','weighted_hierarchy','total_training')}
    m.macro_hparams['lambda_hierarchy']=.9
    newloss,logs=m(images,*tokens,valid,completed);n=m.hns_graph
    for key in ('alignment','original_sparsity','raw_hierarchy','V_DF','V_3D'):
        torch.testing.assert_close(n[key],g[key],atol=0,rtol=0)
    for key in ('weighted_align','weighted_sparse'):
        torch.testing.assert_close(n[key],g[key],atol=0,rtol=0)
        for x,y in zip(grad(n[key]),old[key]):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(n['weighted_hierarchy'],.9*g['weighted_hierarchy'],atol=0,rtol=0)
    torch.testing.assert_close(newloss,(n['weighted_align']+n['weighted_sparse'])+n['weighted_hierarchy'],atol=0,rtol=0)
    for x,y,h in zip(grad(newloss),old['total_training'],old['weighted_hierarchy']):
        if x is None:assert y is None;continue
        hh=torch.zeros_like(x) if h is None else h
        torch.testing.assert_close(x-y,-.1*hh,rtol=3e-4,atol=1e-5)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    assert logs['inc_weight']==0 and not logs['inclusion_enabled']
    assert n['lambda_h']==min(1.,completed/200)


def test_independent_commands_in_isolated_process():
    # Shared historical helpers are rebound by configure; keep this test isolated.
    code='''from recovery import said_e2_hierarchy090_500 as c
c.configure()
for smoke in (True,False):
 cmd=c.base.training_command(c.ARM,smoke)
 assert "--resume" not in cmd and "--max-restarts=0" in cmd
 assert cmd[cmd.index("-m")+1]==c.ENTRY
 assert cmd[cmd.index("--init-state")+1]==str(c.STEP0)
 assert cmd[cmd.index("--max-updates")+1]==("5" if smoke else "500")
 assert cmd[cmd.index("--output-dir")+1]==str(c.RUN/(c.ARM+".smoke5" if smoke else c.ARM)/"step500")
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_full_stream_rejects_LR_data_macro_and_update_drift():
    ref=c.rows(c.BASE_RUN/'step500/steps.jsonl')[:2];actual=copy.deepcopy(ref)
    for r in actual:
        r['macro_lambda_hierarchy']=.9;r['macro_weighted_hierarchy']*=.9
        r['loss']=sum(r['macro_weighted_'+k] for k in ('align','sparse','hierarchy'))
    assert c.matched_stream(actual,ref,c.ARM)['records']==2048
    for mutate in (lambda r:r['actual_lrs'].__setitem__('backbone',99),
                   lambda r:r.__setitem__('step',1),
                   lambda r:r.__setitem__('macro_lambda_hierarchy',1),
                   lambda r:r['rank_health'][0].__setitem__('stream_sha256','wrong'),
                   lambda r:r['rank_health'][0]['sampling']['K'].__setitem__(0,999)):
        broken=copy.deepcopy(actual);mutate(broken[1])
        with pytest.raises(AssertionError):c.matched_stream(broken,ref,c.ARM)


def test_production_sources_are_exact_parent():
    from train.train_nested_semantic_mask import code_manifest
    import hashlib
    manifest=code_manifest()
    assert manifest==c.read(c.BASE_RUN/'step500/config.json')['code_sha256']
    for p,h in manifest.items():
        frozen=subprocess.check_output(['git','show',c.PARENT_COMMIT+':'+p],cwd=c.ROOT)
        assert hashlib.sha256(frozen).hexdigest()==h
