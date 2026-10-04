"""Experiment-only exposure of live production losses; no production edits."""
from contextlib import contextmanager
import sys
import torch
from torch import nn
from torch.nn import functional as F
import model.balanced_hparam_search as balanced
import model.nested_fusion_mask as fusion
from model.nested_semantic_mask import gather,world_rank,inclusion_weight

OBJECTIVES=['F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i',
            'F_combined','O_combined','E_combined',
            'native_i2t','native_t2i','native_combined','align_actual','sparse','inc','total']

@contextmanager
def capture_production():
    """Read existing function-return frame locals, leaving all math untouched."""
    captured=[];forward=[]
    prior=sys.getprofile()
    assert prior is None,'Do not replace an existing profiler'
    def observe(frame,event,arg):
        if event=='return':
            if frame.f_code is fusion.fusion_view_terms.__code__:
                values=frame.f_locals
                captured.append({key:values[key] for key in ['ci','ct','zero','world','n']})
            elif frame.f_code is balanced.BalancedSearch.forward.__code__:
                values=frame.f_locals
                forward.append({key:values[key] for key in ['af','ao','ae','sf','so','se','align','sparse','inc_sum','valid_count','loss','weight']})
    sys.setprofile(observe)
    try:yield captured,forward
    finally:sys.setprofile(prior)


def production_components(model,images,views,valid,completed=200):
    with capture_production() as (captured,forward):
        loss,logs=model(images,*views,valid,completed)
    assert len(captured)==3 and len(forward)==1
    f=forward[0];out={}
    for label,v in zip(['F','O','E'],captured):
        factor=v['world']/v['n']
        out[label+'_i2t']=factor*(v['ci']+v['zero'])
        out[label+'_t2i']=factor*(v['ct']+v['zero'])
        out[label+'_combined']=f[{'F':'af','O':'ao','E':'ae'}[label]]
    out.update(align_actual=f['align'],sparse=f['sparse'],
               inc=f['weight']*world_rank()[0]/f['valid_count']*f['inc_sum'],total=loss)
    return out,logs


def native_components(model,images,tokens,full_correctness=False,precision='bf16'):
    world,rank=world_rank()
    with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda and precision=='bf16'):
        z=model.clip.encode_image(images)
        t=model.clip.encode_text(tokens)
    z,t=z.float(),t.float();zg=gather(z)
    # Q[j,i] agrees with production t2i rows and evaluator native full towers.
    scores=100*(F.normalize(t,dim=-1)@F.normalize(zg,dim=-1).T)
    matrix=gather(scores)
    n=len(matrix);assert matrix.shape==(n,n)
    labels=rank*len(z)+torch.arange(len(z),device=z.device)
    image_rows=matrix.T[rank*len(z):(rank+1)*len(z)]
    zero=(zg.sum()+matrix.sum()+scores.sum()+t.sum())*0
    ci=F.cross_entropy(image_rows,labels,reduction='sum')
    ct=F.cross_entropy(scores,labels,reduction='sum')
    out={'native_i2t':world/n*(ci+zero),'native_t2i':world/n*(ct+zero),
         'native_combined':world/n*(ci+ct+zero)}
    checks={'n_candidates':n,'shape':list(matrix.shape),'positive_mapping':'rank-major diagonal matched pairs','scale':100}
    if full_correctness:
        tg=gather(t,False)
        direct=100*(F.normalize(tg,dim=-1)@F.normalize(zg.detach(),dim=-1).T)
        matrix_difference=float((matrix.detach()-direct).abs().max())
        torch.testing.assert_close(matrix.detach(),direct,atol=3e-5,rtol=2e-6)
        rankings={}
        for name,m in [('T2I',matrix),('I2T',matrix.T)]:
            ranked=torch.argsort(m.detach(),dim=-1,descending=True)[:,0]
            direct_rank=torch.stack([torch.argsort(row,descending=True)[0] for row in (direct if name=='T2I' else direct.T)])
            assert torch.equal(ranked,direct_rank),(name,'top1 direct ranking mismatch')
            rankings[name]=float((ranked==torch.arange(n,device=m.device)).float().mean())
        checks.update(full_embedding_matrix_numerically_equal=True,full_embedding_matrix_max_abs_difference=matrix_difference,direct_top1_rankings_equal=True,top1=rankings)
    logs={k:float(__import__('model.nested_semantic_mask',fromlist=['global_sum']).global_sum(v)/world) for k,v in out.items()}
    return out,logs,checks


class AuditObjective(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.last_losses=None;self.last_checks=None
    def forward(self,images,tokens_f,tokens_o,tokens_e,valid,objective,check_native=False):
        if objective.startswith('native'):
            losses,logs,checks=native_components(self.model,images,tokens_f,check_native)
            self.last_checks=checks
        else:
            losses,logs=production_components(self.model,images,[tokens_f,tokens_o,tokens_e],valid)
        self.last_losses=logs
        return losses[objective]
