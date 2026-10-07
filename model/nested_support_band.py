"""BBNS: squared support bounds with opposite detached edge references."""
import math

import torch

EDGES=('Dall_F','D3_Dall')


def validate_bands(bands):
    assert isinstance(bands,dict) and set(bands)==set(EDGES)
    result={}
    for edge in EDGES:
        values={k:float(v) for k,v in bands[edge].items()}
        assert set(values)=={'kappa','tau_low','tau_high'}
        assert all(math.isfinite(v) and 0<=v<=1 for v in values.values())
        assert values['tau_low']<values['tau_high']
        result[edge]=values
    return result


def support_band_edge(child,parent,band):
    child_ref=child.detach();parent_ref=parent.detach()
    coverage=(child_ref*parent).sum(-1)/(child_ref.sum(-1)+1e-6)
    relative=(child*parent_ref).sum(-1)/(parent_ref.sum(-1)+1e-6)
    cover=torch.relu(band['kappa']-coverage).square()
    lower=torch.relu(band['tau_low']-relative).square()
    upper=torch.relu(relative-band['tau_high']).square()
    return dict(coverage=coverage,relative_support=relative,cover_loss=cover,
        lower_band_loss=lower,upper_band_loss=upper,
        zero_refine_band=((relative>=band['tau_low'])&(relative<=band['tau_high'])).float(),
        zero_edge_band=((coverage>=band['kappa'])&(relative>=band['tau_low'])&(relative<=band['tau_high'])).float())


def bbns_regularizer(omega_f,pf,pd,p3,valid,bands,world,count):
    """World/count differentiable local sums, averaged by DDP.

    No child global sparsity, independent inclusion, ramp or beta.
    F is normalized across all examples; edges across valid nested examples.
    """
    means={}
    edge_total=omega_f.new_zeros(())
    for name,child,parent in [('Dall_F',pd,pf),('D3_Dall',p3,pd)]:
        edge=support_band_edge(child,parent,bands[name])
        local={k:world/count*v[valid].sum() for k,v in edge.items()}
        local['percent_inside_zero_loss_band']=100*local['zero_edge_band']
        local['percent_inside_refinement_band']=100*local['zero_refine_band']
        edge_loss=local['cover_loss']+local['lower_band_loss']+local['upper_band_loss']
        edge_total=edge_total+edge_loss
        means.update({name+'_'+k:v for k,v in local.items()})
        means[name+'_edge_loss']=edge_loss
    total=omega_f/3+edge_total
    means.update(Omega_F=omega_f,total_band_edges=edge_total,total_nested_regularizer=total)
    return total,means
