"""Frozen Balanced architecture with the five authorized training coefficients."""
import copy
import hashlib
import json
import math

import torch
from model.nested_fusion_mask import NestedFusionMask, fusion_view_terms
from model.hard_nested_sparsity import hns_terms,hard_telemetry,hierarchy_weight
from model.nested_semantic_mask import gather, global_sum, inclusion, inclusion_weight, world_rank


DEFAULTS = dict(fusion_lr=1e-4, visual_mask_lr_scale=1., view_weights=[1.,1.,1.],
                sparsity_scale=1., inclusion_max=1.)
DIAGNOSTIC_UPDATES = (1,100,200,500,1217,2000,3000,3651,4868)


def hparams(config):
    result = {key: copy.deepcopy(config.get(key,value)) for key,value in DEFAULTS.items()}
    result['view_weights'] = [float(value) for value in result['view_weights']]
    assert len(result['view_weights']) == 3 and all(math.isfinite(v) and v>=0 for v in result['view_weights'])
    assert sum(result['view_weights']) > 0, 'Alignment weights must have positive total'
    for key in DEFAULTS:
        if key != 'view_weights':
            result[key] = float(result[key])
            assert math.isfinite(result[key]) and (result[key]>=0 if key=='inclusion_max' else result[key]>0), key
    return result


def trial_id(config):
    return hashlib.sha256(json.dumps(hparams(config),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def detail_chain_inclusion(pf, dall, ds):
    """Only Dall->F and Ds->Dall; each child is detached on its edge."""
    return .5 * (torch.relu(dall.detach()-pf).mean(-1) +
                 torch.relu(ds.detach()-dall).mean(-1))


class BalancedSearch(NestedFusionMask):
    def __init__(self, clip, *, search_hparams=None, inclusion_hierarchy='siblings',
                 view_sparsity_weights=(1.,2.,2.), hns_enabled=False, hns_beta=(2.,2.),
                 hns_detach_child=False, **options):
        super().__init__(clip, **options)
        assert self.fusion=='balanced_stack' and self.visual=='patch'
        self.search_hparams = hparams(search_hparams or {})
        self.summary_t2i_weight = float((search_hparams or {}).get("summary_t2i_weight", 1.))
        assert 0 < self.summary_t2i_weight <= 1.
        assert inclusion_hierarchy in ('siblings', 'detail_chain')
        self.inclusion_hierarchy = inclusion_hierarchy
        self.hns_enabled = bool(hns_enabled)
        self.hns_detach_child = bool(hns_detach_child)
        assert not self.hns_detach_child or self.hns_enabled
        self.hns_beta = tuple(float(v) for v in hns_beta)
        assert len(self.hns_beta)==2 and all(math.isfinite(v) and v>=0 for v in self.hns_beta)
        if self.hns_enabled:
            assert inclusion_hierarchy == 'detail_chain'
            assert self.search_hparams['inclusion_max'] == 0.
            assert self.search_hparams['sparsity_scale'] == 1.
            assert self.search_hparams['view_weights'] == [1.35,1.35,.30]
        self.view_sparsity_weights = [float(v) for v in view_sparsity_weights]
        assert len(self.view_sparsity_weights)==3
        assert all(math.isfinite(v) and v>0 for v in self.view_sparsity_weights)
        assert (math.isclose(sum(self.view_sparsity_weights),5.,abs_tol=1e-12)
                or self.view_sparsity_weights in ([.5,1.,1.5],[1.,2.,3.])), 'Unreviewed sparsity coefficients'
        if self.hns_enabled:assert self.view_sparsity_weights == [1.,2.,2.]

    def optimizer_groups(self):
        text_ids = {id(p) for p in self.clip.mask_net.parameters() if p.requires_grad}
        visual_ids = {id(p) for p in self.fusion_branch.visual_blocks.parameters() if p.requires_grad}
        fusion_ids = {id(p) for p in self.fusion_branch.parameters() if p.requires_grad} - visual_ids
        groups = {name: [] for name in ('backbone','text_mask_and_shared_pool','visual_mask','fusion_adapter')}
        for _,p in self.named_parameters():
            if p.requires_grad:
                target = ('text_mask_and_shared_pool' if id(p) in text_ids else 'visual_mask' if id(p) in visual_ids
                          else 'fusion_adapter' if id(p) in fusion_ids else 'backbone')
                groups[target].append(p)
        assert sum(len(p) for p in groups.values())==len({id(p) for params in groups.values() for p in params})
        peaks = (1e-6,1e-3,1e-3*self.search_hparams['visual_mask_lr_scale'],self.search_hparams['fusion_lr'])
        return [dict(params=groups[name],name=name,lr=lr,peak_lr=lr,
                     weight_decay=.01 if name=='backbone' else 0.,
                     param_names=[n for n,p in self.named_parameters() if any(p is q for q in groups[name])])
                for name,lr in zip(groups,peaks)]

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        valid_global = gather(valid,False)
        valid_count = int(valid_global.sum())
        z,visual = self.encode_visual(images)
        global_z,global_visual = gather(z),tuple(gather(x) for x in visual)
        diagnostics = completed+1 in DIAGNOSTIC_UPDATES
        def terms(tokens, enabled, enabled_global, t2i_weight=1.):
            text,condition = self.encode_view(tokens)
            values = fusion_view_terms(self,z,text,visual,condition,enabled,enabled_global,
                                       global_z,global_visual,diagnostics,
                                       **({"t2i_weight": t2i_weight} if t2i_weight != 1. else {}))
            logs = values[-1]
            if self.inclusion_hierarchy == 'detail_chain':
                # Read-only gate moments on every update, including last50.
                with torch.no_grad():
                    g = self.fusion_branch.balanced_gate(visual[1].detach(), condition[1].detach(), paired=True)[enabled]
                    moments = global_sum(torch.stack((g.sum(), g.square().sum(),
                        ((g<.05)|(g>.95)).float().sum()))) / (int(enabled_global.sum())*g.shape[-1])
                    logs.update(g_mean=moments[0],g_variance=moments[1]-moments[0].square(),g_saturation=moments[2])
            if diagnostics:
                gate=logs['_diagnostic_gate']
                ut,uv = condition[1].detach(),visual[1].detach()
                statistics = torch.stack((((gate<.05)|(gate>.95)).float().mean(-1)[enabled].sum(),
                                          ut.norm(dim=-1)[enabled].sum(),uv.norm(dim=-1)[enabled].sum(),
                                          (ut-uv).norm(dim=-1)[enabled].sum()))
                statistics=global_sum(statistics)/int(enabled_global.sum())
                logs.update(g_saturation=statistics[0],u_T_norm=statistics[1],u_V_norm=statistics[2],
                            u_T_minus_u_V_norm=statistics[3])
            return values
        af,sf,mf,pf,lf=terms(tokens_f,torch.ones_like(valid),torch.ones_like(valid_global))
        gate_f=lf.pop('_diagnostic_gate',None)
        logs={'F_'+key:value for key,value in lf.items()}
        hp=self.search_hparams
        inclusion_enabled=hp['inclusion_max']>0
        weight=hp['inclusion_max']*inclusion_weight(self.arm,completed) if inclusion_enabled and valid_count>=2 else 0.
        if valid_count>=2:
            ao,so,mo,po,lo=terms(tokens_o,valid,valid_global,self.summary_t2i_weight)
            ae,se,me,pe,le=terms(tokens_e,valid,valid_global)
            gate_o,gate_e=lo.pop('_diagnostic_gate',None),le.pop('_diagnostic_gate',None)
            if gate_f is not None:
                values=global_sum(torch.stack(((gate_f-gate_o).abs().mean(-1)[valid].sum(),
                                              (gate_f-gate_e).abs().mean(-1)[valid].sum(),
                                              (gate_o-gate_e).abs().mean(-1)[valid].sum())))/valid_count
                logs.update(g_F_P_abs_difference=values[0],g_F_R_abs_difference=values[1],g_P_R_abs_difference=values[2])
            if inclusion_enabled:
                inc_sum=(detail_chain_inclusion(pf,po,pe) if self.inclusion_hierarchy == 'detail_chain'
                         else inclusion(pf,po,pe))[valid].sum()
            else:
                # INC0: retain the raw violation statistic without an autograd
                # graph, loss term, or evaluation of the inclusion schedule.
                with torch.no_grad():
                    inc_sum=(detail_chain_inclusion(pf,po,pe) if self.inclusion_hierarchy == 'detail_chain'
                             else inclusion(pf,po,pe))[valid].sum()
            wf,wp,wr=hp['view_weights']
            align=(10/3*(af+ao+ae) if hp['view_weights']==[1.,1.,1.] else
                   10/(wf+wp+wr)*(wf*af+wp*ao+wr*ae))
            if self.summary_t2i_weight != 1.:
                assert hp["view_weights"] == [1.,1.,1.]
                align = align * (6/(5+self.summary_t2i_weight))
            # Preserve default arithmetic; explicit coefficients are absolute,
            # including WeakSparse's mass3, and are never normalized.
            cf,co,ce=self.view_sparsity_weights
            sparse=((sf+2*so+2*se)/3 if self.view_sparsity_weights==[1.,2.,2.] else
                    (cf*sf+co*so+ce*se)/3)
            loss=align+hp['sparsity_scale']*sparse
            if self.hns_enabled:
                world=world_rank()[0]
                surcharge,hns=hns_terms(mf,mo,me,valid,completed,world,valid_count,self.hns_beta,self.hns_detach_child)
                loss=loss+surcharge
                values=global_sum(torch.stack((align.detach(),sparse.detach(),hns['V_DF'].detach(),
                    hns['V_3D'].detach(),surcharge.detach())))/world
                logs.update(HNS_enabled=True,HNS_align=values[0],HNS_original_sparse=values[1],
                    V_DF_hard=values[2],V_3D_hard=values[3],HNS_surcharge=values[4],
                    HNS_regularizer=values[1]+values[4],lambda_h=hns['lambda_h'],
                    HNS_beta_DF=self.hns_beta[0],HNS_beta_3D=self.hns_beta[1],
                    HNS_detach_child=self.hns_detach_child)
                telemetry,width=hard_telemetry(mf,mo,me,valid)
                totals=global_sum(torch.stack(list(telemetry.values())))
                totals=dict(zip(telemetry,totals))
                logs['HNS_mask_width']=width
                logs['HNS_valid_count']=valid_count
                for edge in ('DF','3D'):
                    for key in ('violation_count','child_only_count','parent_only_count','intersection_count'):
                        logs['HNS_'+edge+'_'+key]=totals[edge+'_'+key]
                    logs['HNS_'+edge+'_hard_violation_ratio']=totals[edge+'_violation_count']/(valid_count*width)
                    logs['HNS_'+edge+'_IoU']=totals[edge+'_IoU_sum']/valid_count
                    logs['HNS_'+edge+'_exact_equality_ratio']=totals[edge+'_exact_equal_samples']/valid_count
                    logs['HNS_'+edge+'_coordinate_equality_ratio']=totals[edge+'_equal_coordinates']/(valid_count*width)
                for name in ('F','Dall','D3'):
                    logs['HNS_'+name+'_keep']=totals[name+'_selected_count']/(valid_count*width)
                logs['HNS_gap_F_D']=logs['HNS_F_keep']-logs['HNS_Dall_keep']
                logs['HNS_gap_D_D3']=logs['HNS_Dall_keep']-logs['HNS_D3_keep']
                logs['HNS_triple_exact_equality_ratio']=totals['triple_equal_samples']/valid_count
                logs['HNS_triple_coordinate_equality_ratio']=totals['triple_equal_coordinates']/(valid_count*width)
                # Enabled only by a separate read-only gradient audit process.
                if getattr(self,'capture_hns_graph',False):
                    self.hns_graph=dict(alignment=align,original_sparsity=sparse,
                        V_DF=hns['V_DF'],V_3D=hns['V_3D'],total_HNS=sparse+surcharge,
                        total_training=loss,masks=dict(F=mf,Dall=mo,D3=me),
                        probabilities=dict(F=pf,Dall=po,D3=pe),lambda_h=hns['lambda_h'],beta=self.hns_beta)
            if inclusion_enabled:
                loss=loss+weight*world_rank()[0]/valid_count*inc_sum
            violation=.5*((mo.detach()>mf.detach()).float().mean(-1)+(me.detach()>mf.detach()).float().mean(-1))
            iou=(mo.detach()*me.detach()).sum(-1)/((mo.detach()+me.detach())>0).sum(-1).clamp_min(1)
            extra=global_sum(torch.stack((inc_sum.detach(),violation[valid].sum(),iou[valid].sum())))/valid_count
            logs.update({'O_'+key:value for key,value in lo.items()})
            logs.update({'E_'+key:value for key,value in le.items()})
            logs.update(inc=extra[0],inclusion_loss=weight*extra[0],hard_inclusion_violation=extra[1],oe_iou=extra[2])
            if self.inclusion_hierarchy == 'detail_chain':
                with torch.no_grad():
                    f,o,e = mf.detach(),mo.detach(),me.detach()
                    fo_iou = (f*o).sum(-1)/((f+o)>0).sum(-1).clamp_min(1)
                    fo_violation = (o>f).float().mean(-1)
                    oe_violation = (e>o).float().mean(-1)
                    chain = global_sum(torch.stack((fo_iou[valid].sum(),fo_violation[valid].sum(),
                        oe_violation[valid].sum())))/valid_count
                logs.update(F_Dall_mask_iou=chain[0],Dall_Ds_mask_iou=extra[2],
                    Dall_F_hard_violation=chain[1],Ds_Dall_hard_violation=chain[2],
                    hard_inclusion_violation=.5*(chain[1]+chain[2]))
        else:
            loss=10*af+hp['sparsity_scale']*sf
            if self.hns_enabled:
                logs.update(HNS_enabled=True,lambda_h=hierarchy_weight(completed),
                    HNS_beta_DF=self.hns_beta[0],HNS_beta_3D=self.hns_beta[1],HNS_detach_child=self.hns_detach_child,
                    HNS_align=global_sum(10*af)/world_rank()[0],HNS_original_sparse=global_sum(sf)/world_rank()[0],
                    HNS_surcharge=0.,HNS_regularizer=global_sum(sf)/world_rank()[0],V_DF_hard=0.,V_3D_hard=0.)
            logs.update(inc=0.,inclusion_loss=0.,hard_inclusion_violation=0.,oe_iou=0.,O_candidates=0,E_candidates=0)
        logs.update(loss=global_sum(loss)/world_rank()[0],inc_weight=weight,inclusion_enabled=inclusion_enabled,valid_global=valid_count,
                    fusion=self.fusion,visual=self.visual,condition_mode=self.condition_mode,shuffle_shift=0,
                    nonfinite=global_sum((~torch.isfinite(loss)).sum()))
        return loss,{key:value.detach() if torch.is_tensor(value) else value for key,value in logs.items()}


def migrate_legacy_optimizer(old_state, module, optimizer):
    """Split the reviewed three-group B0 optimizer by parameter name, preserving moments/step/LR."""
    assert module.search_hparams==hparams({}), 'Only B0 may migrate a legacy optimizer'
    groups=old_state['param_groups']
    assert [g['name'] for g in groups]==['backbone','shared_mask','joint_adapter']
    mask_ids={id(p) for p in module.mask_parameters()}
    auxiliary_ids={id(p) for p in module.fusion_branch.parameters()}-mask_ids
    old_names={name:[] for name in ('backbone','shared_mask','joint_adapter')}
    for name,p in module.named_parameters():
        if p.requires_grad:
            target='shared_mask' if id(p) in mask_ids else 'joint_adapter' if id(p) in auxiliary_ids else 'backbone'
            old_names[target].append(name)
    states={}
    for group in groups:
        names=old_names[group['name']]
        assert len(names)==len(group['params'])
        for name,index in zip(names,group['params']):
            if index in old_state['state']:
                states[name]=copy.deepcopy(old_state['state'][index])
    new_state=optimizer.state_dict()
    new_state['state']={}
    for group in new_state['param_groups']:
        old_group=groups[0] if group['name']=='backbone' else groups[2] if group['name']=='fusion_adapter' else groups[1]
        for key in ('lr','betas','eps','weight_decay','amsgrad','maximize','foreach','capturable','differentiable','fused'):
            if key in old_group:
                group[key]=old_group[key]
        for name,index in zip(group['param_names'],group['params']):
            if name in states:new_state['state'][index]=states[name]
    assert len(new_state['state'])==len(old_state['state'])
    optimizer.load_state_dict(new_state)
    return dict(migrated=True,parameter_states=len(states),source_groups=3,destination_groups=4,
                steps=sorted({int(v['step']) for v in states.values()}))
