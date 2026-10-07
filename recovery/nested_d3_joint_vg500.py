"""Audit-frozen joint V/G, fresh common0/local500, native evaluation/publish."""
import json
import math
import statistics

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha,rows
from recovery.nested_d3_vg_audit import regions_from_distributions

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_joint_vg500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-joint-vg500-20261007'
BRANCH='experiment/nested-d3-joint-vg500-v1'
ARM='Joint-VG'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
CONFIG=ROOT/'configs/nested_d3_joint_vg500.json'
REFERENCES={'Anchor':BASE_EXP,
    'INC0':ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1',
    'Coupled v1':ROOT/'experiments/nest_clip_v1/nested_d3_coupled_reg500_v1',
    'BBNS':ROOT/'experiments/nest_clip_v1/nested_d3_bbns500_v1'}
QUALITY=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
EDGE_KEYS=('V','G','V_penalty','G_lower_penalty','G_upper_penalty','V_legal','G_legal','zero_edge','edge_loss')
VG_KEYS=tuple(e+'_'+k for e in ('Dall_F','D3_Dall') for k in EDGE_KEYS)+(
    'F_mean_soft_support','Dall_mean_soft_support','D3_mean_soft_support',
    'Omega_F','total_vg_edges','total_nested_regularizer')
ARMS={}


def gradient_audit():
    import torch
    from model.nested_joint_vg import joint_vg_edge
    region=dict(eps_v=.05,gamma_low=.1,gamma_high=.3)
    cases={}
    for name,child,parent,term in [('upward',[.9,.8],[.3,.4],'V_penalty'),
        ('lower',[.55,.60],[.60,.65],'G_lower_penalty'),
        ('upper',[.2,.3],[.8,.9],'G_upper_penalty'),
        ('inside',[.6,.6],[.75,.75],'G_lower_penalty'),
        ('identical',[.6,.6],[.6,.6],'G_lower_penalty')]:
        c=torch.tensor([child],dtype=torch.float64,requires_grad=True)
        p=torch.tensor([parent],dtype=torch.float64,requires_grad=True)
        values=joint_vg_edge(c,p,region)
        gc,gp=torch.autograd.grad(values[term].sum(),(c,p))
        assert torch.isfinite(gc).all() and torch.isfinite(gp).all()
        if name in ('upward','lower','upper'):
            assert torch.count_nonzero(gc) and torch.count_nonzero(gp)
        if name in ('upward','lower'):assert (gc>0).all() and (gp<0).all()
        if name=='upper':assert (gc<0).all() and (gp>0).all()
        cases[name]=dict(child=child,parent=parent,term=term,
            values={k:v.item() for k,v in values.items()},child_gradient=gc.tolist(),parent_gradient=gp.tolist())
    assert cases['identical']['values']['G']==0 and cases['identical']['values']['G_lower_penalty']>0
    return dict(passed=True,epsilon=1e-6,joint_parent_child_gradients=True,no_stop_gradient=True,cases=cases,
        exact_equality_caveat='At exact equality PyTorch ReLU has zero subgradient: lower penalty is positive, but its endpoint gradients are zero. Positive gamma_low discourages collapse away from this kink; it does not mathematically guarantee escape from exact collapse.')


def vg_audit(actual,regions):
    assert len(actual)==500
    for row in actual:
        assert row['regularizer_mode']=='joint_vg' and row['inc_weight']==row['inclusion_loss']==0
        assert row['inclusion_enabled'] is False and row['old_inclusion_schedule_applied'] is False
        assert row['independent_child_sparse_applied']==row['independent_inclusion_applied']==0
        edges=0.
        for name in ('Dall_F','D3_Dall'):
            total=sum(row[name+'_'+key] for key in ('V_penalty','G_lower_penalty','G_upper_penalty'))
            assert math.isclose(total,row[name+'_edge_loss'],rel_tol=3e-6,abs_tol=1e-7)
            assert 0<=row[name+'_zero_edge']<=min(row[name+'_V_legal'],row[name+'_G_legal'])<=1
            edges+=total
        assert math.isclose(edges,row['total_vg_edges'],rel_tol=3e-6,abs_tol=1e-7)
        assert math.isclose(row['total_nested_regularizer'],row['Omega_F']/3+edges,rel_tol=3e-6,abs_tol=1e-7)
    means=lambda records:{k:statistics.fmean(r[k] for r in records) for k in VG_KEYS}
    keys=VG_KEYS+('F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i',
        'F_keep_ratio','O_keep_ratio','E_keep_ratio','F_Dall_mask_iou','Dall_Ds_mask_iou',
        'Dall_F_hard_violation','Ds_Dall_hard_violation')
    return dict(passed=True,updates=500,regions=regions,regions_frozen_from_Anchor_only=True,
        no_old_child_global_sparse=True,no_old_independent_inclusion=True,no_ramp_beta_or_stop_gradient=True,
        formula='alignment + Omega_F/3 + mean_valid(edge_Dall_F + edge_D3_Dall)',
        normalization='Differentiable world/valid_count local sums, DDP gradient averaging',
        last50=means(actual[-50:]),all500=means(actual),
        diagnostic_steps={str(r['step']):{k:r[k] for k in keys} for r in actual if r['step'] in (1,100,200,500)},
        per_update_sanitized=[dict(step=r['step'],**{k:r[k] for k in VG_KEYS}) for r in actual])


def classify(q,base,stable,trivial):
    d={k:q[k]-base[k] for k in QUALITY};eps=1e-6
    if trivial:return 'JOINT_VG_NEGATIVE'
    if (q['Score5']>71.158309 and q['J_long3']>=74.978516 and q['Urban_T2I']>=89.6
        and q['Short4']>=65.30 and stable):return 'JOINT_VG_STRONG_POSITIVE'
    if d['Score5']>=.05-eps and d['J_long3']>=-eps and d['Urban_T2I']>=-.1-eps:
        return 'JOINT_VG_POSITIVE'
    flat=all(abs(d[k])<=tol+eps for k,tol in [('Score5',.05),('J_long3',.05),('Urban_T2I',.1),('Short4',.15)])
    if flat and stable:return 'JOINT_VG_STRUCTURAL_SUCCESS'
    if d['Score5']<-.15 and d['J_long3']<-.15:return 'JOINT_VG_NEGATIVE'
    return 'JOINT_VG_TRADEOFF'


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    frozen=json.loads((EXP/'ANCHOR_VG_AUDIT.json').read_text());regions=ARMS[ARM]['vg_regions']
    assert regions==regions_from_distributions(frozen['soft_distributions'])==frozen['frozen_regions']
    assert json.loads(CONFIG.read_text())==json.loads((EXP/'config.json').read_text())==search.arm_config(ARM)
    result=json.loads((EXP/'RESULTS.json').read_text());diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text());gradient=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    actual=rows(RUN_ROOT/ARM/'step500/steps.jsonl');reference=rows(BASE_RUN/'step500/steps.jsonl')
    assert search.matched_stream(actual,reference,ARM)['records']==512000
    audit=vg_audit(actual,regions);routing=gradient_audit();means=audit['last50']
    stable=all(means[e+'_V_legal']>=.50 and means[e+'_G_legal']>=.50 and means[e+'_zero_edge']>=.40 for e in regions)
    collapsed=all(means[e+'_G']<.005 for e in regions)
    inflated=all(means[v+'_mean_soft_support']>.99 for v in ('F','Dall','D3'))
    oversparse=masks['keep_ratio']['D3']<.50
    trivial=collapsed or inflated or oversparse
    audit.update(many_samples_inside_feasible_region=stable,parent_child_collapse=collapsed,
        all_masks_near_one=inflated,D3_over_sparse=oversparse,trivial_solution=trivial,
        checkpoint_audit_sha256=sha(EXP/'ANCHOR_VG_AUDIT.json'),frozen_config_sha256=sha(CONFIG),
        hit_rate_units='Fractions of valid samples; multiply100 for percent')
    diag['joint_vg_last50']=means;diag['joint_vg_diagnostic_steps']=audit['diagnostic_steps']
    masks.update(regularizer_mode='joint_vg',inclusion_edges=[],joint_vg_edges=['Dall->F','D3->Dall'],
        ramp200=False,inclusion_max=0,detached_child=False,detached_parent=False,
        inclusion_loss_active=False,raw_inc_is_telemetry_only=True,
        legacy_config_inclusion_max_unused=1,old_inclusion_ramp_applied=False,vg_regions=regions,
        coverage_hierarchy_ordered=masks['keep_ratio']['F']>masks['keep_ratio']['Dall']>masks['keep_ratio']['D3'])
    q=quality(result);references={};deltas={};recalls={};mask_deltas={};gradient_deltas={}
    for name,path in REFERENCES.items():
        r=json.loads((path/'RESULTS.json').read_text());value=quality(r)
        references[name]=dict(scores=value,path=str(path/'RESULTS.json'),sha256=sha(path/'RESULTS.json'))
        deltas[name]={k:q[k]-value[k] for k in QUALITY};recalls[name]=recall_delta(result,r)
        old=json.loads((path/'MASK_HIERARCHY_AUDIT.json').read_text())
        od=json.loads((path/'TRAINING_DIAGNOSTICS.json').read_text())
        mask_deltas[name]=dict(last50={k:v-old['last50'][k] for k,v in masks['last50'].items()},
            keep_ratio={k:v-od['last50_views'][k]['keep_ratio'] for k,v in masks['keep_ratio'].items()})
        g=json.loads((path/'GRADIENT_SPOTCHECK.json').read_text())
        ratio=g['weighted_mean_gradient_norms']['D3']/g['weighted_mean_gradient_norms']['Dall']
        gradient_deltas[name]=gradient['weighted_lowest_Dall_ratio']-ratio
    status=classify(q,references['Anchor']['scores'],stable,trivial)
    result.update(classification=status,references=references,quality_delta_vs_references_pp=deltas,
        recall_delta_vs_references_pp=recalls,vg_audit_passed=True,joint_gradient_audit_passed=True,
        feasible_region_behavior=stable,trivial_solution=trivial,gradient_ratio_delta_vs_references=gradient_deltas,
        sole_changed_method='visual_support_regularization',frozen_vg_regions=regions)
    masks['delta_vs_references']=mask_deltas
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),('VG_AUDIT',audit),
        ('GRADIENT_AUDIT',dict(passed=True,joint_edge_gradient_routing=routing,backbone_alignment_spotcheck=gradient))]:
        dump(EXP/(name+'.json'),value)
    original=(EXP/'REPORT.md').read_text().replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'One frozen Joint-VG arm; classification after complete native evaluation and feasible-region review.').replace(
        'Inclusion max:1; zero bypasses the schedule and inclusion autograd graph.',
        'Old inclusion/ramp disabled; legacy config inclusion_max1 is unused. Joint V/G gradients train both endpoints.').replace(
        'Soft detached-child inclusion does not impose zero hard-mask violations.',
        'V/G jointly optimize both endpoints; hard violations remain telemetry.').replace(
        'absolute sparsity coefficients:[1.0, 2.0, 2.0] (mass5.0)',
        'Joint-VG regularizer:Omega_F/3 + two V/G edges; old child coefficients unused')
    lines=['','## Joint V/G frozen feasible-region comparison','',f'Classification: `{status}`.',
        'Loss: alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall). No stop-gradient,child global sparsity,old inclusion,ramp,beta or online region changes.',
        f'Frozen regions: `{regions}`; eps_v=P80(V),gamma_low=P20(G),gamma_high=P80(G) from fixed16384 Anchor training samples.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in [(n,v['scores']) for n,v in references.items()]+[(ARM,q)]:
        lines.append('| '+name+' | '+' | '.join(f'{value[k]:.6f}' for k in QUALITY)+' |')
    lines+=['','| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,delta in deltas.items():lines.append('| '+name+' | '+' | '.join(f'{delta[k]:+.6f}' for k in QUALITY)+' |')
    for name in references:
        lines+=['',f'All native recalls vs {name} (pp):','',
            '| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |','|---|---|---|']
        for dataset,v in recalls[name].items():
            values=[' / '.join(f'{v[d][k]:+.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
            lines.append(f'| {dataset} | {values[0]} | {values[1]} |')
    lines+=['',f'Last50 V/G telemetry: `{means}`.',f'Feasible-region behavior:{stable}; collapse:{collapsed}; all-near-one:{inflated}; D3-over-sparse:{oversparse}.',
        f'Keep/IoU/violation changes: `{mask_deltas}`.',
        f'Weighted D3/Dall gradient ratio:{gradient["weighted_lowest_Dall_ratio"]}; deltas:`{gradient_deltas}`.',
        routing['exact_equality_caveat'],
        'Diagnostic steps1/100/200/500 and last50 are in VG_AUDIT.json and TRAINING_DIAGNOSTICS.json.',
        'All512000 sample/text/token/K/index/LR trajectories match Anchor. Final500 full checkpoint and strict native bare immutable.',
        'Stopped500; no full,quantile/lambda/beta/K/weight search,SG version or other experiment. Binary/raw paths,size,SHA,time ranges in RUNTIME_STATS.json.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    global ARMS
    audit=json.loads((EXP/'ANCHOR_VG_AUDIT.json').read_text())
    assert audit['passed'] and audit['records']>=16384 and audit['no_parameter_updates'] and audit['checkpoint_unchanged']
    regions=regions_from_distributions(audit['soft_distributions']);assert regions==audit['frozen_regions']
    ARMS={ARM:dict(axis='joint_vg',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',
        regularizer_mode='joint_vg',vg_regions=regions,experiment_dir=str(EXP))}
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.nested_d3_joint_vg500'
    search.PHASE_PREFIX='formal-nested-d3-joint-vg500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/nested_d3_joint_vg500.py',
        'recovery/nested_d3_vg_audit.py','tests/test_nested_d3_joint_vg500.py','model/nested_joint_vg.py',
        'recovery/check_stage500_publish.py',str(CONFIG.relative_to(ROOT))}
    search.EXTRA_SOURCES.update(str((EXP/n).relative_to(ROOT)) for n in (
        'ANCHOR_VG_AUDIT.json','ANCHOR_VG_AUDIT.md','ANCHOR_VG_COHORT.json'))
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report frozen joint V/G Nested D3 local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    for name in ('VG_AUDIT.json','GRADIENT_AUDIT.json','ANCHOR_VG_AUDIT.json','ANCHOR_VG_AUDIT.md','ANCHOR_VG_COHORT.json'):
        if name not in runner.REPORT_NAMES:runner.REPORT_NAMES+=(name,)
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
