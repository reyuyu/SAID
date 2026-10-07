"""Frozen audit-derived BBNS, fresh common0/local500, native evaluation/publish."""
import json
import math
import statistics

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha,rows
from recovery.nested_d3_support_audit import bands_from_distributions

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_bbns500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-bbns500-20261007'
BRANCH='experiment/nested-d3-bbns500-v1'
ARM='BBNS'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
OFF_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
COUPLED_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_coupled_reg500_v1'
CONFIG=ROOT/'configs/nested_d3_bbns500.json'
QUALITY=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
EDGE_KEYS=('coverage','relative_support','cover_loss','lower_band_loss','upper_band_loss',
           'zero_refine_band','zero_edge_band','percent_inside_zero_loss_band',
           'percent_inside_refinement_band','edge_loss')
ARMS={}


def routing_audit():
    import torch
    from model.nested_support_band import support_band_edge
    cases={}
    for name,child,parent in [('lower',[.2,.2],[.9,.9]),('upper',[.9,.9],[.3,.3]),
                              ('inside',[.5,.5],[.9,.9])]:
        c=torch.tensor([child],dtype=torch.float64,requires_grad=True)
        p=torch.tensor([parent],dtype=torch.float64,requires_grad=True)
        values=support_band_edge(c,p,dict(kappa=.8,tau_low=.4,tau_high=.6))
        gc,gp=torch.autograd.grad(values['cover_loss'].sum(),(c,p),allow_unused=True,retain_graph=True)
        rc,rp=torch.autograd.grad((values['lower_band_loss']+values['upper_band_loss']).sum(),
                                 (c,p),allow_unused=True)
        assert gc is None and rp is None
        if name=='lower':assert (rc<0).all()
        if name=='upper':assert (rc>0).all() and (gp<0).all()
        if name=='inside':assert values['cover_loss'].item()==0 and torch.count_nonzero(rc)==0
        cases[name]=dict(child=child,parent=parent,coverage=values['coverage'].item(),
            relative_support=values['relative_support'].item(),cover_loss=values['cover_loss'].item(),
            lower_band_loss=values['lower_band_loss'].item(),upper_band_loss=values['upper_band_loss'].item(),
            cover_child_gradient=None,cover_parent_gradient=gp.tolist(),
            refine_child_gradient=rc.tolist(),refine_parent_gradient=None)
    return dict(passed=True,epsilon=1e-6,cover_trains_parent_only=True,refinement_trains_child_only=True,
        below_band_gradient_descent_expands_child=True,above_band_gradient_descent_shrinks_child=True,cases=cases)


def band_audit(actual,bands):
    assert len(actual)==500
    for row in actual:
        assert row['regularizer_mode']=='bbns' and row['inc_weight']==row['inclusion_loss']==0
        assert row['inclusion_enabled'] is False
        assert row['independent_child_sparse_applied']==row['independent_inclusion_applied']==0
        edges=0.
        for name in ('Dall_F','D3_Dall'):
            total=sum(row[name+'_'+key] for key in ('cover_loss','lower_band_loss','upper_band_loss'))
            assert math.isclose(total,row[name+'_edge_loss'],rel_tol=3e-6,abs_tol=1e-7)
            assert 0<=row[name+'_zero_edge_band']<=row[name+'_zero_refine_band']<=1
            assert math.isclose(row[name+'_percent_inside_zero_loss_band'],100*row[name+'_zero_edge_band'],rel_tol=3e-6,abs_tol=1e-6)
            edges+=total
        assert math.isclose(row['total_nested_regularizer'],row['Omega_F']/3+edges,rel_tol=3e-6,abs_tol=1e-7)
    keys=tuple(name+'_'+key for name in ('Dall_F','D3_Dall') for key in EDGE_KEYS)+('Omega_F','total_band_edges','total_nested_regularizer')
    means=lambda records:{k:statistics.fmean(r[k] for r in records) for k in keys}
    return dict(passed=True,updates=500,bands=bands,bands_frozen_from_Anchor_only=True,
        no_old_child_global_sparse=True,no_old_independent_inclusion=True,no_inclusion_ramp_or_beta=True,
        formula='alignment + Omega_F/3 + mean_valid(edge_Dall_F + edge_D3_Dall)',
        normalization='Differentiable world/valid_count local sums, DDP gradient averaging',
        last50=means(actual[-50:]),all500=means(actual),
        diagnostic_steps={str(r['step']):{k:r[k] for k in keys} for r in actual if r['step'] in (1,100,200,500)},
        per_update_sanitized=[dict(step=r['step'],**{k:r[k] for k in keys}) for r in actual])


def classify(q,base,keep,band):
    d={k:q[k]-base[k] for k in QUALITY};eps=1e-6
    if q['Score5']>71.158309 and q['J_long3']>=74.978516 and q['Urban_T2I']>=89.6 and q['Short4']>=65.30:
        return 'BBNS_STRONG_POSITIVE'
    if d['Score5']>=.05-eps and d['J_long3']>=-eps and d['Urban_T2I']>=-.1-eps and band:
        return 'BBNS_POSITIVE'
    flat=all(abs(d[k])<=tol+eps for k,tol in [('Score5',.05),('J_long3',.05),('Urban_T2I',.1),('Short4',.15)])
    if flat and keep['D3']>=.77 and keep['F']>keep['Dall']>keep['D3'] and band:
        return 'BBNS_REGULARIZATION_SUCCESS'
    if all(d[k]<-eps for k in ('Score5','J_long3','Urban_T2I')):return 'BBNS_NEGATIVE'
    return 'BBNS_TRADEOFF'


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    anchor_audit=json.loads((EXP/'ANCHOR_SUPPORT_AUDIT.json').read_text())
    bands=ARMS[ARM]['support_bands']
    assert bands==bands_from_distributions(anchor_audit['soft_distributions'])==anchor_audit['frozen_bands']
    assert json.loads(CONFIG.read_text())==json.loads((EXP/'config.json').read_text())==search.arm_config(ARM)
    result=json.loads((EXP/'RESULTS.json').read_text());diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text());gradient=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    actual=rows(RUN_ROOT/ARM/'step500/steps.jsonl');reference=rows(BASE_RUN/'step500/steps.jsonl')
    assert search.matched_stream(actual,reference,ARM)['records']==512000
    audit=band_audit(actual,bands);routing=json.loads((EXP/'GRADIENT_ROUTING_AUDIT.json').read_text())
    assert routing['passed'] and routing==routing_audit()
    means=audit['last50']
    mean_support_inside=all(bands[e]['tau_low']<=means[e+'_relative_support']<=bands[e]['tau_high'] and
        means[e+'_coverage']>=bands[e]['kappa'] for e in bands)
    hit=all(means[e+'_zero_refine_band']>=.50 and means[e+'_zero_edge_band']>=.40 for e in bands)
    expected=mean_support_inside and hit
    audit.update(mean_support_inside_bounds=mean_support_inside,many_samples_zero_band=hit,
        expected_band_behavior=expected,anchor_audit_sha256=sha(EXP/'ANCHOR_SUPPORT_AUDIT.json'),
        frozen_config_sha256=sha(CONFIG),zero_band_units='Fraction of valid samples; multiply100 for percent')
    diag['support_band_last50']=means
    masks.update(regularizer_mode='bbns',old_inclusion_edges_applied=[],BBNS_edges=['Dall->F','D3->Dall'],
        inclusion_edges=[],ramp200=False,inclusion_max=0,detached_child=False,
        BBNS_cover_child_detached=True,BBNS_refine_parent_detached=True,
        legacy_config_inclusion_max_unused=1,old_inclusion_ramp_applied=False,support_bands=bands,support_last50=means,
        coverage_hierarchy_ordered=masks['keep_ratio']['F']>masks['keep_ratio']['Dall']>masks['keep_ratio']['D3'])
    q=quality(result);references={};deltas={};recalls={};mask_deltas={};gradient_deltas={}
    for name,path in [('Anchor',BASE_EXP),('INC0',OFF_EXP),('Coupled v1',COUPLED_EXP)]:
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
    status=classify(q,references['Anchor']['scores'],masks['keep_ratio'],expected)
    result.update(classification=status,references=references,quality_delta_vs_references_pp=deltas,
        recall_delta_vs_references_pp=recalls,support_band_audit_passed=True,gradient_routing_audit_passed=True,
        expected_band_behavior=expected,gradient_ratio_delta_vs_references=gradient_deltas,
        sole_changed_method='visual_support_regularization',frozen_bands=bands)
    masks['delta_vs_references']=mask_deltas
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('SUPPORT_BAND_AUDIT',audit)]:dump(EXP/(name+'.json'),value)
    original=(EXP/'REPORT.md').read_text().replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'One frozen BBNS arm, classified after complete native evaluation and support review.').replace(
        'Inclusion max:1; zero bypasses the schedule and inclusion autograd graph.',
        'Old inclusion is disabled in BBNS; the legacy config value1 is unused. No old ramp/autograd loss.').replace(
        'Soft detached-child inclusion does not impose zero hard-mask violations.',
        'BBNS uses detached child coverage and detached parent refinement; hard violations remain telemetry.')
    original=original.replace(
        'absolute sparsity coefficients:[1.0, 2.0, 2.0] (mass5.0)',
        'BBNS regularizer:Omega_F/3 + two support-band edges; old child coefficients unused')
    lines=['','## BBNS frozen support band comparison','',f'Classification: `{status}`.',
        'Formula: alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall); each edge cover+lower+upper squared hinges. No child global sparse,old inclusion,ramp,beta or online band adjustment.',
        f'Before training, independently per edge: `{bands}`; kappa=P20(C),tau_low=P20(R),tau_high=P80(R) from fixed16384 successful Anchor samples only.',
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
    lines+=['',f'Last50 support/hit/loss telemetry: `{means}`.',
        f'Mean support inside bounds:{mean_support_inside}; many samples zero band:{hit}; expected behavior:{expected}.',
        f'Keep/IoU/violation changes vs references: `{mask_deltas}`.',
        f'Weighted D3/Dall gradient ratio:{gradient["weighted_lowest_Dall_ratio"]}; deltas:`{gradient_deltas}`.',
        'Gradient routing manual tensor audit passed; GRADIENT_ROUTING_AUDIT.json contains separate loss/gradient cases.',
        'Anchor audit stats and exact pretraining thresholds in ANCHOR_SUPPORT_AUDIT.md/JSON; fixed sample IDs in ANCHOR_SUPPORT_COHORT.json.',
        'Operational classification thresholds were declared before training in SEARCH_PLAN.json, never selected against retrieval.',
        'All512000 sample/text/token/K/index/LR trajectories match Anchor. Final500 full checkpoint and strict native bare immutable.',
        'Stopped500; no full,new bands/beta/K/weights/other experiment. Binary/raw paths,size,SHA,time ranges in RUNTIME_STATS.json.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    global ARMS
    audit=json.loads((EXP/'ANCHOR_SUPPORT_AUDIT.json').read_text())
    assert audit['passed'] and audit['records']>=16384 and audit['no_parameter_updates'] and audit['checkpoint_unchanged']
    bands=bands_from_distributions(audit['soft_distributions']);assert bands==audit['frozen_bands']
    ARMS={ARM:dict(axis='support_band',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',
        regularizer_mode='bbns',support_bands=bands,experiment_dir=str(EXP))}
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.nested_d3_bbns500'
    search.PHASE_PREFIX='formal-nested-d3-bbns500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/nested_d3_bbns500.py',
        'recovery/nested_d3_support_audit.py','tests/test_nested_d3_bbns500.py','model/nested_support_band.py',
        'recovery/check_stage500_publish.py',str(CONFIG.relative_to(ROOT))}
    search.EXTRA_SOURCES.update(str((EXP/n).relative_to(ROOT)) for n in (
        'ANCHOR_SUPPORT_AUDIT.json','ANCHOR_SUPPORT_AUDIT.md','ANCHOR_SUPPORT_COHORT.json','GRADIENT_ROUTING_AUDIT.json'))
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report frozen audit-derived Nested D3 BBNS local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    for name in ('SUPPORT_BAND_AUDIT.json','GRADIENT_ROUTING_AUDIT.json','ANCHOR_SUPPORT_AUDIT.json',
                 'ANCHOR_SUPPORT_AUDIT.md','ANCHOR_SUPPORT_COHORT.json'):
        if name not in runner.REPORT_NAMES:runner.REPORT_NAMES+=(name,)
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
