"""One fresh local500 coupled edge regularizer, native evaluation and publication."""
import json
import math
import statistics

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha,rows

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_coupled_reg500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-coupled-reg500-20261007'
BRANCH='experiment/nested-d3-coupled-reg500-v1'
ARM='CoupledNested'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
OFF_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
HALF_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc05_500_v1'
ARMS={ARM:dict(axis='regularizer',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',
    regularizer_mode='coupled_nested',experiment_dir=str(EXP))}
QUALITY=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
REG_KEYS=('L_in_Dall_F','L_out_Dall_F','L_in_D3_Dall','L_out_D3_Dall','Omega_F',
    'total_inside_sparsity','total_outside_penalty','total_nested_regularizer','old_global_sparse_counterfactual',
    'Dall_F_inside_parent_keep_ratio','D3_Dall_inside_parent_keep_ratio',
    'Dall_F_outside_probability_mass','D3_Dall_outside_probability_mass',
    'Dall_F_outside_mass_ratio','D3_Dall_outside_mass_ratio')


def routing_example():
    """Small analytic example published with the trained regularizer audit."""
    import torch
    from model.balanced_hparam_search import nested_edge_terms
    c=torch.tensor([[.8,.2]],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([[.1,.9]],dtype=torch.float64,requires_grad=True)
    inside,outside=nested_edge_terms(c,p)
    outside_grad=torch.autograd.grad(outside.sum(),(c,p),allow_unused=True)
    inside_grad=torch.autograd.grad(inside.sum(),(c,p),allow_unused=True)
    assert outside_grad[0] is None and inside_grad[1] is None
    assert torch.equal(outside_grad[1],torch.tensor([[-.5,0.]],dtype=torch.float64))
    assert torch.allclose(inside_grad[0],p.detach()/(p.detach().sum()+1e-6))
    return dict(passed=True,child=c.tolist(),parent=p.tolist(),inside=inside.item(),outside=outside.item(),
        outside_child_gradient=None,outside_parent_gradient=outside_grad[1].tolist(),
        inside_child_gradient=inside_grad[0].tolist(),inside_parent_gradient=None,
        epsilon=1e-6,parent_detached_in_inside=True,child_detached_in_outside=True)


def regularizer_audit(actual,reference):
    assert len(actual)==len(reference)==500
    matched=[]
    for a,b in zip(actual,reference):
        assert a['step']==b['step'] and a['valid_global']>=2
        lam=min(1.,(a['step']-1)/200.)
        assert a['inc_weight']==b['inc_weight']==lam
        assert a['independent_child_sparse_applied']==a['independent_inclusion_applied']==0
        expected=.5*lam*(a['L_out_Dall_F']+a['L_out_D3_Dall'])
        assert math.isclose(a['total_outside_penalty'],expected,rel_tol=2e-6,abs_tol=1e-7)
        assert math.isclose(expected,a['inclusion_loss'],rel_tol=2e-6,abs_tol=1e-7)
        old=(a['F_sparse']+2*a['O_sparse']+2*a['E_sparse'])/3
        assert math.isclose(old,a['old_global_sparse_counterfactual'],rel_tol=2e-6,abs_tol=1e-7)
        assert math.isclose(a['total_nested_regularizer'],a['Omega_F']/3+a['total_inside_sparsity']+expected,
                            rel_tol=2e-6,abs_tol=1e-7)
        anchor_sparse=(b['F_sparse']+2*b['O_sparse']+2*b['E_sparse'])/3
        matched.append(dict(step=a['step'],nested_regularizer=a['total_nested_regularizer'],
            conditional_inside=a['total_inside_sparsity'],outside_penalty=expected,Omega_F=a['Omega_F'],
            actual_Anchor_global_sparse=anchor_sparse,actual_Anchor_inclusion=b['inclusion_loss'],
            counterfactual_old_global_sparse_same_current_masks=old,
            delta_vs_matched_Anchor_total=a['total_nested_regularizer']-anchor_sparse-b['inclusion_loss']))
    means=lambda rs,ks:{k:statistics.fmean(r[k] for r in rs) for k in ks}
    return dict(passed=True,updates=500,old_child_global_sparse_applied=False,independent_inclusion_applied=False,
        outside_coefficient_exact_Anchor=True,soft_parent_and_soft_child_inside=True,
        F_original_global_Hard_ST_sparsity=True,world_over_valid_count_DDP_normalization=True,
        ramp_convention='lambda(u)=min((u-1)/200,1); original completed-update convention',
        formula='Omega_F/3 + 2/3*(in_Dall_F+in_D3_Dall) + 1/2*lambda*(out_Dall_F+out_D3_Dall)',
        routing_example=routing_example(),last50=means(actual[-50:],REG_KEYS),
        matched_last50=means(matched[-50:],tuple(k for k in matched[0] if k!='step')),
        matched_all500=matched,comparison_population='Identical steps and samples, learned masks may differ',
        exact_low_support_caveat='Conditional support is normalized: uniformly tiny nonzero parent support does not guarantee tiny inside loss; relative low-support coordinates have small contributions.')


def classify(q,base,coverage):
    d={k:q[k]-base[k] for k in QUALITY};eps=1e-6
    if (q['Score5']>71.158309 and q['J_long3']>=74.978516 and q['Urban_T2I']>=89.6 and
        q['Short4']>=65.30 and coverage):return 'COUPLED_NESTED_STRONG_POSITIVE'
    if coverage and d['Score5']>=.05-eps and d['J_long3']>=.05-eps and d['Urban_T2I']>=-.1-eps:
        return 'COUPLED_NESTED_POSITIVE'
    if coverage and all(abs(d[k])<=tol+eps for k,tol in
        [('Score5',.05),('J_long3',.05),('Urban_T2I',.1),('Short4',.15)]):
        return 'REGULARIZER_SIMPLIFICATION_POSITIVE'
    if all(d[k]<-eps for k in ('Score5','J_long3','Urban_T2I')):return 'COUPLED_NESTED_NEGATIVE'
    return 'COUPLED_NESTED_TRADEOFF'


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    result=json.loads((EXP/'RESULTS.json').read_text())
    diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    gradient=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    actual=rows(RUN_ROOT/ARM/'step500/steps.jsonl');reference=rows(BASE_RUN/'step500/steps.jsonl')
    assert search.matched_stream(actual,reference,ARM)['records']==512000
    audit=regularizer_audit(actual,reference)
    keep=masks['keep_ratio'];coverage=keep['F']>keep['Dall']>keep['D3']>0.
    audit['coverage_mean_ordered']=coverage
    diag['coupled_regularizer_last50']=audit['last50']
    masks.update(regularizer_mode='coupled_nested',independent_inclusion_applied=False,
        soft_parent_detached_inside=True,inside_outside_last50={k:v for k,v in audit['last50'].items()
            if 'parent' in k or 'mass' in k},mean_coverage_ordered=coverage)
    q=quality(result);references={};deltas={};recalls={};mask_deltas={};grad_deltas={}
    for name,path in [('Anchor',BASE_EXP),('INC0',OFF_EXP),('INC0.5',HALF_EXP)]:
        r=json.loads((path/'RESULTS.json').read_text());value=quality(r)
        references[name]=dict(scores=value,path=str(path/'RESULTS.json'),sha256=sha(path/'RESULTS.json'))
        deltas[name]={k:q[k]-value[k] for k in QUALITY};recalls[name]=recall_delta(result,r)
        old=json.loads((path/'MASK_HIERARCHY_AUDIT.json').read_text())
        old_diag=json.loads((path/'TRAINING_DIAGNOSTICS.json').read_text())
        mask_deltas[name]=dict(last50={k:v-old['last50'][k] for k,v in masks['last50'].items()},
            keep_ratio={k:v-old_diag['last50_views'][k]['keep_ratio'] for k,v in keep.items()})
        g=json.loads((path/'GRADIENT_SPOTCHECK.json').read_text())
        ratio=g['weighted_mean_gradient_norms']['D3']/g['weighted_mean_gradient_norms']['Dall']
        grad_deltas[name]=dict(weighted_D3_Dall_ratio=gradient['weighted_lowest_Dall_ratio']-ratio,
            raw_norms={k:v-g['mean_gradient_norms'][k] for k,v in gradient['mean_gradient_norms'].items()})
    status=classify(q,references['Anchor']['scores'],coverage)
    result.update(classification=status,references=references,quality_delta_vs_references_pp=deltas,
        recall_delta_vs_references_pp=recalls,sole_changed_config_key='regularizer_mode',
        regularizer_audit_passed=True,gradient_delta_vs_references=grad_deltas)
    masks['delta_vs_references']=mask_deltas
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('REGULARIZER_AUDIT',audit)]:dump(EXP/(name+'.json'),value)
    original=(EXP/'REPORT.md').read_text().replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'One coupled arm only; classified after complete evaluation and audit.').replace(
        'absolute sparsity coefficients:[1.0, 2.0, 2.0] (mass5.0)',
        'coupled edge coefficients:[1,2,2]; child global sparsity replaced by conditional inside terms')
    lines=['','## Coupled regularizer comparison','',f'Classification: `{status}`.',
        'Only regularizer_mode changes. F global Hard-ST sparsity remains; child inside terms use soft detached parent and soft child. No extra child global sparsity or independent inclusion.',
        'Outside expanded coefficient is 1/2*lambda per edge, identical to Anchor. Ramp uses original completed updates: update200=.995; update201 onwards=1.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in [(n,v['scores']) for n,v in references.items()]+[(ARM,q)]:
        lines.append('| '+name+' | '+' | '.join(f'{value[k]:.6f}' for k in QUALITY)+' |')
    lines+=['','| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,delta in deltas.items():lines.append('| '+name+' | '+' | '.join(f'{delta[k]:+.6f}' for k in QUALITY)+' |')
    for name in ('Anchor','INC0'):
        lines+=['',f'All native recalls vs {name} (pp):','',
            '| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |','|---|---|---|']
        for dataset,v in recalls[name].items():
            values=[' / '.join(f'{v[d][k]:+.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
            lines.append(f'| {dataset} | {values[0]} | {values[1]} |')
    lines+=['',f'Last50 regularizer terms: `{audit["last50"]}`.',
        f'Matched actual Anchor magnitudes and same-current-mask counterfactual: `{audit["matched_last50"]}`.',
        f'Gradient routing manual tensor audit: `{audit["routing_example"]}`.',
        f'Hierarchy deltas (fractions; multiply violations by100 for pp): `{mask_deltas}`.',
        f'Gradient deltas: `{grad_deltas}`; mean coverage ordered:{coverage}.',
        audit['exact_low_support_caveat'],
        'Conditional/inside statistics are telemetry, not additional losses. Original last50 CE and fixed8-batch gradient protocol remain unchanged.',
        'Stopped500; no full,other beta/budget/K/weights/optimizer or experiments. Binary/raw artifact paths,size,SHA/time ranges in RUNTIME_STATS.json.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.nested_d3_coupled_reg500'
    search.PHASE_PREFIX='formal-nested-d3-coupled-reg500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/nested_d3_coupled_reg500.py',
        'tests/test_nested_d3_coupled_reg500.py','recovery/check_stage500_publish.py'}
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report fresh Nested D3 coupled regularizer local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    if 'REGULARIZER_AUDIT.json' not in runner.REPORT_NAMES:
        runner.REPORT_NAMES+=('REGULARIZER_AUDIT.json',)
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
