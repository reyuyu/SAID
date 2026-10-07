"""Fresh fixed-K3 Anchor arm: original inclusion chain/ramp with max0.5."""
import json

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha,rows

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc05_500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-inc05-500-20261007'
BRANCH='experiment/nested-d3-inc05-500-v1'
ARM='INC0.5'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
OFF_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
ARMS={ARM:dict(axis='inclusion',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',
    inclusion_max=.5,experiment_dir=str(EXP))}
VIOLATIONS=('Dall_F_hard_violation','D3_Dall_hard_violation')
IOUS=('F_Dall_mask_iou','Dall_D3_mask_iou')
QUALITY=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')


def hierarchy_position(masks,anchor,off):
    value=masks['last50'];base=anchor['last50'];zero=off['last50']
    eps=1e-8
    violation_between={k:base[k]-eps<=value[k]<zero[k]-eps for k in VIOLATIONS}
    iou_between={k:zero[k]+eps<value[k]<=base[k]+eps for k in IOUS}
    stronger=all(zero[k]-value[k]>=.0025-eps for k in VIOLATIONS) and all(value[k]-zero[k]>=.005-eps for k in IOUS)
    return dict(violation_between_INC0_and_Anchor=violation_between,
        IoU_between_INC0_and_Anchor=iou_between,
        all_four_between=all(violation_between.values()) and all(iou_between.values()),
        clearly_stronger_than_INC0=stronger,
        mean_coverage_ordered=masks['keep_ratio']['F']>masks['keep_ratio']['Dall']>masks['keep_ratio']['D3'])


def dominates(a,b):
    eps=1e-6
    return all(a[k]>=b[k]-eps for k in QUALITY) and any(a[k]>b[k]+eps for k in QUALITY)


def classify(q,base,off,position):
    eps=1e-6
    if (q['Score5']>=71.10-eps and q['J_long3']>=74.95-eps and q['Urban_T2I']>=89.6-eps and
        q['Short4']>=65.30-eps and position['clearly_stronger_than_INC0']):return 'SOFT_INCLUSION_STRONG_POSITIVE'
    delta={k:q[k]-base[k] for k in QUALITY}
    if (delta['Score5']>eps and delta['J_long3']>=-.05-eps and delta['Urban_T2I']>=-.1-eps and
        all(position['violation_between_INC0_and_Anchor'].values())):return 'SOFT_INCLUSION_POSITIVE'
    if dominates(base,q) or dominates(off,q):return 'NO_IMPROVEMENT'
    if any(delta[k]>eps for k in QUALITY) and any(delta[k]<-eps for k in QUALITY):return 'SOFT_INCLUSION_TRADEOFF'
    return 'NO_IMPROVEMENT'


def alignment_share(view):
    return view['alignment_share_percent'] if 'alignment_share_percent' in view else 100*view['weighted_alignment_loss_share']


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    result=json.loads((EXP/'RESULTS.json').read_text())
    baseline=json.loads((BASE_EXP/'RESULTS.json').read_text());off=json.loads((OFF_EXP/'RESULTS.json').read_text())
    q,base,zero=map(quality,(result,baseline,off))
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    old_masks=json.loads((BASE_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    off_masks=json.loads((OFF_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    grad=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    position=hierarchy_position(masks,old_masks,off_masks)
    actual=rows(RUN_ROOT/ARM/'step500/steps.jsonl');reference=rows(BASE_RUN/'step500/steps.jsonl')
    proof=search.matched_stream(actual,reference,ARM);assert proof['records']==512000
    assert all(r['inc_weight']==.5*min(1.,(r['step']-1)/200.) for r in actual)
    assert masks['inclusion_max']==.5 and masks['ramp200'] and masks['detached_child']
    assert masks['inclusion_edges']==['Dall->F','lowest->Dall']
    loss_audit=dict(passed=True,optimizer_updates=500,inclusion_max=.5,all500_weights_exactly_half_Anchor=True,
        formula_and_detached_child_unchanged=True,ramp_completed_updates=200,
        weight_by_update={str(u):actual[u-1]['inc_weight'] for u in (1,5,100,200,201,500)},
        last50_raw_inclusion=masks['last50']['inc'],last50_applied_inclusion=.5*masks['last50']['inc'],
        optimizer_definition_groups_counters_and_scheduler_exact=True,
        optimizer_moments_and_model_values_cross_run_bitwise_required=False)
    status=classify(q,base,zero,position)
    gradient_deltas={};share_deltas={}
    for name,path in [('Anchor',BASE_EXP),('INC0',OFF_EXP)]:
        g=json.loads((path/'GRADIENT_SPOTCHECK.json').read_text());d=json.loads((path/'TRAINING_DIAGNOSTICS.json').read_text())
        ratio=g['weighted_mean_gradient_norms']['D3']/g['weighted_mean_gradient_norms']['Dall']
        gradient_deltas[name]=dict(weighted_D3_Dall_ratio=grad['weighted_lowest_Dall_ratio']-ratio,
            raw_norms={v:n-g['mean_gradient_norms'][v] for v,n in grad['mean_gradient_norms'].items()})
        share_deltas[name]={v:alignment_share(s)-alignment_share(d['last50_views'][v]) for v,s in diag['last50_views'].items()}
    result.update(classification=status,quality_delta_vs_INC0_pp={k:q[k]-zero[k] for k in QUALITY},
        recall_delta_vs_INC0_pp=recall_delta(result,off),trajectory_reference='Nested D3 Balanced Anchor',
        sole_changed_config_key='inclusion_max',inclusion_loss_audit=loss_audit,hierarchy_position=position,
        gradient_delta_vs_references=gradient_deltas,last50_alignment_share_delta_vs_references_pp=share_deltas,
        references={name:dict(result_path=str(path/'RESULTS.json'),result_sha256=sha(path/'RESULTS.json'),scores=value)
            for name,path,value in [('Anchor',BASE_EXP,base),('INC0',OFF_EXP,zero)]})
    masks.update(hierarchy_position=position,delta_vs_INC0={k:v-off_masks['last50'][k] for k,v in masks['last50'].items()},
        keep_ratio_delta_vs_INC0={k:v-off_masks['keep_ratio'][k] for k,v in masks['keep_ratio'].items()},
        hard_violation_delta_pp={name:{k:100*(masks['last50'][k]-old['last50'][k]) for k in VIOLATIONS}
            for name,old in [('Anchor',old_masks),('INC0',off_masks)]})
    diag['inclusion_loss_audit']=loss_audit
    for name,value in [('RESULTS',result),('MASK_HIERARCHY_AUDIT',masks),('TRAINING_DIAGNOSTICS',diag)]:dump(EXP/(name+'.json'),value)
    original=(EXP/'REPORT.md').read_text().replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'This one arm is classified after complete native evaluation/review.')
    lines=['','## Inclusion max0/0.5/1 comparison','',f'Classification: `{status}`.',
        'Only inclusion_max1 ->0.5. Same detached-child two-edge chain, original200-completed-update linear ramp and hard masks. No model/data/sampling/objective source edit for INC0.5.',
        'At update u, weight=0.5*min((u-1)/200,1): update200=.4975,update201 onwards=.5, exactly half the original Anchor convention.',
        'Every512000 IDs,F/Dall/D3 strings/tokens,K,selected indices and LR match Anchor. Optimizer definition/group order,counters and scheduler are frozen; learned parameter/moment values may differ.',
        '', '| Model | Inclusion max | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,maximum,value in [('INC0',0,zero),('INC0.5',.5,q),('Anchor',1,base)]:
        lines.append('| '+name+' | '+str(maximum)+' | '+' | '.join(f'{value[k]:.6f}' for k in QUALITY)+' |')
    lines+=['','| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in [('Anchor',base),('INC0',zero)]:lines.append('| '+name+' | '+' | '.join(f'{q[k]-value[k]:+.6f}' for k in QUALITY)+' |')
    lines+=['','| Dataset | Delta I2T R@1/5/10 vs INC0(pp) | Delta T2I R@1/5/10 vs INC0(pp) |','|---|---|---|']
    for name,v in result['recall_delta_vs_INC0_pp'].items():
        recalls=[' / '.join(f'{v[d][k]:+.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I')]
        lines.append(f'| {name} | {recalls[0]} | {recalls[1]} |')
    lines+=['','| Hierarchy | INC0 | INC0.5 | Anchor |','|---|---:|---:|---:|']
    for k in VIOLATIONS+IOUS:
        lines.append('| '+k+' | '+' | '.join(f'{m["last50"][k]:.6f}' for m in (off_masks,masks,old_masks))+' |')
    lines+=['',f'Hierarchy position: `{position}`; hard violation deltas(pp): `{masks["hard_violation_delta_pp"]}`.',
        f'Raw/weighted gradient changes vs references: `{gradient_deltas}`.',
        f'Applied/raw inclusion audit: `{loss_audit}`.',
        'Operational thresholds frozen in SEARCH_PLAN.json before launch. Mean coverage ordering is distinct from sample-level coordinate inclusion.',
        'Checkpoint,bare,images and raw large logs stay local; paths/sizes/SHA/time windows in RUNTIME_STATS.json.',
        'Stopped at exactly500; no full,0.25/0.75,schedule,sparsity,K,alignment or other experiment.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.nested_d3_inc05_500'
    search.PHASE_PREFIX='formal-nested-d3-inc05-500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/nested_d3_inc05_500.py',
        'tests/test_nested_d3_inc05_500.py','recovery/check_stage500_publish.py'}
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report isolated Nested D3 inclusion max0.5 local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
