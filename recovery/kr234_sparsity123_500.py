"""One fresh KR234 arm: literal sparsity1/2/3, full frozen KR234 stream proof."""
import json
from pathlib import Path

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha

EXP=ROOT/'experiments/nest_clip_v1/kr234_sparsity123_500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/kr234-sparsity123-500-20261007'
BRANCH='experiment/kr234-sparsity123-500-v1'
ARM='KR234-S123'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_local_search500_v1/KR234'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-local-search500-20261007/KR234'
FIXED_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
ARMS={ARM:dict(axis='sparsity',weights=[1.35,1.35,.30],r=3.,
    mode='nested_detail_kr234',sparsity_weights=[1.,2.,3.],strict_lowest_reference=True,
    experiment_dir=str(EXP))}


def classify(q,base,mask,base_mask):
    """Predeclared operational meanings for 'marked' decline/hierarchy."""
    d={k:q[k]-base[k] for k in q};eps=1e-6
    keep=mask['keep_ratio'];old=base_mask['keep_ratio']
    sparse=keep['Dk']<old['Dk']-eps
    hierarchy=(sparse and keep['F']>=keep['Dall']>=keep['Dk'] and
        keep['Dall']-keep['Dk']>=old['Dall']-old['Dk']-eps and
        mask['last50']['Dk_Dall_hard_violation']<=base_mask['last50']['Dk_Dall_hard_violation']+.001)
    if d['Score5']>eps and d['J_long3']>=-eps and d['Urban_T2I']>=-.2-eps and hierarchy:
        return 'KR234_S123_STRONG_POSITIVE'
    long_gain=max(d['J_long3'],d['J_long'])>eps
    if sparse and long_gain and (d['Urban_T2I']<-.2-eps or d['Short4']<-.15-eps):
        return 'SPARSITY_TRADEOFF'
    if old['Dk']-keep['Dk']>=.01 and d['Score5']<-eps and d['J_long3']<-eps and d['Urban_T2I']<-eps:
        return 'SPARSITY_OVERREGULARIZED'
    return 'NO_IMPROVEMENT'


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    result=json.loads((EXP/'RESULTS.json').read_text())
    baseline=json.loads((BASE_EXP/'RESULTS.json').read_text())
    fixed=json.loads((FIXED_EXP/'RESULTS.json').read_text())
    q,base,anchor=map(quality,(result,baseline,fixed))
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    old_masks=json.loads((BASE_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    old_diag=json.loads((BASE_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    grad=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    old_grad=json.loads((BASE_EXP/'GRADIENT_SPOTCHECK.json').read_text())
    status=classify(q,base,masks,old_masks)
    result.update(classification=status,quality_delta_vs_anchor_pp={k:q[k]-anchor[k] for k in q},
        quality_delta_vs_KR234_pp={k:q[k]-base[k] for k in q},
        recall_delta_vs_anchor_pp=recall_delta(result,fixed),
        recall_delta_vs_KR234_pp=recall_delta(result,baseline),
        trajectory_reference='KR234',sole_changed_config_key='view_sparsity_weights',
        literal_sparsity_coefficients=[1.,2.,3.],sparsity_mass=6,coefficients_normalized=False,
        references={name:dict(result_path=str(path/'RESULTS.json'),result_sha256=sha(path/'RESULTS.json'),
            scores=quality(payload)) for name,path,payload in [('Anchor',FIXED_EXP,fixed),('KR234',BASE_EXP,baseline)]},
        gradient_delta_vs_KR234=dict(weighted_Dk_Dall_ratio=grad['weighted_lowest_Dall_ratio']-old_grad['weighted_lowest_Dall_ratio'],
            raw_norms={v:g-old_grad['mean_gradient_norms'][v] for v,g in grad['mean_gradient_norms'].items()}),
        last50_alignment_share_delta_vs_KR234_pp={v:d['alignment_share_percent']-old_diag['last50_views'][v]['alignment_share_percent']
            for v,d in diag['last50_views'].items()})
    # Existing review used KR234 as the trajectory reference. Keep that mask
    # evidence explicit instead of calling it the fixed-K Anchor.
    masks['delta_vs_KR234']=masks.pop('delta_vs_anchor')
    masks['keep_ratio_delta_vs_KR234']=masks.pop('keep_ratio_delta_vs_anchor')
    dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks);dump(EXP/'RESULTS.json',result)
    original=(EXP/'REPORT.md').read_text()
    original=original.replace('Delta vs Anchor(pp)','Delta vs KR234(pp)').replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'This one arm is classified after complete native evaluation/review.')
    lines=['','## Fixed-K Anchor and KR234 comparison','',f'Classification: `{status}`.',
        'Only literal sparsity changes1/2/2 ->1/2/3; mass5 ->6; no normalization or extra loss scale.',
        'Both the first5 gate and complete512000-record proof match KR234 IDs,F,Dall,K,Dk indices,text/tokens and LR exactly.',
        '', '| Model | Sparsity | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    keys=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
    for name,coef,value in [('Anchor fixedK3','1/2/2',anchor),('KR234','1/2/2',base),(ARM,'1/2/3',q)]:
        lines.append('| '+name+' | '+coef+' | '+' | '.join(f'{value[k]:.6f}' for k in keys)+' |')
    lines+=['','| Reference | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,old in [('Anchor',anchor),('KR234',base)]:
        lines.append('| '+name+' | '+' | '.join(f'{q[k]-old[k]:+.6f}' for k in keys)+' |')
    lines+=['',f'Dk keep ratio:{old_masks["keep_ratio"]["Dk"]:.6f} -> {masks["keep_ratio"]["Dk"]:.6f}.',
        f'Weighted Dk/Dall gradient ratio:{old_grad["weighted_lowest_Dall_ratio"]:.6f} -> {grad["weighted_lowest_Dall_ratio"]:.6f}.',
        f'Mask changes vs KR234:`{masks["delta_vs_KR234"]}`.',
        'Operational thresholds were frozen in SEARCH_PLAN.json before launch. Strong hierarchy means a lower Dk keep, ordered F/Dall/Dk keeps, no smaller parent-child keep gap, and Dk outside-Dall violation increase<=.1pp. Soft inclusion remains unchanged; no hard projection is imposed.',
        'Complete five-set/directional/recall deltas vs both references are in RESULTS.json. Raw/bare/full weights stay local.',
        'Stopped at exactly500; no full,other sparsity/K/alignment/inclusion experiment or combination launched.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):
        (EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.kr234_sparsity123_500'
    search.PHASE_PREFIX='formal-kr234-sparsity123-500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/kr234_sparsity123_500.py',
        'tests/test_kr234_sparsity123_500.py','recovery/check_stage500_publish.py'}
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report isolated KR234 literal sparsity1/2/3 local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
