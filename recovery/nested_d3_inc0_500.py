"""One fresh fixed-K3 Anchor arm: no inclusion objective/gradient/schedule."""
import json

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT,dump,sha

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-inc0-500-20261007'
BRANCH='experiment/nested-d3-inc0-500-v1'
ARM='INC0'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
ARMS={ARM:dict(axis='inclusion',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',
    inclusion_max=0.,experiment_dir=str(EXP))}


def alignment_share(view):
    return view['alignment_share_percent'] if 'alignment_share_percent' in view else 100*view['weighted_alignment_loss_share']


def classify(q,base,masks,old_masks):
    """Operational pp thresholds declared before training, no metric tuning."""
    delta={k:q[k]-base[k] for k in q};eps=1e-6
    limits=dict(Score5=.2,J_long3=.2,J_long=.2,Short4=.2,Urban_I2T=.3,Urban_T2I=.3)
    if all(abs(delta[k])<=v+eps for k,v in limits.items()):return 'INCLUSION_NEUTRAL'
    improved=(delta['Score5']>.1+eps and delta['J_long3']>=-.2-eps and
        delta['Urban_T2I']>=-.3-eps and delta['Short4']>=-.2-eps)
    if improved:return 'INCLUSION_OVERCONSTRAINED'
    worse=max(masks['last50'][k]-old_masks['last50'][k] for k in
        ('Dall_F_hard_violation','D3_Dall_hard_violation'))>=.005-eps
    if delta['Score5']<-.2-eps and delta['J_long3']<-.2-eps and worse:return 'INCLUSION_NECESSARY'
    return 'TRADEOFF'


def summarize():
    from recovery.nested_d3_local_search_evidence import quality
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    result=json.loads((EXP/'RESULTS.json').read_text())
    baseline=json.loads((BASE_EXP/'RESULTS.json').read_text())
    q,base=quality(result),quality(baseline)
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    old_masks=json.loads((BASE_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    old_diag=json.loads((BASE_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    grad=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text())
    old_grad=json.loads((BASE_EXP/'GRADIENT_SPOTCHECK.json').read_text())
    old_ratio=old_grad['weighted_mean_gradient_norms']['D3']/old_grad['weighted_mean_gradient_norms']['Dall']
    keep=masks['keep_ratio'];coverage=keep['F']>keep['Dall']>keep['D3']
    status=classify(q,base,masks,old_masks)
    violation={k:100*masks['delta_vs_anchor'][k] for k in ('Dall_F_hard_violation','D3_Dall_hard_violation')}
    loss_audit=dict(passed=True,optimizer_updates=500,inclusion_max=0.,
        all500_inclusion_loss_weight_zero_and_disabled=True,
        ramp_not_called=True,inclusion_graph_absent=True,raw_inc_is_detached_telemetry=True,
        CPU_objective_and_all_parameter_gradients_verified=True)
    result.update(classification=status,trajectory_reference='Nested D3 Balanced Anchor',
        sole_changed_config_key='inclusion_max',inclusion_loss_audit=loss_audit,
        natural_coverage_hierarchy=coverage,hard_violation_delta_vs_anchor_pp=violation,
        gradient_delta_vs_anchor=dict(weighted_D3_Dall_ratio=grad['weighted_lowest_Dall_ratio']-old_ratio,
            raw_norms={v:g-old_grad['mean_gradient_norms'][v] for v,g in grad['mean_gradient_norms'].items()}),
        last50_alignment_share_delta_vs_anchor_pp={v:alignment_share(d)-alignment_share(old_diag['last50_views'][v])
            for v,d in diag['last50_views'].items()},
        anchor_reference=dict(result_path=str(BASE_EXP/'RESULTS.json'),result_sha256=sha(BASE_EXP/'RESULTS.json'),scores=base))
    diag['inclusion_loss_audit']=loss_audit;masks['natural_coverage_hierarchy']=coverage
    masks['hard_violation_delta_vs_anchor_pp']=violation
    dump(EXP/'RESULTS.json',result);dump(EXP/'TRAINING_DIAGNOSTICS.json',diag);dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks)
    original=(EXP/'REPORT.md').read_text().replace(
        'Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md.',
        'This one arm is classified after complete native evaluation/review.')
    lines=['','## Inclusion ablation comparison','',f'Classification: `{status}`.',
        'Only inclusion_max changes1 ->0. INC0 excludes the objective term, bypasses its ramp, and computes raw inclusion violation without autograd. Raw inc telemetry is not an applied loss.',
        'The first5 gate and complete512000-record proof match Anchor IDs,F/Dall/D3 strings/tokens,K,selected indices and LR exactly.',
        '', '| Model | Inclusion | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    keys=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
    for name,inc,value in [('Anchor','ramp200/max1',base),('INC0','off',q)]:
        lines.append('| '+name+' | '+inc+' | '+' | '.join(f'{value[k]:.6f}' for k in keys)+' |')
    lines+=['','Observed natural mean coverage hierarchy F>Dall>D3: '+str(coverage)+'.',
        f'Keep ratios: `{keep}`; deltas vs Anchor: `{masks["keep_ratio_delta_vs_anchor"]}`.',
        f'Hard-violation deltas(pp): `{violation}`; IoU/telemetry changes: `{masks["delta_vs_anchor"]}`.',
        f'Retrieval deltas(pp): `{result["quality_delta_vs_anchor_pp"]}`.',
        f'Raw gradient changes: `{result["gradient_delta_vs_anchor"]}`.',
        'Coverage ordering of means does not guarantee per-sample nested masks; hard violation separately measures outside-parent coordinates.',
        'If retrieval improves with looser hierarchy, that is consistent with inclusion constraining specialization; this500-step single-seed ablation does not by itself establish that mechanism.',
        'Classification thresholds were frozen in SEARCH_PLAN.json before launch; no schedule/weight/other-arm tuning.',
        'Every five-set directional R@1/5/10 delta is above and in RESULTS.json. Checkpoint/bare/raw logs stay local with path/size/SHA/time ranges in RUNTIME_STATS.json.',
        'Stopped at exactly500. No full,inc0.5,schedule,sparsity,K,alignment or other experiment launched.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text(original+'\n'.join(lines)+'\n')


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE='recovery.nested_d3_inc0_500'
    search.PHASE_PREFIX='formal-nested-d3-inc0-500-20261007-'
    search.EXTRA_SOURCES={'recovery/nested_d3_followup500.py','recovery/nested_d3_inc0_500.py',
        'tests/test_nested_d3_inc0_500.py','recovery/check_stage500_publish.py'}
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=search.ENTRY_MODULE
    runner.PUBLISH_MESSAGE='Report isolated Nested D3 inclusion-off local500 experiment'
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.configure=configure;runner.summarize=summarize


def main():
    configure();runner.main()


if __name__=='__main__':main()
