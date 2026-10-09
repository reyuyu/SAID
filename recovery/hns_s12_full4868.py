"""Frozen S12 only:2434->3651->4868, evaluate at both boundaries, then stop."""
import hashlib
import json
from pathlib import Path
import subprocess

from recovery import hns_s12_2434_validation as protocol
from recovery.s02_nfs500 import ROOT,dump,sha,now,STEP0,STEP0_SHA

PROJECT=protocol.PROJECT
BRANCH='experiment/hns-s12-full4868-v1'
MOTHER='origin/experiment/hns-s12-2434-validation-v1'
BASE=ROOT/'experiments/nest_clip_v1/hns_s12_2434_validation_v1'
PARENT=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-2434-validation-v1/step2434/training/step002434.pt'
PARENT_SHA='4cde7d4b91af80755215f20687b80856266fa1faed4b2db864a88ae72b030975'
EXP=ROOT/'experiments/nest_clip_v1/hns_s12_full4868_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-full4868-v1'
TARGETS=(3651,4868)
ENTRY='recovery.hns_s12_full4868'
HALF_BRANCH='origin/experiment/nested-d3-hns-half-full-v1'
HALF_FILE='experiments/nest_clip_v1/nested_d3_hns_half_full_v1/RESULTS.json'
DIAG_FILE='experiments/nest_clip_v1/epoch_curve_eval_v1/TRAINING_DIAGNOSTICS_COMPARISON.json'
EXPECTED_SCORES={3651:{'D3_Balanced':73.830182,'HNS_v1':73.774418,'HNS_Half':73.706809},
                 4868:{'D3_Balanced':73.781065,'HNS_v1':73.638084,'HNS_Half':73.660579}}
CODE=protocol.CODE+('recovery/hns_s12_full4868.py','tests/test_hns_s12_full4868.py')


def predecessor(target):
    assert target in TARGETS
    return (PARENT,2434) if target==3651 else (protocol.segment(3651)/'training/step003651.pt',3651)


def phase(target):return protocol.local.IMAGES.parent/f'formal-hns-s12-full4868-v1-{target}'


def archived(branch,path):
    blob=subprocess.check_output(['git','show',branch+':'+path],cwd=ROOT)
    return json.loads(blob),dict(branch=branch,commit=protocol.common.git('rev-parse',branch),
        path=path,sha256=hashlib.sha256(blob).hexdigest())


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0,1,2,3});assert protocol.common.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'No overwrite or implicit retry'
    proof=protocol.identity(PARENT,2434)
    assert proof['sha256']==PARENT_SHA and proof['data_cursor']==dict(next_epoch=2,next_batch=0)
    assert sha(STEP0)==STEP0_SHA
    old=protocol.common.read(BASE/'step2434/RESULTS.json');assert old['completed_steps']==2434
    assert old['evaluation_checkpoint_immutable'] and old['checkpoint']['sha256']==PARENT_SHA
    assert old['strict_export']['checkpoint_sha256']==PARENT_SHA and old['strict_export']['passed']
    assert protocol.common.read(BASE/'step2434/VALIDATION.json')['passed']
    mother=protocol.common.git('rev-parse',MOTHER)
    subprocess.run(['git','merge-base','--is-ancestor',proof['git_head'],mother],cwd=ROOT,check=True)
    for path,digest in proof['sources'].items():
        for commit in (mother,proof['git_head']):
            assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)).hexdigest()==digest
    ready=protocol.common.read(protocol.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    protocol.common.evaluator_proof()
    history,history_proof=archived(protocol.REFERENCE_BRANCH,protocol.REFERENCE_FILE)
    half,half_proof=archived(HALF_BRANCH,HALF_FILE)
    history_diag,diag_proof=archived(protocol.REFERENCE_BRANCH,DIAG_FILE)
    assert history['status']=='COMPLETE' and half['status']=='COMPLETE'
    selected={m:{str(n):history['models'][m][str(n)] for n in TARGETS} for m in ('HNS_v1','D3_Balanced')}
    selected['HNS_Half']={str(n):half['models'][str(n)] for n in TARGETS}
    for n in TARGETS:
        for model,expected in EXPECTED_SCORES[n].items():
            assert abs(selected[model][str(n)]['scores_percent']['Score5']-expected)<.000001
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'config.json',protocol.common.read(BASE/'config.json'))
    dump(EXP/'RESUME_PROVENANCE.json',dict(**proof,mother_commit=mother,evaluated_checkpoint_immutable=True,
        local_only=True,image_root=str(protocol.local.IMAGES),NFS_fallback=False,production_source_changes=[],
        original500_sha256=protocol.common.read(BASE/'RESUME_PROVENANCE.json')['sha256'],
        predecessor_resume_evidence=protocol.common.read(BASE/'step2434/RESUME_GATE.json')))
    dump(EXP/'REFERENCES.json',dict(models=selected,provenance=[history_proof,half_proof],
        matched_nodes_only=True,quoted_historical_Score5_verified=True))
    dump(EXP/'REFERENCE_DIAGNOSTICS.json',dict(provenance=diag_proof,
        models={m:{str(n):history_diag[m][str(n)] for n in TARGETS} for m in ('HNS_v1','D3_Balanced')},
        S12_epoch2_masks=protocol.common.read(BASE/'step2434/MASK_HIERARCHY_AUDIT.json'),
        S12_epoch2_results=old))
    for n in TARGETS:
        (EXP/f'step{n}').mkdir();protocol.segment(n).mkdir()
        dump(protocol.segment(n)/'resume-reference.json',protocol.resume_reference(predecessor(n)[1]))
    paths=protocol.local.path_proof();dump(RUN/'local-path-proof-5000.json',paths)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=paths['passed'],count=paths['count'],
        image_root=str(protocol.local.IMAGES),NFS_fallback=False,raw_path=str(RUN/'local-path-proof-5000.json')))
    dump(EXP/'PLAN.json',dict(order=['resume2434->3651','audit/export/verify/five-eval/report3651',
        'resume full3651->4868','audit/export/verify/five-eval/report4868','STOP'],targets=list(TARGETS),
        horizon=4868,macro=[10,1.2,1],view_weights=[1.35,1.35,.3],sparsity=[1,2,2],beta=[2,2],
        no_SG=True,soft_inclusion=0,K=3,ramp200_unchanged=True,local_only=True,full_restore=True,
        initial_cursor=dict(next_epoch=2,next_batch=0),no_replay_prior2434_images_or_updates=True,
        evaluation_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        scientific_interpretation_only_after4868=True,result_based_weight_switch=False,other_arms=False,
        cache='Disposable /root Docker overlay; retain persistent NFS originals'))
    protocol.state('PREPARED')


def combined(completed):
    refs=protocol.common.read(EXP/'REFERENCES.json')['models']
    results={str(n):protocol.common.read(EXP/f'step{n}/RESULTS.json') for n in completed}
    final=completed==list(TARGETS)
    scientific={}
    if final:
        evidence=protocol.common.read(EXP/'REFERENCE_DIAGNOSTICS.json')
        previous=evidence['S12_epoch2_results'];prior_masks=evidence['S12_epoch2_masks']
        for n in TARGETS:
            node=results[str(n)];masks=protocol.common.read(EXP/f'step{n}/MASK_HIERARCHY_AUDIT.json')
            gradient=protocol.common.read(EXP/f'step{n}/GRADIENT_AUDIT.json')
            comparisons=node['comparisons'];native={}
            for model in ('HNS_v1','D3_Balanced'):
                base=evidence['models'][model][str(n)]
                native[model]=dict(keep_delta={v:masks['keep_ratios'][v]-base['views'][v]['keep'] for v in ('F','Dall','D3')},
                    reference_last50=base)
            scientific[str(n)]=dict(
                Q1_Score5_vs_HNS_pp=comparisons['HNS_v1']['quality_delta_pp']['Score5'],
                Q2_Score5_vs_same_node_Balanced_pp=comparisons['D3_Balanced']['quality_delta_pp']['Score5'],
                Q3_Short4_delta_pp={m:v['quality_delta_pp']['Short4'] for m,v in comparisons.items()},
                Q4_Long_DCI_T2I_delta_pp={m:v['recall_delta_pp']['Long-DCI']['T2I']['R@1'] for m,v in comparisons.items()},
                Long_DCI_T2I_change_vs_previous_S12_node_pp=100*(node['metrics']['Long-DCI']['T2I']['R@1']-previous['metrics']['Long-DCI']['T2I']['R@1']),
                Q5_mask=dict(keep=masks['keep_ratios'],keep_change_vs_previous_S12_node={v:masks['keep_ratios'][v]-prior_masks['keep_ratios'][v] for v in ('F','Dall','D3')},
                    same_node_baselines=native,lower_keep_alone_does_not_prove_better_evidence=True,
                    excessive_shrink_requires_retrieval_and_mask_diagnostics_together=True),
                Q6_structure=dict(mask_telemetry=masks,group_component_gradients=gradient['group_diagnostics'],
                    causal_HNS_benefit_not_identified_without_same_node_INC0_control=True))
            previous=node;prior_masks=masks
        scientific['S12_E3_exceeds_Balanced_E3_best']=results['3651']['scores_percent']['Score5']>refs['D3_Balanced']['3651']['scores_percent']['Score5']
        scientific['S12_exceeds_HNS_at_both_late_nodes']=all(results[str(n)]['comparisons']['HNS_v1']['quality_delta_pp']['Score5']>0 for n in TARGETS)
        dump(EXP/'SCIENTIFIC_DIAGNOSTICS.json',scientific)
    dump(EXP/'RESULTS.json',dict(status='COMPLETED' if final else 'PARTIAL',nodes=results,references=refs,
        frozen_macro=[10,1.2,1],stop_updates=4868,automatic_continuation=False,scientific_diagnostics=scientific,
        interim_results_did_not_change_E4_configuration=True))
    lines=['# HNS-S12 static weights:2434 ->3651 ->4868','',
        'Fixed macros10/1.2/1. No phase switching, additional arm, or training from bare weights. Initial checkpoint SHA: `'+PARENT_SHA+'`.','',
        '| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |','|---:|---|---:|---:|---:|---:|---|']
    for n in completed:
        for model,value in [(m,refs[m][str(n)]) for m in refs]+[('HNS-S12',results[str(n)])]:
            q=protocol.common.quality(value)
            lines.append(f'| {n} | {model} | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} |')
        lines+=['',f'## Step{n}: matched-node deltas','']
        for model,c in results[str(n)]['comparisons'].items():
            lines.append(model+': '+json.dumps(c['quality_delta_pp'])+' pp. Long-DCI directional R1 delta: '+json.dumps({dr:c['recall_delta_pp']['Long-DCI'][dr]['R@1'] for dr in ('I2T','T2I')})+' pp.')
            gain=c['quality_delta_pp']['Score5']
            if 0<gain<.05:lines.append('Positive Score5 difference below0.05pp: weak single-seed signal.')
        lines+=['','| Dataset | I2T R1 / R5 / R10 (%) | T2I R1 / R5 / R10 (%) |','|---|---|---|']
        for ds,directions in results[str(n)]['metrics'].items():
            values=[' / '.join(f'{100*directions[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append(f'| {ds} | {values[0]} | {values[1]} |')
        mask=protocol.common.read(EXP/f'step{n}/MASK_HIERARCHY_AUDIT.json')
        lines+=['','Last50 mask telemetry: '+json.dumps(mask)+'.','']
    if final:
        lines+=['## Final scientific diagnosis','']
        lines.append('S12 exceeds HNS-v1 at both late nodes: '+str(scientific['S12_exceeds_HNS_at_both_late_nodes'])+'.')
        lines.append('S12 E3 exceeds Balanced E3 best: '+str(scientific['S12_E3_exceeds_Balanced_E3_best'])+'.')
        for n in TARGETS:
            q=scientific[str(n)]
            lines.append(f'Step{n}: Short4 deltas '+json.dumps(q['Q3_Short4_delta_pp'])+'; Long-DCI T2I deltas '+json.dumps(q['Q4_Long_DCI_T2I_delta_pp'])+'; own previous-node T2I change '+str(q['Long_DCI_T2I_change_vs_previous_S12_node_pp'])+'pp.')
            lines.append('Keep changes vs previous S12 node: '+json.dumps(q['Q5_mask']['keep_change_vs_previous_S12_node'])+'. Same-node baseline masks and hierarchy/sparsity gradient norms/cosines are in SCIENTIFIC_DIAGNOSTICS.json. Lower keep is not evidence-selection quality by itself.')
        lines.append('Hierarchy loss, violations, IoU and gradients describe structural pressure; causal HNS benefit is not established by this single fixed-S12 trajectory alone.')
    else:lines+=['E3 complete. E4 configuration remains fixed and queued; final scientific interpretation waits for4868.']
    lines+=['All30 recalls and their matched-node deltas are in node RESULTS.json. Model/optimizer/sampler/CPU-CUDA-Python-NumPy RNG/loader restoration and next-five text/token/indices/LR evidence are in node RESUME_GATE.json.',
        'Training/evaluator sources remain byte-identical to the evaluated2434 run; only control and read-only audit node support changed. Native four-GPU evaluator math/batch64/protocol unchanged.',
        'Local-only disposable /root image cache; NFS originals retained. Full checkpoints, bare weights, raw logs stay persistent server-local and are never uploaded.',
        'Stop4868 after final evaluation/report/sync; no new arms or continuation.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def configure():
    protocol.BRANCH=BRANCH;protocol.MOTHER=MOTHER;protocol.PARENT=PARENT;protocol.PARENT_SHA=PARENT_SHA
    protocol.EXP=EXP;protocol.RUN=RUN;protocol.ENTRY=ENTRY;protocol.TARGETS=TARGETS;protocol.CODE=CODE
    protocol.predecessor=predecessor;protocol.phase=phase;protocol.prepare=prepare;protocol.combined=combined


def main():
    configure();protocol.main()


if __name__=='__main__':main()
