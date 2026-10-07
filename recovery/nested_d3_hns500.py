"""One detached INC0-derived HNS smoke5 -> fresh formal500 -> audit/eval/publish."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery import s02_local500 as local
from recovery.hns_preflight import BASE_EXP,BASE_RUN,EXP,BASE_SHA
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,dump,sha,rows,now

RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008'
SMOKE=RUN_ROOT.parent/(RUN_ROOT.name+'.smoke5')
SMOKE_PHASE=search.LOCAL/(RUN_ROOT.name+'.smoke5-phases')
BRANCH='experiment/nested-d3-hard-nested-sparsity500-v1'
ENTRY='recovery.nested_d3_hns500'
ARMS={'HNS':dict(axis='hns',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_d3',experiment_dir=str(EXP))}
EXTRA={'recovery/nested_d3_hns500.py','recovery/hns_preflight.py','recovery/hns_ddp_correctness.py',
    'recovery/hns_gradient_audit.py','recovery/nested_d3_followup500.py',
    'tests/test_nested_d3_hns500.py','configs/nested_d3_hns500.json','recovery/check_stage500_publish.py'}
STATIC=('BASELINE_PROVENANCE.json','MATCHED_PREFLIGHT.json','DDP_CORRECTNESS.json','CORRECTNESS.md')
REPORTS=runner.REPORT_NAMES+('GRADIENT_AUDIT.json','INC0_HIERARCHY_REFERENCE.json','SMOKE_EVIDENCE.json',
    'DECISION.json','HNS_FORMAL_ACCEPTANCE.json','FORMAL_PROVENANCE.json','COMMANDS.json')
OriginalSupervisor=search.Supervisor


def torchrun(module,*args):
    return [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
        '--max-restarts=0','-m',module,*map(str,args)]


def smoke_worker():
    from train import train_nested_semantic_mask as trainer
    search.activate('HNS');local.RUN,local.PHASE=SMOKE,SMOKE_PHASE
    trainer.sampling_diagnostics=search.observe_selection
    local.worker()


def smoke_proof():
    import torch
    path=SMOKE/'step500';accept=json.loads((path/'acceptance.json').read_text())
    assert accept['passed'] and all(r['completed_updates']==r['updates_this_run']==5 and
        r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
    stream=search.matched_stream(rows(path/'steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],'HNS')
    assert stream['records']==5120
    checkpoint=path/'step000005.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['config'];search.frozen_config(cfg,'HNS')
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==STEP0_SHA
    assert cfg['run_type']=='smoke' and cfg['max_updates']==5 and payload['completed_steps']==5
    assert payload['scheduler_horizon']==4868 and payload['next_epoch']==0 and payload['next_batch']==5
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={5}
    reference=torch.load(BASE_RUN/'step500/step000005.pt',map_location='cpu',weights_only=False)
    assert payload['optimizer']['param_groups']==reference['optimizer']['param_groups']
    for key in ('component_initialization','data','parameter_counts','optimizer_groups'):
        if key in reference['config']:assert cfg[key]==reference['config'][key],key
    assert len(payload['rng_per_rank'])==4
    assert all(torch.isfinite(v).all() for state in (payload['model'],payload['adapter']) for v in state.values())
    assert all(torch.isfinite(v).all() for s in payload['optimizer']['state'].values() for v in s.values())
    value=dict(passed=True,fresh_common0=True,resume=None,formal_must_restart_common0=True,
        records=5120,stream_proof=stream,acceptance=accept,optimizer_groups_exact_INC0=True,
        initialization_and_parameter_counts_exact=True,optimizer_steps=[5],horizon=4868,
        checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,uploaded=False),
        local_runtime=str(SMOKE),local_phase_logs=str(SMOKE_PHASE),reviewed_utc=now())
    dump(EXP/'SMOKE_EVIDENCE.json',value);return value


class HnsSupervisor(OriginalSupervisor):
    def run(self):
        assert json.loads((EXP/'MATCHED_PREFLIGHT.json').read_text())['passed']
        assert json.loads((EXP/'DDP_CORRECTNESS.json').read_text())['passed']
        formal_run,formal_phase=local.RUN,local.PHASE
        assert not SMOKE.exists() and not SMOKE_PHASE.exists()
        SMOKE.mkdir();SMOKE_PHASE.mkdir()
        local.RUN,local.PHASE=SMOKE,SMOKE_PHASE
        dump(SMOKE/'launch-provenance.json',dict(common0_SHA256=STEP0_SHA,resume=None,stop_updates=5,
            source_sha256={p:sha(ROOT/p) for p in search.source_paths()},source_base_commit=BASE_SHA))
        try:
            self.execute('smoke5',torchrun(ENTRY,'--smoke-worker','--config',search.CONFIG,'--init-state',STEP0,
                '--index-dir',local.INDEX,'--image-root',local.IMAGES,'--output-dir',SMOKE/'step500',
                '--run-type','smoke','--max-updates',5),training=True)
            smoke_proof()
            self.execute('INC0-hierarchy-reference',torchrun('recovery.hns_gradient_audit','--baseline-hierarchy'))
        finally:local.RUN,local.PHASE=formal_run,formal_phase
        assert json.loads((EXP/'SMOKE_EVIDENCE.json').read_text())['passed']
        runner.state('FORMAL500_RUNNING','HNS',smoke_passed=True,fresh_common0=True)
        super().run()
        self.execute('HNS-gradient-audit500',torchrun('recovery.hns_gradient_audit'))
        dump(search.RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
            commands=self.commands,started_utc=self.started,ended_utc=now()))


def quality_raw(result):
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import scores
    value=scores(result['metrics'])
    return dict(Score5=value['Score5_R1'],J_long3=value['J_long3'],J_long=value['J_long'],
        Short4=statistics.fmean(result['metrics'][ds][dr]['R@1'] for ds in ('COCO','Flickr30k-test1k') for dr in ('I2T','T2I')),
        Urban_I2T=result['metrics']['Urban-1k']['I2T']['R@1'],
        Urban_T2I=result['metrics']['Urban-1k']['T2I']['R@1'])


def assess(q,baseline,telemetry,old,matched):
    """Evidence rules fixed before launch; conservative thresholds in raw fractions."""
    delta={k:q[k]-baseline[k] for k in q}
    edges=('HNS_DF_hard_violation_ratio','HNS_3D_hard_violation_ratio')
    improvement=all(telemetry[k]<=.75*old[k] for k in edges)
    common_inflation=all(matched['HNS_'+v+'_keep']>=.05 for v in ('F','Dall','D3'))
    equality_rise=any(matched[k]>=.15 for k in ('HNS_DF_exact_equality_ratio','HNS_3D_exact_equality_ratio','HNS_triple_exact_equality_ratio'))
    loss_of_gaps=telemetry['HNS_gap_F_D']<=.002 and telemetry['HNS_gap_D_D3']<=.002
    collapse=common_inflation or equality_rise or loss_of_gaps
    acceptable=delta['J_long3']>=-.002 and delta['Short4']>=-.002 and delta['Urban_T2I']>=-.003
    if q['Score5']>baseline['Score5'] and acceptable and improvement and not collapse:status='STRONG_POSITIVE'
    elif abs(delta['Score5'])<=.002 and acceptable and improvement and not collapse:status='STRUCTURAL_POSITIVE'
    elif collapse:status='NEGATIVE'
    elif improvement and delta['Score5']<-.002:status='TRADEOFF'
    else:status='NEGATIVE'
    return dict(classification=status,quality_delta_raw_fraction=delta,both_violations_down_at_least25percent=improvement,
        common_keep_increase_at_least5pp=common_inflation,matched_exact_equality_increase_at_least15pp=equality_rise,
        both_mean_gaps_at_most_point002=loss_of_gaps,collapse_warning=collapse,native_tradeoff_acceptable=acceptable,
        thresholds='Conservative predeclared evidence conventions, not statistical significance. Joint inflation>=5pp on all views; exact equality rise>=15pp on matched1024; both gaps<=.002; retrieval guards J_long3/Short4 -0.2pp,UrbanT2I -0.3pp.',
        automatic_full=False,automatic_other_arms=False,wait_for_human=True)


def summarize():
    from recovery.nested_d3_local_search_evidence import recall_delta
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    result=json.loads((EXP/'RESULTS.json').read_text());assert result['completed_steps']==500
    assert json.loads((EXP/'VALIDATION.json').read_text())['passed']
    audit=json.loads((EXP/'GRADIENT_AUDIT.json').read_text());assert audit['passed']
    steps=rows(search.RUN/'step500/steps.jsonl');assert len(steps)==500
    keys=[k for k in steps[0] if k.startswith('HNS_') or k in ('V_DF_hard','V_3D_hard','lambda_h')]
    def telemetry(records):return {k:statistics.fmean(float(r[k]) for r in records) for k in keys}
    selected={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,300,400,500)}
    last50=telemetry(steps[-50:]);curve=[dict(step=r['step'],**{k:r[k] for k in keys}) for r in steps]
    diag=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text());diag['HNS']=dict(selected_steps=selected,last50=last50,all500=curve)
    dump(EXP/'TRAINING_DIAGNOSTICS.json',diag)
    baseline_paths={'INC0':BASE_EXP/'RESULTS.json',
        'Anchor':ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/RESULTS.json',
        'KR234':ROOT/'experiments/nest_clip_v1/nested_d3_local_search500_v1/KR234/RESULTS.json',
        'Joint-VG':EXP/'references/JOINT_VG_RESULTS.json'}
    baseline_results={name:json.loads(p.read_text()) for name,p in baseline_paths.items()}
    q=quality_raw(result);baseline_quality={name:quality_raw(r) for name,r in baseline_results.items()}
    old_diag=json.loads((BASE_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    old_masks=json.loads((BASE_EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    old={'HNS_'+v+'_keep':old_diag['last50_views'][v]['keep_ratio'] for v in ('F','Dall','D3')}
    old.update(HNS_DF_hard_violation_ratio=old_masks['last50']['Dall_F_hard_violation'],
        HNS_3D_hard_violation_ratio=old_masks['last50']['D3_Dall_hard_violation'])
    old['HNS_gap_F_D']=old['HNS_F_keep']-old['HNS_Dall_keep'];old['HNS_gap_D_D3']=old['HNS_Dall_keep']-old['HNS_D3_keep']
    decision=assess(q,baseline_quality['INC0'],last50,old,audit['matched_cohort_delta_vs_INC0'])
    comparisons={name:dict(quality_delta_pp={k:100*(q[k]-base[k]) for k in q},
        recall_delta_pp=recall_delta(result,baseline_results[name]),source=str(baseline_paths[name]),
        source_sha256=sha(baseline_paths[name])) for name,base in baseline_quality.items()}
    masks=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks.update(HNS_last50=last50,HNS_selected_steps=selected,INC0_exact_last50=old,
        HNS_delta_vs_INC0={k:last50[k]-v for k,v in old.items()},
        matched1024_equality_and_endpoint_audit='GRADIENT_AUDIT.json',
        equality_definition='Entire512-dimensional binary mask equal per valid sample; coordinate equality separate',
        regularizer_edges=['Dall->F','D3->Dall'],mask_to_mask_stop_gradient=False,
        keep_population_note='HNS paired telemetry uses valid samples for all views. Archived INC0 F keep includes all samples; Dall/D3 and both violations use valid samples. Direct F last50 difference is descriptive with this population caveat. Collapse decisions use the exactly matched1024 valid cohort instead.')
    dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks)
    metrics,raw,sources=native_metrics(search.RUN/'step500');assert metrics==result['metrics']
    for name,data in raw.items():dump(EXP/'evaluations'/f'{name}.json',data)
    saved=json.loads((search.RUN/'supervisor-result.json').read_text());launch=json.loads((search.RUN/'launch-provenance.json').read_text())
    dump(EXP/'FORMAL_PROVENANCE.json',launch);dump(EXP/'COMMANDS.json',saved['commands'])
    dump(EXP/'HNS_FORMAL_ACCEPTANCE.json',dict(passed=True,acceptance=saved['acceptance'],
        first_five_gate=json.loads((search.RUN/'first-five-gate.json').read_text()),
        stream_proof=json.loads((search.RUN/'full-stream-proof.json').read_text()),
        strict_export=result['strict_export'],checkpoint_sha256=result['checkpoint_sha256']))
    stats=json.loads((EXP/'RUNTIME_STATS.json').read_text())
    extra=[p for p in SMOKE.rglob('*') if p.is_file()]+[p for p in SMOKE_PHASE.glob('*') if p.is_file()]
    extra+=[search.RUN/'HNS-gradient-audit500.log',search.RUN/'commands.json',search.RUN/'supervisor-result.json']
    stats['additional_local_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
        time_range_utc=[saved['started_utc'],saved['ended_utc']]) for p in extra]
    stats['additional_command_intervals']=saved['commands'];dump(EXP/'RUNTIME_STATS.json',stats)
    result.update(classification=decision['classification'],quality_raw_fraction=q,baseline_quality_raw_fraction=baseline_quality,
        all_baseline_comparisons=comparisons,HNS_last50=last50,decision=decision,
        matched_hierarchy_gradient_audit='GRADIENT_AUDIT.json',base_commit=BASE_SHA)
    dump(EXP/'RESULTS.json',result);dump(EXP/'DECISION.json',decision)
    lines=['# Hard Nested Sparsity: INC0-matched local500','',f'Classification: `{decision["classification"]}`.',
        f'Base remote INC0 commit `{BASE_SHA}`. Fresh common0 smoke5 and independent formal500; exactly500 updates, horizon4868, local-only.',
        'Original alignment and global sparsity preserved; only adjacent illegal-support ReLU surcharge added. No mask-to-mask stop-gradient. Actual masks have512 dimensions.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in {**baseline_quality,'HNS':q}.items():lines.append('| '+name+' | '+' | '.join(f'{100*x:.6f}' for x in value.values())+' |')
    lines+=['','## Five frozen native benchmarks','', '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for dataset,data in metrics.items():lines.append('| '+dataset+' | '+' | '.join(' / '.join(f'{100*d[k]:.6f}' for k in ('R@1','R@5','R@10')) for d in data.values())+' |')
    lines+=['','## All baseline deltas','']
    for name,comparison in comparisons.items():lines += [f'### vs {name}','', 'Quality delta(pp): `'+str(comparison['quality_delta_pp'])+'`.','',
        'All dataset/direction/R1/R5/R10 delta(pp): `'+json.dumps(comparison['recall_delta_pp'],sort_keys=True)+'`.','']
    directions=audit['endpoint_gradient_direction']
    direction_deltas=sorted((delta['R@1'],dataset,direction) for dataset,data in comparisons['INC0']['recall_delta_pp'].items() for direction,delta in data.items())
    failure_evidence=dict(parent_inflation={v:masks['HNS_delta_vs_INC0']['HNS_'+v+'_keep'] for v in ('F','Dall')},
        child_shrink=masks['HNS_delta_vs_INC0']['HNS_D3_keep'],mask_homogenization_warning=decision['collapse_warning'],
        hierarchy_gradient_conflicts=audit['group_cosines'],both_violation_improvement=decision['both_violations_down_at_least25percent'])
    result['scientific_failure_evidence']=failure_evidence;result['dataset_direction_R1_changes_sorted_pp']=direction_deltas
    dump(EXP/'RESULTS.json',result)
    lines+=['## Scientific questions','',
        'A. Relative to INC0: `'+str(comparisons['INC0']['quality_delta_pp'])+'`; violation/keep changes: `'+str(masks['HNS_delta_vs_INC0'])+'`. Decision uses retrieval, both violations and collapse evidence together.',
        'B. Actual endpoint gradient descent directions: `'+json.dumps(directions,sort_keys=True)+'`. Shared mask parameter groups are reported in GRADIENT_AUDIT.json; there are no independent per-view mask branches. Endpoint directions do not establish the cause of density changes across training.',
        'C. Joint inflation/equality evidence: `'+str({k:v for k,v in decision.items() if k not in ('quality_delta_raw_fraction','thresholds')})+'`. Exact pair/triple equality uses whole512-coordinate masks and a matched1024 cohort. Last50 population is separate.',
        'D. Compared with Anchor soft inclusion: `'+str(comparisons['Anchor']['quality_delta_pp'])+'`. This one seed at500 updates supports only this tested formulation, not a general ranking of hard versus soft inclusion.',
        'E. Dataset/direction R1 changes vs INC0 sorted from greatest loss to greatest gain(pp): `'+str(direction_deltas)+'`. All R5/R10 deltas are above. This identifies where the observed aggregate differences arise.',
        'F. Mechanistic evidence: `'+json.dumps(failure_evidence,sort_keys=True)+'`. Shared-group cosines measure gradient conflicts, not causation. Native encoder hidden detach is inherited unchanged: regularizer gradients to native visual/text backbone are zero; alignment updates them. Density/equality alone cannot establish over-constraint or over-shrink. No automatic continuation/variant; human review decides next steps.',
        '', 'Actual ramp(step1/100/200/300/400/500): `'+str({k:v['lambda_h'] for k,v in selected.items()})+'`.',
        'Selected-step and last50 counts/IoU/exact equality/coordinate equality/keep/gaps: TRAINING_DIAGNOSTICS.json. Full raw scalar curves, checkpoint, bare and logs remain server-local with size/SHA/time ranges in RUNTIME_STATS.json.',
        'Correctness,1000 real matched preflight,independent smoke,five-step hard gate,512000 stream match,optimizer/RNG/cursor acceptance,strict export and all five raw evaluator JSONs are included. No training images,weights,cache or credentials uploaded.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publication_paths():
    return [EXP/name for name in (*REPORTS,'REPORT.md','SEARCH_SUMMARY.md')]+list((EXP/'evaluations').glob('*.json'))


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS=ARMS;search.ENTRY_MODULE=ENTRY;search.PHASE_PREFIX='formal-nested-d3-hns500-20261008-'
    search.EDITED={'model/balanced_hparam_search.py','train/train_nested_semantic_mask.py'}
    search.EXTRA_SOURCES=EXTRA|{str((EXP/n).relative_to(ROOT)) for n in STATIC}
    search.Supervisor=HnsSupervisor
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS=ARMS;runner.ENTRY=ENTRY;runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.PUBLISH_MESSAGE='Report isolated INC0-derived Hard Nested Sparsity local500 experiment'
    runner.configure=configure;runner.summarize=summarize;runner.publication_paths=publication_paths


def main():
    configure()
    if '--smoke-worker' in sys.argv:
        sys.argv.remove('--smoke-worker');smoke_worker();return
    runner.main()


if __name__=='__main__':main()
