"""Fresh HNS-SG smoke5 -> fresh500 -> native evaluation -> matched audit -> publish."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery import nested_d3_hns500 as old
from recovery import hns_preflight as preflight
from recovery import s02_local500 as local
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, sha, rows, now

BASE_SHA='bdfd647f7b7e1a2518e4b65a8a959bc65c5606ad'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS'
INC0_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
ANCHOR_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
EXP=ROOT/'experiments/nest_clip_v1/nested_d3_hns_sg500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns-sg500-v1'
SMOKE=RUN_ROOT.parent/(RUN_ROOT.name+'.smoke5')
SMOKE_PHASE=search.LOCAL/(RUN_ROOT.name+'.smoke5-phases')
BRANCH='experiment/nested-d3-hns-sg500-v1'
ENTRY='recovery.hns_sg500'
ARMS={'HNS-SG':dict(axis='hns_sg',beta=[2.,2.],weights=[1.35,1.35,.30],r=2.,
    mode='nested_detail_d3',experiment_dir=str(EXP))}
CODE={'model/hard_nested_sparsity.py','model/balanced_hparam_search.py',
    'train/train_nested_semantic_mask.py','tools/nest_clip.py','recovery/nested_detail_gradients.py',
    'recovery/hns_ddp_correctness.py','recovery/hns_sg500.py','recovery/hns_sg_gradient_audit.py',
    'tests/test_hns_sg500.py','configs/nested_d3_hns_sg500.json','recovery/check_stage500_publish.py'}
STATIC=('BASELINE_PROVENANCE.json','MATCHED_PREFLIGHT.json','DDP_CORRECTNESS.json','CORRECTNESS.md')
REPORTS=runner.REPORT_NAMES+STATIC+('CPU_TESTS.json','SEARCH_PLAN.json','GRADIENT_AUDIT.json',
    'SMOKE_EVIDENCE.json','DECISION.json','HNS_FORMAL_ACCEPTANCE.json','FORMAL_PROVENANCE.json','COMMANDS.json')
OPTIONAL_REPORTS=('SMOKE_VERIFIER_RECOVERY.json','FORMAL_GATE_RECOVERY.json','POST500_RECOVERY.json')
OriginalConfig=search.arm_config
OriginalInvariants=search.checkpoint_invariants
OriginalStream=search.matched_stream
OriginalSupervisor=search.Supervisor


def read(p):return json.loads(p.read_text())


def audit_command(run,experiment):
    # torchrun's own argparse otherwise interprets module --run as --run-path.
    return [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
        '--max-restarts=0','-m','--','recovery.hns_sg_gradient_audit',
        '--run',str(run),'--experiment',str(experiment)]


def git_blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+str(path)],cwd=ROOT)


def arm_config(arm):
    assert arm=='HNS-SG'
    return dict(read(BASE_EXP/'config.json'),hns_detach_child=True)


def checkpoint_invariants(current,reference):
    assert current['config']['hns_detach_child'] is True
    assert current['config']['runtime_model']['hns_detach_child'] is True
    assert reference['config'].get('hns_detach_child',False) is False
    # Copy only metadata; model/AdamW tensors must not be duplicated during
    # the bounded first-five control gate.
    value=dict(current,config=dict(current['config'],runtime_model=dict(current['config']['runtime_model'])))
    value['config']['runtime_model'].pop('hns_detach_child')
    proof=OriginalInvariants(value,reference)
    proof['sole_math_change']='Child detach within hard hierarchy only'
    return proof


def matched_stream(actual,reference,arm):
    proof=OriginalStream(actual,reference,arm)
    assert all(r['HNS_detach_child'] is True for r in actual)
    for r in actual:
        assert r['inc_weight']==r['inclusion_loss']==0 and r['inclusion_enabled'] is False
        assert r['HNS_enabled'] is True and [r['HNS_beta_DF'],r['HNS_beta_3D']]==[2.,2.]
        assert r['lambda_h']==min(1.,(r['step']-1)/200.)
        surcharge=r['lambda_h']*(2*r['V_DF_hard']+2*r['V_3D_hard'])/3
        assert abs(r['HNS_surcharge']-surcharge)<2e-6
        assert abs(r['HNS_regularizer']-(r['HNS_original_sparse']+surcharge))<2e-6
        expected=(10/3)*sum(w*(r[p+'_i2t']+r[p+'_t2i']) for w,p in zip([1.35,1.35,.3],['F','O','E']))
        assert abs(r['HNS_align']-expected)<2e-5
        assert abs(r['loss']-(r['HNS_align']+r['HNS_original_sparse']+r['HNS_surcharge']))<2e-5
    proof.update(all_actual_records_HNS_SG_flag_verified=True,old_soft_inclusion_disabled=True,
        HNS_beta=[2.,2.],HNS_ramp_updates=200,alignment_and_total_literal_formula_verified=True)
    return proof


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=BASE_EXP,BASE_RUN
    search.ARMS,search.ENTRY_MODULE=ARMS,ENTRY
    search.PHASE_PREFIX='formal-hns-sg500-v1-'
    base_cfg=read(BASE_RUN/'step500/config.json')
    search.EDITED={p for p,h in base_cfg['code_sha256'].items() if sha(ROOT/p)!=h}
    search.EXTRA_SOURCES=CODE|{str((EXP/n).relative_to(ROOT)) for n in STATIC}
    search.arm_config=arm_config;search.checkpoint_invariants=checkpoint_invariants
    search.matched_stream=matched_stream;search.Supervisor=Supervisor
    old.EXP,old.BASE_EXP,old.BASE_RUN,old.BASE_SHA=EXP,BASE_EXP,BASE_RUN,BASE_SHA
    old.SMOKE,old.SMOKE_PHASE=SMOKE,SMOKE_PHASE
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS,runner.ENTRY=ARMS,ENTRY
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.PUBLISH_MESSAGE='Report fresh HNS-SG500 native evaluation and matched child-detach gradient audit'
    runner.configure=configure;runner.summarize=summarize;runner.publication_paths=publication_paths


def isolation():
    frozen={}
    for p in preflight.FROZEN_SOURCES:
        digest=hashlib.sha256(git_blob(BASE_SHA,p)).hexdigest()
        assert sha(ROOT/p)==digest,('Frozen source changed',p)
        frozen[p]=digest
    assert read(EXP/'config.json')==dict(json.loads(git_blob(BASE_SHA,str((BASE_EXP/'config.json').relative_to(ROOT)))),hns_detach_child=True)
    for n in ('config.json','RESULTS.json','TRAINING_DIAGNOSTICS.json','GRADIENT_AUDIT.json','MASK_HIERARCHY_AUDIT.json'):
        assert (BASE_EXP/n).read_bytes()==git_blob(BASE_SHA,str((BASE_EXP/n).relative_to(ROOT)))
    assert sha(STEP0)==STEP0_SHA
    baseline=read(BASE_EXP/'RESULTS.json')
    assert sha(BASE_RUN/'step500/step000500.pt')==baseline['checkpoint_sha256']
    full={}
    for branch,folder,name in (
        ('experiment/nested-d3-hns-full-v1','nested_d3_hns_full_v1','FULL_RESULTS.json'),
        ('experiment/nested-detail-d3-balanced-full-v1','nested_detail_d3_balanced_full_v1','RESULTS.json')):
        head=subprocess.check_output(['git','rev-parse','origin/'+branch],cwd=ROOT,text=True).strip()
        p=ROOT/'experiments/nest_clip_v1'/folder/name
        assert p.read_bytes()==git_blob(head,str(p.relative_to(ROOT)))
        full[branch]=dict(commit=head,path=str(p),sha256=sha(p),results=read(p).get('scores_percent',read(p).get('quality_percent')))
    cfg=read(BASE_RUN/'step500/config.json')
    return dict(passed=True,base_commit=BASE_SHA,base_branch='experiment/nested-d3-hard-nested-sparsity500-v1',
        fetched_remote_HEAD=subprocess.check_output(['git','rev-parse','origin/experiment/nested-d3-hard-nested-sparsity500-v1'],cwd=ROOT,text=True).strip(),
        common0_SHA256=STEP0_SHA,checkpoint_identity=baseline['checkpoint_sha256'],
        sole_configuration_addition={'hns_detach_child':True},frozen_source_SHA256=frozen,
        full_results_provenance=full,optimizer_groups=cfg['parameter_counts']['optimizer_groups'],
        parameter_counts=cfg['parameter_counts'],precision='fp32 parameters,bf16 encoder,no scaler',
        candidate_protocol='F all global examples; Dall/D3 all valid global examples',
        native_backbone_hidden_detach_preserved=True,hierarchy_child_detach_only=True,mask_width=512,
        no_NFS_fallback=True,baseline_only_reference_not_resume=True)


def matched_preflight():
    configure()
    preflight.BASE_SHA,preflight.BASE_EXP,preflight.BASE_RUN=BASE_SHA,BASE_EXP,BASE_RUN
    preflight.EXP,preflight.RAW=EXP,RUN_ROOT.parent/(RUN_ROOT.name+'.preflight')
    preflight.source_isolation=isolation
    preflight.main()


def smoke_proof():
    import torch
    path=SMOKE/'step500';accept=read(path/'acceptance.json')
    assert accept['passed'] and all(r['completed_updates']==r['updates_this_run']==5 and
        r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
    proof=matched_stream(rows(path/'steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],'HNS-SG')
    checkpoint=path/'step000005.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['config'];search.frozen_config(cfg,'HNS-SG')
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==STEP0_SHA
    assert cfg['run_type']=='smoke' and cfg['max_updates']==5 and payload['completed_steps']==5
    assert payload['scheduler_horizon']==4868 and payload['next_epoch']==0 and payload['next_batch']==5
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={5}
    reference=torch.load(BASE_RUN/'step500/step000005.pt',map_location='cpu',weights_only=False)
    assert payload['optimizer']['param_groups']==reference['optimizer']['param_groups']
    for key in ('component_initialization','data','parameter_counts','optimizer_groups'):
        if key in reference['config']:assert cfg[key]==reference['config'][key],key
    assert len(payload['rng_per_rank'])==4
    assert all(torch.isfinite(v).all() for st in (payload['model'],payload['adapter']) for v in st.values())
    assert all(torch.isfinite(v).all() for st in payload['optimizer']['state'].values() for v in st.values())
    from train.train_nested_semantic_mask import code_manifest
    assert cfg['code_sha256']==code_manifest(), 'Completed smoke production source changed'
    dump(EXP/'SMOKE_EVIDENCE.json',dict(passed=True,fresh_common0=True,resume=None,
        formal_must_restart_common0=True,stream_proof=proof,acceptance=accept,horizon=4868,
        checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,uploaded=False),
        completed_smoke_reverified_after_control_only_arm_name_fix=True,reviewed_utc=now()))


class Supervisor(OriginalSupervisor):
    def run(self):
        assert read(EXP/'MATCHED_PREFLIGHT.json')['passed']
        assert read(EXP/'DDP_CORRECTNESS.json')['passed'] and read(EXP/'DDP_CORRECTNESS.json')['detach_child']
        formal_run,formal_phase=local.RUN,local.PHASE
        if SMOKE.exists():
            recovery=read(EXP/'SMOKE_VERIFIER_RECOVERY.json')
            assert recovery['formal_updates']==0 and recovery['production_sources_unchanged']
            assert not search.RUN.exists() and read(EXP/'SMOKE_EVIDENCE.json')['passed']
            smoke_proof()
        else:
            assert not SMOKE_PHASE.exists()
            SMOKE.mkdir();SMOKE_PHASE.mkdir()
            local.RUN,local.PHASE=SMOKE,SMOKE_PHASE
            dump(SMOKE/'launch-provenance.json',dict(common0_SHA256=STEP0_SHA,resume=None,stop_updates=5,
                source_sha256={p:sha(ROOT/p) for p in search.source_paths()},source_base_commit=BASE_SHA))
            try:
                self.execute('smoke5',old.torchrun(ENTRY,'--smoke-worker','--config',search.CONFIG,'--init-state',STEP0,
                    '--index-dir',local.INDEX,'--image-root',local.IMAGES,'--output-dir',SMOKE/'step500',
                    '--run-type','smoke','--max-updates',5),training=True)
                smoke_proof()
            finally:local.RUN,local.PHASE=formal_run,formal_phase
        assert read(EXP/'SMOKE_EVIDENCE.json')['passed']
        runner.state('FORMAL500_RUNNING','HNS-SG',smoke_passed=True,fresh_common0=True)
        super().run()
        self.execute('HNS-SG-matched-gradient-audit500',audit_command(search.RUN,EXP))
        dump(search.RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
            commands=self.commands,started_utc=self.started,ended_utc=now()))


def decide(q,base,healthy):
    delta={k:100*(q[k]-base[k]) for k in q}
    if delta['Score5']>0 and delta['J_long3']>=-.20:status='STRONG_POSITIVE'
    elif abs(delta['Score5'])<=.05 and (delta['Short4']>=.10 or delta['J_long3']>=.10 or healthy):status='POSITIVE'
    elif healthy or delta['Short4']>=.10 or delta['J_long3']>=.10:status='MIXED'
    else:status='NEGATIVE'
    return dict(classification=status,delta_vs_HNS_v1_pp=delta,
        interpretation='Predeclared descriptive conventions: J_long3 not markedly down means <=0.20pp decline; Score5 basically tied means <=0.05pp; clear Short4/long recovery >=0.10pp. Raw Score5 strict greater controls STRONG_POSITIVE. No violation threshold veto.',
        optimization_advantage=healthy,automatic_full=False,automatic_other_arms=False,wait_for_human=True)


def summarize():
    from recovery.nested_d3_local_search_evidence import recall_delta
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    result=read(EXP/'RESULTS.json');assert result['completed_steps']==500 and read(EXP/'VALIDATION.json')['passed']
    audit=read(EXP/'GRADIENT_AUDIT.json');assert audit['passed']
    steps=rows(search.RUN/'step500/steps.jsonl');assert len(steps)==500
    keys=[k for k in steps[0] if k.startswith('HNS_') or k in ('V_DF_hard','V_3D_hard','lambda_h')]
    last50={k:statistics.fmean(float(r[k]) for r in steps[-50:]) for k in keys}
    selected={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,300,400,500)}
    diag=read(EXP/'TRAINING_DIAGNOSTICS.json')
    diag['HNS']=dict(selected_steps=selected,last50=last50)
    diag['SG_verification']=dict(child_output_hierarchy_gradient_exactly_zero=True,
        parent_expansion_verified=True,source='GRADIENT_AUDIT.json',entire_child_not_detached=True,
        Dall_also_parent='Dall receives D3->Dall parent gradient; its child gradient on Dall->F alone is zero')
    dump(EXP/'TRAINING_DIAGNOSTICS.json',diag)
    q=old.quality_raw(result)
    bases={'INC0':read(INC0_EXP/'RESULTS.json'),'Anchor':read(ANCHOR_EXP/'RESULTS.json'),'HNS-v1':read(BASE_EXP/'RESULTS.json')}
    comparisons={n:dict(quality_delta_pp={k:100*(q[k]-old.quality_raw(b)[k]) for k in q},
        recall_delta_pp=recall_delta(result,b)) for n,b in bases.items()}
    previous=read(BASE_EXP/'TRAINING_DIAGNOSTICS.json')['HNS']['last50']
    delta={k:v-previous[k] for k,v in last50.items() if k in previous}
    common=all(delta['HNS_'+v+'_keep']>=.05 for v in ('F','Dall','D3'))
    converged=last50['HNS_gap_F_D']<=.002 and last50['HNS_gap_D_D3']<=.002
    modes=audit['checkpoints']['HNS-SG']['modes']
    vg='shared_visual_mask'
    sg_cos=modes['HNS-SG']['group_cosines'][vg]['cosine_hierarchy_original_sparsity']
    no_cos=modes['HNS-v1']['group_cosines'][vg]['cosine_hierarchy_original_sparsity']
    healthy=sg_cos is not None and no_cos is not None and sg_cos>no_cos+.05 and not common and not converged
    decision=decide(q,old.quality_raw(bases['HNS-v1']),healthy)
    masks=read(EXP/'MASK_HIERARCHY_AUDIT.json')
    masks.update(HNS_last50=last50,HNS_selected_steps=selected,HNS_delta_vs_v1=delta,
        regularizer_edges=['Dall->F','D3->Dall'],hierarchy_child_detach=True,
        common_inflation_at_least5pp=common,both_keep_gaps_at_most_point002=converged,
        population='HNS last50 valid-only paired positive masks; matched1024 same cohort audit also provided')
    dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks)
    metrics,raw,_=native_metrics(search.RUN/'step500');assert metrics==result['metrics']
    for name,data in raw.items():dump(EXP/'evaluations'/f'{name}.json',data)
    saved=read(search.RUN/'supervisor-result.json');launch=read(search.RUN/'launch-provenance.json')
    dump(EXP/'FORMAL_PROVENANCE.json',launch);dump(EXP/'COMMANDS.json',saved['commands'])
    dump(EXP/'HNS_FORMAL_ACCEPTANCE.json',dict(passed=True,acceptance=saved['acceptance'],
        first_five_gate=read(search.RUN/'first-five-gate.json'),stream_proof=read(search.RUN/'full-stream-proof.json'),
        strict_export=result['strict_export'],checkpoint_sha256=result['checkpoint_sha256']))
    stats=read(EXP/'RUNTIME_STATS.json')
    extra=[p for p in SMOKE.rglob('*') if p.is_file()]+[p for p in SMOKE_PHASE.glob('*') if p.is_file()]
    extra+=[search.RUN/'HNS-SG-matched-gradient-audit500.log',search.RUN/'commands.json',search.RUN/'supervisor-result.json']
    stats['additional_local_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
        time_range_utc=[saved['started_utc'],saved['ended_utc']]) for p in extra]
    stats['additional_command_intervals']=saved['commands'];dump(EXP/'RUNTIME_STATS.json',stats)
    questions=dict(Q1_score_above_HNS_v1=q['Score5']>old.quality_raw(bases['HNS-v1'])['Score5'],
        Q2_short_recovery_and_long_tradeoffs=comparisons['HNS-v1'],
        Q3_parent_keep_delta={v:delta['HNS_'+v+'_keep'] for v in ('F','Dall')},
        Q4_D3_keep_delta=delta['HNS_D3_keep'],
        Q5_violations={e:last50['HNS_'+e+'_hard_violation_ratio'] for e in ('DF','3D')},
        Q6_same_checkpoint_visual_mask_cosine=dict(SG=sg_cos,no_SG=no_cos,delta=None if sg_cos is None or no_cos is None else sg_cos-no_cos),
        shared_parameter_caveat='Zero direct child-output gradient does not prevent shared-parameter changes from changing that child; density changes alone do not prove causality.')
    result.update(classification=decision['classification'],quality_raw_fraction=q,all_baseline_comparisons=comparisons,
        HNS_last50=last50,HNS_delta_vs_v1=delta,decision=decision,scientific_questions=questions,
        common_inflation=common,convergence_warning=converged,base_commit=BASE_SHA,
        gradient_audit='GRADIENT_AUDIT.json',checkpoint_SHA256=result['checkpoint_sha256'],bare_SHA256=result['strict_export']['bare_sha256'])
    dump(EXP/'RESULTS.json',result);dump(EXP/'DECISION.json',decision)
    lines=['# HNS-SG: hard hierarchy with detached child at500','',f'Classification: `{decision["classification"]}`.',
        'Fresh common0 independent smoke5, then fresh common0 formal500; exactly500 updates, horizon4868, local-only.',
        'Sole mathematical change: ReLU(child.detach()-parent) on the actual Hard-ST positive masks. Original alignment1.35/1.35/.30 and sparse1/2/2 unchanged; beta2/2, ramp200; old soft inclusion0.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in {**bases,'HNS-SG':result}.items():lines.append('| '+name+' | '+' | '.join(f'{100*x:.6f}' for x in old.quality_raw(r).values())+' |')
    lines+=['','| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for name,m in metrics.items():lines.append('| '+name+' | '+' | '.join(' / '.join(f'{100*m[d][k]:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I'))+' |')
    for name,c in comparisons.items():lines+=['',f'## vs {name}','', 'Quality deltas(pp): `'+json.dumps(c['quality_delta_pp'])+'`.','All recall deltas(pp): `'+json.dumps(c['recall_delta_pp'])+'`.']
    lines+=['','## Q1–Q6','',json.dumps(questions,sort_keys=True),
        '', 'Last50 mask telemetry: `'+json.dumps(last50)+'`.',
        'Common inflation >=5pp on all views: '+str(common)+'; both keep gaps <=.002: '+str(converged)+'. These descriptive warnings do not veto retrieval.',
        'Endpoint hierarchy gradients: `'+json.dumps(modes['HNS-SG']['endpoint_gradient_direction'])+'`.',
        'Matched same-graph SG/no-SG forward scalar equality, per-edge zero direct child gradients, unchanged parent gradients, original sparsity and alignment gradients, all six component/group norms and cosines are in GRADIENT_AUDIT.json for BOTH immutable HNS-v1 and HNS-SG checkpoints.',
        'Shared mask parameters mean Dall is child on DF but parent on3D. Total hierarchy can update Dall as parent; child detach is edge-local. Original native hidden detach preserved: hierarchy/sparsity native-backbone gradients are zero, alignment remains active.',
        'Selection conventions: '+decision['interpretation'],
        'HNS full vs D3 Balanced full provenance retained in BASELINE_PROVENANCE.json; this500 experiment cannot establish the cause of their long-term gap. One seed; no statistical synergy or general causal claim.',
        f'Checkpoint SHA256: `{result["checkpoint_sha256"]}`; bare SHA256: `{result["strict_export"]["bare_sha256"]}`.',
        'Large checkpoints/weights/preflight tokens/raw logs remain server-local, inventoried by path/size/SHA/time range in RUNTIME_STATS.json and MATCHED_PREFLIGHT.json. /root is disposable; NFS originals retained.',
        'No full4868 or any other experiment starts. Stop and wait for human decision.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publication_paths():
    return [EXP/n for n in (*REPORTS,'SEARCH_SUMMARY.md')]+[
        EXP/n for n in OPTIONAL_REPORTS if (EXP/n).is_file()]+list((EXP/'evaluations').glob('*.json'))


def main():
    configure()
    if '--preflight' in sys.argv:matched_preflight();return
    if '--smoke-worker' in sys.argv:
        sys.argv.remove('--smoke-worker');search.activate('HNS-SG')
        from train import train_nested_semantic_mask as trainer
        trainer.sampling_diagnostics=search.observe_selection
        local.RUN,local.PHASE=SMOKE,SMOKE_PHASE;local.worker();return
    runner.main()


if __name__=='__main__':main()
