"""Final two frozen HNS arms, with E1 publication before E2 and no full run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from recovery import nested_d3_local_search as search
from recovery import nested_d3_hns500 as old
from recovery import nested_d3_followup500 as publication
from recovery import hns_preflight as preflight
from recovery import s02_local500 as local
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, sha, rows, now

BASE_SHA = 'bdfd647f7b7e1a2518e4b65a8a959bc65c5606ad'
BASE_EXP = ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1'
BASE_RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS'
INC0_EXP, INC0_RUN = preflight.BASE_EXP, preflight.BASE_RUN
ENTRY = 'recovery.hns_final_two500'
RUN_ROOT = ROOT/'runtime/SAID-nest-clip-v1/hns-final-two500-20261008'
MAIN_LOG = RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
IDENTITY = RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
ARMS = {
    'HNS-Weak': dict(axis='hns_beta', beta=[1.,1.], weights=[1.35,1.35,.30], r=2.,
        mode='nested_detail_d3', experiment_dir=str(ROOT/'experiments/nest_clip_v1/nested_d3_hns_weak500_v1'),
        branch='experiment/nested-d3-hns-weak500-v1'),
    'HNS-InnerFocus': dict(axis='hns_beta', beta=[1.,2.], weights=[1.35,1.35,.30], r=2.,
        mode='nested_detail_d3', experiment_dir=str(ROOT/'experiments/nest_clip_v1/nested_d3_hns_innerfocus500_v1'),
        branch='experiment/nested-d3-hns-innerfocus500-v1'),
}
CODE = {'model/hard_nested_sparsity.py','model/balanced_hparam_search.py',
    'train/train_nested_semantic_mask.py','tools/nest_clip.py','recovery/nested_detail_gradients.py',
    'recovery/nested_d3_local_search.py','recovery/hns_gradient_audit.py',
    'recovery/hns_ddp_correctness.py','recovery/hns_final_two500.py',
    'recovery/check_stage500_publish.py','tests/test_hns_final_two500.py'}
STATIC = ('BASELINE_PROVENANCE.json','MATCHED_PREFLIGHT.json','DDP_CORRECTNESS.json',
    'INC0_HIERARCHY_REFERENCE.json','HNS_V1_GRADIENT_REFERENCE.json')
REPORTS = publication.REPORT_NAMES + ('GRADIENT_AUDIT.json','SMOKE_EVIDENCE.json',
    'HNS_FORMAL_ACCEPTANCE.json','FORMAL_PROVENANCE.json','COMMANDS.json','DECISION.json')
OriginalSupervisor = search.Supervisor


def read(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=120).strip()


def configure(arm):
    spec=ARMS[arm];exp=Path(spec['experiment_dir'])
    search.EXP, search.RUN_ROOT, search.BRANCH = exp, RUN_ROOT, spec['branch']
    search.ANCHOR_EXP, search.ANCHOR_RUN = BASE_EXP, BASE_RUN
    search.ARMS, search.ENTRY_MODULE = ARMS, ENTRY
    search.PHASE_PREFIX = 'formal-hns-final-two500-20261008-'
    search.EDITED = {'model/hard_nested_sparsity.py','model/balanced_hparam_search.py','train/train_nested_semantic_mask.py'}
    search.EXTRA_SOURCES = CODE | {str((exp/n).relative_to(ROOT)) for n in STATIC}
    search.activate(arm)
    old.EXP, old.BASE_EXP, old.BASE_RUN, old.BASE_SHA = exp, BASE_EXP, BASE_RUN, BASE_SHA
    old.SMOKE = RUN_ROOT/(arm+'.smoke5')
    old.SMOKE_PHASE = search.LOCAL/(RUN_ROOT.name+'-'+arm+'.smoke5-phases')
    publication.EXP, publication.RUN_ROOT, publication.BRANCH = exp, RUN_ROOT, spec['branch']
    publication.ARMS = {arm:spec}
    publication.MAIN_LOG = MAIN_LOG
    publication.PUBLISH_MESSAGE = 'Report fresh common0 '+arm+' local500 and immutable native evaluation'
    publication.publication_paths = lambda: report_paths(arm)


def isolation():
    """Read pinned Git blobs; code changes are audited below and via unit tests."""
    arm=search.ARM;exp=search.ARM_EXP;frozen={}
    for p in preflight.FROZEN_SOURCES:
        digest=hashlib.sha256(subprocess.check_output(['git','show',BASE_SHA+':'+p],cwd=ROOT)).hexdigest()
        assert sha(ROOT/p)==digest,('Frozen source changed',p)
        frozen[p]=digest
    base=json.loads(subprocess.check_output(['git','show',BASE_SHA+':'+str((BASE_EXP/'config.json').relative_to(ROOT))],cwd=ROOT))
    assert read(exp/'config.json')==dict(base,hns_beta=ARMS[arm]['beta'])
    for name in ('RESULTS.json','REPORT.md','TRAINING_DIAGNOSTICS.json','GRADIENT_AUDIT.json','MASK_HIERARCHY_AUDIT.json'):
        blob=subprocess.check_output(['git','show',BASE_SHA+':'+str((BASE_EXP/name).relative_to(ROOT))],cwd=ROOT)
        assert (BASE_EXP/name).read_bytes()==blob
    result=read(BASE_EXP/'RESULTS.json');cfg=read(BASE_RUN/'step500/config.json')
    assert result['evaluation_checkpoint_immutable'] and sha(BASE_RUN/'step500/step000500.pt')==result['checkpoint_sha256']
    assert sha(STEP0)==STEP0_SHA
    assert cfg['init_sha256']==STEP0_SHA and cfg['horizon']==4868 and cfg['workers']==8
    assert cfg['view_weights']==[1.35,1.35,.30] and cfg['sampling_mode']=='nested_detail_d3' and cfg['inclusion_max']==0
    assert cfg['batch_size']==256 and cfg['world_size']==4 and cfg['accumulation']==1
    return dict(passed=True,base_commit=BASE_SHA,base_branch='experiment/nested-d3-hard-nested-sparsity500-v1',
        fetched_remote_HEAD=git('rev-parse','origin/experiment/nested-d3-hard-nested-sparsity500-v1'),
        sole_configuration_addition={'hns_beta':ARMS[arm]['beta']},
        common0_SHA256=STEP0_SHA,checkpoint_identity=result['checkpoint_sha256'],
        baseline_raw_quality=old.quality_raw(result),INC0_raw_quality=old.quality_raw(read(INC0_EXP/'RESULTS.json')),
        source_and_report_provenance={n:dict(path=str(BASE_EXP/n),sha256=sha(BASE_EXP/n)) for n in ('RESULTS.json','REPORT.md')},
        frozen_source_SHA256=frozen,optimizer_groups=cfg['parameter_counts']['optimizer_groups'],
        parameter_counts=cfg['parameter_counts'],precision='fp32 parameters,bf16 encoder,no scaler',
        mask_width=512,mask_to_mask_stop_gradient=False,original_hidden_detach_preserved=True,
        no_NFS_fallback=True,no_copy_or_full_decode_audit=True)


def matched_preflight(arm):
    configure(arm)
    preflight.BASE_SHA, preflight.BASE_EXP, preflight.BASE_RUN = BASE_SHA, BASE_EXP, BASE_RUN
    preflight.EXP = search.ARM_EXP
    preflight.RAW = RUN_ROOT/(arm+'.preflight')
    preflight.source_isolation = isolation
    for source,name in ((BASE_EXP/'INC0_HIERARCHY_REFERENCE.json','INC0_HIERARCHY_REFERENCE.json'),
                        (BASE_EXP/'GRADIENT_AUDIT.json','HNS_V1_GRADIENT_REFERENCE.json')):
        assert not (search.ARM_EXP/name).exists()
        # Small immutable reviewed JSON reference only; no image/checkpoint copies.
        dump(search.ARM_EXP/name,read(source))
        assert read(search.ARM_EXP/name)==read(source)
    preflight.main()


class Supervisor(OriginalSupervisor):
    def run(self):
        exp=search.ARM_EXP;arm=search.ARM
        assert read(exp/'MATCHED_PREFLIGHT.json')['passed'] and read(exp/'DDP_CORRECTNESS.json')['passed']
        assert read(exp/'DDP_CORRECTNESS.json')['beta']==ARMS[arm]['beta']
        formal_run,formal_phase=local.RUN,local.PHASE
        assert not old.SMOKE.exists() and not old.SMOKE_PHASE.exists()
        old.SMOKE.mkdir(parents=True);old.SMOKE_PHASE.mkdir()
        local.RUN,local.PHASE=old.SMOKE,old.SMOKE_PHASE
        dump(old.SMOKE/'launch-provenance.json',dict(common0_SHA256=STEP0_SHA,resume=None,stop_updates=5,
            beta=ARMS[arm]['beta'],source_sha256={p:sha(ROOT/p) for p in search.source_paths()}))
        try:
            self.execute('smoke5',old.torchrun(ENTRY,'--arm',arm,'--smoke-worker','--config',search.CONFIG,
                '--init-state',STEP0,'--index-dir',local.INDEX,'--image-root',local.IMAGES,
                '--output-dir',old.SMOKE/'step500','--run-type','smoke','--max-updates',5),training=True)
            smoke_proof(arm)
        finally:
            local.RUN,local.PHASE=formal_run,formal_phase
        state('FORMAL500_RUNNING',arm)
        super().run()
        self.execute('HNS-gradient-audit500',old.torchrun('recovery.hns_gradient_audit',
            '--run',search.RUN,'--experiment',exp))
        dump(search.RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
            commands=self.commands,started_utc=self.started,ended_utc=now()))


def smoke_proof(arm):
    import torch
    path=old.SMOKE/'step500';accept=read(path/'acceptance.json')
    assert accept['passed'] and all(r['completed_updates']==r['updates_this_run']==5 and r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
    proof=search.matched_stream(rows(path/'steps.jsonl'),rows(BASE_RUN/'step500/steps.jsonl')[:5],arm)
    checkpoint=path/'step000005.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['config'];search.frozen_config(cfg,arm)
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['init_sha256']==STEP0_SHA
    assert cfg['run_type']=='smoke' and cfg['max_updates']==5 and payload['completed_steps']==5
    assert payload['scheduler_horizon']==4868 and payload['next_epoch']==0 and payload['next_batch']==5
    assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={5}
    baseline=torch.load(BASE_RUN/'step500/step000005.pt',map_location='cpu',weights_only=False)
    assert payload['optimizer']['param_groups']==baseline['optimizer']['param_groups']
    for key in ('component_initialization','data','parameter_counts','optimizer_groups'):
        if key in baseline['config']:assert cfg[key]==baseline['config'][key],key
    assert len(payload['rng_per_rank'])==4
    assert all(torch.isfinite(v).all() for st in (payload['model'],payload['adapter']) for v in st.values())
    assert all(torch.isfinite(v).all() for st in payload['optimizer']['state'].values() for v in st.values())
    dump(search.ARM_EXP/'SMOKE_EVIDENCE.json',dict(passed=True,fresh_common0=True,resume=None,
        formal_must_restart_common0=True,stream_proof=proof,acceptance=accept,horizon=4868,beta=cfg['hns_beta'],
        checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,uploaded=False),reviewed_utc=now()))


def summarize_arm(arm):
    from recovery.nested_d3_local_search_evidence import recall_delta
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    exp=search.ARM_EXP;result=read(exp/'RESULTS.json');assert result['completed_steps']==500
    assert read(exp/'VALIDATION.json')['passed']
    audit=read(exp/'GRADIENT_AUDIT.json');assert audit['passed'] and audit['beta']==ARMS[arm]['beta']
    steps=rows(search.RUN/'step500/steps.jsonl');assert len(steps)==500
    keys=[k for k in steps[0] if k.startswith('HNS_') or k in ('V_DF_hard','V_3D_hard','lambda_h')]
    last50={k:statistics.fmean(float(r[k]) for r in steps[-50:]) for k in keys}
    selected={str(r['step']):{k:r[k] for k in keys} for r in steps if r['step'] in (1,100,200,300,400,500)}
    diag=read(exp/'TRAINING_DIAGNOSTICS.json')
    diag['HNS']=dict(selected_steps=selected,last50=last50,all500=[dict(step=r['step'],**{k:r[k] for k in keys}) for r in steps])
    dump(exp/'TRAINING_DIAGNOSTICS.json',diag)
    q=old.quality_raw(result);bases={'INC0':read(INC0_EXP/'RESULTS.json'),'HNS-v1':read(BASE_EXP/'RESULTS.json')}
    comparisons={n:dict(quality_delta_pp={k:100*(q[k]-old.quality_raw(b)[k]) for k in q},recall_delta_pp=recall_delta(result,b)) for n,b in bases.items()}
    masks=read(exp/'MASK_HIERARCHY_AUDIT.json')
    masks.update(HNS_last50=last50,HNS_selected_steps=selected,beta=ARMS[arm]['beta'],
        regularizer_edges=['Dall->F','D3->Dall'],mask_to_mask_stop_gradient=False,
        equality_definition='Entire512-dimensional binary mask equal per valid sample; coordinate equality separate',
        keep_population_note='HNS paired telemetry is valid-only; archived INC0 F keep is all samples. Matched1024 valid cohort available in gradient audit.')
    dump(exp/'MASK_HIERARCHY_AUDIT.json',masks)
    metrics,raw,_=native_metrics(search.RUN/'step500');assert metrics==result['metrics']
    for name,data in raw.items():dump(exp/'evaluations'/f'{name}.json',data)
    saved=read(search.RUN/'supervisor-result.json');launch=read(search.RUN/'launch-provenance.json')
    dump(exp/'FORMAL_PROVENANCE.json',launch);dump(exp/'COMMANDS.json',saved['commands'])
    dump(exp/'HNS_FORMAL_ACCEPTANCE.json',dict(passed=True,acceptance=saved['acceptance'],
        first_five_gate=read(search.RUN/'first-five-gate.json'),stream_proof=read(search.RUN/'full-stream-proof.json'),
        strict_export=result['strict_export'],checkpoint_sha256=result['checkpoint_sha256']))
    stats=read(exp/'RUNTIME_STATS.json')
    extra=[p for p in old.SMOKE.rglob('*') if p.is_file()]+[p for p in old.SMOKE_PHASE.glob('*') if p.is_file()]
    extra+=[search.RUN/'HNS-gradient-audit500.log',search.RUN/'commands.json',search.RUN/'supervisor-result.json']
    stats['additional_local_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,time_range_utc=[saved['started_utc'],saved['ended_utc']]) for p in extra]
    stats['additional_command_intervals']=saved['commands'];dump(exp/'RUNTIME_STATS.json',stats)
    decision=dict(classification='COMPLETED_FIXED_ARM_AWAITING_FINAL_COMPARISON',
        selection_rule=read(exp/'SEARCH_PLAN.json')['selection_rule'],automatic_full=False,automatic_third_arm=False,
        mask_behaviour_is_scientific_result_not_a_stop_condition=True)
    result.update(classification=decision['classification'],quality_raw_fraction=q,all_baseline_comparisons=comparisons,
        HNS_last50=last50,decision=decision,base_commit=BASE_SHA,beta=ARMS[arm]['beta'])
    dump(exp/'RESULTS.json',result);dump(exp/'DECISION.json',decision)
    lines=[f'# {arm}: frozen HNS local500','',f'Beta {ARMS[arm]["beta"]}; original sparse [1,2,2], alignment [1.35,1.35,.30], fixed K3, old inclusion0.',
        'Independent common0 smoke5 then fresh formal500; resume=None, horizon4868. Actual Hard-ST ReLU adjacent-edge surcharge, joint parent/child gradients.',
        'All512000 samples/strings/tokens/indices and LR verified against pinned HNS-v1. Local-only; no full, copy or full decode audit.',
        '', '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for name,m in metrics.items():lines.append('| '+name+' | '+' | '.join(' / '.join(f'{100*m[d][k]:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I'))+' |')
    lines+=['','Raw quality: `'+json.dumps(q)+'`.']
    for n,c in comparisons.items():lines+=['',f'## vs {n}','', 'Quality deltas(pp): `'+json.dumps(c['quality_delta_pp'])+'`.','All R1/R5/R10 deltas(pp): `'+json.dumps(c['recall_delta_pp'])+'`.']
    lines+=['','## Structure and gradients','', 'Last50 valid-pair telemetry: `'+json.dumps(last50)+'`.',
        'Selected1/100/200/300/400/500 and full500 curve: TRAINING_DIAGNOSTICS.json. Equality is whole mask, not coordinate equality.',
        'Readonly immutable1024 audit: GRADIENT_AUDIT.json; original8-batch alignment protocol: GRADIENT_SPOTCHECK.json.',
        'Measured hierarchy norm ratios vs HNS-v1: `'+json.dumps(audit['hierarchy_gradient_norm_ratio_vs_HNS_v1'])+'`.',
        'Measured weighted inner/outer norm ratios: `'+json.dumps(audit['weighted_inner_outer_gradient_norm_ratio'])+'`.',
        'Endpoint parent expansion and child contraction: `'+json.dumps(audit['endpoint_gradient_direction'])+'`.',
        'No independent view-specific parameter branches exist; output norms and shared parameter groups are distinguished. Original hidden-input detach means regularizers have zero native-backbone gradient; preserved unchanged.',
        'Do not assume trained-model gradients scale exactly with beta; masks, sigmoid derivatives and shared-parameter conflicts change. One seed at500 does not establish causation.',
        '', 'Checkpoint/bare/raw logs stay server-local. RUNTIME_STATS.json records paths, sizes, SHA256 and time intervals. /root disposable overlay; NFS originals retained.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')


def champion(quality,hierarchy):
    """Predeclared raw-max tie cohort avoids nontransitive pairwise ties."""
    raw_max=max(quality,key=lambda n:quality[n]['Score5'])
    top=quality[raw_max]['Score5']
    tied=[n for n,q in quality.items() if top-q['Score5']<.0005]
    chosen=max(tied,key=lambda n:(quality[n]['J_long3'],quality[n]['J_long'],quality[n]['Urban_T2I'],quality[n]['Short4'],-hierarchy[n]))
    long=max(quality,key=lambda n:(quality[n]['J_long3'],quality[n]['J_long']))
    return dict(BEST_SCORE5_CANDIDATE=chosen,BEST_LONG_CANDIDATE=long,raw_Score5_maximum=raw_max,
        approximate_tie_cohort=tied,tie_threshold_raw=.0005,automatic_full=False,automatic_third_arm=False)


def combined():
    candidates={'INC0':INC0_EXP,'HNS-v1':BASE_EXP,**{n:Path(a['experiment_dir']) for n,a in ARMS.items()}}
    results={n:read(p/'RESULTS.json') for n,p in candidates.items()}
    quality={n:old.quality_raw(r) for n,r in results.items()}
    telemetry={n:read(p/'TRAINING_DIAGNOSTICS.json')['HNS']['last50'] for n,p in candidates.items() if n!='INC0'}
    d=read(INC0_EXP/'TRAINING_DIAGNOSTICS.json');m=read(INC0_EXP/'MASK_HIERARCHY_AUDIT.json')['last50']
    inc={'HNS_'+v+'_keep':d['last50_views'][v]['keep_ratio'] for v in ('F','Dall','D3')}
    inc.update(HNS_DF_hard_violation_ratio=m['Dall_F_hard_violation'],HNS_3D_hard_violation_ratio=m['D3_Dall_hard_violation'])
    # Exactly matched real cohort supplies baseline equality, not guessed zeros.
    inc.update({k:v for k,v in read(BASE_EXP/'INC0_HIERARCHY_REFERENCE.json')['telemetry'].items() if 'equality' in k})
    telemetry['INC0']=inc
    choice=champion(quality,{n:t['HNS_DF_hard_violation_ratio']+t['HNS_3D_hard_violation_ratio'] for n,t in telemetry.items()})
    value=dict(status='BOTH_ARMS_COMPLETED',quality_raw_fraction=quality,structure=telemetry,**choice,
        metrics={n:r['metrics'] for n,r in results.items()},beta={'INC0':[0,0],'HNS-v1':[2,2],**{n:a['beta'] for n,a in ARMS.items()}},
        selection_rule=read(search.EXP/'SEARCH_PLAN.json')['selection_rule'],
        INC0_equality_population='Matched1024 step500; baseline last50 equality was not archived. Other INC0 structure is archived last50; F keep all samples.',
        automatic_followup=False)
    lines=['# Final HNS local500 comparison','', '| Model | Beta DF/3D | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |','|---|---|---:|---:|---:|---:|---|']
    for n in candidates:
        q=quality[n];lines.append('| '+n+' | '+str(value['beta'][n])+' | '+' | '.join(f'{100*q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {100*q["Urban_I2T"]:.6f} / {100*q["Urban_T2I"]:.6f} |')
    lines+=['','| Model | COCO I/T | Urban I/T | Flickr I/T | DOCCI I/T | Long I/T |','|---|---|---|---|---|---|']
    for n,r in results.items():lines.append('| '+n+' | '+' | '.join(' / '.join(f'{100*r["metrics"][ds][dr]["R@1"]:.6f}' for dr in ('I2T','T2I')) for ds in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI'))+' |')
    lines+=['','| Model | F keep | Dall keep | D3 keep | DF violation | 3D violation | F=D | D=D3 | all equal |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,t in telemetry.items():lines.append('| '+n+' | '+' | '.join(f'{t[k]:.8f}' for k in ('HNS_F_keep','HNS_Dall_keep','HNS_D3_keep','HNS_DF_hard_violation_ratio','HNS_3D_hard_violation_ratio','HNS_DF_exact_equality_ratio','HNS_3D_exact_equality_ratio','HNS_triple_exact_equality_ratio'))+' |')
    def delta(a,b):return {k:100*(quality[a][k]-quality[b][k]) for k in quality[a]}
    questions={
        'Q1':{'contrast':'Weak vs v1; uniform pressure reduction', 'quality_delta_pp':delta('HNS-Weak','HNS-v1')},
        'Q2':{'contrast':'Inner vs Weak (inner only) and Inner vs v1 (outer only)', 'Inner_minus_Weak_pp':delta('HNS-InnerFocus','HNS-Weak'),'Inner_minus_v1_pp':delta('HNS-InnerFocus','HNS-v1')},
        'Q3':{'contrast':'Inner minus Weak isolates inner coefficient1->2; inspect J_long3,J_long and all three long datasets','delta_pp':delta('HNS-InnerFocus','HNS-Weak')},
        'Q4':{'contrast':'Inner minus v1 isolates outer2->1; inspect Short4 and individual COCO/Flickr alongside long datasets','delta_pp':delta('HNS-InnerFocus','HNS-v1')},
        'Q5':{n:{'keep_delta_vs_v1':{v:telemetry[n]['HNS_'+v+'_keep']-telemetry['HNS-v1']['HNS_'+v+'_keep'] for v in ('F','Dall','D3')},'exact_equality':{k:v for k,v in telemetry[n].items() if 'exact_equality' in k}} for n in ARMS},
        'Q6':{'uniform_contrast_pp':delta('HNS-Weak','HNS-v1'),'inner_granularity_contrast_pp':delta('HNS-InnerFocus','HNS-Weak'),
            'limitation':'One seed,500 updates; these measured contrasts suggest local effects only, not general causation or full-training ranking.'}}
    value['scientific_questions']=questions
    lines+=['','## Scientific contrasts Q1–Q6','', 'INC0 equality uses a matched1024 audit because historical last50 whole-mask equality was not recorded. INC0 F keep includes all samples; new HNS keep is valid-only. Comparable matched1024 diagnostics are in each gradient audit.']
    for n,data in questions.items():lines+=['',n+': `'+json.dumps(data)+'`.']
    lines+=['','## Selection','',json.dumps(choice),
        'Highest raw Score5 primary; candidates within strictly<.05pp of raw maximum form a tie cohort, then J_long3,J_long,UrbanT2I,Short4,lower summed violations. Long candidate: J_long3 then J_long. Structure does not override a clear Score5 advantage.',
        'Recommend BEST_SCORE5_CANDIDATE for human consideration of full training. No full/third beta/second seed/other variant started.']
    dump(search.EXP/'FINAL_COMPARISON.json',value)
    (search.EXP/'SEARCH_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return value


def report_paths(arm):
    exp=Path(ARMS[arm]['experiment_dir'])
    extra=['FINAL_COMPARISON.json','SEARCH_SUMMARY.md'] if arm=='HNS-InnerFocus' else []
    return [exp/n for n in REPORTS+tuple(extra)]+list((exp/'evaluations').glob('*.json'))


def publish_preflight(arm):
    paths=[search.ARM_EXP/n for n in STATIC]
    assert not git('diff','--cached','--name-only')
    subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
    from recovery.check_stage500_publish import inspect
    assert inspect()['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Verify '+arm+' matched real1000 and four-rank objective'],cwd=ROOT,check=True)
    push_verify(ARMS[arm]['branch'])


def push_verify(branch):
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],cwd=ROOT,check=True,timeout=120)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/'+branch)==git('rev-parse','FETCH_HEAD')


def state(status,arm,**extra):
    dump(Path(ARMS[arm]['experiment_dir'])/'QUEUE_STATE.json',dict(status=status,active_arm=arm,
        queue=list(ARMS),runner_pid=os.getpid(),session=os.getsid(0),main_log=str(MAIN_LOG),updated_utc=now(),
        automatic_full=False,automatic_third_arm=False,**extra))


def run_queue():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('HARD_STOP supervisor signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock_path=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]
        try:
            for arm in ARMS:
                configure(arm)
                if completed:
                    receipt=read(Path(ARMS[completed[-1]]['experiment_dir'])/'GITHUB_RECEIPT.json')
                    assert receipt['passed'] and receipt['remote_HEAD_matches_local']
                    assert git('rev-parse','origin/'+receipt['branch'])==receipt['commit']==git('rev-parse','HEAD')
                    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    subprocess.run(['git','switch','-c',ARMS[arm]['branch']],cwd=ROOT,check=True)
                    state('CORRECTNESS_AND_PREFLIGHT',arm)
                    command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc-per-node=4','--max-restarts=0',
                        '-m','recovery.hns_ddp_correctness','--beta',*map(str,ARMS[arm]['beta']),'--output',str(search.ARM_EXP/'DDP_CORRECTNESS.json')]
                    subprocess.run(command,cwd=ROOT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
                    matched_preflight(arm);publish_preflight(arm)
                assert git('branch','--show-current')==ARMS[arm]['branch']
                state('SMOKE5_RUNNING',arm)
                Supervisor().run();summarize_arm(arm)
                if arm==list(ARMS)[-1]:combined()
                state('PUBLISHING',arm)
                receipt=publication.publish();completed.append(arm)
                state('COMPLETED_AND_SYNCED',arm,completed_arms=completed,github=receipt)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            state('BOTH_COMPLETED_AND_SYNCED_GPU_IDLE',arm,completed_arms=completed,GPU_idle=True)
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',arm,completed_arms=completed,error=type(error).__name__+': '+str(error),automatic_retry=False)
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=list(ARMS),default='HNS-Weak')
    for flag in ('worker','gradient','smoke-worker','preflight','detach'):parser.add_argument('--'+flag,action='store_true')
    args,remaining=parser.parse_known_args();configure(args.arm)
    if args.worker or args.gradient or args.smoke_worker:
        sys.argv=[sys.argv[0],*remaining]
        if args.gradient:search.gradient()
        else:
            if args.smoke_worker:
                from train import train_nested_semantic_mask as trainer
                trainer.sampling_diagnostics=search.observe_selection
                local.RUN,local.PHASE=old.SMOKE,old.SMOKE_PHASE
                local.worker()
            else:search.worker()
        return
    assert not remaining
    if args.preflight:matched_preflight(args.arm);return
    if args.detach:
        assert not IDENTITY.exists() and not (RUN_ROOT/'HNS-Weak').exists()
        assert git('branch','--show-current')==ARMS['HNS-Weak']['branch']
        for arm in ARMS:
            exp=Path(ARMS[arm]['experiment_dir']);assert read(exp/'CPU_TESTS.json')['passed']
            assert read(exp/'config.json')==dict(read(BASE_EXP/'config.json'),hns_beta=ARMS[arm]['beta'])
        with MAIN_LOG.open('ab',buffering=0) as handle:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',ENTRY],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                close_fds=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4'))
        value=dict(pid=p.pid,session=p.pid,queue=list(ARMS),main_log=str(MAIN_LOG),start_new_session=True,
            stdin='/dev/null',shell_session_independent=True,started_utc=now())
        dump(IDENTITY,value);print(json.dumps(value),flush=True);return
    run_queue()


if __name__=='__main__':main()
