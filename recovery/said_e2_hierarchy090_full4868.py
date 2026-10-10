"""One complete-state H0.9 continuation500->4868; final-only native evaluation.

No production source or scientific configuration is changed. The reviewed
resume protocol restores full state; this controller adds trajectory-wide
E2 sampler/LR checks and preserves checkpoint lineage and evidence.
"""
import hashlib
import functools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from recovery import hns_s12_2434_validation as protocol
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now, distribution
from recovery.s02_local_full import expected_lrs, stream

PROJECT = Path('/opt/data/private/lklk/SAID')
BRANCH = 'experiment/said-e2-hierarchy090-full4868-v1'
PARENT_BRANCH = 'origin/experiment/said-e2-hierarchy090-500-v1'
PARENT_COMMIT = 'f2244c4310177937fb8ddf5d5cf9172c4ca8904a'
PARENT = PROJECT/'runtime/SAID-nest-clip-v1/said-e2-hierarchy090-500-v1/E2-Hierarchy090/step500/step000500.pt'
PARENT_SHA = '1e71b433b4dd49977be701fc269f9852e023443560b23c2d88d80e111773fd90'
BASE_EXP = ROOT/'experiments/nest_clip_v1/said_e2_hierarchy090_500_v1/E2-Hierarchy090'
EXP = ROOT/'experiments/nest_clip_v1/said_e2_hierarchy090_full4868_v1'
RUN = PROJECT/'runtime/SAID-nest-clip-v1/said-e2-hierarchy090-full4868-v1'
ENTRY = 'recovery.said_e2_hierarchy090_full4868'
TARGETS = (4868,)
CODE = (ENTRY.replace('.', '/')+'.py', 'tests/test_said_e2_hierarchy090_full4868.py')
GROUPS = ('backbone', 'text_mask_and_shared_pool', 'visual_mask', 'fusion_adapter')
REFERENCE_ROOT = PROJECT/'runtime/SAID-nest-clip-v1'
REFERENCE_LOGS = (
    REFERENCE_ROOT/'hns-s12-uniform-e1-1217-v1/step1217/training/steps.jsonl',
    *(REFERENCE_ROOT/f'hns-s12-uniform-full4868-v1/step{n}/training/steps.jsonl'
      for n in (2434,3651,4868)))
ORIGINAL_WORKER = protocol.worker
ORIGINAL_IDENTITY = protocol.identity
ORIGINAL_RESUME_GATE = protocol.resume_gate


def normalize_json(value):
    """Normalize JSON representation without losing colliding object keys.

    Numeric histogram keys become strings in historical logs. Detect any
    collisions before serialization; values and fields remain exact.
    """
    def check_keys(item, path='$'):
        if isinstance(item, dict):
            seen = set()
            for key, child in item.items():
                encoded = next(iter(json.loads(json.dumps({key: None}, ensure_ascii=False, allow_nan=False))))
                if encoded in seen:
                    raise ValueError(f'JSON key collision at {path}/{encoded}')
                seen.add(encoded)
                check_keys(child, path+'/'+encoded)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):check_keys(child, path+'/'+str(index))
    check_keys(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def segment(target):
    assert target == 4868
    return RUN/'step4868'


def phase(target):
    assert target == 4868
    return protocol.local.IMAGES.parent/'formal-said-e2-hierarchy090-full4868-v1'


def predecessor(target):
    assert target == 4868
    return PARENT, 500


def frozen(cfg):
    expected = protocol.common.read(BASE_EXP/'config.json')
    assert all(cfg.get(k) == v for k,v in expected.items()), 'Frozen H0.9 configuration drift'
    assert [cfg[k] for k in protocol.common.MACRO_KEYS] == [10.,1.2,.9]
    assert cfg['view_weights'] == [1.35,1.35,.3]
    assert cfg['view_sparsity_weights'] == [5/3]*3
    assert cfg['epochs'] == 4 and cfg['batch_size'] == 256 and cfg['workers'] == 8
    assert cfg['checkpoint_encoders'] and cfg['hns_enabled'] and cfg['inclusion_max'] == 0


def identity(path, completed):
    import torch
    proof = ORIGINAL_IDENTITY(path, completed)
    p = torch.load(path, map_location='cpu', weights_only=False)
    assert p['scheduler']['last_update_index'] == completed-1
    assert p['scheduler']['optimizer_lrs'] == {g['name']:g['lr'] for g in p['optimizer']['param_groups']}
    assert tuple(g['name'] for g in p['optimizer']['param_groups']) == GROUPS
    if completed == 500:
        assert proof['sha256'] == PARENT_SHA
        assert p['hierarchy090_experiment']['lambda_hierarchy'] == .9
    else:
        lineage = p['resume_lineage']
        assert lineage['parent'] == str(PARENT) and lineage['sha256'] == PARENT_SHA
        assert lineage['first_update'] == 501
        assert p['config']['start_updates'] == 500
        assert p['hierarchy090_continuation']['lambda_hierarchy'] == .9
    assert p['config']['trial_id'] == protocol.common.read(BASE_EXP/'config.json')['trial_id']
    proof.update(full_model_adapter_AdamW_RNG_loader_sampler_present=True,
                 lambda_hierarchy=.9, initialization_not_reset=True)
    return proof


@functools.lru_cache(maxsize=1)
def reference_rows():
    data = [r for path in REFERENCE_LOGS for r in rows(path)]
    assert [r['step'] for r in data] == list(range(501,4869)), 'Reference missing/duplicate updates'
    return data


def check_row(actual, reference):
    """Compare data/LR invariants, never model-dependent outputs or pixels."""
    step = actual['step']
    assert reference['step'] == step and actual['s'] == step-1
    assert actual['epoch'] == reference['epoch'] == (step-1)//1217
    assert actual['nonfinite'] == 0 and math.isfinite(actual['loss'])
    assert actual['lambda_h'] == 1 and actual['HNS_enabled']
    assert actual['inclusion_loss'] == actual['inc_weight'] == 0
    assert [actual['macro_'+k] for k in protocol.common.MACRO_KEYS] == [10.,1.2,.9]
    assert [actual['HNS_sparse_coeff_'+v] for v in ('F','Dall','D3')] == [5/3]*3
    assert math.isclose(actual['macro_weighted_hierarchy'], .9*actual['macro_raw_hierarchy'], rel_tol=4e-6, abs_tol=2e-6)
    assert math.isclose(actual['loss'], sum(actual['macro_weighted_'+k] for k in ('align','sparse','hierarchy')), rel_tol=4e-6, abs_tol=2e-6)
    assert actual['actual_lrs'] == reference['actual_lrs'] == dict(zip(GROUPS, expected_lrs(step-1, protocol.common.read(BASE_EXP/'config.json'))))
    old = {h['rank']:h for h in reference['rank_health']}
    assert {h['rank'] for h in actual['rank_health']} == set(old) == {0,1,2,3}
    count = 0
    for h in actual['rank_health']:
        ref = old[h['rank']]
        assert h['updates'] == step-500 and h['gradients_finite']
        assert h['batch'] == ref['batch'] == (180 if step%1217 == 0 else 256)
        assert h['stream_sha256'] == ref['stream_sha256'] and h['sampling'] == ref['sampling']
        count += h['batch']
    return count


def full_stream_proof(actual, reference):
    assert [r['step'] for r in actual] == [r['step'] for r in reference] == list(range(501,4869))
    count = sum(check_row(a,b) for a,b in zip(actual,reference))
    assert count == 4*1245904-512000 == 4471616
    compact = [[a['step'], h['rank'], h['stream_sha256'], h['sampling']]
               for a in actual for h in a['rank_health']]
    digest = hashlib.sha256(json.dumps(compact, separators=(',',':'), ensure_ascii=False).encode()).hexdigest()
    return dict(passed=True,first_update=501,last_update=4868,updates_this_run=4368,
        sample_positions=count,all_sample_ID_F_Dall_D3_text_tokens_K_detail_indices_exact=True,
        all_native_LRs_exact=True,all_epoch_batch_cursors_exact=True,no_duplicate_optimizer_updates=True,
        tails_per_rank={str(n):180 for n in (1217,2434,3651,4868)},stream_sha256=digest,
        reference_logs=[dict(path=str(p),sha256=sha(p)) for p in REFERENCE_LOGS],
        comparison='Real E2 training logs; no gradient/model/pixel equality requirement')


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0,1,2,3})
    assert protocol.common.git('branch','--show-current') == BRANCH
    assert not EXP.exists() and not RUN.exists(), 'No overwrite, duplicate run or automatic retry'
    assert protocol.common.git('rev-parse',PARENT_BRANCH) == PARENT_COMMIT
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+PARENT_BRANCH.removeprefix('origin/')],cwd=ROOT,text=True).split()[0]
    assert remote == PARENT_COMMIT
    assert sha(PARENT) == PARENT_SHA and sha(STEP0) == STEP0_SHA
    proof = identity(PARENT,500)
    original = protocol.common.read(BASE_EXP/'RESULTS.json')
    assert original['completed_steps'] == 500 and original['checkpoint']['sha256'] == PARENT_SHA
    assert original['strict_export']['passed']
    for name,digest in proof['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',PARENT_COMMIT+':'+name],cwd=ROOT)).hexdigest() == digest
    ready = protocol.common.read(protocol.local.IMAGES.parent/'full-ready.json')
    assert ready['status'] == 'LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator = protocol.common.evaluator_proof()
    reference = reference_rows()
    for r in reference:
        assert r['actual_lrs'] == dict(zip(GROUPS, expected_lrs(r['step']-1, protocol.common.read(BASE_EXP/'config.json'))))
    refs = protocol.common.read(ROOT/'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1/step4868/RESULTS.json')
    q = protocol.common.quality(refs)
    for key,value in dict(Score5=73.812186642,J_long3=78.193644403,J_long=87.055002253,Short4=67.240,Urban_I2T=93.9,Urban_T2I=92.9).items():
        assert math.isclose(q[key],value,abs_tol=1e-5), (key,q[key])
    EXP.mkdir(parents=True); segment(4868).mkdir(parents=True)
    (EXP/'step4868').mkdir()
    dump(EXP/'config.json',protocol.common.read(BASE_EXP/'config.json'))
    dump(EXP/'RESUME_PROVENANCE.json',dict(**proof,parent_commit=PARENT_COMMIT,parent_branch=PARENT_BRANCH,
        production_source_changes=[],resume_only=True,common0_used_only_for_native_identity_not_resume=True))
    dump(segment(4868)/'resume-reference.json',protocol.resume_reference(500))
    dump(EXP/'REFERENCES.json',dict(models={'E2_Uniform':{'4868':refs}},
        prior_hierarchy090_500=original,evaluator_sha256=evaluator,
        reference_results_sha256=sha(ROOT/'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1/step4868/RESULTS.json')))
    protected=[PARENT,Path(refs['checkpoint']['path']),Path(refs['bare']['path'])]
    expected=[PARENT_SHA,refs['checkpoint']['sha256'],refs['bare']['sha256']]
    protected_proof=[dict(path=str(p),sha256=sha(p)) for p in protected]
    assert [v['sha256'] for v in protected_proof]==expected
    dump(EXP/'PROTECTED_ARTIFACTS.json',dict(passed=True,files=protected_proof))
    paths=protocol.local.path_proof();dump(RUN/'local-path-proof-5000.json',paths)
    assert paths['passed']
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=paths['count'],image_root=str(protocol.local.IMAGES),NFS_fallback=False))
    dump(EXP/'PLAN.json',dict(initial_updates=500,stop_updates=4868,updates_this_run=4368,horizon=4868,
        lambda_hierarchy=.9,only_arm='E2-Hierarchy090',intermediate_checkpoints=[1217,2434,3651],
        final_only_five_set_evaluation=True,public_test_selection=False,other_arms=False,
        automatic_fifth_epoch=False,automatic_retry=False,complete_state_restore=True))
    tests=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q',CODE[1],'tests/test_said_e2_hierarchy090.py','tests/test_hns_s12_2434_validation.py']
    with (RUN/'cpu-tests.log').open('w') as handle:
        check=subprocess.run(tests,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    dump(EXP/'CPU_TESTS.json',dict(passed=check.returncode==0,returncode=check.returncode,command=tests,summary=(RUN/'cpu-tests.log').read_text(),utc=now()))
    assert check.returncode == 0
    protocol.state('PREPARED')


def worker(target):
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from recovery import nested_d3_local_search as search
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import training_phase_timing as timing
    references={r['step']:r for r in reference_rows()}
    context={}
    old_opt,old_rates,old_observe,old_save=trainer.build_optimizer,trainer.optimizer_learning_rates,search.observe_selection,trainer.atomic_save
    def optimizer(module):
        context['optimizer']=old_opt(module);return context['optimizer']
    def rates(module,completed,horizon):
        context.update(module=module,step=completed+1)
        values=old_rates(module,completed,horizon)
        assert list(values) == expected_lrs(completed,protocol.common.read(BASE_EXP/'config.json'))
        assert dict(zip(GROUPS,values)) == references[completed+1]['actual_lrs']
        return values
    def observe(batch):
        step=context['step'];opt=context['optimizer'];module=context['module']
        assert {int(v['step']) for v in opt.state.values()} == {step}, 'Optimizer counter discontinuity'
        value=old_observe(batch)
        ref=next(h for h in references[step]['rank_health'] if h['rank']==dist.get_rank())
        try:
            assert normalize_json(value) == normalize_json(ref['sampling']), 'Full-stage K/detail/token sampling mismatch'
        except (AssertionError, ValueError, TypeError) as error:
            dump(segment(target)/f'sampling-failure-{step}-rank{dist.get_rank()}.json',
                 dict(step=step,rank=dist.get_rank(),error=repr(error),
                      actual_repr=repr(value),reference=ref['sampling'],passed=False))
            raise
        if step<=505 or step%1217==0:
            difference=trainer.parameter_agreement(module)
            receipt=dict(passed=difference==0,step=step,rank=dist.get_rank(),optimizer_counters=[step],
                updates_this_run=step-500,parameter_difference=difference,gradients_finite=True,
                actual_lrs={g['name']:g['lr'] for g in opt.param_groups})
            dump(segment(target)/f'update-{step}-rank{dist.get_rank()}.json',receipt)
            assert receipt['passed']
        return value
    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            payload['hierarchy090_continuation']=dict(lambda_hierarchy=.9,first_update=501,stop_update=4868,
                updates_this_run=payload['completed_steps']-500,horizon=4868,parent_sha256=PARENT_SHA,
                controller_sha256=sha(Path(__file__)),production_source_changes=[])
        return old_save(payload,path)
    class ReferenceIterator(timing.TimedIterator):
        def __init__(self,iterator,recorder):
            super().__init__(iterator,recorder);self.audit_position=0;self.audit_epoch=iterator._dataset.epoch
        def __next__(self):
            value=super().__next__();self.audit_position+=1
            step=self.audit_epoch*1217+self.audit_position
            if step>500:
                ref=next(h for h in references[step]['rank_health'] if h['rank']==self.recorder.rank)
                assert len(value['sample_id']) == ref['batch']
                assert stream(value) == ref['stream_sha256'], 'Pre-update full-stage sample/text/token mismatch'
            return value
    trainer.build_optimizer,trainer.optimizer_learning_rates,trainer.atomic_save=optimizer,rates,save
    search.observe_selection=observe;timing.TimedIterator=ReferenceIterator
    ORIGINAL_WORKER(target)


def resume_gate(target):
    receipt=ORIGINAL_RESUME_GATE(target)
    updates=[protocol.common.read(segment(target)/f'update-{s}-rank{r}.json') for s in range(501,506) for r in range(4)]
    assert all(v['passed'] and v['optimizer_counters']==[v['step']] for v in updates)
    receipt.update(post_update_checks=updates,no_repeated_first500_updates=True)
    dump(EXP/'step4868/RESUME_GATE.json',receipt)
    return receipt


def report(target,supervisor,result):
    from recovery.nested_d3_local_search_evidence import diagnostics
    from recovery import nested_d3_local_search as search
    runtime=segment(target);dest=EXP/'step4868';records=rows(runtime/'training/steps.jsonl')
    proof=full_stream_proof(records,reference_rows());dump(EXP/'FULL_STREAM_AND_LR_PROOF.json',proof)
    search.ARMS['H090']=dict(mode='nested_detail_d3',weights=[1.35,1.35,.3])
    diag,masks=diagnostics(records,'H090');diag['last50_steps']=[r['step'] for r in records[-50:]]
    keys=[k for k in records[0] if k.startswith(('HNS_','macro_')) or k in ('loss','lambda_h','V_DF_hard','V_3D_hard')]
    means={k:statistics.fmean(float(r[k]) for r in records[-50:]) for k in keys}
    diag['last50_loss_components']=means
    baseline=reference_rows()[-50:]
    diag['last50_delta_vs_E2']={k:means[k]-statistics.fmean(float(r[k]) for r in baseline) for k in keys}
    masks.update(keep_ratios_valid_population={v:means['HNS_'+v+'_keep'] for v in ('F','Dall','D3')},
        density_order_F_Dall_D3=means['HNS_F_keep']>=means['HNS_Dall_keep']>=means['HNS_D3_keep'],
        valid_population_hierarchy={k:means[k] for k in keys if 'violation' in k or 'IoU' in k or 'equality' in k})
    gradient=protocol.common.read(dest/'GRADIENT_AUDIT.json');assert gradient['passed'] and gradient['checkpoint']['unchanged']
    baseline_gradient=protocol.common.read(ROOT/'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1/step4868/GRADIENT_AUDIT.json')
    assert gradient['sample_ids_sha256']==baseline_gradient['sample_ids_sha256']
    diag['matched_gradient_probe_sample_ids_sha256']=gradient['sample_ids_sha256']
    diag['gradient_group_delta_vs_E2']={name:{k:value-baseline_gradient['group_diagnostics'][name][k]
        for k,value in values.items() if isinstance(value,(int,float)) and not isinstance(value,bool)
        and isinstance(baseline_gradient['group_diagnostics'][name].get(k),(int,float))}
        for name,values in gradient['group_diagnostics'].items()}
    accept=protocol.common.read(runtime/'training/acceptance.json')
    assert accept['passed'] and all(r['completed_updates']==4868 and r['updates_this_run']==4368 and r['max_parameter_difference_from_rank0']==0 and r['final_nccl_all_reduce']==10 for r in accept['ranks'])
    cycles=rows(runtime/'training/cycle_timing.jsonl')
    systems=[r['system'] for r in rows(runtime/'resource-telemetry.jsonl')]
    stats=dict(full_cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),
        GPU_peak_GiB={str(rank):max(h['peak_allocated_gib'] for row in records for h in row['rank_health'] if h['rank']==rank) for rank in range(4)},
        oom_kill=max(s['memory_events'].get('oom_kill',0) for s in systems),commands=supervisor.commands,
        local_only=True,DDP_NCCL_passed=True,updates_this_run=4368)
    assert stats['oom_kill']==0
    log=(runtime/'train.log').read_text(errors='replace')
    assert not any(message in log for message in ('Image failure sample=','Missing local sample=','Input/output error'))
    identities={str(n):identity(runtime/f'training/step{n:06d}.pt',n) for n in (1217,2434,3651,4868)}
    assert sha(PARENT)==PARENT_SHA
    protected=protocol.common.read(EXP/'PROTECTED_ARTIFACTS.json')
    assert all(sha(v['path'])==v['sha256'] for v in protected['files'])
    dump(dest/'PROTECTED_ARTIFACTS_UNCHANGED.json',dict(passed=True,files=protected['files'],utc=now()))
    for name,value in [('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),('RUNTIME_STATS',stats),('CHECKPOINT_IDENTITIES',identities)]:dump(dest/(name+'.json'),value)
    dump(dest/'VALIDATION.json',dict(passed=True,acceptance=accept,resume_gate=resume_gate(target),
        full_stream=proof,strict_export=result['strict_export'],parent500_immutable=True,production_sources_exact=True))


def combined(completed):
    assert completed==[4868]
    result=protocol.common.read(EXP/'step4868/RESULTS.json')
    baseline=protocol.common.read(EXP/'REFERENCES.json')['models']['E2_Uniform']['4868']
    comp=result['comparisons']['E2_Uniform']
    models={'E2-Uniform@4868':baseline,'E2-Hierarchy090@4868':result}
    quality={name:dict(protocol.common.quality(r)) for name,r in models.items()}
    for q in quality.values():q['Urban_Mean']=(q['Urban_I2T']+q['Urban_T2I'])/2
    delta={k:quality['E2-Hierarchy090@4868'][k]-quality['E2-Uniform@4868'][k] for k in quality['E2-Uniform@4868']}
    dump(EXP/'HIERARCHY090_FULL_NATIVE_RESULTS.json',dict(models=models,qualities=quality,delta_pp=delta,
        recall_delta_pp=comp['recall_delta_pp'],scientific_status='EXPLORATORY',stop_updates=4868))
    lines=['# E2-Hierarchy090: complete-state continuation500 to4868','',
        'One frozen arm; lambda_hierarchy=0.9.4368 new updates from the original complete500 checkpoint. Native horizon4868, ramp200, AdamW moments, RNG and data cursor restored. No intermediate public evaluation or test-based checkpoint selection.',
        '', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name,q in quality.items():
        lines.append('| '+name+' | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {q["Urban_I2T"]:.3f}/{q["Urban_T2I"]:.3f} | {q["Urban_Mean"]:.3f} |')
    lines+=['','Delta(pp): `'+json.dumps(delta)+'`.','',
        '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) | ΔI2T R1/R5/R10(pp) | ΔT2I R1/R5/R10(pp) |','|---|---|---|---|---|']
    for ds,m in result['metrics'].items():
        vals=[' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
        vals+=[' / '.join(f'{comp["recall_delta_pp"][ds][dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
        lines.append('| '+ds+' | '+' | '.join(vals)+' |')
    diagnostics=protocol.common.read(EXP/'step4868/TRAINING_DIAGNOSTICS.json')
    gradient=protocol.common.read(EXP/'step4868/GRADIENT_AUDIT.json')
    lines+=['',f'Both long metrics improve: {delta["J_long3"]>0 and delta["J_long"]>0}. Both Urban directions improve: {delta["Urban_I2T"]>0 and delta["Urban_T2I"]>0}.',
        'Last50 mask/loss baseline deltas: `'+json.dumps(diagnostics['last50_delta_vs_E2'])+'`.',
        'Gradient group norms/cosines: `'+json.dumps(gradient['group_diagnostics'])+'`.',
        'Matched gradient probe differences against E2: `'+json.dumps(diagnostics['gradient_group_delta_vs_E2'])+'`.',
        'All4471616 continued sample positions match real E2 logs including four180/rank epoch tails; all4368 native LR/counters are continuous. Checkpoint1217/2434/3651/4868 full identities are preserved. Production source manifest is unchanged.',
        'Strict bare export has exact native image/text outputs; normalized native retrieval only, no mask/rerank/ensemble/TTA. Parent and final training weights remain immutable through evaluation.',
        'Exploratory single-seed comparison on repeatedly observed public benchmarks. Independent validation NOT_ESTABLISHED;0.1pp Urban is one query. Mask structure and gradient cosine alone do not prove retrieval benefit; no unbiased SOTA or significance claim.',
        'STOP4868. No second arm, H1.1, extra seed or fifth epoch. Large weights/raw logs/data remain server-local.']
    (EXP/'HIERARCHY090_FULL4868_REPORT.md').write_text('\n'.join(lines)+'\n')


def configure():
    protocol.BRANCH=BRANCH;protocol.MOTHER=PARENT_BRANCH;protocol.PARENT=PARENT;protocol.PARENT_SHA=PARENT_SHA
    protocol.EXP=EXP;protocol.RUN=RUN;protocol.ENTRY=ENTRY;protocol.TARGETS=TARGETS;protocol.CODE=CODE
    protocol.segment=segment;protocol.phase=phase;protocol.predecessor=predecessor
    protocol.frozen=frozen;protocol.identity=identity;protocol.prepare=prepare;protocol.worker=worker
    protocol.resume_gate=resume_gate;protocol.report=report;protocol.combined=combined


def main():
    configure();protocol.main()


if __name__=='__main__':main()
