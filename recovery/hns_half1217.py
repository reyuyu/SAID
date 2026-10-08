"""Pinned HNS500 intervention: halve only hierarchy on updates501..1217.

The opt-in worker schedule replaces the two imported schedule references in its
own process. Production source/default HNS, sampler, optimizer and exporter are
unchanged. Full-state restoration is the existing reviewed HNS continuation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading

from recovery import hns_full as original
from recovery import s02_local_full as full, s02_local500 as local
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, PYTHON, dump, now, rows, sha, distribution
from recovery.s02_full_stage import LOCAL, IMAGES

ENTRY = 'recovery.hns_half1217'
BRANCH = 'experiment/nested-d3-hns-half1217-v1'
EXP = ROOT/'experiments/nest_clip_v1/nested_d3_hns_half1217_v1'
RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns-half1217-v1'
TRAIN = RUN/'step1217'
REVIEW = RUN/'reviewed'
PHASE = LOCAL/'formal-hns-half1217-v1-phase'
CONFIG = EXP/'config.json'
MAIN_LOG = RUN/'runner.log'
PARENT, PARENT_SHA = original.PARENT, original.PARENT_SHA
HNS_RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1'
BALANCED_RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced-full-20261006'
BASELINES = {
    'HNS-v1': (HNS_RUN/'step4868/step001217.pt', 'a9e8c1e5daa0220c5df24dad015a47aa182886da3c01995000b3259c6bab6873'),
    'D3 Balanced': (BALANCED_RUN/'step4868/step001217.pt', 'b5e59502d0c7dbec8687c6c4cff944b19e72d8cfed181f55732d0add0c31a789'),
}
SCORE_KEYS = original.SCORE_KEYS
CODE = ('recovery/hns_half1217.py', 'tests/test_hns_half1217.py', 'recovery/check_stage500_publish.py')
STATIC = ('config.json', 'PLAN.json', 'RESUME_PROVENANCE.json', 'BASELINE_PROVENANCE.json', 'CPU_TESTS.json')
REPORTS = ('REPORT.md', 'RESULTS.json', 'comparison1217.json', 'RESUME_GATE.json',
           'TRAINING_DIAGNOSTICS.json', 'MASK_HIERARCHY_AUDIT.json', 'RUNTIME_STATS.json',
           'EXPORT_AUDIT.json', 'VALIDATION.json', 'COMMANDS.json')
ORIGINAL_FROZEN = original.frozen_config


def read(p):
    return json.loads(Path(p).read_text())


def half_weight(completed):
    assert isinstance(completed, int) and completed >= 0
    # Trainer passes completed BEFORE the current update: 500 means update501.
    return min(1., completed/200.) if completed < 500 else .5


def install_schedule():
    from model import hard_nested_sparsity as terms, balanced_hparam_search as model
    terms.hierarchy_weight = model.hierarchy_weight = half_weight


def frozen_config(cfg):
    ORIGINAL_FROZEN(cfg)
    assert not cfg.get('hns_detach_child', False), 'Half must retain child contraction'
    assert cfg.get('hns_half_after500', False) in (False, True)


def validate_resume(previous, current, proof, native):
    frozen_config(current)
    assert current['hns_half_after500'] is True
    assert not previous['config'].get('hns_half_after500', False)
    assert current['resume'] == str(PARENT) and current['max_updates'] == 1217
    assert current['horizon'] == 4868
    assert previous['completed_steps'] == previous['global_step'] == 500
    assert previous['trajectory_root'] == str(original.BASE_RUN)
    assert previous['data_cursor'] == dict(next_epoch=0, next_batch=500)
    old, new = previous['config']['code_sha256'], current['code_sha256']
    assert old == proof['predecessor_sources'] and new == proof['current_sources']
    changed = {p for p in old if old[p] != new[p]}
    assert changed == original.MIGRATED == set(proof['audited_equivalent_changes'])
    assert all(proof['audited_equivalent_changes'][p] == [old[p], new[p]] for p in changed)
    # Explicitly allow ONLY the pinned equivalent code migration and this
    # process-local post500 schedule intervention. Do not modify stored parent.
    normalized = dict(previous, config=dict(previous['config'], code_sha256=new))
    return native(normalized, current)


def configure():
    for key in ('EXP', 'RUN', 'TRAIN', 'REVIEW', 'PHASE', 'CONFIG', 'MAIN_LOG', 'ENTRY', 'BRANCH'):
        setattr(original, key, globals()[key])
    original.frozen_config = frozen_config
    original.validate_resume = validate_resume
    original.configure()


def checkpoint_identity(path, expected=None, step=1217):
    import torch
    actual = sha(path)
    if expected: assert actual == expected
    p = torch.load(path, map_location='cpu', weights_only=False)
    assert p['completed_steps'] == p['global_step'] == p['scheduler']['completed_steps'] == step
    assert p['scheduler_horizon'] == p['scheduler']['horizon'] == 4868
    assert p['data_cursor'] == dict(next_epoch=step//1217, next_batch=step%1217)
    assert len(p['rng_per_rank']) == 4 and p['adapter'] and p['optimizer']['state']
    assert {int(s['step']) for s in p['optimizer']['state'].values()} == {step}
    for rng in p['rng_per_rank']:
        assert all(k in rng for k in ('python', 'numpy', 'cpu', 'cuda', 'loader_generator'))
    if step == 1217 and path == TRAIN/'step001217.pt':
        frozen_config(p['config'])
        assert p['config']['hns_half_after500'] and p['config']['max_updates'] == 1217
        assert p['config']['resume'] == str(PARENT) and p['config']['start_updates'] == 500
        assert p['config']['parent_checkpoint_sha256'] == PARENT_SHA
    return dict(passed=True, path=str(path), sha256=actual, bytes=path.stat().st_size,
        step=step, scheduler_horizon=4868, cursor=p['data_cursor'], rng_ranks=4,
        optimizer_counters=[step], config=p['config'], uploaded=False)


def prepare():
    configure()
    assert not RUN.exists() and not PHASE.exists()
    EXP.mkdir(exist_ok=True); REVIEW.mkdir(parents=True)
    cfg = read(original.BASE_EXP/'config.json')
    cfg['hns_half_after500'] = True
    dump(CONFIG, cfg)
    proof = original.identity()
    proof.update(user_authorization='HNS-v1 evaluated500 -> HNS-Half1217, exactly717 new updates',
        intervention='Only process-local hierarchy_weight becomes .5 from completed=500 (update501)',
        scaler='None in original: fp32 parameters, bf16 encoder autocast, no GradScaler')
    dump(EXP/'RESUME_PROVENANCE.json', proof)
    refs = original.reference()
    prior = read(HNS_RUN/'reviewed/resume-stream-reference.json')
    assert refs['records'] == prior['records']
    for item in refs['records'].values():
        old = read(HNS_RUN/f"reviewed/resume-batch-{item['step']}-rank{item['rank']}.json")
        assert old['passed'] and old['stream_sha256'] == item['stream_sha256']
        lr = read(HNS_RUN/f"reviewed/resume-lr-{item['step']}-rank{item['rank']}.json")
        assert lr['passed'] and lr['lrs'] == full.expected_lrs(item['step']-1, cfg)
    dump(REVIEW/'resume-stream-reference.json', refs)
    dump(REVIEW/'prelaunch-local-path-proof-5000.json', local.path_proof())
    assert read(LOCAL/'full-ready.json')['verification']['passed']
    baselines = {n:checkpoint_identity(p, digest) for n,(p,digest) in BASELINES.items()}
    branches = {'HNS-v1':'experiment/nested-d3-hns-full-v1', 'D3 Balanced':'experiment/nested-detail-d3-balanced-full-v1'}
    for n,b in branches.items(): baselines[n]['remote_branch_commit'] = original.git('rev-parse', 'origin/'+b)
    dump(EXP/'BASELINE_PROVENANCE.json', baselines)
    dump(EXP/'PLAN.json', dict(updates=[501,1217], new_updates=717, horizon=4868,
        parent=str(PARENT), parent_sha256=PARENT_SHA, local_root=str(IMAGES), NFS_fallback=False,
        sole_change='hierarchy surcharge multiplier1 -> .5 for optimizer updates501..1217',
        boundary='completed is before current update:499->update500 weight1;500->update501 weight.5',
        alignment=[1.35,1.35,.30], sparse=[1,2,2], beta=[2,2], detach_child=False, old_soft_inclusion=0,
        intervention_statement='This is a branch-from-step500 intervention experiment, not an independently trained 0–1217 run.',
        classification=dict(material_long_decline_pp=.05, near_equal_Score5_pp=.05,
            primary='retrieval', violation_veto=False, density_alone_healthy=False),
        evaluation='Re-evaluate all three1217 checkpoints using identical current strict native evaluator',
        automatic_full=False, automatic_other_experiments=False,
        cache='Ephemeral /root Docker overlay; NFS persistent source retained; no copy or full audit'))
    state('PREPARED')


def state(status, **extra):
    dump(EXP/'STATE.json', dict(status=status, runner_pid=os.getpid(), updated_utc=now(),
        stop_updates=1217, automatic_other_experiments=False, **extra))


def worker():
    configure()
    assert read(CONFIG)['hns_half_after500'] is True
    install_schedule()
    original.worker()


def train_command():
    return [str(ROOT/'.venv/bin/torchrun'), '--standalone', '--nnodes=1', '--nproc-per-node=4',
        '--max-restarts=0', '-m', '--', ENTRY, '--worker', '--config', str(CONFIG),
        '--init-state', str(STEP0), '--resume', str(PARENT), '--index-dir', str(original.INDEX),
        '--image-root', str(IMAGES), '--output-dir', str(TRAIN), '--run-type', 'formal', '--max-updates', '1217']


class Supervisor(full.Supervisor):
    def evaluate1217(self, name, checkpoint, device='cuda:0'):
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores
        ASSETS = pipeline.ASSETS
        destination = RUN/'evaluations'/name.replace(' ', '_')
        destination.mkdir(parents=True, exist_ok=True)
        digest = sha(checkpoint); bare = destination/'student_step1217.pt'
        self.execute(name+'-export', [PYTHON,'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),
            '--output',str(bare),'--expect-updates','1217'])
        self.execute('verify-'+name,[PYTHON,'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),
            '--bare',str(bare),'--output',str(destination/'export-check.json'),
            '--index-dir',str(original.INDEX),'--image-root',str(IMAGES)])
        for dataset,folder in (('coco',ASSETS/'evaluation/coco/val2017'),('urban',ASSETS/'evaluation/Urban1k/Urban1k')):
            self.execute(name+'-'+dataset,[PYTHON,'-m','tools.eval_nest_native','--checkpoint',str(bare),
                '--dataset',dataset,'--root',str(folder),'--device',device,'--batch-size','64',
                '--output',str(destination/(dataset+'_native.json'))])
        bench=ASSETS/'retrieval_benchmarks'
        for ds,manifest,folder in (('flickr_test1k','flickr30k_test1k.jsonl','flickr30k'),
            ('docci','docci_test.jsonl','docci'),('long_dci','long_dci_reconstructed.jsonl','dci')):
            self.execute(name+'-'+ds,[PYTHON,'-m','experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real',
                '--checkpoint',str(bare),'--device',device,'--batch-size','64','--output-dir',str(destination/ds),
                f"{ds}:{bench/'manifests'/manifest}:{bench/folder/'images'}"])
        assert sha(checkpoint)==digest
        metrics,raw,sources=native_metrics(destination)
        calculated=scores(metrics)
        percent={k:v*100 for k,v in calculated.items()}
        percent['Score5']=percent['Score5_R1']
        percent['Short4']=100*statistics.fmean(metrics[ds][dr]['R@1'] for ds in ('COCO','Flickr30k-test1k') for dr in ('I2T','T2I'))
        export=read(destination/'export-check.json'); assert export['passed'] and export['checkpoint_sha256']==digest
        for ds,p in sources.items():dump(EXP/'evaluations'/name.replace(' ','_')/(ds+'.json'),read(p))
        value=dict(step=1217,metrics=metrics,scores_percent=percent,strict_export=export,
            checkpoint=str(checkpoint),checkpoint_sha256=digest,evaluation_checkpoint_immutable=True,
            bare=str(bare),evaluation_device=device,native_full_caption_only=True,finished_utc=now())
        dump(REVIEW/(name.replace(' ','_')+'-results.json'),value)
        return value

    def run(self):
        from train.train_nested_semantic_mask import code_manifest
        from recovery.resource_stall_v2 import system_snapshot
        configure(); self.train=TRAIN
        assert not TRAIN.exists() and not PHASE.exists()
        assert sha(PARENT)==PARENT_SHA and sha(STEP0)==STEP0_SHA
        assert read(EXP/'CPU_TESTS.json')['passed']
        assert read(EXP/'RESUME_PROVENANCE.json')['current_sources']==code_manifest()
        assert read(LOCAL/'full-ready.json')['verification']['passed']
        assert not system_snapshot()['memory_events'].get('oom_kill',0)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        forbidden = {'train.train_nested_semantic_mask','recovery.s02_full_stage','recovery.s02_stage500',
                     'recovery.s02_local500','recovery.s02_nfs500'}
        for p in Path('/proc').iterdir():
            if p.name.isdigit() and int(p.name) != os.getpid():
                try: args = (p/'cmdline').read_bytes().split(b'\0')
                except OSError: continue
                assert not any(a.decode(errors='replace') in forbidden for a in args), 'Concurrent training/copy/audit'
        PHASE.mkdir()
        sources={p:sha(ROOT/p) for p in CODE}; sources.update(code_manifest())
        for p in ('recovery/hns_full.py','recovery/s02_local_full.py','recovery/s02_local500.py',
            'recovery/s02_full_local_data.py','recovery/nested_d3_local_search.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py',
            'tools/nest_clip.py','tools/eval_nest_native.py',
            'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'):
            sources[p]=sha(ROOT/p)
        dump(RUN/'launch-provenance.json',dict(started_utc=self.started,git_head=original.git('rev-parse','HEAD'),
            source_sha256=sources,stop_updates=1217,local_image_root=str(IMAGES),NFS_fallback=False))
        try:
            state('TRAINING_501_TO_1217')
            self.execute('train1217',train_command(),training=True)
            self.acceptance=read(TRAIN/'acceptance.json')
            assert self.acceptance['passed'] and all(r['completed_updates']==1217 and r['updates_this_run']==717
                and r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            original.resume_gate()
            proof=checkpoint_identity(TRAIN/'step001217.pt')
            dump(REVIEW/'final-checkpoint-proof.json',proof)
            stream_audit()
            state('MATCHED_THREE_WAY_EVALUATION',completed_steps=1217)
            # Training has stopped. Each independent model uses one identical
            # A100 for the same evaluator/batch64 protocol. Serialize common
            # control receipts; raw logs and export/output directories differ.
            lock=threading.Lock(); native_dump=full.dump
            def locked_dump(path,value):
                with lock: native_dump(path,value)
            full.dump=locked_dump
            tasks=[('HNS-Half',TRAIN/'step001217.pt'),*[(n,p) for n,(p,_) in BASELINES.items()]]
            try:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pending=[(name,pool.submit(self.evaluate1217,name,p,f'cuda:{i}')) for i,(name,p) in enumerate(tasks)]
                    self.result={name:f.result() for name,f in pending}
            finally:
                full.dump=native_dump
        except BaseException as e:
            self.error=type(e).__name__+': '+str(e); print(self.error,flush=True)
            raise
        finally:
            dump(RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
                started_utc=self.started,ended_utc=now()))


def stream_audit():
    actual=rows(TRAIN/'steps.jsonl')
    reference={r['step']:r for r in rows(HNS_RUN/'step4868/steps.jsonl')}
    assert [r['step'] for r in actual]==list(range(501,1218))
    count=0
    for row in actual:
        ref=reference[row['step']]
        assert row['epoch']==ref['epoch']==0 and row['s']==ref['s']==row['step']-1
        assert row['actual_lrs']==ref['actual_lrs'] and row['lambda_h']==.5
        assert not row['HNS_detach_child'] and row['inc_weight']==row['inclusion_loss']==0
        assert math.isfinite(row['loss']) and row['nonfinite']==0
        for health in row['rank_health']:
            old=next(h for h in ref['rank_health'] if h['rank']==health['rank'])
            assert health['sampling']==old['sampling'], 'Actual IDs/F/Dall/D3/tokens/indices trajectory drift'
            assert health['gradients_finite'] and health['updates']==row['step']-500
            count+=health['batch']
    assert count==716*1024+720
    result=dict(passed=True,new_updates=717,records=count,all_actual_sampling_payloads_exact_HNS_full=True,
        all_LRs_exact=True,all_lambda_h_point5=True,original_sparsity_and_alignment_unchanged=True)
    dump(REVIEW/'stream-proof.json',result)
    return result


def delta(a,b):
    return dict(scores_delta_pp={k:a['scores_percent'][k]-b['scores_percent'][k] for k in SCORE_KEYS},
        recall_delta_pp={ds:{dr:{k:100*(v-b['metrics'][ds][dr][k]) for k,v in rec.items()}
            for dr,rec in dirs.items()} for ds,dirs in a['metrics'].items()})


def classify(d,short_benefit):
    if d['Score5']<0 and d['J_long3']<0 and d['J_long']<0: return 'NEGATIVE'
    if d['Score5']>0 and d['J_long3']>=-.05 and d['J_long']>=-.05: return 'PROMISING_FOR_FULL'
    if abs(d['Score5'])<.05 and short_benefit: return 'MIXED'
    return 'MIXED'


def diagnostic(records):
    from recovery.nested_detail_d3_balanced_full_evidence import view_means
    means={k:statistics.fmean(r[k] for r in records) for k,v in records[0].items()
        if isinstance(v,(int,float)) and (k.startswith(('F_','O_','E_','HNS_','V_')) or
            k in ('loss','alignment','sparsity','inclusion_loss','inc_weight','lambda_h'))}
    means['unweighted_hierarchy_surcharge']=statistics.fmean((2*r['V_DF_hard']+2*r['V_3D_hard'])/3 for r in records)
    return dict(scalars=means,views=view_means(records))


def report():
    saved=read(RUN/'supervisor-result.json'); assert not saved['error']
    results=saved['result']; current=results['HNS-Half']
    comparisons={n:delta(current,results[n]) for n in BASELINES}
    d=comparisons['HNS-v1']['scores_delta_pp'];status=classify(d,d['Short4']>0)
    continuation=rows(TRAIN/'steps.jsonl'); fullrows=rows(HNS_RUN/'step4868/steps.jsonl')
    initial=rows(original.BASE_RUN/'step500/steps.jsonl')[-50:]
    diag=dict(points={str(s):diagnostic([next(r for r in continuation if r['step']==s)]) for s in (501,600,800,1000,1217)},
        last50=diagnostic(continuation[-50:]),HNS_v1_last50_1217=diagnostic(fullrows[1167:1217]),
        parent_last50_500=diagnostic(initial),
        gradient_groups_last50={k:statistics.fmean(h['gradient_norms'][k] for r in continuation[-50:] for h in r['rank_health'])
            for k in continuation[-1]['rank_health'][0]['gradient_norms']})
    old=diag['parent_last50_500']['scalars']
    changes={n:{k:v['scalars'][k]-old[k] for k in old if k.startswith(('HNS_','V_')) and k in v['scalars']}
        for n,v in [('HNS-Half',diag['last50']),('HNS-v1',diag['HNS_v1_last50_1217'])]}
    mask_keys=[k for k in old if any(t in k for t in ('keep','gap','IoU','violation','equality','surcharge'))]
    masks=dict(parent500={k:old[k] for k in mask_keys},
        final1217={n:{k:v['scalars'][k] for k in mask_keys} for n,v in
            [('HNS-Half',diag['last50']),('HNS-v1',diag['HNS_v1_last50_1217'])]},delta500_to1217=changes,
        interpretation='Density/violation alone does not prove healthy masks; retrieval primary')
    cycles=rows(TRAIN/'cycle_timing.jsonl'); assert len(cycles)==717
    phases={r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
    assert all([r['step'] for r in rec]==list(range(501,1218)) for rec in phases.values())
    systems=[r['system'] for r in rows(RUN/'full-resource-telemetry.jsonl') if r['command']=='train1217']
    waits=[max(phases[r][i]['data_wait_s'] for r in range(4)) for i in range(717)]
    inventory=list(RUN.rglob('*.log'))+list(RUN.rglob('*.jsonl'))+list(RUN.rglob('*.pt'))+list(PHASE.glob('*.jsonl'))
    stats=dict(full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        data_wait_seconds=distribution(waits),steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),
        steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        GPU_peak_GiB={str(r):max(h['peak_allocated_gib'] for row in continuation for h in row['rank_health'] if h['rank']==r) for r in range(4)},
        peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),
        peak_file_cache_bytes=max(s['file'] for s in systems),
        oom_kill=max(s['memory_events'].get('oom_kill',0) for s in systems),
        training_io_error_count=0,pod_supervisor_anomaly=False,scope='717 new updates; original500 not repeated',
        PSI_scope='host /proc/pressure, cgroupv1',memory_interpretation='Includes file cache; not RSS',
        raw_local_inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
            time_range_utc=[saved['started_utc'],saved['ended_utc']]) for p in inventory if p != MAIN_LOG])
    for kind in ('io_PSI','memory_PSI'):
        stats[kind]={v:distribution([s[kind][v]['avg10'] for s in systems if v in s[kind]]) for v in ('some','full')}
    pathproof=[r for p in PHASE.glob('image-paths-*.jsonl') for r in rows(p)]
    assert pathproof and all(Path(r['actual_path']).is_relative_to(IMAGES) and not r['NFS_fallback'] for r in pathproof)
    stats['local_path_proof']=dict(passed=True,count=len(pathproof),NFS_fallback=False)
    assert stats['oom_kill']==0
    result=dict(status=status,completed_steps=1217,new_updates=717,models=results,comparisons=comparisons,
        hypothesis_supported=status=='PROMISING_FOR_FULL',recommend_full=status=='PROMISING_FOR_FULL',
        no_post1217_training=True,no_additional_experiments=True,stream_proof=stream_audit(),
        intervention_statement=read(EXP/'PLAN.json')['intervention_statement'])
    dump(EXP/'RESULTS.json',result);dump(EXP/'comparison1217.json',dict(models=results,comparisons=comparisons))
    dump(EXP/'TRAINING_DIAGNOSTICS.json',diag);dump(EXP/'MASK_HIERARCHY_AUDIT.json',masks)
    dump(EXP/'RUNTIME_STATS.json',stats);dump(EXP/'EXPORT_AUDIT.json',{n:r['strict_export'] for n,r in results.items()})
    dump(EXP/'COMMANDS.json',read(RUN/'full-commands.json'))
    dump(EXP/'VALIDATION.json',dict(passed=True,resume=original.resume_gate(),stream=result['stream_proof'],
        checkpoint=read(REVIEW/'final-checkpoint-proof.json'),all_three_checkpoint_immutable=True))
    lines=['# HNS-Half1217','',result['intervention_statement'],'',f'Status: `{status}`. Exactly717 new updates501–1217; no additional training.',
        'Only hierarchy surcharge is halved. Alignment/sparsity/beta2/2/no-SG/optimizer/RNG/sampler/horizon4868 frozen.',
        'Schedule uses completed BEFORE update:completed499/update500=1;completed500/update501=.5.',
        'All three checkpoints re-exported and evaluated with identical evaluator sources, splits, batch64 and native normalized image/full-caption embeddings. No masks/gates/rerank/ensemble/TTA.',
        '', '| Model @1217 | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    for n,r in results.items():lines.append('| '+n+' | '+' | '.join(f'{r["scores_percent"][k]:.6f}' for k in SCORE_KEYS)+' |')
    for n,r in results.items():
        lines.extend(['',f'## {n}','', '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|'])
        for ds,m in r['metrics'].items():lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    lines.extend(['','All aggregate and directional R1/R5/R10 deltas:comparison1217.json.',
        'Deltas vs HNS-v1: `'+json.dumps(comparisons['HNS-v1']['scores_delta_pp'])+'`.',
        'Deltas vs D3 Balanced: `'+json.dumps(comparisons['D3 Balanced']['scores_delta_pp'])+'`.',
        'Matched actual sampling payloads (IDs/F/Dall/D3 texts/tokens/indices) and LR for all717 updates are exact against existing HNS full logs. First501–505 full-state gates:RESUME_GATE.json.',
        'Same fixed production graph tests prove forward violations unchanged, hierarchy gradient halved, alignment/sparsity gradients unchanged. Default production sources remain unchanged.',
        'Full recommendation: '+str(result['recommend_full'])+'. Retrieval is primary; no violation veto or density-alone health claim. Single seed; material-long-decline threshold0.05pp declared before training.',
        'Mask points/last50 and500→1217 changes:TRAINING_DIAGNOSTICS.json, MASK_HIERARCHY_AUDIT.json.',
        'Parent/final/baseline checkpoint and bare SHA256 records:RESUME_PROVENANCE.json, BASELINE_PROVENANCE.json, RESULTS.json. Binary assets never uploaded.',
        'Raw log local paths/sizes/SHA256/time ranges:RUNTIME_STATS.json. `/root` ephemeral cache; NFS originals retained. No copy/full image audit.',
        'No training beyond1217 or additional experiment was started.'])
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(final=False):
    assert original.git('branch','--show-current')==BRANCH
    assert not original.git('diff','--cached','--name-only')
    paths=[ROOT/p for p in CODE]+[EXP/p for p in STATIC]
    if final:
        paths += [EXP/p for p in REPORTS]+list((EXP/'evaluations').rglob('*.json'))
        assert all(sha(ROOT/p)==digest for p,digest in read(RUN/'launch-provenance.json')['source_sha256'].items())
    subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
    assert all(p.is_file() and p.stat().st_size<=1024*1024 for p in paths)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
    from recovery.check_stage500_publish import inspect
    review=inspect(); assert review['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if original.git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Report HNS-Half matched1217 intervention' if final else 'Prepare pinned HNS-Half500->1217 intervention'],cwd=ROOT,check=True)
    head=original.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    remote=original.git('rev-parse','origin/'+BRANCH)
    assert remote==head==original.git('rev-parse','FETCH_HEAD')
    receipt=dict(passed=True,branch=BRANCH,commit=head,remote_HEAD=remote,remote_HEAD_matches_local=True,
        push_success=True,fetch_success=True,final=final,checked_utc=now(),publication_check=review)
    dump(EXP/'GITHUB_RECEIPT.json',receipt)
    return receipt


def run():
    configure();signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('HARD_STOP supervisor signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            Supervisor().run();state('REPORTING',completed_steps=1217);report()
            state('PUBLISHING',completed_steps=1217);receipt=publish(True)
            state('COMPLETED_AND_SYNCED',completed_steps=1217,github=receipt)
        except BaseException as e:
            state('STOPPED_WITH_EVIDENCE',error=type(e).__name__+': '+str(e),automatic_retry=False)
            raise


def detach():
    assert not (RUN/'runner.json').exists()
    assert read(EXP/'GITHUB_RECEIPT.json')['commit']==original.git('rev-parse','HEAD')
    with MAIN_LOG.open('ab',buffering=0) as output:
        p=subprocess.Popen([PYTHON,'-u','-m',ENTRY],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
    dump(RUN/'runner.json',dict(pid=p.pid,session=p.pid,main_log=str(MAIN_LOG),stop_updates=1217,
        queue=['HNS-Half501..1217','three-way five-set evaluation','report','GitHub sync'],automatic_other_experiments=False))
    print(p.pid,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--prepare',action='store_true')
    p.add_argument('--publish',action='store_true');p.add_argument('--detach',action='store_true')
    args,remaining=p.parse_known_args()
    if args.worker:sys.argv=[sys.argv[0],*remaining];worker()
    elif args.prepare:prepare()
    elif args.publish:publish()
    elif args.detach:detach()
    else:run()


if __name__=='__main__':main()
