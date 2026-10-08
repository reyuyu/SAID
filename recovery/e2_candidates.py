"""Two pinned E2 candidates, isolated mother worktrees, sequential GPU ownership.

Production trainer/model/data/export/evaluator files stay byte-identical to each
mother. The only B intervention is an opt-in process-local ramp replacement.
All binaries/raw evidence live in canonical persistent runtime, never in Git.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from recovery.s02_nfs500 import dump, rows, sha, now, distribution

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/opt/data/private/lklk/SAID')
WORKTREES = Path('/opt/data/private/lklk/said-e2-worktrees')
PYTHON = str(PROJECT/'.venv/bin/python')
STEP0 = PROJECT/'runtime/SAID-nest-clip-v1/shared/step000000.pt'
STEP0_SHA = '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
IMAGES = Path('/root/said_s02_stage500/ShareGPT4V')
INDEX = IMAGES.parent/'data_index'
PARENT = PROJECT/'runtime/SAID-nest-clip-v1/nested-d3-hns-half1217-v1/step1217/step001217.pt'
PARENT_SHA = '6be7d283daa5e033eaba34d3b3e213e9528d0057ed1df5755398696bafcbb34d'
KEYS = ('Score5', 'J_long3', 'J_long', 'Short4')
SPEC = {
    'A': dict(name='HNS-Half', directory='nested_d3_hns_half2434_v1', worktree='hns-half2434',
              branch='experiment/nested-d3-hns-half2434-v1', mother='f7062162a065c5cb6b141a2edfed969d3692f2da',
              config='experiments/nest_clip_v1/nested_d3_hns_half1217_v1/config.json',
              segments=[(1217,2434)], source_run='nested-d3-hns-full-v1/step4868'),
    'B': dict(name='Balanced-Ramp500', directory='nested_d3_balanced_ramp500_2434_v1', worktree='balanced-ramp500',
              branch='experiment/nested-d3-balanced-ramp500-2434-v1', mother='6ad60c40147a80b7fabe21a85d2343758a109586',
              config='experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/config.json',
              segments=[(0,500),(500,1217),(1217,2434)],
              source_run='nested-detail-d3-balanced-full-20261006/step4868'),
}
CODE = ('recovery/e2_candidates.py', 'tests/test_e2_candidates.py',
        'tools/eval_five_parallel.py', 'tests/test_eval_five_parallel.py')
QUEUE_RUN = PROJECT/'runtime/SAID-nest-clip-v1/e2-final-candidates-v1'
LIMIT = 2434
INITIAL_A_START = 1217
ENTRY = 'recovery.e2_candidates'
PUBLICATION_TITLE = 'strict E2 validation'


def parent_for_segment(candidate, start):
    if candidate == 'A' and start == INITIAL_A_START:
        return PARENT
    return paths(candidate)[1]/f'step{start}'/f'step{start:06d}.pt'


def read(path): return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=120).strip()


def paths(candidate):
    s=SPEC[candidate]
    return (ROOT/'experiments/nest_clip_v1'/s['directory'],
            PROJECT/'runtime/SAID-nest-clip-v1'/s['directory'])


def ramp500(arm, completed):
    # Same completed-before-update convention as the original ramp200.
    return min(1.,completed/500.) if arm=='A3' else 0.


def install_schedule(candidate):
    from model import balanced_hparam_search as model
    if candidate=='A':
        from recovery.hns_half1217 import install_schedule as original
        original()
    else:
        from model import nested_semantic_mask as terms
        terms.inclusion_weight=model.inclusion_weight=ramp500


def mother_config(candidate):
    s=SPEC[candidate]
    return json.loads(subprocess.check_output(['git','show',s['mother']+':'+s['config']],cwd=ROOT))


def assert_frozen(candidate, config):
    baseline=mother_config(candidate)
    assert all(config.get(k)==v for k,v in baseline.items()), 'Mother config drift'
    assert config['view_weights']==[1.35,1.35,.3]
    assert config.get('view_sparsity_weights',[1.,2.,2.])==[1.,2.,2.]
    assert config['sampling_mode']=='nested_detail_d3' and config['inclusion_hierarchy']=='detail_chain'
    assert config['workers']==8 and config['epochs']==4 and config['seed']==0
    if candidate=='A':
        assert config['hns_enabled'] and config['hns_half_after500']
        assert config['inclusion_max']==0 and config.get('hns_beta',[2.,2.])==[2.,2.]
        assert not config.get('hns_detach_child',False)
    else:
        assert not config.get('hns_enabled',False) and config['inclusion_max']==1
        assert config['inclusion_ramp_steps']==500


def checkpoint_identity(path, step, expected=None):
    import torch
    digest=sha(path)
    if expected: assert digest==expected
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['completed_steps']==p['global_step']==p['scheduler']['completed_steps']==step
    assert p['scheduler_horizon']==p['scheduler']['horizon']==4868
    assert p['data_cursor']==dict(next_epoch=step//1217,next_batch=step%1217)
    assert p['next_epoch']==step//1217 and p['next_batch']==step%1217
    assert p['model'] and p['adapter'] is not None and p['optimizer']['state']
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={step}
    assert p['scaler']==dict(enabled=False,dtype='bfloat16',state=None)
    assert len(p['rng_per_rank'])==4
    rng=[]
    for rank,state in enumerate(p['rng_per_rank']):
        assert all(k in state for k in ('python','numpy','cpu','cuda','loader_generator'))
        rng.append(dict(rank=rank,keys=sorted(state),
            hashes={k:hashlib.sha256(state[k].numpy().tobytes()).hexdigest() for k in ('cpu','cuda','loader_generator')}))
    result=dict(passed=True,path=str(path),sha256=digest,bytes=Path(path).stat().st_size,
        completed_steps=step,scheduler=p['scheduler'],scaler=p['scaler'],sampler=p['sampler'],
        data_cursor=p['data_cursor'],optimizer_counters=[step],rng_per_rank=rng,
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_count':len(g['params'])}
            for g in p['optimizer']['param_groups']],configuration=p['config'],uploaded=False)
    return result


def references():
    """Fetched remote Git blobs must equal local raw JSON before use."""
    sources={
        'epochs':('analysis/d3-hns-epoch-curve-eval-v1','experiments/nest_clip_v1/epoch_curve_eval_v1/RESULTS.json'),
        'epoch_diag':('analysis/d3-hns-epoch-curve-eval-v1','experiments/nest_clip_v1/epoch_curve_eval_v1/TRAINING_DIAGNOSTICS_COMPARISON.json'),
        'half':('experiment/nested-d3-hns-half1217-v1','experiments/nest_clip_v1/nested_d3_hns_half1217_v1/RESULTS.json'),
        'balanced500':('experiment/nested-detail-d3-balanced-full-v1','experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/RESULTS.json'),
        'inc0':('experiment/nested-d3-inc0-500-v1','experiments/nest_clip_v1/nested_d3_inc0_500_v1/RESULTS.json'),
        'hns500':('experiment/nested-d3-hns-full-v1','experiments/nest_clip_v1/nested_d3_hns500_v1/RESULTS.json'),
    }
    verified={};data={}
    for key,(branch,path) in sources.items():
        remote='origin/'+branch
        blob=subprocess.check_output(['git','show',remote+':'+path],cwd=ROOT)
        assert blob==(PROJECT/path).read_bytes(), ('Historical raw JSON differs from fetched Git',key)
        data[key]=json.loads(blob)
        verified[key]=dict(branch=branch,commit=git('rev-parse',remote),path=str(PROJECT/path),
            sha256=hashlib.sha256(blob).hexdigest(),raw_JSON_exact_fetched_Git=True)
    points={
        'D3 Balanced':dict(data['epochs']['models']['D3_Balanced'],**{'500':data['balanced500']}),
        'HNS-v1':dict(data['epochs']['models']['HNS_v1'],**{'500':data['hns500']}),
        'HNS-Half':{'500':data['hns500'],'1217':data['half']['models']['HNS-Half']},
        'INC0':{'500':data['inc0']},
    }
    compact={name:{step:dict(scores_percent={k:v['scores_percent'][k] for k in KEYS},metrics=v['metrics'])
        for step,v in entries.items()} for name,entries in points.items()}
    return dict(passed=True,sources=verified,models=compact,epoch_diagnostics=data['epoch_diag'])


def evaluator_sources():
    validation=read(PROJECT/'engineering/parallel_five_eval_v1/COMPARISON.json')
    receipt=read(PROJECT/'engineering/parallel_five_eval_v1/PARALLEL_RUN.json')
    assert validation['passed'] and validation['metrics_exact'] and validation['scores_exact']
    frozen=dict(validation['unchanged_evaluator_source_sha256'])
    frozen['tools/eval_five_parallel.py']=receipt['scheduler_source_sha256']
    for path,h in frozen.items(): assert sha(ROOT/path)==h, path
    return frozen


def prepare(candidate):
    import torch
    from train.train_nested_semantic_mask import code_manifest
    from recovery import s02_local500 as local
    from tools.eval_five_parallel import require_gpu_idle
    torch.set_num_threads(4)
    exp,run=paths(candidate);s=SPEC[candidate]
    assert git('branch','--show-current')==s['branch']
    assert git('merge-base',s['mother'],'HEAD')==s['mother']
    assert not run.exists(), 'Never overwrite a prepared trajectory'
    require_gpu_idle({0,1,2,3})
    assert sha(STEP0)==STEP0_SHA
    ready=read(IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    exp.mkdir(parents=True,exist_ok=False);run.mkdir(parents=True)
    cfg=mother_config(candidate)
    if candidate=='B': cfg['inclusion_ramp_steps']=500
    assert_frozen(candidate,cfg);dump(exp/'config.json',cfg)
    native=code_manifest()
    for path,h in native.items():
        assert hashlib.sha256(subprocess.check_output(['git','show',s['mother']+':'+path],cwd=ROOT)).hexdigest()==h
    ref=references();dump(exp/'BASELINE_PROVENANCE.json',ref)
    provenance=dict(passed=True,mother_commit=s['mother'],production_sources=native,
        evaluated_sources=evaluator_sources(),common0=str(STEP0),common0_sha256=STEP0_SHA,
        fresh_common0=candidate=='B',resume=str(PARENT) if candidate=='A' else None,
        scaler='Original BF16 autocast, FP32 parameters; GradScaler disabled, state=None',
        image_root=str(IMAGES),index_dir=str(INDEX),NFS_fallback=False,checked_utc=now())
    if candidate=='A':
        identity=checkpoint_identity(PARENT,1217,PARENT_SHA)
        old=read(PROJECT/'experiments/nest_clip_v1/nested_d3_hns_half1217_v1/RESULTS.json')['models']['HNS-Half']
        assert identity['sha256']==old['checkpoint_sha256']==old['strict_export']['checkpoint_sha256']
        assert old['evaluation_checkpoint_immutable'] and identity['configuration']['code_sha256']==native
        assert_frozen(candidate,identity['configuration']);provenance['parent']=identity
        assert identity['configuration']['hns_half_after500']
    dump(exp/('RESUME_PROVENANCE.json' if candidate=='A' else 'FRESH_START_PROOF.json'),provenance)
    proof=local.path_proof();dump(run/'prelaunch-local-path-proof-5000.json',proof)
    dump(exp/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],NFS_fallback=False,
        evidence=str(run/'prelaunch-local-path-proof-5000.json'),sha256=sha(run/'prelaunch-local-path-proof-5000.json')))
    dump(exp/'PLAN.json',dict(candidate=candidate,name=s['name'],branch=s['branch'],mother=s['mother'],
        segments=s['segments'],stop=2434,horizon=4868,next_update=1218 if candidate=='A' else 1,
        sole_change='Continue existing Half1217 with hierarchy weight .5' if candidate=='A' else 'soft inclusion ramp length200->500',
        ramp_convention=('Existing Half schedule, every new update1218..2434 hierarchy_weight=.5' if candidate=='A' else
            'completed BEFORE current update; update1=0, update200=199/500, update500=499/500, update501=1'),
        alignment=[1.35,1.35,.3],sparsity=[1,2,2],K=3,
        pause_for_evaluation=[stop for start,stop in s['segments']],
        evaluation='Validated four-GPU scheduler; unchanged native bare evaluator batch64',
        GPU_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        technical_failure_stop=True,scores_do_not_change_plan=True,automatic3651=False,automatic4868=False,third_experiment=False,
        thresholds=dict(material_long_decline_pp=.05,promising_Score5_gap_pp=.05),
        cache='Disposable /root overlay; NFS source retained; no copying/full audit'))
    set_state(candidate,'PREPARED')


def set_state(candidate,status,**extra):
    exp,run=paths(candidate)
    dump(exp/'STATE.json',dict(status=status,candidate=candidate,pid=os.getpid(),updated_utc=now(),
        stop_updates=LIMIT,automatic_continuation=False,**extra))


def baseline_rows(candidate):
    return rows(PROJECT/'runtime/SAID-nest-clip-v1'/SPEC[candidate]['source_run']/'steps.jsonl')[:LIMIT]


def verify_stream(candidate, actual, start):
    from recovery.nested_detail_d3_balanced500 import digest_indices
    from train.nested_semantic_data import sample_partial_detail_indices
    from recovery.s02_local_full import expected_lrs
    refs={r['step']:r for r in baseline_rows(candidate)}
    cfg=read(paths(candidate)[0]/'config.json');count=0
    assert [r['step'] for r in actual]==list(range(start+1,start+len(actual)+1))
    for row in actual:
        old=refs[row['step']]
        assert row['epoch']==old['epoch'] and row['actual_lrs']==old['actual_lrs']
        assert list(row['actual_lrs'].values())==expected_lrs(row['step']-1,cfg)
        assert math.isfinite(row['loss']) and row['nonfinite']==0
        if candidate=='A':
            assert row['lambda_h']==.5 and row['inc_weight']==0 and row['inclusion_loss']==0
            assert math.isclose(row['HNS_surcharge'],.5*(2*row['V_DF_hard']+2*row['V_3D_hard'])/3,rel_tol=4e-6,abs_tol=1e-7)
        else:
            assert math.isclose(row['inc_weight'],ramp500('A3',row['step']-1),rel_tol=1e-6,abs_tol=1e-7)
        assert {h['rank'] for h in row['rank_health']}=={0,1,2,3}
        for h in row['rank_health']:
            prior=next(x for x in old['rank_health'] if x['rank']==h['rank'])
            assert h['stream_sha256']==prior['stream_sha256'], ('IDs/text/token drift',row['step'],h['rank'])
            sample=h['sampling'];old_sample=prior['sampling']
            assert all(sample[k]==old_sample[k] for k in ('sample_ids','n','K'))
            chosen=[sample_partial_detail_indices(n,0,row['epoch'],sid) for n,sid in zip(sample['n'],sample['sample_ids'])]
            assert sample['D3_selected_sentence_indices_sha256']==digest_indices(sample['sample_ids'],chosen)
            assert h['gradients_finite'] and h['batch']==prior['batch']
            assert h['updates']==row['step']-start
            count+=h['batch']
    return dict(passed=True,updates=len(actual),samples=count,first_update=start+1,
        actual_IDs_F_Dall_D3_strings_tokens_indices_exact=True,LR_trajectory_exact=True,
        source=str(PROJECT/'runtime/SAID-nest-clip-v1'/SPEC[candidate]['source_run']/'steps.jsonl'))


def worker(candidate):
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from recovery import s02_local500 as local
    from recovery.s02_local_full import stream,expected_lrs
    from recovery.nested_detail_d3_balanced500 import digest_indices
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate,training_phase_timing as timing
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_train_gate import require_live_supervisor
    exp,run=paths(candidate)
    output=Path(sys.argv[sys.argv.index('--output-dir')+1]);stop=int(sys.argv[sys.argv.index('--max-updates')+1])
    resume=Path(sys.argv[sys.argv.index('--resume')+1]) if '--resume' in sys.argv else None
    start=int(read(run/'active-segment.json')['start'])
    phase=Path(os.environ['SAID_S02_PHASE_LOCAL'])
    supervisor=int(os.environ['SAID_FULL_SUPERVISOR_PID']);require_live_supervisor(supervisor)
    cfg=read(exp/'config.json');assert_frozen(candidate,cfg);assert (start,stop) in SPEC[candidate]['segments']
    if candidate=='A':
        assert resume==parent_for_segment(candidate,start)
        if start==INITIAL_A_START: assert sha(resume)==PARENT_SHA
        else: assert sha(resume)==read(exp/f'step{start}_RESULTS.json')['checkpoint']['sha256']
    elif start==0: assert resume is None
    else: assert resume==run/f'step{start}'/f'step{start:06d}.pt'
    install_schedule(candidate);gate.RUN=run
    context=dict(optimizer=None,previous=None,restored=False,checked=False,first5=False)
    original_validate,original_optimizer=trainer.validate_resume_payload,trainer.build_optimizer
    original_restore,original_rates,original_save=trainer.restore_rng_state,trainer.optimizer_learning_rates,trainer.atomic_save
    original_sampling,original_loader=trainer.sampling_diagnostics,trainer.DataLoader
    references_by_step={r['step']:r for r in baseline_rows(candidate) if start<r['step']<=start+5}

    def validate(previous,current,*args):
        assert_frozen(candidate,current)
        assert previous['completed_steps']==previous['global_step']==start
        assert previous['config']['code_sha256']==current['code_sha256']
        assert previous['scaler']==dict(enabled=False,dtype='bfloat16',state=None)
        assert previous['scheduler']['completed_steps']==start and previous['scheduler']['horizon']==4868
        assert {int(s['step']) for s in previous['optimizer']['state'].values()}=={start}
        if candidate=='B': assert previous['config']['inclusion_ramp_steps']==500
        context['previous']=previous
        return original_validate(previous,current,*args)

    def optimizer(module):
        result=original_optimizer(module);assert not result.state
        context['optimizer']=result;return result

    def restore(state):
        original_restore(state)
        assert gate.exact_state(trainer.rng_state(),{k:state[k] for k in ('python','numpy','cpu','cuda')})
        context['restored']=True

    def save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            gate.resumable_metadata(payload)
            payload['e2_candidate']=candidate
            payload['schedule_intervention']=dict(hns_half_after500=True) if candidate=='A' else dict(inclusion_ramp_steps=500)
        return original_save(payload,path)

    def sampling(batch):
        value=original_sampling(batch)
        value['D3_selected_sentence_indices_sha256']=digest_indices(batch['sample_id'].tolist(),batch['detail_indices'])
        return value

    def rates(module,completed,horizon):
        require_live_supervisor(supervisor);assert horizon==4868 and start<=completed<stop<=LIMIT
        if completed==start and not context['checked']:
            previous=context['previous']
            if resume:
                assert context['restored'] and previous is not None
                assert gate.exact_state(module.clip.state_dict(),previous['model'])
                assert gate.exact_state(trainer.auxiliary_module(module).state_dict(),previous['adapter'])
                assert gate.exact_state(context['optimizer'].state_dict(),previous['optimizer'])
                saved=previous['rng_per_rank'][dist.get_rank()]
                assert torch.equal(module._loader_epoch_generator_state,saved['loader_generator'])
                assert gate.exact_state(trainer.rng_state(),{k:saved[k] for k in ('python','numpy','cpu','cuda')})
            else:
                assert not context['optimizer'].state
                current_cfg=read(output/'config.json')
                original_cfg=read(PROJECT/'runtime/SAID-nest-clip-v1'/SPEC[candidate]['source_run']/'config.json')
                assert current_cfg['init_sha256']==STEP0_SHA
                for key in ('component_initialization','parameter_counts','runtime_model','data','horizon','code_sha256'):
                    assert current_cfg[key]==original_cfg[key], ('Fresh initialization/optimizer drift',key)
            agreement=trainer.parameter_agreement(module);assert agreement==0
            proof=dict(rank=dist.get_rank(),passed=True,resume=bool(resume),next_update=start+1,
                model_adapter_optimizer_RNG_loader_exact=bool(resume),fresh_empty_AdamW=not resume,
                parameter_difference=agreement,cursor=dict(next_epoch=start//1217,next_batch=start%1217),
                optimizer_counters=[start] if start else [],scheduler_horizon=4868,scaler_disabled_BF16=True)
            gathered=[None]*4;dist.all_gather_object(gathered,proof)
            if dist.get_rank()==0:dump(run/f'initial-gate-{start}.json',dict(passed=True,ranks=gathered))
            context.update(checked=True,previous=None)
        if completed==start+5 and not context['first5']:
            trainer.save_checkpoint(module,context['optimizer'],read(output/'config.json'),completed,output)
            difference=trainer.parameter_agreement(module);assert difference==0
            result=None
            if dist.get_rank()==0:
                try:
                    proof=verify_stream(candidate,rows(output/'steps.jsonl')[:5],start)
                    identity=checkpoint_identity(output/f'step{completed:06d}.pt',completed)
                    actual_groups=identity['configuration']['parameter_counts']['optimizer_groups']
                    reference_cfg=read(PROJECT/'runtime/SAID-nest-clip-v1'/SPEC[candidate]['source_run']/'config.json')
                    assert actual_groups==reference_cfg['parameter_counts']['optimizer_groups']
                    result=dict(passed=True,stream=proof,optimizer_counters=identity['optimizer_counters'],
                        rank_parameter_difference=difference,gate='BEFORE_UPDATE'+str(completed+1))
                except Exception as error:result=dict(passed=False,error=repr(error))
                dump(run/f'first-five-gate-{start}.json',result)
                print(json.dumps(dict(event='FIRST_FIVE_GATE',**result)),flush=True)
            shared=[result];dist.broadcast_object_list(shared,src=0);assert shared[0]['passed'],shared[0]
            context['first5']=True
        if candidate=='A' and completed in (1600,2000):
            trainer.save_checkpoint(module,context['optimizer'],read(output/'config.json'),completed,output)
        values=original_rates(module,completed,horizon)
        assert list(values)==expected_lrs(completed,cfg)
        return values

    class HeartbeatIterator(timing.TimedIterator):
        def __init__(self,iterator,recorder):
            super().__init__(iterator,recorder)
            self.position=0;self.epoch=iterator._dataset.epoch
        def __next__(self):
            self.recorder.begin();self.position+=1;step=self.epoch*1217+self.position
            event=dict(rank=self.recorder.rank,step=step,started_monotonic=time.monotonic(),utc=now(),replay_without_update=step<=start)
            path=phase/f'heartbeat-rank{self.recorder.rank}.json';dump(path,dict(event,state='DATA_WAIT'))
            try:value=next(self.iterator)
            except StopIteration:
                self.recorder.current=None;dump(path,dict(event,state='EXHAUSTED'));raise
            except BaseException:dump(path,dict(event,state='DATA_WAIT_FAILED'));raise
            self.recorder.current.update(step=step,data_wait_s=time.monotonic()-event['started_monotonic'])
            if start<step<=start+5:
                expected=next(h for h in references_by_step[step]['rank_health'] if h['rank']==self.recorder.rank)
                assert stream(value)==expected['stream_sha256']
                assert value['sample_id'].tolist()==expected['sampling']['sample_ids']
                from train.nested_semantic_data import sample_partial_detail_indices
                chosen=[sample_partial_detail_indices(n,0,self.epoch,sid) for n,sid in zip(expected['sampling']['n'],value['sample_id'].tolist())]
                assert value['detail_indices']==chosen
                dump(run/f'batch-{step}-rank{self.recorder.rank}.json',dict(passed=True,step=step,rank=self.recorder.rank,
                    stream_sha256=expected['stream_sha256'],epoch=self.epoch,batch_cursor=self.position-1,
                    sample_ids_text_tokens_indices_exact=True))
            dump(path,dict(event,state='ACTIVE_STEP'));return value

    trainer.validate_resume_payload,trainer.build_optimizer=validate,optimizer
    trainer.restore_rng_state,trainer.optimizer_learning_rates,trainer.atomic_save=restore,rates,save
    trainer.sampling_diagnostics,trainer.NestedDataset=sampling,local.LoggedLocalDataset
    trainer.DataLoader=lambda *a,**kw:original_loader(*a,**kw,timeout=60)
    timing.TimedIterator=HeartbeatIterator
    recorder=timing.install(str(phase),int(os.environ['RANK']))
    try:trainer.main()
    finally:
        recorder.finish()
        if dist.is_initialized():dist.destroy_process_group()
    dump(phase/f'heartbeat-rank{os.environ["RANK"]}.json',dict(state='TRAINING_COMPLETE',utc=now()))


def training_command(candidate,start,stop):
    exp,run=paths(candidate)
    cmd=[str(PROJECT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
        '-m','--',ENTRY,'--worker','--candidate',candidate,
        '--config',str(exp/'config.json'),'--init-state',str(STEP0),'--index-dir',str(INDEX),
        '--image-root',str(IMAGES),'--output-dir',str(run/f'step{stop}'),'--run-type','formal','--max-updates',str(stop)]
    parent=parent_for_segment(candidate,start)
    if start:cmd.extend(['--resume',str(parent)])
    return cmd


class Supervisor:
    def __init__(self,candidate):
        from recovery import s02_local500 as local,s02_local_full as full
        self.candidate=candidate;self.exp,self.run=paths(candidate)
        local.RUN=full.RUN=self.run
        self.supervisor=full.Supervisor()
        self.started=time.monotonic()

    def execute(self,name,command,phase=None,training=False):
        from recovery import s02_local_full as full
        if phase is not None:full.PHASE=phase
        self.supervisor.execute(name,command,training=training)
        dump(self.exp/'COMMANDS.json',self.supervisor.commands)

    def evaluate(self,step,checkpoint):
        from tools.eval_five_parallel import require_gpu_idle
        from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short
        evaluator_sources();require_gpu_idle({0,1,2,3})
        identity=checkpoint_identity(checkpoint,step)
        out=self.run/'evaluations'/f'step{step}';out.mkdir(parents=True,exist_ok=False)
        bare=out/f'student_step{step}.pt'
        self.execute(f'export{step}',[PYTHON,'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),'--output',str(bare),'--expect-updates',str(step)])
        self.execute(f'verify{step}',[PYTHON,'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),
            '--bare',str(bare),'--output',str(out/'export-check.json'),'--index-dir',str(INDEX),'--image-root',str(IMAGES)])
        check=read(out/'export-check.json');bare_sha=sha(bare)
        assert check['passed'] and check['strict_load'] and check['optimizer_steps']==[step]
        assert check['image_max_abs']==check['text_max_abs']==0
        assert check['checkpoint_sha256']==identity['sha256'] and check['bare_sha256']==bare_sha
        set_state(self.candidate,'EVALUATING',update=step)
        self.execute(f'five-eval{step}',[PYTHON,'-m','tools.eval_five_parallel','--checkpoint',str(bare),
            '--training-checkpoint',str(checkpoint),'--output-dir',str(out)])
        metrics,raw,outputs=native_metrics(out)
        aggregate=with_short(dict(metrics=metrics,scores=scores(metrics)))
        percent={k:100*aggregate[{'Score5':'Score5_R1','Short4':'Short4_R1'}.get(k,k)] for k in KEYS}
        assert sha(checkpoint)==identity['sha256'] and sha(bare)==bare_sha
        receipt=read(out/'EVAL_PARALLEL_RUN.json');assert receipt['status']=='COMPLETED'
        for dataset,path in outputs.items():dump(self.exp/f'evaluations/step{step}/{dataset}.json',read(path))
        dump(self.exp/f'evaluations/step{step}/EVAL_PARALLEL_RUN.json',receipt)
        value=dict(update=step,metrics=metrics,scores_percent=percent,strict_export=check,
            checkpoint=identity,bare=dict(path=str(bare),sha256=bare_sha,bytes=bare.stat().st_size,uploaded=False),
            checkpoint_unchanged=True,evaluation_wall_seconds=receipt['wall_seconds'])
        dump(self.exp/f'step{step}_RESULTS.json',value);return value


def summarize(records):
    from recovery.nested_detail_d3_balanced_full_evidence import view_means
    scalars={k:statistics.fmean(r[k] for r in records) for k,v in records[0].items()
        if isinstance(v,(int,float)) and (k.startswith(('F_','O_','E_','HNS_','V_','Dall_','Ds_')) or
            k in ('loss','alignment','sparsity','inclusion_loss','inc','inc_weight','lambda_h'))}
    return dict(steps=[records[0]['step'],records[-1]['step']],scalars=scalars,views=view_means(records),
        gradient_groups={k:statistics.fmean(h['gradient_norms'][k] for r in records for h in r['rank_health'])
            for k in records[0]['rank_health'][0]['gradient_norms']})


def delta(current,reference):
    return dict(scores_delta_pp={k:current['scores_percent'][k]-reference['scores_percent'][k] for k in KEYS},
        recall_delta_pp={ds:{dr:{k:100*(v-reference['metrics'][ds][dr][k]) for k,v in rec.items()}
            for dr,rec in directions.items()} for ds,directions in current['metrics'].items()})


def classify(current,balanced):
    d=delta(current,balanced)['scores_delta_pp']
    if d['Score5']>0 and min(d['J_long3'],d['J_long'])>=-.05: return 'STRONG_CANDIDATE'
    advantages=[v for ds in delta(current,balanced)['recall_delta_pp'].values() for dr in ds.values() for k,v in dr.items() if k=='R@1']
    if abs(d['Score5'])<.05 and (d['J_long3']>0 or d['J_long']>0 or max(advantages)>=.1):return 'PROMISING'
    return 'NEGATIVE'


def runtime_stats(candidate):
    exp,run=paths(candidate);cycles=[];phases=[];steps=[]
    for start,stop in SPEC[candidate]['segments']:
        cycles+=rows(run/f'step{stop}/cycle_timing.jsonl')
        steps+=rows(run/f'step{stop}/steps.jsonl')
        phase=IMAGES.parent/f'e2-{candidate}-to{stop}-phase'
        phases.extend(p for rank in range(4) for p in rows(phase/f'rank{rank}.jsonl'))
    wait={}
    for p in phases:wait[p['step']]=max(wait.get(p['step'],0),p['data_wait_s'])
    telemetry=rows(run/'full-resource-telemetry.jsonl')
    systems=[r['system'] for r in telemetry]
    errors=[]
    for start,stop in SPEC[candidate]['segments']:
        text=(run/f'train{stop}.log').read_text(errors='replace')
        errors.extend(re.findall(r'Image failure sample=\d+[^\n]*|Input/output error[^\n]*|Missing local sample=\d+[^\n]*',text))
    oom=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0)
    assert not errors and oom==0
    result=dict(full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        data_wait_seconds=distribution(list(wait.values())),steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),
        steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        GPU_peak_GiB={str(rank):max(h['peak_allocated_gib'] for r in steps for h in r['rank_health'] if h['rank']==rank) for rank in range(4)},
        peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),peak_file_cache_bytes=max(s['file'] for s in systems),
        oom_kill=oom,training_io_errors=errors,pod_supervisor_anomaly=False,
        scope='Actual new updates only, checkpoint/export/evaluation times separate',
        raw_assets=[dict(path=str(p),bytes=p.stat().st_size,uploaded=False) for p in run.rglob('*') if p.is_file()])
    for kind in ('io_PSI','memory_PSI'):
        result[kind]={v:distribution([s[kind][v]['avg10'] for s in systems if v in s[kind]]) for v in ('some','full')}
    return result


def report(candidate,results,started):
    exp,run=paths(candidate);allsteps=[];checks=[]
    for start,stop in SPEC[candidate]['segments']:
        segment=rows(run/f'step{stop}/steps.jsonl');assert len(segment)==stop-start
        checks.append(verify_stream(candidate,segment,start));allsteps+=segment
        assert read(run/f'initial-gate-{start}.json')['passed']
        assert read(run/f'first-five-gate-{start}.json')['passed']
    refs=read(exp/'BASELINE_PROVENANCE.json');final=results['2434']
    comparisons={name:delta(final,refs['models'][name]['2434']) for name in ('D3 Balanced','HNS-v1')}
    status=classify(final,refs['models']['D3 Balanced']['2434'])
    diag=dict(last50=summarize(allsteps[-50:]),points={str(step):summarize([next(r for r in allsteps if r['step']==step)])
        for step in (1600,2000,2434) if any(r['step']==step for r in allsteps)})
    if candidate=='A':
        parent_rows=rows(PARENT.parent/'steps.jsonl');diag['parent1217_last50']=summarize(parent_rows[-50:])
    else:
        diag['stage_last50']={str(step):summarize([r for r in allsteps if step-50<r['step']<=step]) for step in (500,1217,2434)}
    diag['E2_reference_diagnostics']={name:refs['epoch_diagnostics'][name]['2434'] for name in ('D3_Balanced','HNS_v1')}
    runtime=runtime_stats(candidate);runtime['total_experiment_wall_seconds']=time.monotonic()-started
    output=dict(status=status,name=SPEC[candidate]['name'],completed_steps=2434,
        new_updates=sum(b-a for a,b in SPEC[candidate]['segments']),models=results,
        comparisons=comparisons,
        stage_comparisons={step:{name:delta(value,entries[step]) for name,entries in refs['models'].items() if step in entries}
            for step,value in results.items()},
        stream_proofs=checks,recommend_E3=status=='STRONG_CANDIDATE',
        stopped2434=True,automatic3651=False,automatic4868=False,no_third_experiment=True,
        phase_evaluations_paused_training=True,evaluator_math_modified=False)
    dump(exp/'RESULTS.json',output);dump(exp/'TRAINING_DIAGNOSTICS.json',diag)
    dump(exp/'MASK_HIERARCHY_AUDIT.json',dict(last50=diag['last50'],references=diag['E2_reference_diagnostics']))
    dump(exp/'RUNTIME_STATS.json',runtime);dump(exp/'VALIDATION.json',dict(passed=True,streams=checks))
    inventory=[checkpoint_identity(p,int(p.stem[4:])) for p in run.glob('step*/*.pt')]
    dump(exp/'CHECKPOINT_INVENTORY.json',inventory)
    dump(exp/'EXPORT_AUDIT.json',{step:value['strict_export'] for step,value in results.items()})
    if candidate=='B':dump(exp/'SCHEDULE_AUDIT.json',dict(passed=True,convention='completed-before-current-update',
        original={str(s):min(1.,(s-1)/200.) for s in (1,100,200,500,501,1217,2434)},
        new={str(s):ramp500('A3',s-1) for s in (1,100,200,500,501,1217,2434)},
        measured_all_updates_exact=True,inclusion_formula_child_detach_edges_unchanged=True))
    lines=[f'# {SPEC[candidate]["name"]} E2 validation','',f'Status: `{status}`. Stop exactly2434. No3651/full/third experiment.',
        'One seed; retrieval is primary. Material long decline threshold0.05pp; promising Score5 gap0.05pp, declared before launch.',
        'Checkpoint boundaries pause all DDP ranks for strict export and five-dataset evaluation, then restore complete state without changing the schedule.',
        'Production trainer/model/data/export/evaluator sources equal the respective mother. B changes only the process-local ramp200 to500 with completed-before-current-update convention.',
        'Native normalized image/full-caption embedding retrieval only. GPU0 COCO;1 DOCCI;2 Long;3 Flickr then Urban. Batch64 unchanged.',
        '', '| Step | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    trajectory=dict(refs['models']['HNS-Half']) if candidate=='A' else {}
    trajectory.update(results)
    for step,value in sorted(trajectory.items(),key=lambda x:int(x[0])):
        lines.append('| '+step+' | '+' | '.join(f'{value["scores_percent"][k]:.6f}' for k in KEYS)+' |')
    for step,value in results.items():
        lines+=['',f'## Step{step}','', '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|']
        for ds,m in value['metrics'].items():
            lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
        for name,points in refs['models'].items():
            if step in points:lines+=['',name+' delta(pp): `'+json.dumps(delta(value,points[step])['scores_delta_pp'])+'`.']
    lines+=['','All30 recalls and deltas: RESULTS.json. Keep/violation/IoU/CE/share/group-gradient/gate: TRAINING_DIAGNOSTICS.json.',
        'No inference mask/gate/rerank/ensemble/TTA. GradScaler remains disabled under original BF16 autocast.',
        'Source/checkpoint/RNG/cursor identity and full actual sample/text/token/index/LR gates are recorded. No cross-run bitwise-loss requirement.',
        '`/root` disposable image cache; persistent NFS originals retained. Binaries/raw logs stay in canonical runtime, never Git.',
        'Recommendation only; this runner cannot continue to3651 or launch any extra experiment.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return output


def unified():
    a=WORKTREES/SPEC['A']['worktree']/'experiments/nest_clip_v1'/SPEC['A']['directory']
    b=paths('B')[0];refs=read(b/'BASELINE_PROVENANCE.json')
    models={name:refs['models'][name]['2434'] for name in ('D3 Balanced','HNS-v1')}
    models.update({SPEC['A']['name']:read(a/'RESULTS.json')['models']['2434'],SPEC['B']['name']:read(b/'RESULTS.json')['models']['2434']})
    incumbent=models['D3 Balanced'];improved=[name for name in ('HNS-Half','Balanced-Ramp500') if models[name]['scores_percent']['Score5']>incumbent['scores_percent']['Score5']]
    best=max(improved,key=lambda name:models[name]['scores_percent']['Score5']) if improved else 'D3 Balanced'
    value=dict(models=models,delta_vs_balanced={name:delta(v,incumbent) for name,v in models.items()},
        classifications={name:classify(v,incumbent) for name,v in models.items() if name not in ('D3 Balanced','HNS-v1')},
        BEST_E2_CANDIDATE=best,decision='KEEP_D3_BALANCED' if not improved else best,
        RECOMMEND_CONTINUE_TO_3651=bool(improved) and classify(models[best],incumbent)=='STRONG_CANDIDATE',E3_target_Score5=73.83018163819824,
        no3651_started=True,no4868_started=True,no_third_experiment=True,
        E2_training_diagnostics={SPEC['A']['name']:read(a/'TRAINING_DIAGNOSTICS.json')['last50'],SPEC['B']['name']:read(b/'TRAINING_DIAGNOSTICS.json')['last50'],
            **{name:refs['epoch_diagnostics'][key]['2434'] for name,key in [('D3 Balanced','D3_Balanced'),('HNS-v1','HNS_v1')]}})
    dump(b/'UNIFIED_COMPARISON.json',value)
    lines=['# Final E2 candidates','', '| Model @2434 | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    for name,result in models.items():lines.append('| '+name+' | '+' | '.join(f'{result["scores_percent"][k]:.6f}' for k in KEYS)+' |')
    lines+=['', '| Model @2434 | COCO I/T | Urban I/T | Flickr I/T | DOCCI I/T | Long I/T |','|---|---:|---:|---:|---:|---:|']
    for name,result in models.items():
        lines.append('| '+name+' | '+' | '.join(' / '.join(f'{100*result["metrics"][ds][dr]["R@1"]:.6f}' for dr in ('I2T','T2I'))
            for ds in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI'))+' |')
    lines+=['',f'BEST_E2_CANDIDATE: `{best}`. Decision: `{value["decision"]}`.',
        'Retrieval is primary; mask appearance cannot override Score5. Exact30 recalls and all deltas: UNIFIED_COMPARISON.json.',
        'RECOMMEND_CONTINUE_TO_3651: '+str(value['RECOMMEND_CONTINUE_TO_3651'])+'. Recommendation only; no extra training started.',
        'E3 target:73.830182. E2 alone cannot establish whether it will beat E3.']
    (b/'FINAL_COMPARISON.md').write_text('\n'.join(lines)+'\n')


def publish(candidate,setup=False):
    from recovery import check_stage500_publish as checker
    exp,run=paths(candidate);branch=SPEC[candidate]['branch']
    assert git('branch','--show-current')==branch
    files=[ROOT/p for p in CODE]+[p for p in exp.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
    relative=[str(p.relative_to(ROOT)) for p in files]
    assert set(git('diff','--cached','--name-only').splitlines())<=set(relative)
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
    if git('diff','--cached','--name-only'):
        previous=Path.cwd();os.chdir(ROOT)
        try:checker.ALLOWED=set(relative);check=checker.inspect()
        finally:os.chdir(previous)
        assert check['passed'];subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',('Prepare ' if setup else 'Report ')+SPEC[candidate]['name']+' '+PUBLICATION_TITLE],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],cwd=ROOT,check=True,timeout=120)
    remote=git('rev-parse','origin/'+branch);assert remote==head==git('rev-parse','FETCH_HEAD')
    dump(run/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(branch=branch,commit=head,remote_HEAD=remote,remote_HEAD_matches_local=True,checked_utc=now()))


def run(candidate):
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    from recovery.resource_stall_v2 import system_snapshot
    def stop(sig,frame): raise RuntimeError('Candidate supervisor interrupted; preserve progress')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    exp,runtime=paths(candidate);started=time.monotonic();s=Supervisor(candidate);results={}
    assert read(exp/'CPU_TESTS.json')['passed'];assert sha(STEP0)==STEP0_SHA
    assert read(exp/'STATE.json')['status']=='PREPARED','No implicit retries/duplicate candidate runs'
    proof=read(exp/('RESUME_PROVENANCE.json' if candidate=='A' else 'FRESH_START_PROOF.json'))
    assert proof['production_sources']==code_manifest();evaluator_sources()
    assert not system_snapshot()['memory_events'].get('oom_kill',0)
    try:
        for start,stop in SPEC[candidate]['segments']:
            require_gpu_idle({0,1,2,3})
            phase=IMAGES.parent/f'e2-{candidate}-to{stop}-phase';phase.mkdir(exist_ok=False)
            dump(runtime/'active-segment.json',dict(start=start,stop=stop,phase=str(phase)))
            set_state(candidate,'TRAINING',start=start,stop=stop,phase=str(phase))
            s.execute(f'train{stop}',training_command(candidate,start,stop),phase=phase,training=True)
            acceptance=read(runtime/f'step{stop}/acceptance.json')
            assert acceptance['passed'] and all(r['completed_updates']==stop and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
            stream=verify_stream(candidate,rows(runtime/f'step{stop}/steps.jsonl'),start)
            dump(exp/f'step{stop}_STREAM_PROOF.json',stream)
            checkpoint=runtime/f'step{stop}/step{stop:06d}.pt'
            results[str(stop)]=s.evaluate(stop,checkpoint)
            dump(exp/'PROGRESS.json',dict(models=results,stop=2434,scores_do_not_change_plan=True))
        report(candidate,results,started)
        if candidate=='B':unified()
        require_gpu_idle({0,1,2,3});set_state(candidate,'COMPLETE',completed_steps=2434)
        publish(candidate)
        dump(runtime/'completed.json',dict(status='COMPLETED_AND_SYNCED',candidate=candidate,stop=2434,finished_utc=now()))
    except BaseException as error:
        set_state(candidate,'FAILED',error=repr(error));raise


def queue():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    QUEUE_RUN.mkdir(parents=True,exist_ok=True)
    def stop(sig,frame):raise RuntimeError('Sequential runner interrupted; preserve progress')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (QUEUE_RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (QUEUE_RUN/'QUEUE_STATE.json').exists(),'No duplicate launch'
        state=dict(pid=os.getpid(),order=['A train1218..2434','A evaluate/report/push/fetch','B fresh0..500/eval','B resume501..1217/eval','B resume1218..2434/eval','unified report/push/fetch','STOP'],
            durable=True,started_utc=now(),automatic3651=False,automatic4868=False,third_experiment=False)
        child=None
        try:
            for candidate in ('A','B'):
                wt=WORKTREES/SPEC[candidate]['worktree']
                command=[PYTHON,'-m','recovery.e2_candidates','--run','--candidate',candidate]
                log=QUEUE_RUN/(candidate+'.log')
                with log.open('xb') as handle:
                    child=subprocess.Popen(command,cwd=wt,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
                    dump(QUEUE_RUN/'QUEUE_STATE.json',dict(state,status='RUNNING',candidate=candidate,child_pid=child.pid,log=str(log)))
                    rc=child.wait()
                if rc:raise RuntimeError(f'Candidate{candidate} failed rc={rc}; second candidate blocked, evidence={log}')
                assert read(paths_for_worktree(candidate)/'completed.json')['status']=='COMPLETED_AND_SYNCED'
            dump(QUEUE_RUN/'QUEUE_STATE.json',dict(state,status='COMPLETED_AND_SYNCED',finished_utc=now(),no_post2434_training=True))
        except BaseException as error:
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=40)
            dump(QUEUE_RUN/'QUEUE_STATE.json',dict(state,status='FAILED',error=repr(error),finished_utc=now()));raise


def paths_for_worktree(candidate):
    return PROJECT/'runtime/SAID-nest-clip-v1'/SPEC[candidate]['directory']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',choices=('A','B'),default='A')
    for name in ('prepare','worker','run','queue','publish-setup'):p.add_argument('--'+name,action='store_true')
    args,remaining=p.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*remaining];worker(args.candidate)
    elif args.prepare:prepare(args.candidate)
    elif args.publish_setup:publish(args.candidate,setup=True)
    elif args.run:run(args.candidate)
    elif args.queue:queue()
    else:p.error('Choose an action')


if __name__=='__main__':main()
