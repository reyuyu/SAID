"""One frozen S12 trajectory: resume500->1217, evaluate, resume1217->2434, stop."""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from recovery import hns_macro_fourarm as common, s02_local500 as local
from recovery.s02_local_full import stream, expected_lrs
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now, distribution

PROJECT=Path('/opt/data/private/lklk/SAID')
BRANCH='experiment/hns-s12-2434-validation-v1'
MOTHER='origin/experiment/hns-static-strength-twoarm500-v1'
BASE_EXP=ROOT/'experiments/nest_clip_v1/hns_static_strength_twoarm500_v1/E1-HNS-S12'
BASE_RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-static-strength-twoarm500-v1/E1-HNS-S12'
PARENT=BASE_RUN/'step500/step000500.pt'
PARENT_SHA='cc0b27c871f717c8eb821b1e7a425a8e70ba82f771ee82f0061ca2e6652a172d'
EXP=ROOT/'experiments/nest_clip_v1/hns_s12_2434_validation_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-2434-validation-v1'
ENTRY='recovery.hns_s12_2434_validation'
TARGETS=(1217,2434)
REFERENCE_BRANCH='origin/analysis/d3-hns-epoch-curve-eval-v1'
REFERENCE_FILE='experiments/nest_clip_v1/epoch_curve_eval_v1/RESULTS.json'
CODE=('recovery/hns_s12_2434_validation.py','recovery/hns_macro_gradient.py','tests/test_hns_s12_2434_validation.py')


def phase(target):return local.IMAGES.parent/f'formal-hns-s12-2434-validation-v1-{target}'


def segment(target):
    assert target in TARGETS
    return RUN/f'step{target}'


def predecessor(target):
    assert target in TARGETS
    return (PARENT,500) if target==1217 else (segment(1217)/'training/step001217.pt',1217)


def configure(target):
    local.RUN=segment(target);local.PHASE=phase(target);local.CONFIG=EXP/'config.json'


def state(status,**extra):
    dump(EXP/'STATE.json',dict(status=status,updated_utc=now(),runner_pid=os.getpid(),
        targets=list(TARGETS),stop_updates=2434,automatic_full=False,other_arms=False,**extra))


def frozen(cfg):
    expected=common.read(BASE_EXP/'config.json')
    assert all(cfg.get(k)==v for k,v in expected.items()),'S12 config drift'
    assert tuple(cfg[k] for k in common.MACRO_KEYS)==(10.,1.2,1.)
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0 and cfg['sampling_mode']=='nested_detail_d3'
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg.get('view_sparsity_weights',[1,2,2])==[1,2,2]
    assert cfg.get('hns_beta',[2,2])==[2,2] and not cfg.get('hns_detach_child',False)
    assert not cfg.get('hns_half_after500',False)


def identity(path,completed):
    import torch
    from train.train_nested_semantic_mask import code_manifest
    p=torch.load(path,map_location='cpu',weights_only=False);frozen(p['config'])
    assert p['completed_steps']==p['global_step']==p['scheduler']['completed_steps']==completed
    assert p['scheduler_horizon']==p['scheduler']['horizon']==p['config']['horizon']==4868
    cursor=dict(next_epoch=completed//1217,next_batch=completed%1217)
    assert p['data_cursor']==cursor and p['next_epoch']==cursor['next_epoch'] and p['next_batch']==cursor['next_batch']
    assert p['sampler']==dict(type='DistributedSampler',seed=0,world_size=4,batch_size=256,batches_per_epoch=1217)
    assert p['adapter'] is not None and p['optimizer']['state']
    assert {int(v['step']) for v in p['optimizer']['state'].values()}=={completed}
    assert code_manifest()==p['config']['code_sha256'],'Original production code must remain byte-identical'
    assert len(p['rng_per_rank'])==4
    for s in p['rng_per_rank']:
        assert set(('python','numpy','cpu','cuda','loader_generator'))<=set(s)
    for component in (p['model'],p['adapter']):
        assert all(torch.isfinite(t).all() for t in component.values())
    for s in p['optimizer']['state'].values():
        assert all(torch.isfinite(t).all() for t in s.values())
    return dict(passed=True,path=str(path),sha256=sha(path),size_bytes=Path(path).stat().st_size,
        completed_steps=completed,scheduler_horizon=4868,scheduler=p['scheduler'],data_cursor=cursor,sampler=p['sampler'],
        optimizer_steps=[completed],optimizer_groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],
            parameter_count=len(g['params'])) for g in p['optimizer']['param_groups']],
        rng_per_rank=[dict(rank=r,keys=sorted(s),**{k+'_sha256':hashlib.sha256(s[k].numpy().tobytes()).hexdigest()
            for k in ('cpu','cuda','loader_generator')}) for r,s in enumerate(p['rng_per_rank'])],
        sources=p['config']['code_sha256'],git_head=p['config']['git_head'],uploaded=False)


def resume_reference(completed):
    """CPU-only next five batches from the unchanged sampler; no image decoding."""
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(local.INDEX,local.IMAGES,'nested_detail_d3',0)
    epoch,batch=divmod(completed,1217);out={};before=torch.get_rng_state().clone()
    offsets=np.load(local.INDEX/'offsets.npy',mmap_mode='r')
    with (local.INDEX/'records.jsonl').open('rb') as h:
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
            sampler.set_epoch(epoch)
            ids=list(itertools.islice(iter(sampler),batch*256,(batch+5)*256))
            for i in range(5):
                chosen=ids[i*256:(i+1)*256];views=[]
                for index in chosen:
                    h.seek(int(offsets[index]));rec=json.loads(h.read(int(offsets[index+1]-offsets[index])))
                    views.append(sampled_text_views(rec['caption'],'nested_detail_d3',0,epoch,index+1000))
                b=dict(sample_id=torch.tensor([index+1000 for index in chosen]),views=[v['views'] for v in views])
                b.update({k:torch.stack([v[k] for v in views]) for k in ('tokens_f','tokens_o','tokens_e')})
                out[f'{completed+i+1}:{rank}']=dict(stream_sha256=stream(b),sample_ids=b['sample_id'].tolist(),
                    detail_indices=[v['detail_indices'] for v in views],epoch=epoch,batch_cursor=batch+i)
    assert torch.equal(before,torch.get_rng_state())
    return dict(passed=True,parent_updates=completed,records=out,training_RNG_untouched=True)


def training_command(target):
    parent,_=predecessor(target)
    return common.torchrun(ENTRY,'--worker','--target',target,'--config',EXP/'config.json',
        '--init-state',STEP0,'--resume',parent,'--index-dir',local.INDEX,'--image-root',local.IMAGES,
        '--output-dir',segment(target)/'training','--run-type','formal','--max-updates',target)


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0,1,2,3});assert common.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'No overwrite/retry'
    proof=identity(PARENT,500);assert proof['sha256']==PARENT_SHA and sha(STEP0)==STEP0_SHA
    old=common.read(BASE_EXP/'RESULTS.json')
    assert old['completed_steps']==500 and old['checkpoint']['sha256']==PARENT_SHA
    assert old['checkpoint_unchanged'] and old['strict_export']['checkpoint_sha256']==PARENT_SHA
    mother=common.git('rev-parse',MOTHER)
    assert proof['git_head']==common.read(BASE_EXP/'FORMAL_PROVENANCE.json')['git_head']
    subprocess.run(['git','merge-base','--is-ancestor',proof['git_head'],mother],cwd=ROOT,check=True)
    for path,digest in proof['sources'].items():
        for commit in (mother,proof['git_head']):
            assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)).hexdigest()==digest
    ready=common.read(local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    common.evaluator_proof();EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'config.json',common.read(BASE_EXP/'config.json'))
    dump(EXP/'RESUME_PROVENANCE.json',dict(**proof,mother_commit=mother,evaluated_checkpoint_immutable=True,
        local_only=True,image_root=str(local.IMAGES),NFS_fallback=False,production_source_changes=[]))
    reference=json.loads(subprocess.check_output(['git','show',REFERENCE_BRANCH+':'+REFERENCE_FILE],cwd=ROOT))
    assert reference['status']=='COMPLETE'
    selected={m:{str(n):reference['models'][m][str(n)] for n in TARGETS} for m in ('HNS_v1','D3_Balanced')}
    dump(EXP/'REFERENCES.json',dict(branch=REFERENCE_BRANCH,commit=common.git('rev-parse',REFERENCE_BRANCH),models=selected))
    for n in TARGETS:
        (EXP/f'step{n}').mkdir();segment(n).mkdir()
        dump(segment(n)/'resume-reference.json',resume_reference(predecessor(n)[1]))
    paths=local.path_proof();dump(RUN/'local-path-proof-5000.json',paths)
    dump(EXP/'LOCAL_ONLY_PROOF.json',dict(passed=paths['passed'],count=paths['count'],image_root=str(local.IMAGES),
        NFS_fallback=False,raw_path=str(RUN/'local-path-proof-5000.json')))
    dump(EXP/'PLAN.json',dict(order=['resume500->1217','audit/export/verify/evaluate/report1217',
        'resume1217->2434','audit/export/verify/evaluate/report2434','STOP'],targets=list(TARGETS),horizon=4868,
        macro=[10,1.2,1],view_weights=[1.35,1.35,.3],sparsity=[1,2,2],beta=[2,2],no_SG=True,
        soft_inclusion=0,K=3,ramp200_unchanged=True,local_only=True,full_restore=True,
        cache='Disposable /root Docker overlay; retain persistent NFS originals',other_arms=False,automatic3651=False,automatic4868=False))
    state('PREPARED')


def worker(target):
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate, training_phase_timing as timing
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_train_gate import require_live_supervisor
    from recovery.nested_d3_local_search import observe_selection
    from recovery.hns_static_strength_twoarm import endpoint_streak
    from model.balanced_hparam_search import BalancedSearch
    configure(target);parent,start=predecessor(target);runtime=segment(target)
    require_live_supervisor(int(os.environ['SAID_FULL_SUPERVISOR_PID']))
    context=dict(previous=None,optimizer=None,checked=False)
    original_validate=trainer.validate_resume_payload;original_optimizer=trainer.build_optimizer
    original_rates=trainer.optimizer_learning_rates;original_save=trainer.atomic_save
    original_forward=BalancedSearch.forward;endpoint=None;streak=0
    def validate(p,c,*args):
        frozen(c);assert c['max_updates']==target and c['resume']==str(parent)
        assert p['completed_steps']==start and p['config']['code_sha256']==c['code_sha256']
        assert p['scheduler']['completed_steps']==start and p['scheduler']['horizon']==4868
        assert p['data_cursor']==dict(next_epoch=start//1217,next_batch=start%1217)
        assert sha(parent)==common.read(runtime/'parent-identity.json')['sha256']
        context['previous']=p
        return original_validate(p,c,*args)
    def optimizer(module):
        context['optimizer']=original_optimizer(module);return context['optimizer']
    def save(p,path):
        if 'model' in p and 'optimizer' in p:
            gate.RUN=RUN;gate.resumable_metadata(p)
            p['resume_lineage']=dict(parent=str(parent),sha256=common.read(runtime/'parent-identity.json')['sha256'],
                first_update=start+1,root_checkpoint=str(PARENT),root_sha256=PARENT_SHA)
        return original_save(p,path)
    def rates(module,completed,horizon):
        require_live_supervisor(int(os.environ['SAID_FULL_SUPERVISOR_PID']))
        assert horizon==4868 and start<=completed<target
        if completed==start and not context['checked']:
            p=context['previous'];rank=dist.get_rank();rng=p['rng_per_rank'][rank]
            assert gate.exact_state(module.clip.state_dict(),p['model'])
            assert gate.exact_state(trainer.auxiliary_module(module).state_dict(),p['adapter'])
            assert gate.exact_state(context['optimizer'].state_dict(),p['optimizer'])
            assert gate.exact_state(trainer.rng_state(),{k:rng[k] for k in ('python','numpy','cpu','cuda')})
            assert torch.equal(module._loader_epoch_generator_state,rng['loader_generator'])
            diff=trainer.parameter_agreement(module);proofs=[None]*4
            dist.all_gather_object(proofs,dict(rank=rank,model_adapter_optimizer_exact=True,RNG_exact=True,
                loader_generator_exact=True,parameter_difference=diff,data_cursor=p['data_cursor'],optimizer_counters=[start]))
            assert all(r['parameter_difference']==0 for r in proofs)
            if rank==0:dump(runtime/'RESTORE_AUDIT.json',dict(passed=True,next_update=start+1,ranks=proofs))
            context.update(checked=True,previous=None)
        values=original_rates(module,completed,horizon)
        if completed<start+5:
            assert list(values)==expected_lrs(completed,common.read(EXP/'config.json'))
            dump(runtime/f'lr-{completed+1}-rank{dist.get_rank()}.json',dict(passed=True,step=completed+1,lrs=list(values)))
        return values
    def guarded_forward(module,*args,**kwargs):
        nonlocal endpoint,streak
        loss,logs=original_forward(module,*args,**kwargs)
        endpoint,streak=endpoint_streak(logs,endpoint,streak)
        if streak>=5:
            if dist.get_rank()==0:dump(runtime/'MASK_FAILURE.json',dict(endpoint=endpoint,consecutive=streak))
            raise RuntimeError('HARD_STOP: all three supports at same endpoint for five updates')
        return loss,logs
    class HeartbeatIterator(timing.TimedIterator):
        def __init__(self,iterator,recorder):
            super().__init__(iterator,recorder);self.position=0;self.epoch=iterator._dataset.epoch
        def __next__(self):
            self.recorder.begin();self.position+=1;step=self.epoch*1217+self.position
            stamp=time.monotonic();path=phase(target)/f'heartbeat-rank{self.recorder.rank}.json'
            event=dict(rank=self.recorder.rank,step=step,started_monotonic=stamp,utc=now(),replay_without_update=step<=start)
            dump(path,dict(event,state='DATA_WAIT'))
            try:value=next(self.iterator)
            except StopIteration:
                self.recorder.current=None;dump(path,dict(event,state='EXHAUSTED'));raise
            self.recorder.current['step']=step;self.recorder.current['data_wait_s']=time.monotonic()-stamp
            if start<step<=start+5:
                expected=common.read(runtime/'resume-reference.json')['records'][f'{step}:{self.recorder.rank}']
                assert stream(value)==expected['stream_sha256']
                assert value['sample_id'].tolist()==expected['sample_ids'] and value['detail_indices']==expected['detail_indices']
                dump(runtime/f'batch-{step}-rank{self.recorder.rank}.json',dict(passed=True,step=step,rank=self.recorder.rank,
                    epoch=self.epoch,batch_cursor=self.position-1,stream_sha256=expected['stream_sha256'],IDs_text_tokens_indices_exact=True))
            dump(path,dict(event,state='ACTIVE_STEP'));return value
    trainer.validate_resume_payload=validate;trainer.build_optimizer=optimizer;trainer.optimizer_learning_rates=rates
    trainer.atomic_save=save;trainer.sampling_diagnostics=observe_selection;trainer.NestedDataset=local.LoggedLocalDataset
    BalancedSearch.forward=guarded_forward;original_loader=trainer.DataLoader
    trainer.DataLoader=lambda *args,**kwargs:original_loader(*args,**kwargs,timeout=60)
    timing.TimedIterator=HeartbeatIterator;recorder=timing.install(str(phase(target)),int(os.environ['RANK']))
    try:trainer.main()
    finally:
        recorder.finish()
        if dist.is_initialized():dist.destroy_process_group()
    dump(phase(target)/f'heartbeat-rank{os.environ["RANK"]}.json',dict(state='TRAINING_COMPLETE',utc=now()))


def resume_gate(target):
    runtime=segment(target);start=predecessor(target)[1]
    restored=common.read(runtime/'RESTORE_AUDIT.json');assert restored['passed']
    batches=[common.read(runtime/f'batch-{s}-rank{r}.json') for s in range(start+1,start+6) for r in range(4)]
    lrs=[common.read(runtime/f'lr-{s}-rank{r}.json') for s in range(start+1,start+6) for r in range(4)]
    assert all(v['passed'] for v in batches+lrs)
    proof=dict(passed=True,restore=restored,batches=batches,lrs=lrs,records=5120)
    dump(EXP/f'step{target}/RESUME_GATE.json',proof);return proof


def evaluate(supervisor,target):
    from tools.eval_five_parallel import require_gpu_idle
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short
    require_gpu_idle({0,1,2,3});common.evaluator_proof();runtime=segment(target);dest=EXP/f'step{target}'
    checkpoint=runtime/f'training/step{target:06d}.pt';proof=identity(checkpoint,target)
    out=runtime/'evaluations';out.mkdir();bare=out/f'student_step{target}.pt';python=PROJECT/'.venv/bin/python'
    supervisor.execute('export',[str(python),'-m','tools.nest_clip','export','--checkpoint',str(checkpoint),
        '--output',str(bare),'--expect-updates',str(target)])
    supervisor.execute('verify',[str(python),'-m','tools.nest_clip','verify-export','--checkpoint',str(checkpoint),
        '--bare',str(bare),'--output',str(out/'export-check.json'),'--index-dir',str(local.INDEX),'--image-root',str(local.IMAGES)])
    check=common.read(out/'export-check.json')
    assert check['passed'] and check['strict_load'] and check['optimizer_steps']==[target]
    assert check['image_max_abs']==check['text_max_abs']==0
    assert check['checkpoint_sha256']==proof['sha256']
    state('EVALUATING',target=target)
    supervisor.execute('five-eval',[str(python),'-m','tools.eval_five_parallel','--checkpoint',str(bare),
        '--training-checkpoint',str(checkpoint),'--output-dir',str(out)])
    metrics,raw,_=native_metrics(out);aggregate=with_short(dict(metrics=metrics,scores=scores(metrics)))
    percent={k:100*aggregate[{'Score5':'Score5_R1','Short4':'Short4_R1'}.get(k,k)] for k in ('Score5','J_long3','J_long','Short4')}
    assert sha(checkpoint)==proof['sha256'] and sha(bare)==check['bare_sha256']
    result=dict(completed_steps=target,metrics=metrics,scores_percent=percent,checkpoint=proof,strict_export=check,
        bare=dict(path=str(bare),sha256=check['bare_sha256'],uploaded=False),evaluation_checkpoint_immutable=True)
    refs=common.read(EXP/'REFERENCES.json')['models']
    result['comparisons']={m:common.compare(result,refs[m][str(target)]) for m in refs}
    for ds,value in raw.items():dump(dest/'evaluations'/f'{ds}.json',value)
    dump(dest/'evaluations/EVAL_PARALLEL_RUN.json',common.read(out/'EVAL_PARALLEL_RUN.json'))
    dump(dest/'RESULTS.json',result);dump(dest/'EXPORT_AUDIT.json',check);return result


def report(target,supervisor,result):
    from recovery.nested_d3_local_search_evidence import diagnostics
    runtime=segment(target);dest=EXP/f'step{target}';start=predecessor(target)[1]
    records=rows(runtime/'training/steps.jsonl');assert [r['step'] for r in records]==list(range(start+1,target+1))
    for row in records:
        assert row['nonfinite']==0 and row['HNS_enabled'] and row['inc_weight']==row['inclusion_loss']==0
        assert row['lambda_h']==1 and tuple(row['macro_'+k] for k in common.MACRO_KEYS)==(10,1.2,1)
        expected=sum(row['macro_weighted_'+k] for k in ('align','sparse','hierarchy'))
        assert math.isclose(row['loss'],expected,abs_tol=2e-6,rel_tol=4e-6)
        assert row['actual_lrs']==dict(zip(('backbone','text_mask_and_shared_pool','visual_mask','fusion_adapter'),expected_lrs(row['step']-1,common.read(EXP/'config.json'))))
    from recovery import nested_d3_local_search as search
    search.ARMS['S12']=dict(mode='nested_detail_d3',weights=[1.35,1.35,.3])
    diag,masks=diagnostics(records,'S12');diag['last50_steps']=[r['step'] for r in records[-50:]]
    keys=[k for k in records[0] if k.startswith(('HNS_','macro_')) or k in ('loss','lambda_h')]
    diag['last50_loss_components']={k:statistics.fmean(float(r[k]) for r in records[-50:]) for k in keys}
    masks['keep_ratios']={v:t['keep_ratio'] for v,t in diag['last50_views'].items()}
    gradient=common.read(dest/'GRADIENT_AUDIT.json');assert gradient['passed'] and gradient['checkpoint']['unchanged']
    cycles=rows(runtime/'training/cycle_timing.jsonl');waits={}
    for rank in range(4):
        for row in rows(phase(target)/f'rank{rank}.jsonl'):
            if start<row['step']<=target:waits[row['step']]=max(waits.get(row['step'],0),row['data_wait_s'])
    systems=[r['system'] for r in rows(runtime/'resource-telemetry.jsonl')]
    stats=dict(full_cycle_seconds=distribution([r['four_rank_max_seconds'] for r in cycles]),
        data_wait_seconds=distribution(list(waits.values())),steps_gt3s=sum(r['four_rank_max_seconds']>3 for r in cycles),
        steps_gt10s=sum(r['four_rank_max_seconds']>10 for r in cycles),peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),
        peak_file_cache_bytes=max(s['file'] for s in systems),oom_kill=max(s['memory_events'].get('oom_kill',0) for s in systems),
        GPU_peak_GiB={str(rank):max(h['peak_allocated_gib'] for row in records for h in row['rank_health'] if h['rank']==rank) for rank in range(4)},
        commands=supervisor.commands,local_only=True,updates_this_segment=target-start)
    for kind in ('io_PSI','memory_PSI'):
        stats[kind]={scope:distribution([s[kind][scope]['avg10'] for s in systems]) for scope in ('some','full')}
    assert stats['oom_kill']==0
    paths=[p for f in phase(target).glob('image-paths-*.jsonl') for p in rows(f)]
    assert paths and all(Path(p['actual_path']).is_relative_to(local.IMAGES) and not p['NFS_fallback'] for p in paths)
    for name,value in [('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),('RUNTIME_STATS',stats)]:dump(dest/(name+'.json'),value)
    dump(dest/'VALIDATION.json',dict(passed=True,acceptance=common.read(runtime/'training/acceptance.json'),
        resume_gate=resume_gate(target),frozen_macros_all_updates=True,local_only=True,strict_export=result['strict_export']))


def combined(completed):
    refs=common.read(EXP/'REFERENCES.json')['models'];results={str(n):common.read(EXP/f'step{n}/RESULTS.json') for n in completed}
    dump(EXP/'RESULTS.json',dict(status='COMPLETED' if len(completed)==2 else 'PARTIAL',nodes=results,references=refs,
        frozen_macro=[10,1.2,1],stop_updates=2434,automatic_continuation=False))
    lines=['# Frozen HNS-S12 continuation:500 ->1217 ->2434','',
        '| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |','|---:|---|---:|---:|---:|---:|---|']
    for n in completed:
        for model,value in [('HNS-v1',refs['HNS_v1'][str(n)]),('D3 Balanced',refs['D3_Balanced'][str(n)]),('HNS-S12',results[str(n)])]:
            q=common.quality(value);lines.append(f'| {n} | {model} | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} |')
        lines+=['',f'## Step{n} deltas and interpretation','']
        for model,c in results[str(n)]['comparisons'].items():
            d=c['quality_delta_pp'];r=c['recall_delta_pp'];lines.append(f'{model}: '+json.dumps(d)+'. Long-DCI directional R1 deltas: '+json.dumps({dr:r['Long-DCI'][dr]['R@1'] for dr in ('I2T','T2I')})+'.')
            lines.append(f'Short4 benefit retained vs {model}: {d["Short4"]>0}; Urban T2I decrease: {d["Urban_T2I"]<0}; Long-DCI R1 decreases in either direction: '+str(any(r['Long-DCI'][dr]['R@1']<0 for dr in ('I2T','T2I')))+'.')
        lines+=['','| Dataset | I2T R1 / R5 / R10 (%) | T2I R1 / R5 / R10 (%) |','|---|---|---|']
        for ds,directions in results[str(n)]['metrics'].items():
            values=[' / '.join(f'{100*directions[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I')]
            lines.append(f'| {ds} | {values[0]} | {values[1]} |')
        masks=common.read(EXP/f'step{n}/MASK_HIERARCHY_AUDIT.json')
        lines+=['','Last50 mask telemetry: '+json.dumps(masks)+'.']
        lines+=['','Full30 recalls and deltas, last50 keep/violations/losses, component/group gradients and cosines are in node JSONs.','']
    lines+=['All production source hashes match the evaluated S12@500 checkpoint. Macro10/1.2/1 is static; native horizon4868/ramp200 and original method/optimizer/data are frozen.',
        'Both segment starts audit exact model/adapter/optimizer/CPU-CUDA-Python-NumPy RNG/loader generator and cursor, plus next five samples/text/tokens/indices/LR. Evaluation starts only after training and gradient processes exit.',
        'Training images are local-only disposable /root overlay; persistent NFS originals are retained. Checkpoints/bare/raw logs are never uploaded.',
        'Only this S12 trajectory is authorized. Stop2434; no3651/4868 or other arms.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(setup=False):
    from recovery import check_stage500_publish as checker
    assert common.git('branch','--show-current')==BRANCH
    allowed=list(CODE)+[str(p.relative_to(ROOT)) for p in EXP.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
    assert set(common.git('diff','--cached','--name-only').splitlines())<=set(allowed)
    subprocess.run(['git','add','--',*allowed],cwd=ROOT,check=True)
    if common.git('diff','--cached','--name-only'):
        previous=Path.cwd();os.chdir(ROOT)
        try:checker.ALLOWED=set(allowed);assert checker.inspect()['passed']
        finally:os.chdir(previous)
        subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',('Prepare' if setup else 'Report')+' frozen HNS-S12 validation to2434'],cwd=ROOT,check=True)
    head=common.git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    assert head==common.git('rev-parse','origin/'+BRANCH)==common.git('rev-parse','FETCH_HEAD')
    dump(RUN/('SETUP_GITHUB_RECEIPT.json' if setup else 'GITHUB_RECEIPT.json'),dict(passed=True,branch=BRANCH,
        commit=head,remote_HEAD=head,remote_HEAD_matches_local=True,checked_utc=now()))


def run():
    from tools.eval_five_parallel import require_gpu_idle
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Supervisor interrupted; preserve evidence')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert common.read(EXP/'STATE.json')['status']=='PREPARED' and common.read(EXP/'CPU_TESTS.json')['passed']
        completed=[]
        try:
            for target in TARGETS:
                require_gpu_idle({0,1,2,3});configure(target);phase(target).mkdir(exist_ok=False)
                parent,start=predecessor(target);proof=identity(parent,start)
                assert start!=500 or proof['sha256']==PARENT_SHA
                dump(segment(target)/'parent-identity.json',proof)
                supervisor=local.Supervisor();state('TRAINING',target=target,parent_updates=start,completed_nodes=completed)
                supervisor.execute('train',training_command(target),training=True)
                accept=common.read(segment(target)/'training/acceptance.json')
                assert accept['passed'] and all(r['completed_updates']==target and r['updates_this_run']==target-start and r['max_parameter_difference_from_rank0']==0 for r in accept['ranks'])
                resume_gate(target);require_gpu_idle({0,1,2,3});state('GRADIENT_AUDIT',target=target)
                supervisor.execute('gradient',common.torchrun('recovery.hns_macro_gradient','--checkpoint',segment(target)/f'training/step{target:06d}.pt',
                    '--expect-updates',target,'--output',EXP/f'step{target}/GRADIENT_AUDIT.json'))
                result=evaluate(supervisor,target);report(target,supervisor,result);completed.append(target);combined(completed)
                require_gpu_idle({0,1,2,3});state('NODE_COMPLETED' if target==1217 else 'COMPLETED_GPU_IDLE',target=target,completed_nodes=completed)
                publish()
            dump(RUN/'completed.json',dict(status='COMPLETED_AND_SYNCED',nodes=completed,stop=2434,GPU_idle=True,finished_utc=now()))
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',completed_nodes=completed,error=repr(error),automatic_retry=False);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=int,choices=TARGETS,default=1217)
    for flag in ('prepare','worker','run','launch','publish-setup'):p.add_argument('--'+flag,action='store_true')
    args,remaining=p.parse_known_args()
    if args.worker:sys.argv=[sys.argv[0],*remaining];worker(args.target)
    elif args.prepare:prepare()
    elif args.run:run()
    elif args.publish_setup:publish(True)
    elif args.launch:
        assert common.read(EXP/'STATE.json')['status']=='PREPARED' and common.read(EXP/'CPU_TESTS.json')['passed']
        assert common.read(RUN/'SETUP_GITHUB_RECEIPT.json')['commit']==common.git('rev-parse','HEAD')
        with (RUN/'runner.log').open('xb') as log:
            child=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m',ENTRY,'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,session=child.pid,durable=True,targets=list(TARGETS),log=str(RUN/'runner.log'),started_utc=now())
        dump(RUN/'DETACHED_LAUNCH.json',launch);print(json.dumps(launch),flush=True)
    else:p.error('Choose action')


if __name__=='__main__':main()
