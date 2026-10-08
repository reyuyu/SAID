"""Continue the evaluated Half@2434 trajectory, evaluate E3, then finish E4.

Only orchestration changes. The production trainer, Half schedule, data sampler,
strict exporter and validated four-GPU evaluators remain unchanged.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from recovery import e2_candidates as e

MOTHER = '7f5c59f11e09e136fea0126961a6bb47860047c9'
BRANCH = 'experiment/nested-d3-hns-half-full-v1'
DIRECTORY = 'nested_d3_hns_half_full_v1'
PARENT = e.PROJECT/'runtime/SAID-nest-clip-v1/nested_d3_hns_half2434_v1/step2434/step002434.pt'
PARENT_SHA = '1bc0e6f088f3976c79db9cb3d729967d57f9228f4098772b91f66e00696b1ca8'
SEGMENTS = [(2434,3651),(3651,4868)]
ORDER = ['restore2434', 'train2435..3651', 'strict export/evaluate/report/sync3651',
         'restore3651', 'train3652..4868', 'strict export/evaluate/report/sync4868', 'STOP']


def configure():
    e.LIMIT = 4868
    e.INITIAL_A_START = 2434
    e.ENTRY = 'recovery.hns_half_full'
    e.PUBLICATION_TITLE = 'E3/E4 continuation with epoch evaluation'
    e.PARENT, e.PARENT_SHA = PARENT, PARENT_SHA
    e.SPEC['A'] = dict(name='HNS-Half', directory=DIRECTORY, worktree='hns-half-full',
        branch=BRANCH, mother=MOTHER,
        config='experiments/nest_clip_v1/nested_d3_hns_half2434_v1/config.json',
        segments=list(SEGMENTS), source_run='nested-d3-hns-full-v1/step4868')
    e.CODE = ('recovery/e2_candidates.py', 'tests/test_e2_candidates.py',
              'recovery/hns_half_full.py', 'tests/test_hns_half_full.py',
              'tools/eval_five_parallel.py', 'tests/test_eval_five_parallel.py')


def prepare():
    import torch
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    from recovery.s02_local500 import path_proof
    torch.set_num_threads(4)
    exp,run=e.paths('A')
    assert e.git('branch','--show-current')==BRANCH
    assert e.git('merge-base',MOTHER,'HEAD')==MOTHER
    assert not run.exists() and not exp.exists(), 'Never overwrite an existing trajectory'
    require_gpu_idle({0,1,2,3})
    assert e.sha(e.STEP0)==e.STEP0_SHA
    ready=e.read(e.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    cfg=e.mother_config('A');e.assert_frozen('A',cfg)
    identity=e.checkpoint_identity(PARENT,2434,PARENT_SHA)
    old=e.read(e.ROOT/'experiments/nest_clip_v1/nested_d3_hns_half2434_v1/step2434_RESULTS.json')
    assert old['checkpoint_unchanged']
    assert identity['sha256']==old['checkpoint']['sha256']==old['strict_export']['checkpoint_sha256']
    native=code_manifest()
    assert identity['configuration']['code_sha256']==native
    e.assert_frozen('A',identity['configuration'])
    for path,digest in native.items():
        assert hashlib.sha256(subprocess.check_output(['git','show',MOTHER+':'+path],cwd=e.ROOT)).hexdigest()==digest
    evaluated=e.evaluator_sources();refs=e.references()
    refs['models']['HNS-Half']['2434']=dict(scores_percent=old['scores_percent'],metrics=old['metrics'])
    refs['sources']['half2434']=dict(commit=MOTHER,path=str(PARENT),sha256=PARENT_SHA,
        evaluated_checkpoint_immutable=True)
    exp.mkdir(parents=True);run.mkdir(parents=True)
    e.dump(exp/'config.json',cfg)
    e.dump(exp/'BASELINE_PROVENANCE.json',refs)
    e.dump(exp/'RESUME_PROVENANCE.json',dict(passed=True,parent=identity,mother_commit=MOTHER,
        production_sources=native,evaluated_sources=evaluated,common0_sha256=e.STEP0_SHA,
        fresh_common0=False,image_root=str(e.IMAGES),NFS_fallback=False,
        next_update=2435,strict_all_rank_state_restore=True,checked_utc=e.now()))
    proof=path_proof();e.dump(run/'prelaunch-local-path-proof-5000.json',proof)
    e.dump(exp/'LOCAL_ONLY_PROOF.json',dict(passed=True,count=proof['count'],NFS_fallback=False,
        evidence=str(run/'prelaunch-local-path-proof-5000.json'),sha256=e.sha(run/'prelaunch-local-path-proof-5000.json')))
    e.dump(exp/'PLAN.json',dict(segments=SEGMENTS,order=ORDER,stop=4868,next_update=2435,horizon=4868,
        parent=str(PARENT),parent_sha256=PARENT_SHA,alignment=[1.35,1.35,.3],sparsity=[1,2,2],K=3,
        hierarchy_weight=.5,hns_beta=[2,2],detach_child=False,soft_inclusion=0,
        evaluation_steps=[3651,4868],pause_training_for_evaluation=True,batch64_evaluation=True,
        GPU_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        production_math_unchanged=True,scores_do_not_change_plan=True,technical_failure_blocks_continuation=True,
        third_experiment=False,NFS_copy=False,full_audit=False,
        cache='/root disposable overlay; original NFS images retained'))
    e.set_state('A','PREPARED')


def report(results,started):
    exp,run=e.paths('A');allsteps=[];checks=[];completed=[]
    for start,stop in SEGMENTS:
        if str(stop) not in results:break
        records=e.rows(run/f'step{stop}/steps.jsonl')
        assert len(records)==stop-start
        checks.append(e.verify_stream('A',records,start));allsteps+=records;completed.append((start,stop))
        assert e.read(run/f'initial-gate-{start}.json')['passed']
        assert e.read(run/f'first-five-gate-{start}.json')['passed']
    refs=e.read(exp/'BASELINE_PROVENANCE.json');latest=max(int(k) for k in results)
    output=dict(status='COMPLETE' if latest==4868 else 'E3_EVALUATED_CONTINUING_TO_E4',
        name='HNS-Half',completed_steps=latest,new_updates=len(allsteps),models=results,
        trajectory=dict(refs['models']['HNS-Half'],**results),
        stage_comparisons={step:{name:e.delta(value,entries[step]) for name,entries in refs['models'].items() if step in entries}
            for step,value in results.items()},stream_proofs=checks,
        phase_evaluations_paused_training=True,evaluator_math_modified=False,stop=4868,no_extra_experiments=True)
    e.dump(exp/'RESULTS.json',output)
    diag=dict(last50=e.summarize(allsteps[-50:]),epoch_last50={str(stop):e.summarize(
        [r for r in allsteps if stop-50<r['step']<=stop]) for start,stop in completed},
        parent2434_last50=e.summarize(e.rows(PARENT.parent/'steps.jsonl')[-50:]),
        epoch_references=refs['epoch_diagnostics'])
    e.dump(exp/'TRAINING_DIAGNOSTICS.json',diag)
    e.dump(exp/'MASK_HIERARCHY_AUDIT.json',dict(last50=diag['last50'],epoch_last50=diag['epoch_last50']))
    original=e.SPEC['A']['segments'];e.SPEC['A']['segments']=completed
    try:runtime=e.runtime_stats('A')
    finally:e.SPEC['A']['segments']=original
    runtime['total_continuation_wall_seconds']=time.monotonic()-started
    e.dump(exp/'RUNTIME_STATS.json',runtime)
    e.dump(exp/'VALIDATION.json',dict(passed=True,streams=checks))
    e.dump(exp/'EXPORT_AUDIT.json',{step:value['strict_export'] for step,value in results.items()})
    e.dump(exp/'CHECKPOINT_INVENTORY.json',[value['checkpoint'] for value in results.values()])
    lines=['# HNS-Half E3 / E4 continuation','',
        'Exact evaluated Half@2434 parent restored. Hierarchy stays0.5; no SG; beta2/2; old soft inclusion0.',
        'All production trainer/model/data/export/evaluator sources unchanged. Complete optimizer, scheduler, RNG and loader cursor restored per rank.',
        'Training pauses at3651 for strict native five-dataset evaluation, then restores3651 and ends exactly4868.',
        'GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. Evaluation batch64 unchanged.',
        '', '| Step | Score5 | J_long3 | J_long | Short4 |','|---|---:|---:|---:|---:|']
    for step,v in sorted(output['trajectory'].items(),key=lambda x:int(x[0])):
        lines.append('| '+step+' | '+' | '.join(f'{v["scores_percent"][k]:.6f}' for k in e.KEYS)+' |')
    for step,v in results.items():
        lines+=['',f'## Step{step}','', '| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |','|---|---|---|']
        for ds,m in v['metrics'].items():
            lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
        for name,d in output['stage_comparisons'][step].items():
            lines+=['',name+' deltas(pp): `'+e.json.dumps(d['scores_delta_pp'])+'`.']
    lines+=['','All30 recall deltas: RESULTS.json. Resource statistics: RUNTIME_STATS.json. Full stream gates: VALIDATION.json.',
        'Checkpoints/bare/raw logs remain in persistent canonical runtime and are never uploaded. /root is disposable cache; NFS originals retained.',
        'Ordinary slow steps warn; true I/O/CUDA/DDP/nonfinite/OOM-kill/>60s/supervisor failures stop. No additional experiments.']
    (exp/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    from tools.eval_five_parallel import require_gpu_idle
    from train.train_nested_semantic_mask import code_manifest
    from recovery.resource_stall_v2 import system_snapshot
    exp,runtime=e.paths('A')
    def interrupt(sig,frame):raise RuntimeError('Supervisor interrupted; preserve progress')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    with (runtime/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert e.read(exp/'STATE.json')['status']=='PREPARED','No implicit retry'
        assert e.read(exp/'CPU_TESTS.json')['passed']
        assert e.read(exp/'RESUME_PROVENANCE.json')['production_sources']==code_manifest()
        assert e.sha(PARENT)==PARENT_SHA
        e.evaluator_sources();assert not system_snapshot()['memory_events'].get('oom_kill',0)
        e.dump(runtime/'LAUNCH.json',dict(pid=os.getpid(),session_id=os.getsid(0),durable=True,order=ORDER,
            branch=BRANCH,git_commit=e.git('rev-parse','HEAD'),started_utc=e.now()))
        supervisor=e.Supervisor('A');started=time.monotonic();results={}
        try:
            for start,stop in SEGMENTS:
                require_gpu_idle({0,1,2,3})
                phase=e.IMAGES.parent/f'e2-A-to{stop}-phase';phase.mkdir(exist_ok=False)
                parent=e.parent_for_segment('A',start)
                identity=e.checkpoint_identity(parent,start,PARENT_SHA if start==2434 else results[str(start)]['checkpoint']['sha256'])
                e.dump(exp/f'RESUME_FROM_{start}.json',identity)
                e.dump(runtime/'active-segment.json',dict(start=start,stop=stop,phase=str(phase)))
                e.set_state('A','TRAINING',start=start,stop=stop,phase=str(phase))
                supervisor.execute(f'train{stop}',e.training_command('A',start,stop),phase=phase,training=True)
                acceptance=e.read(runtime/f'step{stop}/acceptance.json')
                assert acceptance['passed'] and all(r['completed_updates']==stop and r['max_parameter_difference_from_rank0']==0 for r in acceptance['ranks'])
                stream=e.verify_stream('A',e.rows(runtime/f'step{stop}/steps.jsonl'),start)
                e.dump(exp/f'step{stop}_STREAM_PROOF.json',stream)
                results[str(stop)]=supervisor.evaluate(stop,runtime/f'step{stop}/step{stop:06d}.pt')
                e.dump(exp/'PROGRESS.json',dict(models=results,stop=4868,scores_do_not_change_plan=True))
                report(results,started)
                e.set_state('A','COMPLETE' if stop==4868 else 'E3_EVALUATED',completed_steps=stop)
                e.publish('A')
            require_gpu_idle({0,1,2,3})
            e.dump(runtime/'completed.json',dict(status='COMPLETED_AND_SYNCED',stop=4868,finished_utc=e.now()))
        except BaseException as error:
            e.set_state('A','FAILED',error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepare','worker','run','publish-setup','launch'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--candidate',choices=('A',),default='A')
    args,remaining=p.parse_known_args();configure()
    if args.worker:sys.argv=[sys.argv[0],*remaining];e.worker('A')
    elif args.prepare:prepare()
    elif args.run:run()
    elif args.publish_setup:e.publish('A',setup=True)
    elif args.launch:
        exp,runtime=e.paths('A')
        assert e.read(exp/'STATE.json')['status']=='PREPARED'
        receipt=e.read(runtime/'SETUP_GITHUB_RECEIPT.json')
        assert receipt['remote_HEAD_matches_local'] and receipt['commit']==e.git('rev-parse','HEAD')
        command=[e.PYTHON,'-m','recovery.hns_half_full','--run']
        log=runtime/'runner.log'
        with log.open('xb') as handle:
            child=subprocess.Popen(command,cwd=e.ROOT,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        launch=dict(pid=child.pid,log=str(log),command=command,order=ORDER,
            detached_session=True,created_utc=e.now())
        e.dump(runtime/'DETACHED_LAUNCH.json',launch)
        print(e.json.dumps(launch),flush=True)
    else:p.error('Choose action')


if __name__=='__main__':main()
