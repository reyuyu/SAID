"""Resume the evaluated HNS-v1 trajectory, stop4868, native evaluation, publish.

Only control and evidence wrappers; the production objective/loader are unchanged.
Large raw evidence/checkpoints stay on persistent NFS; images remain local-only.
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

from recovery import s02_local_full as full
from recovery import s02_local500 as local
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, PYTHON, now, dump, rows, sha, distribution
from recovery.s02_full_stage import LOCAL, IMAGES

ENTRY = 'recovery.hns_full'
BRANCH = 'experiment/nested-d3-hns-full-v1'
BASE_SHA = 'bdfd647f7b7e1a2518e4b65a8a959bc65c5606ad'
BASE_EXP = ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1'
BASE_RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS'
PARENT = BASE_RUN/'step500/step000500.pt'
PARENT_SHA = 'efb161ba1b5847bc919750821c421595afaac4490676c0dad20790e21f77bada'
EXP = ROOT/'experiments/nest_clip_v1/nested_d3_hns_full_v1'
RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1'
TRAIN = RUN/'step4868'
REVIEW = RUN/'reviewed'
PHASE = LOCAL/'formal-hns-full-v1-phase'
INDEX = LOCAL/'data_index'
CONFIG = EXP/'config.json'
MAIN_LOG = RUN/'runner.log'
MIGRATED = {'model/hard_nested_sparsity.py', 'model/balanced_hparam_search.py',
            'train/train_nested_semantic_mask.py'}
STATIC = ('config.json', 'PLAN.json', 'RESUME_PROVENANCE.json', 'CPU_TESTS.json')
REPORTS = ('FULL_RESULTS.md', 'FULL_RESULTS.json', 'FULL_RUNTIME_STATS.json',
           'TRAINING_DIAGNOSTICS.json', 'MASK_HIERARCHY_AUDIT.json', 'VALIDATION.json',
           'RESUME_GATE.json')
CODE = ('recovery/hns_full.py', 'tests/test_hns_full.py', 'recovery/check_stage500_publish.py')
SCORE_KEYS = ('Score5', 'J_long3', 'J_long', 'Short4')


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=120).strip()


def configure():
    full.RUN, full.PARENT, full.PARENT_SHA = RUN, PARENT, PARENT_SHA
    full.PHASE, full.REVIEW, full.FINAL, full.CONFIG = PHASE, REVIEW, TRAIN, CONFIG
    local.RUN, local.PHASE, local.CONFIG = RUN, PHASE, CONFIG


def frozen_config(config):
    expected = read(BASE_EXP/'config.json')
    assert all(config.get(k) == v for k, v in expected.items()), 'HNS-v1 frozen config drift'
    assert config.get('hns_beta', [2., 2.]) == [2., 2.], 'Not HNS-v1 beta2/2'
    assert config.get('view_sparsity_weights', [1., 2., 2.]) == [1., 2., 2.]
    assert config['view_weights'] == [1.35, 1.35, .30]
    assert config['sampling_mode'] == 'nested_detail_d3'
    assert config['hns_enabled'] and config['inclusion_max'] == 0


def validate_resume(previous, current, proof, original):
    """Strict native validator after an explicitly pinned, equivalent code migration.

    Do not mutate the payload: real predecessor hashes remain in child provenance.
    Default beta2/2 was compared to fetched original HNS-v1 loss/every gradient.
    """
    frozen_config(current)
    assert current['resume'] == str(PARENT)
    assert previous['completed_steps'] == previous['global_step'] == 500
    assert previous['trajectory_root'] == str(BASE_RUN)
    assert previous['data_cursor'] == dict(next_epoch=0, next_batch=500)
    assert current['max_updates'] == 4868 and current['horizon'] == 4868
    old_code = previous['config']['code_sha256']
    new_code = current['code_sha256']
    assert old_code == proof['predecessor_sources'] and new_code == proof['current_sources']
    changed = {p for p in old_code if old_code[p] != new_code[p]}
    assert changed == MIGRATED == set(proof['audited_equivalent_changes'])
    for p in changed:
        assert proof['audited_equivalent_changes'][p] == [old_code[p], new_code[p]]
    normalized = dict(previous, config=dict(previous['config'], code_sha256=new_code))
    return original(normalized, current)


def identity():
    import torch
    from train.train_nested_semantic_mask import code_manifest
    result = read(BASE_EXP/'RESULTS.json')
    actual = sha(PARENT)
    assert actual == PARENT_SHA == result['checkpoint_sha256'] == result['strict_export']['checkpoint_sha256']
    assert result['evaluation_checkpoint_immutable'] and result['step'] == 500
    payload = torch.load(PARENT, map_location='cpu', weights_only=False)
    frozen_config(payload['config'])
    assert payload['completed_steps'] == payload['global_step'] == 500
    assert payload['scheduler_horizon'] == payload['scheduler']['horizon'] == 4868
    assert payload['next_epoch'] == 0 and payload['next_batch'] == 500
    assert payload['data_cursor'] == dict(next_epoch=0, next_batch=500)
    assert len(payload['rng_per_rank']) == 4 and payload['adapter'] is not None
    assert {int(s['step']) for s in payload['optimizer']['state'].values()} == {500}
    current, old = code_manifest(), payload['config']['code_sha256']
    assert set(current) == set(old)
    assert {p for p in old if old[p] != current[p]} == MIGRATED
    for p, digest in old.items():
        assert hashlib.sha256(subprocess.check_output(['git','show',BASE_SHA+':'+p], cwd=ROOT)).hexdigest() == digest
    for p, digest in current.items():
        assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p], cwd=ROOT)).hexdigest() == digest
    for state in payload['rng_per_rank']:
        assert all(k in state for k in ('cpu', 'cuda', 'python', 'numpy', 'loader_generator'))
    return dict(passed=True, checkpoint=str(PARENT), sha256=actual, size_bytes=PARENT.stat().st_size,
        completed_steps=500, scheduler_horizon=4868, scheduler=payload['scheduler'],
        sampler=payload['sampler'], data_cursor=payload['data_cursor'], optimizer_counters=[500],
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'} | {'parameter_count':len(g['params'])}
                          for g in payload['optimizer']['param_groups']],
        rng_per_rank=[dict(rank=r, keys=sorted(s), CPU_sha256=hashlib.sha256(s['cpu'].numpy().tobytes()).hexdigest(),
            CUDA_sha256=hashlib.sha256(s['cuda'].numpy().tobytes()).hexdigest(),
            loader_generator_sha256=hashlib.sha256(s['loader_generator'].numpy().tobytes()).hexdigest())
            for r,s in enumerate(payload['rng_per_rank'])],
        predecessor_sources=old, current_sources=current,
        audited_equivalent_changes={p:[old[p],current[p]] for p in sorted(MIGRATED)},
        migration_reason='Beta parameterization defaults2/2, telemetry, beta resume guard only; fetched HNS-v1 loss/every gradient exact unit test',
        baseline_commit=BASE_SHA, baseline_results_sha256=sha(BASE_EXP/'RESULTS.json'),
        evaluated_checkpoint_unchanged=True, local_image_root=str(IMAGES), NFS_fallback=False,
        user_authorization='HNS-v1 full4epoch; continue this evaluated trajectory500->4868', checked_utc=now())


def reference():
    """CPU-only frozen501..505 replay, never touch the future training process RNG."""
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset = FullLocalDataset(INDEX, IMAGES, 'nested_detail_d3', 0)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    offsets = np.load(INDEX/'offsets.npy', mmap_mode='r')
    result = {}
    with (INDEX/'records.jsonl').open('rb') as handle:
        for rank in range(4):
            sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, seed=0, shuffle=True, drop_last=False)
            sampler.set_epoch(0)
            indices = list(itertools.islice(iter(sampler), 500*256, 505*256))
            for offset in range(5):
                chosen = indices[offset*256:(offset+1)*256]
                views = []
                for index in chosen:
                    handle.seek(int(offsets[index]))
                    record = json.loads(handle.read(int(offsets[index+1]-offsets[index])))
                    views.append(sampled_text_views(record['caption'], 'nested_detail_d3', 0, 0, index+1000))
                batch = dict(sample_id=torch.tensor([i+1000 for i in chosen]), views=[v['views'] for v in views])
                batch.update({k:torch.stack([v[k] for v in views]) for k in ('tokens_f','tokens_o','tokens_e')})
                result[f'{501+offset}:{rank}'] = dict(step=501+offset, rank=rank,
                    sample_ids=batch['sample_id'].tolist(), stream_sha256=full.stream(batch),
                    detail_indices=[v['detail_indices'] for v in views])
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
    after = np.random.get_state()
    assert before[2][0] == after[0] and np.array_equal(before[2][1],after[1]) and before[2][2:] == after[2:]
    return dict(passed=True, first_update=501, next_batch=500, horizon=4868,
                training_RNG_untouched=True, records=result, selection='Frozen sampler and visible nested_detail_d3 tokenizer')


def state(status, **extra):
    dump(EXP/'STATE.json', dict(status=status, updated_utc=now(), runner_pid=os.getpid(),
        run=str(RUN), stop_updates=4868, automatic_other_experiments=False, **extra))


def prepare():
    assert not TRAIN.exists() and not PHASE.exists() and not (RUN/'runner.json').exists()
    EXP.mkdir(exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    dump(CONFIG, read(BASE_EXP/'config.json'))
    dump(EXP/'RESUME_PROVENANCE.json', identity())
    dump(REVIEW/'resume-stream-reference.json', reference())
    path_proof = local.path_proof()
    dump(REVIEW/'prelaunch-local-path-proof-5000.json', path_proof)
    ready = read(LOCAL/'full-ready.json')
    assert ready['status'] == 'LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    dump(EXP/'PLAN.json', dict(method='HNS-v1 exact evaluated500 continuation', resume=str(PARENT),
        parent_sha256=PARENT_SHA, updates=[501,4868], epochs=4, horizon=4868,
        alignment=[1.35,1.35,.30], sparsity=[1,2,2], old_inclusion_max=0, HNS_beta=[2,2], HNS_ramp=200,
        checkpoints=[1217,2434,3651,4868], local_root=str(IMAGES), NFS_fallback=False,
        local_cache='Disposable Docker overlay. NFS remains source of truth; never delete originals.',
        local_path_proof=dict(passed=path_proof['passed'], checked=5000,
            local_path=str(REVIEW/'prelaunch-local-path-proof-5000.json'),
            sha256=sha(REVIEW/'prelaunch-local-path-proof-5000.json')),
        evaluation=['COCO canonical','Urban1k','Flickr30k test1K','DOCCI','Long-DCI7602'],
        inference='normalize(native image embedding) @ normalize(native full-caption text embedding).T',
        stop_after_final_evaluation=True, automatic_other_experiments=False,
        performance_policy='>3s warning; local I/O failure, CUDA/OOM, DDP, NaN/Inf, oom_kill, >60s step, supervisor failure hard stop',
        resumed_loader='Replay consumed500 batches without updates, preserving worker/loader state; first executed update501'))
    state('PREPARED')


def worker():
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate, training_phase_timing as timing
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_train_gate import require_live_supervisor
    from recovery.nested_d3_local_search import observe_selection
    configure()
    supervisor = int(os.environ['SAID_FULL_SUPERVISOR_PID'])
    require_live_supervisor(supervisor)
    proof = read(EXP/'RESUME_PROVENANCE.json')
    ref = read(REVIEW/'resume-stream-reference.json')['records']
    context = dict(optimizer=None, previous=None, restored=False, checked=False)
    original_validate, original_optimizer = trainer.validate_resume_payload, trainer.build_optimizer
    original_restore, original_rates, original_save = trainer.restore_rng_state, trainer.optimizer_learning_rates, trainer.atomic_save

    def validate(previous, current, *args):
        assert sha(PARENT) == PARENT_SHA
        completed = validate_resume(previous, current, proof, original_validate)
        context['previous'] = previous
        return completed

    def optimizer(module):
        context['optimizer'] = original_optimizer(module)
        return context['optimizer']

    def restore(rng):
        original_restore(rng)
        assert gate.exact_state(trainer.rng_state(), {k:rng[k] for k in ('python','numpy','cpu','cuda')})
        if context['previous'] is not None:
            context['restored'] = True

    def save(payload, path):
        if 'model' in payload and 'optimizer' in payload:
            gate.RUN = RUN
            payload = gate.resumable_metadata(payload)
            payload['resume_lineage'] = dict(parent=str(PARENT), sha256=PARENT_SHA, first_update=501)
        return original_save(payload, path)

    def rates(module, completed, horizon):
        require_live_supervisor(supervisor)
        assert horizon == 4868 and 500 <= completed < 4868
        if completed == 500 and not context['checked']:
            previous = context['previous']
            assert previous is not None and context['restored']
            assert gate.exact_state(module.clip.state_dict(), previous['model'])
            assert gate.exact_state(trainer.auxiliary_module(module).state_dict(), previous['adapter'])
            assert gate.exact_state(context['optimizer'].state_dict(), previous['optimizer'])
            assert torch.equal(module._loader_epoch_generator_state, previous['rng_per_rank'][dist.get_rank()]['loader_generator'])
            assert gate.exact_state(trainer.rng_state(), {k:previous['rng_per_rank'][dist.get_rank()][k] for k in ('python','numpy','cpu','cuda')})
            difference = trainer.parameter_agreement(module)
            proofs = [None]*4
            dist.all_gather_object(proofs, dict(rank=dist.get_rank(), model_adapter_optimizer_exact=True,
                RNG_exact_before_update501=True, loader_epoch_RNG_exact=True, parameter_difference=difference,
                data_cursor=previous['data_cursor'], optimizer_counters=[500]))
            assert all(r['parameter_difference'] == 0 for r in proofs)
            if dist.get_rank() == 0:
                dump(REVIEW/'STEP500_RESUME_AUDIT.json', dict(passed=True, ranks=proofs, next_update=501))
            context.update(checked=True, previous=None)
        values = original_rates(module, completed, horizon)
        if completed < 505:
            assert list(values) == full.expected_lrs(completed, read(CONFIG))
            dump(REVIEW/f'resume-lr-{completed+1}-rank{dist.get_rank()}.json', dict(passed=True, step=completed+1, lrs=list(values)))
        return values

    class HeartbeatIterator(timing.TimedIterator):
        def __init__(self, iterator, recorder):
            super().__init__(iterator, recorder)
            self.position, self.epoch = 0, iterator._dataset.epoch

        def __next__(self):
            self.recorder.begin(); self.position += 1
            step = self.epoch*1217+self.position
            event = dict(rank=self.recorder.rank, step=step, started_monotonic=time.monotonic(), utc=now(),
                replay_without_update=step<=500)
            path = PHASE/f'heartbeat-rank{self.recorder.rank}.json'
            dump(path,dict(event,state='DATA_WAIT'))
            try:
                value = next(self.iterator)
            except StopIteration:
                self.recorder.current=None; dump(path,dict(event,state='EXHAUSTED')); raise
            except BaseException:
                dump(path,dict(event,state='DATA_WAIT_FAILED')); raise
            self.recorder.current['step'] = step
            self.recorder.current['data_wait_s'] = time.monotonic()-event['started_monotonic']
            if 501<=step<=505:
                expected = ref[f'{step}:{self.recorder.rank}']
                assert value['sample_id'].tolist() == expected['sample_ids']
                assert full.stream(value) == expected['stream_sha256']
                assert value['detail_indices'] == expected['detail_indices']
                dump(REVIEW/f'resume-batch-{step}-rank{self.recorder.rank}.json', dict(passed=True,
                    step=step, rank=self.recorder.rank, stream_sha256=expected['stream_sha256'],
                    sample_ids_F_Dall_D3_tokens_indices_exact=True, epoch=self.epoch, batch_cursor=self.position-1))
            dump(path,dict(event,state='ACTIVE_STEP'))
            return value

    trainer.validate_resume_payload, trainer.build_optimizer = validate, optimizer
    trainer.restore_rng_state, trainer.optimizer_learning_rates, trainer.atomic_save = restore, rates, save
    trainer.sampling_diagnostics = observe_selection
    trainer.NestedDataset = local.LoggedLocalDataset
    original_loader = trainer.DataLoader
    trainer.DataLoader = lambda *args, **kwargs: original_loader(*args, **kwargs, timeout=60)
    timing.TimedIterator = HeartbeatIterator
    recorder = timing.install(str(PHASE), int(os.environ['RANK']))
    try:
        trainer.main()
        dump(PHASE/f'heartbeat-rank{os.environ["RANK"]}.json', dict(state='TRAINING_COMPLETE', utc=now()))
    finally:
        recorder.finish()
        if dist.is_initialized(): dist.destroy_process_group()


def resume_gate():
    restored = read(REVIEW/'STEP500_RESUME_AUDIT.json')
    batches = [read(p) for p in sorted(REVIEW.glob('resume-batch-*.json'))]
    lrs = [read(p) for p in sorted(REVIEW.glob('resume-lr-*.json'))]
    assert restored['passed'] and len(batches) == len(lrs) == 20
    assert {(b['step'],b['rank']) for b in batches} == set(itertools.product(range(501,506),range(4)))
    assert all(r['passed'] for r in batches+lrs)
    value = dict(passed=True, restoration=restored, batches=batches, LR_gates=lrs, no_epoch0_update_restart=True)
    dump(EXP/'RESUME_GATE.json', value)
    return value


class Supervisor(full.Supervisor):
    def run(self):
        import torch
        from train.train_nested_semantic_mask import code_manifest
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        configure(); self.train = TRAIN
        assert not TRAIN.exists() and not PHASE.exists(), 'Never overwrite a trajectory'
        assert sha(PARENT) == PARENT_SHA and sha(STEP0) == STEP0_SHA
        assert read(EXP/'CPU_TESTS.json')['passed']
        assert read(LOCAL/'full-ready.json')['verification']['passed']
        assert read(EXP/'RESUME_PROVENANCE.json')['current_sources'] == code_manifest()
        assert not system_snapshot()['memory_events'].get('oom_kill',0)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPUs busy'
        forbidden = {'train.train_nested_semantic_mask', 'recovery.s02_full_stage',
            'recovery.s02_local500', 'recovery.s02_nfs500', 'recovery.hns_final_two500_continue'}
        for p in Path('/proc').iterdir():
            if p.name.isdigit() and int(p.name) != os.getpid():
                try: args = (p/'cmdline').read_bytes().split(b'\0')
                except OSError: continue
                assert not any(a.decode(errors='replace') in forbidden for a in args), 'Concurrent copy/training/audit'
        PHASE.mkdir()
        sources = {p:sha(ROOT/p) for p in CODE}
        sources.update(code_manifest())
        for p in ('recovery/s02_local_full.py','recovery/s02_local500.py','recovery/s02_full_local_data.py',
                  'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py',
                  'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py',
                  'tools/nest_clip.py','tools/eval_nest_native.py',
                  'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'):
            sources[p] = sha(ROOT/p)
        dump(RUN/'full-launch-provenance.json', dict(started_utc=self.started, supervisor_pid=os.getpid(),
            resume=str(PARENT), parent_sha256=PARENT_SHA, stop_updates=4868, horizon=4868,
            git_head=git('rev-parse','HEAD'), source_sha256=sources, local_image_root=str(IMAGES), NFS_fallback=False))
        command = [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
            '--max-restarts=0','-m',ENTRY,'--worker','--config',str(CONFIG),'--init-state',str(STEP0),
            '--resume',str(PARENT),'--index-dir',str(INDEX),'--image-root',str(IMAGES),
            '--output-dir',str(TRAIN),'--run-type','formal','--max-updates','4868']
        try:
            state('TRAINING_501_TO_4868', launch_commit=git('rev-parse','HEAD'))
            self.execute('train4868', command, training=True)
            self.acceptance = read(TRAIN/'acceptance.json')
            assert self.acceptance['passed'] and all(r['completed_updates']==4868 and r['updates_this_run']==4368
                and r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            resume_gate()
            checkpoint_proofs = []
            for step in (1217,2434,3651,4868):
                p = TRAIN/f'step{step:06d}.pt'
                payload = torch.load(p,map_location='cpu',weights_only=False)
                checkpoint_proofs.append(checkpoint_proof(payload,step,p))
                del payload
            dump(REVIEW/'checkpoint-proofs.json', checkpoint_proofs)
            state('STRICT_NATIVE_FIVE_DATASET_EVALUATION', completed_steps=4868)
            pipeline.RUN, pipeline.EXP = RUN, EXP
            self.result = pipeline.Supervisor.evaluate(self,4868)
            score = self.result['scores_percent']
            score.update(Score5=score['Score5_R1'],Short4=score['Short4_R1'])
            self.result['status'] = full.classify(score['Score5'],score['J_long3'])
        except BaseException as error:
            self.error = type(error).__name__+': '+str(error)
            print(self.error,flush=True)
        finally:
            dump(RUN/'full-supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
                started_utc=self.started,ended_utc=now()))
        if self.error: raise RuntimeError(self.error)


def checkpoint_proof(payload, step, path):
    import torch
    frozen_config(payload['config'])
    assert payload['completed_steps'] == payload['global_step'] == payload['scheduler']['completed_steps'] == step
    assert payload['scheduler_horizon'] == payload['scheduler']['horizon'] == 4868
    assert payload['data_cursor'] == dict(next_epoch=step//1217,next_batch=step%1217)
    assert payload['config']['resume'] == str(PARENT) and payload['config']['start_updates'] == 500
    assert payload['config']['parent_checkpoint_sha256'] == PARENT_SHA
    assert payload['trajectory_root'] == str(RUN) and len(payload['rng_per_rank']) == 4
    assert {int(s['step']) for s in payload['optimizer']['state'].values()} == {step}
    assert all('loader_generator' in r for r in payload['rng_per_rank'])
    assert all(torch.isfinite(v).all() for state in (payload['model'],payload['adapter']) for v in state.values())
    assert all(torch.isfinite(v).all() for state in payload['optimizer']['state'].values() for v in state.values())
    return dict(passed=True,step=step,path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                cursor=payload['data_cursor'],optimizer_counters=[step],rng_ranks=4,uploaded=False)


def stream_audit(steps):
    import torch
    from torch.utils.data import DistributedSampler
    from recovery.s02_full_local_data import FullLocalDataset
    from train.nested_semantic_data import sample_partial_detail_indices
    from recovery.nested_d3_local_search import indices_digest
    assert [r['step'] for r in steps] == list(range(1,4869))
    before = torch.get_rng_state().clone()
    dataset = FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    cfg = read(CONFIG); count = 0
    for epoch in range(4):
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
            sampler.set_epoch(epoch); indices=list(sampler)
            for batch,row in enumerate(steps[epoch*1217:(epoch+1)*1217]):
                assert row['epoch']==epoch and row['s']==row['step']-1 and math.isfinite(row['loss']) and row['nonfinite']==0
                expected=[i+1000 for i in indices[batch*256:(batch+1)*256]]
                h=next(h for h in row['rank_health'] if h['rank']==rank); s=h['sampling']
                assert h['batch']==len(expected)==(180 if batch==1216 else 256)
                assert h['updates']==(row['step'] if row['step']<=500 else row['step']-500)
                assert h['gradients_finite'] and s['sample_ids']==expected and s['nested_d3_exact']
                selected=[sample_partial_detail_indices(n,0,epoch,sid) for n,sid in zip(s['n'],expected)]
                assert s['lowest_selected_sentence_indices_sha256']==indices_digest(expected,selected)
                assert list(row['actual_lrs'].values())==full.expected_lrs(row['step']-1,cfg)
                assert row['HNS_enabled'] and row['inc_weight']==row['inclusion_loss']==0
                assert row['lambda_h']==min(1.,(row['step']-1)/200.)
                count+=len(expected)
    assert count==4*(1245901+3) and torch.equal(before,torch.get_rng_state())
    return dict(passed=True,records=count,steps=4868,epochs=4,tail_per_rank=180,
                all_sample_ids_selected_indices_LR_exact=True,first500_from_pinned_baseline=True,
                native_objective_unchanged=True,training_RNG_untouched=True)


def runtime_stats(steps, saved):
    cycles=rows(TRAIN/'cycle_timing.jsonl')
    assert [c['step'] for c in cycles]==list(range(501,4869))
    phases={r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
    assert all([p['step'] for p in records]==list(range(501,4869)) for records in phases.values())
    lookup={r:{p['step']:p for p in records} for r,records in phases.items()}
    telemetry=[s for s in rows(RUN/'full-resource-telemetry.jsonl') if s['command']=='train4868']
    systems=[s['system'] for s in telemetry]
    waits=[max(lookup[r][c['step']]['data_wait_s'] for r in range(4)) for c in cycles]
    paths=[r for p in PHASE.glob('image-paths-*.jsonl') for r in rows(p)]
    assert paths and all(Path(p['actual_path']).is_relative_to(IMAGES) and not p['NFS_fallback'] for p in paths)
    assert not max((s['memory_events'].get('oom_kill',0) for s in systems),default=0)
    result=dict(scope='Continuation501..4868; loader replay/checkpoint overhead excluded from update distribution',
        full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
        data_wait_seconds_slowest_rank=distribution(waits),
        data_wait_seconds_by_rank={str(r):distribution([p['data_wait_s'] for p in rec]) for r,rec in phases.items()},
        steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),
        steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
        GPU_peak_memory_GiB={str(r):max(h['peak_allocated_gib'] for row in steps for h in row['rank_health'] if h['rank']==r) for r in range(4)},
        peak_cgroup_memory_bytes=max(s['memory_current'] for s in systems),
        peak_file_cache_bytes=max(s['file'] for s in systems),peak_anon_bytes=max(s.get('anon',0) for s in systems),
        oom_kill=0,real_image_IO_errors=0,pod_supervisor_anomaly_observed=False,
        wall_seconds_including_replay_checkpoints=max(r['seconds'] for r in saved['acceptance']['ranks']),
        checkpoint_timing=rows(TRAIN/'checkpoint_timing.jsonl'),
        local_runtime_path_proof=dict(passed=True,count=len(paths),NFS_fallback=False),
        started_utc=saved['started_utc'],finished_utc=saved['ended_utc'],
        memory_interpretation='memory.current includes file cache; it is not process RSS.',
        PSI_scope='Host /proc/pressure on cgroupv1, not per-cgroup PSI')
    for kind in ('io_PSI','memory_PSI'):
        result[kind]={pressure:distribution([s[kind][pressure]['avg10'] for s in systems if pressure in s[kind]])
                      for pressure in ('some','full')}
    result['GPU_utilization_percent']={str(r):distribution([g['gpu_percent'] for s in telemetry for g in s['gpu']
        if isinstance(g,dict) and g.get('index')==r and g.get('gpu_percent') is not None]) for r in range(4)}
    initial=rows(BASE_RUN/'step500/cycle_timing.jsonl')
    result['whole_trajectory_full_cycle_seconds']=distribution([c['four_rank_max_seconds'] for c in initial+cycles])
    inventory=[p for p in RUN.rglob('*') if p.is_file()]+[p for p in PHASE.rglob('*') if p.is_file()]
    result['local_assets_not_uploaded']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
        time_range_utc=[saved['started_utc'],saved['ended_utc']]) for p in sorted(inventory) if p.name not in ('runner.log',)]
    # runner.log is still being written; snapshot identity explicitly identified.
    result['active_runner_log_snapshot']=dict(path=str(MAIN_LOG),bytes=MAIN_LOG.stat().st_size,sha256=sha(MAIN_LOG),snapshot_utc=now(),uploaded=False)
    result['local_image_mirror']=dict(path=str(IMAGES),uploaded=False,disposable=True,NFS_originals_retained=True)
    return result


def comparisons(result):
    from recovery.full_evidence import RANDOMK, RANDOMK_R1
    comparisons={}
    for name,path in (
        ('HNS-v1@500',BASE_EXP/'RESULTS.json'),
        ('S02 full',ROOT/'recovery/FULL_RESULTS.json'),
        ('D3 Balanced full',ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/RESULTS.json')):
        if path.exists():
            base=read(path); score=base.get('scores_percent',{})
            if all(k in score for k in SCORE_KEYS) and base.get('metrics'):
                comparisons[name]=dict(source=str(path),sha256=sha(path),
                    scores_delta_pp={k:result['scores_percent'][k]-score[k] for k in SCORE_KEYS},
                    recall_delta_pp={ds:{dr:{k:100*(v-base['metrics'][ds][dr][k]) for k,v in metrics.items()}
                        for dr,metrics in directions.items()} for ds,directions in result['metrics'].items()})
    comparisons['RandomK formal4epoch']=dict(
        scores_delta_pp={k:result['scores_percent'][k]-RANDOMK[k] for k in SCORE_KEYS},
        R1_delta_pp={ds:{dr:100*result['metrics'][ds][dr]['R@1']-old[i] for i,dr in enumerate(('I2T','T2I'))}
                     for ds,old in RANDOMK_R1.items()},
        R5_R10_delta='Not available in the supplied formal reference; no invented values')
    return comparisons


def report():
    saved=read(RUN/'full-supervisor-result.json')
    assert not saved['error'] and saved['result'] and saved['acceptance']['passed']
    continuation=rows(TRAIN/'steps.jsonl'); baseline=rows(BASE_RUN/'step500/steps.jsonl')
    validation=stream_audit(baseline+continuation)
    stats=runtime_stats(continuation,saved)
    result=dict(saved['result'],completed_steps=4868,stop_exact4868=True,continued_exact_HNS_v1=True,
        resume=str(PARENT),parent_sha256=PARENT_SHA,local_image_root=str(IMAGES),NFS_fallback=False,
        checkpoint_proofs=read(REVIEW/'checkpoint-proofs.json'),stream_validation=validation,
        automatic_other_experiments=False,checkpoint_uploaded=False,bare_uploaded=False)
    result['comparisons']=comparisons(result)
    last50=continuation[-50:]
    means={k:statistics.fmean(r[k] for r in last50) for k,v in last50[0].items()
        if isinstance(v,(int,float)) and (k.startswith(('F_','O_','E_','HNS_','V_')) or k in ('loss','alignment','sparsity','inclusion_loss','inc_weight','lambda_h'))}
    from recovery.nested_detail_d3_balanced_full_evidence import view_means
    diagnostics=dict(last50=means,views_last50=view_means(last50),
        gradient_group_norms_last50={name:statistics.fmean(h['gradient_norms'][name] for row in last50
            for h in row['rank_health']) for name in last50[0]['rank_health'][0]['gradient_norms']},
        epochs=[dict(epoch=e,updates=1217,last50_loss=statistics.fmean(r['loss'] for r in (baseline+continuation)[(e+1)*1217-50:(e+1)*1217])) for e in range(4)],
        note='Native backbone/adapter group gradient norms in local per-step raw logs; no additional optimizer updates')
    hierarchy={k:v for k,v in means.items() if any(t in k for t in ('IoU','iou','violation','keep','gap','equality'))}
    dump(EXP/'FULL_RESULTS.json',result);dump(EXP/'FULL_RUNTIME_STATS.json',stats)
    dump(EXP/'TRAINING_DIAGNOSTICS.json',diagnostics);dump(EXP/'MASK_HIERARCHY_AUDIT.json',dict(last50=hierarchy))
    dump(EXP/'VALIDATION.json',dict(passed=True,stream=validation,resume_gate=resume_gate(),
        checkpoint_proofs=result['checkpoint_proofs'],strict_export=result['strict_export'],all_evaluation_checkpoint_SHA_unchanged=True))
    lines=['# HNS-v1 full4epoch','',f"Status: `{result['status']}`. Exactly4868 updates, continued pinned/evaluated500 without resetting state.",'',
        'Native bare CLIP, normalized image/full-caption embeddings. Local-only images; no mask/gate/rerank inference. No additional experiment starts.', '',
        '| Dataset | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,metrics in result['metrics'].items():
        values=[100*metrics[d][k] for d in ('I2T','T2I') for k in ('R@1','R@5','R@10')]
        lines.append('| '+ds+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','Scores (%): `'+json.dumps({k:result['scores_percent'][k] for k in SCORE_KEYS})+'`.','',
        'All available baseline score/recall deltas are in FULL_RESULTS.json. HNS-v1@500 is a learning-curve comparison; full baselines compare different completed training methods.', '',
        'Checkpoint SHA256: `'+result['checkpoint_sha256']+'`; bare SHA256: `'+result['strict_export']['bare_sha256']+'`.', '',
        'Runtime: `'+json.dumps({k:stats[k] for k in ('full_cycle_seconds','data_wait_seconds_slowest_rank','GPU_peak_memory_GiB','peak_cgroup_memory_bytes','oom_kill','real_image_IO_errors')})+'`.', '',
        'Code migration: explicitly pinned default-beta2/2 equivalent predecessor sources; fetched original v1 loss/every parameter gradient exact CPU test. RESUME_GATE.json verifies actual model/adapter/AdamW/RNG/loader restoration and501–505 IDs/text/tokens/indices/LR.', '',
        'Checkpoints, bare weights, raw logs and stream/path proofs remain local. FULL_RUNTIME_STATS.json lists paths/sizes/SHA256/time ranges. `/root` images are disposable Docker overlay; NFS originals remain the durable source of truth.', '',
        'One seed only. Full outcome does not establish statistical significance. No automatic hyperparameter search or new experiment.']
    (EXP/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')


def publish(final=False):
    assert git('branch','--show-current')==BRANCH, 'Branch changed; refuse publication'
    assert not git('diff','--cached','--name-only'), 'Unrelated staged changes'
    paths=[ROOT/p for p in CODE]+[EXP/p for p in STATIC]
    if final:
        paths += [EXP/p for p in REPORTS]
        for p,digest in read(RUN/'full-launch-provenance.json')['source_sha256'].items():
            assert sha(ROOT/p)==digest, ('Launch source changed',p)
    assert all(p.is_file() and p.stat().st_size<=1024*1024 for p in paths)
    subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
    from recovery.check_stage500_publish import inspect
    review=inspect();assert review['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Report HNS-v1 full4868 native retrieval' if final else 'Prepare pinned HNS-v1 exact500->4868 continuation'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    remote=git('rev-parse','refs/remotes/origin/'+BRANCH)
    assert head==remote==git('rev-parse','FETCH_HEAD')
    for p in paths:
        assert subprocess.check_output(['git','show',head+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
    receipt=dict(passed=True,branch=BRANCH,commit=head,remote_HEAD=remote,remote_HEAD_matches_local=True,
                 push_success=True,fetch_success=True,publication_check=review,checked_utc=now(),final=final)
    dump(EXP/'GITHUB_RECEIPT.json',receipt)
    return receipt


def run():
    configure()
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame): raise RuntimeError('HARD_STOP supervisor signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            Supervisor().run()
            state('GENERATING_FULL_EVIDENCE',completed_steps=4868)
            report()
            state('PUBLISHING',completed_steps=4868)
            receipt=publish(final=True)
            state('COMPLETED_AND_SYNCED',completed_steps=4868,github=receipt)
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',error=type(error).__name__+': '+str(error),automatic_retry=False,
                  checkpoints_retained=True)
            raise


def detach():
    assert not (RUN/'runner.json').exists() and read(EXP/'CPU_TESTS.json')['passed']
    assert read(EXP/'GITHUB_RECEIPT.json')['commit']==git('rev-parse','HEAD')
    with MAIN_LOG.open('ab',buffering=0) as handle:
        p=subprocess.Popen([PYTHON,'-u','-m',ENTRY],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
    value=dict(pid=p.pid,session=p.pid,main_log=str(MAIN_LOG),entry=ENTRY,start_new_session=True,
        stdin='/dev/null',resume=str(PARENT),parent_sha256=PARENT_SHA,updates=[501,4868],
        sequential_queue=['HNS-v1 continue4868','strict native five-dataset evaluation','report','GitHub commit/push/fetch'],
        automatic_other_experiments=False,started_utc=now())
    dump(RUN/'runner.json',value);print(json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--detach',action='store_true')
    parser.add_argument('--publish',action='store_true');args,remaining=parser.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*remaining];worker()
    else:
        assert not remaining
        if args.prepare: prepare()
        elif args.publish: publish()
        elif args.detach: detach()
        else: run()


if __name__=='__main__': main()
