"""One common0 HNS-v1 Pure3 trajectory; native mathematics, hard fail on drift."""
import argparse
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from recovery.s02_nfs500 import ROOT, STEP0_SHA, dump, sha, now
from recovery.s02_full_stage import LOCAL, IMAGES, INDEX_SHA, MANIFEST_SHA

ORIGINAL = Path('/opt/data/private/lklk/SAID')
BASE = '45468e1bbbcee52f6c069db79ee8266d529ca7a9'
EVAL_BASE = 'e305275c24bb6a80c2e87a674d50239d6a0f22f5'
BRANCH = 'experiment/hns-v1-pure3epoch3651-v1'
RUN = ORIGINAL / 'runtime/SAID-nest-clip-v1/hns-v1-pure3epoch3651-v1'
EXP = ROOT / 'experiments/nest_clip_v1/hns_v1_pure3epoch3651_v1'
STEP0 = ORIGINAL / 'runtime/SAID-nest-clip-v1/shared/step000000.pt'
INDEX = LOCAL / 'data_index'
TRAIN = RUN / 'step3651'
CONFIG = EXP / 'config.json'
REFERENCE500 = ORIGINAL / 'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008/HNS/step500'
REFERENCEFULL = ORIGINAL / 'runtime/SAID-nest-clip-v1/nested-d3-hns-full-v1/step4868'
ENTRY = 'recovery.hns_v1_pure3'
HORIZON = 3651
GROUPS = ('backbone', 'text_mask_and_shared_pool', 'visual_mask', 'fusion_adapter')
EVAL_FILES = ('tools/eval_hyfl_native.py', 'tools/retrieval_bounded.py', 'tools/eval_resource_queue.py',
              'tools/audit_hyfl_data.py', 'tools/flickr_full_protocol.py', 'tools/validate_hyfl_inference.py',
              'tools/eval_five_parallel.py', 'tests/test_retrieval_bounded.py')


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def expected_config():
    original = json.loads(subprocess.check_output(['git', 'show', BASE +
                         ':experiments/nest_clip_v1/nested_d3_hns_full_v1/config.json'], cwd=ROOT))
    return dict(original, epochs=3, four_epoch_followup=False, save_initial_checkpoint=True)


def validate_config(config):
    expected = expected_config()
    assert all(config.get(k) == v for k, v in expected.items()), 'Pure3 frozen config drift'
    assert config.get('view_sparsity_weights', [1., 2., 2.]) == [1., 2., 2.]
    assert config.get('hns_beta', [2., 2.]) == [2., 2.]
    assert not config.get('hns_detach_child', False)
    assert not any(k in config for k in ('lambda_align', 'lambda_sparse', 'lambda_hierarchy'))


def normalize_json(value):
    """Strict complete-object comparison; reject lossy JSON key collisions."""
    def check(x):
        if isinstance(x, dict):
            keys = [str(k) for k in x]
            if len(keys) != len(set(keys)):
                raise ValueError('JSON key collision')
            for v in x.values(): check(v)
        elif isinstance(x, (list, tuple)):
            for v in x: check(v)
    check(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def analytic_lrs(s, horizon=HORIZON):
    backbone = (1e-6 * (s + 1) / 200 if s < 200 else
                1e-6 * (1 + math.cos(math.pi * (s - 200) / (horizon - 200))) / 2)
    factor = (1 + math.cos(math.pi * s / horizon)) / 2
    return [backbone, 1e-3 * factor, 1e-3 * factor, 2e-4 * factor]


def source_proof():
    from train.train_nested_semantic_mask import code_manifest
    sources = code_manifest()
    sources.update({p: sha(ROOT / p) for p in ('model/simple_tokenizer.py', 'model/bpe_simple_vocab_16e6.txt.gz',
        'recovery/s02_full_local_data.py', 'train/said_cvssl_data.py', 'tools/nest_clip.py',
        'tools/eval_nest_native.py', 'tools/urban1k_retrieval.py', 'eval/retrieval/coco_retrieval.py',
        'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py')})
    for path, actual in sources.items():
        old = subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT)
        assert hashlib.sha256(old).hexdigest() == actual, ('Original HNS source drift', path)
    for path in EVAL_FILES:
        old = subprocess.check_output(['git', 'show', EVAL_BASE + ':' + path], cwd=ROOT)
        assert hashlib.sha256(old).hexdigest() == sha(ROOT / path), ('Frozen Full eval drift', path)
    return dict(passed=True, training_base_commit=BASE, production_source_sha256=sources,
                evaluation_base_commit=EVAL_BASE,
                copied_frozen_evaluation_sources={p: sha(ROOT / p) for p in EVAL_FILES},
                production_math_modified=False)


def prepare_reference():
    """Keep actual historical stream evidence separate from reconstruction."""
    reference = {}
    paths = [REFERENCE500 / 'steps.jsonl', REFERENCEFULL / 'steps.jsonl']
    evidence = []
    for path in paths:
        assert path.is_file(), ('Historical log missing', path)
        evidence.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                step = row['step']
                if not 1 <= step <= HORIZON: continue
                simplified = dict(step=step, epoch=row['epoch'], s=row['s'],
                    ranks=[dict(rank=h['rank'], batch=h['batch'], stream_sha256=h['stream_sha256'],
                                sampling=h['sampling']) for h in row['rank_health']])
                if step in reference:
                    assert simplified == reference[step], ('Conflicting historical stream', step)
                else: reference[step] = simplified
    assert sorted(reference) == list(range(1, HORIZON + 1))
    for rank in range(4):
        with (RUN / f'reference-rank{rank}.jsonl').open('x') as out:
            for step in range(1, HORIZON + 1):
                row = reference[step]
                assert row['epoch'] == (step - 1) // 1217 and row['s'] == step - 1
                h = next(h for h in row['ranks'] if h['rank'] == rank)
                assert h['batch'] == (180 if step % 1217 == 0 else 256)
                out.write(json.dumps(dict(step=step, epoch=row['epoch'], **h), ensure_ascii=False) + '\n')
    return dict(passed=True, scope='Actual historical HNS-v1 steps1..3651; no reconstructed substitutes',
                logs=evidence, records=HORIZON * 4,
                historical_stream_sha256=hashlib.sha256(json.dumps(reference, sort_keys=True,
                                           ensure_ascii=False, allow_nan=False).encode()).hexdigest())


def prepare():
    import torch
    from recovery.s02_local500 import path_proof
    from recovery.s02_full_local_data import FullLocalDataset
    assert git('branch', '--show-current') == BRANCH
    assert not EXP.exists() and not RUN.exists(), 'Existing Pure3 task: do not duplicate'
    EXP.mkdir(parents=True); RUN.mkdir(parents=True)
    assert sha(STEP0) == STEP0_SHA
    p = torch.load(STEP0, map_location='cpu', weights_only=False)
    assert p['completed_steps'] == 0 and not p['optimizer']['state']
    assert p['provenance']['source'] == 'OpenAI CLIP + original random MaskNetwork'
    from train.train_nested_semantic_mask import state_digest, training_horizon
    initial = dict(passed=True, original=str(STEP0), sha256=STEP0_SHA, completed_steps=0,
                   optimizer_state_empty=True, model_sha256=state_digest(p['model']),
                   rng_cpu_sha256=hashlib.sha256(p['rng_cpu'].numpy().tobytes()).hexdigest(),
                   original_adapter_absent=True, adapter_method='Frozen constructor: clone common Mask blocks; seeded1763 adapter; zero gate',
                   actual_four_rank_initial_state_proof='Written separately before the smoke and formal first updates')
    dump(EXP / 'COMMON0_IDENTITY.json', initial)
    del p
    config = expected_config(); validate_config(config); dump(CONFIG, config)
    old = read(ROOT / 'experiments/nest_clip_v1/nested_d3_hns_full_v1/config.json')
    changes = {k: dict(before=old.get(k), after=v) for k, v in config.items() if old.get(k) != v}
    assert set(changes) == {'epochs', 'four_epoch_followup', 'save_initial_checkpoint'}
    dump(EXP / 'CONFIG_DIFF.json', dict(passed=True, differences=changes,
         mathematical_changes=['epochs4->3', 'derived scheduler horizon4868->3651'],
         control_only_changes=['four_epoch_followup=False', 'save_initial_checkpoint=True'],
         preserved_legacy_experiment_name=config['experiment_name'], identity='HNS-v1-Pure3',
         effective_view_sparsity_weights=[1, 2, 2], HNS_beta=[2, 2], HNS_ramp=200))
    assert training_horizon(config, 1217) == HORIZON
    dump(EXP / 'SOURCE_PROOF.json', source_proof())
    ready = read(LOCAL / 'full-ready.json')
    assert ready['verification']['passed'] and ready['manifest']['sha256'] == MANIFEST_SHA
    dataset = FullLocalDataset(INDEX, IMAGES, 'nested_detail_d3', 0)
    assert len(dataset) == 1245901 and dataset.metadata['records_sha256'] == INDEX_SHA
    assert sha(INDEX / 'records.jsonl') == INDEX_SHA
    proof = path_proof(5000)
    dump(RUN / 'prelaunch-local-path-proof.json', proof)
    dump(EXP / 'DATA_PREFLIGHT.json', dict(passed=True, records=len(dataset), index_sha256=INDEX_SHA,
        offsets_sha256=sha(INDEX / 'offsets.npy'), metadata_sha256=sha(INDEX / 'metadata.json'),
        manifest_sha256=MANIFEST_SHA, local_root=str(IMAGES), NFS_fallback=False,
        real_existing_paths_checked=proof['count'], actual_history_reference=prepare_reference()))
    dump(EXP / 'PLAN.json', dict(model='HNS-v1 Pure3', branch=BRANCH, common0=str(STEP0),
        common0_sha256=STEP0_SHA, one_seed=0, one_training_arm=True, epochs=3,
        scheduler_horizon=HORIZON, max_updates=HORIZON, resume=None, updates=[1, HORIZON],
        checkpoints=[0, 500, 1217, 2434, 3651], smoke='Independent common0 smoke5; then restart common0',
        evaluation=['COCO-Val5K', 'Flickr30k-Test1K', 'DOCCI', 'Long-DCI', 'Urban-1k', 'Flickr30k-Full'],
        DCI_excluded=True, intermediate_public_evaluation=False,
        production_mathematics='Historical HNS-v1: alignment10/sum(w)*weighted symmetric CE; sparse(F+2Dall+2D3)/3; ramp*(2V_DF+2V_3D)/3',
        original_detach=True, no_soft_inclusion=True, stop_after_final_evaluation=True,
        automatic_retry=False, automatic_next_experiment=False))
    state('PREPARED')


def state(status, **extra):
    dump(EXP / 'STATE.json', dict(status=status, utc=now(), pid=os.getpid(), run=str(RUN),
                                stop_updates=HORIZON, automatic_extra_training=False, **extra))


def worker():
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from recovery import s02_local500 as local
    from recovery.s02_local_full import stream as batch_stream
    from recovery.nested_d3_local_search import observe_selection
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import training_phase_timing as timing
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.reproduction_train_gate import exact_state
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_train_gate import require_live_supervisor
    rank = int(os.environ['RANK'])
    supervisor = int(os.environ['SAID_FULL_SUPERVISOR_PID'])
    phase = Path(os.environ['SAID_S02_PHASE_LOCAL'])
    run_kind = 'smoke' if '--run-type' in sys.argv and sys.argv[sys.argv.index('--run-type') + 1] == 'smoke' else 'formal'
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    local.INDEX = INDEX
    historical_config = read(REFERENCE500 / 'config.json')
    reference = {r['step']: r for r in (json.loads(s) for s in (RUN / f'reference-rank{rank}.jsonl').read_text().splitlines())}
    context = dict(optimizer=None, saved500=False)
    old_build, old_save = trainer.build_optimizer, trainer.atomic_save
    old_rates, old_validate = trainer.optimizer_learning_rates, trainer.validate_resume_payload

    def optimizer(module):
        opt = old_build(module)
        assert not opt.state
        context['optimizer'] = opt
        return opt

    def save(payload, path):
        if 'optimizer' in payload and 'model' in payload:
            n = payload['completed_steps']; cfg = payload['config']; validate_config(cfg)
            assert payload['scheduler_horizon'] == HORIZON
            payload.update(global_step=n,
                scheduler=dict(type='native_manual_learning_rate_function', horizon=HORIZON,
                               completed_steps=n, last_update_index=n-1,
                               optimizer_lrs={g['name']: g['lr'] for g in payload['optimizer']['param_groups']}),
                sampler=dict(type='DistributedSampler', seed=0, world_size=4, batch_size=256,
                             batches_per_epoch=1217, num_samples=cfg['sampler_num_samples'], drop_last=False),
                scaler=dict(enabled=False, dtype='bfloat16', state=None),
                data_cursor=dict(next_epoch=payload['next_epoch'], next_batch=payload['next_batch']),
                trajectory_root=str(RUN),
                initialization_lineage=dict(common0=str(STEP0), sha256=STEP0_SHA,
                                            horizon=HORIZON, optimizer_reset=False))
            payload['control_source_sha256'] = {p: sha(ROOT / p) for p in
                ('recovery/hns_v1_pure3.py', 'recovery/hns_v1_pure3_evidence.py',
                 'tools/eval_hns_pure3_flickrfull.py')}
            if n == 0:
                common = torch.load(STEP0, map_location='cpu', weights_only=False)
                assert exact_state(payload['model'], common['model']), 'Initial model differs from common0'
                assert not payload['optimizer']['state']
                assert cfg['component_initialization'] == historical_config['component_initialization']
                assert cfg['parameter_counts'] == historical_config['parameter_counts']
                for r in payload['rng_per_rank']:
                    assert torch.equal(r['cpu'], common['rng_cpu']), 'Historical initial CPU RNG drift'
                    assert torch.equal(r['loader_generator'], torch.Generator().manual_seed(0).get_state())
                proof = dict(passed=True, kind=run_kind, common0_model_exact=True,
                             adapter_components_exact=cfg['component_initialization'],
                             optimizer_state_empty=True, optimizer_groups=[{k:v for k,v in g.items() if k!='params'}
                                  for g in payload['optimizer']['param_groups']],
                             RNG_CPU_common0_exact=True, loader_seed0_exact=True, rng_ranks=4,
                             before_first_optimizer_update=True, horizon=HORIZON,
                             checkpoint=str(path), model_sha256=trainer.state_digest(payload['model']),
                             adapter_sha256=trainer.state_digest(payload['adapter']))
                dump(RUN / f'{run_kind}-STEP0_IDENTITY.json', proof)
        return old_save(payload, path)

    def rates(module, completed, horizon):
        require_live_supervisor(supervisor)
        assert horizon == HORIZON and 0 <= completed < HORIZON
        values = old_rates(module, completed, horizon)
        assert list(values) == analytic_lrs(completed), 'Native vs analytic3651 LR mismatch'
        if run_kind == 'formal' and completed == 500 and not context['saved500']:
            trainer.save_checkpoint(module, context['optimizer'], read(output / 'config.json'), 500, output)
            context['saved500'] = True
        if completed in (0, 4, 499, 1216, 2433, 3650):
            difference = trainer.parameter_agreement(module)
            assert difference == 0, ('Rank parameters differ', completed)
            dump(RUN / f'{run_kind}-agreement-step{completed}-rank{rank}.json',
                 dict(passed=True, rank=rank, completed_steps=completed, maximum_difference=0))
        return values

    def validate(previous, current, *args):
        validate_config(current)
        assert Path(current['resume']).resolve().is_relative_to(TRAIN.resolve())
        assert previous['trajectory_root'] == str(RUN) and previous['scheduler_horizon'] == HORIZON
        return old_validate(previous, current, *args)

    class Iterator(timing.TimedIterator):
        def __init__(self, iterator, recorder):
            super().__init__(iterator, recorder)
            self.position = 0; self.epoch = iterator._dataset.epoch
            assert self.epoch in (0, 1, 2), 'Fourth epoch read forbidden'
        def __next__(self):
            self.recorder.begin(); self.position += 1
            step = self.epoch * 1217 + self.position
            event = dict(rank=rank, step=step, epoch=self.epoch, started_monotonic=time.monotonic(), utc=now())
            path = phase / f'heartbeat-rank{rank}.json'
            dump(path, dict(event, state='DATA_WAIT'))
            try: batch = next(self.iterator)
            except StopIteration:
                self.recorder.current = None; dump(path, dict(event, state='EXHAUSTED')); raise
            except BaseException:
                dump(path, dict(event, state='DATA_WAIT_FAILED')); raise
            self.recorder.current['step'] = step
            self.recorder.current['data_wait_s'] = time.monotonic() - event['started_monotonic']
            assert step <= HORIZON
            expected = reference[step]
            actual_sampling = observe_selection(batch)
            assert expected['epoch'] == self.epoch and len(batch['sample_id']) == expected['batch']
            assert batch_stream(batch) == expected['stream_sha256'], f'Step{step}/rank{rank} F/Dall/D3/token/ID mismatch'
            assert normalize_json(actual_sampling) == normalize_json(expected['sampling']), f'Step{step}/rank{rank} complete K/detail sampling mismatch'
            dump(path, dict(event, state='ACTIVE_STEP'))
            with (phase / f'stream-rank{rank}.jsonl').open('a') as out:
                out.write(json.dumps(dict(step=step, epoch=self.epoch, batch_cursor=self.position-1,
                    rank=rank, batch=len(batch['sample_id']), stream_sha256=expected['stream_sha256'],
                    full_sampling_digest=hashlib.sha256(json.dumps(normalize_json(actual_sampling), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    matches_real_historical_log=True)) + '\n')
            return batch

    trainer.build_optimizer = optimizer; trainer.atomic_save = save
    trainer.optimizer_learning_rates = rates; trainer.validate_resume_payload = validate
    trainer.sampling_diagnostics = observe_selection; trainer.NestedDataset = local.LoggedLocalDataset
    original_loader = trainer.DataLoader
    trainer.DataLoader = lambda *a, **kw: original_loader(*a, **kw, timeout=60)
    timing.TimedIterator = Iterator
    recorder = timing.install(str(phase), rank)
    try:
        trainer.main()
        dump(phase / f'heartbeat-rank{rank}.json', dict(state='TRAINING_COMPLETE', utc=now()))
    except BaseException as error:
        dump(RUN / f'{run_kind}-FAILURE-rank{rank}.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        recorder.finish()
        if dist.is_initialized(): dist.destroy_process_group()


def execute_controller():
    import torch
    from recovery import s02_local_full as control
    from recovery.resource_stall_v2 import system_snapshot
    from tools.eval_five_parallel import require_gpu_idle
    control.RUN = RUN
    supervisor = control.Supervisor()
    lock = (RUN / '.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep the historical 60s heartbeat and actual OOM/NaN/DDP stop policy.
    try:
        source_proof()
        assert sha(STEP0) == STEP0_SHA and read(EXP / 'CPU_TESTS.json')['passed']
        assert read(EXP / 'DDP_CORRECTNESS.json')['passed']
        require_gpu_idle((0, 1, 2, 3))
        assert not system_snapshot()['memory_events'].get('oom_kill', 0)
        for kind, limit in [('smoke', 5), ('formal', HORIZON)]:
            phase = LOCAL / ('hns-v1-pure3epoch3651-v1-' + kind + '-phases')
            assert not phase.exists(); phase.mkdir()
            control.PHASE = phase
            output = RUN / 'smoke5' if kind == 'smoke' else TRAIN
            assert not output.exists()
            state('SMOKE5' if kind == 'smoke' else 'FORMAL_TRAINING', phase=str(phase), max_updates=limit)
            command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nnodes=1', '--nproc-per-node=4',
                '--max-restarts=0', '-m', ENTRY, '--worker', '--config', str(CONFIG), '--init-state', str(STEP0),
                '--index-dir', str(INDEX), '--image-root', str(IMAGES), '--output-dir', str(output),
                '--run-type', kind if kind == 'smoke' else 'formal', '--max-updates', str(limit)]
            supervisor.execute(kind + str(limit), command, training=True)
            acceptance = read(output / 'acceptance.json')
            assert acceptance['passed'] and all(r['completed_updates'] == r['updates_this_run'] == limit
                and r['max_parameter_difference_from_rank0'] == 0 for r in acceptance['ranks'])
            p = torch.load(output / f'step{limit:06d}.pt', map_location='cpu', weights_only=False)
            assert p['completed_steps'] == limit and p['scheduler_horizon'] == HORIZON
            assert {int(s['step']) for s in p['optimizer']['state'].values()} == {limit}
            assert p['data_cursor'] == dict(next_epoch=limit // 1217, next_batch=limit % 1217)
            del p
            dump(EXP / ('SMOKE_ACCEPTANCE.json' if kind == 'smoke' else 'DDP_ACCEPTANCE.json'),
                 dict(passed=True, kind=kind, acceptance=acceptance, horizon=HORIZON,
                      fresh_common0=True, updates=limit, phase=str(phase), output=str(output)))
            require_gpu_idle((0, 1, 2, 3))
        from recovery.hns_v1_pure3_evidence import training_audit, evaluate, summarize
        training_audit()
        state('FINAL_EVALUATION', completed_steps=HORIZON)
        evaluate(supervisor)
        summarize()
        state('EVALUATED_READY_FOR_GITHUB', completed_steps=HORIZON)
    except BaseException as error:
        dump(EXP / 'FAILURE_AUDIT.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now(),
             no_automatic_retry=True, original_common0_unchanged=sha(STEP0) == STEP0_SHA))
        state('ENGINEERING_FAILURE', error=repr(error))
        raise
    finally:
        dump(RUN / 'controller-final.json', dict(commands=supervisor.commands, utc=now(), owned_workers_terminated=True))
        lock.close()


def main():
    if '--worker' in sys.argv:
        sys.argv.remove('--worker'); worker(); return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'run'])
    args = p.parse_args()
    if args.phase == 'prepare': prepare()
    else: execute_controller()


if __name__ == '__main__': main()
