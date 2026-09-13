"""Background runner for PG-CLIP v0.1 (Pre-Projection Gated CLIP).

    python tools/pgclip_runner.py --run-dir <dir> [--init-state <shared init>]

Sequence with real exit codes and an atomically written ``run_status.json`` after every phase::

    train (exactly 500 updates) -> verify the exact step-500 checkpoint -> export the bare student
        -> COCO canonical -> Urban-1k -> conclusion

Operational rules (the same ones the frozen v0.2 runner uses):

* an exclusive ``O_CREAT|O_EXCL`` lock holding the PID refuses a second start; a stale lock whose PID
  is gone is reported and only cleared with ``--clear-stale-lock``;
* the four GPUs are re-queried before training; an ``nvidia-smi`` failure is treated as "not idle",
  never as idle, and a non-empty compute-application list refuses the run;
* the process detaches from the calling shell (``setsid``/``stdin=/dev/null`` are applied by the
  caller), writes its own log and records the PID, run id, start time, implementation SHA and the
  actual commands (credentials stripped);
* a failed phase stops the sequence: an incomplete checkpoint is never evaluated, and a run that has
  already trained to 500 can only re-run export/eval/report (never retrain to fix a tool);
* the dashboard only reads the status file -- this runner never reads the dashboard.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model.pgclip import (ARM, GATE_HEADS, GATE_LAYERS, GATE_MODE, GATE_OUT, GATE_SEED,  # noqa: E402
                          GATE_WIDTH, IMAGE_CHUNK_DEFAULT, LAMBDA_GLOBAL, LAMBDA_PREPROJ,
                          LAMBDA_SPARSE, OBJECTIVE, PHASE, TEXT_CHUNK_DEFAULT, config_dict,
                          file_sha256, git_head)

TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_pgclip.py')
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_pgclip_student.py')
CANONICAL_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'

SHARED_INIT_DEFAULT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/'
                       'cvssl_initial.pt')
SHARE4V_ROOT = '/root/datasets/ShareGPT4V'
SHARE4V_JSON = 'share-captioner_coco_lcs_sam_1246k_1107.json'
COCO_ROOT = '/root/datasets/coco'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
REQUIRED_GPUS = 4
SENSITIVE_KEYS = ('password', 'passwd', 'token', 'secret', 'key', 'credential')
ATTEMPT_FIELDS = ('train_exit_code', 'export_exit_code', 'coco_exit_code', 'urban_exit_code',
                  'failure_reason', 'checkpoint_problems', 'checkpoint_verified', 'conclusion')
# the frozen S0@500 reference the promotion gate is defined against
GATE = {'coco_i2t_r1': 0.6058, 'coco_t2i_r1': 0.41236}


def write_status(path, **fields):
    payload = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            payload = {}
    payload.update(fields)
    payload['updated_at'] = time.time()
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return payload


def sanitise_argv(argv):
    cleaned, redact_next = [], False
    for token in argv:
        if redact_next:
            cleaned.append('<redacted>')
            redact_next = False
            continue
        lowered = token.lower()
        if any(key in lowered for key in SENSITIVE_KEYS) and '=' in token:
            cleaned.append(token.split('=', 1)[0] + '=<redacted>')
            continue
        if lowered in ('--password', '--token', '--secret'):
            cleaned.append(token)
            redact_next = True
            continue
        cleaned.append(token)
    return cleaned


def acquire_lock(path, clear_stale=False):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                holder = json.load(handle)
        except (ValueError, OSError):
            holder = {'pid': None}
        pid = holder.get('pid')
        alive = False
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit('REFUSING to start: %s is held by live pid %s (started %s)'
                             % (path, pid, holder.get('started_at')))
        if not clear_stale:
            raise SystemExit('REFUSING to start: %s holds a stale lock (pid %s is gone); '
                             're-run with --clear-stale-lock to replace it' % (path, pid))
        os.unlink(path)
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, json.dumps({'pid': os.getpid(), 'started_at': time.time(),
                                 'started_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                                ).encode('utf-8'))
    return handle


def release_lock(handle, path):
    try:
        os.close(handle)
    except OSError:
        pass
    try:
        os.unlink(path)
    except OSError:
        pass


def gpu_report():
    """``(ok, detail)`` -- a failing nvidia-smi is never treated as an idle machine."""
    try:
        listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return False, 'nvidia-smi could not be executed: %s' % error
    if listing.returncode != 0:
        return False, 'nvidia-smi -L exited %d: %s' % (listing.returncode, listing.stderr.strip())
    devices = [line for line in listing.stdout.splitlines() if line.startswith('GPU ')]
    if len(devices) < REQUIRED_GPUS:
        return False, 'only %d GPU(s) visible, %d required' % (len(devices), REQUIRED_GPUS)
    try:
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return False, 'nvidia-smi compute-app query failed: %s' % error
    if busy.returncode != 0:
        return False, 'nvidia-smi compute-app query exited %d' % busy.returncode
    running = [line for line in busy.stdout.splitlines() if line.strip()]
    if running:
        return False, 'GPUs are busy: %s' % '; '.join(running)
    return True, '%d GPUs idle' % len(devices)


def training_environment():
    env = dict(os.environ)
    env.update({
        'SHARE4V_DATA_ROOT': SHARE4V_ROOT,
        'SHARE4V_JSON': SHARE4V_JSON,
        'SHARE4V_FULL_AUDIT': '/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json',
        'COCO_DATA_ROOT': COCO_ROOT,
        'NCCL_SOCKET_IFNAME': 'lo',
        'GLOO_SOCKET_IFNAME': 'lo',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    })
    return env


def verify_coco_environment():
    """The COCO and Urban-1k absolute paths must exist before any evaluation is attempted.

    Urban-1k uses the upstream layout ``<root>/image`` + ``<root>/caption`` (paired by stem); a
    missing directory is an error and no other 1k set is substituted for it.
    """
    annotations = os.path.join(COCO_ROOT, 'annotations', 'captions_val2017.json')
    images = os.path.join(COCO_ROOT, 'val2017')
    if not os.path.isfile(annotations):
        raise SystemExit('COCO annotations missing at the absolute path %s' % annotations)
    if not os.path.isdir(images):
        raise SystemExit('COCO val2017 images missing at %s' % images)
    urban_images = os.path.join(URBAN_ROOT, 'image')
    urban_captions = os.path.join(URBAN_ROOT, 'caption')
    if not os.path.isdir(urban_images) or not os.path.isdir(urban_captions):
        raise SystemExit('Urban-1k layout not found under %s (expected image/ and caption/; no '
                         'substitute 1k set is used)' % URBAN_ROOT)
    image_stems = {os.path.splitext(name)[0] for name in os.listdir(urban_images)}
    caption_stems = {os.path.splitext(name)[0] for name in os.listdir(urban_captions)}
    if image_stems != caption_stems or not image_stems:
        raise SystemExit('Urban-1k image/caption stems do not match under %s (%d/%d)'
                         % (URBAN_ROOT, len(image_stems), len(caption_stems)))
    return annotations, images


def resolve_torchrun(python, torchrun):
    if os.sep in torchrun or (os.altsep and os.altsep in torchrun):
        if not os.path.exists(torchrun):
            raise SystemExit('--torchrun %s does not exist' % torchrun)
        return torchrun
    sibling = os.path.join(os.path.dirname(os.path.abspath(python)), torchrun)
    if not os.path.exists(sibling):
        raise SystemExit('cannot find %r next to --python %s; pass an absolute --torchrun'
                         % (torchrun, python))
    return sibling


def run_command(argv, log_path, env=None):
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write('\n$ %s\n' % ' '.join(sanitise_argv(argv)))
        handle.flush()
        process = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default='pgclip_v01')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--init-state', default=SHARED_INIT_DEFAULT)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--save-steps', dest='save_steps', default='0,20,100,250,500')
    parser.add_argument('--python', default='/root/miniconda3/envs/said-smartclip/bin/python')
    parser.add_argument('--torchrun', default='torchrun')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup-length', type=int, default=200)
    parser.add_argument('--clip-lr', type=float, default=1e-6)
    parser.add_argument('--gate-lr', type=float, default=1e-3)
    parser.add_argument('--image-chunk', type=int, default=IMAGE_CHUNK_DEFAULT)
    parser.add_argument('--text-chunk', type=int, default=TEXT_CHUNK_DEFAULT)
    parser.add_argument('--qp-checkpoint', type=int, default=1)
    parser.add_argument('--lock-file', default=None)
    parser.add_argument('--clear-stale-lock', action='store_true')
    parser.add_argument('--skip-gpu-check', action='store_true',
                        help='only for a re-run of the evaluation phases; never for training')
    parser.add_argument('--allow-missing-gate-state', action='store_true',
                        help='continue with the student-only export/evaluation when a checkpoint '
                             'predates schema v1.1 and does not carry the gate tensors; the loss is '
                             'recorded in the status file and the report, never hidden')
    parser.add_argument('--phases', default='train,export,coco,urban')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    args.torchrun = resolve_torchrun(args.python, args.torchrun)
    status_path = os.path.join(run_dir, 'run_status.json')
    log_path = os.path.join(run_dir, 'runner.log')
    lock_path = args.lock_file or (run_dir + '.lock')
    checkpoint = os.path.join(run_dir, 'pgclip_%s_step%06d.pt' % (ARM, args.steps))
    student = os.path.join(run_dir, 'student_%06d.pt' % args.steps)
    evaluation_dir = os.path.join(run_dir, 'evaluation')
    phases = [phase.strip() for phase in args.phases.split(',') if phase.strip()]
    implementation_sha = git_head(REPO)

    lock = acquire_lock(lock_path, clear_stale=args.clear_stale_lock)
    started = time.time()
    try:
        previous = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as handle:
                    previous = json.load(handle)
            except (ValueError, OSError):
                previous = {}
        history = list(previous.get('attempt_history') or [])
        if previous.get('phase') in ('failed', 'complete'):
            history.append({'phase': previous.get('phase'),
                            'finished_at': previous.get('finished_at'),
                            'exit_codes': previous.get('exit_codes'),
                            'failure_reason': previous.get('failure_reason')})
        reset = {field: None for field in ATTEMPT_FIELDS}
        write_status(status_path, run_id=args.run_id, phase='starting', pid=os.getpid(),
                     started_at=started,
                     started_at_iso=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(started)),
                     arm=ARM, objective=OBJECTIVE, phase_name=PHASE, gate_mode=GATE_MODE,
                     lambda_global=LAMBDA_GLOBAL, lambda_preproj=LAMBDA_PREPROJ,
                     lambda_sparse=LAMBDA_SPARSE, max_steps=args.steps, run_dir=run_dir,
                     completed_steps=0, exit_codes={}, resume=args.resume,
                     save_steps=args.save_steps, implementation_sha=implementation_sha,
                     attempt=int(previous.get('attempt') or 0) + 1, attempt_history=history[-5:],
                     lock_file=lock_path, phases=phases, **reset)
        if 'train' in phases:
            ok, detail = gpu_report()
            write_status(status_path, gpu_check=detail, gpu_ok=bool(ok))
            if not ok and not args.skip_gpu_check:
                write_status(status_path, phase='failed',
                             failure_reason='GPU pre-check refused to start: %s' % detail)
                raise SystemExit('REFUSING to start training: %s' % detail)
            if not os.path.isfile(args.init_state):
                write_status(status_path, phase='failed',
                             failure_reason='shared init missing: %s' % args.init_state)
                raise SystemExit('shared init missing: %s' % args.init_state)
            write_status(status_path, init_state=args.init_state,
                         init_file_sha256=file_sha256(args.init_state))
        try:
            annotations, images = verify_coco_environment()
        except SystemExit as error:
            write_status(status_path, phase='failed',
                         failure_reason='data pre-check refused to start: %s' % error)
            raise
        write_status(status_path, coco_annotations=annotations, coco_images=images,
                     urban_root=URBAN_ROOT)
        env = training_environment()
        exit_codes = {}

        # ---- phase 1: training -------------------------------------------------
        if 'train' in phases:
            if os.path.exists(checkpoint):
                raise SystemExit('REFUSING to retrain: %s already exists' % checkpoint)
            port = 35600 + (os.getpid() % 300)
            train_argv = [args.torchrun, '--nproc_per_node=%d' % args.nproc,
                          '--master_port=%d' % port, TRAIN_SCRIPT,
                          '--base_model', 'B16', '--batch-size', str(args.batch_size),
                          '--epochs', str(args.epochs), '--lr', str(args.clip_lr),
                          '--gate_lr', str(args.gate_lr), '--weight_decay', '1e-2',
                          '--warmup_length', str(args.warmup_length), '--seed', '0',
                          '--init_state', args.init_state, '--output_dir', run_dir,
                          '--max_steps', str(args.steps),
                          '--save_completed_steps', args.save_steps,
                          '--log_every', '10', '--heavy_log_every', '25',
                          '--grad_health_steps', str(args.steps),
                          '--image_chunk', str(args.image_chunk),
                          '--text_chunk', str(args.text_chunk),
                          '--qp_checkpoint', str(args.qp_checkpoint),
                          '--num_workers', '8', '--amp_dtype', 'bf16']
            if args.resume:
                train_argv += ['--resume', args.resume]
            write_status(status_path, phase='training', commands={'train': sanitise_argv(train_argv)})
            exit_codes['train'] = run_command(train_argv, log_path, env=env)
            write_status(status_path, train_exit_code=exit_codes['train'], exit_codes=exit_codes,
                         completed_steps=_completed_steps(run_dir))
            if exit_codes['train'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='training exited %d; no evaluation attempted'
                                            % exit_codes['train'])
                raise SystemExit('training failed with exit code %d' % exit_codes['train'])

        # ---- phase 2: checkpoint verification ----------------------------------
        if not os.path.isfile(checkpoint):
            write_status(status_path, phase='failed',
                         failure_reason='expected checkpoint missing: %s' % checkpoint)
            raise SystemExit('missing checkpoint %s' % checkpoint)
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        identity = config_dict()
        problems = []
        if payload.get('objective') != OBJECTIVE:
            problems.append('objective=%r' % payload.get('objective'))
        if payload.get('arm') != ARM:
            problems.append('arm=%r' % payload.get('arm'))
        if payload.get('phase') != PHASE:
            problems.append('phase=%r' % payload.get('phase'))
        if payload.get('gate_mode') != GATE_MODE:
            problems.append('gate_mode=%r' % payload.get('gate_mode'))
        if int(payload.get('completed_steps', -1)) != args.steps:
            problems.append('completed_steps=%r' % payload.get('completed_steps'))
        weights = payload.get('loss_weights') or {}
        for name, want in (('global', LAMBDA_GLOBAL), ('preproj', LAMBDA_PREPROJ),
                           ('sparse', LAMBDA_SPARSE)):
            if float(weights.get(name, float('nan'))) != float(want):
                problems.append('loss_weights[%s]=%r' % (name, weights.get(name)))
        config = payload.get('config') or {}
        for name in ('objective', 'arm', 'gate_mode', 'lambda_global', 'lambda_preproj',
                     'lambda_sparse', 'gate_width', 'gate_out', 'gate_layers', 'gate_heads',
                     'fixed_scale', 'norm_eps'):
            got = config.get(name, payload.get(name))
            if got != identity[name]:
                problems.append('%s=%r' % (name, got))
        recorded_batch = int(payload.get('batch_size_per_gpu', -1))
        if recorded_batch != args.batch_size:
            problems.append('batch_size_per_gpu=%r' % recorded_batch)
        if int(payload.get('world_size', -1)) != args.nproc:
            problems.append('world_size=%r' % payload.get('world_size'))
        gate = payload.get('gate') or {}
        # the gate identity lives in the config block of every checkpoint (and, from schema v1.1 on,
        # flat at the top level too); the first production run predates the flat keys, so both shapes
        # are accepted here and the *tensors* are checked as well
        gate_config = config.get('gate') or {}
        identity_gate = {'gate_mode': config.get('gate_mode', payload.get('gate_mode')),
                         'gate_width': config.get('gate_width', payload.get('gate_width')),
                         'gate_out': config.get('gate_out', payload.get('gate_out')),
                         'gate_layers': config.get('gate_layers', payload.get('gate_layers')),
                         'gate_heads': config.get('gate_heads', payload.get('gate_heads')),
                         'gate_seed': config.get('gate_seed', payload.get('gate_seed'))}
        for name, want in (('gate_mode', GATE_MODE), ('gate_width', GATE_WIDTH),
                           ('gate_out', GATE_OUT), ('gate_layers', GATE_LAYERS),
                           ('gate_heads', GATE_HEADS), ('gate_seed', GATE_SEED)):
            if identity_gate.get(name) != want:
                problems.append('gate[%s]=%r' % (name, identity_gate.get(name)))
        if gate_config.get('mode') and 'straight-through' not in gate_config['mode']:
            problems.append('gate mode is not the hard straight-through gate: %r'
                            % gate_config.get('mode'))
        import torch as _torch
        gate_tensors = payload.get('gate_state')
        if not isinstance(gate_tensors, dict) or not gate_tensors:
            # schema v1.0 of the first production run wrote the gate TENSORS to the same key that
            # carries the gate description, so the trained weights were not persisted. The primary
            # evaluation uses the bare CLIP student only, so the run can continue with an explicit
            # flag; the loss is recorded in the status and in the report instead of being hidden.
            legacy = payload.get('gate')
            if isinstance(legacy, dict) and legacy and all(
                    hasattr(value, 'shape') for value in legacy.values()):
                gate_tensors = legacy
            else:
                gate_tensors = None
        gate_state_missing = gate_tensors is None
        missing_note = None
        if gate_state_missing:
            if not args.allow_missing_gate_state:
                problems.append('the checkpoint carries no gate tensors; re-run with '
                                '--allow-missing-gate-state to continue with the student-only '
                                'export and evaluation')
            else:
                missing_note = ('the trained gate weights are absent from this checkpoint '
                                '(schema v1.0 key collision); the student export and both '
                                'evaluations do not use the gate')
        else:
            shapes = {name: tuple(value.shape) for name, value in gate_tensors.items()}
            if shapes.get('projection.weight') != (GATE_OUT, GATE_WIDTH):
                problems.append('gate projection.weight shape %r' % (shapes.get('projection.weight'),))
            if shapes.get('projection.bias') != (GATE_OUT,):
                problems.append('gate projection.bias shape %r' % (shapes.get('projection.bias'),))
            if not any(name.startswith('stem.') for name in shapes):
                problems.append('the gate stem tensors are missing')
            clip_tensors = payload.get('clip') or {}
            if tuple(clip_tensors.get('visual.proj', _torch.empty(0)).shape) != (GATE_OUT, GATE_WIDTH):
                problems.append('clip.visual.proj shape is not %r'
                                % ((GATE_OUT, GATE_WIDTH),))
            shared = [name for name in gate_tensors if name in clip_tensors]
            if shared:
                problems.append('the gate duplicates clip parameters: %r' % shared[:3])
        if config.get('lr') != args.clip_lr or config.get('gate_lr') != args.gate_lr:
            problems.append('lr=%r gate_lr=%r' % (config.get('lr'), config.get('gate_lr')))
        horizon = payload.get('lr_horizon_steps')
        if horizon is None:
            horizon = config.get('lr_horizon_steps')
        if horizon is None or int(horizon) <= 0 or int(horizon) != int(
                config.get('lr_horizon_steps', -2)):
            problems.append('lr_horizon_steps mismatch (%r vs %r)'
                            % (payload.get('lr_horizon_steps'), config.get('lr_horizon_steps')))
        write_status(status_path, checkpoint=checkpoint, checkpoint_verified=not problems,
                     checkpoint_problems=problems,
                     completed_steps=int(payload.get('completed_steps', -1)),
                     lr_horizon_steps=horizon,
                     gate_state_missing=gate_state_missing,
                     gate_state_note=(missing_note if gate_state_missing
                                      and args.allow_missing_gate_state else None),
                     checkpoint_gate_tensors=(sorted(gate_tensors)[:6]
                                              if isinstance(gate_tensors, dict) else None),
                     gate_config=gate_config or gate, loss_weights=weights)
        if problems:
            write_status(status_path, phase='failed',
                         failure_reason='checkpoint verification failed: %s' % problems)
            raise SystemExit('checkpoint verification failed: %s' % problems)
        del payload

        # ---- phase 3: bare student export --------------------------------------
        if 'export' in phases or 'coco' in phases or 'urban' in phases:
            if not os.path.exists(student):
                write_status(status_path, phase='exporting')
                export_argv = [args.python, EXPORT_SCRIPT, '--checkpoint', checkpoint,
                               '--out', student, '--expect-steps', str(args.steps),
                               '--expect-objective', OBJECTIVE, '--expect-arm', ARM]
                write_status(status_path, commands=dict(_status_commands(status_path),
                                                        export=sanitise_argv(export_argv)))
                exit_codes['export'] = run_command(export_argv, log_path, env=env)
                write_status(status_path, export_exit_code=exit_codes['export'],
                             exit_codes=exit_codes)
                if exit_codes['export'] != 0:
                    write_status(status_path, phase='failed',
                                 failure_reason='export exited %d' % exit_codes['export'])
                    raise SystemExit('export failed with exit code %d' % exit_codes['export'])
            if not os.path.isfile(student):
                write_status(status_path, phase='failed',
                             failure_reason='exported student missing: %s' % student)
                raise SystemExit('missing student %s' % student)
            write_status(status_path, student=student, student_sha256=file_sha256(student))

        # ---- phase 4/5: the two evaluations ------------------------------------
        if 'coco' in phases:
            os.makedirs(evaluation_dir, exist_ok=True)
            canonical_out = os.path.join(evaluation_dir, '%s_step%06d_canonical.json'
                                         % (ARM, args.steps))
            coco_argv = [args.python, CANONICAL_EVAL, '--label', ARM,
                         '--gap_anti_temperature', '1.0', '--sharegpt4v_manifest', '',
                         '--data_root', SHARE4V_ROOT, '--image_root', SHARE4V_ROOT,
                         '--image_batch_size', '64', '--canonical', '--canonical_only', '--coco',
                         '--canonical_tags', str(args.steps),
                         '--canonical_names', '%s@%d' % (ARM, args.steps),
                         '--checkpoints', '%d:%s' % (args.steps, student),
                         '--coco_root', os.path.join(COCO_ROOT, 'val2017'),
                         '--output', canonical_out]
            write_status(status_path, phase='evaluating_coco',
                         commands=dict(_status_commands(status_path),
                                       coco=sanitise_argv(coco_argv)))
            exit_codes['coco'] = run_command(coco_argv, log_path, env=env)
            write_status(status_path, coco_exit_code=exit_codes['coco'], exit_codes=exit_codes,
                         coco_output=canonical_out)
            if exit_codes['coco'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='COCO canonical evaluation exited %d'
                                            % exit_codes['coco'])
                raise SystemExit('COCO evaluation failed')

        if 'urban' in phases:
            os.makedirs(evaluation_dir, exist_ok=True)
            urban_out = os.path.join(evaluation_dir, '%s_step%06d_urban1k.json'
                                     % (ARM, args.steps))
            urban_argv = [args.python, URBAN_EVAL, '--checkpoint', student,
                          '--label', '%s@%d' % (ARM, args.steps),
                          '--expect-steps', str(args.steps), '--base_model', 'ViT-B/16',
                          '--device', 'cuda', '--batch_size', '64',
                          '--urban_root', URBAN_ROOT, '--out', urban_out]
            write_status(status_path, phase='evaluating_urban',
                         commands=dict(_status_commands(status_path),
                                       urban=sanitise_argv(urban_argv)))
            exit_codes['urban'] = run_command(urban_argv, log_path, env=env)
            write_status(status_path, urban_exit_code=exit_codes['urban'], exit_codes=exit_codes,
                         urban_output=urban_out)
            if exit_codes['urban'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='Urban-1k evaluation exited %d' % exit_codes['urban'])
                raise SystemExit('Urban-1k evaluation failed')

        conclusion = _conclusion(run_dir, args.steps)
        write_status(status_path, phase='complete', exit_codes=exit_codes, conclusion=conclusion,
                     finished_at=time.time(), wall_seconds=time.time() - started, student=student,
                     implementation_sha=implementation_sha)
        print('RUNNER_COMPLETE ' + json.dumps({'exit_codes': exit_codes, 'conclusion': conclusion},
                                              sort_keys=True), flush=True)
    finally:
        release_lock(lock, lock_path)


def _status_commands(status_path):
    try:
        with open(status_path, 'r', encoding='utf-8') as handle:
            return json.load(handle).get('commands') or {}
    except (ValueError, OSError):
        return {}


def _completed_steps(run_dir):
    summary = os.path.join(run_dir, 'run_summary.json')
    try:
        with open(summary, 'r', encoding='utf-8') as handle:
            return int(json.load(handle).get('completed_steps', 0))
    except (ValueError, OSError, TypeError):
        return None


def _conclusion(run_dir, steps):
    """The frozen promotion gate, recomputed from the produced files (never invented)."""
    canonical = os.path.join(run_dir, 'evaluation', '%s_step%06d_canonical.json' % (ARM, steps))
    try:
        with open(canonical, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        inner = payload['canonical']['%s@%d' % (ARM, steps)]['coco_val2017']
    except (ValueError, OSError, KeyError, TypeError) as error:
        return {'verdict': 'unknown', 'reason': 'cannot read %s (%s)' % (canonical, error)}
    i2t, t2i = inner['image2text_R1'], inner['text2image_R1']
    passed = (i2t >= GATE['coco_i2t_r1'] and t2i >= GATE['coco_t2i_r1']
              and (i2t > GATE['coco_i2t_r1'] or t2i > GATE['coco_t2i_r1']))
    return {'verdict': 'PROMISING_AT_500' if passed else 'FAIL',
            'coco_i2t_r1': i2t, 'coco_t2i_r1': t2i,
            'i2t_delta_points': (i2t - GATE['coco_i2t_r1']) * 100.0,
            't2i_delta_points': (t2i - GATE['coco_t2i_r1']) * 100.0,
            'gate': 'COCO I2T R@1 >= 0.6058 and T2I R@1 >= 0.41236, at least one strictly higher',
            'urban_is_separate': True}


if __name__ == '__main__':
    main()
