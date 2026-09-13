"""Background runner for the S0-TriMask-HS v0.2 experiment (task section 11).

    python tools/trimask_hs_runner.py --run-dir <dir> --init-state <shared init>

Sequence, with real exit codes preserved and an atomically written ``run_status.json`` after every
phase::

    train (500 updates) -> verify the exact step-500 checkpoint -> export the bare student
        -> COCO canonical -> Urban-1k -> final status

Rules that matter operationally:

* an exclusive lock file (``O_CREAT|O_EXCL`` holding the PID) refuses a second start; a stale lock
  whose PID is gone is reported and only cleared with ``--clear-stale-lock``;
* the four GPUs are re-checked before starting: ``nvidia-smi`` failing is treated as "not idle", never
  as idle, and a non-empty compute-application list refuses the run;
* the process detaches from the calling shell (``setsid``/``stdin=/dev/null`` are applied by the
  caller), writes its own log, and records the PID, the run id, the start time and the actual
  commands (with credentials stripped) into the status file;
* a failed training phase stops the sequence: an incomplete checkpoint is never evaluated;
* the dashboard only ever reads the status file -- this runner never reads the dashboard.
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
TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_said_trimask.py')
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_trimask_student.py')
CANONICAL_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'

from model.said_trimask import (LOSS_PROFILE_DEFAULT, LOSS_PROFILES,  # noqa: E402
                                profile_lambdas, profile_names)

# the default profile's names, kept as module constants so nothing about the frozen v0.2 run changes
ARM = 'S0_TriMask_HS'
OBJECTIVE = 'smartclip_trimask_hs'
GATE_MODE = 'hard_st'
SHARED_INIT_DEFAULT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/'
                       'cvssl_initial.pt')
SHARE4V_ROOT = '/root/datasets/ShareGPT4V'
SHARE4V_JSON = 'share-captioner_coco_lcs_sam_1246k_1107.json'
COCO_ROOT = '/root/datasets/coco'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
REQUIRED_GPUS = 4
SENSITIVE_KEYS = ('password', 'passwd', 'token', 'secret', 'key', 'credential')


# --------------------------------------------------------------------------- status file
def write_status(path, **fields):
    """Atomic: write a temporary file in the same directory, then ``os.replace`` it."""
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


# fields that describe one attempt and must never leak from an earlier failed run into a new one
ATTEMPT_FIELDS = ('train_exit_code', 'export_exit_code', 'coco_exit_code', 'urban_exit_code',
                  'failure_reason', 'checkpoint_problems', 'checkpoint_verified', 'conclusion')


def sanitise_argv(argv):
    """The command actually run, with anything credential-looking removed."""
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


# --------------------------------------------------------------------------- lock
def acquire_lock(path, clear_stale=False):
    """``O_CREAT|O_EXCL`` lock holding the PID; returns the file handle or raises."""
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
                             're-run with --clear-stale-lock to replace it'
                             % (path, pid))
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


# --------------------------------------------------------------------------- environment
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
    """The COCO annotation file must exist as an absolute path: the earlier S0-XPool evaluation
    failed because an unset COCO_DATA_ROOT silently fell back to a relative path."""
    annotations = os.path.join(COCO_ROOT, 'annotations', 'captions_val2017.json')
    images = os.path.join(COCO_ROOT, 'val2017')
    if not os.path.isfile(annotations):
        raise SystemExit('COCO annotations missing at the absolute path %s' % annotations)
    if not os.path.isdir(images):
        raise SystemExit('COCO val2017 images missing at %s' % images)
    return annotations, images


def resolve_torchrun(python, torchrun):
    """Resolve ``torchrun`` next to ``--python`` instead of trusting ``PATH``.

    The host has an unrelated ``/usr/local/bin/torchrun`` (python3.11) earlier in ``PATH``; using it
    launches the training script in the wrong interpreter and fails on the first import. A bare name
    therefore means "the launcher that ships with ``--python``".
    """
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
    """Run a subprocess, tee its output into the runner log, and return the real exit code."""
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write('\n$ %s\n' % ' '.join(sanitise_argv(argv)))
        handle.flush()
        process = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default='s0_trimask_hs_v02')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--init-state', default=SHARED_INIT_DEFAULT)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--resume', default=None,
                        help='continue from this checkpoint (default: train from the shared init)')
    parser.add_argument('--save-steps', dest='save_steps', default=None,
                        help='default: 0,20,100,250,<steps> for a fresh run and 750,<steps> when '
                             '--resume is given')
    parser.add_argument('--python', default='/root/miniconda3/envs/said-smartclip/bin/python')
    parser.add_argument('--torchrun', default='torchrun',
                        help='a bare name means "the torchrun next to --python"; PATH is never '
                             'trusted because an unrelated system torchrun may come first')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup-length', type=int, default=200)
    parser.add_argument('--lambda-sparse-t', type=float, default=None)
    parser.add_argument('--loss-profile', dest='loss_profile', default=LOSS_PROFILE_DEFAULT,
                        choices=list(LOSS_PROFILES),
                        help='default = the frozen 10/1/1 + 2/0.2 weighting under the v0.2 names; '
                             'balanced = all three alignment terms at 10 and both sparsity terms at '
                             '2, under its own arm/objective/phase names')
    parser.add_argument('--lock-file', default=None)
    parser.add_argument('--clear-stale-lock', action='store_true')
    parser.add_argument('--skip-gpu-check', action='store_true',
                        help='only for a re-run of the evaluation phases; never for training')
    parser.add_argument('--phases', default='train,export,coco,urban',
                        help='subset/order of phases to execute')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    # the loss profile owns the names and the five coefficients; the default profile reproduces the
    # v0.2 experiment exactly, so nothing about the frozen @500 run changes
    arm, objective, phase_name = profile_names(args.loss_profile, GATE_MODE)
    lambdas = profile_lambdas(args.loss_profile, GATE_MODE)
    if args.lambda_sparse_t is None:
        args.lambda_sparse_t = lambdas['lambda_sparse_t']
    elif float(args.lambda_sparse_t) != float(lambdas['lambda_sparse_t']):
        raise SystemExit('--lambda-sparse-t %r contradicts --loss-profile %s (which fixes it at %r)'
                         % (args.lambda_sparse_t, args.loss_profile, lambdas['lambda_sparse_t']))
    args.torchrun = resolve_torchrun(args.python, args.torchrun)
    if args.save_steps is None:
        args.save_steps = ('750,%d' % args.steps) if args.resume \
            else ('0,20,100,250,%d' % args.steps)
    save_steps = args.save_steps
    status_path = os.path.join(run_dir, 'run_status.json')
    log_path = os.path.join(run_dir, 'runner.log')
    lock_path = args.lock_file or (run_dir + '.lock')
    checkpoint = os.path.join(run_dir, 'trimask_%s_step%06d.pt' % (arm, args.steps))
    student = os.path.join(run_dir, 'student_%06d.pt' % args.steps)
    evaluation_dir = os.path.join(run_dir, 'evaluation')
    phases = [phase.strip() for phase in args.phases.split(',') if phase.strip()]

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
                     arm=arm, objective=objective, text_gate_mode=GATE_MODE,
                     loss_profile=args.loss_profile, phase_name=phase_name, **lambdas,
                     max_steps=args.steps,
                     run_dir=run_dir, completed_steps=0, exit_codes={},
                     resume=args.resume, save_steps=save_steps,
                     attempt=int(previous.get('attempt') or 0) + 1,
                     attempt_history=history[-5:], lock_file=lock_path, phases=phases,
                     **reset)
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
        annotations, images = verify_coco_environment()
        write_status(status_path, coco_annotations=annotations, coco_images=images)

        env = training_environment()
        exit_codes = {}

        # ---- phase 1: training -------------------------------------------------
        if 'train' in phases:
            if os.path.exists(checkpoint):
                raise SystemExit('REFUSING to retrain: %s already exists' % checkpoint)
            port = 35500 + (os.getpid() % 400)
            train_argv = [args.torchrun, '--nproc_per_node=%d' % args.nproc,
                          '--master_port=%d' % port, TRAIN_SCRIPT,
                          '--text_gate_mode', GATE_MODE,
                          '--loss_profile', args.loss_profile,
                          '--lambda_1', str(lambdas['lambda_1']),
                          '--lambda_2', str(lambdas['lambda_2']),
                          '--lambda_3', str(lambdas['lambda_3']),
                          '--lambda_sparse_i', str(lambdas['lambda_sparse_i']),
                          '--lambda_sparse_t', str(args.lambda_sparse_t),
                          '--base_model', 'B16', '--batch-size', str(args.batch_size),
                          '--epochs', str(args.epochs), '--lr', '1e-6', '--mask_lr', '1e-3',
                          '--weight_decay', '1e-2', '--warmup_length', str(args.warmup_length),
                          '--seed', '0', '--init_state', args.init_state,
                          '--output_dir', run_dir, '--max_steps', str(args.steps),
                          '--save_completed_steps', save_steps,
                          '--log_every', '10', '--heavy_log_every', '25',
                          '--grad_health_steps', str(args.steps),
                          '--grad_checkpoint_views', '1', '--num_workers', '8',
                          '--amp_dtype', 'bf16']
            if args.resume:
                train_argv += ['--resume', args.resume]
            write_status(status_path, phase='training', commands={'train': sanitise_argv(train_argv)})
            exit_codes['train'] = run_command(train_argv, log_path, env=env)
            write_status(status_path, train_exit_code=exit_codes['train'],
                         exit_codes=exit_codes,
                         completed_steps=_completed_steps(run_dir))
            if exit_codes['train'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='training exited %d; no evaluation attempted'
                                            % exit_codes['train'])
                raise SystemExit('training failed with exit code %d' % exit_codes['train'])

        # ---- phase 2: checkpoint verification ----------------------------------
        summary_path = os.path.join(run_dir, 'run_summary.json')
        if not os.path.isfile(checkpoint):
            write_status(status_path, phase='failed',
                         failure_reason='expected checkpoint missing: %s' % checkpoint)
            raise SystemExit('missing checkpoint %s' % checkpoint)
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        problems = []
        if payload.get('objective') != objective:
            problems.append('objective=%r' % payload.get('objective'))
        if payload.get('arm') != arm:
            problems.append('arm=%r' % payload.get('arm'))
        if payload.get('text_gate_mode') != GATE_MODE:
            problems.append('text_gate_mode=%r' % payload.get('text_gate_mode'))
        if (payload.get('loss_profile') or LOSS_PROFILE_DEFAULT) != args.loss_profile:
            problems.append('loss_profile=%r' % payload.get('loss_profile'))
        if float(payload.get('lambda_sparse_t', -1)) != float(args.lambda_sparse_t):
            problems.append('lambda_sparse_t=%r' % payload.get('lambda_sparse_t'))
        if int(payload.get('completed_steps', -1)) != args.steps:
            problems.append('completed_steps=%r' % payload.get('completed_steps'))
        config = payload.get('config') or {}
        # every coefficient of the requested profile, not a hard-coded 10/1/1: the profile is the
        # experiment identity, and a checkpoint trained with other weights must never pass here
        for name in ('lambda_1', 'lambda_2', 'lambda_3', 'lambda_sparse_i', 'lambda_sparse_t'):
            if float(config.get(name, float('nan'))) != float(lambdas[name]):
                problems.append('%s=%r' % (name, config.get(name)))
        if int(payload.get('lr_horizon_steps', -1)) != int(config.get('lr_horizon_steps', -2)):
            problems.append('lr_horizon_steps mismatch')
        write_status(status_path, checkpoint=checkpoint, checkpoint_verified=not problems,
                     checkpoint_problems=problems,
                     completed_steps=int(payload.get('completed_steps', -1)),
                     lr_horizon_steps=payload.get('lr_horizon_steps'))
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
                               '--expect-gate-mode', GATE_MODE,
                               '--expect-loss-profile', args.loss_profile]
                write_status(status_path,
                             commands={'export': sanitise_argv(export_argv)})
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

        # ---- phase 4/5: the two evaluations ------------------------------------
        if 'coco' in phases:
            os.makedirs(evaluation_dir, exist_ok=True)
            canonical_out = os.path.join(evaluation_dir, '%s_step%06d_canonical.json'
                                         % (arm, args.steps))
            coco_argv = [args.python, CANONICAL_EVAL, '--label', arm,
                         '--gap_anti_temperature', '1.0', '--sharegpt4v_manifest', '',
                         '--data_root', SHARE4V_ROOT, '--image_root', SHARE4V_ROOT,
                         '--image_batch_size', '64', '--canonical', '--canonical_only', '--coco',
                         '--canonical_tags', str(args.steps),
                         '--canonical_names', '%s@%d' % (arm, args.steps),
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
                                     % (arm, args.steps))
            urban_argv = [args.python, URBAN_EVAL, '--checkpoint', student,
                          '--label', '%s@%d' % (arm, args.steps),
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

        conclusion = _conclusion(run_dir, args.steps, arm)
        write_status(status_path, phase='complete', exit_codes=exit_codes,
                     conclusion=conclusion, finished_at=time.time(),
                     wall_seconds=time.time() - started, student=student)
        print('RUNNER_COMPLETE ' + json.dumps({'exit_codes': exit_codes,
                                               'conclusion': conclusion}, sort_keys=True),
              flush=True)
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


def _conclusion(run_dir, steps, arm=ARM):
    """The gate verdict, recomputed from the produced evaluation files (never invented)."""
    canonical = os.path.join(run_dir, 'evaluation', '%s_step%06d_canonical.json' % (arm, steps))
    try:
        with open(canonical, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        inner = payload['canonical']['%s@%d' % (arm, steps)]['coco_val2017']
    except (ValueError, OSError, KeyError, TypeError) as error:
        return {'verdict': 'unknown', 'reason': 'cannot read %s (%s)' % (canonical, error)}
    i2t, t2i = inner['image2text_R1'], inner['text2image_R1']
    passed = (i2t >= 0.6058 and t2i >= 0.41236 and (i2t > 0.6058 or t2i > 0.41236))
    return {'verdict': 'PROMISING_AT_500' if passed else 'FAIL',
            'coco_i2t_r1': i2t, 'coco_t2i_r1': t2i,
            'gate': 'COCO I2T R@1 >= 0.6058 and T2I R@1 >= 0.41236, at least one strictly higher'}


if __name__ == '__main__':
    main()
