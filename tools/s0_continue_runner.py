"""Background runner for the S0 continuation (500 -> 1000 optimizer updates).

    python tools/s0_continue_runner.py --run-dir <dir> --resume <S0@500 checkpoint>

Sequence, with real exit codes and an atomically written ``run_status.json`` after every phase::

    train (resume at batch 500, replay-verified) -> verify the exact step-1000 checkpoint
        -> export the bare student -> COCO canonical -> Urban-1k -> final status

Same structure and status format as ``tools/trimask_hs_runner.py`` in the S0-TriMask worktrees, so
the read-only dashboard can display both continuations with the same code. Kept as a separate file
because the S0 arm uses a different trainer, a different objective/arm name and a different
checkpoint payload.
"""
import argparse
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_said_cls_cvssl.py')
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_bare_student.py')
CANONICAL_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'
S0_500 = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/ddpfix_step500_S0_smartclip/'
          'cvssl_S0_smartclip_step000500.pt')
SHARED_INIT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/'
               'cvssl_initial.pt')
SHARE4V_ROOT = '/root/datasets/ShareGPT4V'
COCO_ROOT = '/root/datasets/coco'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
ARM = 'S0_smartclip'
OBJECTIVE = 'said_cls_cvssl'
LR_HORIZON = 3651
REQUIRED_GPUS = 4
SENSITIVE_KEYS = ('password', 'passwd', 'token', 'secret', 'key', 'credential')
ATTEMPT_FIELDS = ('train_exit_code', 'export_exit_code', 'coco_exit_code', 'urban_exit_code',
                  'failure_reason', 'checkpoint_problems', 'checkpoint_verified', 'conclusion')


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
        if '=' in token and any(key in lowered for key in SENSITIVE_KEYS):
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
            raise SystemExit('REFUSING to start: %s is held by live pid %s' % (path, pid))
        if not clear_stale:
            raise SystemExit('REFUSING to start: %s holds a stale lock (pid %s is gone); re-run with '
                             '--clear-stale-lock' % (path, pid))
        os.unlink(path)
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, json.dumps({'pid': os.getpid(), 'started_at': time.time()}).encode('utf-8'))
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
    try:
        listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return False, 'nvidia-smi could not be executed: %s' % error
    if listing.returncode != 0:
        return False, 'nvidia-smi -L exited %d' % listing.returncode
    devices = [line for line in listing.stdout.splitlines() if line.startswith('GPU ')]
    if len(devices) < REQUIRED_GPUS:
        return False, 'only %d GPU(s) visible, %d required' % (len(devices), REQUIRED_GPUS)
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=30)
    if busy.returncode != 0:
        return False, 'nvidia-smi compute-app query exited %d' % busy.returncode
    running = [line for line in busy.stdout.splitlines() if line.strip()]
    if running:
        return False, 'GPUs are busy: %s' % '; '.join(running)
    return True, '%d GPUs idle' % len(devices)


def verify_coco_environment():
    annotations = os.path.join(COCO_ROOT, 'annotations', 'captions_val2017.json')
    images = os.path.join(COCO_ROOT, 'val2017')
    if not os.path.isfile(annotations):
        raise SystemExit('COCO annotations missing at %s' % annotations)
    if not os.path.isdir(images):
        raise SystemExit('COCO val2017 images missing at %s' % images)
    return annotations, images


def run_command(argv, log_path, env=None):
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write('\n$ %s\n' % ' '.join(sanitise_argv(argv)))
        handle.flush()
        return subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default='s0_continue_1000')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--resume', default=S0_500)
    parser.add_argument('--init-state', default=SHARED_INIT)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save-steps', dest='save_steps', default=None)
    parser.add_argument('--grad-probe-steps', dest='grad_probe_steps', default='',
                        help="steps at which to run the trainer's gradient probe. Default '' (no "
                             'probe), which is what the frozen S0 run used: the probe keeps several '
                             'retained backward graphs alive and OOMed all four ranks at 79.1/79.3 '
                             'GiB when it was scheduled at step 750. It yields diagnostics only, so '
                             'the continuation runs without it rather than changing the budget.')
    parser.add_argument('--python', default='/root/miniconda3/envs/said-smartclip/bin/python')
    parser.add_argument('--torchrun', default='torchrun')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup-length', type=int, default=200)
    parser.add_argument('--lock-file', default=None)
    parser.add_argument('--clear-stale-lock', action='store_true')
    parser.add_argument('--phases', default='train,export,coco,urban')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    sibling = os.path.join(os.path.dirname(os.path.abspath(args.python)), args.torchrun)
    args.torchrun = args.torchrun if os.sep in args.torchrun else sibling
    if not os.path.exists(args.torchrun):
        raise SystemExit('torchrun not found: %s' % args.torchrun)
    save_steps = args.save_steps or ('750,%d' % args.steps)
    status_path = os.path.join(run_dir, 'run_status.json')
    log_path = os.path.join(run_dir, 'runner.log')
    lock_path = args.lock_file or (run_dir + '.lock')
    checkpoint = os.path.join(run_dir, 'cvssl_%s_step%06d.pt' % (ARM, args.steps))
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
                     arm=ARM, objective=OBJECTIVE, max_steps=args.steps, run_dir=run_dir,
                     completed_steps=0, exit_codes={}, attempt=int(previous.get('attempt') or 0) + 1,
                     attempt_history=history[-5:], lock_file=lock_path, phases=phases,
                     resume=args.resume, save_steps=save_steps, **reset)

        if 'train' in phases:
            ok, detail = gpu_report()
            write_status(status_path, gpu_check=detail, gpu_ok=bool(ok))
            if not ok:
                write_status(status_path, phase='failed',
                             failure_reason='GPU pre-check refused to start: %s' % detail)
                raise SystemExit('REFUSING to start training: %s' % detail)
            for path in (args.init_state, args.resume):
                if not os.path.isfile(path):
                    write_status(status_path, phase='failed',
                                 failure_reason='required input missing: %s' % path)
                    raise SystemExit('missing %s' % path)
        annotations, images = verify_coco_environment()
        write_status(status_path, coco_annotations=annotations, coco_images=images)

        env = dict(os.environ)
        env.update({'SHARE4V_DATA_ROOT': SHARE4V_ROOT,
                    'SHARE4V_JSON': 'share-captioner_coco_lcs_sam_1246k_1107.json',
                    'SHARE4V_FULL_AUDIT':
                        '/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json',
                    'COCO_DATA_ROOT': COCO_ROOT, 'NCCL_SOCKET_IFNAME': 'lo',
                    'GLOO_SOCKET_IFNAME': 'lo',
                    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
        exit_codes = {}

        if 'train' in phases:
            if os.path.exists(checkpoint):
                raise SystemExit('REFUSING to retrain: %s already exists' % checkpoint)
            port = 36000 + (os.getpid() % 400)
            train_argv = [args.torchrun, '--nproc_per_node=%d' % args.nproc,
                          '--master_port=%d' % port, TRAIN_SCRIPT,
                          '--arm', ARM, '--lambda_U', '0', '--lambda_align', '10',
                          '--lambda_sparse', '2', '--ddp_gradient_averaging', '1',
                          '--base_model', 'B16', '--batch-size', str(args.batch_size),
                          '--epochs', str(args.epochs), '--lr', '1e-6', '--mask_lr', '1e-3',
                          '--weight_decay', '1e-2', '--warmup_length', str(args.warmup_length),
                          '--seed', '0', '--init_state', args.init_state,
                          '--output_dir', run_dir, '--max_steps', str(args.steps),
                          '--save_completed_steps', save_steps, '--log_every', '10',
                          '--grad_checkpoint_views', '1', '--num_workers', '8', '--amp_dtype', 'bf16',
                          '--resume', args.resume]
            if args.grad_probe_steps:
                train_argv += ['--grad_probe_steps', args.grad_probe_steps]
            write_status(status_path, phase='training',
                         commands={'train': sanitise_argv(train_argv)})
            exit_codes['train'] = run_command(train_argv, log_path, env=env)
            write_status(status_path, train_exit_code=exit_codes['train'], exit_codes=exit_codes,
                         completed_steps=_completed_steps(run_dir))
            if exit_codes['train'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='training exited %d; no evaluation attempted'
                                            % exit_codes['train'])
                raise SystemExit('training failed with exit code %d' % exit_codes['train'])

        if not os.path.isfile(checkpoint):
            write_status(status_path, phase='failed',
                         failure_reason='expected checkpoint missing: %s' % checkpoint)
            raise SystemExit('missing checkpoint %s' % checkpoint)
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = payload.get('config') or {}
        problems = []
        if payload.get('objective') != OBJECTIVE:
            problems.append('objective=%r' % payload.get('objective'))
        if config.get('arm') != ARM:
            problems.append('arm=%r' % config.get('arm'))
        if int(payload.get('completed_steps', -1)) != args.steps:
            problems.append('completed_steps=%r' % payload.get('completed_steps'))
        if int(payload.get('lr_horizon_steps', -1)) != LR_HORIZON:
            problems.append('lr_horizon_steps=%r' % payload.get('lr_horizon_steps'))
        if float(config.get('lambda_U', -1)) != 0.0:
            problems.append('lambda_U=%r' % config.get('lambda_U'))
        write_status(status_path, checkpoint=checkpoint, checkpoint_verified=not problems,
                     checkpoint_problems=problems,
                     completed_steps=int(payload.get('completed_steps', -1)),
                     lr_horizon_steps=payload.get('lr_horizon_steps'))
        if problems:
            write_status(status_path, phase='failed',
                         failure_reason='checkpoint verification failed: %s' % problems)
            raise SystemExit('checkpoint verification failed: %s' % problems)
        del payload

        if not os.path.exists(student):
            write_status(status_path, phase='exporting')
            export_argv = [args.python, EXPORT_SCRIPT, '--checkpoint', checkpoint,
                           '--out', student, '--expect-steps', str(args.steps),
                           '--expect-objective', OBJECTIVE, '--expect-arm', ARM]
            write_status(status_path, commands=dict(_commands(status_path),
                                                    export=sanitise_argv(export_argv)))
            exit_codes['export'] = run_command(export_argv, log_path, env=env)
            write_status(status_path, export_exit_code=exit_codes['export'], exit_codes=exit_codes)
            if exit_codes['export'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='export exited %d' % exit_codes['export'])
                raise SystemExit('export failed')
        if not os.path.isfile(student):
            write_status(status_path, phase='failed',
                         failure_reason='exported student missing: %s' % student)
            raise SystemExit('missing student')

        if 'coco' in phases:
            os.makedirs(evaluation_dir, exist_ok=True)
            canonical_out = os.path.join(evaluation_dir,
                                         '%s_step%06d_canonical.json' % (ARM, args.steps))
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
                         commands=dict(_commands(status_path), coco=sanitise_argv(coco_argv)))
            exit_codes['coco'] = run_command(coco_argv, log_path, env=env)
            write_status(status_path, coco_exit_code=exit_codes['coco'], exit_codes=exit_codes,
                         coco_output=canonical_out)
            if exit_codes['coco'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='COCO evaluation exited %d' % exit_codes['coco'])
                raise SystemExit('COCO evaluation failed')

        if 'urban' in phases:
            os.makedirs(evaluation_dir, exist_ok=True)
            urban_out = os.path.join(evaluation_dir,
                                     '%s_step%06d_urban1k.json' % (ARM, args.steps))
            urban_argv = [args.python, URBAN_EVAL, '--checkpoint', student,
                          '--label', '%s@%d' % (ARM, args.steps),
                          '--expect-steps', str(args.steps), '--base_model', 'ViT-B/16',
                          '--device', 'cuda', '--batch_size', '64',
                          '--urban_root', URBAN_ROOT, '--out', urban_out]
            write_status(status_path, phase='evaluating_urban',
                         commands=dict(_commands(status_path), urban=sanitise_argv(urban_argv)))
            exit_codes['urban'] = run_command(urban_argv, log_path, env=env)
            write_status(status_path, urban_exit_code=exit_codes['urban'], exit_codes=exit_codes,
                         urban_output=urban_out)
            if exit_codes['urban'] != 0:
                write_status(status_path, phase='failed',
                             failure_reason='Urban-1k evaluation exited %d' % exit_codes['urban'])
                raise SystemExit('Urban-1k evaluation failed')

        write_status(status_path, phase='complete', exit_codes=exit_codes,
                     conclusion=_conclusion(run_dir, args.steps), finished_at=time.time(),
                     wall_seconds=time.time() - started, student=student)
        print('RUNNER_COMPLETE ' + json.dumps({'exit_codes': exit_codes}, sort_keys=True),
              flush=True)
    finally:
        release_lock(lock, lock_path)


def _commands(status_path):
    try:
        with open(status_path, 'r', encoding='utf-8') as handle:
            return json.load(handle).get('commands') or {}
    except (ValueError, OSError):
        return {}


def _completed_steps(run_dir):
    try:
        with open(os.path.join(run_dir, 'run_summary.json'), 'r', encoding='utf-8') as handle:
            return int(json.load(handle).get('completed_steps', 0))
    except (ValueError, OSError, TypeError):
        return None


def _conclusion(run_dir, steps):
    path = os.path.join(run_dir, 'evaluation', '%s_step%06d_canonical.json' % (ARM, steps))
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        inner = payload['canonical']['%s@%d' % (ARM, steps)]['coco_val2017']
    except (ValueError, OSError, KeyError, TypeError) as error:
        return {'verdict': 'unknown', 'reason': 'cannot read %s (%s)' % (path, error)}
    i2t, t2i = inner['image2text_R1'], inner['text2image_R1']
    return {'coco_i2t_r1': i2t, 'coco_t2i_r1': t2i,
            'note': 'the frozen promotion gate is defined at 500 updates; this run is a 1000-step '
                    'budget and the gate is NOT applied to it'}


if __name__ == '__main__':
    main()
