"""Stage runner for S0-GlobalOnly v0.1.

    python tools/globalonly_runner.py --run-dir <dir> [--stages ...] [--dry-run]

Stages, each a real subprocess with a real exit code recorded in ``run_status.json``:

    train  -> verify -> export -> coco -> urban -> report

Operational rules (the ones the frozen arms use):

* the four GPUs are re-queried before training; an ``nvidia-smi`` failure counts as "not idle";
* an exclusive lock file holding the PID refuses a second concurrent start;
* ``train`` refuses to run if the final checkpoint already exists (never retrain to fix a tool);
* a failed stage stops the sequence: an unverified checkpoint is never evaluated;
* nothing is written outside the run directory, and the report only records numbers it read from
  files, each with the file it came from.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_global_only.py')
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_global_only_student.py')
COCO_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'
ARM = 'S0_GLOBAL_ONLY'
STEPS = 500
SAVE_STEPS = '0,100,250,500'
SHARED_INIT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt')
COCO_ROOT = '/root/datasets/coco/val2017'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
PYTHON = sys.executable
STAGES = ('train', 'verify', 'export', 'coco', 'urban', 'report')
STAGE_EXIT = {'train': 1, 'verify': 2, 'export': 3, 'coco': 4, 'urban': 5, 'report': 6}


def sha256_of(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head():
    try:
        return subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        return 'unknown'


def state_digest(state_dict):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            digest.update(key.encode('utf-8'))
            digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def gpu_preflight():
    try:
        rows = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=60)
    except Exception as error:
        return False, 'nvidia-smi failed: %s' % error, {}
    if rows.returncode != 0:
        return False, 'nvidia-smi exited %d' % rows.returncode, {}
    devices = [line for line in rows.stdout.splitlines() if line.strip()]
    busy = []
    for line in devices:
        parts = [p.strip() for p in line.split(',')]
        try:
            used = float(parts[1].split()[0])
        except (IndexError, ValueError):
            return False, 'cannot read memory from %r' % line, {}
        if used > 512.0:
            busy.append(line)
    if len(devices) < 4:
        return False, 'only %d GPUs visible' % len(devices), {'devices': devices}
    if busy:
        return False, 'GPUs are not idle: %s' % busy, {'devices': devices}
    return True, '4 GPUs idle', {'devices': devices}


def write_status(path, **fields):
    payload = {}
    if os.path.isfile(path):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}
    payload.update(fields)
    payload['updated_at'] = time.time()
    with open(path + '.tmp', 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(path + '.tmp', path)
    return payload


def run_stage(name, argv, log_path, env=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as handle:
        handle.write('\n# stage=%s start=%s\n$ %s\n'
                     % (name, time.strftime('%Y-%m-%dT%H:%M:%S'), ' '.join(argv)))
        handle.flush()
        started = time.time()
        process = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    return process.returncode, time.time() - started


def verify_checkpoint(path, expect_steps):
    import torch
    problems = []
    if not os.path.isfile(path):
        return ['checkpoint missing: %s' % path]
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        return ['the checkpoint is not a dict']
    if int(payload.get('completed_steps', -1)) != expect_steps:
        problems.append('completed_steps=%r expected %d' % (payload.get('completed_steps'),
                                                            expect_steps))
    state = payload.get('clip') or payload.get('model')
    if not isinstance(state, dict) or not state:
        problems.append('no clip/model state dict')
    elif payload.get('clip_state_digest') != state_digest(state):
        problems.append('clip_state_digest does not match the stored state')
    optimizer = payload.get('optimizer') or {}
    if not isinstance(optimizer, dict) or not optimizer.get('state'):
        problems.append('the optimizer state is missing or empty')
    if len(list(optimizer.get('state', {}).values())[:1]):
        first = list(optimizer['state'].values())[0]
        if 'exp_avg' not in first:
            problems.append('the optimizer state entries are not AdamW entries')
    if payload.get('arm') != ARM:
        problems.append('arm=%r expected %r' % (payload.get('arm'), ARM))
    return problems


def main():
    parser = argparse.ArgumentParser(description='S0-GlobalOnly v0.1 stage runner')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stages', default=','.join(STAGES))
    parser.add_argument('--init-state', default=SHARED_INIT)
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--torchrun', default='torchrun')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-length', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--master-port', type=int, default=35711)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    status_path = os.path.join(run_dir, 'run_status.json')
    checkpoint = os.path.join(run_dir, '%s_step%06d.pt' % (ARM, STEPS))
    student_dir = os.path.join(run_dir, 'student_export')
    evaluation_dir = os.path.join(run_dir, 'evaluation')
    stages = tuple(s.strip() for s in args.stages.split(',') if s.strip())

    train_argv = [args.torchrun, '--nproc_per_node=%d' % args.nproc,
                  '--master_port=%d' % args.master_port, TRAIN_SCRIPT,
                  '--base_model', 'B16', '--batch-size', str(args.batch_size),
                  '--epochs', str(args.epochs), '--lr', str(args.lr),
                  '--weight_decay', str(args.weight_decay),
                  '--warmup_length', str(args.warmup_length), '--seed', '0',
                  '--init_state', args.init_state, '--output_dir', run_dir,
                  '--max_steps', str(STEPS), '--save_completed_steps', SAVE_STEPS,
                  '--log_every', '10', '--num_workers', str(args.num_workers),
                  '--amp_dtype', 'bf16']
    export_argv = [args.python, EXPORT_SCRIPT, '--checkpoint', checkpoint,
                   '--output_dir', student_dir, '--expect_steps', str(STEPS)]
    coco_argv = [args.python, COCO_EVAL, '--label', ARM, '--gap_anti_temperature', '1.0',
                 '--sharegpt4v_manifest', '', '--data_root', '/root/datasets',
                 '--image_root', '/root/datasets', '--image_batch_size', '64',
                 '--canonical', '--canonical_only', '--coco', '--canonical_tags', str(STEPS),
                 '--canonical_names', '%s@%d' % (ARM, STEPS),
                 '--checkpoints', '%d:%s' % (STEPS, os.path.join(student_dir,
                                                                's0_global_only_student.pt')),
                 '--coco_root', COCO_ROOT,
                 '--output', os.path.join(evaluation_dir, '%s_step%06d_canonical.json'
                                          % (ARM, STEPS))]
    urban_argv = [args.python, URBAN_EVAL,
                  '--checkpoint', os.path.join(student_dir, 's0_global_only_student.pt'),
                  '--label', '%s@%d' % (ARM, STEPS), '--expect-steps', str(STEPS),
                  '--base_model', 'ViT-B/16', '--device', 'cuda', '--batch_size', '64',
                  '--urban_root', URBAN_ROOT,
                  '--out', os.path.join(evaluation_dir, '%s_step%06d_urban1k.json'
                                        % (ARM, STEPS))]

    if args.dry_run:
        for name, argv in (('train', train_argv), ('export', export_argv), ('coco', coco_argv),
                           ('urban', urban_argv)):
            if name in stages or name == 'train':
                print('DRY_RUN stage=%s\n  %s' % (name, ' '.join(argv)))
        print('DRY_RUN verify: %s' % checkpoint)
        print('DRY_RUN report: %s' % os.path.join(REPO, 'docs', 's0_global_only_v01',
                                                  'results.json'))
        return 0

    lock_path = os.path.join(run_dir, 'runner.lock')
    if os.path.exists(lock_path):
        try:
            with open(lock_path) as handle:
                old = int(json.load(handle).get('pid', -1))
            os.kill(old, 0)
            raise SystemExit('another runner holds the lock (pid %d)' % old)
        except (OSError, ValueError):
            pass
    with open(lock_path, 'w') as handle:
        json.dump({'pid': os.getpid(), 'started': time.time(), 'argv': sys.argv}, handle)

    write_status(status_path, arm=ARM, run_dir=run_dir, stages=list(stages),
                 implementation_sha=git_head(), init_state=args.init_state,
                 init_file_sha256=sha256_of(args.init_state) if os.path.isfile(args.init_state)
                 else None,
                 steps=STEPS, save_steps=SAVE_STEPS, batch_size_per_gpu=args.batch_size,
                 world_size=args.nproc, global_batch=args.batch_size * args.nproc,
                 exit_codes={}, phase='preflight')
    exit_codes = {}

    ok, reason, report = gpu_preflight()
    write_status(status_path, gpu_check=reason, gpu_report=report, gpu_ok=ok)
    if not ok and 'train' in stages:
        write_status(status_path, phase='failed', failure_reason='GPU preflight refused: %s' % reason)
        return 10

    env = dict(os.environ, NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo',
               COCO_DATA_ROOT='/root/datasets/coco', SHARE4V_DATA_ROOT='/root/datasets/ShareGPT4V',
               SHARE4V_JSON='share-captioner_coco_lcs_sam_1246k_1107.json',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    if 'train' in stages:
        if os.path.exists(checkpoint):
            raise SystemExit('REFUSING to retrain: %s already exists' % checkpoint)
        write_status(status_path, phase='training', commands={'train': train_argv})
        exit_codes['train'], wall = run_stage('train', train_argv,
                                              os.path.join(log_dir, 'train.log'), env=env)
        write_status(status_path, train_exit_code=exit_codes['train'], exit_codes=exit_codes,
                     train_wall_seconds=wall)
        if exit_codes['train'] != 0:
            write_status(status_path, phase='failed',
                         failure_reason='training exited %d' % exit_codes['train'])
            return STAGE_EXIT['train']

    if 'verify' in stages:
        write_status(status_path, phase='verifying')
        problems = verify_checkpoint(checkpoint, STEPS)
        write_status(status_path, verification_problems=problems,
                     checkpoint=checkpoint,
                     checkpoint_sha256=sha256_of(checkpoint) if os.path.isfile(checkpoint) else None)
        exit_codes['verify'] = 0 if not problems else STAGE_EXIT['verify']
        write_status(status_path, exit_codes=exit_codes, verify_exit_code=exit_codes['verify'])
        if problems:
            write_status(status_path, phase='failed',
                         failure_reason='checkpoint verification failed: %s' % problems)
            return exit_codes['verify']

    if 'export' in stages:
        write_status(status_path, phase='exporting')
        exit_codes['export'], wall = run_stage('export', export_argv,
                                               os.path.join(log_dir, 'export.log'), env=env)
        write_status(status_path, exit_codes=exit_codes, export_exit_code=exit_codes['export'],
                     export_wall_seconds=wall)
        if exit_codes['export'] != 0:
            write_status(status_path, phase='failed', failure_reason='export exited %d'
                         % exit_codes['export'])
            return STAGE_EXIT['export']

    os.makedirs(evaluation_dir, exist_ok=True)
    if 'coco' in stages:
        write_status(status_path, phase='evaluating_coco', commands={'coco': coco_argv})
        exit_codes['coco'], wall = run_stage('coco', coco_argv,
                                             os.path.join(log_dir, 'coco.log'), env=env)
        write_status(status_path, exit_codes=exit_codes, coco_exit_code=exit_codes['coco'],
                     coco_wall_seconds=wall)
        if exit_codes['coco'] != 0:
            write_status(status_path, phase='failed', failure_reason='COCO evaluation exited %d'
                         % exit_codes['coco'])
            return STAGE_EXIT['coco']

    if 'urban' in stages:
        write_status(status_path, phase='evaluating_urban', commands={'urban': urban_argv})
        exit_codes['urban'], wall = run_stage('urban', urban_argv,
                                              os.path.join(log_dir, 'urban.log'), env=env)
        write_status(status_path, exit_codes=exit_codes, urban_exit_code=exit_codes['urban'],
                     urban_wall_seconds=wall)
        if exit_codes['urban'] != 0:
            write_status(status_path, phase='failed', failure_reason='Urban-1k evaluation exited %d'
                         % exit_codes['urban'])
            return STAGE_EXIT['urban']

    if 'report' in stages:
        results = {'arm': ARM, 'objective': 'global_alignment_only', 'run_id': 's0_global_only_v01',
                   'run_dir': run_dir, 'implementation_sha': git_head(),
                   'stage_exit_codes': exit_codes, 'created_at': time.time()}
        coco_path = os.path.join(evaluation_dir, '%s_step%06d_canonical.json' % (ARM, STEPS))
        urban_path = os.path.join(evaluation_dir, '%s_step%06d_urban1k.json' % (ARM, STEPS))
        if os.path.isfile(coco_path):
            with open(coco_path) as handle:
                payload = json.load(handle)
            inner = next(iter(payload['canonical'].values()))
            results['coco'] = {'source': coco_path, 'raw': inner['coco_val2017'],
                               'checkpoint_sha256': inner.get('checkpoint_sha256')}
        else:
            results['coco'] = {'source': coco_path, 'error': 'missing'}
        if os.path.isfile(urban_path):
            with open(urban_path) as handle:
                payload = json.load(handle)
            nested = payload.get('urban1k') or {}
            results['urban'] = {'source': urban_path,
                                'i2t': nested.get('image2text'), 't2i': nested.get('text2image'),
                                'checkpoint_sha256': payload.get('checkpoint_sha256')}
        else:
            results['urban'] = {'source': urban_path, 'error': 'missing'}
        summary_path = os.path.join(run_dir, 'run_summary.json')
        if os.path.isfile(summary_path):
            with open(summary_path) as handle:
                results['train'] = json.load(handle)
        docs = os.path.join(REPO, 'docs', 's0_global_only_v01')
        os.makedirs(docs, exist_ok=True)
        with open(os.path.join(docs, 'results.json'), 'w') as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        exit_codes['report'] = 0
        write_status(status_path, phase='complete', exit_codes=exit_codes, results=results)
        print('RUNNER_COMPLETE ' + json.dumps({'exit_codes': exit_codes}, sort_keys=True),
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
