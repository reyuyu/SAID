"""Background stage runner for the CG-CLIP v0.1 arm (``CG_CLIP_V01``, run id ``cgclip_v01``).

    python tools/cgclip_runner.py --run-dir /root/SAID-cgclip-v01/runs_salu/cgclip_v01

Sequence, in this fixed order, each stage a real subprocess whose real exit code is recorded::

    train  (exactly 500 optimizer updates, 4 GPUs)
      -> verify  (the step-500 checkpoint must be COMPLETE; no bypass exists)
      -> export  (bare CLIP student from the verified checkpoint)
      -> coco    (canonical COCO retrieval on the exported student)
      -> urban   (Urban-1k retrieval on the exported student)
      -> report  (results.json + run_status.json assembled from the files actually read)

Exit codes::

    0  the selected stages completed
    1  train failed (or the trainer produced no usable checkpoint)
    2  verify refused the checkpoint (incomplete / unreadable / digest mismatch)
    3  export failed
    4  COCO evaluation failed
    5  Urban-1k evaluation failed
    6  report could not assemble the results
    10 a pre-flight refusal (lock held, fewer than 4 GPUs, busy GPU, missing data root, missing
       shared init, resume check failed); nothing was launched

Operational rules, mirrored from the frozen PG-CLIP v0.2 runner (``tools/pgclip_runner.py``):

* an exclusive ``O_CREAT|O_EXCL`` lock holds the PID, the timestamp, the argv, the git HEAD and the
  resolved command lines; a live holder refuses a second start, and a stale lock is reported and only
  replaced with ``--clear-stale-lock``;
* the four GPUs are re-queried with ``nvidia-smi`` before training; an ``nvidia-smi`` failure is
  treated as "not idle", never as idle, and a non-empty compute-application list refuses the run.
  The observed per-GPU memory and utilisation are recorded in the status file;
* ``--detach`` re-executes this runner through ``subprocess.Popen(..., start_new_session=True)`` with
  ``stdin`` at ``/dev/null``, so a launched run outlives the SSH session; the detached PID is recorded
  in the same lock;
* every stage writes its own log under ``<run-dir>/logs/`` and the runner appends its own log line
  there too; ``run_status.json`` is rewritten atomically after every step with honest exit codes,
  wall times, start/end timestamps and the attempt history;
* a failed stage stops the pipeline -- the artifacts and the status file are kept so a human can
  inspect them -- and a stage that is already complete is never repeated (a run that has trained to
  500 cannot be retrained to fix a downstream tool).

Single-configuration policy: this arm is ONE configuration, exactly 500 updates, no sweeps, no extra
arms, no epoch extension and no automatic control run. The values that define the experiment are
declared as ``LOCKED`` below and the runner refuses to start with a non-default value instead of
running a second configuration. ``--steps/--save-steps/--base-model/--seed`` therefore have no
purpose beyond documentation and cannot be used to extend the run.

There is deliberately NO ``--allow-missing`` style escape hatch here (the PG runner's
``--allow-missing-gate-state`` is not ported): a missing or incomplete checkpoint is a hard failure.
Standard library only, plus ``torch`` inside the checkpoint-schema helpers.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------- experiment identity
# Imported lazily/probably from the repository itself; the values below are the frozen CG-CLIP v0.1
# identity and are asserted against model/cgclip.py whenever that module can be imported.
ARM = 'CG_CLIP_V01'
RUN_ID = 'cgclip_v01'
OBJECTIVE = 'clip_native_caption_gated_cls'
PHASE = 'cgclip-v0.1'
GATE_KIND = 'caption_gated_final_cls_attention'
LAMBDA_GLOBAL = 5.0
LAMBDA_ATTENTION = 5.0
LAMBDA_SPARSE = 1.0
FIXED_SCALE = 100.0
NORM_EPS = 1e-6
GATE_KEY_DIM = 64
GATE_SEED = 0

# model/cgclip.py:RESUME_CRITICAL_KEYS -- the config identity a resume must match exactly
RESUME_CRITICAL_KEYS = ('objective', 'arm', 'gate_kind', 'gate_key_dim', 'gate_seed',
                        'lambda_global', 'lambda_attention', 'lambda_sparse', 'fixed_scale',
                        'norm_eps')
CONFIG_EXPECTED = {
    'objective': OBJECTIVE, 'arm': ARM, 'gate_kind': GATE_KIND, 'gate_key_dim': GATE_KEY_DIM,
    'gate_seed': GATE_SEED, 'lambda_global': LAMBDA_GLOBAL, 'lambda_attention': LAMBDA_ATTENTION,
    'lambda_sparse': LAMBDA_SPARSE, 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
}
# the exact gate tensors model/cgclip.py:CaptionGate must have written
GATE_TENSOR_SHAPES = {'query.weight': (64, 512), 'key.weight': (64, 768), 'bias': ()}
GATE_TENSOR_NAMES = tuple(sorted(GATE_TENSOR_SHAPES))

# --------------------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_cgclip.py')
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_cgclip_student.py')
CANONICAL_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'

SHARED_INIT_DEFAULT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/'
                       'cvssl_initial.pt')
SHARE4V_ROOT = '/root/datasets/ShareGPT4V'
SHARE4V_JSON = 'share-captioner_coco_lcs_sam_1246k_1107.json'
COCO_ROOT = '/root/datasets/coco'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
PYTHON_DEFAULT = '/root/miniconda3/envs/said-smartclip/bin/python'
# the launcher of the frozen trainer invocation; the default is the required module form and
# ``--torchrun`` only exists so an operator can point at an absolute torchrun launcher path
TORCHRUN_DEFAULT = 'torchrun'
MASTER_PORT = 29571
REQUIRED_GPUS = 4
DEFAULT_RUN_DIR = '/root/SAID-cgclip-v01/runs_salu/cgclip_v01'

STAGES = ('train', 'verify', 'export', 'coco', 'urban', 'report')
STAGE_EXIT_CODE = {'train': 1, 'verify': 2, 'export': 3, 'coco': 4, 'urban': 5, 'report': 6}
PREFLIGHT_EXIT_CODE = 10

# The one and only configuration of this arm. A non-default value is refused, not honoured.
LOCKED = {
    'steps': 500,
    'save_steps': '0,20,100,250,500',
    'base_model': 'B16',
    'epochs': 3,
    'batch_size': 256,
    'lr': 1e-6,
    'gate_lr': 1e-3,
    'weight_decay': 1e-2,
    'warmup_length': 200,
    'seed': 0,
    'nproc': 4,
    'image_chunk': 32,
    'text_chunk': 64,
    'num_workers': 8,
    'amp_dtype': 'bf16',
    'log_every': 10,
    'heavy_log_every': 25,
    'cond_checkpoint': 1,
    'master_port': MASTER_PORT,
}

SALU_LOG_NAME = 'salu_log.jsonl'
CHECKPOINT_TEMPLATE = '%s_step%06d.pt'
# applied to every stage command: a failed stage stops the pipeline, it never retries or extends
SENSITIVE_KEYS = ('password', 'passwd', 'token', 'secret', 'key', 'credential')
METADATA_KEYS = ('epoch', 'step_in_epoch', 'lr_horizon_steps', 'batch_size_per_gpu', 'world_size',
                 'precision', 'chunking', 'config', 'digests', 'visual_spec')


# --------------------------------------------------------------------------- small helpers
def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    """Same rule as model/cgclip.py:file_sha256, re-implemented so importing torch is not needed."""
    return sha256_of(path)


def git_head(repo=None):
    repo = repo or REPO
    try:
        result = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return result.stdout.strip() or 'unknown'


def is_hex_digest(value, length=64):
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in '0123456789abcdefABCDEF' for character in value)


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


def write_status(path, **fields):
    """Atomic read-modify-write of the status file, so the dashboard only ever sees valid JSON."""
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
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return payload


def read_status(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return {}


def iso(timestamp):
    return time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(timestamp))


# --------------------------------------------------------------------------- lock
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
                             % (path, pid, holder.get('started_at_iso')))
        if not clear_stale:
            raise SystemExit('REFUSING to start: %s holds a stale lock (pid %s is gone); '
                             're-run with --clear-stale-lock to replace it' % (path, pid))
        os.unlink(path)
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, json.dumps({'pid': os.getpid(),
                                 'started_at': time.time(),
                                 'started_at_iso': iso(time.time())}).encode('utf-8'))
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


def update_lock(path, **fields):
    """Record argv / git HEAD / command lines in the lock file we hold (best effort)."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (ValueError, OSError):
        payload = {}
    payload.update(fields)
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except OSError:
        pass


# --------------------------------------------------------------------------- GPU pre-flight
def _query_gpu_rows():
    """Per-GPU identity/memory/utilisation rows plus the raw CSV echo; raises on a failed query."""
    listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30)
    devices = [line for line in listing.stdout.splitlines() if line.startswith('GPU ')]
    query = subprocess.run(['nvidia-smi',
                            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
    if listing.returncode != 0:
        raise RuntimeError('nvidia-smi -L exited %d: %s'
                           % (listing.returncode, listing.stderr.strip()))
    if query.returncode != 0:
        raise RuntimeError('nvidia-smi --query-gpu exited %d: %s'
                           % (query.returncode, query.stderr.strip()))
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 5:
            raise RuntimeError('cannot parse nvidia-smi row %r' % line)
        rows.append({'index': parts[0], 'name': parts[1], 'memory_used_mib': parts[2],
                     'memory_total_mib': parts[3], 'utilisation_gpu_percent': parts[4]})
    return devices, rows, query.stdout.strip()


def gpu_preflight():
    """``(ok, detail, report)`` -- a failing nvidia-smi is never treated as an idle machine.

    Refuses when nvidia-smi cannot be executed, when fewer than four devices are visible, when a
    compute application is registered, when a GPU is already holding a large amount of memory, or
    when the per-GPU query cannot be parsed. The full observation lands in the status file either way.
    """
    try:
        devices, rows, raw = _query_gpu_rows()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return False, 'nvidia-smi could not be used: %s' % error, {'error': str(error)}
    report = {'nvidia_smi_visible_devices': len(devices), 'per_gpu': rows,
              'query_echo': raw, 'required_gpus': REQUIRED_GPUS,
              'query': 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu'}
    if len(devices) < REQUIRED_GPUS:
        return (False, 'only %d GPU(s) visible, %d required' % (len(devices), REQUIRED_GPUS),
                report)
    if len(rows) != len(devices):
        return (False, 'the per-GPU query returned %d row(s) for %d device(s)'
                % (len(rows), len(devices)), report)
    try:
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return False, 'nvidia-smi compute-app query failed: %s' % error, report
    if busy.returncode != 0:
        return False, 'nvidia-smi compute-app query exited %d' % busy.returncode, report
    running = [line for line in busy.stdout.splitlines() if line.strip()]
    report['compute_applications'] = running
    if running:
        return False, 'GPUs are busy: %s' % '; '.join(running), report
    occupied = []
    for row in rows:
        try:
            used = float(row['memory_used_mib'])
        except (TypeError, ValueError):
            return False, 'cannot read memory.used from nvidia-smi row %r' % row, report
        if used > 512.0:
            occupied.append('%s: %s MiB' % (row['index'], row['memory_used_mib']))
    if occupied:
        return False, 'GPUs already hold memory: %s' % '; '.join(occupied), report
    return True, '%d GPUs idle' % len(devices), report


# --------------------------------------------------------------------------- stage commands
def build_commands(args, run_dir):
    """The exact argv of every stage, built before anything is launched (also used by --dry-run)."""
    python = args.python
    torchrun = resolve_torchrun(python, args.torchrun)
    checkpoint = os.path.join(run_dir, CHECKPOINT_TEMPLATE % (ARM, args.steps))
    student_dir = os.path.join(run_dir, 'student_export')
    evaluation_dir = os.path.join(run_dir, 'evaluation')
    canonical_out = os.path.join(evaluation_dir, '%s_step%06d_canonical.json'
                                 % (ARM, args.steps))
    urban_out = os.path.join(evaluation_dir, '%s_step%06d_urban1k.json' % (ARM, args.steps))
    commands = {
        'train': [torchrun, '--nproc_per_node=%d' % args.nproc,
                  '--master_port=%d' % args.master_port, TRAIN_SCRIPT,
                  '--base_model', args.base_model, '--batch-size', str(args.batch_size),
                  '--epochs', str(args.epochs), '--lr', str(args.lr),
                  '--gate_lr', str(args.gate_lr), '--weight_decay', str(args.weight_decay),
                  '--warmup_length', str(args.warmup_length), '--seed', str(args.seed),
                  '--init_state', args.init_state, '--output_dir', run_dir,
                  '--max_steps', str(args.steps), '--save_completed_steps', args.save_steps,
                  '--log_every', str(args.log_every), '--heavy_log_every', str(args.heavy_log_every),
                  '--num_workers', str(args.num_workers), '--amp_dtype', args.amp_dtype,
                  '--image_chunk', str(args.image_chunk), '--text_chunk', str(args.text_chunk),
                  '--cond_checkpoint', str(args.cond_checkpoint)],
        'export': [python, EXPORT_SCRIPT, '--checkpoint', checkpoint,
                   '--output_dir', student_dir],
        'coco': [python, CANONICAL_EVAL, '--label', ARM,
                 '--gap_anti_temperature', '1.0', '--sharegpt4v_manifest', '',
                 '--data_root', SHARE4V_ROOT, '--image_root', SHARE4V_ROOT,
                 '--image_batch_size', '64', '--canonical', '--canonical_only', '--coco',
                 '--canonical_tags', str(args.steps),
                 '--canonical_names', '%s@%d' % (ARM, args.steps),
                 '--checkpoints', '%d:%s' % (args.steps, '<STUDENT_EXPORT>'),
                 '--coco_root', os.path.join(COCO_ROOT, 'val2017'),
                 '--output', canonical_out],
        'urban': [python, URBAN_EVAL, '--checkpoint', '<STUDENT_EXPORT>',
                  '--label', '%s@%d' % (ARM, args.steps),
                  '--expect-steps', str(args.steps), '--base_model', 'ViT-B/16',
                  '--device', 'cuda', '--batch_size', '64',
                  '--urban_root', URBAN_ROOT, '--out', urban_out],
    }
    return commands, checkpoint, student_dir, canonical_out, urban_out


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


def verify_data_environment():
    """The COCO and Urban-1k absolute paths must exist before any evaluation is attempted.

    Urban-1k uses the upstream layout ``<root>/image`` + ``<root>/caption`` (paired by stem); a missing
    directory is an error and no other 1k set is substituted for it.
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
    if not os.path.isfile(TRAIN_SCRIPT):
        raise SystemExit('the trainer is absent from this worktree: %s' % TRAIN_SCRIPT)
    if not os.path.isfile(EXPORT_SCRIPT):
        raise SystemExit('the export tool is absent from this worktree: %s' % EXPORT_SCRIPT)
    if not os.path.isfile(CANONICAL_EVAL):
        raise SystemExit('the canonical COCO evaluator is absent: %s' % CANONICAL_EVAL)
    if not os.path.isfile(URBAN_EVAL):
        raise SystemExit('the Urban-1k evaluator is absent: %s' % URBAN_EVAL)
    return annotations, images


def run_stage(name, argv, run_dir, log_dir, stage_state, env=None):
    """Run one stage subprocess, appending to its own ``logs/<stage>.log``; return the exit code."""
    log_path = os.path.join(log_dir, '%s.log' % name)
    started = time.time()
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write('\n# stage=%s start=%s cwd=%s\n' % (name, iso(started), REPO))
        handle.write('$ %s\n' % ' '.join(sanitise_argv(argv)))
        handle.flush()
        process = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    finished = time.time()
    record = {'exit_code': process.returncode, 'log': log_path,
              'started_at': started, 'started_at_iso': iso(started),
              'finished_at': finished, 'finished_at_iso': iso(finished),
              'wall_seconds': finished - started,
              'attempt': int(stage_state.get(name, {}).get('attempts') or 0) + 1}
    return process.returncode, record


# --------------------------------------------------------------------------- state digests
def state_digest(state_dict):
    """``model/cgclip.py:state_digest`` re-implemented: sha256 over sorted keys, key bytes then
    ``value.detach().float().cpu().numpy().tobytes()``.

    Re-implementing it here (instead of importing) is what makes the verify stage an independent
    check: the digest written by the trainer is recomputed from the loaded tensors by this code.
    """
    import torch
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('state_digest needs a non-empty dict')
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise ValueError('state_digest entry %r is not a tensor (%s)'
                             % (key, type(value).__name__))
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_summary(name, payload, problems):
    """Validate one optimizer payload: real ``state`` dict and ``param_groups``, both non-empty."""
    optimizer = payload.get(name)
    summary = {'present': isinstance(optimizer, dict)}
    if not summary['present']:
        problems.append('%s is missing or is not a dict (%s)'
                        % (name, type(optimizer).__name__))
        return summary
    state = optimizer.get('state')
    groups = optimizer.get('param_groups')
    if not isinstance(state, dict) or not state:
        problems.append('%s.state is missing or empty (%s)'
                        % (name, type(state).__name__))
        summary['state_entries'] = 0
    else:
        summary['state_entries'] = len(state)
        expansions = sum(len(entry) for entry in state.values()
                         if isinstance(entry, dict))
        summary['state_tensor_entries'] = expansions
        if summary['state_entries'] < 2:
            problems.append('%s.state carries only %d parameter entr(ies), which cannot be the real '
                            'AdamW state of a trained model'
                            % (name, summary['state_entries']))
    if not isinstance(groups, list) or not groups:
        problems.append('%s.param_groups is missing or empty (%s)'
                        % (name, type(groups).__name__))
    else:
        summary['param_groups'] = len(groups)
        summary['param_group_keys'] = sorted({key for group in groups
                                             if isinstance(group, dict) for key in group})
    return summary


def verify_checkpoint(path, expected_steps, salu_log):
    """Full schema verification of the step-500 training checkpoint. Returns a report dict.

    Every check appends a human-readable entry to ``problems``; a non-empty ``problems`` list is a
    stage failure. There is no flag that softens any of these checks.
    """
    import torch
    report = {'path': path, 'file_sha256': sha256_of(path), 'problems': []}
    problems = report['problems']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        problems.append('the checkpoint is not a dict but %s' % type(payload).__name__)
        return report
    report['top_level_keys'] = sorted(str(key) for key in payload)

    # --- training progress ---------------------------------------------------
    completed = payload.get('completed_steps')
    report['completed_steps'] = completed
    try:
        numeric_completed = int(completed)
    except (TypeError, ValueError):
        numeric_completed = None
    if numeric_completed != expected_steps:
        problems.append('completed_steps=%r expected %d' % (completed, expected_steps))

    # --- configuration identity (model/cgclip.py:check_resume_compatible critical keys) ---
    config = payload.get('config')
    if not isinstance(config, dict) or not config:
        problems.append('config is missing, empty or not a dict (%s)' % type(config).__name__)
        config = {}
    report['config'] = config
    critical = payload.get('resume_critical') if isinstance(payload.get('resume_critical'),
                                                            dict) else {}
    identity = {}
    for key in RESUME_CRITICAL_KEYS:
        want = CONFIG_EXPECTED[key]
        got = config.get(key, payload.get(key))
        identity[key] = got
        if isinstance(want, float) or isinstance(got, float):
            try:
                same = float(want) == float(got)
            except (TypeError, ValueError):
                same = False
        else:
            same = want == got
        if not same:
            problems.append('config identity %s=%r != %r' % (key, got, want))
        if critical.get(key, got) != got:
            problems.append('resume_critical[%s]=%r disagrees with config[%s]=%r'
                            % (key, critical.get(key), key, got))
    report['config_identity'] = identity

    # --- gate description and gate init report -------------------------------
    gate_config = payload.get('gate_config')
    if not isinstance(gate_config, dict) or not gate_config:
        problems.append('gate_config is missing, empty or not a dict (%s)'
                        % type(gate_config).__name__)
    else:
        report['gate_config'] = gate_config
        if gate_config.get('gate_kind') not in (None, GATE_KIND):
            problems.append('gate_config[gate_kind]=%r != %r'
                            % (gate_config.get('gate_kind'), GATE_KIND))
        if 'straight-through' not in str(gate_config.get('gate_forward', '')):
            problems.append('gate_config[gate_forward] does not describe the hard straight-through '
                            'gate: %r' % gate_config.get('gate_forward'))
    gate_report = payload.get('gate_init_report')
    if not isinstance(gate_report, dict) or not gate_report:
        problems.append('gate_init_report is missing, empty or not a dict (%s)'
                        % type(gate_report).__name__)
    else:
        report['gate_init_report_keys'] = sorted(str(key) for key in gate_report)

    # --- gate tensors --------------------------------------------------------
    gate_state = payload.get('gate_state')
    if not isinstance(gate_state, dict) or not gate_state:
        problems.append('gate_state is missing, empty or not a dict (%s); the trained gate tensors '
                        'must be present in this schema and there is no bypass'
                        % type(gate_state).__name__)
        gate_state = None
    else:
        non_tensors = sorted(str(key) for key, value in gate_state.items()
                             if not torch.is_tensor(value))
        if non_tensors:
            problems.append('gate_state carries non-tensor entries %r; gate_state must be the gate '
                            'tensor dict and must not be a description dict' % non_tensors[:6])
            gate_state = None
        else:
            shapes = {key: tuple(value.shape) for key, value in gate_state.items()}
            report['gate_state_tensors'] = {key: list(shape) for key, shape in sorted(shapes.items())}
            for name, shape in sorted(GATE_TENSOR_SHAPES.items()):
                got = shapes.get(name)
                if got is None:
                    problems.append('gate_state[%s] is missing (the %s gate tensor)'
                                    % (name, GATE_KIND))
                elif got != shape:
                    problems.append('gate_state[%s] shape %r != %r' % (name, got, shape))
            extra = sorted(set(shapes) - set(GATE_TENSOR_SHAPES))
            if extra:
                problems.append('gate_state carries unexpected tensors %r that are not part of the '
                                'CaptionGate module' % extra)
            for name, shape in sorted(shapes.items()):
                if not all(isinstance(size, int) for size in shape):
                    problems.append('gate_state[%s] has a non-integer shape %r' % (name, shape))
            if 'bias' in shapes and shapes['bias'] not in ((), (1,)):
                problems.append('gate_state[bias] is not the single trainable scalar but %r'
                                % (shapes['bias'],))

    # --- clip tensors --------------------------------------------------------
    clip_state = payload.get('clip')
    if not isinstance(clip_state, dict) or not clip_state:
        problems.append('clip is missing, empty or not a dict (%s)' % type(clip_state).__name__)
        clip_state = None
    else:
        non_tensors = sorted(str(key) for key, value in clip_state.items()
                             if not torch.is_tensor(value))
        if non_tensors:
            problems.append('clip carries non-tensor entries %r' % non_tensors[:6])
            clip_state = None
        else:
            report['clip_tensors'] = len(clip_state)
    if clip_state is not None and gate_state is not None:
        shared = sorted(set(clip_state) & set(gate_state))
        if shared:
            problems.append('clip and gate_state share parameter names %r' % shared[:6])

    # --- optimizer states ----------------------------------------------------
    report['optimizers'] = {
        'optimizer_clip': _optimizer_summary('optimizer_clip', payload, problems),
        'optimizer_gate': _optimizer_summary('optimizer_gate', payload, problems),
    }

    # --- recomputed digests --------------------------------------------------
    recorded_gate = payload.get('gate_state_digest')
    recorded_clip = payload.get('clip_state_digest')
    report['recorded_digests'] = {'gate_state_digest': recorded_gate,
                                  'clip_state_digest': recorded_clip}
    recomputed = {}
    if gate_state is not None:
        recomputed['gate_state'] = state_digest(gate_state)
    if clip_state is not None:
        recomputed['clip_state'] = state_digest(clip_state)
    report['recomputed_digests'] = recomputed
    for label, recorded, recomputed_value in (('gate_state_digest', recorded_gate,
                                               recomputed.get('gate_state')),
                                              ('clip_state_digest', recorded_clip,
                                               recomputed.get('clip_state'))):
        if not is_hex_digest(recorded):
            problems.append('%s is not a 64-character hex digest: %r' % (label, recorded))
            continue
        if recomputed_value is None:
            problems.append('%s cannot be recomputed: the matching state dict is unusable' % label)
            continue
        if recorded.lower() != recomputed_value.lower():
            problems.append('%s mismatch: recorded %s, recomputed from the loaded tensors %s'
                            % (label, recorded, recomputed_value))

    # --- remaining required metadata ----------------------------------------
    for key in METADATA_KEYS:
        value = payload.get(key)
        if value is None:
            problems.append('%s is missing' % key)
            continue
        if key in ('config', 'digests', 'visual_spec', 'chunking') and (
                not isinstance(value, dict) or not value):
            problems.append('%s is present but empty or not a dict (%s)'
                            % (key, type(value).__name__))
    for key in ('epoch', 'step_in_epoch'):
        value = payload.get(key)
        report[key] = value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append('%s is not a number: %r' % (key, value))
    horizon = payload.get('lr_horizon_steps')
    report['lr_horizon_steps'] = horizon
    if not isinstance(horizon, (int, float)) or isinstance(horizon, bool) or int(horizon) <= 0:
        problems.append('lr_horizon_steps is not a positive number: %r' % (horizon,))
    batch = payload.get('batch_size_per_gpu')
    report['batch_size_per_gpu'] = batch
    if not isinstance(batch, (int, float)) or int(batch) <= 0:
        problems.append('batch_size_per_gpu is not a positive number: %r' % (batch,))
    world = payload.get('world_size')
    report['world_size'] = world
    if not isinstance(world, (int, float)) or int(world) != LOCKED['nproc']:
        problems.append('world_size=%r expected %d' % (world, LOCKED['nproc']))
    precision = payload.get('precision')
    report['precision'] = precision
    if not isinstance(precision, dict) or not precision:
        problems.append('precision is missing, empty or not a dict (%s)'
                        % type(precision).__name__)
    elif 'bf16' not in json.dumps(precision).lower():
        problems.append('precision does not record the bf16 autocast dtype: %r' % (precision,))
    chunking = payload.get('chunking')
    if isinstance(chunking, dict) and chunking:
        report['chunking'] = chunking
        recorded = {'image': chunking.get('image_chunk', chunking.get('image')),
                    'text': chunking.get('text_chunk', chunking.get('text'))}
        wanted = {'image': LOCKED['image_chunk'], 'text': LOCKED['text_chunk']}
        for name, want in wanted.items():
            got = recorded[name]
            if got is not None and int(got) != int(want):
                problems.append('chunking[%s]=%r expected %d' % (name, got, want))
    spec = payload.get('visual_spec')
    if isinstance(spec, dict) and spec:
        report['visual_spec_keys'] = sorted(str(key) for key in spec)
        if spec.get('visual_layers') not in (None, 12):
            problems.append('visual_spec[visual_layers]=%r is not the ViT-B/16 depth'
                            % spec.get('visual_layers'))
    digests = payload.get('digests')
    if isinstance(digests, dict) and digests:
        report['digests'] = digests

    reported_schema = payload.get('schema_version')
    report['schema_version'] = reported_schema
    report['payload_class'] = type(payload).__name__

    # --- the run log must have reached the same step -------------------------
    log_report = {'path': salu_log, 'exists': os.path.isfile(salu_log)}
    if not log_report['exists']:
        problems.append('the run log is missing: %s' % salu_log)
    else:
        matched_line = None
        records = 0
        with open(salu_log, 'r', encoding='utf-8') as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                records += 1
                if not isinstance(record, dict):
                    continue
                try:
                    if int(record.get('completed_steps')) == expected_steps and matched_line is None:
                        matched_line = number
                except (TypeError, ValueError):
                    continue
        log_report['records'] = records
        log_report['line_with_completed_steps_%d' % expected_steps] = matched_line
        if matched_line is None:
            problems.append('%s has no record with completed_steps == %d (%d record(s) read)'
                            % (salu_log, expected_steps, records))
    report['run_log'] = log_report
    report['verified'] = not problems
    return report


# --------------------------------------------------------------------------- discovery helpers
def discover_checkpoint(run_dir, steps):
    """The step checkpoint: the arm-prefixed exact name first, then both glob patterns."""
    exact = os.path.join(run_dir, CHECKPOINT_TEMPLATE % (ARM, steps))
    pattern_names = [CHECKPOINT_TEMPLATE % (ARM, steps), CHECKPOINT_TEMPLATE % (ARM.lower(), steps)]
    if os.path.isfile(exact):
        return exact, 'exact', []
    for base in (run_dir, os.path.join(run_dir, 'checkpoints')):
        if not os.path.isdir(base):
            continue
        names = set(os.listdir(base))
        for pattern in pattern_names:
            candidate = os.path.join(base, pattern)
            if pattern in names and os.path.isfile(candidate):
                return candidate, 'glob', []
    suffix = 'step%06d.pt' % steps
    matches = []
    for root, _directories, files in os.walk(run_dir):
        for name in files:
            if name.endswith(suffix):
                matches.append(os.path.join(root, name))
    if not matches:
        return None, None, []
    matches.sort()
    return matches[0], 'walk', matches[1:]


def discover_student(student_dir):
    """The exported student: the *.pt/*.pth files the export tool actually produced."""
    if not os.path.isdir(student_dir):
        return None, []
    candidates = sorted(os.path.join(student_dir, name) for name in os.listdir(student_dir)
                        if name.endswith(('.pt', '.pth', '.bin')))
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda path: (os.path.getsize(path), path), reverse=True)
    return ranked[0], ranked[1:]


def _completed_steps(run_dir):
    summary = os.path.join(run_dir, 'run_summary.json')
    try:
        with open(summary, 'r', encoding='utf-8') as handle:
            return int(json.load(handle).get('completed_steps', 0))
    except (ValueError, OSError, TypeError):
        return None


def read_json(path):
    """``(payload, error)`` -- never raises; the error text is recorded in the results."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle), None
    except (ValueError, OSError) as error:
        return None, str(error)


def read_coco_result(path):
    """Both the flat and the nested shape the canonical evaluator has used are accepted."""
    payload, error = read_json(path)
    if error:
        return None, error
    inner = payload
    try:
        canonical = payload['canonical']
        key = next(iter(canonical))
        inner = canonical[key]['coco_val2017']
        source_key = 'canonical.%s.coco_val2017' % key
    except (KeyError, TypeError, StopIteration):
        if isinstance(payload.get('coco_val2017'), dict):
            inner = payload['coco_val2017']
            source_key = 'coco_val2017'
        elif isinstance(payload.get('coco'), dict):
            inner = payload['coco']
            source_key = 'coco'
        else:
            return None, 'no COCO block has been found in %s' % path
    if not isinstance(inner, dict):
        return None, 'the COCO block in %s is not a dict' % path
    return {'source': path, 'source_key': source_key,
            'image2text_R1': inner.get('image2text_R1'),
            'text2image_R1': inner.get('text2image_R1'),
            'raw_keys': sorted(str(key) for key in inner), 'raw': inner}, None


def read_urban_result(path):
    payload, error = read_json(path)
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, 'the Urban-1k result is not a dict'
    result = {'source': path, 'raw_keys': sorted(str(key) for key in payload)}
    for name in ('image2text_R1', 'text2image_R1', 'i2t_r1', 't2i_r1', 'i2t', 't2i'):
        if name in payload:
            result[name] = payload[name]
    if 'image2text_R1' not in result and 'i2t_r1' not in result and 'i2t' not in result:
        return None, 'no image-to-text R@1 has been found in %s' % path
    return result, None


# --------------------------------------------------------------------------- argument handling
def locked_value(args, name):
    """A locked argument must still carry its frozen default; anything else is refused."""
    value = getattr(args, name.replace('-', '_'))
    want = LOCKED[name]
    if isinstance(want, float) or isinstance(value, float):
        same = abs(float(value) - float(want)) <= 0.0
    else:
        same = str(value) == str(want)
    return same, value, want


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--run-dir', default=DEFAULT_RUN_DIR,
                        help='absolute output run directory; the default is the frozen cgclip_v01 '
                             'run directory of this arm')
    parser.add_argument('--init-state', default=SHARED_INIT_DEFAULT,
                        help='the shared frozen CV-SSL initialisation, the only allowed start point')
    parser.add_argument('--stages', default=','.join(STAGES),
                        help='comma-separated subset of: %s (always executed in this order)'
                             % ','.join(STAGES))
    parser.add_argument('--resume-from', default=None, choices=STAGES,
                        help='start at this stage and continue through the end of --stages; a '
                             'training resume needs a fully verified checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the exact stage commands and exit without executing anything')
    parser.add_argument('--detach', action='store_true',
                        help='re-execute detached (start_new_session=True, stdin=/dev/null) so the '
                             'run survives the calling shell; the PID is recorded in the lock')
    parser.add_argument('--internal-detached-child', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--python', default=PYTHON_DEFAULT)
    parser.add_argument('--torchrun', default=TORCHRUN_DEFAULT,
                        help='torchrun launcher, either the module form (default) or an absolute '
                             'path to a torchrun executable')
    parser.add_argument('--nproc', type=int, default=LOCKED['nproc'])
    parser.add_argument('--steps', type=int, default=LOCKED['steps'])
    parser.add_argument('--save-steps', dest='save_steps', default=LOCKED['save_steps'])
    parser.add_argument('--base-model', dest='base_model', default=LOCKED['base_model'])
    parser.add_argument('--epochs', type=int, default=LOCKED['epochs'])
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=LOCKED['batch_size'])
    parser.add_argument('--lr', type=float, default=LOCKED['lr'])
    parser.add_argument('--gate-lr', dest='gate_lr', type=float, default=LOCKED['gate_lr'])
    parser.add_argument('--weight-decay', dest='weight_decay', type=float,
                        default=LOCKED['weight_decay'])
    parser.add_argument('--warmup-length', dest='warmup_length', type=int,
                        default=LOCKED['warmup_length'])
    parser.add_argument('--seed', type=int, default=LOCKED['seed'])
    parser.add_argument('--log-every', dest='log_every', type=int, default=LOCKED['log_every'])
    parser.add_argument('--heavy-log-every', dest='heavy_log_every', type=int,
                        default=LOCKED['heavy_log_every'])
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=LOCKED['num_workers'])
    parser.add_argument('--amp-dtype', dest='amp_dtype', default=LOCKED['amp_dtype'])
    parser.add_argument('--image-chunk', dest='image_chunk', type=int, default=LOCKED['image_chunk'])
    parser.add_argument('--text-chunk', dest='text_chunk', type=int, default=LOCKED['text_chunk'])
    parser.add_argument('--cond-checkpoint', dest='cond_checkpoint', type=int,
                        default=LOCKED['cond_checkpoint'])
    parser.add_argument('--master-port', dest='master_port', type=int,
                        default=LOCKED['master_port'])
    parser.add_argument('--lock-file', default=None)
    parser.add_argument('--clear-stale-lock', action='store_true')
    parser.add_argument('--skip-gpu-check', action='store_true',
                        help='only for a re-run that does not include the train stage; training '
                             'always requires an idle 4-GPU machine')
    args = parser.parse_args()

    # the frozen-configuration guard can only work if every locked name really is an option
    registered = {option.lstrip('-').replace('-', '_') for action in parser._actions
                  for option in action.option_strings}
    unregistered = sorted(name for name in LOCKED if name not in registered)
    if unregistered:
        raise SystemExit('runner defect: locked option(s) %r are not registered on the parser'
                         % unregistered)

    run_dir = os.path.abspath(args.run_dir)
    stages = [stage.strip() for stage in args.stages.split(',') if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        parser.error('unknown stage(s) %r; the only stages are %s' % (unknown, ','.join(STAGES)))
    stages = [stage for stage in STAGES if stage in stages]
    if args.resume_from:
        stages = [stage for stage in stages if STAGES.index(stage) >= STAGES.index(args.resume_from)]
    if not stages:
        parser.error('no stage selected')
    if 'train' in stages and args.skip_gpu_check:
        parser.error('--skip-gpu-check is never valid together with the train stage')
    for name in LOCKED:
        same, value, want = locked_value(args, name)
        if not same:
            parser.error('%s=%r is not this experiment: CG-CLIP v0.1 is ONE configuration with '
                         'exactly %d updates (%s=%r). Sweeps, extra arms and epoch extension are '
                         'not available from this runner.'
                         % (name, value, LOCKED['steps'], name, want))
    log_dir = os.path.join(run_dir, 'logs')
    status_path = os.path.join(run_dir, 'run_status.json')
    lock_path = args.lock_file or os.path.join(run_dir, 'runner.lock')
    salu_log = os.path.join(run_dir, SALU_LOG_NAME)
    docs_dir = os.path.join(REPO, 'docs', RUN_ID)
    results_path = os.path.join(docs_dir, 'results.json')

    resolution_error = None
    try:
        args.torchrun = resolve_torchrun(args.python, args.torchrun)
    except SystemExit as error:
        resolution_error = str(error)
    if resolution_error is not None and not args.dry_run:
        print('PREFLIGHT_REFUSAL %s' % resolution_error, file=sys.stderr)
        raise SystemExit(PREFLIGHT_EXIT_CODE)
    try:
        commands, checkpoint, student_dir, canonical_out, urban_out = build_commands(args, run_dir)
    except SystemExit as error:
        print('PREFLIGHT_REFUSAL %s' % error, file=sys.stderr)
        raise SystemExit(PREFLIGHT_EXIT_CODE)

    # ---- dry run: print the exact command lines, execute nothing -------------
    if args.dry_run:
        print('DRY_RUN repo=%s run_dir=%s stages=%s' % (REPO, run_dir, ','.join(stages)))
        print('DRY_RUN init_state=%s' % args.init_state)
        print('DRY_RUN lock_file=%s' % lock_path)
        print('DRY_RUN status_file=%s' % status_path)
        if resolution_error is not None:
            print('DRY_RUN WARNING launcher not resolvable on this machine (nothing was executed): '
                  '%s' % resolution_error)
        for stage in stages:
            if stage in ('verify', 'report'):
                print('DRY_RUN stage=%s (in-process, no subprocess): %s'
                      % (stage, 'verify the step-%d checkpoint' % args.steps if stage == 'verify'
                         else 'assemble %s and %s' % (results_path, status_path)))
                continue
            print('DRY_RUN stage=%s log=%s' % (stage, os.path.join(log_dir, '%s.log' % stage)))
            print('DRY_RUN $ %s' % ' '.join(sanitise_argv(commands[stage])))
        print('DRY_RUN nothing was executed')
        return 0

    # ---- detached re-execution ---------------------------------------------
    if args.detach and not args.internal_detached_child:
        os.makedirs(log_dir, exist_ok=True)
        argv = [sys.executable] + [token for token in sys.argv[1:]
                                   if token not in ('--detach',)] + ['--internal-detached-child']
        master_log = os.path.join(log_dir, 'runner_master.log')
        with open(master_log, 'ab') as handle, open(os.devnull, 'rb') as devnull:
            handle.write(('\n# detached launch %s\n' % iso(time.time())).encode('utf-8'))
            handle.flush()
            process = subprocess.Popen(argv, stdin=devnull, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
        print('DETACHED pid=%d master_log=%s (the run continues without this shell)'
              % (process.pid, master_log), flush=True)
        return 0

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    env = training_environment()
    runner_log = os.path.join(log_dir, 'runner.log')

    def note(message):
        with open(runner_log, 'a', encoding='utf-8') as handle:
            handle.write('[%s] %s\n' % (iso(time.time()), message))
        print(message, flush=True)

    lock = acquire_lock(lock_path, clear_stale=args.clear_stale_lock)
    started = time.time()
    note('runner start pid=%d stages=%s resume_from=%s run_dir=%s'
         % (os.getpid(), ','.join(stages), args.resume_from, run_dir))
    try:
        update_lock(lock_path, run_id=args.run_id, run_dir=run_dir, stages=stages,
                    resume_from=args.resume_from, argv=sanitise_argv(sys.argv),
                    git_head=git_head(REPO),
                    commands={name: sanitise_argv(argv) for name, argv in commands.items()},
                    python=args.python, torchrun=args.torchrun, detached=args.detach)
        previous = read_status(status_path)
        history = list(previous.get('attempt_history') or [])
        if previous.get('phase') in ('failed', 'complete'):
            history.append({'phase': previous.get('phase'),
                            'finished_at': previous.get('finished_at'),
                            'exit_codes': previous.get('exit_codes'),
                            'failure_reason': previous.get('failure_reason')})
        stage_state = dict(previous.get('stage_state') or {})
        exit_codes = dict(previous.get('exit_codes') or {})
        write_status(status_path, run_id=args.run_id, arm=ARM, objective=OBJECTIVE,
                     phase_name=PHASE, gate_kind=GATE_KIND, run_dir=run_dir, repo=REPO,
                     pid=os.getpid(), started_at=started, started_at_iso=iso(started),
                     stages=stages, resume_from=args.resume_from, phase='starting',
                     max_steps=args.steps, save_steps=args.save_steps,
                     batch_size=args.batch_size, batch_size_per_gpu=args.batch_size,
                     world_size=args.nproc, clip_lr=args.lr, gate_lr=args.gate_lr,
                     weight_decay=args.weight_decay, warmup_length=args.warmup_length,
                     seed=args.seed, epochs=args.epochs, base_model=args.base_model,
                     loss_weights={'global': LAMBDA_GLOBAL, 'attention': LAMBDA_ATTENTION,
                                   'sparse': LAMBDA_SPARSE},
                     implementation_sha=git_head(REPO), attempt=int(previous.get('attempt') or 0) + 1,
                     attempt_history=history[-5:], lock_file=lock_path,
                     run_log_path=salu_log, commands={name: sanitise_argv(argv)
                                                      for name, argv in commands.items()})

        # ---- pre-flight: GPUs, shared init, data roots, resume eligibility ----
        report_context = {'init_state': args.init_state}
        if 'train' in stages:
            ok, detail, gpu = gpu_preflight()
            write_status(status_path, gpu_check=detail, gpu_ok=bool(ok), gpu=gpu)
            report_context['gpu'] = gpu
            report_context['gpu_check'] = detail
            if not ok:
                write_status(status_path, phase='failed',
                             failure_reason='GPU pre-check refused to start: %s' % detail)
                note('PREFLIGHT_REFUSAL GPU pre-check: %s' % detail)
                raise SystemExit(PREFLIGHT_EXIT_CODE)
            if not os.path.isfile(args.init_state):
                write_status(status_path, phase='failed',
                             failure_reason='shared init missing: %s' % args.init_state)
                note('PREFLIGHT_REFUSAL shared init missing: %s' % args.init_state)
                raise SystemExit(PREFLIGHT_EXIT_CODE)
            write_status(status_path, init_state=args.init_state,
                         init_file_sha256=sha256_of(args.init_state))
        if 'train' in stages and os.path.exists(checkpoint):
            write_status(status_path, phase='failed',
                         failure_reason='REFUSING to retrain: %s already exists' % checkpoint)
            note('PREFLIGHT_REFUSAL refusing to retrain, checkpoint exists: %s' % checkpoint)
            raise SystemExit(PREFLIGHT_EXIT_CODE)

        # a training resume is only allowed from a checkpoint that fully verifies
        resume_allowed = None
        if 'train' in stages and args.resume_from == 'train':
            resume_allowed = None
            note('resume-from train requested without --resume: training will restart from the '
                 'shared init only if no partial state exists')
        try:
            annotations, coco_images = verify_data_environment()
        except SystemExit as error:
            write_status(status_path, phase='failed',
                         failure_reason='data pre-check refused to start: %s' % error)
            note('PREFLIGHT_REFUSAL data pre-check: %s' % error)
            raise SystemExit(PREFLIGHT_EXIT_CODE)
        write_status(status_path, coco_annotations=annotations, coco_images=coco_images,
                     urban_root=URBAN_ROOT)

        # ---- stage 1: training -------------------------------------------------
        if 'train' in stages:
            write_status(status_path, phase='training', stage='train')
            code, record = run_stage('train', commands['train'], run_dir, log_dir, stage_state, env)
            exit_codes['train'] = code
            stage_state['train'] = dict(stage_state.get('train') or {}, **record)
            write_status(status_path, train_exit_code=code, exit_codes=exit_codes,
                         stage_state=stage_state, completed_steps=_completed_steps(run_dir))
            if code != 0:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='training exited %d; no verification, export or '
                                            'evaluation attempted' % code)
                note('STAGE_FAILED train exit=%d; pipeline stopped' % code)
                raise SystemExit(STAGE_EXIT_CODE['train'])
            note('stage train ok in %.1fs' % record['wall_seconds'])

        # ---- stage 2: checkpoint verification ----------------------------------
        if 'verify' in stages:
            write_status(status_path, phase='verifying', stage='verify')
            started_verify = time.time()
            found, how, alternatives = discover_checkpoint(run_dir, args.steps)
            if found is None:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='the step-%d checkpoint was not found under %s '
                                            '(looked for %s)'
                                            % (args.steps, run_dir,
                                               CHECKPOINT_TEMPLATE % (ARM, args.steps)))
                note('STAGE_FAILED verify: no step-%d checkpoint under %s' % (args.steps, run_dir))
                raise SystemExit(STAGE_EXIT_CODE['verify'])
            checkpoint = found
            note('verify checkpoint=%s (found via %s)' % (checkpoint, how))
            try:
                verify = verify_checkpoint(checkpoint, args.steps, salu_log)
            except (OSError, RuntimeError, ValueError, EOFError) as error:
                verify = {'path': checkpoint, 'problems': ['the checkpoint could not be loaded: %s'
                                                           % error]}
            except Exception as error:                                       # pragma: no cover
                verify = {'path': checkpoint, 'problems': ['the checkpoint could not be read with '
                                                           'torch: %s' % error]}
            finished_verify = time.time()
            stage_state['verify'] = dict(stage_state.get('verify') or {}, **{
                'attempts': int((stage_state.get('verify') or {}).get('attempts') or 0) + 1,
                'started_at': started_verify, 'started_at_iso': iso(started_verify),
                'finished_at': finished_verify, 'finished_at_iso': iso(finished_verify),
                'wall_seconds': finished_verify - started_verify,
                'checkpoint': checkpoint, 'discovered_via': how,
                'exit_code': 0 if verify.get('verified') else STAGE_EXIT_CODE['verify']})
            exit_codes['verify'] = 0 if verify.get('verified') else STAGE_EXIT_CODE['verify']
            write_status(status_path, checkpoint=checkpoint,
                         checkpoint_discovered_via=how,
                         checkpoint_alternatives=alternatives,
                         checkpoint_verified=bool(verify.get('verified')),
                         checkpoint_problems=verify.get('problems'),
                         checkpoint_report=verify,
                         completed_steps=verify.get('completed_steps'),
                         lr_horizon_steps=verify.get('lr_horizon_steps'),
                         gate_state_tensors=verify.get('gate_state_tensors'),
                         gate_tensor_names=GATE_TENSOR_NAMES,
                         recorded_digests=verify.get('recorded_digests'),
                         recomputed_digests=verify.get('recomputed_digests'),
                         run_log_completed_steps=verify.get('run_log'),
                         exit_codes=exit_codes, stage_state=stage_state)
            if not verify.get('verified'):
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='checkpoint verification failed (hard failure, no '
                                            'bypass): %s' % verify.get('problems'))
                note('STAGE_FAILED verify: %s' % verify.get('problems'))
                raise SystemExit(STAGE_EXIT_CODE['verify'])
            note('stage verify ok in %.1fs' % (finished_verify - started_verify))
        else:
            # downstream stages may run on their own, but they still need a verified checkpoint
            found, how, alternatives = discover_checkpoint(run_dir, args.steps)
            if found is None:
                write_status(status_path, phase='failed',
                             failure_reason='the step-%d checkpoint was not found under %s'
                                            % (args.steps, run_dir))
                note('PREFLIGHT_REFUSAL no step-%d checkpoint under %s' % (args.steps, run_dir))
                raise SystemExit(PREFLIGHT_EXIT_CODE)
            checkpoint = found

        # ---- stage 3: bare student export --------------------------------------
        if 'export' in stages:
            write_status(status_path, phase='exporting', stage='export')
            code, record = run_stage('export', commands['export'], run_dir, log_dir, stage_state,
                                     env)
            exit_codes['export'] = code
            stage_state['export'] = dict(stage_state.get('export') or {}, **record)
            write_status(status_path, export_exit_code=code, exit_codes=exit_codes,
                         stage_state=stage_state, student_export_dir=student_dir)
            if code != 0:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='student export exited %d; no evaluation attempted'
                                            % code)
                note('STAGE_FAILED export exit=%d; pipeline stopped' % code)
                raise SystemExit(STAGE_EXIT_CODE['export'])
            student, extra_students = discover_student(student_dir)
            if student is None:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='the export stage exited 0 but produced no *.pt/*.pth '
                                            'file under %s' % student_dir)
                note('STAGE_FAILED export produced no student file under %s' % student_dir)
                raise SystemExit(STAGE_EXIT_CODE['export'])
            write_status(status_path, student=student, student_sha256=sha256_of(student),
                         student_files=sorted([student] + list(extra_students)))
            note('stage export ok, student=%s' % student)

        # ---- stage 4: canonical COCO retrieval ---------------------------------
        if 'coco' in stages:
            os.makedirs(os.path.dirname(canonical_out), exist_ok=True)
            student, _extra = discover_student(student_dir)
            if student is None:
                write_status(status_path, phase='failed',
                             failure_reason='no exported student to evaluate under %s' % student_dir)
                note('PREFLIGHT_REFUSAL no exported student under %s' % student_dir)
                raise SystemExit(PREFLIGHT_EXIT_CODE)
            coco_argv = [student if token == '<STUDENT_EXPORT>' else token
                         for token in commands['coco']]
            write_status(status_path, phase='evaluating_coco', stage='coco', student=student,
                         commands=dict(read_status(status_path).get('commands') or {},
                                       coco=sanitise_argv(coco_argv)))
            code, record = run_stage('coco', coco_argv, run_dir, log_dir, stage_state, env)
            exit_codes['coco'] = code
            stage_state['coco'] = dict(stage_state.get('coco') or {}, **record)
            write_status(status_path, coco_exit_code=code, exit_codes=exit_codes,
                         stage_state=stage_state, coco_output=canonical_out)
            if code != 0:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='COCO canonical evaluation exited %d' % code)
                note('STAGE_FAILED coco exit=%d; pipeline stopped' % code)
                raise SystemExit(STAGE_EXIT_CODE['coco'])
            note('stage coco ok in %.1fs' % record['wall_seconds'])

        # ---- stage 5: Urban-1k retrieval ---------------------------------------
        if 'urban' in stages:
            os.makedirs(os.path.dirname(urban_out), exist_ok=True)
            student, _extra = discover_student(student_dir)
            if student is None:
                write_status(status_path, phase='failed',
                             failure_reason='no exported student to evaluate under %s' % student_dir)
                note('PREFLIGHT_REFUSAL no exported student under %s' % student_dir)
                raise SystemExit(PREFLIGHT_EXIT_CODE)
            urban_argv = [student if token == '<STUDENT_EXPORT>' else token
                          for token in commands['urban']]
            write_status(status_path, phase='evaluating_urban', stage='urban',
                         commands=dict(read_status(status_path).get('commands') or {},
                                       urban=sanitise_argv(urban_argv)))
            code, record = run_stage('urban', urban_argv, run_dir, log_dir, stage_state, env)
            exit_codes['urban'] = code
            stage_state['urban'] = dict(stage_state.get('urban') or {}, **record)
            write_status(status_path, urban_exit_code=code, exit_codes=exit_codes,
                         stage_state=stage_state, urban_output=urban_out)
            if code != 0:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='Urban-1k evaluation exited %d' % code)
                note('STAGE_FAILED urban exit=%d; pipeline stopped' % code)
                raise SystemExit(STAGE_EXIT_CODE['urban'])
            note('stage urban ok in %.1fs' % record['wall_seconds'])

        # ---- stage 6: results assembly -----------------------------------------
        if 'report' in stages:
            write_status(status_path, phase='reporting', stage='report')
            started_report = time.time()
            problems = []
            run_summary, summary_error = read_json(os.path.join(run_dir, 'run_summary.json'))
            if summary_error:
                problems.append('run_summary.json unreadable: %s' % summary_error)
            checkpoint_report = read_status(status_path).get('checkpoint_report')
            if not isinstance(checkpoint_report, dict) or not checkpoint_report.get('verified'):
                # the verify stage may not have been selected: re-verify so the report is honest
                try:
                    checkpoint_report = verify_checkpoint(checkpoint, args.steps, salu_log)
                except Exception as error:                                   # pragma: no cover
                    checkpoint_report = {'path': checkpoint,
                                         'problems': ['verification could not run: %s' % error],
                                         'verified': False}
            coco_result, coco_error = read_coco_result(canonical_out)
            urban_result, urban_error = read_urban_result(urban_out)
            if coco_error:
                problems.append('COCO result unavailable: %s' % coco_error)
            if urban_error:
                problems.append('Urban-1k result unavailable: %s' % urban_error)
            results = {
                'arm': ARM, 'run_id': args.run_id, 'phase': PHASE,
                'created_at': time.time(), 'created_at_iso': iso(time.time()),
                'repo': REPO, 'run_dir': run_dir, 'implementation_sha': git_head(REPO),
                'configuration': {'objective': OBJECTIVE, 'gate_kind': GATE_KIND,
                                  'max_steps': args.steps, 'save_steps': args.save_steps,
                                  'base_model': args.base_model, 'epochs': args.epochs,
                                  'batch_size_per_gpu': args.batch_size, 'world_size': args.nproc,
                                  'clip_lr': args.lr, 'gate_lr': args.gate_lr,
                                  'weight_decay': args.weight_decay,
                                  'warmup_length': args.warmup_length, 'seed': args.seed,
                                  'loss_weights': {'global': LAMBDA_GLOBAL,
                                                   'attention': LAMBDA_ATTENTION,
                                                   'sparse': LAMBDA_SPARSE},
                                  'sweeps': 0, 'extra_arms': 0},
                'init_state': {'path': args.init_state,
                               'sha256': (sha256_of(args.init_state)
                                          if os.path.isfile(args.init_state) else None),
                               'only_start_point': True},
                'train': None, 'checkpoint_verification': None,
                'coco': None, 'urban': None, 'problems': problems,
            }
            if run_summary is None:
                results['train'] = {'source': None, 'error': summary_error,
                                    'completed_steps': _completed_steps(run_dir)}
            else:
                results['train'] = {'source': os.path.join(run_dir, 'run_summary.json'),
                                    'completed_steps': run_summary.get('completed_steps'),
                                    'raw': run_summary}
            results['checkpoint_verification'] = checkpoint_report
            results['coco'] = coco_result or {'source': canonical_out, 'error': coco_error}
            results['urban'] = urban_result or {'source': urban_out, 'error': urban_error}
            results['stage_logs'] = {stage: os.path.join(log_dir, '%s.log' % stage)
                                     for stage in STAGES if stage != 'report'}
            results['stage_exit_codes'] = exit_codes
            os.makedirs(docs_dir, exist_ok=True)
            temporary = results_path + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            os.replace(temporary, results_path)
            finished_report = time.time()
            stage_state['report'] = dict(stage_state.get('report') or {}, **{
                'attempts': int((stage_state.get('report') or {}).get('attempts') or 0) + 1,
                'started_at': started_report, 'started_at_iso': iso(started_report),
                'finished_at': finished_report, 'finished_at_iso': iso(finished_report),
                'wall_seconds': finished_report - started_report,
                'results_json': results_path,
                'exit_code': STAGE_EXIT_CODE['report'] if problems else 0})
            write_status(status_path, results_json=results_path,
                         results_problems=problems,
                         run_summary_source=results['train']['source'],
                         coco_source=canonical_out, urban_source=urban_out,
                         stage_state=stage_state)
            if problems:
                write_status(status_path, phase='failed', finished_at=time.time(),
                             wall_seconds=time.time() - started,
                             failure_reason='the results could not be assembled from the real '
                                            'artifacts: %s' % problems)
                note('STAGE_FAILED report: %s' % problems)
                raise SystemExit(STAGE_EXIT_CODE['report'])
            note('stage report ok, wrote %s' % results_path)

        write_status(status_path, phase='complete', exit_codes=exit_codes,
                     stage_state=stage_state, finished_at=time.time(),
                     wall_seconds=time.time() - started, results_json=results_path,
                     run_log_path=salu_log)
        note('RUNNER_COMPLETE %s' % json.dumps({'exit_codes': exit_codes,
                                                'results': results_path}, sort_keys=True))
        return 0
    finally:
        release_lock(lock, lock_path)


if __name__ == '__main__':
    raise SystemExit(main())
