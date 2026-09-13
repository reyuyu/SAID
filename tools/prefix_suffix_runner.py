"""Background stage runner for the S0-Suffix v0.1 experiment (``objective=said_prefix_suffix_v01``).

    python tools/prefix_suffix_runner.py --dry-run
    python tools/prefix_suffix_runner.py --run-root /root/SAID-s0-suffix-v01 --detach

The experiment is TWO arms -- ``S0_SUFFIX_NATIVE`` and ``S0_SUFFIX_MASK`` -- each exactly 500
optimizer updates. The arms are executed STRICTLY SEQUENTIALLY, never concurrently: the four GPUs
are never shared between arms, and no second torchrun is ever started while one is alive. Every stage
is a real subprocess whose real exit code is recorded, in this fixed order::

    train   (exactly 500 updates, 4 GPUs, one arm at a time)
      -> verify  (the step-500 checkpoint must be COMPLETE; no bypass exists)
      -> export  (bare native student from the verified checkpoint; suffix mask absent)
      -> coco    (canonical COCO retrieval on the exported student)
      -> urban   (Urban-1k retrieval on the exported student)
      -> report  (run_status.json + conclusion.json assembled from the files actually read)

Exit codes::

    0  the selected stages completed
    1  train failed (or the trainer produced no usable checkpoint)
    2  verify refused the checkpoint (incomplete / unreadable / digest mismatch)
    3  export failed
    4  COCO evaluation failed
    5  Urban-1k evaluation failed
    6  report could not assemble the results
    10 a pre-flight refusal (lock held, fewer than 4 GPUs, busy GPU, missing data root, missing
       shared init, a step-500 checkpoint that would force a retrain); nothing was launched

Operational rules, mirrored from the frozen CG-CLIP v0.1 runner (``tools/cgclip_runner.py``):

* an exclusive ``O_CREAT|O_EXCL`` lock per run directory holds the PID, the timestamp, the argv, the
  git HEAD and the resolved command lines; a live holder refuses a second start, and a stale lock is
  reported and only replaced with ``--clear-stale-lock``;
* the four GPUs are re-queried with ``nvidia-smi`` before EVERY arm's training; an ``nvidia-smi``
  failure is treated as "not idle", never as idle, a non-empty compute-application list refuses the
  run, and a GPU already holding more than 512 MiB refuses the run. The observed per-GPU memory and
  utilisation are recorded in the status files;
* ``--detach`` re-executes this runner through ``subprocess.Popen(..., start_new_session=True)`` with
  ``stdin`` at ``/dev/null``, so a launched run outlives the SSH session;
* every stage writes its own log under ``<run dir>/logs/`` and ``run_status.json`` is rewritten
  atomically after every stage with honest exit codes, wall times, start/end timestamps and the
  attempt history;
* a failed stage stops the pipeline but KEEPS every artifact and the status file; training is never
  repeated, and a run whose step-500 checkpoint already exists is NEVER retrained -- the runner
  refuses and says so, and ``--resume-from`` re-runs only the export / evaluation / report stages.
  Those downstream stages are re-runnable: ``--resume-from export`` re-exports the verified
  checkpoint, while a resume that did not ask for ``export`` reuses the student already on disk and
  never rewrites it (the exporter itself refuses to overwrite a non-identical student).

Two-arm policy: this experiment is ONE configuration per arm, exactly 500 updates, no sweeps, no
extra arms, no epoch extension and no control run. The values that define the experiment are declared
as ``LOCKED`` below and the runner refuses to start with a non-default value instead of running a
second configuration. There is deliberately NO ``--allow-missing-*`` style escape hatch and no
``--skip-gpu-check`` style bypass: a missing, incomplete or bf16-inconsistent checkpoint is a hard
failure, and training always requires an idle 4-GPU machine.

The checkpoint schema the verify stage asserts (see also
``tools/diag/export_suffix_student.py``, which reads the same names)::

    clip_state / clip            non-empty dict of tensors (the native CLIP state)
    suffix_mask_state            the suffix mask tensor dict for S0_SUFFIX_MASK, explicitly None for
                                 S0_SUFFIX_NATIVE
    suffix_mask_config           non-empty DESCRIPTOR dict, a separate key from the tensors
    clip_state_digest            sha256 over sorted keys of clip_state
    suffix_mask_state_digest     sha256 over sorted keys of suffix_mask_state
    optimizer_clip / optimizer_mask (/ optimizer_suffix for the mask arm)
    scheduler_config / scheduler_state, completed_steps, data_cursor, rng_states, loss_config,
    sampling_config, provenance

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
OBJECTIVE = 'said_prefix_suffix_v01'
PHASE = 's0-suffix-v0.1'
ARM_NATIVE = 'S0_SUFFIX_NATIVE'
ARM_MASK = 'S0_SUFFIX_MASK'
# the arm -> run id / run directory mapping is FIXED by the experiment identity and is not derived
# from anything at run time; ``--run-root`` only moves the parent that holds both run directories
ARM_RUN_IDS = {ARM_NATIVE: 's0_suffix_native', ARM_MASK: 's0_suffix_mask'}
ARM_ORDER = (ARM_NATIVE, ARM_MASK)
# the frozen S0@500 reference used by the conclusion gate; these are the only threshold numbers here
S0_AT_500_COCO_I2T_R1 = 0.6058
S0_AT_500_COCO_T2I_R1 = 0.41236
# query-count equivalences of the frozen evaluators (one query = this many percentage points)
QUERY_EQUIVALENCE_PP = {'coco_image_query': 0.02, 'coco_text_query': 0.004, 'urban1k_query': 0.1}

# --------------------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TRAIN_SCRIPT = os.path.join(REPO, 'train', 'train_said_prefix_suffix.py')
TRAIN_MODULE = 'train.train_said_prefix_suffix'
EXPORT_SCRIPT = os.path.join(REPO, 'tools', 'diag', 'export_suffix_student.py')
CANONICAL_EVAL = os.path.join(REPO, 'tools', 'phase30a_fixed_cohort_eval.py')
URBAN_EVAL = '/root/SAID-gap-completion/tools/eval_urban1k_cls.py'

SHARED_INIT_DEFAULT = ('/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/'
                       'cvssl_initial.pt')
SHARE4V_ROOT = '/root/datasets/ShareGPT4V'
SHARE4V_JSON = 'share-captioner_coco_lcs_sam_1246k_1107.json'
COCO_ROOT = '/root/datasets/coco'
URBAN_ROOT = '/root/datasets/Urban1k/Urban1k'
PYTHON_DEFAULT = '/root/miniconda3/envs/said-smartclip/bin/python'
# the launcher of the frozen trainer invocation: the torchrun binary next to --python, never PATH
TORCHRUN_DEFAULT = 'torchrun'
MASTER_PORT = 29581
REQUIRED_GPUS = 4
MAX_IDLE_MEMORY_MIB = 512.0
DEFAULT_RUN_ROOT = '/root/SAID-s0-suffix-v01'
RUNS_SUBDIR = 'runs_salu'

SALU_LOG_NAME = 'salu_log.jsonl'
CHECKPOINT_TEMPLATE = '%s_step%06d.pt'
STUDENT_DIR_NAME = 'student_export'
STUDENT_FILE_NAME = 's0_suffix_student_step000500.pt'
STUDENT_PLACEHOLDER = '<STUDENT>'

STAGES = ('train', 'verify', 'export', 'coco', 'urban', 'report')
STAGE_EXIT_CODE = {'train': 1, 'verify': 2, 'export': 3, 'coco': 4, 'urban': 5, 'report': 6}
PREFLIGHT_EXIT_CODE = 10

# The one and only configuration of each arm. A non-default value is refused, not honoured.
LOCKED = {
    'steps': 500,
    'save_steps': '0,20,100,250,500',
    'base_model': 'B16',
    'epochs': 3,
    'batch_size': 256,
    'lr': 1e-6,
    'mask_lr': 1e-3,
    'suffix_lr': 1e-4,
    'weight_decay': 1e-2,
    'warmup_length': 200,
    'seed': 0,
    'lambda_suffix': 1.0,
    'nproc': 4,
    'image_chunk': 16,
    'text_chunk': 32,
    'num_workers': 8,
    'amp_dtype': 'bf16',
    'log_every': 10,
    'heavy_log_every': 25,
    'suffix_checkpoint': 1,
    'master_port': MASTER_PORT,
}
# The first optimizer update of every run is step 1, so 500 updates present 500 GLOBAL batches. The
# global batch is the per-GPU batch (256) times the four ranks: 500 * 1024 = 512000 synchronized
# pairs. The per-rank figure (500 * 256 = 128000) is never accepted in its place.
LOCAL_BATCH_SIZE = 256
WORLD_SIZE = 4
PAIRS_PER_STEP = LOCAL_BATCH_SIZE * WORLD_SIZE
CHECKPOINT_TAIL = 'step%06d.pt'

# the suffix mask module of S0-Suffix v0.1: two linear layers, 1024 -> 512 -> 512
SUFFIX_MASK_TENSOR_SHAPES = {'layer1.weight': (512, 1024), 'layer1.bias': (512,),
                             'layer2.weight': (512, 512), 'layer2.bias': (512,)}
SUFFIX_MASK_TENSOR_NAMES = tuple(sorted(SUFFIX_MASK_TENSOR_SHAPES))
# the checkpoint key names the verify stage reads, in resolution order; the name actually used is
# recorded in the verification report
CLIP_STATE_KEYS = ('clip_state', 'clip', 'native_clip_state', 'clip_state_dict')
SUFFIX_MASK_STATE_KEYS = ('suffix_mask_state', 'suffix_state', 'suffix_mask_tensors')
SUFFIX_MASK_CONFIG_KEYS = ('suffix_mask_config', 'suffix_config', 'suffix_mask_descriptor')
CLIP_STATE_DIGEST_KEYS = ('clip_state_digest', 'clip_digest')
SUFFIX_MASK_DIGEST_KEYS = ('suffix_mask_state_digest', 'suffix_mask_digest')
OTHER_STATE_KEYS = ('optimizer_clip', 'optimizer_mask', 'optimizer_suffix')
# any top-level key carrying one of these prefixes is a training-only suffix tensor and must live
# under the suffix mask state key, never beside it
SUFFIX_KEY_PREFIXES = ('suffix_mask.', 'suffix_mask', 'suffix.', 'suffix_', 'prefix_suffix.',
                       'suffix_branch.', 'suffix_projection.')
# number columns the run log must have carried by the end of step 500
LOG_PRESENTATION_KEYS = ('global_presentations', 'global_pair_presentations', 'global_pairs')
LOG_SYNCHRONIZED_KEYS = ('synchronized_pair_presentations', 'synchronized_pair_presentations_count',
                         'synchronized_pairs')
LOG_PAIR_SIZE_KEYS = ('pair_size', 'synchronized_pair_size', 'world_batch_size',
                      'global_batch_size', 'batch_size')
LOG_COMPLETED_KEYS = ('completed_steps', 'step', 'global_step', 'optimizer_updates')

SENSITIVE_KEYS = ('password', 'passwd', 'token', 'secret', 'key', 'credential')
# fields that describe one attempt of one arm and must never leak from an earlier failed arm
ATTEMPT_FIELDS = ('failure_reason', 'exit_codes', 'stage_state', 'train_exit_code',
                  'verify_exit_code', 'export_exit_code', 'coco_exit_code', 'urban_exit_code',
                  'report_exit_code', 'checkpoint', 'checkpoint_verified', 'checkpoint_problems',
                  'checkpoint_report', 'student', 'student_sha256', 'conclusion',
                  'conclusion_verdict')


# --------------------------------------------------------------------------- small helpers
def sha256_of(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo=None):
    try:
        result = subprocess.run(['git', '-C', repo or REPO, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return result.stdout.strip() or 'unknown'


def is_hex_digest(value, length=64):
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in '0123456789abcdefABCDEF' for character in value)


def sanitise_argv(argv):
    """The command actually run, with anything credential-looking removed from the record."""
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


def iso(timestamp):
    return time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(timestamp))


def read_json(path):
    """``(payload, error)`` -- never raises; the error text is recorded beside the numbers."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle), None
    except (ValueError, OSError) as error:
        return None, str(error)


def write_status(path, **fields):
    """Atomic read-modify-write of a status file, so a dashboard only ever sees valid JSON."""
    payload = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            payload = {}
    payload.update(fields)
    payload['updated_at'] = time.time()
    payload['updated_at_iso'] = iso(payload['updated_at'])
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return payload


def read_status(path):
    payload, _error = read_json(path)
    return payload if isinstance(payload, dict) else {}


def salu_log_path(run_dir):
    """The run log the verify stage reads: ``<run dir>/salu_log.jsonl``."""
    return os.path.join(run_dir, SALU_LOG_NAME)


def completed_steps_from_summary(run_dir):
    """The trainer's own ``run_summary.json`` step count, or None; never a substitute for verify."""
    payload, error = read_json(os.path.join(run_dir, 'run_summary.json'))
    if error or not isinstance(payload, dict):
        return None
    try:
        return int(payload.get('completed_steps', 0))
    except (TypeError, ValueError):
        return None


# the name used inside the stage loop; kept as an alias so the trainer's own summary is read by
# exactly one implementation
_completed_steps = completed_steps_from_summary


def resolve_lock_path(override, run_dir, arm):
    """The per-arm lock: an explicit ``--lock-file`` is suffixed, otherwise it lives in the arm dir.

    The lock is per run directory, so two invocations of DIFFERENT arms are refused by their own
    directory lock, while one invocation of both arms holds one lock at a time.
    """
    if override:
        return '%s.%s' % (override, ARM_RUN_IDS[arm])
    return os.path.join(run_dir, 'runner.lock')


# --------------------------------------------------------------------------- lock
def acquire_lock(path, clear_stale=False):
    """``O_CREAT|O_EXCL`` lock holding the PID; returns the open handle or raises SystemExit."""
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
    os.write(handle, json.dumps({'pid': os.getpid(), 'started_at': time.time(),
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
    payload['updated_at_iso'] = iso(time.time())
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
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

    Refuses when nvidia-smi cannot be executed, when fewer than four devices are visible, when the
    per-GPU query cannot be parsed, when a compute application is registered, or when a GPU already
    holds more than ``MAX_IDLE_MEMORY_MIB``. The full observation is returned for the status file
    either way, so the observed values survive a refusal.
    """
    try:
        devices, rows, raw = _query_gpu_rows()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return False, 'nvidia-smi could not be used: %s' % error, {'error': str(error)}
    report = {'nvidia_smi_visible_devices': len(devices), 'per_gpu': rows,
              'query_echo': raw, 'required_gpus': REQUIRED_GPUS,
              'max_idle_memory_mib': MAX_IDLE_MEMORY_MIB,
              'query': 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu',
              'observed_at_iso': iso(time.time())}
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
        return False, 'GPUs are busy (the two arms must never compete for the four GPUs): %s' \
            % '; '.join(running), report
    occupied = []
    for row in rows:
        try:
            used = float(row['memory_used_mib'])
        except (TypeError, ValueError):
            return False, 'cannot read memory.used from nvidia-smi row %r' % row, report
        if used > MAX_IDLE_MEMORY_MIB:
            occupied.append('%s: %s MiB' % (row['index'], row['memory_used_mib']))
    if occupied:
        return False, 'GPUs already hold more than %.0f MiB: %s' \
            % (MAX_IDLE_MEMORY_MIB, '; '.join(occupied)), report
    return True, '%d GPUs idle (max %.0f MiB used, no compute application)' \
        % (len(devices), max(float(row['memory_used_mib']) for row in rows)), report


# --------------------------------------------------------------------------- launchers
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


def required_inputs(args, commands):
    """``(missing, present)`` lists of the repository inputs every execution needs.

    ``--dry-run`` prints this instead of refusing, so a worktree that is still receiving the trainer
    can still be inspected; every real execution refuses on a missing input.
    """
    wanted = [(CANONICAL_EVAL, 'canonical COCO evaluator'),
              (URBAN_EVAL, 'Urban-1k evaluator'),
              (args.init_state, 'shared frozen CV-SSL initialisation')]
    checked = []
    missing = []
    for path, description in wanted:
        entry = {'path': path, 'description': description, 'exists': os.path.exists(path)}
        checked.append(entry)
        if not entry['exists']:
            missing.append(entry)
    for stage in ('train', 'export'):
        for token in commands[ARM_ORDER[0]][stage]:
            if token.startswith(os.sep) and token.endswith('.py') and not os.path.isfile(token):
                missing.append({'path': token, 'description': '%s entry point' % stage,
                                'exists': False})
                break
    return missing, checked


# --------------------------------------------------------------------------- stage commands
def build_commands(args, run_dir, arm):
    """The exact argv of every stage of ONE arm, built before anything is launched."""
    python = args.python
    torchrun = args.torchrun
    checkpoint = os.path.join(run_dir, CHECKPOINT_TEMPLATE % (arm, args.steps))
    student_dir = os.path.join(run_dir, STUDENT_DIR_NAME)
    student = os.path.join(student_dir, STUDENT_FILE_NAME)
    evaluation_dir = os.path.join(run_dir, 'evaluation')
    canonical_out = os.path.join(evaluation_dir, '%s_step%06d_canonical.json' % (arm, args.steps))
    urban_out = os.path.join(evaluation_dir, '%s_step%06d_urban1k.json' % (arm, args.steps))
    commands = {
        # the frozen trainer contract: the module form, exactly one configuration per arm
        'train': [torchrun, '--nproc_per_node=%d' % args.nproc,
                  '--master_port=%d' % args.master_port, TRAIN_MODULE,
                  '--arm', arm, '--base_model', args.base_model,
                  '--batch-size', str(args.batch_size), '--epochs', str(args.epochs),
                  '--lr', str(args.lr), '--mask_lr', str(args.mask_lr),
                  '--suffix_lr', str(args.suffix_lr), '--weight_decay', str(args.weight_decay),
                  '--warmup_length', str(args.warmup_length), '--seed', str(args.seed),
                  '--lambda_suffix', str(args.lambda_suffix), '--init_state', args.init_state,
                  '--output_dir', run_dir, '--max_steps', str(args.steps),
                  '--save_completed_steps', args.save_steps,
                  '--log_every', str(args.log_every),
                  '--heavy_log_every', str(args.heavy_log_every),
                  '--num_workers', str(args.num_workers), '--amp_dtype', args.amp_dtype,
                  '--image_chunk', str(args.image_chunk), '--text_chunk', str(args.text_chunk),
                  '--suffix_checkpoint', str(args.suffix_checkpoint)],
        'export': [python, EXPORT_SCRIPT, '--checkpoint', checkpoint,
                   '--output_dir', student_dir, '--expect-steps', str(args.steps),
                   '--expect-arm', arm],
        # the STUDENT placeholder is embedded inside the token ``500:<STUDENT>``; the caller
        # substitutes it inside every token, never by whole-token matching
        'coco': [python, CANONICAL_EVAL, '--label', arm,
                 '--gap_anti_temperature', '1.0', '--sharegpt4v_manifest', '',
                 '--data_root', SHARE4V_ROOT, '--image_root', SHARE4V_ROOT,
                 '--image_batch_size', '64', '--canonical', '--canonical_only', '--coco',
                 '--canonical_tags', str(args.steps),
                 '--canonical_names', '%s@%d' % (arm, args.steps),
                 '--checkpoints', '%d:%s' % (args.steps, STUDENT_PLACEHOLDER),
                 '--coco_root', os.path.join(COCO_ROOT, 'val2017'),
                 '--output', canonical_out],
        'urban': [python, URBAN_EVAL, '--checkpoint', STUDENT_PLACEHOLDER,
                  '--label', '%s@%d' % (arm, args.steps),
                  '--expect-steps', str(args.steps), '--base_model', 'ViT-B/16',
                  '--device', 'cuda', '--batch_size', '64',
                  '--urban_root', URBAN_ROOT, '--out', urban_out],
    }
    paths = {'checkpoint': checkpoint, 'student_dir': student_dir, 'student': student,
             'canonical_out': canonical_out, 'urban_out': urban_out}
    return commands, paths


def substitute_student(argv, student):
    """Replace the placeholder inside each token and refuse if any occurrence survives."""
    resolved = [token.replace(STUDENT_PLACEHOLDER, student) for token in argv]
    left = [token for token in resolved if STUDENT_PLACEHOLDER in token]
    if left:
        raise SystemExit('a stage command still holds an unresolved student placeholder: %r' % left)
    return resolved


def discover_checkpoint(run_dir, arm, steps):
    """The step checkpoint: exact name first, then the same name lower-cased, then a walk.

    Returns ``(path_or_None, how, alternatives)``. A checkpoint of ANOTHER arm is never accepted:
    the walk only matches the tail ``step%06d.pt`` and the arm is checked from the file name.
    """
    exact = os.path.join(run_dir, CHECKPOINT_TEMPLATE % (arm, steps))
    if os.path.isfile(exact):
        return exact, 'exact', []
    names = (CHECKPOINT_TEMPLATE % (arm, steps), CHECKPOINT_TEMPLATE % (arm.lower(), steps))
    for base in (run_dir, os.path.join(run_dir, 'checkpoints')):
        if not os.path.isdir(base):
            continue
        present = set(os.listdir(base))
        for name in names:
            candidate = os.path.join(base, name)
            if name in present and os.path.isfile(candidate):
                return candidate, 'glob', []
    matches = []
    for root, _directories, files in os.walk(run_dir):
        for name in files:
            if not name.endswith(CHECKPOINT_TAIL % steps):
                continue
            if name.startswith(arm) or name.startswith(arm.lower()):
                matches.append(os.path.join(root, name))
    if not matches:
        return None, None, []
    matches.sort()
    return matches[0], 'walk', matches[1:]


def discover_step500_checkpoints(run_dir):
    """Every ``*step000500.pt`` under a run directory: used only to refuse a retrain."""
    if not os.path.isdir(run_dir):
        return []
    found = []
    for root, _directories, files in os.walk(run_dir):
        found.extend(os.path.join(root, name) for name in files
                     if name.endswith(CHECKPOINT_TAIL % 500))
    return sorted(found)


def discover_student(student_dir):
    """The exported student: the *.pt/*.pth/*.bin files the export tool actually produced."""
    if not os.path.isdir(student_dir):
        return None, []
    candidates = sorted(os.path.join(student_dir, name) for name in os.listdir(student_dir)
                        if name.endswith(('.pt', '.pth', '.bin')))
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda path: (os.path.getsize(path), path), reverse=True)
    return ranked[0], ranked[1:]


def run_stage(name, argv, log_dir, stage_state, env=None):
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
              'attempt': int((stage_state.get(name) or {}).get('attempts') or 0) + 1}
    return process.returncode, record


# --------------------------------------------------------------------------- digest / schema
def state_digest(state_dict):
    """sha256 over sorted keys: key bytes then ``value.detach().float().cpu().numpy().tobytes()``.

    Re-implemented here (instead of importing the trainer) so the verify stage is an INDEPENDENT
    recomputation of the digest the trainer wrote: what the trainer claims and what the bytes on
    disk contain are compared by two different pieces of code.
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


def resolve_key(payload, candidates, what):
    for name in candidates:
        if name in payload:
            return name, payload[name]
    return None, None


def describe_dict_problem(name, value):
    """A description dict must be a non-empty dict and must not smuggle tensors in."""
    import torch
    if not isinstance(value, dict) or not value:
        return '%s is empty or not a dict (%s)' % (name, type(value).__name__)
    tensor_keys = sorted(str(key) for key, item in value.items() if torch.is_tensor(item))
    if tensor_keys:
        return ('%s carries tensor entries %r: the tensors and their description are separate keys '
                'and must never be merged into one dict' % (name, tensor_keys[:6]))
    return None


def optimizer_summary(name, payload, problems):
    """Validate one optimizer payload: a real ``state`` dict and non-empty ``param_groups``."""
    optimizer = payload.get(name)
    summary = {'present': isinstance(optimizer, dict)}
    if not summary['present']:
        problems.append('%s is missing or is not a dict (%s)' % (name, type(optimizer).__name__))
        return summary
    state = optimizer.get('state')
    groups = optimizer.get('param_groups')
    if not isinstance(state, dict) or not state:
        problems.append('%s.state is missing or empty (%s)' % (name, type(state).__name__))
        summary['state_entries'] = 0
    else:
        summary['state_entries'] = len(state)
        summary['state_tensor_entries'] = sum(len(entry) for entry in state.values()
                                              if isinstance(entry, dict))
        if summary['state_entries'] < 2:
            problems.append('%s.state carries only %d parameter entr(ies), which cannot be the real '
                            'AdamW state of a trained model'
                            % (name, summary['state_entries']))
    if not isinstance(groups, list) or not groups:
        problems.append('%s.param_groups is missing or empty (%s)' % (name, type(groups).__name__))
    else:
        summary['param_groups'] = len(groups)
        summary['param_group_keys'] = sorted({key for group in groups if isinstance(group, dict)
                                              for key in group})
    return summary


def _scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def _log_entries(node, node_path, wanted):
    """Every ``wanted`` key anywhere in a log record, most specific path first."""
    found = []

    def walk(item, path):
        if isinstance(item, dict):
            for key in sorted(item):
                child = '%s.%s' % (path, key)
                if key in wanted:
                    found.append((path.count('.') + len(wanted) * 0, child, item[key]))
                walk(item[key], child)
        elif isinstance(item, list):
            for index, element in enumerate(item):
                walk(element, '%s[%d]' % (path, index))

    walk(node, node_path)
    found.sort(key=lambda entry: (-entry[0], entry[1]))
    return [{'path': entry[1], 'value': entry[2]} for entry in found]


def read_run_log(salu_log, expected_steps, expected_pairs):
    """Read ``salu_log.jsonl`` and prove it reached step 500 with the GLOBAL presentation counts.

    Returns ``(report, problems)``. The expected figure is the GLOBAL one
    (``expected_steps * LOCAL_BATCH_SIZE * WORLD_SIZE`` = 512000); the per-rank figure
    (``500 * 256`` = 128000) is never accepted, and the log is searched most-specific-path-first.
    """
    problems = []
    per_rank = int(expected_steps) * LOCAL_BATCH_SIZE
    report = {'path': salu_log, 'exists': os.path.isfile(salu_log),
              'expected_global_presentations': expected_pairs,
              'expected_synchronized_pair_presentations': expected_pairs,
              'per_rank_figure_that_is_not_accepted': per_rank,
              'rule': ('global_presentations and synchronized_pair_presentations are GLOBAL pair '
                       'presentations: %d updates * %d per-GPU pairs * %d ranks = %d, never the '
                       'per-rank %d' % (expected_steps, LOCAL_BATCH_SIZE, WORLD_SIZE,
                                        expected_pairs, per_rank))}
    if not report['exists']:
        problems.append('the run log is missing: %s' % salu_log)
        return report, problems
    records = 0
    matched_lines = []
    with open(salu_log, 'r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                report.setdefault('unparseable_lines', []).append(number)
                continue
            records += 1
            if not isinstance(record, dict):
                continue
            entries = _log_entries(record, 'record', set(LOG_COMPLETED_KEYS))
            for entry in entries:
                try:
                    if int(entry['value']) == int(expected_steps):
                        matched_lines.append({'line': number, 'path': entry['path'],
                                              'value': entry['value']})
                        break
                except (TypeError, ValueError):
                    continue
    report['records'] = records
    report['lines_with_completed_steps_%d' % expected_steps] = matched_lines[:10]
    if not matched_lines:
        problems.append('%s has no record with completed_steps == %d (%d record(s) read)'
                        % (salu_log, expected_steps, records))
        return report, problems

    # the presentation counts: search the records that reached the final step first, then all records
    chosen = None
    for number, line in enumerate(open(salu_log, 'r', encoding='utf-8'), start=1):
        if number not in {entry['line'] for entry in matched_lines}:
            continue
        try:
            chosen = json.loads(line)
        except ValueError:
            chosen = None
        if chosen is not None:
            report['presentation_source_line'] = number
            break
    sources = [candidate for candidate in (chosen,) if isinstance(candidate, dict)]
    if not sources:
        sources = []
        with open(salu_log, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    sources.append(record)
    for label, keys, expected in (('global_presentations', LOG_PRESENTATION_KEYS, expected_pairs),
                                  ('synchronized_pair_presentations', LOG_SYNCHRONIZED_KEYS,
                                   expected_pairs)):
        entry = None
        for source in sources:
            candidates = _log_entries(source, 'record', set(keys))
            if candidates:
                entry = candidates[0]
                break
        if entry is None:
            problems.append('%s: no %r column has been found in any record of %s'
                            % (label, list(keys), salu_log))
            continue
        report[label] = entry
        try:
            got = int(entry['value'])
        except (TypeError, ValueError):
            problems.append('%s=%r is not an integer (%s)' % (label, entry['value'], entry['path']))
            continue
        if got != expected:
            hint = ''
            if got == int(expected / 4):
                hint = (' -- that is the PER-RANK figure (%d/4); the gate is on the GLOBAL figure %d'
                        % (expected, expected))
            problems.append('%s=%d at %s, expected %d%s'
                            % (label, got, entry['path'], expected, hint))
    # the pair size the trainer recorded, when a record carries one, explains the arithmetic and is
    # recorded beside the observed presentation counts (it is never used to loosen them)
    for source in sources:
        sizes = _log_entries(source, 'record', set(LOG_PAIR_SIZE_KEYS))
        if sizes:
            report['pair_size_observed'] = sizes[0]
            break
    return report, problems


# --------------------------------------------------------------------------- the verify stage
def verify_checkpoint(path, arm, expected_steps, salu_log, native_key_set=None):
    """Full schema verification of the step-500 training checkpoint. Returns a report dict.

    Every check appends a human-readable entry to ``problems``; a non-empty ``problems`` list is a
    stage failure. There is no flag anywhere that softens any of these checks.
    """
    import torch
    expected_pairs = int(expected_steps) * PAIRS_PER_STEP
    report = {'path': path, 'arm': arm, 'expected_steps': int(expected_steps),
              'file_sha256': None, 'file_size_bytes': None, 'problems': [], 'checks': {}}
    problems = report['problems']
    checks = report['checks']
    if not os.path.isfile(path):
        problems.append('the checkpoint is missing: %s' % path)
        checks['file_exists'] = False
        return report
    checks['file_exists'] = True
    report['file_size_bytes'] = os.path.getsize(path)
    report['file_sha256'] = sha256_of(path)
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:                                                   # noqa: BLE001
        problems.append('the checkpoint could not be loaded: %s' % error)
        checks['loads'] = False
        return report
    checks['loads'] = True
    if not isinstance(payload, dict):
        problems.append('the checkpoint is not a dict but %s' % type(payload).__name__)
        checks['is_dict'] = False
        return report
    checks['is_dict'] = True
    report['top_level_keys'] = sorted(str(key) for key in payload)
    report['payload_class'] = type(payload).__name__

    # --- training progress ---------------------------------------------------
    completed = payload.get('completed_steps')
    report['completed_steps'] = completed
    try:
        numeric_completed = int(completed)
    except (TypeError, ValueError):
        numeric_completed = None
    checks['completed_steps_present'] = completed is not None
    checks['completed_steps_is_%d' % expected_steps] = numeric_completed == int(expected_steps)
    if numeric_completed != int(expected_steps):
        problems.append('completed_steps=%r expected %d: only a finished %d-update checkpoint may '
                        'be exported or evaluated' % (completed, int(expected_steps),
                                                      int(expected_steps)))

    # --- objective / arm identity -------------------------------------------
    report['objective'] = payload.get('objective')
    report['arm_in_payload'] = payload.get('arm')
    checks['objective'] = payload.get('objective') == OBJECTIVE
    if not checks['objective']:
        problems.append('objective=%r expected %r' % (payload.get('objective'), OBJECTIVE))
    checks['arm'] = payload.get('arm') == arm
    if not checks['arm']:
        problems.append('arm=%r expected %r' % (payload.get('arm'), arm))

    # --- native CLIP state ---------------------------------------------------
    clip_key, clip_state = resolve_key(payload, CLIP_STATE_KEYS, 'clip state')
    report['clip_state_key'] = clip_key
    if clip_key is None:
        problems.append('no clip state key is present (looked for %r)' % list(CLIP_STATE_KEYS))
        clip_state = None
    elif not isinstance(clip_state, dict) or not clip_state:
        problems.append('%s is empty or not a dict (%s)'
                        % (clip_key, type(clip_state).__name__))
        clip_state = None
    else:
        non_tensors = sorted(str(key) for key, value in clip_state.items()
                             if not torch.is_tensor(value))
        if non_tensors:
            problems.append('%s carries non-tensor entries %r; the clip state must be the tensor '
                            'dict' % (clip_key, non_tensors[:6]))
            clip_state = None
        else:
            report['clip_tensor_count'] = len(clip_state)
            report['mask_net_tensors'] = sorted(key for key in clip_state
                                                if key.startswith('mask_net.'))
            checks['mask_net_present_in_native_key_set'] = bool(report['mask_net_tensors'])
            leaked = sorted(key for key in clip_state if key.startswith(SUFFIX_KEY_PREFIXES))
            checks['suffix_tensors_absent_from_clip_state'] = not leaked
            if leaked:
                problems.append('the clip state already carries suffix-alignment tensors %r: the '
                                'training-only suffix branch must live under the suffix mask state '
                                'key, never inside the native CLIP state' % leaked[:6])
            if native_key_set is not None:
                report['native_key_set_source'] = native_key_set['source']
                report['native_key_set_tensor_count'] = native_key_set['count']
                missing = sorted(set(native_key_set['keys']) - set(clip_state))
                extra = sorted(set(clip_state) - set(native_key_set['keys']))
                checks['clip_state_covers_native_key_set'] = not missing
                if missing:
                    problems.append('the clip state is missing %d tensor(s) of the native CLIP key '
                                    'set including %r; the student could not be a complete native '
                                    'student' % (len(missing), missing[:6]))
                checks['clip_state_has_no_foreign_keys'] = not extra
                if extra:
                    problems.append('the clip state carries %d key(s) that are not part of the '
                                    'native CLIP key set: %r' % (len(extra), extra[:6]))

    # --- suffix mask state ---------------------------------------------------
    mask_key, mask_value = resolve_key(payload, SUFFIX_MASK_STATE_KEYS, 'suffix mask state')
    report['suffix_mask_state_key'] = mask_key
    checks['suffix_mask_state_present'] = mask_key is not None
    if mask_key is None:
        problems.append('the suffix mask state key is MISSING entirely (looked for %r); it must be '
                        'present and explicitly None for %s and a non-empty tensor dict for %s'
                        % (list(SUFFIX_MASK_STATE_KEYS), ARM_NATIVE, ARM_MASK))
    report['suffix_mask_state_is_none'] = mask_value is None
    if arm == ARM_NATIVE:
        checks['suffix_mask_state_is_none_for_native_arm'] = mask_value is None
        if mask_value is not None:
            problems.append('arm %s must carry %s=None, found %s: the flat arm has no suffix module '
                            'and a populated state here would mean the checkpoint is not the arm it '
                            'claims to be' % (ARM_NATIVE, mask_key, type(mask_value).__name__))
    elif arm == ARM_MASK:
        if not isinstance(mask_value, dict) or not mask_value:
            checks['suffix_mask_state_is_tensor_dict'] = False
            problems.append('arm %s must carry a non-empty suffix mask tensor dict under %s, found '
                            '%s' % (ARM_MASK, mask_key, type(mask_value).__name__))
        else:
            non_tensors = sorted(str(key) for key, value in mask_value.items()
                                 if not torch.is_tensor(value))
            checks['suffix_mask_state_is_tensor_dict'] = not non_tensors
            if non_tensors:
                problems.append('the suffix mask state carries non-tensor entries %r; it must be '
                                'the suffix mask TENSOR dict, never a description dict'
                                % non_tensors[:6])
            else:
                shapes = {str(key): tuple(value.shape) for key, value in mask_value.items()}
                report['suffix_mask_state_tensors'] = {key: list(shape)
                                                       for key, shape in sorted(shapes.items())}
                got = {str(key).split('.', 1)[-1]: tuple(value.shape)
                       for key, value in mask_value.items()}
                expected = {name: list(shape)
                            for name, shape in sorted(SUFFIX_MASK_TENSOR_SHAPES.items())}
                absent = [name for name in SUFFIX_MASK_TENSOR_NAMES if name not in got]
                wrong = {name: list(shape) for name, shape in sorted(got.items())
                         if SUFFIX_MASK_TENSOR_SHAPES.get(name) != shape}
                checks['suffix_mask_shapes_match_module'] = not absent and not wrong
                if absent:
                    problems.append('the suffix mask state is missing the module tensor(s) %r '
                                    '(expected shapes %r)' % (absent, expected))
                if wrong:
                    problems.append('the suffix mask state tensor shape(s) %r do not match the '
                                    'suffix mask module %r' % (wrong, expected))
                extra = sorted(name for name in got if name not in SUFFIX_MASK_TENSOR_SHAPES)
                checks['suffix_mask_has_no_extra_tensors'] = not extra
                if extra:
                    problems.append('the suffix mask state carries unexpected tensors %r that are '
                                    'not part of the suffix mask module' % extra)
                non_float = sorted(name for name, value in mask_value.items()
                                   if not value.is_floating_point())
                checks['suffix_mask_tensors_are_float'] = not non_float
                if non_float:
                    problems.append('the suffix mask tensor(s) %r are not floating point'
                                    % non_float)
    else:
        problems.append('unknown arm %r: the only arms are %r' % (arm, list(ARM_ORDER)))

    # --- the description dict must be a SEPARATE key -------------------------
    config_key, mask_config = resolve_key(payload, SUFFIX_MASK_CONFIG_KEYS, 'suffix mask config')
    report['suffix_mask_config_key'] = config_key
    checks['suffix_mask_config_present'] = config_key is not None
    if config_key is None:
        problems.append('the suffix mask description key is MISSING (looked for %r); the module '
                        'description must be a key of its own'
                        % list(SUFFIX_MASK_CONFIG_KEYS))
    else:
        checks['suffix_mask_config_is_separate_key'] = config_key not in (clip_key, mask_key)
        if config_key in (clip_key, mask_key):
            problems.append('the suffix mask DESCRIPTION is stored under %r, which is also a tensor '
                            'key: a key collision of exactly this kind already destroyed one '
                            'experiment\'s gate weights. The description and the tensors must be '
                            'two different keys.' % config_key)
        problem = describe_dict_problem(config_key, mask_config)
        checks['suffix_mask_config_is_description_dict'] = problem is None
        if problem:
            problems.append(problem)
        else:
            report['suffix_mask_config'] = {str(key): mask_config[key] for key in sorted(mask_config)
                                            if _scalar(mask_config[key])}
            report['suffix_mask_config_keys'] = sorted(str(key) for key in mask_config)
    if clip_key is not None and mask_key is not None:
        checks['clip_and_mask_keys_are_distinct'] = clip_key != mask_key
        if clip_key == mask_key:
            problems.append('the clip state and the suffix mask state are the SAME key (%r)' % clip_key)

    # --- optimizer states ----------------------------------------------------
    expected_optimizers = list(OTHER_STATE_KEYS[:2]) + ([OTHER_STATE_KEYS[2]]
                                                       if arm == ARM_MASK else [])
    report['expected_optimizers'] = expected_optimizers
    report['optimizers'] = {name: optimizer_summary(name, payload, problems)
                            for name in expected_optimizers}
    if arm == ARM_NATIVE:
        checks['optimizer_suffix_absent_or_none_for_native_arm'] = (
            payload.get(OTHER_STATE_KEYS[2]) in (None, {}))
        if not checks['optimizer_suffix_absent_or_none_for_native_arm']:
            problems.append('%s is populated on arm %s; the flat arm trains no suffix module'
                            % (OTHER_STATE_KEYS[2], ARM_NATIVE))
    for name in ('scheduler_config', 'scheduler_state'):
        value = payload.get(name)
        checks[name] = isinstance(value, dict) and bool(value)
        if not checks[name]:
            problems.append('%s is missing, empty or not a dict (%s)'
                            % (name, type(value).__name__))
        elif name == 'scheduler_config':
            report['scheduler_config'] = {str(key): value[key] for key in sorted(value)
                                          if _scalar(value[key])}

    # --- the run log: step 500 and the GLOBAL presentation counts ------------
    log_report, log_problems = read_run_log(salu_log, expected_steps, expected_pairs)
    report['run_log'] = log_report
    problems.extend(log_problems)
    checks['run_log_reports_step_%d' % expected_steps] = not log_problems

    # --- recomputed digests --------------------------------------------------
    clip_digest_key, recorded_clip = resolve_key(payload, CLIP_STATE_DIGEST_KEYS, 'clip digest')
    mask_digest_key, recorded_mask = resolve_key(payload, SUFFIX_MASK_DIGEST_KEYS, 'mask digest')
    report['recorded_digest_keys'] = {'clip_state_digest': clip_digest_key,
                                      'suffix_mask_state_digest': mask_digest_key}
    report['recorded_digests'] = {'clip_state_digest': recorded_clip,
                                  'suffix_mask_state_digest': recorded_mask}
    recomputed = {}
    if clip_state is not None:
        try:
            recomputed['clip_state_digest'] = state_digest(clip_state)
        except ValueError as error:
            problems.append('the clip state digest cannot be recomputed: %s' % error)
    if arm == ARM_MASK and isinstance(mask_value, dict) and mask_value:
        try:
            recomputed['suffix_mask_state_digest'] = state_digest(mask_value)
        except ValueError as error:
            problems.append('the suffix mask digest cannot be recomputed: %s' % error)
    report['recomputed_digests'] = recomputed
    checks['digest_recomputation_is_deterministic'] = (
        recomputed == {key: (state_digest(clip_state) if key == 'clip_state_digest'
                             else state_digest(mask_value))
                       for key in recomputed})
    for label, recorded, value in (('clip_state_digest', recorded_clip,
                                    recomputed.get('clip_state_digest')),
                                   ('suffix_mask_state_digest', recorded_mask,
                                    recomputed.get('suffix_mask_state_digest'))):
        if label == 'suffix_mask_state_digest' and arm == ARM_NATIVE:
            checks['native_arm_records_no_suffix_digest'] = recorded is None
            if recorded is not None:
                problems.append('%s=%r is recorded on arm %s while the suffix mask state is None'
                                % (label, recorded, ARM_NATIVE))
            continue
        if recorded is None:
            checks['%s_present' % label] = False
            problems.append('%s is missing from the header' % label)
            continue
        checks['%s_present' % label] = True
        if not is_hex_digest(recorded):
            problems.append('%s is not a 64-character hex digest: %r' % (label, recorded))
            continue
        if value is None:
            problems.append('%s cannot be recomputed: the matching state dict is unusable' % label)
            continue
        checks['%s_matches_recomputed' % label] = recorded.lower() == value.lower()
        if recorded.lower() != value.lower():
            problems.append('%s mismatch: recorded %s, recomputed from the loaded tensors %s'
                            % (label, recorded, value))

    # --- remaining required metadata ----------------------------------------
    required = ('data_cursor', 'rng_states', 'loss_config', 'sampling_config', 'provenance')
    for name in required:
        value = payload.get(name)
        present = value is not None
        checks['%s_present' % name] = present
        if not present:
            problems.append('%s is missing (None or absent)' % name)
            continue
        if name in ('loss_config', 'sampling_config', 'provenance'):
            problem = describe_dict_problem(name, value) if isinstance(value, dict) else None
            if isinstance(value, dict):
                if problem:
                    problems.append(problem)
                else:
                    report[name] = {str(key): value[key] for key in sorted(value)
                                    if _scalar(value[key])}
            else:
                checks['%s_is_dict' % name] = False
                problems.append('%s is not a dict (%s)' % (name, type(value).__name__))
        elif name == 'rng_states':
            if not isinstance(value, dict) or not value:
                checks['rng_states_is_dict'] = False
                problems.append('rng_states is empty or not a dict (%s)' % type(value).__name__)
            else:
                checks['rng_states_is_dict'] = True
                report['rng_state_count'] = len(value)
        elif name == 'data_cursor':
            report['data_cursor'] = value if _scalar(value) else 'non-scalar (%s)' \
                % type(value).__name__
    return report


def native_key_set_reference(base_model):
    """The native CLIP key set as an independent cross-check, when the model can be built.

    ``torch``/``longclip`` may be unavailable in the runner interpreter; that only removes this
    cross-check, it never removes a check: the failure text is recorded and the verify stage still
    requires every other condition.
    """
    try:
        import argparse as _argparse
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from model import longclip
        model, _preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                     args=_argparse.Namespace())
        keys = sorted(model.state_dict())
        del model
    except Exception as error:                                                   # noqa: BLE001
        return {'source': 'model/longclip.py:%s' % base_model, 'keys': None, 'count': None,
                'unavailable': str(error)}
    return {'source': 'model/longclip.py:%s (freshly built)' % base_model, 'keys': keys,
            'count': len(keys)}


# --------------------------------------------------------------------------- result readers
def read_coco_result(path):
    """Read the canonical COCO R@1 pair from the shapes the frozen evaluator has produced.

    The evaluator nests ``canonical.<name>.coco_val2017``; older or flat layouts put
    ``coco_val2017`` or ``coco`` at the top level. Both are read, neither is assumed, and the number
    always carries the path it came from.
    """
    payload, error = read_json(path)
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, 'the COCO result at %s is not a dict' % path
    inner, source_key = None, None
    canonical = payload.get('canonical')
    if isinstance(canonical, dict) and canonical:
        for name in sorted(canonical):
            entry = canonical[name]
            if isinstance(entry, dict) and isinstance(entry.get('coco_val2017'), dict):
                inner, source_key = entry['coco_val2017'], 'canonical.%s.coco_val2017' % name
                break
    if inner is None and isinstance(payload.get('coco_val2017'), dict):
        inner, source_key = payload['coco_val2017'], 'coco_val2017'
    if inner is None and isinstance(payload.get('coco'), dict):
        inner, source_key = payload['coco'], 'coco'
    if inner is None:
        return None, 'no COCO block has been found in %s (top-level keys: %s)' \
            % (path, sorted(str(key) for key in payload))
    result = {'source': path, 'source_key': source_key,
              'raw_keys': sorted(str(key) for key in inner),
              'image2text_R1': inner.get('image2text_R1'),
              'text2image_R1': inner.get('text2image_R1')}
    for label in ('image2text_R5', 'image2text_R10', 'text2image_R5', 'text2image_R10',
                  'image_representation', 'global_pool', 'n_images', 'n_captions', 'protocol'):
        if label in inner:
            result[label] = inner[label]
    return result, None


def read_urban_result(path):
    """Read the Urban-1k R@1 pair from whichever layout the frozen evaluator produced.

    The evaluator nests its metrics as ``urban1k.image2text.R1`` / ``urban1k.text2image.R1``; older
    or alternative layouts put flat ``image2text_R1`` / ``text2image_R1`` keys at the top level, so
    both are read and neither is assumed. Nothing is computed here.
    """
    payload, error = read_json(path)
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, 'the Urban-1k result at %s is not a dict' % path
    result = {'source': path, 'raw_keys': sorted(str(key) for key in payload)}

    def first_number(*candidates):
        for candidate in candidates:
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
        return None

    nested = payload.get('urban1k') if isinstance(payload.get('urban1k'), dict) else {}
    image2text = nested.get('image2text') if isinstance(nested.get('image2text'), dict) else {}
    text2image = nested.get('text2image') if isinstance(nested.get('text2image'), dict) else {}
    i2t = first_number(image2text.get('R1'), payload.get('image2text_R1'), payload.get('i2t_r1'),
                       payload.get('i2t'), payload.get('i2t_R1'), payload.get('image_to_text_R1'))
    t2i = first_number(text2image.get('R1'), payload.get('text2image_R1'), payload.get('t2i_r1'),
                       payload.get('t2i'), payload.get('t2i_R1'), payload.get('text_to_image_R1'))
    if i2t is None or t2i is None:
        return None, ('no image-to-text / text-to-image R@1 pair has been found in %s (top-level '
                      'keys: %s)' % (path, result['raw_keys']))
    result['image2text_R1'] = i2t
    result['text2image_R1'] = t2i
    result['image2text_R5'] = first_number(image2text.get('R5'), payload.get('image2text_R5'))
    result['image2text_R10'] = first_number(image2text.get('R10'), payload.get('image2text_R10'))
    result['text2image_R5'] = first_number(text2image.get('R5'), payload.get('text2image_R5'))
    result['text2image_R10'] = first_number(text2image.get('R10'), payload.get('text2image_R10'))
    for label, source in (('n_images', nested.get('n_images')),
                          ('n_captions', nested.get('n_captions')),
                          ('n_queries', nested.get('n_queries')),
                          ('protocol', nested.get('protocol')),
                          ('checkpoint_sha256', payload.get('checkpoint_sha256')),
                          ('completed_steps', payload.get('completed_steps'))):
        if source is not None:
            result[label] = source
    result['reading'] = ('Urban-1k is reported separately from the COCO gate and is never combined '
                         'with it into one verdict')
    return result, None


# --------------------------------------------------------------------------- report
def build_conclusion(arm, run_dir, paths, steps, report_status):
    """The frozen gate on the UNROUNDED numbers, plus the arm difference and its resolution.

    Nothing here is invented: every number is read from a named file, every threshold is a module
    constant, and the file each number came from is recorded beside it.
    """
    coco_result, coco_error = read_coco_result(paths['canonical_out'])
    urban_result, urban_error = read_urban_result(paths['urban_out'])
    problems = []
    if coco_error:
        problems.append('COCO result unavailable: %s' % coco_error)
    if urban_error:
        problems.append('Urban-1k result unavailable: %s' % urban_error)

    gate = {'rule': ('COCO I2T R@1 >= %r AND T2I R@1 >= %r AND (I2T R@1 > %r OR T2I R@1 > %r)'
                     % (S0_AT_500_COCO_I2T_R1, S0_AT_500_COCO_T2I_R1, S0_AT_500_COCO_I2T_R1,
                        S0_AT_500_COCO_T2I_R1)),
            'frozen_s0_at_500': {'coco_image2text_R1': S0_AT_500_COCO_I2T_R1,
                                 'coco_text2image_R1': S0_AT_500_COCO_T2I_R1},
            'rounded_or_approximated': False,
            'source': paths['canonical_out']}
    i2t = coco_result.get('image2text_R1') if coco_result else None
    t2i = coco_result.get('text2image_R1') if coco_result else None
    if isinstance(i2t, (int, float)) and isinstance(t2i, (int, float)):
        i2t, t2i = float(i2t), float(t2i)
        comparisons = {'coco_image2text_R1_at_least_frozen': i2t >= S0_AT_500_COCO_I2T_R1,
                       'coco_text2image_R1_at_least_frozen': t2i >= S0_AT_500_COCO_T2I_R1,
                       'at_least_one_strictly_greater': (i2t > S0_AT_500_COCO_I2T_R1
                                                         or t2i > S0_AT_500_COCO_T2I_R1)}
        passed = all(comparisons.values())
        verdict = 'PASSED_AT_500' if passed else 'FAILED_AT_500'
        gate.update({'observed': {'coco_image2text_R1': i2t, 'coco_text2image_R1': t2i},
                     'comparisons': comparisons})
    else:
        verdict = 'FAILED_AT_500'
        gate.update({'observed': {'coco_image2text_R1': i2t, 'coco_text2image_R1': t2i},
                     'comparisons': None,
                     'verdict_reason': 'the COCO R@1 pair could not be read, so nothing passed the '
                                       'gate'})

    peer_arm = ARM_MASK if arm == ARM_NATIVE else ARM_NATIVE
    peer_dir = os.path.join(os.path.dirname(run_dir),
                            ARM_RUN_IDS[peer_arm])
    peer_coco = os.path.join(peer_dir, 'evaluation',
                             '%s_step%06d_canonical.json' % (peer_arm, steps))
    peer_urban = os.path.join(peer_dir, 'evaluation',
                              '%s_step%06d_urban1k.json' % (peer_arm, steps))
    peer = {'peer_arm': peer_arm, 'peer_run_dir': peer_dir, 'peer_run_dir_exists':
            os.path.isdir(peer_dir)}
    if not peer['peer_run_dir_exists']:
        peer['mask_minus_native'] = None
        peer['reason'] = ('the peer run directory %s does not exist (the two arms run strictly '
                          'sequentially, so the difference is only computable once both have been '
                          'evaluated)' % peer_dir)
    else:
        peer_coco_result, peer_coco_error = read_coco_result(peer_coco)
        peer_urban_result, peer_urban_error = read_urban_result(peer_urban)
        peer['peer_coco_source'] = peer_coco
        peer['peer_urban_source'] = peer_urban
        peer['peer_coco_error'] = peer_coco_error
        peer['peer_urban_error'] = peer_urban_error
        this = {'coco_image2text_R1': i2t, 'coco_text2image_R1': t2i,
                'urban_image2text_R1': urban_result.get('image2text_R1') if urban_result else None,
                'urban_text2image_R1': urban_result.get('text2image_R1') if urban_result else None}
        other = {'coco_image2text_R1': peer_coco_result.get('image2text_R1')
                 if peer_coco_result else None,
                 'coco_text2image_R1': peer_coco_result.get('text2image_R1')
                 if peer_coco_result else None,
                 'urban_image2text_R1': peer_urban_result.get('image2text_R1')
                 if peer_urban_result else None,
                 'urban_text2image_R1': peer_urban_result.get('text2image_R1')
                 if peer_urban_result else None}
        if arm == ARM_MASK:
            native, mask = other, this
        else:
            native, mask = this, other
        differences = {}
        for label in sorted(this):
            if isinstance(native.get(label), (int, float)) and isinstance(mask.get(label), (int, float)):
                differences[label] = float(mask[label]) - float(native[label])
            else:
                differences[label] = None
        peer['native'] = native
        peer['mask'] = mask
        peer['mask_minus_native'] = differences
        peer['sign_convention'] = 'mask - native: positive means the suffix-mask arm scored higher'
        peer['resolution'] = {
            'coco_image_query_pp': QUERY_EQUIVALENCE_PP['coco_image_query'],
            'coco_text_query_pp': QUERY_EQUIVALENCE_PP['coco_text_query'],
            'urban1k_query_pp': QUERY_EQUIVALENCE_PP['urban1k_query'],
            'coco_image2text_R1_delta_pp': _pp(differences.get('coco_image2text_R1'),
                                               QUERY_EQUIVALENCE_PP['coco_image_query']),
            'coco_text2image_R1_delta_pp': _pp(differences.get('coco_text2image_R1'),
                                               QUERY_EQUIVALENCE_PP['coco_text_query']),
            'urban_image2text_R1_delta_pp': _pp(differences.get('urban_image2text_R1'),
                                                QUERY_EQUIVALENCE_PP['urban1k_query']),
            'reading': ('a difference smaller than one query of the frozen cohort is not a measured '
                        'difference, it is rounding: 1 COCO image query = 0.02 pp, 1 COCO text '
                        'query = 0.004 pp, 1 Urban-1k query = 0.1 pp'),
        }
        if differences.get('coco_image2text_R1') is None:
            peer['reason'] = ('the peer COCO R@1 could not be read: %s' % peer_coco_error)

    return {
        'arm': arm,
        'objective': OBJECTIVE,
        'phase': PHASE,
        'run_id': report_status.get('run_id'),
        'run_dir': run_dir,
        'completed_steps': steps,
        'created_at_iso': iso(time.time()),
        'gate': gate,
        'verdict': verdict,
        'gate_verdict': verdict,
        'coco': coco_result or {'source': paths['canonical_out'], 'error': coco_error},
        'urban': urban_result or {'source': paths['urban_out'], 'error': urban_error},
        'arm_comparison': peer,
        'query_equivalences': {
            'one_coco_image_query_pp': QUERY_EQUIVALENCE_PP['coco_image_query'],
            'one_coco_text_query_pp': QUERY_EQUIVALENCE_PP['coco_text_query'],
            'one_urban1k_query_pp': QUERY_EQUIVALENCE_PP['urban1k_query'],
            'statement': ('1 COCO image query = 0.02 pp, 1 COCO text query = 0.004 pp, 1 Urban-1k '
                          'query = 0.1 pp of R@1; every difference below these values is resolution, '
                          'not a result'),
        },
        'sources': {'checkpoint': report_status.get('checkpoint'),
                    'student': report_status.get('student'),
                    'coco_result': paths['canonical_out'],
                    'urban_result': paths['urban_out'],
                    'salu_log': os.path.join(run_dir, SALU_LOG_NAME),
                    'run_status': os.path.join(run_dir, 'run_status.json')},
        'problems': problems,
        'not_run': ['any training beyond the frozen 500 updates', 'any sweep, grid or extra arm',
                    'any evaluation of a checkpoint other than the verified step-500 student'],
    }


def _pp(delta, per_query_pp):
    """A difference expressed in percentage points and in whole queries of the frozen cohort."""
    if not isinstance(delta, (int, float)) or not per_query_pp:
        return None
    return {'delta_pp': delta * 100.0, 'delta_queries': delta / (per_query_pp / 100.0)}


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
    parser.add_argument('--arms', default=','.join(ARM_ORDER),
                        help='comma-separated subset of %s; the arms are executed STRICTLY '
                             'SEQUENTIALLY, one full pipeline at a time' % ','.join(ARM_ORDER))
    parser.add_argument('--run-root', default=DEFAULT_RUN_ROOT,
                        help='parent directory holding <%s>/<run id> for every arm' % RUNS_SUBDIR)
    parser.add_argument('--init-state', default=SHARED_INIT_DEFAULT,
                        help='the shared frozen CV-SSL initialisation, the only allowed start point')
    parser.add_argument('--stages', default=','.join(STAGES),
                        help='comma-separated subset of: %s (always executed in this order, per arm)'
                             % ','.join(STAGES))
    parser.add_argument('--resume-from', default=None, choices=STAGES,
                        help='start every arm at this stage and continue through the end of '
                             '--stages; this is the only way to re-run export / eval / report '
                             'without touching training')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the exact per-arm stage commands and exit without executing '
                             'anything (no lock is taken, nothing is created)')
    parser.add_argument('--detach', action='store_true',
                        help='re-execute detached (start_new_session=True, stdin=/dev/null) so the '
                             'run survives the calling shell; the PID is recorded in the lock')
    parser.add_argument('--internal-detached-child', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--python', default=PYTHON_DEFAULT)
    parser.add_argument('--torchrun', default=TORCHRUN_DEFAULT,
                        help='the torchrun launcher; a bare name means "the torchrun next to '
                             '--python", PATH is never trusted')
    parser.add_argument('--nproc', type=int, default=LOCKED['nproc'])
    parser.add_argument('--steps', type=int, default=LOCKED['steps'])
    parser.add_argument('--save-steps', dest='save_steps', default=LOCKED['save_steps'])
    parser.add_argument('--base-model', dest='base_model', default=LOCKED['base_model'])
    parser.add_argument('--epochs', type=int, default=LOCKED['epochs'])
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=LOCKED['batch_size'])
    parser.add_argument('--lr', type=float, default=LOCKED['lr'])
    parser.add_argument('--mask-lr', dest='mask_lr', type=float, default=LOCKED['mask_lr'])
    parser.add_argument('--suffix-lr', dest='suffix_lr', type=float, default=LOCKED['suffix_lr'])
    parser.add_argument('--weight-decay', dest='weight_decay', type=float,
                        default=LOCKED['weight_decay'])
    parser.add_argument('--warmup-length', dest='warmup_length', type=int,
                        default=LOCKED['warmup_length'])
    parser.add_argument('--seed', type=int, default=LOCKED['seed'])
    parser.add_argument('--lambda-suffix', dest='lambda_suffix', type=float,
                        default=LOCKED['lambda_suffix'])
    parser.add_argument('--log-every', dest='log_every', type=int, default=LOCKED['log_every'])
    parser.add_argument('--heavy-log-every', dest='heavy_log_every', type=int,
                        default=LOCKED['heavy_log_every'])
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=LOCKED['num_workers'])
    parser.add_argument('--amp-dtype', dest='amp_dtype', default=LOCKED['amp_dtype'])
    parser.add_argument('--image-chunk', dest='image_chunk', type=int, default=LOCKED['image_chunk'])
    parser.add_argument('--text-chunk', dest='text_chunk', type=int, default=LOCKED['text_chunk'])
    parser.add_argument('--suffix-checkpoint', dest='suffix_checkpoint', type=int,
                        default=LOCKED['suffix_checkpoint'])
    parser.add_argument('--master-port', dest='master_port', type=int,
                        default=LOCKED['master_port'])
    parser.add_argument('--lock-file', default=None,
                        help='override the lock path (default <run dir>/runner.lock and '
                             '<run root>/suffix_runner.lock)')
    parser.add_argument('--clear-stale-lock', action='store_true',
                        help='replace a lock whose recorded PID is gone; a live holder is never '
                             'overridden')
    args = parser.parse_args()

    # the frozen-configuration guard can only work if every locked name really is an option
    registered = {option.lstrip('-').replace('-', '_') for action in parser._actions
                  for option in action.option_strings}
    unregistered = sorted(name for name in LOCKED if name not in registered)
    if unregistered:
        raise SystemExit('runner defect: locked option(s) %r are not registered on the parser'
                         % unregistered)

    arms = [arm.strip() for arm in args.arms.split(',') if arm.strip()]
    unknown_arms = [arm for arm in arms if arm not in ARM_ORDER]
    if unknown_arms:
        parser.error('unknown arm(s) %r; the only arms are %s' % (unknown_arms, ','.join(ARM_ORDER)))
    # the arms always run in the frozen order, whatever order they were listed in
    arms = [arm for arm in ARM_ORDER if arm in arms]
    if not arms:
        parser.error('no arm selected')
    if len(set(arms)) != len(arms):
        parser.error('duplicate arm in --arms')

    stages = [stage.strip() for stage in args.stages.split(',') if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        parser.error('unknown stage(s) %r; the only stages are %s' % (unknown, ','.join(STAGES)))
    stages = [stage for stage in STAGES if stage in stages]
    # the stages the operator actually asked for, before --resume-from trims the head: this is what
    # decides whether an existing student export is re-exported or reused
    requested_stages = list(stages)
    if args.resume_from:
        stages = [stage for stage in stages if STAGES.index(stage) >= STAGES.index(args.resume_from)]
    if not stages:
        parser.error('no stage selected')

    for name in LOCKED:
        same, value, want = locked_value(args, name)
        if not same:
            parser.error('%s=%r is not this experiment: S0-Suffix v0.1 is ONE configuration per arm '
                         'with exactly %d updates (%s=%r). Sweeps, extra arms and epoch extension '
                         'are not available from this runner.'
                         % (name, value, LOCKED['steps'], name, want))

    run_root = os.path.abspath(args.run_root)
    runs_dir = os.path.join(run_root, RUNS_SUBDIR)
    run_dirs = {arm: os.path.join(runs_dir, ARM_RUN_IDS[arm]) for arm in ARM_ORDER}
    runner_status_path = os.path.join(run_root, 'suffix_runner_status.json')
    runner_log = os.path.join(run_root, 'logs', 'runner.log')
    lock_path_default = args.lock_file or os.path.join(run_root, 'suffix_runner.lock')

    # ---- resolve the launcher before anything else ---------------------------
    resolution_error = None
    try:
        args.torchrun = resolve_torchrun(args.python, args.torchrun)
    except SystemExit as error:
        resolution_error = str(error)
    if resolution_error is not None and not args.dry_run:
        print('PREFLIGHT_REFUSAL %s' % resolution_error, file=sys.stderr)
        raise SystemExit(PREFLIGHT_EXIT_CODE)

    built = {}
    for arm in ARM_ORDER:
        commands, paths = build_commands(args, run_dirs[arm], arm)
        built[arm] = {'commands': commands, 'paths': paths, 'run_dir': run_dirs[arm]}
    missing_inputs, checked_inputs = required_inputs(args, {arm: built[arm]['commands']
                                                            for arm in ARM_ORDER})

    # ---- dry run: print the exact command lines, execute nothing -------------
    if args.dry_run:
        print('DRY_RUN experiment=%s objective=%s repo=%s' % (PHASE, OBJECTIVE, REPO))
        print('DRY_RUN run_root=%s runs_dir=%s stages=%s arms=%s resume_from=%s'
              % (run_root, runs_dir, ','.join(stages), ','.join(arms), args.resume_from))
        print('DRY_RUN python=%s torchrun=%s init_state=%s' % (args.python, args.torchrun,
                                                               args.init_state))
        print('DRY_RUN lock_file=%s runner_status=%s runner_log=%s'
              % (lock_path_default, runner_status_path, runner_log))
        print('DRY_RUN stages run strictly sequentially, one arm at a time; no second torchrun is '
              'started while one arm is alive')
        if resolution_error is not None:
            print('DRY_RUN WARNING launcher not resolvable on this machine (nothing was executed): '
                  '%s' % resolution_error)
        for entry in checked_inputs:
            print('DRY_RUN input exists=%s %s (%s)'
                  % (entry['exists'], entry['path'], entry['description']))
        for entry in missing_inputs:
            print('DRY_RUN WARNING MISSING %s (%s) -- an execution would refuse with exit code %d'
                  % (entry['path'], entry['description'], PREFLIGHT_EXIT_CODE))
        for arm in arms:
            spec = built[arm]
            print('DRY_RUN arm=%s run_dir=%s' % (arm, spec['run_dir']))
            if 'train' in stages:
                print('DRY_RUN arm=%s existing step-%d checkpoints=%r (a run that already owns one '
                      'is never retrained)' % (arm, args.steps,
                                               discover_step500_checkpoints(spec['run_dir'])))
            for stage in stages:
                if stage in ('verify', 'report'):
                    print('DRY_RUN arm=%s stage=%s (in-process, no subprocess): %s'
                          % (arm, stage,
                             'verify the step-%d checkpoint %s'
                             % (args.steps, spec['paths']['checkpoint']) if stage == 'verify'
                             else 'assemble %s and %s'
                             % (os.path.join(spec['run_dir'], 'run_status.json'),
                                os.path.join(spec['run_dir'], 'conclusion.json'))))
                    continue
                argv = spec['commands'][stage]
                if stage in ('coco', 'urban'):
                    argv = substitute_student(argv, spec['paths']['student'])
                print('DRY_RUN arm=%s stage=%s log=%s' % (arm, stage,
                                                          os.path.join(spec['run_dir'], 'logs',
                                                                       '%s.log' % stage)))
                print('DRY_RUN $ %s' % ' '.join(sanitise_argv(argv)))
        print('DRY_RUN nothing was executed')
        return 0

    # ---- detached re-execution ---------------------------------------------
    if args.detach and not args.internal_detached_child:
        os.makedirs(os.path.dirname(runner_log), exist_ok=True)
        argv = [sys.executable] + [token for token in sys.argv[1:] if token != '--detach'] \
            + ['--internal-detached-child']
        master_log = os.path.join(os.path.dirname(runner_log), 'runner_master.log')
        with open(master_log, 'ab') as handle, open(os.devnull, 'rb') as devnull:
            handle.write(('\n# detached launch %s\n' % iso(time.time())).encode('utf-8'))
            handle.flush()
            process = subprocess.Popen(argv, stdin=devnull, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
        print('DETACHED pid=%d master_log=%s (the run continues without this shell)'
              % (process.pid, master_log), flush=True)
        return 0

    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(runner_log), exist_ok=True)
    env = training_environment()
    started = time.time()
    attempt_history = []
    runner_exit = 0
    arm_results = {}

    def note(message):
        with open(runner_log, 'a', encoding='utf-8') as handle:
            handle.write('[%s] %s\n' % (iso(time.time()), message))
        print(message, flush=True)

    def update_runner(**fields):
        write_status(runner_status_path, run_id=PHASE, objective=OBJECTIVE, run_root=run_root,
                     runs_dir=runs_dir, repo=REPO, arms={arm: run_dirs[arm] for arm in arms},
                     arm_run_ids=dict(ARM_RUN_IDS), stages=stages, resume_from=args.resume_from,
                     pid=os.getpid(), lock_file=lock_path_default, runner_log=runner_log,
                     implementation_sha=git_head(REPO), **fields)

    try:
        update_runner(phase='starting', started_at=started, started_at_iso=iso(started),
                      attempt_history=attempt_history[-5:])

        # ---- pre-flight: missing inputs, data roots, retrain refusal ------------
        if missing_inputs:
            detail = '; '.join('%s (%s)' % (entry['path'], entry['description'])
                               for entry in missing_inputs)
            update_runner(phase='failed', finished_at=time.time(),
                          wall_seconds=time.time() - started,
                          failure_reason='required input(s) are absent from this worktree: %s'
                                         % detail,
                          required_inputs=checked_inputs)
            note('PREFLIGHT_REFUSAL missing required input(s): %s' % detail)
            raise SystemExit(PREFLIGHT_EXIT_CODE)
        try:
            annotations, coco_images = verify_data_environment()
        except SystemExit as error:
            update_runner(phase='failed', finished_at=time.time(),
                          wall_seconds=time.time() - started,
                          failure_reason='data pre-check refused to start: %s' % error)
            note('PREFLIGHT_REFUSAL data pre-check: %s' % error)
            raise SystemExit(PREFLIGHT_EXIT_CODE)
        update_runner(coco_annotations=annotations, coco_images=coco_images,
                      urban_root=URBAN_ROOT, required_inputs=checked_inputs,
                      init_state=args.init_state)

        # a run that already owns a step-500 checkpoint is NEVER retrained, not even in the same
        # invocation: the refusal happens before the lock of that arm is taken
        retrain_offenders = {}
        if 'train' in stages:
            for arm in arms:
                found = discover_step500_checkpoints(run_dirs[arm])
                if found:
                    retrain_offenders[arm] = found
            if retrain_offenders:
                update_runner(phase='failed', finished_at=time.time(),
                              wall_seconds=time.time() - started,
                              failure_reason='REFUSING to retrain: a step-%d checkpoint already '
                                             'exists under %r; re-run only export/eval/report with '
                                             '--resume-from export (or remove the run directory '
                                             'explicitly if a new training run is really wanted)'
                                             % (args.steps, retrain_offenders),
                              retrain_offenders=retrain_offenders)
                note('PREFLIGHT_REFUSAL refusing to retrain: %s' % retrain_offenders)
                raise SystemExit(PREFLIGHT_EXIT_CODE)

        for arm in arms:
            spec = built[arm]
            run_dir = spec['run_dir']
            commands, paths = spec['commands'], spec['paths']
            log_dir = os.path.join(run_dir, 'logs')
            status_path = os.path.join(run_dir, 'run_status.json')
            lock_path = resolve_lock_path(args.lock_file, run_dir, arm)
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            arm_started = time.time()

            update_runner(phase='arm_starting', active_arm=arm,
                          **{'%s_phase' % arm.lower(): 'starting'})
            try:
                lock = acquire_lock(lock_path, clear_stale=args.clear_stale_lock)
            except SystemExit as error:
                write_status(status_path, run_id=ARM_RUN_IDS[arm], arm=arm, objective=OBJECTIVE,
                             phase_name=PHASE, run_dir=run_dir, repo=REPO, phase='failed',
                             failure_reason=str(error), stages=stages)
                arm_results[arm] = {'exit_code': PREFLIGHT_EXIT_CODE, 'phase': 'refused',
                                    'failure_reason': str(error)}
                update_runner(phase='failed', active_arm=arm, arms_state=arm_results,
                              finished_at=time.time(), wall_seconds=time.time() - started,
                              failure_reason='arm %s refused to start: %s' % (arm, error))
                note('PREFLIGHT_REFUSAL arm %s: %s' % (arm, error))
                runner_exit = PREFLIGHT_EXIT_CODE
                break

            try:
                previous = read_status(status_path)
                history = list(previous.get('attempt_history') or [])
                if previous.get('phase') in ('failed', 'complete'):
                    history.append({'phase': previous.get('phase'),
                                    'finished_at': previous.get('finished_at'),
                                    'exit_codes': previous.get('exit_codes'),
                                    'failure_reason': previous.get('failure_reason')})
                stage_state = dict(previous.get('stage_state') or {})
                exit_codes = dict(previous.get('exit_codes') or {})
                attempt = int(previous.get('attempt') or 0) + 1
                update_lock(lock_path, run_id=ARM_RUN_IDS[arm], arm=arm, objective=OBJECTIVE,
                            run_dir=run_dir, stages=stages, resume_from=args.resume_from,
                            argv=sanitise_argv(sys.argv), git_head=git_head(REPO),
                            python=args.python, torchrun=args.torchrun, detached=args.detach,
                            commands={name: sanitise_argv(argv) for name, argv in commands.items()})
                write_status(status_path, run_id=ARM_RUN_IDS[arm], arm=arm, objective=OBJECTIVE,
                             phase_name=PHASE, run_dir=run_dir, repo=REPO, pid=os.getpid(),
                             started_at=arm_started, started_at_iso=iso(arm_started),
                             stages=stages, resume_from=args.resume_from, phase='starting',
                             max_steps=args.steps, save_steps=args.save_steps,
                             batch_size=args.batch_size, world_size=args.nproc,
                             clip_lr=args.lr, mask_lr=args.mask_lr, suffix_lr=args.suffix_lr,
                             weight_decay=args.weight_decay, warmup_length=args.warmup_length,
                             seed=args.seed, epochs=args.epochs, base_model=args.base_model,
                             lambda_suffix=args.lambda_suffix, init_state=args.init_state,
                             attempt=attempt, attempt_history=history[-5:], lock_file=lock_path,
                             run_log_path=os.path.join(run_dir, SALU_LOG_NAME),
                             commands={name: sanitise_argv(argv) for name, argv in commands.items()},
                             **{field: None for field in ATTEMPT_FIELDS})
                note('arm %s start pid=%d stages=%s run_dir=%s' % (arm, os.getpid(),
                                                                   ','.join(stages), run_dir))

                # ---- stage 1: training -----------------------------------------
                if 'train' in stages:
                    ok, detail, gpu = gpu_preflight()
                    write_status(status_path, gpu_check=detail, gpu_ok=bool(ok), gpu=gpu)
                    update_runner(active_arm=arm, phase='training',
                                  **{'%s_gpu_check' % arm.lower(): detail,
                                     '%s_gpu' % arm.lower(): gpu})
                    if not ok:
                        write_status(status_path, phase='failed',
                                     failure_reason='GPU pre-check refused to start: %s' % detail)
                        note('PREFLIGHT_REFUSAL arm %s GPU pre-check: %s' % (arm, detail))
                        raise SystemExit(PREFLIGHT_EXIT_CODE)
                    if not os.path.isfile(args.init_state):
                        write_status(status_path, phase='failed',
                                     failure_reason='shared init missing: %s' % args.init_state)
                        note('PREFLIGHT_REFUSAL arm %s shared init missing: %s'
                             % (arm, args.init_state))
                        raise SystemExit(PREFLIGHT_EXIT_CODE)
                    write_status(status_path, phase='training', stage='train',
                                 init_file_sha256=sha256_of(args.init_state),
                                 train_checkpoint_target=paths['checkpoint'])
                    code, record = run_stage('train', commands['train'], log_dir, stage_state, env)
                    exit_codes['train'] = code
                    stage_state['train'] = dict(stage_state.get('train') or {}, **record)
                    write_status(status_path, train_exit_code=code, exit_codes=exit_codes,
                                 stage_state=stage_state, completed_steps=_completed_steps(run_dir))
                    if code != 0:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='training exited %d; no verification, export or '
                                                    'evaluation attempted' % code)
                        note('STAGE_FAILED arm %s train exit=%d; pipeline stopped' % (arm, code))
                        raise SystemExit(STAGE_EXIT_CODE['train'])
                    note('arm %s stage train ok in %.1fs' % (arm, record['wall_seconds']))

                # ---- stage 2: checkpoint verification ---------------------------
                if 'verify' in stages:
                    write_status(status_path, phase='verifying', stage='verify')
                    started_verify = time.time()
                    found, how, alternatives = discover_checkpoint(run_dir, arm, args.steps)
                    if found is None:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='the step-%d checkpoint was not found under %s '
                                                    '(looked for %s)'
                                                    % (args.steps, run_dir,
                                                       CHECKPOINT_TEMPLATE % (arm, args.steps)))
                        note('STAGE_FAILED arm %s verify: no step-%d checkpoint under %s'
                             % (arm, args.steps, run_dir))
                        raise SystemExit(STAGE_EXIT_CODE['verify'])
                    checkpoint = found
                    note('arm %s verify checkpoint=%s (found via %s)' % (arm, checkpoint, how))
                    # the native CLIP key set is an INDEPENDENT cross-check of the clip state: it is
                    # looked up before the verification so the checkpoint is loaded exactly once
                    reference = read_status(status_path).get('native_key_set')
                    if not (isinstance(reference, dict) and reference.get('keys')):
                        reference = native_key_set_reference(args.base_model)
                        write_status(status_path, native_key_set=reference)
                        if not reference.get('keys'):
                            note('arm %s verify: the native key-set cross-check is unavailable (%s); '
                                 'the remaining checks still run' % (arm,
                                                                     reference.get('unavailable')))
                    try:
                        verify = verify_checkpoint(checkpoint, arm, args.steps,
                                                   salu_log_path(run_dir), native_key_set=reference)
                    except (OSError, RuntimeError, ValueError, EOFError) as error:
                        verify = {'path': checkpoint, 'arm': arm,
                                  'problems': ['the checkpoint could not be loaded: %s' % error],
                                  'verified': False}
                    except Exception as error:                                  # noqa: BLE001
                        verify = {'path': checkpoint, 'arm': arm,
                                  'problems': ['the checkpoint could not be read with torch: %s'
                                               % error],
                                  'verified': False}
                    verify['verified'] = not verify.get('problems')
                    finished_verify = time.time()
                    stage_state['verify'] = dict(stage_state.get('verify') or {}, **{
                        'attempts': int((stage_state.get('verify') or {}).get('attempts') or 0) + 1,
                        'started_at': started_verify, 'started_at_iso': iso(started_verify),
                        'finished_at': finished_verify, 'finished_at_iso': iso(finished_verify),
                        'wall_seconds': finished_verify - started_verify,
                        'checkpoint': checkpoint, 'discovered_via': how,
                        'exit_code': 0 if verify.get('verified')
                        else STAGE_EXIT_CODE['verify']})
                    exit_codes['verify'] = 0 if verify.get('verified') \
                        else STAGE_EXIT_CODE['verify']
                    write_status(status_path, checkpoint=checkpoint,
                                 checkpoint_discovered_via=how,
                                 checkpoint_alternatives=alternatives,
                                 checkpoint_verified=bool(verify.get('verified')),
                                 checkpoint_problems=verify.get('problems'),
                                 checkpoint_report=verify,
                                 completed_steps=verify.get('completed_steps'),
                                 suffix_mask_state_tensors=verify.get('suffix_mask_state_tensors'),
                                 suffix_mask_tensor_names=list(SUFFIX_MASK_TENSOR_NAMES),
                                 recorded_digests=verify.get('recorded_digests'),
                                 recomputed_digests=verify.get('recomputed_digests'),
                                 run_log_presentations=verify.get('run_log'),
                                 exit_codes=exit_codes, stage_state=stage_state)
                    update_runner(active_arm=arm, phase='verifying',
                                  **{'%s_checkpoint' % arm.lower(): checkpoint,
                                     '%s_checkpoint_verified' % arm.lower():
                                     bool(verify.get('verified'))})
                    if not verify.get('verified'):
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='checkpoint verification failed (hard failure, '
                                                    'no bypass): %s' % verify.get('problems'))
                        note('STAGE_FAILED arm %s verify: %s' % (arm, verify.get('problems')))
                        raise SystemExit(STAGE_EXIT_CODE['verify'])
                    note('arm %s stage verify ok in %.1fs' % (arm, finished_verify - started_verify))
                else:
                    found, how, alternatives = discover_checkpoint(run_dir, arm, args.steps)
                    if found is None:
                        write_status(status_path, phase='failed',
                                     failure_reason='the step-%d checkpoint was not found under %s'
                                                    % (args.steps, run_dir))
                        note('PREFLIGHT_REFUSAL arm %s no step-%d checkpoint under %s'
                             % (arm, args.steps, run_dir))
                        raise SystemExit(PREFLIGHT_EXIT_CODE)
                    checkpoint = found

                # ---- stage 3: bare student export -------------------------------
                if 'export' in stages:
                    write_status(status_path, phase='exporting', stage='export')
                    if os.path.isfile(paths['student']) and 'export' not in requested_stages:
                        # the operator resumed at a later stage: the existing student is the artifact
                        # of the verified checkpoint and is reused, never silently rewritten
                        note('arm %s export: %s already exists and --stages did not ask for an '
                             'export; the export stage is skipped, use --stages export (or '
                             '--resume-from export) to re-export it' % (arm, paths['student']))
                    else:
                        code, record = run_stage('export', commands['export'], log_dir, stage_state,
                                                 env)
                        exit_codes['export'] = code
                        stage_state['export'] = dict(stage_state.get('export') or {}, **record)
                        write_status(status_path, export_exit_code=code, exit_codes=exit_codes,
                                     stage_state=stage_state, student_export_dir=paths['student_dir'])
                        if code != 0:
                            write_status(status_path, phase='failed', finished_at=time.time(),
                                         wall_seconds=time.time() - started,
                                         failure_reason='student export exited %d; no evaluation '
                                                        'attempted' % code)
                            note('STAGE_FAILED arm %s export exit=%d; pipeline stopped' % (arm, code))
                            raise SystemExit(STAGE_EXIT_CODE['export'])
                        note('arm %s stage export ok in %.1fs' % (arm, record['wall_seconds']))
                    student, extra_students = discover_student(paths['student_dir'])
                    if student is None:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='the export stage produced no *.pt/*.pth file '
                                                    'under %s' % paths['student_dir'])
                        note('STAGE_FAILED arm %s export produced no student file under %s'
                             % (arm, paths['student_dir']))
                        raise SystemExit(STAGE_EXIT_CODE['export'])
                    if extra_students:
                        note('arm %s export produced extra file(s) %r; the student used is %s'
                             % (arm, extra_students, student))
                    write_status(status_path, student=student, student_sha256=sha256_of(student),
                                 student_files=sorted([student] + list(extra_students)))
                    update_runner(active_arm=arm, phase='exporting',
                                  **{'%s_student' % arm.lower(): student})
                    note('arm %s stage export ok, student=%s' % (arm, student))

                # ---- stage 4: canonical COCO retrieval --------------------------
                if 'coco' in stages:
                    os.makedirs(os.path.dirname(paths['canonical_out']), exist_ok=True)
                    student, _extra = discover_student(paths['student_dir'])
                    if student is None:
                        write_status(status_path, phase='failed',
                                     failure_reason='no exported student to evaluate under %s'
                                                    % paths['student_dir'])
                        note('PREFLIGHT_REFUSAL arm %s no exported student under %s'
                             % (arm, paths['student_dir']))
                        raise SystemExit(PREFLIGHT_EXIT_CODE)
                    coco_argv = substitute_student(commands['coco'], student)
                    if any(STUDENT_PLACEHOLDER in token for token in coco_argv):
                        raise SystemExit('the COCO command still holds an unresolved student '
                                         'placeholder')
                    write_status(status_path, phase='evaluating_coco', stage='coco', student=student,
                                 commands=dict(read_status(status_path).get('commands') or {},
                                               coco=sanitise_argv(coco_argv)))
                    code, record = run_stage('coco', coco_argv, log_dir, stage_state, env)
                    exit_codes['coco'] = code
                    stage_state['coco'] = dict(stage_state.get('coco') or {}, **record)
                    write_status(status_path, coco_exit_code=code, exit_codes=exit_codes,
                                 stage_state=stage_state, coco_output=paths['canonical_out'])
                    if code != 0:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='COCO canonical evaluation exited %d' % code)
                        note('STAGE_FAILED arm %s coco exit=%d; pipeline stopped' % (arm, code))
                        raise SystemExit(STAGE_EXIT_CODE['coco'])
                    note('arm %s stage coco ok in %.1fs' % (arm, record['wall_seconds']))

                # ---- stage 5: Urban-1k retrieval --------------------------------
                if 'urban' in stages:
                    os.makedirs(os.path.dirname(paths['urban_out']), exist_ok=True)
                    student, _extra = discover_student(paths['student_dir'])
                    if student is None:
                        write_status(status_path, phase='failed',
                                     failure_reason='no exported student to evaluate under %s'
                                                    % paths['student_dir'])
                        note('PREFLIGHT_REFUSAL arm %s no exported student under %s'
                             % (arm, paths['student_dir']))
                        raise SystemExit(PREFLIGHT_EXIT_CODE)
                    urban_argv = substitute_student(commands['urban'], student)
                    write_status(status_path, phase='evaluating_urban', stage='urban',
                                 commands=dict(read_status(status_path).get('commands') or {},
                                               urban=sanitise_argv(urban_argv)))
                    code, record = run_stage('urban', urban_argv, log_dir, stage_state, env)
                    exit_codes['urban'] = code
                    stage_state['urban'] = dict(stage_state.get('urban') or {}, **record)
                    write_status(status_path, urban_exit_code=code, exit_codes=exit_codes,
                                 stage_state=stage_state, urban_output=paths['urban_out'])
                    if code != 0:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='Urban-1k evaluation exited %d' % code)
                        note('STAGE_FAILED arm %s urban exit=%d; pipeline stopped' % (arm, code))
                        raise SystemExit(STAGE_EXIT_CODE['urban'])
                    note('arm %s stage urban ok in %.1fs' % (arm, record['wall_seconds']))

                # ---- stage 6: conclusion ----------------------------------------
                if 'report' in stages:
                    write_status(status_path, phase='reporting', stage='report')
                    started_report = time.time()
                    report_status = read_status(status_path)
                    conclusion = build_conclusion(arm, run_dir, paths, args.steps, report_status)
                    conclusion['stage_exit_codes'] = dict(exit_codes)
                    conclusion['stage_logs'] = {stage: os.path.join(log_dir, '%s.log' % stage)
                                                for stage in STAGES if stage != 'report'}
                    temporary = os.path.join(run_dir, 'conclusion.json.tmp')
                    with open(temporary, 'w', encoding='utf-8') as handle:
                        json.dump(conclusion, handle, ensure_ascii=False, indent=2, sort_keys=True,
                                  default=str)
                    os.replace(temporary, os.path.join(run_dir, 'conclusion.json'))
                    finished_report = time.time()
                    stage_state['report'] = dict(stage_state.get('report') or {}, **{
                        'attempts': int((stage_state.get('report') or {}).get('attempts') or 0) + 1,
                        'started_at': started_report, 'started_at_iso': iso(started_report),
                        'finished_at': finished_report, 'finished_at_iso': iso(finished_report),
                        'wall_seconds': finished_report - started_report,
                        'conclusion_json': os.path.join(run_dir, 'conclusion.json'),
                        'exit_code': STAGE_EXIT_CODE['report'] if conclusion['problems'] else 0})
                    exit_codes['report'] = stage_state['report']['exit_code']
                    write_status(status_path, conclusion=conclusion,
                                 conclusion_verdict=conclusion['verdict'],
                                 conclusion_json=os.path.join(run_dir, 'conclusion.json'),
                                 conclusion_problems=conclusion['problems'],
                                 report_exit_code=exit_codes['report'],
                                 exit_codes=exit_codes, stage_state=stage_state)
                    if conclusion['problems']:
                        write_status(status_path, phase='failed', finished_at=time.time(),
                                     wall_seconds=time.time() - started,
                                     failure_reason='the conclusion could not be assembled from the '
                                                    'real artifacts: %s' % conclusion['problems'])
                        note('STAGE_FAILED arm %s report: %s' % (arm, conclusion['problems']))
                        raise SystemExit(STAGE_EXIT_CODE['report'])
                    note('arm %s stage report ok, verdict=%s, wrote %s'
                         % (arm, conclusion['verdict'],
                            os.path.join(run_dir, 'conclusion.json')))

                write_status(status_path, phase='complete', exit_codes=exit_codes,
                             stage_state=stage_state, finished_at=time.time(),
                             wall_seconds=time.time() - started)
                arm_results[arm] = {'exit_code': 0, 'phase': 'complete',
                                    'exit_codes': dict(exit_codes),
                                    'conclusion_verdict': read_status(status_path)
                                    .get('conclusion_verdict'),
                                    'completed_steps': read_status(status_path)
                                    .get('completed_steps')}
                update_runner(active_arm=arm, phase='arm_complete', arms_state=arm_results,
                              **{'%s_phase' % arm.lower(): 'complete'})
                note('ARM_COMPLETE %s %s' % (arm, json.dumps(arm_results[arm], sort_keys=True,
                                                             default=str)))
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 0
                arm_results[arm] = {'exit_code': code, 'phase': 'failed',
                                    'failure_reason': read_status(status_path).get('failure_reason'),
                                    'exit_codes': dict(read_status(status_path).get('exit_codes')
                                                       or {})}
                update_runner(phase='arm_failed', active_arm=arm, arms_state=arm_results,
                              **{'%s_phase' % arm.lower(): 'failed'})
                runner_exit = code or 0
                attempt_history.append({'arm': arm, 'exit_code': runner_exit,
                                        'finished_at': time.time(),
                                        'failure_reason': arm_results[arm]['failure_reason']})
                note('arm %s stopped with exit code %d; artifacts and the status file are kept'
                     % (arm, runner_exit))
                break
            finally:
                release_lock(lock, lock_path)

        final_phase = 'complete' if runner_exit == 0 and all(
            result.get('phase') == 'complete' for result in arm_results.values()) else 'failed'
        update_runner(phase=final_phase, arms_state=arm_results, exit_codes={
            arm: result.get('exit_code') for arm, result in arm_results.items()},
            attempt_history=attempt_history[-5:], finished_at=time.time(),
            wall_seconds=time.time() - started,
            failure_reason=None if final_phase == 'complete'
            else (arm_results.get(list(arm_results)[-1], {}).get('failure_reason')
                  if arm_results else 'no arm completed'))
        note('RUNNER_%s %s' % (final_phase.upper(),
                               json.dumps({'exit_codes': {arm: result.get('exit_code')
                                                          for arm, result in arm_results.items()},
                                           'arms_state': arm_results}, sort_keys=True,
                                          default=str)))
    finally:
        # a crash still leaves a status file that says the runner stopped
        if not read_status(runner_status_path).get('phase') in ('complete', 'failed'):
            update_runner(phase='failed', failure_reason='the runner stopped without completing its '
                                                         'selected stages', finished_at=time.time(),
                          wall_seconds=time.time() - started)
    return runner_exit


if __name__ == '__main__':
    raise SystemExit(main())
