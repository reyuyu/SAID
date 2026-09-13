"""Read-only data layer for the S0-TriMask / S0-TriMask-HS training dashboard.

Transport-independent on purpose: :mod:`tools.serve_training_dashboard` is only one possible front
end (the repository's older Streamlit dashboard could import this module instead). Nothing here
imports torch, loads a checkpoint, runs a forward pass or touches the GPU: every value is read from
small files that the trainer or the runner already wrote.

Safety model:

* a run id is only ever resolved through the registry that was registered at start-up; the client
  never supplies a path, so no request can escape the registered run directories;
* only whitelisted file names inside a registered directory are opened;
* JSONL is read incrementally from a cached byte offset, a trailing partial line is left for the
  next poll, a corrupt complete line becomes a warning instead of an exception, and a shrunken or
  rotated file resets its own cursor.
"""
import json
import math
import os
import re
import time

RUN_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
# every file this module may open, relative to a registered run directory
RUN_FILES = {
    'status': 'run_status.json',
    'config': 'config.json',
    'summary': 'run_summary.json',
    'log': 'salu_log.jsonl',
    'mask_snapshot': 'mask_snapshot.json',
    'export_verification': 'export_verification.json',
}
EVALUATION_FILES = {
    'coco': 'evaluation/{name}_canonical.json',
    'urban1k': 'evaluation/{name}_urban1k.json',
}
# read-only offline diagnostics (the geometry probe): fixed directory, fixed file names, written by
# `tools/diag/trimask_hs_geometry_probe.py` and never by this service
DIAGNOSTICS_DIR = 'diagnostics'
DIAGNOSTICS_FILES = {
    'probe': 'hs_mask_geometry_probe.json',
    'manifest': 'manifest.json',
    'queries_csv': 'hs_mask_geometry_probe_queries.csv',
}
DIAGNOSTIC_LABELS = {'available': '已诊断', 'missing': '未诊断'}
# fixed reminders attached to every diagnostics response, so the page cannot present the numbers
# without them
DIAGNOSTIC_REMINDERS = [
    '坐标方差不是语义信息比例：它只描述 mask 取值在 caption 与坐标两个方向上的分布。',
    '余弦更高不等于排名更好：读出分数上升可能同时抬高正例与难负例。',
    '只读前向干预不等于重新训练后的因果结论：本轮没有产生任何新的 optimizer update。',
]
# the runner names its evaluation files "<arm>_step<NNNNNN>_<suffix>"; if the registered prefix does
# not match, the fixed suffix inside the fixed evaluation/ directory is used instead, so a run whose
# files were named by the trainer is still found without any client-supplied path
EVALUATION_SUFFIX = {'coco': '_canonical.json', 'urban1k': '_urban1k.json'}
MAX_RECORDS_PER_READ = 20000
MAX_LINES_PER_POLL = 4000
MAX_LINE_BYTES = 1 << 20
PHASES = ('not_started', 'training', 'exporting', 'evaluating_coco', 'evaluating_urban',
          'complete', 'failed', 'unknown')
PHASE_LABELS = {
    'not_started': '未开始', 'training': '训练中', 'exporting': '导出中',
    'evaluating_coco': 'COCO评估中', 'evaluating_urban': 'Urban评估中',
    'complete': '完成', 'failed': '失败', 'unknown': '未知',
}

# reference numbers used by the evaluation panel; every one of them is a measured value from a
# frozen result file, and the source is named next to it in the UI
BASELINES = {
    'S0@500': {
        'coco': {'i2t_r1': 0.6058, 'i2t_r5': 0.8220, 'i2t_r10': 0.8906,
                 't2i_r1': 0.41236, 't2i_r5': 0.67092, 't2i_r10': 0.7662},
        'urban1k': {'i2t_r1': 0.87000, 'i2t_r5': 0.97100, 'i2t_r10': 0.98800,
                    't2i_r1': 0.84200, 't2i_r5': 0.96700, 't2i_r10': 0.98100},
        'source': '/root/SAID-token-v1/docs/said_token_v1/fix_v2_results.json',
    },
    'S0_TriMask@500 (soft, v0.1)': {
        'coco': {'i2t_r1': 0.6028, 'i2t_r5': 0.8224, 'i2t_r10': 0.8902,
                 't2i_r1': 0.41444, 't2i_r5': 0.67196, 't2i_r10': 0.76644},
        'urban1k': {'i2t_r1': 0.87100, 'i2t_r5': 0.97500, 'i2t_r10': 0.98700,
                    't2i_r1': 0.83700, 't2i_r5': 0.96200, 't2i_r10': 0.98300},
        'source': '/root/SAID-s0-trimask-v01/docs/said_trimask/results.json',
    },
}
GATE = {'coco_i2t_r1': 0.6058, 'coco_t2i_r1': 0.41236,
        'rule': 'COCO I2T R@1 >= 0.6058 且 COCO T2I R@1 >= 0.41236，且至少一项严格提高（原始精度）'}


def json_safe(value):
    """Non-finite floats become ``None`` so the page never receives a fake 0 or an invalid token."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RunNotFound(KeyError):
    pass


class RunRegistry:
    """The only way a request can name a directory."""

    def __init__(self):
        self._runs = {}

    def register(self, run_id, directory, label=None, demo=False, objective=None, arm=None,
                 evaluation_prefix=None):
        if not RUN_ID_PATTERN.match(run_id or ''):
            raise ValueError('invalid run id %r' % (run_id,))
        resolved = os.path.realpath(directory)
        self._runs[run_id] = {
            'run_id': run_id,
            'directory': resolved,
            'label': label or run_id,
            'demo': bool(demo),
            'objective': objective,
            'arm': arm,
            'evaluation_prefix': evaluation_prefix or run_id,
        }
        return self._runs[run_id]

    def resolve(self, run_id):
        if not RUN_ID_PATTERN.match(run_id or '') or run_id not in self._runs:
            raise RunNotFound(run_id)
        return self._runs[run_id]
    def ids(self):
        return sorted(self._runs)

    def entries(self):
        return [self._runs[key] for key in self.ids()]

    def path(self, run_id, key):
        if key not in RUN_FILES:
            raise RunNotFound(key)
        return os.path.join(self.resolve(run_id)['directory'], RUN_FILES[key])

    def evaluation_path(self, run_id, dataset):
        """Resolve an evaluation file inside the registered run directory (never a client path).

        The registered prefix wins; if it does not exist, the most recently modified file with the
        fixed dataset suffix inside the fixed ``evaluation/`` subdirectory is used, and the caller
        reports which file it read.
        """
        if dataset not in EVALUATION_FILES:
            raise RunNotFound(dataset)
        record = self.resolve(run_id)
        explicit = os.path.join(record['directory'],
                                EVALUATION_FILES[dataset].format(name=record['evaluation_prefix']))
        if os.path.exists(explicit):
            return explicit
        directory = os.path.join(record['directory'], 'evaluation')
        suffix = EVALUATION_SUFFIX[dataset]
        try:
            candidates = [name for name in os.listdir(directory) if name.endswith(suffix)]
        except OSError:
            return explicit

        def sort_key(name):
            try:
                modified = os.path.getmtime(os.path.join(directory, name))
            except OSError:
                modified = 0.0
            return (modified, name)

        if not candidates:
            return explicit
        return os.path.join(directory, sorted(candidates, key=sort_key)[-1])

    def diagnostics_path(self, run_id, key='probe'):
        """Resolve one whitelisted diagnostics file inside the registered run directory."""
        if key not in DIAGNOSTICS_FILES:
            raise RunNotFound(key)
        return os.path.join(self.resolve(run_id)['directory'], DIAGNOSTICS_DIR,
                            DIAGNOSTICS_FILES[key])


def read_json(path):
    """``(value, error)`` -- a missing or half-written file is reported, never raised."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, None
    except (ValueError, OSError) as error:
        return None, '%s: %s' % (os.path.basename(path), error)


class MetricsReader:
    """Incremental, cached JSONL reader with index cursors and no unbounded rescans.

    A cursor is a record index. The cache keeps the parsed records of each file plus the byte offset
    already consumed, so a poll reads only the appended bytes. ``after=0`` on a large history is
    still capped by ``MAX_RECORDS_PER_READ``; the client then continues from the returned cursor.
    """

    def __init__(self):
        self._cache = {}

    def _state(self, path):
        return self._cache.setdefault(path, {'records': [], 'offset': 0, 'partial': b'',
                                             'warnings': [], 'size': 0})

    def _refresh(self, path):
        state = self._state(path)
        try:
            size = os.path.getsize(path)
        except OSError:
            return state
        if size < state['offset']:
            state.update({'records': [], 'offset': 0, 'partial': b'',
                          'size': size})
            state['warnings'].append('日志被截断或轮换，已重置读取位置: %s' % os.path.basename(path))
        if size == state['offset']:
            state['size'] = size
            return state
        with open(path, 'rb') as handle:
            handle.seek(state['offset'])
            chunk = handle.read()
            state['offset'] += len(chunk)
        state['size'] = size
        data = state['partial'] + chunk
        lines = data.split(b'\n')
        state['partial'] = lines.pop()          # an unfinished last line waits for the next poll
        parsed = 0
        for raw in lines:
            if parsed >= MAX_LINES_PER_POLL:
                state['partial'] = b'\n'.join(lines[parsed:]) + b'\n' + state['partial']
                state['offset'] -= sum(len(item) + 1 for item in lines[parsed:])
                break
            parsed += 1
            line = raw.strip()
            if not line:
                continue
            if len(line) > MAX_LINE_BYTES:
                state['warnings'].append('超长日志行已跳过 (%d bytes)' % len(line))
                continue
            text = line.decode('utf-8', 'replace')
            if text.startswith('LOG '):
                text = text[4:]
            try:
                record = json.loads(text)
            except ValueError:
                state['warnings'].append('损坏的完整日志行已跳过: %s' % text[:80])
                continue
            if not isinstance(record, dict):
                state['warnings'].append('非对象的日志行已跳过')
                continue
            state['records'].append({key: json_safe(value) for key, value in record.items()})
        state['warnings'] = state['warnings'][-20:]
        return state

    def read(self, path, after=0, limit=MAX_RECORDS_PER_READ):
        state = self._refresh(path)
        after = max(0, int(after or 0))
        records = state['records'][after:after + limit]
        return {
            'records': records,
            'cursor': after + len(records),
            'available': len(state['records']),
            'warnings': list(state['warnings']),
        }


class DashboardData:
    """Everything the pages and the JSON API need, with no side effects on the run."""

    def __init__(self, registry, clock=time.time):
        self.registry = registry
        self.reader = MetricsReader()
        self.clock = clock

    # ---------------------------------------------------------------- runs
    def list_runs(self):
        return [self._summary(record) for record in self.registry.entries()]

    def _summary(self, record):
        run_id = record['run_id']
        status, _ = read_json(os.path.join(record['directory'], RUN_FILES['status']))
        cursor = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        last = None
        if cursor['available']:
            tail = self.reader.read(self.registry.path(run_id, 'log'),
                                    after=cursor['available'] - 1, limit=1)
            last = tail['records'][0] if tail['records'] else None
        warnings = list(cursor['warnings'])
        if status:
            phase = status.get('phase', 'unknown')
        elif last:
            # records exist but no status file: this run was not launched by the runner, so its
            # phase genuinely cannot be known -- it is reported as unknown, never as "not started"
            phase = 'unknown'
            warnings.append('缺少 run_status.json（该 run 不是由 runner 启动），阶段无法判定')
        else:
            phase = 'not_started'
        return {
            'run_id': run_id, 'label': record['label'], 'demo': record['demo'],
            'directory': record['directory'],
            'phase': phase,
            'phase_label': PHASE_LABELS.get(phase, '未知'),
            'completed_steps': (status or {}).get('completed_steps',
                                                  (last or {}).get('completed_steps')),
            'objective': record['objective'] or (status or {}).get('objective'),
            'arm': record['arm'] or (status or {}).get('arm'),
            'warnings': warnings,
        }

    # ---------------------------------------------------------------- status
    def status(self, run_id):
        record = self.registry.resolve(run_id)
        status, error = read_json(os.path.join(record['directory'], RUN_FILES['status']))
        config, _ = read_json(os.path.join(record['directory'], RUN_FILES['config']))
        summary, _ = read_json(os.path.join(record['directory'], RUN_FILES['summary']))
        metrics = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        tail = None
        if metrics['available']:
            slice_ = self.reader.read(self.registry.path(run_id, 'log'),
                                      after=metrics['available'] - 1, limit=1)
            tail = slice_['records'][0] if slice_['records'] else None
        phase = 'not_started'
        if status:
            phase = status.get('phase', 'unknown')
        elif tail:
            phase = 'training'
        total = ((status or {}).get('max_steps')
                 or (config or {}).get('max_steps') or 500)
        completed = ((status or {}).get('completed_steps')
                     or (tail or {}).get('completed_steps') or 0)
        steps_per_second = None
        eta_seconds = None
        recent = self._recent_steps(run_id, 25)
        if recent:
            mean = sum(recent) / len(recent)
            if mean > 0:
                steps_per_second = 1.0 / mean
                remaining = max(0, total - completed)
                # only publish an ETA once there is a real sample of steady-state steps; the UI
                # labels it "估计" and hides it while the run is still warming up
                eta_seconds = remaining * mean if len(recent) >= 10 else None
        started = (status or {}).get('started_at')
        elapsed = None
        if started:
            try:
                elapsed = max(0.0, self.clock() - float(started))
            except (TypeError, ValueError):
                elapsed = None
        return {
            'run_id': run_id, 'label': record['label'], 'demo': record['demo'],
            'phase': phase, 'phase_label': PHASE_LABELS.get(phase, '未知'),
            'status': status, 'status_error': error,
            'objective': ((config or {}).get('objective') or record['objective']
                          or (status or {}).get('objective')),
            'arm': (config or {}).get('arm') or record['arm'] or (status or {}).get('arm'),
            'git_head': (config or {}).get('git_head') or (summary or {}).get('git_head'),
            'text_gate_mode': ((config or {}).get('text_gate_mode')
                               or (status or {}).get('text_gate_mode')),
            # the loss weighting is part of a run's identity: two runs of the same gate mode can
            # differ only here, so the panel must be able to show it
            'loss_profile': ((config or {}).get('loss_profile')
                             or (status or {}).get('loss_profile') or 'default'),
            'lambda_1': (config or {}).get('lambda_1', (status or {}).get('lambda_1')),
            'lambda_2': (config or {}).get('lambda_2', (status or {}).get('lambda_2')),
            'lambda_3': (config or {}).get('lambda_3', (status or {}).get('lambda_3')),
            'lambda_sparse_i': (config or {}).get('lambda_sparse_i',
                                                  (status or {}).get('lambda_sparse_i')),
            'lambda_sparse_t': (config or {}).get('lambda_sparse_t'),
            'completed_steps': completed, 'max_steps': total,
            'progress': (completed / total) if total else None,
            'lr_horizon_steps': (config or {}).get('lr_horizon_steps'),
            'world_size': (config or {}).get('world_size'),
            'batch_size_per_gpu': (config or {}).get('batch_size_per_gpu'),
            'global_pairs': (tail or {}).get('global_pairs'),
            'last_sec_per_step': (tail or {}).get('sec_per_step'),
            'samples_per_sec': (tail or {}).get('samples_per_sec'),
            'peak_memory_gb': (tail or {}).get('peak_memory_gb'),
            'mean_sec_per_step': steps_per_second and round(1.0 / steps_per_second, 4),
            'eta_seconds': eta_seconds,
            'eta_is_estimate': eta_seconds is not None,
            'elapsed_seconds': elapsed,
            'last_logged_step': (tail or {}).get('completed_steps'),
            'run_summary': summary,
            'warnings': (metrics['warnings'] if status is None else []) + (
                [error] if error else []),
        }

    def _recent_steps(self, run_id, count):
        metrics = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        total = metrics['available']
        if total < 5:
            return []
        slice_ = self.reader.read(self.registry.path(run_id, 'log'),
                                  after=max(0, total - count), limit=count)
        values = [record.get('sec_per_step') for record in slice_['records']]
        return [float(value) for value in values
                if isinstance(value, (int, float)) and value and value > 0]

    # ---------------------------------------------------------------- metrics
    def metrics(self, run_id, after=0, limit=MAX_RECORDS_PER_READ):
        self.registry.resolve(run_id)
        block = self.reader.read(self.registry.path(run_id, 'log'), after=after, limit=limit)
        block['run_id'] = run_id
        return block

    # ---------------------------------------------------------------- masks
    def masks(self, run_id):
        self.registry.resolve(run_id)
        snapshot, error = read_json(self.registry.path(run_id, 'mask_snapshot'))
        latest_heavy = None
        metrics = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        if metrics['available']:
            slice_ = self.reader.read(self.registry.path(run_id, 'log'),
                                      after=max(0, metrics['available'] - 60),
                                      limit=60)
            for record in reversed(slice_['records']):
                if record.get('heavy_diagnostics'):
                    latest_heavy = {key: value for key, value in record.items()
                                    if key.startswith('mask_') or key.startswith('text_gate_')
                                    or key in ('completed_steps', 'adv_gap',
                                               'third_not_worse_fraction')}
                    break
        return {'run_id': run_id, 'snapshot': snapshot, 'snapshot_error': error,
                'latest_heavy_scalars': latest_heavy}

    # ---------------------------------------------------------------- diagnostics
    def diagnostics(self, run_id):
        """The read-only offline geometry probe, or an explicit 未诊断 when it has not been run.

        Only the three whitelisted files inside ``diagnostics/`` are ever opened, the response is
        built from the file's own content plus its mtime, and nothing here can start a forward pass:
        a page refresh cannot re-run the probe or touch the GPU.
        """
        self.registry.resolve(run_id)
        probe_path = self.registry.diagnostics_path(run_id, 'probe')
        payload, error = read_json(probe_path)
        try:
            updated_at = os.path.getmtime(probe_path)
        except OSError:
            updated_at = None
        files = {}
        for key in DIAGNOSTICS_FILES:
            path = self.registry.diagnostics_path(run_id, key)
            files[key] = {'file': os.path.basename(path), 'available': os.path.isfile(path)}
        result = {
            'run_id': run_id,
            'available': payload is not None,
            'status': DIAGNOSTIC_LABELS['available'] if payload is not None
                      else DIAGNOSTIC_LABELS['missing'],
            'file': os.path.basename(probe_path),
            'directory': DIAGNOSTICS_DIR,
            'files': files,
            'error': error,
            'updated_at': updated_at,
            'updated_at_iso': (time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(updated_at))
                               if updated_at else None),
            'reminders': DIAGNOSTIC_REMINDERS,
        }
        if payload is None:
            # no fabricated zeros: every field the page would show is explicitly absent
            for key in ('scope', 'read_only', 'new_optimizer_updates', 'not_a_canonical_evaluation',
                        'checkpoint', 'manifest', 'variance', 'replacements', 'geometry',
                        'hard_queries', 'not_run', 'tie_rule', 'dtypes_and_precision',
                        'mask_statistics', 'reconciliation_with_trained_scoring_form', 'timing'):
                result[key] = None
            result['new_optimizer_updates'] = None
            return result

        replacements = payload.get('diagnostic_B_replacements') or {}
        variants = {}
        for name, entry in (replacements.get('variants') or {}).items():
            variants[name] = {key: ({k: v for k, v in value.items() if k != 'per_query_rank'}
                                    if isinstance(value, dict) else value)
                              for key, value in entry.items()}
        checkpoint = payload.get('checkpoint') or {}
        manifest = payload.get('manifest') or {}
        result.update({
            'scope': payload.get('scope'),
            'read_only': payload.get('read_only'),
            'new_optimizer_updates': payload.get('new_optimizer_updates'),
            'not_a_canonical_evaluation': payload.get('not_a_canonical_evaluation'),
            'tie_rule': payload.get('tie_rule'),
            'not_run': payload.get('not_run'),
            'timing': payload.get('timing'),
            'dtypes_and_precision': payload.get('dtypes_and_precision'),
            'mask_statistics': payload.get('mask_statistics'),
            'reconciliation_with_trained_scoring_form':
                payload.get('reconciliation_with_trained_scoring_form'),
            'checkpoint': {
                'sha256_before': checkpoint.get('sha256_before'),
                'sha256_after': checkpoint.get('sha256_after'),
                'sha256_unchanged': checkpoint.get('sha256_unchanged'),
                'identity': checkpoint.get('identity'),
                'meta': checkpoint.get('meta'),
                'clip_tensors': checkpoint.get('clip_tensors'),
                'text_mask_net_tensors': checkpoint.get('text_mask_net_tensors'),
                'parameter_state_unchanged': (payload.get('parameter_state') or {}).get('unchanged'),
                'run_status_unchanged': (payload.get('run_status') or {}).get('unchanged'),
            },
            'manifest': {'count': manifest.get('count'), 'sha256': manifest.get('sha256'),
                         'selection': manifest.get('selection'), 'source': manifest.get('source'),
                         'image_ids': manifest.get('image_ids'),
                         'annotation_ids': manifest.get('annotation_ids'),
                         'skipped_captions': manifest.get('skipped_captions')},
            'variance': payload.get('diagnostic_A_variance'),
            'replacements': {'variants': variants,
                             'identity_checks': replacements.get('identity_checks'),
                             'shuffled_aggregate': replacements.get('shuffled_aggregate'),
                             'shuffle_permutations': replacements.get('shuffle_permutations'),
                             'not_used': replacements.get('not_used')},
            'geometry': payload.get('diagnostic_C_geometry'),
            'hard_queries': payload.get('hard_queries'),
        })
        return result

    # ---------------------------------------------------------------- evaluation
    def evaluation(self, run_id):
        self.registry.resolve(run_id)
        result = {'run_id': run_id, 'datasets': {}, 'baselines': BASELINES, 'gate': GATE}
        for dataset in EVALUATION_FILES:
            path = self.registry.evaluation_path(run_id, dataset)
            payload, error = read_json(path)
            result['datasets'][dataset] = {
                'available': payload is not None,
                'error': error,
                'file': os.path.basename(path),
                'metrics': self._evaluation_metrics(dataset, payload),
                'raw': payload if payload is not None else None,
            }
        coco = result['datasets']['coco']['metrics']
        verdict = None
        if coco:
            passed = (coco['i2t_r1'] is not None and coco['t2i_r1'] is not None
                      and coco['i2t_r1'] >= GATE['coco_i2t_r1']
                      and coco['t2i_r1'] >= GATE['coco_t2i_r1']
                      and (coco['i2t_r1'] > GATE['coco_i2t_r1']
                           or coco['t2i_r1'] > GATE['coco_t2i_r1']))
            verdict = {
                'verdict': 'PROMISING_AT_500' if passed else 'FAIL',
                'i2t_pass': coco['i2t_r1'] >= GATE['coco_i2t_r1'],
                't2i_pass': coco['t2i_r1'] >= GATE['coco_t2i_r1'],
                'i2t_delta_points': (coco['i2t_r1'] - GATE['coco_i2t_r1']) * 100.0,
                't2i_delta_points': (coco['t2i_r1'] - GATE['coco_t2i_r1']) * 100.0,
            }
        result['verdict'] = verdict
        return result

    @staticmethod
    def _evaluation_metrics(dataset, payload):
        if not payload:
            return None
        try:
            if dataset == 'coco':
                canonical = payload.get('canonical') or {}
                inner = next(iter(canonical.values()))
                block = inner['coco_val2017']
                return {
                    'i2t_r1': block['image2text_R1'], 'i2t_r5': block['image2text_R5'],
                    'i2t_r10': block['image2text_R10'], 't2i_r1': block['text2image_R1'],
                    't2i_r5': block['text2image_R5'], 't2i_r10': block['text2image_R10'],
                    'checkpoint_sha256': inner.get('checkpoint_sha256'),
                    'label': next(iter(canonical)),
                    'protocol': 'COCO val2017 全量 5000 图 / 25000 文本，native CLS/EOS',
                }
            block = payload['urban1k']
            return {
                'i2t_r1': block['image2text']['R1'], 'i2t_r5': block['image2text']['R5'],
                'i2t_r10': block['image2text']['R10'], 't2i_r1': block['text2image']['R1'],
                't2i_r5': block['text2image']['R5'], 't2i_r10': block['text2image']['R10'],
                'checkpoint_sha256': payload.get('checkpoint_sha256'),
                'label': payload.get('label'),
                'protocol': 'Urban-1k 1000 图 / 1000 文本，native CLS/EOS',
            }
        except (KeyError, TypeError, StopIteration):
            return None

    # ---------------------------------------------------------------- logs
    def logs(self, run_id, limit=200):
        """The tail of the run log plus the runner's own error/warning lines."""
        self.registry.resolve(run_id)
        metrics = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        total = metrics['available']
        slice_ = self.reader.read(self.registry.path(run_id, 'log'),
                                  after=max(0, total - limit), limit=limit)
        status, status_error = read_json(self.registry.path(run_id, 'status'))
        issues = []
        for record in slice_['records'][-limit:]:
            if record.get('grads_finite') == 0.0:
                issues.append({'level': 'error', 'step': record.get('completed_steps'),
                               'text': '出现非有限梯度，已拒绝该次更新'})
        for warning in slice_['warnings']:
            issues.append({'level': 'warning', 'step': None, 'text': warning})
        if status_error:
            issues.append({'level': 'warning', 'step': None, 'text': status_error})
        if status:
            for key, label in (('train_exit_code', '训练'), ('export_exit_code', '导出'),
                               ('coco_exit_code', 'COCO 评估'), ('urban_exit_code', 'Urban 评估')):
                code = status.get(key)
                if isinstance(code, int) and code != 0:
                    issues.append({'level': 'error', 'step': status.get('completed_steps'),
                                   'text': '%s 退出码 %d' % (label, code)})
        return {
            'run_id': run_id,
            'records': slice_['records'],
            'issues': issues,
            'commands': (status or {}).get('commands'),
            'warnings': slice_['warnings'],
        }

    def metrics_csv(self, run_id, columns=None):
        """A small CSV of the scalar history for download (no tensors, no captions)."""
        self.registry.resolve(run_id)
        side = self.reader.read(self.registry.path(run_id, 'log'), after=0, limit=1)
        total = side['available']
        slice_ = self.reader.read(self.registry.path(run_id, 'log'), after=max(0, total - 5000),
                                  limit=5000)
        records = slice_['records']
        if not records:
            return 'completed_steps\n'
        if columns is None:
            columns = ['completed_steps', 'loss_total', 'loss_1', 'loss_1_i2t', 'loss_1_t2i',
                       'loss_2', 'loss_2_i2t', 'loss_2_t2i', 'loss_3', 'loss_3_i2t', 'loss_3_t2i',
                       'loss_sparse_i', 'loss_sparse_t', 'weighted_loss_1', 'weighted_loss_2',
                       'weighted_loss_3', 'weighted_loss_sparse_i', 'weighted_loss_sparse_t',
                       'mask_i_keep_ratio', 'mask_t_mean', 'text_gate_hT_zero_fraction',
                       'mask_intersection_intersection_count_mean',
                       'mask_intersection_jaccard_mean', 'adv_gap', 'third_not_worse_fraction',
                       'sec_per_step', 'peak_memory_gb']
            columns = [name for name in columns
                       if any(name in record for record in records)] or ['completed_steps']
        lines = [','.join(columns)]
        for record in records:
            row = []
            for name in columns:
                value = record.get(name)
                row.append('' if value is None else str(value))
            lines.append(','.join(row))
        return '\n'.join(lines) + '\n'
