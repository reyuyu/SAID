"""Audit the three real four-GPU smokes before any formal run starts."""
import hashlib
import json
from pathlib import Path

import torch


SERVER = Path('/root/lk_projects/SAID-nest-clip-v1')
ROOT = SERVER / 'jointmask_fast_v1' / 'smoke'
HERE = Path(__file__).resolve().parent
GROUPS = ('T-fast', 'TI-fast', 'TI-Shuffle-fast')


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, tuple):
        return isinstance(right, tuple) and len(left) == len(right) and all(
            rng_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            rng_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            rng_equal(left[key], right[key]) for key in left)
    try:
        return bool(left == right)
    except ValueError:
        return bool((left == right).all())


rows = {}
configs = {}
acceptance = {}
checkpoint_summary = {}
for group in GROUPS:
    directory = ROOT / group
    rows[group] = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    configs[group] = json.loads((directory / 'config.json').read_text())
    assert configs[group]['runtime_model'] == dict(image_chunk=128, text_chunk=128, checkpoint_pair_blocks=False, checkpoint_encoders=True, encoder_checkpoint_active=True, condition_mode=configs[group]['condition_mode'], arm='A3')
    acceptance[group] = json.loads((directory / 'acceptance.json').read_text())
    assert [row['step'] for row in rows[group]] == list(range(1, 6))
    assert acceptance[group]['passed']
    assert all(rank['completed_updates'] == rank['updates_this_run'] == 5
               for rank in acceptance[group]['ranks'])
    assert all(rank['max_parameter_difference_from_rank0'] == 0
               for rank in acceptance[group]['ranks'])
    assert all(row['F_candidates'] == row['O_candidates'] == row['E_candidates'] == 1024
               for row in rows[group])
    assert all(row['valid_global'] == 1024 and row['nonfinite'] == 0 for row in rows[group])
    assert all(len(row['rank_health']) == 4 for row in rows[group])
    assert all(rank['batch'] == 256 and rank['gradients_finite']
               for row in rows[group] for rank in row['rank_health'])
    step0 = torch.load(directory / 'step000000.pt', map_location='cpu', weights_only=False)
    assert step0['completed_steps'] == 0 and not step0['optimizer']['state']
    checkpoint_summary[group] = dict(
        step0_sha256=file_sha(directory / 'step000000.pt'),
        step5_sha256=file_sha(directory / 'step000005.pt'),
        model_state_sha256=state_digest(step0['model']),
        adapter_state_sha256=(state_digest(step0['adapter']) if step0['adapter'] else None),
        optimizer_state_entries=len(step0['optimizer']['state']),
        optimizer_groups=len(step0['optimizer']['param_groups']),
        init_sha256=step0['config']['init_sha256'],
        rng_per_rank=step0['rng_per_rank'])
    del step0

assert len({checkpoint_summary[group]['model_state_sha256'] for group in GROUPS}) == 1
assert checkpoint_summary['T-fast']['adapter_state_sha256'] is None
assert checkpoint_summary['TI-fast']['adapter_state_sha256'] == checkpoint_summary['TI-Shuffle-fast']['adapter_state_sha256']
assert checkpoint_summary['T-fast']['optimizer_groups'] == 2
assert checkpoint_summary['TI-fast']['optimizer_groups'] == checkpoint_summary['TI-Shuffle-fast']['optimizer_groups'] == 3
assert all(checkpoint_summary[group]['init_sha256'] ==
           '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
           for group in GROUPS)
assert all(rng_equal(checkpoint_summary['T-fast']['rng_per_rank'], checkpoint_summary[group]['rng_per_rank'])
           for group in GROUPS[1:])

for step in range(5):
    streams = [[health['stream_sha256'] for health in rows[group][step]['rank_health']]
               for group in GROUPS]
    assert streams[0] == streams[1] == streams[2]
old_rows = [json.loads(line) for line in
            (SERVER / 'randomk500/smoke/A3-RandomK/steps.jsonl').read_text().splitlines()]
assert len(old_rows) >= 5
for step in range(5):
    assert ([health['stream_sha256'] for health in rows['T-fast'][step]['rank_health']] ==
            [health['stream_sha256'] for health in old_rows[step]['rank_health']])

assert rows['T-fast'][0]['loss'] == rows['TI-fast'][0]['loss'] == rows['TI-Shuffle-fast'][0]['loss']
assert all(row['F_delta_abs_mean'] == 0 for row in rows['T-fast'])
for group in ('TI-fast', 'TI-Shuffle-fast'):
    first = rows[group][0]
    assert first['F_delta_abs_mean'] == 0
    assert all(health['adapter_gradient_norms']['WI'] == 0 and
               health['adapter_gradient_norms']['WT'] == 0 and
               health['adapter_gradient_norms']['WO'] > 0
               for health in first['rank_health'])
    assert all(health['adapter_gradient_norms']['WI'] > 0 and
               health['adapter_gradient_norms']['WT'] > 0 and
               health['adapter_gradient_norms']['WO'] > 0
               for row in rows[group][1:] for health in row['rank_health'])
    assert all(row['F_delta_abs_mean'] > 0 for row in rows[group][1:])
assert all(0 < row['shuffle_shift'] < row['F_candidates'] for row in rows['TI-Shuffle-fast'])
assert all(row['shuffle_shift'] == 0 for group in ('T-fast', 'TI-fast') for row in rows[group])

config_ignored = {'condition_mode', 'experiment_name', 'config', 'output_dir', 'ranks',
                  'runtime_model', 'adapter_initialization'}
normalized = []
for group in GROUPS:
    normalized.append({key: value for key, value in configs[group].items() if key not in config_ignored})
assert normalized[0] == normalized[1] == normalized[2]

for group in GROUPS:
    checkpoint_summary[group].pop('rng_per_rank')
result = dict(
    passed=True,
    groups=list(GROUPS),
    tests=dict(
        four_ranks_five_updates=True,
        parameter_agreement_exact=True,
        global_candidates=1024,
        batch_per_rank=256,
        common_model_step0_exact=True,
        ti_shuffle_adapter_step0_exact=True,
        empty_fresh_optimizer_state=True,
        construction_rng_replayed=True,
        all_three_streams_equal=True,
        original_a3_randomk_first5_streams_equal=True,
        runtime_fast_config_asserted=True,
        zero_delta_step1_equal_loss=True,
        adapter_gradient_schedule=True,
        shuffle_fixed_point_free=True),
    checkpoints=checkpoint_summary,
    steps={group: [dict(step=row['step'], loss=row['loss'],
                               delta=row['F_delta_abs_mean'], shift=row['shuffle_shift'],
                               max_seconds=max(rank['seconds'] for rank in row['rank_health']))
                         for row in rows[group]] for group in GROUPS},
    resources={group: dict(device_identities=configs[group]['ranks'],
                           acceptance=acceptance[group]['ranks'])
               for group in GROUPS})
(HERE / 'smoke-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
