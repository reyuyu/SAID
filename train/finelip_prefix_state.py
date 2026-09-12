"""FP0 update clock and boundary-only resume validation."""
import math


def group_size(index, length, accumulation=4):
    return min(accumulation, length - (index // accumulation) * accumulation)


def lr_factor(update, horizon, warmup=200):
    if update < warmup:
        return (update + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (update - warmup) / (horizon - warmup)))


def validate_resume(payload, config, length):
    if payload['objective'] != 'finelip_prefix' or payload['arm'] != 'FP0':
        raise ValueError('not an FP0 checkpoint')
    for key in ('batch_size', 'world_size', 'accumulation', 'seed', 'workers',
                'schedule_epochs', 'lr_horizon_updates', 'manifest_sha256', 'init_sha256'):
        if payload['config'][key] != config[key]:
            raise ValueError('resume configuration mismatch: ' + key)
    if payload['accumulation_pending'] != 0:
        raise ValueError('checkpoint must be at an optimizer boundary')
    index = payload['next_batch_index']
    if not 0 <= index <= length or (index != length and index % config['accumulation']):
        raise ValueError('invalid resume cursor')
    epoch = payload['epoch']
    expected_micro = epoch * length + index
    expected_updates = epoch * math.ceil(length / config['accumulation']) + math.ceil(index / config['accumulation'])
    if (payload['micro_step'], payload['optimizer_step']) != (expected_micro, expected_updates):
        raise ValueError('resume clock and cursor disagree')
    return (epoch + 1, 0) if index == length else (epoch, index)
