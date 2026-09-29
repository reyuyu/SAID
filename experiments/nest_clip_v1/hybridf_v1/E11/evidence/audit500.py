"""Verify the saved E11 step500 snapshot from the user-interrupted longer run."""
from collections import deque
import json
import math
from pathlib import Path

import torch

from audit import BASE, EV, REPO, ROOT, equal, read, stream
from train.nested_semantic_data import file_sha
from train.train_nested_semantic_mask import learning_rates


def main():
    run = BASE / 'formal'
    config = read(run / 'config.json')
    execution = read(REPO / 'experiments/nest_clip_v1/hybridf_v1/E11/evidence/formal.execution.json')
    assert execution['exit_code'] != 0
    assert config['run_type'] == 'formal' and config['start_updates'] == 0
    assert config['max_updates'] == config['horizon'] == 3651 and config['resume'] is None
    assert config['arm'] == 'A3' and config['full_native_mix'] == .25
    assert config['sampling_mode'] == 'random_k' and config['sampling_seed'] == 0
    assert config['training_records'] == 1245901 and config['batches_per_epoch'] == 1217
    assert config['batch_size'] == 256 and config['world_size'] == 4
    assert config['init_sha256'] == '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
    for path, digest in config['code_sha256'].items():
        assert file_sha(REPO / path) == digest, path
    assert len({card['uuid'] for card in config['ranks']}) == 4

    console = (REPO / 'experiments/nest_clip_v1/hybridf_v1/E11/evidence/formal.console.txt').read_text()
    assert 'SignalException: Process ' in console and ' got signal: 2' in console
    assert 'ncclSystemError' not in console and 'NCCL WARN' not in console
    rows = stream([run / 'steps.jsonl'])
    reference = stream([ROOT / 'randomk500/A3-RandomK/steps.jsonl'])
    window = deque(maxlen=50)
    milestones = {}
    max_error = 0.
    sample_rows = 0
    for i in range(1, 501):
        row = next(rows)
        old = next(reference)
        assert row['step'] == old['step'] == i and row['s'] == i - 1
        assert row['epoch'] == old['epoch'] == 0 and row['full_native_mix'] == .25
        assert row['F_native_candidates'] == row['F_candidates'] == 1024
        assert row['valid_global'] == old['valid_global']
        assert row['nonfinite'] == 0
        assert all(math.isfinite(x) for x in row.values() if isinstance(x, (int, float)))
        assert (row['lr_backbone'], row['lr_mask']) == learning_rates(i - 1, 3651)
        weight = min(1, (i - 1) / 200)
        assert row['inc_weight'] == weight
        hybrid = .75 * (row['F_mask_i2t'] + row['F_mask_t2i']) + .25 * (
            row['F_native_i2t'] + row['F_native_t2i'])
        assert math.isclose(row['F_hybrid'], hybrid, rel_tol=2e-6, abs_tol=2e-5)
        align = (10 / 3) * (hybrid + row['O_i2t'] + row['O_t2i'] +
                            row['E_i2t'] + row['E_t2i'])
        sparse = (row['F_sparse'] + 2 * row['O_sparse'] + 2 * row['E_sparse']) / 3
        error = abs(align + sparse + weight * row['inc'] - row['loss'])
        assert math.isclose(align + sparse + weight * row['inc'], row['loss'],
                            rel_tol=2e-5, abs_tol=2e-4)
        max_error = max(max_error, error)
        for health, prior in zip(row['rank_health'], old['rank_health']):
            assert health['rank'] == prior['rank'] and health['updates'] == i
            assert health['batch'] == prior['batch'] == 256 and health['gradients_finite']
            assert all(math.isfinite(x) for x in health['gradient_norms'].values())
            assert health['stream_sha256'] == prior['stream_sha256']
            for key in ('sample_ids', 'n', 'K', 'sample_id_sha256', 'full_view_sha256',
                        'local_views_sha256', 'split_sha256'):
                assert health['sampling'][key] == prior['sampling'][key], (i, health['rank'], key)
            sample_rows += health['batch']
        row['common_loss'] = row['loss'] - weight * row['inc']
        window.append(row)
        if i in (1, 100, 200, 201, 300, 400, 500):
            milestones[str(i)] = {k: v for k, v in row.items() if k != 'rank_health'}
    assert next(rows)['step'] > 500
    common = torch.load(ROOT / 'shared/step000000.pt', map_location='cpu', weights_only=False)
    zero = torch.load(run / 'step000000.pt', map_location='cpu', weights_only=False)
    original = torch.load(ROOT / 'randomk500/A3-RandomK/step000000.pt', map_location='cpu', weights_only=False)
    assert equal(common['model'], zero['model']) and equal(common['optimizer'], zero['optimizer'])
    for ours, prior in zip(zero['rng_per_rank'], original['rng_per_rank']):
        assert equal(ours['python'], prior['python'])
        assert torch.equal(ours['cpu'], prior['cpu']) and torch.equal(ours['cuda'], prior['cuda'])
        assert ours['numpy'][0] == prior['numpy'][0]
        assert (ours['numpy'][1] == prior['numpy'][1]).all()
        assert ours['numpy'][2:] == prior['numpy'][2:]
    del common, zero, original
    checkpoints = []
    for n in (0, 100, 200, 300, 400, 500):
        path = run / f'step{n:06d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['completed_steps'] == n
        assert payload['scheduler_horizon'] == payload['stop_updates'] == 3651
        assert payload['config']['full_native_mix'] == .25
        assert len(payload['rng_per_rank']) == 4
        assert all(int(x['step']) == n for x in payload['optimizer']['state'].values())
        del payload
        checkpoints.append(dict(step=n, path=str(path), sha256=file_sha(path)))
    keys = (key for key, value in window[-1].items()
            if isinstance(value, (int, float)) and key not in ('step', 's', 'epoch'))
    result = dict(passed=True, audited_snapshot_step=500, horizon=3651,
                  planned_max_updates=3651, actual_last_logged_step=max(x['step'] for x in stream([run / 'steps.jsonl'])),
                  note='User requested stop after step500; SIGINT deliberately interrupted a run that had already advanced further. Only the saved step500 checkpoint is evaluated. Final cross-rank parameter comparison was not executed at step500.',
                  code_commit=config['git_head'], execution=execution, config=config,
                  compared_rank_streams=2000, sample_rows=sample_rows,
                  sample_F_P_R_K_streams_identical_to_A3=True,
                  step0_model_optimizer_rng_equal=True,
                  max_loss_reconstruction_abs_error=max_error,
                  milestones=milestones,
                  last50={key: sum(row[key] for row in window) / len(window) for key in keys},
                  checkpoints=checkpoints,
                  raw_log=dict(path=str(run / 'steps.jsonl'), sha256=file_sha(run / 'steps.jsonl')))
    destination = EV / 'formal500-audit.json'
    with destination.open('x') as stream_out:
        json.dump(result, stream_out, indent=2)
        stream_out.write('\n')
    print(json.dumps(dict(passed=True, step=500, actual_stop=result['actual_last_logged_step'],
                          compared_rank_streams=2000, max_error=max_error)))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
