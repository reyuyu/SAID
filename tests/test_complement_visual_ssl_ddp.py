"""Single-process vs 2-rank CVSSL gradient equivalence (including unequal valid counts).

The 2-rank run is launched with ``torchrun --nproc_per_node=2`` on gloo/CPU by this test; each rank
writes its own gradient, and the test then checks the two statements that matter:

* ``sum_r grad_r == grad_single``      -- the trainer's convention (no DDP wrapper, ``1/V_global``);
* ``mean_r grad_r == grad_single``     -- the DDP-averaged convention (``ddp_gradient_averaging=1``
  multiplies by the world size so the post-average gradient is the global mean).

A rank with zero valid anchors must still take part in every collective: the test includes a run
where one rank's anchors are all invalid and requires it to finish (no deadlock).
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, 'tests', '_cvssl_ddp_worker.py')
PYTHON = sys.executable
PORT = 29531


def _run(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise AssertionError('command failed: %s\n%s\n%s'
                             % (' '.join(command), result.stdout[-2000:], result.stderr[-2000:]))


def _single(out, uneven):
    _run([PYTHON, WORKER, '--mode', 'single', '--out', out, '--uneven', str(int(uneven))])


def _two_rank(out_dir, uneven, ddp_scaling):
    env = {'NCCL_SOCKET_IFNAME': 'lo', 'GLOO_SOCKET_IFNAME': 'lo',
           'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(PORT)}
    _run([PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=2',
          '--master_port', str(PORT), WORKER, '--mode', 'rank',
          '--out', os.path.join(out_dir, 'rank{rank}.json'), '--uneven', str(int(uneven)),
          '--ddp_scaling', str(int(ddp_scaling))], env=env)


@pytest.mark.parametrize('uneven', [False, True])
def test_two_rank_cvssl_gradient_matches_single_process(tmp_path, uneven):
    single_path = str(tmp_path / 'single.json')
    _single(single_path, uneven)
    _two_rank(str(tmp_path), uneven, ddp_scaling=0)
    single = json.load(open(single_path))
    ranks = [json.load(open(str(tmp_path / ('rank%d.json' % rank)))) for rank in range(2)]

    summed = torch.tensor(ranks[0]['grad_theta']) + torch.tensor(ranks[1]['grad_theta'])
    reference = torch.tensor(single['grad_theta'])
    scale = max(float(reference.abs().max()), 1e-8)
    assert float((summed - reference).abs().max()) / scale < 1e-5
    # the real global mean loss is identical on both sides
    assert abs(single['global_mean_loss'] - ranks[0]['global_mean_loss']) < 1e-6
    assert abs(single['valid_ab'] - ranks[0]['valid_ab']) < 1e-6
    if uneven:
        local_ab = [rank['valid_local_ab'] for rank in ranks]
        assert local_ab[0] != local_ab[1], \
            'this case must actually have unequal local valid counts, got %r' % (local_ab,)
        # and the global count is still the sum of both ranks
        assert abs(sum(local_ab) - single['valid_ab']) < 1e-6


@pytest.mark.parametrize('uneven', [False, True])
def test_ddp_scaled_variant_is_the_rank_average(tmp_path, uneven):
    single_path = str(tmp_path / 'single.json')
    _single(single_path, uneven)
    _two_rank(str(tmp_path), uneven, ddp_scaling=1)
    single = json.load(open(single_path))
    ranks = [json.load(open(str(tmp_path / ('rank%d.json' % rank)))) for rank in range(2)]
    assert all(abs(rank['loss_scale'] - 2.0) < 1e-9 for rank in ranks)
    averaged = 0.5 * (torch.tensor(ranks[0]['grad_theta']) + torch.tensor(ranks[1]['grad_theta']))
    reference = torch.tensor(single['grad_theta'])
    scale = max(float(reference.abs().max()), 1e-8)
    assert float((averaged - reference).abs().max()) / scale < 1e-5


def test_rank_with_no_valid_anchor_does_not_deadlock(tmp_path):
    """Both anchors of rank 1 are made invalid; every collective must still complete."""
    worker = os.path.join(REPO_ROOT, 'tests', '_cvssl_no_valid_worker.py')
    env = {'NCCL_SOCKET_IFNAME': 'lo', 'GLOO_SOCKET_IFNAME': 'lo',
           'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(PORT + 1)}
    _run([PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=2',
          '--master_port', str(PORT + 1), worker, '--out',
          os.path.join(str(tmp_path), 'rank{rank}.json')], env=env)
    for rank in range(2):
        payload = json.load(open(str(tmp_path / ('rank%d.json' % rank))))
        assert payload['finite']
        if rank == 1:
            assert payload['valid_local'] == 0.0
