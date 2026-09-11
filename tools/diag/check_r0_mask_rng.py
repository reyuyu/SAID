"""Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.

R0 RNG semantics: is the random mask of a given sample_id identical in single-process and
2-rank DDP? (per-sample matched independence)

The 2-rank path consumes the generator once per rank with the same seed, while the single-process
path consumes it once for the whole global batch. Because the permutation is now derived from
``mask_seed + 1000003 * sample_id``, a given sample must receive the same permutation in both
layouts. Before that fix the mask depended on the local batch layout and the two disagreed.
"""
import os
import sys

import torch

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

from model.complement_visual_ssl import build_arm_mask  # noqa: E402

DIM = 16


def random_mask(m_s, sample_ids, seed=99, mask_seed=0):
    """Return only the R0 mask for ``m_s`` and the given global sample ids."""
    generator = torch.Generator().manual_seed(seed)
    produced = build_arm_mask(m_s, 'random', generator=generator, sample_ids=sample_ids,
                              mask_seed=mask_seed)
    mask = produced[0]
    if not torch.is_tensor(mask):            # defensive: never silently compare diagnostics
        raise TypeError('build_arm_mask did not return a tensor first, got %r' % type(mask))
    return mask


def make_matrix(rows, generator_seed=5):
    generator = torch.Generator().manual_seed(generator_seed)
    return (torch.rand((rows, DIM), generator=generator) > 0.5).float()


def main():
    results = {}
    m_s_global = make_matrix(6)
    complement = 1.0 - m_s_global

    single_mask = random_mask(m_s_global, torch.arange(6))
    # rank0 of a 2-rank run holds global samples 0..2
    rank0_mask = random_mask(m_s_global.narrow(0, 0, 3), torch.arange(0, 3))
    results['rank0 rows 0..2 == single rows 0..2'] = bool(
        torch.equal(single_mask.narrow(0, 0, 3), rank0_mask))

    # rank1 holds global samples 3..5 and consumes its own generator from row 0
    rank1_mask = random_mask(m_s_global.narrow(0, 3, 3), torch.arange(3, 6))
    results['rank1 rows 0..2 == single rows 3..5'] = bool(
        torch.equal(single_mask.narrow(0, 3, 3), rank1_mask))

    results['mask keeps the sample value histogram'] = bool(
        torch.equal(single_mask.sort(dim=-1).values, complement.sort(dim=-1).values))
    results['at least one row is actually permuted'] = bool(
        not torch.equal(single_mask, complement))

    for key, value in results.items():
        print('%-45s %s' % (key, 'PASS' if value else 'FAIL'))
    if not all(results.values()):
        raise SystemExit('R0 random-mask semantics check FAILED')
    print('R0 random-mask semantics: PASS')


if __name__ == '__main__':
    main()
