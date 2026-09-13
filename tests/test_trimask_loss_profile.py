"""Tests for the loss-weight profiles of the S0-TriMask experiments.

They exist to make two things impossible: (a) the frozen default weighting silently changing, and
(b) a checkpoint trained under one weighting being resumed or evaluated as another. Everything here
is CPU-only and touches no dataset, no GPU and no checkpoint on disk.
"""
import argparse
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

from model.said_trimask import (ARM, ARM_HS, HARD_GATE, LAMBDA_1, LAMBDA_2, LAMBDA_3,  # noqa: E402
                                LAMBDA_SPARSE_I, LAMBDA_SPARSE_T_HS, LOSS_PROFILE_BALANCED,
                                LOSS_PROFILE_DEFAULT, OBJECTIVE, OBJECTIVE_HS, PHASE_HS,
                                SOFT_GATE, all_profile_names, profile_lambdas, profile_names)
from train_said_trimask import check_checkpoint_compatibility, resolve_experiment  # noqa: E402

BALANCED_ARM, BALANCED_OBJECTIVE, BALANCED_PHASE = profile_names(LOSS_PROFILE_BALANCED, HARD_GATE)


# ------------------------------------------------------------------ the default profile is frozen
def test_default_profile_is_byte_identical_to_the_frozen_v01_and_v02_experiments():
    assert profile_names(LOSS_PROFILE_DEFAULT, SOFT_GATE) == (ARM, OBJECTIVE, 's0-trimask-v0.1')
    assert profile_names(LOSS_PROFILE_DEFAULT, HARD_GATE) == (ARM_HS, OBJECTIVE_HS, PHASE_HS)
    hard = profile_lambdas(LOSS_PROFILE_DEFAULT, HARD_GATE)
    soft = profile_lambdas(LOSS_PROFILE_DEFAULT, SOFT_GATE)
    assert hard == {'lambda_1': LAMBDA_1, 'lambda_2': LAMBDA_2, 'lambda_3': LAMBDA_3,
                    'lambda_sparse_i': LAMBDA_SPARSE_I, 'lambda_sparse_t': LAMBDA_SPARSE_T_HS}
    assert hard == {'lambda_1': 10.0, 'lambda_2': 1.0, 'lambda_3': 1.0,
                    'lambda_sparse_i': 2.0, 'lambda_sparse_t': 0.2}
    assert soft['lambda_sparse_t'] == 0.0
    assert soft['lambda_1'] == 10.0


# ------------------------------------------------------------------ the balanced profile
def test_balanced_profile_sets_every_alignment_term_to_10_and_both_sparsity_terms_to_2():
    lambdas = profile_lambdas(LOSS_PROFILE_BALANCED, HARD_GATE)
    assert lambdas == {'lambda_1': 10.0, 'lambda_2': 10.0, 'lambda_3': 10.0,
                       'lambda_sparse_i': 2.0, 'lambda_sparse_t': 2.0}
    assert len(set([lambdas['lambda_1'], lambdas['lambda_2'], lambdas['lambda_3']])) == 1
    assert lambdas['lambda_sparse_i'] == lambdas['lambda_sparse_t']


def test_balanced_profile_has_its_own_names_and_never_collides_with_the_default_ones():
    assert (BALANCED_ARM, BALANCED_OBJECTIVE, BALANCED_PHASE) == ('S0_TriMask_HS_BAL',
                                                                  'smartclip_trimask_hs_bal',
                                                                  's0-trimask-hs-bal-v0.3')
    default_names = {profile_names(LOSS_PROFILE_DEFAULT, mode) for mode in (SOFT_GATE, HARD_GATE)}
    balanced_names = (BALANCED_ARM, BALANCED_OBJECTIVE, BALANCED_PHASE)
    assert BALANCED_ARM not in {name[0] for name in default_names}
    assert BALANCED_OBJECTIVE not in {name[1] for name in default_names}
    assert BALANCED_PHASE not in {name[2] for name in default_names}
    assert balanced_names in all_profile_names()
    assert 'S0_TriMask_HS_BAL' in {name[0] for name in all_profile_names()}


def test_balanced_profile_is_only_defined_for_the_hard_gate():
    with pytest.raises(ValueError):
        profile_lambdas(LOSS_PROFILE_BALANCED, SOFT_GATE)
    with pytest.raises(ValueError):
        profile_names(LOSS_PROFILE_BALANCED, SOFT_GATE)


def test_unknown_profile_is_rejected_everywhere():
    with pytest.raises(ValueError):
        profile_lambdas('something_else', HARD_GATE)
    with pytest.raises(ValueError):
        profile_names('something_else', HARD_GATE)
    args = argparse.Namespace(loss_profile='something_else', text_gate_mode=HARD_GATE, arm=None,
                              objective=None, lambda_1=None, lambda_2=None, lambda_3=None,
                              lambda_sparse_i=None, lambda_sparse_t=None)
    with pytest.raises(SystemExit):
        resolve_experiment(args)


# ------------------------------------------------------------------ argument resolution
def _args(**overrides):
    values = {'loss_profile': LOSS_PROFILE_DEFAULT, 'text_gate_mode': HARD_GATE, 'arm': None,
              'objective': None, 'lambda_1': None, 'lambda_2': None, 'lambda_3': None,
              'lambda_sparse_i': None, 'lambda_sparse_t': None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolve_experiment_returns_the_default_weighting_unchanged():
    arm, objective, phase, lambdas = resolve_experiment(_args())
    assert (arm, objective, phase) == (ARM_HS, OBJECTIVE_HS, PHASE_HS)
    assert lambdas == profile_lambdas(LOSS_PROFILE_DEFAULT, HARD_GATE)


def test_resolve_experiment_returns_the_balanced_weighting_with_its_own_names():
    arm, objective, phase, lambdas = resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED))
    assert (arm, objective, phase) == (BALANCED_ARM, BALANCED_OBJECTIVE, BALANCED_PHASE)
    assert lambdas['lambda_1'] == lambdas['lambda_2'] == lambdas['lambda_3'] == 10.0
    assert lambdas['lambda_sparse_i'] == lambdas['lambda_sparse_t'] == 2.0


def test_resolve_experiment_refuses_a_lambda_that_contradicts_the_profile():
    for name in ('lambda_1', 'lambda_2', 'lambda_3', 'lambda_sparse_i', 'lambda_sparse_t'):
        with pytest.raises(SystemExit):
            resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED, **{name: 0.5}))
    # the frozen weighting likewise cannot be re-weighted in place
    with pytest.raises(SystemExit):
        resolve_experiment(_args(lambda_2=10.0))
    # ... but an explicit value that agrees is accepted
    arm, _, _, lambdas = resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED, lambda_1=10.0))
    assert lambdas['lambda_1'] == 10.0 and arm == BALANCED_ARM


def test_resolve_experiment_refuses_a_contradicting_arm_or_objective():
    # the default hard-gate profile expects ARM_HS / OBJECTIVE_HS, so the *soft* names contradict it
    with pytest.raises(SystemExit):
        resolve_experiment(_args(arm=ARM))
    with pytest.raises(SystemExit):
        resolve_experiment(_args(objective=OBJECTIVE))
    # the balanced profile expects its own names, so the default ones contradict it
    with pytest.raises(SystemExit):
        resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED, arm=ARM_HS))
    with pytest.raises(SystemExit):
        resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED, objective=OBJECTIVE_HS))
    arm, objective, _, _ = resolve_experiment(_args(loss_profile=LOSS_PROFILE_BALANCED,
                                                    arm=BALANCED_ARM,
                                                    objective=BALANCED_OBJECTIVE))
    assert (arm, objective) == (BALANCED_ARM, BALANCED_OBJECTIVE)


def test_soft_mode_still_requires_zero_text_sparsity():
    _, _, _, lambdas = resolve_experiment(_args(text_gate_mode=SOFT_GATE))
    assert lambdas['lambda_sparse_t'] == 0.0


# ------------------------------------------------------------------ checkpoint compatibility
def _payload(profile, lambdas, arm=ARM_HS, objective=OBJECTIVE_HS, mode=HARD_GATE):
    config = dict(lambdas)
    payload = {'arm': arm, 'objective': objective, 'text_gate_mode': mode,
               'lambda_sparse_t': lambdas['lambda_sparse_t'], 'config': config}
    if profile is not None:
        payload['loss_profile'] = profile
    return payload


def test_checkpoint_without_a_profile_key_is_read_as_the_default_profile():
    """Every checkpoint that predates profiles was trained with the default weighting."""
    lambdas = profile_lambdas(LOSS_PROFILE_DEFAULT, HARD_GATE)
    payload = _payload(None, lambdas)                    # no 'loss_profile' key at all
    check_checkpoint_compatibility(payload, ARM_HS, OBJECTIVE_HS, HARD_GATE,
                                   lambdas['lambda_sparse_t'],
                                   loss_profile=LOSS_PROFILE_DEFAULT, lambdas=lambdas)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(payload, ARM_HS, OBJECTIVE_HS, HARD_GATE,
                                       lambdas['lambda_sparse_t'],
                                       loss_profile=LOSS_PROFILE_BALANCED, lambdas=lambdas)


def test_balanced_checkpoint_cannot_be_resumed_as_default_and_vice_versa():
    balanced = profile_lambdas(LOSS_PROFILE_BALANCED, HARD_GATE)
    payload = _payload(LOSS_PROFILE_BALANCED, balanced, arm=BALANCED_ARM,
                       objective=BALANCED_OBJECTIVE)
    # as the balanced run: accepted
    check_checkpoint_compatibility(payload, BALANCED_ARM, BALANCED_OBJECTIVE, HARD_GATE,
                                   balanced['lambda_sparse_t'],
                                   loss_profile=LOSS_PROFILE_BALANCED, lambdas=balanced)
    # as the default run: refused on the arm/objective first, and on the profile if the names matched
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(payload, ARM_HS, OBJECTIVE_HS, HARD_GATE, 0.2,
                                       loss_profile=LOSS_PROFILE_DEFAULT,
                                       lambdas=profile_lambdas(LOSS_PROFILE_DEFAULT, HARD_GATE))
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(payload, BALANCED_ARM, BALANCED_OBJECTIVE, HARD_GATE,
                                       balanced['lambda_sparse_t'],
                                       loss_profile=LOSS_PROFILE_DEFAULT, lambdas=balanced)


def test_every_coefficient_is_checked_on_resume_not_only_the_text_sparsity():
    balanced = profile_lambdas(LOSS_PROFILE_BALANCED, HARD_GATE)
    for name in ('lambda_1', 'lambda_2', 'lambda_3', 'lambda_sparse_i'):
        drifted = dict(balanced)
        drifted[name] = balanced[name] * 2.0
        payload = _payload(LOSS_PROFILE_BALANCED, drifted, arm=BALANCED_ARM,
                           objective=BALANCED_OBJECTIVE)
        with pytest.raises(ValueError):
            check_checkpoint_compatibility(payload, BALANCED_ARM, BALANCED_OBJECTIVE, HARD_GATE,
                                           drifted['lambda_sparse_t'],
                                           loss_profile=LOSS_PROFILE_BALANCED, lambdas=balanced)


def test_gate_mode_and_arm_mismatches_are_still_refused():
    lambdas = profile_lambdas(LOSS_PROFILE_DEFAULT, HARD_GATE)
    payload = _payload(LOSS_PROFILE_DEFAULT, lambdas)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(payload, ARM, OBJECTIVE, HARD_GATE,
                                       lambdas['lambda_sparse_t'],
                                       loss_profile=LOSS_PROFILE_DEFAULT, lambdas=lambdas)
    with pytest.raises(ValueError):
        check_checkpoint_compatibility(payload, ARM_HS, OBJECTIVE_HS, SOFT_GATE, 0.0,
                                       loss_profile=LOSS_PROFILE_DEFAULT,
                                       lambdas=profile_lambdas(LOSS_PROFILE_DEFAULT, SOFT_GATE))
