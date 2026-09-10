"""Phase 2.8B tests: ``--lr_total_steps`` decouples early stopping from the LR horizon."""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_salu import lr_scale, parse_args, resolve_total_steps  # noqa: E402

STEPS_PER_EPOCH = 1216          # measured from the full ShareGPT4V loader (1,245,901 / 1024)
THREE_EPOCHS = 3 * STEPS_PER_EPOCH


def test_default_behaviour_is_unchanged_with_max_steps():
    args = parse_args([])
    args.max_steps = 500
    args.epochs = 3
    stop_steps, lr_total_steps = resolve_total_steps(args, STEPS_PER_EPOCH)
    # old behaviour: --max_steps truncates the run *and* shrinks the cosine horizon
    assert (stop_steps, lr_total_steps) == (500, 500)
    assert lr_total_steps == (args.max_steps if args.max_steps is not None
                              else args.epochs * STEPS_PER_EPOCH)


def test_default_behaviour_without_max_steps_uses_epochs():
    args = parse_args([])
    args.max_steps = None
    args.epochs = 3
    stop_steps, lr_total_steps = resolve_total_steps(args, STEPS_PER_EPOCH)
    assert (stop_steps, lr_total_steps) == (THREE_EPOCHS, THREE_EPOCHS)


def test_lr_total_steps_only_truncates_the_run():
    args = parse_args([])
    args.max_steps = 500
    args.epochs = 3
    args.lr_total_steps = THREE_EPOCHS
    stop_steps, lr_total_steps = resolve_total_steps(args, STEPS_PER_EPOCH)
    assert stop_steps == 500                 # run length unchanged
    assert lr_total_steps == THREE_EPOCHS    # horizon untouched
    # the LR at step 499 is the one a full 3-epoch run would use there
    assert lr_scale(499, args.warmup_length, lr_total_steps) == pytest.approx(
        lr_scale(499, args.warmup_length, THREE_EPOCHS))
    assert lr_scale(499, args.warmup_length, THREE_EPOCHS) > 0.9
    assert lr_scale(499, args.warmup_length, 500) < 1e-3      # the confound this flag avoids


def test_rejects_non_positive_lr_total_steps():
    args = parse_args([])
    for value in (0, -10):
        args.lr_total_steps = value
        with pytest.raises(ValueError):
            resolve_total_steps(args, STEPS_PER_EPOCH)


def test_cli_default_is_none_and_documented():
    defaults = parse_args([])
    assert defaults.lr_total_steps is None
    assert defaults.max_steps is None
    assert defaults.epochs == 3
    parsed = parse_args(['--max_steps', '500', '--lr_total_steps', '3648'])
    assert parsed.max_steps == 500 and parsed.lr_total_steps == 3648
