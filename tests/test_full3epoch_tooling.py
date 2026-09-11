"""Phase 3.0A.2 tests: runner pairing logic and summary consistency.

This phase adds only a runner and summarisers, so the tests pin down the parts that could
silently produce a wrong scientific table: the ``A<step>`` / ``C<step>`` tag resolution used
for the matched cross-arm comparisons, and the ordering of the matched checkpoint trajectory.
No core model, training or Gap code is touched.
"""
import importlib.util
import os
import sys
import tokenize

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'phase30a_fixed_cohort_eval_2',
        os.path.join(REPO_ROOT, 'tools', 'phase30a_fixed_cohort_eval.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summaries():
    """Load the summariser module without executing its ``__main__`` body.

    The summariser reads run artifacts at import time, so the tests only need its declared
    constants (``MATCHED``, ``REPORT_KEYS``), which are defined before any file access.
    """
    path = os.path.join(REPO_ROOT, 'tools', 'phase30a_2_summaries.py')
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    namespace = {'__name__': 'phase30a_2_summaries_test'}
    # stop before the first statement that touches the filesystem
    cutoff = source.index('if not os.path.exists(RAW):')
    exec(compile(source[:cutoff], path, 'exec'), namespace)
    return namespace


# --------------------------------------------------------------------------- #
# cross-arm tag resolution
# --------------------------------------------------------------------------- #
def test_resolve_tag_accepts_the_spellings_the_runs_actually_use():
    runner = load_runner()
    available = {'initial', 'A100', 'C100', 'A1216', 'C1216', 'Aend', 'Cend'}
    assert runner.resolve_tag(available, 'C100') == 'C100'
    assert runner.resolve_tag(available, 'C') == 'C100'          # short form
    assert runner.resolve_tag(available, 'A1216') == 'A1216'
    assert runner.resolve_tag(available, 'Aend') == 'Aend'
    assert runner.resolve_tag(available, 'initial') == 'initial'
    assert runner.resolve_tag(available, 'B500') is None          # genuinely absent
    assert runner.resolve_tag({'A_step100'}, 'A') == 'A_step100'


def test_cross_arm_comparisons_pair_every_matched_step():
    """C<step> - A<step> must be emitted for each matched step and each required scorer."""
    runner = load_runner()
    torch = pytest.importorskip('torch')

    n = 24
    tags = ['initial'] + ['A%s' % step for step in
                          ('100', '200', '500', '1216', '2432', '3000', 'end')] + \
           ['C%s' % step for step in ('100', '200', '500', '1216', '2432', '3000', 'end')]
    torch.manual_seed(0)
    by_tag = {}
    for tag in tags:
        matrices = {name: torch.randn(n, n) for name in runner.CROSS_ARM_SCORERS}
        by_tag[tag] = {'matrices': matrices}

    comparisons = runner.cross_arm_comparisons(by_tag, replicates=200, seed=7)
    # 7 matched steps (100..end) x 6 scorers; the initial state is the same checkpoint in both
    # arms so it is skipped rather than compared with itself
    assert len(comparisons) == 7 * len(runner.CROSS_ARM_SCORERS)
    for scorer in runner.CROSS_ARM_SCORERS:
        for step in ('100', '200', '500', '1216', '2432', '3000', 'end'):
            key = 'C%s_minus_A%s_%s' % (step, step, scorer)
            assert key in comparisons, key
            entry = comparisons[key]
            assert entry['left']['checkpoint'] == 'C%s' % step
            assert entry['right']['checkpoint'] == 'A%s' % step
            assert entry['step'] == step and entry['scorer'] == scorer
            assert entry['R@1']['query_count'] == n
            assert 'mcnemar_R@1' in entry
    assert not any(name.startswith('Cinitial') for name in comparisons)


def test_cross_arm_comparisons_are_invariant_to_input_order():
    """The summary reads the comparisons by name, so the order of dict insertion must not
    change any number."""
    runner = load_runner()
    torch = pytest.importorskip('torch')
    n = 16
    tags = ['A200', 'C200']
    torch.manual_seed(3)
    by_tag = {tag: {'matrices': {name: torch.randn(n, n)
                                 for name in runner.CROSS_ARM_SCORERS}} for tag in tags}
    first = runner.cross_arm_comparisons(by_tag, replicates=200, seed=11)
    second = runner.cross_arm_comparisons(dict(reversed(list(by_tag.items()))),
                                          replicates=200, seed=11)
    assert first == second


def test_cross_arm_steps_cover_the_required_checkpoints():
    runner = load_runner()
    for step in ('100', '200', '500', '1216', '2432', 'end'):
        assert step in runner.CROSS_ARM_STEPS
    for scorer in ('unsaid_raw', 'unsaid_centered', 'global'):
        assert scorer in runner.CROSS_ARM_SCORERS


# --------------------------------------------------------------------------- #
# summary module
# --------------------------------------------------------------------------- #
def test_matched_checkpoints_are_ordered_and_use_the_recording_steps():
    namespace = load_summaries()
    matched = namespace['MATCHED']
    labels = [entry[0] for entry in matched]
    assert labels == ['step0', 'step100', 'step200', 'step500', 'step1216', 'step2432',
                      'step3000', 'step_end']
    steps = [entry[3] for entry in matched]
    recorded = [value for value in steps if value is not None]
    assert recorded == sorted(recorded)                 # monotone
    assert steps[0] == 0
    assert steps[-1] is None                             # step_end borrows the checkpoint step
    # the initial row is the same checkpoint in both arms
    assert matched[0][1] is None and matched[0][2] is None
    # every non-initial row has both arms
    for label, a_tag, c_tag, _ in matched[1:]:
        assert a_tag is not None and c_tag is not None, label
        assert a_tag.startswith('A') and c_tag.startswith('C'), label


def test_summary_declares_no_training_during_evaluation():
    namespace = load_summaries()
    source = open(os.path.join(REPO_ROOT, 'tools', 'phase30a_2_summaries.py'),
                  encoding='utf-8').read()
    assert 'optimizer_steps' in source
    assert 'training_performed_during_evaluation' in source
    assert "c_u_is_evaluation_target_only" in source
    assert namespace['REPORT_KEYS'][0] == 'R@1'
    assert 'unsaid_raw' in namespace['SCORERS']
    assert 'unsaid_centered' in namespace['SCORERS']


# --------------------------------------------------------------------------- #
# no method change
# --------------------------------------------------------------------------- #
def test_no_training_code_in_the_new_tooling():
    import tokenize

    for relative in (os.path.join('tools', 'phase30a_2_summaries.py'),
                     os.path.join('tools', 'phase30a_fixed_cohort_eval.py')):
        pieces = []
        with open(os.path.join(REPO_ROOT, relative), encoding='utf-8') as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                pieces.append(token.string)
        code = ' '.join(pieces)
        for banned in ('optimizer', 'backward', 'AdamW', 'GradScaler', 'lr_scheduler'):
            assert banned not in code, (relative, banned)


def test_the_full_epoch_arm_script_uses_the_matched_configuration():
    source = open(os.path.join(REPO_ROOT, 'tools', 'exp_full3epoch_arm.sh'),
                  encoding='utf-8').read()
    for required in ('--objective_mode gap_completion', '--lambda_global 0',
                     '--lambda_unsaid 0', '--gap_anti_temperature 1.0',
                     '--said_loss_mode identifiable', '--said_feature_source residual',
                     '--epochs 3', '--max_steps 3648', '--lr_total_steps 3648',
                     '--warmup_length 200', '--batch_size 256', '--seed 0',
                     '--amp_dtype bf16', '--strict_manifest'):
        assert required in source, required
    # both arms must be launched from the same frozen initial state, i.e. no --resume
    assert '--resume' not in source
    # the forbidden method changes must not appear
    for banned in ('uss', 'USS', 'lambda_uss', 'texts_unsaid', 'centered_pooling', 'tau_sweep'):
        assert banned not in source, banned
