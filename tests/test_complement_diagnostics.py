"""Phase 3.0A.1c tests: complement emergence diagnostics.

Covers the phase-brief acceptance points:

* zero ``lambda_gap_discover`` / ``lambda_global_absorb`` are legal matched controls
* checkpoint ``phase`` / ``objective_mode`` metadata, and resume mismatch rejection
  (including "no silent resume of a legacy checkpoint into gap mode")
* patch-homogeneity metrics are finite, and correct on inputs with known geometry
* raw-pooling metrics are finite and expose what the L2 normalisation hides
* global-relation metrics are finite
* none of the diagnostics can reach a loss
* the legacy checkpoint / logging behaviour is unchanged
* a temperature sweep can run on one fixed batch with identical (I, C_S) and weights
"""
import json
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import complement_diagnostics as cd  # noqa: E402
from model.gap_completion import soft_anti_said_attention, unsaid_feature_from_attention  # noqa: E402
from test_gap_completion import build_model, gap_forward, make_batch  # noqa: E402

import train_salu  # noqa: E402

DIM = 8
PATCHES = 6


def parse(**overrides):
    argv = ['--lambda_global', '0', '--lambda_unsaid', '0']
    for key, value in overrides.items():
        argv.extend(['--' + key, str(value)])
    return train_salu.parse_args(argv)


# --------------------------------------------------------------------------- #
# patch homogeneity metrics
# --------------------------------------------------------------------------- #
def test_identical_patches_are_maximally_homogeneous():
    patches = torch.ones(3, PATCHES, DIM)
    patches[:, 1:] *= 2.0                        # still exactly parallel to patch 0
    global_feature = torch.randn(3, DIM)
    metrics = cd.patch_homogeneity_metrics(patches, global_feature)
    assert float(metrics['patch_pair_cosine_mean']) == pytest.approx(1.0, abs=1e-4)
    # the off-diagonal entries are all exactly 1, so the spread is fp32 noise
    assert float(metrics['patch_pair_cosine_std']) == pytest.approx(0.0, abs=1e-3)
    # mutually parallel unit vectors are indistinguishable across patches
    assert float(metrics['patch_to_global_cosine_std']) == pytest.approx(0.0, abs=1e-5)
    # one direction only: the centroid is the patches themselves -> no spread
    assert float(metrics['patch_centered_energy']) == pytest.approx(0.0, abs=1e-5)


def test_orthogonal_patches_are_minimally_homogeneous():
    torch.manual_seed(0)
    patches = torch.eye(PATCHES, DIM).unsqueeze(0).repeat(2, 1, 1)   # mutually orthogonal
    global_feature = torch.randn(2, DIM)
    metrics = cd.patch_homogeneity_metrics(patches, global_feature)
    assert float(metrics['patch_pair_cosine_mean']) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics['patch_pair_cosine_std']) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics['patch_centered_energy']) > 0.3


def test_patch_pair_mean_matches_a_brute_force_gram():
    """The closed form must equal the explicit off-diagonal Gram mean."""
    torch.manual_seed(1)
    patches = torch.randn(4, PATCHES, DIM)
    metrics = cd.patch_homogeneity_metrics(patches, torch.randn(4, DIM))
    unit = F.normalize(patches, dim=-1)
    gram = torch.bmm(unit, unit.transpose(1, 2))
    total = gram.sum(dim=(1, 2)) - torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    brute_force = total / float(PATCHES * (PATCHES - 1))
    assert torch.allclose(metrics['patch_pair_cosine_mean'], brute_force.mean(), atol=1e-6)
    # and the std against the explicit masked computation
    explicit = []
    for index in range(patches.shape[0]):
        matrix = gram[index].clone()
        matrix.fill_diagonal_(float('nan'))
        explicit.append(matrix[~torch.isnan(matrix)].std(unbiased=True))
    assert torch.allclose(metrics['patch_pair_cosine_std'],
                          torch.stack(explicit).mean(), atol=1e-5)


def test_patch_homogeneity_metrics_are_finite_including_degenerate_input():
    torch.manual_seed(2)
    for patches in (torch.randn(3, PATCHES, DIM),
                    torch.zeros(2, PATCHES, DIM),          # all-zero patches
                    torch.ones(2, 1, DIM)):                # single patch: no pairs
        metrics = cd.patch_homogeneity_metrics(patches, torch.randn(patches.shape[0], DIM))
        for key, value in metrics.items():
            assert torch.isfinite(torch.as_tensor(value)), key
        assert 0.0 <= float(metrics['patch_centered_energy'])


def test_patch_homogeneity_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        cd.patch_homogeneity_metrics(torch.randn(2, DIM), torch.randn(2, DIM))
    with pytest.raises(ValueError):
        cd.patch_homogeneity_metrics(torch.randn(2, PATCHES, DIM), torch.randn(2, PATCHES, DIM))


# --------------------------------------------------------------------------- #
# raw pooling metrics
# --------------------------------------------------------------------------- #
def test_raw_pool_metrics_expose_what_normalisation_hides():
    """Two different attentions can give the same unit z but very different raw pools."""
    torch.manual_seed(3)
    patches = torch.randn(2, PATCHES, DIM)
    concentrated = torch.zeros(2, PATCHES)
    concentrated[:, 0] = 1.0
    spread = torch.full((2, PATCHES), 1.0 / PATCHES)
    metrics = cd.raw_pooling_metrics(concentrated, spread, patches)
    for value in metrics.values():
        assert torch.isfinite(value)
    # the concentrated pool is one patch, the spread pool averages them: smaller norm
    assert float(metrics['said_raw_pool_norm']) > float(metrics['unsaid_raw_pool_norm'])
    assert -1.0 <= float(metrics['raw_pool_cosine']) <= 1.0


def test_identical_attention_gives_raw_pool_cosine_one():
    torch.manual_seed(4)
    patches = torch.randn(3, PATCHES, DIM)
    attention = torch.softmax(torch.randn(3, PATCHES), dim=-1)
    metrics = cd.raw_pooling_metrics(attention, attention, patches)
    assert float(metrics['raw_pool_cosine']) == pytest.approx(1.0, abs=1e-5)
    assert float(metrics['raw_pool_norm_ratio']) == pytest.approx(1.0, abs=1e-5)
    assert float(metrics['said_raw_pool_norm']) == pytest.approx(
        float(metrics['unsaid_raw_pool_norm']), abs=1e-6)


def test_raw_pool_metrics_reject_wrong_shapes():
    with pytest.raises(ValueError):
        cd.raw_pooling_metrics(torch.randn(2, PATCHES, 1), torch.randn(2, PATCHES),
                               torch.randn(2, PATCHES, DIM))


# --------------------------------------------------------------------------- #
# global relation metrics
# --------------------------------------------------------------------------- #
def test_global_relation_metrics_are_exact_cosines():
    torch.manual_seed(5)
    global_feature = torch.randn(4, DIM)
    said = torch.randn(4, DIM)
    unsaid = torch.randn(4, DIM)
    metrics = cd.global_relation_metrics(global_feature, said, unsaid)
    unit_g = F.normalize(global_feature, dim=-1)
    unit_s = F.normalize(said, dim=-1)
    unit_u = F.normalize(unsaid, dim=-1)
    assert torch.allclose(metrics['cos_global_said'], (unit_g * unit_s).sum(-1).mean(), atol=1e-6)
    assert torch.allclose(metrics['cos_global_unsaid'], (unit_g * unit_u).sum(-1).mean(), atol=1e-6)
    assert torch.allclose(metrics['cos_said_unsaid'], (unit_s * unit_u).sum(-1).mean(), atol=1e-6)
    for value in metrics.values():
        assert torch.isfinite(value)


# --------------------------------------------------------------------------- #
# the diagnostics are measurement only
# --------------------------------------------------------------------------- #
def test_diagnostics_never_depend_on_gradients_and_never_enter_a_loss():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    for key in cd.COMPLEMENT_DIAGNOSTIC_KEYS:
        value = out[key]
        assert torch.is_tensor(value), key
        assert value.requires_grad is False, key
        assert value.grad_fn is None, key
        assert torch.isfinite(value), key

    # L_total is unchanged whether the diagnostics are computed or not: the loss
    # expression only ever references the three official terms.
    terms = (out['loss_said'], out['loss_gap_discover'], out['loss_global_absorb'])
    expected = terms[0] + terms[1] + terms[2]
    assert torch.allclose(out['loss_total'], expected, atol=1e-6)
    out['loss_total'].backward()
    for name, parameter in model.said_router.named_parameters():
        assert parameter.grad is not None, name


def test_diagnostics_reach_the_temperature_sweep_writer():
    """Every key a sweep record needs is present in the model output."""
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    required = set(cd.COMPLEMENT_DIAGNOSTIC_KEYS) | {
        'said_attention_entropy', 'unsaid_attention_entropy',
        'said_unsaid_attention_overlap', 'said_unsaid_attention_jsd',
        'unsaid_novel_component_norm', 'gap_before_mean', 'gap_after_mean',
        'gap_reduction_mean', 'gap_closure_ratio_mean', 'gap_closure_positive_fraction',
        'said_unsaid_feature_cosine',
    }
    missing = sorted(key for key in required if out.get(key) is None)
    assert missing == []
    # and every one of them survives a JSON round trip as a real number
    record = train_salu.build_gap_log_fields(out)
    for key in required:
        assert isinstance(record[key], float), key
    assert json.loads(json.dumps(record, sort_keys=True))['patch_pair_cosine_mean'] is not None


# --------------------------------------------------------------------------- #
# temperature sweep identity: one batch, one set of weights, different tau
# --------------------------------------------------------------------------- #
def test_temperature_sweep_reuses_one_batch_and_only_changes_tau():
    torch.manual_seed(9)
    model = build_model()
    images, texts = make_batch()
    batch_sha = train_salu.batch_identity_sha256(images, texts)
    weights_before = {name: parameter.detach().clone()
                      for name, parameter in model.named_parameters()}

    records = {}
    for tau in (2.0, 1.0, 0.5, 0.25, 0.10):
        out = gap_forward(model, images, texts, gap_anti_temperature=tau)
        records[tau] = {
            'batch_sha256': train_salu.batch_identity_sha256(images, texts),
            'gap_anti_temperature': float(out['gap_anti_temperature']),
            'jsd': float(out['said_unsaid_attention_jsd']),
            'overlap': float(out['said_unsaid_attention_overlap']),
            'cos_said_unsaid': float(out['cos_said_unsaid']),
            'novel_norm': float(out['unsaid_novel_component_norm']),
            'closure': float(out['gap_closure_ratio_mean']),
            'gap_after': float(out['gap_after_mean']),
        }

    # identical batch and identical weights for every temperature, no optimizer step
    shas = {record['batch_sha256'] for record in records.values()}
    assert len(shas) == 1 and shas.pop() == batch_sha
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), weights_before[name]), name

    # and tau really changes the anti-Said side (a non-degenerate sweep)
    assert records[0.10]['jsd'] != records[1.0]['jsd']
    assert records[0.10]['overlap'] != records[1.0]['overlap']
    for tau, record in records.items():
        assert record['gap_anti_temperature'] == tau
        for key in ('jsd', 'overlap', 'cos_said_unsaid', 'novel_norm', 'closure', 'gap_after'):
            assert math.isfinite(record[key]), (tau, key)
        assert -1.0 <= record['cos_said_unsaid'] <= 1.0, tau


def test_batch_identity_sha_is_stable_and_content_addressed():
    torch.manual_seed(10)
    images = torch.randn(3, 3, 4, 4)
    texts = ['a cat.', 'a dog.', 'a bird.']
    first = train_salu.batch_identity_sha256(images, texts)
    assert first == train_salu.batch_identity_sha256(images, texts)
    assert first != train_salu.batch_identity_sha256(images, ['a cat.', 'a dog.', 'a bird'])
    assert first != train_salu.batch_identity_sha256(images + 1e-6, texts)
    assert len(first) == 64


# --------------------------------------------------------------------------- #
# checkpoint / resume objective metadata
# --------------------------------------------------------------------------- #
def test_gap_checkpoints_carry_the_phase30a_phase_and_objective():
    args = parse(objective_mode='gap_completion')
    phase, objective = train_salu.objective_checkpoint_metadata(args)
    assert phase == 'phase3.0a-gap-completion'
    assert objective == 'gap_completion'


def test_legacy_checkpoint_metadata_is_unchanged():
    args = train_salu.parse_args([])
    phase, objective = train_salu.objective_checkpoint_metadata(args)
    assert phase == 'phase2-said-only'
    assert objective is None                  # legacy checkpoints gain no new field


def test_resume_rejects_an_objective_mismatch():
    gap_args = parse(objective_mode='gap_completion')
    legacy_args = train_salu.parse_args([])

    # matching objectives resume
    assert train_salu.validate_resume_objective({'objective_mode': 'gap_completion'},
                                                gap_args) == 'gap_completion'
    assert train_salu.validate_resume_objective({'objective_mode': 'legacy'},
                                                legacy_args) == 'legacy'
    # mismatch is refused in both directions
    with pytest.raises(ValueError):
        train_salu.validate_resume_objective({'objective_mode': 'legacy'}, gap_args)
    with pytest.raises(ValueError):
        train_salu.validate_resume_objective({'objective_mode': 'gap_completion'}, legacy_args)


def test_resume_of_a_pre_phase30a_checkpoint():
    """No objective_mode metadata = legacy checkpoint."""
    gap_args = parse(objective_mode='gap_completion')
    legacy_args = train_salu.parse_args([])
    assert train_salu.validate_resume_objective({}, legacy_args) == 'legacy'
    assert train_salu.validate_resume_objective({'phase': 'phase2-said-only'},
                                                legacy_args) == 'legacy'
    # gap mode must never silently continue a legacy run
    with pytest.raises(ValueError):
        train_salu.validate_resume_objective({}, gap_args)
    with pytest.raises(ValueError):
        train_salu.validate_resume_objective({'phase': 'phase2-said-only'}, gap_args)


def test_saved_checkpoint_round_trips_its_objective(tmp_path):
    class _FakeDDP:
        def __init__(self, module):
            self.module = module

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    for objective in ('gap_completion', 'legacy'):
        args = parse(objective_mode=objective)
        path = os.path.join(str(tmp_path), 'ckpt_%s.pt' % objective)
        train_salu.save_checkpoint(path, _FakeDDP(model), optimizer, scaler, 7, 0, args,
                                   [1e-6, 1e-4], 3648)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['step'] == 7
        if objective == 'gap_completion':
            assert payload['phase'] == 'phase3.0a-gap-completion'
            assert payload['objective_mode'] == 'gap_completion'
            assert train_salu.validate_resume_objective(payload, args) == 'gap_completion'
        else:
            assert payload['phase'] == 'phase2-said-only'
            assert 'objective_mode' not in payload
            assert train_salu.validate_resume_objective(payload, args) == 'legacy'
        # saving always writes the CLIP-only companion
        assert os.path.exists(os.path.join(str(tmp_path),
                                           'clip_only_' + os.path.basename(path)))


def test_parse_step_list_argument():
    assert train_salu.parse_step_list('0,20,50,100') == [0, 20, 50, 100]
    assert train_salu.parse_step_list('') == []
    assert train_salu.parse_step_list(None) == []
    assert train_salu.parse_step_list([3]) == [3]
    assert train_salu.parse_args([]).grad_contribution_steps == [0]


# --------------------------------------------------------------------------- #
# per-term gradient diagnostics run on the training graph
# --------------------------------------------------------------------------- #
def test_grad_contribution_norms_build_their_own_graph():
    """They must not reuse the training graph (DDP frees it) and must leave buffers clean."""
    model = build_model()
    images, texts = make_batch()
    args = parse(objective_mode='gap_completion')
    out = gap_forward(model, images, texts)
    out['loss_total'].backward()                   # exactly as the training loop does
    full = train_salu.grad_norm_summary(model)
    for key in ('grad_norm_backbone', 'grad_norm_said_router', 'grad_norm_total'):
        assert math.isfinite(full[key]) and full[key] > 0.0, key
    assert full['grad_norm_said_router'] > 0.0     # L_said trains the router

    model.zero_grad(set_to_none=True)
    contributions = train_salu.grad_contribution_norms(model, images, texts, args, [0], 0)
    assert contributions, 'step 0 must be a diagnostic step by default'
    for name in ('gap', 'absorb'):
        value = contributions['grad_norm_%s_contribution' % name]
        assert math.isfinite(value) and value > 0.0, name
    # the absorb term has no path to the Said router (target detached, z_S frozen)
    assert contributions['grad_norm_absorb_contribution_router'] == 0.0
    assert contributions['grad_norm_gap_contribution_router'] == 0.0
    # and the buffers are left clean for the next real step
    assert all(parameter.grad is None for parameter in model.parameters())


def test_grad_contribution_norms_are_skipped_off_the_diagnostic_steps():
    model = build_model()
    images, texts = make_batch()
    args = parse(objective_mode='gap_completion')
    out = gap_forward(model, images, texts)
    out['loss_total'].backward()
    model.zero_grad(set_to_none=True)
    assert train_salu.grad_contribution_norms(model, images, texts, args, [20, 50, 100], 7) == {}
    assert train_salu.grad_contribution_norms(model, images, texts, args, [], 0) == {}


def test_grad_norm_summary_accepts_a_ddp_wrapped_model():
    """DDP exposes params through ``.module``; the summary must work on either form.

    The training loop passes the DDP wrapper, so ``grad_norm_summary`` and
    ``grad_contribution_norms`` must not assume a bare ``SALUModel``.
    """
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    out['loss_total'].backward()
    args = parse(objective_mode='gap_completion')

    class _Wrapper:
        """Minimal stand-in for DDP: attributes live behind ``.module``, as in PyTorch."""

        def __init__(self, module):
            self.module = module

        def __call__(self, *fargs, **fkwargs):
            return self.module(*fargs, **fkwargs)

        def forward(self, *fargs, **fkwargs):
            return self.module.forward(*fargs, **fkwargs)

        def zero_grad(self, set_to_none=True):
            return self.module.zero_grad(set_to_none=set_to_none)

        def parameters(self):
            return self.module.parameters()

        def no_sync(self):
            # DDP's context manager; the test never initialises a process group, so the
            # helper must fall back to a no-op instead of calling into the module
            raise AssertionError('no_sync must not be called without a process group')

    wrapped = _Wrapper(model)
    summary = train_salu.grad_norm_summary(wrapped)
    assert summary['grad_norm_total'] > 0.0
    assert summary['grad_norm_said_router'] > 0.0
    model.zero_grad(set_to_none=True)
    contributions = train_salu.grad_contribution_norms(wrapped, images, texts, args, [0], 0)
    assert contributions['grad_norm_gap_contribution'] > 0.0
    assert train_salu.unwrap_model(wrapped) is model
    assert train_salu.unwrap_model(model) is model


def test_grad_contribution_norms_use_a_fresh_context_per_term():
    """Both terms must run: a single reused (single-use) context would break the second.

    The parameters are also passed keyword-free positionally here, so an accidental
    signature/argument drift is caught rather than silently tolerated.
    """
    model = build_model()
    images, texts = make_batch()
    args = parse(objective_mode='gap_completion')
    model.zero_grad(set_to_none=True)
    contributions = train_salu.grad_contribution_norms(model, images, texts, args, [0], 0)
    assert set(contributions) == {
        'grad_norm_gap_contribution', 'grad_norm_gap_contribution_backbone',
        'grad_norm_gap_contribution_router', 'grad_norm_gap_contribution_weight',
        'grad_norm_absorb_contribution', 'grad_norm_absorb_contribution_backbone',
        'grad_norm_absorb_contribution_router', 'grad_norm_absorb_contribution_weight',
    }
    for key, value in contributions.items():
        assert math.isfinite(value), key


def test_ddp_no_sync_degrades_to_a_no_op_without_a_process_group():
    """Outside distributed init there is no no_sync; the helper must not call into it."""
    model = build_model()
    assert not (torch.distributed.is_available() and torch.distributed.is_initialized())
    for _ in range(3):                     # repeatable, i.e. a fresh context every time
        with train_salu.ddp_no_sync(model):
            pass
    with train_salu.ddp_no_sync(model):
        pass


def test_grad_contribution_with_zero_weight_still_measures_the_representation():
    """A zero weight is a matched control: the term is measured, just not in L_total."""
    model = build_model()
    images, texts = make_batch()
    args = parse(objective_mode='gap_completion', lambda_gap_discover='0')
    out = gap_forward(model, images, texts, lambda_gap_discover=0.0)
    assert torch.allclose(out['loss_total'], out['loss_said'] + out['loss_global_absorb'],
                          atol=1e-6)
    model.zero_grad(set_to_none=True)
    contributions = train_salu.grad_contribution_norms(model, images, texts, args, [0], 0)
    assert contributions['grad_norm_gap_contribution_weight'] == 0.0
    assert contributions['grad_norm_gap_contribution'] > 0.0
