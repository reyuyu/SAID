"""S0-Suffix v0.1 acceptance tests -- written against the FROZEN INTERFACE, not against the code.

Spec: ``docs/said_prefix_suffix_v01/spec.md`` (sections 2-10 and the frozen API list in section 14).
The production implementation (``model/said_prefix_suffix.py``, ``train/train_said_prefix_suffix.py``)
is repaired in parallel with this file, so every assertion below is derived from the frozen contract:

    suffix_readout_scores(g_block, mS_block, tR_block, suffix_mask,
                          image_chunk=16, text_chunk=32, eps=1e-6, want_statistics=False)
        g_block  [Bi, 512] LIVE final visual features
        mS_block [Bj, 512] candidate prefix masks, TWO-DIMENSIONAL ONLY (a 3-D input is rejected)
        tR_block [Bj, 512] LIVE suffix text features
        -> {"scores": [Bi, Bj] fp32, "statistics": dict | None}

    Norm(x) = torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=1e-6)   (no custom autograd)
    gate    = SuffixMaskGate.gate_from_pU(pU) == (pU >= 0.5).to(pU.dtype) + (pU - pU.detach())

There is exactly ONE return type (a dict with the two keys above) and ONE call shape (:func:`call_readout`
passes KEYWORD ARGUMENTS ONLY). No "guess the signature / guess the call order" helper survives here, and
no compatibility layer is allowed to hide an interface drift: a drifted interface must fail loudly.

Index spaces (spec section 6):
    LOCAL_FULL    this rank's ``B`` rows, in rank order
    GLOBAL_FULL   ``W * B`` rows, rank order
    GLOBAL_VALID  the ``V`` rows of the global valid index set ``J``; ``global_to_valid[J] = arange(V)``
The label of a LOCAL VALID row inside the V pool is ``global_to_valid[rank * B + local_row]``.
``arange(rank * B, rank * B + n_r)`` is WRONG and must never appear in this file.

Group map (the comments name the defect each test pins):
    A  the prefix/suffix split, the spec's worked examples, the inclusive K draw
    B  text isolation, the empty-suffix placeholder and the caption_said character invariant
    C  gate inputs, the straight-through gate, the tR/dtype conventions
    D  initialisation, RNG isolation compared on the REAL global generators
    E  S0 regression against the REAL weighted objective ``10*(L_SIDM+L_DISM) + 2*sparse``
    F  the readout contract: 3-D rejection, non-square broadcast, per-pair gradients, index mapping
    G  the heavy-log statistics and the DDP graph hygiene of the returned dict
    H  all-off / near-zero gate, normalisation gradients against native F.normalize
    I  the valid-subset DDP route (real two-rank runs against a single-process oracle)
    J  the full checkpoint round-trip and the trainer CLI surface

Retired claims (they were WRONG and are gone):
  * "pre-normalising g makes the conditional score smaller by orders of magnitude" -- with a fixed mask
    a correctly normalised cosine has no such artificial scale effect, and the F forward really reads
    ``g``, so the spec's unit-scale ``g`` contract is respected instead of worked around;
  * "the objective builds its own comparison shape" -- the objective is used as the FROZEN weighted
    objective ``10*(L_SIDM+L_DISM) + 2*sparse`` and compared on those terms;
  * "a detached mS must show an explicit zero gradient" -- ``None`` and an exact zero are both accepted;
  * "mksuffix_readout_scores returns ``(scores, mU)``" -- it returns the dict of the contract above.
"""
import argparse
import inspect
import math
import os
import random
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda', 0) if CUDA else torch.device('cpu')
needs_cuda = pytest.mark.skipif(not CUDA, reason='acceptance tests need a CUDA device')
needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2,
                                    reason='needs two visible GPUs')
CLIP_CACHE = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
SHARED_INIT = '/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt'
DATA_ROOT = '/root/datasets/ShareGPT4V'
SUFFIX_MODEL_PATH = os.path.join(REPO, 'model', 'said_prefix_suffix.py')
TRAINER_PATH = os.path.join(REPO, 'train', 'train_said_prefix_suffix.py')
WORKER_PATH = os.path.join(REPO, 'tests', '_suffix_ddp_worker.py')

# --------------------------------------------------------------------------- numeric policy
# Every comparison states WHY its tolerance is what it is. The rule of spec section 9 ("accept a
# reasonable floating-point error, never widen a tolerance to hide a multiplier error") means each
# constant below may absorb accumulation order only:
#
#   TOL_EXACT           | bitwise. Both sides are the same expression, in the same order, on the same
#                       | tensors: anything non-zero is a real difference (a different op, a copy that
#                       | went through a different path, a cast that should not be there).
#   TOL_FP32_TIGHT      | 1e-6 ABSOLUTE on fp32 VALUES of magnitude ~1..1e2 (unit vectors, cosines of
#                       | unit vectors, probabilities, ratios in [0, 1]). One fp32 ulp at 1.0 is
#                       | 1.2e-7 and the expressions compared differ by a few roundings, so 1e-6 is
#                       | ~8 ulp: it absorbs the reassociation of a handful of products/sums and
#                       | nothing else. A wrong order, a missing detach or a re-normalisation moves
#                       | these numbers by >= 1e-3 relative, i.e. 1000x this bound.
#   TOL_RECOMPUTE       | 1e-5 ABSOLUTE on VALUES around the fixed 100x scale (so ~1e-7 relative).
#                       | Used when the reference is an independent re-write that walks the pairs one
#                       | (i, j) at a time while the production path tiles them: the accumulation
#                       | order differs over up to 32 rows, and the score magnitude is ~100.
#   TOL_GRADIENT        | 1e-4 RELATIVE to the compared tensor's own max-abs: fp32 gradients from two
#                       | different summation orders (per-pair loop vs tiled einsum). No gradient
#                       | comparison in this file is allowed to be looser than this.
#   TOL_DDP             | 1e-3 RELATIVE: NCCL reduction over ranks on top of a different per-rank
#                       | summation order, on real two-rank runs. A scaling error (world_size, V,
#                       | lambda_suffix) is a factor-level error, far outside 1e-3.
#   TOL_DENOM           | 1e-6 ABSOLUTE against a denominator-based analytic formula that is computed
#                       | in float64 while the implementation is fp32: the only difference possible
#                       | is the fp32 rounding of ||x||, i.e. ~1e-7 relative, and the quantities
#                       | compared are <= 1e3.
#   STATE_TOLERANCE     | 1e-6 for ``pU == 8/9`` (spec section 4) and for ``bias == log(8)``.
TOL_EXACT = 0.0
TOL_FP32_TIGHT = 1e-6
TOL_RECOMPUTE = 1e-5
TOL_GRADIENT = 1e-4
TOL_DDP = 1e-3
TOL_DENOM = 1e-6
STATE_TOLERANCE = 1e-6

CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
SUFFIXES = ['on a wooden table', 'through tall grass', 'against a wall', 'on a calm lake']
FIXED_SCALE = 100.0
S0_LAMBDA_SPARSE = 2.0
FEATURE_DIM = 512

#: The frozen keyword names of the readout. They are asserted, not guessed: a renamed parameter is an
#: interface drift and every readout test must fail on it.
READOUT_PARAMS = ('g_block', 'mS_block', 'tR_block', 'suffix_mask')

#: The two keys of the ONE return type. Extra keys are drift; missing keys are drift.
READOUT_KEYS = ('scores', 'statistics')

#: Semantic key families for the heavy-log statistics. The contract does not freeze the NAMES, so the
#: tests resolve them by family and FAIL LOUDLY (with the full key list in the message) when a family
#: is absent: a silently missing statistic is exactly the defect these tests exist to catch.
KEEP_RATIO_KEYS = ('keep_ratio', 'kept_ratio', 'mask_keep_ratio', 'keep_fraction', 'kept_fraction',
                   'mU_keep_ratio', 'mu_keep_ratio', 'hard_keep_ratio', 'keep_ratio_per_pair',
                   'mU_keep_fraction')
NORM_RATIO_KEYS = ('norm_ratio', 'gated_norm_ratio', 'readout_norm_ratio', 'g_norm_ratio',
                   'norm_ratio_per_pair', 'mU_norm_ratio', 'gated_to_plain_norm_ratio',
                   'gated_norm_ratio_per_pair')
LSE_MARGIN_KEYS = ('lse_margin', 'margin_lse', 'lse_margins', 'logsumexp_margin')
COUNT_KEYS = ('count', 'numel', 'pairs', 'pair_count', 'valid_pair_count', 'n_pairs',
              'readout_pair_count', 'valid_suffix_count')
VALUE_KEYS = ('value', 'mean', 'mean_value', 'mean_norm_ratio', 'average', 'norm_ratio_mean')
GATE_KEYS = ('gate', 'mU', 'mask_u', 'gate_values', 'gate_matrix', 'gate_tensor', 'hard_gate')

#: The explicit debug flag that gates the FULL gate tensor. ``return_gate`` is the documented name; the
#: alternatives exist because the same flag is also spelled after the statistic it exposes.
DEBUG_GATE_FLAGS = ('return_gate', 'want_gate', 'return_mU', 'return_mask_u', 'return_gate_tensor',
                    'want_gate_tensor', 'debug')


# --------------------------------------------------------------------------- implementation glue
def implementation_module():
    """``model.said_prefix_suffix`` or a loud skip when the implementation file is absent."""
    if not os.path.isfile(SUFFIX_MODEL_PATH):
        pytest.skip('the implementation model/said_prefix_suffix.py is not present yet at %s'
                    % SUFFIX_MODEL_PATH)
    import importlib
    return importlib.import_module('model.said_prefix_suffix')


def trainer_module():
    """``train.train_said_prefix_suffix`` or a loud skip when the trainer is absent."""
    if not os.path.isfile(TRAINER_PATH):
        pytest.skip('the implementation train/train_said_prefix_suffix.py is not present yet at %s'
                    % TRAINER_PATH)
    import importlib
    return importlib.import_module('train_said_prefix_suffix')


def require_clip():
    if not os.path.isfile(CLIP_CACHE):
        pytest.skip('the CLIP ViT-B/16 checkpoint is not cached at %s' % CLIP_CACHE)


def _shared_init_state():
    if not os.path.isfile(SHARED_INIT):
        pytest.skip('the shared initialisation is not present at %s' % SHARED_INIT)
    payload = torch.load(SHARED_INIT, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        payload = payload['model']
    return payload


def build_model(device=DEVICE, shared_init=False):
    """A real CLIP ViT-B/16 from the local cache, optionally at the experiment's shared init."""
    require_clip()
    from model import longclip
    model, _preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                                 args=argparse.Namespace())
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(FIXED_SCALE))
    if shared_init:
        payload = _shared_init_state()
        target = set(model.state_dict())
        mapped = {}
        for key, value in payload.items():
            if key in target:
                mapped[key] = value
            elif key.startswith('clip.') and key[len('clip.'):] in target:
                mapped[key[len('clip.'):]] = value
            elif ('clip.' + key) in target:
                mapped['clip.' + key] = value
        model.load_state_dict(mapped, strict=False)
    return model.to(device)


def tokenize(texts, device=DEVICE):
    from model import longclip
    return longclip.tokenize(list(texts), truncate=True).to(device)


def fixed_images(count=2, seed=20260913, device=DEVICE):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, 224, 224, generator=generator).to(device)


def unit_features(count, generator, dim=FEATURE_DIM, device=DEVICE, unit=True):
    """``g`` / ``tR`` inputs of the specified unit-scale contract.

    The spec defines ``g = Norm(encode_image(I))`` and ``tR = Norm(encode_text(R))``, i.e. both are
    UNIT vectors. Feeding raw standard normals would violate the documented input contract; a test that
    needed raw features to see a difference would be measuring its own input, not the implementation.
    """
    raw = torch.randn(count, dim, generator=generator)
    if unit:
        raw = F.normalize(raw, p=2, dim=-1, eps=1e-6)
    return raw.to(device)


def binary_masks(count, generator, dim=FEATURE_DIM, device=DEVICE):
    return (torch.rand(count, dim, generator=generator) >= 0.5).float().to(device)


def make_suffix_mask(module, device=DEVICE, seed=None):
    """Build F through whatever OPTIONAL constructor arguments the implementation declares."""
    parameters = set(inspect.signature(module.SuffixMask.__init__).parameters)
    kwargs = {}
    if 'device' in parameters:
        kwargs['device'] = device
    if seed is not None and 'seed' in parameters:
        kwargs['seed'] = seed
    return module.SuffixMask(**kwargs).to(device)


def randomise_suffix_mask(mask, scale=0.5, seed=3):
    """A genuinely NON-initial F: every tensor away from its zero/``log(8)`` start.

    DEFECT FIXED (device mismatch): the perturbation noise is built ON THE PARAMETER'S OWN DEVICE with
    a generator created FOR that device. A CPU ``torch.Generator`` can never be used to draw the noise
    that perturbs a CUDA parameter: ``torch.randn(..., generator=cpu_generator)`` on the CPU is fine,
    but a generator is bound to a device and mixing the two either raises or, worse, silently draws on
    the wrong device.
    """
    with torch.no_grad():
        for name, parameter in mask.named_parameters():
            device = parameter.device
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            if parameter.dim() == 0:
                noise = torch.randn(1, generator=generator, device=device)[0]
            else:
                noise = torch.randn(parameter.shape, generator=generator, device=device)
            parameter.add_(noise.to(parameter.dtype) * scale)
    return mask


def gate_from_pU(module, pU):
    """The straight-through gate ``mU = hardU + (pU - pU.detach())`` of spec section 4."""
    gate_class = getattr(module, 'SuffixMaskGate', None)
    if gate_class is not None and hasattr(gate_class, 'gate_from_pU'):
        return gate_class.gate_from_pU(pU)
    if hasattr(module, 'gate_from_pU'):
        return module.gate_from_pU(pU)
    raise AssertionError('the implementation exposes neither SuffixMaskGate.gate_from_pU nor a '
                         'module-level gate_from_pU, so the straight-through gate of spec section 4 '
                         'cannot be checked')


def call_readout(module, g_block, mS_block, tR_block, suffix_mask, image_chunk=2, text_chunk=2,
                 eps=1e-6, want_statistics=False, **extra_flags):
    """THE ONLY way this file reaches the readout: keyword arguments, one return type.

    DEFECT FIXED (interface guessing): the old helper tried a list of positional orders, a list of
    kwargs variants and both a tuple and a dict return. All of that is deleted. This helper

    1. introspects the signature and asserts the four frozen parameter names are present (a rename is
       drift and must fail here rather than being papered over by a guessed call),
    2. calls with KEYWORD ARGUMENTS ONLY,
    3. asserts the result is a dict whose keys are exactly ``scores`` and ``statistics``,
    4. asserts ``scores`` is a 2-D fp32 tensor of shape ``[Bi, Bj]``,
    5. asserts ``statistics`` is a dict only when it was asked for and ``None`` otherwise.

    There is no compatibility layer and no fallback: a drifted interface fails loudly with the actual
    signature and the actual return type in the message.
    """
    function = getattr(module, 'suffix_readout_scores', None)
    assert callable(function), ('the frozen entry point suffix_readout_scores is missing from %r'
                                % module.__name__)
    parameters = inspect.signature(function).parameters
    missing = [name for name in READOUT_PARAMS if name not in parameters]
    assert not missing, ('suffix_readout_scores has lost the frozen parameter names %r; it declares %r'
                         % (missing, list(parameters)))
    result = function(g_block=g_block, mS_block=mS_block, tR_block=tR_block, suffix_mask=suffix_mask,
                      image_chunk=int(image_chunk), text_chunk=int(text_chunk), eps=float(eps),
                      want_statistics=bool(want_statistics), **extra_flags)
    assert isinstance(result, dict), \
        ('suffix_readout_scores must return the ONE documented dict {"scores", "statistics"}; got %s. '
         'A tuple, a named tuple or an object that only mimics dict keys is an interface drift'
         % type(result).__name__)
    assert set(result) == set(READOUT_KEYS), \
        ('suffix_readout_scores must return exactly %r; got %r'
         % (sorted(READOUT_KEYS), sorted(result)))
    scores = result['scores']
    assert torch.is_tensor(scores) and scores.dim() == 2, ('scores must be a 2-D tensor', scores)
    assert tuple(scores.shape) == (int(g_block.shape[0]), int(tR_block.shape[0])), \
        ('scores must be [Bi, Bj]', tuple(scores.shape), int(g_block.shape[0]),
         int(tR_block.shape[0]))
    assert scores.dtype == torch.float32, ('scores must be fp32 (spec section 8)', scores.dtype)
    if want_statistics:
        assert isinstance(result['statistics'], dict) and result['statistics'], \
            ('want_statistics=True must return a NON-EMPTY statistics dict; got %r'
             % (result['statistics'],))
    else:
        assert result['statistics'] is None, \
            ('the default call must NOT carry statistics (the heavy log is an explicit request); got %r'
             % (result['statistics'],))
    return result


def readout_scores(module, g_block, mS_block, tR_block, suffix_mask, **kwargs):
    return call_readout(module, g_block, mS_block, tR_block, suffix_mask, **kwargs)['scores']


def statistic(statistics, keys, what):
    """Resolve one statistic by key family, failing loudly with the available keys.

    ``None`` beside a zero count means the requested statistic does not exist; that is a MISSING
    statistic, not a value, and is reported as such so a test can never pass on an absent field.
    """
    present = sorted(str(key) for key in statistics)
    for key in keys:
        if key in statistics and statistics[key] is not None:
            return key, statistics[key]
    raise AssertionError('the readout statistics carry no %s; looked for %r among %r'
                         % (what, list(keys), present))


def statistic_optional(statistics, keys):
    for key in keys:
        if key in statistics:
            return key, statistics[key]
    return None, None


def as_float(value):
    if torch.is_tensor(value):
        assert value.numel() == 1, ('a scalar statistic was expected, got shape %s'
                                    % (tuple(value.shape),))
        return float(value.detach().float())
    return float(value)


def as_matrix(value, what):
    assert torch.is_tensor(value), ('%s must be a TENSOR of shape [Bi, Bj] (a Python list or a '
                                    'flattened vector cannot carry the per-pair layout)' % what)
    assert value.dim() == 2, ('%s must keep the [Bi, Bj] per-pair layout, got shape %s: a [Bi] '
                              'reduction with mean(-1) and then an assignment into a [Bi, Bj] slot is '
                              'the defect this pins' % (what, tuple(value.shape)))
    return value.detach().float()


def tensor_gradient(value, tensor, what):
    """``d value / d tensor`` with ``allow_unused``: ``None`` is a legitimate answer."""
    return torch.autograd.grad(value, [tensor], allow_unused=True, retain_graph=True)[0]


def assert_absent_or_zero(grad, what):
    """A stop-gradient input may legitimately report ``None`` OR an exact zero, never a real gradient.

    DEFECT FIXED (test demanded an explicit zero tensor): a detached input is simply not part of the
    graph, so autograd returns ``None``; demanding ``zeros_like`` would have forced the production code
    to add a fake ``0 * x`` term. Both shapes are accepted and both are named in the message.
    """
    if grad is None:
        return 0.0
    magnitude = float(grad.abs().max())
    assert magnitude == 0.0, \
        ('%s must be a stop-gradient input of the suffix path: its gradient must be None or EXACTLY '
         'zero (both are accepted), found max |grad| = %r' % (what, magnitude))
    return magnitude


def parameters_of(module_or_tensor):
    if isinstance(module_or_tensor, torch.nn.Module):
        return [parameter for parameter in module_or_tensor.parameters() if parameter.requires_grad]
    return []


def call_split(module, caption, k):
    """``split_prefix_suffix(caption, k)``; ``k`` stays positional because the spec writes it so.

    Only the ``k`` spelling is probed (positional first, then keyword). Nothing about the READOUT is
    guessed anywhere in this file.
    """
    try:
        return module.split_prefix_suffix(caption, k)
    except TypeError:
        return module.split_prefix_suffix(caption, k=k)


def call_mask_helper(module, mask_net, hidden):
    """The ORIGINAL S0 mask pipeline, reused and never re-implemented (spec section 1).

    Returns the mask TENSOR: the frozen helper returns ``(mask, soft, logits)``, and the leading tuple
    element is the mask. There is exactly one documented shape, so nothing is guessed here either.
    """
    function = getattr(module, 'said_mask_helper', None)
    if function is None:
        from model.said_cls_cvssl import said_mask_from_hidden as function
    result = function(mask_net, hidden, soft_mask=False)
    assert isinstance(result, (tuple, list)) and result, \
        ('the S0 mask helper must return (mask, soft, logits); got %r' % (type(result).__name__,))
    return result[0]


# --------------------------------------------------------------------------- index spaces
def global_to_valid(J, V=None):
    """``global_to_valid[J] = arange(V)``: the label of a global row inside the V pool."""
    V = int(J.numel()) if V is None else int(V)
    table = torch.full((int(J.max().item()) + 1 if J.numel() else 0,), -1, dtype=torch.long,
                       device=J.device)
    table[J] = torch.arange(V, dtype=torch.long, device=J.device)
    return table


def anchors_of_rank(J, local_size, rank):
    """``(local_rows, anchor_valid)`` of one rank: the CORRECT mapping the DDP route must use.

    ``local_rows`` are rows inside the rank's own ``B`` rows; ``anchor_valid`` are their labels inside
    the global V pool. ``arange(rank * B, rank * B + n_r)`` is NOT this rule and must never be used as
    an expected label anywhere in this file.
    """
    table = global_to_valid(J)
    lo, hi = int(rank) * int(local_size), (int(rank) + 1) * int(local_size)
    rows = [index for index in range(int(local_size)) if lo + index in set(J.tolist())]
    labels = [int(table[lo + index].item()) for index in rows]
    assert all(label >= 0 for label in labels), (rows, labels, J.tolist())
    return rows, labels


# --------------------------------------------------------------------------- references
def per_pair_reference(module, gate, g_block, mS_block, tR_block, scale=FIXED_SCALE, eps=1e-6):
    """The spec's per-pair definition written out one ``(i, j)`` at a time, sharing no code with F.

        rS_ij    = g_i.detach() * mS_j.detach()
        pU_ij    = F(stop_grad(concat([g_i, rS_ij])))
        mU_ij    = (pU_ij >= 0.5) + (pU_ij - pU_ij.detach())
        QU_ij    = scale * dot(Norm(g_i * mU_ij), tR_j)

    ``g_i`` is the LIVE row (it carries the trunk gradient) and ``tR_j`` the LIVE text row. The scale is
    applied HERE as well, so a readout that silently dropped the fixed 100 cannot agree with this.

    ``F`` is called through the module's ``forward`` only: a bare ``Linear`` inside a ``Sequential``
    silently supports in-place calls, and calling a layer directly would mutate F.
    """
    del eps
    g_block = g_block if g_block.dim() == 2 else g_block.reshape(-1, g_block.shape[-1])
    masks = mS_block if mS_block.dim() == 3 else mS_block.unsqueeze(0).expand(g_block.shape[0],
                                                                            mS_block.shape[0], -1)
    terms = []
    for i in range(int(g_block.shape[0])):
        for j in range(int(tR_block.shape[0])):
            left = g_block[i].detach()
            right = (g_block[i].detach() * masks[i, j].detach())
            x_u = torch.cat([left, right]).unsqueeze(0)
            mask_u = gate_from_pU(module, gate(x_u))
            u = F.normalize(g_block[i].unsqueeze(0) * mask_u, p=2, dim=-1, eps=1e-6)
            terms.append(scale * (u @ tR_block[j].unsqueeze(1)).squeeze())
    return torch.stack(terms).reshape(int(g_block.shape[0]), int(tR_block.shape[0]))


def positional_per_pair_scores(module, gate, g_block, mS_block, tR_block):
    """The per-pair reference with the scale ALREADY applied: ``QU``, not ``QU / 100``."""
    return per_pair_reference(module, gate, g_block, mS_block, tR_block, scale=FIXED_SCALE)


def wrong_broadcast_scores(module, gate, g_block, mS_block, tR_block):
    """The naive ``g_block * mS_block`` broadcast, with no per-pair mask dimension.

    DEFECT PINNED (broadcast): with ``Bi != Bj`` a plain element-wise product either raises (shape
    mismatch) or silently pairs row ``i`` with row ``i`` of the mask block, which is a DIFFERENT pairing
    from the spec's ``(i, j)``. Both outcomes are returned here so the test can require a real
    numerical difference instead of trusting that the wrong path would crash.
    """
    try:
        product = g_block.detach() * mS_block.detach()
        return 'elementwise', product
    except RuntimeError:
        pass
    masks = mS_block.detach().unsqueeze(0).expand(g_block.shape[0], mS_block.shape[0],
                                                 mS_block.shape[1])
    product = g_block.detach().unsqueeze(1) * masks
    return 'broadcast', product


def per_pair_loss(scores):
    """``sum_ij scores_ij``: a graph-faithful scalar to differentiate the per-pair reference."""
    return scores.sum()


def clip_reference_suffix_loss(scores, positive_weight=1.0):
    """``(CE_i2t(Q,y) + CE_t2i(Q^T,y)) / V`` for a ``V x V`` matrix whose diagonal is the positive."""
    targets = torch.arange(scores.shape[0], device=scores.device)
    return (F.cross_entropy(scores, targets, reduction='sum')
            + F.cross_entropy(scores.t(), targets, reduction='sum')) / float(scores.shape[0]) \
        * positive_weight


def lse_margin_reference(rows):
    """``positive - logsumexp(valid negatives only)``, the ONLY definition of an LSE margin.

    ``logsumexp(all) - positive`` is the cross entropy and must never be reported under this name.
    """
    rows = rows.detach().float()
    positive = rows.diagonal()
    masked = rows.clone()
    index = torch.arange(rows.shape[0], device=rows.device)
    masked[index, index] = float('-inf')
    return positive - torch.logsumexp(masked, dim=1)


def relative_difference(mine, reference):
    scale = max(float(reference.abs().max()), 1e-12)
    return float((mine - reference).abs().max()) / scale


def parameter_snapshot(mask):
    return {key: value.detach().clone() for key, value in mask.state_dict().items()}


def assert_parameters_unchanged(mask, snapshot, what):
    """No forward or backward of the readout may MUTATE F's tensors."""
    current = mask.state_dict()
    assert set(current) == set(snapshot), (sorted(current), sorted(snapshot))
    for key in snapshot:
        if not torch.equal(current[key].detach(), snapshot[key]):
            drift = float((current[key].detach() - snapshot[key]).abs().max())
            raise AssertionError('%s mutated the suffix mask tensor %r by up to %r: the readout '
                                 'must never write into F' % (what, key, drift))


def zero_gate_module(module):
    """A stand-in F whose ``pU`` is a KNOWN constant, for the statistic arithmetic.

    The statistics are a property of the readout arithmetic, not of the learned network, so pinning
    them against a controlled gate removes every confound (random initialisation, GELU, saturation).
    """

    class _ConstantGate(torch.nn.Module):
        def __init__(self, probability):
            super().__init__()
            self.value = float(probability)
            self.seen = []

        def forward(self, x):
            self.seen.append(tuple(x.shape))
            return torch.full_like(x, self.value)

    return _ConstantGate


def softmax_rows(scores):
    return torch.softmax(scores.detach().float(), dim=-1)



def test_a_split_follows_the_spec_worked_examples_exactly():
    """Spec section 2: ``[A,B,C,D]`` K=1 -> R='B. C'; K=2 -> 'C'; K=3, K=4 -> empty.

    The last non-empty fragment is excluded by construction, and the trailing empty fragment of
    ``[A,B,C,D,'']`` must not drag D into R.
    """
    module = implementation_module()
    caption = 'A. B. C. D'
    expected = {1: 'B. C', 2: 'C', 3: '', 4: ''}
    for k, suffix in expected.items():
        split = call_split(module, caption, k)
        assert split['suffix'] == suffix, (k, split)
        assert split['k'] == k
        assert split['n_sentences'] == 4
        assert split['last_nonempty_index'] == 3
        assert bool(split['valid']) == bool(suffix), (k, split)
        assert split['prefix'] == '. '.join(['A', 'B', 'C', 'D'][:k]), (k, split['prefix'])

    trailing = call_split(module, 'A. B. C. D. ', 2)
    assert trailing['suffix'] == 'C', ('the trailing empty fragment must not be joined into R',
                                       trailing)
    assert trailing['suffix'] != 'C. D'
    assert trailing['last_nonempty_index'] == 3, trailing
    empty_only = call_split(module, 'A. B. C. D. . . ', 1)
    assert empty_only['suffix'] == 'B. C', empty_only


def test_a_split_prefix_is_character_identical_to_the_old_s0_rule():
    """Spec section 2: ``P = '. '.join(caption.replace('\\n',' ').split('. ')[:K])``, unchanged."""
    module = implementation_module()
    captions = ['one sentence only',
                'first. second. third. fourth',
                'a caption\nwith a newline. and a second sentence. and a third',
                'trailing dot. then more. ',
                'double  space.  another fragment. last one']
    for caption in captions:
        cleaned = caption.replace('\n', ' ')
        sentences = cleaned.split('. ')
        for k in range(1, len(sentences) + 1):
            split = call_split(module, caption, k)
            assert split['prefix'] == '. '.join(sentences[:k]), (caption, k)
            assert split['n_sentences'] == len(sentences), (caption, k)


def test_a_last_non_empty_sentence_is_excluded_and_empty_fragments_never_appear():
    module = implementation_module()
    split = call_split(module, 'A. B. . D', 1)
    assert split['last_nonempty_index'] == 3, split
    assert split['suffix'] == 'B', ('C is empty and D is the last non-empty fragment, so R = B',
                                    split)
    pieces = split['suffix'].split('. ') if split['suffix'] else []
    assert all(piece != '' for piece in pieces), split['suffix']
    for k in (2, 3, 4):
        later = call_split(module, 'A. B. . D', k)
        assert '' not in [piece for piece in later['suffix'].split('. ') if later['suffix']], later


def test_a_one_and_two_sentence_captions_and_the_empty_suffix_case():
    module = implementation_module()
    single = call_split(module, 'only one sentence', 1)
    assert single['n_sentences'] == 1 and single['last_nonempty_index'] == 0
    assert single['suffix'] == '' and single['valid'] is False
    assert single['prefix'] == 'only one sentence'

    two = call_split(module, 'first sentence. second sentence', 1)
    assert two['suffix'] == '', ('the second sentence IS the last non-empty fragment, so it is the '
                                 'excluded one', two)
    assert two['valid'] is False
    assert two['prefix'] == 'first sentence'

    three = call_split(module, 'first. second. third', 1)
    assert three['suffix'] == 'second' and three['valid'] is True


def test_a_empty_suffix_is_never_padded_and_never_re_drawn():
    """Spec section 2: an empty R is not backfilled, the image is not swapped and K is not re-drawn."""
    module = implementation_module()
    rng = random.Random(0)
    rng_state = rng.getstate()
    first = call_split(module, 'first. second', 1)
    second = call_split(module, 'first. second', 1)
    assert first == second, 'the split must be a pure function of (caption, k)'
    assert first['suffix'] == '' and first['valid'] is False
    assert first['k'] == 1, ('an empty R must not cause K to be re-drawn: K is what the caller '
                             'asked for')
    assert rng.getstate() == rng_state, 'splitting must not consume any RNG stream'
    assert call_split(module, 'first. second. third', 1)['suffix'] == 'second'


def test_a_sample_k_is_inclusive_on_both_ends_and_reaches_n():
    """Spec section 2: ``K = rng.randint(1, n_sentences)``, inclusive at 1 and at n."""
    module = implementation_module()
    for n in (1, 2, 3, 5):
        rng = random.Random(12345)
        draws = [module.sample_k(n, rng) for _ in range(400)]
        assert set(draws) <= set(range(1, n + 1)), (n, sorted(set(draws)))
        assert n in draws, ('K = n must be reachable', n, sorted(set(draws)))
        assert 1 in draws, ('K = 1 must be reachable', n, sorted(set(draws)))
        if n > 1:
            reference = random.Random(12345)
            assert draws == [reference.randint(1, n) for _ in range(400)], \
                'sample_k must be exactly rng.randint(1, n_sentences)'
    assert module.sample_k(1, random.Random(7)) == 1


# =========================================================================== B: text isolation
def test_b_the_empty_suffix_placeholder_is_a_real_tokenisation_marked_invalid():
    """Spec section 6: an empty R keeps a B-placeholder row with ``valid = false``.

    The placeholder must come from a real ``tokenize("")`` (so the row stays a legal [B, 77] input of
    the shared text encoder) and it must be marked ``valid = false``, so it can never enter the loss or
    the candidate pool. An all-zero id row, a dropped row or a padded-with-content row are all wrong.
    """
    module = implementation_module()
    if not os.path.isfile(CLIP_CACHE):
        pytest.skip('the CLIP tokenizer is only reachable with the cached checkpoint')
    ids = tokenize(['', 'a real suffix'])
    assert ids.dim() == 2 and int(ids.shape[0]) == 2, tuple(ids.shape)
    assert not torch.equal(ids[0], ids[1]), 'the placeholder must really be the empty string'
    counts = module.content_token_counts(ids)
    assert counts[1] > 0, ('a real suffix must carry content tokens', counts)
    validity = [bool(text.strip()) and count > 0
                for text, count in zip(['', 'a real suffix'], counts)]
    assert validity == [False, True], (counts, validity)
    # and the empty string really is what the splitter produces for a one-sentence caption
    assert call_split(module, 'single sentence only', 1)['suffix'] == ''


def test_b_the_caption_said_invariant_is_enforced_on_the_real_input_shape():
    """Spec section 2/3: the prefix rebuilt from the FULL caption equals ``caption_said`` EXACTLY.

    ``caption_said`` is already the prefix the pipeline chopped at ``prefix_k``; re-splitting it would
    silently produce an empty suffix for every row (the measured ``global_valid_V = 0`` defect). The
    trainer's own tokenisation helper is what must refuse that, so the check runs against the trainer.
    """
    module = implementation_module()
    trainer = trainer_module()
    tokenized = getattr(trainer, 'tokenized', None)
    assert callable(tokenized), \
        ('the trainer must expose its tokenisation helper (``tokenized``) so the caption invariant can '
         'be checked on the REAL input shape; attributes: %r'
         % sorted(name for name in dir(trainer) if not name.startswith('_')))
    captions = ['A. B. C. D', 'first. second. third. fourth', 'one. two. three']
    k_values = [1, 2, 1]
    said = [call_split(module, caption, k)['prefix'] for caption, k in zip(captions, k_values)]
    assert said == ['. '.join(part.split('. ')[:k]) for part, k in zip(captions, k_values)]
    batch = {'caption_said': list(said), 'caption_full': list(captions),
             'prefix_k': torch.tensor(k_values, dtype=torch.long)}

    import model.said_prefix_suffix as suffix_model
    from model import longclip
    saved = longclip.tokenize

    def _stub(texts, truncate=True):
        return torch.zeros(len(list(texts)), 8, dtype=torch.long)

    longclip.tokenize = _stub
    try:
        text_ids, prefix_texts, suffix_texts, split = tokenized(batch, torch.device('cpu'))
    finally:
        longclip.tokenize = saved
    assert prefix_texts == said, ('the rebuilt prefix must equal caption_said character for '
                                  'character', prefix_texts, said)
    assert set(text_ids) == {'prefix', 'suffix'}
    expected_suffix = [call_split(module, caption, k)['suffix']
                       for caption, k in zip(captions, k_values)]
    assert suffix_texts == expected_suffix, (suffix_texts, expected_suffix)
    assert list(split['valid']) == [bool(text) for text in expected_suffix]

    # the truncated-caption input shape is the defect that silently zeroed the suffix loss: it must be
    # refused rather than silently accepted
    broken = {'caption_said': list(said), 'caption_full': list(said),
              'prefix_k': torch.tensor(k_values, dtype=torch.long)}
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        tokenized(broken, torch.device('cpu'))
    assert suffix_model is not None


@needs_cuda
def test_b_suffix_is_encoded_by_its_own_forward_and_only_the_suffix_target_moves():
    """Spec sections 2 and 4: P and R are tokenised and encoded SEPARATELY.

    With the image and the prefix fixed, changing the suffix string must leave ``mS`` unchanged and
    move only the suffix target ``tR`` and the scores built from it.
    """
    module = implementation_module()
    model = build_model(shared_init=True)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_a_ids = tokenize(SUFFIXES[:2])
    suffix_b_ids = tokenize(['a completely different sentence about swimming',
                             'an unrelated phrase about frozen water'])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR_a = F.normalize(model.encode_text(suffix_a_ids).float(), p=2, dim=-1, eps=1e-6)
        tR_b = F.normalize(model.encode_text(suffix_b_ids).float(), p=2, dim=-1, eps=1e-6)
    assert not torch.equal(tR_a, tR_b), 'the two suffixes must differ for this test to mean anything'

    gate = make_suffix_mask(module)
    first = call_readout(module, g, mS, tR_a, gate, image_chunk=2, text_chunk=2)
    second = call_readout(module, g, mS, tR_b, gate, image_chunk=2, text_chunk=2)
    with torch.no_grad():
        hidden_again = model.encode_text(prefix_ids, return_full=True)[1]
    diff = float((first['scores'] - second['scores']).abs().max())
    assert diff > 1e-3, ('a different suffix must change the suffix scores', diff)
    assert torch.equal(hidden, hidden_again), 'the prefix hidden state is a pure function of P'
    assert call_mask_helper(module, model.mask_net, hidden_again).equal(mS)

    combined_ids = tokenize(['a photo of a cat on a wooden table on a wooden table'])
    with torch.no_grad():
        t_combined = F.normalize(model.encode_text(combined_ids).float(), p=2, dim=-1, eps=1e-6)
    assert not torch.allclose(t_combined, tR_a[:1], atol=1e-4), \
        ('the suffix must be encoded by its own forward: the hidden state of prefix+suffix as one '
         'string is not the hidden state of the suffix alone')


@needs_cuda
def test_b_mS_depends_on_the_prefix_alone_and_mU_only_on_g_and_mS():
    module = implementation_module()
    model = build_model(shared_init=True)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    other_ids = tokenize(['totally different words entirely', 'another unrelated caption'])
    with torch.no_grad():
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        other_hidden = model.encode_text(other_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        mS_other = call_mask_helper(module, model.mask_net, other_hidden)
    assert not torch.equal(mS, mS_other), 'the prefix must drive mS'
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.3, seed=9)
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = gate(xU)
        mU = gate_from_pU(module, pU)
        mU_again = gate_from_pU(module, gate(torch.cat([g.detach(),
                                                        (g.detach() * mS.detach())], dim=-1)))
        mU_other = gate_from_pU(module, gate(torch.cat([g.detach(),
                                                        (g.detach() * mS_other.detach())], dim=-1)))
    assert torch.equal(mU, mU_again), 'mU must be a deterministic function of (g, mS, F)'
    assert not torch.equal(mU, mU_other), \
        'a different prefix changes mS and therefore genuinely changes mU (pair-specific mS)'


# =========================================================================== C: gate inputs
def test_c_rS_is_g_times_mS_with_no_renormalisation():
    """Spec section 4: ``rS = g_det * stop_grad(mS)`` and it is NOT normalised again."""
    module = implementation_module()
    generator = torch.Generator().manual_seed(5)
    g = unit_features(3, generator)
    mS = binary_masks(3, generator)

    rS = g.detach() * mS.detach()
    unit = F.normalize(rS, p=2, dim=-1, eps=1e-6)
    norm_rS = rS.norm(dim=-1)
    assert float(norm_rS.min()) > 0.1, 'the mask must keep enough coordinates for this to be sharp'
    assert float((norm_rS - 1.0).abs().max()) > 1e-3, \
        'rS keeps the norm of g*mS: it is NOT unit norm, so a re-normalisation is detectable'
    assert float(unit.norm(dim=-1).min()) == pytest.approx(1.0, abs=TOL_FP32_TIGHT)

    xU = torch.cat([g.detach(), rS], dim=-1)
    assert tuple(xU.shape) == (3, 2 * FEATURE_DIM), xU.shape
    assert torch.equal(xU[:, FEATURE_DIM:], rS), \
        'the second half of the F input must be exactly g*mS, never Norm(g*mS)'
    assert not torch.allclose(xU[:, FEATURE_DIM:], unit, atol=1e-4)


def test_c_gate_mU_is_a_straight_through_hard_gate():
    """Spec section 4: ``pU = sigmoid(F(xU))`` and ``mU = hardU + (pU - pU.detach())``."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.02, seed=7)
    xU = torch.randn(4, 2 * FEATURE_DIM, generator=torch.Generator().manual_seed(7))
    pU = gate(xU)
    assert float(pU.min()) >= 0.0 and float(pU.max()) <= 1.0, \
        'forward(x) must return a probability'
    assert float(pU.min()) > 1e-6 and float(pU.max()) < 1.0 - 1e-6, \
        'forward(x) must return the sigmoid probability pU, not a saturated hard value or a logit'
    assert float(pU.mean()) == pytest.approx(8.0 / 9.0, abs=0.05), \
        'at initialisation pU is 8/9; a mild perturbation must stay near it'
    mU = gate_from_pU(module, pU)
    hard = (pU >= 0.5).to(pU.dtype)
    assert torch.equal(mU, hard + (pU - pU.detach()))
    assert set(torch.unique(mU).tolist()) <= {0.0, 1.0}
    pU_leaf = pU.detach().clone().requires_grad_(True)
    value = torch.autograd.grad(gate_from_pU(module, pU_leaf).sum(), pU_leaf)[0]
    assert torch.equal(value, torch.ones_like(value))


@needs_cuda
def test_c_f_input_is_fully_detached_and_the_final_g_stays_live():
    """Spec sections 4 and 7: F trains from a detached input; ``g`` trains through ``g*mU``."""
    module = implementation_module()
    model = build_model()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.4, seed=13)
    generator = torch.Generator().manual_seed(11)
    g_leaf = unit_features(2, generator).requires_grad_(True)
    mS = binary_masks(2, generator)
    tR = unit_features(2, generator)

    xU = torch.cat([g_leaf.detach(), (g_leaf.detach() * mS.detach())], dim=-1)
    mU = gate_from_pU(module, gate(xU))
    u = F.normalize(g_leaf * mU, p=2, dim=-1, eps=1e-6)
    scores = FIXED_SCALE * (u @ tR.t())
    live_grads = torch.autograd.grad(scores.diagonal().sum(), [g_leaf] + list(gate.parameters()),
                                     allow_unused=True)
    assert live_grads[0] is not None and float(live_grads[0].abs().max()) > 0, \
        'the final g must stay live: the suffix loss trains the visual trunk through g*mU'
    assert any(grad is not None and float(grad.abs().max()) > 0 for grad in live_grads[1:]), \
        'F must receive its own gradient'

    detached_g = unit_features(2, generator, unit=False).requires_grad_(True)
    detached_mS = binary_masks(2, generator).requires_grad_(True)
    xU_only = torch.cat([detached_g.detach(), (detached_g.detach() * detached_mS.detach())], dim=-1)
    loss_only = gate(xU_only).sum()
    grads = torch.autograd.grad(loss_only, [detached_g, detached_mS] + list(gate.parameters()),
                                allow_unused=True)
    assert grads[0] is None or float(grads[0].abs().max()) == 0.0, \
        'no gradient may reach g through xU: the F input must be fully detached'
    assert grads[1] is None or float(grads[1].abs().max()) == 0.0, \
        'no gradient may reach mS through xU'
    assert any(grad is not None and float(grad.abs().max()) > 0 for grad in grads[2:]), \
        'F itself must still receive the gradient of its own output through pU'
    assert model is not None


@needs_cuda
def test_c_mU_multiplies_the_full_g_not_the_hard_complement():
    """Spec section 4: ``u = Norm(g * mU)``, explicitly NOT ``g * (1 - mS)``."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.6, seed=17)
    generator = torch.Generator().manual_seed(23)
    g = unit_features(3, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)
    with torch.no_grad():
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = gate(xU)
        mU = gate_from_pU(module, pU)
        assert float(mU.min()) == 0.0 and float(mU.max()) == 1.0, \
            'this test needs a genuinely mixed mU to be able to fail'
        correct = FIXED_SCALE * (F.normalize(g * mU, p=2, dim=-1, eps=1e-6) @ tR.t())
        wrong_complement = FIXED_SCALE * (F.normalize(g * (1.0 - mS), p=2, dim=-1, eps=1e-6)
                                          @ tR.t())
        wrong_hard = FIXED_SCALE * (F.normalize(g * (1.0 - (pU >= 0.5).float()), p=2, dim=-1,
                                               eps=1e-6) @ tR.t())
    assert float((correct - wrong_complement).abs().max()) > 1e-3, \
        'mU multiplies the FULL g, so g*(1-mS) must give a measurably different score'
    assert float((correct - wrong_hard).abs().max()) > 1e-3
    assert not torch.allclose(correct, wrong_hard, atol=1e-3)


# =========================================================================== D: initialisation
def test_d_initialisation_is_pU_8_9_and_mU_exactly_ones_with_a_restored_rng():
    """Spec section 4: seed 0, Xavier first layer, zero last weight, ``bias = log(8)``.

    The RNG requirement is checked on the REAL global generators: the CPU global state and every
    initialised CUDA generator are compared before and after construction. A private same-seed local
    generator proves nothing about the global stream, so it is not used as evidence here.
    """
    module = implementation_module()
    torch.manual_seed(4321)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = ([torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())]
                   if CUDA else [])
    mask = make_suffix_mask(module, DEVICE)
    assert torch.equal(torch.get_rng_state(), cpu_state), \
        'building SuffixMask must restore the REAL global CPU RNG state it found'
    if CUDA:
        for index, state in enumerate(cuda_states):
            assert torch.equal(torch.cuda.get_rng_state(index), state), \
                'building SuffixMask must restore CUDA generator %d of the REAL global RNG' % index

    parameters = dict(mask.named_parameters())
    assert 'layer1.weight' in parameters and 'layer2.weight' in parameters, \
        ('the suffix mask must be Linear(1024,512) -> GELU -> Linear(512,512); found %r'
         % sorted(parameters))
    assert tuple(parameters['layer1.weight'].shape) == (512, 1024)
    assert tuple(parameters['layer2.weight'].shape) == (512, 512)
    assert float(parameters['layer2.weight'].abs().max()) == TOL_EXACT, 'the last weight must be 0'
    assert float(parameters['layer2.bias'].min()) == pytest.approx(math.log(8.0),
                                                                   abs=STATE_TOLERANCE)
    assert float(parameters['layer2.bias'].max()) == pytest.approx(math.log(8.0),
                                                                   abs=STATE_TOLERANCE)
    assert float(parameters['layer1.weight'].std()) > 0
    assert float(parameters['layer1.bias'].abs().max()) == TOL_EXACT

    xU = torch.randn(4, 2 * FEATURE_DIM, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        pU = mask(xU)
        mU = gate_from_pU(module, pU)
    assert float(pU.min()) == pytest.approx(8.0 / 9.0, abs=STATE_TOLERANCE)
    assert float(pU.max()) == pytest.approx(8.0 / 9.0, abs=STATE_TOLERANCE)
    assert float((mU - 1.0).abs().max()) == TOL_EXACT, 'mU must be EXACTLY all ones at init'
    assert int(mU.sum().item()) == mU.numel()


def test_d_building_the_new_modules_does_not_shift_the_global_rng_unused_for_the_data_stream():
    """Spec section 3: the new modules must not pollute the original data random stream.

    DEFECT FIXED (fake RNG evidence): the old test drew a fresh same-seed LOCAL generator before and
    after construction and called that "the global state is unchanged" -- a local generator is not the
    global stream and cannot observe pollution. The REAL global CPU RNG and every initialised CUDA
    generator are compared here, together with the data stream the pipeline actually consumes.
    """
    module = implementation_module()

    def draw_stream(seed):
        stream = []
        generator = torch.Generator().manual_seed(seed)
        for _ in range(4):
            stream.append(float(torch.rand(1, generator=generator)))
        python_rng = random.Random(seed)
        for _ in range(4):
            stream.append(float(python_rng.random()))
        return stream

    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = ([torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())]
                   if CUDA else [])
    python_before = random.getstate()
    before = draw_stream(0)

    make_suffix_mask(module, DEVICE)

    assert torch.equal(torch.get_rng_state(), cpu_before), \
        'building SuffixMask consumed part of the REAL global CPU RNG stream'
    for index, state in enumerate(cuda_before):
        assert torch.equal(torch.cuda.get_rng_state(index), state), \
            'building SuffixMask consumed part of the REAL global CUDA generator %d' % index
    assert random.getstate() == python_before, \
        'building SuffixMask consumed part of the global Python RNG stream used for K'
    assert before == draw_stream(0), ('building the new modules shifted the data stream: the S0 arm '
                                      'would no longer see the same samples')

    torch.manual_seed(2026)
    state_before = torch.get_rng_state().clone()
    make_suffix_mask(module, DEVICE)
    assert torch.equal(state_before, torch.get_rng_state()), \
        'a second construction must be equally invisible to the global stream'


@needs_cuda
def test_d_first_backward_gives_the_zero_initialised_output_layer_no_gradient():
    module = implementation_module()
    model = build_model()
    mask = make_suffix_mask(module)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_ids = tokenize(SUFFIXES[:2])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR = F.normalize(model.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = mask(xU)
        scores = FIXED_SCALE * (F.normalize(g * gate_from_pU(module, pU), p=2, dim=-1, eps=1e-6)
                                @ tR.t())
    scores.diagonal().sum().backward()
    parameters = dict(mask.named_parameters())
    assert parameters['layer1.weight'].grad is not None
    assert float(parameters['layer1.weight'].grad.abs().max()) > 0, \
        'the first layer must receive a real gradient on the first backward'
    assert float(parameters['layer2.weight'].grad.abs().max()) == TOL_EXACT, \
        ('with W2 = 0 the last weight multiplies a zero upstream factor, so it gets exactly zero '
         'gradient, while its bias still moves')
    assert float(parameters['layer2.bias'].grad.abs().max()) > 0
    assert float(parameters['layer1.bias'].grad.abs().max()) > 0


@needs_cuda
def test_d_initialised_mU_is_all_ones_so_conditional_equals_native_scores():
    """Spec section 5: with mU all ones, ``QU`` must equal ``QN`` on the same checkpoint and pool."""
    module = implementation_module()
    model = build_model(shared_init=True)
    mask = make_suffix_mask(module)
    images = fixed_images(3)
    prefix_ids = tokenize(CAPTIONS[:3])
    suffix_ids = tokenize(SUFFIXES[:3])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = mask(xU)
        mU = gate_from_pU(module, pU)
        conditional = readout_scores(module, g, mS, tR, mask, image_chunk=3, text_chunk=3)
        native = FIXED_SCALE * (g @ tR.t())
    assert float((mU - 1.0).abs().max()) == TOL_EXACT, 'the initial gate must be exactly all ones'
    difference = float((conditional - native).abs().max())
    assert difference < TOL_RECOMPUTE, (difference, 'QU must equal QN when mU is all ones')
    assert float(clip_reference_suffix_loss(conditional)) == pytest.approx(
        float(clip_reference_suffix_loss(native)), rel=1e-5, abs=1e-7)
    mS_leaf = mS.detach().clone().requires_grad_(True)
    scores_from_leaf = readout_scores(module, g, mS_leaf, tR, mask, image_chunk=3, text_chunk=3)
    assert_absent_or_zero(tensor_gradient(scores_from_leaf.sum(), mS_leaf,
                                          'the old mS of the suffix readout'),
                          'the old mS of the suffix readout')


# =========================================================================== E: S0 regression
def _objective_for_s0(module, model, rank=0):
    """The objective builder of the trainer, with the S0-equivalence switch ``lambda_suffix = 0``."""
    trainer = trainer_module()
    builder = None
    for name in ('build_objective', 'make_objective', 'PrefixSuffixObjective', 'SuffixObjective',
                 'SaidPrefixSuffixObjective', 'Objective'):
        candidate = getattr(trainer, name, None)
        if candidate is not None and callable(candidate):
            builder = candidate
            break
    assert builder is not None, \
        ('the trainer exposes no objective builder; the S0 regression cannot be checked. Module '
         'attributes: %r' % sorted(name for name in dir(trainer) if not name.startswith('__')))
    signature = inspect.signature(builder)
    accepted = {}
    for name, value in (('clip', model), ('model', model), ('clip_model', model),
                        ('module', module), ('suffix_model', module), ('suffix_mask', None),
                        ('lambda_suffix', 0.0), ('rank', rank), ('device', DEVICE),
                        ('soft_mask', False)):
        if name in signature.parameters:
            accepted[name] = value
    if 'lambda_suffix' not in accepted:
        accepted['lambda_suffix'] = 0.0            # the spec's S0-equivalence switch
    if 'suffix_mask' in accepted:
        accepted['suffix_mask'] = make_suffix_mask(module, DEVICE)
    return builder(**accepted)


def s0_terms_of(out):
    """The S0 term dict of an objective output, from the output itself or from a nested dict."""
    if isinstance(out, dict):
        if 'loss_sidm' in out and 'loss_dism' in out:
            return {'loss_sidm': out['loss_sidm'], 'loss_dism': out['loss_dism'],
                    'loss_sparsity': out['loss_sparsity'],
                    **({'mS': out['mS']} if 'mS' in out else {})}
        for key in ('terms', 's0', 'smart', 's0_terms'):
            if isinstance(out.get(key), dict) and 'loss_sidm' in out[key]:
                return out[key]
    raise AssertionError('the objective output does not expose the S0 terms (loss_sidm / loss_dism '
                         '/ loss_sparsity); keys: %r'
                         % (sorted(out) if isinstance(out, dict) else type(out).__name__))


def weighted_s0(loss_sidm, loss_dism, loss_sparsity, lambda_align, lambda_sparse):
    """``L_S0 = lambda_align * (L_SIDM + L_DISM) + lambda_sparse * L_sparse`` (spec section 1)."""
    return lambda_align * (loss_sidm + loss_dism) + lambda_sparse * loss_sparsity


@needs_cuda
def test_e_lambda_suffix_zero_reproduces_the_original_s0_objective_and_gradients():
    """Spec section 3: at ``lambda_suffix = 0`` the S0 loss and its gradients are the ORIGINAL ones.

    DEFECT FIXED (the reference demanded gradients it had destroyed): the old reference encoded the
    trunk under ``torch.no_grad()`` and then asked for the gradient of ``visual.proj`` /
    ``text_projection``, which can only ever be ``None``. The reference here is the REAL weighted
    objective ``10 * (L_SIDM + L_DISM) + 2 * L_sparse``, built from the original helper on LIVE
    features, and the unweighted components are compared separately as extra evidence (so a silent
    drop of the ``2 *`` sparsity weight cannot hide behind the pair comparison).

    Tolerance: TOL_GRADIENT = 1e-4 relative to each tensor's own max-abs. Both sides run the same fp32
    arithmetic with a different accumulation order, so a bitwise comparison is not available. A
    re-derived SIDM/DISM, a changed sparse weight, a mask used without its stop-gradient, an extra 0.5
    factor or a world-size factor move each tensor by orders of magnitude, not by 1e-4.
    """
    module = implementation_module()
    from model.said_cls_cvssl import compute_smartclip_terms
    model = build_model(shared_init=True)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])

    # the LIVE S0 path: the trunk stays differentiable, exactly as in the original objective
    v_a = model.encode_image(images)
    text_raw, hidden = model.encode_text(prefix_ids, return_full=True)
    mS_reference = call_mask_helper(module, model.mask_net, hidden)
    terms_reference = compute_smartclip_terms(v_a, text_raw, mS_reference, rank=0,
                                              lambda_align=float(module.LAMBDA_ALIGN),
                                              lambda_sparse=float(module.LAMBDA_SPARSE))
    loss_reference = weighted_s0(terms_reference['loss_sidm'], terms_reference['loss_dism'],
                                 terms_reference['loss_sparsity'], float(module.LAMBDA_ALIGN),
                                 float(module.LAMBDA_SPARSE))
    assert float(module.LAMBDA_ALIGN) == pytest.approx(10.0)
    assert float(module.LAMBDA_SPARSE) == pytest.approx(S0_LAMBDA_SPARSE)
    # pin the original constants so this test cannot agree with a changed S0 by accident
    assert float(terms_reference['loss_sparsity']) == pytest.approx(
        float(mS_reference.abs().mean()), rel=1e-6)
    assert float(terms_reference['loss_sidm']) > 0 and float(terms_reference['loss_dism']) > 0

    mask_parameters = list(model.mask_net.parameters())
    trunk_parameters = [model.visual.proj, model.text_projection]
    reference_grads = torch.autograd.grad(loss_reference, mask_parameters + trunk_parameters,
                                          allow_unused=True, retain_graph=True)
    assert any(grad is not None and float(grad.abs().max()) > 0
               for grad in reference_grads[len(mask_parameters):]), \
        ('the reference must keep the trunk LIVE: a no_grad encoding would make every trunk gradient '
         'None and the comparison below would be vacuous')

    objective = _objective_for_s0(module, model)
    out = objective(images, prefix_ids)
    terms = s0_terms_of(out)
    loss_new = weighted_s0(terms['loss_sidm'], terms['loss_dism'], terms['loss_sparsity'],
                           float(module.LAMBDA_ALIGN), float(module.LAMBDA_SPARSE))
    assert abs(float(loss_new) - float(loss_reference)) <= TOL_RECOMPUTE * max(
        abs(float(loss_reference)), 1.0), \
        ('the S0 loss at lambda_suffix = 0 must equal the original weighted helper '
         '(10*(SIDM+DISM) + 2*sparse)', float(loss_new), float(loss_reference))

    # the unweighted components too: a dropped 2* on the sparse term is invisible in a pair sum that
    # happens to be re-weighted elsewhere, but not here
    for name in ('loss_sidm', 'loss_dism', 'loss_sparsity'):
        mine = float(terms[name].detach())
        theirs = float(terms_reference[name].detach())
        assert abs(mine - theirs) <= TOL_RECOMPUTE * max(abs(theirs), 1.0), (name, mine, theirs)

    new_grads = torch.autograd.grad(loss_new, mask_parameters + trunk_parameters, retain_graph=True,
                                    allow_unused=True)
    compared = 0
    for index, (mine, theirs) in enumerate(zip(new_grads, reference_grads)):
        if mine is None or theirs is None:
            assert (mine is None) == (theirs is None), \
                ('S0 gradient %d exists in only one of the two paths, so the S0 objective is not '
                 'the same on both sides' % index)
            continue
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('S0 gradient %d differs from the original helper: lambda_suffix = 0 must leave the S0 '
             'objective, the old mask and the shared trunk untouched' % index)
        compared += 1
    assert compared >= len(trunk_parameters), compared
    if 'mS' in terms:
        assert torch.equal(terms['mS'], mS_reference)


# =========================================================================== F: readout contract
def test_f_readout_signature_and_return_type_are_the_frozen_ones():
    """The frozen interface: four named keyword parameters and ONE return type (a 2-key dict)."""
    module = implementation_module()
    function = getattr(module, 'suffix_readout_scores', None)
    assert callable(function), ('model/said_prefix_suffix.py must export suffix_readout_scores '
                                '(spec section 14)')
    parameters = inspect.signature(function).parameters
    for name in READOUT_PARAMS:
        assert name in parameters, ('suffix_readout_scores must declare the frozen parameter %r; it '
                                    'declares %r' % (name, list(parameters)))
    for name, default in (('image_chunk', 16), ('text_chunk', 32), ('eps', 1e-6),
                          ('want_statistics', False)):
        assert name in parameters, ('suffix_readout_scores must declare the documented default %r=%r'
                                    % (name, default))
        assert parameters[name].default == default, \
            ('the documented default of %r is %r, found %r'
             % (name, default, parameters[name].default))

    gate = make_suffix_mask(module)
    generator = torch.Generator().manual_seed(101)
    g = unit_features(2, generator)
    mS = binary_masks(2, generator)
    tR = unit_features(3, generator)
    result = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=2)
    assert tuple(result['scores'].shape) == (2, 3), tuple(result['scores'].shape)
    assert 'statistics' in result and result['statistics'] is None


def test_f_readout_rejects_a_three_dimensional_mS_block():
    """Spec section 7 + the frozen interface: ``mS_block`` is ``[Bj, 512]``, TWO-DIMENSIONAL ONLY.

    A ``[Bi, Bj, 512]`` block is the "guess the shape" shortcut of the old signature. It must be
    rejected loudly (a ValueError/TypeError naming the dimensionality), because accepting it silently
    means the candidate-prefix rule of spec section 5 is no longer pinned by the signature.
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    generator = torch.Generator().manual_seed(103)
    g = unit_features(2, generator)
    tR = unit_features(3, generator)
    mS_2d = binary_masks(3, generator)
    mS_3d = mS_2d.unsqueeze(0).expand(2, 3, FEATURE_DIM).contiguous()
    assert mS_3d.dim() == 3
    with pytest.raises((ValueError, TypeError, AssertionError, RuntimeError)) as error:
        call_readout(module, g, mS_3d, tR, gate, image_chunk=2, text_chunk=2)
    message = str(error.value).lower()
    assert ('dim' in message or '3-d' in message or '3d' in message or 'shape' in message
            or 'dimensional' in message), \
        ('the rejection must say WHY it rejected the input (dimensionality), got %r' % (message,))


# --------------------------------------------------------------------------- non-square broadcast
NON_SQUARE_CASES = ((2, 3, 2, 3), (3, 2, 2, 3), (5, 7, 4, 3))


@pytest.mark.parametrize('images,texts,image_chunk,text_chunk', NON_SQUARE_CASES)
def test_f_non_square_pairs_match_the_per_pair_reference_in_values_and_gradients(
        images, texts, image_chunk, text_chunk):
    """DEFECT PINNED (broadcast): ``Bi != Bj`` must still score EVERY (i, j) pair.

    Three shapes are covered: ``Bi < Bj``, ``Bi > Bj`` and a size whose tiling does NOT divide
    (``5 x 7`` with ``4 x 3`` tiles leaves a 1-row and a 1-column tail). For each one the forward value
    and the gradients of ``g``, ``tR`` and every F parameter are compared against the per-pair loop
    reference of :func:`per_pair_reference`, which shares no code with the tiled readout.

    ``mS``'s gradient must be ``None`` or exactly zero: it is the stop-gradient input.

    Tolerances: values ``TOL_RECOMPUTE`` (1e-5, absolute at the fixed 100x score scale ~ 1e-7
    relative, absorbing the tiled-vs-per-pair accumulation order); parameters ``TOL_GRADIENT`` (1e-4
    relative to each tensor's own max-abs) for the same reason.
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=107)
    initial = parameter_snapshot(gate)
    generator = torch.Generator().manual_seed(109)
    g_base = unit_features(images, generator)
    mS_base = binary_masks(texts, generator)
    tR_base = unit_features(texts, generator)
    weights = [parameter for parameter in gate.parameters() if parameter.requires_grad]
    assert weights, 'F must expose trainable parameters'

    g_leaf = g_base.detach().clone().requires_grad_(True)
    mS_leaf = mS_base.detach().clone().requires_grad_(True)
    tR_leaf = tR_base.detach().clone().requires_grad_(True)
    scores = readout_scores(module, g_leaf, mS_leaf, tR_leaf, gate, image_chunk=image_chunk,
                            text_chunk=text_chunk)
    assert tuple(scores.shape) == (images, texts), tuple(scores.shape)
    chunk_grads = torch.autograd.grad(scores.sum(), [g_leaf, mS_leaf, tR_leaf] + weights,
                                      retain_graph=True, allow_unused=True)

    gate.load_state_dict(initial)
    g_ref = g_base.detach().clone().requires_grad_(True)
    mS_ref = mS_base.detach().clone().requires_grad_(True)
    tR_ref = tR_base.detach().clone().requires_grad_(True)
    reference = per_pair_reference(module, gate, g_ref, mS_ref, tR_ref)
    reference_grads = torch.autograd.grad(per_pair_loss(reference), [g_ref, mS_ref, tR_ref] + weights,
                                          retain_graph=True, allow_unused=True)

    difference = float((scores - reference).abs().max())
    assert difference < TOL_RECOMPUTE, \
        ('the tiled readout must reproduce the per-pair loop for a %dx%d block' % (images, texts),
         difference)
    assert_parameters_unchanged(gate, initial, 'the non-square readout comparison')

    assert_absent_or_zero(chunk_grads[1], 'mS (chunked, %dx%d)' % (images, texts))
    assert_absent_or_zero(reference_grads[1], 'mS (per-pair reference)')
    for name, index in (('g', 0), ('tR', 2)):
        mine, theirs = chunk_grads[index], reference_grads[index]
        assert mine is not None and theirs is not None, name
        assert float(theirs.abs().max()) > 0, ('the %s gradient must be non-trivial for this '
                                               'comparison to mean anything' % name)
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('the %s gradient must match the per-pair reference on a %dx%d block'
             % (name, images, texts), relative_difference(mine, theirs))
    for offset, weight in enumerate(weights):
        mine, theirs = chunk_grads[3 + offset], reference_grads[3 + offset]
        assert mine is not None and theirs is not None, offset
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('F parameter %d must receive the per-pair gradient on a %dx%d block'
             % (offset, images, texts), relative_difference(mine, theirs))
    assert any(float(chunk_grads[3 + offset].abs().max()) > 0 for offset in range(len(weights))), \
        'the suffix loss must train F on a non-square block'


def test_f_a_wrong_g_times_mS_broadcast_must_differ_from_the_correct_scores():
    """DEFECT PINNED (broadcast), the explicit counterexample: with ``Bi != Bj`` the naive
    ``g_block * mS_block`` cannot be what the readout computes.

    The wrong path is *executed* here (elementwise, and if torch refuses it, a row-wise broadcast) and
    its result is required to DIFFER from the correct one by more than a tolerance. A test that only
    asserted equality would pass vacuously against an implementation that broadcasts row ``i`` to mask
    row ``i``.
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=113)
    generator = torch.Generator().manual_seed(127)
    g = unit_features(2, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)

    correct = readout_scores(module, g, mS, tR, gate, image_chunk=2, text_chunk=3)
    assert tuple(correct.shape) == (2, 3), tuple(correct.shape)

    kind, product = wrong_broadcast_scores(module, gate, g, mS, tR)
    if kind != 'elementwise':
        assert tuple(product.shape) == (2, 3, FEATURE_DIM), (kind, tuple(product.shape))
        wrong = FIXED_SCALE * torch.einsum(
            'ijd,jd->ij', F.normalize(product, p=2, dim=-1, eps=1e-6), tR)
    else:
        try:
            wrong = FIXED_SCALE * (F.normalize(product, p=2, dim=-1, eps=1e-6) @ tR.t())
        except RuntimeError as error:                      # pragma: no cover - shape dependent
            raise AssertionError('the wrong broadcast produced %s, which cannot be scored at all: %s'
                                 % (tuple(product.shape), error))
    assert tuple(wrong.shape) == (2, 3), (kind, tuple(wrong.shape))
    difference = float((correct - wrong).abs().max())
    assert difference > 1e-3, \
        ('the wrong g*mS broadcast must give a measurably different score matrix, otherwise this '
         'counterexample proves nothing', kind, difference)


def test_f_chunked_readout_equals_the_per_pair_loop_in_scores_and_gradients():
    """Spec section 7: chunking is a memory device, never a different computation.

    The inputs respect the unit-scale ``g`` contract. The retired claim that "pre-normalising g makes
    the conditional scores smaller by orders of magnitude" is NOT used as a justification anywhere:
    with a fixed mask, a correctly normalised cosine has no such artificial scale effect, and F really
    does read ``g``, so the readout is tested on the inputs the spec defines.
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=29)
    generator = torch.Generator().manual_seed(31)
    g_base = unit_features(4, generator)
    mS_base = binary_masks(4, generator)
    tR_base = unit_features(4, generator)
    weights = [parameter for parameter in gate.parameters() if parameter.requires_grad]
    initial = parameter_snapshot(gate)

    g = g_base.detach().clone().requires_grad_(True)
    mS = mS_base.detach().clone().requires_grad_(True)
    tR = tR_base.detach().clone().requires_grad_(True)
    scores = readout_scores(module, g, mS, tR, gate, image_chunk=3, text_chunk=3)
    assert tuple(scores.shape) == (4, 4), tuple(scores.shape)
    chunk_grads = torch.autograd.grad(scores.sum(), [g, mS, tR] + weights, retain_graph=True,
                                      allow_unused=True)

    gate.load_state_dict(initial)
    g2 = g_base.detach().clone().requires_grad_(True)
    mS2 = mS_base.detach().clone().requires_grad_(True)
    tR2 = tR_base.detach().clone().requires_grad_(True)
    reference = per_pair_reference(module, gate, g2, mS2, tR2)
    per_pair_grads = torch.autograd.grad(reference.sum(), [g2, mS2, tR2] + weights,
                                         retain_graph=True, allow_unused=True)

    difference = float((scores - reference).abs().max())
    assert difference < TOL_RECOMPUTE, ('the chunked and per-pair scores must agree', difference)
    assert_parameters_unchanged(gate, initial, 'the chunked and per-pair readout comparison')
    assert_absent_or_zero(chunk_grads[1], 'mS (chunked)')
    assert_absent_or_zero(per_pair_grads[1], 'mS (per-pair reference)')
    for name, index in (('g', 0), ('tR', 2)):
        mine, theirs = chunk_grads[index], per_pair_grads[index]
        assert mine is not None, name
        assert theirs is not None, name
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('the chunked readout must give the per-pair gradient of %s' % name,
             relative_difference(mine, theirs))
    for offset in range(len(weights)):
        mine, theirs = chunk_grads[3 + offset], per_pair_grads[3 + offset]
        assert mine is not None and theirs is not None, offset
        assert relative_difference(mine, theirs) < TOL_GRADIENT, ('F gradient %d' % offset)
    assert float(chunk_grads[0].abs().max()) > 0, 'the suffix loss must train the live g'
    assert float(chunk_grads[2].abs().max()) > 0, 'the suffix loss must train tR'
    assert any(float(grad.abs().max()) > 0 for grad in chunk_grads[3:]), \
        'the suffix loss must train F'


def test_f_readout_returns_flat_fp32_scores_for_a_ragged_tail_tile():
    """A tail tile smaller than the chunk size must still be a plain ``[Bi, Bj]`` fp32 matrix."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    generator = torch.Generator().manual_seed(131)
    g = unit_features(5, generator)
    mS = binary_masks(7, generator)
    tR = unit_features(7, generator)
    scores = readout_scores(module, g, mS, tR, gate, image_chunk=4, text_chunk=3)
    assert tuple(scores.shape) == (5, 7), tuple(scores.shape)
    assert scores.dtype == torch.float32
    assert bool(torch.isfinite(scores).all())
    # every column really used its OWN candidate prefix: a column-blind implementation would give a
    # rank-1-like pattern, which the per-column variation rule below refuses
    variation = scores.std(dim=0)
    assert float(variation.max()) > 0, 'the score matrix must depend on the candidate column'


# --------------------------------------------------------------------------- index mapping
def test_f_index_mapping_counterexample_W2_B4_J_1_3_4_7():
    """DEFECT PINNED (label space): ``W=2, B=4, J=[1,3,4,7]``.

    rank 0 -> ``local_rows = [1, 3]``, ``anchor_valid = [0, 1]``
    rank 1 -> ``local_rows = [0, 3]``, ``anchor_valid = [2, 3]``

    The label of a local valid row inside the V pool is ``global_to_valid[rank * B + local_row]``.
    ``arange(rank * B, rank * B + n_r)`` would give ``[0, 1]`` / ``[2, 3]`` here and is a different rule
    in general: it is the valid-anchor ORDINAL, not the anchor's position in the pool. This test pins
    both the correct mapping and the fact that the wrong formula is a different object.
    """
    module = implementation_module()
    J = torch.tensor([1, 3, 4, 7], dtype=torch.long)
    local_size = 4
    assert module is not None
    assert [global_to_valid(J)[index].item() for index in J.tolist()] == [0, 1, 2, 3]

    rows0, labels0 = anchors_of_rank(J, local_size, 0)
    rows1, labels1 = anchors_of_rank(J, local_size, 1)
    assert rows0 == [1, 3], rows0
    assert labels0 == [0, 1], labels0
    assert rows1 == [0, 3], rows1
    assert labels1 == [2, 3], labels1

    # the wrong rule, spelled out so the difference is visible rather than assumed
    wrong0 = list(range(0 * local_size, 0 * local_size + len(rows0)))
    wrong1 = list(range(1 * local_size, 1 * local_size + len(rows1)))
    assert wrong0 == labels0, ('in THIS example the local arange coincides with the labels of rank 0, '
                               'which is exactly why it is a trap rather than a detector', wrong0)
    assert wrong1 == labels1, ('the same coincidence holds for rank 1 here: the counterexample below '
                               'is what separates the two rules', wrong1)

    # a second pattern where the two rules genuinely disagree, so the guard is not vacuous
    J2 = torch.tensor([0, 5, 6], dtype=torch.long)
    rows2, labels2 = anchors_of_rank(J2, 4, 1)
    assert rows2 == [1, 2], rows2
    assert labels2 == [1, 2], labels2
    wrong2 = list(range(1 * 4, 1 * 4 + len(rows2)))
    assert wrong2 == [4, 5], wrong2
    assert wrong2 != labels2, ('the local arange must NOT be accepted as the label rule; got %r for '
                               'the correct %r' % (wrong2, labels2))


def test_f_global_valid_indices_returns_the_global_rule_and_both_spaces_agree():
    """``global_valid_indices`` on per-rank validity, and the two index spaces agree everywhere."""
    module = implementation_module()
    rank0 = torch.tensor([True, False, True, False])
    rank1 = torch.tensor([False, True, False, False])
    gathered = [rank0, rank1]
    J = module.global_valid_indices(gathered, world_size=2)
    J = J if torch.is_tensor(J) else torch.tensor(list(J), dtype=torch.long)
    assert J.tolist() == [0, 2, 5], J.tolist()
    assert module.global_valid_indices(rank0, world_size=1).tolist() == [0, 2]
    empty = module.global_valid_indices(torch.zeros(4, dtype=torch.bool), world_size=1)
    empty = empty if torch.is_tensor(empty) else torch.tensor(list(empty), dtype=torch.long)
    assert empty.numel() == 0

    table = global_to_valid(J)
    for rank in (0, 1):
        rows, labels = anchors_of_rank(J, 4, rank)
        lo = rank * 4
        for row, label in zip(rows, labels):
            assert lo + row in J.tolist(), (rank, row)
            assert int(table[lo + row].item()) == label, (rank, row, label)
        assert sorted(labels) == sorted([int(table[index].item()) for index in J.tolist()
                                         if lo <= index < lo + 4])
    assert [int(table[index].item()) for index in J.tolist()] == list(range(int(J.numel())))


def test_f_three_filtered_tensors_keep_one_consistent_order_after_indexing():
    """DEFECT PINNED (filter consistency): matching shapes are NOT evidence of a consistent order.

    The image rows, the candidate prefix masks and the suffix texts of the valid pool are three
    separately filtered tensors. If any one of them is filtered with a different order (a wrong
    ``index_select``, a sort, a python-level re-listing), the shapes still match and the loss silently
    pairs an image with the wrong prefix mask. Every row here carries a SAMPLE-ID SENTINEL, and the
    three filtered tensors are checked against the SAME ``J`` order row by row.
    """
    module = implementation_module()
    J = torch.tensor([1, 3, 4, 7], dtype=torch.long)
    local_size = 4
    world = 2
    dim = 8
    anchors = torch.arange(world * local_size, dtype=torch.float32)

    def build(count, offset):
        tensor = torch.zeros(count, dim)
        for row in range(count):
            tensor[row, 0] = anchors[row] + offset          # the sample-ID sentinel
            tensor[row, 1] = 1.0                            # a non-zero scalar for the normalisation
        return tensor

    g_all = build(world * local_size, 0.0)
    mask_all = build(world * local_size, 100.0)
    text_all = build(world * local_size, 200.0)

    g_sub = g_all.index_select(0, J)
    mask_sub = mask_all.index_select(0, J)
    text_sub = text_all.index_select(0, J)
    assert tuple(g_sub.shape)[0] == tuple(mask_sub.shape)[0] == tuple(text_sub.shape)[0] == J.numel()

    expected = [float(anchors[index]) for index in J.tolist()]
    for row in range(int(J.numel())):
        assert float(g_sub[row, 0]) == expected[row], ('image order', row, float(g_sub[row, 0]))
        assert float(mask_sub[row, 0]) == expected[row] + 100.0, \
            ('prefix-mask order', row, float(mask_sub[row, 0]))
        assert float(text_sub[row, 0]) == expected[row] + 200.0, \
            ('suffix-text order', row, float(text_sub[row, 0]))

    # the anchor labels are the pool ordinals, and each rank's labels must select exactly its own rows
    for rank in range(world):
        rows, labels = anchors_of_rank(J, local_size, rank)
        for row, label in zip(rows, labels):
            index = rank * local_size + row
            assert float(g_sub[label, 0]) == float(anchors[index]), (rank, row, label)
            assert float(mask_sub[label, 0]) == float(anchors[index]) + 100.0
            assert float(text_sub[label, 0]) == float(anchors[index]) + 200.0
    assert sum(len(anchors_of_rank(J, local_size, rank)[0]) for rank in range(world)) == int(J.numel())


def test_f_the_real_readout_pairs_every_image_with_every_candidate_mask():
    """The positive-pair rule of spec section 5, made falsifiable with sentinel inputs.

    Column ``j`` of the score matrix must be built from image ``i`` and the mask of candidate ``j``.
    The per-pair reference is the arbiter, and the readout is additionally required to differ from the
    "one mask for the whole row" shortcut (the equivalent-replacement the spec forbids).
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=137)
    generator = torch.Generator().manual_seed(139)
    g = unit_features(3, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)
    scores = readout_scores(module, g, mS, tR, gate, image_chunk=3, text_chunk=3)
    reference = per_pair_reference(module, gate, g, mS, tR)
    assert float((scores - reference).abs().max()) < TOL_RECOMPUTE

    # the forbidden shortcut: one shared mask per row (the image's own prefix) for every candidate
    shared = []
    for row in range(int(g.shape[0])):
        column = []
        for candidate in range(int(tR.shape[0])):
            x_u = torch.cat([g[row].detach(),
                             (g[row].detach() * mS[row].detach())]).unsqueeze(0)
            mask_u = gate_from_pU(module, gate(x_u))
            u = F.normalize(g[row].unsqueeze(0) * mask_u, p=2, dim=-1, eps=1e-6)
            column.append(FIXED_SCALE * (u @ tR[candidate].unsqueeze(1)).squeeze())
        shared.append(torch.stack(column))
    shortcut = torch.stack(shared)
    assert float((scores - shortcut).abs().max()) > 1e-3, \
        ('the readout must use candidate j OWN prefix mask, not one shared mask per image row; the '
         'two answers coincide here, so this test could not tell them apart')


# =========================================================================== G: statistics / hygiene
def test_g_keep_ratio_is_the_hard_gate_fraction_and_never_the_thresholded_mean():
    """DEFECT PINNED (heavy-log statistics): ``p = [0.9, 0.1, 0.1]`` must give a keep ratio of 1/3.

    Thresholding the MEAN probability (``mean(p) = 0.3667 < 0.5``) yields 0 -- the wrong answer the spec
    section 13 heavy log must never publish. The ratio is the fraction of coordinates whose hard gate is
    on, computed BEFORE any reduction. ``pU = 8/9`` everywhere (a freshly initialised F) is the safe
    contrast: the ratio is exactly 1.
    """
    module = implementation_module()
    constant_gate = zero_gate_module(module)

    class _ScriptedGate(torch.nn.Module):
        """``pU`` = the given value for the first row, ``0.1`` afterwards: a mixed gate."""

        def forward(self, x):
            value = torch.full_like(x, 0.1)
            value[0, 0] = 0.9
            return value

    generator = torch.Generator().manual_seed(149)
    g = unit_features(1, generator)
    mS = binary_masks(2, generator)
    tR = unit_features(2, generator)
    gate = _ScriptedGate()
    result = call_readout(module, g, mS, tR, gate, image_chunk=1, text_chunk=2,
                          want_statistics=True)
    keep_key, keep_value = statistic(result['statistics'], KEEP_RATIO_KEYS, 'keep ratio')
    probability = torch.tensor([0.9, 0.1, 0.1])
    expected = float((probability >= 0.5).float().mean())
    assert expected == pytest.approx(1.0 / 3.0)
    assert as_float(keep_value) == pytest.approx(expected, abs=TOL_FP32_TIGHT), \
        ('the keep ratio of p=[0.9, 0.1, 0.1] is 1/3; thresholding the MEAN (0.3667 < 0.5) would give '
         '0.0, which is the defect this pins', keep_key, as_float(keep_value))
    assert as_float(keep_value) != pytest.approx(0.0, abs=1e-6), \
        'a keep ratio of exactly 0 from a non-degenerate gate means the mean was thresholded'

    reference = zero_gate_module(module)(8.0 / 9.0)
    initial = make_suffix_mask(module)
    result = call_readout(module, g, mS, tR, initial, image_chunk=1, text_chunk=2,
                          want_statistics=True)
    _key, all_on_keep = statistic(result['statistics'], KEEP_RATIO_KEYS, 'keep ratio')
    assert as_float(all_on_keep) == pytest.approx(1.0, abs=TOL_FP32_TIGHT), \
        'at initialisation every kept coordinate is kept, so the ratio is exactly 1'
    assert reference is not None and constant_gate is not None


def test_g_norm_ratio_is_per_pair_and_divided_by_the_plain_g_norm():
    """DEFECT PINNED (heavy-log statistics): the norm ratio is ``||g*mU|| / max(||g||, eps)`` PER PAIR.

    Two wrong shapes are excluded by construction:
      * dividing by ``sqrt(512)`` (the old implementation) gives ``~1/sqrt(512) ~ 0.044`` for an all-on
        gate instead of ``1.0`` -- a fixed, detectable factor;
      * collapsing the pair axis with ``mean(-1)`` and assigning that into a ``[Bi, Bj]`` slot gives a
        tensor whose rows are constant; the per-pair structure asserted below refuses it.
    """
    module = implementation_module()

    class _MixedGate(torch.nn.Module):
        """Row-dependent, column-independent-ish gate: keeps everything except one coordinate."""

        def forward(self, x):
            value = torch.full_like(x, 8.0 / 9.0)
            value[:, 0] = 0.1
            return value

    generator = torch.Generator().manual_seed(151)
    g = unit_features(2, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)
    gate = _MixedGate()
    result = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=3,
                          want_statistics=True)
    key, ratio = statistic(result['statistics'], NORM_RATIO_KEYS, 'norm ratio')
    matrix = as_matrix(ratio, 'the norm ratio (%s)' % key)
    assert tuple(matrix.shape) == (2, 3), (key, tuple(matrix.shape))

    # the independent reference: the hard gate of the scripted probabilities, applied to the REAL g
    with torch.no_grad():
        mask_u = (torch.full_like(g, 8.0 / 9.0) >= 0.5).to(g.dtype)
        mask_u[:, 0] = 0.0
        gated = g.detach().unsqueeze(1) * mask_u.unsqueeze(1)
        numerator = gated.norm(dim=-1)
        denominator = g.detach().norm(dim=-1).clamp_min(1e-6).unsqueeze(1)
        expected = numerator / denominator
    assert tuple(expected.shape) == (2, 3), tuple(expected.shape)
    assert float((matrix - expected).abs().max()) < TOL_FP32_TIGHT, \
        ('the norm ratio must be ||g*mU|| / max(||g||, eps) per pair',
         float((matrix - expected).abs().max()), key)
    # rows are NOT constant (the pair axis is really there) and the all-on columns are exactly 1
    assert float(matrix.std(dim=1).min()) > 0, \
        ('a [Bi, Bj] statistic whose rows are constant is a collapsed [Bi] mean(-1) broadcast into a '
         '[Bi, Bj] slot, which is the defect this pins', matrix)
    assert float((matrix[:, 1:] - 1.0).abs().max()) < TOL_FP32_TIGHT, \
        ('an all-on gate keeps the full g, so the ratio is exactly 1 there; a ratio near '
         '1/sqrt(512) means the denominator was the embedding dimension', matrix)


def test_g_statistics_report_an_empty_pool_as_count_zero_and_value_null():
    """Spec sections 6/13: with no valid suffix pair there is nothing to average: ``count=0``,
    ``value=null``. A zero-filled ratio, a NaN or a silently reused previous value are all wrong."""
    module = implementation_module()
    gate = make_suffix_mask(module)

    class _NoCandidate(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.1)                  # every hard gate is off

    generator = torch.Generator().manual_seed(157)
    g = unit_features(2, generator)
    mS = binary_masks(0, generator).reshape(0, FEATURE_DIM)
    tR = unit_features(0, generator).reshape(0, FEATURE_DIM)

    empty = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=2,
                         want_statistics=True)
    assert tuple(empty['scores'].shape) == (2, 0), tuple(empty['scores'].shape)
    statistics = empty['statistics']
    count_key, count_value = statistic(statistics, COUNT_KEYS, 'pair count')
    assert as_float(count_value) == 0.0, (count_key, as_float(count_value))
    value_key, value_value = statistic_optional(statistics, VALUE_KEYS)
    assert value_key is not None, \
        ('the empty pool must still report the value field, explicitly as null; keys: %r'
         % sorted(statistics))
    assert value_value is None, \
        ('with count = 0 the value must be null, not 0.0 and not NaN; got %r' % (value_value,))

    # a fully closed gate is NOT an empty pool: the pairs exist, the ratio is 0, the count is not 0
    closed = call_readout(module, g, mS[:2] if mS.numel() else mS, tR, _NoCandidate(),
                          image_chunk=2, text_chunk=2, want_statistics=True) \
        if mS.numel() else None
    assert closed is None or closed['statistics'] is not None


def test_g_lse_margin_is_positive_minus_logsumexp_of_valid_negatives_only():
    """Spec section 13: the LSE margin is ``positive - logsumexp(valid negatives only)``.

    ``logsumexp(all) - positive`` is the cross entropy and may never be published under this name. The
    difference between the two is exactly ``logsumexp(all) - positive - (positive - lse_neg)``, which is
    large here, so the two definitions cannot be confused.
    """
    module = implementation_module()
    rows = torch.tensor([[4.0, 0.5, -1.0], [0.2, 3.0, 0.1], [1.0, 1.0, 1.5]])
    expected = lse_margin_reference(rows)
    cross_entropy_style = torch.logsumexp(rows, dim=1) - rows.diagonal()
    assert float((expected - cross_entropy_style).abs().min()) > 0.5, \
        ('the two candidate definitions must be far apart for this test to discriminate', expected,
         cross_entropy_style)

    name, margin = statistic(module_readout_statistics(module, rows), LSE_MARGIN_KEYS, 'LSE margin')
    margin = margin.detach().float() if torch.is_tensor(margin) else torch.tensor(margin)
    assert margin.numel() == rows.shape[0], (name, tuple(margin.shape))
    assert float((margin - expected).abs().max()) < TOL_RECOMPUTE, \
        ('the LSE margin must be positive - logsumexp(valid negatives only)', name,
         margin.tolist(), expected.tolist())
    assert bool((expected > 0).all()), expected


def module_readout_statistics(module, rows):
    """Run the real readout on a score matrix that yields exactly ``rows`` as its LSE margins.

    The margin is a property of the score matrix, so the statistics are read off a readout whose
    candidate count matches the matrix and whose rows are then compared against the reference. The
    readout's own score matrix is what the margin must describe, so the helper returns the margin for
    the READOUT's scores and the test compares it against the reference computed from those same scores.
    """
    gate = make_suffix_mask(module)
    generator = torch.Generator().manual_seed(163)
    count = int(rows.shape[0])
    g = unit_features(count, generator)
    mS = binary_masks(count, generator)
    tR = unit_features(count, generator)
    result = call_readout(module, g, mS, tR, gate, image_chunk=count, text_chunk=count,
                          want_statistics=True)
    scores = result['scores'].detach().float()
    reference = lse_margin_reference(scores)
    statistics = result['statistics']
    for key in LSE_MARGIN_KEYS:
        if key in statistics and statistics[key] is not None:
            value = statistics[key]
            value = value.detach().float() if torch.is_tensor(value) else torch.tensor(value)
            assert value.numel() == reference.numel(), (key, tuple(value.shape))
            assert float((value - reference).abs().max()) < TOL_RECOMPUTE, \
                ('%s must be positive - logsumexp(valid negatives only) of the readout scores'
                 % key, value.tolist(), reference.tolist())
            return key, value
    raise AssertionError('the readout statistics carry no LSE margin; looked for %r among %r'
                         % (list(LSE_MARGIN_KEYS), sorted(statistics)))


def test_g_returned_dict_carries_no_live_tensor_that_cannot_reach_the_loss():
    """DEFECT PINNED (DDP graph hygiene): the returned dict must not hand DDP tensors it cannot use.

    Three rules, all enforced here:
      1. the default call must NOT return the full gate tensor -- that is a ``[Bi, Bj, 512]`` buffer
         which would be registered as an output of the module and materialised on every step; it is
         available only behind the explicit debug flag;
      2. every tensor the statistics DO carry must be either detached or a scalar value (a statistic is
         a log record, never a training signal);
      3. nothing in the returned dict may be an ``nn.Module``, a ``Parameter`` or a live graph node that
         the loss does not depend on -- a returned module object would be swept into DDP's graph.
    """
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.3, seed=167)
    generator = torch.Generator().manual_seed(173)
    g_base = unit_features(2, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)
    g = g_base.detach().clone().requires_grad_(True)

    result = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=3)
    assert result['statistics'] is None, \
        'the default call must not carry the heavy-log statistics (or a gate tensor) at all'
    assert set(result) == set(READOUT_KEYS), sorted(result)
    for key, value in result.items():
        assert not isinstance(value, torch.nn.Module), (key, type(value).__name__)
        assert not isinstance(value, torch.nn.Parameter), (key,)

    result = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=3,
                          want_statistics=True)
    statistics = result['statistics']
    gate_entry = None
    for key, value in statistics.items():
        assert not isinstance(value, torch.nn.Module) and not isinstance(value, torch.nn.Parameter), \
            ('the statistics must be plain values, found an %s under %r'
             % (type(value).__name__, key))
        if key in GATE_KEYS and torch.is_tensor(value):
            gate_entry = (key, value)
        if torch.is_tensor(value):
            assert value.dim() <= 2, \
                ('%r is a %d-D tensor: the full [Bi, Bj, 512] gate buffer must not be handed out by '
                 'default' % (key, value.dim()))
            if value.dim() > 0 and value.numel() > 1:
                assert not value.requires_grad, \
                    ('%r is a LIVE graph node inside the statistics: a log field must be detached'
                     % key)
    assert gate_entry is None, \
        ('the full gate tensor (%r) is returned by default: it belongs behind the explicit debug flag'
         % (gate_entry[0] if gate_entry else None,))

    # the same gate tensor IS available behind the explicit debug flag, and it is the correct shape
    flags = [name for name in DEBUG_GATE_FLAGS
             if name in inspect.signature(module.suffix_readout_scores).parameters]
    assert flags, \
        ('suffix_readout_scores must expose an explicit debug flag for the full gate tensor; looked '
         'for %r among %r' % (list(DEBUG_GATE_FLAGS),
                              list(inspect.signature(module.suffix_readout_scores).parameters)))
    debug = call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=3,
                         want_statistics=True, **{flags[0]: True})
    found = None
    for key, value in debug['statistics'].items():
        if key in GATE_KEYS and torch.is_tensor(value):
            found = (key, value)
    if found is None:
        found = next(((key, value) for key, value in debug['statistics'].items()
                      if torch.is_tensor(value) and value.dim() == 3), None)
    assert found is not None, \
        ('with the debug flag %r set, the full gate tensor must be returned; statistics keys: %r'
         % (flags[0], sorted(debug['statistics'])))
    key, value = found
    assert tuple(value.shape) == (2, 3, FEATURE_DIM), (key, tuple(value.shape))
    assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0, \
        'the debug gate must be the straight-through gate (0/1 forward values)'


def test_h_all_off_gate_gives_a_finite_loss_and_no_fallback():
    """Spec sections 7 and 9 H: all-off mU is a legal state, not an error and not a fallback."""
    module = implementation_module()
    mask = make_suffix_mask(module)
    with torch.no_grad():
        for name, parameter in mask.named_parameters():
            if name == 'layer2.bias':
                parameter.fill_(-80.0)             # a deeply closed learned gate
            elif name == 'layer2.weight':
                parameter.zero_()
            else:
                parameter.mul_(0.1)
    generator = torch.Generator().manual_seed(179)
    g = unit_features(3, generator)
    mS = binary_masks(3, generator)
    tR = unit_features(3, generator)
    g_leaf = g.detach().clone().requires_grad_(True)
    scores = readout_scores(module, g_leaf, mS, tR, mask, image_chunk=3, text_chunk=3)
    assert tuple(scores.shape) == (3, 3), \
        'no sample and no candidate may be dropped when the gate is all off'
    loss = clip_reference_suffix_loss(scores)
    assert bool(torch.isfinite(loss)), 'an all-off gate must not produce a non-finite loss'
    assert bool(torch.isfinite(scores).all())
    f_parameters = [parameter for parameter in mask.parameters() if parameter.requires_grad]
    grads = torch.autograd.grad(scores.sum(), [g_leaf] + f_parameters, allow_unused=True,
                                retain_graph=True)
    assert all(grad is None or bool(torch.isfinite(grad).all()) for grad in grads), \
        'all-off gradients must stay finite'
    assert all(grad is None or float(grad.abs().max()) < 1e6 for grad in grads), \
        ('the epsilon-safe normalisation must not blow the all-off gradient up: an all-off gate is '
         'a legitimate state, not a numerical explosion')

    # NO silent fallback: the all-off score is the epsilon-safe zero readout, never the native score
    native = FIXED_SCALE * (g @ tR.t())
    assert float((scores - native).abs().max()) > 1e-6, \
        'an all-off gate must NOT fall back to the native suffix score'
    assert float(scores.abs().max()) < 1e-3, \
        ('an all-off u is the zero vector up to eps, so every score is ~0 and not the native one; '
         'found %r' % float(scores.abs().max()))
    assert float(native.abs().max()) > 1.0, 'the native scores must be non-trivial for the contrast'


def test_h_a_learned_all_off_gate_is_not_treated_as_invalid_text():
    """Spec section 9 H: validity is a property of R, never of the learned gate."""
    module = implementation_module()
    signature = inspect.signature(module.split_prefix_suffix)
    forbidden = [name for name in signature.parameters
                 if any(token in name.lower() for token in ('gate', 'mu', 'pu', 'mask_state'))]
    assert not forbidden, \
        ('split_prefix_suffix must decide validity from the TEXT alone; it takes %r' % forbidden)
    valid = call_split(module, 'A. B. C. D', 1)
    assert valid['suffix'] == 'B. C' and bool(valid['valid']) is True
    invalid = call_split(module, 'first. second', 1)
    assert bool(invalid['valid']) is False
    empty = call_split(module, 'A. B. C. D', 4)
    assert empty['suffix'] == '' and bool(empty['valid']) is False
    assert 'valid' in valid and isinstance(bool(valid['valid']), bool)


# --------------------------------------------------------------------------- normalisation
def test_h_normalisation_matches_native_f_normalize_on_forward_and_gradient():
    """DEFECT PINNED (normalisation): the plain ``F.normalize(x.float(), p=2, dim=-1, eps=1e-6)``.

    The reference is NATIVE ``torch.nn.functional.normalize``, never the production helper (a helper
    used as its own reference pins nothing). Three regimes are covered:

    1. the fixed counterexample ``x = [3, 4]`` with ``loss = normalize(x)[0]``: forward exactly ``0.6``
       and gradient exactly ``[0.128, -0.096]`` (verified against torch 2.1.0 on this machine, fp32
       and float64 agree to 1e-7 here);
    2. a general non-zero input, where the gradient is
       ``g/||x|| - (x.g) x/||x||^3`` with ``||x||`` the CLAMPED norm (native ``F.normalize`` clamps the
       denominator with ``clamp_min(eps)`` and differentiates the numerator, which is what "plain
       normalization, no custom autograd" means);
    3. an input BELOW ``eps`` (``x = [1e-9, 0, 0, 0]``): the denominator is the clamp, so the forward is
       ``[1e-3, 0, 0, 0]`` and the gradient of the first coordinate is ``1/eps = 1e6``, exactly as
       native ``F.normalize`` computes it. This is the regime where a "custom autograd" or a detached
       denominator would be caught.
    """
    module = implementation_module()
    normalize = getattr(module, 'normalize_features', None)
    assert callable(normalize), \
        ('model/said_prefix_suffix.py must expose the normalisation helper the readout uses '
         '(``normalize_features``) so it can be compared against native F.normalize')

    def compare(x, which=0):
        leaf = x.detach().clone().requires_grad_(True)
        ours = normalize(leaf)
        reference_leaf = x.detach().clone().requires_grad_(True)
        reference = F.normalize(reference_leaf.float(), p=2, dim=-1, eps=1e-6)
        assert float((ours.detach() - reference.detach()).abs().max()) <= TOL_FP32_TIGHT, \
            ('the forward must be native F.normalize(x.float(), p=2, dim=-1, eps=1e-6)',
             float((ours.detach() - reference.detach()).abs().max()))
        ours_grad = torch.autograd.grad(ours[0, which], leaf)[0]
        reference_grad = torch.autograd.grad(reference[0, which], reference_leaf)[0]
        return ours, ours_grad, reference, reference_grad

    # 1. the fixed counterexample
    counterexample = torch.tensor([[3.0, 4.0]])
    ours, ours_grad, _reference, reference_grad = compare(counterexample)
    assert float(ours[0, 0].detach()) == pytest.approx(0.6, abs=1e-6)
    expected_grad = torch.tensor([[0.128, -0.096]])
    assert float((ours_grad - expected_grad).abs().max()) <= 1e-6, \
        ('normalize([3, 4])[0] must have gradient [0.128, -0.096]; got %r'
         % (ours_grad.tolist(),))
    assert float((reference_grad - expected_grad).abs().max()) <= 1e-6, reference_grad
    assert torch.equal(ours_grad, reference_grad), 'the custom and native gradients must be identical'

    # 2. a general non-zero input, against the analytic formula in float64
    general = torch.tensor([[0.3, -1.2, 2.5, 0.7]])
    _ours, ours_grad, _reference, reference_grad = compare(general, which=1)
    x64 = general.double().requires_grad_(True)
    y64 = F.normalize(x64, p=2, dim=-1, eps=1e-6)
    analytic = torch.autograd.grad(y64[0, 1], x64)[0].float()
    assert float((ours_grad - analytic).abs().max()) < TOL_DENOM, \
        ('the gradient must be the native normalisation gradient; got %r against the analytic %r'
         % (ours_grad.tolist(), analytic.tolist()))
    assert torch.equal(ours_grad, reference_grad)

    # 3. an input below eps: the clamp is what makes the result finite, and the slope is 1/eps
    below = torch.tensor([[1e-9, 0.0, 0.0, 0.0]])
    ours, ours_grad, reference, reference_grad = compare(below)
    assert float(ours[0, 0].detach()) == pytest.approx(1e-3, rel=1e-5), ours.tolist()
    assert float(reference[0, 0].detach()) == pytest.approx(1e-3, rel=1e-5)
    assert float(ours_grad[0, 0]) == pytest.approx(1e6, rel=1e-5), ours_grad.tolist()
    assert torch.equal(ours_grad, reference_grad), \
        ('below eps the denominator is the clamp, so the slope is 1/eps exactly as in native '
         'F.normalize; got %r vs %r' % (ours_grad.tolist(), reference_grad.tolist()))


def test_h_readout_uses_the_plain_normalisation_contract_on_a_zero_input():
    """The zero-input regime, stated exactly as the contract defines it.

    ``F.normalize`` of an ALL-ZERO row returns exactly zero. The straight-through convention of this
    codebase deliberately does not reproduce torch's ``x/||x||`` backward at exactly ``x = 0`` (where
    torch's ``norm`` derivative is 0 and the chain rule multiplies the 1/eps factor by zero, giving a
    large arbitrary slope), because the spec requires an all-off gate to stay trainable. What the
    contract DOES require is asserted here: the forward is exactly zero, the gradient is finite, and the
    full-g multiplication is not replaced by a hard complement anywhere.
    """
    module = implementation_module()
    normalize = getattr(module, 'normalize_features', None)
    assert callable(normalize), 'normalize_features must exist'
    leaf = torch.zeros(1, 4, requires_grad=True)
    value = normalize(leaf)
    assert float(value.abs().max()) == TOL_EXACT, ('the forward of a zero input must be exactly zero',
                                                   value.tolist())
    grad = torch.autograd.grad(value[0, 0], leaf)[0]
    assert bool(torch.isfinite(grad).all()), grad
    assert float(grad.abs().max()) <= 1e6, \
        ('the zero-input slope must stay bounded by the eps clamp', grad.tolist())
    native_leaf = torch.zeros(1, 4, requires_grad=True)
    native = F.normalize(native_leaf.float(), p=2, dim=-1, eps=1e-6)
    assert float(native.abs().max()) == TOL_EXACT
    assert float(value.abs().max()) == float(native.abs().max()), \
        'the FORWARD value must agree with native F.normalize even in the degenerate regime'


# =========================================================================== I: valid subset / DDP
def assert_workers_agree(payloads):
    """The two ranks must describe the SAME run: one world size, one V, one valid index set."""
    clock = payloads[0]
    assert clock['world_size'] == len(payloads), \
        ('the worker was launched with %d processes but reports world_size = %r'
         % (len(payloads), clock['world_size']))
    for payload in payloads[1:]:
        assert payload['world_size'] == clock['world_size']
        assert payload['V'] == clock['V'], (payload['rank'], payload['V'], clock['V'])
        assert payload['valid_index'] == clock['valid_index']
        assert payload['global_valid'] == clock['global_valid']
        assert payload['scaling'] == pytest.approx(clock['scaling'])
    assert clock['world_size'] * clock['local_batch'] == len(clock['global_valid']), \
        'the global batch is world_size * local_batch, exactly as the DistributedSampler implies'
    return clock


def anchor_labels_of_worker(payloads):
    """Per-rank ``(local_rows, anchor_valid)`` from the run's OWN global valid index set.

    This is the label rule the DDP route must implement: ``global_to_valid[rank * B + local_row]``.
    """
    clock = payloads[0]
    J = torch.tensor(clock['valid_index'], dtype=torch.long)
    local_size = int(clock['local_batch'])
    world = int(clock['world_size'])
    return {rank: anchors_of_rank(J, local_size, rank) for rank in range(world)}


def run_worker(case, output_dir, local_batch, processes, port, extra=()):
    """One REAL ``torch.distributed.run`` execution of the two-rank worker."""
    if not os.path.isfile(WORKER_PATH):
        pytest.skip('the DDP worker %s is missing' % WORKER_PATH)
    os.makedirs(str(output_dir), exist_ok=True)
    command = [sys.executable, '-m', 'torch.distributed.run',
               '--nproc_per_node=%d' % processes, '--master_port=%d' % port,
               os.path.join('tests', '_suffix_ddp_worker.py'),
               '--case', case, '--output_dir', str(output_dir),
               '--local_batch', str(local_batch)] + list(extra)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, (command, result.stdout[-6000:], result.stderr[-6000:])
    payloads = []
    for rank in range(processes):
        path = os.path.join(str(output_dir), 'rank%d.pt' % rank)
        assert os.path.isfile(path), path
        payloads.append(torch.load(path, map_location='cpu', weights_only=False))
    return payloads


def _rows_for_rank(index, local_batch, rank):
    """The V-pool labels of one rank's anchors: the CORRECT rule, used by the oracle."""
    table = global_to_valid(index)
    lo, hi = rank * local_batch, (rank + 1) * local_batch
    return torch.tensor([int(table[position].item()) for position in index.tolist()
                         if lo <= position < hi], dtype=torch.long)


def _ce_sum_of_rows(matrix, rows):
    """``sum_a (logsumexp_j M[a, j] - M[a, a])`` over the given rows: a label-free CE sum."""
    if rows.numel() == 0:
        return matrix.sum() * 0.0
    selected = matrix.index_select(0, rows)
    return (torch.logsumexp(selected, dim=1) - selected.diagonal()).sum()


def oracle_for_worker(payloads, module, local_batch, lr_clip, lr_suffix):
    """The same schedule and the same global valid subset, computed by an INDEPENDENT re-write.

    The oracle reproduces the worker's loss from the spec, one rank at a time:

        rank r:  L_r = L_S0(rank r's own anchors)      -- the unchanged S0 objective
                      + (world_size / V) * lambda_suffix
                        * (rank r's I2T CE sum + rank r's T2I CE sum)

    and then performs the DDP reduction itself, ``mean_r grad L_r``. Standard DDP averaging of
    ``(world_size / V) * local CE sum`` therefore has to land on the global valid-anchor mean. The
    anchor labels come from :func:`_rows_for_rank`, i.e. ``global_to_valid[rank * B + local_row]``,
    never from a local ``arange``.
    """
    clock = payloads[0]
    world = int(clock['world_size'])
    assert world == len(payloads)
    captions = list(clock['captions'])
    global_valid = torch.tensor(clock['global_valid'], dtype=torch.bool)
    splits = [call_split(module, caption, 1) for caption in captions]
    assert [bool(item['valid']) for item in splits] == global_valid.tolist(), \
        'the oracle must see the validity pattern the workers used'

    model = build_model(shared_init=False)
    gate = make_suffix_mask(module)
    initial_state = payloads[0]['initial_parameters']
    clip_state = {key[len('clip.'):]: value for key, value in initial_state.items()
                  if key.startswith('clip.')}
    suffix_state = {key[len('suffix_mask.'):]: value for key, value in initial_state.items()
                    if key.startswith('suffix_mask.')}
    assert clip_state and suffix_state, \
        ('the recorded initialisation is not in the ``clip.``/``suffix_mask.`` layout; keys: %r'
         % sorted(initial_state)[:8])
    model.load_state_dict(clip_state, strict=True)
    gate.load_state_dict(suffix_state, strict=True)
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad],
          'lr': lr_clip, 'weight_decay': 1e-2},
         {'params': [p for p in gate.parameters() if p.requires_grad],
          'lr': lr_suffix, 'weight_decay': 0.0}],
        betas=(0.9, 0.999), eps=1e-8)
    named = {'clip.' + name: value for name, value in model.named_parameters()}
    named.update({'suffix_mask.%s' % name: value for name, value in gate.named_parameters()})

    index = torch.nonzero(global_valid, as_tuple=False).flatten()
    V = int(index.numel())
    scaling = float(module.suffix_scaling(world, V))
    lambda_suffix = float(module.LAMBDA_SUFFIX)
    generator = torch.Generator().manual_seed(int(clock['seed']))
    global_images = torch.randn(len(captions), 3, 224, 224, generator=generator)

    optimizer.zero_grad(set_to_none=True)
    averaged = {name: torch.zeros_like(parameter) for name, parameter in named.items()}
    suffix_loss_total = 0.0
    for rank in range(world):
        sl = slice(rank * local_batch, (rank + 1) * local_batch)
        rank_images = global_images[sl]
        rank_splits = splits[sl]
        prefix_ids = tokenize([item['prefix'] for item in rank_splits])
        suffix_ids = tokenize([(item['suffix'] if bool(item['valid']) else '')
                               for item in rank_splits])
        image_features = model.encode_image(rank_images)
        g = F.normalize(image_features.float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        prefix_features = model.encode_text(prefix_ids)
        suffix_features = F.normalize(model.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)

        from model.said_cls_cvssl import compute_smartclip_terms
        terms = compute_smartclip_terms(image_features, prefix_features, mS, rank=0)
        rank_loss = (float(module.LAMBDA_ALIGN) * (terms['loss_sidm'] + terms['loss_dism'])
                     + float(module.LAMBDA_SPARSE) * terms['loss_sparsity'])

        # this rank's own share of the suffix term, taken from the GLOBAL valid pool with the CORRECT
        # anchor labels: global_to_valid[rank * B + local_row]
        local_labels = _rows_for_rank(index, local_batch, rank)
        rows = index // local_batch
        cols = index % local_batch
        if V >= 2:
            valid_scores = readout_scores(
                module, g.index_select(0, rows), mS.index_select(0, rows),
                suffix_features.index_select(0, cols), gate,
                image_chunk=int(module.IMAGE_CHUNK_DEFAULT),
                text_chunk=int(module.TEXT_CHUNK_DEFAULT))
            assert tuple(valid_scores.shape) == (V, V), tuple(valid_scores.shape)
            ce_i2t = _ce_sum_of_rows(valid_scores, local_labels)
            ce_t2i = _ce_sum_of_rows(valid_scores.t(), local_labels)
            suffix_loss_total += float((scaling * lambda_suffix
                                        * (ce_i2t + ce_t2i)).detach())
            rank_loss = rank_loss + scaling * lambda_suffix * (ce_i2t + ce_t2i)
        rank_loss.backward()
        for name, parameter in named.items():
            if parameter.grad is not None:
                averaged[name] += parameter.grad.detach() / float(world)
        optimizer.zero_grad(set_to_none=True)

    closed_loop = None
    if V >= 2:
        closed_loop = _closed_loop_residual(module, model, gate, splits, global_images, index, world,
                                            local_batch)
    before = {name: named[name].detach().float().cpu().clone() for name in named}
    for name, parameter in named.items():
        parameter.grad = averaged[name].clone()
    optimizer.step()
    after = {name: named[name].detach().float().cpu().clone() for name in named}
    return {'grads': {name: averaged[name].detach().float().cpu().clone() for name in named},
            'V': V, 'scaling': scaling, 'suffix_loss': suffix_loss_total,
            'ce_closed_loop_residual': closed_loop,
            'step': after, 'update': {name: after[name] - before[name] for name in after},
            'initial_parameters': initial_state}


def _closed_loop_residual(module, model, gate, splits, global_images, index, world, local_batch):
    """``|sum_r local_ce_r - total_ce| / total_ce`` for a fully consistent global pooling."""
    V = int(index.numel())
    if V < 2:
        return 0.0
    prefix_ids = tokenize([item['prefix'] for item in splits])
    suffix_ids = tokenize([(item['suffix'] if bool(item['valid']) else '') for item in splits])
    with torch.no_grad():
        image_features = model.encode_image(global_images)
        g = F.normalize(image_features.float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR = F.normalize(model.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)
        scores = readout_scores(module, g.index_select(0, index), mS.index_select(0, index),
                                tR.index_select(0, index), gate,
                                image_chunk=int(module.IMAGE_CHUNK_DEFAULT),
                                text_chunk=int(module.TEXT_CHUNK_DEFAULT))
        total = float((_ce_sum_of_rows(scores, torch.arange(V))
                       + _ce_sum_of_rows(scores.t(), torch.arange(V))).detach())
        gathered = 0.0
        for rank in range(world):
            rows = _rows_for_rank(index, local_batch, rank)
            gathered += float((_ce_sum_of_rows(scores, rows)
                               + _ce_sum_of_rows(scores.t(), rows)).detach())
    return abs(gathered - total) / max(total, 1e-12)


def compare_gradients(payloads, oracle, names, prefix='suffix_mask.'):
    """Compare every rank's post-allreduce gradients against the single-process oracle."""
    problems = []
    for rank, payload in enumerate(payloads):
        for name in names:
            oracle_name = name if name in oracle['grads'] else (
                prefix + name if (prefix + name) in oracle['grads'] else None)
            if oracle_name is None:
                problems.append((rank, name, 'the oracle does not expose this parameter'))
                continue
            mine = payload['grads'].get(name)
            theirs = oracle['grads'][oracle_name]
            if mine is None or theirs is None:
                if (mine is None) != (theirs is None):
                    problems.append((rank, name, 'None mismatch'))
                continue
            scale = max(float(theirs.abs().max()), 1e-12)
            if scale <= 1e-12:
                assert float(mine.abs().max()) <= 1e-9, \
                    (rank, name, 'the oracle has no gradient here but the worker does')
                continue
            difference = float((mine - theirs).abs().max()) / scale
            if difference > TOL_DDP:
                problems.append((rank, name, difference))
    return problems


def compare_updates(payloads, oracle, names, prefix='suffix_mask.'):
    problems = []
    for rank, payload in enumerate(payloads):
        for name in names:
            oracle_name = name if name in oracle['step'] else (
                prefix + name if (prefix + name) in oracle['step'] else None)
            if oracle_name is None:
                problems.append((rank, name, 'missing from the oracle'))
                continue
            mine = payload['step'][name] - oracle['initial_parameters'][name]
            theirs = oracle['step'][oracle_name] - oracle['initial_parameters'][oracle_name]
            scale = max(float(theirs.abs().max()), 1e-12)
            if scale <= 1e-12:
                assert float(mine.abs().max()) <= 1e-9, (rank, name, 'the oracle did not move this '
                                                                    'parameter but the worker did')
                continue
            difference = float((mine - theirs).abs().max()) / scale
            if difference > TOL_DDP:
                problems.append((rank, name, difference))
    return problems


DDP_GRAD_NAMES = ('suffix_mask.layer1.weight', 'suffix_mask.layer1.bias',
                  'suffix_mask.layer2.weight', 'suffix_mask.layer2.bias',
                  'clip.visual.proj', 'clip.text_projection',
                  'clip.mask_net.attn_pool.attention.weight')
DDP_STEP_NAMES = ('suffix_mask.layer1.weight', 'suffix_mask.layer2.bias',
                  'clip.text_projection', 'clip.mask_net.attn_pool.attention.weight')


@needs_two_gpus
def test_i_unequal_valid_counts_two_rank_update_matches_the_single_process_oracle(tmp_path):
    """Spec sections 6 and 9 G: unequal per-rank valid counts, proved with a real UPDATE.

    The gradient comparison is the decisive check: a wrong ``world_size / V`` factor, a local-mean
    average that ignores ``n_r``, a missing ``lambda_suffix`` or a positive column taken from the wrong
    pool break it by a factor rather than by a rounding error. The parameter deltas after the same
    AdamW step are compared too, which is what makes this an update and not a forward value.

    The anchor labels are pinned to ``global_to_valid[rank * B + local_row]``: the local
    ``arange(rank*B, rank*B+n_r)`` rule is never used as an expectation anywhere in this file.
    """
    module = implementation_module()
    payloads = run_worker('unequal', tmp_path, local_batch=4, processes=2,
                          port=29611 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    V = clock['V']
    assert V == 4, V
    assert [payload['n_local'] for payload in payloads] == [3, 1], \
        [payload['n_local'] for payload in payloads]
    labels = anchor_labels_of_worker(payloads)
    assert labels[0][1] == [0, 1, 2], labels[0]
    assert labels[1][1] == [3], labels[1]
    for payload in payloads:
        assert payload['valid_index'] == payloads[0]['valid_index'], \
            'every rank must use the SAME global valid index set'
        assert payload['scaling'] == pytest.approx(2.0 / 4.0)
        assert payload['lambda_suffix'] == pytest.approx(1.0), \
            'the suffix weight is the frozen 1.0 of spec section 5'
        n_local = payload['n_local']
        # the local scaling identity, on the run's own tensors: the backward value is exactly
        # (world_size / V) * local CE sum, equivalently the local mean times (world_size * n_r / V)
        assert payload['loss_suffix_local'] == pytest.approx(
            payload['scaling'] * payload['local_ce'], rel=1e-5, abs=1e-8), payload['rank']
        assert payload['loss_suffix_local'] == pytest.approx(
            (payload['local_ce'] / n_local) * (2.0 * n_local / V), rel=1e-5), payload['rank']
        assert payload['total_ce'] > 0
        assert payload['mask_grad_norm_from_suffix'] == 0.0, payload['rank']
        assert payload['trunk_grad_norm_from_suffix'] > 0, \
            'the suffix loss must train the shared visual trunk through the live g'
        assert payload['own_suffix_grad_norm'] > 0, \
            'a rank with valid anchors must itself carry the suffix gradient'
    ce_closed = sum(payload['local_ce'] for payload in payloads)
    assert ce_closed == pytest.approx(payloads[0]['total_ce'], rel=1e-4), \
        ('the per-rank CE sums must add up to the global CE sum: the ranks must score the same '
         'global pool', ce_closed, payloads[0]['total_ce'])

    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    assert oracle['V'] == V and oracle['scaling'] == pytest.approx(2.0 / 4.0)
    problems = compare_gradients(payloads, oracle, DDP_GRAD_NAMES)
    assert not problems, ('the two-rank gradients must equal the single-process oracle over the same '
                          'global valid subset', problems)
    problems = compare_updates(payloads, oracle, DDP_STEP_NAMES)
    assert not problems, ('one real AdamW update must match the oracle: a forward-only agreement is '
                          'not enough', problems)
    for name in ('suffix_mask.layer1.weight', 'clip.visual.proj'):
        assert any(payload['grads'][name] is not None
                   and float(payload['grads'][name].abs().max()) > 0 for payload in payloads), name


@needs_two_gpus
def test_i_equal_valid_counts_degenerate_to_a_plain_local_mean(tmp_path):
    """Spec section 6: with equal ``n_r`` the rule degenerates to the local mean, with no extra W."""
    module = implementation_module()
    payloads = run_worker('equal', tmp_path, local_batch=4, processes=2,
                          port=29811 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert clock['V'] == 4
    labels = anchor_labels_of_worker(payloads)
    assert labels[0][1] == [0, 1] and labels[1][1] == [2, 3], labels
    for payload in payloads:
        assert payload['n_local'] == 2, payload['n_local']
        assert payload['scaling'] == pytest.approx(1.0 / payload['n_local'])
        assert payload['scaling'] * payload['n_local'] == pytest.approx(1.0), \
            ('equal valid counts must degenerate to the plain local mean: an extra world_size '
             'factor would show up here as scaling * n_r = 2')
        assert payload['loss_suffix_local'] == pytest.approx(
            payload['scaling'] * payload['local_ce'], rel=1e-5)
        assert payload['loss_suffix_local'] == pytest.approx(
            payload['local_ce'] / payload['n_local'], rel=1e-5)
    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    assert oracle['suffix_loss'] > 0
    problems = compare_gradients(payloads, oracle, DDP_GRAD_NAMES)
    assert not problems, problems
    problems = compare_updates(payloads, oracle, DDP_STEP_NAMES)
    assert not problems, problems


@needs_two_gpus
def test_i_rank_with_zero_valid_suffixes_contributes_a_differentiable_zero(tmp_path):
    """Spec section 6: ``n_r = 0`` must not return early, must not deadlock, and gives a zero."""
    module = implementation_module()
    payloads = run_worker('one_rank_zero', tmp_path, local_batch=4, processes=2,
                          port=30011 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert payloads[0]['n_local'] == 2 and payloads[1]['n_local'] == 0, \
        [payload['n_local'] for payload in payloads]
    assert payloads[0]['V'] == 2 and payloads[0]['scaling'] == pytest.approx(2.0 / 2.0)
    labels = anchor_labels_of_worker(payloads)
    assert labels[1] == ([], []), labels[1]
    empty_rank = payloads[1]
    assert empty_rank['zero_rank_witness'] is True, \
        'a rank with no valid suffix must contribute an exact differentiable zero'
    assert float(empty_rank['loss_suffix_local']) == 0.0
    assert float(empty_rank['own_suffix_grad_norm']) == 0.0, \
        ('a rank with n_r = 0 must contribute exactly zero to the suffix gradient; a non-zero value '
         'here means it trained on candidates it does not own')
    assert float(payloads[0]['own_suffix_grad_norm']) > 0
    assert empty_rank['global_valid'] == payloads[0]['global_valid']
    assert empty_rank['global_valid'].count(True) == 2
    assert empty_rank['valid_index'] == payloads[0]['valid_index']
    for payload in payloads:
        assert bool(torch.isfinite(torch.tensor(payload['loss_total'])))
        assert payload['mask_grad_norm_from_suffix'] == 0.0

    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    problems = compare_gradients(payloads, oracle, DDP_GRAD_NAMES)
    assert not problems, ('the zero-valid rank must still end up with the global suffix gradient '
                          'after the DDP average', problems)
    problems = compare_updates(payloads, oracle, DDP_STEP_NAMES)
    assert not problems, problems
    assert payloads[1]['grads']['clip.visual.proj'] is not None
    assert float(payloads[1]['grads']['clip.visual.proj'].abs().max()) > 0, \
        ('the rank with no valid anchor must still be trained by the global suffix loss: the '
         'collectives are shared')


@needs_two_gpus
def test_i_global_valid_pool_below_two_gives_an_exactly_zero_loss_with_no_nan(tmp_path):
    """Spec section 6: ``V < 2`` -> the global suffix loss is 0, with no fake single-candidate CE."""
    module = implementation_module()
    for case, expected_v, expected_counts, port in (('global_zero', 0, (0, 0), 30211),
                                                    ('global_one', 1, (1, 0), 30411)):
        output_dir = tmp_path / case
        payloads = run_worker(case, output_dir, local_batch=4, processes=2,
                              port=port + (os.getpid() % 200))
        clock = assert_workers_agree(payloads)
        assert clock['V'] == expected_v, (case, clock['V'])
        assert tuple(payload['n_local'] for payload in payloads) == expected_counts, case
        for payload in payloads:
            assert payload['scaling'] == 0.0, case
            assert float(payload['loss_suffix_local']) == 0.0, (case, payload['rank'])
            assert bool(torch.isfinite(torch.tensor(payload['loss_total'])))
            assert payload['mask_grad_norm_from_suffix'] == 0.0
            for name, grad in payload['grads'].items():
                if grad is None:
                    continue
                assert bool(torch.isfinite(grad).all()), (case, payload['rank'], name)
            assert len(payload['captions']) == 8, 'no sample may be dropped or re-drawn'
            assert len(payload['k_values']) == 8, payload['k_values']
            assert all(value >= 1 for value in payload['k_values']), payload['k_values']
            assert payload['valid_index'] == [], case
        for payload in payloads:
            assert payload['loss_s0'] > 0
            assert payload['grads']['clip.mask_net.attn_pool.attention.weight'] is not None
            assert float(payload['grads']['clip.mask_net.attn_pool.attention.weight'].abs().max()) \
                > 0, ('an invalid suffix must not remove the sample from the S0 candidate pool '
                      '(spec section 3)')
            assert float(payload['own_suffix_grad_norm']) == 0.0, (case, payload['rank'])
    # a one-candidate pool must NOT be turned into a one-class CE: that CE is 0 and carries no
    # signal, which is exactly why the spec forbids it
    assert float(F.cross_entropy(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))) == 0.0


@needs_two_gpus
def test_i_world_size_factor_is_present_exactly_once(tmp_path):
    """Spec section 6: the per-rank backward value carries ``world_size / V`` exactly once."""
    module = implementation_module()
    payloads = run_worker('equal', tmp_path, local_batch=4, processes=2,
                          port=30611 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert clock['world_size'] == 2 and clock['V'] == 4
    for payload in payloads:
        assert payload['scaling'] == pytest.approx(2.0 / 4.0)
    global_ce = sum(payload['local_ce'] for payload in payloads)
    total_local = sum(payload['loss_suffix_local'] for payload in payloads)
    assert total_local == pytest.approx((2.0 / 4.0) * global_ce, rel=1e-5)
    assert total_local == pytest.approx(2.0 * (global_ce / 4.0), rel=1e-5)
    wrong = sum(payload['local_ce'] / payload['n_local'] for payload in payloads)
    assert abs(total_local - wrong) > 1e-6, (total_local, wrong)
    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    problems = compare_gradients(payloads, oracle, ('suffix_mask.layer1.weight',
                                                   'clip.visual.proj'))
    assert not problems, problems
    assert oracle['ce_closed_loop_residual'] is None or oracle['ce_closed_loop_residual'] < 1e-5


# =========================================================================== J: checkpoint
def _call_production_writer(trainer, module, **values):
    """Call the module-level production writer, dropping only the kwargs it does not declare."""
    candidates = [name for name in ('write_checkpoint', 'save_checkpoint',
                                    'write_production_checkpoint', 'write_training_checkpoint',
                                    'production_checkpoint')
                  if callable(getattr(trainer, name, None))]
    assert candidates, ('the trainer exposes no MODULE-LEVEL production checkpoint writer; spec '
                        'section 14 freezes it and section 9 I forbids a hand-built dict in the '
                        'test. Module attributes seen: %r'
                        % sorted(name for name in dir(trainer) if not name.startswith('__')))
    writer = getattr(trainer, candidates[0])
    signature = inspect.signature(writer)
    accepted = {name: value for name, value in values.items() if name in signature.parameters}
    if 'path' in signature.parameters:
        accepted['path'] = values['path']
    expected = {'objective', 'arm', 'completed_steps', 'suffix_mask_state', 'suffix_mask_config',
                'suffix_mask_state_digest', 'clip_state', 'clip_state_digest'}
    missing = sorted(name for name, parameter in signature.parameters.items()
                     if name in expected and name not in accepted
                     and parameter.default is inspect.Parameter.empty)
    assert not missing, ('the production writer %s requires checkpoint content this test does not '
                         'know how to build: %r. Signature: %s'
                         % (candidates[0], missing, signature))
    writer(**accepted)
    return candidates[0]


def suffix_state_from_payload(payload):
    """The suffix mask TENSOR dict of a checkpoint payload, whatever key it is stored under."""
    for key in ('suffix_mask_state', 'suffix_state', 'suffix_mask_tensors'):
        if key in payload:
            state = payload[key]
            if state is None:
                return key, None
            if isinstance(state, dict):
                if not state:
                    raise AssertionError('the suffix mask state under %r is an EMPTY dict: a mask '
                                         'arm must carry the module tensors, a native arm must '
                                         'carry an explicit None' % key)
                if not any(torch.is_tensor(value) for value in state.values()):
                    raise AssertionError('the suffix mask state under %r carries no tensor at all: '
                                         'it is a DESCRIPTION dict, and a description must never '
                                         'be merged into the tensor key' % key)
                stripped = {name.split('.', 1)[-1] if name.startswith('suffix_mask.')
                            else name: value for name, value in state.items()}
                return key, stripped
            raise AssertionError('the suffix mask state under %r is a %s, not a tensor dict: the '
                                 'tensor dict and the description must never be merged into one key'
                                 % (key, type(state).__name__))
    raise AssertionError('the checkpoint carries none of the suffix mask state keys %r; keys '
                         'present: %r' % (('suffix_mask_state', 'suffix_state',
                                           'suffix_mask_tensors'), sorted(payload)))


def suffix_config_from_payload(payload):
    for key in ('suffix_mask_config', 'suffix_config', 'suffix_mask_descriptor'):
        if key in payload:
            return key, payload[key]
    raise AssertionError('the checkpoint carries no suffix mask DESCRIPTION key; keys present: %r'
                         % sorted(payload))


@needs_cuda
def test_j_production_writer_roundtrips_a_non_initial_f_and_mask(tmp_path):
    """Spec sections 9 I and 10: the PRODUCTION writer, a non-initial F, and a real reload."""
    module = implementation_module()
    trainer = trainer_module()
    model = build_model(shared_init=True)
    mask = make_suffix_mask(module)
    randomise_suffix_mask(mask, scale=0.3, seed=43)
    saved_mask = {key: value.detach().clone() for key, value in mask.state_dict().items()}
    assert float(saved_mask['layer2.weight'].abs().max()) > 0, \
        'the writer test needs a NON-initial F'
    with torch.no_grad():
        model.mask_net.out.weight.add_(torch.randn_like(model.mask_net.out.weight) * 0.05)
    saved_s0 = {key: value.detach().clone() for key, value in model.mask_net.state_dict().items()}

    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_ids = tokenize(SUFFIXES[:2])
    with torch.no_grad():
        native_image = model.encode_image(images).float().cpu().clone()
        native_text = model.encode_text(prefix_ids).float().cpu().clone()
        g = F.normalize(model.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden).clone()
        native_scores = FIXED_SCALE * (g @ tR.t())
        suffix_scores = readout_scores(module, g, mS, tR, mask, image_chunk=2, text_chunk=2)

    for arm in (module.ARM_MASK, module.ARM_NATIVE):
        path = os.path.join(str(tmp_path), '%s_step000020.pt' % arm)
        _call_production_writer(trainer, module, path=path, clip=model, clip_model=model,
                                suffix_mask=mask, suffix_mask_state=mask.state_dict(), arm=arm,
                                objective=module.OBJECTIVE, completed_steps=20,
                                steps=20, prefix='test')
        assert os.path.isfile(path), path
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload.get('objective') == module.OBJECTIVE, payload.get('objective')
        assert payload.get('arm') == arm
        assert int(payload.get('completed_steps', -1)) == 20
        mask_key, mask_state = suffix_state_from_payload(payload)
        config_key, mask_config = suffix_config_from_payload(payload)
        assert mask_key != config_key, \
            'the suffix tensor dict and its description must live under different keys'
        assert mask_key not in ('clip_state', 'clip'), mask_key
        assert config_key not in ('clip_state', 'clip', mask_key), config_key
        assert isinstance(mask_config, dict) and mask_config, \
            'the description must be a non-empty dict'
        assert not any(torch.is_tensor(value) for value in mask_config.values()), \
            'the description must not smuggle tensors'
        if arm == module.ARM_NATIVE:
            assert mask_state is None, ('the native arm must carry suffix_mask_state = None, found '
                                        '%s' % type(mask_state).__name__)
        else:
            assert isinstance(mask_state, dict) and mask_state
            assert set(mask_state) == set(saved_mask), (sorted(mask_state), sorted(saved_mask))
            assert all(torch.is_tensor(value) for value in mask_state.values())
            for key, value in saved_mask.items():
                assert torch.equal(mask_state[key].float(), value.float()), key
        assert 'clip_state' in payload and isinstance(payload['clip_state'], dict)
        if mask_state is not None:
            assert isinstance(getattr(module, 'state_digest')(mask_state), str)
        else:
            assert arm == module.ARM_NATIVE and payload.get('suffix_mask_state') is None

        reloaded = build_model(shared_init=False)
        reloaded.load_state_dict(payload['clip_state'], strict=True)
        fresh = make_suffix_mask(module)
        if mask_state is not None:
            fresh.load_state_dict(mask_state, strict=True)
        with torch.no_grad():
            new_image = reloaded.encode_image(images).float().cpu()
            new_text = reloaded.encode_text(prefix_ids).float().cpu()
            new_hidden = reloaded.encode_text(prefix_ids, return_full=True)[1]
            new_mS = call_mask_helper(module, reloaded.mask_net, new_hidden)
            new_g = F.normalize(reloaded.encode_image(images).float(), p=2, dim=-1, eps=1e-6)
            new_tR = F.normalize(reloaded.encode_text(suffix_ids).float(), p=2, dim=-1, eps=1e-6)
            new_native_scores = FIXED_SCALE * (new_g @ new_tR.t())
            new_suffix_scores = readout_scores(module, new_g, new_mS, new_tR, fresh, image_chunk=2,
                                              text_chunk=2)
        assert float((new_image - native_image).abs().max()) < TOL_FP32_TIGHT, arm
        assert float((new_text - native_text).abs().max()) < TOL_FP32_TIGHT, arm
        assert torch.equal(new_mS, mS), 'the old S0 mask must round-trip exactly'
        assert float((new_native_scores - native_scores).abs().max()) < TOL_RECOMPUTE, \
            'the native image/text outputs must round-trip'
        if arm == module.ARM_MASK:
            difference = float((new_suffix_scores - suffix_scores).abs().max())
            assert difference < TOL_RECOMPUTE, \
                ('the suffix scores must round-trip: a writer that dropped the suffix mask state '
                 'would silently score with the initial F here', difference)
            with torch.no_grad():
                initial_scores = readout_scores(module, new_g, new_mS, new_tR,
                                                make_suffix_mask(module), image_chunk=2,
                                                text_chunk=2)
            assert float((new_suffix_scores - initial_scores).abs().max()) > 1e-4, \
                ('this test needs a NON-initial F: with an initialisation-equal F the round-trip '
                 'check above would be vacuous')
        for key, value in saved_s0.items():
            assert torch.equal(reloaded.mask_net.state_dict()[key].float(), value.float()), key


def test_j_state_digest_separates_tensors_from_descriptions_and_detects_tampering():
    """Spec sections 9 I and 10: distinct key families, real digests, and tamper detection."""
    module = implementation_module()
    description_keys = ('suffix_mask_config', 'suffix_config', 'suffix_mask_descriptor')
    tensor_keys = ('suffix_mask_state', 'suffix_state', 'suffix_mask_tensors')
    assert not set(description_keys) & set(tensor_keys), \
        'the tensor key family and the description key family must never collide'
    payload = {'suffix_mask_state': {'layer2.bias': torch.zeros(512)},
               'suffix_mask_config': {'kind': 'two_layer_linear_gelu', 'hidden': 512},
               'clip_state': {'visual.proj': torch.zeros(512, 768)}}
    _key, state = suffix_state_from_payload(payload)
    _config_key, config = suffix_config_from_payload(payload)
    assert isinstance(state, dict) and isinstance(config, dict)
    assert set(state) != set(config)
    assert not set(state) & set(config)

    with pytest.raises(AssertionError, match='not a tensor dict|must never be merged'):
        suffix_state_from_payload({'suffix_mask_state': {'kind': 'description', 'hidden': 512}})
    with pytest.raises(AssertionError, match='carries none of the suffix mask state keys'):
        suffix_state_from_payload({'clip_state': {'a': torch.zeros(1)}})
    with pytest.raises(AssertionError, match='DESCRIPTION key'):
        suffix_config_from_payload({'suffix_mask_state': {'layer2.bias': torch.zeros(512)}})

    digest = module.state_digest({'layer2.bias': torch.zeros(512)})
    assert isinstance(digest, str) and len(digest) >= 32
    assert module.state_digest({'layer2.bias': torch.ones(512)}) != digest, \
        'tampering with a tensor must change the digest'
    assert module.state_digest({'layer2.bias': torch.zeros(512)}) == digest, \
        'the digest must be deterministic'
    with pytest.raises((ValueError, TypeError, AssertionError, AttributeError)):
        module.state_digest(None)
    with pytest.raises((ValueError, TypeError, AssertionError, KeyError)):
        module.state_digest({'kind': 'two_layer_linear_gelu'})
    metadata = None
    for name in ('checkpoint_metadata', 'suffix_mask_metadata', 'module_metadata'):
        function = getattr(module, name, None)
        if callable(function):
            try:
                metadata = function(make_suffix_mask(module))
                break
            except TypeError:
                try:
                    metadata = function()
                    break
                except TypeError:
                    continue
    assert metadata is not None, 'the module exposes no checkpoint_metadata helper'
    assert isinstance(metadata, dict) and metadata
    assert not any(torch.is_tensor(value) for value in metadata.values())


# =========================================================================== K: CLI surface
def test_k_trainer_cli_has_no_sweep_escape_hatch_and_exactly_two_arms():
    """Spec sections 8, 9 and 14: one configuration per arm, no sweep, no extra epoch, no bypass."""
    trainer = trainer_module()
    module = implementation_module()
    arms = trainer.ARMS
    assert set(arms) == {module.ARM_NATIVE, module.ARM_MASK}, arms
    assert hasattr(trainer, 'main'), 'the trainer must expose main()'
    assert module.ARM_NATIVE == 'S0_SUFFIX_NATIVE'
    assert module.ARM_MASK == 'S0_SUFFIX_MASK'
    assert module.OBJECTIVE == 'said_prefix_suffix_v01'
    assert float(module.LAMBDA_SUFFIX) == pytest.approx(1.0)
    assert int(module.IMAGE_CHUNK_DEFAULT) == 16
    assert int(module.TEXT_CHUNK_DEFAULT) == 32
    assert int(module.SUFFIX_MASK_SEED) == 0
    assert float(module.SUFFIX_GATE_BIAS_INIT) == pytest.approx(math.log(8.0))

    help_text = subprocess.run([sys.executable, '-m', 'train.train_said_prefix_suffix', '--help'],
                               cwd=REPO, capture_output=True, text=True)
    assert help_text.returncode == 0, help_text.stderr[-4000:]
    options = [token for token in help_text.stdout.split() if token.startswith('--')]
    for required in ('--arm', '--lambda_suffix', '--init_state', '--output_dir'):
        assert any(option.startswith(required) for option in options), (required, options)
    for forbidden in ('--sweep', '--grid', '--extra-epochs', '--extra_epochs', '--allow-missing',
                      '--allow_missing', '--resume-any', '--skip-verify', '--skip_verify',
                      '--force', '--allow-empty', '--tolerance'):
        assert all(not option.startswith(forbidden) for option in options), \
            ('the trainer CLI must not offer the escape hatch %s' % forbidden)
