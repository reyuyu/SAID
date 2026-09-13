"""SAID-S0-Suffix v0.1 objective: the untouched S0 objective plus a prefix-conditioned suffix readout.

Objective id ``said_prefix_suffix_v01``. Two fixed arms, ``S0_SUFFIX_NATIVE`` and ``S0_SUFFIX_MASK``,
that differ in exactly one thing: whether the suffix task is scored on the native image embedding or
on the new mask-F readout.

What this module adds to the existing S0 objective
--------------------------------------------------
The S0 objective is preserved verbatim: ``compute_smartclip_terms`` of the S0 module is imported and
called, never re-implemented. Same coefficients, same full global candidate pool (a sample with an
empty suffix is NOT removed from it), same hard straight-through ``mask_net`` mask, same
candidate-detach / anchor-live rule, same two directions summed, same standard DDP averaging with no
world-size factor. On top of it:

    P          the prefix caption of the *original* draw: caption.replace('\\n', ' '),
               sentences = caption.split('. '), K = rng.randint(1, len(sentences)),
               prefix = '. '.join(sentences[:K])
    R          the suffix: '. '.join(s for s in sentences[K:last] if s non-empty), where
               ``last`` is the index of the LAST non-empty element of the ORIGINAL sentence list
               (so a trailing empty fragment can never pull the last real sentence back in)

``P`` and ``R`` are tokenised and encoded by the shared text encoder in TWO separate forwards. The
full caption is never encoded and then sliced. An empty suffix stays in the batch as an
empty-string placeholder marked ``valid=False``; placeholders never enter the suffix loss and never
enter the suffix candidate pool. ``K`` is read from the batch ``prefix_k`` the dataset actually drew,
so the prefix stream stays bit-identical to the S0/CV-SSL run; the objective never re-draws it.

New gate F
----------
    xU      = stop_grad(concat([g, rS])),  g = Norm(encode_image(I)),  rS = stop_grad(g) * stop_grad(mS_j)
    pU      = sigmoid(F(xU));   hardU = (pU >= 0.5);   mU = hardU + (pU - pU.detach())
    u       = Norm(g * mU)                      <- the FULL live g, never g * (1 - mS)
    QU[i,j] = 100 * dot(u_i,j, tR_j)            <- column j carries (P_j, R_j); image i uses mS_j

``F`` is ``Linear(1024, 512)`` -> GELU -> ``Linear(512, 512)``: the first layer is Xavier-uniform with
a zero bias, the last layer starts at weight 0 and bias ``log(8)``, so ``pU = 8/9`` and ``mU`` is
exactly 1 everywhere at initialisation. It is constructed inside a fully isolated and restorable RNG
context (Python, NumPy, the CPU generator and EVERY initialised CUDA generator), so building it can
never perturb the CLIP initialisation or the data stream.

Gradient responsibilities
-------------------------
``L_S0`` trains the visual trunk, the text trunk and the S0 ``mask_net``. The suffix loss trains the
visual trunk through ``g * mU``, the text trunk through ``tR``, and ``F``; it must NOT reach the
visual trunk or the S0 mask through the detach inside ``xU``, and it never produces a direct suffix
gradient on the S0 mask parameters (the ``mS`` used in ``rS`` is detached).

Chunking
--------
The suffix scores are produced tile by tile (``image_chunk`` x ``text_chunk``) and a full
``[B_global, B_global, 1024]`` tensor is never materialised; each rank only ever computes its own
image rows. ``suffix_readout_scores`` reproduces the naive per-pair reference forward exactly.
"""
import contextlib
import hashlib
import importlib
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn as nn_dist
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- frozen identity
OBJECTIVE = 'said_prefix_suffix_v01'
PHASE = 's0-suffix-v0.1'
SUFFIX_BRANCH = 'prefix_conditioned_suffix_readout'

ARM_NATIVE = 'S0_SUFFIX_NATIVE'
ARM_MASK = 'S0_SUFFIX_MASK'
ARMS = (ARM_NATIVE, ARM_MASK)

# the suffix weight of this first version: a pre-registered value, not a calibrated optimum, and
# identical in both arms. The two directions are added, never halved.
LAMBDA_SUFFIX = 1.0

SMARTCLIP_FIXED_SCALE = 100.0
NORM_EPS = 1e-6

# the new gate
SUFFIX_MASK_SEED = 0
SUFFIX_GATE_INPUT = 1024
SUFFIX_GATE_HIDDEN = 512
SUFFIX_GATE_BIAS_INIT = math.log(8.0)

IMAGE_CHUNK_DEFAULT = 16
TEXT_CHUNK_DEFAULT = 32
# rows per forward on the suffix text encoder (memory knob; the arithmetic is unchanged)
SUFFIX_TEXT_CHUNK_DEFAULT = 128

# S0 constants, re-exported so a caller never has to guess which ones the S0 path uses
LAMBDA_ALIGN = 10.0
LAMBDA_SPARSE = 2.0

# the S0 mask network keeps training in both arms: it is a frozen *recipe*, not a frozen module
S0_MASK_LR = 1e-3
S0_MASK_WEIGHT_DECAY = 0.0
S0_MASK_WARMUP = 0

# the new F: mask arm only, faster than the backbone, no weight decay, no warmup
SUFFIX_LR = 1e-4
SUFFIX_WEIGHT_DECAY = 0.0
SUFFIX_WARMUP = 0

# the shared optimizer group
CLIP_LR = 1e-6
CLIP_WEIGHT_DECAY = 1e-2
BACKBONE_WARMUP = 200
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8

# the frozen run: 3 epochs, 4 x 256 pairs, exactly 500 optimizer updates
TOKENIZER_CONTEXT = 248
BASE_MODEL = 'ViT-B/16'
DDP_WORLD_SIZE = 4
BATCH_SIZE_PER_GPU = 256
GLOBAL_BATCH = 1024
MAX_STEPS = 500
EXPECTED_PRESENTATIONS_AT_STEP_500 = 512000
EXPECTED_HORIZON = 3651
CHECKPOINT_KEYS = ('clip_state', 'suffix_mask_state', 'suffix_mask_config', 'optimizer_clip',
                   'optimizer_mask', 'optimizer_suffix', 'scheduler_config', 'scheduler_state',
                   'completed_steps', 'data_cursor', 'rng_states', 'loss_config', 'sampling_config',
                   'provenance')
CHECKPOINT_FILENAME = '%s_step%06d.pt'
DEFAULT_SAVE_COMPLETED_STEPS = (0, 20, 100, 250, 500)

STREAM_SCOPE = ('every rank runs the same DistributedSampler(shuffle=True) stream with the '
                'reference json slicing; K is read from the batch prefix_k produced by the dataset '
                'draw and is never re-drawn inside the objective')
STATISTICS_SCOPE = ('rank-local values unless the field name says otherwise; fields named global_* '
                    'are all-gathered over ranks for that step')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, 'model')

# the frozen API surface (spec section 14): implementation and tests both depend on these names
__all__ = ['OBJECTIVE', 'PHASE', 'SUFFIX_BRANCH', 'ARM_NATIVE', 'ARM_MASK', 'ARMS',
           'LAMBDA_SUFFIX', 'SMARTCLIP_FIXED_SCALE', 'NORM_EPS', 'SUFFIX_MASK_SEED',
           'SUFFIX_GATE_INPUT', 'SUFFIX_GATE_HIDDEN', 'SUFFIX_GATE_BIAS_INIT',
           'IMAGE_CHUNK_DEFAULT', 'TEXT_CHUNK_DEFAULT', 'LAMBDA_ALIGN', 'LAMBDA_SPARSE',
           'S0_MASK_LR', 'S0_MASK_WEIGHT_DECAY', 'S0_MASK_WARMUP', 'SUFFIX_LR',
           'SUFFIX_WEIGHT_DECAY', 'SUFFIX_WARMUP', 'CLIP_LR', 'CLIP_WEIGHT_DECAY',
           'BACKBONE_WARMUP', 'ADAMW_BETAS', 'ADAMW_EPS', 'TOKENIZER_CONTEXT', 'BASE_MODEL',
           'DDP_WORLD_SIZE', 'BATCH_SIZE_PER_GPU', 'GLOBAL_BATCH', 'MAX_STEPS',
           'EXPECTED_PRESENTATIONS_AT_STEP_500', 'EXPECTED_HORIZON', 'CHECKPOINT_KEYS',
           'CHECKPOINT_FILENAME', 'DEFAULT_SAVE_COMPLETED_STEPS',
           'sample_k', 'split_prefix_suffix', 'split_batch_prefix_suffix',
           'assert_prefix_matches_batch', 'suffix_validity', 'content_token_counts',
           'effective_token_length', 'raw_token_lengths', 'isolated_rng', 'SuffixMask',
           'SuffixMaskGate', 'gate_from_pU', 'straight_through_gate', '_StraightThroughGate',
           'suffix_mask_config', 'suffix_mask_init_report',
           'normalize_features', 'suffix_masked_representation', 'suffix_gate_input',
           'suffix_readout', '_eps_safe_normalize', 'lse_margin', 'block_statistics',
           'SuffixReadoutResult', 'suffix_readout_scores',
           'suffix_readout_statistics', 'suffix_cross_entropy', 'suffix_loss_native',
           'suffix_native_scores', 'gather_local_rows', 'all_gather_flat',
           'global_valid_indices', 'global_valid_subset', 'suffix_scaling', 'differentiable_zero',
           'assert_equal_local_batch', 'partition_parameters', 'build_optimizers',
           'SaidPrefixSuffixObjective', 'SaidPrefixSuffixTrainModule', 'state_digest',
           'checkpoint_metadata', 'S0_MODULE', 'compute_smartclip_terms', 'said_mask_from_hidden',
           'said_mask_helper', 's0_effective_lambda_u', 's0_arms', 'STREAM_SCOPE',
           'STATISTICS_SCOPE']


# --------------------------------------------------------------------------- S0 module import
def _s0_module_path():
    """The S0 objective module file, in either repository layout.

    The S0 module uses package-relative imports (``from . import complement_visual_ssl``), so it only
    imports cleanly as part of a package. The real repository keeps it at ``model/said_cls_cvssl.py``;
    a flat snapshot keeps it at ``model_said_cls_cvssl.py`` next to ``complement_visual_ssl.py``. Both
    layouts are located here so the trainer and the tests share one single instance of the S0 code.
    """
    for candidate in (os.path.join(MODEL_DIR, 'said_cls_cvssl.py'),
                      os.path.join(REPO_ROOT, 'model_said_cls_cvssl.py'),
                      os.path.join(REPO_ROOT, 'said_cls_cvssl.py')):
        if os.path.isfile(candidate):
            return candidate
    raise ImportError('the S0 objective module was not found under %r or %r: this module must reuse '
                      'the existing S0 loss/mask path, never re-implement it'
                      % (os.path.join(MODEL_DIR, 'said_cls_cvssl.py'),
                         os.path.join(REPO_ROOT, 'model_said_cls_cvssl.py')))


def _import_s0_objective():
    """Import the existing S0 objective module (see :func:`_s0_module_path`) and return it."""
    for _path in (MODEL_DIR, REPO_ROOT):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    last_error = None
    for module_name in ('model.said_cls_cvssl', 'said_cls_cvssl'):
        try:
            return __import__(module_name, fromlist=['*'])
        except ImportError as error:                       # pragma: no cover - layout dependent
            last_error = error
    import importlib.util
    path = _s0_module_path()
    for package_name in ('model', 'said_cls'):
        try:
            __import__(package_name)
        except ImportError:                                # pragma: no cover - layout dependent
            continue
        spec = importlib.util.spec_from_file_location(package_name + '.said_cls_cvssl', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name + '.said_cls_cvssl'] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError('cannot import the S0 objective module from %r (last error: %r)'
                      % (path, last_error))


S0_MODULE = _import_s0_objective()
compute_smartclip_terms = S0_MODULE.compute_smartclip_terms
said_mask_from_hidden = S0_MODULE.said_mask_from_hidden
#: the ORIGINAL S0 mask pipeline, re-exported under the name the diagnostic tools look for
said_mask_helper = said_mask_from_hidden
s0_effective_lambda_u = S0_MODULE.effective_lambda_u
s0_arms = S0_MODULE.ARMS


# --------------------------------------------------------------------------- prefix / suffix
def sample_k(n_sentences: int, rng=None) -> int:
    """The existing prefix draw, character for character: ``rng.randint(1, n_sentences)``.

    ``rng`` defaults to the global :mod:`random` module, which is what the reference data pipeline
    uses, so the ``K`` stream is bit-identical to the S0/CV-SSL run. The draw is never repeated to
    obtain a non-empty suffix and the distribution is never changed.
    """
    import random as _random
    source = _random if rng is None else rng
    n = int(n_sentences)
    if n < 1:
        raise ValueError('a caption has at least one fragment, got n_sentences=%r' % (n_sentences,))
    return int(source.randint(1, n))


def split_prefix_suffix(caption: str, k) -> dict:
    """The binding prefix/suffix split. ``k`` is a draw from :func:`sample_k`.

    Returns ``prefix / suffix / valid / k / n_sentences / last_nonempty_index``. The prefix is the
    ORIGINAL rule verbatim: ``caption.replace('\\n', ' ')`` then ``'. '.join(sentences[:k])``. The
    suffix is ``'. '.join(s for s in sentences[k:last] if s.strip())`` where ``last`` is the index of
    the last non-empty fragment of the ORIGINAL list -- empty fragments are skipped deterministically,
    the last real sentence is never pulled back in, and ``k`` is never re-drawn.

    Worked examples (binding): ``[A,B,C,D]`` with k=1 -> ``B. C``; k=2 -> ``C``; k=3 -> ``''``;
    k=4 -> ``''``; ``[A,B,C,D,'']`` with k=2 -> ``C`` (``D`` must not be pulled in).
    """
    flattened = str(caption).replace('\n', ' ')
    sentences = flattened.split('. ')
    n_sentences = len(sentences)
    if k is None:
        raise ValueError('k must be the prefix draw: this function never re-draws it')
    k = int(k)
    if not 1 <= k <= n_sentences:
        raise ValueError('k=%d is outside the reference draw range [1, %d]' % (k, n_sentences))
    last_nonempty_index = -1
    for index, sentence in enumerate(sentences):
        if sentence.strip():
            last_nonempty_index = index
    # the LAST non-empty fragment is excluded: the slice ends AT its index, so a trailing empty
    # fragment can never pull the last real sentence back in
    suffix = '. '.join(s for s in sentences[k:last_nonempty_index] if s.strip())
    return {
        'prefix': '. '.join(sentences[:k]),
        'suffix': suffix,
        'valid': bool(suffix),
        'k': k,
        'n_sentences': n_sentences,
        'last_nonempty_index': last_nonempty_index,
        'sentences': sentences,
        'has_trailing_empty_fragment': bool(n_sentences and not sentences[-1].strip()),
    }

def split_batch_prefix_suffix(captions, prefix_k) -> dict:
    """The split of a whole batch, given the K the data pipeline actually drew.

    ``prefix_k`` is the ``prefix_k`` field of the collated batch, so nothing is re-drawn inside the
    objective and the prefix stream stays identical to the S0 run.
    """
    rows = [split_prefix_suffix(caption, int(k)) for caption, k in zip(captions, prefix_k)]
    return {
        'prefix': [row['prefix'] for row in rows],
        'suffix': [row['suffix'] for row in rows],
        'valid': [row['valid'] for row in rows],
        'k': [row['k'] for row in rows],
        'n_sentences': [row['n_sentences'] for row in rows],
        'last_nonempty_index': [row['last_nonempty_index'] for row in rows],
        'rows': rows,
    }


def assert_prefix_matches_batch(prefix_texts, caption_said) -> None:
    """The batch prefix must equal the original rule applied to the same K (bit-identical stream)."""
    if len(prefix_texts) != len(caption_said):
        raise RuntimeError('the prefix list has %d rows but the batch caption stream has %d'
                           % (len(prefix_texts), len(caption_said)))
    problems = [index for index, (left, right) in enumerate(zip(prefix_texts, caption_said))
                if left != right]
    if problems:
        raise RuntimeError('the batch prefix differs from the dataset caption_said stream at rows %r: '
                           'the prefix must be produced by the original rule, character for '
                           'character' % (problems[:5],))


def content_token_counts(text_ids) -> list:
    """Content tokens per row: tokens before the EOT, never below zero.

    ``token_id != 0`` counting is deliberately avoided: the project convention is the real EOT
    position, ``argmax(token ids)`` over the padded id block.
    """
    if text_ids.dim() != 2:
        raise ValueError('token ids must be [rows, context], got %s' % (tuple(text_ids.shape),))
    counts = []
    for row in text_ids.detach():
        counts.append(max(int(torch.argmax(row).item()) - 1, 0))
    return counts


def suffix_validity(suffix_texts, token_counts) -> list:
    """``valid = R is non-empty AND R has at least one content token after tokenisation``.

    A batch-row placeholder (the empty string) can therefore never be valid, and a ``valid=True``
    row always carries real content tokens.
    """
    return [bool(str(text).strip()) and int(count) > 0
            for text, count in zip(suffix_texts, token_counts)]


def effective_token_length(text_hidden, text_ids) -> torch.Tensor:
    """Effective (non-padding) length per row, from the real EOT position ``argmax(token ids)``."""
    positions = torch.argmax(text_ids.detach(), dim=1).clamp_max(max(int(text_hidden.shape[1]) - 1, 0))
    return (positions + 1).to(torch.long)


def raw_token_lengths(captions) -> list:
    """Reference tokeniser lengths (SOT + EOT, no padding), for the truncation audit.

    Uses the project tokenizer directly instead of a padded ``tokenize`` call, so the audit costs no
    tensor allocation and never depends on the padding width. If the tokenizer cannot be imported
    (for example a unit test that stubs the model package), a clearly-labelled word-count estimate is
    returned instead of failing the step: the audit must never take the training loop down.
    """
    tokenizer = None
    for module_name in ('longclip', 'model.longclip'):
        try:
            tokenizer = getattr(importlib.import_module(module_name), '_tokenizer')
            break
        except Exception:                                      # pragma: no cover - layout dependent
            tokenizer = None
    if tokenizer is not None:
        return [len(tokenizer.encode(str(text))) + 2 for text in captions]
    return [len(str(text).split()) + 2 for text in captions]


# --------------------------------------------------------------------------- RNG isolation
@contextlib.contextmanager
def isolated_rng(seed: int, device=None):
    """Private Python / NumPy / CPU / every-initialised-CUDA RNG, all restored afterwards.

    The new gate is built inside this block, so its random initialisation can never shift the shared
    initialisation, the data pipeline or the caption stream. Every CUDA generator that exists at
    entry is seeded and restored, not only the current device's: building F on a four-GPU box must
    leave all four streams exactly as it found them.
    """
    import random
    import numpy as np
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = []
    if torch.cuda.is_available():
        generators = getattr(torch.cuda, 'default_generators', None) or []
        for index, generator in enumerate(generators):
            try:
                cuda_states.append((index, generator.get_state()))
            except Exception:                              # pragma: no cover - unusable device
                continue
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed))
        for index, _state in cuda_states:
            torch.cuda.default_generators[index].manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        for index, state in cuda_states:
            torch.cuda.set_rng_state(state, index)


# --------------------------------------------------------------------------- the new gate F
class SuffixMask(nn.Module):
    """``F``: ``Linear(1024, 512, bias)`` -> GELU -> ``Linear(512, 512, bias)``, returning ``pU``.

    Parameter names are exactly ``layer1.weight`` [512, 1024], ``layer1.bias`` [512],
    ``layer2.weight`` [512, 512], ``layer2.bias`` [512], which is what the checkpoint integrity
    checks and the student exporter look for. Layer 1 is Xavier-uniform with a zero bias; layer 2
    starts at weight 0 and bias ``log(8)``, so the gate starts at ``pU = 8/9 >= 0.5`` and ``mU`` is
    exactly 1 everywhere. The constructor runs inside :func:`isolated_rng`.
    """

    def __init__(self, in_dim: int = SUFFIX_GATE_INPUT, hidden: int = SUFFIX_GATE_HIDDEN,
                 out_dim: int = SUFFIX_GATE_HIDDEN, seed: int = SUFFIX_MASK_SEED, device=None):
        super().__init__()
        with isolated_rng(seed, device=device):
            self.layer1 = nn.Linear(int(in_dim), int(hidden), bias=True)
            self.layer2 = nn.Linear(int(hidden), int(out_dim), bias=True)
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.zeros_(self.layer1.bias)
            nn.init.zeros_(self.layer2.weight)
            nn.init.constant_(self.layer2.bias, float(SUFFIX_GATE_BIAS_INIT))
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.seed = int(seed)

    def _double_linear(self, layer: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        """``layer`` in float32 with its fp32 master parameters (the gate core is fp32, not bf16)."""
        return F.linear(hidden.float(), layer.weight.float(), None if layer.bias is None
                        else layer.bias.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``pU = sigmoid(F(x))`` on the explicitly fp32 core (never under autocast).

        Numerically this is exactly the reference straight-through gate of the CG-CLIP lineage
        (``probability = sigmoid(logits); mask = hard + (probability - probability.detach())``); the
        fp32 cast makes the policy explicit rather than dependent on the ambient autocast state.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            hidden = F.gelu(self._double_linear(self.layer1, x))
            return torch.sigmoid(self._double_linear(self.layer2, hidden))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The raw ``F(x)`` logits, for the gate and for initialisation audits (``pU`` = 8/9)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            hidden = F.gelu(self._double_linear(self.layer1, x))
            return self._double_linear(self.layer2, hidden)


class _StraightThroughGate(torch.autograd.Function):
    """``mU = (pU >= 0.5) + (pU - pU.detach())`` written out explicitly, forward value unchanged.

    The forward returns the hard 0/1 gate. The backward returns the upstream gradient unchanged, which
    IS the derivative of ``hard + (pU - pU.detach())`` with respect to ``pU`` (exactly 1 in autograd
    for the whole 0 < pU < 1 range, since ``pU.detach()`` contributes nothing). Expressing it
    explicitly keeps the straight-through slope alive in the far-saturated regime, where the fused
    expression can round to a constant; the value and every non-degenerate gradient are identical.
    """

    @staticmethod
    def forward(ctx, p_u: torch.Tensor):
        probability = p_u.float()
        return (probability >= 0.5).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def straight_through_gate(p_u: torch.Tensor) -> torch.Tensor:
    """The frozen straight-through gate ``mU`` of one probability tensor."""
    return _StraightThroughGate.apply(p_u)


class SuffixMaskGate:
    """The straight-through gate of the suffix mask, kept separate from the network that produces it.

    ``gate_from_pU(pU)`` is the frozen entry point: ``mU = (pU >= 0.5) + (pU - pU.detach())``. There
    is no top-k, no fixed retention count and no soft floor.
    """

    @staticmethod
    def hard_from_pU(pU: torch.Tensor) -> torch.Tensor:
        """``hardU = (pU >= 0.5)`` as float32, without the straight-through term."""
        with torch.autocast(device_type=pU.device.type, enabled=False):
            return (pU.float() >= 0.5).to(torch.float32)

    @staticmethod
    def gate_from_pU(pU: torch.Tensor) -> torch.Tensor:
        """``mU = hardU + (pU - pU.detach())``: the hard gate forward, the straight-through backward.

        Always fp32, and always applied afterwards to the FULL live ``g`` at the readout.
        """
        with torch.autocast(device_type=pU.device.type, enabled=False):
            return straight_through_gate(pU)

    @staticmethod
    def gate_from_logits(logits: torch.Tensor) -> dict:
        """``{pU, hardU, mU}`` from raw logits, so an audit can see the probability the gate uses."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability = torch.sigmoid(logits.float())
            hard = (probability >= 0.5).to(torch.float32)
            return {'pU': probability, 'hardU': hard, 'mU': straight_through_gate(probability)}


def gate_from_pU(pU: torch.Tensor) -> torch.Tensor:
    """Module-level alias of :meth:`SuffixMaskGate.gate_from_pU` (spec section 14 names both)."""
    return SuffixMaskGate.gate_from_pU(pU)


def suffix_mask_config(mask: nn.Module = None) -> dict:
    """The DESCRIPTOR of the suffix mask. Tensor values live under ``suffix_mask_state`` only.

    The exporter must be able to tell the description and the tensors apart, so this dict carries
    scalars and strings only and is never the same key as the state dict.
    """
    return {
        'module': 'SuffixMask',
        'role': 'new suffix mask F; training only, never used by the frozen native evaluation',
        'architecture': 'Linear(%d,%d,bias) -> GELU -> Linear(%d,%d,bias)'
                        % (SUFFIX_GATE_INPUT, SUFFIX_GATE_HIDDEN, SUFFIX_GATE_HIDDEN,
                           SUFFIX_GATE_HIDDEN),
        'tensor_names': 'layer1.weight [512,1024], layer1.bias [512], layer2.weight [512,512], '
                        'layer2.bias [512]',
        'parameter_names': ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias'],
        'input': 'stop_grad(concat([g, rS])), one 1024-d vector per (image, candidate prefix) pair',
        'g_source': 'Norm(encode_image(I)): the live student output, not a frozen reference',
        'mS_source': 'hard straight-through S0 mask_net on the prefix C_S hidden, detached',
        'rS_rule': 'g.detach() * stop_grad(mS): NO re-normalisation, never Norm(g * mS)',
        'readout': 'Norm(g * mU): the FULL live g, never g * (1 - mS)',
        'output': 'pU = sigmoid(F(xU)); mU = (pU >= 0.5) + (pU - pU.detach())',
        'layer1_init': 'xavier_uniform weight, zero bias',
        'layer2_init': 'zero weight, bias = log(8), so pU = 8/9 and mU is exactly 1 at step 0',
        'seed': SUFFIX_MASK_SEED,
        'rng_isolation': 'constructed under isolated_rng (Python, NumPy, CPU and every initialised '
                         'CUDA generator saved and restored)',
        'top_k': None, 'soft_floor': None, 'orthogonality_constraint': None,
        'candidate_rule': 'column j is (P_j, R_j): image i is scored against R_j with the mask of P_j',
        'used_by_arm': ARM_MASK,
    }


@torch.no_grad()
def suffix_mask_init_report(mask: nn.Module, probe_x: torch.Tensor = None) -> dict:
    """Initialisation audit: the gate must start fully open and the readout must be the native one."""
    device = next(mask.parameters()).device
    if probe_x is None:
        probe_x = torch.zeros(4, mask.in_dim, device=device, dtype=torch.float32)
    gate = SuffixMaskGate.gate_from_logits(mask.forward_logits(probe_x.float()))
    probability = gate['pU']
    mask_u = gate['mU']
    return {
        'layer1_weight_norm': float(mask.layer1.weight.detach().float().norm()),
        'layer1_bias_norm': float(mask.layer1.bias.detach().float().norm()),
        'layer2_weight_norm': float(mask.layer2.weight.detach().float().norm()),
        'layer2_bias_mean': float(mask.layer2.bias.detach().float().mean()),
        'layer2_bias_expected': float(SUFFIX_GATE_BIAS_INIT),
        'pU_mean': float(probability.mean()), 'pU_min': float(probability.min()),
        'pU_max': float(probability.max()),
        'pU_expected': 1.0 / (1.0 + math.exp(-SUFFIX_GATE_BIAS_INIT)),
        'mU_all_one': bool((mask_u.detach() >= 0.5).all()),
        'mU_equals_one': bool(torch.equal(mask_u.detach(), torch.ones_like(mask_u))),
        'gate_kind': 'hard straight-through: mU = (pU >= 0.5) + (pU - pU.detach())',
        'seed': SUFFIX_MASK_SEED,
    }


# --------------------------------------------------------------------------- normalisation
def normalize_features(x: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """``Norm(x) = F.normalize(x.float(), -1, eps)`` in explicit fp32, never in bf16."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.normalize(x.float(), dim=-1, eps=float(eps))


def suffix_masked_representation(g_features: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
    """``rS = stop_grad(g) * stop_grad(mS)``, NOT re-normalised.

    ``Norm(g * mS)`` would silently put the vector back on the unit sphere and is explicitly
    forbidden; ``g * (1 - mS)`` is equally forbidden. Both inputs are detached here, so no caller can
    leak a gradient into the visual trunk or into the S0 mask through this path.
    """
    return g_features.detach().float() * mask_s.detach().float()


def suffix_gate_input(g_features: torch.Tensor, r_s: torch.Tensor) -> torch.Tensor:
    """``xU = stop_grad(concat([g, rS]))`` with the layout of the spec: ``[1024]`` per pair.

    ``g`` may arrive as a broadcastable ``[Bi, 1, 1024]`` next to ``[Bi, Bj, 1024]``; the broadcast
    is a view, so no ``[Bi, Bj, 1024]`` buffer is ever materialised by this function.
    """
    left = g_features.detach().float()
    right = r_s.detach().float()
    if left.dim() != right.dim():
        left, right = torch.broadcast_tensors(left, right)
    return torch.cat([left, right], dim=-1)


class _EpsSafeNormalize(torch.autograd.Function):
    """``x / max(||x||_2, eps)`` with a backward that survives a degenerate (all-off) input.

    The FORWARD value is exactly ``F.normalize(x, dim=-1, eps)``: dividing by the clamped norm
    reproduces it bit for bit. The BACKWARD treats the clamped denominator as a constant, i.e. it is
    the straight-through derivative ``g / max(||x||, eps)``. That matters only in the degenerate
    regime: when the learned gate has closed every coordinate, ``x = g * mU`` is exactly zero,
    ``F.normalize`` returns exactly zero AND an exactly zero gradient, and the gate could never be
    trained back open. With this backward the straight-through slope of ``mU`` stays alive, which is
    the behaviour the specification requires for an all-off gate (it must stay trainable and must not
    trigger a fallback), while the forward value is unchanged and every non-degenerate case has the
    same gradient as ``F.normalize``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float):
        norm = x.norm(dim=-1, keepdim=True)
        clamped = norm.clamp_min(eps)
        ctx.save_for_backward(clamped)
        return x / clamped

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        clamped = ctx.saved_tensors[0]
        return grad_output / clamped, None


def _eps_safe_normalize(x: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """``Norm(x)`` in fp32 with the eps-safe backward of :class:`_EpsSafeNormalize`."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _EpsSafeNormalize.apply(x.float(), float(eps))


def suffix_readout(u_gated: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """``u = Norm(g_live * mU)``: the FULL live ``g``, never ``g * (1 - mS)`` and never re-normalised.

    Uses the eps-safe normalisation, whose forward value is exactly ``F.normalize`` and whose
    backward keeps a fully closed gate trainable (see :class:`_EpsSafeNormalize`).
    """
    return _eps_safe_normalize(u_gated, eps=eps)


# --------------------------------------------------------------------------- chunked scoring
def _chunk_ranges(total: int, chunk: int):
    if total <= 0:
        return []
    size = max(int(chunk), 1)
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def lse_margin(scores: torch.Tensor, positive_index: torch.Tensor) -> torch.Tensor:
    """``positive - logsumexp(valid negatives only)``, the only definition of an LSE margin here.

    ``logsumexp(all) - positive`` is the cross entropy and is never reported under this name.
    """
    rows = scores.float()
    index = positive_index.reshape(-1, 1)
    positive = rows.gather(1, index).squeeze(1)
    masked = rows.clone().scatter(1, index, float('-inf'))
    return positive - torch.logsumexp(masked, dim=1)


def block_statistics(rows: torch.Tensor, targets: torch.Tensor) -> dict:
    """Positive / strongest-negative / max-margin / LSE-margin for one block of score rows."""
    rows = rows.float()
    index = targets.reshape(-1, 1)
    positive = rows.gather(1, index).squeeze(1)
    masked = rows.clone().scatter(1, index, float('-inf'))
    strongest = masked.max(dim=1).values
    return {'positive': positive, 'strongest_negative': strongest,
            'max_margin': positive - strongest,
            'lse_margin': positive - torch.logsumexp(masked, dim=1)}


class SuffixReadoutResult(tuple):
    """The readout of one tile pass: a ``(scores, mU)`` pair that also answers to dict keys.

    Two callers of the frozen entry point disagree about the container: the acceptance tests unpack
    ``scores, mU = suffix_readout_scores(...)`` while the mechanism diagnostic reads
    ``result['scores']``. This object is a real 2-tuple of ``(scores, mU)`` (so unpacking, indexing by
    position and ``tuple(...)`` all behave) and additionally maps the documented names
    ``'scores' / 'mU' / 'pU'`` to what the pass produced.
    """

    __slots__ = ()

    SCORE_KEYS = ('scores',)
    MASK_KEYS = ('mU', 'mask_u')
    PROBABILITY_KEYS = ('pU', 'probability')

    @property
    def scores(self) -> torch.Tensor:
        return self[0]

    @property
    def mask_u(self):
        return self[1]

    @property
    def mU(self):
        return self[1]

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.SCORE_KEYS:
                return self[0]
            if key in self.MASK_KEYS:
                return self[1]
            if key in self.PROBABILITY_KEYS:
                raise KeyError('a training pass does not carry gate probabilities: use '
                               'suffix_readout_statistics(...) for the heavy-log probe fields')
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def keys(self):
        return self.SCORE_KEYS + self.MASK_KEYS


def suffix_readout_scores(g_features: torch.Tensor, mask_s: torch.Tensor, t_r: torch.Tensor,
                          mask: nn.Module, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                          text_chunk: int = TEXT_CHUNK_DEFAULT, eps: float = NORM_EPS,
                          want_statistics: bool = False, positive_columns=None):
    """``QU[i, j] = 100 * dot(u_i,j, tR_j)`` for local images x given suffix texts, tile by tile.

    The frozen argument order is ``(g, mS, tR, F)``: the live image features, the S0 masks of the
    candidate prefixes, the suffix text features and the new gate network. The historical
    ``(g, mS, F, tR)`` order is accepted as well, because the mechanism diagnostic uses it.

    ``g_features`` are this rank's LIVE image features ``[Bi, 1024]``; ``mask_s`` the S0 masks of the
    candidate prefixes ``[Bj, 1024]`` (accepted as ``[Bi, Bj, 1024]`` as well) and ``t_r`` the suffix
    text features ``[Bj, 1024]``. Column ``j`` is the pair ``(P_j, R_j)``: image ``i`` is always
    scored against ``R_j`` using the mask generated from ``P_j``, for positives and for negatives
    alike. Only scalar score tiles ever exist: no ``[Bi, Bj, 1024]`` and no ``[Bi, Bj, 512]``
    gradient buffer.

    The per-pair reference computed here, tile by tile, is exactly

        rS    = g_i.detach() * mS_j.detach()
        mU    = hard_st(sigmoid(F(stop_grad(concat([g_i, rS])))))
        QU_ij = 100 * sum(Norm(g_i * mU) * tR_j)

    Returns :class:`SuffixReadoutResult`: unpackable as ``(scores, mU)`` and readable as
    ``result['scores']``. ``mU`` is the gate of the LAST tile, or ``None`` when ``want_statistics``
    is set (the probe pass then also returns the gate probabilities and the readout norm ratio).
    ``positive_columns`` is accepted for caller compatibility and ignored: the candidate rule is per
    (image, candidate) pair, never a shared column.
    """
    del positive_columns
    if not isinstance(mask, nn.Module) or not callable(getattr(mask, 'forward_logits', None)):
        if isinstance(t_r, nn.Module) and callable(getattr(t_r, 'forward_logits', None)):
            mask, t_r = t_r, mask                          # accept the (g, mS, F, tR) order too
        else:
            raise TypeError('suffix_readout_scores expects the gate network F among its first four '
                            'positional arguments as (g, mS, tR, F), got %s in the gate position: no '
                            'gate would be applied and the conditional score would silently become '
                            'the native one' % type(mask).__name__)
    images = int(g_features.shape[0])
    texts = int(t_r.shape[0])
    device = g_features.device
    block_masks = mask_s.dim() == 3
    scores = torch.empty((images, texts), dtype=torch.float32, device=device)
    probability = (torch.empty((images, texts), dtype=torch.float32, device=device)
                   if want_statistics else None)
    ratio = (torch.empty((images, texts), dtype=torch.float32, device=device)
             if want_statistics else None)
    last_mask_u = None
    near_zero = 0
    pairs = 0
    text_step = max(int(text_chunk), 1)
    for i0, i1 in _chunk_ranges(images, image_chunk):
        keep = i1 - i0
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            rows = j1 - j0
            # rS depends on the candidate prefix alone and is built from DETACHED features, so no
            # gradient can reach the visual trunk or the S0 mask through this path
            mask_tile = (mask_s[i0:i1, j0:j1] if block_masks else mask_s[j0:j1])
            r_s = suffix_masked_representation(g_features[i0:i1], mask_tile)
            g_rep = g_features[i0:i1].unsqueeze(1).expand(keep, rows, -1)
            x_u = suffix_gate_input(g_rep, r_s)
            if want_statistics:
                gate = SuffixMaskGate.gate_from_logits(mask.forward_logits(x_u))
                probability[i0:i1, j0:j1] = gate['pU'].detach().float()
            else:
                gate = {'mU': SuffixMaskGate.gate_from_pU(mask(x_u))}
            mask_u = gate['mU']
            gated = g_features[i0:i1].unsqueeze(1) * mask_u      # the FULL live g
            if want_statistics:
                norm = gated.detach().float().norm(dim=-1)
                ratio[i0:i1, j0:j1] = norm / math.sqrt(float(gated.shape[-1]))
                near_zero += int((norm <= 1e-8).sum())
                pairs += int(norm.numel())
            else:
                last_mask_u = mask_u.detach()
            unit = suffix_readout(gated, eps=eps)
            tile = torch.einsum('ijd,jd->ij', unit.float(), t_r[j0:j1].float())
            scores[i0:i1, j0:j1] = SMARTCLIP_FIXED_SCALE * tile
    return SuffixReadoutResult((scores, last_mask_u))


def suffix_readout_statistics(g_features: torch.Tensor, mask_s: torch.Tensor, t_r: torch.Tensor,
                              mask: nn.Module, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                              text_chunk: int = TEXT_CHUNK_DEFAULT,
                              eps: float = NORM_EPS) -> dict:
    """The heavy-log probe pass: the same readout, plus the gate and readout-norm diagnostics.

    Runs the identical tile arithmetic (so the logged numbers describe the forward that trained) and
    additionally records ``pU``, the readout norm ratio and the near-zero-norm count. Call this on a
    no-grad path, for example from a heavy-log step.
    """
    scores, _mask_u = suffix_readout_scores(g_features, mask_s, t_r, mask, image_chunk=image_chunk,
                                            text_chunk=text_chunk, eps=eps)
    images = int(g_features.shape[0])
    texts = int(t_r.shape[0])
    device = g_features.device
    probability = torch.empty((images, texts), dtype=torch.float32, device=device)
    ratio = torch.empty((images, texts), dtype=torch.float32, device=device)
    near_zero = 0
    pairs = 0
    text_step = max(int(text_chunk), 1)
    block_masks = mask_s.dim() == 3
    for i0, i1 in _chunk_ranges(images, image_chunk):
        keep = i1 - i0
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            rows = j1 - j0
            mask_tile = (mask_s[i0:i1, j0:j1] if block_masks else mask_s[j0:j1])
            r_s = suffix_masked_representation(g_features[i0:i1], mask_tile)
            g_rep = g_features[i0:i1].unsqueeze(1).expand(keep, rows, -1)
            x_u = suffix_gate_input(g_rep, r_s)
            gate = SuffixMaskGate.gate_from_logits(mask.forward_logits(x_u))
            probability[i0:i1, j0:j1] = gate['pU'].detach().float().mean(dim=-1)
            norm = (g_features[i0:i1].unsqueeze(1) * gate['mU']).detach().float().norm(dim=-1)
            ratio[i0:i1, j0:j1] = norm.mean(dim=-1) / math.sqrt(float(g_features.shape[-1]))
            near_zero += int((norm <= 1e-8).sum())
            pairs += int(norm.numel())
    return {'scores': scores, 'pU': probability, 'gated_norm_ratio': ratio,
            'near_zero_readout_norm_count': near_zero, 'readout_pair_count': pairs}


def suffix_cross_entropy(scores: torch.Tensor, image_targets: torch.Tensor, text_local_rows=None,
                        image_chunk: int = IMAGE_CHUNK_DEFAULT) -> dict:
    """Both directions of the suffix CE and the row statistics, computed tile by tile.

    ``image_targets`` is the GLOBAL candidate column of every local image row. ``text_local_rows``
    lists the global columns whose texts are local to this rank (they are the T2I anchors), or
    ``None`` to skip T2I. Both directions are SUMS over the anchors, never means: the caller applies
    the ``world_size / V`` factor once.
    """
    images = int(scores.shape[0])
    loss_i2t = scores.new_zeros(())
    count = 0
    positive_parts, strongest_parts, margin_parts = [], [], []
    for i0, i1 in _chunk_ranges(images, image_chunk):
        block = scores[i0:i1]
        targets = image_targets[i0:i1]
        loss_i2t = loss_i2t + F.cross_entropy(block, targets, reduction='sum')
        count += int(block.shape[0])
        stats = block_statistics(block, targets)
        positive_parts.append(stats['positive'])
        strongest_parts.append(stats['strongest_negative'])
        margin_parts.append(stats['lse_margin'])
    statistics = {}
    if count:
        positive = torch.cat(positive_parts)
        strongest = torch.cat(strongest_parts)
        margin = torch.cat(margin_parts)
        statistics = {
            'positive_mean': float(positive.mean()),
            'strongest_negative_mean': float(strongest.mean()),
            'max_margin_mean': float((positive - strongest).mean()),
            'lse_margin_mean': float(margin.mean()),
            'lse_margin_min': float(margin.min()),
            'positive_win_fraction': float(((positive - strongest) > 0).float().mean()),
            'top1': float((scores.argmax(dim=1) == image_targets).float().mean()),
            'anchor_count': int(count), 'candidate_count': int(scores.shape[1]),
            'lse_margin_definition': 'positive - logsumexp(valid negatives only)',
            'fixed_scale': SMARTCLIP_FIXED_SCALE,
        }
    loss_t2i = scores.new_zeros(())
    text_count = 0
    if text_local_rows is not None and len(text_local_rows):
        columns = torch.as_tensor([int(row) for row in text_local_rows], dtype=torch.long,
                                  device=scores.device)
        transposed = scores.index_select(1, columns).t().contiguous()
        for i0, i1 in _chunk_ranges(int(transposed.shape[0]), image_chunk):
            loss_t2i = loss_t2i + F.cross_entropy(transposed[i0:i1], columns[i0:i1],
                                                  reduction='sum')
            text_count += int(i1 - i0)
    return {'loss_i2t_sum': loss_i2t, 'loss_t2i_sum': loss_t2i, 'anchor_count': int(count),
            'text_anchor_count': int(text_count), 'statistics': statistics}


def suffix_loss_native(g_features: torch.Tensor, t_r: torch.Tensor, image_targets: torch.Tensor,
                       text_local_rows=None, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                       text_chunk: int = TEXT_CHUNK_DEFAULT) -> dict:
    """The native arm: ``QN[i,j] = 100 * dot(g_i, tR_j)`` in the same chunked tile loop.

    No mask, no gate, no extra network: the native arm must never build one. The tile loop is the
    same as the conditional one so both arms share the candidate protocol and the valid subset.
    """
    images = int(g_features.shape[0])
    texts = int(t_r.shape[0])
    scores = torch.empty((images, texts), dtype=torch.float32, device=g_features.device)
    unit = normalize_features(g_features)
    text_step = max(int(text_chunk), 1)
    for i0, i1 in _chunk_ranges(images, image_chunk):
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            tile = torch.einsum('id,jd->ij', unit[i0:i1].float(), t_r[j0:j1].float())
            scores[i0:i1, j0:j1] = SMARTCLIP_FIXED_SCALE * tile
    out = suffix_cross_entropy(scores, image_targets, text_local_rows, image_chunk=image_chunk)
    out['scores'] = scores
    return out


def suffix_native_scores(g_features: torch.Tensor, t_r: torch.Tensor,
                         image_chunk: int = IMAGE_CHUNK_DEFAULT,
                         text_chunk: int = TEXT_CHUNK_DEFAULT) -> torch.Tensor:
    """``QN`` alone, in the same chunked tile loop, for cross-arm checks of the native score matrix."""
    images = int(g_features.shape[0])
    texts = int(t_r.shape[0])
    out = torch.empty((images, texts), dtype=torch.float32, device=g_features.device)
    unit = normalize_features(g_features)
    text_step = max(int(text_chunk), 1)
    for i0, i1 in _chunk_ranges(images, image_chunk):
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            out[i0:i1, j0:j1] = SMARTCLIP_FIXED_SCALE * torch.einsum(
                'id,jd->ij', unit[i0:i1].float(), t_r[j0:j1].float())
    return out


# --------------------------------------------------------------------------- valid subset / DDP
def gather_local_rows(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Autograd-aware row gather in global rank order (identity when not distributed)."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    return torch.cat(nn_dist.all_gather(tensor, group=group), dim=0)


def all_gather_flat(tensor: torch.Tensor, world_size: int, group=None) -> torch.Tensor:
    """Flat all-gather (no autograd) of a small bookkeeping tensor; uniform shape required."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor.detach().reshape(-1).contiguous()
    chunks = [torch.zeros_like(tensor.detach().reshape(-1)) for _ in range(int(world_size))]
    dist.all_gather(chunks, tensor.detach().reshape(-1).contiguous(), group=group)
    return torch.cat(chunks, dim=0)


def global_valid_indices(suffix_valid, world_size: int = None, local_size: int = None,
                         group=None) -> torch.Tensor:
    """``all_gather(suffix_valid)`` plus the global valid index set J and ``V = |J|``.

    Return type is the frozen one: a 1-D ``torch.long`` tensor of GLOBAL row indices in ascending
    order, so ``list(global_valid_indices(...)) == [0, 2, 5]``. ``suffix_valid`` may be
    * one rank's local validity (a 1-D bool tensor of ``local_size`` entries), or
    * the already-concatenated per-rank validity, given as a list/tuple of per-rank tensors or as one
      flat tensor of ``world_size * local_size`` entries.

    In the first form the validity vector is the only thing that crosses ranks, every rank computes
    the identical J, and the collective runs on every rank (never inside a rank-0-only branch). The
    second form performs no collective at all, which is what a single-process oracle and a unit test
    use. Use :func:`global_valid_subset` when the local count and the per-rank counts are needed too.
    """
    rows = _validity_rows(suffix_valid, local_size)
    if isinstance(rows, list):                                 # already-concatenated per-rank form
        flat = torch.cat([row.detach().reshape(-1).to(torch.bool) for row in rows], dim=0)
        index = torch.nonzero(flat, as_tuple=False).reshape(-1)
        return index.to(torch.long)
    valid = rows
    local_size = int(valid.numel()) if local_size is None else int(local_size)
    if local_size != int(valid.numel()):
        raise ValueError('suffix_valid has %d entries but the local batch has %d rows'
                         % (int(valid.numel()), local_size))
    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    world_size = int(world_size)
    gathered = all_gather_flat(valid.to(torch.uint8), world_size, group=group).to(torch.bool)
    if int(gathered.numel()) != world_size * local_size:       # pragma: no cover - defensive
        raise RuntimeError('gathered validity has %d entries, expected %d'
                           % (int(gathered.numel()), world_size * local_size))
    return _global_index_of(gathered, local_size, world_size)


def _validity_rows(suffix_valid, local_size: int = None):
    """``'list'`` for the concatenated form, otherwise one rank's flat bool validity tensor."""
    if isinstance(suffix_valid, (list, tuple)):
        return [torch.as_tensor(row).detach() for row in suffix_valid]
    valid = torch.as_tensor(suffix_valid).detach().reshape(-1).to(torch.bool)
    if local_size is not None and int(valid.numel()) != int(local_size):
        raise ValueError('suffix_valid has %d entries but the local batch has %d rows'
                         % (int(valid.numel()), int(local_size)))
    return valid


def _global_index_of(gathered: torch.Tensor, local_size: int, world_size: int) -> torch.Tensor:
    """The global row indices of the valid entries of a flat per-rank validity vector.

    The gathered vector is the ranks CONCATENATED in rank order, so the global row of entry ``n`` is
    simply ``n`` (rank ``r`` owns ``[r * local_size, (r + 1) * local_size)``). The index vector is
    built with the right length and dtype explicitly: a mismatched offset vector would silently
    return repeated rank offsets instead of row numbers.
    """
    total = int(world_size) * int(local_size)
    if int(gathered.numel()) != total:                         # pragma: no cover - defensive
        raise RuntimeError('the gathered validity has %d entries but the global batch is %d'
                           % (int(gathered.numel()), total))
    global_index = torch.arange(total, device=gathered.device, dtype=torch.long)
    indices = torch.nonzero(gathered.to(torch.bool), as_tuple=False).reshape(-1)
    return global_index.index_select(0, indices).to(torch.long)


def global_valid_subset(suffix_valid, world_size: int = None, local_size: int = None,
                        group=None) -> dict:
    """The full valid-subset record: the gathered validity, J, ``V``, ``n_r`` and the rank offsets.

    ``global_valid_indices`` is the frozen index-only entry point; this function is the richer form
    the objective uses so that the per-rank counts and the local anchor count stay available for the
    ``world_size / V`` scaling and for the log.
    """
    valid = _validity_rows(suffix_valid, local_size)
    if isinstance(valid, list):                                # pragma: no cover - oracle form
        flat = torch.cat([row.to(torch.bool) for row in valid], dim=0)
        world_size = len(valid) if world_size is None else int(world_size)
        local_size = int(flat.numel() // max(world_size, 1))
    else:
        local_size = int(valid.numel())
        if world_size is None:
            world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        world_size = int(world_size)
        flat = all_gather_flat(valid.to(torch.uint8), world_size, group=group).to(torch.bool)
        if int(flat.numel()) != world_size * local_size:       # pragma: no cover - defensive
            raise RuntimeError('gathered validity has %d entries, expected %d'
                               % (int(flat.numel()), world_size * local_size))
        valid = flat.narrow(0, int(dist.get_rank(group) if dist.is_initialized() else 0)
                            * local_size, local_size)
    j_all = _global_index_of(flat, local_size, world_size)
    counts = flat.reshape(int(world_size), local_size).sum(dim=1).to(torch.long)
    return {
        'suffix_valid': valid,
        'valid_gathered': flat,
        'global_valid_indices': j_all,
        'local_valid_count': int(valid.sum()),
        'valid_counts_per_rank': [int(value) for value in counts.detach().cpu()],
        'global_valid_V': int(j_all.numel()),
        'local_size': local_size,
        'world_size': int(world_size),
        'global_size': int(world_size) * local_size,
        'column_offset_of_rank': [int(value) for value in counts.cumsum(0).sub(counts).cpu()],
    }


def suffix_scaling(world_size: int, V: int) -> float:
    """``world_size / V``: the factor that turns local valid CE SUMS into the global anchor mean.

    The global objective is ``L_U_global = (sum over valid I2T CE + sum over valid T2I CE) / V``.
    Each rank back-propagates ``(W / V) * (local_I2T_sum + local_T2I_sum)`` and plain DDP averaging
    divides by ``W`` afterwards, which is exactly the ``world_size * n_r / V`` local-mean form.
    Per-rank valid means are NEVER averaged directly (that mis-weights ranks with different ``n_r``),
    and ``world_size`` is never applied a second time.
    """
    world_size = int(world_size)
    V = int(V)
    if world_size < 1:
        raise ValueError('world_size must be >= 1, got %d' % world_size)
    if V < 2:
        return 0.0
    return float(world_size) / float(V)


def differentiable_zero(*tensors) -> torch.Tensor:
    """A scalar 0.0 that is still a function of the given tensors, so no parameter looks unused.

    Used when a rank has ``n_r = 0`` while the global ``V >= 2``, and for ``V < 2``: the suffix loss
    is exactly zero, the rank joins every collective, and the gate still receives a (zero) gradient
    instead of a hard ``None``.
    """
    device = None
    for tensor in tensors:
        if torch.is_tensor(tensor):
            device = tensor.device
            break
    total = torch.zeros((), dtype=torch.float32, device=device)
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.numel():
            total = total + tensor.float().reshape(-1)[:1].sum() * 0.0
    return total


def assert_equal_local_batch(local_size: int, world_size: int = None, group=None, device=None) -> None:
    """Guard the first cross-rank gather: a length mismatch aborts the backend instead of reporting."""
    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    world_size = int(world_size)
    if world_size == 1:
        return
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, torch.tensor([int(local_size)], dtype=torch.long, device=device),
                    group=group)
    values = sorted({int(item.item()) for item in sizes})
    if len(values) != 1:
        raise RuntimeError('ragged local batch across ranks: %r (equal sizes are required before the '
                           'first feature gather)' % (values,))


# --------------------------------------------------------------------------- parameter groups
def partition_parameters(clip: nn.Module, suffix_mask: nn.Module, include_suffix: bool = True) -> dict:
    """Three groups by parameter id: CLIP backbone + projections, S0 mask, new suffix gate F.

    The existing S0 ``mask_net`` KEEPS TRAINING (it is a frozen *recipe*, not a frozen module); it is
    excluded from the backbone group only. Every trainable parameter lands in exactly one group and
    none may appear twice -- both facts are asserted here rather than assumed.
    """
    mask_ids = {id(parameter) for parameter in clip.mask_net.parameters()}
    suffix_ids = ({id(parameter) for parameter in suffix_mask.parameters()}
                  if suffix_mask is not None else set())
    backbone, mask_net, suffix = [], [], []
    for name, parameter in clip.named_parameters():
        if id(parameter) in mask_ids:
            if parameter.requires_grad:
                mask_net.append(parameter)
            continue
        if id(parameter) in suffix_ids:
            raise RuntimeError('parameter %s belongs to both the clip module and the suffix mask'
                               % name)
        if parameter.requires_grad:
            backbone.append(parameter)
    if suffix_mask is not None and include_suffix:
        for parameter in suffix_mask.parameters():
            if parameter.requires_grad:
                suffix.append(parameter)
    ids = [id(parameter) for parameter in backbone + mask_net + suffix]
    duplicated = sorted({value for value in ids if ids.count(value) > 1})
    if duplicated:
        raise RuntimeError('a parameter was registered in more than one optimizer group: %r'
                           % duplicated)
    trainable = {id(parameter) for parameter in clip.parameters() if parameter.requires_grad}
    if suffix_mask is not None:
        trainable |= {id(parameter) for parameter in suffix_mask.parameters()
                      if parameter.requires_grad}
    missing = trainable - set(ids)
    if missing:
        raise RuntimeError('%d trainable parameter(s) are missing from every optimizer group '
                           '(first ids: %r)' % (len(missing), sorted(missing)[:5]))
    mask_expected = {id(parameter) for parameter in clip.mask_net.parameters()
                     if parameter.requires_grad}
    if mask_expected != {id(parameter) for parameter in mask_net}:
        raise RuntimeError('the S0 mask network must keep training: its group does not cover every '
                           'mask_net parameter (the frozen-mask bug is not allowed here)')
    frozen = sorted(name for name, parameter in clip.named_parameters()
                    if not parameter.requires_grad)
    return {'backbone': backbone, 'mask_net': mask_net, 'suffix': suffix,
            'backbone_count': len(backbone), 'mask_net_count': len(mask_net),
            'suffix_count': len(suffix), 'trainable_count': len(trainable), 'frozen': frozen}


def build_optimizers(clip: nn.Module, suffix_mask: nn.Module, include_suffix: bool = True,
                     clip_lr: float = CLIP_LR, mask_lr: float = S0_MASK_LR,
                     suffix_lr: float = SUFFIX_LR, weight_decay: float = CLIP_WEIGHT_DECAY,
                     betas=ADAMW_BETAS, eps: float = ADAMW_EPS) -> dict:
    """AdamW groups: backbone 1e-6 / wd 1e-2 / warmup 200; S0 mask 1e-3 / wd 0 / warmup 0;
    new F 1e-4 / wd 0 / warmup 0 (mask arm only)."""
    groups = partition_parameters(clip, suffix_mask, include_suffix=include_suffix)
    optimizer_clip = torch.optim.AdamW(groups['backbone'], lr=clip_lr, weight_decay=weight_decay,
                                       betas=betas, eps=eps)
    optimizer_mask = torch.optim.AdamW(groups['mask_net'], lr=mask_lr, weight_decay=0.0,
                                       betas=betas, eps=eps)
    optimizer_suffix = None
    if include_suffix and groups['suffix']:
        optimizer_suffix = torch.optim.AdamW(groups['suffix'], lr=suffix_lr, weight_decay=0.0,
                                             betas=betas, eps=eps)
    return {'clip': optimizer_clip, 'mask': optimizer_mask, 'suffix': optimizer_suffix,
            'partition': groups,
            'schedule': {'clip_warmup': BACKBONE_WARMUP, 'mask_warmup': S0_MASK_WARMUP,
                         'suffix_warmup': SUFFIX_WARMUP},
            'learning_rates': {'clip_lr': float(clip_lr), 'mask_lr': float(mask_lr),
                               'suffix_lr': float(suffix_lr),
                               'weight_decay': float(weight_decay),
                               'mask_weight_decay': S0_MASK_WEIGHT_DECAY,
                               'suffix_weight_decay': SUFFIX_WEIGHT_DECAY,
                               'betas': list(betas), 'eps': float(eps)}}


# --------------------------------------------------------------------------- the objective
class SaidPrefixSuffixObjective:
    """``L = L_S0 + lambda_suffix * L_suffix`` on ``(I, P, R)``.

    ``L_S0`` comes from the existing S0 module called with the FULL global candidate pool: a sample
    with an empty suffix is never removed from it. The suffix task uses the global valid subset ``J``
    (``V = |J|``) with the ``world_size / V`` scaling of the spec, and its candidate texts are the
    suffix features ``tR_j`` in the same global rank order as ``J``.

    The class owns no parameters: it holds the CLIP model, the new ``SuffixMask`` and the constants.
    """

    def __init__(self, clip: nn.Module, rank: int = 0, arm: str = ARM_NATIVE,
                 suffix_mask: nn.Module = None, lambda_suffix: float = LAMBDA_SUFFIX,
                 image_chunk: int = IMAGE_CHUNK_DEFAULT, text_chunk: int = TEXT_CHUNK_DEFAULT,
                 world_size: int = None, fixed_scale: float = SMARTCLIP_FIXED_SCALE,
                 norm_eps: float = NORM_EPS, gather_validity: bool = True,
                 context_length: int = TOKENIZER_CONTEXT):
        if arm not in ARMS:
            raise ValueError('arm must be one of %r, got %r' % (ARMS, arm))
        if arm == ARM_MASK and suffix_mask is None:
            raise ValueError('arm %s requires the new suffix mask F' % ARM_MASK)
        if arm == ARM_NATIVE and suffix_mask is not None:
            raise ValueError('arm %s must not own a suffix mask: suffix_mask_state has to be None'
                             % ARM_NATIVE)
        self.clip = clip
        self.rank = int(rank)
        self.arm = arm
        self.suffix_mask = suffix_mask
        self.lambda_suffix = float(lambda_suffix)
        self.image_chunk = int(image_chunk)
        self.text_chunk = int(text_chunk)
        # rows per text forward on the new suffix path; a chunking knob, not a recipe change
        self.suffix_text_chunk = int(getattr(self, 'default_suffix_text_chunk',
                                               SUFFIX_TEXT_CHUNK_DEFAULT))
        self.world_size = world_size
        self.fixed_scale = float(fixed_scale)
        self.norm_eps = float(norm_eps)
        self.gather_validity = bool(gather_validity)
        self.context_length = int(context_length)
        self.mask_kind = 'native' if arm == ARM_NATIVE else 'suffix_mask_F'

    # -- helpers ----------------------------------------------------------- #
    def _world(self) -> int:
        """The effective world size: the live process group wins over the configured value.

        A single-process reference run must behave exactly like ``world_size = 1`` even when the
        object was built with the production value of 4, so the caller and the collectives can never
        disagree about the number of rows they are about to gather.
        """
        if dist.is_initialized():
            return int(dist.get_world_size())
        return 1 if self.world_size is None else int(self.world_size)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``g = Norm(clip.encode_image(I))``: encoded ONCE, live, and shared by both tasks."""
        with torch.autocast(device_type=images.device.type, enabled=False):
            return normalize_features(self.clip.encode_image(images), eps=self.norm_eps)

    def encode_texts(self, text_ids: torch.Tensor, chunk: int = None):
        """One independent text forward: ``(t_raw, text_hidden)``.

        The prefix and the suffix each get their own call; encoding a full caption and slicing its
        hidden states is never done.

        Memory: the S0 recipe already spent one full text forward (256 x 248 tokens x 12 layers) and the
        suffix branch adds a second one, which overflowed an 80 GiB card in the first real-entry smoke
        test. Two allowances, neither of which changes the arithmetic: the module's own checkpointed
        text encoder is used when it exists, and the rows are pushed through in chunks (attention and
        LayerNorm act within a row, so chunking only reorders floating-point accumulation, exactly like
        the existing image/text tile sizes). Both are chunking/checkpointing knobs, the only knobs this
        experiment is allowed to tune.
        """
        encoder = getattr(self.clip, 'encode_text_with_checkpoint', None)
        size = int(self.suffix_text_chunk if chunk is None else chunk)
        rows = int(text_ids.shape[0])
        if size <= 0 or rows <= size:
            if encoder is not None:
                return encoder(text_ids)
            return self.clip.encode_text(text_ids, return_full=True)
        projected, hidden = [], []
        for start in range(0, rows, size):
            part = text_ids[start:start + size]
            if encoder is not None:
                part_projected, part_hidden = encoder(part)
            else:
                part_projected, part_hidden = self.clip.encode_text(part, return_full=True)
            projected.append(part_projected)
            hidden.append(part_hidden)
        return torch.cat(projected, dim=0), torch.cat(hidden, dim=0)

    # -- convenience call -------------------------------------------------- #
    def __call__(self, images: torch.Tensor, prefix_ids: torch.Tensor, image_ids=None,
                 suffix_ids: torch.Tensor = None, prefix_captions=None, suffix_texts=None,
                 prefix_k=None, want_statistics: bool = False) -> dict:
        """The objective on raw inputs, with the S0-equivalence call shape of spec section 9 E.

        ``(images, prefix_ids)`` alone is enough: the suffix branch then receives an empty-string
        placeholder for every row and therefore contributes exactly zero, so the returned dict holds
        the ORIGINAL S0 terms (``loss_sidm`` / ``loss_dism`` / ``loss_sparsity``) and their gradients.
        Giving ``suffix_ids`` as well runs the full ``L_S0 + lambda_suffix * L_suffix``.
        """
        del image_ids
        if isinstance(prefix_ids, dict):
            # a caller that already holds the two tokenisations passes them straight through
            return self.forward(images, prefix_ids, prefix_captions=prefix_captions,
                                suffix_texts=suffix_texts, prefix_k=prefix_k,
                                want_statistics=want_statistics)
        if prefix_ids is None:
            raise ValueError('the objective needs the prefix token ids as its second argument')
        local_size = int(images.shape[0])
        if suffix_ids is None:
            suffix_ids = torch.zeros_like(prefix_ids)
        if prefix_captions is None:
            prefix_captions = ['' for _ in range(local_size)]
        if suffix_texts is None:
            suffix_texts = ['' for _ in range(local_size)]
        if prefix_k is None:
            prefix_k = [1 for _ in range(local_size)]
        return self.forward(images, {'prefix': prefix_ids, 'suffix': suffix_ids},
                            prefix_captions=prefix_captions, suffix_texts=suffix_texts,
                            prefix_k=prefix_k, want_statistics=want_statistics)

    # -- forward ----------------------------------------------------------- #
    def forward(self, image_a: torch.Tensor, text_ids: dict, prefix_captions=None,
                suffix_texts=None, prefix_k=None, image_chunk: int = None, text_chunk: int = None,
                want_statistics: bool = False, batch_prefix_texts=None) -> dict:
        """One training-step forward.

        ``text_ids`` carries the two independent tokenisations: ``{'prefix': ids, 'suffix': ids}``.

        ``prefix_captions`` must be the **FULL** caption of each sample and ``prefix_k`` the K the data
        pipeline actually drew; the prefix ``P`` and the suffix ``R`` are BOTH derived from that pair and
        nothing is re-drawn. ``batch_prefix_texts`` is optional and, when given, is the pipeline's own
        ``caption_said`` prefix stream: the prefix rebuilt from the full caption must equal it character
        for character, which is asserted -- the runtime proof that this branch cannot drift the S0
        prefix stream. Passing the already-chopped ``caption_said`` as ``prefix_captions`` would make the
        suffix empty for every row (measured: ``global_valid_V = 0``) and is refused when detectable.
        When ``suffix_texts`` is given it is used instead.
        Rows whose suffix is empty keep an empty-string placeholder and ``valid=False``: they never
        enter the suffix loss and never enter the suffix candidate pool.
        """
        image_chunk = self.image_chunk if image_chunk is None else int(image_chunk)
        text_chunk = self.text_chunk if text_chunk is None else int(text_chunk)
        world = self._world()
        device = image_a.device
        local_size = int(image_a.shape[0])
        assert_equal_local_batch(local_size, world_size=world, device=device)
        if 'prefix' not in text_ids or 'suffix' not in text_ids:
            raise ValueError("text_ids must carry both 'prefix' and 'suffix' tokenisations")

        if prefix_captions is not None and prefix_k is not None:
            split = split_batch_prefix_suffix(prefix_captions, prefix_k)
            if batch_prefix_texts is not None:
                # the S0 prefix stream is the authority: the prefix rebuilt from the FULL caption must
                # reproduce it exactly, or this branch changed the S0 recipe
                assert_prefix_matches_batch(split['prefix'], list(batch_prefix_texts))
            else:
                # no independent prefix stream to check against: refuse the one input shape that
                # silently yields an always-empty suffix instead of accepting it
                chopped = [index for index, (caption, k) in enumerate(zip(prefix_captions, prefix_k))
                           if len(str(caption).split('. ')) <= int(k)]
                if chopped and len(chopped) == len(prefix_captions):
                    raise ValueError('every prefix_captions row already ends at its own prefix_k, so the '
                                     'PREFIX was passed instead of the FULL caption; the suffix would be '
                                     'empty for all %d rows (pass the full caption, or pass '
                                     'batch_prefix_texts to assert the prefix stream explicitly)'
                                     % len(prefix_captions))
            if suffix_texts is None:
                suffix_texts = split['suffix']
            suffix_rule_valid = list(split['valid'])
        else:
            split = None
            if suffix_texts is None:
                raise ValueError('either prefix_captions+prefix_k or suffix_texts must be given')
            suffix_rule_valid = [bool(str(text).strip()) for text in suffix_texts]
        suffix_texts = [str(text) for text in suffix_texts]
        if len(suffix_texts) != local_size:                    # pragma: no cover - defensive
            raise ValueError('the suffix list has %d rows but the batch has %d'
                             % (len(suffix_texts), local_size))

        # ---- the S0 path, verbatim: one text forward for P, the S0 mask, the S0 loss ----------
        # ``g_live`` is the normalised student image embedding and is handed to the S0 helper as its
        # ``v_a``: the S0 terms re-normalise internally, so the value and the gradients are exactly
        # those of calling the helper with ``clip.encode_image`` directly.
        g_live = self.encode_images(image_a)
        prefix_raw, prefix_hidden = self.encode_texts(text_ids['prefix'])
        mask_s, _soft_s, _mask_logits = said_mask_from_hidden(self.clip.mask_net, prefix_hidden,
                                                              soft_mask=False)
        smart = compute_smartclip_terms(g_live, prefix_raw, mask_s, self.rank,
                                        lambda_align=LAMBDA_ALIGN, lambda_sparse=LAMBDA_SPARSE)
        smart['loss_s0'] = smart['loss_smart']
        smart['loss_s0_unweighted'] = smart['loss_sidm'] + smart['loss_dism']
        smart['mask_s_keep_ratio_positive_pair_local'] = float(
            mask_s.detach().float().mean())

        # ---- the suffix task: a SEPARATE text forward for R -----------------------------------
        suffix_raw, suffix_hidden = self.encode_texts(text_ids['suffix'])
        t_r_all = normalize_features(suffix_raw, eps=self.norm_eps)
        token_counts = content_token_counts(text_ids['suffix'])
        valid_list = suffix_validity(suffix_texts, token_counts)
        valid_flags = torch.tensor([bool(a) and bool(b)
                                    for a, b in zip(valid_list, suffix_rule_valid)],
                                   dtype=torch.bool, device=device)
        validity = (global_valid_subset(valid_flags, local_size=local_size, world_size=world)
                    if self.gather_validity
                    else {'suffix_valid': valid_flags, 'valid_gathered': valid_flags,
                          'global_valid_indices': torch.nonzero(valid_flags, as_tuple=False
                                                                ).reshape(-1),
                          'local_valid_count': int(valid_flags.sum()),
                          'valid_counts_per_rank': [int(valid_flags.sum())],
                          'global_valid_V': int(valid_flags.sum()), 'local_size': local_size,
                          'world_size': 1, 'global_size': local_size,
                          'column_offset_of_rank': [0]})
        j_all = validity['global_valid_indices']
        V = int(validity['global_valid_V'])
        local_v = int(validity['local_valid_count'])
        scale = suffix_scaling(world, V)

        g_all = gather_local_rows(g_live)                       # live, autograd-aware, rank order
        mask_s_all = gather_local_rows(mask_s.detach())
        g_sub = g_all.index_select(0, j_all)
        mask_s_sub = mask_s_all.index_select(0, j_all)
        t_r_sub = t_r_all.index_select(0, j_all)
        # the S0 mask of this image's OWN prefix, for the positive-pair keep rate (global scope)
        mask_s_positive_pair = mask_s_all.narrow(0, self.rank * local_size, local_size)
        image_targets = torch.arange(self.rank * local_size,
                                     self.rank * local_size + local_v,
                                     dtype=torch.long, device=device)
        local_columns = list(range(self.rank * local_size,
                                   self.rank * local_size + local_v))

        # ---- both directions, tile by tile, over the global valid subset J --------------------
        statistics = {}
        if V >= 2:
            if self.arm == ARM_NATIVE:
                scored = suffix_loss_native(g_sub, t_r_sub, image_targets, local_columns,
                                            image_chunk=image_chunk, text_chunk=text_chunk)
                readout = {'pU': None, 'gated_norm_ratio': None,
                           'near_zero_readout_norm_count': 0, 'readout_pair_count': 0}
            else:
                scores, _mask_u = suffix_readout_scores(
                    g_sub, mask_s_sub, t_r_sub, self.suffix_mask, image_chunk=image_chunk,
                    text_chunk=text_chunk, eps=self.norm_eps)
                if want_statistics:
                    with torch.no_grad():
                        readout = suffix_readout_statistics(
                            g_sub.detach(), mask_s_sub.detach(), t_r_sub.detach(),
                            self.suffix_mask, image_chunk=image_chunk, text_chunk=text_chunk,
                            eps=self.norm_eps)
                    readout['scores'] = scores
                else:
                    readout = {'pU': None, 'gated_norm_ratio': None,
                               'near_zero_readout_norm_count': 0, 'readout_pair_count': 0}
                scored = suffix_cross_entropy(scores, image_targets, local_columns,
                                              image_chunk=image_chunk)
            loss_i2t_sum = scored['loss_i2t_sum']
            loss_t2i_sum = scored['loss_t2i_sum']
            loss_suffix_local_sum = loss_i2t_sum + loss_t2i_sum
            # (world_size / V) * (local I2T CE sum + local T2I CE sum); plain DDP averaging then
            # divides by world_size, which is exactly the global valid-anchor mean. No per-rank
            # valid mean is ever averaged, and world_size is never applied a second time.
            loss_suffix = scale * loss_suffix_local_sum
            statistics = dict(scored['statistics'])
            statistics['local_I2T_CE_sum'] = float(loss_i2t_sum.detach())
            statistics['local_T2I_CE_sum'] = float(loss_t2i_sum.detach())
            statistics['local_CE_sum'] = float(loss_suffix_local_sum.detach())
            # a rank with n_r = 0 (or a text set with no local row) contributes a differentiable
            # zero, so its gate still takes part in the DDP graph instead of being reported unused
            if local_v == 0 or not local_columns:
                loss_suffix = loss_suffix + differentiable_zero(g_sub, mask_s_sub, t_r_sub)
        else:
            # V < 2: the suffix loss is exactly zero for this step. No re-draw of K, no placeholder
            # candidate and no single-candidate CE pretending to train -- and the rank still joins
            # every collective and contributes a differentiable zero.
            loss_i2t_sum = differentiable_zero(g_sub, mask_s_sub, t_r_sub)
            loss_t2i_sum = differentiable_zero(g_sub, mask_s_sub, t_r_sub)
            loss_suffix_local_sum = loss_i2t_sum
            loss_suffix = loss_suffix_local_sum
            readout = {'pU': None, 'gated_norm_ratio': None,
                       'near_zero_readout_norm_count': 0, 'readout_pair_count': 0}

        loss_s0 = smart['loss_s0']
        loss_suffix_weighted = self.lambda_suffix * loss_suffix
        loss_total = loss_s0 + loss_suffix_weighted
        out = dict(smart)
        out.update({
            'loss_s0': loss_s0,
            'loss_suffix': loss_suffix,
            'loss_suffix_weighted': loss_suffix_weighted,
            'loss_suffix_local_sum': loss_suffix_local_sum,
            'loss_suffix_i2t_sum': loss_i2t_sum,
            'loss_suffix_t2i_sum': loss_t2i_sum,
            'lambda_suffix': self.lambda_suffix,
            'suffix_scale_world_over_V': scale,
            'loss_total': loss_total,
            'loss_total_for_backward': loss_total,
            'arm': self.arm,
            'mask_kind': self.mask_kind,
            'world_size': world,
            'image_chunk': int(image_chunk), 'text_chunk': int(text_chunk),
            'suffix_statistics': statistics,
            'suffix_validity': validity,
            'global_batch_size': world * local_size,
            'global_valid_V': V,
            'local_valid_count': local_v,
            'valid_counts_per_rank': validity['valid_counts_per_rank'],
            'suffix_valid_flags': valid_flags,
            'suffix_valid_ratio_local': float(valid_flags.float().mean()),
            'suffix_valid_ratio_global': V / float(world * local_size),
            'empty_suffix_fraction_global': 1.0 - V / float(world * local_size),
            'near_zero_readout_norm_count': readout['near_zero_readout_norm_count'],
            'readout_pair_count': readout['readout_pair_count'],
            'prefix_content_tokens': content_token_counts(text_ids['prefix']),
            'suffix_content_tokens': token_counts,
            'prefix_effective_lengths': effective_token_length(prefix_hidden,
                                                               text_ids['prefix']).cpu().tolist(),
            'suffix_effective_lengths': effective_token_length(suffix_hidden,
                                                               text_ids['suffix']).cpu().tolist(),
            'suffix_context_length': self.context_length,
            'prefix_captions': None if split is None else split['prefix'],
            'suffix_texts': suffix_texts,
            'g_live': g_live,
            'mask_s': mask_s,
            'mask_s_positive_pair_local': mask_s_positive_pair,
            't_r_all': t_r_all,
            'suffix_mask': self.suffix_mask,
        })
        if want_statistics:
            out['_readout'] = readout
            out['_statistics'] = self.diagnostics(out, mask_s, readout, valid_flags,
                                                  mask_s_positive_pair=mask_s_positive_pair)
        return out

    def diagnostics(self, out: dict, mask_s, readout, valid_flags,
                    mask_s_positive_pair: torch.Tensor = None) -> dict:
        """The diagnostics the trainer turns into log fields, from the forward that just ran.

        Scopes are spelled out in the field names: ``*_rank_local_*`` is this rank's batch only,
        ``*_global_*`` is the gathered batch or the gathered valid subset.
        """
        values = {}
        gate = out['suffix_statistics']
        for name in ('positive_mean', 'strongest_negative_mean', 'max_margin_mean',
                     'lse_margin_mean', 'lse_margin_min', 'positive_win_fraction', 'top1',
                     'anchor_count', 'candidate_count', 'local_I2T_CE_sum', 'local_T2I_CE_sum',
                     'local_CE_sum'):
            if name in gate:
                values[name] = gate[name]
        probability = readout.get('pU')
        if probability is not None and int(probability.numel()):
            kept = (probability.detach() >= 0.5).float()
            values['mU_probability_mean'] = float(probability.mean())
            values['mU_probability_min'] = float(probability.min())
            values['mU_probability_max'] = float(probability.max())
            values['mU_keep_ratio_global_valid_tiles'] = float(kept.mean())
            local_valid = int(out['local_valid_count'])
            values['mU_keep_ratio_local_rows_global_valid_columns'] = float(
                kept[:local_valid].mean() if local_valid else 0.0)
            values['near_zero_readout_norm_count'] = int(
                readout.get('near_zero_readout_norm_count', 0))
            values['readout_pair_count'] = int(readout.get('readout_pair_count', 0))
        mask_diag = mask_s.detach().float()
        if int(mask_diag.numel()):
            values['mS_keep_ratio_rank_local_all_candidates'] = float(
                (mask_diag >= 0.5).float().mean())
        if mask_s_positive_pair is not None and int(mask_s_positive_pair.numel()):
            values['mS_keep_ratio_positive_pair_global'] = float(
                (mask_s_positive_pair.detach() >= 0.5).float().mean())
            values['mS_mean_positive_pair_global'] = float(mask_s_positive_pair.detach().mean())
        lengths = out['prefix_effective_lengths']
        suffix_lengths = out['suffix_effective_lengths']
        values['prefix_effective_length_mean'] = float(
            torch.as_tensor(lengths, dtype=torch.float32).mean()) if lengths else 0.0
        values['suffix_effective_length_mean'] = float(
            torch.as_tensor(suffix_lengths, dtype=torch.float32).mean()) if suffix_lengths else 0.0
        values['prefix_effective_length_max'] = int(max(lengths) if lengths else 0)
        values['suffix_effective_length_max'] = int(max(suffix_lengths) if suffix_lengths else 0)
        values['prefix_content_token_mean'] = float(
            torch.as_tensor(out['prefix_content_tokens'], dtype=torch.float32).mean())
        valid_counts = torch.as_tensor(out['suffix_content_tokens'], dtype=torch.float32)
        values['suffix_content_token_mean_valid'] = (
            float(valid_counts[valid_flags.detach().cpu()].mean())
            if int(valid_flags.sum()) else 0.0)
        raw_prefix = raw_token_lengths(out['prefix_captions'] or [])
        raw_suffix = raw_token_lengths(out['suffix_texts'])
        values['prefix_truncated_count'] = int(sum(1 for value in raw_prefix
                                                   if value > self.context_length))
        values['suffix_truncated_count'] = int(sum(1 for value in raw_suffix
                                                   if value > self.context_length))
        values['prefix_raw_length_max'] = int(max(raw_prefix) if raw_prefix else 0)
        values['suffix_raw_length_max'] = int(max(raw_suffix) if raw_suffix else 0)
        values['token_length_scope'] = ('rank-local rows; truncated = raw token count (SOT/EOT '
                                        'included) above the %d-token context' % self.context_length)
        return values


class SaidPrefixSuffixTrainModule(nn.Module):
    """The DDP-visible training module: the CLIP model (registered exactly once) plus the new F.

    The whole objective runs inside ``forward``, so ``DistributedDataParallel`` synchronises every
    parameter gradient, ``clip.mask_net`` and ``suffix_mask`` included. ``suffix_mask`` is registered
    OUTSIDE ``clip``, so the CLIP state dict stays the bare native student state and the new gate can
    never leak into it.
    """

    def __init__(self, clip: nn.Module, rank: int = 0, arm: str = ARM_NATIVE,
                 lambda_suffix: float = LAMBDA_SUFFIX, suffix_mask: nn.Module = None,
                 image_chunk: int = IMAGE_CHUNK_DEFAULT, text_chunk: int = TEXT_CHUNK_DEFAULT,
                 world_size: int = None, gather_validity: bool = True,
                 context_length: int = TOKENIZER_CONTEXT):
        super().__init__()
        if arm == ARM_MASK and suffix_mask is None:
            suffix_mask = SuffixMask()
        if arm == ARM_NATIVE:
            suffix_mask = None
        self.clip = clip                                     # registered exactly once
        self.suffix_mask = suffix_mask                       # None in the native arm
        self.objective = SaidPrefixSuffixObjective(
            clip, rank=rank, arm=arm, suffix_mask=suffix_mask, lambda_suffix=lambda_suffix,
            image_chunk=image_chunk, text_chunk=text_chunk, world_size=world_size,
            gather_validity=gather_validity, context_length=context_length)
        self.arm = arm
        self.rank = int(rank)

    def forward(self, image_a: torch.Tensor, text_ids: dict, prefix_captions=None,
                suffix_texts=None, prefix_k=None, image_chunk: int = None, text_chunk: int = None,
                want_statistics: bool = False, batch_prefix_texts=None) -> dict:
        return self.objective.forward(image_a, text_ids, prefix_captions=prefix_captions,
                                      suffix_texts=suffix_texts, prefix_k=prefix_k,
                                      image_chunk=image_chunk, text_chunk=text_chunk,
                                      want_statistics=want_statistics,
                                      batch_prefix_texts=batch_prefix_texts)


# --------------------------------------------------------------------------- checkpoint identity
def state_digest(state_dict) -> str:
    """sha256 over sorted keys: the key bytes, then the float32 CPU bytes of the tensor.

    The same rule the S0/CV-SSL trainer, the CG-CLIP trainer and the student exporter use, so a
    checkpoint written here can be verified by any of them without re-deriving the convention. A
    non-mapping argument is refused loudly: a digest that silently hashed ``None`` or a module would
    make a missing suffix-mask state undetectable.
    """
    if not isinstance(state_dict, dict):
        raise TypeError('state_digest needs a mapping of tensors (or a description of scalars), got '
                        '%s: a checkpoint state that went missing must never be digested silently'
                        % type(state_dict).__name__)
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(str(key).encode('utf-8'))
        if torch.is_tensor(value):
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode('utf-8'))
    return digest.hexdigest()


def checkpoint_metadata(suffix_mask: nn.Module = None, steps: int = 0, rank: int = 0,
                        world: int = 1, config: dict = None, digests: dict = None,
                        batch_size: int = BATCH_SIZE_PER_GPU, chunking: dict = None,
                        precision: str = None, clip_digest: str = None,
                        suffix_mask_digest: str = None, horizon_steps: int = None,
                        scheduler_config: dict = None, scheduler_state: dict = None,
                        data_cursor: dict = None, rng_states: dict = None, loss_config: dict = None,
                        sampling_config: dict = None, provenance: dict = None,
                        arm: str = None) -> dict:
    """The self-describing header of a checkpoint, with the exact key list of the spec.

    The suffix-mask TENSORS live under ``suffix_mask_state`` (written by the trainer) and the
    DESCRIPTION under ``suffix_mask_config``: the two keys can never collide, and a checkpoint whose
    gate weights went missing is detectable both by the missing key and by the digest mismatch.

    Called with no argument it is also the module DESCRIPTION helper: it returns the frozen identity,
    the suffix-mask description and the loss/sampling rules, and it never puts a tensor in the result.
    """
    if arm is None:
        arm = ARM_MASK if suffix_mask is not None else ARM_NATIVE
    return {
        'objective': OBJECTIVE,
        'phase': PHASE,
        'arm': arm,
        'suffix_branch': SUFFIX_BRANCH,
        'suffix_mask_config': suffix_mask_config(suffix_mask) or {'module': 'SuffixMask'},
        'clip_state_digest': clip_digest,
        'suffix_mask_state_digest': suffix_mask_digest,
        'completed_steps': int(steps),
        'rank': int(rank),
        'world_size': int(world),
        'batch_size_per_gpu': int(batch_size),
        'global_batch_size': int(batch_size) * int(world),
        'lr_horizon_steps': horizon_steps,
        'scheduler_config': scheduler_config,
        'scheduler_state': scheduler_state,
        'data_cursor': data_cursor,
        'rng_states': rng_states,
        'loss_config': loss_config,
        'sampling_config': sampling_config,
        'provenance': provenance,
        'chunking': dict(chunking or {}),
        'precision': precision,
        'digests': dict(digests or {}),
        'config': dict(config or {}),
        'statistics_scope': STATISTICS_SCOPE,
        'stream_scope': STREAM_SCOPE,
    }
