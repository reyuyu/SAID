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
    g_det   = stop_grad(g_block)                       # [Bi, 512]
    m_det   = stop_grad(mS_block)                      # [Bj, 512]
    rS      = g_det[:, None, :] * m_det[None, :, :]    # [Bi, Bj, 512]
    g_pair  = g_det[:, None, :].expand(-1, Bj, -1)     # [Bi, Bj, 512]
    xU      = concat([g_pair, rS], -1)                 # [Bi, Bj, 1024]  <-- ONLY the concat is 1024
    logits  = F.forward_logits(xU)                     # PRE-sigmoid; forward() = sigmoid of this
    pU      = sigmoid(logits)                          # [Bi, Bj, 512]
    hardU   = (pU >= 0.5);   mU = hardU + (pU - pU.detach())
    u       = Norm(g_block[:, None, :] * mU)           # LIVE g, mU NOT detached
    QU[i,j] = 100 * (u * tR_block[None, :, :]).sum(-1) <- column j carries (P_j, R_j)

``g_block`` and ``tR_block`` are 512-d feature blocks; only ``xU`` is 1024-d. ``F`` is
``Linear(1024, 512)`` -> GELU -> ``Linear(512, 512)``: the first layer is Xavier-uniform with a zero
bias, the last layer starts at weight 0 and bias ``log(8)``, so ``pU = 8/9`` and ``mU`` is exactly 1
everywhere at initialisation. It is constructed inside a fully isolated and restorable RNG context
(Python, NumPy, the CPU generator and EVERY initialised CUDA generator), so building it can never
perturb the CLIP initialisation or the data stream.

Index spaces
------------
Exactly three spaces exist and they are never mixed. The code names them ``LOCAL_FULL``
(this rank's ``B`` rows), ``GLOBAL_FULL`` (the ``W*B`` rows concatenated in rank order) and
``GLOBAL_VALID`` (the ``V`` rows left after the validity filter, ``J``). ``J`` is a
``GLOBAL_FULL`` index set: it may select from gathered ``W*B`` tensors only, never from a
``[B, 512]`` LOCAL tensor and never a second time from an already filtered ``[V, 512]`` tensor.
:func:`valid_pool_mapping` is the pure, CPU-testable map from LOCAL rows to ``GLOBAL_VALID`` labels.

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
``[B, V]`` block of suffix scores, which is then gathered row by row. Each rank therefore computes
its own image rows only, and the gate network is executed exactly once per (image, candidate) pair --
never a second time to obtain the log statistics.
"""
import contextlib
import hashlib
import importlib
import math
import os
import sys

import torch
import torch.distributed as dist
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
           'SuffixMaskGate', 'gate_from_pU', 'straight_through_gate',
           'suffix_mask_config', 'suffix_mask_init_report',
           'normalize_features', 'suffix_masked_representation', 'suffix_gate_input',
           'suffix_pair_gate_input', 'suffix_readout', 'lse_margin', 'block_statistics',
           'statistics_accumulator', 'merge_statistics', 'finalize_statistics',
           'tile_readout_statistics', 'full_grid_statistics', 'valid_pool_mapping',
           'assert_global_full_index_space', 'positive_pair_selection',
           'suffix_readout_scores', 'suffix_readout_statistics', 'suffix_readout_scores_with_gates',
           'suffix_cross_entropy', 'suffix_cross_entropy_sum', 'native_grid_statistics',
           'full_grid_statistics', 'suffix_native_scores', 'suffix_native_scores_local', 'differentiable_gather_rows', 'gather_local_rows',
           'all_gather_flat', 'global_valid_indices', 'global_valid_subset', 'suffix_scaling',
           'differentiable_zero', 'assert_equal_local_batch', 'partition_parameters',
           'build_optimizers', 'SaidPrefixSuffixObjective', 'SaidPrefixSuffixTrainModule',
           'state_digest', 'checkpoint_metadata', 'S0_MODULE', 'compute_smartclip_terms',
           'said_mask_from_hidden', 'said_mask_helper', 's0_effective_lambda_u', 's0_arms',
           'STREAM_SCOPE', 'STATISTICS_SCOPE']


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
        """``pU = sigmoid(F(x))``: the PRE-sigmoid logits (``forward_logits``) put through sigmoid.

        ``forward`` adds nothing to ``forward_logits`` and the readout calls ``forward_logits``
        directly, so the gate the training step uses and the gate an audit reads are the same
        function of the same parameters -- there is no second, differently-computed probability.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.forward_logits(x).sigmoid()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The raw PRE-sigmoid ``F(x)`` logits: ``[..., 1024] -> [..., 512]``.

        The suffix readout takes the gate from here, so a gate audit and the training forward cannot
        disagree about which quantity was thresholded. Explicit fp32 core, never under autocast.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            hidden = F.gelu(self._double_linear(self.layer1, x))
            return self._double_linear(self.layer2, hidden)


def straight_through_gate(p_u: torch.Tensor) -> torch.Tensor:
    """The frozen gate, written out exactly: ``mU = (pU >= 0.5) + (pU - pU.detach())``.

    The hard value is the forward and the sigmoid slope is the backward, both of them ordinary
    autograd of that one expression. There is deliberately NO custom ``torch.autograd.Function``
    here: the expression already carries the straight-through slope, and a hand-written backward
    would only be able to reproduce it or get it wrong. It is also NOT a fix for sigmoid saturation
    -- far from the threshold ``pU - pU.detach()`` has a vanishing sigmoid derivative like any other
    sigmoid term, and nothing in this module claims otherwise or requires a non-zero gate gradient
    for an all-closed sample.
    """
    with torch.autocast(device_type=p_u.device.type, enabled=False):
        probability = p_u.float()
        return (probability >= 0.5).to(probability.dtype) + (probability - probability.detach())


class SuffixMaskGate:
    """The straight-through gate of the suffix mask, kept separate from the network that produces it.

    ``gate_from_pU(pU)`` is the frozen entry point: ``mU = (pU >= 0.5) + (pU - pU.detach())``. There
    is no top-k, no fixed retention count and no soft floor, and no custom backward of any kind.
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
            probability = pU.float()
            return (probability >= 0.5).to(probability.dtype) + (probability - probability.detach())

    @staticmethod
    def gate_from_logits(logits: torch.Tensor) -> dict:
        """``{pU, hardU, mU}`` from raw logits, so an audit can see the probability the gate uses."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability = torch.sigmoid(logits.float())
            hard = (probability >= 0.5).to(torch.float32)
            return {'pU': probability, 'hardU': hard,
                    'mU': (probability >= 0.5).to(probability.dtype)
                          + (probability - probability.detach())}


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
        'input': 'stop_grad(concat([g, rS])): one 1024-d vector per (image, candidate) pair, built '
                 'from the two 512-d blocks g and rS (only the concatenation is 1024-d)',
        'g_source': 'Norm(encode_image(I)): the live student output, 512-d, not a frozen reference',
        'mS_source': 'hard straight-through S0 mask_net on the prefix C_S hidden, 512-d, detached',
        'rS_rule': 'g.detach() * mS.detach(), one 512-d block per pair: NO re-normalisation, never '
                   'Norm(g * mS)',
        'readout': 'Norm(g * mU) with the 512-d FULL live g, never g * (1 - mS)',
        'output': 'pU = sigmoid(F.forward_logits(xU)); mU = (pU >= 0.5) + (pU - pU.detach()), '
                  'both 512-d per pair',
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
    """``xU = stop_grad(concat([g, rS]))`` along the last dimension: ``[1024]`` per pair.

    ``g_features`` is a 512-d block and ``r_s`` a 512-d block (the concatenation is the only 1024-d
    tensor in this module).
    """
    return torch.cat([g_features.detach().float(), r_s.detach().float()], dim=-1)


def suffix_pair_gate_input(g_block: torch.Tensor, mS_block: torch.Tensor) -> torch.Tensor:
    """The image x candidate-prefix pair tensor ``xU`` of the mask arm: ``[Bi, Bj, 1024]``.

    The two inputs are used through their OWN axes and are never element-wise multiplied with each
    other: ``g_block * mS_block`` is only correct-looking when ``Bi == Bj`` and otherwise broadcasts
    ``g_j * mS_j`` into the wrong image rows, so it is not written anywhere in this module.

        g_det = g_block.detach()[:, None, :]        # [Bi, 1, 512]
        m_det = mS_block.detach()[None, :, :]       # [1, Bj, 512]
        rS    = g_det * m_det                       # [Bi, Bj, 512]
        xU    = cat([g_det.expand(-1, Bj, -1), rS], dim=-1)     # [Bi, Bj, 1024]

    ``g_det.expand`` is a view, so no ``[Bi, Bj, 512]`` buffer of the plain image block is
    materialised and no chunk size is allowed to hide a broadcast of the wrong shape.
    """
    if g_block.dim() != 2 or mS_block.dim() != 2:
        raise ValueError('the pair readout takes a 2-D image block [Bi, 512] and a 2-D candidate '
                         'prefix-mask block [Bj, 512]; got %s and %s'
                         % (tuple(g_block.shape), tuple(mS_block.shape)))
    g_det = g_block.detach().float()[:, None, :]                # [Bi, 1, 512]
    m_det = mS_block.detach().float()[None, :, :]               # [1, Bj, 512]
    r_s = g_det * m_det                                         # [Bi, Bj, 512]
    return suffix_gate_input(g_det.expand(-1, int(m_det.shape[1]), -1), r_s)


def suffix_readout(u_gated: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """``u = Norm(g_live * mU)``: the FULL live ``g``, never ``g * (1 - mS)``.

    Exactly :func:`normalize_features` -- the STANDARD ``F.normalize`` autograd, no custom backward,
    no detached denominator and no straight-through on this path. An all-closed sample therefore
    reads out as an exactly zero ``u`` (``F.normalize`` of a zero input), and that is the intended
    value, not a case to be patched.
    """
    return normalize_features(u_gated, eps=eps)


# --------------------------------------------------------------------------- chunked scoring
def _chunk_ranges(total: int, chunk: int):
    if total <= 0:
        return []
    size = max(int(chunk), 1)
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def lse_margin(scores: torch.Tensor, positive_index: torch.Tensor) -> torch.Tensor:
    """``positive - logsumexp(negatives among the columns given)``: the only LSE margin here.

    The caller must pass candidate columns that are all valid (the suffix pool is the valid subset
    ``J``), so the sum really is over valid negatives. ``logsumexp(all) - positive`` is the cross
    entropy and is never reported under this name.
    """
    rows = scores.float()
    index = positive_index.reshape(-1, 1)
    positive = rows.gather(1, index).squeeze(1)
    masked = rows.clone().scatter(1, index, float('-inf'))
    return positive - torch.logsumexp(masked, dim=1)


def block_statistics(rows: torch.Tensor, targets: torch.Tensor) -> dict:
    """Positive / strongest-negative / max-margin / LSE-margin for one block of score rows.

    Every column of ``rows`` must be a valid candidate (the valid subset ``J`` is the only pool this
    module scores), so ``masked`` holds valid negatives only.
    """
    rows = rows.float()
    index = targets.reshape(-1, 1)
    positive = rows.gather(1, index).squeeze(1)
    masked = rows.clone().scatter(1, index, float('-inf'))
    strongest = masked.max(dim=1).values
    return {'positive': positive, 'strongest_negative': strongest,
            'max_margin': positive - strongest,
            'lse_margin': positive - torch.logsumexp(masked, dim=1)}


# --------------------------------------------------------------------------- tile statistics
_SUM_KEYS = ('positive_pairs', 'positive_keep_sum', 'positive_probability_sum', 'positive_norm_sum',
             'positive_norm_ratio_sum', 'positive_near_zero', 'all_pairs', 'all_keep_sum',
             'all_probability_sum', 'all_norm_sum', 'all_norm_ratio_sum', 'all_near_zero',
             'anchor_count', 'candidate_count')


def statistics_accumulator() -> dict:
    """A FRESH accumulator of per-tile gate/readout statistics.

    The heavy log must not re-run F over the whole ``V x V`` grid to obtain numbers the scoring tile
    already had: every tile feeds this accumulator while it is scored, and :func:`finalize_statistics`
    turns the sums into counts plus weighted means. Because the sums and their counts are carried
    together, tiles of different sizes are weighted by their own element counts and unweighted means
    of differently sized tiles are never averaged.

    ``positive_pairs`` counts the (image, suffix) pairs that are genuine positives -- an image and
    the suffix of its OWN row -- and is kept strictly separate from ``all_pairs``. ``margin_*`` fields
    stay absent until :func:`full_grid_statistics` has really seen the score grid.
    """
    positive = dict((key, 0.0) for key in _SUM_KEYS)
    everything = dict((key, 0.0) for key in _SUM_KEYS)
    return {'positive_pair': positive, 'all_pair': everything, 'margin_positive': None,
            'margin_strongest_negative': None, 'margin_max_margin': None,
            'margin_lse_margin': None, 'margin_anchor_count': 0, 'margin_candidate_count': 0,
            'extreme_units': 0, 'extreme_probability_min': None, 'extreme_probability_max': None,
            'scope': {'positive_pair': 'image i with the suffix of its own row (P_i, R_i)',
                      'all_pair': 'every (image i, valid suffix j) pair scored in this pass',
                      'margin': 'not measured in this pass'}}


def statistics_log_fields(statistics: dict) -> dict:
    """The flat, ALWAYS-NUMERIC fields of a statistics dict, for a JSON log record.

    The nested statistics dict uses ``None`` for "not measured"; a JSON log line cannot, because a
    reader must never see "not measured" dressed up as ``0 % kept``. This helper therefore names the
    case explicitly instead of hiding it: ``gate_statistics_measured`` says whether any gate element
    was scored at all, and the numeric fields are exactly the measured numbers (or a clearly labelled
    default with ``measured = false``).
    """
    sides = statistics.get('sides') or {}
    all_pair = sides.get('all_pair') or {}
    positive = sides.get('positive_pair') or {}
    pairs = int(all_pair.get('pair_count') or 0)
    measured = pairs > 0
    return {
        'gate_statistics_measured': bool(measured),
        'gate_statistics_note': ('measured = the pass scored at least one (image, valid suffix) pair; '
                                 'when false every gate/readout field below is a default and NOT a '
                                 'measured zero'),
        'readout_pair_count': pairs,
        'positive_pair_count': int(positive.get('pair_count') or 0),
        'mU_keep_ratio_all_valid_pairs': all_pair.get('keep_ratio'),
        'mU_probability_mean': all_pair.get('probability_mean'),
        'mU_probability_min': all_pair.get('probability_min'),
        'mU_probability_max': all_pair.get('probability_max'),
        'gated_readout_norm_mean': all_pair.get('readout_norm_mean'),
        'gated_readout_norm_ratio_mean': all_pair.get('readout_norm_ratio_mean'),
        'near_zero_readout_norm_count': int(all_pair.get('near_zero_readout_norm_count') or 0),
        'mU_keep_ratio_positive_pair': positive.get('keep_ratio'),
        'mU_probability_mean_positive_pair': positive.get('probability_mean'),
        'gated_readout_norm_ratio_mean_positive_pair': positive.get('readout_norm_ratio_mean'),
        'gate_statistics_scope': statistics.get('gates_scope') or statistics.get('native_grid_scope'),
    }


def _accumulate(acc: dict, key: str, value) -> None:
    if value is None:
        return
    if torch.is_tensor(value):
        acc[key] = acc[key] + value.detach().float()
    else:
        acc[key] = acc[key] + float(value)


def _merge_into(target: dict, item: dict) -> dict:
    """Add one accumulator or finalized statistics dict into ``target`` (in place, O(1) fields)."""
    for side in ('positive_pair', 'all_pair'):
        for key in _SUM_KEYS:
            _accumulate(target[side], key, item[side][key])
    for key in ('margin_anchor_count', 'margin_candidate_count'):
        _accumulate(target, key, item.get(key, 0) or 0)
    for key in ('margin_positive', 'margin_strongest_negative', 'margin_max_margin',
                'margin_lse_margin', 'margin_positive_win_fraction', 'margin_top1'):
        value = item.get(key)
        if value is None:
            continue
        weight = int(item.get('margin_anchor_count') or 0)
        total, count = target[key] if target[key] is not None else (0.0, 0)
        target[key] = (total + float(value) * weight, count + weight)
    scope = dict(item.get('scope') or {})
    if scope.get('margin', 'not measured in this pass') != 'not measured in this pass':
        target['scope']['margin'] = scope['margin']
    return target


def merge_statistics(statistics_list) -> dict:
    """Merge several accumulators (or finalized dicts) by summing sides and counts.

    Both input forms are accepted, so a caller may merge tile accumulators together, or merge a
    finished statistics dict of one pass into another, without re-deriving which form it holds. The
    merge is a pure sum of counts and sums, which is exactly what makes differently sized tiles carry
    their own weight; a field that was never measured stays absent instead of becoming a zero.
    """
    merged = statistics_accumulator()
    for item in statistics_list:
        if item is not None:
            _merge_into(merged, item)
    return merged


def finalize_statistics(accumulator: dict) -> dict:
    """Add the count-weighted means to an accumulator (idempotent: a final dict is returned as is).

    Every mean is a SUM divided by the count it was summed over, so a tile of 4 pairs and a tile of
    128 pairs contribute in proportion to their pair counts and no unweighted mean of differently
    sized tiles is ever averaged. ``pair_count == 0`` yields ``None`` -- "not measured" is never
    reported as ``0`` (an all-closed gate at ``p=[0.9, 0.1, 0.1]`` has keep ratio ``1/3``, and a pass
    with no positive pair at all has NO keep ratio, not a zero one).
    """
    if 'sides' in accumulator:
        return accumulator
    sides = {}
    for side, pair_key in (('positive_pair', 'positive_pairs'), ('all_pair', 'all_pairs')):
        raw = accumulator[side]
        pairs = int(raw[pair_key])
        prefix = 'positive' if side == 'positive_pair' else 'all'
        sides[side] = {
            'pair_count': pairs,
            'probability_mean': (float(raw[prefix + '_probability_sum']) / pairs) if pairs else None,
            'keep_ratio': (float(raw[prefix + '_keep_sum']) / pairs) if pairs else None,
            'readout_norm_mean': (float(raw[prefix + '_norm_sum']) / pairs) if pairs else None,
            'readout_norm_ratio_mean': (float(raw[prefix + '_norm_ratio_sum']) / pairs) if pairs
            else None,
            'near_zero_readout_norm_count': int(raw[prefix + '_near_zero']),
        }
    accumulator['sides'] = sides
    accumulator['anchor_count'] = int(accumulator['all_pair']['anchor_count'])
    accumulator['candidate_count'] = int(accumulator['all_pair']['candidate_count'])
    margin_anchor_count = int(accumulator.get('margin_anchor_count') or 0)
    for key, field in (('margin_positive', 'positive_mean'),
                       ('margin_strongest_negative', 'strongest_negative_mean'),
                       ('margin_max_margin', 'max_margin_mean'),
                       ('margin_lse_margin', 'lse_margin_mean'),
                       ('margin_positive_win_fraction', 'positive_win_fraction'),
                       ('margin_top1', 'top1')):
        value = accumulator.get(key)
        if value is None:
            accumulator[field] = None
        elif isinstance(value, tuple):
            total, count = value
            accumulator[field] = (total / count) if count else None
        else:
            # a margin that was measured on the complete grid: keep the plain mean it already is
            accumulator[field] = float(value) if margin_anchor_count else None
    accumulator['definitions'] = {
        'keep_ratio': '(pU >= 0.5) fraction of the GATE ELEMENTS, not of a mean over the 512 '
                      'dimensions: p=[0.9,0.1,0.1] has keep ratio 1/3, never 0',
        'probability_mean': 'pU summed over every gate element and divided by the element count '
                            '(never a mean over the 512 dimensions first)',
        'readout_norm_ratio': '||g_i * mU_ij|| / max(||g_i||, eps) on the 512-d pair vectors: never '
                              'divided by sqrt(512) and never collapsed from [Bi, Bj] to [Bi]',
        'lse_margin': 'positive - logsumexp(valid negatives only); logsumexp(all) - positive is CE '
                      'and is never reported under this name',
        'weighting': 'every mean is sum / pair count, so tiles of different sizes are weighted by '
                     'their own element counts',
        'positive_pair': 'image i with the suffix of its own row (P_i, R_i), separate from all_pair',
    }
    return accumulator


def positive_pair_selection(image_anchors, text_anchors):
    """The (row, column) positions of the genuine positive pairs of a score block.

    ``image_anchors[r]`` is the GLOBAL_FULL row of image block row ``r`` and ``text_anchors[c]`` the
    GLOBAL_FULL row of candidate column ``c``: the pair is positive when the two global rows are the
    SAME sample. Indexing by position instead would silently mislabel every pass whose anchors are
    not a contiguous ``rank*B + arange(n_r)`` block, which is exactly the ``n_r = 0`` / unequal
    ``n_r`` case this round has to get right.
    """
    rows = torch.as_tensor(image_anchors, dtype=torch.long).reshape(-1)
    columns = torch.as_tensor(text_anchors, dtype=torch.long).reshape(-1)
    if not int(rows.numel()) or not int(columns.numel()):
        return [], []
    matches = (rows.reshape(-1, 1) == columns.reshape(1, -1)).nonzero(as_tuple=False)
    return (matches[:, 0].tolist(), matches[:, 1].tolist())


def tile_readout_statistics(probability, readout, g_block, mS_block, image_anchors, text_anchors,
                            eps: float = NORM_EPS) -> dict:
    """The statistics of ONE scored tile, taken from the quantities the tile already computed.

    ``probability`` is the full ``[Bi, Bj, 512]`` gate probability grid of the tile (NOT reduced over
    the 512 dimensions), ``readout`` the ``[Bi, Bj, 512]`` ``g_i * mU_ij`` product and ``g_block`` the
    ``[Bi, 512]`` image block the tile was built from. Nothing is re-run here.
    """
    values = {'positive_pair': dict((key, 0.0) for key in _SUM_KEYS),
              'all_pair': dict((key, 0.0) for key in _SUM_KEYS),
              'margin_positive': None, 'margin_strongest_negative': None, 'margin_max_margin': None,
              'margin_lse_margin': None, 'margin_anchor_count': 0, 'margin_candidate_count': 0,
              'scope': {'positive_pair': 'image i with the suffix of its own row (P_i, R_i)',
                        'all_pair': 'every (image i, valid suffix j) pair of this pass',
                        'margin': 'not measured in this pass'}}
    with torch.no_grad():
        probability = probability.detach().float()
        hard = (probability >= 0.5).float()
        kept = hard.sum(dim=-1)
        summed = probability.sum(dim=-1)
        g_norm = g_block.detach().float().norm(dim=-1)                     # [Bi]
        readout_norm = readout.detach().float().norm(dim=-1)               # [Bi, Bj]
        ratio = readout_norm / g_norm.reshape(-1, 1).clamp_min(float(eps))
        near_zero = (readout_norm <= 1e-8)
        pairs = int(probability.shape[0]) * int(probability.shape[1])
        all_pair = values['all_pair']
        all_pair['all_pairs'] = float(pairs)
        all_pair['all_keep_sum'] = float(kept.sum())
        all_pair['all_probability_sum'] = float(summed.sum())
        all_pair['all_norm_sum'] = float(readout_norm.sum())
        all_pair['all_norm_ratio_sum'] = float(ratio.sum())
        all_pair['all_near_zero'] = float(near_zero.sum())
        all_pair['anchor_count'] = float(probability.shape[0])
        all_pair['candidate_count'] = float(probability.shape[1])
        rows, columns = positive_pair_selection(image_anchors, text_anchors)
        if rows:
            row_index = torch.as_tensor(rows, dtype=torch.long, device=probability.device)
            column_index = torch.as_tensor(columns, dtype=torch.long, device=probability.device)
            picked_keep = kept[row_index, column_index]
            picked_sum = summed[row_index, column_index]
            picked_norm = readout_norm[row_index, column_index]
            picked_ratio = ratio[row_index, column_index]
            positive = values['positive_pair']
            positive['positive_pairs'] = float(len(rows))
            positive['positive_keep_sum'] = float(picked_keep.sum())
            positive['positive_probability_sum'] = float(picked_sum.sum())
            positive['positive_norm_sum'] = float(picked_norm.sum())
            positive['positive_norm_ratio_sum'] = float(picked_ratio.sum())
            positive['positive_near_zero'] = float(near_zero[row_index, column_index].sum())
    return values


def full_grid_statistics(scores, targets, image_chunk: int = IMAGE_CHUNK_DEFAULT) -> dict:
    """Positive / negative margin statistics of a COMPLETE score grid, tile by tile.

    Only call this with the real grid (every column a valid candidate and every row an anchor a
    target exists for); with a partial grid the fields stay ``None`` instead of reporting a margin
    that was never measured. Both directions are reduced to per-anchor vectors here, so no
    ``[V, V]`` intermediate beyond the grid itself is built.
    """
    grid = scores.detach().float()
    anchors = int(grid.shape[0])
    anchors_with_target = min(anchors, int(torch.as_tensor(targets).reshape(-1).numel()))
    values = {'positive': [], 'strongest_negative': [], 'max_margin': [], 'lse_margin': []}
    for start, stop in _chunk_ranges(anchors_with_target, image_chunk):
        stats = block_statistics(grid[start:stop],
                                 torch.as_tensor(targets).reshape(-1)[start:stop])
        for key in values:
            values[key].append(stats[key])
    if not values['positive']:
        return {'margin_positive': None, 'margin_strongest_negative': None,
                'margin_max_margin': None, 'margin_lse_margin': None,
                'margin_positive_win_fraction': None, 'margin_top1': None,
                'margin_anchor_count': 0, 'margin_candidate_count': int(grid.shape[1]),
                'scope': {'margin': 'no anchor with a valid target in this pass'}}
    stacked = dict((key, torch.cat(parts)) for key, parts in values.items())
    return {
        'margin_positive': float(stacked['positive'].mean()),
        'margin_strongest_negative': float(stacked['strongest_negative'].mean()),
        'margin_max_margin': float(stacked['max_margin'].mean()),
        'margin_lse_margin': float(stacked['lse_margin'].mean()),
        'margin_lse_margin_min': float(stacked['lse_margin'].min()),
        'margin_positive_win_fraction': float((stacked['max_margin'] > 0).float().mean()),
        'margin_top1': float((grid.argmax(dim=1) == torch.as_tensor(targets).reshape(-1)
                              ).float()[:anchors_with_target].mean()),
        'margin_anchor_count': int(stacked['positive'].numel()),
        'margin_candidate_count': int(grid.shape[1]),
        'scope': {'margin': 'complete score grid: positive - logsumexp(valid negatives only), '
                            'candidates = the valid suffix pool J (V columns)'},
    }


# --------------------------------------------------------------------------- index spaces
def assert_global_full_index_space(t_r_global_full, mS_global_full, J, world_size: int,
                                   local_size: int) -> None:
    """Hard guard BEFORE any GPU gather: every tensor an index set is applied to is ``GLOBAL_FULL``.

    The first smoke test died with ``indexSelectLargeIndex srcIndex < srcSelectDimSize`` because a
    GLOBAL_FULL index set was applied to tensors of different spaces. The message names the space
    that disagreed instead of leaving a CUDA assertion to be decoded later.
    """
    expected = int(world_size) * int(local_size)
    rows = int(t_r_global_full.shape[0])
    mask_rows = int(mS_global_full.shape[0])
    if rows != expected or mask_rows != expected:
        raise RuntimeError('index-space mismatch before the GPU gather: J lives in GLOBAL_FULL '
                           '[0, %d) but tR_global_full has %d rows (%s) and mS_global_full has %d '
                           'rows (%s); expected W*B = %d rows in BOTH. A LOCAL [B, 512] tensor '
                           'must be gathered before J is applied, and an already filtered '
                           '[V, 512] tensor must not be selected a second time.'
                           % (expected, rows, 'LOCAL_FULL' if rows == int(local_size) else 'unknown',
                              mask_rows,
                              'LOCAL_FULL' if mask_rows == int(local_size) else 'unknown', expected))
    index = torch.as_tensor(J).reshape(-1)
    if int(index.numel()):
        if int(index.min()) < 0 or int(index.max()) >= expected:
            raise RuntimeError('index-space mismatch: J holds GLOBAL_FULL indices and must lie in '
                               '[0, %d) (= W*B), got min %d max %d. Never repair this with J %% B '
                               'or J // B: the fix is to gather the tensor into GLOBAL_FULL first.'
                               % (expected, int(index.min()), int(index.max())))


def valid_pool_mapping(valid_local, J, world_size: int, local_size: int, device=None) -> dict:
    """The explicit map from LOCAL rows to ``GLOBAL_VALID`` (compressed-pool) labels.

        global_to_valid = full([W*B], -1, long); global_to_valid[J] = arange(V)
        local_rows      = nonzero(valid_local)          # local rows of THIS rank that are valid
        anchor_global   = rank*B + local_rows           # their GLOBAL_FULL rows
        anchor_valid    = global_to_valid[anchor_global]  # their GLOBAL_VALID labels

    ``rank*B + arange(n_r)`` is NEVER used: it assumes the valid local rows are the first ``n_r``
    rows and is not the label of a compressed pool. Worked counterexample (binding, pure CPU):
    ``W=2, B=4, J=[1,3,4,7]`` gives rank 0 ``local_rows=[1,3]`` -> ``anchor_valid=[0,1]`` and rank 1
    ``local_rows=[0,3]`` -> ``anchor_valid=[2,3]`` (``rank*B + arange(2)`` would have said ``[0,1]``
    for rank 1, which is wrong).
    """
    world_size = int(world_size)
    local_size = int(local_size)
    flag = torch.as_tensor(valid_local).detach().reshape(-1).to(torch.bool)
    if int(flag.numel()) != local_size:
        raise ValueError('the local validity has %d entries but the local batch has %d rows'
                         % (int(flag.numel()), local_size))
    index = torch.as_tensor(J).detach().reshape(-1).to(torch.long)
    total = world_size * local_size
    if int(index.numel()):
        if int(index.min()) < 0 or int(index.max()) >= total:
            raise RuntimeError('J must hold GLOBAL_FULL indices in [0, %d), got min %d max %d'
                               % (total, int(index.min()), int(index.max())))
    anchor_device = device if device is not None else flag.device
    global_to_valid = torch.full((total,), -1, dtype=torch.long, device=anchor_device)
    if int(index.numel()):
        global_to_valid[index.to(anchor_device)] = torch.arange(
            int(index.numel()), dtype=torch.long, device=anchor_device)
    local_rows = torch.nonzero(flag.to(anchor_device), as_tuple=False).reshape(-1).to(torch.long)
    position = int(dist.get_rank()) if dist.is_initialized() else None
    if position is None:
        position = 0
    anchor_global = (int(position) * local_size) + local_rows
    anchor_valid = global_to_valid.index_select(0, anchor_global)
    if int(anchor_valid.numel()) and int(anchor_valid.min()) < 0:
        bad = global_to_valid.index_select(0, anchor_global).lt(0).nonzero(as_tuple=False)
        raise RuntimeError('%d valid local row(s) map to no GLOBAL_VALID label: J and the local '
                           'validity disagree about which rows are valid (rank %d rows %r). J is a '
                           'GLOBAL_FULL index set, so rank r owns rows [r*B, (r+1)*B).'
                           % (int(bad.numel()), int(position),
                              [int(value) for value in local_rows[:8]]))
    return {'global_to_valid': global_to_valid, 'local_rows': local_rows,
            'anchor_global': anchor_global, 'anchor_valid': anchor_valid,
            'rank': int(position), 'V': int(index.numel()),
            'space_note': 'local_rows/anchor_global are LOCAL_FULL/GLOBAL_FULL; anchor_valid is '
                          'GLOBAL_VALID (the compressed pool J)'}


# --------------------------------------------------------------------------- differentiable gather
class _DifferentiableAllGather(torch.autograd.Function):
    """``all_gather`` in rank order with a backward that gives every rank its own slice of ``grad``.

    The suffix objective needs the gathered rows to be differentiable: the I2T term of one rank is
    scored against the candidate texts of every rank. The gradient path this introduces is
    mathematically equal to the standard DDP path (for the T2I term the per-rank CE sum is
    ``sum over this rank's own columns``, which is exact, and DDP's averaging divides by ``W``, which
    the ``W / V`` loss scale already accounts for), so no extra collective is inserted here: the
    module-wide DDP averaging remains the only reduction of record.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int, group):
        if not dist.is_initialized() or dist.get_world_size(group) == 1:
            return (tensor,)
        flat = [torch.zeros_like(tensor) for _ in range(int(world_size))]
        dist.all_gather(flat, tensor.contiguous(), group=group)
        return tuple(flat)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not dist.is_initialized() or len(grad_outputs) == 1:
            return grad_outputs[0], None, None
        rank = dist.get_rank(ctx.group) if getattr(ctx, 'group', None) is not None \
            else dist.get_rank()
        rank = min(max(int(rank), 0), len(grad_outputs) - 1)
        return grad_outputs[rank], None, None


def differentiable_gather_rows(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather the rows of every rank into ``GLOBAL_FULL`` order, keeping the autograd path.

    Identity when the process group is absent or has a single member, so a single-process run and the
    distributed run share this one code path.
    """
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    return torch.cat(list(_DifferentiableAllGather.apply(tensor, dist.get_world_size(group), group)),
                     dim=0)


def _mask_arm_scores(g_block: torch.Tensor, mS_block: torch.Tensor, tR_block: torch.Tensor,
                     suffix_mask: nn.Module, image_chunk: int, text_chunk: int, eps: float,
                     want_statistics: bool, image_anchors=None, text_anchors=None,
                     return_gates: bool = False) -> dict:
    """The mask arm tile loop: ``[Bi, V]`` of conditional scores and the statistics of the same pass.

    Every tile is scored exactly once and the statistics are taken from that one execution, so the
    heavy log never re-runs F over the whole ``V x V`` grid.
    """
    images = int(g_block.shape[0])
    texts = int(tR_block.shape[0])
    device = g_block.device
    scores = torch.empty((images, texts), dtype=torch.float32, device=device)
    accumulator = statistics_accumulator()
    gates = (torch.empty((images, texts, int(mS_block.shape[-1])), dtype=torch.float32,
                         device=device) if return_gates else None)
    text_step = max(int(text_chunk), 1)
    image_rows = None if image_anchors is None else torch.as_tensor(image_anchors).reshape(-1)
    text_rows = None if text_anchors is None else torch.as_tensor(text_anchors).reshape(-1)
    for i0, i1 in _chunk_ranges(images, image_chunk):
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            x_u = suffix_pair_gate_input(g_block[i0:i1], mS_block[j0:j1])   # [Bi, Bj, 1024]
            logits = suffix_mask.forward_logits(x_u)                        # [Bi, Bj, 512]
            gate = SuffixMaskGate.gate_from_logits(logits)
            mask_u = gate['mU']                                             # [Bi, Bj, 512]
            gated = g_block[i0:i1][:, None, :] * mask_u                     # LIVE g, [Bi, Bj, 512]
            unit = suffix_readout(gated, eps=eps)                           # fp32 F.normalize
            scores[i0:i1, j0:j1] = (SMARTCLIP_FIXED_SCALE
                                    * (unit * tR_block[j0:j1][None, :, :]).sum(-1)).float()
            if want_statistics:
                tile_anchors = None if image_rows is None else image_rows[i0:i1]
                tile_columns = None if text_rows is None else text_rows[j0:j1]
                _merge_into(accumulator, tile_readout_statistics(gate['pU'], gated, g_block[i0:i1],
                                                                 mS_block[j0:j1], tile_anchors,
                                                                 tile_columns, eps=eps))
                if return_gates:
                    gates[i0:i1, j0:j1] = gate['pU'].detach().float()
    out = {'scores': scores, 'statistics': None}
    if want_statistics:
        accumulator['all_pair']['anchor_count'] = float(images)
        accumulator['all_pair']['candidate_count'] = float(texts)
        statistics = finalize_statistics(accumulator)
        statistics['gates_scope'] = ('tile-local: the rows are the local image block and the columns '
                                     'the valid suffix pool of this rank, so no full global gate '
                                     'tensor is materialised or claimed')
        statistics['full_global_gate_grid'] = False
        if return_gates:
            statistics['pU'] = gates
        out['statistics'] = statistics
    return out


def suffix_readout_scores(g_block: torch.Tensor, mS_block: torch.Tensor, tR_block: torch.Tensor,
                          suffix_mask: nn.Module, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                          text_chunk: int = TEXT_CHUNK_DEFAULT, eps: float = NORM_EPS,
                          want_statistics: bool = False) -> dict:
    """``QU[i, j] = 100 * dot(u_i,j, tR_j)`` for local images x given suffixes, tile by tile.

    FROZEN PRODUCTION INTERFACE. ONE return type, always a dict::

        {'scores': [Bi, Bj] fp32, 'statistics': dict or None}

    There is no tuple form, no "guess which argument is which" and no automatic swap of ``tR_block``
    and ``suffix_mask``: the order is always ``(g_block, mS_block, tR_block, suffix_mask)`` and a
    missing gate network is an ERROR, because silently scoring without the gate would turn the
    conditional readout into the native one.

        g_block  : [Bi, 512] the LIVE final visual features (the gradient must reach the trunk)
        mS_block : [Bj, 512] CANDIDATE prefix masks, TWO-DIMENSIONAL ONLY (a 3-D input is refused)
        tR_block : [Bj, 512] LIVE suffix text features, one row per candidate column
        eps      : the ``F.normalize`` epsilon, ``1e-6``

    Column ``j`` is the pair ``(P_j, R_j)``: image ``i`` is scored against ``R_j`` with the mask
    generated from ``P_j``, for positives and negatives alike. The per-pair expression, tile by tile,
    is exactly

        rS    = g_i.detach() * mS_j.detach()                    # [Bi, Bj, 512]
        xU    = concat([g_i.detach(), rS])                      # [Bi, Bj, 1024]
        pU    = sigmoid(F.forward_logits(xU))                   # [Bi, Bj, 512]
        mU    = (pU >= 0.5) + (pU - pU.detach())
        QU_ij = 100 * sum(Norm(g_i_live * mU_ij) * tR_j)

    Gradient contract: ``g_block`` is LIVE in the final ``Norm(g * mU)``; ``tR_block`` is LIVE;
    ``mS_block`` is detached on this path (the S0 mask gets no direct suffix gradient); the ``g`` and
    ``rS`` inside ``xU`` are detached. ``statistics`` is ``None`` unless ``want_statistics`` is set,
    and then it holds the counts and count-weighted means measured BY THIS PASS -- never a second
    execution of F over the global grid.
    """
    if not isinstance(suffix_mask, nn.Module):
        raise TypeError('suffix_readout_scores takes the gate network as its fourth argument '
                        '(g_block, mS_block, tR_block, suffix_mask); got %s, which would score the '
                        'suffixes with NO gate and silently return the native readout'
                        % type(suffix_mask).__name__)
    if not callable(getattr(suffix_mask, 'forward_logits', None)):
        raise TypeError('the gate network of suffix_readout_scores must expose forward_logits(x) '
                        'returning the PRE-sigmoid logits, so the training gate and any audit read '
                        'the same quantity; got %s' % type(suffix_mask).__name__)
    if g_block.dim() != 2 or mS_block.dim() != 2 or tR_block.dim() != 2:
        raise ValueError('suffix_readout_scores takes THREE 2-D blocks -- g_block [Bi, 512], '
                         'mS_block [Bj, 512] and tR_block [Bj, 512] -- and no 3-D input: got %s, '
                         '%s, %s' % (tuple(g_block.shape), tuple(mS_block.shape),
                                     tuple(tR_block.shape)))
    if int(mS_block.shape[0]) != int(tR_block.shape[0]):
        raise ValueError('the candidate prefix masks and the candidate suffix features must have the '
                         'same number of columns (one (P_j, R_j) pair per column), got %d and %d'
                         % (int(mS_block.shape[0]), int(tR_block.shape[0])))
    return _mask_arm_scores(g_block, mS_block, tR_block, suffix_mask, image_chunk, text_chunk, eps,
                            bool(want_statistics))


def suffix_readout_statistics(g_block: torch.Tensor, mS_block: torch.Tensor, tR_block: torch.Tensor,
                             suffix_mask: nn.Module, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                             text_chunk: int = TEXT_CHUNK_DEFAULT,
                             eps: float = NORM_EPS) -> dict:
    """The heavy-log statistics of one tile pass, REUSING the pass's own in-tile measurements.

    A thin wrapper over :func:`suffix_readout_scores` with ``want_statistics=True``: the gate is
    executed exactly once per (image, candidate) pair, and the returned dict carries the same counts
    and count-weighted means the scoring tile produced. Nothing is re-run over a global grid and no
    "last tile's mU" is ever returned as if it were every gate.
    """
    result = suffix_readout_scores(g_block, mS_block, tR_block, suffix_mask,
                                   image_chunk=image_chunk, text_chunk=text_chunk, eps=eps,
                                   want_statistics=True)
    return result['statistics']


def suffix_readout_scores_with_gates(g_block: torch.Tensor, mS_block: torch.Tensor,
                                     tR_block: torch.Tensor, suffix_mask: nn.Module,
                                     image_chunk: int = IMAGE_CHUNK_DEFAULT,
                                     text_chunk: int = TEXT_CHUNK_DEFAULT, eps: float = NORM_EPS):
    """The SAME production readout, additionally returning the tile's own gate probabilities.

    This is the explicit small-test / debug entry point: the full ``[Bi, Bj, 512]`` gate tensor is
    materialised only when a caller asks for it BY NAME, because a tensor that does not contribute to
    the loss corrupts DDP's unused-parameter bookkeeping and because "the last tile's mU" must never
    be handed back as if it were every gate. The production entry point returns no gate tensor at all.
    """
    result = _mask_arm_scores(g_block, mS_block, tR_block, suffix_mask, image_chunk, text_chunk, eps,
                              True, return_gates=True)
    return {'scores': result['scores'], 'pU': result['statistics']['pU'],
            'statistics': result['statistics'],
            'scope': 'small-test / debug only: one gate probability per (image, candidate, unit) '
                     'element of THIS pass, never a global claim'}


def suffix_cross_entropy_sum(block_scores: torch.Tensor, targets) -> torch.Tensor:
    """``sum``-reduced CE of ONE score block, over the anchors the targets really label.

    ``block_scores`` is ``[anchors, V]`` with one row per anchor of the block and ``targets`` the
    anchor's own column inside ``V``: the reduction is ``reduction='sum'`` over exactly those rows,
    never over a ``V``-row block carrying fewer targets. ``torch.nn.functional.cross_entropy`` with
    ``reduction='sum'`` is the CE core and is computed in explicit fp32.
    """
    rows = int(block_scores.shape[0])
    columns = int(block_scores.shape[1])
    target = torch.as_tensor(targets, device=block_scores.device).reshape(-1).to(torch.long)
    if int(target.numel()) != rows:
        raise RuntimeError('the CE block has %d anchor rows but %d targets; a block of V rows must '
                           'never be passed with only n_r targets -- slice the block to the anchors '
                           'first' % (rows, int(target.numel())))
    if not rows:
        return block_scores.float().sum() * 0.0
    if int(target.max()) >= columns or int(target.min()) < 0:
        raise RuntimeError('a CE target %d is outside the %d candidate columns of the block: the '
                           'targets must be GLOBAL_VALID columns of the V-candidate pool'
                           % (int(target.max()), columns))
    with torch.autocast(device_type=block_scores.device.type, enabled=False):
        return F.cross_entropy(block_scores.float(), target, reduction='sum')


def native_grid_statistics(scores_valid: torch.Tensor, world_size: int, V: int) -> dict:
    """The statistics of the native arm, from the complete ``[V, V]`` valid grid it already has.

    The native arm builds no gate at all, so there is nothing to reuse from a tile pass: the grid is
    the score matrix itself and the margins are the real global ones. The gate fields stay at
    ``count = 0, value = null`` because this arm measures no gate.
    """
    accumulator = statistics_accumulator()
    accumulator['all_pair']['anchor_count'] = float(int(scores_valid.shape[0]))
    accumulator['all_pair']['candidate_count'] = float(int(scores_valid.shape[1]))
    statistics = finalize_statistics(accumulator)
    statistics.update(full_grid_statistics(scores_valid,
                                           torch.arange(int(scores_valid.shape[0]),
                                                        device=scores_valid.device,
                                                        dtype=torch.long)))
    statistics['gates_scope'] = ('the native arm has no gate: pU/mU statistics are count = 0, '
                                 'value = null, never a zero that looks measured')
    statistics['full_global_gate_grid'] = False
    statistics['native_grid_scope'] = ('the complete V x V valid grid of both directions (world_size '
                                       '= %d, V = %d)' % (int(world_size), int(V)))
    return statistics


def suffix_cross_entropy(scores: torch.Tensor, image_targets: torch.Tensor, text_local_rows=None,
                        image_chunk: int = IMAGE_CHUNK_DEFAULT) -> dict:
    """Both directions of the suffix CE and the row statistics of a COMPLETE valid grid.

    Only valid for a score block whose ROWS and COLUMNS are the same global valid pool: every column
    is then a valid candidate and ``image_targets`` labels one column of every row. The row count and
    the target count must match -- a ``[V, V]`` block handed over with only ``n_r`` targets is refused
    instead of letting the CE iterate rows that no target labels. The OBJECTIVE of this module does
    not use this helper (it slices ``Q_i2t`` and ``Q_t2i`` itself); it is kept for the single-process
    oracle, which really does hold the complete grid.

    Both directions are SUMS over the anchors, never means: the caller applies the
    ``world_size / V`` factor once.
    """
    images = int(scores.shape[0])
    targets_all = torch.as_tensor(image_targets).reshape(-1)
    if int(targets_all.numel()) != images:
        raise RuntimeError('suffix_cross_entropy got a %d-row score block with %d targets: the '
                           'targets must label EVERY row (slice the block to the anchors instead of '
                           'passing V rows with n_r targets)' % (images, int(targets_all.numel())))
    loss_i2t = scores.new_zeros(())
    count = 0
    positive_parts, strongest_parts, margin_parts = [], [], []
    for i0, i1 in _chunk_ranges(images, image_chunk):
        block = scores[i0:i1].float()
        targets = targets_all[i0:i1]
        with torch.autocast(device_type=scores.device.type, enabled=False):
            loss_i2t = loss_i2t + F.cross_entropy(block, targets.to(scores.device),
                                                  reduction='sum')
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
            'top1': float((scores.argmax(dim=1) == targets_all.to(scores.device)).float().mean()),
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
            with torch.autocast(device_type=scores.device.type, enabled=False):
                loss_t2i = loss_t2i + F.cross_entropy(transposed[i0:i1].float(),
                                                      columns[i0:i1], reduction='sum')
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


def suffix_native_scores_local(g_block: torch.Tensor, tR_block: torch.Tensor,
                               image_chunk: int = IMAGE_CHUNK_DEFAULT,
                               text_chunk: int = TEXT_CHUNK_DEFAULT) -> torch.Tensor:
    """The native arm's per-rank score block ``Q_local = 100 * g_local @ tR_valid.T``: ``[B, V]``.

    ``g_block`` is the LIVE ``[B, 512]`` image block of this rank and ``tR_block`` the LIVE
    ``[V, 512]`` valid suffix features; no global grid is ever built per rank.
    """
    images = int(g_block.shape[0])
    texts = int(tR_block.shape[0])
    out = torch.empty((images, texts), dtype=torch.float32, device=g_block.device)
    text_step = max(int(text_chunk), 1)
    for i0, i1 in _chunk_ranges(images, image_chunk):
        for j0 in range(0, texts, text_step):
            j1 = min(j0 + text_step, texts)
            tile = (g_block[i0:i1].float() * tR_block[j0:j1].float()).sum(-1)
            out[i0:i1, j0:j1] = SMARTCLIP_FIXED_SCALE * tile
    return out


# --------------------------------------------------------------------------- valid subset / DDP
def gather_local_rows(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Autograd-aware row gather in global rank order (identity when not distributed).

    Kept as the name the earlier callers used; it is :func:`differentiable_gather_rows` and nothing
    else, so there is exactly ONE gather implementation behind both names.
    """
    return differentiable_gather_rows(tensor, group=group)


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
        """``g = Norm(clip.encode_image(I))``: encoded ONCE, live, and shared by both tasks.

        Precision: the visual trunk runs under the NORMAL ambient autocast policy of the run (bf16 in
        this experiment, fp32 master weights untouched) and only the normalisation is forced to fp32.
        Wrapping the whole trunk in ``autocast(enabled=False)`` would silently change the trunk's
        arithmetic relative to the S0 recipe; computing ``g_raw`` under the ambient policy and then
        normalising in fp32 keeps S0 bit-comparable while making the readout precision explicit.
        """
        g_raw = self.clip.encode_image(images)
        return normalize_features(g_raw, eps=self.norm_eps)

    def encode_texts(self, text_ids: torch.Tensor, chunk: int = None):
        """One independent text forward: ``(t_raw, text_hidden)``.

        The prefix and the suffix each get their own call; encoding a full caption and slicing its
        hidden states is never done. The trunk itself keeps the run's ambient autocast policy (only
        the downstream normalisation and the suffix scoring are explicitly fp32).

        Memory: the S0 recipe already spent one full text forward (256 x 248 tokens x 12 layers) and the
        suffix branch adds a second one, which overflowed an 80 GiB card in the first real-entry smoke
        test. Two allowances, neither of which changes the arithmetic: the module's own checkpointed
        text encoder is used when it exists, and the rows are pushed through in chunks. Chunking keeps
        the token sequence and its ORDER unchanged (the rows are a contiguous slice of ``text_ids`` and
        the outputs are concatenated back in the same order); attention and LayerNorm act within a row,
        so a chunk boundary can only reorder floating-point accumulation, exactly like the existing
        image/text tile sizes.

        Activation checkpointing is NOT re-declared here: this method calls whatever encoder the
        module provides (``encode_text_with_checkpoint``) and passes no checkpointing flag of its own,
        so it can neither change a shared helper's default from reentrant to non-reentrant nor claim a
        reentrancy the underlying helper does not use. A caller that needs the explicit non-reentrant
        form calls the encoder itself with ``use_reentrant=False, preserve_rng_state=True``.
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
        J = validity['global_valid_indices']                     # GLOBAL_FULL indices, [W*B)
        V = int(validity['global_valid_V'])
        local_v = int(validity['local_valid_count'])
        scale = suffix_scaling(world, V)

        # ---- ONE index-space discipline ------------------------------------------------------
        # LOCAL_FULL    : this rank's B rows
        # GLOBAL_FULL   : the W*B rows gathered in rank order (J lives HERE)
        # GLOBAL_VALID  : the V rows left after the validity filter (anchor_valid lives HERE)
        # The first smoke test crashed with `indexSelectLargeIndex srcIndex < srcSelectDimSize`
        # because J was applied to tensors that were not all in GLOBAL_FULL. Every tensor J selects
        # from is therefore created here, in GLOBAL_FULL, and checked BEFORE the GPU gather:
        tR_local = normalize_features(suffix_raw, eps=self.norm_eps)      # LOCAL_FULL [B, 512]
        tR_global_full = differentiable_gather_rows(tR_local)             # GLOBAL_FULL [W*B, 512]
        mS_global_full = differentiable_gather_rows(mask_s.detach())      # GLOBAL_FULL [W*B, 512]
        assert_global_full_index_space(tR_global_full, mS_global_full, J, world, local_size)
        if int(tR_global_full.shape[0]) != world * local_size:
            raise RuntimeError('the suffix candidate pool is not the whole global batch: '
                               'tR_global_full has %d rows but the global batch is W*B = %d. The '
                               'suffix texts must be encoded one row per GLOBAL_FULL row (an '
                               'empty-string placeholder for an invalid suffix), otherwise a '
                               'GLOBAL_FULL index set cannot address them.'
                               % (int(tR_global_full.shape[0]), world * local_size))
        tR_valid = tR_global_full.index_select(0, J)                      # GLOBAL_VALID [V, 512]
        mS_valid = mS_global_full.index_select(0, J)                      # GLOBAL_VALID [V, 512]
        # ---- the explicit GLOBAL_VALID labelling of THIS rank's valid rows --------------------
        pool = valid_pool_mapping(valid_flags, J, world, local_size, device=device)
        local_rows = pool['local_rows']                     # LOCAL_FULL rows of this rank
        anchor_valid = pool['anchor_valid']                 # their GLOBAL_VALID labels in [0, V)
        anchor_global = pool['anchor_global']               # their GLOBAL_FULL rows
        # the S0 mask of this image's OWN prefix, for the positive-pair keep rate (GLOBAL_FULL)
        mS_positive_pair_global = mS_global_full.narrow(0, self.rank * local_size, local_size)

        # ---- both directions, from the [B, V] block this rank really owns ---------------------
        statistics = {}
        if V >= 2:
            if self.arm == ARM_NATIVE:
                Q_local = suffix_native_scores_local(g_live, tR_valid,
                                                     image_chunk=image_chunk,
                                                     text_chunk=text_chunk)
                readout_statistics = None
            else:
                read = suffix_readout_scores(g_live, mS_valid, tR_valid, self.suffix_mask,
                                             image_chunk=image_chunk, text_chunk=text_chunk,
                                             eps=self.norm_eps,
                                             want_statistics=bool(want_statistics))
                Q_local = read['scores']
                readout_statistics = read['statistics']
            Q_i2t = (Q_local.index_select(0, local_rows) if int(local_rows.numel())
                     else Q_local[:0])
            targets_i2t = anchor_valid
            # GLOBAL_FULL rows of every rank's [B, V] block, then the V rows of the valid pool
            Q_global_full = differentiable_gather_rows(Q_local)           # [W*B, V]
            Q_valid = Q_global_full.index_select(0, J)                    # [V, V]
            Q_t2i = (Q_valid.index_select(1, anchor_valid).t().contiguous()
                     if int(anchor_valid.numel()) else Q_valid[:0, :].t().contiguous())
            targets_t2i = anchor_valid
            loss_i2t_sum = suffix_cross_entropy_sum(Q_i2t, targets_i2t)
            loss_t2i_sum = suffix_cross_entropy_sum(Q_t2i, targets_t2i)
            loss_suffix_local_sum = loss_i2t_sum + loss_t2i_sum
            # (world_size / V) * (local I2T CE sum + local T2I CE sum), applied EXACTLY ONCE; plain
            # DDP averaging then divides by world_size, which is exactly the global valid-anchor
            # mean. No per-rank valid mean is ever averaged and world_size is never applied twice.
            loss_suffix = scale * loss_suffix_local_sum
            statistics = (readout_statistics if readout_statistics is not None
                          else native_grid_statistics(Q_valid, world, V))
            statistics['local_I2T_CE_sum'] = float(loss_i2t_sum.detach())
            statistics['local_T2I_CE_sum'] = float(loss_t2i_sum.detach())
            statistics['local_CE_sum'] = float(loss_suffix_local_sum.detach())
            statistics['suffix_score_scope'] = (
                'per rank: Q_local [B, V] (%s arm) -> I2T over this rank\'s valid rows -> '
                'differentiable gather -> the V x V global valid grid, never a full grid per rank'
                % self.mask_kind)
            # a rank with n_r = 0 (or a text set with no local row) contributes a differentiable
            # zero, so its gate still takes part in the DDP graph instead of being reported unused
            if local_v == 0 or not int(anchor_valid.numel()):
                loss_suffix = loss_suffix + differentiable_zero(Q_local, tR_valid)
        else:
            # V < 2: the suffix loss is exactly zero for this step. No re-draw of K, no placeholder
            # candidate and no single-candidate CE pretending to train -- and the rank still joins
            # every collective and contributes a differentiable zero.
            loss_i2t_sum = differentiable_zero(tR_global_full)
            loss_t2i_sum = differentiable_zero(tR_global_full)
            loss_suffix_local_sum = loss_i2t_sum + loss_t2i_sum
            loss_suffix = loss_suffix_local_sum
            statistics = statistics_accumulator()
            statistics['all_pair']['candidate_count'] = float(V)
            statistics = finalize_statistics(statistics)
            statistics['local_I2T_CE_sum'] = 0.0
            statistics['local_T2I_CE_sum'] = 0.0
            statistics['local_CE_sum'] = 0.0
            statistics['suffix_score_scope'] = ('V < 2: no suffix pair was scored this step, so '
                                                'every statistic here is count = 0, value = null')

        loss_s0 = smart['loss_s0']
        loss_suffix_weighted = self.lambda_suffix * loss_suffix
        loss_total = loss_s0 + loss_suffix_weighted
        # mS keep rates, DETACHED and reduced here: a live tensor or the F module itself is never put
        # into this dict, because an output tensor that does not contribute to the loss corrupts
        # DDP's unused-parameter bookkeeping and a stored Module keeps the gate alive in the graph.
        mask_s_local_detached = mask_s.detach().float()
        mask_s_positive_pair = mS_positive_pair_global.detach().float()
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
            'suffix_valid_flags': valid_flags.detach(),
            'suffix_valid_ratio_local': float(valid_flags.float().mean()),
            'suffix_valid_ratio_global': V / float(world * local_size),
            'empty_suffix_fraction_global': 1.0 - V / float(world * local_size),
            'near_zero_readout_norm_count': int(
                statistics['sides']['all_pair']['near_zero_readout_norm_count']),
            'readout_pair_count': int(statistics['sides']['all_pair']['pair_count']),
            'index_spaces': {
                'LOCAL_FULL': 'this rank\'s B rows',
                'GLOBAL_FULL': 'the W*B rows gathered in rank order; J lives here',
                'GLOBAL_VALID': 'the V rows after the validity filter; anchor_valid lives here',
                'J_max': int(J.max()) if int(J.numel()) else None,
                'global_full_rows': int(tR_global_full.shape[0]),
                'anchor_valid': [int(value) for value in anchor_valid[:8]],
            },
            'prefix_content_tokens': content_token_counts(text_ids['prefix']),
            'suffix_content_tokens': token_counts,
            'prefix_effective_lengths': effective_token_length(prefix_hidden,
                                                               text_ids['prefix']).cpu().tolist(),
            'suffix_effective_lengths': effective_token_length(suffix_hidden,
                                                               text_ids['suffix']).cpu().tolist(),
            'suffix_context_length': self.context_length,
            'prefix_captions': None if split is None else split['prefix'],
            'suffix_texts': suffix_texts,
            'mask_s_keep_ratio_rank_local_all_candidates': float(
                (mask_s_local_detached >= 0.5).float().mean()),
            'mask_s_positive_pair_local': mask_s_positive_pair.detach(),
        })
        if want_statistics:
            out['_statistics'] = self.diagnostics(out, mask_s_local_detached, statistics,
                                                  valid_flags.detach(),
                                                  mask_s_positive_pair=mask_s_positive_pair)
        return out

    def diagnostics(self, out: dict, mask_s_detached, statistics: dict, valid_flags,
                    mask_s_positive_pair: torch.Tensor = None) -> dict:
        """The diagnostics the trainer turns into log fields, from the forward that just ran.

        Scales follow the field names: ``*_rank_local_*`` is this rank's batch only,
        ``*_positive_pair_*`` is the (image, own suffix) diagonal and ``*_all_pair_*`` is every scored
        pair. Every gate/readout number comes from the tile statistics measured during the SAME
        forward; nothing is re-run and no full gate tensor is claimed.
        """
        values = {}
        gate = out['suffix_statistics']
        for name in ('positive_mean', 'strongest_negative_mean', 'max_margin_mean',
                     'lse_margin_mean', 'lse_margin_min', 'positive_win_fraction', 'top1',
                     'anchor_count', 'candidate_count', 'margin_anchor_count',
                     'margin_candidate_count', 'local_I2T_CE_sum', 'local_T2I_CE_sum',
                     'local_CE_sum'):
            if name in gate and gate[name] is not None:
                values[name] = gate[name]
        sides = gate.get('sides') or {}
        all_pair = sides.get('all_pair') or {}
        pairs = int(all_pair.get('pair_count') or 0)
        values['readout_pair_count'] = pairs
        values['mU_keep_ratio_all_valid_pairs'] = all_pair.get('keep_ratio')
        values['mU_probability_mean'] = all_pair.get('probability_mean')
        values['gated_readout_norm_ratio_mean'] = all_pair.get('readout_norm_ratio_mean')
        values['near_zero_readout_norm_count'] = int(
            all_pair.get('near_zero_readout_norm_count') or 0)
        # the trainer's earlier column names, now filled with the count-weighted global-valid numbers
        values['mU_keep_ratio_global_valid_tiles'] = all_pair.get('keep_ratio')
        values['mU_keep_ratio_local_rows_global_valid_columns'] = all_pair.get('keep_ratio')
        values['mU_keep_ratio_note'] = (
            'keep ratio = (pU >= 0.5) fraction of the GATE ELEMENTS of the scored tiles (p=[0.9,'
            '0.1,0.1] is 1/3); the value is null, never 0, when no pair was scored')
        positive = sides.get('positive_pair') or {}
        values['mU_keep_ratio_positive_pair'] = positive.get('keep_ratio')
        values['mU_probability_mean_positive_pair'] = positive.get('probability_mean')
        values['gated_readout_norm_ratio_mean_positive_pair'] = positive.get(
            'readout_norm_ratio_mean')
        values['positive_pair_count'] = int(positive.get('pair_count') or 0)
        mask_diag = mask_s_detached.detach().float()
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
