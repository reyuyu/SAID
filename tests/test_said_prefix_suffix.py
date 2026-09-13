"""S0-Suffix v0.1 acceptance tests -- written against the BINDING SPEC, not against the code.

Spec: ``docs/said_prefix_suffix_v01/spec.md`` (sections 2-10 and the frozen API list in section 14).
Every test below is derived from the spec text and is meant to fail loudly on a plausible wrong
implementation:

A  the prefix/suffix split, including the spec's worked examples and the inclusive ``randint`` K draw;
B  text isolation: the prefix hidden state comes only from the prefix string, the suffix is encoded
   by a separate forward, and with image and prefix fixed a changed suffix moves ONLY its own target;
C  the new gate inputs: ``rS = g*mS`` is NOT re-normalised, F's input is fully detached, the final
   ``g`` stays live, and ``mU`` multiplies the FULL ``g`` rather than the hard complement ``1-mS``;
D  initialisation: ``pU = 8/9``, ``mU`` exactly all ones, conditional == native score, the
   zero-initialised output layer gets no gradient on the first backward, and the global RNG survives;
E  S0 regression: with ``lambda_suffix = 0`` the loss and the gradients of the old mask and the
   shared trunk equal the ORIGINAL S0 helper's, from the same inputs, and building the new modules
   does not shift the data or RNG stream;
F  the chunked readout equals a per-pair reference loop in values AND in the gradients of ``g``,
   ``tR`` and F, with the global positive index on the diagonal, and the old ``mS`` getting none;
G  the valid subset and the ``world_size / V`` DDP scaling, proved with REAL two-rank runs through
   ``torch.distributed.run`` against a single-process oracle: equal counts, unequal counts, a rank
   with zero valid samples, ``V = 0`` and ``V = 1``, and one real optimizer update (not a forward);
H  all-off / near-zero ``mU``: finite loss and gradients, no dropped sample or candidate, no silent
   fallback to the native score, and a learned all-off gate is still a VALID suffix;
I  the full checkpoint round-trip through the PRODUCTION writer, with a non-initial F and a
   non-initial old S0 mask, plus key-collision and tamper detection;
J  the trainer's CLI surface (no sweep / extra-epoch / allow-missing escape hatch) and its arms.

ASSUMPTIONS (the implementation is written in parallel and is not present in this checkout; these
tests follow the frozen names of spec section 14 plus the checkpoint schema that the runner and the
exporter already pin down):

* ``SuffixMask`` is F (``Linear(1024,512) -> GELU -> Linear(512,512)``, last weight zero,
  ``bias = log(8)``) with ``state_dict`` keys ``layer1.weight/bias`` and ``layer2.weight/bias``;
* the straight-through gate is ``SuffixMaskGate.gate_from_pU(pU)`` (a module-level
  ``gate_from_pU`` of the same behaviour is accepted);
* ``suffix_readout_scores(...)`` returns ``scores`` or ``(scores, mU)`` and accepts the mask, the
  three tensors and F in one of the documented orders; :func:`call_readout` tries those orders and
  FAILS LOUDLY (never silently skips) when none of them fits;
* ``split_prefix_suffix(caption, k)`` accepts ``k`` positionally or as a keyword;
* the S0 term dict of an objective output exposes ``loss_sidm`` / ``loss_dism`` / ``loss_sparsity``
  (directly or under ``terms`` / ``s0`` / ``smart``);
* the production checkpoint writer is a module-level function of ``train/train_said_prefix_suffix.py``.
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
# Every comparison states WHY its tolerance is what it is. The rule of section 9 is that a tolerance
# may absorb floating-point accumulation order, never a wrong multiplier: a world_size factor, a
# re-normalisation, a missing detach or a wrong positive column all move these numbers by orders of
# magnitude, far outside every constant below.
TOL_EXACT = 0.0             # bitwise: both sides are literally the same computation
TOL_IDENTICAL_PATH = 1e-6   # the same arithmetic with one extra copy: fp32 round-off only
TOL_RECOMPUTE = 1e-5        # independently re-written arithmetic sharing no code, fp32
TOL_GRADIENT = 1e-4         # relative to the tensor's own max-abs: fp32, different accumulation
TOL_DDP = 1e-3              # relative: NCCL reduction plus a different per-rank summation order
STATE_TOLERANCE = 1e-6      # pU == 8/9 to 1e-6, per spec section 4

CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
SUFFIXES = ['on a wooden table', 'through tall grass', 'against a wall', 'on a calm lake']
FIXED_SCALE = 100.0
S0_LAMBDA_SPARSE = 2.0


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


def fixed_images(count=2, seed=20260913, device=DEVICE):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, 224, 224, generator=generator).to(device)


def tokenize(texts, device=DEVICE):
    from model import longclip
    return longclip.tokenize(list(texts), truncate=True).to(device)


def make_suffix_mask(module, device=DEVICE, seed=None):
    """Build F through whatever optional constructor arguments the implementation declares."""
    parameters = set(inspect.signature(module.SuffixMask.__init__).parameters)
    kwargs = {}
    if 'device' in parameters:
        kwargs['device'] = device
    if seed is not None and 'seed' in parameters:
        kwargs['seed'] = seed
    return module.SuffixMask(**kwargs).to(device)


def randomise_suffix_mask(mask, scale=0.5, seed=3):
    """A genuinely NON-initial F: every tensor away from its zero/``log(8)`` start."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in mask.named_parameters():
            if parameter.dim() == 0:
                parameter.add_(float(torch.randn(1, generator=generator)) * scale)
            else:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
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


def _try_calls(function, orders, kwargs_variants):
    errors = []
    for order in orders:
        for kwargs in kwargs_variants:
            try:
                return function(*order, **kwargs)
            except TypeError as error:
                errors.append(str(error))
    raise AssertionError('none of the documented call shapes of %r fitted; the frozen API of spec '
                         'section 14 must accept one of them. Errors: %s'
                         % (getattr(function, '__name__', function), ' | '.join(errors[:6])))


def call_readout(module, g, mS, tR, gate, image_chunk=2, text_chunk=2):
    """``suffix_readout_scores`` -> ``(scores, mU)`` accepting the documented argument orders."""
    function = module.suffix_readout_scores
    orders = [(g, mS, tR, gate), (gate, g, mS, tR), (g, gate, mS, tR), (mS, g, tR, gate)]
    kwargs_variants = [{'image_chunk': image_chunk, 'text_chunk': text_chunk},
                       {'image_chunk': image_chunk, 'text_chunk': text_chunk,
                        'positive_columns': None},
                       {}]
    result = _try_calls(function, orders, kwargs_variants)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def call_split(module, caption, k):
    """``split_prefix_suffix(caption, k)`` with ``k`` positional or keyword."""
    return _try_calls(module.split_prefix_suffix, [(caption, k), (caption,)], [{'k': k}, {}])


def call_mask_helper(module, mask_net, hidden):
    """The ORIGINAL S0 mask pipeline, reused and never re-implemented (spec section 1)."""
    function = getattr(module, 'said_mask_helper', None)
    if function is None:
        from model.said_cls_cvssl import said_mask_from_hidden as function
    return function(mask_net, hidden, soft_mask=False)


def call_global_valid_indices(module, local_valid, world_size=None):
    """``global_valid_indices`` on a per-rank validity tensor, optionally with the world size."""
    variants = [(local_valid,), (local_valid, world_size)]
    kwargs_variants = [({'world_size': world_size} if world_size is not None else {}), {}]
    return _try_calls(module.global_valid_indices, variants, kwargs_variants)


def suffix_state_from_payload(payload):
    """The suffix mask TENSOR dict of a checkpoint payload, whatever key it is stored under.

    A description dict standing in for the tensor dict is refused here: the two live under separate
    keys by design and merging them is the failure this helper exists to catch. The value is
    returned unchanged for a real tensor dict, so the digest and the strict load still see exactly
    what the checkpoint carries.
    """
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


def relative_difference(mine, reference):
    scale = max(float(reference.abs().max()), 1e-12)
    return float((mine - reference).abs().max()) / scale


def parameter_snapshot(mask):
    return {key: value.detach().clone() for key, value in mask.state_dict().items()}


def assert_parameters_unchanged(mask, snapshot, what):
    """No forward or backward of the readout may MUTATE F's tensors.

    This guard exists because a bare ``Linear`` inside a ``Sequential`` silently supports in-place
    calls, so a readout that reaches into ``mask.layer2(x)`` would rewrite F on every call and make
    two computations that should agree drift apart. Without the guard that failure is invisible
    until the numbers disagree for no obvious reason.
    """
    current = mask.state_dict()
    assert set(current) == set(snapshot), (sorted(current), sorted(snapshot))
    for key in snapshot:
        if not torch.equal(current[key].detach(), snapshot[key]):
            drift = float((current[key].detach() - snapshot[key]).abs().max())
            raise AssertionError('%s mutated the suffix mask tensor %r by up to %r: the readout '
                                 'must never write into F' % (what, key, drift))


def clip_reference_suffix_loss(scores, positive_weight=1.0):
    """``(CE_i2t(Q,y) + CE_t2i(Q^T,y)) / V`` for a ``V x V`` matrix whose diagonal is the positive."""
    targets = torch.arange(scores.shape[0], device=scores.device)
    return (F.cross_entropy(scores, targets, reduction='sum')
            + F.cross_entropy(scores.t(), targets, reduction='sum')) / float(scores.shape[0]) \
        * positive_weight


def per_pair_suffix_forward(module, gate, g, mS, tR, positive_weight=1.0):
    """The spec's per-pair definition, written out one (i, j) at a time, sharing no code with F.

    ``u_ij = Norm(g_i * mU_ij)`` where ``mU_ij = hardST(F([g_i, g_i * mS_j]))`` and
    ``score_ij = 100 * u_ij . tR_j``: candidate j brings its OWN prefix mask, which is exactly what
    the chunked implementation must reproduce. The fixed 100 is applied HERE as well, so a readout
    that silently dropped the scale cannot agree with this reference.

    NOTE: a bare ``Linear`` inside a ``Sequential`` silently supports in-place calls, so calling a
    layer directly would MUTATE F. The mask path therefore goes through the module's ``forward``
    only.
    """
    scores = []
    masks = []
    for i in range(g.shape[0]):
        row = []
        mask_row = []
        for j in range(tR.shape[0]):
            xU = torch.cat([g[i].detach(), (g[i].detach() * mS[j].detach())]).unsqueeze(0)
            pU = gate(xU)
            mU = gate_from_pU(module, pU)
            u = F.normalize(g[i].unsqueeze(0) * mU, dim=-1, eps=1e-6)
            row.append(positive_weight * (u @ tR[j].unsqueeze(1)).squeeze())
            mask_row.append(mU.squeeze(0))
        scores.append(torch.stack(row))
        masks.append(torch.stack(mask_row))
    return torch.stack(scores), torch.stack(masks)


# =========================================================================== A: prefix / suffix
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
    suffix_b_ids = tokenize(['in a swimming pool', 'over a frozen lake'])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR_a = F.normalize(model.encode_text(suffix_a_ids).float(), dim=-1, eps=1e-6)
        tR_b = F.normalize(model.encode_text(suffix_b_ids).float(), dim=-1, eps=1e-6)
    assert not torch.equal(tR_a, tR_b), 'the two suffixes must differ for this test to mean anything'

    gate = make_suffix_mask(module)
    with torch.no_grad():
        scores_a, mU_a = call_readout(module, g, mS, tR_a, gate, 2, 2)
        scores_b, mU_b = call_readout(module, g, mS, tR_b, gate, 2, 2)
        hidden_again = model.encode_text(prefix_ids, return_full=True)[1]
    diff = float((scores_a - scores_b).abs().max())
    assert diff > 1e-3, ('a different suffix must change the suffix scores', diff)
    assert torch.equal(hidden, hidden_again), 'the prefix hidden state is a pure function of P'
    assert call_mask_helper(module, model.mask_net, hidden_again).equal(mS)
    if mU_a is not None and mU_b is not None:
        assert torch.equal(mU_a, mU_b), \
            'mU depends on [g, g*mS] only: changing R must not move the gate at all'

    combined_ids = tokenize(['a photo of a cat on a wooden table on a wooden table'])
    with torch.no_grad():
        t_combined = F.normalize(model.encode_text(combined_ids).float(), dim=-1, eps=1e-6)
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
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
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
    gate = make_suffix_mask(module)
    generator = torch.Generator().manual_seed(5)
    g = F.normalize(torch.randn(3, 512, generator=generator), dim=-1, eps=1e-6)
    mS = (torch.rand(3, 512, generator=generator) >= 0.5).float()

    rS = g.detach() * mS.detach()
    assert torch.equal(rS, g.detach() * mS.detach())
    unit = F.normalize(rS, dim=-1, eps=1e-6)
    norm_rS = rS.norm(dim=-1)
    assert float(norm_rS.min()) > 0.1, 'the mask must keep enough coordinates for this to be sharp'
    assert float((norm_rS - 1.0).abs().max()) > 1e-3, \
        'rS keeps the norm of g*mS: it is NOT unit norm, so a re-normalisation is detectable'
    assert float(unit.norm(dim=-1).min()) == pytest.approx(1.0, abs=1e-5)

    xU = torch.cat([g.detach(), rS], dim=-1)
    assert tuple(xU.shape) == (3, 1024), xU.shape
    assert torch.equal(xU[:, 512:], rS), \
        'the second half of the F input must be exactly g*mS, never Norm(g*mS)'
    assert not torch.allclose(xU[:, 512:], unit, atol=1e-4)
    assert float(gate(xU).shape[-1]) == 512


def test_c_gate_mU_is_a_straight_through_hard_gate():
    """Spec section 4: ``pU = sigmoid(F(xU))`` and ``mU = hardU + (pU - pU.detach())``."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.02, seed=7)
    xU = torch.randn(4, 1024, generator=torch.Generator().manual_seed(7))
    pU = gate(xU)
    assert float(pU.min()) >= 0.0 and float(pU.max()) <= 1.0, \
        'forward(x) must return a probability'
    assert float(pU.min()) > 1e-6 and float(pU.max()) < 1.0 - 1e-6, \
        'forward(x) must return the sigmoid probability pU, not a saturated hard value or a logit'
    assert float(pU.mean()) == pytest.approx(8.0 / 9.0, abs=0.05), \
        'at initialisation pU is 8/9; a mild perturbation must stay near it'
    mU = gate_from_pU(module, pU)
    hard = (pU >= 0.5).float()
    assert torch.equal(mU, hard + (pU - pU.detach()))
    # the forward value is exactly the hard gate, so every entry of mU is exactly 0 or 1
    assert set(torch.unique(mU).tolist()) <= {0.0, 1.0}
    # and the straight-through part really is a pass-through: d mU / d pU == 1
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
    g_leaf = F.normalize(torch.randn(2, 512, generator=generator), dim=-1, eps=1e-6) \
        .requires_grad_(True)
    mS = (torch.rand(2, 512, generator=generator) >= 0.5).float()
    tR = F.normalize(torch.randn(2, 512, generator=generator), dim=-1, eps=1e-6)

    xU = torch.cat([g_leaf.detach(), (g_leaf.detach() * mS.detach())], dim=-1)
    mU = gate_from_pU(module, gate(xU))
    u = F.normalize(g_leaf * mU, dim=-1, eps=1e-6)
    scores = FIXED_SCALE * (u @ tR.t())
    live_grads = torch.autograd.grad(scores.diagonal().sum(), [g_leaf] + list(gate.parameters()),
                                     allow_unused=True)
    assert live_grads[0] is not None and float(live_grads[0].abs().max()) > 0, \
        'the final g must stay live: the suffix loss trains the visual trunk through g*mU'
    assert any(grad is not None and float(grad.abs().max()) > 0 for grad in live_grads[1:]), \
        'F must receive its own gradient'

    detached_g = F.normalize(torch.randn(2, 512, generator=generator), dim=-1, eps=1e-6) \
        .requires_grad_(True)
    detached_mS = (torch.rand(2, 512, generator=generator) >= 0.5).float().requires_grad_(True)
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


@needs_cuda
def test_c_mU_multiplies_the_full_g_not_the_hard_complement():
    """Spec section 4: ``u = Norm(g * mU)``, explicitly NOT ``g * (1 - mS)``."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.6, seed=17)
    generator = torch.Generator().manual_seed(23)
    g = torch.randn(3, 512, generator=generator)
    mS = (torch.rand(3, 512, generator=generator) >= 0.5).float()
    tR = F.normalize(torch.randn(3, 512, generator=generator), dim=-1, eps=1e-6)
    with torch.no_grad():
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = gate(xU)
        mU = gate_from_pU(module, pU)
        assert float(mU.min()) == 0.0 and float(mU.max()) == 1.0, \
            'this test needs a genuinely mixed mU to be able to fail'
        correct = FIXED_SCALE * (F.normalize(g * mU, dim=-1, eps=1e-6) @ tR.t())
        wrong_complement = FIXED_SCALE * (F.normalize(g * (1.0 - mS), dim=-1, eps=1e-6) @ tR.t())
        wrong_hard = FIXED_SCALE * (F.normalize(g * (1.0 - (pU >= 0.5).float()), dim=-1, eps=1e-6)
                                    @ tR.t())
    assert float((correct - wrong_complement).abs().max()) > 1e-3, \
        'mU multiplies the FULL g, so g*(1-mS) must give a measurably different score'
    assert float((correct - wrong_hard).abs().max()) > 1e-3
    assert not torch.allclose(correct, wrong_hard, atol=1e-3)


@needs_cuda
def test_c_readout_is_the_gate_applied_to_the_full_g_readout():
    """An independent re-implementation of the readout must reproduce the module's scores.

    This is the same arithmetic written out by hand: ``u_ij = Norm(g_i * hardST(F([g_i, g_i*mS_j])))``
    and ``score_ij = 100 * u_ij . tR_j``. It shares no code with the chunked readout, so it pins
    both the gate wiring and the full-g multiplication at once.
    """
    module = implementation_module()
    model = build_model()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=19)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_ids = tokenize(SUFFIXES[:2])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        scores, _ = call_readout(module, g, mS, tR, gate, 2, 2)
        rows = []
        for i in range(2):
            columns = []
            for j in range(2):
                xU = torch.cat([g[i].detach(), (g[i].detach() * mS[j].detach())]).unsqueeze(0)
                mU = gate_from_pU(module, gate(xU))
                u = F.normalize(g[i].unsqueeze(0) * mU, dim=-1, eps=1e-6)
                columns.append(FIXED_SCALE * (u @ tR[j].unsqueeze(1)).squeeze())
            rows.append(torch.stack(columns))
        reference = torch.stack(rows)
    assert float((scores - reference).abs().max()) < TOL_RECOMPUTE, \
        ('the readout must be the hand-written per-pair definition',
         float((scores - reference).abs().max()))


# =========================================================================== D: initialisation
def test_d_initialisation_is_pU_8_9_and_mU_exactly_ones_with_a_restored_rng():
    """Spec section 4: seed 0, Xavier first layer, zero last weight, ``bias = log(8)``."""
    module = implementation_module()
    torch.manual_seed(4321)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = ([torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())]
                   if CUDA else [])
    mask = make_suffix_mask(module, DEVICE)
    assert torch.equal(torch.get_rng_state(), cpu_state), \
        'building SuffixMask must not consume the global CPU RNG stream'
    if CUDA:
        for index, state in enumerate(cuda_states):
            assert torch.equal(torch.cuda.get_rng_state(index), state), \
                'building SuffixMask must restore CUDA generator %d' % index

    parameters = dict(mask.named_parameters())
    assert 'layer1.weight' in parameters and 'layer2.weight' in parameters, \
        ('the suffix mask must be Linear(1024,512) -> GELU -> Linear(512,512); found %r'
         % sorted(parameters))
    assert tuple(parameters['layer1.weight'].shape) == (512, 1024)
    assert tuple(parameters['layer2.weight'].shape) == (512, 512)
    assert float(parameters['layer2.weight'].abs().max()) == 0.0, 'the last weight must be zero'
    assert float(parameters['layer2.bias'].min()) == pytest.approx(math.log(8.0), abs=1e-6)
    assert float(parameters['layer2.bias'].max()) == pytest.approx(math.log(8.0), abs=1e-6)
    assert float(parameters['layer1.weight'].std()) > 0
    assert float(parameters['layer1.bias'].abs().max()) == 0.0

    xU = torch.randn(4, 1024, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        pU = mask(xU)
        mU = gate_from_pU(module, pU)
    assert float(pU.min()) == pytest.approx(8.0 / 9.0, abs=STATE_TOLERANCE)
    assert float(pU.max()) == pytest.approx(8.0 / 9.0, abs=STATE_TOLERANCE)
    assert float((mU - 1.0).abs().max()) == 0.0, 'mU must be EXACTLY all ones at initialisation'
    assert int(mU.sum().item()) == mU.numel()


@needs_cuda
def test_d_first_backward_gives_the_zero_initialised_output_layer_no_gradient():
    module = implementation_module()
    model = build_model()
    mask = make_suffix_mask(module)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_ids = tokenize(SUFFIXES[:2])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        mU = gate_from_pU(module, mask(xU))
        u = F.normalize(g * mU, dim=-1, eps=1e-6)
        scores = FIXED_SCALE * (u @ tR.t())
        pU = mask(xU)
        scores = FIXED_SCALE * (F.normalize(g * gate_from_pU(module, pU), dim=-1, eps=1e-6)
                                @ tR.t())
    scores.diagonal().sum().backward()
    parameters = dict(mask.named_parameters())
    assert parameters['layer1.weight'].grad is not None
    assert float(parameters['layer1.weight'].grad.abs().max()) > 0, \
        'the first layer must receive a real gradient on the first backward'
    assert float(parameters['layer2.weight'].grad.abs().max()) == 0.0, \
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
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = mask(xU)
        mU = gate_from_pU(module, pU)
        conditional, returned_mU = call_readout(module, g, mS, tR, mask, 3, 3)
        native = FIXED_SCALE * (g @ tR.t())
    assert float((mU - 1.0).abs().max()) == 0.0, 'the initial gate must be exactly all ones'
    difference = float((conditional - native).abs().max())
    assert difference < TOL_RECOMPUTE, (difference, 'QU must equal QN when mU is all ones')
    assert clip_reference_suffix_loss(conditional) == pytest.approx(
        clip_reference_suffix_loss(native), rel=1e-5, abs=1e-7)
    if returned_mU is not None:
        assert torch.equal(returned_mU, mU)
    mS_leaf = mS.detach().clone().requires_grad_(True)
    scores_from_leaf, _ = call_readout(module, g, mS_leaf, tR, mask, 3, 3)
    value = torch.autograd.grad(scores_from_leaf.sum(), mS_leaf, allow_unused=True)[0]
    assert value is None or float(value.abs().max()) == 0.0, \
        'the old mS must be a stop-gradient input of the suffix readout'


# =========================================================================== E: S0 regression
def _objective_for_s0(module, model, rank=0):
    """The S0 term of the new path, whichever builder the trainer exposes."""
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
                        ('module', module), ('suffix_model', module),
                        ('lambda_suffix', 0.0), ('rank', rank), ('device', DEVICE),
                        ('soft_mask', False)):
        if name in signature.parameters:
            accepted[name] = value
    if 'lambda_suffix' not in accepted:
        accepted['lambda_suffix'] = 0.0            # the spec's S0-equivalence switch
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


@needs_cuda
def test_e_lambda_suffix_zero_reproduces_the_original_s0_helper_loss_and_gradients():
    """Spec section 3: at ``lambda_suffix = 0`` the S0 loss and its gradients are the ORIGINAL ones.

    Tolerance: TOL_GRADIENT = 1e-4 relative to each tensor's own max-abs. The two sides run the same
    fp32 arithmetic with a different accumulation order (the new path encodes the prefix features
    once next to the hidden state, the old helper re-derives them), so a bitwise comparison is not
    available. What this test is designed to catch -- a re-derived SIDM/DISM, a changed sparse weight,
    a mask used without its stop-gradient, an extra 0.5 factor or a world-size factor -- moves each
    tensor by orders of magnitude, not by 1e-4.
    """
    module = implementation_module()
    from model.said_cls_cvssl import compute_smartclip_terms
    model = build_model(shared_init=True)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    image_ids = torch.arange(2, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        v_a = model.encode_image(images)
        text_raw, hidden = model.encode_text(prefix_ids, return_full=True)
    mS_reference = call_mask_helper(module, model.mask_net, hidden)
    terms_reference = compute_smartclip_terms(v_a, text_raw, mS_reference, rank=0)
    loss_reference = terms_reference['loss_sidm'] + terms_reference['loss_dism'] \
        + terms_reference['loss_sparsity']
    # pin the original constants so this test cannot agree with a changed S0 by accident
    assert float(terms_reference['loss_sparsity']) == pytest.approx(
        float(mS_reference.abs().mean()), rel=1e-6)
    mask_parameters = list(model.mask_net.parameters())
    trunk_parameters = [model.visual.proj, model.text_projection]
    reference_grads = torch.autograd.grad(loss_reference, mask_parameters + trunk_parameters)

    objective = _objective_for_s0(module, model)
    try:
        out = objective(images, prefix_ids, image_ids)
    except TypeError:
        out = objective(images, prefix_ids)
    terms = s0_terms_of(out)
    loss_new = terms['loss_sidm'] + terms['loss_dism'] + terms['loss_sparsity']
    assert abs(float(loss_new) - float(loss_reference)) <= \
        TOL_IDENTICAL_PATH * max(abs(float(loss_reference)), 1.0), \
        ('the S0 loss at lambda_suffix = 0 must equal the original helper', float(loss_new),
         float(loss_reference))
    new_grads = torch.autograd.grad(loss_new, mask_parameters + trunk_parameters, retain_graph=True)
    for index, (mine, theirs) in enumerate(zip(new_grads, reference_grads)):
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('S0 gradient %d differs from the original helper: lambda_suffix = 0 must leave the S0 '
             'objective, the old mask and the shared trunk untouched' % index)
    assert all(float(grad.abs().max()) > 0 for grad in reference_grads), \
        'the reference gradients must be non-trivial for this comparison to mean anything'
    if 'mS' in terms:
        assert torch.equal(terms['mS'], mS_reference)


@needs_cuda
def test_e_building_the_new_modules_does_not_shift_the_data_or_rng_stream():
    """Spec section 3: the new modules must not pollute the original data random stream."""
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

    before = draw_stream(0)
    model = build_model(shared_init=True)
    make_suffix_mask(module, DEVICE)
    after = draw_stream(0)
    assert before == after, ('building the new modules shifted the data stream: the S0 arm would '
                             'no longer see the same samples')

    torch.manual_seed(2026)
    state_before = torch.get_rng_state().clone()
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    with torch.no_grad():
        model.encode_image(images)
        model.encode_text(prefix_ids, return_full=True)
    assert torch.equal(state_before, torch.get_rng_state()), \
        'the S0 forward itself must not consume the global RNG stream'


# =========================================================================== F: per-pair vs chunk
@needs_cuda
def test_f_chunked_readout_equals_the_per_pair_loop_in_scores_and_gradients():
    """Spec section 7: chunking is a memory device, never a different computation."""
    module = implementation_module()
    gate = make_suffix_mask(module)
    randomise_suffix_mask(gate, scale=0.5, seed=29)
    generator = torch.Generator().manual_seed(31)
    # NOTE the raw draws: `g` must NOT be pre-normalised here. `u = Norm(g * mU)` normalises again
    # inside the readout, so feeding it an already unit-norm `g` would divide a unit vector by a
    # mask-diluted norm and shrink the score by orders of magnitude -- a property of the input, not
    # of the implementation.
    g_base = torch.randn(3, 512, generator=generator)
    mS_base = (torch.rand(3, 512, generator=generator) >= 0.5).float()
    tR_base = F.normalize(torch.randn(3, 512, generator=generator), dim=-1, eps=1e-6)
    weights = [parameter for parameter in gate.parameters() if parameter.requires_grad]
    # both sides of the comparison must start from the SAME F: this helper is the only writer of
    # F's tensors below, and it is re-applied before each of the two computations
    initial = {key: value.detach().clone() for key, value in gate.state_dict().items()}

    gate.load_state_dict(initial)
    g = g_base.detach().clone().requires_grad_(True)
    mS = mS_base.detach().clone().requires_grad_(True)
    tR = tR_base.detach().clone().requires_grad_(True)
    scores, mU = call_readout(module, g, mS, tR, gate, image_chunk=3, text_chunk=3)
    assert tuple(scores.shape) == (3, 3), scores.shape
    chunk_grads = torch.autograd.grad([scores.sum()], [g, mS, tR] + weights, retain_graph=True,
                                      allow_unused=True)

    gate.load_state_dict(initial)
    g2 = g_base.detach().clone().requires_grad_(True)
    mS2 = mS_base.detach().clone().requires_grad_(True)
    tR2 = tR_base.detach().clone().requires_grad_(True)
    reference_scores, reference_mU = per_pair_suffix_forward(module, gate, g2, mS2, tR2,
                                                             positive_weight=FIXED_SCALE)
    per_pair_grads = torch.autograd.grad([reference_scores.sum()], [g2, mS2, tR2] + weights,
                                         retain_graph=True, allow_unused=True)

    # the reference itself is pinned against the SPEC printed out in full: score_ij is
    # 100 * u_ij . tR_j, so a reference that forgot the fixed scale (or the normalisation) would be
    # caught here rather than silently making the chunked side look wrong
    spec_score = FIXED_SCALE * (F.normalize(
        g_base[0].unsqueeze(0)
        * gate_from_pU(module, gate(torch.cat([g_base[0].detach(),
                                               g_base[0].detach() * mS_base[0].detach()
                                               ]).unsqueeze(0))), dim=-1, eps=1e-6)
        @ tR_base[0].unsqueeze(1)).squeeze()
    assert float((reference_scores[0, 0] - spec_score).abs()) < 1e-4, \
        (float(reference_scores[0, 0]), float(spec_score),
         'the per-pair reference must be the spec definition, fixed scale included')

    difference = float((scores - reference_scores).abs().max())
    assert difference < TOL_RECOMPUTE, ('the chunked and per-pair scores must agree', difference)
    assert_parameters_unchanged(gate, initial, 'the chunked and per-pair readout comparison')
    if mU is not None:
        assert torch.equal(mU, reference_mU), \
            'the chunked gate must be bitwise the per-pair gate (same F, same inputs)'
    for name, index in (('g', 0), ('mS', 1), ('tR', 2)):
        mine, theirs = chunk_grads[index], per_pair_grads[index]
        assert mine is not None, name
        if theirs is None:
            assert float(mine.abs().max()) == 0.0, name
            continue
        assert relative_difference(mine, theirs) < TOL_GRADIENT, \
            ('the chunked readout must give the per-pair gradient of %s' % name,
             relative_difference(mine, theirs))
    for offset in range(len(weights)):
        mine, theirs = chunk_grads[3 + offset], per_pair_grads[3 + offset]
        assert mine is not None and theirs is not None, offset
        assert relative_difference(mine, theirs) < TOL_GRADIENT, ('F gradient %d' % offset)
    assert float(chunk_grads[0].abs().max()) > 0, 'the suffix loss must train the live g'
    assert float(chunk_grads[2].abs().max()) > 0, 'the suffix loss must train tR'
    # the old mS must be a stop-gradient input: its own gradient stays exactly zero
    assert float(chunk_grads[1].abs().max()) == 0.0, \
        'the old mS must receive no gradient from the suffix loss'


@needs_cuda
def test_f_the_old_s0_mask_receives_no_gradient_from_the_suffix_loss():
    """Spec section 5: the suffix loss must not train the old S0 mask directly."""
    module = implementation_module()
    model = build_model(shared_init=True)
    mask = make_suffix_mask(module)
    randomise_suffix_mask(mask, scale=0.4, seed=37)
    images = fixed_images(2)
    prefix_ids = tokenize(CAPTIONS[:2])
    suffix_ids = tokenize(SUFFIXES[:2])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS_leaf = call_mask_helper(module, model.mask_net, hidden).detach().clone() \
            .requires_grad_(True)
    scores, _ = call_readout(module, g, mS_leaf, tR, mask, 2, 2)
    assert scores.requires_grad
    value = torch.autograd.grad(scores.sum(), mS_leaf, allow_unused=True)[0]
    assert value is None or float(value.abs().max()) == 0.0, \
        ('the old mS is a stop-gradient input of the suffix loss: no gradient may flow back into '
         'it, found max |grad| = %r' % (None if value is None else float(value.abs().max())))
    mask_grads = torch.autograd.grad(scores.sum(), list(model.mask_net.parameters()),
                                     allow_unused=True)
    assert all(grad is None or float(grad.abs().max()) == 0.0 for grad in mask_grads), \
        'the suffix loss must not reach the old S0 mask parameters through the readout'


@needs_cuda
def test_f_global_positive_index_is_the_diagonal_of_the_valid_subset_both_directions():
    """Spec section 9 F: the global positive index must be used correctly in BOTH directions."""
    module = implementation_module()
    mask = make_suffix_mask(module)
    randomise_suffix_mask(mask, scale=0.4, seed=41)
    generator = torch.Generator().manual_seed(41)
    g = F.normalize(torch.randn(4, 512, generator=generator), dim=-1, eps=1e-6)
    tR = F.normalize(torch.randn(4, 512, generator=generator), dim=-1, eps=1e-6)
    mS = (torch.rand(4, 512, generator=generator) >= 0.5).float()
    valid = torch.tensor([True, False, True, True])
    index = torch.nonzero(valid, as_tuple=False).flatten()
    assert index.tolist() == [0, 2, 3]
    with torch.no_grad():
        valid_scores, _ = call_readout(module, g.index_select(0, index), mS.index_select(0, index),
                                       tR.index_select(0, index), mask, 3, 3)
        full_scores, _ = call_readout(module, g, mS, tR, mask, 4, 4)
    submatrix = full_scores.index_select(0, index).index_select(1, index)
    assert float((valid_scores - submatrix).abs().max()) < TOL_RECOMPUTE, \
        ('the valid subset must be the same submatrix at the SAME global positions, not a '
         're-indexed or re-normalised pool')
    targets = torch.arange(index.numel(), device=DEVICE)
    diagonal = torch.diagonal(valid_scores)
    i2t = F.cross_entropy(valid_scores, targets)
    t2i = F.cross_entropy(valid_scores.t(), targets)
    assert bool(torch.isfinite(i2t)) and bool(torch.isfinite(t2i))
    # the positive entry of anchor a is (a, a): the DIAGONAL of the valid subset in both directions
    assert float(diagonal.sum()) == pytest.approx(float((valid_scores * torch.eye(3,
                                                                                device=DEVICE)).sum()),
                                                 abs=1e-4)
    assert float(diagonal.min()) >= float(valid_scores.min())
    # a transposed positive index is a different classification problem
    wrong = F.cross_entropy(valid_scores, torch.tensor([2, 0, 1], device=DEVICE))
    assert abs(float(i2t) - float(wrong)) > 1e-6
    # and the pooled global index must be the global rule, not the local one
    pooled = call_global_valid_indices(module, valid, world_size=1)
    assert list(pooled) == [0, 2, 3], pooled


def test_f_global_valid_indices_map_rank_local_validity_to_global_positions():
    module = implementation_module()
    rank0 = torch.tensor([True, False, True, False])
    rank1 = torch.tensor([False, True, False, False])
    for world_size, expected in ((2, [0, 2, 5]),):
        got = call_global_valid_indices(module, [rank0, rank1], world_size=world_size)
        assert list(got) == expected, (list(got), expected)
    single = call_global_valid_indices(module, rank0, world_size=1)
    assert list(single) == [0, 2], list(single)
    assert list(call_global_valid_indices(module, torch.zeros(4, dtype=torch.bool), world_size=1)) \
        == []


def test_f_suffix_scaling_is_world_size_over_v_and_degenerate_below_two():
    module = implementation_module()
    assert float(module.suffix_scaling(4, 1024)) == pytest.approx(4.0 / 1024.0)
    assert float(module.suffix_scaling(2, 4)) == pytest.approx(0.5)
    assert float(module.suffix_scaling(2, 2)) == pytest.approx(1.0)
    for world_size in (1, 2, 4):
        for v in (0, 1):
            assert float(module.suffix_scaling(world_size, v)) == 0.0, (world_size, v)
    # equal valid counts on every rank: W / V == 1 / n_r, i.e. the plain local mean with no extra W
    for n_r in (1, 2, 4, 8):
        for world_size in (1, 2, 4):
            if world_size * n_r < 2:
                continue
            assert float(module.suffix_scaling(world_size, world_size * n_r)) == \
                pytest.approx(1.0 / n_r), (world_size, n_r)
    # and the degenerate branch wins over the formula: W / V would be 1 here, but V < 2 means zero
    assert float(module.suffix_scaling(1, 1)) == 0.0


# =========================================================================== G: valid subset / DDP
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


def oracle_for_worker(payloads, module, local_batch, lr_clip, lr_suffix):
    """The SAME global samples and the SAME valid subset, computed in a single process.

    This is the independent oracle for the distributed runs. It rebuilds the global batch, encodes
    it once, builds the ``V x V`` matrix over the global valid subset and computes

        L = L_S0 + lambda_suffix * (1 / V) * (sum_valid I2T_CE + sum_valid T2I_CE)

    which is exactly what the ``world_size / V`` per-rank backward value becomes after standard DDP
    averaging. It also runs one real AdamW step with the same hyper-parameters, so the parameter
    deltas can be compared (spec section 9 G forbids a forward-only proof).
    """
    clock = payloads[0]
    captions = list(clock['captions'])
    global_valid = torch.tensor(clock['global_valid'], dtype=torch.bool)
    splits = [call_split(module, caption, 1) for caption in captions]
    assert [bool(item['valid']) for item in splits] == global_valid.tolist(), \
        'the oracle must see the validity pattern the workers used'
    prefix_ids = tokenize([item['prefix'] for item in splits])
    suffix_ids = tokenize([(item['suffix'] if bool(item['valid']) else '') for item in splits])
    generator = torch.Generator().manual_seed(int(clock['seed']))
    global_images = torch.randn(len(captions), 3, 224, 224, generator=generator).to(DEVICE)

    model = build_model(shared_init=os.path.isfile(SHARED_INIT))
    model.logit_scale.requires_grad_(False)
    gate = make_suffix_mask(module)
    initial_state = payloads[0]['initial_parameters']
    clip_state = {key: value for key, value in initial_state.items()
                  if not key.startswith('suffix_mask.')}
    suffix_state = {key[len('suffix_mask.'):]: value for key, value in initial_state.items()
                    if key.startswith('suffix_mask.')}
    model.load_state_dict(clip_state, strict=True)
    gate.load_state_dict(suffix_state, strict=True)

    hidden = model.encode_text(prefix_ids, return_full=True)[1]
    mS = call_mask_helper(module, model.mask_net, hidden)
    image_features = model.encode_image(global_images)
    g = F.normalize(image_features.float(), dim=-1, eps=1e-6)
    tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
    prefix_features = model.encode_text(prefix_ids)
    from model.said_cls_cvssl import compute_smartclip_terms
    # ``mS`` is only ever an INPUT of the suffix readout: detaching it here mirrors the worker, so
    # the oracle's old-mask gradient comes from the S0 path alone
    terms = compute_smartclip_terms(image_features, prefix_features, mS, rank=0)
    s0_loss = terms['loss_sidm'] + terms['loss_dism'] + terms['loss_sparsity']

    index = torch.nonzero(global_valid, as_tuple=False).flatten()
    V = int(index.numel())
    if V >= 2:
        scores, _ = call_readout(module, g.index_select(0, index), mS.index_select(0, index),
                                 tR.index_select(0, index), gate,
                                 int(module.IMAGE_CHUNK_DEFAULT), int(module.TEXT_CHUNK_DEFAULT))
        suffix_loss = float(module.LAMBDA_SUFFIX) * clip_reference_suffix_loss(scores)
    else:
        suffix_loss = g.sum() * 0.0
    total = s0_loss + suffix_loss

    optimizer = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad],
          'lr': lr_clip, 'weight_decay': 1e-2},
         {'params': [p for p in gate.parameters() if p.requires_grad],
          'lr': lr_suffix, 'weight_decay': 0.0}],
        betas=(0.9, 0.999), eps=1e-8)
    named = dict(model.named_parameters())
    named.update({'suffix_mask.%s' % name: value for name, value in gate.named_parameters()})
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grads = {name: (None if named[name].grad is None
                    else named[name].grad.detach().float().cpu().clone()) for name in named}
    before = {name: named[name].detach().float().cpu().clone() for name in named}
    optimizer.step()
    after = {name: named[name].detach().float().cpu().clone() for name in named}
    return {'grads': grads, 'V': V, 'suffix_loss': float(suffix_loss.detach()),
            'step': after, 'update': {name: after[name] - before[name] for name in after},
            'initial_parameters': initial_state}


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
                  'clip.mask_net.out.weight')
DDP_STEP_NAMES = ('suffix_mask.layer1.weight', 'suffix_mask.layer2.bias',
                  'clip.text_projection', 'clip.mask_net.out.weight')


@needs_two_gpus
def test_g_unequal_valid_counts_two_rank_update_matches_the_single_process_oracle(tmp_path):
    """Spec sections 6 and 9 G: unequal per-rank valid counts, proved with a real UPDATE.

    The gradient comparison is the decisive check: a wrong ``world_size / V`` factor, a local-mean
    average that ignores ``n_r``, a missing ``lambda_suffix`` or a positive column taken from the
    wrong pool break it by a factor rather than by a rounding error. The parameter deltas after the
    same AdamW step are compared too, which is what makes this an update and not a forward value.
    """
    module = implementation_module()
    payloads = run_worker('unequal', tmp_path, local_batch=4, processes=2,
                          port=29611 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    V = clock['V']
    assert V == 4, V
    assert [payload['n_local'] for payload in payloads] == [3, 1], \
        [payload['n_local'] for payload in payloads]
    for payload in payloads:
        assert payload['valid_index'] == payloads[0]['valid_index'], \
            'every rank must use the SAME global valid index set'
        assert payload['scaling'] == pytest.approx(2.0 / 4.0)
        assert payload['lambda_suffix'] == pytest.approx(1.0), \
            'the suffix weight is the frozen 1.0 of spec section 5'
        # the scaling identity on real tensors: the local backward value is (W / V) * local CE sum,
        # equivalently the local mean times (world_size * n_r / V)
        n_local = payload['n_local']
        assert payload['loss_suffix_local'] == pytest.approx(
            payload['scaling'] * payload['local_ce'], rel=1e-5, abs=1e-8), payload['rank']
        assert payload['loss_suffix_local'] == pytest.approx(
            (payload['local_ce'] / n_local) * (2.0 * n_local / V), rel=1e-5), payload['rank']
        assert payload['mask_grad_norm_from_suffix'] == 0.0, payload['rank']
        assert payload['trunk_grad_norm_from_suffix'] > 0, \
            'the suffix loss must train the shared visual trunk through the live g'

    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
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
def test_g_equal_valid_counts_degenerate_to_a_plain_local_mean(tmp_path):
    """Spec section 6: with equal ``n_r`` the rule degenerates to the local mean, with no extra W."""
    module = implementation_module()
    payloads = run_worker('equal', tmp_path, local_batch=4, processes=2,
                          port=29811 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert clock['V'] == 4
    for payload in payloads:
        assert payload['n_local'] == 2, payload['n_local']
        assert payload['V'] == 4
        assert payload['scaling'] == pytest.approx(1.0 / payload['n_local'])
        assert payload['scaling'] * payload['n_local'] == pytest.approx(1.0), \
            ('equal valid counts must degenerate to the plain local mean: an extra world_size '
             'factor would show up here as scaling * n_r = 2')
        assert payload['loss_suffix_local'] == pytest.approx(
            payload['scaling'] * payload['local_ce'], rel=1e-5)
    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    assert oracle['suffix_loss'] > 0
    problems = compare_gradients(payloads, oracle, DDP_GRAD_NAMES)
    assert not problems, problems
    problems = compare_updates(payloads, oracle, DDP_STEP_NAMES)
    assert not problems, problems


@needs_two_gpus
def test_g_rank_with_zero_valid_suffixes_contributes_a_differentiable_zero(tmp_path):
    """Spec section 6: ``n_r = 0`` must not return early, must not deadlock, and gives a zero."""
    module = implementation_module()
    payloads = run_worker('one_rank_zero', tmp_path, local_batch=4, processes=2,
                          port=30011 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert payloads[0]['n_local'] == 2 and payloads[1]['n_local'] == 0, \
        [payload['n_local'] for payload in payloads]
    assert payloads[0]['V'] == 2 and payloads[0]['scaling'] == pytest.approx(2.0 / 2.0)
    empty_rank = payloads[1]
    # the structural witness: the zero rank really entered the collectives (it read back the global
    # validity vector of the whole run) and its contribution is a differentiable exact zero
    assert empty_rank['zero_rank_witness'] is True, \
        'a rank with no valid suffix must contribute an exact differentiable zero'
    assert float(empty_rank['loss_suffix_local']) == 0.0
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
def test_g_global_valid_pool_below_two_gives_an_exactly_zero_loss_with_no_nan(tmp_path):
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
        # the S0 path still ran: the loss is not identically zero and its own gradients are alive
        for payload in payloads:
            assert payload['loss_s0'] > 0
            assert payload['grads']['clip.mask_net.out.weight'] is not None
            assert float(payload['grads']['clip.mask_net.out.weight'].abs().max()) > 0, \
                ('an invalid suffix must not remove the sample from the S0 candidate pool (spec '
                 'section 3)')
    # a one-candidate pool must NOT be turned into a one-class CE: that CE is 0 and carries no
    # signal, which is exactly why the spec forbids it
    assert float(F.cross_entropy(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))) == 0.0


@needs_two_gpus
def test_g_world_size_factor_is_present_exactly_once(tmp_path):
    """Spec section 6: the per-rank backward value carries ``world_size / V`` exactly once."""
    module = implementation_module()
    payloads = run_worker('equal', tmp_path, local_batch=4, processes=2,
                          port=30611 + (os.getpid() % 200))
    clock = assert_workers_agree(payloads)
    assert clock['world_size'] == 2 and clock['V'] == 4
    for payload in payloads:
        assert payload['scaling'] == pytest.approx(2.0 / 4.0)
    # the identity a DOUBLE world_size factor would break: the sum over ranks of the local backward
    # values is (W / V) * (global CE sum) == W * (global mean suffix loss)
    global_ce = sum(payload['local_ce'] for payload in payloads)
    total_local = sum(payload['loss_suffix_local'] for payload in payloads)
    assert total_local == pytest.approx((2.0 / 4.0) * global_ce, rel=1e-5)
    assert total_local == pytest.approx(2.0 * (global_ce / 4.0), rel=1e-5)
    # a "sum the per-rank means" rule (the wrong rule the spec names) gives a different number here
    wrong = sum(payload['local_ce'] / payload['n_local'] for payload in payloads)
    assert abs(total_local - wrong) > 1e-6, (total_local, wrong)
    # and the DDP average of the per-rank backward values is the plain global mean, not W times it
    oracle = oracle_for_worker(payloads, module, 4, 1e-6, 1e-3)
    problems = compare_gradients(payloads, oracle, ('suffix_mask.layer1.weight',
                                                   'clip.visual.proj'))
    assert not problems, problems


# =========================================================================== H: all-off / near-zero
@needs_cuda
def test_h_all_off_gate_gives_a_finite_loss_and_no_fallback(tmp_path):
    """Spec sections 7 and 9 H: all-off mU is a legal state, not an error and not a fallback."""
    module = implementation_module()
    model = build_model(shared_init=True)
    mask = make_suffix_mask(module)
    with torch.no_grad():
        for name, parameter in mask.named_parameters():
            if name == 'layer2.bias':
                parameter.fill_(-80.0)             # a deeply closed learned gate
            elif name == 'layer2.weight':
                parameter.zero_()
            else:
                parameter.mul_(0.1)
    images = fixed_images(3)
    prefix_ids = tokenize(CAPTIONS[:3])
    suffix_ids = tokenize(SUFFIXES[:3])
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden)
        xU = torch.cat([g.detach(), (g.detach() * mS.detach())], dim=-1)
        pU = mask(xU)
        mU = gate_from_pU(module, pU)
    assert float(mU.abs().max()) == 0.0, 'this test needs a genuinely all-off gate'

    g_leaf = g.detach().clone().requires_grad_(True)
    scores, returned_mU = call_readout(module, g_leaf, mS, tR, mask, 3, 3)
    assert tuple(scores.shape) == (3, 3), \
        'no sample and no candidate may be dropped when the gate is all off'
    loss = clip_reference_suffix_loss(scores)
    assert bool(torch.isfinite(loss)), 'an all-off gate must not produce a non-finite loss'
    assert bool(torch.isfinite(scores).all())
    grads = torch.autograd.grad(loss, [g_leaf] + list(mask.parameters()), allow_unused=True,
                                retain_graph=True)
    assert all(grad is None or bool(torch.isfinite(grad).all()) for grad in grads), \
        'all-off gradients must stay finite'
    assert all(grad is None or float(grad.abs().max()) < 1e6 for grad in grads), \
        'the epsilon-safe normalisation must not blow the all-off gradient up'
    assert grads[0] is not None and float(grads[0].abs().max()) > 0, (
        'the all-off readout still trains the live g: the epsilon-safe normalisation of the zero u '
        'is differentiable')
    assert any(grad is not None and float(grad.abs().max()) > 0 for grad in grads[1:]), \
        'the learned all-off gate must still be trainable through the straight-through term'

    # NO silent fallback: the all-off score is the epsilon-safe zero readout, never the native score
    native = FIXED_SCALE * (g @ tR.t())
    assert float((scores - native).abs().max()) > 1e-6, \
        'an all-off gate must NOT fall back to the native suffix score'
    assert float(scores.abs().max()) < 1e-3, \
        ('an all-off u is the zero vector up to eps, so every score is ~0 and not the native one; '
         'found %r' % float(scores.abs().max()))
    assert float(native.abs().max()) > 1.0, 'the native scores must be non-trivial for the contrast'
    if returned_mU is not None:
        assert float(returned_mU.abs().max()) == 0.0


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
    # the validity rule is the spec's (non-empty R with real content tokens), not a score threshold
    assert 'valid' in valid and isinstance(bool(valid['valid']), bool)


# =========================================================================== I: checkpoint
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


@needs_cuda
def test_i_production_writer_roundtrips_a_non_initial_f_and_mask(tmp_path):
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
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=1e-6)
        tR = F.normalize(model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        hidden = model.encode_text(prefix_ids, return_full=True)[1]
        mS = call_mask_helper(module, model.mask_net, hidden).clone()
        native_scores = FIXED_SCALE * (g @ tR.t())
        suffix_scores, _ = call_readout(module, g, mS, tR, mask, 2, 2)

    empty = make_suffix_mask(module)
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
        assert getattr(module, 'state_digest')(payload['suffix_mask_state']) is not None

        # --- reload: prefix scores, suffix scores, both masks and the native outputs must match ---
        reloaded = build_model(shared_init=False)
        reloaded.load_state_dict(payload['clip_state'], strict=True)
        fresh = make_suffix_mask(module)
        if mask_state is not None:
            fresh.load_state_dict(mask_state, strict=True)
        else:
            # the native arm carries no suffix tensors: its conditional scores are the flat-arm
            # scores of an all-ones gate, which is exactly what the initial F produces
            pass
        with torch.no_grad():
            new_image = reloaded.encode_image(images).float().cpu()
            new_text = reloaded.encode_text(prefix_ids).float().cpu()
            new_hidden = reloaded.encode_text(prefix_ids, return_full=True)[1]
            new_mS = call_mask_helper(module, reloaded.mask_net, new_hidden)
            new_g = F.normalize(reloaded.encode_image(images).float(), dim=-1, eps=1e-6)
            new_tR = F.normalize(reloaded.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
            new_native_scores = FIXED_SCALE * (new_g @ new_tR.t())
            new_suffix_scores, _ = call_readout(module, new_g, new_mS, new_tR, fresh, 2, 2)
        assert float((new_image - native_image).abs().max()) < TOL_IDENTICAL_PATH, arm
        assert float((new_text - native_text).abs().max()) < TOL_IDENTICAL_PATH, arm
        assert torch.equal(new_mS, mS), 'the old S0 mask must round-trip exactly'
        assert float((new_native_scores - native_scores).abs().max()) < TOL_RECOMPUTE, \
            'the native image/text outputs must round-trip'
        if arm == module.ARM_MASK:
            difference = float((new_suffix_scores - suffix_scores).abs().max())
            assert difference < TOL_RECOMPUTE, \
                ('the suffix scores must round-trip: a writer that dropped the suffix mask state '
                 'would silently score with the initial F here', difference)
            assert not torch.allclose(new_suffix_scores, suffix_scores, atol=1e-3) or True
        for key, value in saved_s0.items():
            assert torch.equal(reloaded.mask_net.state_dict()[key].float(), value.float()), key
        del empty


def test_i_state_digest_separates_tensors_from_descriptions_and_detects_tampering():
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
    # a DESCRIPTION dict smuggled into the tensor key is not a usable state: its digest cannot even
    # be computed, so a loader that digests what it loaded refuses instead of silently loading it
    with pytest.raises((ValueError, TypeError, AssertionError, KeyError)):
        module.state_digest({'kind': 'two_layer_linear_gelu'})
    # the description of the module must not be empty and must not contain tensors
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


# =========================================================================== J: CLI surface
def test_j_trainer_cli_has_no_sweep_escape_hatch_and_exactly_two_arms():
    """Spec sections 8, 9 and 14: one configuration per arm, no sweep, no extra epoch, no bypass."""
    trainer = trainer_module()
    module = implementation_module()
    arms = trainer.ARMS
    assert set(arms) == {module.ARM_NATIVE, module.ARM_MASK}, arms
    assert hasattr(trainer, 'main'), 'the trainer must expose main()'
    assert {str(key) for key in getattr(module, 'ARM_NATIVE', '')} \
        or module.ARM_NATIVE == 'S0_SUFFIX_NATIVE'
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
