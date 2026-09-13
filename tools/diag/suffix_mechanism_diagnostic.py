"""Read-only mechanism diagnostic for S0-Suffix v0.1 (the new prefix-conditioned suffix gate F).

    python tools/diag/suffix_mechanism_diagnostic.py --build-manifest cohort.json
    python tools/diag/suffix_mechanism_diagnostic.py \
        --checkpoint runs_salu/s0_suffix_mask/S0_SUFFIX_MASK_step000500.pt \
        --output diag/suffix_mechanism_step000500.json \
        --run_dir runs_salu/s0_suffix_mask

WHAT THIS MEASURES
------------------
Four scoring modes, all on ONE fixed cohort manifest and ONE fixed prefix/suffix split, all scored
on the SAME candidate protocol as training (column ``j`` carries ``(P_j, R_j)``: image ``i`` is
scored against the suffix text ``R_j`` using the S0 mask produced by the prefix ``P_j`` -- never a
single "correct" prefix shared by a whole row):

* ``NORMAL``            real images + each candidate's own prefix mask -> ``mU`` -> suffix scores;
* ``U_ALL_ONES``        ``mU`` forced to all ones; asserted to reproduce the native suffix scores
                        ``100 * dot(Norm(g), tR)`` on the very same pool (a self-check of this tool,
                        so it is an assertion, not a reported number only);
* ``PREFIX_SHUFFLED``   only the SOURCE of the candidate prefix masks is permuted (a fixed,
                        recorded derangement); images, suffix texts and the label diagonal are
                        untouched and ``mU`` is regenerated;
* ``IMAGE_SHUFFLED``    the input images are permuted and ``g``, ``rS`` and ``mU`` are all
                        recomputed from the permuted images. This is a VISUAL-DEPENDENCE
                        INTERVENTION, never an evaluation set: the true pair is still ``(i, i)``.

WHAT THIS IS NOT
----------------
The cohort comes from the SAME ShareGPT4V manifest as training (selection by ``json_index``), the
training pipeline has NO holdout mechanism at all, and this cohort is therefore a FIXED DIAGNOSTIC
SPLIT INSIDE THE TRAINING DISTRIBUTION. It is NOT an external validation set and nothing here may be
read as generalization. The ``reading`` block of the output repeats this and forbids the readings the
spec forbids (different masks do not demonstrate semantic decoupling; a shuffled-condition drop does
not prove pure visual complementarity; ``U_ALL_ONES`` close to ``NORMAL`` means the new selection
adds little).

READ-ONLY BY CONSTRUCTION
-------------------------
No optimizer is imported or constructed, no parameter is updated, the CLIP and suffix-mask state
digests are recomputed and asserted UNCHANGED after every forward pass, the tool refuses any
``--output`` inside ``--run_dir``, and it writes its own status JSON (it never touches another run's
``run_status.json``). It ends by printing exactly one machine-readable JSON line.

The mask arm is the only arm with a suffix gate, so the checkpoint must be ``S0_SUFFIX_MASK`` at
``completed_steps == 500``; anything else is refused loudly rather than analysed.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (REPO, os.path.join(REPO, 'model')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
from model import said_prefix_suffix as sps                                     # noqa: E402

# --------------------------------------------------------------------------- experiment identity
OBJECTIVE = sps.OBJECTIVE
ARM_NATIVE = sps.ARM_NATIVE
ARM_MASK = sps.ARM_MASK
PROBE_NAME = 'suffix_mechanism_diagnostic'

# --------------------------------------------------------------------------- fixed diagnostic cohort
DEFAULT_SOURCE_MANIFEST = '/root/SAID/outputs/validation/sharegpt4v1k_manifest.json'
DEFAULT_MANIFEST_PATH = os.path.join(REPO, 'outputs', 'validation',
                                     'suffix_mech_cohort64_manifest.json')
DEFAULT_CHECKPOINT = os.path.join(REPO, 'runs_salu', 's0_suffix_mask',
                                  'S0_SUFFIX_MASK_step000500.pt')
DEFAULT_DATA_ROOT = os.environ.get('SHARE4V_DATA_ROOT', '/root/datasets/ShareGPT4V')
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT
DEFAULT_COHORT_SIZE = 64
DEFAULT_EXPECT_STEPS = 500
DEFAULT_BASE_MODEL = sps.BASE_MODEL
COHORT_SCHEMA_VERSION = 1
COHORT_PROTOCOL = 's0-suffix-fixed-diagnostic-cohort-v1'
SOURCE_CAPTION_SHAPES = ('dict', 'list', 'str')

# the split of this diagnostic: K is FIXED (no draw), the training random K is untouched
SPLIT_RULE = ('the experiment rule of model/said_prefix_suffix.py: caption.replace("\\n", " ") then '
              'sentences = caption.split(". "); prefix P = ". ".join(sentences[:K]); suffix R = '
              '". ".join(s for s in sentences[K:last_nonempty_index + 1] if s is non-empty), where '
              'last_nonempty_index is the index of the LAST non-empty fragment of the ORIGINAL '
              'sentence list, so a trailing empty fragment can never pull the last real sentence '
              'back in')
K_RULE = ('K = max(1, L // 2) with L = last_nonempty_index + 1 for that caption; this K is FIXED '
          'for the diagnostic and is NOT the training draw K = rng.randint(1, len(sentences)): the '
          'diagnostic split changes no training random stream and is not a training sample of the '
          'K distribution')
VALIDITY_RULE = ('R is non-empty AND tokenises to at least one content token (tokens before the real '
                 'EOT, content_count = argmax(token ids) - 1 > 0, the project convention; token id '
                 '!= 0 is never used)')
COHORT_HONESTY = (
    'This cohort is drawn from the SAME ShareGPT4V manifest as training (selection by json_index), '
    'the training pipeline train/said_cvssl_data.py has NO holdout mechanism at all, and the '
    'manifest is therefore a FIXED DIAGNOSTIC SPLIT INSIDE THE TRAINING DISTRIBUTION. It is NOT an '
    'external validation set, it is not held out, and no number computed on it may be reported as '
    'generalization. It exists only so that the four modes below are scored on one frozen pool with '
    'one frozen P/R split.')
SOURCE_MANIFEST_NOTE = ('the caption source is the source file itself, read at build time and '
                        'included per sample; no caption is resampled, and the source sha256 is '
                        're-checked whenever the stored manifest is reused')
CAPTION_PREFERENCE = ('full_dense', 'caption', 'text', 'detailed', 'long', 'full')

# the permutation seeds are part of the frozen configuration of this diagnostic
SPLIT_SEED = 20260914
PREFIX_PERMUTATION_SEED = 20260915
IMAGE_PERMUTATION_SEED = 20260916
PERMUTATION_RULE = ('a deterministic DERANGEMENT of the 0..n-1 index list, drawn with '
                    'random.Random(seed).shuffle inside a saved/restored global RNG context; the '
                    'explicit index list is recorded in the output. Fixed points are rejected so '
                    'that the shuffled mode can never silently degenerate into the normal one; if '
                    'the retry budget is exhausted a cyclic shift is used, which is a derangement '
                    'for n >= 2 as well')

MODES = ('NORMAL', 'U_ALL_ONES', 'PREFIX_SHUFFLED', 'IMAGE_SHUFFLED')
PERMUTED_MODES = ('PREFIX_SHUFFLED', 'IMAGE_SHUFFLED')
PROBABILITY_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
ALL_ONES_TOLERANCE = 1e-3
MASK_PROBABILITY_THRESHOLD = 0.5
NEAR_THRESHOLD_DISTANCE = 0.05
MAX_FIXED_POINTS_IN_PERMUTATION = 0
CORRELATION_EPS = 1e-12

ABSOLUTE_READING = (
    'Different masks do not by themselves demonstrate semantic decoupling: two masks can differ '
    'elementwise while the readouts they produce stay dominated by the shared image direction, and '
    'nothing here measures what the masked-out coordinates mean semantically.')
SHUFFLE_READING = (
    'Degradation under a shuffled prefix source or under shuffled images is NOT proof of pure '
    'visual complementarity: it shows only that the scores depend on that input. A loss under '
    'shuffling is consistent with the gate reading the conditioning signal, with the readout being '
    'dominated by the full-image term, or with plain overfitting to the training pairing, and this '
    'diagnostic cannot separate those explanations.')
U_ALL_ONES_READING = (
    'A U_ALL_ONES result close to NORMAL would mean the new selection adds little: forcing the gate '
    'fully open recovers almost the same ranking, so the gate is not the source of any difference '
    'between the two arms.')
COHORT_READING = (
    'This is a fixed diagnostic split inside the training distribution (same ShareGPT4V manifest as '
    'training, selection by json_index, no holdout anywhere in the training pipeline). It is NOT an '
    'external validation set and no generalization claim may be based on it.')
SMALL_POOL_READING = (
    'The cohort holds at most 64 samples: differences of one or two ranks are within the noise of '
    'such a pool, no significance test is reported here, and no claim of a stable effect may be made '
    'from these numbers.')


# --------------------------------------------------------------------------- small helpers
def sha256_of(path, block=1 << 20):
    """sha256 of a whole file, streamed (never loading the file into memory)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo=None):
    """The repository HEAD, or ``'unknown'``: provenance only, never fatal."""
    try:
        result = subprocess.run(['git', '-C', repo or REPO, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return result.stdout.strip() or 'unknown'


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    """Write a JSON file atomically (temporary file plus ``os.replace``)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def check_output_path(output, run_dir):
    """Refuse any output path inside the run directory: this tool never writes into a run."""
    if run_dir is None:
        return None
    resolved_run = os.path.abspath(run_dir)
    resolved_output = os.path.abspath(output)
    if resolved_output == resolved_run or resolved_output.startswith(resolved_run + os.sep):
        raise SystemExit('refusing to write %s: the output must NOT live inside --run_dir %s (this '
                         'tool is read-only with respect to the run directory, and a diagnostic must '
                         'never overwrite another run\'s status file)'
                         % (resolved_output, resolved_run))
    return resolved_run


def check_status_path(status_output, run_dir):
    """The status file is this tool's OWN file: a run's ``run_status.json`` is never touched."""
    resolved = check_output_path(status_output, run_dir)
    if resolved is not None:
        forbidden = os.path.join(resolved, 'run_status.json')
        if os.path.abspath(status_output) == forbidden:
            raise SystemExit('refusing to write %s: that is the run\'s own run_status.json. This '
                             'diagnostic writes its own separate status file (use --status-output to '
                             'choose it)' % forbidden)
    return resolved


def tokenize_texts(texts, device='cpu'):
    """The project tokenisation: ``longclip.tokenize(..., truncate=True)`` at the 248-token context.

    The trainer tokenises the prefix and the suffix exactly this way (two independent forwards), so
    the diagnostic reproduces the training token stream rather than inventing one.
    """
    return longclip.tokenize([str(text) for text in texts], truncate=True).to(device)


def document_caption(record, index, wanted, allow_fallback):
    """The caption text of one source-manifest record, for whichever shape the file actually has.

    Handles ``captions`` as a dict (preferred keys for the full/detailed caption), as a list (first
    entry) and as a plain string, plus a top-level ``caption``/``text`` fallback. The shape actually
    found is reported back so the manifest can document it instead of assuming it.
    """
    value, keys = None, None
    if isinstance(record, dict):
        value = record.get('captions', None)
        keys = record.get('caption_keys', None)
    if isinstance(value, dict):
        for name in CAPTION_PREFERENCE:
            if name in value and isinstance(value[name], str) and value[name].strip():
                return value[name], 'dict', name
        if not allow_fallback:
            raise SystemExit('record %d: the captions dict has none of the expected keys %r (present: '
                             '%r); refusing to guess a caption'
                             % (index, list(CAPTION_PREFERENCE), sorted(str(key) for key in value)))
        for name in sorted(value):
            item = value[name]
            if isinstance(item, str) and item.strip():
                return item, 'dict_fallback_key', str(name)
        raise SystemExit('record %d: the captions dict carries no non-empty string caption' % index)
    if isinstance(value, list):
        for position, item in enumerate(value):
            if isinstance(item, str) and item.strip():
                return item, 'list', position
        raise SystemExit('record %d: the captions list carries no non-empty string caption' % index)
    if isinstance(value, str) and value.strip():
        return value, 'str', None
    if isinstance(record, dict):
        for name in ('caption', 'text', 'value'):
            item = record.get(name, None)
            if isinstance(item, str) and item.strip():
                return item, 'record_field', name
    if keys is not None:
        raise SystemExit('record %d: captions has an unsupported shape (caption_keys=%r)'
                         % (index, keys))
    raise SystemExit('record %d: the record carries no usable caption under captions/caption/text '
                     '(keys: %r)' % (index, sorted(str(key) for key in record)
                                     if isinstance(record, dict) else type(record).__name__))


def document_image_path(record, index):
    """``image_path`` (relative to the image root) of one source-manifest record."""
    if not isinstance(record, dict):
        raise SystemExit('record %d is not a dict (%s)' % (index, type(record).__name__))
    for name in ('image_path', 'image', 'image_name'):
        item = record.get(name, None)
        if isinstance(item, str) and item.strip():
            return item
    raise SystemExit('record %d carries no image path under image_path/image (keys: %r)'
                     % (index, sorted(str(key) for key in record)))


def document_json_index(record, index):
    """``json_index`` of one source-manifest record, or the manifest position when it is absent."""
    if isinstance(record, dict) and 'json_index' in record:
        try:
            return int(record['json_index']), True
        except (TypeError, ValueError):
            raise SystemExit('record %d has a non-integer json_index=%r'
                             % (index, record['json_index']))
    return int(index), False


def build_manifest(source_path, cohort_size):
    """Build the fixed diagnostic cohort manifest from the source ShareGPT4V manifest.

    The first ``cohort_size`` samples **in the source file's own order** whose suffix ``R`` is
    non-empty and tokenises to at least one content token are kept. If fewer qualify, the actual
    number is recorded; nothing is topped up from the training set and no sample is replaced.
    """
    if not os.path.isfile(source_path):
        raise SystemExit('source manifest not found: %s' % source_path)
    with open(source_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get('samples', None)
        if records is None:
            for name in ('records', 'data', 'manifest'):
                if isinstance(payload.get(name), list):
                    records = payload[name]
                    break
        header = {key: payload[key] for key in sorted(payload) if key != 'samples'}
    else:
        records, header = payload, {}
    if not isinstance(records, list) or not records:
        raise SystemExit('%s has no list of samples (got %s); refusing to guess the manifest shape'
                         % (source_path, type(records).__name__))
    header_note = ('the source manifest header is copied for provenance only; the caption text used '
                   'here always comes from the source records themselves')
    shapes, samples, rejected = {}, [], []
    seen_index = set()
    for index, record in enumerate(records):
        if len(samples) >= int(cohort_size):
            break
        caption, shape, shape_key = document_caption(record, index, CAPTION_PREFERENCE,
                                                     allow_fallback=False)
        shapes[shape] = shapes.get(shape, 0) + 1
        json_index, index_was_present = document_json_index(record, index)
        image_path = document_image_path(record, index)
        flattened = str(caption).replace('\n', ' ')
        sentences = flattened.split('. ')
        last_nonempty_index = -1
        for position, sentence in enumerate(sentences):
            if sentence.strip():
                last_nonempty_index = position
        l_value = last_nonempty_index + 1
        k = max(1, l_value // 2)
        if not 1 <= k <= len(sentences):
            raise SystemExit('record %d: the fixed K rule gives K=%d, outside [1, %d]; the rule '
                             'K = max(1, L // 2) with L = last_nonempty_index + 1 cannot be applied '
                             'to this caption' % (index, k, len(sentences)))
        split = sps.split_prefix_suffix(flattened, k)
        if split['last_nonempty_index'] != last_nonempty_index:
            raise SystemExit('record %d: the module split reports last_nonempty_index=%d while this '
                             'tool computed %d' % (index, split['last_nonempty_index'],
                                                   last_nonempty_index))
        prefix_ids = tokenize_texts([split['prefix']])
        suffix_ids = tokenize_texts([split['suffix']])
        prefix_content = int(sps.content_token_counts(prefix_ids)[0])
        suffix_content = int(sps.content_token_counts(suffix_ids)[0])
        prefix_raw = int(len(longclip._tokenizer.encode(split['prefix'])) + 2)
        suffix_raw = int(len(longclip._tokenizer.encode(split['suffix'])) + 2)
        valid = bool(sps.suffix_validity([split['suffix']], [suffix_content])[0])
        if not valid:
            rejected.append({'position_in_source': int(index), 'json_index': int(json_index),
                             'image_path': image_path, 'K': int(k),
                             'n_sentences': int(split['n_sentences']),
                             'suffix': split['suffix'],
                             'reason': ('empty suffix' if not split['suffix'].strip()
                                        else 'suffix has no content token after tokenisation')})
            continue
        if json_index in seen_index:
            raise SystemExit('the source manifest repeats json_index=%d: the cohort may not hold the '
                             'same source sample twice' % json_index)
        seen_index.add(json_index)
        samples.append({
            'cohort_index': len(samples),
            'position_in_source': int(index),
            'json_index': int(json_index),
            'json_index_was_present_in_source': bool(index_was_present),
            'image_path': image_path,
            'K': int(k),
            'L_last_nonempty_plus_one': int(l_value),
            'n_sentences': int(split['n_sentences']),
            'last_nonempty_index': int(last_nonempty_index),
            'has_trailing_empty_fragment': bool(split['has_trailing_empty_fragment']),
            'prefix': split['prefix'],
            'suffix': split['suffix'],
            'prefix_content_tokens': prefix_content,
            'suffix_content_tokens': suffix_content,
            'prefix_raw_token_count': prefix_raw,
            'suffix_raw_token_count': suffix_raw,
            'prefix_truncated': bool(prefix_raw > sps.TOKENIZER_CONTEXT),
            'suffix_truncated': bool(suffix_raw > sps.TOKENIZER_CONTEXT),
            'caption_shape': shape,
            'caption_shape_key': shape_key,
            'valid': True,
        })

    manifest = {
        'schema_version': COHORT_SCHEMA_VERSION,
        'protocol': COHORT_PROTOCOL,
        'probe': PROBE_NAME,
        'objective': OBJECTIVE,
        'arm_this_cohort_belongs_to': ARM_MASK,
        'source_manifest_path': os.path.abspath(source_path),
        'source_manifest_sha256': sha256_of(source_path),
        'source_manifest_record_count': len(records),
        'source_manifest_header': header,
        'source_manifest_header_note': header_note,
        'source_manifest_caption_shapes_observed': shapes,
        'source_manifest_caption_keys_used': sorted({str(sample['caption_shape_key'])
                                                     for sample in samples}),
        'source_manifest_caption_key_preference': list(CAPTION_PREFERENCE),
        'source_manifest_note': SOURCE_MANIFEST_NOTE,
        'image_root': DEFAULT_IMAGE_ROOT,
        'cohort_requested_size': int(cohort_size),
        'cohort_size': len(samples),
        'cohort_full_size_reached': bool(len(samples) == int(cohort_size)),
        'cohort_selection': ('the first %d samples IN THE SOURCE FILE\'S OWN ORDER whose suffix R is '
                             'non-empty and tokenises to at least one content token; nothing is '
                             'topped up from the training set and no rejected sample is replaced'
                             % int(cohort_size)),
        'split_rule': SPLIT_RULE,
        'k_rule': K_RULE,
        'validity_rule': VALIDITY_RULE,
        'split_seed': SPLIT_SEED,
        'split_seed_note': ('recorded for provenance only: the K rule of this cohort is deterministic '
                            'and draws nothing, so no RNG consumed any seed here'),
        'tokenizer': {'name': 'longclip SimpleTokenizer via longclip.tokenize',
                      'context_length': sps.TOKENIZER_CONTEXT, 'truncate': True,
                      'content_token_rule': 'argmax(token ids) - 1, clamped at 0 (the real EOT '
                                            'position), never token_id != 0 counting',
                      'raw_token_count_rule': 'len(longclip._tokenizer.encode(text)) + 2, the SOT '
                                              'and EOT included (the project audit convention)'},
        'candidate_protocol': ('column j is (P_j, R_j): image i is scored against the suffix text R_j '
                               'using the mask generated from the prefix P_j, positives and '
                               'negatives alike; the true pair of the fixed cohort is (i, i)'),
        'true_pair_is_diagonal': True,
        'rejected_samples_before_the_cut': rejected,
        'honest_statement': COHORT_HONESTY,
        'samples': samples,
        'built_utc': utc_now(),
    }
    return manifest


def load_cohort_manifest(path, expect_source):
    """Load an existing cohort manifest and re-check that its source file is still the same bytes."""
    if not os.path.isfile(path):
        raise SystemExit('cohort manifest not found: %s (run once with --build-manifest %s)'
                         % (path, path))
    with open(path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get('protocol') != COHORT_PROTOCOL:
        raise SystemExit('%s is not a %s cohort manifest (protocol=%r)'
                         % (path, COHORT_PROTOCOL, manifest.get('protocol')
                            if isinstance(manifest, dict) else None))
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples:
        raise SystemExit('%s carries no samples' % path)
    source = manifest.get('source_manifest_path')
    if expect_source is not None and os.path.abspath(expect_source) != os.path.abspath(source or ''):
        raise SystemExit('the cohort manifest was built from %s but --source_manifest says %s; a '
                         'cohort built from a different source file must never be reused silently'
                         % (source, os.path.abspath(expect_source)))
    if not source or not os.path.isfile(source):
        raise SystemExit('the cohort manifest records source_manifest_path=%r, which does not exist'
                         % (source,))
    digest = sha256_of(source)
    if digest != manifest.get('source_manifest_sha256'):
        raise SystemExit('the source manifest %s changed: recorded sha256=%s recomputed=%s; rebuild '
                         'the cohort explicitly with --build-manifest'
                         % (source, manifest.get('source_manifest_sha256'), digest))
    for position, sample in enumerate(samples):
        for key in ('json_index', 'image_path', 'K', 'prefix', 'suffix'):
            if key not in sample:
                raise SystemExit('cohort sample %d has no %r field' % (position, key))
        if not str(sample['suffix']).strip():
            raise SystemExit('cohort sample %d carries an EMPTY suffix: the frozen cohort must only '
                             'hold samples with a non-empty suffix, so this manifest is not the one '
                             'this tool built' % position)
    indices = [int(sample['json_index']) for sample in samples]
    if len(set(indices)) != len(indices):
        raise SystemExit('the cohort manifest repeats a json_index (%d samples, %d distinct): a '
                         'repeated source sample would silently duplicate a candidate'
                         % (len(indices), len(set(indices))))
    return manifest


# --------------------------------------------------------------------------- permutations
def derangement(indices, seed):
    """A deterministic derangement of ``indices`` plus the seed context it was drawn in.

    Fixed points are rejected (a shuffled mode that silently keeps an element in place would blur
    into the normal mode); after a bounded number of retries a cyclic shift is used, which is also a
    derangement for ``n >= 2``. The explicit list is returned so the output records the exact
    permutation that was applied.
    """
    order = [int(value) for value in indices]
    count = len(order)
    if count < 2:
        raise SystemExit('a permutation of %d element(s) cannot be a derangement' % count)
    state = random.getstate()
    try:
        generator = random.Random(int(seed))
        drawn = None
        for _attempt in range(64):
            candidate = list(order)
            generator.shuffle(candidate)
            if all(left != right for left, right in zip(candidate, order)):
                drawn = candidate
                break
        if drawn is None:
            drawn = order[1:] + order[:1]
    finally:
        random.setstate(state)
    fixed = [left for left, right in zip(order, drawn) if left == right]
    if fixed:
        raise SystemExit('the permutation keeps %d element(s) in place: %r' % (len(fixed), fixed))
    return drawn


# --------------------------------------------------------------------------- scoring helpers
def rank_of_true_pair(rows, labels):
    """Per-row rank of the true column: the number of candidates strictly above the positive.

    Ties are counted as candidates above, so the reported rank is never optimistic; the rank is 0
    exactly when the positive is the strict argmax, which is what makes ``R@1`` the mean of
    ``rank == 0``.
    """
    rows = rows.float()
    positive = rows.gather(1, labels.reshape(-1, 1))
    return (rows > positive).sum(dim=1).to(torch.long)


def summarise_ranks(ranks):
    """Mean / median / top-1 statistics of one rank list."""
    values = ranks.to(torch.float32)
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'min': int(ranks.min()),
        'max': int(ranks.max()),
        'R@1': float((ranks == 0).float().mean()),
        'top5_fraction': float((ranks < 5).float().mean()),
        'anchor_count': int(ranks.numel()),
    }


def score_table(scores):
    """I2T / T2I statistics of one score matrix, with the true pair on the diagonal.

    ``scores[i, j]`` is image ``i`` against the candidate ``j``; the fixed cohort's true pair is
    ``i == j`` in BOTH directions (image ``i`` belongs to ``(P_i, R_i)``, and the text of candidate
    ``j`` belongs to image ``j``). The T2I direction therefore transposes the matrix and uses the
    same diagonal labels -- the permutation modes permute the INPUT of the scores, never the labels.
    """
    count = int(scores.shape[0])
    if int(scores.shape[1]) != count:
        raise SystemExit('the score matrix is %s, not square: the fixed cohort scores every candidate '
                         'against every image' % (tuple(scores.shape),))
    identity = torch.arange(count, device=scores.device, dtype=torch.long)
    out = {'score_shape': [count, count], 'true_pair_is_diagonal': True,
           'positive_index_I2T': 'row i, column i', 'positive_index_T2I': 'transposed row j, '
                                                                        'column j'}
    for direction, rows in (('I2T', scores.float()), ('T2I', scores.float().t().contiguous())):
        statistics = sps.block_statistics(rows, identity)
        ranks = rank_of_true_pair(rows, identity)
        out[direction] = {
            'R@1': float((ranks == 0).float().mean()),
            'positive_mean': float(statistics['positive'].mean()),
            'strongest_negative_mean': float(statistics['strongest_negative'].mean()),
            'max_margin_mean': float(statistics['max_margin'].mean()),
            'lse_margin_mean': float(statistics['lse_margin'].mean()),
            'lse_margin_min': float(statistics['lse_margin'].min()),
            'positive_win_fraction': float((statistics['max_margin'] > 0).float().mean()),
            'rank': summarise_ranks(ranks),
            'per_anchor_rank': [int(value) for value in ranks.tolist()],
            'lse_margin_definition': 'positive - logsumexp(valid negatives only); logsumexp(all) - '
                                     'positive is the cross entropy and is never reported here',
            'anchor_count': count,
            'candidate_count': count,
            'valid_negatives_per_anchor': count - 1,
            'fixed_scale': sps.SMARTCLIP_FIXED_SCALE,
        }
    return out


def native_scores_from_features(g_unit, t_r):
    """``QN[i, j] = 100 * dot(Norm(g_i), tR_j)``: the native suffix scores of the SAME pool.

    This is the reference the ``U_ALL_ONES`` mode must reproduce, computed from the already
    normalised features so the comparison isolates the gate and nothing else.
    """
    rows = sps.suffix_readout(t_r, eps=sps.NORM_EPS)
    return sps.SMARTCLIP_FIXED_SCALE * (g_unit.float() @ rows.float().t())


def mask_pair_statistics(pU, mU, mask_s):
    """Gate statistics of one mode: keep rates, all-on/all-off fractions, probability distribution.

    ``pU[i, j]`` and ``mU[i, j]`` belong to the pair (image ``i``, candidate ``j``); the per-row and
    per-column means are kept separately because the two aggregate different things (a row averages
    over candidate prefixes, a column averages over images).
    """
    probability = pU.detach().float()
    hard = (mU.detach().float() >= MASK_PROBABILITY_THRESHOLD)
    pair_keep = hard.float().mean(dim=1)
    column_keep = mask_s.detach().float().mean(dim=1)
    quantile_axis = torch.tensor(PROBABILITY_QUANTILES, device=probability.device,
                                 dtype=torch.float32)
    return {
        'pair_count': int(probability.shape[0] * probability.shape[1]),
        'keep_rate_overall': float(hard.float().mean()),
        'keep_rate_per_pair_mean': float(pair_keep.mean()),
        'keep_rate_per_pair_min': float(pair_keep.min()),
        'keep_rate_per_pair_max': float(pair_keep.max()),
        'keep_rate_per_pair': [float(value) for value in pair_keep.tolist()],
        'keep_rate_per_candidate_mean': float(hard.float().mean(dim=0).mean()),
        'keep_rate_per_candidate': [float(value) for value in hard.float().mean(dim=0).tolist()],
        'all_on_fraction': float(hard.all(dim=1).float().mean()),
        'all_off_fraction': float((~hard).all(dim=1).float().mean()),
        'all_on_pair_count': int(hard.all(dim=1).sum()),
        'all_off_pair_count': int((~hard).any(dim=1).logical_not().sum()),
        'probability_mean': float(probability.mean()),
        'probability_std': float(probability.std(unbiased=False)),
        'probability_min': float(probability.min()),
        'probability_max': float(probability.max()),
        'probability_quantiles_1_5_25_50_75_95_99': [
            float(value) for value in torch.quantile(probability, quantile_axis)],
        'probability_quantile_levels': list(PROBABILITY_QUANTILES),
        'near_threshold_distance': NEAR_THRESHOLD_DISTANCE,
        'near_threshold_fraction': float(
            ((probability - MASK_PROBABILITY_THRESHOLD).abs() < NEAR_THRESHOLD_DISTANCE)
            .float().mean()),
        'threshold': MASK_PROBABILITY_THRESHOLD,
        'mask_s_keep_rate_per_candidate_mean': float(column_keep.mean()),
        'mask_s_keep_rate_overall': float(mask_s.detach().float().mean()),
        'gate_kind': 'mU = (pU >= 0.5) + (pU - pU.detach()), hard forward with the soft slope',
    }


def overlap_statistics(pU, mask_s):
    """Overlap of the OLD S0 mask ``mS`` and the NEW gate ``mU`` (both are per-candidate here).

    Two resolutions are reported because ``mU`` lives per (image, candidate) pair while ``mS`` lives
    per candidate prefix: the per-column mean of the hard gate (the comparable aggregate) and the
    hard-diagonal elements (the pairs whose image really owns that prefix). Elementwise agreement,
    Pearson correlation and the cosine of the two derived readouts are computed on both.
    """
    hard = (pU.detach().float() >= MASK_PROBABILITY_THRESHOLD).float()
    mask = mask_s.detach().float()
    count = int(hard.shape[0])
    diagonal = torch.arange(count, device=hard.device, dtype=torch.long)
    column_mean = hard.mean(dim=0)
    diagonal_hard = hard[diagonal, diagonal]

    def pair(left, right, scope):
        left = left.reshape(-1).float()
        right = right.reshape(-1).float()
        left_centred = left - left.mean()
        right_centred = right - right.mean()
        denominator = (left_centred.norm() * right_centred.norm()).clamp_min(CORRELATION_EPS)
        return {
            'scope': scope,
            'elementwise_agreement_fraction': float((left >= MASK_PROBABILITY_THRESHOLD)
                                                    .eq(right >= MASK_PROBABILITY_THRESHOLD)
                                                    .float().mean()),
            'mean_absolute_difference': float((left - right).abs().mean()),
            'pearson_correlation': float((left_centred * right_centred).sum() / denominator),
            'mU_keep_rate': float((left >= MASK_PROBABILITY_THRESHOLD).float().mean()),
            'mS_keep_rate': float((right >= MASK_PROBABILITY_THRESHOLD).float().mean()),
        }

    return {
        'column_mean_mU_versus_mS': pair(column_mean, mask.mean(dim=0),
                                         'per candidate prefix: the hard gate averaged over the '
                                         'image rows against mS of the same prefix'),
        'diagonal_mU_versus_mS': pair(diagonal_hard, mask[diagonal],
                                      'the true pair (i, i) of the fixed cohort only'),
        'hard_gate_fraction_of_mS_kept': float(
            (hard * (mask >= MASK_PROBABILITY_THRESHOLD).float().unsqueeze(0)).sum()
            / (mask >= MASK_PROBABILITY_THRESHOLD).float().sum().clamp_min(1.0)),
        'hard_gate_fraction_kept_outside_mS': float(
            (hard * (mask < MASK_PROBABILITY_THRESHOLD).float().unsqueeze(0)).sum()
            / (mask < MASK_PROBABILITY_THRESHOLD).float().sum().clamp_min(1.0)),
    }


def readout_cosine(g_unit, mU):
    """Cosine of the conditional suffix readout against the native one, in chunks.

    ``Norm(g * mU)`` is compared with ``Norm(g)``; because both factors are unit length, the cosine
    is ``100 * dot(Norm(g_i * mU_ij), g_i) / 100``, so the chunked computation never materialises a
    second ``[n, n, 1024]`` tensor. Row ``i`` holds the candidate prefixes for the image ``i``.
    """
    unit = g_unit.detach().float()
    count = int(unit.shape[0])
    cosine = torch.zeros((count, count), dtype=torch.float32, device=unit.device)
    scale = float(sps.SMARTCLIP_FIXED_SCALE)
    for start in range(0, count, sps.IMAGE_CHUNK_DEFAULT):
        stop = min(start + sps.IMAGE_CHUNK_DEFAULT, count)
        gated = unit[start:stop].unsqueeze(1) * mU.detach().float()[start:stop]
        rows = sps.suffix_readout(gated, eps=sps.NORM_EPS)
        cosine[start:stop] = (rows * unit[start:stop].unsqueeze(1)).sum(dim=-1) * scale / scale
    diagonal = torch.arange(count, device=unit.device, dtype=torch.long)
    return {
        'cosine_Norm_gmU_versus_Norm_g_pair_mean': float(cosine.mean()),
        'cosine_Norm_gmU_versus_Norm_g_diagonal_mean': float(cosine[diagonal, diagonal].mean()),
        'cosine_Norm_gmU_versus_Norm_g_diagonal_per_pair': [
            float(value) for value in cosine[diagonal, diagonal].tolist()],
        'definition': 'cos(Norm(g_i * mU_ij), Norm(g_i)); 1.0 means the gate changed nothing in '
                      'direction, which is what an all-ones gate gives',
        'scope': 'all (image, candidate) pairs of the fixed cohort, plus the (i, i) diagonal',
    }


def forced_mask_readout(g_unit, t_r, mask):
    """``suffix_readout_scores`` with a mask identically one, so ``mU`` is one by construction.

    The same ``suffix_readout_scores`` entry point as the normal mode is used, so the ``U_ALL_ONES``
    self-check compares two runs of the production readout rather than two implementations of it.
    """
    count = int(g_unit.shape[0])
    ones = torch.ones((count, int(g_unit.shape[1])), dtype=torch.float32, device=g_unit.device)
    return sps.suffix_readout_scores(g_unit, ones, mask, t_r,
                                     image_chunk=sps.IMAGE_CHUNK_DEFAULT,
                                     text_chunk=sps.TEXT_CHUNK_DEFAULT,
                                     eps=sps.NORM_EPS, want_statistics=True)


# --------------------------------------------------------------------------- model / checkpoint
def load_clip(base_model, clip_state, device):
    """A freshly built native CLIP with the checkpoint's clip state loaded STRICTLY."""
    model, preprocess = longclip.load_from_clip(base_model, device='cpu', download_root=None,
                                                args=argparse.Namespace())
    native_keys = set(model.state_dict())
    extra = sorted(set(clip_state) - native_keys)
    absent = sorted(native_keys - set(clip_state))
    if extra or absent:
        raise SystemExit('the checkpoint clip state is not the native CLIP state: keys the model '
                         'cannot accept %r, model tensors missing from the state %r'
                         % (extra[:5], absent[:5]))
    model.load_state_dict(clip_state, strict=True)
    return model.to(device).eval(), preprocess


def verify_checkpoint(payload, checkpoint, expect_steps):
    """Refuse anything that is not a finished S0_SUFFIX_MASK step-500 checkpoint with real F tensors."""
    if not isinstance(payload, dict):
        raise SystemExit('checkpoint %s is not a dict payload (got %s)'
                         % (checkpoint, type(payload).__name__))
    problems = []
    if payload.get('objective') != OBJECTIVE:
        problems.append('objective=%r expected %r' % (payload.get('objective'), OBJECTIVE))
    if payload.get('arm') != ARM_MASK:
        problems.append('arm=%r expected %r: only the mask arm owns the suffix gate F, so only it '
                        'can be analysed here' % (payload.get('arm'), ARM_MASK))
    try:
        completed = int(payload.get('completed_steps'))
    except (TypeError, ValueError):
        completed = None
    if completed is None:
        problems.append('completed_steps=%r is not an integer' % (payload.get('completed_steps'),))
    elif completed != int(expect_steps):
        problems.append('completed_steps=%r expected %d: only the finished run may be analysed'
                        % (payload.get('completed_steps'), int(expect_steps)))
    if 'clip_state' not in payload:
        problems.append('the clip_state key is missing entirely')
    elif not isinstance(payload['clip_state'], dict) or not payload['clip_state']:
        problems.append('clip_state is empty or not a dict (%s)'
                        % type(payload['clip_state']).__name__)
    mask_state = payload.get('suffix_mask_state', 'MISSING')
    if mask_state == 'MISSING':
        problems.append('the suffix_mask_state key is MISSING entirely: without the gate tensors the '
                        'checkpoint cannot be the mask arm and F cannot be rebuilt')
    elif not isinstance(mask_state, dict) or not mask_state:
        problems.append('suffix_mask_state must be a non-empty tensor dict for %s, found %s'
                        % (ARM_MASK, type(mask_state).__name__))
    else:
        non_tensors = sorted(str(key) for key, value in mask_state.items()
                             if not torch.is_tensor(value))
        if non_tensors:
            problems.append('suffix_mask_state carries non-tensor entries %r: it must be the gate '
                            'TENSOR dict, never a description dict' % non_tensors[:6])
        else:
            expected = {'layer1.weight': (512, 1024), 'layer1.bias': (512,),
                        'layer2.weight': (512, 512), 'layer2.bias': (512,)}
            got = {str(key).split('.', 1)[-1]: tuple(value.shape)
                   for key, value in mask_state.items()}
            absent = sorted(name for name in expected if name not in got)
            wrong = {name: list(got[name]) for name in sorted(got)
                     if name in expected and expected[name] != got[name]}
            if absent or wrong:
                problems.append('suffix_mask_state does not match the SuffixMask module: absent %r, '
                                'wrong shapes %r' % (absent, wrong))
    if problems:
        raise SystemExit('checkpoint verification failed for %s: %s' % (checkpoint, problems))
    return completed


def resolve_digest(payload, key):
    """The digest the checkpoint header records for a state, or ``None`` when it records none."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


# --------------------------------------------------------------------------- the four modes
def run_mode(mode, count, objective, mask, features, mask_s, permutations, image_chunk, text_chunk):
    """Score the fixed cohort in ONE mode, on the fixed P/R split and the fixed label diagonal.

    Every mode returns the same structure: the score matrix, the gate probabilities and the hard
    gate actually used, the ``mS`` that produced ``rS``, and masks that say WHICH source produced
    each score and each gate. The label diagonal is never permuted: a permutation moves the INPUT of
    the scores (the images, or the source of the candidate prefix masks), never the truth.
    """
    identity = list(range(count))
    if mode in ('NORMAL', 'U_ALL_ONES'):
        source_index, image_index = identity, identity
    elif mode == 'PREFIX_SHUFFLED':
        source_index, image_index = list(permutations['PREFIX_SHUFFLED']), identity
    elif mode == 'IMAGE_SHUFFLED':
        source_index, image_index = identity, list(permutations['IMAGE_SHUFFLED'])
    else:
        raise SystemExit('unknown mode %r' % (mode,))

    image_tensor = features['images']
    with torch.no_grad():
        if mode == 'IMAGE_SHUFFLED':
            # g, rS and mU are ALL recomputed from the permuted images
            order = torch.as_tensor(image_index, dtype=torch.long, device=image_tensor.device)
            g_live = objective.encode_images(image_tensor.index_select(0, order))
            g_unit = sps.normalize_features(g_live, eps=sps.NORM_EPS)
        else:
            g_unit = features['g_unit']
        mask_s_used = mask_s
        text_used = features['t_r']

        if mode == 'U_ALL_ONES':
            readout = forced_mask_readout(g_unit, text_used, mask)
            scores = readout['scores']
            probability = readout['pU'].detach().float()
            mask_matrix = torch.ones_like(probability)
            forced = True
        else:
            source_tensor = torch.as_tensor(source_index, dtype=torch.long, device=mask_s.device)
            mask_s_effective = mask_s_used.index_select(0, source_tensor)
            readout = sps.suffix_readout_scores(g_unit, mask_s_effective, mask, text_used,
                                                image_chunk=int(image_chunk),
                                                text_chunk=int(text_chunk),
                                                eps=sps.NORM_EPS, want_statistics=True)
            scores = readout['scores']
            probability = readout['pU'].detach().float()
            mask_matrix = sps.SuffixMaskGate.gate_from_pU(probability).detach().float()
            forced = False

    return {
        'mode': mode,
        'scores': scores,
        'pU': probability,
        'mU': mask_matrix,
        'mask_s_used': mask_s_used,
        'mask_source_index': source_index,
        'image_index': image_index,
        'forced_all_ones': forced,
        'gated_norm_ratio_mean': float(readout['gated_norm_ratio'].mean()),
        'near_zero_readout_norm_count': int(readout['near_zero_readout_norm_count']),
        'readout_pair_count': int(readout['readout_pair_count']),
    }


def mode_payload(result, features, u_all_ones_check):
    """Everything the output records for one mode: scores, ranks, gate statistics, overlap, cosines."""
    scores = result['scores'].detach().float()
    count = int(scores.shape[0])
    payload = {
        'scores_100x': [[float(value) for value in row] for row in scores.tolist()],
        'statistics': score_table(scores),
        'mask_pair_statistics': mask_pair_statistics(result['pU'], result['mU'],
                                                     result['mask_s_used']),
        'mask_overlap_old_mS_vs_new_mU': overlap_statistics(result['pU'], result['mask_s_used']),
        'readout_cosine': readout_cosine(features['g_unit'], result['mU']),
        'gated_readout_norm_ratio_mean': result['gated_norm_ratio_mean'],
        'near_zero_readout_norm_count': result['near_zero_readout_norm_count'],
        'readout_pair_count': result['readout_pair_count'],
        'indexing': {
            'candidate_source_index': list(result['mask_source_index']),
            'image_source_index': list(result['image_index']),
            'candidate_source_is_identity': bool(result['mask_source_index']
                                                 == list(range(count))),
            'image_source_is_identity': bool(result['image_index'] == list(range(count))),
            'label_diagonal': 'the true pair is (i, i) in BOTH directions for every mode: no '
                              'permutation ever moves the labels',
            'prefix_mask_source': ('the mask of candidate column j comes from cohort sample '
                                   'candidate_source_index[j]' if not result['forced_all_ones']
                                   else 'mU is forced to one, so no prefix mask reaches the readout'),
            'candidate_rule': ('column j carries (P_j, R_j): image i is scored against R_j with the '
                               'mask generated from P_j; a row never shares one "correct" prefix'),
        },
        'forced_all_ones': bool(result['forced_all_ones']),
    }
    if result['forced_all_ones']:
        payload['u_all_ones_self_check'] = u_all_ones_check
    return payload


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build-manifest', default=None, metavar='OUT_JSON',
                        help='build the FIXED diagnostic cohort manifest from the source manifest '
                             'and write it to this path, then exit (no checkpoint is needed)')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST_PATH,
                        help='the fixed cohort manifest: built by --build-manifest, reused (after a '
                             'source sha256 re-check) on every later run (default %s)'
                             % DEFAULT_MANIFEST_PATH)
    parser.add_argument('--source_manifest', default=DEFAULT_SOURCE_MANIFEST,
                        help='the ShareGPT4V validation manifest the cohort is drawn from; its '
                             'sha256 is recorded and re-checked (default %s)'
                             % DEFAULT_SOURCE_MANIFEST)
    parser.add_argument('--cohort-size', type=int, default=DEFAULT_COHORT_SIZE,
                        help='how many samples the cohort holds at most: the first N in the source '
                             'file order with a non-empty suffix; if fewer qualify the actual number '
                             'is recorded and NOTHING is topped up from the training set '
                             '(default %d)' % DEFAULT_COHORT_SIZE)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='the mask arm checkpoint: S0_SUFFIX_MASK at completed_steps==500, with '
                             'clip_state and a real suffix_mask_state (default %s)'
                             % DEFAULT_CHECKPOINT)
    parser.add_argument('--expect-steps', type=int, default=DEFAULT_EXPECT_STEPS,
                        help='required completed_steps; %d by default' % DEFAULT_EXPECT_STEPS)
    parser.add_argument('--output', default=None,
                        help='JSON file holding the full diagnostic record; must NOT be inside '
                             '--run_dir (default: <repo>/outputs/diag/'
                             'suffix_mechanism_step%06d.json)')
    parser.add_argument('--status-output', default=None,
                        help='the tool\'s OWN status JSON; must NOT be inside --run_dir and is never '
                             'another run\'s run_status.json (default: <output> with a _status '
                             'suffix)')
    parser.add_argument('--run_dir', default=None,
                        help='the training run directory, used only for the read-only assertion and '
                             'provenance; the tool never writes into it')
    parser.add_argument('--image_root', default=DEFAULT_IMAGE_ROOT,
                        help='root the manifest image_path values are relative to (default %s)'
                             % DEFAULT_IMAGE_ROOT)
    parser.add_argument('--base_model', default=DEFAULT_BASE_MODEL,
                        help='native CLIP architecture to rebuild (default %s)' % DEFAULT_BASE_MODEL)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--weight-dtype', default='float32', choices=['float32'],
                        help='the master-weight dtype; only float32 exists in this experiment')
    parser.add_argument('--image_chunk', type=int, default=sps.IMAGE_CHUNK_DEFAULT)
    parser.add_argument('--text_chunk', type=int, default=sps.TEXT_CHUNK_DEFAULT)
    parser.add_argument('--modes', default=','.join(MODES),
                        help='comma-separated subset of %s; the U_ALL_ONES self-check runs whenever '
                             'U_ALL_ONES is selected (default: all four)' % ','.join(MODES))
    parser.add_argument('--prefix-permutation-seed', type=int, default=PREFIX_PERMUTATION_SEED)
    parser.add_argument('--image-permutation-seed', type=int, default=IMAGE_PERMUTATION_SEED)
    parser.add_argument('--all-ones-tolerance', type=float, default=ALL_ONES_TOLERANCE,
                        help='tight tolerance of the U_ALL_ONES self-check: the forced-all-ones '
                             'scores must equal 100*dot(Norm(g), tR) within this value and the '
                             'check is ASSERTED, not merely reported (default %r)'
                             % ALL_ONES_TOLERANCE)
    parser.add_argument('--allow-fewer-samples', action='store_true',
                        help='accept a cohort smaller than --cohort-size (the count is always '
                             'recorded; a smaller cohort is never padded or topped up)')
    args = parser.parse_args()

    # ---- the manifest capability: build and stop, or load and re-verify ----------------------
    manifest_path = os.path.abspath(args.manifest)
    if args.build_manifest:
        target = os.path.abspath(args.build_manifest)
        if args.run_dir is not None:
            check_output_path(target, args.run_dir)
        built = build_manifest(os.path.abspath(args.source_manifest), int(args.cohort_size))
        write_json(target, built)
        print('BUILT_COHORT_MANIFEST %s samples=%d requested=%d full=%s'
              % (target, built['cohort_size'], built['cohort_requested_size'],
                 built['cohort_full_size_reached']))
        print('SUFFIX_MECHANISM_DIAGNOSTIC ' + json.dumps({
            'probe': PROBE_NAME, 'phase': 'build_manifest', 'read_only': True,
            'new_optimizer_updates': 0, 'objective': OBJECTIVE,
            'manifest_path': target, 'manifest_sha256': sha256_of(target),
            'source_manifest_path': built['source_manifest_path'],
            'source_manifest_sha256': built['source_manifest_sha256'],
            'cohort_size': built['cohort_size'],
            'cohort_requested_size': built['cohort_requested_size'],
            'cohort_full_size_reached': built['cohort_full_size_reached'],
            'k_rule': built['k_rule'], 'split_rule': built['split_rule'],
            'caption_shapes_observed': built['source_manifest_caption_shapes_observed'],
            'honest_statement': built['honest_statement'], 'utc': utc_now()}, sort_keys=True))
        return

    if not os.path.isfile(manifest_path):
        raise SystemExit('the cohort manifest %s does not exist: build it once with '
                         '--build-manifest %s (the cohort must be frozen before any scoring, so it is '
                         'never built implicitly here)' % (manifest_path, manifest_path))
    manifest = load_cohort_manifest(manifest_path, args.source_manifest)
    cohort = manifest['samples']
    count = len(cohort)
    if count < 2:
        raise SystemExit('the cohort holds %d sample(s): at least 2 are needed for a retrieval '
                         'diagnostic (a single candidate has no negatives)' % count)
    if count < int(args.cohort_size) and not args.allow_fewer_samples:
        raise SystemExit('the cohort holds %d samples but --cohort-size is %d; pass '
                         '--allow-fewer-samples to accept the recorded count explicitly'
                         % (count, int(args.cohort_size)))
    for position, sample in enumerate(cohort):
        path = os.path.join(args.image_root, str(sample['image_path']))
        if not os.path.isfile(path):
            raise SystemExit('cohort sample %d image is missing: %s (image root %s)'
                             % (position, path, args.image_root))

    modes = [item.strip().upper() for item in str(args.modes).split(',') if item.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise SystemExit('unknown mode(s) %r; the modes are %r' % (unknown, list(MODES)))
    if not modes:
        raise SystemExit('--modes selected nothing')

    # ---- the checkpoint ----------------------------------------------------------------------
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit('checkpoint not found: %s' % checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completed = verify_checkpoint(payload, checkpoint, args.expect_steps)
    clip_state = payload['clip_state']
    mask_state = payload['suffix_mask_state']

    output = os.path.abspath(args.output) if args.output else os.path.join(
        REPO, 'outputs', 'diag', 'suffix_mechanism_step%06d.json' % completed)
    run_dir = check_output_path(output, args.run_dir)
    status_output = os.path.abspath(args.status_output) if args.status_output else \
        os.path.splitext(output)[0] + '_status.json'
    check_status_path(status_output, args.run_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    device = torch.device(args.device)
    use_autocast = device.type == 'cuda'
    started = time.time()

    model, _preprocess = load_clip(args.base_model, clip_state, device)
    mask = sps.SuffixMask().to(device)
    missing, unexpected = mask.load_state_dict(mask_state, strict=True)
    if list(missing) or list(unexpected):
        raise SystemExit('the suffix mask state did not load cleanly: missing=%r unexpected=%r'
                         % (list(missing), list(unexpected)))
    mask.eval()
    objective = sps.SaidPrefixSuffixObjective(model, rank=0, arm=ARM_MASK, suffix_mask=mask,
                                              image_chunk=int(args.image_chunk),
                                              text_chunk=int(args.text_chunk), world_size=1,
                                              gather_validity=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in mask.parameters():
        parameter.requires_grad_(False)

    clip_digest_before = sps.state_digest(model.state_dict())
    mask_digest_before = sps.state_digest(mask.state_dict())
    stored_clip_digest = resolve_digest(payload, 'clip_state_digest')
    stored_mask_digest = resolve_digest(payload, 'suffix_mask_state_digest')

    # ---- the fixed cohort through the encoder, ONCE ------------------------------------------
    from PIL import Image

    caption_texts = [str(sample['prefix']) + '. ' + str(sample['suffix']) for sample in cohort]
    prefix_texts = [str(sample['prefix']) for sample in cohort]
    suffix_texts = [str(sample['suffix']) for sample in cohort]
    prefix_ids = tokenize_texts(prefix_texts, device)
    suffix_ids = tokenize_texts(suffix_texts, device)
    prefix_content = sps.content_token_counts(prefix_ids)
    suffix_content = sps.content_token_counts(suffix_ids)
    if not all(value > 0 for value in suffix_content):
        raise SystemExit('at least one cohort suffix has no content token after tokenisation (%r): '
                         'the manifest was built with the same rule, so this cohort and this '
                         'tokenisation disagree' % suffix_content)
    if not all(value > 0 for value in prefix_content):
        raise SystemExit('at least one cohort prefix has no content token after tokenisation (%r)'
                         % prefix_content)

    tensors = []
    for sample in cohort:
        with Image.open(os.path.join(args.image_root, str(sample['image_path']))) as image:
            tensors.append(_preprocess(image.convert('RGB')))
    images = torch.stack(tensors).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            g_live = objective.encode_images(images)
            prefix_raw, prefix_hidden = objective.encode_texts(prefix_ids)
            suffix_raw, _suffix_hidden = objective.encode_texts(suffix_ids)
        g_unit = sps.normalize_features(g_live, eps=sps.NORM_EPS)
        t_r = sps.normalize_features(suffix_raw, eps=sps.NORM_EPS)
        mask_s, _soft_s, _mask_logits = sps.said_mask_from_hidden(model.mask_net, prefix_hidden,
                                                                 soft_mask=False)
        mask_s = mask_s.detach().float()
        native_scores = native_scores_from_features(g_unit, t_r)
        prefix_native_scores = sps.suffix_readout_scores(
            g_unit, mask_s, None, t_r, image_chunk=int(args.image_chunk),
            text_chunk=int(args.text_chunk), eps=sps.NORM_EPS)['scores'].detach().float()

    permutations = {
        'PREFIX_SHUFFLED': derangement(range(count), args.prefix_permutation_seed),
        'IMAGE_SHUFFLED': derangement(range(count), args.image_permutation_seed),
    }
    features = {'images': images, 'g_unit': g_unit, 't_r': t_r}

    native_reference = {
        'definition': 'QN[i, j] = 100 * dot(Norm(g_i), tR_j) on the same images and the same suffix '
                      'texts of the fixed cohort, both norms at eps=%r' % sps.NORM_EPS,
        'statistics': score_table(native_scores),
        'scores_100x': [[float(value) for value in row] for row in native_scores.tolist()],
        'unmasked_readout_definition': ('suffix_readout_scores with mS forced to one: the same '
                                        'readout entry point with no prefix conditioning, which '
                                        'isolates the mask from the readout arithmetic'),
        'unmasked_readout_statistics': score_table(prefix_native_scores),
        'unmasked_readout_vs_native_max_abs_diff': float(
            (prefix_native_scores - native_scores).abs().max()),
        'unmasked_readout_vs_native_note': ('the module never renormalises ``g * mS`` (``Norm(g*mS)`` '
                                            'is forbidden), so with ``mS`` forced to one those '
                                            'products are already unit length and this readout '
                                            'coincides with the native scores up to round-off. The '
                                            'difference measures readout arithmetic only; it is NOT '
                                            'the native-versus-conditional gap, which is reported '
                                            'per mode in ``readout_cosine``.'),
    }

    modes_output = {}
    u_all_ones_check = None
    for mode in modes:
        result = run_mode(mode, count, objective, mask, features, mask_s, permutations,
                          int(args.image_chunk), int(args.text_chunk))
        if mode == 'U_ALL_ONES':
            difference = float((result['scores'].detach().float() - native_scores).abs().max())
            u_all_ones_check = {
                'max_abs_difference_from_native_scores': difference,
                'tolerance': float(args.all_ones_tolerance),
                'passed': bool(difference <= float(args.all_ones_tolerance)),
                'reference': '100 * dot(Norm(g_i), tR_j) computed from the same g and tR',
                'why_this_is_an_assertion': ('if forcing the gate open did not reproduce the native '
                                             'scores, the tool would be measuring its own bug rather '
                                             'than the mask, so the run is aborted instead of merely '
                                             'reporting the number'),
            }
            if not u_all_ones_check['passed']:
                raise SystemExit('the U_ALL_ONES self-check FAILED: forcing mU to one gives scores '
                                 'that differ from the native suffix scores by %r (tolerance %r). '
                                 'The forced-all-ones readout must equal 100*dot(Norm(g), tR), so '
                                 'this tool is not measuring what it claims to measure.'
                                 % (difference, float(args.all_ones_tolerance)))
        modes_output[mode] = mode_payload(result, features, u_all_ones_check)

    # ---- read-only assertion AFTER every forward pass ---------------------------------------
    clip_digest_after = sps.state_digest(model.state_dict())
    mask_digest_after = sps.state_digest(mask.state_dict())
    if clip_digest_before != clip_digest_after:
        raise SystemExit('the CLIP parameter digest changed (%s -> %s): this tool must stay '
                         'read-only' % (clip_digest_before, clip_digest_after))
    if mask_digest_before != mask_digest_after:
        raise SystemExit('the suffix-mask parameter digest changed (%s -> %s): this tool must stay '
                         'read-only' % (mask_digest_before, mask_digest_after))

    manifest_digest = sha256_of(manifest_path)
    summary = {
        'probe': PROBE_NAME,
        'objective': OBJECTIVE,
        'arm': payload.get('arm'),
        'read_only': True,
        'new_optimizer_updates': 0,
        'cohort_size': count,
        'cohort_full_size_reached': bool(manifest.get('cohort_full_size_reached')),
        'modes': list(modes),
        'completed_steps': completed,
        'checkpoint_sha256': sha256_of(checkpoint),
        'clip_state_digest': clip_digest_after,
        'suffix_mask_state_digest': mask_digest_after,
        'parameter_digests_unchanged': True,
        'manifest_path': manifest_path,
        'manifest_sha256': manifest_digest,
        'source_manifest_sha256': manifest.get('source_manifest_sha256'),
        'device': str(device),
        'base_model': args.base_model,
        'prefix_permutation_seed': int(args.prefix_permutation_seed),
        'image_permutation_seed': int(args.image_permutation_seed),
        'split_seed': SPLIT_SEED,
    }
    for mode in modes:
        summary['%s_I2T_R@1' % mode] = modes_output[mode]['statistics']['I2T']['R@1']
        summary['%s_T2I_R@1' % mode] = modes_output[mode]['statistics']['T2I']['R@1']
    if 'NORMAL' in modes_output:
        summary['NORMAL_mU_keep_rate'] = \
            modes_output['NORMAL']['mask_pair_statistics']['keep_rate_overall']
        summary['NORMAL_mU_keep_rate_per_pair_mean'] = \
            modes_output['NORMAL']['mask_pair_statistics']['keep_rate_per_pair_mean']
        summary['NORMAL_mU_all_on_fraction'] = \
            modes_output['NORMAL']['mask_pair_statistics']['all_on_fraction']
        summary['NORMAL_mU_probability_mean'] = \
            modes_output['NORMAL']['mask_pair_statistics']['probability_mean']
        summary['NORMAL_mS_mU_elementwise_agreement'] = \
            modes_output['NORMAL']['mask_overlap_old_mS_vs_new_mU'][
                'column_mean_mU_versus_mS']['elementwise_agreement_fraction']
        summary['NORMAL_mS_mU_correlation'] = \
            modes_output['NORMAL']['mask_overlap_old_mS_vs_new_mU'][
                'column_mean_mU_versus_mS']['pearson_correlation']
        summary['NORMAL_cosine_conditional_vs_native_readout'] = \
            modes_output['NORMAL']['readout_cosine'][
                'cosine_Norm_gmU_versus_Norm_g_diagonal_mean']
    if u_all_ones_check is not None:
        summary['U_ALL_ONES_max_abs_difference_from_native'] = \
            u_all_ones_check['max_abs_difference_from_native_scores']
        summary['U_ALL_ONES_self_check_passed'] = u_all_ones_check['passed']
    summary['utc'] = utc_now()

    record = {
        'probe': PROBE_NAME,
        'objective': OBJECTIVE,
        'phase': sps.PHASE,
        'arm': payload.get('arm'),
        'read_only': True,
        'new_optimizer_updates': 0,
        'cohort': {
            'manifest_path': manifest_path,
            'manifest_sha256': manifest_digest,
            'source_manifest_path': manifest.get('source_manifest_path'),
            'source_manifest_sha256': manifest.get('source_manifest_sha256'),
            'cohort_size': count,
            'cohort_requested_size': manifest.get('cohort_requested_size'),
            'cohort_full_size_reached': bool(manifest.get('cohort_full_size_reached')),
            'selection': manifest.get('cohort_selection'),
            'split_rule': manifest.get('split_rule'),
            'k_rule': manifest.get('k_rule'),
            'validity_rule': manifest.get('validity_rule'),
            'honest_statement': manifest.get('honest_statement'),
            'is_external_validation_set': False,
            'is_inside_the_training_distribution': True,
            'true_pair_is_diagonal': True,
            'candidate_protocol': manifest.get('candidate_protocol'),
            'samples': [{'cohort_index': sample['cohort_index'],
                         'json_index': sample['json_index'],
                         'image_path': sample['image_path'],
                         'K': sample['K'],
                         'n_sentences': sample['n_sentences'],
                         'last_nonempty_index': sample['last_nonempty_index'],
                         'prefix': sample['prefix'],
                         'suffix': sample['suffix'],
                         'prefix_content_tokens': sample.get('prefix_content_tokens'),
                         'suffix_content_tokens': sample.get('suffix_content_tokens')}
                        for sample in cohort],
            'prefix_content_tokens': prefix_content,
            'suffix_content_tokens': suffix_content,
            'tokenizer_context_length': sps.TOKENIZER_CONTEXT,
        },
        'modes': modes_output,
        'native_reference': native_reference,
        'u_all_ones_self_check': u_all_ones_check,
        'permutations': {
            'rule': PERMUTATION_RULE,
            'PREFIX_SHUFFLED': {
                'seed': int(args.prefix_permutation_seed),
                'candidate_source_index': permutations['PREFIX_SHUFFLED'],
                'permutes': 'ONLY the source of the candidate prefix masks; images, suffix texts and '
                            'the label diagonal are unchanged',
            },
            'IMAGE_SHUFFLED': {
                'seed': int(args.image_permutation_seed),
                'image_source_index': permutations['IMAGE_SHUFFLED'],
                'permutes': 'the input images; g, rS and mU are all recomputed from the permuted '
                            'images; P/R and the labels stay fixed',
            },
        },
        'provenance': {
            'checkpoint_path': checkpoint,
            'checkpoint_sha256': sha256_of(checkpoint),
            'checkpoint_file_size_bytes': os.path.getsize(checkpoint),
            'completed_steps': completed,
            'objective_in_checkpoint': payload.get('objective'),
            'arm_in_checkpoint': payload.get('arm'),
            'clip_state_key': 'clip_state',
            'suffix_mask_state_key': 'suffix_mask_state',
            'clip_state_digest_recomputed': clip_digest_after,
            'clip_state_digest_stored': stored_clip_digest,
            'clip_state_digest_matches_stored': stored_clip_digest == clip_digest_after,
            'suffix_mask_state_digest_recomputed': mask_digest_after,
            'suffix_mask_state_digest_stored': stored_mask_digest,
            'suffix_mask_state_digest_matches_stored': stored_mask_digest == mask_digest_after,
            'parameter_digests_unchanged_after_the_forwards': True,
            'suffix_mask_config_from_checkpoint': {
                key: value for key, value in (payload.get('suffix_mask_config') or {}).items()
                if isinstance(value, (str, int, float, bool)) or value is None},
            'checkpoint_git_head': payload.get('git_head'),
            'git_head': git_head(REPO),
            'device': str(device),
            'device_name': (torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'),
            'torch_version': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_device_count': int(torch.cuda.device_count()),
            'run_dir': run_dir,
            'run_dir_not_written': True,
            'status_output': status_output,
            'status_output_is_this_tools_own_file': True,
            'never_touched_another_run_status_json': True,
            'manifest_path': manifest_path,
            'manifest_sha256': manifest_digest,
            'source_manifest_path': manifest.get('source_manifest_path'),
            'source_manifest_sha256': manifest.get('source_manifest_sha256'),
            'source_manifest_sha256_rechecked': True,
            'modes': list(modes),
            'mode_seeds': {'split_seed': SPLIT_SEED,
                           'prefix_permutation_seed': int(args.prefix_permutation_seed),
                           'image_permutation_seed': int(args.image_permutation_seed)},
            'image_root': os.path.abspath(args.image_root),
            'base_model': args.base_model,
            'weight_dtype': args.weight_dtype,
            'autocast': ('bfloat16 for the visual and text encoders' if use_autocast
                         else 'disabled (cpu)'),
            'fp32_core': 'the gate F, the g normalisation, the suffix scoring and the CE cores',
            'image_chunk': int(args.image_chunk),
            'text_chunk': int(args.text_chunk),
            'all_ones_tolerance': float(args.all_ones_tolerance),
            'wall_seconds': time.time() - started,
            'recorded_utc': utc_now(),
        },
        'reading': {
            'different_masks_are_not_decoupling': ABSOLUTE_READING,
            'shuffled_conditioning_is_not_proof': SHUFFLE_READING,
            'u_all_ones_close_to_normal': U_ALL_ONES_READING,
            'cohort_is_not_a_validation_set': COHORT_READING,
            'small_pool': SMALL_POOL_READING,
            'scope': ('mechanism diagnostic on ONE mask-arm checkpoint and ONE fixed cohort; it does '
                      'not replace the frozen native COCO / Urban-1k evaluation and it decides '
                      'nothing about the frozen gate'),
        },
        'not_run': [
            'any training or optimizer update',
            'any import or construction of an optimizer',
            'any write inside --run_dir',
            'any write to another run\'s run_status.json',
            'the native arm as a second checkpoint (the suffix gate exists only in the mask arm)',
            'any significance test (the pool is at most 64 samples and no claim of a stable effect '
            'is made)',
            'any external-validation or generalization claim',
        ],
        'not_claimed': [
            'that the suffix equals all of the Unsaid content',
            'that the two masks are orthogonal or together cover the image',
            'that semantic decoupling follows from the masks differing',
            'that a drop under shuffling proves pure visual complementarity',
            'that any COCO / Urban-1k conclusion follows from this diagnostic',
        ],
    }

    write_json(output, record)
    status = {
        'probe': PROBE_NAME,
        'objective': OBJECTIVE,
        'arm': payload.get('arm'),
        'state': 'completed',
        'read_only': True,
        'new_optimizer_updates': 0,
        'checkpoint': checkpoint,
        'checkpoint_sha256': summary['checkpoint_sha256'],
        'completed_steps': completed,
        'manifest': manifest_path,
        'manifest_sha256': manifest_digest,
        'cohort_size': count,
        'modes': list(modes),
        'output': output,
        'output_sha256': sha256_of(output),
        'run_dir': run_dir,
        'run_dir_not_written': True,
        'wrote_own_status_file': True,
        'updated_utc': utc_now(),
    }
    write_json(status_output, status)

    print('WROTE %s' % output)
    print('WROTE %s' % status_output)
    print('SUFFIX_MECHANISM_DIAGNOSTIC ' + json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
