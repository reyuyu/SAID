"""Two-rank S0-Suffix v0.1 worker: one REAL optimizer update of the valid-subset DDP route.

Run through ``torch.distributed.run --nproc_per_node=2``. Every rank builds the SAME CLIP ViT-B/16
from the local cache (plus the shared initialisation when it is present) and consumes a
deterministic slice of one global batch of ``world * local_batch`` samples: rank ``r`` takes
``[r * local_batch, (r + 1) * local_batch)``, exactly like the trainer's ``DistributedSampler``.

Why this file exists (spec section 6, test section 9 G): the suffix objective is computed over the
GLOBAL valid subset, so the per-rank backward value must be

    L_U_backward_local = (world_size / V) * (local I2T CE sum + local T2I CE sum)

and standard DDP averaging of that over the ranks produces the global valid-anchor mean
``L_U_global = (sum_valid I2T_CE + sum_valid T2I_CE) / V``. The worker therefore DOES wrap the
training module in ``DistributedDataParallel``, so the gradients it writes are the real
post-allreduce gradients, and it performs one real ``AdamW`` step whose parameter deltas the test
compares against a single-process oracle (spec section 9 G forbids a forward-only proof).

Five validity cases are driven from ``tests/test_said_prefix_suffix.py``:

    equal          equal per-rank valid counts (2 and 2, global V = 4)
    unequal        unequal per-rank valid counts (3 and 1, global V = 4)
    one_rank_zero  rank 1 contributes nothing while global V = 2 >= 2
    global_zero    global V = 0
    global_one     global V = 1

The worker itself never decides what "correct" is. It records the exact global inputs, the exact
validity pattern, the rank-local gradients, the post-step parameters, and its own witnesses for the
two structural rules that cannot be checked from outside:

1. a rank with ``n_r = 0`` must not skip its collectives. The witness is that the rank read back the
   REAL global validity vector from the gather (impossible if it returned early) and that its suffix
   term is a differentiable exact zero (the graph exists, the value is 0), so the backward pass it
   runs together with its peer is the shared one. An early ``return`` before the collectives would
   hang the peer, which shows up as a run that never finishes rather than as a wrong number;
2. the objective's local suffix term must equal ``(world_size / V) * local_ce_sum`` exactly, which
   the worker records so the test can verify the scaling identity on real tensors.
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                       # noqa: E402
from model import said_prefix_suffix as suffix_model                             # noqa: E402
from model.said_cls_cvssl import compute_smartclip_terms                         # noqa: E402
import train_said_prefix_suffix as trainer                                       # noqa: E402

# captions whose K = 1 split leaves a NON-EMPTY suffix: four sentences, the last non-empty fragment
# is the fourth, so K = 1 yields R = "second. third" and valid = True
VALID_CAPTIONS = (
    'alpha one. beta two. gamma three. delta four',
    'epsilon start. zeta middle. eta more. theta end',
    'iota begin. kappa body. lambda next. mu finish',
    'nu open. xi inside. omicron after. pi close',
    'rho first. sigma second. tau third. upsilon last',
    'phi head. chi torso. psi tail. omega final',
    'kappa lead. lambda centre. mu tail. nu stop',
    'omicron top. pi upper. rho lower. sigma bottom',
)
# captions whose K = 1 split leaves an EMPTY suffix: their only sentence is the last non-empty
# fragment, so there is nothing after it (the second one also carries a trailing empty fragment,
# which the deterministic skip must ignore)
INVALID_CAPTIONS = (
    'only one sentence with no successor',
    'single sentence then a trailing fragment .',
    'a lone sentence that ends the caption',
    'one more sentence with nothing after it',
    'another solitary caption without a follow up',
    'yet another single sentence and nothing more',
    'a final lonely sentence closing the text',
    'one last caption that has no successor at all',
)

#: the raw per-rank validity patterns of the five cases; the index inside a rank is the position
#: inside that rank's slice of the global batch, and the ranks are concatenated in rank order
CASE_PATTERNS = {
    'equal': ([1, 0, 1, 0], [1, 0, 1, 0]),
    'unequal': ([1, 1, 1, 0], [0, 1, 0, 0]),
    'one_rank_zero': ([1, 0, 1, 0], [0, 0, 0, 0]),
    'global_zero': ([0, 0, 0, 0], [0, 0, 0, 0]),
    'global_one': ([1, 0, 0, 0], [0, 0, 0, 0]),
}

#: the parameters the test compares. ``suffix_mask.*`` is the new F, ``clip.visual.conv1`` the
#: shared visual trunk, ``clip.text_projection`` the shared text trunk and ``clip.mask_net`` the OLD
#: S0 mask, which the suffix loss must never touch directly.
GRAD_NAMES = ('suffix_mask.layer1.weight', 'suffix_mask.layer1.bias',
              'suffix_mask.layer2.weight', 'suffix_mask.layer2.bias',
              'clip.visual.conv1.weight', 'clip.visual.proj', 'clip.text_projection',
              'clip.textual.positional_embedding', 'clip.mask_net.out.weight',
              'clip.mask_net.proj.weight')
STEP_NAMES = ('suffix_mask.layer1.weight', 'suffix_mask.layer2.bias',
              'clip.visual.conv1.weight', 'clip.visual.proj', 'clip.text_projection',
              'clip.mask_net.out.weight')


# --------------------------------------------------------------------------- deterministic inputs
def build_captions(case, local_batch, world):
    """The exact global caption list of ``case``: a property of the TEXT alone, no RNG involved.

    The captions are drawn from one shared pool in global order, so different cases reuse the same
    texts and the only difference between the runs is the validity PATTERN.
    """
    if case not in CASE_PATTERNS:
        raise SystemExit('unknown case %r; known cases are %r' % (case, sorted(CASE_PATTERNS)))
    patterns = CASE_PATTERNS[case]
    if len(patterns) != world:
        raise SystemExit('case %r declares %d ranks but world_size is %d'
                         % (case, len(patterns), world))
    flags = [flag for pattern in patterns for flag in pattern]
    if len(flags) != world * local_batch:
        raise SystemExit('case %r declares %d samples but the global batch is %d'
                         % (case, len(flags), world * local_batch))
    valid_pool = list(VALID_CAPTIONS)
    invalid_pool = list(INVALID_CAPTIONS)
    captions = []
    for flag in flags:
        pool = valid_pool if flag else invalid_pool
        if not pool:
            raise SystemExit('case %r needs %d %s caption(s) but only %d distinct text(s) exist'
                             % (case, flags.count(bool(flag)), 'valid' if flag else 'invalid',
                                len(VALID_CAPTIONS if flag else INVALID_CAPTIONS)))
        captions.append(pool.pop(0))
    return captions


def build_images(global_count, seed):
    """The deterministic global image batch: identical on every rank, sliced per rank."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(global_count, 3, 224, 224, generator=generator)


def load_shared_init(model, path):
    """Load the shared initialisation the way the trainer does; returns the number of tensors."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        payload = payload['model']
    target = set(model.state_dict())
    mapped = {}
    for key, value in payload.items():
        if key in target:
            mapped[key] = value
        elif key.startswith('clip.') and key[len('clip.'):] in target:
            mapped[key[len('clip.'):]] = value
        elif ('clip.' + key) in target:
            mapped['clip.' + key] = value
    if not mapped:
        raise SystemExit('the shared init %s matches no model tensor' % path)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    bad = [key for key in missing if not key.startswith('mask_net')]
    if bad or list(unexpected):
        raise SystemExit('the shared init did not load cleanly: missing %r unexpected %r'
                         % (bad, list(unexpected)))
    return len(mapped)


# --------------------------------------------------------------------------- the objective module
def _gate(module, pU):
    """``mU = hardU + (pU - pU.detach())`` through whatever frozen name the module exposes."""
    gate_class = getattr(module, 'SuffixMaskGate', None)
    if gate_class is not None and hasattr(gate_class, 'gate_from_pU'):
        return gate_class.gate_from_pU(pU)
    if hasattr(module, 'gate_from_pU'):
        return module.gate_from_pU(pU)
    raise SystemExit('the implementation exposes no gate_from_pU: the frozen API of spec section 14 '
                     'cannot be exercised')


def _try_calls(function, orders, kwargs_variants):
    errors = []
    for order in orders:
        for kwargs in kwargs_variants:
            try:
                return function(*order, **kwargs)
            except TypeError as error:
                errors.append('%s' % (error,))
    raise SystemExit('none of the documented call shapes of %r fitted; the frozen API of spec '
                     'section 14 must accept one of them (%s)' % (function.__name__, errors[:4]))


def call_readout(module, g, mS, tR, gate, image_chunk, text_chunk):
    """``suffix_readout_scores`` -> ``(scores, mU)``, whatever argument order it declares."""
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


def call_mask_helper(module, mask_net, hidden):
    """The ORIGINAL S0 mask pipeline used unchanged (spec section 1)."""
    function = getattr(module, 'said_mask_helper', None)
    if function is None:
        from model.said_cls_cvssl import said_mask_from_hidden as function
    return function(mask_net, hidden, soft_mask=False)


class SuffixObjective(torch.nn.Module):
    """The DDP-visible training module: the trunk, the old S0 mask, F, and the suffix loss.

    The loss implemented here is the SPEC's loss and not the trainer's copy of it, because the
    worker has to produce a gradient the test can compare against an independent single-process
    oracle. The production entry points that are reachable are recorded in ``production_entry_points``
    so the report can state which of them were exercised.
    """

    def __init__(self, model, suffix_mask, rank, world, local_batch, readout=None):
        super().__init__()
        self.model = model
        self.suffix_mask = suffix_mask
        self.rank = int(rank)
        self.world = int(world)
        self.local_batch = int(local_batch)
        self.lambda_suffix = float(suffix_model.LAMBDA_SUFFIX)
        self.image_chunk = int(suffix_model.IMAGE_CHUNK_DEFAULT)
        self.text_chunk = int(suffix_model.TEXT_CHUNK_DEFAULT)
        # the readout is a FUNCTION of (g, mS, tR, F, chunking) and owns the [V, V] pairing of the
        # candidate prefix masks (spec section 5: column j is generated from (P_j, R_j)). This call
        # shape is the documented one for ``suffix_readout_scores``; the alternative shape used
        # below the fallback lets the worker still run if the implementation takes it instead.
        self.readout = readout or 'sequence'

    # -- feature level ------------------------------------------------------------------------
    def encode(self, images, prefix_ids, suffix_ids):
        g_raw = self.model.encode_image(images)
        g = F.normalize(g_raw.float(), dim=-1, eps=1e-6)
        prefix_hidden = self.model.encode_text(prefix_ids, return_full=True)[1]
        prefix_features = self.model.encode_text(prefix_ids)
        tR = F.normalize(self.model.encode_text(suffix_ids).float(), dim=-1, eps=1e-6)
        return g_raw, g, prefix_hidden, prefix_features, tR

    def s0_loss(self, g_raw, prefix_hidden, prefix_features):
        """The unchanged S0 objective on this rank's ``prefix`` string (the S0 arm's text input)."""
        mS = call_mask_helper(self, self.model.mask_net, prefix_hidden)
        terms = compute_smartclip_terms(g_raw, prefix_features, mS, self.rank)
        return terms['loss_sidm'] + terms['loss_dism'] + terms['loss_sparsity'], mS

    # -- the suffix branch --------------------------------------------------------------------
    def suffix_loss(self, g, prefix_hidden, tR, global_valid):
        """The global-valid-pool suffix loss with the spec's ``world_size / V`` backward value.

        ``g``, ``prefix_hidden`` and ``tR`` are the rank-LOCAL tensors; they are gathered with
        autograd so that the suffix gradient reaches the shared trunk on every rank. The gather is
        entered by every rank, whatever ``n_r`` is.
        """
        device = g.device
        global_valid = global_valid.to(device)
        V = int(global_valid.sum().item())
        scaling = float(suffix_model.suffix_scaling(self.world, V))
        expected = 0.0 if V < 2 else float(self.world) / float(V)
        if abs(scaling - expected) > 1e-12:
            raise SystemExit('suffix_scaling(%d, %d) = %r, expected %r'
                             % (self.world, V, scaling, expected))

        hidden_all = torch.cat(dist.nn.all_gather(prefix_hidden), dim=0)
        g_all = torch.cat(dist.nn.all_gather(g), dim=0)
        tR_all = torch.cat(dist.nn.all_gather(tR), dim=0)
        mS_all = call_mask_helper(self, self.model.mask_net, hidden_all)

        if V < 2:
            # no fake single-candidate CE and no re-draw: the global suffix loss is exactly zero,
            # but it stays a differentiable part of this rank's graph
            zero = (g_all.sum() + mS_all.sum() + tR_all.sum()) * 0.0
            return zero, scaling, V, torch.zeros(0, dtype=torch.long, device=device), None, 0.0

        index = torch.nonzero(global_valid, as_tuple=False).flatten()
        rows = index // self.local_batch
        cols = index % self.local_batch
        g_valid = g_all.index_select(0, rows)
        tR_valid = tR_all.index_select(0, cols)
        if self.readout == 'sequence':
            # the valid pool in GLOBAL ORDER: image i paired with candidate j, and column j carries
            # the prefix mask of candidate j, which is exactly what the readout must reproduce
            scores, mU = call_readout(self, g_valid, mS_all.index_select(0, rows), tR_valid,
                                      self.suffix_mask, self.image_chunk, self.text_chunk)
        else:
            # the matrix form: the full [rows, cols, 512] mask block of the valid pool
            mask_block = mS_all.index_select(0, rows).unsqueeze(1).expand(
                rows.numel(), cols.numel(), mS_all.shape[-1])
            scores, mU = call_readout(self, g_valid, mask_block, tR_valid,
                                      self.suffix_mask, self.image_chunk, self.text_chunk)
        if tuple(scores.shape) != (V, V):
            raise SystemExit('the readout returned a %s matrix for a global valid pool of V = %d'
                             % (tuple(scores.shape), V))
        targets = torch.arange(V, device=device)
        ce_i2t = F.cross_entropy(scores, targets, reduction='sum')
        ce_t2i = F.cross_entropy(scores.t(), targets, reduction='sum')
        # only this rank's own anchors contribute; a rank with no valid anchor adds a zero that
        # still carries the graph of the collectives it participated in
        local_mask = (index >= self.rank * self.local_batch) & \
                     (index < (self.rank + 1) * self.local_batch)
        if bool(local_mask.any()):
            local_ce = _sum_rows(scores, local_mask, targets) \
                + _sum_rows(scores.t(), local_mask, targets)
        else:
            local_ce = (scores.sum() + ce_i2t + ce_t2i) * 0.0
        # the spec's scaling, checked on real tensors before the loss is returned: the local
        # backward value is (world_size / V) times THIS RANK's CE sum, equivalently the local mean
        # times (world_size * n_r / V). A double world_size factor fails this immediately.
        n_local = int(local_mask.sum().item())
        global_ce = ce_i2t + ce_t2i
        if abs(float(local_ce.detach()) - float((scaling * global_ce).detach())) > 1e-4:
            raise SystemExit('rank %d: the local CE sum %r is not (scaling x global CE sum) = %r'
                             % (self.rank, float(local_ce.detach()),
                                float((scaling * global_ce).detach())))
        del n_local
        del ce_i2t, ce_t2i
        loss = scaling * self.lambda_suffix * local_ce
        return loss, scaling, V, index, mU, float(local_ce.detach())


def _sum_rows(matrix, mask, targets):
    """The CE sum over the anchors selected by ``mask`` (the spec's per-anchor CE, summed)."""
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() == 0:
        return matrix.sum() * 0.0
    return F.cross_entropy(matrix.index_select(0, selected),
                           targets.index_select(0, selected), reduction='sum')


def main():
    parser = argparse.ArgumentParser(description='two-rank S0-Suffix v0.1 update worker')
    parser.add_argument('--case', required=True, choices=sorted(CASE_PATTERNS))
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--local_batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--lr_suffix', type=float, default=1e-3)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--init_state', default='/root/SAID-gap-completion/runs_salu/'
                                               'said_cls_cvssl/shared_init/cvssl_initial.pt')
    parser.add_argument('--clip_cache', default=os.path.expanduser('~/.cache/clip/ViT-B-16.pt'))
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    device = torch.device('cuda', local_rank)

    if not os.path.isfile(args.clip_cache):
        raise SystemExit('the CLIP checkpoint is not cached at %s' % args.clip_cache)

    from model import longclip as _longclip
    model, _preprocess = _longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                                  args=argparse.Namespace())
    # ``logit_scale`` is not part of this objective (the score scale is the fixed 100) and the
    # trainer never trains it: freezing it here keeps every DDP-registered parameter genuinely used
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(100.0))
    model.logit_scale.requires_grad_(False)
    loaded = load_shared_init(model, args.init_state) if os.path.isfile(args.init_state) else 0
    model = model.to(device).train()
    suffix_mask = suffix_model.SuffixMask().to(device)
    for parameter in suffix_mask.parameters():
        parameter.requires_grad_(True)

    captions = build_captions(args.case, args.local_batch, world)
    global_images = build_images(world * args.local_batch, args.seed)
    local_images = global_images[rank * args.local_batch:(rank + 1) * args.local_batch].to(device)
    local_captions = captions[rank * args.local_batch:(rank + 1) * args.local_batch]

    # one fixed K = 1 draw for every rank: the validity pattern is then a property of the text
    splits = [_split(caption, 1) for caption in local_captions]
    local_valid = torch.tensor([bool(item['valid']) for item in splits], dtype=torch.bool,
                               device=device)
    local_prefix_ids = _tokenize([item['prefix'] for item in splits], device)
    local_suffix_ids = _tokenize([(item['suffix'] if bool(item['valid']) else '')
                                  for item in splits], device)

    # --- collectives FIRST: every rank participates, including a rank with n_r = 0 ---------------
    global_valid = torch.zeros(world * args.local_batch, dtype=torch.bool, device=device)
    dist.all_gather_into_tensor(global_valid, local_valid)
    global_valid_list = global_valid.detach().cpu().tolist()
    local_valid_list = local_valid.detach().cpu().tolist()
    if local_valid_list != global_valid_list[rank * args.local_batch:
                                             (rank + 1) * args.local_batch]:
        raise SystemExit('rank %d read back a global validity vector that contradicts its own local '
                         'slice: the gather is not the one this rank contributed to' % rank)
    # the STRUCTURAL collective-participation requirement of spec section 6: a rank whose slice is
    # entirely invalid must still take part in every all_gather of the step. The witness here is
    # that this rank receives the gather's real output rather than a locally faked one.
    gathered_validity = [torch.zeros_like(local_valid) for _ in range(world)]
    dist.all_gather(gathered_validity, local_valid)
    for other, piece in enumerate(gathered_validity):
        expected_slice = global_valid_list[other * args.local_batch:
                                           (other + 1) * args.local_batch]
        if piece.detach().cpu().tolist() != expected_slice:
            raise SystemExit('rank %d saw a wrong slice for rank %d in the validity gather'
                             % (rank, other))
    if not bool(local_valid.any()):
        # this rank contributes no anchor: it must STILL be inside the collectives, which means it
        # must be able to see the peers' valid suffixes gathered below
        if not any(bool(piece.any()) for other, piece in enumerate(gathered_validity)
                   if other != rank):
            raise SystemExit('rank %d is invalid-only but the global pool is empty, which '
                             'contradicts the case definition' % rank)
    gathered = [torch.zeros_like(local_suffix_ids) for _ in range(world)]
    dist.all_gather(gathered, local_suffix_ids)
    global_suffix_ids = torch.cat(gathered, dim=0)

    V = int(global_valid.sum().item())
    n_local = int(local_valid.sum().item())

    objective = SuffixObjective(model, suffix_mask, rank, world, args.local_batch).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        objective, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ddp_model._set_static_graph()

    g_raw, g, prefix_hidden, prefix_features, tR = objective.encode(local_images, local_prefix_ids,
                                                                    local_suffix_ids)
    s0_value, mS = objective.s0_loss(g_raw, prefix_hidden, prefix_features)
    suffix_value, scaling, V, index, mU, local_ce = ddp_model.module.suffix_loss(
        g, prefix_hidden, tR, global_valid)
    total = s0_value + suffix_value
    if not bool(torch.isfinite(total)):
        raise SystemExit('rank %d produced a non-finite total loss' % rank)
    if V < 2 and float(suffix_value.detach()) != 0.0:
        raise SystemExit('V = %d must give an EXACTLY zero suffix loss, got %r'
                         % (V, float(suffix_value.detach())))

    optimizer = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad],
          'lr': args.lr_clip, 'weight_decay': 1e-2},
         {'params': [p for p in suffix_mask.parameters() if p.requires_grad],
          'lr': args.lr_suffix, 'weight_decay': 0.0}],
        betas=(0.9, 0.999), eps=1e-8)

    named = dict(model.named_parameters())
    named.update({'suffix_mask.%s' % key: value
                  for key, value in suffix_mask.named_parameters()})
    for name in GRAD_NAMES + STEP_NAMES:
        if name not in named:
            raise SystemExit('parameter %r is missing from the worker model; present: %r'
                             % (name, sorted(named)[:12]))

    # the exact parameterisation this rank started from: the single-process oracle loads it, so the
    # gradient and update comparisons cannot be confounded by two different initialisations
    initial_parameters = {name: parameter.detach().float().cpu().clone()
                          for name, parameter in named.items()}

    optimizer.zero_grad(set_to_none=True)
    total.backward()

    local_grads = {}
    for name in GRAD_NAMES:
        grad = named[name].grad
        local_grads[name] = None if grad is None else grad.detach().float().cpu().clone()

    # --- the structural witnesses, measured on the graph that was just back-propagated ----------
    # a rank with n_r = 0 makes a differentiable exactly-zero contribution
    zero_witness = (float(suffix_value.detach()) == 0.0) if n_local == 0 else None
    # the OLD S0 mask must receive nothing from the suffix loss (spec section 5)
    mask_grads = torch.autograd.grad(suffix_value, list(model.mask_net.parameters()),
                                     retain_graph=True, allow_unused=True)
    mask_grad_norm_from_suffix = float(sum(
        0.0 if grad is None else float(grad.detach().float().pow(2).sum()) for grad in mask_grads))
    # the suffix loss must train the shared trunk (the live g path)
    trunk_grad = torch.autograd.grad(suffix_value, [model.visual.proj], retain_graph=True,
                                     allow_unused=True)[0]
    trunk_grad_norm_from_suffix = (0.0 if trunk_grad is None
                                   else float(trunk_grad.detach().float().pow(2).sum()))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    local_step = {name: named[name].detach().float().cpu().clone() for name in STEP_NAMES}

    payload = {
        'rank': rank, 'world_size': world, 'local_batch': args.local_batch,
        'case': args.case, 'seed': args.seed,
        'lambda_suffix': float(suffix_model.LAMBDA_SUFFIX),
        'scaling': scaling, 'V': V, 'n_local': n_local,
        'valid_index': index.detach().cpu().tolist(),
        'global_valid': global_valid_list, 'local_valid': local_valid_list,
        'captions': captions, 'local_captions': local_captions,
        'prefixes': [item['prefix'] for item in splits],
        'suffixes': [item['suffix'] for item in splits],
        'k_values': [int(item['k']) for item in splits],
        'local_ce': float(local_ce),
        'loss_suffix_local': float(suffix_value.detach()),
        'loss_s0': float(s0_value.detach()),
        'loss_total': float(total.detach()),
        'mask_grad_norm_from_suffix': mask_grad_norm_from_suffix,
        'trunk_grad_norm_from_suffix': trunk_grad_norm_from_suffix,
        'zero_rank_witness': zero_witness,
        'zero_rank_gathered_global_valid': global_valid_list,
        'grads': local_grads, 'step': local_step,
        'initial_parameters': initial_parameters,
        'mU_all_ones': (None if mU is None else bool(float((mU - 1.0).abs().max()) == 0.0)),
        'production_entry_points': {
            'trainer_has_main': hasattr(trainer, 'main'),
            'trainer_has_ARMS': hasattr(trainer, 'ARMS'),
            'writer_name': next((name for name in ('write_checkpoint', 'save_checkpoint',
                                                   'write_production_checkpoint')
                                 if callable(getattr(trainer, name, None))), None),
        },
        'init_state_loaded': loaded,
        'init_state_path': args.init_state if os.path.isfile(args.init_state) else None,
    }

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()
    torch.save(payload, os.path.join(args.output_dir, 'rank%d.pt' % rank))
    dist.barrier()
    if rank == 0:
        with open(os.path.join(args.output_dir, 'worker_info.json'), 'w', encoding='utf-8') as fh:
            json.dump({'case': args.case, 'world_size': world, 'local_batch': args.local_batch,
                       'V': V, 'seed': args.seed, 'scaling': scaling,
                       'lambda_suffix': float(suffix_model.LAMBDA_SUFFIX),
                       'init_state_loaded': loaded}, fh, indent=2, sort_keys=True)
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------- small helpers
def _tokenize(texts, device):
    from model import longclip
    return longclip.tokenize(list(texts), truncate=True).to(device)


def _split(caption, k):
    try:
        return suffix_model.split_prefix_suffix(caption, k)
    except TypeError:
        return suffix_model.split_prefix_suffix(caption, k=k)


if __name__ == '__main__':
    main()
