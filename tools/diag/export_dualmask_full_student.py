"""Strict bare-student export for Dual-Mask-Full v0.1 (`S0_DUALMASK_FULL_V01`).

The frozen COCO/Urban consumers take a bare CLIP student, exactly as in the accepted Clean v0.1
export: only ``clip_state`` is written, and the suffix gate stays in the training checkpoint.

The tool refuses to run unless the checkpoint is a real 500-update masked checkpoint whose recorded
coefficients are the ones this ablation was launched with, and it separates two probes that are easy
to confuse: loading the same trained state into two independent models (a real losslessness check,
expected to be exactly zero) and comparing the trained state against the initialization (expected to
be non-zero, which is what shows the trained weights are actually the ones being exported).
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (REPO, os.path.join(REPO, 'train')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
from model.dual_mask_suffix import DualMaskSuffixTrainModule                    # noqa: E402
import train_dual_mask_suffix as trainer                                        # noqa: E402

ARM = 'S0_DUALMASK_FULL_V01'
EXPECTED_STEPS = 500
EXPECTED_LAMBDA_SUFFIX = 10.0
EXPECTED_LAMBDA_U_SPARSE = 2.0


def ensure_process_group():
    """The module forward uses autograd-aware ``all_gather``, so a one-process group must exist.

    This is the same single-process fallback the trainer uses; it makes no collective do any real
    communication (world size 1) and it never touches the checkpoint.
    """
    if not dist.is_available() or dist.is_initialized():
        return
    init_file = '/tmp/dmfull_export_pg_%d' % os.getpid()
    try:
        os.remove(init_file)
    except FileNotFoundError:
        pass
    dist.init_process_group(backend='gloo', init_method='file://' + init_file, rank=0, world_size=1)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(device, lambda_suffix, lambda_u_sparse):
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                       args=argparse.Namespace())
    module = DualMaskSuffixTrainModule(model, suffix_mode='masked', rank=0, image_chunk=16,
                                       text_chunk=32, lambda_suffix=lambda_suffix,
                                       lambda_u_sparse=lambda_u_sparse)
    return module.to(device)


def probe(module, device, seed=20260914):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(2, 3, 224, 224, generator=generator).to(device)
    text = longclip.tokenize(['a photo of a cat', 'a dog on grass'], truncate=True).to(device)
    suffix = longclip.tokenize(['and it is running', 'on the sofa'], truncate=True).to(device)
    with torch.no_grad():
        g = module.clip.encode_image(images)
        t = module.clip.encode_text(text)
        s = module.clip.encode_text(suffix)
        out = module(images, text, suffix, torch.ones(2, dtype=torch.bool, device=device),
                     torch.arange(2, device=device))
    return {'image': g.float().cpu(), 'text': t.float().cpu(), 'suffix': s.float().cpu(),
            'loss_total': out['loss_total'].detach().float().cpu().reshape(1),
            'u_sparse_global': out['loss_u_sparse_global'].detach().float().cpu().reshape(1)}


def main():
    parser = argparse.ArgumentParser(description='export the %s bare student' % ARM)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--expect_steps', type=int, default=EXPECTED_STEPS)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit('the checkpoint is not a dict: %r' % type(payload))
    problems = []
    steps = int(payload.get('completed_steps', -1))
    if steps != args.expect_steps:
        problems.append('completed_steps=%r expected %d' % (payload.get('completed_steps'),
                                                            args.expect_steps))
    state = payload.get('clip_state')
    if not isinstance(state, dict) or not state:
        problems.append('no clip_state mapping in the checkpoint')
    gate_state = payload.get('suffix_gate_state')
    if not isinstance(gate_state, dict) or not gate_state:
        problems.append('the masked checkpoint has no suffix_gate_state')
    optimizers = payload.get('optimizer_states')
    if not isinstance(optimizers, dict) or not all(
            isinstance(optimizers.get(key), dict) and optimizers[key].get('state')
            for key in ('clip', 'mask', 'suffix')):
        problems.append('the optimizer states are missing or empty')
    config = payload.get('config') or {}
    if config.get('suffix_mode') != 'masked':
        problems.append('suffix_mode=%r expected masked' % config.get('suffix_mode'))
    for key, expected in (('suffix_lambda', EXPECTED_LAMBDA_SUFFIX),
                          ('u_sparsity_lambda', EXPECTED_LAMBDA_U_SPARSE)):
        if float(config.get(key, float('nan'))) != expected:
            problems.append('%s=%r expected %r' % (key, config.get(key), expected))
    if not config.get('training_sha'):
        problems.append('the checkpoint config has no training_sha')
    if problems:
        raise SystemExit('checkpoint verification failed for %s: %s' % (args.checkpoint, problems))

    device = torch.device(args.device)
    ensure_process_group()
    module = build(device, EXPECTED_LAMBDA_SUFFIX, EXPECTED_LAMBDA_U_SPARSE)
    before = probe(module, device)                       # initialization, not trained
    trainer.load_checkpoint(args.checkpoint, module)     # strict clip + strict gate
    after = probe(module, device)
    twin = build(device, EXPECTED_LAMBDA_SUFFIX, EXPECTED_LAMBDA_U_SPARSE)
    trainer.load_checkpoint(args.checkpoint, twin)
    twin_probe = probe(twin, device)
    lossless = {key: float((after[key] - twin_probe[key]).abs().max()) for key in after}
    trained_vs_init = {key: float((before[key] - after[key]).abs().max()) for key in after}

    os.makedirs(args.output_dir, exist_ok=True)
    student_path = os.path.join(args.output_dir, 'bare_student_step500.pt')
    if os.path.isfile(student_path):
        raise SystemExit('refusing to overwrite %s' % student_path)
    torch.save(state, student_path)
    # a fresh native CLIP must load the bare student strictly and agree with the full model
    control, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                         args=argparse.Namespace())
    control = control.to(device)
    control.load_state_dict(torch.load(student_path, map_location='cpu', weights_only=False),
                            strict=True)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(20260914)
        images = torch.randn(2, 3, 224, 224, generator=generator).to(device)
        text = longclip.tokenize(['a photo of a cat', 'a dog on grass'], truncate=True).to(device)
        student_out = {'image': control.encode_image(images).float().cpu(),
                       'text': control.encode_text(text).float().cpu()}
    student_delta = {key: float((student_out[key] - after[key]).abs().max()) for key in student_out}

    metadata = {
        'arm': ARM, 'objective': payload.get('config', {}).get('objective'),
        'source_checkpoint': os.path.abspath(args.checkpoint),
        'source_checkpoint_sha256': sha256_of(args.checkpoint),
        'completed_steps': steps,
        'checkpoint_config': {k: config.get(k) for k in
                              ('objective', 'suffix_mode', 'suffix_lambda', 'u_sparsity_lambda',
                               'u_sparsity', 'total_objective', 'training_sha', 'lr_horizon_steps',
                               'loader_batches', 'world_size', 'batch_size_per_gpu', 'seed')},
        'clip_tensor_count': len(state),
        'suffix_gate_tensor_count': len(gate_state),
        'student_file': student_path,
        'student_key': 'clip_state_only',
        'student_sha256': sha256_of(student_path),
        'probe_after_two_independent_loads_max_abs_diff': lossless,
        'probe_after_two_independent_loads_note': (
            'the same trained state loaded into two independent models; this is the real '
            'losslessness check and must be exactly zero'),
        'probe_trained_vs_initialization_max_abs_diff': trained_vs_init,
        'probe_trained_vs_initialization_note': (
            'expected to be clearly non-zero: it shows the exported weights are the trained ones'),
        'bare_student_vs_full_model_native_max_abs_diff': student_delta,
        'bare_student_vs_full_model_native_note': (
            'encode_image / encode_text of a fresh CLIP loaded strictly from the bare student '
            'against the same interfaces of the full trained model'),
        'gate_still_in_checkpoint_only': True,
    }
    metadata_path = os.path.join(args.output_dir, 'bare_student_metadata.json')
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print('EXPORTED ' + student_path, flush=True)
    print('LOSSLESS ' + json.dumps(lossless, sort_keys=True), flush=True)
    print('TRAINED_VS_INIT ' + json.dumps(trained_vs_init, sort_keys=True), flush=True)
    print('BARE_VS_FULL ' + json.dumps(student_delta, sort_keys=True), flush=True)
    print('EXPORT_METADATA ' + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
