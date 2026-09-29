"""Real ViT-B/16 shape, compatibility, conditioning, pooling and export-path checks."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import longclip
from model.nested_joint_input import VisualInputProjection, joint_logits
from train.nested_semantic_data import NestedDataset, file_sha


def state_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init-state', required=True)
    p.add_argument('--index-dir', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    initial = torch.load(args.init_state, map_location='cpu', weights_only=False)
    incompatible = clip.load_state_dict(initial['model'], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    clip = clip.float().cuda().eval()
    dataset = NestedDataset(args.index_dir, args.image_root, 'random_k', 0)
    dataset.set_epoch(0)
    samples = [dataset[i] for i in (0, 1)]
    images = torch.stack([sample['image'] for sample in samples]).cuda()
    tokens = torch.stack([sample['tokens_f'] for sample in samples]).cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        native = clip.encode_image(images)
        z, cls, patch = clip.encode_image_with_joint_tokens(images)
        text, hidden = clip.encode_text(tokens, return_full=True)
        projected_from_cls = cls @ clip.visual.proj
    torch.testing.assert_close(z, native, atol=0, rtol=0)
    torch.testing.assert_close(z, projected_from_cls, atol=0, rtol=0)
    assert cls.shape == (2, 768) and patch.shape == (2, 196, 768)
    assert hidden.shape == (2, 248, 512)

    projection = VisualInputProjection(768, 512).cuda()
    records = {}
    for visual_mode, raw in [('cls', cls[:, None]), ('patch', patch)]:
        visual = projection(raw.detach().float())
        repeated_hidden = hidden[:1].detach().float().expand(2, -1, -1).contiguous()
        for readout in ('all', 'text'):
            logits, weights = joint_logits(clip.mask_net, visual, repeated_hidden, readout,
                                            checkpoint_block=False, return_pool_weights=True)
            probabilities = torch.sigmoid(logits)
            mask_difference = float((probabilities[0] - probabilities[1]).abs().mean())
            assert mask_difference > 0
            pool_visual_mass = (float(weights[:, :visual.shape[1]].sum(1).mean())
                                if readout == 'all' else 0.)
            native_score = 100 * (F.normalize(z.float(), dim=-1) *
                                  F.normalize(text.float(), dim=-1)).sum(-1)
            masked_score = 100 * (F.normalize(z.float() * probabilities, dim=-1) *
                                  F.normalize(text.float(), dim=-1)).sum(-1)
            records[f'{visual_mode}-{readout}'] = {
                'joint_length': visual.shape[1] + hidden.shape[1],
                'mask_change_same_text_other_image_mean_abs': mask_difference,
                'pool_visual_mass': pool_visual_mass,
                'pool_text_mass': 1. - pool_visual_mass,
                'native_auxiliary_score_gap_mean_abs':
                    float((native_score - masked_score).abs().mean()),
            }
    loss = sum(torch.sigmoid(joint_logits(
        clip.mask_net, projection(patch.detach().float()), hidden.detach().float(), mode,
        checkpoint_block=False)).mean() for mode in ('all', 'text'))
    loss.backward()
    assert projection.projection.weight.grad is not None
    assert all(parameter.grad is not None for parameter in clip.mask_net.parameters())
    assert cls.grad is None and patch.grad is None and hidden.grad is None

    output = {
        'passed': True,
        'init_sha256': file_sha(args.init_state),
        'strict_load': True,
        'native_image_max_abs': float((z - native).abs().max()),
        'native_projection_max_abs': float((z - projected_from_cls).abs().max()),
        'shapes': {'X_I': [2, 197, 768], 'h_cls': list(cls.shape),
                   'h_patch': list(patch.shape), 'H_text': list(hidden.shape),
                   'z': list(z.shape)},
        'projection_state_sha256': state_sha(projection),
        'projection_parameters': sum(p.numel() for p in projection.parameters()),
        'condition_hidden_detached': True,
        'projection_gradient_finite': bool(torch.isfinite(projection.projection.weight.grad).all()),
        'mask_gradients_finite': all(torch.isfinite(p.grad).all()
                                     for p in clip.mask_net.parameters()),
        'diagnostics': records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
