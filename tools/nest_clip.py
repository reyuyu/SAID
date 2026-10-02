"""Prepare shared untrained initialization/data, or strictly export a bare student."""
import argparse
import json
from pathlib import Path

import torch

from model import longclip
from model.nested_semantic_mask import NestedSemanticMask
from model.nested_vcp_mask import NestedVCPMask
from model.nested_fusion_mask import NestedFusionMask
from model.balanced_hparam_search import BalancedSearch,hparams
from model.backbone import load_native_state
from train.nested_semantic_data import prepare_index, file_sha
from train.train_nested_semantic_mask import (seed_all, build_optimizer, atomic_save,
                                                auxiliary_module)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--annotation', required=True)
    q.add_argument('--index-dir', required=True)
    q.add_argument('--init-state', required=True)
    q = sub.add_parser('export')
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--expect-updates', required=True, type=int)
    q = sub.add_parser('verify-export')
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--bare', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--index-dir', required=True)
    q.add_argument('--image-root', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        seed_all(0)
        model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
        module = NestedSemanticMask(model, checkpoint_encoders=False)
        optimizer = build_optimizer(module)
        destination = Path(args.init_state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(source='OpenAI CLIP + original random MaskNetwork', seed=0,
                          original_clip_sha256=file_sha(Path.home()/'.cache/clip/ViT-B-16.pt'),
                          constructor='longclip.load_from_clip; unchanged MaskNetwork; no training',
                          frozen_position=not model.positional_embedding.requires_grad,
                          trainable_residual_position=model.positional_embedding_res.requires_grad)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         completed_steps=0, rng_cpu=torch.get_rng_state(), provenance=provenance), destination)
        meta = prepare_index(args.annotation, args.index_dir)
        result = dict(init_sha256=file_sha(destination), provenance=provenance, data=meta)
        destination.with_suffix('.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif args.command == 'export':
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert payload['completed_steps'] == args.expect_updates
        model, _, backbone = load_native_state(payload['model'], (payload.get('config') or {}).get('base_model'))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        atomic_save(model.state_dict(), args.output)
        print(json.dumps(dict(output=args.output, sha256=file_sha(args.output), strict_load=True,backbone=backbone)))
    else:
        from train.nested_semantic_data import NestedDataset
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = payload['config']
        model, _, backbone = load_native_state(payload['model'],config.get('base_model'))
        module_class = (NestedFusionMask if config.get('condition_mode') == 'dual_branch' else
                        NestedVCPMask if config.get('condition_mode') == 'vcp_mask'
                        else NestedSemanticMask)
        model_options = ({'fusion': config['fusion'], 'visual': config['visual']}
                         if config.get('condition_mode') == 'dual_branch' else {})
        if config.get('hparam_search'):
            module_class=BalancedSearch
            model_options['search_hparams']=hparams(config)
        module = module_class(model, arm=config['arm'], checkpoint_encoders=False,
                              image_chunk=config.get('image_chunk', 32),
                              text_chunk=config.get('text_chunk', 64),
                              condition_mode=config.get('condition_mode', 'text_only'),
                              shuffle_seed=config.get('shuffle_seed', 0),
                              checkpoint_pair_blocks=config.get(
                                  'checkpoint_pair_blocks', True), **model_options).eval()
        auxiliary = auxiliary_module(module)
        if auxiliary is None:
            assert payload.get('adapter') is None
        else:
            auxiliary.load_state_dict(payload['adapter'], strict=True)
        optimizer = build_optimizer(module)
        optimizer.load_state_dict(payload['optimizer'])
        assert len(optimizer.param_groups) == (4 if config.get('hparam_search') else 2 if auxiliary is None else 3)
        steps = sorted({int(v['step']) for v in optimizer.state.values()})
        assert steps == ([] if payload['completed_steps']==0 else [payload['completed_steps']])
        student, _, student_backbone = load_native_state(torch.load(args.bare, map_location='cpu', weights_only=True),backbone['base_model'])
        assert student_backbone == backbone
        student.eval()
        dataset = NestedDataset(args.index_dir, args.image_root)
        samples = [dataset[i] for i in (0,1)]
        images = torch.stack([s['image'] for s in samples])
        tokens = torch.stack([s['tokens_f'] for s in samples])
        with torch.no_grad():
            a, b = module.clip.encode_image(images), student.encode_image(images)
            c, d = module.clip.encode_text(tokens), student.encode_text(tokens)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(c,d,atol=0,rtol=0)
        result = dict(passed=True, strict_load=True, optimizer_steps=steps,
                      backbone=backbone,
                      image_max_abs=float((a-b).abs().max()), text_max_abs=float((c-d).abs().max()),
                      checkpoint_sha256=file_sha(args.checkpoint), bare_sha256=file_sha(args.bare))
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result))


if __name__ == '__main__':
    main()
