"""Observe untouched official entry point/evaluator; do not change retrieval math."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from .common import (CHECKPOINT, DATA, OFFICIAL, RUN, DeBiasCLIPNativeAdapter,
                     dump, official_import, official_runtime, reconstruct)


def metric_standard(raw):
    return {d: {f'R@{k}': float(raw[f'{prefix}_R@{k}']) for k in (1, 5, 10)}
            for d, prefix in [('I2T', 'image_to_text'), ('T2I', 'text_to_image')]}


def norms(features):
    x = features.float().norm(dim=-1)
    return {'shape': list(features.shape), 'dtype': str(features.dtype),
            'norm_mean': float(x.mean()), 'norm_std': float(x.std()),
            'norm_min': float(x.min()), 'norm_max': float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urban-only', action='store_true')
    ap.add_argument('--precision', choices=['amp', 'fp32'], default='amp')
    a = ap.parse_args()
    official_runtime()
    train = official_import('training.train')
    params = official_import('training.params_new')
    main_module = official_import('main')
    data_module = official_import('training.data')
    from open_clip import get_input_dtype
    from open_clip_train.precision import get_autocast
    tag = ('urban-' if a.urban_only else 'base-full-')+a.precision
    destination = RUN / tag
    destination.mkdir(exist_ok=True)
    argv = ['--data-root-dir', str(DATA), '--csv-img-base-path', str(DATA),
            '--force-custom-clip', '--eval-data', 'base_full', '--model', 'ViT-B-16-longclip',
            '--pretrained', str(CHECKPOINT), '--force-quick-gelu',
            '--logs', str(destination/'logs'), '--name', tag]
    if a.precision != 'amp':
        argv += ['--precision', a.precision]
    captured = {'datasets': {}, 'model': None}
    current = {}
    original_retrieval = train.retrieval_on_split
    original_scores = train.compute_similarity_scores_original_clip
    original_create = main_module.create_model_and_transforms
    factory = official_import('local_clip.factory')
    raw_payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    raw_state = {k.removeprefix('module.'): v for k, v in raw_payload['state_dict'].items()}
    del raw_payload
    original_load = torch.nn.Module.load_state_dict
    loading = []
    def spy_load(module, state, strict=True, *args, **kwargs):
        assert strict is True
        result = original_load(module, state, strict=strict, *args, **kwargs)
        actual = module.state_dict()
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.equal(actual[k].cpu(), v.cpu()) for k, v in state.items())
        differences = [{'key': k, 'max_abs': float((actual[k].cpu()-v.cpu()).abs().max())}
                       for k, v in raw_state.items() if not torch.equal(actual[k].cpu(), v.cpu())]
        loading.append({'strict': True, 'tensor_count': len(state), 'missing_keys': [],
                        'unexpected_keys': [], 'equal_effective_official_state': True,
                        'raw_checkpoint_mismatches': differences})
        return result
    def observe_model(model, tokenizer):
        captured['model'] = model
        captured['tokenizer'] = tokenizer
        adapter = DeBiasCLIPNativeAdapter(model, tokenizer)
        original_forward = model.forward
        def forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if current:
                for kwarg, key in [('image', 'image_features'), ('text', 'text_features')]:
                    if kwargs.get(kwarg) is not None:
                        value = out[key].detach()
                        current[key].append(value.cpu())
                        if current['dataset'] == 'Urban1k':
                            alt = (adapter.encode_image_native(kwargs[kwarg]) if kwarg == 'image'
                                   else adapter.encode_text_native(kwargs[kwarg]))
                            current[key+'_errors'].append(float((alt-value).abs().max()))
                current['logit_scale'] = float(out['logit_scale'])
            return out
        model.forward = forward
    def construct(*args, **kwargs):
        result = original_create(*args, **kwargs)
        tokenizer = factory.get_tokenizer('ViT-B-16-longclip', is_siglip='none')
        observe_model(result[0], tokenizer)
        captured['transform'] = repr(result[2])
        return result
    def scores(*args, **kwargs):
        result = original_scores(*args, **kwargs)
        if current['dataset'] == 'Urban1k':
            current['official_scores'] = result[0].detach()
            current['image_ids'] = result[1].detach()
        return result
    def retrieval(keyword, model, txt_loader, img_loader, img2txt, txt2img, args, epoch,
                  metrics, device, input_dtype, autocast):
        root = Path(txt_loader.dataset.root_dir).name
        name = {'coco': 'COCO', 'flickr30k-images': 'Flickr-full', 'urban1k': 'Urban1k',
                'docci': 'DOCCI', 'dci': 'DCI-full'}[root]
        current.clear()
        current.update(dataset=name, image_features=[], text_features=[],
                       image_features_errors=[], text_features_errors=[])
        result = original_retrieval(keyword, model, txt_loader, img_loader, img2txt, txt2img,
                                    args, epoch, metrics, device, input_dtype, autocast)
        im = torch.cat(current['image_features'])
        tx = torch.cat(current['text_features'])
        record = {'dataset': name, 'metrics': metric_standard(result), 'official_raw_metrics': result,
                  'n_images': len(im), 'n_captions': len(tx), 'image_embedding': norms(im),
                  'text_embedding': norms(tx), 'similarity_shape': [len(im), len(tx)],
                  'logit_scale_exp': current['logit_scale'], 'precision': args.precision,
                  'image_encoding_batch': img_loader.batch_size, 'text_encoding_batch': txt_loader.batch_size,
                  'official_dataset_type': type(txt_loader.dataset).__name__,
                  'official_evaluator': 'training.train.retrieval_on_split / compute_retrieval'}
        if name == 'Urban1k':
            score = current['official_scores']
            plain = im @ tx.T
            scaled = current['logit_scale'] * im @ tx.T
            score_difference = float((scaled-score).abs().max())
            assert score_difference == 0
            adapter_errors = {k: max(current[k+'_errors']) for k in ['image_features', 'text_features']}
            assert max(adapter_errors.values()) == 0
            _, _, mapped_i, mapped_t = data_module.get_urban1k_dataset(args, None, tokenizer=captured['tokenizer'], root_dir=str(DATA/'urban1k'))
            image_ids = current['image_ids']
            cap_ids = torch.arange(len(tx))
            mapped_i, mapped_t = train.remap_indices(image_ids, cap_ids, mapped_i, mapped_t)
            direct = train.compute_retrieval(scaled, mapped_t, mapped_i)
            assert metric_standard(direct) == record['metrics']
            plain_raw = train.compute_retrieval(plain, mapped_t, mapped_i)
            record['embedding_audit'] = {'adapter_max_abs': adapter_errors,
                'direct_scaled_score_max_abs': score_difference,
                'direct_scaled_recall_equal_official': True,
                'plain_unscaled_recall': metric_standard(plain_raw),
                'plain_vs_scaled_top1_i2t_mismatches': int((plain.argmax(1) != scaled.argmax(1)).sum()),
                'plain_vs_scaled_top1_t2i_mismatches': int((plain.argmax(0) != scaled.argmax(0)).sum()),
                'plain_vs_scaled_note': 'Global positive scale is rank-invariant in exact arithmetic; finite-precision ties are measured, not assumed.'}
            records = txt_loader.dataset.old_data
            torch.save({'image_features': im, 'text_features': tx,
                        'image_ids': image_ids, 'raw_rows': records,
                        'official_scaled_scores': score}, destination/'urban-embeddings.pt')
            r1 = record['metrics']['T2I']['R@1']*100
            delta = r1-93.0
            record.update(published_T2I_R1_percent=93.0, delta_to_published_T2I_pp=delta,
                correct_T2I_queries=round(record['metrics']['T2I']['R@1']*len(tx)),
                published_equivalent_correct_T2I_queries=930,
                delta_correct_T2I_queries=round(record['metrics']['T2I']['R@1']*len(tx))-930,
                classification='REPRODUCED' if abs(delta)<.05 else 'NEAR-REPRODUCED' if abs(delta)<=.200001 else 'NOT REPRODUCED')
        captured['datasets'][name] = record
        dump(destination/(name+'.json'), record)
        print(json.dumps({'dataset': name, 'metrics': record['metrics'],
                          'counts': [len(im), len(tx)], 'precision': args.precision}), flush=True)
        current.clear()
        return result
    with patch.object(torch.nn.Module, 'load_state_dict', spy_load), \
         patch.object(main_module, 'create_model_and_transforms', construct), \
         patch.object(train, 'retrieval_on_split', retrieval), \
         patch.object(train, 'compute_similarity_scores_original_clip', scores):
        if a.urban_only:
            args = params.parse_args(argv)
            args.device = 'cuda:0'
            args.rank = 0
            args.world_size = 1
            args.distributed = False
            model, _, transform = original_create('ViT-B-16-longclip', str(CHECKPOINT),
                precision=a.precision, device=args.device, force_custom_clip=True,
                force_quick_gelu=True, output_dict=True, longclip_keep_length=20, longclip_pca_dim=32)
            model.eval()
            tokenizer = factory.get_tokenizer('ViT-B-16-longclip', is_siglip='none')
            observe_model(model, tokenizer)
            captured['transform'] = repr(transform)
            txt, img, img2txt, txt2img = data_module.get_urban1k_dataset(args, transform, tokenizer,
                                                                      root_dir=str(DATA/'urban1k'))
            train.retrieval_on_split('', model, txt.dataloader, img.dataloader, img2txt, txt2img,
                args, 0, {}, torch.device(args.device), get_input_dtype(args.precision), get_autocast(args.precision))
        else:
            main_module.main(argv)
    result = {'status': 'COMPLETE', 'mode': tag, 'argv': argv,
              'datasets': captured['datasets'], 'strict_loading': loading,
              'preprocess': captured['transform'], 'official_sources_modified': False,
              'no_training_or_finetuning': True}
    dump(destination/'results.json', result)
    print(json.dumps({'status': 'COMPLETE', 'mode': tag, 'datasets': list(result['datasets'])}), flush=True)


if __name__ == '__main__':
    main()
