"""Infer and strictly validate the two supported Balanced CLIP backbones."""
import argparse

import torch

from model import longclip
from model.model_longclip import build_model


SPECS = {
    'ViT-B/16': dict(visual_width=768, text_width=512, embedding_dim=512,
                    patch_count=196, patch_size=16, mask_heads=8,
                    visual_layers=12,text_layers=12,visual_heads=12,vocab_size=49408),
    'ViT-L/14': dict(visual_width=1024, text_width=768, embedding_dim=768,
                    patch_count=256, patch_size=14, mask_heads=12,
                    visual_layers=24,text_layers=12,visual_heads=16,vocab_size=49408),
}
L14_URL = 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'
L14_SHA256 = L14_URL.split('/')[-2]


def infer_base_model(state):
    signature = (int(state['visual.conv1.weight'].shape[-1]),
                 int(state['visual.conv1.weight'].shape[0]),
                 int(state['text_projection'].shape[0]),
                 int(state['text_projection'].shape[1]),
                 int(state['visual.positional_embedding'].shape[0])-1)
    for name, spec in SPECS.items():
        if signature == (spec['patch_size'], spec['visual_width'], spec['text_width'],
                         spec['embedding_dim'], spec['patch_count']):
            return name
    raise ValueError(f'Unsupported Balanced CLIP tensor signature: {signature}')


def validate_backbone(model, base_model):
    spec = SPECS[base_model]
    actual = dict(visual_width=int(model.visual.proj.shape[0]),
                  text_width=int(model.text_projection.shape[0]),
                  embedding_dim=int(model.text_projection.shape[1]),
                  patch_count=int(model.visual.positional_embedding.shape[0])-1,
                  patch_size=int(model.visual.conv1.weight.shape[-1]),
                  mask_heads=model.mask_net.resblocks[0].attn.num_heads,
                  visual_layers=len(model.visual.transformer.resblocks),
                  text_layers=len(model.transformer.resblocks),
                  visual_heads=model.visual.transformer.resblocks[0].attn.num_heads,
                  vocab_size=int(model.token_embedding.weight.shape[0]))
    assert actual == spec, (base_model, actual, spec)
    assert model.visual.input_resolution == 224 and model.context_length == 248
    assert model.positional_embedding.shape == (248,spec['text_width'])
    assert model.positional_embedding_res.shape == (248,spec['text_width'])
    assert len(model.mask_net.resblocks) == 1
    assert model.transformer.resblocks[0].attn.num_heads == spec['mask_heads']
    return dict(base_model=base_model,input_resolution=224,context_length=248,
                visual_tokens=spec['patch_count']+1,**actual)


def load_native_state(state, base_model=None, device='cpu'):
    inferred = infer_base_model(state)
    if base_model not in (None,'auto'):
        assert base_model == inferred, f'Checkpoint is {inferred}, requested {base_model}'
    assert state['positional_embedding'].shape[0] == 248
    model = build_model(dict(state),load_from_clip=False,args=argparse.Namespace()).float()
    model.load_state_dict(state,strict=True)
    model.positional_embedding.requires_grad_(False)
    metadata = validate_backbone(model,inferred)
    return model.to(device).eval(),longclip._transform(224),metadata


def load_native_checkpoint(path, base_model=None, device='cpu'):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    state = payload['model'] if isinstance(payload,dict) and 'model' in payload else payload
    model, preprocess, metadata = load_native_state(state,base_model,device)
    if 'model' in payload:
        metadata.update(completed_steps=payload.get('completed_steps'),config=payload.get('config'))
        declared = (payload.get('config') or {}).get('base_model',metadata['base_model'])
        assert declared == metadata['base_model']
    return model,preprocess,metadata
