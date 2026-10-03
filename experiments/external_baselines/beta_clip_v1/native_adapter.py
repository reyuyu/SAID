"""Official beta-CLIP CLS/EOS adapter, with a complete strict fine-tuned load."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F
import torchvision.transforms as T

OFFICIAL=Path('/root/lk_projects/B-CLIP-official')
COMMIT='7be4476f84654b0febe7224d5788868d61ecae8b'
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/beta_clip_official_v1')
EXP=Path(__file__).resolve().parent


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def official_imports():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=OFFICIAL,text=True).strip()==COMMIT
    if str(OFFICIAL) not in sys.path:sys.path.insert(0,str(OFFICIAL))
    return importlib.import_module('models_tome'),importlib.import_module('tokenizer')


def official_transform():
    return T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),
        T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])


def reconstruct(checkpoint,device='cpu'):
    models,tokens=official_imports()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=payload.get('args',{})
    args=dict(vars(args) if isinstance(args,argparse.Namespace) else args)
    assert args['model']=='CLIP_VITB16_OPENAI' and args['context_length']==248
    assert args['beta']==.5 and args['max_concepts']==30 and args['fg_loss_fn']=='cls+tcil'
    mode=args['tcil_loss_mode']
    ce=(mode=='k_positives_ce' or (mode=='1_k_positives' and args.get('use_softmax_for_multi_positives') is True))
    bce=mode=='k_positives_bce'
    assert ce or bce,('Unknown fine-grained objective identity',mode)
    ctor_parameters=inspect.signature(models.CLIP.__init__).parameters
    kwargs={key:value for key,value in args.items() if key in ctor_parameters and key not in (
        'self','embed_dim','vision_width','vision_model','vocab_size','transformer_width','transformer_heads','transformer_layers','kwargs')}
    kwargs.setdefault('attn_fn','softmax');kwargs.setdefault('attn_fn_alpha',1.0);kwargs.setdefault('global_pool','')
    assert kwargs['use_text_conditioned_patches'] and kwargs['use_text_eos'] and kwargs['use_text_concepts']
    import timm
    original_create=timm.create_model
    initialization=[]
    def construct_architecture(*arguments,**options):
        # The complete official fine-tuned state will overwrite EVERY tensor.
        # Avoid a redundant network bootstrap; strict loading proves nothing is
        # retained from temporary initialization. Architecture/forward are official.
        initialization.append(dict(model_name=arguments[0],requested_pretrained=options.get('pretrained'),
                                   effective_pretrained=False))
        options['pretrained']=False
        return original_create(*arguments,**options)
    with patch.object(timm,'create_model',construct_architecture):
        model=getattr(models,args['model'])(**kwargs)
    state=payload['state_dict']
    stripped={key.removeprefix('module.'):value for key,value in state.items()}
    converted=model.convert_state_dict(stripped)
    model.resize_text_pos_embed()
    expected=set(model.state_dict());actual=set(converted)
    assert expected==actual,dict(missing=sorted(expected-actual),unexpected=sorted(actual-expected))
    result=model.load_state_dict(converted,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loaded=model.state_dict()
    assert all(torch.equal(loaded[key].cpu(),converted[key].cpu()) for key in loaded)
    shape={key:list(loaded[key].shape) for key in (
        'visual.pos_embed','visual.patch_embed.proj.weight','image_projection','text_projection','positional_embedding','positional_embedding_res')}
    assert shape['visual.patch_embed.proj.weight']==[768,3,16,16] and shape['visual.pos_embed']==[1,197,768]
    assert shape['image_projection']==[768,512] and shape['text_projection']==[512,512]
    assert shape['positional_embedding']==shape['positional_embedding_res']==[248,512]
    model=model.float().to(device).eval()
    metadata=dict(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha256(checkpoint),
        size_bytes=Path(checkpoint).stat().st_size,epoch=payload.get('epoch'),identity='CE' if ce else 'BCE',
        actual_tcil_loss_mode=mode,legacy_softmax_mode=mode=='1_k_positives',args={k:str(v) if isinstance(v,Path) else v for k,v in args.items()},
        construction_kwargs=kwargs,official_commit=COMMIT,strict_load=True,missing_keys=[],unexpected_keys=[],
        official_conversions=['Strip only leading module.','Official convert_state_dict (conditioner keys exempt)','Official resize_text_pos_embed before strict load'],
        all_loaded_tensors_equal_checkpoint=True,tensors_loaded=len(loaded),
        vision_tensors=sum(k.startswith('visual.') for k in loaded),text_tensors=sum(k.startswith(('transformer.','token_embedding.','ln_final.')) for k in loaded),
        conditioner_tensors=sum(k.startswith('text_conditioned_patches_block.') for k in loaded),
        parameter_shapes=shape,timm_bootstrap=initialization,no_temporary_initializer_weights_retained=True,
        no_training=True,readme_args_difference={'tcil_loss_mode':dict(script='k_positives_ce' if ce else 'k_positives_bce',checkpoint=mode)})
    del payload,state,stripped,converted,loaded
    return model,tokens.SimpleTokenizer(context_length=248),official_transform(),metadata


class BetaCLIPNativeAdapter:
    """Query-independent official CLS path; never execute the TCI conditioner."""
    def __init__(self,model,tokenizer):
        self.model=model;self.tokenizer=tokenizer

    def tokenize(self,texts):
        tokens=self.tokenizer(texts)
        return tokens.unsqueeze(0) if tokens.ndim==1 else tokens

    @torch.no_grad()
    def image_raw(self,images):
        # Official CLS evaluation uses the saved image-block flags, not the
        # conditioner. The CLS token remains independent of every text query.
        features=self.model.encode_image_by_block(images)
        return features[:,0] if features.ndim==3 else features

    @torch.no_grad()
    def text_raw(self,tokens):return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_image_native(self,images):return F.normalize(self.image_raw(images).float(),dim=-1)

    @torch.no_grad()
    def encode_text_native(self,texts):
        device=next(self.model.parameters()).device
        return F.normalize(self.text_raw(self.tokenize(texts).to(device)).float(),dim=-1)
