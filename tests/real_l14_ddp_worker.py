"""Small real OpenAI L14 global reference using the existing named-gradient tolerances."""
import argparse
import json
import os
from pathlib import Path
import types

import torch
import torch.distributed as dist
from torch.nn import functional as F

from model import longclip
from model.backbone import load_native_state
from model.balanced_hparam_search import BalancedSearch,hparams
from model.nested_semantic_mask import hard_st,inclusion
from tests.test_nested_fusion import explicit_logits
from train.nested_semantic_data import NestedDataset
from tests import l14_ddp_worker as reference


def same_shape_image_encoder(self,images,**options):
    outputs=[type(self).encode_image(self,images[start:start+1],**options) for start in range(len(images))]
    if isinstance(outputs[0],tuple):
        return tuple(torch.cat([o[index] for o in outputs]) for index in range(len(outputs[0])))
    return torch.cat(outputs)


def same_shape_text_encoder(self,tokens,**options):
    outputs=[type(self).encode_text(self,tokens[start:start+1],**options) for start in range(len(tokens))]
    if isinstance(outputs[0],tuple):
        return tuple(torch.cat([o[index] for o in outputs]) for index in range(len(outputs[0])))
    return torch.cat(outputs)


def same_shape_mask_blocks(self,tokens):
    return torch.cat([type(self).forward(self,tokens[:,start:start+1])
                      for start in range(tokens.shape[1])],dim=1)


def same_shape_pool(self,tokens):
    return torch.cat([type(self).forward(self,tokens[start:start+1]) for start in range(len(tokens))])


def same_shape_visual_branch(self,hidden):
    return torch.cat([type(self).encode_visual(self,hidden[start:start+1]) for start in range(len(hidden))])


def global_loss_with_one_image_forward(module,images,views,valid,completed):
    z,hidden=module.clip.encode_image(images,return_token_hidden=True)
    visual=module.fusion_branch.encode_visual(hidden[:,1:])
    uv=module.clip.mask_net.attn_pool(visual)
    terms=[];positive=[];probabilities=[]
    for tokens in views if int(valid.sum())>=2 else views[:1]:
        text,ht=module.clip.encode_text(tokens,return_full=True)
        text_tokens=module.clip.mask_net.resblocks(ht.detach().permute(1,0,2)).permute(1,0,2)
        ut=module.clip.mask_net.attn_pool(text_tokens)
        joined=torch.cat((ut[:,None].expand(-1,len(uv),-1),uv[None].expand(len(ut),-1,-1)),-1)
        gate=module.fusion_branch.gate(joined).sigmoid()
        probability=(gate*ut[:,None]+(1-gate)*uv[None]).sigmoid()
        mask=hard_st(probability)
        score=torch.cat([100*(F.normalize(z[None]*mask[q:q+1],dim=-1,eps=1e-6)*
                             F.normalize(text[q:q+1],dim=-1,eps=1e-6)[:,None]).sum(-1)
                         for q in range(len(text))])
        enabled=torch.ones_like(valid) if not terms else valid
        labels=torch.arange(len(z),device=z.device)[enabled]
        ce=sum(F.cross_entropy(score.T[q:q+1].masked_fill(~enabled[None],-torch.inf),q.view(1))+
               F.cross_entropy(score[q:q+1].masked_fill(~enabled[None],-torch.inf),q.view(1))
               for q in labels)/len(labels)
        terms.append(ce)
        positive.append(mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean())
        probabilities.append(probability.diagonal(dim1=0,dim2=1).T)
    hp=module.search_hparams
    if int(valid.sum())<2:
        return 10*terms[0]+hp['sparsity_scale']*positive[0]
    return (10/sum(hp['view_weights'])*sum(w*ce for w,ce in zip(hp['view_weights'],terms))+
            hp['sparsity_scale']*(positive[0]+2*positive[1]+2*positive[2])/3+
            hp['inclusion_max']*min(1,completed/200)*inclusion(*probabilities)[valid].mean())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--init-state',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    local,rank,world=[int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE')]
    assert world==2
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl')
    payload=torch.load(args.init_state,map_location='cpu',weights_only=False)
    dataset=NestedDataset('/root/lk_projects/SAID-nest-clip-v1/data_index',
                          '/root/lk_projects/SAID-assets/training/ShareGPT4V','random_k',0)
    samples=[]
    for index in range(100):
        sample=dataset[index]
        if bool(sample['valid']):
            samples.append(sample)
        if len(samples)==2:
            break
    assert len(samples)==2

    def make_model(search_hparams=None,image_chunk=3,text_chunk=2):
        clip,_,metadata=load_native_state(payload['model'],'ViT-L/14')
        assert metadata['visual_layers']==24 and metadata['text_layers']==12
        clip.encode_image=types.MethodType(same_shape_image_encoder,clip)
        clip.encode_text=types.MethodType(same_shape_text_encoder,clip)
        clip.mask_net.resblocks.forward=types.MethodType(same_shape_mask_blocks,clip.mask_net.resblocks)
        clip.mask_net.attn_pool.forward=types.MethodType(same_shape_pool,clip.mask_net.attn_pool)
        module=BalancedSearch(clip,search_hparams=search_hparams,fusion='balanced_stack',visual='patch',
                              checkpoint_encoders=False,image_chunk=image_chunk,text_chunk=text_chunk)
        module.fusion_branch.load_state_dict(payload['adapter'],strict=True)
        module.fusion_branch.encode_visual=types.MethodType(same_shape_visual_branch,module.fusion_branch)
        return module

    def inputs(batch,device='cpu'):
        assert batch==2
        images=torch.stack([sample['image'] for sample in samples]).to(device)
        views=[torch.stack([sample[key] for sample in samples]).to(device)
               for key in ('tokens_f','tokens_o','tokens_e')]
        return images,views

    reference.make_l14_model=make_model
    reference.l14_inputs=inputs
    reference.weighted_reference=global_loss_with_one_image_forward
    try:
        result=reference.run_case(local,rank,0,'real_l14_all_valid',1,[1,1],hparams(payload['config']))
        records=[None]*world
        dist.all_gather_object(records,result)
        if rank==0:
            output=dict(passed=True,world_size=world,base_model='ViT-L/14',input_resolution=224,
                        context_length=248,native_encoders='real OpenAI L14,24 vision/12 text blocks',
                        precision='FP32 global-reference comparison; production encoder BF16 checked in resource probe',
                        encoder_batch_shape=1,global_candidate_count=2,
                        reference_note='Identical native/mask branch batch shape isolates distributed reduction; full global loss remains two candidates',
                        reference_image_forward='Once per logical global batch, independent explicit pair masks and CE',
                        deterministic_reference_kernels=True,
                        input_source='First two valid frozen ShareGPT4V samples at epoch0, RandomK seed0',
                        sample_ids=[int(sample['sample_id']) for sample in samples],
                        synthetic_stress_failures_preserved=True,
                        initial_state=args.init_state,cases=records)
            Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps(dict(passed=True,real_l14=True)),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
