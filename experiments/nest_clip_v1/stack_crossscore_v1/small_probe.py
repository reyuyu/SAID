"""Bounded mask-only shape/compute probe; never a full-update resource gate."""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from model.model_longclip import MaskNetwork
from model.nested_fusion_mask import FusionBranch, pool_summary, stack_logits
from model.nested_semantic_mask import hard_st


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(801)
    rows=[]
    for name,fusion,length in [('S-CLS','stack_pool',1),('C-CLS','crossscore_flat',1),
                               ('S-PATCH','stack_pool',196),('C-PATCH','crossscore_flat',196)]:
        mask=MaskNetwork(512,1,8).cuda()
        branch=FusionBranch(mask.resblocks,768,512,fusion,length).cuda()
        optimizer=torch.optim.AdamW(list(mask.parameters())+list(branch.parameters()),lr=1e-4,weight_decay=0)
        hv=torch.randn(8,length,768,device='cuda')
        ht=torch.randn(4,248,512,device='cuda')
        z=torch.randn(8,512,device='cuda')
        text=torch.randn(4,512,device='cuda')
        timings=[]
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick=time.perf_counter()
            zv=branch.encode_visual(hv)
            zt=mask.resblocks(ht.permute(1,0,2)).permute(1,0,2)
            if fusion=='stack_pool':
                pool=mask.attn_pool.attention
                logits=stack_logits(pool_summary(zv,pool.weight,pool.bias),pool_summary(zt,pool.weight,pool.bias))
            else:
                logits=branch.cross_logits(branch.query(zv),branch.text_condition(zt))
            probability=logits.sigmoid()
            scores=100*(F.normalize(z[None]*hard_st(probability),dim=-1,eps=1e-6)*
                        F.normalize(text,dim=-1,eps=1e-6)[:,None]).sum(-1)
            loss=scores.square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-tick)
        rows.append(dict(group=name,image_batch=8,text_batch=4,text_slots=248,visual_slots=length,
                         width=512,rank=64,mask_only_seconds=timings,warmup_steps=1,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         readout_weights=branch.readout.weight.numel() if hasattr(branch,'readout') else None,
                         full_local_contracted_tensor_gib=(256*512*length*64*4/2**30) if fusion=='crossscore_flat' else None,
                         keep_ratio=float((probability>=.5).float().mean())))
        del mask,branch,optimizer,hv,ht,z,text,zv,zt,logits,probability,scores,loss
        torch.cuda.empty_cache()
    out=Path(__file__).resolve().parent/'evidence/small-probe.json'
    out.write_text(json.dumps(dict(scope='synthetic single-GPU mask-only shape probe; excludes CLIP, DDP and real data; not speed acceptance',
                                   groups=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
