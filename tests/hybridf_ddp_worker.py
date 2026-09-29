"""Two-rank NCCL check of the complete mixed loss against global-batch reference."""
import copy
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.nested_semantic_mask import NestedSemanticMask
from model.said_cls_cvssl import said_mask_from_hidden
from tests.test_nested_semantic_mask import TinyCLIP
from tests.test_nested_hybridf import reference


def main():
    local,rank,world=[int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE')]
    assert world==2
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl')
    results=[]
    for validity in ([1,1,1,1,0,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[0,0,0,1,0,0]):
        torch.manual_seed(19);clip=TinyCLIP().cuda();ref=copy.deepcopy(clip)
        module=NestedSemanticMask(clip,arm='A3',checkpoint_encoders=False,full_native_mix=.25)
        def encode(tokens):
            t,h=clip.encode_text(tokens,return_full=True);m,p,l=said_mask_from_hidden(clip.mask_net,h)
            return t,m,p,l
        module.encode_view=encode
        def image(x):
            with torch.autocast('cuda',enabled=False):return clip.visual(x)
        clip.encode_image=image
        ddp=DDP(module,device_ids=[local],output_device=local,find_unused_parameters=True,static_graph=False)
        images=torch.randn(6,8,device='cuda');tokens=[torch.randint(0,30,(6,6),device='cuda') for _ in range(3)]
        valid=torch.tensor(validity,dtype=torch.bool,device='cuda');section=slice(rank*3,(rank+1)*3)
        loss,logs=ddp(images[section],*[t[section] for t in tokens],valid[section],200)
        expected=reference(ref,images,tokens,valid,200)
        loss.backward();expected.backward()
        torch.testing.assert_close(logs['loss'],expected.detach(),atol=1e-4,rtol=1e-5)
        errors=[]
        for (name,p),(_,q) in zip(clip.named_parameters(),ref.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=5e-5,msg=name)
            errors.append(float((p.grad-q.grad).abs().max()))
        torch.optim.SGD(clip.parameters(),lr=1e-4).step();torch.optim.SGD(ref.parameters(),lr=1e-4).step()
        sgd=max(float((p-q).abs().max()) for p,q in zip(clip.parameters(),ref.parameters()))
        assert sgd<5e-6
        torch.optim.AdamW(clip.parameters(),lr=1e-4).step();torch.optim.AdamW(ref.parameters(),lr=1e-4).step()
        for (name,p),(_,q) in zip(clip.named_parameters(),ref.named_parameters()):
            if name.endswith('attn.in_proj_bias'):
                w=p.numel()//3
                assert max(float(p.grad[w:2*w].abs().max()),float(q.grad[w:2*w].abs().max()))<1e-4
                torch.testing.assert_close(p[w:2*w],q[w:2*w],atol=2.01e-4,rtol=0)
                torch.testing.assert_close(p[:w],q[:w],atol=5e-6,rtol=3e-5)
                torch.testing.assert_close(p[2*w:],q[2*w:],atol=5e-6,rtol=3e-5)
            elif name=='mask_net.attn_pool.attention.bias':
                assert max(float(p.grad.abs().max()),float(q.grad.abs().max()))<1e-4
                torch.testing.assert_close(p,q,atol=2.01e-4,rtol=0)
            else:torch.testing.assert_close(p,q,atol=5e-6,rtol=3e-5,msg=name)
        results.append(dict(validity=validity,loss_error=float((logs['loss']-expected.detach()).abs()),
                            max_gradient_error=max(errors),max_sgd_update_error=sgd,
                            max_adamw_update_error=max(float((p-q).abs().max()) for p,q in zip(clip.parameters(),ref.parameters())),
                            native_candidates=logs['F_native_candidates'],reconstruction_error=float(logs['loss_reconstruction_abs_error'])))
        dist.barrier()
    if rank==0:print(json.dumps(dict(passed=True,world_size=2,backend='nccl',cases=results),indent=2),flush=True)
    dist.barrier();torch.cuda.synchronize();dist.destroy_process_group()


if __name__=='__main__':main()
