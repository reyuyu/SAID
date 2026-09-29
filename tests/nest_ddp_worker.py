"""torchrun two-rank reference test: gather backward + DDP average + optimizer."""
import copy
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_semantic_mask import NestedSemanticMask
from tests.test_nested_semantic_mask import TinyCLIP, global_reference


def main():
    local, rank, world = [int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE')]
    assert world == 2
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl')
    output=[]
    for arm in ('A2','A3'):
        for validity in ([1,1,1,1,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0], [0,0,0,1,0,0]):
            torch.manual_seed(19)
            clip=TinyCLIP().cuda()
            reference=copy.deepcopy(clip)
            module=NestedSemanticMask(clip,arm=arm,checkpoint_encoders=False).cuda()
            # Tiny tests stay FP32 to isolate distributed arithmetic from BF16.
            def encode(tokens):
                from model.said_cls_cvssl import said_mask_from_hidden
                t,h=clip.encode_text(tokens,return_full=True)
                m,p,l=said_mask_from_hidden(clip.mask_net,h)
                return t,m,p,l
            module.encode_view=encode
            ddp=DDP(module,device_ids=[local],output_device=local,find_unused_parameters=True,static_graph=False)
            images=torch.randn(6,8,device='cuda')
            tokens=[torch.randint(0,30,(6,6),device='cuda') for _ in range(3)]
            valid=torch.tensor(validity,device='cuda',dtype=torch.bool)
            section=slice(rank*3,(rank+1)*3)
            # Image autocast in forward: disable it by giving TinyCLIP a FP32
            # native encoder, exactly the same arithmetic used by reference.
            def image_encode(x):
                with torch.autocast('cuda',enabled=False):
                    return clip.visual(x)
            clip.encode_image=image_encode
            loss,logs=ddp(images[section],*[t[section] for t in tokens],valid[section],200)
            expected=global_reference(reference,images,tokens,valid,arm,200)
            loss.backward(); expected.backward()
            loss_error=float((logs['loss']-expected.detach()).abs())
            errors=[]
            for (name,p),(_,q) in zip(clip.named_parameters(),reference.named_parameters()):
                assert (p.grad is None)==(q.grad is None),name
                if p.grad is not None:
                    torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=5e-5,msg=name)
                    errors.append(float((p.grad-q.grad).abs().max()))
            torch.testing.assert_close(logs['loss'],expected.detach(),atol=1e-4,rtol=1e-5)
            a=torch.optim.SGD(clip.parameters(),lr=1e-4)
            b=torch.optim.SGD(reference.parameters(),lr=1e-4)
            a.step(); b.step()
            update_error=max(float((p-q).abs().max()) for p,q in zip(clip.parameters(),reference.parameters()))
            for p,q in zip(clip.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,atol=5e-6,rtol=3e-5)
            # Also exercise AdamW: isolate the mathematically zero attention
            # key-bias gradient, where FP32 cancellation / eps amplifies tiny
            # differences up to two learning rates. All other entries stay tight.
            a=torch.optim.AdamW(clip.parameters(),lr=1e-4)
            b=torch.optim.AdamW(reference.parameters(),lr=1e-4)
            a.step(); b.step()
            adam_error=0.
            for (name,p),(_,q) in zip(clip.named_parameters(),reference.named_parameters()):
                adam_error=max(adam_error,float((p-q).abs().max()))
                if name.endswith('attn.in_proj_bias'):
                    width=p.numel()//3
                    assert max(float(p.grad[width:2*width].abs().max()),float(q.grad[width:2*width].abs().max())) < 1e-4
                    torch.testing.assert_close(p[width:2*width],q[width:2*width],atol=2.01e-4,rtol=0)
                    torch.testing.assert_close(p[:width],q[:width],atol=5e-6,rtol=3e-5)
                    torch.testing.assert_close(p[2*width:],q[2*width:],atol=5e-6,rtol=3e-5)
                elif name == 'mask_net.attn_pool.attention.bias':
                    # A scalar added to every pooling softmax logit is also
                    # an exact shift-invariant null direction.
                    assert max(float(p.grad.abs().max()),float(q.grad.abs().max())) < 1e-4
                    torch.testing.assert_close(p,q,atol=2.01e-4,rtol=0)
                else:
                    torch.testing.assert_close(p,q,atol=5e-6,rtol=3e-5,msg=name)
            output.append(dict(arm=arm,validity=validity,loss_error=loss_error,
                               max_gradient_error=max(errors),max_sgd_update_error=update_error,
                               max_adamw_update_error=adam_error))
            dist.barrier()
    if rank==0:
        print(json.dumps(dict(passed=True,world_size=world,backend='nccl',cases=output),indent=2),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
