"""Two-rank fusion reference: explicit masks, named gradients and real AdamW."""
import argparse
import copy
import json
import os
from pathlib import Path
import types

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.nested_fusion_mask import pool_summary, fusion_scores
from model.nested_semantic_mask import gather, hard_st
from tests.test_nested_fusion import make_model, reference_loss, explicit_inputs, explicit_logits
from torch.nn import functional as F
from train.train_nested_semantic_mask import build_optimizer, learning_rates


def fp32_encoders(module):
    def encode_visual(self, images):
        with torch.autocast('cuda', enabled=False):
            z, hidden = self.clip.encode_image(images.float(), return_token_hidden=True)
            zv = self.fusion_branch.encode_visual(hidden[:, :1] if self.visual == 'cls' else hidden[:, 1:])
            if self.fusion in ('stack_pool', 'balanced_stack'):
                pool = self.clip.mask_net.attn_pool.attention
                condition = pool_summary(zv, pool.weight, pool.bias)
            else:
                condition = (self.fusion_branch.projected_queries(zv),)
        return z, condition
    def encode_view(self, tokens):
        with torch.autocast('cuda', enabled=False):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
            zt = self.clip.mask_net.resblocks(hidden.detach().permute(1,0,2)).permute(1,0,2)
            if self.fusion in ('stack_pool', 'balanced_stack'):
                pool = self.clip.mask_net.attn_pool.attention
                condition = pool_summary(zt, pool.weight, pool.bias)
            else:
                keys = self.fusion_branch.projected_keys(zt)
                condition = (self.fusion_branch.contract_keys(keys), keys)
        return text, condition
    module.encode_visual = types.MethodType(encode_visual,module)
    module.encode_view = types.MethodType(encode_view,module)


def theoretical_zero_mask(name, gradient, visual):
    result = torch.zeros_like(gradient,dtype=torch.bool)
    if name.endswith('attn_pool.attention.bias'):
        result.fill_(True)
    elif name.endswith('attn.in_proj_bias'):
        width=gradient.numel()//3
        result[width:2*width]=True
    if visual == 'cls' and 'fusion_branch.visual_blocks.' in name and name.endswith(
            ('attn.in_proj_weight', 'attn.in_proj_bias')):
        # With one token, softmax is exactly [1]; visual Q/K cannot affect the output.
        result[:2 * gradient.shape[0] // 3] = True
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--new-arms', action='store_true')
    args=parser.parse_args()
    local,rank,world=[int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE')]
    assert world==2
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    cases=[('all_valid',2,[1,1,1,1]),('rank1_zero_valid',2,[1,1,0,0]),
           ('V1',2,[1,0,0,0]),('V0',2,[0,0,0,0]),('tail',1,[1,1])]
    results=[]
    arms = ([('balanced_stack', 'patch'), ('cosine_crossscore', 'cls')] if args.new_arms else
            [('stack_pool','cls'),('crossscore_flat','cls'),('stack_pool','patch'),('crossscore_flat','patch')])
    for fusion,visual in arms:
        for index,(case,batch,validity) in enumerate(cases):
            if rank == 0:
                print(json.dumps(dict(testing=fusion,visual=visual,case=case)),flush=True)
            torch.manual_seed(751+index)
            # 32 channels avoid the frequent all-closed degeneracy of an 8-channel fixture.
            # All production initialization rules, including readout bias zero, remain intact.
            module=make_model(fusion,visual,width=32).cuda()
            reference=copy.deepcopy(module)
            fp32_encoders(module)
            fp32_encoders(reference)
            ddp=DDP(module,device_ids=[local],output_device=local,
                    find_unused_parameters=True,static_graph=False)
            images=torch.randn(batch*world,8,device='cuda')
            tokens=[torch.randint(0,31,(batch*world,6),device='cuda') for _ in range(3)]
            valid=torch.tensor(validity,device='cuda',dtype=torch.bool)
            sl=slice(rank*batch,(rank+1)*batch)
            loss,logs=ddp(images[sl],*[x[sl] for x in tokens],valid[sl],61)
            expected=reference_loss(reference,images,tokens,valid,61)
            with torch.no_grad():
                z,vv=module.encode_visual(images[sl])
                t,tt=module.encode_view(tokens[0][sl])
                zg=gather(z,False)
                vg=tuple(gather(x,False) for x in vv)
                score,_,_=fusion_scores(module,zg,t,vg,tt,torch.ones_like(valid),
                                        torch.ones_like(valid[sl]))
                score=gather(score,False)
                rz,rt,rv,rtt=explicit_inputs(reference,images,tokens[0])
                rm=hard_st(explicit_logits(reference,rv,rtt).sigmoid())
                score_reference=100*(F.normalize(rz[None]*rm,dim=-1,eps=1e-6)*
                                     F.normalize(rt,dim=-1,eps=1e-6)[:,None]).sum(-1)
                torch.testing.assert_close(score,score_reference,atol=8e-5,rtol=3e-6)
                score_error=float((score-score_reference).abs().max())
            loss.backward()
            expected.backward()
            actual_named=dict(module.named_parameters())
            reference_named=dict(reference.named_parameters())
            assert actual_named.keys()==reference_named.keys()
            gradients=[]
            for name,actual in actual_named.items():
                target=reference_named[name]
                assert (actual.grad is None)==(target.grad is None),name
                if actual.grad is None:
                    gradients.append(dict(name=name,none=True,max_abs=0))
                    continue
                diff=(actual.grad-target.grad).abs()
                idx=tuple(int(v) for v in torch.unravel_index(diff.argmax(),diff.shape))
                gradients.append(dict(name=name,none=False,max_abs=float(diff.max()),index=idx,
                                      actual_norm=float(actual.grad.norm()),reference_norm=float(target.grad.norm())))
                try:
                    torch.testing.assert_close(actual.grad,target.grad,atol=1.5e-3,rtol=1e-4)
                except AssertionError:
                    print(json.dumps(dict(fusion=fusion,visual=visual,case=case,name=name,
                                          diagnostics=gradients[-1],actual_at_worst=float(actual.grad[idx]),
                                          reference_at_worst=float(target.grad[idx])),indent=2),flush=True)
                    raise
            torch.testing.assert_close(logs['loss'],expected.detach(),atol=3e-4,rtol=3e-5)
            optimizers=[build_optimizer(module),build_optimizer(reference)]
            for optimizer in optimizers:
                for group,rate in zip(optimizer.param_groups,learning_rates(61,3651,True)):
                    group['lr']=rate
                optimizer.step()
            updates=[]
            for name,actual in actual_named.items():
                target=reference_named[name]
                diff=(actual-target).abs()
                exempt=torch.zeros_like(diff,dtype=torch.bool)
                theoretical=torch.zeros_like(diff,dtype=torch.bool)
                if actual.grad is not None:
                    theoretical=theoretical_zero_mask(name,actual.grad,visual)
                    # Retain the prior VCP criterion for AdamW's near-zero sensitivity,
                    # and additionally check the first-update analytic amplification.
                    group=next(g for g in optimizers[0].param_groups if any(p is actual for p in g['params']))
                    ga,gb=actual.grad,target.grad
                    predicted=(group['lr']*(ga/(ga.abs()+1e-8)-gb/(gb.abs()+1e-8))).abs()
                    rounding=2*torch.finfo(actual.dtype).eps*torch.maximum(actual.abs(),target.abs()).clamp_min(.01)
                    exempt=((ga.abs()<2e-5) & (gb.abs()<2e-5) & ((predicted-diff).abs()<=rounding))
                unexplained=diff.masked_fill(exempt,0)
                flat=int(diff.argmax())
                idx=tuple(int(v) for v in torch.unravel_index(diff.argmax(),diff.shape))
                updates.append(dict(name=name,max_abs=float(diff.max()),unexplained_max=float(unexplained.max()),
                                    index=idx,near_zero_gradient=bool(exempt.flatten()[flat]),
                                    theoretical_null_direction=bool(theoretical.flatten()[flat]),
                                    actual_gradient=None if actual.grad is None else float(actual.grad.flatten()[flat]),
                                    reference_gradient=None if target.grad is None else float(target.grad.flatten()[flat])))
                if float(unexplained.max())>2e-6:
                    print(json.dumps(dict(fusion=fusion,visual=visual,case=case,name=name,
                                          gradients=gradients,updates=updates),indent=2),flush=True)
                    raise AssertionError(f'Unexplained AdamW difference {name}: {float(unexplained.max())}')
            results.append(dict(fusion=fusion,visual=visual,case=case,loss_error=float((logs['loss']-expected.detach()).abs()),
                                score_error=score_error,
                                test_width=32,test_readout_bias=0.,
                                gradients=gradients,updates=updates))
            dist.barrier()
            del ddp,module,reference,optimizers
    if rank==0:
        Path(args.output).write_text(json.dumps(dict(passed=True,world_size=2,cases=results),indent=2)+'\n')
        print(json.dumps(dict(passed=True,cases=len(results))),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
