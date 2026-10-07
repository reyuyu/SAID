"""Read-only step500 native-backbone gradient spot-check on8 frozen global batches."""
import argparse
import hashlib
import json
import os
import statistics

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch, hparams
from model.nested_fusion_mask import fusion_view_terms
from model.nested_semantic_mask import gather, global_sum
from recovery.nested_detail500 import RUN, EXP, IMAGES, WEIGHTS
from recovery.s02_full_local_data import FullLocalDataset
from recovery.s02_nfs500 import dump, sha
from recovery.s02_local500 import INDEX
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import CappedSampler, seed_all, setup


def gradient_vector(parameters):
    flat = torch.cat([p.grad.detach().float().flatten() if p.grad is not None else
                      torch.zeros_like(p,dtype=torch.float32).flatten() for p in parameters])
    dist.all_reduce(flat); flat /= dist.get_world_size()
    assert torch.isfinite(flat).all(), 'Nonfinite diagnostic gradient'
    return flat


def cosine(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))


def main():
    seed_all(0); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    rank,local,world,_ = setup(); assert world == 4
    checkpoint = RUN/'step500/step000500.pt'
    checkpoint_sha = sha(checkpoint)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['global_step'] == payload['completed_steps'] == 500
    cfg = payload['config']
    clip,_ = longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(payload['model'],strict=True)
    module = BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),
        inclusion_hierarchy=cfg['inclusion_hierarchy'],hns_enabled=cfg.get('hns_enabled',False),
        hns_beta=cfg.get('hns_beta',[2.,2.]),
        fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=cfg['checkpoint_encoders'],
        image_chunk=cfg['image_chunk'],text_chunk=cfg['text_chunk'],
        shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=cfg['checkpoint_pair_blocks'])
    module.fusion_branch.load_state_dict(payload['adapter'],strict=True)
    del payload
    module = module.cuda(local).train()
    backbone = module.optimizer_groups()[0]['params']
    dataset = FullLocalDataset(INDEX,IMAGES,'nested_detail',0)
    sampler = DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
    sampler.set_epoch(0)
    loader = DataLoader(dataset,batch_size=256,sampler=CappedSampler(sampler,8*256),collate_fn=collate,
        num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(0),timeout=60)
    results = []
    for batch_number,batch in enumerate(loader,1):
        ids = [None]*4; dist.all_gather_object(ids,batch['sample_id'].tolist())
        vectors = {}; losses = {}
        images = batch['image'].cuda(non_blocking=True)
        for label,key in [('F','tokens_f'),('Dall','tokens_o'),('Ds','tokens_e')]:
            module.zero_grad(set_to_none=True)
            tokens = batch[key].cuda(non_blocking=True)
            valid = torch.ones(256,dtype=torch.bool,device='cuda') if label=='F' else batch['valid'].cuda()
            enabled_global = gather(valid,False)
            z,visual = module.encode_visual(images)
            text,condition = module.encode_view(tokens)
            terms = fusion_view_terms(module,z,text,visual,condition,valid,enabled_global,
                gather(z),tuple(gather(x) for x in visual),False)
            raw_ce,logs = terms[0],terms[-1]
            assert torch.isfinite(raw_ce), 'Nonfinite diagnostic CE'
            raw_ce.backward()
            vectors[label] = gradient_vector(backbone)
            losses[label] = float(logs['i2t']+logs['t2i'])
            del terms,raw_ce,z,visual,text,condition
        module.zero_grad(set_to_none=True)
        tokens = batch['tokens_f'].cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            zi = module.clip.encode_image_with_checkpoint(images)
            zt = module.clip.encode_text(tokens)
        zi,zt = F.normalize(zi.float(),dim=-1),F.normalize(zt.float(),dim=-1)
        logits_i,logits_t = 100*zi@gather(zt).T,100*zt@gather(zi).T
        labels = rank*256+torch.arange(256,device='cuda')
        native_loss = F.cross_entropy(logits_i,labels)+F.cross_entropy(logits_t,labels)
        native_loss.backward(); native = gradient_vector(backbone)
        weighted = sum((10/3*WEIGHTS[v])*g for v,g in vectors.items())
        values = dict(batch=batch_number,global_sample_ids_sha256=hashlib.sha256(
            json.dumps(ids,separators=(',',':')).encode()).hexdigest(),raw_combined_CE=losses,
            gradient_norms={v:float(g.norm()) for v,g in vectors.items()},
            native_CE=float(global_sum(native_loss)/world),
            native_gradient_norm=float(native.norm()),weighted_alignment_gradient_norm=float(weighted.norm()),
            cosine_to_native={v:cosine(g,native) for v,g in vectors.items()},
            weighted_alignment_cosine_to_native=cosine(weighted,native),
            all_gradients_finite=True)
        results.append(values)
        if rank == 0:
            print(json.dumps(values),flush=True)
        del vectors,native,weighted,zi,zt,logits_i,logits_t,native_loss
    assert len(results)==8 and sha(checkpoint)==checkpoint_sha
    if rank == 0:
        dump(EXP/'GRADIENT_SPOTCHECK.json',dict(passed=True,checkpoint_sha256=checkpoint_sha,checkpoint_unchanged=True,
            fixed_global_batches=8,global_batch_size=1024,selection='First8 seed0 epoch0 frozen batches at fixed step500 weights',
            no_optimizer_updates=True,read_only_separate_process=True,
            backbone_scope='Exact production backbone optimizer group; zero for unused parameters',
            gradients='Raw directional-summed per-view CE gradients, DDP-equivalent all-reduce/4',
            native_reference='Native F CLIP CE, normalized embeddings, fixed temperature100, both directions',
            mean_gradient_norms={v:statistics.fmean(r['gradient_norms'][v] for r in results) for v in WEIGHTS},
            mean_cosine_to_native={v:statistics.fmean(r['cosine_to_native'][v] for r in results) for v in WEIGHTS},
            mean_weighted_alignment_cosine_to_native=statistics.fmean(r['weighted_alignment_cosine_to_native'] for r in results),
            batches=results))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
