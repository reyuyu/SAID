"""HDSM v0.1 trainer: one shared decoder, one 2048-d Hard-ST gate, four fixed losses."""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model.hd_smartmask import *
from model import longclip
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate

TOKEN_CONTEXT = 248
PRECISION = "fp32 master; bf16 autocast for CLIP encoder; latent/gate/decoder/normalize/score/loss fp32"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0)); local = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    return rank, local, world

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def load_initial(model, path, rank):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if isinstance(payload,dict) and "model" in payload: payload=payload["model"]
    target=set(model.state_dict()); mapped={}
    for key,val in payload.items():
        for candidate in (key, key[5:] if key.startswith("clip.") else "", "clip."+key):
            if candidate in target: mapped[candidate]=val; break
    if not mapped: raise RuntimeError("shared init matched no CLIP tensors")
    missing,unexpected=model.load_state_dict(mapped,strict=False)
    bad=[k for k in missing if not k.startswith("mask_net")]
    if bad or unexpected: raise RuntimeError(f"shared init mismatch missing={bad} unexpected={unexpected}")
    if rank==0: print(f"INIT_STATE_LOADED tensors={len(mapped)} sha256={sha256_file(path)}",flush=True)

class HDSMTrainModule(torch.nn.Module):
    def __init__(self,clip,latent_encoder,decoder,gate,image_chunk=16,text_chunk=32,checkpoint_blocks=True,rank=0):
        super().__init__(); self.clip=clip; self.latent_encoder=latent_encoder; self.decoder=decoder; self.gate=gate
        self.image_chunk=image_chunk; self.text_chunk=text_chunk; self.checkpoint_blocks=checkpoint_blocks; self.rank=rank
    def encode(self,images,text,amp_enabled=True):
        device_type="cuda" if images.is_cuda else "cpu"
        with torch.autocast(device_type=device_type,dtype=torch.bfloat16,enabled=amp_enabled):
            g_raw,t_raw,hidden=self.clip_forward(images,text)
        with torch.autocast(device_type=device_type,enabled=False):
            g=native_unit(g_raw); t=F.normalize(t_raw.float(),dim=-1,eps=NORM_EPS); z=self.latent_encoder(g.float()); full_raw=self.decoder(z); f=F.normalize(full_raw,dim=-1,eps=NORM_EPS); masks,p=self.gate(hidden.float())
        return {"g":g,"t":t,"z":z.float(),"full_raw":full_raw.float(),"f":f,"masks":masks.float(),"p":p.float()}
    def clip_forward(self,images,text):
        g_raw=self.clip.encode_image(images)
        t_raw,hidden=self.clip.encode_text(text,return_full=True)
        return g_raw,t_raw,hidden
    def forward(self,images,text,amp_enabled=True):
        e=self.encode(images,text,amp_enabled); local=e["g"].shape[0]
        if dist.is_initialized():
            sizes=[torch.zeros(1,dtype=torch.long,device=e["g"].device) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes,torch.tensor([local],device=e["g"].device))
            if len({int(x) for x in sizes} ) != 1: raise RuntimeError("uneven local batch")
        tall=gather_rows(e["t"]); mall=gather_rows(e["masks"])
        q=conditional_scores(e["z"],self.decoder,mall,tall,self.image_chunk,self.text_chunk,self.checkpoint_blocks)
        qall=gather_rows(q)
        rank=self.rank; cols=slice(rank*local,(rank+1)*local); targets=global_targets(local,rank,q.device)
        masked_pos=positive_masked_units(e["z"],self.decoder,e["masks"])
        # q has local image rows x global text columns; qall has global image rows. The local
        # text-anchor slice is the corresponding global column range, so targets remain global
        # image indices (this is the same autograd-aware DDP route used by the prior experiments).
        align_i2t=torch.nn.functional.cross_entropy(q,targets)
        align_t2i=torch.nn.functional.cross_entropy(qall[:,cols].t(),targets)
        align=align_i2t+align_t2i
        rec=(e["f"]-e["g"].detach()).square().sum(-1).mean()
        cons=(1.0-(masked_pos*e["f"].detach()).sum(-1)).mean()
        sparse=e["masks"].mean()
        terms={"align_i2t":align_i2t,"align_t2i":align_t2i,"align":align,"rec":rec,"cons":cons,"sparse":sparse,
               "total":LAMBDA_ALIGN*align+rec+LAMBDA_CONS*cons+LAMBDA_SPARSE*sparse,
               "weighted_align":LAMBDA_ALIGN*align,"weighted_rec":rec,"weighted_cons":LAMBDA_CONS*cons,"weighted_sparse":LAMBDA_SPARSE*sparse}
        with torch.no_grad():
            pos=q.gather(1,(targets-rank*local).view(-1,1)).squeeze(1)
            neg=q.clone(); neg.scatter_(1,(targets-rank*local).view(-1,1),float("-inf"))
            terms["stats"]={"positive_mean":float(pos.mean()),"strongest_negative_mean":float(neg.max(1).values.mean()),"max_margin_mean":float((pos-neg.max(1).values).mean()),"top1":float((q.argmax(1)==(targets-rank*local)).float().mean()),"global_batch":int(tall.shape[0]),"local_batch":local}
        return terms|{"encoded":e}

def grad_finite(module):
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters() if p.requires_grad)

def save_ckpt(path,module,optimizer,scheduler_state,step,cursor,config,provenance):
    payload=checkpoint_payload(module.clip,module.latent_encoder,module.decoder,module.gate,optimizer,step,scheduler_state,config,cursor,{"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state() if torch.cuda.is_available() else None},provenance)
    atomic_torch_save(payload,path)

def set_group_lrs(optimizer, step, horizon):
    """Apply the specified warmup/cosine schedule independently to each group."""
    specs=((1e-6,200),(1e-3,0),(1e-4,0))
    for group,(base,warmup) in zip(optimizer.param_groups,specs):
        if step < warmup:
            lr=base*float(step+1)/float(warmup)
        else:
            progress=min(1.0,max(0.0,float(step-warmup)/float(max(1,horizon-warmup))))
            lr=base*0.5*(1.0+math.cos(math.pi*progress))
        group["lr"]=lr

def main():
    p=argparse.ArgumentParser(); p.add_argument("--init-state",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--max-steps",type=int,default=500); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--epochs",type=int,default=3); p.add_argument("--seed",type=int,default=0); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--log-every",type=int,default=10); p.add_argument("--heavy-log-every",type=int,default=25); p.add_argument("--image-chunk",type=int,default=16); p.add_argument("--text-chunk",type=int,default=32); p.add_argument("--no-checkpoint-blocks",action="store_true"); p.add_argument("--base-model",default="ViT-B/16"); args=p.parse_args()
    seed_all(args.seed); rank,local,world=setup_dist(); device=torch.device("cuda",local); os.makedirs(args.output_dir,exist_ok=True)
    clip,_=longclip.load_from_clip(args.base_model,device="cpu",download_root=None,args=args); clip.train(); clip=clip.to(device); load_initial(clip,args.init_state,rank); clip.logit_scale.requires_grad_(False)
    with isolated_rng(GATE_SEED,device):
        latent=nn.Sequential(nn.LayerNorm(CLIP_DIM),nn.Linear(CLIP_DIM,LATENT_DIM),nn.ReLU()).to(device); decoder=nn.Linear(LATENT_DIM,CLIP_DIM,bias=False).to(device); gate=HighDimGate(device=device).to(device)
    nn.init.ones_(latent[0].weight); nn.init.zeros_(latent[0].bias); nn.init.xavier_uniform_(latent[1].weight); nn.init.zeros_(latent[1].bias); nn.init.xavier_uniform_(decoder.weight)
    module=HDSMTrainModule(clip,latent,decoder,gate,args.image_chunk,args.text_chunk,not args.no_checkpoint_blocks,rank).to(device)
    if world>1: module=torch.nn.parallel.DistributedDataParallel(module,device_ids=[local],output_device=local,find_unused_parameters=True)
    raw=module.module if hasattr(module,"module") else module; optimizer,groups=build_optimizer(raw.clip,raw.latent_encoder,raw.decoder,raw.gate)
    dataset=Share4VCvsslDataset(seed=args.seed,augment_view_b=False,strict_manifest=os.environ.get("SHARE4V_FULL_AUDIT")); sampler=torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True,seed=args.seed) if world>1 else torch.utils.data.RandomSampler(dataset)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=args.num_workers,pin_memory=True,collate_fn=cvssl_collate,drop_last=True); horizon=args.epochs*len(loader)
    config=config_dict()|{"batch_size_per_gpu":args.batch_size,"world_size":world,"global_batch":args.batch_size*world,"loader_batches":len(loader),"lr_horizon_steps":horizon,"image_chunk":args.image_chunk,"text_chunk":args.text_chunk,"checkpoint_blocks":not args.no_checkpoint_blocks,"precision":PRECISION,"init_state":args.init_state,"init_file_sha256":sha256_file(args.init_state),"base_model":args.base_model,"gate":raw.gate.config(),"latent_encoder":"LayerNorm(512)->Linear(512,2048)->ReLU","decoder":"Linear(2048,512,bias=False)"}
    log_path=os.path.join(args.output_dir,"salu_log.jsonl"); step=0; start=time.time()
    if rank==0: save_ckpt(os.path.join(args.output_dir,"hdsm_HDSM_V01_step000000.pt"),raw,optimizer,{"horizon":horizon,"step":0},0,{"epoch":0,"step_in_epoch":-1},config,{"git_head":"unknown","run":"HDSM_V01"})
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        if world>1: sampler.set_epoch(epoch)
        for batch_index,batch in enumerate(loader):
            if step>=args.max_steps: break
            set_group_lrs(optimizer,step,horizon); optimizer.zero_grad(set_to_none=True); out=module(batch["image_a"].to(device),longclip.tokenize(batch["caption_said"],truncate=True).to(device),True); loss=out["total"]
            if not torch.isfinite(loss): raise RuntimeError("non-finite HDSM loss")
            loss.backward()
            finite_local=grad_finite(raw)
            if dist.is_initialized():
                finite_flag=torch.tensor([0 if finite_local else 1],device=device,dtype=torch.int32)
                dist.all_reduce(finite_flag,op=dist.ReduceOp.MAX)
                if int(finite_flag.item()): raise RuntimeError("non-finite HDSM gradient on at least one rank; refusing optimizer step")
            elif not finite_local: raise RuntimeError("non-finite HDSM gradient; refusing optimizer step")
            optimizer.step(); step+=1
            if rank==0 and (step%args.log_every==0 or step==1):
                e=out["encoded"]; masked_raw=raw.decoder(e["z"]*e["masks"]); record={"completed_steps":step,"arm":ARM,"objective":OBJECTIVE,"epoch":epoch,"step_in_epoch":batch_index,"align_i2t":float(out["align_i2t"]),"align_t2i":float(out["align_t2i"]),"align":float(out["align"]),"rec":float(out["rec"]),"cons":float(out["cons"]),"sparse":float(out["sparse"]),"weighted_align":float(out["weighted_align"]),"weighted_rec":float(out["weighted_rec"]),"weighted_cons":float(out["weighted_cons"]),"weighted_sparse":float(out["weighted_sparse"]),"total":float(out["total"]),"lr":optimizer.param_groups[0]["lr"],"gate_lr":optimizer.param_groups[1]["lr"],"latent_lr":optimizer.param_groups[2]["lr"],"world_size":world,"global_pairs":args.batch_size*world,"statistics_scope":"rank0_local_batch","sec_per_step":time.time()-start}
                record.update(out["stats"]); record.update(mask_stats(e["masks"],e["p"])); record.update(energy_metrics(e["z"],e["masks"],e["full_raw"],masked_raw))
                with open(log_path,"a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
            if rank==0 and step in (20,100,250,500): save_ckpt(os.path.join(args.output_dir,f"hdsm_HDSM_V01_step{step:06d}.pt"),raw,optimizer,{"horizon":horizon,"step":step},step,{"epoch":epoch,"step_in_epoch":batch_index},config,{"git_head":"unknown","run":"HDSM_V01"})
        if step>=args.max_steps: break
    if rank==0:
        json.dump({"arm":ARM,"objective":OBJECTIVE,"phase":PHASE,"completed_steps":step,"target_steps":args.max_steps,"wall_seconds":time.time()-start,"loader_batches":len(loader),"lr_horizon_steps":horizon,"world_size":world,"global_batch":args.batch_size*world,"config":config},open(os.path.join(args.output_dir,"run_summary.json"),"w"),indent=2)
    if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()

if __name__=="__main__": main()
