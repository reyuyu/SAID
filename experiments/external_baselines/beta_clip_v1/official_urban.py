"""Run untouched official Urban CLS+TCI, then compare query-independent CLS."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from experiments.external_baselines.beta_clip_v1.native_adapter import (
    BetaCLIPNativeAdapter,OFFICIAL,RUN,reconstruct,
)


def aggregate(tensor):
    parts=[torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts,tensor.contiguous());return torch.cat(parts)


def caption_digest(rows):
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--variant',default='ce')
    args=parser.parse_args();rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True  # Explicitly retain official runtime default.
    dist.init_process_group('nccl');assert dist.get_world_size() in (1,4)
    model,tokenizer,transform,metadata=reconstruct(args.checkpoint,device='cuda:'+str(local))
    native=BetaCLIPNativeAdapter(model,tokenizer)
    evaluator=importlib.import_module('validate_urban1k_distributed')
    captures={}
    original_forward=model.forward_patches_then_condition
    def forward(*a,**kw):
        result=original_forward(*a,**kw)
        captures['official_image_raw']=result['cls_embed'].detach()
        captures['official_text_raw']=result['eos_embed'].detach()
        captures['images']=a[0].detach();captures['tokens']=a[1].detach()
        return result
    original_logits=model.get_sim_logits_conditioned
    def logits(*a,**kw):
        result=original_logits(*a,**kw)
        captures['official_cls_i2t']=result[0].detach();captures['official_cls_t2i']=result[1].detach()
        return result
    model.forward_patches_then_condition=forward;model.get_sim_logits_conditioned=logits
    output=RUN/(args.variant+'-official-urban');output.mkdir(parents=True,exist_ok=True)
    working=output/'working';working.mkdir(exist_ok=True);(working/'data').mkdir(exist_ok=True)
    urban=Path('/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k')
    link=working/'data/Urban1k'
    if rank==0 and not link.exists():link.symlink_to(urban,target_is_directory=True)
    dist.barrier(device_ids=[local]);os.chdir(working)
    evaluator.validate_urban1k(model,str(output/'checkpoint_10'),metadata['epoch']-1,0,False,tokenizer,device=local)
    model.forward_patches_then_condition=original_forward;model.get_sim_logits_conditioned=original_logits
    with torch.no_grad():
        images=captures['images'];tokens=captures['tokens']
        image_raw=native.image_raw(images);text_raw=native.text_raw(tokens)
        image_error=float((image_raw-captures['official_image_raw']).abs().max())
        text_error=float((text_raw-captures['official_text_raw']).abs().max())
        normalized_image=F.normalize(image_raw.float(),dim=-1);normalized_text=F.normalize(text_raw.float(),dim=-1)
        all_image,all_text=aggregate(normalized_image),aggregate(normalized_text)
        i2t=normalized_image@all_text.T;t2i=normalized_text@all_image.T
        top1_i=int((i2t.argmax(-1)!=captures['official_cls_i2t'].argmax(-1)).sum())
        top1_t=int((t2i.argmax(-1)!=captures['official_cls_t2i'].argmax(-1)).sum())
        # Compare rankings with identical multiplication order/scale as official.
        scaled_i=(model.logit_scale.exp()*normalized_image)@all_text.T
        scaled_t=(model.logit_scale.exp()*normalized_text)@all_image.T
        score_error=max(float((scaled_i-captures['official_cls_i2t']).abs().max()),float((scaled_t-captures['official_cls_t2i']).abs().max()))
        sorted_i=int((scaled_i.argsort(-1,descending=True)!=captures['official_cls_i2t'].argsort(-1,descending=True)).any(-1).sum())
        sorted_t=int((scaled_t.argsort(-1,descending=True)!=captures['official_cls_t2i'].argsort(-1,descending=True)).any(-1).sum())
        correct=torch.arange(len(images),device=images.device)+rank*len(images)
        counts=torch.tensor([(i2t.argmax(-1)==correct).sum(),(t2i.argmax(-1)==correct).sum(),len(images),top1_i,top1_t,sorted_i,sorted_t],device=images.device)
        dist.all_reduce(counts)
        standard=model.encode_image(images)
        standard=standard[:,0] if standard.ndim==3 else standard
        standard_error=float((standard-image_raw).abs().max())
    local_summary=dict(rank=rank,image_raw_max_abs=image_error,text_raw_max_abs=text_error,
                       official_scaled_score_max_abs=score_error,standard_encode_image_vs_official_cls_max_abs=standard_error)
    summaries=[None]*dist.get_world_size();dist.all_gather_object(summaries,local_summary)
    if rank==0:
        official_file=working/'evaluation_urban1k/checkpoint_10/epoch_10/urban1k_retrieval.json'
        official=json.loads(official_file.read_text())
        adapter_i=100*int(counts[0])/int(counts[2]);adapter_t=100*int(counts[1])/int(counts[2])
        assert int(counts[2])==1000
        assert abs(adapter_i-official['results']['retrieval_acc_cls_i2t'])<1e-6
        assert abs(adapter_t-official['results']['retrieval_acc_cls_t2i'])<1e-6
        assert int(counts[3])==int(counts[4])==0
        assert max(x['image_raw_max_abs'] for x in summaries)==0 and max(x['text_raw_max_abs'] for x in summaries)==0
        names=sorted(p.name for p in (urban/'caption').glob('*.txt'))
        source_captions=[(n,(urban/'caption'/n).read_text().splitlines()[0].strip()) for n in names]
        # Official loader and frozen SAID Urban reader both use lexically sorted
        # stems and the stripped first caption line. Verify raw lists explicitly.
        from tools.urban1k_retrieval import image_caption_pairs,read_captions
        said_pairs=image_caption_pairs(str(urban));said_captions=read_captions(said_pairs)
        official_captions=[c for _,c in source_captions]
        differences=sum(a!=b for a,b in zip(official_captions,said_captions));assert differences==0
        published_i,published_t=(88.6,89.0) if args.variant=='ce' else (92.3,91.8)
        comparisons={kind:max(abs(official['results'][f'retrieval_acc_{kind}_i2t']-published_i),abs(official['results'][f'retrieval_acc_{kind}_t2i']-published_t)) for kind in ('cls','tci')}
        classification={kind:'REPRODUCED' if delta<.05 else 'NEAR' if delta<=.200001 else 'NOT_REPRODUCED' for kind,delta in comparisons.items()}
        result=dict(variant=args.variant,checkpoint=metadata,official_evaluator=official,
            adapter_native_cls=dict(I2T_R1=adapter_i/100,T2I_R1=adapter_t/100),
            consistency=dict(passed=True,top1_i2t_mismatches=int(counts[3]),top1_t2i_mismatches=int(counts[4]),
                scaled_full_ranking_i2t_query_mismatches=int(counts[5]),scaled_full_ranking_t2i_query_mismatches=int(counts[6]),ranks=summaries,
                note='Native adapter uses official query-independent image-block CLS. TCI never used by adapter.'),
            urban_caption_audit=dict(count=1000,official_sha256=caption_digest(official_captions),said_sha256=caption_digest(said_captions),different_count=differences),
            published_reference=dict(I2T_percent=published_i,T2I_percent=published_t),
            reproduction_by_path=classification,max_delta_pp_by_path=comparisons,
            no_training=True)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        torch.save(dict(image_features=all_image.cpu(),text_features=all_text.cpu(),rank_group_order=True),output/'native-cls-embeddings.pt')
        print(json.dumps(dict(official=official['results'],adapter=result['adapter_native_cls'],consistency_passed=True,reproduction=classification)),flush=True)
    dist.barrier(device_ids=[local]);torch.cuda.synchronize(local);dist.destroy_process_group()


if __name__=='__main__':main()
