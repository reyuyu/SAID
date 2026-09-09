"""Standard global CLIP retrieval for the two Phase 2.5 final checkpoints."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from model import longclip
from .local_router_eval import load_checkpoint
from .grounding_root_cause import setup
from .semantic_grounding_eval import write_json, sha256


def collate_coco(batch):
    return torch.stack([x[0] for x in batch]),[c for _,captions in batch for c in captions[:5]]


def recall_metrics(images,texts):
    if len(texts)!=5*len(images):raise ValueError('five captions per image required')
    result={}
    for direction,queries,targets in [('I2T',images,texts),('T2I',texts,images)]:
        hits={k:0 for k in [1,5,10]}
        for start in range(0,len(queries),256):
            top=(queries[start:start+256]@targets.T).topk(10,dim=1).indices
            truth=torch.arange(start,start+len(top),device=queries.device)
            if direction=='I2T':correct=top//5==truth[:,None]
            else:correct=top==truth[:,None]//5
            for k in hits:hits[k]+=int(correct[:,:k].any(1).sum())
        result[direction]={'R@%d'%k:hits[k]/len(queries) for k in hits}
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs_root',type=Path,default=Path('runs_salu/phase25'))
    p.add_argument('--coco_root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/local_evidence_router/coco.json'))
    p.add_argument('--device',default='cuda:0');args=p.parse_args();setup()
    if args.output.exists():raise FileExistsError('use a fresh COCO output')
    summary={}
    for arm in ['residual','attention_delta']:
        path=args.runs_root/arm/'salu_said_only_last.pt'
        model,preprocess=load_checkpoint(path,args.device)
        # Guard the inference contract: retrieval may only use standard encoders.
        def forbidden(*a,**kw):raise AssertionError('local evidence must not participate in retrieval')
        model.encode_router_input=forbidden
        model.clip.encode_image_with_local_evidence=forbidden
        data=CocoCaptions(root=str(args.coco_root/'val2017'),annFile=str(args.coco_root/'annotations/captions_val2017.json'),transform=preprocess)
        assert len(data)==5000
        loader=DataLoader(data,batch_size=32,num_workers=4,shuffle=False,collate_fn=collate_coco)
        images=[];texts=[]
        for index,(tensor,captions) in enumerate(loader):
            images.append(F.normalize(model.encode_image(tensor.to(args.device)),dim=-1))
            texts.append(F.normalize(model.encode_text(longclip.tokenize(captions).to(args.device)),dim=-1))
            if index%40==0:print('COCO',arm,index,'/',len(loader),flush=True)
        result=recall_metrics(torch.cat(images),torch.cat(texts))
        summary[arm]={'metrics':result,'images':5000,'captions':25000,'precision':'fp32',
                      'checkpoint_sha256':sha256(path),'inference':'standard encode_image + encode_text'}
        write_json(args.output,summary)
        print('COCO_RESULT',arm,result,flush=True)
        del model,images,texts;torch.cuda.empty_cache()


if __name__=='__main__':main()
