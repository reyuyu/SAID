#!/usr/bin/env python3
"""Run native retrieval with the published L14 student.

Urban-1k and COCO use the repository's canonical evaluators. For other protocols,
provide a JSONL manifest with image_id, image_path, caption and positive_image_id.
All paths are supplied by the user; datasets are never packaged in this release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from model.backbone import load_native_state
from model import longclip
from tools.urban1k_retrieval import evaluate_urban1k, dataset_fingerprint
from eval.retrieval.coco_retrieval import evaluate_coco

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

@torch.no_grad()
def evaluate_manifest(model, preprocess, manifest, image_root, device, batch):
    rows=[json.loads(x) for x in Path(manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    images=[]; seen=set()
    for r in rows:
        if r['image_id'] not in seen:
            seen.add(r['image_id']); images.append((r['image_id'],r['image_path']))
    ims=[]
    for i in range(0,len(images),batch):
        x=torch.stack([preprocess(Image.open(Path(image_root)/p).convert('RGB')) for _,p in images[i:i+batch]]).to(device)
        ims.append(F.normalize(model.encode_image(x).float(),dim=-1).cpu())
    txt=[]
    for i in range(0,len(rows),batch):
        t=longclip.tokenize([r['caption'] for r in rows[i:i+batch]],context_length=248,truncate=True).to(device)
        txt.append(F.normalize(model.encode_text(t).float(),dim=-1).cpu())
    sim=torch.cat(ims) @ torch.cat(txt).T
    image_ids=[x[0] for x in images]; id_to_i={str(x):i for i,x in enumerate(image_ids)}
    positives=[str(r['positive_image_id']) for r in rows]
    result={'n_images':len(images),'n_captions':len(rows)}
    for k in (1,5,10):
        top=sim.topk(min(k,sim.shape[1]),dim=1).indices
        result[f'I2T_R@{k}']=sum(any(str(rows[j]['positive_image_id'])==str(image_ids[i]) for j in top[i]) for i in range(len(images)))/len(images)
        top2=sim.T.topk(min(k,sim.shape[0]),dim=1).indices
        result[f'T2I_R@{k}']=sum(id_to_i.get(positives[j],-1) in set(top2[j].tolist()) for j in range(len(rows)))/len(rows)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset',choices=['urban1k','coco','manifest'],required=True); p.add_argument('--data-root'); p.add_argument('--manifest'); p.add_argument('--image-root'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--output',required=True); a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model,pre,spec=load_native_state(state,base_model='ViT-L/14',device=a.device)
    out={'checkpoint_sha256':sha(a.checkpoint),'backbone':spec,'native_student_only':True,'tokenizer':'longclip.tokenize(context_length=248, truncate=True)','device':a.device}
    if a.dataset=='urban1k':
        if not a.data_root: p.error('--data-root is required for urban1k')
        out['dataset_fingerprint']=dataset_fingerprint(a.data_root); out['metrics']=evaluate_urban1k(model,pre,root=a.data_root,batch_size=a.batch_size,device=a.device)
    elif a.dataset=='coco':
        if not a.data_root: p.error('--data-root is required for coco')
        ann=str(Path(a.data_root).parent/'annotations/captions_val2017.json'); out['metrics']=evaluate_coco(model,pre,root=a.data_root,ann_file=ann,batch_size=a.batch_size,device=a.device,similarity_chunk=512,image_representation='legacy_cls')
    else:
        if not a.manifest or not a.image_root: p.error('--manifest and --image-root are required')
        out['manifest_sha256']=sha(a.manifest); out['metrics']=evaluate_manifest(model,pre,a.manifest,a.image_root,a.device,a.batch_size)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
