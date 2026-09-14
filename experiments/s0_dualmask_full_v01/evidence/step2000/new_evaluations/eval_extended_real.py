import argparse, json, os, sys, time, hashlib
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, '/root/SAID-gap-completion')
sys.path.insert(0, '/root/SAID-gap-completion/eval/retrieval')
from tools.eval_urban1k_cls import load_student
from model import longclip

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metric(sim, image_ids, positive_ids):
    nimg, ntxt = sim.shape
    id_to_i={str(x):i for i,x in enumerate(image_ids)}
    pos_by_img={str(x):[] for x in image_ids}
    for j,p in enumerate(positive_ids): pos_by_img.setdefault(str(p),[]).append(j)
    out={}
    for k in (1,5,10):
        kk=min(k,ntxt); top=sim.topk(kk,dim=1,largest=True,sorted=True).indices
        i2t=sum(any(int(j) in pos_by_img[str(image_ids[i])] for j in top[i]) for i in range(nimg))/nimg
        kk2=min(k,nimg); top2=sim.t().topk(kk2,dim=1,largest=True,sorted=True).indices
        t2i=sum(id_to_i.get(str(positive_ids[j]),-1) in set(map(int,top2[j])) for j in range(ntxt))/ntxt
        out[f'R@{k}']={'I2T':i2t,'T2I':t2i}
    return out

@torch.no_grad()
def evaluate(model, preprocess, manifest, image_root, device, batch):
    rows=[json.loads(x) for x in open(manifest,encoding='utf-8') if x.strip()]
    images=[]; seen=set()
    for r in rows:
        if r['image_id'] not in seen: seen.add(r['image_id']); images.append((r['image_id'],r['image_path']))
    text=[]
    for s in range(0,len(rows),batch):
        tok=longclip.tokenize([r['caption'] for r in rows[s:s+batch]],truncate=True).to(device)
        text.append(torch.nn.functional.normalize(model.encode_text(tok).float(),dim=-1).cpu())
    tf=torch.cat(text)
    ims=[]
    for s in range(0,len(images),batch):
        x=torch.stack([preprocess(Image.open(Path(image_root)/p).convert('RGB')) for _,p in images[s:s+batch]]).to(device)
        ims.append(torch.nn.functional.normalize(model.encode_image(x).float(),dim=-1).cpu())
    imf=torch.cat(ims)
    sim=imf @ tf.t()
    return len(images),len(rows),metric(sim,[x[0] for x in images],[r['positive_image_id'] for r in rows])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--output-dir',required=True); ap.add_argument('protocols',nargs='+',help='name:manifest:image_root')
    a=ap.parse_args(); os.makedirs(a.output_dir,exist_ok=True); model,pre,meta=load_student(a.checkpoint,'ViT-B/16','cpu'); model.to(a.device).eval(); allout={}
    for spec in a.protocols:
        name,manifest,root=spec.split(':',2); t=time.time(); ni,nt,m=evaluate(model,pre,manifest,root,a.device,a.batch_size); out={'protocol':name,'n_images':ni,'n_captions':nt,'metrics':m,'checkpoint_sha256':sha256(a.checkpoint),'manifest_sha256':sha256(manifest),'elapsed_seconds':time.time()-t,'device':a.device,'native_student_only':True}; Path(a.output_dir,f'{name}.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); allout[name]=out; print(json.dumps(out))
    Path(a.output_dir,'extended_summary.json').write_text(json.dumps(allout,indent=2),encoding='utf-8')
if __name__=='__main__': main()
