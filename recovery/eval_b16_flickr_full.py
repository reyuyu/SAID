"""Native ViT-B/16 Flickr30k-Full evaluator for one frozen bare student."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/opt/data/private/lklk/said-e2-worktrees/hns-v1-l14-full5epoch6085-fromscratch-v1')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

class Images(Dataset):
    def __init__(self, rows, root, pre): self.rows,self.root,self.pre=rows,Path(root),pre
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        iid,path=self.rows[i]
        with Image.open(self.root/path) as im: return iid,self.pre(im.convert('RGB'))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--expect-sha256',required=True); p.add_argument('--job',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    assert sha(a.checkpoint)==a.expect_sha256
    job=json.loads(Path(a.job).read_text()); rows=[json.loads(s) for s in Path(job['manifest']).read_text().splitlines() if s.strip()]
    images=[]; seen=set()
    for r in rows:
        if r['image_id'] not in seen: seen.add(r['image_id']); images.append((r['image_id'],r['image_path']))
    from model import longclip
    model,pre=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(state,strict=True); model=model.float().eval().to(a.device)
    started=time.time(); bs=64
    tf=[]
    for i in range(0,len(rows),bs):
        tok=longclip.tokenize([r['caption'] for r in rows[i:i+bs]],truncate=True).to(a.device)
        tf.append(F.normalize(model.encode_text(tok).float(),dim=-1).cpu())
    im=[]; loader=DataLoader(Images(images,job['image_root'],pre),batch_size=bs,num_workers=4,shuffle=False,pin_memory=True)
    ids=[]
    for iid,x in loader:
        ids.extend(iid); im.append(F.normalize(model.encode_image(x.to(a.device)).float(),dim=-1).cpu())
    tf=torch.cat(tf); im=torch.cat(im)
    positives=[str(r['positive_image_id']) for r in rows]; id_to={str(x):i for i,x in enumerate(ids)}
    it=[set() for _ in ids]
    for j,pid in enumerate(positives): it[id_to[pid]].add(j)
    truth=[{id_to[x]} for x in positives]; out={}
    for name,q,g,t in [('I2T',im,tf,it),('T2I',tf,im,truth)]:
        correct={1:0,5:0,10:0};
        for start in range(0,len(q),256):
            scores=[]
            for gs in range(0,len(g),4096): scores.append(q[start:start+256]@g[gs:gs+4096].T)
            s=torch.cat(scores,1); top=s.topk(10,dim=1).indices
            for i,row in enumerate(top):
                for k in (1,5,10):
                    if set(row[:k].tolist()) & t[start+i]: correct[k]+=1
        out[name]={'query_count':len(q),'candidate_count':len(g),'recall_percent':{str(k):100*correct[k]/len(q) for k in (1,5,10)}}
    result={'dataset':job['name'],'protocol':job['protocol'],'checkpoint_sha256':a.expect_sha256,'bare_student_only':True,'model':'ViT-B/16','n_images':len(images),'n_captions':len(rows),'metrics':out,'manifest_sha256':sha(job['manifest']),'image_content_sha256':job['image_content_sha256'],'elapsed_seconds':time.time()-started,'evaluator_sha256':sha(__file__)}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
