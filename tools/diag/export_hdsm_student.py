"""Export and losslessly verify the native CLIP student from an HDSM checkpoint."""
import argparse, hashlib, os, sys
import torch
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from model import longclip

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def probe(model):
    gen=torch.Generator().manual_seed(20260913)
    images=torch.randn(2,3,224,224,generator=gen)
    texts=longclip.tokenize(['a photo of a cat .','a dog on the grass .'],truncate=True)
    model.eval()
    with torch.no_grad(): return model.encode_image(images),model.encode_text(texts)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True); p.add_argument('--expect-steps',type=int,required=True); p.add_argument('--base-model',default='ViT-B/16'); a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if payload.get('schema_version')!='hdsm-v0.1' or int(payload.get('completed_steps',-1))!=a.expect_steps: raise SystemExit('checkpoint schema/step mismatch')
    state=payload['clip_state']
    model,_=longclip.load_from_clip(a.base_model,device='cpu',download_root=None,args=argparse.Namespace())
    missing,unexpected=model.load_state_dict(state,strict=True)
    if missing or unexpected: raise SystemExit(f'strict native load failed missing={missing} unexpected={unexpected}')
    before=probe(model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
    torch.save({'model':state,'completed_steps':payload['completed_steps'],'objective':payload['config'].get('objective'),'arm':payload['config'].get('arm'),'phase':payload['config'].get('phase'),'source_checkpoint':os.path.abspath(a.checkpoint),'source_checkpoint_sha256':digest(a.checkpoint),'student_state_sha256':digest(a.checkpoint)},a.out)
    fresh,_=longclip.load_from_clip(a.base_model,device='cpu',download_root=None,args=argparse.Namespace()); fresh.load_state_dict(torch.load(a.out,map_location='cpu',weights_only=False)['model'],strict=True)
    after=probe(fresh); di=max(float((before[0]-after[0]).abs().max()),float((before[1]-after[1]).abs().max()))
    print(f'EXPORT_STRICT_LOAD_OK tensors={len(state)} max_abs_diff={di}')
    if di!=0.0: raise SystemExit('lossless probe failed')
    print(f'EXPORT_SHA256 {digest(a.out)}')

if __name__=='__main__': main()
