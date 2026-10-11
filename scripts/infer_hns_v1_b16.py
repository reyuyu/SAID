#!/usr/bin/env python3
"""Native SAID HNS-v1 image/text similarity inference."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from model import longclip

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True)
    p.add_argument('--texts',nargs='+',required=True); p.add_argument('--device',default=None)
    a=p.parse_args(); device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if not Path(a.checkpoint).is_file(): raise SystemExit(f'checkpoint not found: {a.checkpoint}')
    try: image=Image.open(a.image).convert('RGB')
    except Exception as e: raise SystemExit(f'cannot open image {a.image}: {e}')
    model,preprocess=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    try: model.load_state_dict(state,strict=True)
    except RuntimeError as e: raise SystemExit(f'strict SAID student load failed: {e}')
    model=model.float().to(device).eval(); image_tensor=preprocess(image).unsqueeze(0).to(device)
    tokens=longclip.tokenize(a.texts,context_length=248,truncate=True).to(device)
    with torch.no_grad():
        im=torch.nn.functional.normalize(model.encode_image(image_tensor).float(),dim=-1)
        tx=torch.nn.functional.normalize(model.encode_text(tokens).float(),dim=-1)
        scores=(im@tx.T).squeeze(0).cpu().tolist()
    for rank,i in enumerate(sorted(range(len(scores)),key=lambda i:scores[i],reverse=True),1): print(f'{rank}\t{scores[i]:.8f}\t{a.texts[i]}')
    print(f'embedding_dim={im.shape[-1]} device={device} context_length=248')
if __name__=='__main__': main()
