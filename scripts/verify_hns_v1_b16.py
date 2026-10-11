#!/usr/bin/env python3
import argparse,hashlib,json
import torch
from model import longclip
EXPECTED={'student':'a60150575f7e8f7b41effb730c0f9d866757954833a240813d4695d9a1cfcb44','step0':'54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6','training':'83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--student',required=True); p.add_argument('--training'); p.add_argument('--step0'); a=p.parse_args()
 got=sha(a.student); assert got==EXPECTED['student'],f'student SHA256 mismatch: {got}'; s=torch.load(a.student,map_location='cpu',weights_only=True); model,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace()); model.load_state_dict(s,strict=True); assert model.context_length==248 and model.text_projection.shape[1]==512
 result={'passed':True,'student_sha256':got,'context_length':248,'embedding_dim':512}
 for name,path in [('training',a.training),('step0',a.step0)]:
  if path:
   digest=sha(path); assert digest==EXPECTED[name],f'{name} SHA256 mismatch: {digest}'; x=torch.load(path,map_location='cpu',weights_only=False); result[name]={'sha256':digest,'keys':list(x)[:30],'completed_steps':x.get('completed_steps')}
   if name=='training': assert x.get('completed_steps')==3651
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
