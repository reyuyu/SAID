import math, torch
import torch.nn as nn
import torch.nn.functional as F
class PreProjectionGate(nn.Module):
 def __init__(self,width=512,out=768,heads=8,seed=0):
  super().__init__(); devices=torch.random.fork_rng(devices=[]); devices.__enter__(); torch.manual_seed(seed)
  self.stem=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,batch_first=True),1); self.proj=nn.Linear(width,out); nn.init.zeros_(self.proj.weight); nn.init.constant_(self.proj.bias,math.log(8)); devices.__exit__(None,None,None)
 def forward(self,H):
  with torch.autocast(device_type=H.device.type,enabled=False):
   p=torch.sigmoid(self.proj(self.stem(H.detach().float()))); hard=(p>=.5).float(); return hard+p-p.detach(),p

def pg_scores(h,W,t,m):
 with torch.autocast(device_type=h.device.type,enabled=False):
  g=F.normalize(h.float()@W.float(),dim=-1,eps=1e-6); u=F.normalize((h[:,None,:].float()*m[None,:,:].float())@W.float(),dim=-1,eps=1e-6); tn=F.normalize(t.float(),dim=-1,eps=1e-6); return 100*g@tn.T,100*torch.einsum('ijd,jd->ij',u,tn)
def pg_loss(qg,qp,m):
 y=torch.arange(qg.size(0),device=qg.device); return 5*(F.cross_entropy(qg,y)+F.cross_entropy(qg.T,y))+5*(F.cross_entropy(qp,y)+F.cross_entropy(qp.T,y))+m.abs().mean()
