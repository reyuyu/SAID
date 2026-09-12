"""Read-only shared-init native-interface and actual-backbone gradient acceptance."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train'))
from model import longclip
from model.finelip_prefix import FineLIPPrefix
from train_said_cls_cvssl import load_init_state, state_digest

p=argparse.ArgumentParser(); p.add_argument('--init_state',required=True); args=p.parse_args()
args.base_model='ViT-B/16'
torch.manual_seed(0)
clip,_=longclip.load_from_clip(args.base_model,device='cpu',args=args)
load_init_state(clip,args.init_state,0)
digest=state_digest(clip.state_dict())
net=FineLIPPrefix(clip.float(),amp=False).cuda().eval()
tokens=longclip.tokenize(['A red square.',''],context_length=248,truncate=True).cuda()
images=torch.randn(2,3,224,224,device='cuda')
with torch.no_grad():
    gi=net.clip.encode_image(images); gt=net.clip.encode_text(tokens)
    v,t,iw,tw,ai,at=net.encode(images,tokens)
    torch.testing.assert_close(ai,gi,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(at,gt,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(v[:,0],torch.nn.functional.normalize(gi,dim=-1))
    torch.testing.assert_close(t[:,-1],torch.nn.functional.normalize(gt,dim=-1))
    assert v.shape==(2,40,512) and t.shape==(2,40,512)
net.train(); net.amp=True
loss,stats,extra=net(images,tokens,torch.arange(2,device='cuda'),True)
loss.backward()
missing=[name for name,param in net.named_parameters() if param.requires_grad and param.grad is None]
assert not missing,missing
assert all(torch.isfinite(param.grad).all() for param in net.parameters() if param.grad is not None)
print(json.dumps({'native_interface':'PASS','actual_backbone_backward':'PASS','master_dtype':str(next(net.parameters()).dtype),
                  'aggregation_dtype':str(net.image_aggregator.scale.dtype),'score_dtype':str(loss.dtype),
                  'output_shape':list(v.shape),'initial_state_digest':digest,'unconnected_trainable_parameters':missing}))
