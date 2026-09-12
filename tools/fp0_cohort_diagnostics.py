"""Fixed 64-prefix cohort, single feature pass per checkpoint, offline diagnostics only."""
import argparse
import json
import random
import sys
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.append(str(ROOT/'train'))
from model import longclip
from model.finelip_prefix import FineLIPPrefix, scores
from said_cvssl_data import reference_view_a_transform


def diagnostics(checkpoint, manifest_path, output):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=payload['config']
    manifest_path=Path(manifest_path)
    if not manifest_path.exists():
        records=json.loads(Path(config['manifest']).read_text())[1000:1064]
        rng=random.Random(91427)
        cohort=[]
        for i,r in enumerate(records):
            sentences=r['conversations'][1]['value'].replace('\n',' ').split('. ')
            k=rng.randint(1,len(sentences))
            cohort.append({'sample_id':1000+i,'image':r['image'],'prefix_k':k,
                           'caption_said':'. '.join(sentences[:k])})
        manifest_path.write_text(json.dumps({'name':'FP0_fixed64_prefix_seed91427','items':cohort},indent=2))
    cohort=json.loads(manifest_path.read_text())['items']
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    net=FineLIPPrefix(clip.float()).cuda().eval()
    net.load_state_dict(payload['module'],strict=True)
    values={key:[] for key in ('v','t','i','e','ih','th')}
    transform=reference_view_a_transform()
    with torch.no_grad():
        for first in range(0,len(cohort),16):
            batch=cohort[first:first+16]
            images=[]
            for row in batch:
                with Image.open(Path(config['manifest']).parent/row['image']) as im:
                    images.append(transform(im.convert('RGB')))
            tokens=longclip.tokenize([r['caption_said'] for r in batch],context_length=248,truncate=True).cuda()
            v,t,iw,tw,gi,gt=net.encode(torch.stack(images).cuda(),tokens)
            for key,x in [('v',v),('t',t),('i',gi.float()),('e',gt.float()),
                          ('ih',-(iw*iw.clamp_min(1e-30).log()).sum(-1)),
                          ('th',-(tw*tw.clamp_min(1e-30).log()).sum(-1))]: values[key].append(x.cpu())
        values={k:torch.cat(v) for k,v in values.items()}
        v,t=values['v'],values['t']
        native=torch.nn.functional.normalize(values['i'],dim=-1)@torch.nn.functional.normalize(values['e'],dim=-1).T
        local=torch.cat([scores(v[i:i+8],t) for i in range(0,len(v),8)])
        def recall(matrix):
            target=torch.arange(len(matrix))
            return {'I2T_R1':float((matrix.argmax(1)==target).float().mean()),
                    'T2I_R1':float((matrix.argmax(0)==target).float().mean())}
        def slot_cos(z):
            n=z.shape[1]
            return float(((z.sum(1).square().sum(-1)-z.square().sum((1,2)))/(n*(n-1))).mean())
        result={'optimizer_step':payload['optimizer_step'],'n':len(v),'scope':'training-cohort diagnostic; not canonical retrieval',
                'image_scale':net.image_aggregator.scale.item(),'text_scale':net.text_aggregator.scale.item(),
                'image_slot_cos':slot_cos(v[:,1:]),'text_slot_cos':slot_cos(t[:,:-1]),
                'image_aggregation_entropy':float(values['ih'].mean()),'text_aggregation_entropy':float(values['th'].mean()),
                'native_cls_eos':recall(native),'finegrain':recall(local)}
        Path(output).write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(diagnostics(a.checkpoint,a.manifest,a.out)))
