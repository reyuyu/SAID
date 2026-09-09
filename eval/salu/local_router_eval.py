"""Phase 2.5 frozen A/B evaluation; never changes official Phase 2.3 artifacts."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model import longclip
from model.salu_model import SALUModel
from .flickr_entities import load_split
from .grounding_root_cause import setup, logits_from_features
from .local_evidence_metrics import geometry, evaluate, rank_values, correlation
from .semantic_grounding_eval import sha256, write_json

TAGS={'initial':'salu_initial.pt','step100':'salu_said_only_step000100.pt',
      'step200':'salu_said_only_step000200.pt','step400':'salu_said_only_step000400.pt',
      'final':'salu_said_only_last.pt'}


def load_checkpoint(path, device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    args=payload.get('args',{})
    clip,preprocess=longclip.load_from_clip('ViT-B/16',device='cpu')
    model=SALUModel(clip,tau_said=args.get('tau_said',.07),
                    said_loss_mode=args.get('said_loss_mode','positive'),
                    said_feature_source=args.get('said_feature_source','residual'))
    model.load_state_dict(payload['model'],strict=True)
    return model.float().eval().to(device),preprocess


def patch_geometry(direct, router):
    """Per-phrase ranking correlation and |top-k intersection| / k (ties stable)."""
    if direct.shape!=router.shape or direct.ndim!=2:raise ValueError('matching [phrases,patches] arrays required')
    rho=np.asarray([correlation(a,b,True) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
                    for a,b in zip(direct,router)])
    a=np.argsort(-direct,axis=1,kind='stable');b=np.argsort(-router,axis=1,kind='stable')
    rows={'spearman':rho}
    for k in [1,5,10,20]:
        rows['top%d'%k]=np.asarray([len(set(x[:k])&set(y[:k]))/k for x,y in zip(a,b)])
    return rows


@torch.inference_mode()
def infer(model,preprocess,records,image_root,device,output=None):
    direct=[];router=[];check=0.
    for index,im in enumerate(records):
        with Image.open(image_root/(im['id']+'.jpg')) as image:
            tensor=preprocess(image)[None].to(device)
            if output is not None:
                geo=im['geometry'];left,top=geo['crop_xy']
                crop=image.resize(tuple(geo['resized_size']),Image.Resampling.BICUBIC).crop((left,top,left+224,top+224)).convert('RGB')
                (output/'images').mkdir(parents=True,exist_ok=True)
                crop.save(output/'images'/(im['id']+'.png'))
        _,h=model.encode_router_input(tensor);h=h[0].float()
        t=F.normalize(model.encode_text(longclip.tokenize([p['phrase'] for p in im['phrases']],truncate=False).to(device)).float(),dim=-1)
        d=logits_from_features(h,t);r=logits_from_features(h,t,model.said_router)
        if index<8:
            actual=model.said_router(t,h[None].expand(len(t),-1,-1))[0]
            check=max(check,float((actual-torch.softmax(r/model.tau_said,-1)).abs().max()))
        direct.append(d.cpu().numpy());router.append(r.cpu().numpy())
    if check>1e-6:raise RuntimeError('router logit export does not match production attention')
    return np.concatenate(direct),np.concatenate(router),check


def save_variant(root,key,logits,g):
    metrics,rows,attention=evaluate(logits,g,.07)
    (root/'logits').mkdir(exist_ok=True);np.save(root/'logits'/(key+'.npy'),logits)
    folder=root/'attention'/key;folder.mkdir(parents=True,exist_ok=True)
    for ident,a in zip(g['ids'],attention):np.save(folder/(ident+'.npy'),a.reshape(14,14).astype('float32'))
    # Symmetric pair switch contribution, averaged over each phrase's pairs.
    q,d=g['pairs'].T;excess=(attention[q]*(g['coverage'][q]-g['coverage'][d])).sum(1)
    symmetric=excess.reshape(-1,2).mean(1).repeat(2)
    sums=np.bincount(q,weights=symmetric,minlength=len(logits));counts=np.bincount(q,minlength=len(logits))
    rows['switch_margin']=np.divide(sums,counts,out=np.full(len(logits),np.nan),where=counts>0)
    (root/'per_phrase_candidate').mkdir(exist_ok=True)
    np.savez_compressed(root/'per_phrase_candidate'/(key+'.npz'),**rows)
    return metrics,rows


def full_gate(summary):
    a=summary['attention_delta_final_router'];r=summary['residual_final_router']
    positive=all(a[k]['mean']>0 for k in ['mass_gain','semantic_mass_excess','switch_margin'])
    improves=all(a[k]['mean']>r[k]['mean'] for k in ['mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor'])
    return positive and improves


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['small','geometry','full'],required=True)
    p.add_argument('--runs_root',type=Path,default=Path('runs_salu/phase25'))
    p.add_argument('--output',type=Path,default=Path('outputs/local_evidence_router'))
    p.add_argument('--image_root',type=Path,required=True);p.add_argument('--entities_root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();setup()
    root=args.output/args.mode;root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('use a fresh output directory')
    if args.mode=='full':
        summary=json.loads((args.output/'small/summary.json').read_text())
        if not full_gate(summary):raise RuntimeError('Full Flickr gate not met; do not run full val')
    max_images={'small':128,'geometry':400,'full':None}[args.mode]
    records,exclusions=load_split(args.image_root,args.entities_root,'val',max_images)
    g=geometry(records)
    if args.mode=='small':
        previous=json.loads(Path('outputs/local_semantic_evidence/small/manifest.json').read_text())
        assert g['ids']==previous['phrase_ids'] and len(g['ids'])==1908
    tags=list(TAGS) if args.mode=='small' else ['final']
    manifest={'status':'running','mode':args.mode,'phrase_ids':g['ids'],'image_ids':[im['id'] for im in records],
              'num_images':len(records),'num_phrases':len(g['ids']),'pair_count':len(g['pairs'])//2,
              'exclusions':exclusions,'tau':.07,'tags':tags,'checkpoints':{},
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'orientation':'identity primary; rotate180 diagnostic only'}
    write_json(root/'manifest.json',manifest);write_json(root/'per_phrase.json',{p['id']:p for p in g['phrases']})
    summary={};checks={}
    for arm in ['residual','attention_delta']:
        for tag in tags:
            key=arm+'_'+tag;path=args.runs_root/arm/TAGS[tag]
            model,preprocess=load_checkpoint(path,args.device)
            assert model.said_feature_source==arm
            direct,router,check=infer(model,preprocess,records,args.image_root,args.device,root if args.mode!='geometry' else None)
            checks[key]=check;manifest['checkpoints'][key]={'file':path.name,'sha256':sha256(path)}
            if args.mode=='geometry':
                assert len(direct)>=5000
                direct,router=direct[:5000],router[:5000]
                rows=patch_geometry(direct,router)
                attention=np.exp(router/.07-(router/.07).max(1,keepdims=True));attention/=attention.sum(1,keepdims=True)
                mass_gain=(attention*g['coverage'][:5000]).sum(1)-g['coverage'][:5000].mean(1)
                rows['router_mass_gain']=mass_gain
                summary[key]={k:float(np.nanmean(v)) for k,v in rows.items()}
                valid=np.isfinite(rows['spearman'])
                summary[key].update({'phrases':5000,'spearman_valid_phrases':int(valid.sum()),
                    'ranking_vs_mass_gain_pearson':correlation(rows['spearman'][valid],mass_gain[valid]),
                    'ranking_vs_mass_gain_spearman':correlation(rows['spearman'][valid],mass_gain[valid],True)})
                summary[key]['correlation_quartiles']=[]
                for indices in np.array_split(np.flatnonzero(valid)[np.argsort(rows['spearman'][valid])],4):
                    summary[key]['correlation_quartiles'].append({'n':len(indices),'mean_spearman':float(rows['spearman'][indices].mean()),'mean_mass_gain':float(mass_gain[indices].mean())})
                np.savez_compressed(root/(key+'_geometry.npz'),**rows)
            else:
                for mode,logits in [('direct',direct),('router',router)]:
                    if args.mode=='full' and arm=='residual' and mode=='direct':continue
                    summary[key+'_'+mode],_=save_variant(root,key+'_'+mode,logits,g)
                rows=patch_geometry(direct,router);np.savez_compressed(root/(key+'_geometry.npz'),**rows)
            print('EVALUATED',key,flush=True)
            write_json(root/'summary.json',summary)
            del model;torch.cuda.empty_cache()
    manifest['status']='complete';manifest['router_export_max_diff']=checks
    if args.mode=='small':manifest['full_gate']=full_gate(summary)
    if args.mode=='geometry':
        manifest['evaluated_phrase_ids']=g['ids'][:5000]
        manifest['note']='Geometry-only probe on first 5000 phrases; not a full-val grounding benchmark.'
    write_json(root/'manifest.json',manifest)
    print('LOCAL_ROUTER_EVAL_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
