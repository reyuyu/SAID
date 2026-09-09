"""Frozen local semantic evidence evaluation with reusable saved logits."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .flickr_entities import load_split
from .grounding_root_cause import load_model,setup
from .local_evidence import STAGES,extract_stages,project_candidate
from .local_evidence_metrics import geometry,evaluate,core,softmax,pareto_front
from .semantic_grounding_eval import write_json,native_clip_patches

def main():
 p=argparse.ArgumentParser()
 for name in ['image_root','entities_root','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--max_images',type=int,default=128);p.add_argument('--candidates',type=Path);p.add_argument('--postprocess',action='store_true');p.add_argument('--device',default='cuda:0');args=p.parse_args();setup()
 root=args.output;root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists() and not args.postprocess:raise FileExistsError('Use fresh output or --postprocess')
 records,exclusions=load_split(args.image_root,args.entities_root,'val',args.max_images or None)
 g=geometry(records);ids=g['ids'];lookup={k:i for i,k in enumerate(ids)}
 candidates=[f'{b}_block{l:02d}_{s}' for b in ['initial','phase22'] for l in [3,6,9,11] for s in STAGES]
 if args.candidates:candidates=json.loads(args.candidates.read_text())
 auxiliary=[] if args.candidates else [f'{b}_block{l:02d}_x_in' for b in ['initial','phase22'] for l in [3,6,9,11]]
 keys=candidates+auxiliary;original=json.loads(Path('outputs/semantic_grounding/manifest.json').read_text())
 manifest={'status':'running','schema_version':2,'dataset':'Flickr30K Entities','split':'val','num_images':len(records),'num_phrases':len(ids),'image_ids':[im['id'] for im in records],'phrase_ids':ids,'candidates':candidates,'auxiliary':auxiliary,'layers':[3,6,9,11],'stages':list(STAGES),'backbones':['initial','phase22'],'tau':.07,'exclusions':exclusions,'switching_groups':g['groups'],'pair_count':len(g['pairs'])//2,'pair_query_count':len(g['pairs']),'projection':'diagnostic reuse of final ln_post + proj; no intermediate text alignment assumed','checkpoints':original['checkpoints'],'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol':'identity primary; rotate180 diagnostic only; no q/k; frozen weights'}
 if not args.postprocess:
  write_json(root/'manifest.json',manifest);write_json(root/'per_phrase.json',{x['id']:x for x in g['phrases']});(root/'images').mkdir(exist_ok=True);(root/'logits').mkdir(exist_ok=True);checks={}
  for backbone in ['initial','phase22']:
   active=[k for k in keys if k.startswith(backbone+'_')]
   if not active:continue
   model,preprocess,tokenize,salu=load_model(backbone,original,args.device)
   arrays={k:np.empty((len(ids),196),dtype=np.float32) for k in active};maxdiff=0.
   with torch.inference_mode():
    for index,im in enumerate(records):
     with Image.open(args.image_root/(im['id']+'.jpg')) as image:
      tensor=preprocess(image)[None].to(args.device);geo=im['geometry'];left,top=geo['crop_xy'];crop=image.resize(tuple(geo['resized_size']),Image.Resampling.BICUBIC).crop((left,top,left+224,top+224)).convert('RGB')
     crop.save(root/'images'/(im['id']+'.png'))
     layers=sorted({int(k.split('_block')[1][:2]) for k in active});states=extract_stages(model.visual,tensor,layers,include_input=True)
     if index<8 and 11 in states:
      expected=model.encode_image_with_patches(tensor)[1] if backbone=='phase22' else native_clip_patches(model,tensor)
      actual=project_candidate(model.visual,states[11]['residual']);maxdiff=max(maxdiff,float((actual-expected).abs().max()));torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
     positions=[lookup[x['id']] for x in im['phrases']];t=F.normalize(model.encode_text(tokenize([x['phrase'] for x in im['phrases']],truncate=False).to(args.device)).float(),dim=-1)
     for key in active:
      layer=int(key.split('_block')[1][:2]);stage=key.split('_',2)[2];h=F.normalize(project_candidate(model.visual,states[layer][stage])[0].float(),dim=-1);arrays[key][positions]=(t@h.T).cpu().numpy()
     if index%100==0:print(backbone,index+1,'/',len(records),flush=True)
   for key,array in arrays.items():np.save(root/'logits'/(key+'.npy'),array)
   checks[backbone]={'images':min(8,len(records)),'max_abs_diff':maxdiff,'pass':maxdiff<=1e-6};write_json(root/'final_baseline_checks.json',checks)
   del model,salu,arrays;torch.cuda.empty_cache()
 else:
  manifest=json.loads((root/'manifest.json').read_text());assert manifest['phrase_ids']==ids
 np.save(root/'mean_gt_coverage.npy',g['coverage'].mean(0).reshape(14,14));manifest['random_patch_pointing']=float(g['centres'].mean());manifest['mean_gt_area_fraction']=float(g['coverage'].mean())
 summary={};auxsummary={};temperature={}
 for key in keys:
  logits=np.load(root/'logits'/(key+'.npy'));assert np.isfinite(logits).all()
  stats,rows,att=evaluate(logits,g);stats.update({'backbone':key.split('_')[0],'layer':int(key.split('_block')[1][:2]),'stage':key.split('_',2)[2]})
  if key in auxiliary:auxsummary[key]=stats;continue
  summary[key]=stats;folder=root/'attention'/key;folder.mkdir(parents=True,exist_ok=True)
  for ident,a in zip(ids,att):np.save(folder/(ident+'.npy'),a.reshape(14,14).astype(np.float32))
  (root/'per_phrase_candidate').mkdir(exist_ok=True);np.savez_compressed(root/'per_phrase_candidate'/(key+'.npz'),**rows)
  (root/'spatial_priors').mkdir(exist_ok=True);np.save(root/'spatial_priors'/('mean_attention_'+key+'.npy'),att.mean(0).reshape(14,14));np.save(root/'spatial_priors'/('peak_hist_'+key+'.npy'),np.bincount(att.argmax(1),minlength=196).reshape(14,14)/len(att))
  temperature[key]={}
  for tau in [.03,.05,.07,.10,.20]:
   temp,_=core(softmax(logits,tau),g);temperature[key][str(tau)]={m:temp[m] for m in ['mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor']}
  print('metrics',key,stats['pointing']['mean'],stats['mass_gain']['mean'],stats['semantic_mass_excess']['mean'],flush=True)
 manifest['pareto_front']=pareto_front(summary);manifest['status']='complete'
 write_json(root/'per_candidate_metrics.json',summary);write_json(root/('small_summary.json' if args.max_images else 'full_summary.json'),summary);write_json(root/'input_stage_metrics.json',auxsummary);write_json(root/'temperature_summary.json',temperature);write_json(root/'manifest.json',manifest)
 print('LOCAL_EVIDENCE_COMPLETE',len(candidates),len(ids),flush=True)
if __name__=='__main__':main()
