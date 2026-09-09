"""Frozen root-cause probes; never changes benchmark orientation or weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .semantic_grounding_eval import sha256, write_json, direct_attention
from .grounding_metrics import patch_coverage, contains


def setup():
    torch.set_num_threads(4)
    torch.manual_seed(231)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(stage, manifest, device):
    import clip
    from model import longclip
    from model.salu_model import SALUModel
    if stage == 'initial':
        model, preprocess = clip.load('ViT-B/16', device='cpu', jit=False)
        return model.float().eval().to(device), preprocess, clip.tokenize, None
    path = Path('runs_salu')/stage/'salu_said_only_last.pt'
    assert sha256(path) == manifest['checkpoints'][stage]['sha256'], 'checkpoint hash mismatch'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    backbone, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu')
    salu = SALUModel(backbone, tau_said=payload['args'].get('tau_said', .07),
                     said_loss_mode=payload['args'].get('said_loss_mode', 'positive'))
    salu.load_state_dict(payload['model'], strict=True)
    salu.float().eval().to(device)
    return salu.clip, preprocess, longclip.tokenize, salu


def captured_patches(model, tensor, layers=(11,)):
    """Observe the actual visual forward via hooks, not the export helper."""
    outputs = {}
    handles = []
    for i in layers:
        def hook(module, args, output, index=i):
            outputs[index] = output
        handles.append(model.visual.transformer.resblocks[i].register_forward_hook(hook))
    try:
        model.encode_image(tensor)
    finally:
        for handle in handles:
            handle.remove()
    return {i: model.visual.ln_post(x.permute(1,0,2)[:,1:]) @ model.visual.proj
            for i,x in outputs.items()}


def logits_from_features(patches, texts, router=None):
    if router is None:
        return F.normalize(texts, dim=-1) @ F.normalize(patches, dim=-1).T
    q = F.normalize(F.linear(texts, router.q_proj.weight, router.q_proj.bias), dim=-1)
    k = F.normalize(F.linear(patches, router.k_proj.weight, router.k_proj.bias), dim=-1)
    return q @ k.T


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a,b)[0,1])


def choose_phrases(phrases):
    """Fixed 100 distinct images, covering all available official categories."""
    rng = np.random.default_rng(231)
    keys = sorted(phrases)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    selected, images = [], set()
    categories = sorted({c for p in phrases.values() for c in p['categories']})
    for category in categories:
        key = next(k for k in keys if category in phrases[k]['categories'] and phrases[k]['image_id'] not in images)
        selected.append(key); images.add(phrases[key]['image_id'])
    for key in keys:
        if len(selected) == 100:
            break
        if phrases[key]['image_id'] not in images:
            selected.append(key); images.add(phrases[key]['image_id'])
    return selected


def live_probe(root, image_root, output, device):
    manifest = json.loads((root/'manifest.json').read_text())
    phrases = json.loads((root/'per_phrase.json').read_text())
    selected = choose_phrases(phrases)
    write_json(output/'selection.json', {'phrase_ids': selected, 'images': [phrases[k]['image_id'] for k in selected],
                                        'categories': sorted({c for k in selected for c in phrases[k]['categories']}), 'seed':231})
    results = {}
    for stage in ['initial', 'phase21', 'phase22']:
        model, preprocess, tokenize, salu = load_model(stage, manifest, device)
        variants = [stage+'_direct'] if stage == 'initial' else [stage+'_router']
        if stage == 'phase22': variants.append('phase22_direct')
        rows = {v:[] for v in variants}
        with torch.inference_mode():
            for key in selected:
                p = phrases[key]
                with Image.open(image_root/(p['image_id']+'.jpg')) as original:
                    tensor = preprocess(original)[None].to(device)
                h = captured_patches(model, tensor)[11][0]
                t = F.normalize(model.encode_text(tokenize([p['phrase']], truncate=False).to(device)).float(),dim=-1)
                for v in variants:
                    router = salu.said_router if v.endswith('router') else None
                    logits = logits_from_features(h,t,router)
                    a = torch.softmax(logits/(router.tau_said if router else .07),dim=-1)[0].cpu().numpy().reshape(14,14)
                    saved = np.load(root/'attention'/v/(key+'.npy'))
                    diff = np.abs(a-saved)
                    w = patch_coverage(p['boxes'])
                    l = logits[0].cpu().numpy().reshape(14,14)
                    rows[v].append({'phrase_id':key,'max_abs_diff':float(diff.max()), 'mean_abs_diff':float(diff.mean()),
                                    'live_logit_corr':corr(l,w),'rotated_live_logit_corr':corr(np.rot90(l,2),w),
                                    'rotated_gt_corr':corr(l,np.rot90(w,2))})
        for v in variants:
            worst = max(rows[v],key=lambda r:r['max_abs_diff'])
            results[v]={'phrases':100, 'max_abs_diff':worst['max_abs_diff'],
                        'mean_abs_diff':float(np.mean([r['mean_abs_diff'] for r in rows[v]])),
                        'worst_phrase_id':worst['phrase_id'],'pass':worst['max_abs_diff']<=1e-6,'rows':rows[v]}
            print(v,{k:x for k,x in results[v].items() if k!='rows'},flush=True)
        write_json(output/'saved_vs_live.json',results)
        if any(not results[v]['pass'] for v in variants):
            raise RuntimeError('STOP: saved/live mismatch')
        del model,salu
        torch.cuda.empty_cache()


def crop_exif_probe(root,image_root,output,preprocess):
    selection=json.loads((output/'selection.json').read_text())['images']
    manifest=json.loads((root/'manifest.json').read_text())
    phrases=json.loads((root/'per_phrase.json').read_text())
    images={p['id']:p for p in manifest['images']}
    dist={k:0 for k in ['1','3','6','8','missing','other']}
    nontrivial=[]
    for key in images:
        with Image.open(image_root/(key+'.jpg')) as im:
            value=im.getexif().get(274)
            label='missing' if value is None else str(value) if value in [1,3,6,8] else 'other'
            dist[label]+=1
            if value not in [None,1]: nontrivial.append({'image':key,'orientation':value})
    mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
    rows=[]
    (output/'crop_overlays').mkdir(exist_ok=True)
    for key in selection:
        with Image.open(image_root/(key+'.jpg')) as original:
            rgb=((preprocess(original)*std+mean).clamp(0,1)*255).permute(1,2,0).numpy()
            recovered=np.rint(rgb).astype(np.uint8)
            width,height=original.size
            size=(224,int(224*height/width)) if width<=height else (int(224*width/height),224)
            left,top=round((size[0]-224)/2),round((size[1]-224)/2)
            manual=np.array(original.resize(size,Image.Resampling.BICUBIC).crop((left,top,left+224,top+224)).convert('RGB'))
        saved=np.array(Image.open(root/'images'/(key+'.png')).convert('RGB'))
        diff=np.abs(recovered.astype(int)-saved.astype(int))
        geometry=images[key]['geometry']
        match=list(size)==geometry['resized_size'] and [left,top]==geometry['crop_xy']
        box_diff=0.0
        for phrase_id in images[key]['phrase_ids']:
            p=phrases[phrase_id]
            boxes=np.asarray(p['original_boxes'])*np.array([size[0]/width,size[1]/height]*2)
            boxes-=np.array([left,top,left,top])
            boxes=boxes.clip(0,224)
            boxes=boxes[np.all(boxes[:,2:]>boxes[:,:2],axis=1)]
            assert boxes.shape==np.asarray(p['boxes']).shape
            box_diff=max(box_diff,float(np.abs(boxes-np.asarray(p['boxes'])).max()))
        row={'image_id':key,'max_pixel_diff':int(diff.max()),'mean_pixel_diff':float(diff.mean()),
             'box_coordinate_max_diff':box_diff,
             'manual_saved_max_diff':int(np.abs(manual.astype(int)-saved.astype(int)).max()),
             'float_roundtrip_max_diff':float(np.abs(rgb-saved).max()),'geometry_match':match}
        rows.append(row)
        overlay=Image.fromarray(recovered);draw=ImageDraw.Draw(overlay)
        for entity,boxes in images[key]['entities'].items():
            for box in boxes: draw.rectangle(tuple(box),outline='lime',width=1)
        overlay.save(output/'crop_overlays'/(key+'.png'))
    result={'images':100,'max_pixel_diff':max(r['max_pixel_diff'] for r in rows),
            'mean_pixel_diff':float(np.mean([r['mean_pixel_diff'] for r in rows])),
            'geometry_match':all(r['geometry_match'] for r in rows),
            'box_coordinate_max_diff':max(r['box_coordinate_max_diff'] for r in rows),
            'pass':all(r['max_pixel_diff']<=1 and r['manual_saved_max_diff']==0 and r['geometry_match'] and r['box_coordinate_max_diff']<=1e-9 for r in rows),
            'exif':dist,'nontrivial_exif':nontrivial,'rows':rows}
    write_json(output/'crop_exif.json',result)
    return result


def patch_order_probe(model,device):
    v=model.visual
    code=torch.arange(196,device=device).reshape(14,14).float()/195
    x=code.repeat_interleave(16,0).repeat_interleave(16,1)[None,None].repeat(1,3,1,1)
    conv=v.conv1(x)
    isolated=F.conv2d(F.unfold(x,16,stride=16).transpose(1,2).reshape(196,3,16,16),v.conv1.weight).reshape(196,-1)
    flat=conv.flatten(2).permute(0,2,1)[0]
    diff=(flat-isolated).abs().max().item()
    captured={}
    def before_ln(m,args): captured['pre_ln']=args[0].clone()
    handle=v.ln_pre.register_forward_pre_hook(before_ln)
    model.encode_image(x);handle.remove()
    expected=torch.cat([v.class_embedding[None,None],flat[None]],1)+v.positional_embedding
    positional=(captured['pre_ln']-expected).abs().max().item()
    # Verify the actual transformer implementation equals explicit residual
    # updates in every sequence slot. No identity-content assumption is made.
    slots=v.ln_pre(expected).permute(1,0,2)
    actual=v.transformer(slots)
    manual=slots.clone()
    for block in v.transformer.resblocks:
        z=block.ln_1(manual)
        manual=manual+block.attn(z,z,z,need_weights=False,attn_mask=block.attn_mask)[0]
        manual=manual+block.mlp(block.ln_2(manual))
    sequence=(actual-manual).abs().max().item()
    if hasattr(model,'encode_image_with_patches'):
        exported=model.encode_image_with_patches(x)[1]
        expected_patch=v.ln_post(actual.permute(1,0,2)[:,1:])@v.proj
        final_diff=(exported-expected_patch).abs().max().item()
    else:
        from .semantic_grounding_eval import native_clip_patches
        expected_patch=v.ln_post(actual.permute(1,0,2)[:,1:])@v.proj
        final_diff=(native_clip_patches(model,x)-expected_patch).abs().max().item()
    return {'positions':196,'corners_checked':[[0,0],[0,13],[13,0],[13,13]],
            'conv_max_diff':diff,'positional_max_diff':positional,'sequence_max_diff':sequence,
            'final_slot_max_diff':final_diff,'pass':max(diff,positional,sequence,final_diff)<=1e-5,
            'note':'Position embeddings are indexed by raster slots; their learned content is not a spatial label.'}


def later_probes(root,image_root,output,device):
    live=json.loads((output/'saved_vs_live.json').read_text())
    assert len(live)==4 and all(v['pass'] for v in live.values()),'STOP: live prerequisite'
    manifest=json.loads((root/'manifest.json').read_text())
    phrases=json.loads((root/'per_phrase.json').read_text())
    small=json.loads(Path('outputs/semantic_grounding_small/manifest.json').read_text())['images']
    order={}
    # Pipeline gates first. No layerwise inference until all gates pass.
    for stage in ['initial','phase22']:
        model,preprocess,tokenize,salu=load_model(stage,manifest,device)
        with torch.inference_mode():
            if stage=='initial':
                crop=crop_exif_probe(root,image_root,output,preprocess)
                assert crop['pass'],'STOP: crop mismatch'
            order[stage]=patch_order_probe(model,device)
        write_json(output/'patch_order.json',order)
        assert order[stage]['pass'],'STOP: patch order'
        if stage=='phase22':
            import copy
            identity=copy.deepcopy(salu.said_router)
            identity.q_proj=torch.nn.Identity();identity.k_proj=torch.nn.Identity()
            worst=0.;count=0
            with torch.inference_mode():
                for image in small:
                    with Image.open(image_root/(image['id']+'.jpg')) as original: tensor=preprocess(original)[None].to(device)
                    h=captured_patches(model,tensor)[11][0]
                    ids=image['phrase_ids']
                    t=F.normalize(model.encode_text(tokenize([phrases[k]['phrase'] for k in ids],truncate=False).to(device)).float(),dim=-1)
                    a=direct_attention(h,t,.07)
                    b=identity(t,h[None].expand(len(ids),-1,-1))[0]
                    worst=max(worst,float((a-b).abs().max()));count+=len(ids)
            write_json(output/'identity_router.json',{'phrases':count,'max_abs_diff':worst,'pass':worst<=1e-6})
            assert worst<=1e-6,'STOP: identity mismatch'
        del model,salu
        torch.cuda.empty_cache()
    summary={}
    weights={k:patch_coverage(phrases[k]['boxes']) for im in small for k in im['phrase_ids']}
    for stage in ['initial','phase22']:
        model,preprocess,tokenize,salu=load_model(stage,manifest,device)
        rows=[]
        with torch.inference_mode():
            for index,im in enumerate(small):
                with Image.open(image_root/(im['id']+'.jpg')) as original:tensor=preprocess(original)[None].to(device)
                features=captured_patches(model,tensor,(3,6,9,11))
                ids=im['phrase_ids']
                t=F.normalize(model.encode_text(tokenize([phrases[k]['phrase'] for k in ids],truncate=False).to(device)).float(),dim=-1)
                for layer,h in features.items():
                    logits=logits_from_features(h[0],t).cpu().numpy().reshape(-1,14,14)
                    attention=torch.softmax(torch.from_numpy(logits.reshape(-1,196))/.07,-1).numpy().reshape(-1,14,14)
                    for key,l,a in zip(ids,logits,attention):
                        w=weights[key]
                        for orientation in ['identity','rotate180']:
                            grid=a if orientation=='identity' else np.rot90(a,2)
                            r,c=np.unravel_index(grid.argmax(),grid.shape)
                            rows.append({'phrase_id':key,'layer_index':layer,'orientation':orientation,
                                         'pointing':contains(phrases[key]['boxes'],(c+.5)*16,(r+.5)*16),
                                         'gt_mass':float(np.sum(grid*w)),'mass_gain':float(np.sum(grid*w)-w.mean()),
                                         'logit_gt_corr':corr(l if orientation=='identity' else np.rot90(l,2),w)})
                if index%32==0:print('layerwise',stage,index,flush=True)
        table=[]
        for layer in [3,6,9,11]:
            for orientation in ['identity','rotate180']:
                items=[r for r in rows if r['layer_index']==layer and r['orientation']==orientation]
                table.append({'layer_index':layer,'orientation':orientation,'phrases':len(items),
                              **{k:float(np.mean([r[k] for r in items if r[k] is not None])) for k in ['pointing','gt_mass','mass_gain','logit_gt_corr']}})
        summary[stage]=table
        write_json(output/('layerwise_'+stage+'_phrases.json'),rows)
        write_json(output/'layerwise_summary.json',summary)
        del model,salu
        torch.cuda.empty_cache()
    print('ROOT_CAUSE_PROBES_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['live','later'],required=True)
    parser.add_argument('--root',type=Path,default=Path('outputs/semantic_grounding'))
    parser.add_argument('--image_root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/grounding_root_cause'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();setup();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='live':live_probe(args.root,args.image_root,args.output,args.device)
    else:later_probes(args.root,args.image_root,args.output,args.device)


if __name__=='__main__':main()
