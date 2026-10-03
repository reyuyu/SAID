"""Read-only checkpoint discovery; inspect structures before choosing CE/BCE.

The broad scan reads only ZIP pickle metadata, without deserializing objects.
Only plausible beta-CLIP candidates are SHA256-hashed and loaded on the meta
device with the restricted weights-only loader. Model tensors are never edited.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickletools
import subprocess
import zipfile


ROOTS=['/root/lk_projects/B-CLIP-official','/root/lk_projects/SAID','/root','/tmp']
SUFFIXES=('.pt','.pth','.ckpt')
MARKERS=('text_conditioned_patches_block','text_conditioning_mode','tcil_loss_mode',
         'k_positives_ce','k_positives_bce','CLIP_VITB16_OPENAI','CLIP_VITL14_OPENAI')


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def strings_from_zip(path):
    if not zipfile.is_zipfile(path):return [],'non_zip'
    with zipfile.ZipFile(path) as archive:
        files=[n for n in archive.namelist() if n.endswith('data.pkl')]
        if not files:return [],'no_data_pickle'
        raw=archive.read(files[0])
        strings=[]
        for opcode,value,_ in pickletools.genops(raw):
            if opcode.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE','BINUNICODE8') and isinstance(value,str):
                strings.append(value)
        return strings,'zip_pickle_metadata'


def serializable(value):
    if isinstance(value,argparse.Namespace):return serializable(vars(value))
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return repr(value)


def inspect_candidate(path):
    import torch
    record=dict(path=str(path.resolve()),filename=path.name,size_bytes=path.stat().st_size,
                sha256=digest(path),checkpoint_identity='unknown',identity_confidence='unknown')
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            payload=torch.load(path,map_location='meta',weights_only=True)
        record['torch_load_top_level_keys']=list(payload) if isinstance(payload,dict) else None
        args={}
        for name in ('args','config'):
            source=payload.get(name) if isinstance(payload,dict) else None
            if isinstance(source,argparse.Namespace):source=vars(source)
            if isinstance(source,dict):args.update(source)
        record.update(epoch=serializable(payload.get('epoch')),args_config=serializable(args))
        state=payload.get('state_dict',payload.get('model',payload))
        state={k.removeprefix('module.'):v for k,v in state.items() if torch.is_tensor(v)}
        shapes={k:list(v.shape) for k,v in state.items()}
        key_shapes={k:v for k,v in shapes.items() if k in (
            'image_projection','visual.proj','visual.pos_embed','visual.positional_embedding',
            'visual.patch_embed.proj.weight','visual.conv1.weight',
            'positional_embedding','positional_embedding_res','text_projection') or
            'text_conditioned_patches_block' in k}
        mode=args.get('tcil_loss_mode');identity={'k_positives_ce':'CE','k_positives_bce':'BCE'}.get(mode,'unknown')
        conditioned=any('text_conditioned' in key for key in state)
        projection=shapes.get('image_projection',shapes.get('visual.proj'))
        text_projection=shapes.get('text_projection')
        vision_positions=shapes.get('visual.pos_embed',shapes.get('visual.positional_embedding'))
        patch_weight=shapes.get('visual.patch_embed.proj.weight',shapes.get('visual.conv1.weight'))
        b16=(projection==[768,512] and text_projection==[512,512] and
             vision_positions in ([1,197,768],[197,768]) and patch_weight==[768,3,16,16])
        positions=shapes.get('positional_embedding_res',shapes.get('positional_embedding'))
        context=positions[0] if positions else args.get('context_length')
        matches_release=(args.get('beta')==.5 and args.get('max_concepts')==30
                         and args.get('fg_loss_fn')=='cls+tcil'
                         and args.get('model')=='CLIP_VITB16_OPENAI')
        record.update(model=args.get('model'),context_length=context,tcil_loss_mode=mode,beta=args.get('beta'),
            fg_loss_fn=args.get('fg_loss_fn'),max_concepts=args.get('max_concepts'),
            state_dict_key_count=len(state),state_dict_numel=sum(v.numel() for v in state.values()),
            key_parameter_shapes=key_shapes,is_vit_b16=b16,has_beta_clip_conditioner=conditioned,
            official_release_args_match=matches_release,
            checkpoint_identity=identity,identity_confidence=('verified' if identity!='unknown' and conditioned and b16 and context==248 and matches_release else
                                                           'partially_verified' if identity!='unknown' or conditioned else 'unknown'))
    except Exception as exc:record.update(load_error=type(exc).__name__+': '+str(exc),checkpoint_identity='unknown')
    return record


def discover(roots):
    command=['rg','--files','-uuu','-g','*.pt','-g','*.pth','-g','*.ckpt',*roots]
    result=subprocess.run(command,text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError(result.stderr)
    seen=set();files=[];candidates=[];excluded_small=0;errors=[]
    for line in result.stdout.splitlines():
        path=Path(line)
        try:
            stat=path.stat();identity=(stat.st_dev,stat.st_ino)
            if identity in seen:continue
            seen.add(identity)
            if stat.st_size<1<<20:
                excluded_small+=1;continue
            strings,inspection=strings_from_zip(path)
            markers=[marker for marker in MARKERS if any(marker in value for value in strings)]
            related_by_name=any(word in str(path).lower() for word in ('beta-clip','beta_clip','b-clip'))
            record=dict(path=str(path.resolve()),filename=path.name,size_bytes=stat.st_size,
                metadata_inspection=inspection,beta_markers_found=markers,
                metadata_sample=[value for value in strings if value in (
                    'model','state_dict','config','completed_steps','mask_net.resblocks.0.attn.in_proj_weight',
                    'image_projection','text_projection','tcil_loss_mode')][:20])
            if markers or (related_by_name and inspection!='no_data_pickle'):
                candidate=inspect_candidate(path);record['candidate_inspection']=candidate;candidates.append(candidate)
            else:
                record['excluded_reason']='No beta-CLIP model/conditioner/TCIL markers in checkpoint metadata; not selected for evaluation.'
            files.append(record)
        except (OSError,ValueError,zipfile.BadZipFile) as exc:errors.append(dict(path=str(path),error=str(exc)))
    return dict(scanned_utc=datetime.now(timezone.utc).isoformat(),roots=roots,search_command=command,
        included_gitignored_files=True,checkpoint_files_ge_1MiB=len(files),excluded_small_files=excluded_small,
        inspected_files=files,candidates=candidates,scan_errors=errors,
        verified_ce=[x for x in candidates if x['checkpoint_identity']=='CE' and x['identity_confidence']=='verified'],
        verified_bce=[x for x in candidates if x['checkpoint_identity']=='BCE' and x['identity_confidence']=='verified'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+',default=ROOTS)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=discover(args.roots);Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('roots','checkpoint_files_ge_1MiB','excluded_small_files','scan_errors','verified_ce','verified_bce')},indent=2))


if __name__=='__main__':main()
