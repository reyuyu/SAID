"""Strict FP0 checkpoint export and frozen native retrieval protocols, once per target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import longclip


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    config=json.loads((run/'config.json').read_text())
    per_epoch=config['updates_per_epoch']
    summary=json.loads((run/'run_summary.json').read_text())
    if summary['optimizer_step'] != 2*per_epoch:
        raise RuntimeError('exact two-epoch completion required before evaluation chain')
    evaluation=run/'evaluation'; evaluation.mkdir(exist_ok=True)
    exports=run/'export'; exports.mkdir(exist_ok=True)
    for step in (0,1000,per_epoch,2*per_epoch):
        source=run/('FP0_update%06d.pt'%step)
        payload=torch.load(source,map_location='cpu',weights_only=False)
        if payload['objective']!='finelip_prefix' or payload['arm']!='FP0' or payload['optimizer_step']!=step:
            raise ValueError('wrong FP0 checkpoint identity')
        clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        result=clip.load_state_dict(payload['model'],strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        if not payload['model']: raise ValueError('empty student')
        out=exports/('FP0_student_update%06d.pt'%step)
        if not out.exists():
            torch.save({'model':payload['model'],'completed_steps':step,'optimizer_step':step,
                        'objective':'finelip_prefix','arm':'FP0','config':payload['config'],
                        'git_head':payload['git_sha']},out)
        else:
            old=torch.load(out,map_location='cpu',weights_only=False)
            if old['optimizer_step']!=step or any(not torch.equal(v,old['model'][k]) for k,v in payload['model'].items()):
                raise ValueError('existing export differs from exact source')
        provenance={'optimizer_step':step,'parent_checkpoint':str(source),'checkpoint_sha256':file_sha(source),
                    'student_sha256':file_sha(out),'tensors':len(payload['model']),'strict_load':'PASS',
                    'micro_step':payload['micro_step'],'sample_presentations':payload['sample_presentations']}
        (evaluation/('FP0_update%d_provenance.json'%step)).write_text(json.dumps(provenance,indent=2))
        del clip,payload
        label='FP0_update%d'%step
        diag=evaluation/(label+'_diagnostics.json')
        if not diag.exists():
            subprocess.run([sys.executable,str(ROOT/'tools/fp0_cohort_diagnostics.py'),
                            '--checkpoint',str(source),'--manifest',str(evaluation/'fixed64_manifest.json'),
                            '--out',str(diag)],cwd=ROOT,check=True)
        canonical=evaluation/(label+'_canonical.json')
        if not canonical.exists():
            command=[sys.executable,str(ROOT/'tools/phase30a_fixed_cohort_eval.py'),
                     '--label','finelip_prefix_FP0','--gap_anti_temperature','1.0',
                     '--usr_manifest','/root/SAID/outputs/validation/sharegpt4v1k_usr_manifest.json',
                     '--source_manifest','/root/SAID/outputs/validation/sharegpt4v1k_manifest.json',
                     '--sharegpt4v_manifest','/root/SAID/outputs/validation/sharegpt4v1k_manifest.json',
                     '--data_root',os.environ['SHARE4V_DATA_ROOT'],'--image_root',os.environ['SHARE4V_DATA_ROOT'],
                     '--image_batch_size','64','--canonical','--canonical_only','--coco',
                     '--canonical_tags',str(step),'--canonical_names',label,
                     '--checkpoints',str(step)+':'+str(out),'--output',str(canonical)]
            subprocess.run(command,cwd=ROOT,check=True)
        urban=evaluation/(label+'_urban1k.json')
        if not urban.exists():
            subprocess.run([sys.executable,'/root/SAID-gap-completion/tools/eval_urban1k_cls.py',
                            '--checkpoint',str(out),'--label',label,'--expect-steps',str(step),
                            '--base_model','ViT-B/16','--device','cuda','--batch_size','64',
                            '--urban_root','/root/datasets/Urban1k/Urban1k','--out',str(urban)],
                           cwd='/root/SAID-gap-completion',check=True)
        for path in (canonical,urban):
            json.loads(path.read_text())
        print('FP0_EVAL_DONE '+label,flush=True)


if __name__=='__main__': main()
