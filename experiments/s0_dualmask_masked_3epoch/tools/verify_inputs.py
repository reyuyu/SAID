"""Read-only verification for code, required asset bytes and optional evaluation files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BUNDLE=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--init',required=True,type=Path)
    parser.add_argument('--train-json',required=True,type=Path)
    parser.add_argument('--clip-base',type=Path,default=Path.home()/'.cache/clip/ViT-B-16.pt')
    parser.add_argument('--coco-root',required=True,type=Path)
    parser.add_argument('--urban-root',required=True,type=Path)
    parser.add_argument('--eval-files',action='store_true')
    parser.add_argument('--check-training-paths',action='store_true')
    parser.add_argument('--training-image-root',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-kind',choices=['masked500','masked1000'])
    args=parser.parse_args()
    checks=[]
    def check(label,path,expected):
        exists=path.is_file()
        actual=sha(path) if exists else None
        checks.append(dict(label=label,path=str(path),exists=exists,expected_sha256=expected,
                           actual_sha256=actual,passed=actual==expected))
    for rel,value in json.loads((BUNDLE/'manifests/code_sha256.json').read_text()).items():
        check('code:'+rel,args.repo/rel,value)
    assets=json.loads((BUNDLE/'manifests/assets.json').read_text())
    inputs=dict(common_init=args.init,sharegpt4v_annotation=args.train_json,clip_base=args.clip_base,
                coco_annotation=args.coco_root/'annotations/captions_val2017.json')
    if args.checkpoint:
        if not args.checkpoint_kind:
            parser.error('--checkpoint requires --checkpoint-kind')
        inputs[args.checkpoint_kind]=args.checkpoint
    for name,path in inputs.items():
        check(name,path,assets[name]['sha256'])
    if args.eval_files:
        for kind,files in json.loads((BUNDLE/'manifests/evaluation_files_sha256.json').read_text()).items():
            root=args.coco_root/'val2017' if kind=='coco' else args.urban_root
            for rel,value in files.items():
                check(kind+':'+rel,root/rel,value)
    path_report=None
    if args.check_training_paths:
        if args.training_image_root is None:
            parser.error('--check-training-paths requires --training-image-root')
        records=json.loads(args.train_json.read_text())[1000:]
        missing=[r['image'] for r in records if not (args.training_image_root/r['image']).is_file()]
        path_report={'rows':len(records),'missing_count':len(missing),'first_missing':missing[:20],
                     'image_bytes_verified':False}
    failed=[c for c in checks if not c['passed']]
    report={'status':'passed' if not failed and not (path_report and path_report['missing_count']) else 'failed',
            'checks':len(checks),'failed':failed,'training_paths':path_report,
            'scope':'SHA256 for listed inputs/code/eval files; optional training-path existence, not training-image bytes'}
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
