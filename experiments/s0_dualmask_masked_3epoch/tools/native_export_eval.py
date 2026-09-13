"""Portable strict export/evaluation entry; calls the unmodified frozen retrieval functions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

BUNDLE=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()


def read_frozen(name):
    path=BUNDLE/'reference_eval'/name
    spec=importlib.util.spec_from_file_location('frozen_'+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--action',required=True,choices=['export','coco','urban'])
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--step',type=int)
    p.add_argument('--bare-output',type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--coco-root',type=Path)
    p.add_argument('--urban-root',type=Path)
    args=p.parse_args()
    if args.report.exists() or (args.action=='export' and args.bare_output and args.bare_output.exists()):
        p.error('output already exists; choose a fresh output/report path')
    started=time.perf_counter()
    sys.path[:0]=[str(args.repo),str(args.repo/'train')]
    import torch
    from model import longclip
    from model.dual_mask_suffix import DualMaskSuffixTrainModule
    from train_dual_mask_suffix import load_checkpoint, export_bare_student
    model,preprocess=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    report={'action':args.action,'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint)}
    if args.action=='export':
        if args.step is None or args.bare_output is None:
            p.error('export requires --step and --bare-output')
        module=DualMaskSuffixTrainModule(model,'masked')
        payload=load_checkpoint(str(args.checkpoint),module)
        assert payload['completed_steps']==args.step
        assert payload['config']['formal_optimizer_updates']==args.step
        steps={name:[int(s['step']) for s in state['state'].values() if 'step' in s]
               for name,state in payload['optimizer_states'].items()}
        # V<2 can leave gate grads=None: optimizer step counters may legitimately lag.
        assert all(values and min(values)>0 and max(values)<=args.step for values in steps.values())
        args.bare_output.parent.mkdir(parents=True,exist_ok=True)
        export_bare_student(str(args.checkpoint),str(args.bare_output))
        student,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
        student.load_state_dict(torch.load(args.bare_output,map_location='cpu',weights_only=False),strict=True)
        error=max(float((v-student.state_dict()[k]).abs().max()) for k,v in module.clip.state_dict().items())
        assert error==0
        report.update(actual_updates=args.step,training_sha=payload['config']['training_sha'],
            strict_clip=True,strict_suffix_gate=True,strict_bare_student=True,
            bare_state_max_abs_error=error,bare_student=str(args.bare_output),bare_sha256=sha(args.bare_output),
            optimizer_steps={k:{'min':min(v),'max':max(v),'count':len(v)} for k,v in steps.items()})
    else:
        model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False),strict=True)
        model.to(args.device).eval()
        report['strict_bare_student']=True
        if args.action=='coco':
            if args.coco_root is None:
                p.error('coco requires --coco-root')
            annotation=args.coco_root/'annotations/captions_val2017.json'
            assert len(json.loads(annotation.read_text())['images'])==5000
            evaluator=read_frozen('coco_retrieval.py')
            metrics=evaluator.evaluate_coco(model,preprocess,root=str(args.coco_root/'val2017'),
                ann_file=str(annotation),batch_size=64,similarity_chunk=512,device=args.device,
                image_representation='legacy_cls')
            report.update(metrics=metrics,n_images=5000,n_texts=25000,similarity_chunk=512,
                annotation_sha256=sha(annotation),evaluator_sha256=sha(BUNDLE/'reference_eval/coco_retrieval.py'))
        else:
            if args.urban_root is None:
                p.error('urban requires --urban-root')
            evaluator=read_frozen('urban1k_retrieval.py')
            assert len(evaluator.image_caption_pairs(str(args.urban_root)))==1000
            report.update(urban1k=evaluator.evaluate_urban1k(model,preprocess,root=str(args.urban_root),
                batch_size=64,device=args.device),dataset=evaluator.dataset_fingerprint(str(args.urban_root)),
                evaluator_sha256=sha(BUNDLE/'reference_eval/urban1k_retrieval.py'))
    report['elapsed_seconds']=time.perf_counter()-started
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
