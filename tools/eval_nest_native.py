"""NEST export evaluation using existing canonical native retrieval implementations."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from model import longclip
from eval.retrieval.coco_retrieval import evaluate_coco
from tools.urban1k_retrieval import evaluate_urban1k, image_caption_pairs
from train.nested_semantic_data import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', choices=['coco','urban'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    destination=Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    model,preprocess=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
    model=model.float().to(args.device).eval()
    if args.dataset=='coco':
        ann=Path(args.root).parent/'annotations/captions_val2017.json'
        metadata=json.loads(ann.read_text())
        counts=Counter(row['image_id'] for row in metadata['annotations'])
        # The source has 25014 captions; the existing canonical evaluator takes
        # each image's first five, yielding exactly 25000 (same original order).
        assert len(metadata['images'])==5000 and len(counts)==5000 and min(counts.values())>=5
        result=evaluate_coco(model,preprocess,root=args.root,ann_file=str(ann),
                             batch_size=args.batch_size,device=args.device,
                             similarity_chunk=512,image_representation='legacy_cls')
    else:
        assert len(image_caption_pairs(args.root))==1000
        result=evaluate_urban1k(model,preprocess,root=args.root,batch_size=args.batch_size,device=args.device)
    result.update(checkpoint_sha256=file_sha(args.checkpoint), native_only=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
