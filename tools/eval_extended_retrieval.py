#!/usr/bin/env python3
"""Unified, model-agnostic retrieval evaluator.

Feature metrics are usable with CPU arrays in tests.  Real model loading is only
possible with the explicit --execute-model/--model-factory contract.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, pathlib

def cache_identity(checkpoint, manifest_sha, tokenizer, context_length, preprocess, dtype):
    x={'checkpoint':checkpoint,'manifest_sha':manifest_sha,'tokenizer':tokenizer,'context_length':context_length,'preprocess':preprocess,'dtype':dtype}
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _topk(scores,k):
    # Stable descending order makes ties deterministic by candidate input order.
    return sorted(range(len(scores)),key=lambda j:(-float(scores[j]),j))[:k]

def recall_at_k(image_features,text_features,image_ids,text_positive_ids,k=1,query_chunk=0):
    import numpy as np
    im=np.asarray(image_features,dtype=np.float32); tx=np.asarray(text_features,dtype=np.float32)
    im=im/np.maximum(np.linalg.norm(im,axis=1,keepdims=True),1e-12); tx=tx/np.maximum(np.linalg.norm(tx,axis=1,keepdims=True),1e-12)
    image_ids=[str(x) for x in image_ids]; positives=[set(map(str,p)) if isinstance(p,(list,tuple,set)) else {str(p)} for p in text_positive_ids]
    # I2T: image has all captions whose positive image id matches.
    hits_i=[]
    for i in range(len(im)):
        order=_topk(im[i]@tx.T,k); hits_i.append(any(str(image_ids[i]) in positives[j] for j in order))
    # T2I: each caption maps to one or more positive image ids.
    idx={x:j for j,x in enumerate(image_ids)}; hits_t=[]
    for q,p in enumerate(positives): hits_t.append(any(idx.get(x,-1) in _topk(tx[q]@im.T,k) for x in p))
    return float(np.mean(hits_i)),float(np.mean(hits_t))

def evaluate_features(image_features,text_features,image_ids,positive_ids):
    return {f'R@{k}':recall_at_k(image_features,text_features,image_ids,positive_ids,k) for k in (1,5,10)}

def load_manifest(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if not rows: raise ValueError('empty manifest')
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description='Evaluate native bare-student retrieval from explicit features/model factory.')
    p.add_argument('--manifest',required=True); p.add_argument('--image-root',required=True); p.add_argument('--checkpoint'); p.add_argument('--model-factory',help='module:callable returning bare native student'); p.add_argument('--execute-model',action='store_true'); p.add_argument('--device',default='cpu'); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--score-chunk',type=int,default=0); p.add_argument('--cache-dir'); p.add_argument('--output')
    a=p.parse_args(argv)
    if a.execute_model:
        if not a.model_factory or not a.checkpoint: p.error('--execute-model requires --model-factory and --checkpoint')
        mod,name=a.model_factory.split(':',1); factory=getattr(importlib.import_module(mod),name); model=factory(a.checkpoint,device=a.device)
        rows=load_manifest(a.manifest)
        if not hasattr(model,'encode_manifest'):
            raise RuntimeError('model factory must return an adapter with encode_manifest(rows, image_root, batch_size)')
        image_features,text_features,image_ids,positive_ids=model.encode_manifest(rows,a.image_root,a.batch_size)
        result=evaluate_features(image_features,text_features,image_ids,positive_ids)
        payload={'status':'EXPLICIT_MODEL_RUN_OUTPUT','protocols':sorted({r.get('protocol_id') for r in rows}),'rows':len(rows),'metrics':result,'cache_identity':cache_identity(a.checkpoint,'unknown','adapter',248,'native','fp32')}
        if a.output: pathlib.Path(a.output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(json.dumps(payload)); return 0
    print(json.dumps({'status':'MODEL_INFERENCE_NOT_RUN','real_evaluation':'REAL_EVALUATION_NOT_RUN','manifest':a.manifest}))
    return 0
if __name__=='__main__': raise SystemExit(main())
