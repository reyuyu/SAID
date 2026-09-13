#!/usr/bin/env python3
"""Download/check retrieval benchmark metadata and write deterministic manifests.

This module is intentionally model-free.  Downloads happen only with --download;
importing it or asking for --help never imports torch or touches CUDA.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, re, shutil, tarfile, time
from urllib.request import Request, urlopen

SOURCES = {
    "docci": {"url": "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines", "license": "CC BY 4.0", "homepage": "https://google.github.io/docci/"},
    "dci": {"url": "https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz", "sha256": "9caff10cb6324c801d9020638f49925f04de87d897ca8614c599f3c43bef3aeb", "license": "CC-BY-NC", "homepage": "https://github.com/facebookresearch/DCI"},
    "long_dci": {"url": "https://huggingface.co/datasets/mderakhshani/Long-DCI", "license": "repository terms", "homepage": "https://huggingface.co/datasets/mderakhshani/Long-DCI"},
    "flickr30k": {"url": "https://huggingface.co/datasets/nlphuji/flickr30k", "license": "dataset terms", "homepage": "https://huggingface.co/datasets/nlphuji/flickr30k"},
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def safe_member(root, member):
    name=pathlib.PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts: raise ValueError(f'unsafe archive member: {member.name}')
    target=(pathlib.Path(root)/pathlib.Path(*name.parts)).resolve()
    if os.path.commonpath([str(pathlib.Path(root).resolve()),str(target)]) != str(pathlib.Path(root).resolve()): raise ValueError(f'archive escape: {member.name}')
    if member.issym() or member.islnk(): raise ValueError(f'links are not extracted: {member.name}')

def extract_safe(archive, root):
    pathlib.Path(root).mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as t:
        members=t.getmembers()
        for m in members: safe_member(root,m)
        t.extractall(root, members=members)

def download(url, dest, expected=None, rate_mib=20):
    dest=pathlib.Path(dest); part=dest.with_suffix(dest.suffix+'.part'); dest.parent.mkdir(parents=True,exist_ok=True)
    start=part.stat().st_size if part.exists() else 0
    headers={'User-Agent':'SAID-retrieval-data-prep/0.1'}
    if start: headers['Range']=f'bytes={start}-'
    with urlopen(Request(url,headers=headers),timeout=60) as r, open(part,'ab' if start else 'wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
            if rate_mib>0: time.sleep(len(b)/(rate_mib*1024*1024))
    os.replace(part,dest)
    digest=sha256(dest)
    if expected and digest.lower()!=expected.lower(): raise ValueError(f'SHA256 mismatch: {digest} != {expected}')
    return digest

def _row(protocol, image_id, image_path, caption_id, caption, split, positive=None, source=None):
    return {'protocol_id':protocol,'image_id':str(image_id),'image_path':str(image_path),'caption_id':str(caption_id),'caption':caption,'positive_image_id':str(positive if positive is not None else image_id),'split':split,'source':source or protocol}

def validate_rows(rows, root=None):
    seen_i=set(); seen_c=set()
    for r in rows:
        if not r.get('caption','').strip(): raise ValueError(f'empty caption: {r}')
        if r['image_id'] in seen_i and r['protocol_id'] not in ('flickr30k_full','flickr30k_test1k'): raise ValueError(f'duplicate image_id: {r["image_id"]}')
        if r['caption_id'] in seen_c: raise ValueError(f'duplicate caption_id: {r["caption_id"]}')
        seen_i.add(r['image_id']); seen_c.add(r['caption_id'])
        if root:
            p=(pathlib.Path(root)/r['image_path']).resolve()
            if os.path.commonpath([str(pathlib.Path(root).resolve()),str(p)]) != str(pathlib.Path(root).resolve()): raise ValueError(f'path escape: {r["image_path"]}')
    return len(rows)

def write_manifest(rows, path):
    rows=sorted(rows,key=lambda r:(r['protocol_id'],r['image_id'],r['caption_id']))
    validate_rows(rows)
    pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8',newline='') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    return sha256(path),len(rows)

def parse_docci(jsonl, manifest, image_root='.'):
    rows=[]
    with open(jsonl,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line); split=str(x.get('split',x.get('partition',''))).lower()
            if split not in ('test','test_a','test-b','test_b'): continue
            iid=x.get('example_id',x.get('image_id')); cap=x.get('description',x.get('caption',''))
            rows.append(_row('docci_test',iid,str(x.get('image',iid+'.jpg')),iid,cap,'test',iid,'docci'))
    return write_manifest(rows,manifest)

def parse_tsv(path, manifest, protocol='long_dci'):
    rows=[]
    with open(path,encoding='utf-8',newline='') as f:
        sample=f.read(4096); f.seek(0); dialect=csv.Sniffer().sniff(sample,delimiters='\t,')
        for i,r in enumerate(csv.DictReader(f,dialect=dialect)):
            iid=r.get('image_id') or r.get('image') or r.get('id'); cap=r.get('caption') or r.get('text') or r.get('description')
            if iid is None or cap is None: continue
            rows.append(_row(protocol,iid,r.get('image_path',r.get('image',str(iid))),r.get('caption_id',f'{iid}_{i}'),cap,r.get('split','test'),iid,protocol))
    return write_manifest(rows,manifest)

def parse_dci(annotation_dir, manifest, split='all'):
    rows=[]
    for path in sorted(pathlib.Path(annotation_dir).glob('*-data.json')):
        with open(path,encoding='utf-8') as f: x=json.load(f)
        iid=path.name[:-10]; image=x.get('image',iid+'.jpg'); cap=x.get('short_caption') or x.get('extra_caption','')
        rows.append(_row('dci_full',iid,image,iid,cap,split,iid,'dci'))
    return write_manifest(rows,manifest)

def parse_flickr(path, manifest, test1k=False):
    rows=[]
    with open(path,encoding='utf-8',newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            iid=r.get('image_id') or r.get('image') or r.get('filename'); cap=r.get('caption') or r.get('sentence') or r.get('text')
            split=(r.get('split') or '').lower()
            if not iid or not cap or (test1k and split not in ('test','test1k','test_1k')): continue
            rows.append(_row('flickr30k_test1k' if test1k else 'flickr30k_full',iid,r.get('image_path',iid),r.get('caption_id',f'{iid}_{i}'),cap,split or ('test1k' if test1k else 'all'),iid,'flickr30k'))
    return write_manifest(rows,manifest)

def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare retrieval benchmark files (model-free).')
    p.add_argument('--root',default='/root/datasets/retrieval_benchmarks'); p.add_argument('--dataset',choices=sorted(SOURCES)); p.add_argument('--download',action='store_true'); p.add_argument('--check',action='store_true'); p.add_argument('--manifest'); p.add_argument('--input'); p.add_argument('--rate-mib',type=float,default=20)
    a=p.parse_args(argv); root=pathlib.Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if a.download:
        if not a.dataset: p.error('--download requires --dataset')
        s=SOURCES[a.dataset]; out=root/a.dataset/pathlib.Path(s['url'].split('/')[-1] or 'source');
        try:
            d=download(s['url'],out,s.get('sha256'),a.rate_mib); print(json.dumps({'dataset':a.dataset,'status':'DOWNLOADED','path':str(out),'sha256':d}))
        except Exception as e: print(json.dumps({'dataset':a.dataset,'status':'BLOCKED_OR_FAILED','error':str(e)})); return 2
    if a.input and a.manifest:
        if not a.dataset: p.error('--dataset required with --input')
        if a.dataset=='docci': result=parse_docci(a.input,a.manifest)
        elif a.dataset=='long_dci': result=parse_tsv(a.input,a.manifest)
        elif a.dataset=='flickr30k': result=parse_flickr(a.input,a.manifest)
        elif a.dataset=='dci': result=parse_dci(a.input,a.manifest)
        else: result=parse_tsv(a.input,a.manifest,'dci')
        print(json.dumps({'status':'PREPARED','manifest':a.manifest,'sha256':result[0],'rows':result[1]})); return 0
    print(json.dumps({'status':'READY_FOR_EXPLICIT_INPUT','root':str(root),'datasets':sorted(SOURCES)})); return 0
if __name__=='__main__': raise SystemExit(main())
