#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load(path): return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output)
    metrics={'COCO':load(out/'coco_native.json'),'Urban-1k':load(out/'urban_native.json')}
    for name in ('flickr_test1k','docci'):
        metrics[name]=load(out/name/f'{name}.json')
    result={'completed_updates':3651,'checkpoint_sha256':file_sha(out/'step003651.pt'),
            'bare_sha256':file_sha(out/'student_step3651.pt'),'export_check':load(out/'export-check.json'),
            'metrics':metrics,'excluded_evaluations':['DCI Full','Long-DCI']}
    (out/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str(out/'FINAL_RESULTS.json')}))
if __name__=='__main__': main()
