#!/usr/bin/env python3
"""Download ModelScope files through ms-hub and verify the published hash."""
import argparse,hashlib,subprocess
from pathlib import Path
REPO='hongchenyang0814/SAID-HNS-v1-ViT-B16-Epoch3'
EXPECTED='a60150575f7e8f7b41effb730c0f9d866757954833a240813d4695d9a1cfcb44'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',default='checkpoints/hns_v1_b16'); p.add_argument('--revision'); a=p.parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); cmd=['ms-hub','download',REPO,'inference/student_step3651.pt','--local-dir',str(out)]
 if a.revision: cmd += ['--revision',a.revision]
 subprocess.run(cmd,check=True); path=out/'inference/student_step3651.pt'; got=sha(path)
 if got!=EXPECTED: raise SystemExit(f'SHA256 mismatch: {got} != {EXPECTED}')
 print(f'{path} {got}')
if __name__=='__main__': main()
