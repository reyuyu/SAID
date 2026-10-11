#!/usr/bin/env python3
"""Run the native evaluator with an explicit local dataset root."""
import argparse,subprocess,sys
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset',choices=['urban1k','coco'],required=True); p.add_argument('--data-root',required=True); p.add_argument('--output',default='evaluation.json'); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
 dataset='urban' if a.dataset=='urban1k' else 'coco'; subprocess.run([sys.executable,'-m','tools.eval_nest_native','--checkpoint',a.checkpoint,'--dataset',dataset,'--root',a.data_root,'--output',a.output,'--device',a.device,'--batch-size',str(a.batch_size)],check=True)
if __name__=='__main__': main()
