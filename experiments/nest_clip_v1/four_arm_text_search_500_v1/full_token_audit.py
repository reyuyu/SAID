"""CPU replay of unchanged Full packing for scheduled and last50 token lengths."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import mmap
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from train.nested_semantic_data import text_views
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,records,dump

def chunk(item):
    step,ranks=item;torch.set_num_threads(1)
    index=Path('/root/lk_projects/SAID-nest-clip-v1/data_index');offset=np.load(index/'offsets.npy',mmap_mode='r');hist=Counter()
    with (index/'records.jsonl').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for rank in ranks:
            sampling=rank['sampling'];ids=sampling['sample_ids'];strings=[];tokens=[]
            for sid in ids:
                i=sid-1000;caption=json.loads(data[offset[i]:offset[i+1]])['caption'];v=text_views(caption)
                strings.append(v['views'][0]);tokens.append(v['tokens_f'].tolist());hist[int(v['tokens_f'].argmax())+1]+=1
            digest=hashlib.sha256(json.dumps(dict(sample_ids=ids,F=strings,tokens_f=tokens),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            assert digest==sampling['full_view_sha256'],(step,rank['rank'])
    return step,dict(hist)

def summary(h):
    n=sum(h.values());total=sum(int(k)*v for k,v in h.items());return {'count':n,'token_sum':total,'mean':total/n,'histogram':dict(sorted(h.items()))}

def main():
    torch.set_num_threads(1);baseline=load(EXP/'BASELINE.json');rows=records(Path(baseline['root'])/'steps.jsonl')
    steps=sorted(set([1,100,200,500]+list(range(451,501))));hist={}
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(chunk,(step,rows[step-1]['rank_health'])) for step in steps]
        for f in as_completed(futures):
            step,h=f.result();hist[step]=h
            if len(hist)%10==0:print(json.dumps({'Full_replay_steps':len(hist),'total_steps':len(steps)}),flush=True)
    last=Counter()
    for step in range(451,501):last.update(hist[step])
    result={'passed':True,'context':'Full effective tokens incl SOT/EOT; replayed baseline IDs and Full hash checked for each sampled batch; common all arms',
      'matched_steps':len(steps),'matched_rank_batches':len(steps)*4,'steps':{str(s):summary(hist[s]) for s in [1,100,200,500]},'last50':summary(last)}
    dump(EXP/'evidence/FULL_TOKEN_LENGTHS.json',result);print(json.dumps({'passed':True,'Full_last50_mean_tokens':result['last50']['mean']}),flush=True)

if __name__=='__main__':main()
