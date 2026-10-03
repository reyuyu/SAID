"""Same-model text redundancy and off-diagonal ambiguity, with no loss changes."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from model import longclip
from train.nested_semantic_data import sampled_text_views, file_sha

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/nest_clip_v1/balanced_summary_random_detail_500_v1'
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_summary_random_detail_500_v1')
INDEX=Path('/root/lk_projects/SAID-nest-clip-v1/data_index')
SHARED=Path('/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt')


def stats(x):
    x=x.float().cpu().numpy()
    return {'count':len(x),'mean':float(np.mean(x)),'median':float(np.median(x)),
            **{f'p{p}':float(np.percentile(x,p)) for p in [10,25,50,75,90,95]}}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['step0','step500'],required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    pool=np.load(RUN/'audit_subset_indices.npy').tolist()
    candidates=random.Random(0).sample(pool,2048)
    offset=np.load(INDEX/'offsets.npy',mmap_mode='r')
    names=['Full','Summary','All_Detail_raw','All_Detail_visible','Random_Detail','RandomK_P','RandomK_R']
    banks={n:[] for n in names};ids=[]
    with (INDEX/'records.jsonl').open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
            for i in candidates:
                caption=json.loads(data[offset[i]:offset[i+1]])['caption']
                old=sampled_text_views(caption,'random_k',0,0,i+1000)
                new=sampled_text_views(caption,'summary_random_detail',0,0,i+1000)
                if not new['valid']:continue
                allraw=sampled_text_views(caption,'summary_detail',0,0,i+1000)
                visible=longclip.tokenize(['. '.join(new['views'][0].split('. ')[1:])],truncate=False)[0]
                values=[new['tokens_f'],new['tokens_o'],allraw['tokens_e'],visible,new['tokens_e'],old['tokens_o'],old['tokens_e']]
                for n,v in zip(names,values):banks[n].append(v)
                ids.append(i+1000)
                if len(ids)==1024:break
    assert len(ids)==1024
    model,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    if a.stage=='step0':
        assert file_sha(SHARED)=='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
        model.load_state_dict(torch.load(SHARED,map_location='cpu',weights_only=False)['model'],strict=True)
        checkpoint=SHARED
    else:
        checkpoint=RUN/'step500/student_step500.pt'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model=model.float().to('cuda:0').eval()
    features={}
    for name,tokens in banks.items():
        tokens=torch.stack(tokens);values=[]
        for start in range(0,len(tokens),64):
            values.append(F.normalize(model.encode_text(tokens[start:start+64].cuda()).float(),dim=-1))
        features[name]=torch.cat(values)
    provenance={'stage':a.stage,'checkpoint':str(checkpoint),'checkpoint_sha256':file_sha(checkpoint),
        'sample_count':1024,'sample_id_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        'subset':'Fixed seed0, eligible1024 drawn from the audited100k; reused identically at step0/500',
        'tokenizer':'Actual SAID LongCLIP tokenizer, context248','precision':'FP32','same_model_for_all_views':True,
        'no_mask_gate_or_parameter_change':True}
    comparisons=[('F_vs_All_Detail_raw','Full','All_Detail_raw'),('F_vs_All_Detail_visible','Full','All_Detail_visible'),
        ('F_vs_Random_Detail','Full','Random_Detail'),('S_vs_Random_Detail','Summary','Random_Detail'),
        ('S_vs_All_Detail_raw','Summary','All_Detail_raw'),('RandomK_F_vs_R','Full','RandomK_R'),
        ('RandomK_P_vs_R','RandomK_P','RandomK_R')]
    redundancy={name:stats((features[x]*features[y]).sum(-1)) for name,x,y in comparisons}
    ambiguity={}
    for name in ['Summary','All_Detail_raw','All_Detail_visible','Random_Detail','RandomK_P','RandomK_R']:
        sim=features[name]@features[name].T
        mean=(sim.sum(1)-sim.diagonal())/(len(sim)-1)
        sim.fill_diagonal_(-torch.inf)
        top=sim.topk(5,dim=1).values
        ambiguity[name]={'nearest_off_diagonal_cosine':stats(top[:,0]),
                         'top5_nearest_negative_cosine':stats(top.mean(1)),
                         'mean_off_diagonal_cosine':stats(mean)}
    for filename,values in [('TEXT_REDUNDANCY_AUDIT.json',redundancy),('T2I_AMBIGUITY_AUDIT.json',ambiguity)]:
        path=EXP/filename
        current=json.loads(path.read_text()) if path.exists() else {'diagnostic_only':True,'stages':{}}
        current['stages'][a.stage]={'provenance':provenance,'metrics':values}
        path.write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps({'stage':a.stage,'F_Dall':redundancy['F_vs_All_Detail_raw']['mean'],
                      'F_Drandom':redundancy['F_vs_Random_Detail']['mean'],
                      'S_nearest_negative':ambiguity['Summary']['nearest_off_diagonal_cosine']['mean']}),flush=True)


if __name__=='__main__':main()
