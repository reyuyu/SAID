"""Read-only Summary ambiguity on exactly the previous fixed1024 sample IDs."""
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
from train.nested_semantic_data import sampled_text_views,file_sha
from tools.random_detail_text_audit import stats

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B','C','D'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    pool=np.load('/root/lk_projects/SAID-nest-clip-v1/balanced_summary_random_detail_500_v1/audit_subset_indices.npy').tolist()
    candidates=random.Random(0).sample(pool,2048)
    index=Path('/root/lk_projects/SAID-nest-clip-v1/data_index');offset=np.load(index/'offsets.npy',mmap_mode='r')
    banks={'Summary':[],'Detail':[]};ids=[]
    mode='summary_contiguous_detail' if a.arm=='D' else 'summary_random_detail'
    with (index/'records.jsonl').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for i in candidates:
            v=sampled_text_views(json.loads(data[offset[i]:offset[i+1]])['caption'],mode,0,0,i+1000)
            if not v['valid']:continue
            banks['Summary'].append(v['tokens_o']);banks['Detail'].append(v['tokens_e']);ids.append(i+1000)
            if len(ids)==1024:break
    assert len(ids)==1024
    digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    old=Path(__file__).resolve().parents[0]/'../balanced_summary_random_detail_500_v1/T2I_AMBIGUITY_AUDIT.json'
    expected=json.loads(old.read_text())['stages']['step500']['provenance']['sample_id_sha256'];assert digest==expected
    model,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True);model=model.float().cuda().eval()
    result={'arm':a.arm,'diagnostic_only':True,'checkpoint_sha256':file_sha(a.checkpoint),'sample_count':1024,'sample_id_sha256':digest,'sampling_mode':mode,'precision':'FP32','metrics':{}}
    for name,bank in banks.items():
        tokens=torch.stack(bank);features=torch.cat([F.normalize(model.encode_text(tokens[i:i+64].cuda()).float(),dim=-1) for i in range(0,1024,64)])
        sim=features@features.T;mean=(sim.sum(1)-sim.diagonal())/1023;sim.fill_diagonal_(-torch.inf)
        result['metrics'][name]={'nearest_off_diagonal_cosine':stats(sim.max(1).values),'mean_off_diagonal_cosine':stats(mean)}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
