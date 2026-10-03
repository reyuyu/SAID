"""Fixed100k sampling simulation for the two baselines and visible random detail."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import mmap
import multiprocessing
from pathlib import Path
import random

import numpy as np
import torch

from train.nested_semantic_data import sampled_text_views
from tools.summary_detail_audit import distribution

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/nest_clip_v1/balanced_summary_random_detail_500_v1'
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_summary_random_detail_500_v1')
INDEX = Path('/root/lk_projects/SAID-nest-clip-v1/data_index')


def chunk(indices):
    names=['RandomK_P','RandomK_R','Summary','All_Detail_raw_previous','Random_Detail','All_Detail_visible_control']
    h={n:{'tokens':Counter(),'sentences':Counter()} for n in names}
    kd=Counter();km={};ratios=[];coverage=[];raw_coverage=[]
    count=eligible=k1=all_ge3=old_full_mismatch=tail=0
    offsets=np.load(INDEX/'offsets.npy',mmap_mode='r')
    with (INDEX/'records.jsonl').open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
            for i in indices:
                row=json.loads(data[offsets[i]:offsets[i+1]])
                old=sampled_text_views(row['caption'],'random_k',0,0,i+1000)
                allraw=sampled_text_views(row['caption'],'summary_detail',0,0,i+1000)
                new=sampled_text_views(row['caption'],'summary_random_detail',0,0,i+1000)
                assert old['views'][0]==new['views'][0] and torch.equal(old['tokens_f'],new['tokens_f'])
                count+=1
                if not new['valid']:continue
                eligible+=1
                parts=new['views'][0].split('. ');m=new['detail_pool_size'];k=new['K'];indexes=new['detail_indices']
                assert 0 not in indexes and indexes==sorted(set(indexes)) and all(1<=j<len(parts) for j in indexes)
                assert new['views'][2]=='. '.join(parts[j] for j in indexes)
                pool='. '.join(parts[1:])
                from model import longclip
                pool_length=len(longclip._tokenizer.encode(pool))
                raw_length=allraw['untruncated_lengths'][2]-2
                ratios.append(k/m);coverage.append((new['untruncated_lengths'][2]-2)/pool_length)
                raw_coverage.append((new['untruncated_lengths'][2]-2)/raw_length)
                kd[k]+=1;km.setdefault(m,Counter())[k]+=1;k1+=k==1;all_ge3+=k==m and m>=3
                tail+=any(parts[j] not in parts for j in indexes)
                for name,view,key,sentences in [
                    ('RandomK_P',old,'tokens_o',old['K']),('RandomK_R',old,'tokens_e',old['n']-old['K']),
                    ('Summary',new,'tokens_o',1),('All_Detail_raw_previous',allraw,'tokens_e',allraw['n']-1),
                    ('Random_Detail',new,'tokens_e',k)]:
                    h[name]['tokens'][int(view[key].argmax())+1]+=1;h[name]['sentences'][sentences]+=1
                h['All_Detail_visible_control']['tokens'][pool_length+2]+=1
                h['All_Detail_visible_control']['sentences'][m]+=1
    return {'histograms':h,'KD':kd,'KD_by_m':km,'ratios':ratios,'coverage':coverage,'raw_coverage':raw_coverage,
            'count':count,'eligible':eligible,'k1':k1,'all_ge3':all_ge3,'tail_outside_F':tail}


def float_stats(values):
    return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            **{f'p{q}':float(np.percentile(values,q)) for q in [10,25,50,75,90,95]}}


def main():
    torch.set_num_threads(1)
    subset=sorted(random.Random(0).sample(range(1245901),100000))
    h={};kd=Counter();km={};vectors={k:[] for k in ['ratios','coverage','raw_coverage']}
    totals={k:0 for k in ['count','eligible','k1','all_ge3','tail_outside_F']}
    with ProcessPoolExecutor(max_workers=32,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(chunk,subset[s:s+1000]) for s in range(0,100000,1000)]
        for n,f in enumerate(as_completed(futures),1):
            r=f.result()
            for name,data in r['histograms'].items():
                for field,c in data.items():h.setdefault(name,{}).setdefault(field,Counter()).update(c)
            kd.update(r['KD'])
            for m,c in r['KD_by_m'].items():km.setdefault(m,Counter()).update(c)
            for k in vectors:vectors[k].extend(r[k])
            for k in totals:totals[k]+=r[k]
            if n%10==0:print(json.dumps({'chunks_done':n,'chunks_total':100}),flush=True)
    assert totals['count']==100000 and totals['all_ge3']==totals['tail_outside_F']==0
    # All100k match F; fingerprint a random1000's actual IDs/raw F/token bytes.
    checksum=hashlib.sha256();offset=np.load(INDEX/'offsets.npy',mmap_mode='r')
    with (INDEX/'records.jsonl').open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
            for i in random.Random(0).sample(subset,1000):
                caption=json.loads(data[offset[i]:offset[i+1]])['caption']
                a=sampled_text_views(caption,'random_k',0,0,i+1000)
                b=sampled_text_views(caption,'summary_random_detail',0,0,i+1000)
                assert torch.equal(a['tokens_f'],b['tokens_f']) and a['views'][0]==b['views'][0]
                checksum.update(str(i+1000).encode());checksum.update(b['views'][0].encode());checksum.update(b['tokens_f'].numpy().tobytes())
    gate={'passed':True,'samples':1000,'F_raw_equal_count':1000,'F_tokens_bitwise_equal_count':1000,
          'additional_F_equivalence_samples':100000,'seed':0,'digest':checksum.hexdigest()}
    (EXP/'evidence/F_1000_EQUIVALENCE.json').write_text(json.dumps(gate,indent=2)+'\n')
    result={'status':'COMPLETE','subset_count':100000,'subset_seed':0,'sampling_seed':0,'epoch':0,
        'subset_sha256':hashlib.sha256(json.dumps(subset,separators=(',',':')).encode()).hexdigest(),
        'F_equivalence':gate,'visible_valid_fraction':totals['eligible']/totals['count'],
        'views':{name:{f:distribution(c,[10,25,50,75,90,95]) for f,c in data.items()} for name,data in h.items()},
        'KD':distribution(kd,[10,25,50,75,90,95]),'KD_histogram':dict(kd),
        'KD_by_m':{str(m):dict(c) for m,c in sorted(km.items())},'KD_over_m':float_stats(vectors['ratios']),
        'visible_detail_token_coverage':float_stats(vectors['coverage']),
        'raw_detail_token_coverage':float_stats(vectors['raw_coverage']),
        'K1_fallback_fraction':totals['k1']/totals['eligible'],
        'K_equals_m_m_ge3_count':totals['all_ge3'],'raw_tail_outside_visible_F_count':totals['tail_outside_F'],
        'definitions':'Tokens effective incl SOT/EOT; content coverage excludes SOT/EOT; new S/D only from packed visible F. Previous All-Detail raw construction is kept as its own baseline; visible-all control isolates the budget correction.'}
    (EXP/'SAMPLING_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (RUN/'sampling-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    np.save(RUN/'audit_subset_indices.npy',np.array(subset,dtype=np.int64))
    print(json.dumps({'status':'COMPLETE','Random_Detail_mean_tokens':result['views']['Random_Detail']['tokens']['mean'],
                      'All_Detail_mean_tokens':result['views']['All_Detail_raw_previous']['tokens']['mean'],
                      'KD_mean':result['KD']['mean'],'coverage_mean':result['visible_detail_token_coverage']['mean']}),flush=True)


if __name__=='__main__':main()
