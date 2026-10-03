"""CPU-only real-image augmentation replay and four-arm view statistics."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import numpy as np
import torch
from train.nested_semantic_data import NestedDataset
from tools.summary_detail_audit import distribution
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP

MODES=['random_k','interior_random_k','summary_random_detail','summary_contiguous_detail']

def chunk(indices):
    torch.set_num_threads(1)
    datasets={mode:NestedDataset('/root/lk_projects/SAID-nest-clip-v1/data_index','/root/lk_projects/SAID-assets/training/ShareGPT4V',mode,0) for mode in MODES}
    h={mode:{field:Counter() for field in ['n','K','F_tokens','local1_tokens','local2_tokens']} for mode in MODES}
    digest=hashlib.sha256()
    for i in indices:
        samples={}
        for mode,ds in datasets.items():
            random.seed(i);np.random.seed(i);torch.manual_seed(i)
            samples[mode]=ds[i]
            sample=samples[mode]
            for field in ['n','K']:h[mode][field][sample[field]]+=1
            for name,key in [('F_tokens','tokens_f'),('local1_tokens','tokens_o'),('local2_tokens','tokens_e')]:
                if name=='F_tokens' or sample['valid']:h[mode][name][int(sample[key].argmax())+1]+=1
        baseline=samples['random_k']
        for mode,s in samples.items():
            assert torch.equal(s['image'],baseline['image']), (i,mode)
            assert torch.equal(s['tokens_f'],baseline['tokens_f']) and s['views'][0]==baseline['views'][0]
        digest.update(str(i+1000).encode());digest.update(baseline['image'].numpy().tobytes())
    return {'histograms':h,'samples':len(indices),'digest':digest.hexdigest()}

def main():
    torch.set_num_threads(1)
    indices=random.Random(0).sample(range(1245901),1000)
    hist={mode:{field:Counter() for field in ['n','K','F_tokens','local1_tokens','local2_tokens']} for mode in MODES};results=[]
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(chunk,indices[i:i+25]) for i in range(0,1000,25)]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            for mode,fields in r['histograms'].items():
                for field,h in fields.items():hist[mode][field].update(h)
    result={'passed':True,'samples':1000,'sample_indices_sha256':hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),
      'all_four_modes_real_image_tensors_bitwise_equal':True,'all_four_modes_F_equal':True,
      'chunk_image_digests':sorted(r['digest'] for r in results),'view_statistics':{mode:{field:distribution(h,[10,25,50,75,90,95]) for field,h in fields.items()} for mode,fields in hist.items()},
      'scope':'Replay actual real-image augmentation with identical per-sample Python/NumPy/Torch states. Original augmentation source, worker generator and sampler are unchanged; no historical image tensor digests exist. Sampling uses private RNG, so equal pre-image RNG states persist by induction throughout the matched stream.'}
    (EXP/'evidence/image-and-sampling-1000.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'real_image_replay_samples':1000}),flush=True)

if __name__=='__main__':main()
