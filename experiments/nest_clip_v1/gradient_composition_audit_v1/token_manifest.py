"""Replay exact32 official global batches; save token and sentence-choice hashes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import mmap
import numpy as np
import torch
from train.nested_semantic_data import sampled_text_views,sampling_diagnostics,collate
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,INDEX,SEARCH,load,dump,stages


def digest_bytes(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def replay(item):
    step,rank,ids=item;torch.set_num_threads(1)
    offsets=np.load(INDEX/'offsets.npy',mmap_mode='r');samples={m:[] for m in ['random_k','summary_random_detail']}
    with (INDEX/'records.jsonl').open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as records:
        for sid in ids:
            i=sid-1000;record=json.loads(records[offsets[i]:offsets[i+1]])
            for mode in samples:
                views=sampled_text_views(record['caption'],mode,0,0,sid)
                samples[mode].append(dict(image=torch.zeros(1),image_id=0,sample_id=sid,**views))
    result={'batch':step,'rank':rank,'sample_ids':ids,'views':{}}
    full_bytes=None
    for mode,values in samples.items():
        batch=collate(values);diagnostic=sampling_diagnostics(batch)
        result['views'][mode]={'n':batch['n'].tolist(),'K':batch['K'].tolist(),'valid':batch['valid'].tolist(),
          'Full_tokens_sha256':digest_bytes(batch['tokens_f']),'local1_tokens_sha256':digest_bytes(batch['tokens_o']),
          'local2_tokens_sha256':digest_bytes(batch['tokens_e']),'production_digests':{k:diagnostic[k] for k in ['full_view_sha256','local_views_sha256','split_sha256']}}
        if mode=='summary_random_detail':result['views'][mode]['detail_sentence_indices']=batch['detail_indices']
        if full_bytes is None:full_bytes=batch['tokens_f'].contiguous().numpy().tobytes()
        else:assert full_bytes==batch['tokens_f'].contiguous().numpy().tobytes()
    return result,full_bytes


def main():
    torch.set_num_threads(1)
    baseline=load(SEARCH/'BASELINE.json');rows=[json.loads(l) for l in (Path(baseline['root'])/'steps.jsonl').read_text().splitlines()[:32]]
    brows=[json.loads(l) for l in (Path(stages()['B']['checkpoint']).parent/'steps.jsonl').read_text().splitlines()[:32]]
    tasks=[(s+1,rank,rows[s]['rank_health'][rank]['sampling']['sample_ids']) for s in range(32) for rank in range(4)]
    output={};bytes_by_key={}
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(replay,item) for item in tasks]
        for future in as_completed(futures):
            item,data=future.result();key=(item['batch'],item['rank']);output[key]=item;bytes_by_key[key]=data
    batches=[]
    for step in range(1,33):
        ranks=[output[(step,r)] for r in range(4)]
        for rank,item in enumerate(ranks):
            for mode,reference in [('random_k',rows),('summary_random_detail',brows)]:
                for key,value in item['views'][mode]['production_digests'].items():assert value==reference[step-1]['rank_health'][rank]['sampling'][key]
        global_ids=[sid for item in ranks for sid in item['sample_ids']]
        full_sha=hashlib.sha256(b''.join(bytes_by_key[(step,r)] for r in range(4))).hexdigest()
        batches.append(dict(batch=step,ranks=ranks,global_sample_ID_sha256=hashlib.sha256(json.dumps(global_ids,separators=(',',':')).encode()).hexdigest(),global_Full_tokens_sha256=full_sha))
    dump(EXP/'evidence/FIXED_BATCH_TOKEN_MANIFEST.json',dict(passed=True,batches=32,world_size=4,batch_per_rank=256,epoch=0,sampling_seed=0,sampler_seed=0,
      dtype='Actual LongCLIP token tensor bytes, rank-major concatenation for globalFull digest',selection='Exact first32 official epoch0 DistributedSampler batches from matched baseline logs; verified against runtime stage evidence',
      exact_RandomK_and_ArmB_sampling_match_existing500_logs=True,manifest=batches))
    print(json.dumps({'passed':True,'global_batches':32,'rank_batches':128,'samples':32768,'exact_historical_token_and_sampling_digests':True}))

if __name__=='__main__':main()
