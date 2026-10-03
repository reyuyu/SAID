"""Verify final raw metrics, strict exports, resource gates and selection consistency."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from experiments.nest_clip_v1.four_arm_text_search_500_v1.run import EXP,ARMS
from experiments.nest_clip_v1.four_arm_text_search_500_v1.report import selection,GUARD
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores


def main():
    result=load(EXP/'SEARCH_RESULTS.json');baseline=result['baseline'];arms=result['arms']
    assert list(arms)==['A','B','C','D'] and all(r['status']=='COMPLETE' for r in arms.values())
    verified={}
    for arm,r in arms.items():
        directory=EXP/ARMS[arm][0];root=Path(r['checkpoint']).parent
        m,raw,_=native_metrics(root);assert m==r['metrics']
        expected=scores(m);expected['Short4_R1']=statistics.fmean(m[ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I'])
        assert expected==r['scores']
        assert load(directory/'RESULTS.json')==r
        strict=load(directory/'evidence/strict-export.json');assert strict['passed'] and strict['strict_load']
        assert strict['bare_sha256']==r['bare_sha256']
        assert r['config']['horizon']==4868 and r['config']['max_updates']==500 and r['config']['resume'] is None and r['config']['start_updates']==0
        assert r['resources']['passed'] and r['resources']['normal_cycles']==495 and r['resources']['normal_full_cycle_mean_seconds']<=3 and r['resources']['peak_allocated_gib']<=65
        assert load(directory/'evidence/final-matching-and-objective.json')['passed']
        with gzip.open(directory/'evidence/steps.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        assert len(rows)==500 and rows[-1]['step']==500
        diag=load(directory/'TRAINING_DIAGNOSTICS.json');assert all(s in diag['steps'] for s in ['1','100','200','500'])
        assert diag['last50']['metrics']['F_i2t']['observations']==50
        assert diag['last50']['token_lengths']['Full']['count']==51200
        for ds in m:
            for dr in ['I2T','T2I']:
                v=m[ds][dr];assert 0<=v['R@1']<=v['R@5']<=v['R@10']<=1
        for name,value in raw.items():assert load(directory/'raw'/(name+'.json'))==value
        commands={p.stem:load(p) for p in (directory/'commands').glob('*.json')}
        assert all(c['exit_code']==0 for c in commands.values())
        assert len(commands)==9 if arm=='A' else len(commands)==10
        if arm in 'BCD':assert load(directory/'SUMMARY_AMBIGUITY.json')['checkpoint_sha256']==r['bare_sha256']
        verified[arm]={'passed':True,'native30_recalls_verified':True,'strict_export':True,'fresh500_H4868':True,'resources_pass':True,'commands_pass':True,'complete_lossless_log_rows':500}
    d=selection(arms,baseline['scores']['Score5_R1']);stored=load(EXP/'SELECTION.json')
    for key,value in d.items():assert stored[key]==value,key
    assert result['status']==stored['status']=='TRADEOFF'
    assert stored['champion']=='A' and stored['best_observed_Score5_arm']=='B' and not stored['overall_improvement']
    assert arms['A']['scores']['J_long3']>=GUARD and all(arms[a]['scores']['J_long3']<GUARD for a in 'BCD')
    for path in EXP.rglob('*'):
        if path.is_file():
            assert path.stat().st_size<10*1024*1024,(path,path.stat().st_size)
            assert path.suffix not in ['.pt','.pth','.npy','.npz','.safetensors'],path
    dump(EXP/'evidence/final-artifact-verification.json',{'passed':True,'all_four_arms':verified,'selection_consistent':True,'no_large_weights_datasets_embeddings_in_output':True,'status':'TRADEOFF'})
    print(json.dumps({'passed':True,'arms':list(verified),'native_directional_recalls_checked':120,'status':'TRADEOFF'}))

if __name__=='__main__':main()
