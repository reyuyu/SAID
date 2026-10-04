"""One authorized S0.4 experiment: fresh smoke5/fresh500/native5/spot8 only."""
import ast
import gzip
import hashlib
import itertools
import json
import mmap
from pathlib import Path
import random
import shutil
import statistics
import subprocess

import numpy as np
import torch
from torch.utils.data import DistributedSampler
from model import longclip
from model.balanced_hparam_search import trial_id
from train.nested_semantic_data import sampled_text_views,file_sha
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1 import run as base

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/armb_summary04_500_v1')
BRANCH='codex/nest-balanced-armb-summary04-500-v1'
SEARCH=ROOT/'experiments/nest_clip_v1/four_arm_text_search_500_v1'
SOURCE=SEARCH/'arm_B_summary_weight'
WEIGHTS=[1.3,.4,1.3]


def audit_sampling():
    original=subprocess.check_output(['git','show','b85f73a:train/nested_semantic_data.py'],cwd=ROOT,text=True)
    nodes=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name in ['text_views','sample_split_k','sample_detail_indices','sampled_text_views']]
    scope=dict(longclip=longclip,torch=torch,hashlib=hashlib,random=random)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'ArmB_source','exec'),scope)
    offsets=np.load(base.INDEX/'offsets.npy',mmap_mode='r');digest=hashlib.sha256()
    with (base.INDEX/'records.jsonl').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for i in random.Random(0).sample(range(1245901),1000):
            caption=json.loads(data[offsets[i]:offsets[i+1]])['caption']
            a=scope['sampled_text_views'](caption,'summary_random_detail',0,0,i+1000)
            b=sampled_text_views(caption,'summary_random_detail',0,0,i+1000)
            assert a.keys()==b.keys()
            for k in a:assert torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k]
            digest.update(str(i+1000).encode())
            digest.update(json.dumps(b['views'],ensure_ascii=False).encode())
            for k in ['tokens_f','tokens_o','tokens_e']:digest.update(b[k].numpy().tobytes())
            digest.update(json.dumps(b['detail_indices']).encode())
    result=dict(passed=True,samples=1000,seed=0,epoch=0,source_commit='b85f73a4c49df493af8796aa85fdb8790fe92805',
      F_S_D_raw_tokens_K_detail_indices_all_fields_exactly_equal=True,sha256=digest.hexdigest())
    base.dump(EXP/'evidence/SAMPLING_1000.json',result)
    return result


def prepare():
    baseline=base.load(SEARCH/'BASELINE.json');old=base.load(SOURCE/'RESULTS.json');source_config=base.load(SOURCE/'config.json')
    assert old['checkpoint_sha256']=='fbdd8c3785777fa612db6e4d5446d3fb19367d94eb8ddacd3d581dcb64460bce'
    assert file_sha(old['checkpoint'])==old['checkpoint_sha256'] and file_sha(old['bare_student'])==old['bare_sha256']
    assert file_sha(baseline['checkpoint'])==baseline['checkpoint_sha256'] and file_sha(baseline['student'])==baseline['bare_sha256']
    assert file_sha(base.SHARED)==base.INIT_SHA
    config=dict(source_config,view_weights=WEIGHTS,experiment_name='ArmB-Summary0.4-500')
    config['trial_id']=trial_id(config)
    differences={k:{'old':source_config.get(k),'new':v} for k,v in config.items() if source_config.get(k)!=v}
    assert set(differences)=={'view_weights','experiment_name','trial_id'}
    assert sum(WEIGHTS)==3. and 'summary_t2i_weight' not in config
    base.dump(EXP/'config.json',config);base.dump(EXP/'BASELINE.json',baseline);base.dump(EXP/'ARM_B_S06.json',old)
    paths=['model/balanced_hparam_search.py','model/nested_semantic_mask.py','model/nested_fusion_mask.py','model/nested_vcp_mask.py',
      'model/model_longclip.py','model/longclip.py','model/said_cls_cvssl.py','train/train_nested_semantic_mask.py',
      'train/nested_semantic_data.py','train/random_detail_observer.py','train/said_cvssl_data.py',
      'tools/nest_clip.py','tools/eval_nest_native.py','eval/retrieval/coco_retrieval.py','tools/urban1k_retrieval.py',
      'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py']
    hashes={}
    for path in paths:
        original=subprocess.check_output(['git','show','b85f73a:'+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==original,path;hashes[path]=hashlib.sha256(original).hexdigest()
    assert base.load(base.INDEX/'metadata.json')==old['config']['data']
    assert file_sha(base.INDEX/'records.jsonl')==old['config']['data']['records_sha256']
    old_rows=base.records(Path(old['checkpoint']).parent/'steps.jsonl');assert len(old_rows)==500
    for rank in range(4):
        sampler=DistributedSampler(range(1245901),num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
        ids=[i+1000 for i in itertools.islice(iter(sampler),500*256)]
        for step,row in enumerate(old_rows):
            h=hashlib.sha256(json.dumps(ids[step*256:(step+1)*256],separators=(',',':')).encode()).hexdigest()
            assert h==row['rank_health'][rank]['sampling']['sample_id_sha256']
    sampling=audit_sampling()
    proof=dict(passed=True,source_commit='b85f73a4c49df493af8796aa85fdb8790fe92805',
      only_behavior_change='alignment view weights [1.2,.6,1.2] -> [1.3,.4,1.3]',config_differences=differences,
      unchanged_production_sources=hashes,common_step0_sha256=base.INIT_SHA,
      no_resume=True,scheduler_horizon=4868,stop_updates=500,full_training_authorized=False,
      fixed_1000_sampling=sampling,all500x4_sampler_ID_hashes_match_ArmB=True)
    base.dump(EXP/'evidence/preflight.json',proof)
    base.dump(RUN/'state.json',dict(status='READY',stage='preflight passed',branch=BRANCH,started_at=base.now(),stop_updates=500,horizon=4868))
    print(json.dumps({'preflight':'PASS','sampling_1000':'exact','only_weights_changed':WEIGHTS}),flush=True)


def bind():
    base.EXP=EXP;base.RUN=RUN;base.ROOT=ROOT;base.BRANCH=BRANCH
    original=base.stream_check
    arm_rows=base.records(Path(base.load(SOURCE/'RESULTS.json')['checkpoint']).parent/'steps.jsonl')
    def check(row,reference):
        original(row,reference)
        old=arm_rows[row['step']-1]
        for a,b in zip(row['rank_health'],old['rank_health']):
            for key in ['sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256','fixed_first_reference_stream_sha256']:
                assert a['sampling'][key]==b['sampling'][key],(row['step'],a['rank'],key)
            assert a['sampling']['random_detail_sampling']['selected_indices_sha256']==b['sampling']['random_detail_sampling']['selected_indices_sha256']
    base.stream_check=check


def training_diagnostics():
    root=RUN/'step500';rows=base.records(root/'steps.jsonl');cycles=base.records(root/'cycle_timing.jsonl')
    assert len(rows)==500
    def summarize(group):
        keys=sorted(set().union(*(r.keys() for r in group)))
        metrics={k:dict(mean=statistics.fmean(r[k] for r in group if isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool)),observations=sum(isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool) for r in group)) for k in keys if any(isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool) for r in group)}
        raw={label:metrics[p+'_i2t']['mean']+metrics[p+'_t2i']['mean'] for label,p in [('F','F'),('S','O'),('D','E')]}
        weighted={label:w*raw[label] for label,w in zip(['F','S','D'],WEIGHTS)}
        denom=sum(weighted.values())
        token={}
        for row in group:
            for rank in row['rank_health']:
                for label,stat in rank['sampling']['Full_Summary_Detail_token_statistics'].items():
                    v=token.setdefault(label,dict(count=0,token_sum=0));v['count']+=stat['samples'];v['token_sum']+=stat['effective_token_sum']
        for v in token.values():v['mean']=v['token_sum']/v['count']
        return dict(metrics=metrics,raw_combined_CE=raw,weighted_CE_contribution=weighted,
          weighted_alignment_contribution_including_outer10_over3={k:10/3*v for k,v in weighted.items()},
          weighted_CE_share={k:v/denom for k,v in weighted.items()},token_lengths=token)
    errors=[]
    for row in rows:
        align=10/3*sum(w*(row[p+'_i2t']+row[p+'_t2i']) for w,p in zip(WEIGHTS,['F','O','E']))
        total=align+(row['F_sparse']+2*row['O_sparse']+2*row['E_sparse'])/3+row['inc_weight']*row['inc']
        assert __import__('math').isclose(total,row['loss'],abs_tol=3e-4,rel_tol=1e-5)
        errors.append(abs(total-row['loss']))
    diag=dict(steps={str(i):summarize([rows[i-1]]) for i in [1,100,200,500]},last50=summarize(rows[-50:]),
      notes='O=Summary,E=Detail. Gate/F-D IoU are recorded only at scheduled diagnostics; last50 gate has one observation. CE/keep/inclusion/S-D IoU have50 observations. Loss contribution is not gradient contribution.')
    base.dump(EXP/'TRAINING_DIAGNOSTICS.json',diag)
    actual=base.load(root/'config.json');old=base.load(SOURCE/'RESULTS.json')['config']
    assert actual['component_initialization']==old['component_initialization'] and actual['parameter_counts']==old['parameter_counts']
    assert actual['code_sha256']==old['code_sha256']
    normal=[r['four_rank_max_seconds'] for r in cycles if not r['warmup']];accept=base.load(root/'acceptance.json')
    resources=dict(normal_full_cycle_mean_seconds=statistics.fmean(normal),normal_cycles=len(normal),peak_allocated_gib=max(r['peak_allocated_gib'] for r in accept['ranks']),passed=True)
    assert len(normal)==495 and resources['normal_full_cycle_mean_seconds']<=3 and resources['peak_allocated_gib']<=65
    base.dump(EXP/'RESOURCE_SUMMARY.json',resources)
    base.dump(EXP/'evidence/final-training-match.json',dict(passed=True,formal_steps=500,ranks=4,
      F_S_D_raw_tokens_K_sentence_indices_streams_match_ArmB=True,initial_components_match=True,
      parameter_counts_match=True,production_source_hashes_match=True,all500_weighted_losses_reconstructed=True,max_loss_rounding_error=max(errors)))
    for name in ['steps.jsonl','cycle_timing.jsonl']:
        with (root/name).open('rb') as src,(EXP/'evidence'/(name+'.gz')).open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as archive:shutil.copyfileobj(src,archive)
    return resources


def main():
    bind();prepare();base.train('smoke');base.train('formal');resources=training_diagnostics();base.evaluate()
    result=base.load(RUN/'result.json');result['scores']['Short4_R1']=statistics.fmean(result['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I']);result['resources']=resources
    base.dump(EXP/'RESULTS.json',result)
    base.execute('gradient-spot8',[base.TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0','-m','experiments.nest_clip_v1.armb_summary04_500_v1.spot_check'],gpu=True)
    from experiments.nest_clip_v1.armb_summary04_500_v1.report import generate
    generate()

if __name__=='__main__':main()
