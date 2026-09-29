"""Audit and summarize existing A3-RandomK outputs; no model training/evaluation."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.data import DistributedSampler

REPO=Path('/root/lk_projects/SAID')
ROOT=Path('/root/lk_projects/SAID-nest-clip-v1')
EXPERIMENT=ROOT/'randomk500'
EVIDENCE=EXPERIMENT/'evidence'
RUN=EXPERIMENT/'A3-RandomK'
sys.path.insert(0,str(REPO))
from train.nested_semantic_data import sample_split_k, file_sha


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    with Path(p).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def histogram_summary(histogram):
    h={int(k):v for k,v in histogram.items()}
    n=sum(h.values())
    def quantile(q):
        target=max(1,math.ceil(q*n)); total=0
        for k,v in sorted(h.items()):
            total+=v
            if total>=target:return k
    return dict(count=n,min=min(h),p25=quantile(.25),median=quantile(.5),p75=quantile(.75),
                p95=quantile(.95),max=max(h),mean=sum(k*v for k,v in h.items())/n,
                histogram={str(k):h[k] for k in sorted(h)})


def audit():
    cfg=read(RUN/'config.json'); acceptance=read(RUN/'acceptance.json')
    execution=read(EVIDENCE/'train500.execution.json')
    assert execution['exit_code']==0
    console=(EVIDENCE/'train500.console.txt').read_text()
    split=console.index('{"step": 500,')
    assert 'Traceback' not in console[:split] and 'RandomK token boundary overflow' not in console
    shutdown_warning=None
    if 'Traceback' in console:
        import re
        assert '_MultiProcessingDataLoaderIter.__del__' in console[split:] and '_shutdown_workers' in console[split:]
        shutdown_warning=dict(status='completed_with_dataloader_shutdown_warning', after_step500=True,
                              ignored_in_destructor=True, process_exit_code=0,
                              worker_pids=sorted(set(re.findall(r'worker \(pid (\d+)\)',console[split:]))),
                              root_cause='Not determined; no training or sampling change/retry performed',
                              evidence='train500.console.txt')
    assert cfg['arm']=='A3' and cfg['experiment_name']=='A3-RandomK' and cfg['sampling_mode']=='random_k'
    assert cfg['sampling_seed']==cfg['seed']==0 and cfg['horizon']==3651 and cfg['max_updates']==500
    assert cfg['world_size']==4 and cfg['batch_size']==256 and cfg['accumulation']==1
    assert cfg['resume'] is None and cfg['training_records']==1245901 and cfg['batches_per_epoch']==1217
    assert cfg['init_sha256']==read(EVIDENCE/'preflight.json')['initialization_sha256']
    assert cfg['git_head']==read(EVIDENCE/'smoke.execution.json')['git_head']
    for p,h in cfg['code_sha256'].items():assert file_sha(REPO/p)==h,p
    assert acceptance['passed'] and len(acceptance['ranks'])==4
    for h in acceptance['ranks']:assert h['completed_updates']==h['updates_this_run']==500 and h['max_parameter_difference_from_rank0']==0
    actual=rows(RUN/'steps.jsonl'); fixed=rows(ROOT/'formal/A3/steps.jsonl'); smoke=rows(EXPERIMENT/'smoke/A3-RandomK/steps.jsonl')
    assert [r['step'] for r in actual]==list(range(1,501))
    fixed_cfg=read(ROOT/'formal/A3/config.json')
    for k in ('arm','world_size','batch_size','accumulation','epochs','seed','workers','checkpoint_encoders','score_chunk','horizon','data','init_sha256'):
        assert cfg[k]==fixed_cfg[k],k
    samplers=[list(DistributedSampler(range(1245901),num_replicas=4,rank=r,shuffle=True,seed=0,drop_last=False)) for r in range(4)]
    group=defaultdict(Counter); distributions={k:Counter() for k in ('prefix_segments','remainder_segments','prefix_token_lengths','remainder_token_lengths')}
    reasons=Counter(); valid=k1=changed=0; duplicate_steps=[]
    for step,row in enumerate(actual):
        assert row['s']==step and row['epoch']==0 and row['inc_weight']==min(1.,step/200)
        assert math.isfinite(row['loss']) and row['nonfinite']==0
        assert row['F_candidates']==1024 and row['O_candidates']==row['E_candidates']==row['valid_global']
        expected_lr=1e-6*(step+1)/200 if step<200 else .5e-6*(1+math.cos(math.pi*(step-200)/(3651-200)))
        assert math.isclose(row['lr_backbone'],expected_lr,rel_tol=1e-12)
        assert math.isclose(row['lr_mask'],.5e-3*(1+math.cos(math.pi*step/3651)),rel_tol=1e-12)
        assert row['valid_global']==fixed[step]['valid_global']
        if row['duplicate_image_ids']:duplicate_steps.append(row['step'])
        for rank,h in enumerate(row['rank_health']):
            assert h['rank']==rank and h['updates']==step+1 and h['batch']==256 and h['gradients_finite']
            assert all(math.isfinite(v) for v in h['gradient_norms'].values())
            s=h['sampling']; ids=[x+1000 for x in samplers[rank][step*256:(step+1)*256]]
            assert s['sample_ids']==ids
            assert s['fixed_first_reference_stream_sha256']==fixed[step]['rank_health'][rank]['stream_sha256']
            assert len(s['n'])==len(s['K'])==256
            observed=defaultdict(Counter)
            for sample_id,n,k in zip(ids,s['n'],s['K']):
                if k:
                    assert n>=2 and 1<=k<n and k==sample_split_k(n,0,row['epoch'],sample_id)
                    observed[str(n)][str(k)]+=1
                else:assert n in (0,1)
            assert dict(observed)==s['K_histogram_by_n']
            assert s['valid_count']==h['valid']==sum(k>0 for k in s['K'])
            assert s['k1_count']==sum(k==1 for k in s['K'])
            valid+=s['valid_count'];k1+=s['k1_count'];reasons.update(h['reasons'])
            for n,count in s['K_histogram_by_n'].items():group[n].update(count)
            for key in distributions:distributions[key].update(s[key])
            changed+=h['stream_sha256']!=s['fixed_first_reference_stream_sha256']
            if step<5:
                for key in ('sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256'):
                    assert s[key]==smoke[step]['rank_health'][rank]['sampling'][key]
    assert sum(reasons.values())==512000 and valid==reasons['valid'] and 0<k1<valid
    checkpoints=[]
    common=torch.load(ROOT/'shared/step000000.pt',map_location='cpu',weights_only=False)
    old_zero=torch.load(ROOT/'formal/A3/step000000.pt',map_location='cpu',weights_only=False)
    assert sorted(p.name for p in RUN.glob('step*.pt'))==[f'step{s:06d}.pt' for s in (0,100,200,300,400,500)]
    for step in (0,100,200,300,400,500):
        p=RUN/f'step{step:06d}.pt'; ckpt=torch.load(p,map_location='cpu',weights_only=False)
        assert ckpt['completed_steps']==step and ckpt['stop_updates']==500 and ckpt['scheduler_horizon']==3651
        assert len(ckpt['rng_per_rank'])==4 and all(int(s['step'])==step for s in ckpt['optimizer']['state'].values())
        if step==0:
            assert ckpt['optimizer']==common['optimizer']
            assert all(torch.equal(v,ckpt['model'][k]) for k,v in common['model'].items())
            for a,b in zip(ckpt['rng_per_rank'],old_zero['rng_per_rank']):
                assert a['python']==b['python'] and torch.equal(a['cpu'],b['cpu']) and torch.equal(a['cuda'],b['cuda'])
                assert a['numpy'][0]==b['numpy'][0] and (a['numpy'][1]==b['numpy'][1]).all() and a['numpy'][2:]==b['numpy'][2:]
        del ckpt
        checkpoints.append(dict(step=step,server_path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    summary=dict(passed=True,shutdown_warning=shutdown_warning,code_commit=cfg['git_head'],execution=execution,config=cfg,acceptance=acceptance,
                 step0_model_optimizer_match_common=True,step0_rng_matches_fixed_A3=True,
                 compared_reference_rank_streams=2000,all_fixed_reference_streams_match=True,
                 explicit_sampler_sample_ids_match=True,all_logged_K_replay=True,
                 smoke_formal_first5_streams_match=True,changed_local_rank_streams=changed,
                 samples=512000,valid_samples=valid,valid_ratio=valid/512000,k1_samples=k1,k1_ratio=k1/valid,
                 K_histogram_by_n={n:dict(c) for n,c in sorted(group.items(),key=lambda x:int(x[0]))},
                 length_distributions={k:histogram_summary(v) for k,v in distributions.items()},
                 fallback_reasons=dict(reasons),global_valid_min=min(r['valid_global'] for r in actual),
                 global_valid_max=max(r['valid_global'] for r in actual),duplicate_steps=duplicate_steps,
                 checkpoints=checkpoints)
    write(EVIDENCE/'formal-audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','execution','acceptance','K_histogram_by_n','length_distributions','checkpoints')},indent=2))


def mechanism(records):
    fields=['loss','inc','inc_weight','hard_inclusion_violation','oe_iou','valid_global']
    fields += [f'{v}_{k}' for v in ('F','O','E') for k in ('i2t','t2i','keep_ratio','all_open','all_closed')]
    def values(r):
        d={k:r.get(k,0.) for k in fields};d['common_loss']=r['loss']-r['inc_weight']*r['inc']
        for v in ('F','O','E'):d[f'{v}_align']=d[f'{v}_i2t']+d[f'{v}_t2i']
        return d
    return dict(last50={k:statistics.fmean(values(r)[k] for r in records[-50:]) for k in values(records[-1])},
                milestones={str(s):values(records[s-1]) for s in (1,100,200,201,300,400,500)})


def aggregate():
    audit=read(EVIDENCE/'formal-audit.json');assert audit['passed']
    export=read(RUN/'export-check.json');assert export['passed'] and export['optimizer_steps']==[500]
    assert export['image_max_abs']==export['text_max_abs']==0
    raw={'RandomK':{d:read(RUN/f'{d}_native.json') for d in ('coco','urban')},
         'FixedA3':{d:read(ROOT/f'formal/A3/{d}_native.json') for d in ('coco','urban')}}
    for d in ('coco','urban'):assert raw['RandomK'][d]['checkpoint_sha256']==export['bare_sha256']
    baseline={a:{d:read(EVIDENCE/f'baselines/{a}/{d}.json') for d in ('coco','urban')} for a in ('clean','full')}
    def recall(source,dataset,direction,k):return source[f'{direction}_R{k}'] if dataset=='coco' else source[direction][f'R{k}']
    comparison=[]
    for dataset in ('coco','urban'):
        for direction,key in (('I2T','image2text'),('T2I','text2image')):
            for k in (1,5,10):
                values={arm:recall(raw[arm][dataset],dataset,key,k) for arm in raw}
                for arm in baseline:
                    source=baseline[arm][dataset]['metrics' if dataset=='coco' else 'urban1k']
                    values[arm]=recall(source,dataset,key,k)
                assert all(0<=x<=1 for x in values.values())
                comparison.append(dict(dataset=dataset,direction=direction,metric=f'R@{k}',raw_recall=values,
                                       percent={k:100*v for k,v in values.items()},
                                       delta_pp={arm:100*(values['RandomK']-values[arm]) for arm in ('FixedA3','clean','full')}))
    executions={name:read(EVIDENCE/f'{name}.execution.json') for name in ('smoke','train500','export500','verify-export','coco','urban')}
    assert all(x['exit_code']==0 and x['git_head']==audit['code_commit'] for x in executions.values())
    result=dict(code_commit=audit['code_commit'],reference_commit='190b4779bea73ffda891eac0cc0e6e0ca1796637',
                change='Only local text split: fixed_first -> random_k; model arm remains A3',
                formal_audit=audit,smoke_audit=read(EVIDENCE/'smoke-audit.json'),
                export_verification=export,executions=executions,raw_evaluations=raw,
                baselines=baseline,baseline_verification=read(EVIDENCE/'baselines/verification.json'),
                comparison=comparison,
                mechanism={arm:mechanism(rows(p)) for arm,p in [('RandomK',RUN/'steps.jsonl'),('FixedA3',ROOT/'formal/A3/steps.jsonl')]},
                limitations=['One seed, 500 updates, no statistical significance claim',
                             'Compares random-K vs fixed first segment within unchanged A3; does not isolate inclusion loss',
                             'Clean/Full are previously completed step500 methods with different objectives/local-view constructions',
                             'Training alignment tasks differ with text split; common loss alone is not retrieval performance'])
    write(EXPERIMENT/'results.json',result)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    torch.set_num_threads(4)
    if sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='aggregate':aggregate()
    else:raise ValueError(sys.argv[1])
