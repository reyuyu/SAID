"""Stream audit for the complete trajectory: immutable first500 + continuation."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DistributedSampler

REPO=Path('/root/lk_projects/SAID')
ROOT=Path('/root/lk_projects/SAID-nest-clip-v1')
DEST=ROOT/'randomk3epoch'
RUN=DEST/'A3-RandomK'
EVIDENCE=DEST/'evidence'
sys.path.insert(0,str(REPO))
from train.nested_semantic_data import file_sha,sample_split_k
from train.train_nested_semantic_mask import learning_rates


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def log_rows():
    for origin,path in [('parent',ROOT/'randomk500/A3-RandomK/steps.jsonl'),('continuation',RUN/'steps.jsonl')]:
        with path.open() as f:
            for line in f:
                if line.strip():yield origin,json.loads(line)


def main():
    torch.set_num_threads(4)
    config=read(RUN/'config.json');execution=read(EVIDENCE/'train3651.execution.json');acceptance=read(RUN/'acceptance.json')
    assert execution['exit_code']==0 and execution['git_head']==config['git_head']
    assert config['horizon']==config['max_updates']==3651 and config['start_updates']==500
    assert config['updates_planned_this_run']==3151
    assert config['parent_checkpoint_sha256']=='1fb8f630b181a4b93c4b303f5844e43bcf8fd6aff3b0a73d872bbc0c42ed4d40'
    assert read(EVIDENCE/'restore-check.json')['passed']
    for p,h in config['code_sha256'].items():assert file_sha(REPO/p)==h,p
    assert len(acceptance['ranks'])==4 and acceptance['passed']
    for rank in acceptance['ranks']:
        assert rank['completed_updates']==3651 and rank['updates_this_run']==3151 and rank['max_parameter_difference_from_rank0']==0
    groups={e:defaultdict(Counter) for e in range(3)}
    epoch_stats={e:dict(updates=0,samples=0,valid=0,k1=0,duplicate_steps=[],reasons=Counter(),
                       prefix_lengths=Counter(),remainder_lengths=Counter()) for e in range(3)}
    histories={};last50=deque(maxlen=50);active_epoch=None;expected_ids=None;expected_step=1
    compact_path=DEST/'trajectory_summary.jsonl'
    with compact_path.open('x') as compact:
        for origin,row in log_rows():
            assert row['step']==expected_step and row['s']==expected_step-1
            epoch=(expected_step-1)//1217;batch=(expected_step-1)%1217
            assert row['epoch']==epoch
            if active_epoch!=epoch:
                expected_ids=[]
                for rank in range(4):
                    sampler=DistributedSampler(range(1245901),num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
                    sampler.set_epoch(epoch);expected_ids.append(list(sampler))
                active_epoch=epoch
            lr=learning_rates(expected_step-1,3651)
            assert math.isclose(row['lr_backbone'],lr[0],rel_tol=1e-12) and math.isclose(row['lr_mask'],lr[1],rel_tol=1e-12)
            assert row['nonfinite']==0 and all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            assert row['inc_weight']==(min(1.,(expected_step-1)/200) if row['valid_global']>=2 else 0.)
            size=180 if batch==1216 else 256
            assert row['F_candidates']==4*size
            assert row['O_candidates']==row['E_candidates']==(row['valid_global'] if row['valid_global']>=2 else 0)
            stats=epoch_stats[epoch];stats['updates']+=1
            if row['duplicate_image_ids']:stats['duplicate_steps'].append(expected_step)
            lightweight=[]
            for rank,h in enumerate(row['rank_health']):
                assert h['rank']==rank and h['batch']==size and h['gradients_finite']
                assert h['updates']==expected_step-(500 if origin=='continuation' else 0)
                assert all(math.isfinite(v) for v in h['gradient_norms'].values())
                s=h['sampling'];ids=[i+1000 for i in expected_ids[rank][batch*256:(batch+1)*256]]
                assert s['sample_ids']==ids and len(s['K'])==len(s['n'])==size
                calculated=defaultdict(Counter)
                for sample_id,n,k in zip(ids,s['n'],s['K']):
                    if k:
                        assert n>=2 and 1<=k<n and k==sample_split_k(n,0,epoch,sample_id)
                        calculated[str(n)][str(k)]+=1
                    else:assert n in (0,1)
                assert dict(calculated)==s['K_histogram_by_n']
                assert s['valid_count']==sum(k>0 for k in s['K'])==h['valid']
                assert s['k1_count']==sum(k==1 for k in s['K'])
                stats['samples']+=size;stats['valid']+=h['valid'];stats['k1']+=s['k1_count'];stats['reasons'].update(h['reasons'])
                stats['prefix_lengths'].update(s['prefix_token_lengths']);stats['remainder_lengths'].update(s['remainder_token_lengths'])
                for n,hist in calculated.items():groups[epoch][n].update(hist)
                light={k:v for k,v in h.items() if k!='sampling'}
                light['sampling']={k:v for k,v in s.items() if k.endswith('_sha256') or k in ('valid_count','k1_count')}
                lightweight.append(light)
            assert sum(h['valid'] for h in row['rank_health'])==row['valid_global']
            row['common_loss']=row['loss']-row['inc_weight']*row['inc']
            row['rank_health']=lightweight;row['origin']=origin
            compact.write(json.dumps(row,ensure_ascii=False)+'\n')
            last50.append(row)
            if expected_step in (1,500,501,1000,1217,1218,2000,2434,2435,3000,3651):histories[str(expected_step)]=row
            if expected_step%250==0:print(f'Audited through step {expected_step}',flush=True)
            expected_step+=1
    assert expected_step==3652
    for e,stats in epoch_stats.items():
        assert stats['updates']==1217 and stats['samples']==1245904
        assert sum(stats['reasons'].values())==stats['samples']
        stats['valid_ratio']=stats['valid']/stats['samples'];stats['k1_ratio']=stats['k1']/stats['valid']
        stats['K_histogram_by_n']={n:dict(h) for n,h in sorted(groups[e].items(),key=lambda v:int(v[0]))}
    expected_checkpoints=[500]+list(range(600,3601,100))+[3651]
    assert sorted(p.name for p in RUN.glob('step*.pt'))==[f'step{s:06d}.pt' for s in expected_checkpoints]
    checkpoints=[]
    for step in expected_checkpoints:
        path=RUN/f'step{step:06d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['completed_steps']==step and state['scheduler_horizon']==state['stop_updates']==3651
        assert state['next_epoch']==step//1217 and state['next_batch']==step%1217
        assert len(state['rng_per_rank'])==4 and all(int(v['step'])==step for v in state['optimizer']['state'].values())
        del state
        checkpoints.append(dict(step=step,server_path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
        print(f'Checkpoint verified: {step}',flush=True)
    console=(EVIDENCE/'train3651.console.txt').read_text()
    diagnostics=dict(traceback_present='Traceback' in console,worker_abort_present='killed by signal' in console,
                     nccl_error_present='ncclSystemError' in console)
    metric_keys=[k for k,v in last50[-1].items() if isinstance(v,(int,float)) and k not in ('step','s','epoch')]
    result=dict(passed=True,code_commit=config['git_head'],config=config,execution=execution,acceptance=acceptance,
                updates_total=3651,updates_this_run=3151,horizon=3651,restoration=read(EVIDENCE/'restore-check.json'),
                samples_total=sum(x['samples'] for x in epoch_stats.values()),epoch_stats=epoch_stats,
                all_sample_ids_match_distributed_sampler=True,all_K_replay=True,
                epoch_boundary_steps={k:histories[k] for k in ('1217','1218','2434','2435','3651')},
                milestones=histories,last50={k:sum(x[k] for x in last50)/50 for k in metric_keys},
                checkpoints=checkpoints,diagnostics=diagnostics,
                raw_logs=[dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size) for p in
                          (ROOT/'randomk500/A3-RandomK/steps.jsonl',RUN/'steps.jsonl')],
                compact_log=dict(path=str(compact_path),sha256=file_sha(compact_path),
                                 note='Derived projection retaining all update metrics, per-rank health and stream digests; full n/K/sample arrays remain in immutable server logs'))
    write(EVIDENCE/'training-audit.json',result)
    print(json.dumps(dict(passed=True,updates=3651,samples=result['samples_total'],diagnostics=diagnostics)),flush=True)


if __name__=='__main__':main()
