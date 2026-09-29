"""E11 stream, formula and checkpoint verification, without further training."""
from collections import deque
import json
import math
from pathlib import Path
import sys
import torch
REPO=Path('/root/lk_projects/SAID');ROOT=Path('/root/lk_projects/SAID-nest-clip-v1');BASE=ROOT/'hybridf_v1/E11';EV=BASE/'evidence'
sys.path.insert(0,str(REPO))
from train.nested_semantic_data import file_sha
from train.train_nested_semantic_mask import learning_rates
def read(p):return json.loads(Path(p).read_text())
def stream(paths):
    for p in paths:
        with p.open() as f:
            for line in f:
                if line.strip():yield json.loads(line)
def equal(x,y):
    if torch.is_tensor(x):return torch.equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x==y
def audit(mode):
    n=5 if mode=='smoke' else 3651;run=BASE/mode;cfg=read(run/'config.json');acc=read(run/'acceptance.json')
    ex=read(EV/f'{mode}.execution.json');assert ex['exit_code']==0
    assert cfg['arm']=='A3' and cfg['full_native_mix']==.25 and cfg['sampling_mode']=='random_k'
    assert cfg['sampling_seed']==cfg['seed']==0 and cfg['horizon']==3651 and cfg['max_updates']==n
    assert cfg['resume'] is None and cfg['start_updates']==0 and cfg['updates_planned_this_run']==n
    assert cfg['training_records']==1245901 and cfg['batches_per_epoch']==1217
    for k,h in cfg['code_sha256'].items():assert file_sha(REPO/k)==h,k
    assert acc['passed'] and len(acc['ranks'])==4 and len({x['uuid'] for x in cfg['ranks']})==4
    for h in acc['ranks']:assert h['completed_updates']==h['updates_this_run']==n and h['max_parameter_difference_from_rank0']==0
    console=(EV/f'{mode}.console.txt').read_text()
    assert all(x not in console for x in ('Traceback','ncclSystemError','killed by signal','terminate called','RandomK token boundary overflow'))
    reference=stream([ROOT/'randomk500/A3-RandomK/steps.jsonl',ROOT/'randomk3epoch/A3-RandomK/steps.jsonl'])
    last50=deque(maxlen=50);milestones={};max_error=0.;samples=0;valid=0;compared=0
    compact=BASE/f'{mode}_trajectory.jsonl'
    with compact.open('x') as output:
        for i,row in enumerate(stream([run/'steps.jsonl']),1):
            old=next(reference);assert row['step']==old['step']==i and row['s']==i-1 and row['epoch']==(i-1)//1217
            assert row['full_native_mix']==.25 and row['nonfinite']==0
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            assert row['F_native_candidates']==row['F_candidates']==(720 if i%1217==0 else 1024)
            assert row['valid_global']==old['valid_global']
            assert (row['lr_backbone'],row['lr_mask'])==learning_rates(i-1,3651)
            weight=min(1,(i-1)/200) if row['valid_global']>=2 else 0;assert row['inc_weight']==weight
            hybrid=.75*(row['F_mask_i2t']+row['F_mask_t2i'])+.25*(row['F_native_i2t']+row['F_native_t2i'])
            assert math.isclose(hybrid,row['F_hybrid'],rel_tol=2e-6,abs_tol=2e-5)
            if row['valid_global']>=2:
                align=(10/3)*(hybrid+row['O_i2t']+row['O_t2i']+row['E_i2t']+row['E_t2i'])
                sparse=(row['F_sparse']+2*row['O_sparse']+2*row['E_sparse'])/3
            else:align=10*hybrid;sparse=row['F_sparse']
            rebuilt=align+sparse+weight*row['inc'];error=abs(rebuilt-row['loss']);max_error=max(max_error,error)
            assert math.isclose(rebuilt,row['loss'],rel_tol=2e-5,abs_tol=2e-4)
            assert row['loss_reconstruction_abs_error']<2e-4
            lights=[]
            for h,o in zip(row['rank_health'],old['rank_health']):
                assert h['batch']==o['batch'] and h['updates']==i and h['gradients_finite']
                assert all(math.isfinite(v) for v in h['gradient_norms'].values())
                assert h['stream_sha256']==o['stream_sha256']
                for key in ('sample_ids','n','K','sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256'):
                    assert h['sampling'][key]==o['sampling'][key],(i,h['rank'],key)
                samples+=h['batch'];valid+=h['valid'];compared+=1
                light={k:v for k,v in h.items() if k!='sampling'}
                light['sampling']={k:v for k,v in h['sampling'].items() if k.endswith('_sha256') or k in ('k1_count','valid_count')}
                lights.append(light)
            row['rank_health']=lights;row['common_loss']=row['loss']-weight*row['inc']
            output.write(json.dumps(row)+'\n');last50.append(row)
            if i in (1,5,100,200,201,500,1217,1218,2000,2434,2435,3000,3651):milestones[str(i)]=row
            if i%500==0:print('audited',i,flush=True)
    assert i==n
    common=torch.load(ROOT/'shared/step000000.pt',map_location='cpu',weights_only=False)
    zero=torch.load(run/'step000000.pt',map_location='cpu',weights_only=False)
    baseline=torch.load(ROOT/'randomk500/A3-RandomK/step000000.pt',map_location='cpu',weights_only=False)
    assert equal(common['model'],zero['model']) and equal(common['optimizer'],zero['optimizer'])
    for x,y in zip(zero['rng_per_rank'],baseline['rng_per_rank']):
        assert equal(x['python'],y['python']) and torch.equal(x['cpu'],y['cpu']) and torch.equal(x['cuda'],y['cuda'])
        assert x['numpy'][0]==y['numpy'][0] and (x['numpy'][1]==y['numpy'][1]).all() and x['numpy'][2:]==y['numpy'][2:]
    del common,zero,baseline
    expected=[0,5] if mode=='smoke' else [0]+list(range(100,3601,100))+[3651]
    assert sorted(p.name for p in run.glob('step*.pt'))==[f'step{x:06d}.pt' for x in expected]
    checkpoints=[]
    for step in expected:
        p=run/f'step{step:06d}.pt';ckpt=torch.load(p,map_location='cpu',weights_only=False)
        assert ckpt['completed_steps']==step and ckpt['scheduler_horizon']==3651 and ckpt['stop_updates']==n
        assert ckpt['config']['full_native_mix']==.25
        assert all(int(v['step'])==step for v in ckpt['optimizer']['state'].values())
        del ckpt
        checkpoints.append(dict(step=step,path=str(p),sha256=file_sha(p),bytes=p.stat().st_size))
        print('checkpoint verified',step,flush=True)
    keys=[k for k,v in last50[-1].items() if isinstance(v,(int,float)) and k not in ('step','s','epoch')]
    result=dict(passed=True,mode=mode,code_commit=cfg['git_head'],config=cfg,acceptance=acc,execution=ex,
                max_loss_reconstruction_abs_error=max_error,all_native_candidates_global=True,
                sample_rows=samples,valid_sample_rows=valid,compared_rank_streams=compared,
                sample_F_P_R_K_streams_identical_to_A3=True,step0_model_optimizer_equal=True,step0_rng_equal=True,
                losses_last=[r['loss'] for r in last50],last50={k:sum(r[k] for r in last50)/len(last50) for k in keys},
                milestones=milestones,checkpoints=checkpoints,diagnostics='No worker/NCCL/traceback diagnostics',
                raw_log=dict(path=str(run/'steps.jsonl'),sha256=file_sha(run/'steps.jsonl')))
    with (EV/f'{mode}-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,mode=mode,steps=n,max_reconstruction_error=max_error,samples=samples)),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);audit(sys.argv[1])
