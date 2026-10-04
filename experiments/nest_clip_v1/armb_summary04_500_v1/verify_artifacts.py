"""Verify matched500/native5/spot8 artifacts and registered decision."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from experiments.nest_clip_v1.armb_summary04_500_v1.run import EXP,ROOT,RUN,SOURCE,WEIGHTS
from experiments.nest_clip_v1.armb_summary04_500_v1.report import add_short,classify
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
from train.nested_semantic_data import file_sha


def main():
    result=load(EXP/'RESULTS.json');old=load(EXP/'ARM_B_S06.json');baseline=load(EXP/'BASELINE.json')
    assert result['status']=='COMPLETE'
    assert result['config']['view_weights']==WEIGHTS and result['config']['sampling_mode']=='summary_random_detail'
    assert result['config']['horizon']==4868 and result['config']['resume'] is None and result['config']['start_updates']==0 and result['config']['max_updates']==500
    assert result['config']['code_sha256']==old['config']['code_sha256']
    for path,sha in load(EXP/'evidence/preflight.json')['unchanged_production_sources'].items():
        assert file_sha(ROOT/path)==sha
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show','b85f73a:'+path],cwd=ROOT)
    assert load(EXP/'evidence/SAMPLING_1000.json')['passed']
    assert load(EXP/'evidence/final-training-match.json')['passed']
    m,raw,_=native_metrics(RUN/'step500');assert m==result['metrics']
    s=scores(m);s['Short4_R1']=add_short(result)['Short4_R1'];assert s==result['scores']
    assert load(EXP/'DECISION.json')==classify(s,add_short(old),add_short(baseline))
    assert load(EXP/'DECISION.json')['full_training_launched'] is False
    for ds,value in raw.items():assert load(EXP/'raw'/(ds+'.json'))==value
    assert all(0<=m[ds][dr]['R@1']<=m[ds][dr]['R@5']<=m[ds][dr]['R@10']<=1 for ds in m for dr in ['I2T','T2I'])
    with gzip.open(EXP/'evidence/steps.jsonl.gz','rt') as f:steps=[json.loads(line) for line in f]
    assert len(steps)==500 and steps[-1]['step']==500
    prior=records(Path(old['checkpoint']).parent/'steps.jsonl')
    for row,ref in zip(steps,prior):
        for a,b in zip(row['rank_health'],ref['rank_health']):
            for key in ['sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256']:
                assert a['sampling'][key]==b['sampling'][key]
            assert a['sampling']['random_detail_sampling']['selected_indices_sha256']==b['sampling']['random_detail_sampling']['selected_indices_sha256']
    resource=load(EXP/'RESOURCE_SUMMARY.json');assert resource['normal_cycles']==495 and resource['normal_full_cycle_mean_seconds']<=3 and resource['peak_allocated_gib']<=65
    proof=load(EXP/'evidence/spot8-proof.json');assert len(proof['batches'])==8 and proof['optimizer_steps']==0
    for batch in proof['batches']:
        for rank in batch['states']:
            assert rank['before']==rank['after']
            assert rank['update_calls']=={'optimizer_step':0,'scaler_step':0}
        for agreement in batch['DDP_gradient_agreements'].values():assert agreement['four_rank_max_abs_gradient_difference']==0 and agreement['tested_tensors']>=20
    spot=load(EXP/'GRADIENT_SPOT_CHECK.json');assert spot['passed'] and spot['batches']==8 and spot['checkpoint_sha256']==result['checkpoint_sha256']
    assert spot['optimizer_steps']==0 and file_sha(result['checkpoint'])==result['checkpoint_sha256']
    commands={p.stem:load(p) for p in (EXP/'commands').glob('*.json')};assert len(commands)==10 and all(c['exit_code']==0 for c in commands.values())
    for path in EXP.rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):assert path.stat().st_size<10*1024*1024 and path.suffix not in ['.pt','.pth','.npy','.npz','.safetensors']
    dump(EXP/'evidence/FINAL_VERIFICATION.json',dict(passed=True,formal_updates=500,scheduler_horizon=4868,
      exact_ArmB_sampling_all500x4=True,strict_export=True,native_recalls_checked=30,spot_batches=8,
      spot_optimizer_steps=0,resources_pass=True,decision_consistent=True,production_sources_unchanged=True,
      checkpoints_gradients_datasets_not_committed=True,full_training_launched=False))
    print(json.dumps({'passed':True,'updates':500,'native_recalls':30,'spot_batches':8,'status':result['classification']['status']}))

if __name__=='__main__':main()
