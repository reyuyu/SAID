"""Verify two fixed500 runs, native30 recalls each and immutable diagnostics."""
import ast
import gzip
import json
import math
from pathlib import Path
import subprocess
from experiments.nest_clip_v1.armb_summary_dose_500_v1.run import EXP,ROOT,RUN,S04,ARMS
from experiments.nest_clip_v1.armb_summary_dose_500_v1.report import make_curve,gradient_curve,summarize_decision,score_record
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1.run import load,dump,records
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
from train.nested_semantic_data import file_sha


def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
    elif isinstance(value,float):assert math.isfinite(value)


def main():
    source=load(S04/'RESULTS.json');old=records(Path(source['checkpoint']).parent/'steps.jsonl');checks={}
    for key,(directory,weights) in ARMS.items():
        exp=EXP/directory;run=RUN/directory;r=load(exp/'RESULTS.json');assert r['status']=='COMPLETE'
        assert r['config']['view_weights']==weights and r['config']['sampling_mode']=='summary_random_detail'
        assert r['config']['horizon']==4868 and r['config']['resume'] is None and r['config']['start_updates']==0 and r['config']['max_updates']==500
        assert r['config']['component_initialization']==source['config']['component_initialization'] and r['config']['parameter_counts']==source['config']['parameter_counts']
        for path,proof in load(exp/'evidence/preflight.json')['frozen_production_sources'].items():
            assert file_sha(ROOT/path)==proof['current_sha256']
            reference=subprocess.check_output(['git','show','bbe5daa:'+path],cwd=ROOT)
            if not proof['allowed_validation_only_change']:assert (ROOT/path).read_bytes()==reference
            else:
                nodes=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='hparams')]
                assert nodes(ast.parse((ROOT/path).read_bytes()))==nodes(ast.parse(reference))
        assert load(exp/'evidence/preflight.json')['sampling_1000_exact']['passed']
        with gzip.open(exp/'evidence/steps.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        assert len(rows)==500
        for row,ref in zip(rows,old):
            for a,b in zip(row['rank_health'],ref['rank_health']):
                for field in ['sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256']:assert a['sampling'][field]==b['sampling'][field]
                assert a['sampling']['random_detail_sampling']['selected_indices_sha256']==b['sampling']['random_detail_sampling']['selected_indices_sha256']
        m,raw,_=native_metrics(run/'step500');assert m==r['metrics'];assert score_record(r)==r['scores']
        for ds,value in raw.items():assert load(exp/'raw'/(ds+'.json'))==value
        assert all(0<=m[ds][dr]['R@1']<=m[ds][dr]['R@5']<=m[ds][dr]['R@10']<=1 for ds in m for dr in ['I2T','T2I'])
        resource=load(exp/'RESOURCE_SUMMARY.json');assert resource['normal_cycles']==495 and resource['normal_full_cycle_mean_seconds']<=3 and resource['peak_allocated_gib']<=65
        for name,count,objectives in [('GRADIENT_SPOT8',8,5),('LAST50_GRADIENT_SHARES',50,3)]:
            summary=load(exp/(name+'.json'));proof=load(exp/'evidence'/(name+'-proof.json'))
            assert summary['passed'] and summary['batches']==count and summary['optimizer_steps']==0 and summary['checkpoint_sha256']==r['checkpoint_sha256']
            assert len(proof['batches'])==count
            for batch in proof['batches']:
                assert len(batch['gradient_agreements'])==objectives
                for agreement in batch['gradient_agreements'].values():assert agreement['tested_tensors']>=20 and agreement['four_rank_max_abs_gradient_difference']==0
                for state in batch['state_checks']:assert state['before']==state['after'] and state['update_calls']=={'optimizer_step':0,'scaler_step':0}
            if key=='E2':assert summary['groups']['native_backbone_total']['weighted_norm_share_means']['O_combined']==0.
        assert file_sha(r['checkpoint'])==r['checkpoint_sha256']
        if key=='E2':assert load(exp/'TRAINING_DIAGNOSTICS.json')['last50']['weighted_CE_contribution']['S']==0.
        commands={p.stem:load(p) for p in (exp/'commands').glob('*.json')};assert len(commands)==11 and all(v['exit_code']==0 for v in commands.values())
        checks[key]=dict(passed=True,updates=500,strict_export=True,native_recalls=30,live_sampling_batches=2000,spot8_backs=40,last50_replay_backs=150,forward_retained=True)
    curve=make_curve();gradients=gradient_curve();assert load(EXP/'DECISION.json')==summarize_decision(curve,gradients)
    assert [r['S_weight'] for r in curve]==[1.,.6,.4,.2,0.]
    assert load(EXP/'DECISION.json')['no_full_training'] and load(EXP/'DECISION.json')['no_extra_weights_tested']
    finite(load(EXP/'DOSE_RESULTS.json'))
    for path in EXP.rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):assert path.stat().st_size<10*1024*1024 and path.suffix not in ['.pt','.pth','.npy','.npz','.safetensors']
    dump(EXP/'evidence/FINAL_VERIFICATION.json',dict(passed=True,arms=checks,dose_curve_and_decision_consistent=True,no_extra_training=True,no_full_training=True,gradient_diagnostics_updates=0))
    print(json.dumps({'passed':True,'arms':list(checks),'formal_updates':1000,'native_recalls':60,'gradient_updates':0}))

if __name__=='__main__':main()
