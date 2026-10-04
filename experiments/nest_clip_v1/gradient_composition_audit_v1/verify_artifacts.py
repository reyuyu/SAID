"""Validate complete audit evidence, identical inputs and zero parameter updates."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,ROOT,RUN,load,dump,stages
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import OBJECTIVES
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import GROUPS
from train.nested_semantic_data import file_sha


def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
    elif isinstance(value,float):assert math.isfinite(value),value


def main():
    summary=load(EXP/'GRADIENT_SUMMARY.json');assert summary['status']=='COMPLETE' and summary['optimizer_steps']==0
    assert list(summary['stages'])==['0','R','B']
    with gzip.open(EXP/'RAW_BATCH_GRADIENT_STATS.json.gz','rt') as f:raw=json.load(f)
    assert len(raw['batches'])==96;finite(summary);finite(raw)
    token=load(EXP/'evidence/FIXED_BATCH_TOKEN_MANIFEST.json');assert token['passed'] and len(token['manifest'])==32
    source=load(EXP/'evidence/production-sources-unchanged.json')
    for path,sha in source['sources'].items():
        assert file_sha(ROOT/path)==sha
        original=subprocess.check_output(['git','show','b85f73a:'+path],cwd=ROOT);assert (ROOT/path).read_bytes()==original
    assert load(EXP/'evidence/DDP_GLOBAL_CE_ORACLE.json')['passed']
    inputs={};statechecks=gradchecks=0
    for stage in ['0','R','B']:
        batch=load(EXP/'evidence'/('batches_'+stage+'.json'));assert len(batch['batches'])==32
        proof=load(EXP/'evidence'/('stage_'+stage)/'state-proof.json');assert len(proof['batches'])==32
        execution=load(EXP/'evidence'/('stage_'+stage)/'execution.json')
        assert execution['checkpoint_file_SHA256_before_after_equal'] and execution['optimizer_step_calls']==execution['scaler_step_calls']==0
        assert execution['identity']['total_trainable']==156931074
        for i,(row,p,manifest) in enumerate(zip(batch['batches'],proof['batches'],token['manifest'])):
            assert row['batch']==p['batch']==manifest['batch']==i+1
            assert row['global_sample_ID_sha256']==manifest['global_sample_ID_sha256']
            for rank,(input,control,m) in enumerate(zip(row['ranks'],p['state_checks'],manifest['ranks'])):
                statechecks+=1;assert input['rank']==control['rank']==rank
                assert control['unchanged'] and control['before_digest']==control['after_digest']
                assert control['update_calls']=={'optimizer_step':0,'scaler_step':0}
                assert input['sampling']['sample_ids']==m['sample_ids']
                mode='summary_random_detail' if stage=='B' else 'random_k'
                assert input['sampling']['K']==m['views'][mode]['K']
                for key,sha in m['views'][mode]['production_digests'].items():assert input['sampling'][key]==sha
                key=(i,rank);identity=(input['image_tensor_sha256'],input['sampling']['full_view_sha256'])
                if key in inputs:assert inputs[key]==identity
                else:inputs[key]=identity
            assert set(p['gradient_agreements'])==set(OBJECTIVES)
            for agreement in p['gradient_agreements'].values():
                assert agreement['tested_tensors']>=20 and agreement['four_rank_max_abs_gradient_difference']==0 and agreement['DDP_synchronized'];gradchecks+=1
            if i in [0,31]:assert p['repeat_gradient']['bitwise_equal']
        mainrows=[r for r in raw['batches'] if r['stage']==stage];assert len(mainrows)==32
        for row in mainrows:
            assert row['Gram_objective_order']==OBJECTIVES
            assert row['native_correctness']['shape']==[1024,1024]
            for g in GROUPS[2:7]:
                assert row['groups'][g]['norms']['native_combined']==0
                assert row['groups'][g]['pairs']['F_combined__native_combined']['cosine'] is None
                assert row['groups'][g]['pairs']['F_combined__native_combined']['projection_onto_second'] is None
            for g in GROUPS[:7]:
                matrix=row['Gram_FP64'][g]
                for a in range(16):
                    assert matrix[a][a]>=0
                    for b in range(16):assert abs(matrix[a][b]-matrix[b][a])<1e-8
        control=summary['stages'][stage]['FP32_first_batch_control'];assert control['linearity']['relative_L2']<5e-5 and control['model_unchanged']
        assert mainrows[0]['native_correctness']['direct_top1_rankings_equal']
        provenance=load(EXP/'CHECKPOINT_PROVENANCE.json')[stage]
        assert provenance['unchanged'] and provenance['sha256_before']==provenance['sha256_after']==stages()[stage]['sha256']
        assert file_sha(stages()[stage]['checkpoint'])==provenance['sha256_after']
    commands={p.stem:load(p) for p in (EXP/'commands').glob('*.json')}
    for stage in ['0','R','B']:
        assert commands['audit-stage-'+stage]['exit_code']==0
        key='fp32-control-0-v2' if stage=='0' and 'fp32-control-0-v2' in commands else 'fp32-control-'+stage
        assert commands[key]['exit_code']==0
    if 'fp32-control-0-v2' in commands:assert commands['fp32-control-0']['exit_code']!=0  # preserved corrected failure
    assert len(summary['recommendations']['recommended_candidates'])<=3 and summary['recommendations']['no_training_launched']
    dump(EXP/'evidence/FINAL_VERIFICATION.json',{'passed':True,'main_batches':96,'world_size':4,'model_state_before_after_checks':statechecks,
      'main_objective_DDP_agreements':gradchecks,'minimum_tensors_per_agreement':20,'minimum_gradient_tensor_checks':gradchecks*20,
      'identical_image_and_F_inputs_all32x4x3':True,'exact_historical_sampling':True,'native_zero_auxiliary_NA':True,
      'FP32_controls_all3_stages':True,'all_checkpoints_unmodified_SHA256':True,'production_sources_unchanged':True,'optimizer_steps':0})
    print(json.dumps({'passed':True,'main_batches':96,'statechecks':statechecks,'DDP_objective_checks':gradchecks,'optimizer_steps':0}))

if __name__=='__main__':main()
