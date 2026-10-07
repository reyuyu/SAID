"""Pinned INC0 source isolation and1000 local real tensor/text/sampler replay."""
import argparse
import hashlib
import itertools
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,dump,sha,now
from recovery.s02_local500 import INDEX,IMAGES
from recovery.s02_full_local_data import FullLocalDataset

BASE_SHA='e201975982809961cf4f078522e3137203c52726'
EXP=ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1'
RAW=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns500-20261008.preflight'
BASE_EXP=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
BASE_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-inc0-500-20261007/INC0'
FROZEN_SOURCES=('model/nested_fusion_mask.py','model/nested_semantic_mask.py',
    'model/model_longclip.py','model/longclip.py','train/nested_semantic_data.py',
    'train/said_cvssl_data.py','recovery/s02_full_local_data.py','recovery/s02_full_stage.py',
    'recovery/local500_policy.py','tools/eval_nest_native.py',
    'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py',
    'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py')


def git_bytes(path):
    return subprocess.check_output(['git','show',BASE_SHA+':'+path],cwd=ROOT)


def source_isolation():
    frozen={}
    for path in FROZEN_SOURCES:
        original=hashlib.sha256(git_bytes(path)).hexdigest()
        assert sha(ROOT/path)==original,('Frozen source changed',path)
        frozen[path]=original
    base=json.loads(git_bytes('experiments/nest_clip_v1/nested_d3_inc0_500_v1/config.json'))
    cfg=json.loads((EXP/'config.json').read_text())
    assert cfg==dict(base,hns_enabled=True),'Only hns_enabled may differ from fetched INC0 config'
    old=json.loads((BASE_RUN/'step500/config.json').read_text())
    assert old['init_sha256']==STEP0_SHA and old['horizon']==4868 and old['workers']==8
    assert old['view_weights']==[1.35,1.35,.30] and old['sampling_mode']=='nested_detail_d3' and old['inclusion_max']==0
    assert old['batch_size']==256 and old['world_size']==4 and old['accumulation']==1
    result=json.loads(git_bytes('experiments/nest_clip_v1/nested_d3_inc0_500_v1/RESULTS.json'))
    assert result==json.loads((BASE_EXP/'RESULTS.json').read_text())
    assert result['evaluation_checkpoint_immutable'] and sha(BASE_RUN/'step500/step000500.pt')==result['checkpoint_sha256']
    assert sha(STEP0)==STEP0_SHA
    return dict(passed=True,base_branch='experiment/nested-d3-inc0-500-v1',base_commit=BASE_SHA,
        frozen_source_SHA256=frozen,sole_configuration_addition={'hns_enabled':True},
        checkpoint_identity=result['checkpoint_sha256'],common0_SHA256=STEP0_SHA,
        optimizer_groups=old['parameter_counts']['optimizer_groups'],parameter_counts=old['parameter_counts'],
        precision='fp32 parameters, bf16 native encoder autocast, no GradScaler',
        mask_width=512,shared_mask_parameters_across_views=True,
        original_hidden_detach_preserved=True,mask_to_mask_stop_gradient=False,
        candidate_protocol='F all global examples; Dall/D3 all valid global examples',
        scheduler='native manual per-group LR; backbone warmup200; horizon4868',
        native_export_source_before=hashlib.sha256(git_bytes('tools/nest_clip.py')).hexdigest(),
        native_export_constructor_only_adds_HNS_flag=True,
        data_semantics_preprocessing_optimizer_scheduler_evaluation_unchanged=True)


def main():
    started=now();torch.set_num_threads(4)
    assert not RAW.exists(),'Do not silently overwrite/repeat matched preflight'
    RAW.mkdir(parents=True)
    isolation=source_isolation()
    # Compile the actual fetched dataset class, rather than a guessed replica.
    namespace={'__name__':'inc0_fetched_local_dataset','__file__':str(ROOT/'recovery/s02_full_local_data.py')}
    exec(compile(git_bytes('recovery/s02_full_local_data.py'),'INC0@'+BASE_SHA,'exec'),namespace)
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    reference=namespace['FullLocalDataset'](INDEX,IMAGES,'nested_detail_d3',0)
    expected=json.loads((BASE_RUN/'sampling-audit-1000.json').read_text())
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    identities=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),250):
            a,b=dataset[index],reference[index];old=expected[len(identities)]
            assert a['sample_id']==b['sample_id']==old['sample_id'] and old['rank']==rank
            assert dataset.resolved_path(index)==reference.resolved_path(index)
            assert torch.equal(a['image'],b['image']) and torch.isfinite(a['image']).all()
            assert a['views']==b['views'] and a['detail_indices']==b['detail_indices']
            for label,key,pos in [('F','tokens_f',0),('Dall','tokens_o',1),('D3','tokens_e',2)]:
                assert torch.equal(a[key],b[key]) and a[key].tolist()==old['token_ids'][label]
                assert a['views'][pos]==old['strings'][label]
            assert a['detail_indices']==old['sentence_indices']['lowest']
            identities.append(dict(rank=rank,sample_id=a['sample_id'],image_path=str(dataset.resolved_path(index)),
                image_tensor_sha256=hashlib.sha256(a['image'].contiguous().numpy().tobytes()).hexdigest(),
                views=a['views'],token_ids={k:a[k].tolist() for k in ('tokens_f','tokens_o','tokens_e')},
                K=len(a['detail_indices']),selected_indices=a['detail_indices']))
    after=np.random.get_state()
    assert random.getstate()==before[0] and torch.equal(torch.get_rng_state(),before[2])
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    raw=RAW/'matched-real1000.json';dump(raw,identities)
    report=dict(passed=True,records=1000,seed=0,epoch=0,sampler_order_exact=True,
        sample_ids_text_token_K_indices_image_path_and_preprocess_exact=True,global_RNG_unchanged=True,
        image_decode_source='Both dataset classes read only the local mirror, never NFS',
        common_initialization_and_model_optimizer_candidate_scheduler_conditions=isolation,
        started_utc=started,ended_utc=now(),local_raw_artifacts=[dict(path=str(raw),bytes=raw.stat().st_size,
            sha256=sha(raw),uploaded=False,time_range_utc=[started,now()])])
    dump(EXP/'MATCHED_PREFLIGHT.json',report);dump(EXP/'BASELINE_PROVENANCE.json',isolation)
    print(json.dumps(dict(passed=True,records=1000,base_commit=BASE_SHA)),flush=True)


if __name__=='__main__':main()
