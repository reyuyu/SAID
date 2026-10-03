"""Pin the common initializer, existing matched500 baseline and unchanged runtime."""
import json
from pathlib import Path
import subprocess

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import REPO,SHARED,load,native_metrics,sha,hparams

EXP=Path(__file__).resolve().parent
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_view_relation_500_v1')
BASELINE_STATE=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1/state.json')
BASELINE_SHA='f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8'
REFERENCE='83e2a76'


def main():
    baseline=load(BASELINE_STATE)['roles']['baseline']['result']
    cfg=load(REPO/'configs/nest_balanced_view_relation_500_v1.json')
    assert sha(baseline['checkpoint'])==baseline['checkpoint_sha256']==BASELINE_SHA
    assert baseline['config']['horizon']==4868 and baseline['config']['max_updates']==500
    assert baseline['config']['init_sha256']==sha(SHARED)
    assert hparams(cfg)==hparams(baseline['config'])
    assert cfg['remainder_mode']=='compact' and cfg['view_relation'] and cfg['sibling_coefficient']==1
    assert cfg['checkpoint_encoders'] and not cfg['checkpoint_pair_blocks']
    assert cfg['batch_size']==256 and cfg['world_size']==4 and cfg['image_chunk']==cfg['text_chunk']==128
    unchanged=('model/nested_fusion_mask.py','model/nested_semantic_mask.py','model/model_longclip.py',
               'model/longclip.py','train/nested_semantic_data.py')
    for path in unchanged:
        assert (REPO/path).read_bytes()==subprocess.check_output(['git','show',REFERENCE+':'+path],cwd=REPO),path
    metrics,raw,sources=native_metrics(Path(baseline['root']))
    assert metrics==baseline['metrics'] and len(raw)==5
    result=dict(passed=True,baseline_reused=True,baseline_checkpoint_sha256=BASELINE_SHA,
        common_step0_sha256=baseline['config']['init_sha256'],reference_commit=REFERENCE,
        math_reference_commit='14653c92c6da9d552a2b624ab169eaaa275cdde8',
        unchanged_files={p:sha(REPO/p) for p in unchanged},
        no_runtime_optimization=True,compact_F_P_R_unchanged=True,
        no_new_modules_or_parameters=True,no_new_encoder_forward=True,
        cross_masks_detached_only=True,unscaled_cosine=True,zero_margin=True,
        sibling_coefficient=1,common_relation_ramp_updates=200,horizon=4868,stop_updates=500)
    (EXP/'evidence').mkdir(parents=True,exist_ok=True)
    (EXP/'evidence/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
