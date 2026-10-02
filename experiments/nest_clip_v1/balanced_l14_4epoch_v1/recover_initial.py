"""CPU-only reconstruction of the omitted initial checkpoint from verified pristine sources."""
import copy
import datetime as dt
import json
import random

import numpy as np
import torch

from experiments.nest_clip_v1.balanced_l14_4epoch_v1.prepare import EXP,RUN,INIT
from train.nested_semantic_data import file_sha
from train.train_nested_semantic_mask import atomic_save,validate_resume_payload


def main():
    assert not torch.cuda.is_available(), 'Run this helper with CUDA_VISIBLE_DEVICES empty'
    state=json.loads((RUN/'state.json').read_text())
    tid=state['l14_trial']
    root=RUN/'trials'/tid/'step4868'
    destination=root/'step000000.pt'
    assert not destination.exists(), 'Never overwrite a checkpoint'
    config=json.loads((root/'config.json').read_text())
    assert config['start_updates']==0 and config['resume'] is None and config['horizon']==4868
    assert file_sha(INIT)==config['init_sha256']
    initial=torch.load(INIT,map_location='cpu',weights_only=False)
    smoke_path=RUN/'trials'/tid/'smoke/step000005.pt'
    smoke=torch.load(smoke_path,map_location='cpu',weights_only=False)
    assert smoke['config']['code_sha256']==config['code_sha256']
    assert smoke['config']['component_initialization']==config['component_initialization']
    assert not initial['optimizer']['state']
    assert [g['param_names'] for g in initial['optimizer']['param_groups']]==[
        g['param_names'] for g in smoke['optimizer']['param_groups']]
    np_initial=np.random.RandomState(0).get_state()
    loader_initial=torch.Generator().manual_seed(0).get_state()
    for rank in smoke['rng_per_rank']:
        assert torch.equal(rank['cpu'],initial['rng_cpu'])
        assert rank['python']==random.Random(0).getstate()
        assert rank['numpy'][0]==np_initial[0] and np.array_equal(rank['numpy'][1],np_initial[1])
        assert rank['numpy'][2:]==np_initial[2:]
        assert rank['cuda'].numel()==16 and int(rank['cuda'].count_nonzero())==0
        assert torch.equal(rank['loader_generator'],loader_initial)
    provenance=dict(derived=True,trainer_emitted=False,
                    cause='Generic search configuration suppressed save_initial_checkpoint',
                    source_initial=str(INIT),source_initial_sha256=file_sha(INIT),
                    source_smoke_rng=str(smoke_path),source_smoke_sha256=file_sha(smoke_path),
                    verified_cpu_rng_equal_to_initial=True,verified_python_numpy_seed0=True,
                    verified_cuda_seed0_unadvanced=True,verified_loader_epoch_start_seed0=True,
                    empty_optimizer_from_initial=True,actual_formal_config_preserved=True,
                    reconstructed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    replay_scope='Resume-schema and initial-state equivalence; not a bitwise replay claim for ongoing BF16 training')
    payload=dict(model=initial['model'],adapter=initial['adapter'],optimizer=initial['optimizer'],
                 completed_steps=0,scheduler_horizon=4868,stop_updates=4868,
                 rng_per_rank=copy.deepcopy(smoke['rng_per_rank']),next_epoch=0,next_batch=0,
                 config=config,reconstruction=provenance)
    assert validate_resume_payload(payload,config)==0
    atomic_save(payload,destination)
    report=dict(passed=True,checkpoint=str(destination),checkpoint_sha256=file_sha(destination),
                resume_schema_passed=True,provenance=provenance)
    (EXP/'evidence/initial-checkpoint-reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'step000000.reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
