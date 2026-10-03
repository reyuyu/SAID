"""Verify the real continuation checkpoint and the completed four-epoch reference."""
import copy
import json
from pathlib import Path

import torch

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import (
    REPO, SHARED, hparams, load, native_metrics, sha,
)
from train.train_nested_semantic_mask import code_manifest, validate_resume_payload

EXP = Path(__file__).resolve().parent
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_4epoch_v1')
PARENT_RUN = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_500_v1')
BASELINE_RUN = Path('/root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch')
BRANCH = 'codex/nest-balanced-rdrop-4epoch-v1'
BASE = load(REPO/'configs/nest_balanced_rdrop_4epoch_v1.json')


def reference_records():
    parent_state = load(PARENT_RUN/'state.json')
    baseline_state = load(BASELINE_RUN/'state.json')
    assert parent_state['status'] == baseline_state['status'] == 'completed'
    parent = copy.deepcopy(parent_state['trials'][parent_state['rdrop_trial']]['budgets']['500'])
    baseline_trial = baseline_state['trials'][baseline_state['four_epoch_experiment']['trial_id']]
    return parent, copy.deepcopy(baseline_trial['budgets']['4868']), [
        Path(baseline_trial['budgets'][str(stop)]['root'])/'steps.jsonl' for stop in (3651, 4868)]


def main():
    parent, baseline, baseline_logs = reference_records()
    assert hparams(parent['config']) == hparams(baseline['config']) == hparams(BASE)
    assert parent['config']['init_sha256'] == baseline['config']['init_sha256'] == sha(SHARED)
    assert parent['config']['horizon'] == baseline['config']['horizon'] == 4868
    assert parent['config']['remainder_mode'] == 'sentence_drop'
    assert baseline['config'].get('remainder_mode', 'compact') == 'compact'
    assert parent['step_range'] == [1, 500] and parent['matched_streams_equal']
    assert load(PARENT_RUN/'state.json')['tests']['passed']
    for record in (parent, baseline):
        assert sha(record['checkpoint']) == record['checkpoint_sha256']
        assert sha(record['student']) == record['bare_sha256']
        actual, raw, sources = native_metrics(Path(record['root']))
        assert actual == record['metrics'] and len(raw) == 5
    previous = torch.load(parent['checkpoint'], map_location='cpu', weights_only=False)
    current = dict(previous['config'], **BASE)
    current.update(max_updates=4868, resume=parent['checkpoint'], code_sha256=code_manifest())
    assert validate_resume_payload(previous, current) == 500
    assert previous['optimizer']['state'] and previous['adapter'] is not None
    assert all('loader_generator' in state for state in previous['rng_per_rank'])
    result = dict(passed=True, parent_checkpoint=parent['checkpoint'],
                  parent_checkpoint_sha256=parent['checkpoint_sha256'],
                  baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
                  common_step0_sha256=parent['config']['init_sha256'],
                  start_updates=500, stop_updates=4868, new_updates=4368, horizon=4868,
                  next_epoch=previous['next_epoch'], next_batch=previous['next_batch'],
                  model_optimizer_rng_loader_present_and_valid=True, training_code_byte_identical=True,
                  original_sentence_drop_tests_passed=True, baseline_retrained=False,
                  baseline_native_metrics_verified=True, baseline_logs=list(map(str, baseline_logs)))
    (EXP/'evidence').mkdir(parents=True, exist_ok=True)
    (EXP/'evidence/preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
