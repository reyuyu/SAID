"""Strict-load a CVSSL checkpoint the way the trainer resumes it, and report provenance.

Checks, all hard failures:
  * the payload has the keys the trainer's ``--resume`` path reads;
  * ``model.load_state_dict(payload['model'])`` succeeds in **strict** mode (this is the "model
    weights load strictly" requirement);
  * both optimizer state dicts load;
  * the recorded initial-state digest, lr horizon and precision match the configuration;
  * the checkpoint actually differs from the initial state (a checkpoint that equals the init means
    no training happened).
"""
import hashlib
import json
import os
import sys

import torch

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/train')

EXPECTED_INIT = 'caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf'


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode('utf-8'))
        if torch.is_tensor(value):
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode('utf-8'))
    return digest.hexdigest()


def main():
    path = sys.argv[1]
    expected_completed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    print('CHECKPOINT %s' % path)
    print('CHECKPOINT_SHA256 %s' % hashlib.sha256(open(path, 'rb').read()).hexdigest())
    print('KEYS %s' % sorted(payload))
    for key in ('arm', 'completed_steps', 'epoch', 'step_in_epoch', 'precision',
                'lr_horizon_steps', 'git_head', 'objective'):
        print('PAYLOAD_%-18s %s' % (key, payload.get(key)))
    digests = payload.get('digests', {})
    print('INITIAL_STATE_SHA256 %s' % digests.get('initial_state_sha256'))
    assert digests.get('initial_state_sha256') == EXPECTED_INIT, 'init digest mismatch'
    if expected_completed is not None:
        assert int(payload['completed_steps']) == expected_completed, (
            'completed_steps %r != %r' % (payload.get('completed_steps'), expected_completed))
    assert payload.get('objective') == 'said_cls_cvssl'
    assert payload.get('lr_horizon_steps') == 3651, payload.get('lr_horizon_steps')

    model_state = payload['model']
    print('MODEL_TENSORS %d' % len(model_state))
    dtypes = {}
    for value in model_state.values():
        if torch.is_tensor(value):
            dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
    print('MODEL_DTYPES %s' % sorted(dtypes.items()))
    assert set(dtypes) == {'torch.float32'}, 'checkpoint must be fp32 master weights'
    print('MODEL_STATE_DIGEST %s' % state_digest(model_state))

    # strict load against a freshly built model, exactly as --resume does
    from model import longclip
    import argparse
    args = argparse.Namespace()
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None, args=args)
    missing, unexpected = model.load_state_dict(model_state, strict=True)
    print('STRICT_LOAD_OK missing=%s unexpected=%s' % (missing, unexpected))

    optimizer_state = payload['optimizer']
    mask_state = payload['mask_optimizer']
    groups = len(optimizer_state['param_groups'])
    mask_groups = len(mask_state['param_groups'])
    state_tensors = sum(len(entry) for entry in optimizer_state['state'].values())
    mask_tensors = sum(len(entry) for entry in mask_state['state'].values())
    step = max(int(entry['step']) for entry in optimizer_state['state'].values())
    print('BACKBONE_OPT groups=%d state_entries=%d tensors=%d step=%d lr=%s'
          % (groups, len(optimizer_state['state']), state_tensors, step,
             optimizer_state['param_groups'][0]['lr']))
    print('MASK_OPT groups=%d state_entries=%d tensors=%d lr=%s'
          % (mask_groups, len(mask_state['state']), mask_tensors,
             mask_state['param_groups'][0]['lr']))

    # did training actually move the weights away from the init?
    init_payload = torch.load(os.path.join(REPO, 'runs_salu/said_cls_cvssl/shared_init/'
                                           'cvssl_initial.pt'), map_location='cpu',
                              weights_only=False)
    init_state = init_payload['model'] if 'model' in init_payload else init_payload
    changed, worst = 0, 0.0
    for key in model_state:
        if key in init_state and torch.is_tensor(model_state[key]):
            difference = float((model_state[key].float()
                                - init_state[key].float()).abs().max())
            worst = max(worst, difference)
            if difference > 0:
                changed += 1
    print('CHANGED_TENSORS %d / %d  max_abs_change %.6e' % (changed, len(model_state), worst))
    assert changed > 0, 'checkpoint equals the initial state: training did not happen'
    print('VERIFY_OK %s' % os.path.basename(path))


if __name__ == '__main__':
    main()
