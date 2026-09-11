#!/bin/bash
# SAID-CLS-CVSSL v0.1 -- build the ONE frozen initial state all four arms must share.
#
#   bash tools/exp_said_cls_cvssl_init.sh
#
# Starts from the same frozen CLIP initial state the historical SmartCLIP / SALU comparisons use
# (runs_salu/phase30a_2_A_said_only/salu_initial.pt), loads every tensor that exists there, keeps a
# deterministically seeded mask_net if it is absent (it is present in that file), and writes the
# complete state dict -- plus its SHA-256 -- to runs_salu/said_cls_cvssl/shared_init/.
set -u
cd /root/SAID-gap-completion || exit 1
SOURCE=${SOURCE_INIT:-runs_salu/phase30a_2_A_said_only/salu_initial.pt}
OUT=runs_salu/said_cls_cvssl/shared_init
mkdir -p "$OUT"
/root/miniconda3/envs/said-smartclip/bin/python - "$SOURCE" "$OUT/cvssl_initial.pt" <<'PY'
import hashlib
import json
import sys

import torch

sys.path.insert(0, '/root/SAID-gap-completion')
sys.path.insert(0, '/root/SAID-gap-completion/train')

from model import longclip  # noqa: E402


class _Args:
    base_model = 'ViT-B/16'


source, target = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
import random
import numpy as np
random.seed(0)
np.random.seed(0)

model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=_Args())
payload = torch.load(source, map_location='cpu', weights_only=False)
if isinstance(payload, dict) and 'model' in payload:
    payload = payload['model']
tgt = set(model.state_dict())
mapped, ignored = {}, []
for key, value in payload.items():
    if key in tgt:
        mapped[key] = value
    elif key.startswith('clip.') and key[len('clip.'):] in tgt:
        mapped[key[len('clip.'):]] = value
    elif ('clip.' + key) in tgt:
        mapped['clip.' + key] = value
    else:
        ignored.append(key)
missing, unexpected = model.load_state_dict(mapped, strict=False)
bad = [k for k in missing if not k.startswith('mask_net')]
if bad or list(unexpected):
    raise SystemExit('init mapping failed: missing %r unexpected %r' % (bad, list(unexpected)))
state = model.state_dict()
digest = hashlib.sha256()
for key in sorted(state):
    digest.update(key.encode('utf-8'))
    digest.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
torch.save({'model': state, 'source': source, 'sha256': digest.hexdigest(),
            'loaded_tensors': len(mapped), 'mask_net_from_source': len(mapped) - len(
                [k for k in mapped if 'mask_net' in k])}, target)
print('SHARED_INIT %s' % target)
print('shared_initial_state_sha256 %s' % digest.hexdigest())
print('loaded %d tensors, ignored %d, mask_net missing %d'
      % (len(mapped), len(ignored), len([k for k in missing if k.startswith('mask_net')])))
PY
