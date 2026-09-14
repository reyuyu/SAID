import hashlib
import json
import sys
from pathlib import Path

import torch

run_dir = Path(__file__).resolve().parent
checkpoint = run_dir / 'bare_student_step2000.pt'
expected = '54f5c8b601c237e883a79cb82cb85865e3e0ff27e7399a189eb95c21361faf15'
digest = hashlib.sha256()
with checkpoint.open('rb') as source:
    for chunk in iter(lambda: source.read(8 << 20), b''):
        digest.update(chunk)
assert digest.hexdigest() == expected
sys.path.insert(0, '/root/SAID-gap-completion')
from tools.eval_urban1k_cls import load_student
model, preprocess, metadata = load_student(str(checkpoint), 'ViT-B/16', 'cpu')
assert metadata['loaded_tensors'] == 317
assert metadata['missing_keys'] == [] and metadata['unexpected_keys'] == []
assert all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values())
result = {
    'arm': 'S0_DUALMASK_FULL_V01',
    'completed_steps_from_matching_export_metadata': 2000,
    'checkpoint_sha256': digest.hexdigest(),
    'checkpoint_size_bytes': checkpoint.stat().st_size,
    'strict_load': metadata,
    'all_parameters_finite': True,
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'evaluation': 'normalize(encode_image) dot normalize(encode_text)',
}
(run_dir / 'checkpoint_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
