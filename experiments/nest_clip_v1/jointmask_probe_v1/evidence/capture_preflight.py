"""Capture immutable resource, device and software facts for the probe."""
import json
import hashlib
import os
from pathlib import Path
import subprocess

import torch


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


server = Path('/root/lk_projects/SAID-nest-clip-v1')
assets = Path('/root/lk_projects/SAID-assets')
resources = {
    'training_index': server / 'data_index/metadata.json',
    'training_images': assets / 'training/ShareGPT4V',
    'coco': assets / 'evaluation/coco/val2017',
    'urban': assets / 'evaluation/Urban1k/Urban1k',
    'flickr_manifest': assets / 'retrieval_benchmarks/manifests/flickr30k_test1k.jsonl',
    'flickr_images': assets / 'retrieval_benchmarks/flickr30k/images',
    'docci_manifest': assets / 'retrieval_benchmarks/manifests/docci_test.jsonl',
    'docci_images': assets / 'retrieval_benchmarks/docci/images',
    'long_dci_manifest': assets / 'retrieval_benchmarks/manifests/long_dci_reconstructed.jsonl',
    'long_dci_images': assets / 'retrieval_benchmarks/dci/images'}
assert all(path.exists() for path in resources.values())
initial = server / 'shared/step000000.pt'
initial_sha = file_sha(initial)
assert initial_sha == '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
metadata = json.loads(resources['training_index'].read_text())
assert metadata['training_records'] == 1245901
gpu_rows = []
for row in command('nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.free,mig.mode.current',
                   '--format=csv,noheader,nounits').splitlines():
    index, name, uuid, total, free, mig = [value.strip() for value in row.split(',')]
    gpu_rows.append(dict(index=int(index), name=name, uuid=uuid,
                         total_mib=int(total), free_mib=int(free), mig=mig))
assert len(gpu_rows) == 4 and len({row['uuid'] for row in gpu_rows}) == 4
assert all('A100' in row['name'] and row['mig'] == 'Disabled' for row in gpu_rows)
result = dict(
    passed=True,
    git_branch=command('git', 'branch', '--show-current'),
    base_revision='25a5d1242231050ce544c4a1df76613cd6a69930',
    initial_checkpoint=str(initial),
    initial_sha256=initial_sha,
    training_metadata=metadata,
    batches_per_epoch=1217,
    scheduler_horizon=3651,
    formal_stop=500,
    gpu=gpu_rows,
    topology=command('nvidia-smi', 'topo', '-m'),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    nccl_environment={key: value for key, value in os.environ.items() if key.startswith('NCCL')},
    software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                  nccl=torch.cuda.nccl.version()),
    resources={name: str(path) for name, path in resources.items()},
    evaluation_protocol=['coco', 'urban', 'flickr_test1k', 'docci', 'long_dci'],
    excluded_by_user=['dci_full'])
destination = Path(__file__).with_name('preflight.json')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
