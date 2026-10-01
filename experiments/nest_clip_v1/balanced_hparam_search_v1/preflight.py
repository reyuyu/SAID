"""Record resource, data and reference provenance without exposing credentials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    exp=Path(__file__).resolve().parent
    repo=exp.parents[2]
    init=Path('/root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt')
    checksum=sha(init)
    assert checksum=='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
    query=['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,mig.mode.current','--format=csv']
    devices=subprocess.check_output(query,text=True)
    assert len(devices.strip().splitlines())==5 and devices.count('A100')==4
    activity=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'],text=True)
    assert len(activity.strip().splitlines())==1
    metadata=json.loads(Path('/root/lk_projects/SAID-nest-clip-v1/data_index/metadata.json').read_text())
    assert metadata['training_records']==1245901
    reference=repo/'experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json'
    exact=subprocess.check_output(['git','show',f'6bafa8a009af4ffee1100027d5171d78dbeb96d3:{reference.relative_to(repo)}'],cwd=repo)
    assert exact==reference.read_bytes()
    assets=Path('/root/lk_projects/SAID-assets')
    required=[assets/'evaluation/coco/val2017',assets/'evaluation/coco/annotations/captions_val2017.json',
              assets/'evaluation/Urban1k/Urban1k',assets/'retrieval_benchmarks/manifests/flickr30k_test1k.jsonl',
              assets/'retrieval_benchmarks/manifests/docci_test.jsonl',assets/'retrieval_benchmarks/flickr30k/images',
              assets/'retrieval_benchmarks/docci/images']
    assert all(p.exists() for p in required)
    result=dict(passed=True,gpus=devices,active_processes=activity,
                topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
                environment={k:v for k,v in os.environ.items() if k.startswith('NCCL') or k=='CUDA_VISIBLE_DEVICES'},
                torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
                initial_checkpoint_sha256=checksum,data=metadata,reference_commit='6bafa8a009af4ffee1100027d5171d78dbeb96d3',
                reference_sha256=sha(reference),evaluation_assets=[str(p) for p in required],
                parent_result_commit='ec7194ac7ad1e6e8d0bfecabf0cf5831bb632946',
                parent_results_sha256=sha(repo/'experiments/nest_clip_v1/balanced_stack_3epoch_v1/FULL3EPOCH_RESULTS.json'))
    (exp/'evidence/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
