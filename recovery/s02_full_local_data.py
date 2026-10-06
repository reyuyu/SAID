"""Fail-fast native local data resolver; this module never starts training."""
import json
import mmap
from pathlib import Path

from recovery.s02_full_stage import ROOT, LOCAL, IMAGES, INDEX_SHA, MANIFEST_SHA, local_path
from train.nested_semantic_data import NestedDataset


class FullLocalDataset(NestedDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.image_root.resolve()!=IMAGES.resolve():
            raise RuntimeError('Full reproduction requires the disposable local mirror')

    def resolved_path(self,index):
        import numpy as np
        if self._records is None:
            self._file=(self.index_dir/'records.jsonl').open('rb')
            self._records=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)
            self._offsets=np.load(self.index_dir/'offsets.npy',mmap_mode='r')
        record=json.loads(self._records[self._offsets[index]:self._offsets[index+1]])
        path=local_path(record['image'])
        if not path.is_file():
            raise RuntimeError(f'Missing local sample={index+1000}; NFS fallback forbidden')
        return path

    def __getitem__(self,index):
        self.resolved_path(index)  # Before native Image.open; preserve all values/relative paths.
        return super().__getitem__(index)


def frozen_path_proof(count=5000):
    import torch
    from torch.utils.data import DistributedSampler
    import itertools
    dataset=FullLocalDataset(LOCAL/'data_index',IMAGES,'summary_random_detail',0)
    if dataset.metadata['records_sha256']!=INDEX_SHA:
        raise RuntimeError('Local training index drift')
    before=torch.get_rng_state().clone()
    proof=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),count//4):
            path=dataset.resolved_path(index)
            proof.append(dict(rank=rank,sample_id=index+1000,resolved_path=str(path)))
    if len(proof)!=count or not torch.equal(before,torch.get_rng_state()):
        raise RuntimeError('Path-proof count/RNG drift')
    return dict(passed=True,checked=count,local_root=str(IMAGES),NFS_fallback=False,
        seed=0,epoch=0,global_RNG_unchanged=True,rows=proof)


def configure_native_training():
    ready=json.loads((LOCAL/'full-ready.json').read_text())
    if ready['status']!='LOCAL_FULL_TRAINING_DATA_READY' or not ready['verification']['passed']:
        raise RuntimeError('Full-cache admission missing')
    if ready['manifest']['sha256']!=MANIFEST_SHA:
        raise RuntimeError('Full-cache manifest drift')
    from train import train_nested_semantic_mask as trainer
    trainer.NestedDataset=FullLocalDataset
    return dict(image_root=str(IMAGES),index_dir=str(LOCAL/'data_index'),
        relative_paths_unchanged=True,missing='fail-fast',NFS_fallback=False,training_launched=False)
